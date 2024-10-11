"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import flashinfer
import pytest
import torch



@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [54, 97, 512])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA", "ALIBI"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [True, False])
@pytest.mark.parametrize("q_dtype", [torch.float16])
@pytest.mark.parametrize(
    "kv_dtype", [torch.float16, torch.float8_e4m3fn, torch.float8_e5m2]
)
@pytest.mark.parametrize("contiguous_kv", [True, False])
def test_batch_decode_with_tuple_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    q_dtype,
    kv_dtype,
    contiguous_kv,
):
    q = torch.randn(batch_size, num_qo_heads, head_dim).to(0).to(q_dtype)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32).to(0) for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].to(kv_dtype) for i in range(2)]
        for i in range(2):
            kv_data_fp32[i] = kv_data_fp32[i][:, 1, :, 1, :, 1, :]
            kv_data[i] = kv_data[i][:, 1, :, 1, :, 1, :]
            # actual data is stored in non-contiguous memory
            assert (
                kv_data[i].stride(-4)
                != kv_data[i].shape[-3] * kv_data[i].shape[-2] * kv_data[i].shape[-1]
            )
    else:
        kv_data_fp32 = [
            torch.randn(*kv_shape, dtype=torch.float32).to(0) for _ in range(2)
        ]
        kv_data = [kv_data_fp32[i].to(kv_dtype) for i in range(2)]
    kv_data = tuple(kv_data)
    kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * num_pages_per_seq
    kv_indices = torch.arange(0, total_num_pages).to(0).int()
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    ).to(0)

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=logits_soft_cap,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    if return_lse:
        o, _ = wrapper.run_return_lse(q, kv_data)
    else:
        o = wrapper.run(q, kv_data)

    k_cache, v_cache = kv_data_fp32
    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[i]
        ki = torch.cat(
            [
                k_cache[kv_indptr[i] : kv_indptr[i + 1] - 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    k_cache[kv_indptr[i + 1] - 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else k_cache[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        vi = torch.cat(
            [
                v_cache[kv_indptr[i] : kv_indptr[i + 1] - 1]
                .to(torch.float32)  # torch.cat does not support some fp8 types
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    v_cache[kv_indptr[i + 1] - 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else v_cache[kv_indptr[i + 1] - 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        o_ref_i = flashinfer.single_decode_with_kv_cache(
            qi,
            ki,
            vi,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
        )
        torch.testing.assert_close(o[i], o_ref_i, rtol=1e-3, atol=1e-3)


    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    q_dtype,
    kv_dtype,
    contiguous_kv,

if __name__ == "__main__":
    test_batch_decode_with_tuple_paged_kv_cache(
        batch_size=9, kv_len=34, page_size=16, num_kv_heads=1, num_qo_heads=4, head_dim=256, 
        kv_layout="NHD", pos_encoding_mode="ROPE_LLAMA", logits_soft_cap=0.0, return_lse=False, 
        q_dtype=torch.float16, kv_dtype=torch.float16,
        contiguous_kv = True
    )
    