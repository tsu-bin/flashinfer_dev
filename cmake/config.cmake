# Whether to compile fp8 kernels or not.
set(FLASHINFER_ENABLE_FP8 ON)
# Whether to compile bf16 kernels or not.
set(FLASHINFER_ENABLE_BF16 OFF)
# Whether to compile tvm bindings or not.
set(FLASHINFER_TVM_BINDING OFF)
# Whether to compile prefill kernel tests/benchmarks or not.
set(FLASHINFER_PREFILL OFF)
# Whether to compile decode kernel tests/benchmarks or not.
set(FLASHINFER_DECODE ON)
# Whether to compile page kernel tests/benchmarks or not.
set(FLASHINFER_PAGE OFF)
# Whether to compile cascade kernel tests/benchmarks or not.
set(FLASHINFER_CASCADE OFF)
# Whether to compile sampling kernel tests/benchmarks or not.
set(FLASHINFER_SAMPLING OFF)
# Whether to compile normalization kernel tests/benchmarks or not.
set(FLASHINFER_NORMALIZATION OFF)
# Whether to compile fastdiv tests
set(FLASHINFER_FASTDIV_TEST OFF)
# Whether to compile fastdequant tests
set(FLASHINFER_FASTDEQUANT_TEST OFF)
# Whether to compile distributed tests
set(FLASHINFER_DISTRIBUTED OFF)
# The following configurations can impact the binary
# size of the generated library
set(FLASHINFER_GEN_HEAD_DIMS 256 512) # 64 128 256
set(FLASHINFER_GEN_KV_LAYOUTS 0) # 0 1
set(FLASHINFER_GEN_POS_ENCODING_MODES 1) # 0 1 2
set(FLASHINFER_GEN_ALLOW_FP16_QK_REDUCTIONS "false") # "false" "true"
set(FLASHINFER_GEN_MASK_MODES 1) # 0 1 2

set(FLASHINFER_TVM_SOURCE_DIR "/hostShare/tools/tvm_all/tvm-dev/")

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -G -keep")


# Set target cuda architectures for tests/benchmarks, defaults to native.
# "native" is a special value for CMAKE_CUDA_ARCHITECTURES which means use the architectures of the host's GPU.
# it's new in CMake 3.24, if you are using an older of CMake or you want to use a different value, you can
# set its value here. Supported CUDA architctures include 80;86;89;90
# NOTE(Zihao): using "native" might be slow because whenever compile a cuda file with `-arch=native`, nvcc will spawn
# a `__nvcc_device_query` process to get the architecture of the host's GPU, which could stall the compilation process.
# So it's recommended to set it to a specific value if you know the architecture of the target GPU.
# Example:
# set(FLASHINFER_CUDA_ARCHITECTURES 80)
set(FLASHINFER_CUDA_ARCHITECTURES native)
