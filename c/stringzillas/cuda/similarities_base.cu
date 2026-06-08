// Base-tier similarity device TU: emits the 14 explicit entry kernels into ONE PTX blob.
//
// Compiled standalone with `nvcc -ptx -arch=compute_50` (the base floor). The resulting PTX is
// `bin2c`-embedded into the shared object and loaded at runtime via the CUDA Driver API. This TU carries
// no host code - only the `extern "C" __global__` entry kernels from `cuda_maxwell.cuh`.
#define SZ_GPU_TIER sz_cap_cuda_k
#include "stringzillas/similarities/cuda_maxwell.cuh"
