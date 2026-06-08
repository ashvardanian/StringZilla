// Kepler-tier similarity device TU: emits the 14 explicit entry kernels into ONE PTX blob compiled with
// `nvcc -ptx -arch=compute_70`.
#define SZ_GPU_TIER sz_caps_ck_k
#include "stringzillas/similarities/cuda_kepler.cuh"
