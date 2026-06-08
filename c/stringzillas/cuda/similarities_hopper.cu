// Hopper-tier similarity device TU: emits the 14 explicit entry kernels into ONE PTX blob compiled with
// `nvcc -ptx -arch=compute_90a` (DPX add-min/max linear cells).
#define SZ_GPU_TIER sz_caps_ckh_k
#include "stringzillas/similarities/cuda_hopper.cuh"
