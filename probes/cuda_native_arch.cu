/**
 *  @file probes/cuda_native_arch.cu
 *  @author Ash Vardanian
 *  @date August 13, 2026
 *  @brief CUDA probe building a `.cu` source for this machine's own instruction set.
 *
 *  Targeting the native instruction set turns on the host compiler's widest intrinsic headers.
 *  NVCC reads those headers on its device pass even though only host functions use them, and
 *  some pairs of toolkit and host disagree over the builtins inside, so the header is what the
 *  probe carries - the flag alone would compile anywhere.
 */
#include <immintrin.h>

__global__ void sz_probe_kernel_(void) {}

int main(void) { return 0; }
