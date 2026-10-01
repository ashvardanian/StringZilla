/**
 *  @file bench/cross_rocm.hip
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief GPU engine benchmarks - the ROCm kernels against the CPU tiers this build carries.
 */
#include "cross_simt.cuh"

int main() { return sz::bench::bench_simt_main(); }
