/**
 *  @brief CUDA-accelerated fingerprinting utilities for string collections - GPU backends hub.
 *  @file include/stringzillas/fingerprints.cuh
 *  @author Ash Vardanian
 *
 *  This is a thin hub. It first pulls in the CPU hub (for the ISA-agnostic template core in
 *  `fingerprints/serial.hpp`), then the per-GPU-tier specialization header `fingerprints/cuda.cuh`.
 *
 *  @sa fingerprints.hpp
 *  @sa fingerprints/serial.hpp
 *  @sa fingerprints/cuda.cuh
 */
#ifndef STRINGZILLAS_FINGERPRINTS_CUH_
#define STRINGZILLAS_FINGERPRINTS_CUH_

#include "stringzillas/fingerprints.hpp"      // ISA-agnostic template core (via fingerprints/serial.hpp) + CPU backends
#include "stringzillas/fingerprints/cuda.cuh" // Base CUDA tier

#endif // STRINGZILLAS_FINGERPRINTS_CUH_
