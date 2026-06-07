/**
 *  @brief Hardware-accelerated Min-Hash fingerprinting for string collections - CPU backends hub.
 *  @file include/stringzillas/fingerprints.hpp
 *  @author Ash Vardanian
 *
 *  This is a thin hub aggregating the per-backend fingerprint headers. The ISA-agnostic rolling-hash
 *  template core and the serial backend aliases live in `fingerprints/serial.hpp`; CPU SIMD
 *  specializations live in their own per-ISA files. GPU backends are aggregated by `fingerprints.cuh`.
 *
 *  See `fingerprints/serial.hpp` for the detailed description of the rolling-hash and Min-Hashing math.
 *
 *  @sa fingerprints/serial.hpp
 *  @sa fingerprints/haswell.hpp
 *  @sa fingerprints/skylake.hpp
 */
#ifndef STRINGZILLAS_FINGERPRINTS_HPP_
#define STRINGZILLAS_FINGERPRINTS_HPP_

#include "stringzillas/fingerprints/serial.hpp"  // ISA-agnostic rolling-hash template core + serial aliases
#include "stringzillas/fingerprints/haswell.hpp" // AVX2 (Haswell) specializations
#include "stringzillas/fingerprints/skylake.hpp" // AVX-512 (Skylake) specializations

#endif // STRINGZILLAS_FINGERPRINTS_HPP_
