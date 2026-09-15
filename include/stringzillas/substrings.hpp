/**
 *  @brief  Multi-pattern exact and case-folded substring search for CPUs.
 *  @file   include/stringzillas/substrings.hpp
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/substrings.cuh for the GPU backends.
 *
 *  This is a thin hub aggregating the per-backend headers. The shared vocabulary, the automaton, its
 *  construction, and the CPU engines all live in `substrings/serial.hpp`, matching how the `similarities` and
 *  `fingerprints` families are laid out.
 *
 *  There are no per-ISA kernels here, unlike the other families. A transition is one data-dependent load on
 *  a serial dependency chain, so parallelism comes from advancing many haystacks at once rather than from
 *  vector instructions, and per-ISA variation reduces to the chain count taken from `cpu_specs_t`.
 */
#ifndef STRINGZILLAS_SUBSTRINGS_HPP_
#define STRINGZILLAS_SUBSTRINGS_HPP_

#include "stringzillas/substrings/serial.hpp" // Automaton, dictionary builder, and CPU engines

#endif // STRINGZILLAS_SUBSTRINGS_HPP_
