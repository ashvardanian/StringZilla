/**
 *  @brief  Multi-pattern exact and case-folded substring search for CPUs.
 *  @file   include/stringzillas/find_many.hpp
 *  @author Ash Vardanian
 *  @sa     include/stringzillas/find_many.cuh for the GPU backends.
 *
 *  This is a thin hub aggregating the per-backend headers. The shared vocabulary, the automaton, its
 *  construction, and the CPU engines all live in `find_many/serial.hpp`, matching how the `similarities` and
 *  `fingerprints` families are laid out.
 *
 *  There are no per-ISA kernels here, unlike the other families. A transition is one data-dependent load on
 *  a serial dependency chain, so parallelism comes from advancing many haystacks at once rather than from
 *  vector instructions, and per-ISA variation reduces to the chain count taken from `cpu_specs_t`.
 */
#ifndef STRINGZILLAS_FIND_MANY_HPP_
#define STRINGZILLAS_FIND_MANY_HPP_

#include "stringzillas/find_many/serial.hpp" // Automaton, dictionary builder, and CPU engines

#endif // STRINGZILLAS_FIND_MANY_HPP_
