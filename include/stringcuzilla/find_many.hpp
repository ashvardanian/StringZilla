/**
 *  @brief  Hardware-accelerated multi-pattern exact substring search.
 *  @file   find_many.hpp
 *  @author Ash Vardanian
 *
 *  One of the most broadly used algorithms in string processing is the multi-pattern Aho-Corasick
 *  algorithm, that constructs a trie from the patterns, transforms it into a finite state machine,
 *  and then uses it to search for all patterns in the text in a single pass.
 *
 *  One of its biggest issues is the memory consumption, as the naive implementation requires each
 *  state to be proportional to the size of the alphabet, or 256 for byte-level processing. Such dense
 *  representations simplify transition lookup down to a single memory access, but that access can be
 *  expensive if the memory doesn't fir into the CPU caches for really large vocabulary sizes.
 *
 *  Addressing this, we provide a sparse layout variant of the FSM, that uses predicated SIMD instructions
 *  to rapidly probe the transitions and find the next state. This allows us to use a much smaller state,
 *  fitting in L1/L2 caches much more frequently.
 */
#ifndef STRINGZILLA_FIND_MANY_HPP_
#define STRINGZILLA_FIND_MANY_HPP_

#include "types.h"

#include "compare.h" // `sz_compare`
#include "memory.h"  // `sz_copy`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

#pragma endregion // Core API

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_FIND_MANY_HPP_
