/**
 *  @file c/stringzillas/substrings_serial.cpp
 *  @brief Single CPU-variant instantiation unit for the multi-pattern Aho-Corasick engine.
 *  @author Ash Vardanian
 *  @date August 6, 2026
 *
 *  Multi-pattern search has no per-ISA kernels - a transition is one data-dependent load on a serial
 *  dependency chain, so parallelism comes from many independent haystacks rather than from vector
 *  instructions. Unlike the similarity siblings, which get one instantiation unit per ISA, this is the only
 *  one `substrings` needs.
 */
#include "stringzillas/substrings.hpp" // Dictionary + engine

namespace ashvardanian {
namespace stringzillas {

// A `u16` automaton halves the hot-row footprint, fitting twice as many rows in the GPU's shared-memory prefix.
template struct aho_corasick_dictionary<u16_t, std::allocator<char>>;
template struct substrings<u16_t, std::allocator<char>, sz_caps_sp_k>;
template struct aho_corasick_dictionary<u32_t, std::allocator<char>>;
template struct substrings<u32_t, std::allocator<char>, sz_caps_sp_k>;

} // namespace stringzillas
} // namespace ashvardanian
