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

// Both state-id widths, because the engine holds whichever one its needle set fits: a `u16` automaton halves
// the hot-row footprint, so twice as much of it stays cache-resident.
template struct aho_corasick_dictionary<u16_t, std::allocator<char>>;
template struct aho_corasick_dictionary<u32_t, std::allocator<char>>;

// Both capabilities are listed: the serial engine is what the C shim builds for a single-threaded scope, so
// omitting it only moves its instantiation into every consumer translation unit.
template struct substrings<std::allocator<char>, sz_cap_serial_k>;
template struct substrings<std::allocator<char>, sz_caps_sp_k>;

} // namespace stringzillas
} // namespace ashvardanian
