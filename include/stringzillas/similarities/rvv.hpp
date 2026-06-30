/**
 *  @brief RISC-V Vector (RVV 1.0) string-similarity backend.
 *  @file include/stringzillas/similarities/rvv.hpp
 *  @author Ash Vardanian
 *  @sa include/stringzillas/similarities/serial.hpp
 *  @sa include/stringzillas/similarities/neon.hpp
 *
 *  @note Wave 1 foundation. Every engine in this header redirects to the scalar `sz_cap_serial_k`
 *        implementations, so `sz_caps_sr_k` (serial + RVV) is a fully dispatchable capability that runs
 *        at serial speed. The empty region markers below mark where the real vector-length-agnostic
 *        RVV kernels will land, region by region, in later waves - the build stays green throughout.
 */
#ifndef STRINGZILLAS_SIMILARITIES_RVV_HPP_
#define STRINGZILLAS_SIMILARITIES_RVV_HPP_

#include "stringzillas/similarities/serial.hpp"

namespace ashvardanian {
namespace stringzillas {

/*  RVV implementation of the string similarity algorithms for 64-bit RISC-V CPUs.
 *  Until the vector-length-agnostic kernels land, the whole backend forwards to the scalar serial
 *  oracle: the per-pair Myers and the tiny-input Wagner-Fischer horizontal walker inherit the serial
 *  versions, and the diagonal `tile_scorer` cells redirect to the serial cell recurrence. The generic
 *  serial diagonal/horizontal walkers and the generic serial batch engines then drive everything, so
 *  `sz_caps_sr_k` is bit-exact with `sz_cap_serial_k`.
 */
#pragma region RVV Implementation
#if SZ_USE_RVV
// ! Wave 1 emits no RVV intrinsics yet - every kernel below inherits the scalar serial implementation, so no
// ! function in this header carries the vector target. The first real kernel must re-introduce the function-scoped
// ! target attribute that the rest of the RVV backend uses (mirroring `include/stringzilla/find/rvv.h`):
// !   #if defined(__clang__)
// !   #pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
// !   #elif defined(__GNUC__)
// !   #pragma GCC push_options
// !   #pragma GCC target("arch=+v")
// !   #endif
// ! Pushing it now would trip `-Wpragma-clang-attribute` (an unused attribute region) under `-Werror`.

#pragma region Bit Parallel Myers

/**
 *  @brief Redirects the RVV byte Myers/Hyyr\xc3\xb6 unit-cost Levenshtein to the serial scalar version. The
 *         vector-length-agnostic lane-batched Myers will replace this in a later wave; for now the per-pair
 *         and cross-product entry points both score on the serial oracle.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<char, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public levenshtein_distance_myers<char, sz_cap_serial_k> {

    using base_t = levenshtein_distance_myers<char, sz_cap_serial_k>;
    using base_t::base_t;
    using base_t::operator();
};

/**
 *  @brief Redirects the RVV @b rune Myers unit-cost UTF-8 Levenshtein to the serial scalar version. Mirrors
 *         the byte redirect above; the serial rune Myers keeps a rune-keyed `match_masks` hash, so it is
 *         ISA-independent and bit-exact.
 */
template <sz_capability_t capability_>
struct levenshtein_distance_myers<rune_t, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public levenshtein_distance_myers<rune_t, sz_cap_serial_k> {

    using base_t = levenshtein_distance_myers<rune_t, sz_cap_serial_k>;
    using base_t::base_t;
    using base_t::operator();
};

#pragma endregion Bit Parallel Myers

/** @brief Redirects the RVV horizontal-walker specialization to the serial version. The Wagner-Fischer
 *         tiny-input path is not worth vectorizing, so it shares the serial implementation. */
template <typename char_type_, typename score_type_, typename substituter_type_, typename gap_costs_type_,
          sz_similarity_objective_t objective_, sz_similarity_locality_t locality_>
struct horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                         sz_cap_rvv_k, void>
    : public horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                               sz_cap_serial_k, void> {

    using base_t = horizontal_walker<char_type_, score_type_, substituter_type_, gap_costs_type_, objective_, locality_,
                                     sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

#pragma region Uniform Cost Levenshtein

/**
 *  @brief Redirects every RVV diagonal `tile_scorer` cell to the serial version. These four catch-alls
 *         (linear/affine x global/local, generic over the substituter and objective) cover both the
 *         uniform-cost Levenshtein cells and the class-based Needleman-Wunsch / Smith-Waterman cells, so
 *         the generic serial diagonal walker can drive the whole `sz_caps_sr_k` capability at serial speed.
 *         A later wave will replace the hot cell widths with specialized vector-length-agnostic kernels;
 *         these redirects stay as the fallback for the not-yet-vectorized combinations.
 */
template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         linear_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               linear_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, linear_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         linear_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               linear_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_global_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         affine_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               affine_gap_costs_t, objective_, sz_similarity_global_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

template <typename first_iterator_type_, typename second_iterator_type_, typename score_type_,
          typename substituter_type_, sz_similarity_objective_t objective_, sz_capability_t capability_>
struct tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_, affine_gap_costs_t,
                   objective_, sz_similarity_local_k, capability_, std::enable_if_t<(capability_ & sz_cap_rvv_k) != 0>>
    : public tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                         affine_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void> {

    using base_t = tile_scorer<first_iterator_type_, second_iterator_type_, score_type_, substituter_type_,
                               affine_gap_costs_t, objective_, sz_similarity_local_k, sz_cap_serial_k, void>;
    using base_t::base_t;
    using base_t::operator();
};

#pragma endregion // Uniform Cost Levenshtein

#pragma region RVV Inter Sequence Candidate Lanes
// ! Future home of the vector-length-agnostic candidate-lane Levenshtein driver. Until then the generic
// ! serial batch engines score the cross-product per pair via the serial diagonal walker above.
#pragma endregion RVV Inter Sequence Candidate Lanes

#pragma region Weighted Candidate Lane Walker
// ! Future home of the class-based (Needleman-Wunsch / Smith-Waterman) candidate-lane walker.
#pragma endregion Weighted Candidate Lane Walker

#pragma region Cross Product
// ! Future home of the RVV cross-product cell addressing, scoring, and public overloads. The generic serial
// ! `levenshtein_distances` / `needleman_wunsch_scores` / `smith_waterman_scores` engines cover this for now.
#pragma endregion Cross Product

// ! The matching `#pragma clang attribute pop` / `#pragma GCC pop_options` lands here once the first RVV kernel
// ! re-introduces the function-scoped vector target attribute documented at the top of this region.

#endif            // SZ_USE_RVV
#pragma endregion // RVV Implementation

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_SIMILARITIES_RVV_HPP_
