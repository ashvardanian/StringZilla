/**
 *  @file c/stringzillas/substrings.cuh
 *  @brief Multi-pattern Aho-Corasick search shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date August 6, 2026
 */
#ifndef SZS_SUBSTRINGS_CUH_
#define SZS_SUBSTRINGS_CUH_
#include "stringzillas.cuh"

#include <cstddef> // `offsetof`

#pragma region Dispatch

/**
 *  @brief Allocates a `substrings_backends_t` holding an empty @p engine_type_ and publishes the opaque handle.
 *
 *  The engine arrives with no automaton: indexing needs a device, which only `szs_substrings_index` names, and
 *  a default-constructed engine touches neither a driver nor a device allocation.
 */
template <typename engine_type_>
inline sz_status_t emplace_empty_substrings_engine(szs_substrings_t *engine_punned,
                                                   char const **error_message) noexcept {
    auto engine = new (std::nothrow) substrings_backends_t(std::in_place_type_t<engine_type_>());
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate Substrings engine");
    *engine_punned = reinterpret_cast<szs_substrings_t>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

/** @brief Narrows the C policy to the engine's own enumeration, rather than assuming the two orders match. */
inline szs::substrings_overlap_policy_t szs_substrings_policy_(
    szs_substrings_overlap_policy_t overlap_policy) noexcept {
    switch (overlap_policy) {
    case szs_substrings_leftmost_longest_k: return szs::substrings_leftmost_longest_k;
    case szs_substrings_leftmost_first_k: return szs::substrings_leftmost_first_k;
    default: return szs::substrings_overlapping_k;
    }
}

/**
 *  @brief The compiled dictionary's needle count, which every per-needle array is validated against.
 *
 *  Templated on the backends type so the engines' members instantiate where it is called rather than here:
 *  reaching into a CUDA engine at namespace scope pulls its kernel table's addresses ahead of the device
 *  stubs NVCC emits for them, which a Clang host then rejects as a specialization after instantiation.
 */
template <typename backends_type_>
inline size_t szs_substrings_needles_count_(backends_type_ const &engine) noexcept {
    return std::visit([](auto const &engine_variant) { return (size_t)engine_variant.count_needles(); },
                      engine.variants);
}

/**
 *  @brief Runs @p operation on whichever engine and scope the two handles carry: the engine arm chosen by its
 *         own visit, the executor and specs by the scope's, and an engine-to-scope mismatch refused before
 *         either runs.
 *
 *  Every substrings operation reaches its engine through here, so the four cannot drift apart in how they
 *  pair a backend with a device. @sa `szs_levenshtein_cross_` for the same skeleton over one operation.
 */
/**
 *  @brief Refuses an engine whose needles were never indexed, before any operation walks an empty automaton.
 *
 *  Zero needles cannot mean anything else: `aho_corasick_dictionary::try_insert` rejects an empty needle, so an
 *  indexed engine always holds at least one. Without this a search would report zeros and call it success.
 */
inline sz_status_t szs_substrings_require_indexed_(szs_substrings_t engine_punned,
                                                   char const **error_message) noexcept {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto const &engine = *reinterpret_cast<substrings_backends_t const *>(engine_punned);
    if (szs_substrings_needles_count_(engine) != 0) return sz_success_k;
    return propagate_error(sz::status_t::unexpected_dimensions_k, error_message,
                           "No needles indexed; call `szs_substrings_index` first");
}

template <typename operation_type_>
inline sz_status_t szs_substrings_dispatch_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                            char const **error_message, operation_type_ &&operation) noexcept {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(device_punned != nullptr && "Device must be initialized");

    auto &engine = *reinterpret_cast<substrings_backends_t *>(engine_punned);
    auto &device = *reinterpret_cast<device_scope_t *>(device_punned);
    sz::status_t const status = std::visit(
        [&](auto &engine_variant) -> sz::status_t {
            using engine_variant_t = std::decay_t<decltype(engine_variant)>;
            if constexpr (is_gpu_capability(engine_variant_t::capability_k)) {
#if SZ_USE_CUDA
                auto [gpu_scope, scope_status] = gpu_scope_for(device);
                if (scope_status.status != sz::status_t::success_k) return scope_status.status;
                return operation(engine_variant, get_executor(gpu_scope), get_specs(gpu_scope));
#else
                return sz::status_t::missing_gpu_k;
#endif // SZ_USE_CUDA
            }
            // CPU scopes differ only in the executor they hand out, so one visitor covers both.
            else
                return std::visit(
                    [&](auto &scope_variant) -> sz::status_t {
                        using scope_t = std::decay_t<decltype(scope_variant)>;
                        if constexpr (!is_cpu_scope<scope_t>()) return sz::status_t::device_code_mismatch_k;
                        else return operation(engine_variant, get_executor(scope_variant), get_specs(scope_variant));
                    },
                    device.variants);
        },
        engine.variants);
    return propagate_error(status, error_message);
}

/**
 *  @brief Counts every needle's occurrences in every haystack, on whichever scope @p device_punned names.
 *  @sa `szs_levenshtein_cross_` for the same dispatch skeleton.
 */
template <typename haystacks_type_>
inline sz_status_t szs_substrings_count_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                         haystacks_type_ const &haystacks,
                                         szs_substrings_overlap_policy_t overlap_policy, sz_size_t *counts,
                                         sz_size_t *matches_total_out, char const **error_message) noexcept {
    sz_assert_(counts != nullptr && "Counts output cannot be null");
    sz_assert_(matches_total_out != nullptr && "Match-total output cannot be null");

    *matches_total_out = 0;
    if (sz_status_t const indexed = szs_substrings_require_indexed_(engine_punned, error_message);
        indexed != sz_success_k)
        return indexed;
    sz::span<size_t> const counts_span {counts, haystacks.size()};
    szs::substrings_overlap_policy_t const policy = szs_substrings_policy_(overlap_policy);
    size_t matches_total = 0;

    sz_status_t const result = szs_substrings_dispatch_( //
        engine_punned, device_punned, error_message,
        [&](auto &engine_variant, auto &&executor, auto const &specs) -> sz::status_t {
            return engine_variant.try_count(haystacks, policy, counts_span, matches_total, executor, specs);
        });
    *matches_total_out = (sz_size_t)matches_total;
    return result;
}

/**
 *  @brief Locates every needle's occurrences in every haystack. @sa `szs_substrings_count_`.
 *
 *  The caller's array IS the output on every arm, viewed as the C++ match. Capacity checks and host-memory
 *  staging live in the engines, not here.
 *
 *  Capacity is the engine's call alone. It counts before it can check anything anyway, and it reports the
 *  size it wanted rather than zero, which makes `matches_capacity == 0` a single-call size query.
 */
template <typename haystacks_type_>
inline sz_status_t szs_substrings_find_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                        haystacks_type_ const &haystacks,
                                        szs_substrings_overlap_policy_t overlap_policy, szs_substrings_match_t *matches,
                                        sz_size_t matches_capacity, sz_size_t *matches_found_out,
                                        char const **error_message) noexcept {
    sz_assert_(matches_found_out != nullptr && "Matches-found output cannot be null");

    *matches_found_out = 0;
    if (sz_status_t const indexed = szs_substrings_require_indexed_(engine_punned, error_message);
        indexed != sz_success_k)
        return indexed;

    static_assert(
        sizeof(szs::substrings_match_t) == sizeof(szs_substrings_match_t) &&
            offsetof(szs::substrings_match_t, haystack_index) == offsetof(szs_substrings_match_t, haystack_index) &&
            offsetof(szs::substrings_match_t, needle_index) == offsetof(szs_substrings_match_t, needle_index) &&
            offsetof(szs::substrings_match_t, byte_offset) == offsetof(szs_substrings_match_t, byte_offset) &&
            offsetof(szs::substrings_match_t, byte_length) == offsetof(szs_substrings_match_t, byte_length),
        "The C++ match must mirror the C ABI field for field, or this view would misreport matches");

    sz::span<szs::substrings_match_t> const matches_out {reinterpret_cast<szs::substrings_match_t *>(matches),
                                                         matches_capacity};
    szs::substrings_overlap_policy_t const policy = szs_substrings_policy_(overlap_policy);
    size_t matches_found = 0;

    sz_status_t const result = szs_substrings_dispatch_( //
        engine_punned, device_punned, error_message,
        [&](auto &engine_variant, auto &&executor, auto const &specs) -> sz::status_t {
            return engine_variant.try_find(haystacks, policy, matches_out, matches_found, executor, specs);
        });

    // The engines report the capacity they wanted rather than zeroing it, so the refused-buffer arm needs no
    // second walk here - `matches_found` already names the need, which is what makes a zero capacity a size query.
    *matches_found_out = (sz_size_t)matches_found;
    return result;
}

/**
 *  @brief Scores every haystack against the compiled dictionary. @sa `szs_substrings_count_`.
 *
 *  Weights are one per needle rather than a matrix: the dictionary @b is the query, so a second weighting
 *  is a second call. No overlap policy either - term frequencies are raw counts, which is classic BM25.
 */
template <typename haystacks_type_>
inline sz_status_t szs_substrings_score_bm25_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                              haystacks_type_ const &haystacks, sz_f32_t const *document_lengths,
                                              szs_substrings_bm25_t parameters, sz_f32_t const *needle_weights,
                                              sz_f32_t *scores, char const **error_message) noexcept {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(scores != nullptr && "Scores output cannot be null");
    if (sz_status_t const indexed = szs_substrings_require_indexed_(engine_punned, error_message);
        indexed != sz_success_k)
        return indexed;

    // The engines only assert their per-needle array sizes, and asserts compile out of release builds.
    if (needle_weights == nullptr)
        return propagate_error(sz::status_t::unexpected_dimensions_k, error_message,
                               "One weight per needle, no more and no fewer");

    // Normalizing divides by the corpus mean, so without one there is nothing to divide by. Letting that
    // through would drop `length_normalization` and every entry of `document_lengths` without a word.
    if (parameters.length_normalization > 0.0f && !(parameters.average_document_length > 0.0f))
        return propagate_error(sz::status_t::unexpected_dimensions_k, error_message,
                               "BM25 length normalization needs a positive `average_document_length`: " //
                               "pass the corpus mean, or set `length_normalization` to zero");

    auto const needles_count = szs_substrings_needles_count_(*reinterpret_cast<substrings_backends_t *>(engine_punned));
    szs::substrings_bm25_t const engine_parameters {
        parameters.term_frequency_saturation, parameters.length_normalization, parameters.average_document_length};
    sz::span<sz::f32_t> const scores_span {scores, haystacks.size()};
    sz::span<sz::f32_t const> const lengths_span {document_lengths, document_lengths ? haystacks.size() : 0};
    sz::span<sz::f32_t const> const weights_span {needle_weights, needles_count};

    return szs_substrings_dispatch_( //
        engine_punned, device_punned, error_message,
        [&](auto &engine_variant, auto &&executor, auto const &specs) -> sz::status_t {
            return engine_variant.try_score_bm25(haystacks, lengths_span, engine_parameters, weights_span, scores_span,
                                                 executor, specs);
        });
}

/**
 *  @brief Rewrites every haystack into one output tape. @sa `szs_substrings_count_`.
 *
 *  Tape in, tape out, because a rewrite's product is itself a tape. The offsets array is always filled,
 *  even when the byte capacity is refused, so a refused call still names the size it wanted.
 */
template <typename haystacks_type_>
inline sz_status_t szs_substrings_replace_(szs_substrings_t engine_punned, szs_device_scope_t device_punned,
                                           haystacks_type_ const &haystacks,
                                           szs_substrings_overlap_policy_t overlap_policy,
                                           sz_sequence_t const *replacements, sz_ptr_t output_data,
                                           sz_size_t output_data_capacity, sz_size_t *output_offsets,
                                           sz_size_t *output_bytes_written, char const **error_message) noexcept {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(replacements != nullptr && "Replacement collection cannot be null");
    sz_assert_(output_offsets != nullptr && "Offsets output cannot be null");
    sz_assert_(output_bytes_written != nullptr && "Bytes-written output cannot be null");

    *output_bytes_written = 0;
    auto const replacements_container = sz_sequence_as_cpp_container_t {replacements};
    auto const needles_count = szs_substrings_needles_count_(*reinterpret_cast<substrings_backends_t *>(engine_punned));
    if (replacements_container.size() != needles_count)
        return propagate_error(sz::status_t::unexpected_dimensions_k, error_message,
                               "One replacement per needle, no more and no fewer");

    szs::substrings_overlap_policy_t const policy = szs_substrings_policy_(overlap_policy);
    // The engines refuse this too, but only with a bare status - and a rewrite under overlapping matches is
    // the one refusal a caller is likely to hit by misreading the API rather than by mis-sizing a buffer.
    if (sz::status_t const rewritable = szs::substrings_check_rewritable(policy); rewritable != sz::status_t::success_k)
        return propagate_error(rewritable, error_message,
                               "A rewrite needs a cover whose matches share no bytes, so overlapping is refused");

    sz::span<char> const output_span {output_data, output_data_capacity};
    sz::span<size_t> const offsets_span {output_offsets, haystacks.size() + 1};
    size_t bytes_written = 0;

    sz_status_t const result = szs_substrings_dispatch_( //
        engine_punned, device_punned, error_message,
        [&](auto &engine_variant, auto &&executor, auto const &specs) -> sz::status_t {
            return engine_variant.try_replace(haystacks, policy, replacements_container, output_span, offsets_span,
                                              bytes_written, executor, specs);
        });

    // The engines report the capacity they wanted rather than zeroing it, which is what makes a zero
    // capacity a size query rather than a wasted call.
    *output_bytes_written = (sz_size_t)bytes_written;
    return result;
}

#pragma endregion Dispatch

extern "C" {

#pragma region Substrings

SZ_API_RUNTIME sz_status_t szs_substrings_init(                       //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities, //
    szs_substrings_t *engine_punned, char const **error_message) {

    sz_unused_(alloc); // Custom allocator not yet implemented, using default
    sz_assert_(engine_punned != nullptr && *engine_punned == nullptr && "Engine must be uninitialized");

#if SZ_USE_CUDA
    if ((capabilities & sz_cap_cuda_k) == sz_cap_cuda_k)
        return emplace_empty_substrings_engine<szs::substrings_cuda_t>(engine_punned, error_message);
#endif // SZ_USE_CUDA
    if ((capabilities & sz_caps_sp_k) == sz_caps_sp_k)
        return emplace_empty_substrings_engine<szs::substrings_parallel_t>(engine_punned, error_message);
    return emplace_empty_substrings_engine<szs::substrings_serial_t>(engine_punned, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_index(                                          //
    szs_substrings_t engine_punned, sz_sequence_t const *needles,                         //
    szs_substrings_case_sensitivity_t case_sensitivity, szs_device_scope_t device_punned, //
    char const **error_message) {

    sz_assert_(needles != nullptr && "Needle collection cannot be null");

    szs::substrings_case_sensitivity_t const sensitivity = case_sensitivity == szs_substrings_uncased_k
                                                               ? szs::substrings_uncased_k
                                                               : szs::substrings_cased_k;
    auto const needles_container = sz_sequence_as_cpp_container_t {needles};

    // The same dispatch every operation uses, so the hot tier is sized from the scope that will walk it and an
    // engine/scope mismatch is refused here rather than at the first search.
    return szs_substrings_dispatch_( //
        engine_punned, device_punned, error_message,
        [&](auto &engine_variant, auto &&executor, auto const &specs) -> sz::status_t {
            return engine_variant.try_index(needles_container, sensitivity, executor, specs);
        });
}

SZ_API_RUNTIME sz_status_t szs_substrings_count(                                    //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,               //
    sz_sequence_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy, //
    sz_size_t *counts, sz_size_t *matches_total,                                    //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_count_(engine_punned, device_punned, sz_sequence_as_cpp_container_t {haystacks},
                                 overlap_policy, counts, matches_total, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_find(                                            //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                      //
    sz_sequence_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy,        //
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_find_(engine_punned, device_punned, sz_sequence_as_cpp_container_t {haystacks},
                                overlap_policy, matches, matches_capacity, matches_found, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_count_u32tape(                                    //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                       //
    sz_sequence_u32tape_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy, //
    sz_size_t *counts, sz_size_t *matches_total,                                            //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_count_(engine_punned, device_punned, sz_sequence_u32tape_as_cpp_container_t {haystacks},
                                 overlap_policy, counts, matches_total, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_count_u64tape(                                    //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                       //
    sz_sequence_u64tape_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy, //
    sz_size_t *counts, sz_size_t *matches_total,                                            //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_count_(engine_punned, device_punned, sz_sequence_u64tape_as_cpp_container_t {haystacks},
                                 overlap_policy, counts, matches_total, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_find_u32tape(                                     //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                       //
    sz_sequence_u32tape_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy, //
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found,  //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_find_(engine_punned, device_punned, sz_sequence_u32tape_as_cpp_container_t {haystacks},
                                overlap_policy, matches, matches_capacity, matches_found, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_find_u64tape(                                     //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                       //
    sz_sequence_u64tape_t const *haystacks, szs_substrings_overlap_policy_t overlap_policy, //
    szs_substrings_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found,  //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_find_(engine_punned, device_punned, sz_sequence_u64tape_as_cpp_container_t {haystacks},
                                overlap_policy, matches, matches_capacity, matches_found, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_score_bm25(                 //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_t const *haystacks,                                   //
    sz_f32_t const *document_lengths,                                 //
    szs_substrings_bm25_t parameters,                                 //
    sz_f32_t const *needle_weights, sz_f32_t *scores,                 //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_score_bm25_(engine_punned, device_punned, sz_sequence_as_cpp_container_t {haystacks},
                                      document_lengths, parameters, needle_weights, scores, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_score_bm25_u32tape(         //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u32tape_t const *haystacks,                           //
    sz_f32_t const *document_lengths,                                 //
    szs_substrings_bm25_t parameters,                                 //
    sz_f32_t const *needle_weights, sz_f32_t *scores,                 //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_score_bm25_(engine_punned, device_punned, sz_sequence_u32tape_as_cpp_container_t {haystacks},
                                      document_lengths, parameters, needle_weights, scores, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_score_bm25_u64tape(         //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned, //
    sz_sequence_u64tape_t const *haystacks,                           //
    sz_f32_t const *document_lengths,                                 //
    szs_substrings_bm25_t parameters,                                 //
    sz_f32_t const *needle_weights, sz_f32_t *scores,                 //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_score_bm25_(engine_punned, device_punned, sz_sequence_u64tape_as_cpp_container_t {haystacks},
                                      document_lengths, parameters, needle_weights, scores, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_replace_bound(  //
    szs_substrings_t engine_punned,                       //
    sz_sequence_t const *replacements,                    //
    sz_size_t input_bytes, sz_size_t *output_bytes_bound, //
    char const **error_message) {

    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    sz_assert_(replacements != nullptr && "Replacement collection cannot be null");
    sz_assert_(output_bytes_bound != nullptr && "Bound output cannot be null");

    auto &engine = *reinterpret_cast<substrings_backends_t *>(engine_punned);
    auto const replacements_container = sz_sequence_as_cpp_container_t {replacements};
    sz_status_t result = sz_success_k;
    std::visit(
        [&](auto &engine_variant) {
            if (replacements_container.size() != engine_variant.count_needles()) {
                result = propagate_error(sz::status_t::unexpected_dimensions_k, error_message,
                                         "One replacement per needle, no more and no fewer");
                return;
            }

            // The densest rewrite tiles the input with the shortest match there is and swaps each one for
            // the widest replacement there is. The bytes past the last whole match survive verbatim, so they
            // are added rather than dropped - integer division alone under-counts, and a caller sizing an
            // output tape to an under-count would have it overflowed by a rewrite that cannot be refused.
            size_t widest_replacement = 0;
            for (size_t needle_index = 0; needle_index < replacements_container.size(); ++needle_index)
                widest_replacement = sz_max_of_two(widest_replacement, replacements_container[needle_index].size());
            size_t const shortest_match = sz_max_of_two(engine_variant.min_source_match_bytes(), (size_t)1);
            size_t const whole_matches = input_bytes / shortest_match;

            // A dictionary that only ever shrinks still bounds at the input length, never below it.
            *output_bytes_bound = (sz_size_t)sz_max_of_two(
                input_bytes, whole_matches * widest_replacement + input_bytes % shortest_match);
            result = propagate_error(sz::status_t::success_k, error_message);
        },
        engine.variants);
    return result;
}

SZ_API_RUNTIME sz_status_t szs_substrings_replace_u32tape(                           //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                //
    sz_sequence_u32tape_t const *haystacks,                                          //
    szs_substrings_overlap_policy_t overlap_policy,                                  //
    sz_sequence_t const *replacements,                                               //
    sz_ptr_t output_data, sz_size_t output_data_capacity, sz_size_t *output_offsets, //
    sz_size_t *output_bytes_written,                                                 //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_replace_(engine_punned, device_punned, sz_sequence_u32tape_as_cpp_container_t {haystacks},
                                   overlap_policy, replacements, output_data, output_data_capacity, output_offsets,
                                   output_bytes_written, error_message);
}

SZ_API_RUNTIME sz_status_t szs_substrings_replace_u64tape(                           //
    szs_substrings_t engine_punned, szs_device_scope_t device_punned,                //
    sz_sequence_u64tape_t const *haystacks,                                          //
    szs_substrings_overlap_policy_t overlap_policy,                                  //
    sz_sequence_t const *replacements,                                               //
    sz_ptr_t output_data, sz_size_t output_data_capacity, sz_size_t *output_offsets, //
    sz_size_t *output_bytes_written,                                                 //
    char const **error_message) {

    sz_assert_(haystacks != nullptr && "Haystack collection cannot be null");
    return szs_substrings_replace_(engine_punned, device_punned, sz_sequence_u64tape_as_cpp_container_t {haystacks},
                                   overlap_policy, replacements, output_data, output_data_capacity, output_offsets,
                                   output_bytes_written, error_message);
}

SZ_API_RUNTIME void szs_substrings_free(szs_substrings_t engine_punned) {
    sz_assert_(engine_punned != nullptr && "Engine must be initialized");
    auto *engine = reinterpret_cast<substrings_backends_t *>(engine_punned);
    delete engine;
}

#pragma endregion Substrings
}

#endif // SZS_SUBSTRINGS_CUH_
