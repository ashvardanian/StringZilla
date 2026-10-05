/**
 *  @file bench/cross.hpp
 *  @author Ash Vardanian
 *  @date September 29, 2026
 *  @brief Kernel-level benchmarks, shared by the family files and the per-architecture cross files.
 *
 *  Each family's adapters take their kernel by pointer and wrap it into the harness's callables.
 *  The family files run them over the dispatch points, and each `cross_<arch>.cpp` over its
 *  architecture's kernels by name, one section per capability, stress-tested against the serial
 *  kernel of the same operation and logged relative to it, which `cross_serial.cpp` times first.
 *
 *  CPU paths use direct kernels so header-only benchmarks need no compiled dispatch library.
 */
#pragma once
#ifndef STRINGZILLA_BENCH_CROSS_HPP
#define STRINGZILLA_BENCH_CROSS_HPP

#include <bit>     // `std::bit_cast`
#include <cstdint> // `std::uintptr_t`
#include <cstring> // `std::memcpy`

#include <array>         // `std::array`
#include <deque>         // `std::deque`
#include <functional>    // `std::less`
#include <map>           // `std::map`
#include <numeric>       // `std::iota`
#include <optional>      // `std::optional`
#include <stdexcept>     // `std::runtime_error`
#include <string>        // `std::string`
#include <string_view>   // `std::string_view`
#include <unordered_map> // `std::unordered_map`
#include <unordered_set> // `std::unordered_set`
#include <variant>       // `std::variant`
#include <vector>        // `std::vector`

#include <fmt/format.h>

#include "harness.hpp"
#include "substrings.cuh" // `substrings_dictionary_t`, `substrings_counts_from_sz`

namespace ashvardanian::stringzilla::bench {

#pragma region Cross Sections

/** Prints a kernel's @p row against the serial kernel @p serial_name, or keeps it as that one. */
inline void log_kernel(environment_t &env, std::optional<row_t> const &row, std::string_view serial_name) {
    if (!row) return;
    if (row->name == serial_name) {
        print(*row);
        env.serial_rows.insert_or_assign(row->name, *row);
    }
    else if (auto const serial = env.serial_rows.find(serial_name); serial != env.serial_rows.end())
        print(*row, baseline_of(serial->second));
    else print(*row);
}

/** Whether the filter keeps the kernel @p name but dropped the serial row it prints against. */
inline bool needs_serial_row(environment_t const &env, std::string_view name, std::string_view serial_name) {
    return name != serial_name && env.settings.selects(name) && !env.serial_rows.contains(serial_name);
}

/** Keeps @p row, timed under a kept kernel's name, as the serial row @p serial_name, unprinted. */
inline void keep_serial_row(environment_t &env, std::optional<row_t> row, std::string_view serial_name) {
    if (!row) return;
    row->name = serial_name;
    env.serial_rows.insert_or_assign(row->name, std::move(*row));
}

/**
 *  @brief Times the kernel @p name over every token, as @c bench_unary does, stress-tested against
 *      @p serial and logged relative to it, unless it is the serial kernel @p serial_name itself.
 *  @param[in] extras The preprocessing and the check validator, when the kernel needs them.
 */
template <typename serial_type_, typename callable_type_, typename... extras_types_>
void bench_kernel_unary(environment_t &env, corpus_t const &corpus, std::string const &name,
                        std::string const &serial_name, serial_type_ &&serial, callable_type_ &&callable,
                        extras_types_ &&...extras) {
    if (name == serial_name)
        log_kernel(env, bench_unary(env, corpus, name, callable_no_op_t {}, callable, extras...), serial_name);
    else {
        if (needs_serial_row(env, name, serial_name))
            keep_serial_row(env, bench_unary(env, corpus, name, callable_no_op_t {}, serial, extras...), serial_name);
        log_kernel(env, bench_unary(env, corpus, name, serial, callable, extras...), serial_name);
    }
}

/**
 *  @brief Times the kernel @p name over the whole dataset, as @c bench_nullary does, stress-tested
 *      against @p serial and logged relative to it, unless it is the serial kernel @p serial_name.
 *  @param[in] extras The preprocessing and the check validator, when the kernel needs them.
 */
template <typename serial_type_, typename callable_type_, typename... extras_types_>
void bench_kernel_nullary(environment_t &env, corpus_t const &corpus, std::string const &name,
                          std::string const &serial_name, serial_type_ &&serial, callable_type_ &&callable,
                          extras_types_ &&...extras) {
    if (name == serial_name)
        log_kernel(env, bench_nullary(env, corpus, name, callable_no_op_t {}, callable, extras...), serial_name);
    else {
        if (needs_serial_row(env, name, serial_name))
            keep_serial_row(env, bench_nullary(env, corpus, name, callable_no_op_t {}, serial, extras...), serial_name);
        log_kernel(env, bench_nullary(env, corpus, name, serial, callable, extras...), serial_name);
    }
}

#pragma endregion Cross Sections

#pragma region Find

/** Counts the matches of @p matcher_type_ front to back, as @c sz::find_matches_view does, for the
 *  header-only benchmark, which has no C++ layer to borrow the range from. */
template <typename string_type_, typename matcher_type_>
class forward_matches {
    string_type_ haystack_;
    matcher_type_ matcher_;

  public:
    forward_matches(string_type_ haystack, matcher_type_ matcher) noexcept : haystack_(haystack), matcher_(matcher) {}

    std::size_t size() const noexcept {
        std::size_t count = 0;
        for (string_type_ remaining = haystack_;; ++count) {
            std::size_t const position = matcher_(remaining);
            if (position == string_type_::npos) return count;
            remaining.remove_prefix(position + matcher_.skip_length());
        }
    }
};

/** Counts the matches of @p matcher_type_ back to front, as @c sz::rfind_matches_view does. */
template <typename string_type_, typename matcher_type_>
class reverse_matches {
    string_type_ haystack_;
    matcher_type_ matcher_;

  public:
    reverse_matches(string_type_ haystack, matcher_type_ matcher) noexcept : haystack_(haystack), matcher_(matcher) {}

    std::size_t size() const noexcept {
        std::size_t count = 0;
        for (string_type_ remaining = haystack_;; ++count) {
            std::size_t const position = matcher_(remaining);
            if (position == string_type_::npos) return count;
            remaining.remove_suffix(remaining.size() - position - matcher_.needle_length() + matcher_.skip_length());
        }
    }
};

/** Wraps a substring search kernel into something similar to @c sz::matcher_find and compatible
 *  with @c sz::find_matches_view. */
template <sz_kernel_find_t find_func_>
struct matcher_from_sz_find {
    using size_type = std::size_t;
    std::string_view needle_;

    inline matcher_from_sz_find(std::string_view needle = {}) noexcept : needle_(needle) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        sz_cptr_t match_pointer = nullptr;
        find_func_(haystack.data(), haystack.size(), needle_.data(), needle_.size(), &match_pointer, nullptr);
        do_not_optimize(match_pointer);
        if (!match_pointer) return std::string_view::npos; // No match found
        return match_pointer - haystack.data();
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Counts the matches of the @p token_index token, as a needle, across the whole dataset. */
template <template <typename, typename> class range_template_, typename matcher_type_>
auto callable_for_substring_search(corpus_t const &corpus) {
    using matcher_t = matcher_type_;
    using matches_t = range_template_<std::string_view, matcher_t>;
    return [&corpus](std::size_t token_index) -> call_result_t {
        std::string_view haystack = corpus.dataset;
        std::string_view needle = corpus.tokens[token_index];
        matcher_t matcher(needle);
        matches_t matches(haystack, matcher);
        // Drain all matches to ensure the compiler doesn't optimize the search away
        std::size_t count_bytes = haystack.size();
        std::size_t count_matches = matches.size();
        std::size_t count_operations = count_bytes * needle.size();
        do_not_optimize(count_matches);
        return call_result_t {count_bytes, count_matches, count_operations};
    };
}

/** Wraps a byte search kernel into something similar to @c sz::matcher_find and compatible with
 *  @c sz::find_matches_view. */
template <sz_kernel_find_byte_t find_func_>
struct matcher_from_sz_find_byte {
    using size_type = std::size_t;
    char needle_;

    inline matcher_from_sz_find_byte(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        sz_cptr_t match_pointer = nullptr;
        find_func_(haystack.data(), haystack.size(), &needle_, &match_pointer, nullptr);
        do_not_optimize(match_pointer);
        if (!match_pointer) return std::string_view::npos; // No match found
        return match_pointer - haystack.data();
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Counts the spaces, newlines, and nulls in the @p token_index token, as a haystack. */
template <template <typename, typename> class range_template_, typename matcher_type_>
auto callable_for_byte_search(corpus_t const &corpus) {
    using matcher_t = matcher_type_;
    using matches_t = range_template_<std::string_view, matcher_t>;
    return [&corpus](std::size_t token_index) -> call_result_t {
        std::string_view haystack = corpus.tokens[token_index];
        std::size_t count_whitespaces = matches_t(haystack, matcher_t(' ')).size();
        std::size_t count_newlines = matches_t(haystack, matcher_t('\n')).size();
        std::size_t count_nulls = matches_t(haystack, matcher_t(0)).size();
        // As a checksum, mix the counts together
        std::size_t count_matches = count_whitespaces + count_newlines + count_nulls;
        std::size_t count_bytes = haystack.size() * 3; // We've traversed the input 3 times
        do_not_optimize(count_matches);
        return call_result_t {count_bytes, count_matches};
    };
}

/** Wraps a byteset search kernel into something similar to @c sz::matcher_find and compatible
 *  with @c sz::find_matches_view. */
template <sz_kernel_find_byteset_t find_func_>
struct matcher_from_sz_find_byteset {
    using size_type = std::size_t;
    sz_byteset_t needles_;

    matcher_from_sz_find_byteset(std::string_view needles) noexcept {
        sz_byteset_init(&needles_);
        for (char needle : needles) sz_byteset_add(&needles_, needle);
    }
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        sz_cptr_t match_pointer = nullptr;
        find_func_(haystack.data(), haystack.size(), &needles_, &match_pointer, nullptr);
        do_not_optimize(match_pointer);
        if (!match_pointer) return std::string_view::npos; // No match found
        return match_pointer - haystack.data();
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Counts the tabs, HTML specials, and digits in the @p token_index token, as a haystack. */
template <template <typename, typename> class range_template_, typename matcher_type_>
auto callable_for_byteset_search(corpus_t const &corpus) {
    using matcher_t = matcher_type_;
    using matches_t = range_template_<std::string_view, matcher_t>;

    // Built once, outside the timed calls, so the sets never count toward the measurement.
    matcher_t const matcher_tabs(std::string_view("\n\r\v\f", 4));
    matcher_t const matcher_html(std::string_view("</>&'\"=[]", 9));
    matcher_t const matcher_digits(std::string_view("0123456789", 10));
    return [&corpus, matcher_tabs, matcher_html, matcher_digits](std::size_t token_index) -> call_result_t {
        std::string_view haystack = corpus.tokens[token_index];
        std::size_t count_tabs = matches_t(haystack, matcher_tabs).size();
        std::size_t count_html = matches_t(haystack, matcher_html).size();
        std::size_t count_digits = matches_t(haystack, matcher_digits).size();
        // As a checksum, mix the counts together
        std::size_t count_matches = count_tabs + count_html + count_digits;
        std::size_t count_bytes = haystack.size() * 3; // We've traversed the input 3 times
        do_not_optimize(count_matches);
        return call_result_t {count_bytes, count_matches};
    };
}

/** Times one capability's forward and reverse substring search, each corpus word a needle. */
template <sz_kernel_find_t find_, sz_kernel_find_t rfind_>
void bench_find_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.words();
    bench_kernel_unary(env, corpus, fmt::format("sz_find_{}", kit), "sz_find_serial",
                       callable_for_substring_search<forward_matches, matcher_from_sz_find<sz_find_serial>>(corpus),
                       callable_for_substring_search<forward_matches, matcher_from_sz_find<find_>>(corpus));
    bench_kernel_unary(env, corpus, fmt::format("sz_rfind_{}", kit), "sz_rfind_serial",
                       callable_for_substring_search<reverse_matches, matcher_from_sz_find<sz_rfind_serial>>(corpus),
                       callable_for_substring_search<reverse_matches, matcher_from_sz_find<rfind_>>(corpus));
}

/** Times one capability's forward and reverse byte search, each corpus word a haystack. */
template <sz_kernel_find_byte_t find_byte_, sz_kernel_find_byte_t rfind_byte_>
void bench_find_byte_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.words();
    using serial_t = matcher_from_sz_find_byte<sz_find_byte_serial>;
    using rserial_t = matcher_from_sz_find_byte<sz_rfind_byte_serial>;
    bench_kernel_unary(env, corpus, fmt::format("sz_find_byte_{}", kit), "sz_find_byte_serial",
                       callable_for_byte_search<forward_matches, serial_t>(corpus),
                       callable_for_byte_search<forward_matches, matcher_from_sz_find_byte<find_byte_>>(corpus));
    bench_kernel_unary(env, corpus, fmt::format("sz_rfind_byte_{}", kit), "sz_rfind_byte_serial",
                       callable_for_byte_search<reverse_matches, rserial_t>(corpus),
                       callable_for_byte_search<reverse_matches, matcher_from_sz_find_byte<rfind_byte_>>(corpus));
}

/** Times one capability's forward and reverse byteset search, each corpus word a haystack. */
template <sz_kernel_find_byteset_t find_byteset_, sz_kernel_find_byteset_t rfind_byteset_>
void bench_find_byteset_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.words();
    using serial_t = matcher_from_sz_find_byteset<sz_find_byteset_serial>;
    using rserial_t = matcher_from_sz_find_byteset<sz_rfind_byteset_serial>;
    using kernel_t = matcher_from_sz_find_byteset<find_byteset_>;
    using rkernel_t = matcher_from_sz_find_byteset<rfind_byteset_>;
    bench_kernel_unary(env, corpus, fmt::format("sz_find_byteset_{}", kit), "sz_find_byteset_serial",
                       callable_for_byteset_search<forward_matches, serial_t>(corpus),
                       callable_for_byteset_search<forward_matches, kernel_t>(corpus));
    bench_kernel_unary(env, corpus, fmt::format("sz_rfind_byteset_{}", kit), "sz_rfind_byteset_serial",
                       callable_for_byteset_search<reverse_matches, rserial_t>(corpus),
                       callable_for_byteset_search<reverse_matches, rkernel_t>(corpus));
}

#pragma endregion Find

#pragma region Token

/** Wraps a byte-summing kernel into something similar to @c std::accumulate. */
template <sz_kernel_bytesum_t func_>
struct bytesum_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        sz_u64_t bytesum = 0;
        func_(buffer.data(), buffer.size(), &bytesum, nullptr);
        do_not_optimize(bytesum);
        return {buffer.size(), static_cast<check_value_t>(bytesum), 1};
    }
};

/** Wraps a hashing kernel into something similar to @c std::hash. */
template <sz_kernel_hash_t func_>
struct hash_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        sz_u64_t hash = 0;
        func_(buffer.data(), buffer.size(), 0, &hash, nullptr);
        do_not_optimize(hash);
        return {buffer.size(), static_cast<check_value_t>(hash), 1};
    }
};

/** Fixed seed schedule shared by the multi-seed hashing baseline and kernels. */
inline std::array<sz_u64_t, 8> multiway_seeds() noexcept {
    return {0u, 1u, 42u, 314159u, 2654435761u, 11400714819323198485ull, 7u, 8u};
}

/** Hashes one token under every seed in a single multi-seed kernel call. */
template <sz_kernel_hash_multiseed_t func_>
struct hash_multiseed_from_sz {
    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }
    inline call_result_t operator()(std::string_view buffer) const noexcept {
        auto seeds = multiway_seeds();
        decltype(seeds) hashes; // Same std::array type - carries the compile-time seed count.
        func_(buffer.data(), buffer.size(), seeds.data(), seeds.size(), hashes.data(), nullptr);
        sz_u64_t mixed = 0;
        for (sz_u64_t hash : hashes) mixed ^= hash;
        do_not_optimize(mixed);
        return {buffer.size() * seeds.size(), static_cast<check_value_t>(mixed), seeds.size()};
    }
};

/** Wraps hash state initialization, streaming, and folding for streaming benchmarks. */
template <sz_kernel_hash_state_init_t init_, sz_kernel_hash_state_update_t stream_, sz_kernel_hash_state_digest_t fold_>
struct hash_stream_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    call_result_t operator()(std::string_view s) const noexcept {
        sz_hash_state_t state;
        init_(&state, 42, nullptr);
        stream_(&state, s.data(), s.size(), nullptr);
        sz_u64_t hash = 0;
        fold_(&state, &hash, nullptr);
        do_not_optimize(hash);
        return {s.size(), static_cast<check_value_t>(hash), 1};
    }
};

/** Wraps SHA256 state initialization, streaming, and digesting for streaming benchmarks. */
template <sz_kernel_sha256_state_init_t init_, sz_kernel_sha256_state_update_t stream_,
          sz_kernel_sha256_state_digest_t fold_>
struct sha256_stream_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    call_result_t operator()(std::string_view s) const noexcept {
        sz_sha256_state_t state;
        init_(&state, nullptr);
        stream_(&state, s.data(), s.size(), nullptr);
        sz_u8_t digest[STRINGZILLA_SHA256_DIGEST_LENGTH];
        fold_(&state, digest, nullptr);
        // Use first 8 bytes of digest as check value
        sz_u64_t check = 0;
        std::memcpy(&check, digest, sizeof(sz_u64_t));
        do_not_optimize(check);
        return {s.size(), static_cast<check_value_t>(check), 1};
    }
};

/** Number of independent messages driven through one multi-state SHA256 call. */
enum : std::size_t { multistate_lanes_k = 16 };

/**
 *  @brief Lane-length policies, trimming the corpus tokens into a chosen shape.
 *
 *  Lanes in a group advance in lockstep, so the spread of lengths within a group is what the
 *  batched kernels are actually sensitive to - a corpus alone only ever shows whatever spread it
 *  happens to have. Every policy trims rather than extends, so lanes stay inside the tokens the
 *  corpus already owns.
 */
struct sha256_lanes_uniform_t {
    static constexpr char const *name_k = "";
    static inline std::size_t length(std::size_t, std::size_t token_length) noexcept { return token_length; }
};

/** One lane far shorter than the rest, which drops its whole group to the scalar kernel. */
struct sha256_lanes_one_short_t {
    static constexpr char const *name_k = "_one_short";
    static inline std::size_t length(std::size_t lane_index, std::size_t token_length) noexcept {
        return lane_index == 0 ? (token_length < 21 ? token_length : 21) : token_length;
    }
};

/** One lane far longer than the rest, the shape lockstep pays the most for. */
struct sha256_lanes_one_long_t {
    static constexpr char const *name_k = "_one_long";
    static inline std::size_t length(std::size_t lane_index, std::size_t token_length) noexcept {
        return lane_index == 0 ? token_length : token_length / 8;
    }
};

/**
 *  @brief Digests a batch of tokens through one multi-state update and digest.
 *
 *  Update and digest come from the same tier on purpose. Goldmont is legacy-SSE SHA-NI while
 *  Skylake and above are AVX-512, and pairing a legacy-SSE update with a wide digest in one loop
 *  makes every SHA-NI instruction pay an AVX-SSE transition, which reads as a slow kernel rather
 *  than as a mixed measurement. The states start from the serial initializer every tier shares.
 */
template <sz_kernel_sha256_multistate_update_t update_, sz_kernel_sha256_multistate_digest_t digest_, typename lanes_>
struct sha256_multistate_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        sz_string_view_t lanes[multistate_lanes_k];
        sz_sha256_state_t states[multistate_lanes_k];
        sz_u8_t digests[multistate_lanes_k * STRINGZILLA_SHA256_DIGEST_LENGTH];
        std::size_t bytes_passed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            std::string_view const token = corpus.tokens[(token_index + lane_index) % corpus.tokens.size()];
            std::size_t const lane_length = lanes_::length(lane_index, token.size());
            lanes[lane_index].start = token.data();
            lanes[lane_index].length = lane_length;
            sz_sha256_state_init_serial(&states[lane_index], nullptr);
            bytes_passed += lane_length;
        }
        sz_sequence_t texts;
        sz_sequence_from_string_views(lanes, multistate_lanes_k, &texts);
        update_(states, &texts, nullptr);
        digest_(states, multistate_lanes_k, digests, nullptr);
        // Multiplied rather than XOR-ed, so two lanes swapping digests cannot cancel out - lane
        // ordering is exactly what a batched kernel gets wrong.
        sz_u64_t mixed = 0;
        for (std::size_t lane_index = 0; lane_index != multistate_lanes_k; ++lane_index) {
            sz_u64_t lane_word;
            std::memcpy(&lane_word, &digests[lane_index * STRINGZILLA_SHA256_DIGEST_LENGTH], sizeof(sz_u64_t));
            mixed = mixed * 31u + lane_word;
        }
        do_not_optimize(mixed);
        call_result_t result;
        result.bytes_passed = bytes_passed;
        result.check_value = static_cast<check_value_t>(mixed);
        result.operations_count = multistate_lanes_k;
        return result;
    }
};

/** Wraps an equality-checking kernel into something similar to @c std::equal_to.
 *  Assuming that almost any random pair of strings would differ in the very first byte, to make
 *  benchmarks more similar to mixed cases, like Hash Table lookups, where during probing we meet
 *  both differing and equivalent strings. */
template <sz_kernel_equal_t func_>
struct equality_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index], corpus.tokens[corpus.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_bool_t ab = sz_false_k, aa = sz_false_k, bb = sz_false_k, ba = sz_false_k;
        func_(a.data(), b.data(), std::min(a.size(), b.size()), &ab, nullptr);
        func_(a.data(), a.data(), a.size(), &aa, nullptr);
        func_(b.data(), b.data(), b.size(), &bb, nullptr);
        func_(b.data(), a.data(), std::min(a.size(), b.size()), &ba, nullptr);
        std::size_t max_bytes_passed = a.size() + b.size() + std::min(a.size(), b.size());
        check_value_t check_value = ab;
        do_not_optimize(ab);
        do_not_optimize(aa);
        do_not_optimize(bb);
        do_not_optimize(ba);
        return {max_bytes_passed, check_value};
    }
};

/** Wraps an order-checking kernel into something similar to @c std::less.
 *  Assuming that almost any random pair of strings would differ in the very first byte, to make
 *  benchmarks more similar to mixed cases, like Hash Table lookups, where during probing we meet
 *  both differing and equivalent strings. */
template <sz_kernel_order_t func_>
struct ordering_from_sz {

    corpus_t const &corpus;
    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index], corpus.tokens[corpus.tokens.size() - 1 - token_index]);
    }

    inline call_result_t operator()(std::string_view a, std::string_view b) const noexcept {
        sz_ordering_t ab = sz_equal_k, aa = sz_equal_k, bb = sz_equal_k, ba = sz_equal_k;
        func_(a.data(), a.size(), b.data(), b.size(), &ab, nullptr);
        func_(a.data(), a.size(), a.data(), a.size(), &aa, nullptr);
        func_(b.data(), b.size(), b.data(), b.size(), &bb, nullptr);
        func_(b.data(), b.size(), a.data(), a.size(), &ba, nullptr);
        std::size_t max_bytes_passed = 4 * std::min(a.size(), b.size());
        check_value_t check_value = ab + aa * 3 + bb * 9 + ba * 27; // Each can have 3 unique values
        do_not_optimize(ab);
        do_not_optimize(aa);
        do_not_optimize(bb);
        do_not_optimize(ba);
        return {max_bytes_passed, check_value};
    }
};

/** Times one capability's byte sum over every line of the corpus. */
template <sz_kernel_bytesum_t bytesum_>
void bench_bytesum_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    bench_kernel_unary(env, corpus, fmt::format("sz_bytesum_{}", kit), "sz_bytesum_serial",
                       bytesum_from_sz<sz_bytesum_serial> {corpus}, bytesum_from_sz<bytesum_> {corpus});
}

/** Times one capability's single-shot hash over every line of the corpus. */
template <sz_kernel_hash_t hash_>
void bench_hash_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    bench_kernel_unary(env, corpus, fmt::format("sz_hash_{}", kit), "sz_hash_serial",
                       hash_from_sz<sz_hash_serial> {corpus}, hash_from_sz<hash_> {corpus});
}

/** Times one capability's multi-seed hash, every seed of @c multiway_seeds in one call. */
template <sz_kernel_hash_multiseed_t hash_multiseed_>
void bench_hash_multiseed_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    bench_kernel_unary(env, corpus, fmt::format("sz_hash_multiseed_{}", kit), "sz_hash_multiseed_serial",
                       hash_multiseed_from_sz<sz_hash_multiseed_serial> {corpus},
                       hash_multiseed_from_sz<hash_multiseed_> {corpus});
}

/** Times one capability's streaming hash: initialization, one update, and the digest. */
template <sz_kernel_hash_state_init_t init_, sz_kernel_hash_state_update_t update_,
          sz_kernel_hash_state_digest_t digest_>
void bench_hash_stream_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    using serial_t =
        hash_stream_from_sz<sz_hash_state_init_serial, sz_hash_state_update_serial, sz_hash_state_digest_serial>;
    bench_kernel_unary(env, corpus, fmt::format("sz_hash_stream_{}", kit), "sz_hash_stream_serial", serial_t {corpus},
                       hash_stream_from_sz<init_, update_, digest_> {corpus});
}

/** Times one capability's SHA256: initialization, one update, and the digest. */
template <sz_kernel_sha256_state_init_t init_, sz_kernel_sha256_state_update_t update_,
          sz_kernel_sha256_state_digest_t digest_>
void bench_sha256_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    using serial_t = sha256_stream_from_sz<sz_sha256_state_init_serial, sz_sha256_state_update_serial,
                                           sz_sha256_state_digest_serial>;
    bench_kernel_unary(env, corpus, fmt::format("sz_sha256_{}", kit), "sz_sha256_serial", serial_t {corpus},
                       sha256_stream_from_sz<init_, update_, digest_> {corpus});
}

/** Times one capability's multi-state SHA256 against one lane-length shape. */
template <sz_kernel_sha256_multistate_update_t update_, sz_kernel_sha256_multistate_digest_t digest_, typename lanes_>
void bench_sha256_multistate_lanes(environment_t &env, corpus_t const &corpus, std::string_view kit) {
    using serial_t =
        sha256_multistate_from_sz<sz_sha256_multistate_update_serial, sz_sha256_multistate_digest_serial, lanes_>;
    bench_kernel_unary(env, corpus, fmt::format("sz_sha256_multistate_{}{}", kit, lanes_::name_k),
                       fmt::format("sz_sha256_multistate_serial{}", lanes_::name_k), serial_t {corpus},
                       sha256_multistate_from_sz<update_, digest_, lanes_> {corpus});
}

/** Times one capability's multi-state SHA256 over uniform lanes, one short lane, and a long one. */
template <sz_kernel_sha256_multistate_update_t update_, sz_kernel_sha256_multistate_digest_t digest_>
void bench_sha256_multistate_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    bench_sha256_multistate_lanes<update_, digest_, sha256_lanes_uniform_t>(env, corpus, kit);
    bench_sha256_multistate_lanes<update_, digest_, sha256_lanes_one_short_t>(env, corpus, kit);
    bench_sha256_multistate_lanes<update_, digest_, sha256_lanes_one_long_t>(env, corpus, kit);
}

/** Times one capability's equality check over pairs of lines of the corpus. */
template <sz_kernel_equal_t equal_>
void bench_equal_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    bench_kernel_unary(env, corpus, fmt::format("sz_equal_{}", kit), "sz_equal_serial",
                       equality_from_sz<sz_equal_serial> {corpus}, equality_from_sz<equal_> {corpus});
}

/** Times one capability's ordering over pairs of lines of the corpus. */
template <sz_kernel_order_t order_>
void bench_order_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    bench_kernel_unary(env, corpus, fmt::format("sz_order_{}", kit), "sz_order_serial",
                       ordering_from_sz<sz_order_serial> {corpus}, ordering_from_sz<order_> {corpus});
}

#pragma endregion Token

#pragma region Sequence

using strings_t = std::vector<std::string_view>;
using permute_t = std::vector<sz_sorted_idx_t>;

/** Helper function to distill a large @b permute_t object down to one comparable hash integer. */
template <typename entries_type_>
bool is_sorting_permutation(entries_type_ const &entries, permute_t const &permute) {
    return std::is_sorted(permute.begin(), permute.end(),
                          [&](std::size_t i, std::size_t j) { return entries[i] < entries[j]; });
}

/** Helper function to accumulate the total length of all strings in a sequence. */
inline std::size_t accumulate_lengths(strings_t const &strings) {
    return std::accumulate(strings.begin(), strings.end(), (std::size_t)0,
                           [](std::size_t sum, std::string_view const &str) { return sum + str.size(); });
}

/** Reads a string start from a @c strings_t collection. */
inline sz_cptr_t strings_get_start_(void const *handle, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(handle);
    return array[i].data();
}

/** Reads a string length from a @c strings_t collection. */
inline sz_size_t strings_get_length_(void const *handle, sz_size_t i) {
    strings_t const &array = *reinterpret_cast<strings_t const *>(handle);
    return array[i].size();
}

template <sz_kernel_sequence_argsort_t func_>
struct argsort_strings_via_sz {
    strings_t const &input;
    permute_t &output;

    argsort_strings_via_sz(strings_t const &input, permute_t &output) : input(input), output(output) {}
    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);

        // Prepare the sequence structure for the callback.
        sz_sequence_t array;
        array.count = input.size();
        array.handle = &input;
        array.get_start = strings_get_start_;
        array.get_length = strings_get_length_;
        sz_allocator_t allocator;
        if (sz_allocator_init_heap(&allocator) != sz_success_k)
            throw std::runtime_error("The heap allocator could not be initialized.");
        if (func_(&array, 0, sz_false_k, &allocator, output.data(), nullptr) != sz_success_k)
            throw std::runtime_error("The argsort failed.");

        // Prepare stats and hash the permutation to compare with the reference.
        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_sorting_permutation(input, output);
        std::size_t bytes_passed = accumulate_lengths(input);
        return {bytes_passed, checksum, ops_performed};
    }
};

/** Case-fold every token once, so the uncased checksum can validate in folded-byte order without
 *  re-folding on every benchmarked call, which would dominate the measured throughput. */
inline std::vector<std::string> fold_tokens(strings_t const &tokens) {
    std::vector<std::string> folded(tokens.size());
    std::vector<char> scratch;
    for (std::size_t token_index = 0; token_index != tokens.size(); ++token_index) {
        std::string_view const token = tokens[token_index];
        scratch.resize(token.size() * 3 + 4); // worst-case fold expansion (e.g. ß ⇾ ss, ﬀ ⇾ ff)
        sz_size_t folded_length = 0;
        sz_utf8_uncased_fold_serial(token.data(), token.size(), scratch.data(), &folded_length, nullptr);
        folded[token_index].assign(scratch.data(), folded_length);
    }
    return folded;
}

/** Uncased analogue of @c is_sorting_permutation: validates the permutation orders the strings by
 *  their folded forms. UTF-8 byte order matches code-point order, so the folded-byte `<` serves as
 *  the fold key. */
inline bool is_uncased_sorting_permutation(std::vector<std::string> const &folded, permute_t const &permute) {
    return std::is_sorted(permute.begin(), permute.end(),
                          [&](std::size_t i, std::size_t j) { return folded[i] < folded[j]; });
}

template <sz_kernel_sequence_argsort_t func_>
struct argsort_ci_strings_via_sz {
    strings_t const &input;
    std::vector<std::string> const &folded;
    permute_t &output;

    argsort_ci_strings_via_sz(strings_t const &input, std::vector<std::string> const &folded, permute_t &output)
        : input(input), folded(folded), output(output) {}
    call_result_t operator()() const {
        std::iota(output.begin(), output.end(), 0);

        // Prepare the sequence structure for the callback.
        sz_sequence_t array;
        array.count = input.size();
        array.handle = &input;
        array.get_start = strings_get_start_;
        array.get_length = strings_get_length_;
        sz_allocator_t allocator;
        if (sz_allocator_init_heap(&allocator) != sz_success_k)
            throw std::runtime_error("The heap allocator could not be initialized.");
        if (func_(&array, 0, sz_false_k, &allocator, output.data(), nullptr) != sz_success_k)
            throw std::runtime_error("The uncased argsort failed.");

        std::size_t ops_performed = input.size() * std::log2(input.size());
        check_value_t checksum = is_uncased_sorting_permutation(folded, output);
        std::size_t bytes_passed = accumulate_lengths(input);
        return {bytes_passed, checksum, ops_performed};
    }
};

/** Two sets to intersect: every distinct token, and a seeded sample of half as many tokens. */
struct intersect_inputs_t {
    strings_t tokens_a, tokens_b;
    permute_t permute_a, permute_b;

    intersect_inputs_t(corpus_t const &corpus, std::uint32_t seed) {
        std::unordered_set<std::string_view> unique_tokens(corpus.tokens.begin(), corpus.tokens.end());
        tokens_a.assign(unique_tokens.begin(), unique_tokens.end());
        std::mt19937 generator(seed);
        std::sample(unique_tokens.begin(), unique_tokens.end(), //
                    std::back_inserter(tokens_b), corpus.tokens.size() / 2, generator);
        std::size_t const max_tokens_in_intersection = (std::min)(tokens_a.size(), tokens_b.size());
        permute_a.resize(max_tokens_in_intersection), permute_b.resize(max_tokens_in_intersection);
    }
};

template <sz_kernel_sequence_intersect_t func_>
struct intersect_strings_via_sz {
    strings_t const &input_a;
    strings_t const &input_b;
    permute_t &output_a;
    permute_t &output_b;

    explicit intersect_strings_via_sz(intersect_inputs_t &inputs)
        : input_a(inputs.tokens_a), input_b(inputs.tokens_b), output_a(inputs.permute_a), output_b(inputs.permute_b) {}

    call_result_t operator()() const {

        // Prepare the sequence structure for the callback.
        sz_sequence_t array_a, array_b;
        array_a.count = input_a.size();
        array_a.handle = &input_a;
        array_a.get_start = strings_get_start_;
        array_a.get_length = strings_get_length_;
        array_b.count = input_b.size();
        array_b.handle = &input_b;
        array_b.get_start = strings_get_start_;
        array_b.get_length = strings_get_length_;

        sz_size_t intersections = 0;
        sz_allocator_t allocator;
        if (sz_allocator_init_heap(&allocator) != sz_success_k)
            throw std::runtime_error("The heap allocator could not be initialized.");
        if (func_(&array_a, &array_b, &allocator, 0, &intersections, output_a.data(), output_b.data(), nullptr) !=
            sz_success_k)
            throw std::runtime_error("The intersection failed.");

        // Prepare stats
        check_value_t checksum = static_cast<check_value_t>(intersections);
        std::size_t bytes_passed = accumulate_lengths(input_a) + accumulate_lengths(input_b);
        return {bytes_passed, checksum, input_a.size() + input_b.size()};
    }
};

/** Times one capability's argsort of the corpus words, in byte order and in case-folded order. */
template <sz_kernel_sequence_argsort_t argsort_, sz_kernel_sequence_argsort_t argsort_uncased_>
void bench_sequence_argsort_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.words();
    permute_t permute(corpus.tokens.size());
    bench_kernel_nullary(env, corpus, fmt::format("sz_sequence_argsort_{}", kit), "sz_sequence_argsort_serial",
                         argsort_strings_via_sz<sz_sequence_argsort_serial> {corpus.tokens, permute},
                         argsort_strings_via_sz<argsort_> {corpus.tokens, permute});
    std::vector<std::string> const folded = fold_tokens(corpus.tokens);
    bench_kernel_nullary(env, corpus, fmt::format("sz_sequence_argsort_uncased_{}", kit),
                         "sz_sequence_argsort_uncased_serial",
                         argsort_ci_strings_via_sz<sz_sequence_argsort_uncased_serial> {corpus.tokens, folded, permute},
                         argsort_ci_strings_via_sz<argsort_uncased_> {corpus.tokens, folded, permute});
}

/** Times one capability's intersection of every distinct corpus word with a sample of them. */
template <sz_kernel_sequence_intersect_t intersect_>
void bench_sequence_intersect_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.words();
    intersect_inputs_t inputs(corpus, env.settings.seed.value);
    bench_kernel_nullary(env, corpus, fmt::format("sz_sequence_intersect_{}", kit), "sz_sequence_intersect_serial",
                         intersect_strings_via_sz<sz_sequence_intersect_serial> {inputs},
                         intersect_strings_via_sz<intersect_> {inputs});
}

#pragma endregion Sequence

#pragma region Memory

/** Bytes past the dataset the output buffers keep, for the widest shift and unaligned exports. */
constexpr std::size_t max_shift_length = 299;

/** A page-aligned copy of the dataset to write into, with room after it for the widest shift. */
struct dataset_copy_t {
    std::vector<char> storage;

    explicit dataset_copy_t(corpus_t const &corpus) : storage(4096 + corpus.dataset.size() + max_shift_length) {
        std::memcpy(data(), corpus.dataset.data(), corpus.dataset.size());
    }
    dataset_copy_t(dataset_copy_t const &) = delete;
    dataset_copy_t &operator=(dataset_copy_t const &) = delete;

    /** The first page boundary inside @c storage, where the copy starts. */
    char *data() noexcept {
        std::uintptr_t const address = reinterpret_cast<std::uintptr_t>(storage.data());
        return storage.data() + (round_up_to_multiple<4096>(address) - address);
    }
};

/** Wraps a @b memcpy-like kernel into a callable for @c bench_unary. */
template <sz_kernel_copy_t copy_func_, int page_misalignment_ = 0>
struct copy_from_sz {

    corpus_t const &corpus;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - corpus.dataset.data();
        output_offset -= output_offset % sz_default_alignment_k;
        // Ensure unaligned exports if needed
        output_offset += page_misalignment_;
        copy_func_(output + output_offset, slice.data(), slice.size(), nullptr);
        return {slice.size()};
    }
};

/** Wraps a @b memmove-like kernel into a callable for @c bench_unary, shifting forward and back. */
template <sz_kernel_move_t move_func_, int shift_ = 0>
struct move_from_sz {

    corpus_t const &corpus;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - corpus.dataset.data();
        // Shift forward
        move_func_(output + output_offset + shift_, output + output_offset, slice.size(), nullptr);
        // Shift backward to revert the changes
        move_func_(output + output_offset, output + output_offset + shift_, slice.size(), nullptr);
        return {slice.size() * 2};
    }
};

/** Wraps a @b memset-like kernel into a callable for @c bench_unary. */
template <sz_kernel_fill_t fill_func_>
struct fill_from_sz {

    corpus_t const &corpus;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - corpus.dataset.data();
        fill_func_(output + output_offset, slice.size(), slice.front(), nullptr);
        return {slice.size(), static_cast<check_value_t>(slice.front())};
    }
};

/** Wraps a @c std::generate -like kernel into a callable for @c bench_unary. */
template <sz_kernel_fill_random_t fill_func_>
struct fill_random_from_sz {

    corpus_t const &corpus;
    sz_ptr_t output;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - corpus.dataset.data();
        fill_func_(output + output_offset, slice.size(), slice.front(), nullptr);
        char last_random_byte = output[output_offset + slice.size() - 1];
        do_not_optimize(last_random_byte);
        return {slice.size(), static_cast<check_value_t>(last_random_byte)};
    }
};

/** Wraps a lookup-table kernel into something similar to @c std::transform. */
template <sz_kernel_lookup_t lookup_func_>
struct lookup_from_sz {

    corpus_t const &corpus;
    sz_ptr_t output;
    sz_cptr_t lookup_table;

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view slice) const noexcept {
        std::size_t output_offset = slice.data() - corpus.dataset.data();
        lookup_func_(output + output_offset, slice.data(), slice.size(), lookup_table, nullptr);
        return {slice.size(), static_cast<check_value_t>(slice.front())};
    }
};

/** The cyclic rotation of the alphabet every lookup benchmark transforms through. */
inline std::array<unsigned char, 256> rotated_alphabet() noexcept {
    std::array<unsigned char, 256> table {};
    std::iota(table.begin(), table.end(), static_cast<unsigned char>(1)); // The last byte wraps around to 0
    return table;
}

/** Times one capability's copy into cache-line aligned and one-byte shifted output. */
template <sz_kernel_copy_t copy_>
void bench_copy_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();
    log_kernel(env, bench_unary(env, corpus, fmt::format("sz_copy_{}(align)", kit), copy_from_sz<copy_> {corpus, o}),
               "sz_copy_serial(align)");
    log_kernel(env, bench_unary(env, corpus, fmt::format("sz_copy_{}(shift)", kit), copy_from_sz<copy_, 1> {corpus, o}),
               "sz_copy_serial(shift)");
}

/** Times one capability's move, shifting each line forward and back by a byte and a cache line. */
template <sz_kernel_move_t move_>
void bench_move_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();
    log_kernel(env, bench_unary(env, corpus, fmt::format("sz_move_{}(by1)", kit), move_from_sz<move_, 1> {corpus, o}),
               "sz_move_serial(by1)");
    log_kernel(env, bench_unary(env, corpus, fmt::format("sz_move_{}(by64)", kit), move_from_sz<move_, 64> {corpus, o}),
               "sz_move_serial(by64)");
}

/** Times one capability's fill, each line overwritten with its own first byte. */
template <sz_kernel_fill_t fill_>
void bench_fill_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    dataset_copy_t output(corpus);
    log_kernel(env,
               bench_unary(env, corpus, fmt::format("sz_fill_{}", kit), fill_from_sz<fill_> {corpus, output.data()}),
               "sz_fill_serial");
}

/** Times one capability's random fill, each line seeded with its own first byte. */
template <sz_kernel_fill_random_t fill_random_>
void bench_fill_random_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();
    bench_kernel_unary(env, corpus, fmt::format("sz_fill_random_{}", kit), "sz_fill_random_serial",
                       fill_random_from_sz<sz_fill_random_serial> {corpus, o},
                       fill_random_from_sz<fill_random_> {corpus, o});
}

/** Times one capability's lookup, rotating the alphabet of every line by one. */
template <sz_kernel_lookup_t lookup_>
void bench_lookup_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    dataset_copy_t output(corpus);
    std::array<unsigned char, 256> const alphabet = rotated_alphabet();
    log_kernel(
        env,
        bench_unary(env, corpus, fmt::format("sz_lookup_{}", kit),
                    lookup_from_sz<lookup_> {corpus, output.data(), reinterpret_cast<sz_cptr_t>(alphabet.data())}),
        "sz_lookup_serial");
}

#pragma endregion Memory

#pragma region Cipher

/** The message sizes every kernel is measured at, from a short record to a page. */
constexpr std::size_t cipher_message_sizes_[] = {256, 1024, 4096, 16384};

/** Bytes of messages cycled per size, so large that no single message stays in a register file. */
constexpr std::size_t cipher_pool_bytes_ = 1024ull * 1024ull;

/** The chunk sizes the streaming path is measured at, straddling the sixteen-byte block. */
constexpr std::size_t cipher_chunk_sizes_[] = {1, 7, 16, 40};

/** A message pool with a deterministic pattern, since cipher cost never depends on the bytes. */
inline std::vector<char> cipher_pool() {
    std::vector<char> pool(cipher_pool_bytes_ + STRINGZILLA_AES_BLOCK_LENGTH);
    for (std::size_t index = 0; index != pool.size(); ++index) pool[index] = static_cast<char>(index * 31 + 7);
    return pool;
}

/** Counter-mode throughput for one kernel, cycling a pool so each call touches fresh bytes. */
template <sz_kernel_aes256_key_init_t init_, sz_kernel_aes256_ctr_xor_t transform_>
struct ctr_from_sz {

    std::size_t message_bytes;
    std::vector<char> &pool;
    std::vector<char> &target;
    sz_aes256_key_t key;
    sz_u8_t nonce[STRINGZILLA_AES256_NONCE_LENGTH];

    ctr_from_sz(std::size_t configured, std::vector<char> &input, std::vector<char> &output)
        : message_bytes(configured), pool(input), target(output) {
        sz_u8_t secret[STRINGZILLA_AES256_KEY_LENGTH];
        for (std::size_t index = 0; index != STRINGZILLA_AES256_KEY_LENGTH; ++index)
            secret[index] = static_cast<sz_u8_t>(index * 7 + 1);
        for (std::size_t index = 0; index != STRINGZILLA_AES256_NONCE_LENGTH; ++index)
            nonce[index] = static_cast<sz_u8_t>(index * 5 + 2);
        init_(&key, secret, nullptr);
    }

    inline call_result_t operator()(std::size_t call_index) const noexcept {
        std::size_t const messages = cipher_pool_bytes_ / message_bytes;
        std::size_t const offset = (call_index % messages) * message_bytes;
        transform_(&key, nonce, 0, pool.data() + offset, message_bytes, target.data() + offset, nullptr);
        sz_u64_t check = 0;
        std::memcpy(&check, target.data() + offset, sizeof(check));
        do_not_optimize(check);
        call_result_t result;
        result.bytes_passed = message_bytes;
        result.check_value = static_cast<check_value_t>(check);
        result.operations_count = 1;
        return result;
    }
};

/** Authenticated throughput for one kernel, one whole message per call. */
template <sz_kernel_aes256_gcm_key_init_t init_, sz_kernel_aes256_gcm_encrypt_t encrypt_>
struct gcm_from_sz {

    std::size_t message_bytes;
    std::vector<char> &pool;
    std::vector<char> &target;
    sz_aes256_gcm_key_t key;
    sz_u8_t nonce[STRINGZILLA_AES256_NONCE_LENGTH];

    gcm_from_sz(std::size_t configured, std::vector<char> &input, std::vector<char> &output)
        : message_bytes(configured), pool(input), target(output) {
        sz_u8_t secret[STRINGZILLA_AES256_KEY_LENGTH];
        for (std::size_t index = 0; index != STRINGZILLA_AES256_KEY_LENGTH; ++index)
            secret[index] = static_cast<sz_u8_t>(index * 3 + 5);
        for (std::size_t index = 0; index != STRINGZILLA_AES256_NONCE_LENGTH; ++index)
            nonce[index] = static_cast<sz_u8_t>(index + 9);
        init_(&key, secret, nullptr);
    }

    inline call_result_t operator()(std::size_t call_index) const noexcept {
        std::size_t const messages = cipher_pool_bytes_ / message_bytes;
        std::size_t const offset = (call_index % messages) * message_bytes;
        sz_u8_t tag[STRINGZILLA_AES256_TAG_LENGTH];
        encrypt_(&key, nonce, STRINGZILLA_NULL, 0, pool.data() + offset, message_bytes, target.data() + offset, tag,
                 nullptr);
        sz_u64_t check = 0;
        std::memcpy(&check, tag, sizeof(check));
        do_not_optimize(check);
        call_result_t result;
        result.bytes_passed = message_bytes;
        result.check_value = static_cast<check_value_t>(check);
        result.operations_count = 1;
        return result;
    }
};

/**
 *  @brief Seals one message in fixed-size chunks, which is where the partial-block path dominates.
 *
 *  A chunk below the block width leaves bytes staged in the state between calls, so the cost per
 *  byte is the staging rather than the cipher. Sweeping across sixteen shows where the two cross.
 */
template <sz_kernel_aes256_gcm_key_init_t init_, sz_kernel_aes256_gcm_encryptor_init_t begin_,
          sz_kernel_aes256_gcm_encryptor_update_t update_, sz_kernel_aes256_gcm_encryptor_digest_t digest_>
struct gcm_stream_from_sz {

    std::size_t message_bytes;
    std::size_t chunk_bytes;
    std::vector<char> &pool;
    std::vector<char> &target;
    sz_aes256_gcm_key_t key;
    sz_u8_t nonce[STRINGZILLA_AES256_NONCE_LENGTH];

    gcm_stream_from_sz(std::size_t configured, std::size_t chunk, std::vector<char> &input, std::vector<char> &output)
        : message_bytes(configured), chunk_bytes(chunk), pool(input), target(output) {
        sz_u8_t secret[STRINGZILLA_AES256_KEY_LENGTH];
        for (std::size_t index = 0; index != STRINGZILLA_AES256_KEY_LENGTH; ++index)
            secret[index] = static_cast<sz_u8_t>(index * 3 + 5);
        for (std::size_t index = 0; index != STRINGZILLA_AES256_NONCE_LENGTH; ++index)
            nonce[index] = static_cast<sz_u8_t>(index + 9);
        init_(&key, secret, nullptr);
    }

    inline call_result_t operator()(std::size_t call_index) const noexcept {
        std::size_t const messages = cipher_pool_bytes_ / message_bytes;
        std::size_t const offset = (call_index % messages) * message_bytes;
        sz_u8_t tag[STRINGZILLA_AES256_TAG_LENGTH];
        sz_aes256_gcm_encryptor_t encryptor;
        begin_(&encryptor, &key, nonce, nullptr);
        for (std::size_t consumed = 0; consumed < message_bytes; consumed += chunk_bytes) {
            std::size_t const taken = chunk_bytes < message_bytes - consumed ? chunk_bytes : message_bytes - consumed;
            update_(&encryptor, pool.data() + offset + consumed, taken, target.data() + offset + consumed, nullptr);
        }
        digest_(&encryptor, tag, nullptr);
        sz_u64_t check = 0;
        std::memcpy(&check, tag, sizeof(check));
        do_not_optimize(check);
        call_result_t result;
        result.bytes_passed = message_bytes;
        result.check_value = static_cast<check_value_t>(check);
        result.operations_count = 1;
        return result;
    }
};

/** Times one capability's counter mode at every message size. */
template <sz_kernel_aes256_key_init_t key_init_, sz_kernel_aes256_ctr_xor_t ctr_xor_>
void bench_aes256_ctr_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    for (std::size_t message_bytes : cipher_message_sizes_) {
        std::vector<char> pool = cipher_pool();
        std::vector<char> target(pool.size());
        std::string const suffix = ":" + std::to_string(message_bytes);
        using serial_t = ctr_from_sz<sz_aes256_key_init_serial, sz_aes256_ctr_xor_serial>;
        bench_kernel_unary(env, corpus, fmt::format("sz_aes256_ctr_xor_{}{}", kit, suffix),
                           "sz_aes256_ctr_xor_serial" + suffix, serial_t {message_bytes, pool, target},
                           ctr_from_sz<key_init_, ctr_xor_> {message_bytes, pool, target});
    }
}

/** Times one capability's Galois/counter mode at every message size. */
template <sz_kernel_aes256_gcm_key_init_t key_init_, sz_kernel_aes256_gcm_encrypt_t encrypt_>
void bench_aes256_gcm_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    for (std::size_t message_bytes : cipher_message_sizes_) {
        std::vector<char> pool = cipher_pool();
        std::vector<char> target(pool.size());
        std::string const suffix = ":" + std::to_string(message_bytes);
        using serial_t = gcm_from_sz<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encrypt_serial>;
        bench_kernel_unary(env, corpus, fmt::format("sz_aes256_gcm_encrypt_{}{}", kit, suffix),
                           "sz_aes256_gcm_encrypt_serial" + suffix, serial_t {message_bytes, pool, target},
                           gcm_from_sz<key_init_, encrypt_> {message_bytes, pool, target});
    }
}

/** Times one capability's streaming Galois/counter mode over a page, at every chunk size. */
template <sz_kernel_aes256_gcm_key_init_t key_init_, sz_kernel_aes256_gcm_encryptor_init_t begin_,
          sz_kernel_aes256_gcm_encryptor_update_t update_, sz_kernel_aes256_gcm_encryptor_digest_t digest_>
void bench_aes256_gcm_stream_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.lines();
    std::size_t const message_bytes = 4096;
    for (std::size_t chunk_bytes : cipher_chunk_sizes_) {
        std::vector<char> pool = cipher_pool();
        std::vector<char> target(pool.size());
        std::string const suffix = ":chunk" + std::to_string(chunk_bytes);
        using serial_t =
            gcm_stream_from_sz<sz_aes256_gcm_key_init_serial, sz_aes256_gcm_encryptor_init_serial,
                               sz_aes256_gcm_encryptor_update_serial, sz_aes256_gcm_encryptor_digest_serial>;
        bench_kernel_unary(
            env, corpus, fmt::format("sz_aes256_gcm_stream_{}{}", kit, suffix), "sz_aes256_gcm_stream_serial" + suffix,
            serial_t {message_bytes, chunk_bytes, pool, target},
            gcm_stream_from_sz<key_init_, begin_, update_, digest_> {message_bytes, chunk_bytes, pool, target});
    }
}

#pragma endregion Cipher

#pragma region Container

template <typename string_type_, typename other_string_type_>
string_type_ string_cast(other_string_type_ const &other) noexcept {
    return string_type_(other.data(), other.size());
}

/**
 *  @brief Helper function-like object to order string-view convertible objects with StringZilla.
 *  @see Similar to `std::less<std::string_view>`: https://en.cppreference.com/w/cpp/utility/functional/less
 *  @note Unlike the @c sz::less, the structure below supports different hardware backends.
 */
template <sz_kernel_order_t order_>
struct less_from_sz {
    inline bool operator()(std::string_view a, std::string_view b) const noexcept {
        sz_ordering_t ordering = sz_equal_k;
        order_(a.data(), a.size(), b.data(), b.size(), &ordering, nullptr);
        return ordering < 0;
    }
};

/**
 *  @brief Helper function-like object comparing string-view convertible objects with StringZilla.
 *  @see Similar to `std::equal_to<std::string_view>`: https://en.cppreference.com/w/cpp/utility/functional/equal_to
 *  @note Unlike the @c sz::equal_to, the structure below supports different hardware backends.
 */
template <sz_kernel_equal_t equal_>
struct equal_to_from_sz {
    inline bool operator()(std::string_view a, std::string_view b) const noexcept {
        if (a.size() != b.size()) return false;
        sz_bool_t equal = sz_false_k;
        equal_(a.data(), b.data(), b.size(), &equal, nullptr);
        return equal == sz_true_k;
    }
};

/**
 *  @brief Helper function-like object to hash string-view convertible objects with StringZilla.
 *  @see Similar to @c std::hash: https://en.cppreference.com/w/cpp/utility/functional/hash
 *  @note Unlike the @c sz::hash, the structure below supports different hardware backends.
 */
template <sz_kernel_hash_t hash_>
struct hasher_from_sz {
    inline std::size_t operator()(std::string_view str) const noexcept {
        sz_u64_t hash = 0;
        hash_(str.data(), str.size(), 0, &hash, nullptr);
        return hash;
    }
};

template <typename container_type_>
struct callable_for_associative_lookups {

    container_type_ container;
    corpus_t const &corpus;

    inline callable_for_associative_lookups(corpus_t const &corpus) noexcept : corpus(corpus) {}
    void preprocess() {
        using key_type = typename container_type_::key_type;
        for (std::string_view const &key : corpus.tokens) container[string_cast<key_type>(key)]++;
    }

    /** Helper API to produce a delayed construction lambda. */
    inline auto preprocessor() {
        return [this] { preprocess(); };
    }

    /** The actual lookup operation to be benchmarked. */
    call_result_t operator()(std::size_t token_index) const {
        std::string_view key = corpus.tokens[token_index];
        auto counter = container.find(key)->second;
        return {key.size(), static_cast<std::size_t>(counter)};
    }
};

/** Times @c std::map lookups of every word, ordered by one capability's ordering kernel. */
template <sz_kernel_order_t order_>
void bench_map_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.words();
    auto callable = callable_for_associative_lookups<std::map<std::string_view, unsigned, less_from_sz<order_>>>(
        corpus);
    log_kernel(env,
               bench_unary(env, corpus, fmt::format("map<sz_order_{}>::find", kit), callable_no_op_t(), callable,
                           callable.preprocessor()),
               "map<sz_order_serial>::find");
}

/** Times @c std::unordered_map lookups of every word, hashed and compared by capability kernels. */
template <sz_kernel_hash_t hash_, sz_kernel_equal_t equal_>
void bench_unordered_map_kernels(environment_t &env, std::string_view hash_kit, std::string_view equal_kit) {
    corpus_t const &corpus = env.corpora.words();
    auto callable = callable_for_associative_lookups<
        std::unordered_map<std::string_view, unsigned, hasher_from_sz<hash_>, equal_to_from_sz<equal_>>>(corpus);
    log_kernel(
        env,
        bench_unary(env, corpus, fmt::format("unordered_map<sz_hash_{}, sz_equal_{}>::find", hash_kit, equal_kit),
                    callable_no_op_t(), callable, callable.preprocessor()),
        "unordered_map<sz_hash_serial, sz_equal_serial>::find");
}

#pragma endregion Container

#pragma region Levenshtein

/** The query lengths every sweep runs: the slice's median token, and the 1024 bytes whose match
 *  masks fill a 32 KB L1, where the multi-word regime starts. */
inline std::array<std::size_t, 2> levenshtein_query_lengths(corpus_t const &corpus) {
    return {median_token_bytes(corpus), 1024};
}

inline check_value_t levenshtein_check_value(std::span<sz_size_t const> answers) {
    check_value_t accumulators[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::size_t const count = answers.size();
    std::size_t index = 0;
    for (; index + 8 <= count; index += 8)
        for (std::size_t lane = 0; lane != 8; ++lane)
            accumulators[lane] = accumulators[lane] * 31u + (check_value_t)answers[index + lane];
    for (; index != count; ++index) {
        std::size_t const lane = index % 8;
        accumulators[lane] = accumulators[lane] * 31u + (check_value_t)answers[index];
    }
    check_value_t mixed = 0;
    for (std::size_t lane = 0; lane != 8; ++lane) mixed = mixed * 31u + accumulators[lane];
    return mixed;
}

inline check_value_t levenshtein_check_value_rolling_(std::span<sz_size_t const> answers) {
    check_value_t mixed = 0;
    for (sz_size_t const distance : answers) mixed = mixed * 31u + distance;
    return mixed;
}

/** A prepared batch over rolling corpus tokens or a borrowed resident sequence. */
template <typename function_type_>
struct levenshtein_distances_from_sz {
    function_type_ function;
    check_value_t (*check)(std::span<sz_size_t const>);
    std::variant<std::reference_wrapper<corpus_t const>, sz_sequence_t> source;
    std::vector<sz_string_view_t> query_views, views;

    /** Host lengths avoid migrating device tape offsets during the timed readback. */
    std::span<sz_string_view_t const> fixed_views;
    std::optional<std::reference_wrapper<device_backend_t const>> runtime;
    std::optional<device_vector<sz_size_t>> device_distances;
    pinned_vector<sz_size_t> answers;
    sz_levenshtein_engine_t engine {};

    levenshtein_distances_from_sz(auto init, function_type_ function, corpus_t const &corpus, std::size_t query_bytes,
                                  std::size_t candidates, sz_levenshtein_symbol_t symbol)
        : function(function), check(levenshtein_check_value_rolling_), source(std::cref(corpus)), query_views(8),
          views(candidates) {
        for (std::size_t query = 0; query != query_views.size(); ++query) {
            std::string_view const token = corpus.tokens[query % corpus.tokens.size()];
            query_views[query] = {token.data(), std::min(token.size(), query_bytes)};
        }
        prepare_(init, symbol, candidates);
    }

    /** Both borrowed candidate views must outlive the arm. */
    levenshtein_distances_from_sz(auto init, function_type_ function, std::span<sz_string_view_t const> queries,
                                  sz_sequence_t candidates, std::span<sz_string_view_t const> candidate_views,
                                  sz_levenshtein_symbol_t symbol,
                                  std::optional<std::reference_wrapper<device_backend_t const>> runtime = {})
        : function(function), check(levenshtein_check_value), source(candidates),
          query_views(queries.begin(), queries.end()), fixed_views(candidate_views), runtime(runtime) {
        prepare_(init, symbol, candidates.count);
    }

    ~levenshtein_distances_from_sz() noexcept {
        sz_levenshtein_engine_free(&engine, runtime ? runtime->get().stream : nullptr);
    }
    levenshtein_distances_from_sz(levenshtein_distances_from_sz const &) = delete;
    levenshtein_distances_from_sz &operator=(levenshtein_distances_from_sz const &) = delete;

    call_result_t operator()(std::size_t token_index) {
        sz_sequence_t candidates {};
        std::span<sz_string_view_t const> candidate_views = fixed_views;
        if (auto const *rolling = std::get_if<std::reference_wrapper<corpus_t const>>(&source)) {
            corpus_t const &corpus = rolling->get();
            for (std::size_t candidate = 0; candidate != views.size(); ++candidate) {
                std::string_view const token = corpus.tokens[(token_index + 1 + candidate) % corpus.tokens.size()];
                views[candidate] = {token.data(), token.size()};
            }
            sz_sequence_from_string_views(views.data(), views.size(), &candidates);
            candidate_views = views;
        }
        else candidates = std::get<sz_sequence_t>(source);
        sz_stream_t const stream = runtime ? runtime->get().stream : nullptr;
        if (function(&engine, &candidates, runtime ? device_distances->data() : answers.data(), candidates.count,
                     stream) != sz_success_k)
            throw std::runtime_error("The cross-product entry failed.");
        if (runtime) {
            if (copy_device_to_host(*device_distances, std::span<sz_size_t>(answers), runtime->get()) != sz_success_k ||
                sz_stream_synchronize_best(runtime->get().capabilities, stream) != sz_success_k)
                throw std::runtime_error("The answers would not come back.");
        }
        std::size_t bytes = 0, query_symbols = 0;
        for (sz_string_view_t const &view : candidate_views) bytes += view.length;
        for (sz_string_view_t const &view : query_views) query_symbols += view.length;
        return call_result_t(bytes, check(answers), query_symbols * bytes);
    }

  private:
    void prepare_(auto init, sz_levenshtein_symbol_t symbol, std::size_t candidates) {
        sz_allocator_t allocator;
        if (runtime) allocator = runtime->get().unified;
        else if (sz_allocator_init_heap(&allocator) != sz_success_k)
            throw std::runtime_error("The heap allocator could not be initialized.");
        if (candidates && query_views.size() > (std::numeric_limits<std::size_t>::max)() / candidates)
            throw std::bad_alloc();
        std::size_t const count = query_views.size() * candidates;
        sz_stream_t const stream = runtime ? runtime->get().stream : nullptr;
        answers = pinned_vector<sz_size_t>(
            count, 0, pinned_alloc<sz_size_t>(runtime ? runtime->get().pinned : allocator, stream));
        if (runtime) {
            device_distances.emplace(device_alloc<sz_size_t>(runtime->get().device, stream));
            if (device_distances->resize_uninitialized(count) != sz::status_t::success_k)
                throw std::runtime_error("The device would not hold the distances.");
        }
        sz_sequence_t queries {};
        sz_sequence_from_string_views(query_views.data(), query_views.size(), &queries);
        if (init(&engine, &queries, symbol, &allocator, stream) != sz_success_k) {
            sz_levenshtein_engine_free(&engine, stream);
            throw std::runtime_error("The engine could not be prepared.");
        }
    }
};

/** Times one capability's cross-product sweep at every query length, over bytes or over runes, with
 *  each batch prepared by that capability's own init kernel. */
template <sz_kernel_levenshtein_engine_init_t init_, sz_kernel_levenshtein_distances_t distances_>
void bench_levenshtein_distances_kernels(environment_t &env, std::string_view kit, sz_levenshtein_symbol_t symbol) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    std::size_t const candidates = candidates_per_call(env, corpus);
    for (std::size_t const query_bytes : levenshtein_query_lengths(corpus)) {
        std::string const suffix = fmt::format("{}:q{}", symbol == sz_levenshtein_runes_k ? ":utf8" : "", query_bytes);
        bench_kernel_unary(
            env, corpus, fmt::format("sz_levenshtein_distances_{}{}", kit, suffix),
            "sz_levenshtein_distances_serial" + suffix,
            levenshtein_distances_from_sz {sz_levenshtein_engine_init_serial, sz_levenshtein_distances_serial, corpus,
                                           query_bytes, candidates, symbol},
            levenshtein_distances_from_sz {init_, distances_, corpus, query_bytes, candidates, symbol});
    }
}

#pragma endregion Levenshtein

#pragma region Overlap

/**
 *  @brief The window width at which a random query window and a random candidate window collide
 *      about once per query, with H₂ the byte collision entropy of the slice:
 *
 *  @verbatim
 *  width = ⌈log₂(query_bytes × mean candidate bytes) / H₂⌉
 *  @endverbatim
 */
inline std::size_t overlap_width_(corpus_t const &corpus, std::size_t query_bytes) {
    double counts[256] = {};
    for (char const byte : corpus.dataset) counts[static_cast<unsigned char>(byte)] += 1.0;
    double collisions = 0.0;
    for (double const count : counts) collisions += count * count;
    double const total = static_cast<double>(corpus.dataset.size());
    double const collision_entropy = -std::log2(collisions / (total * total));
    std::size_t token_bytes = 0;
    for (std::string_view const token : corpus.tokens) token_bytes += token.size();
    double const mean_candidate_bytes = static_cast<double>(token_bytes) / static_cast<double>(corpus.tokens.size());
    double const width = std::ceil(std::log2(static_cast<double>(query_bytes) * mean_candidate_bytes) /
                                   collision_entropy);
    return width > 1.0 ? static_cast<std::size_t>(width) : 1;
}

/** A query assembled from leading tokens at the width derived from the corpus. */
struct overlap_query_t {

    /** The window width every arm extracts and scores at. */
    std::size_t width;

    /** The dataset's leading tokens, concatenated up to the requested byte count. */
    std::string text;

    overlap_query_t(corpus_t const &corpus, std::size_t query_bytes) : width(overlap_width_(corpus, query_bytes)) {
        for (std::string_view const token : corpus.tokens) {
            if (text.size() >= query_bytes) break;
            text.append(token);
        }
    }
};

struct overlap_corpus_t {
    std::vector<sz_string_view_t> views;
    tape_t candidates;

    overlap_corpus_t(std::span<sz_string_view_t const> views, device_backend_t const &runtime)
        : views(views.begin(), views.end()), candidates(unified_alloc<char>(runtime.unified, runtime.stream)) {
        sz_sequence_t source {};
        sz_sequence_from_string_views(this->views.data(), this->views.size(), &source);
        if (sz::failed(candidates.assign(source))) throw std::runtime_error("Unified memory could not hold the tape.");
        if (sz_stream_synchronize_best(runtime.capabilities, runtime.stream) != sz_success_k)
            throw std::runtime_error("The tape would not reach the device.");
    }
    sz_sequence_t host_candidates() const noexcept {
        sz_sequence_t result {};
        sz_sequence_from_string_views(views.data(), views.size(), &result);
        return result;
    }
    std::size_t bytes() const noexcept {
        std::size_t total = 0;
        for (sz_string_view_t const &view : views) total += view.length;
        return total;
    }
};

inline check_value_t overlap_check_value_bits_(std::span<sz_f32_t const> scores) {
    check_value_t mixed = 0;
    for (sz_f32_t const score : scores) mixed = mixed * 31u + std::bit_cast<sz_u32_t>(score);
    return mixed;
}

inline check_value_t overlap_check_value_quantized_(std::span<sz_f32_t const> scores) {
    check_value_t mixed = 0;
    for (sz_f32_t const score : scores) mixed = mixed * 31u + static_cast<check_value_t>(score * 1048576.0f);
    return mixed;
}

/** The same prepared forest scores rolling corpus tokens or a borrowed resident sequence. */
template <typename function_type_>
struct scores_from_sz {
    function_type_ function;
    check_value_t (*check)(std::span<sz_f32_t const>);
    std::variant<std::reference_wrapper<corpus_t const>, sz_sequence_t> source;
    std::vector<sz_string_view_t> views;

    /** Host lengths keep timed metrics from migrating device tape offsets. */
    std::span<sz_string_view_t const> fixed_views;
    unified_vector<sz_f32_t> scores;
    std::size_t width;
    std::optional<std::reference_wrapper<device_backend_t const>> runtime;
    sz_overlap_engine_t engine {};

    scores_from_sz(auto init, function_type_ function, corpus_t const &corpus, overlap_query_t const &query,
                   std::size_t candidates)
        : function(function), check(overlap_check_value_bits_), source(std::cref(corpus)), views(candidates),
          scores(candidates), width(query.width) {
        prepare_(init, query.text);
    }

    /** The candidate descriptor borrows its tape for the arm's lifetime. */
    scores_from_sz(auto init, function_type_ function, std::string_view query, std::size_t width,
                   sz_sequence_t candidates, std::span<sz_string_view_t const> candidate_views,
                   std::optional<std::reference_wrapper<device_backend_t const>> runtime = {})
        : function(function), check(overlap_check_value_quantized_), source(candidates), fixed_views(candidate_views),
          scores(candidates.count, 0.0f,
                 runtime ? unified_alloc<sz_f32_t>(runtime->get().unified, runtime->get().stream)
                         : unified_alloc<sz_f32_t>()),
          width(width), runtime(runtime) {
        prepare_(init, query);
    }

    ~scores_from_sz() noexcept { sz_overlap_engine_free(&engine, runtime ? runtime->get().stream : nullptr); }
    scores_from_sz(scores_from_sz const &) = delete;
    scores_from_sz &operator=(scores_from_sz const &) = delete;

    call_result_t operator()(std::size_t token_index) {
        sz_sequence_t candidates {};
        std::span<sz_string_view_t const> candidate_views = fixed_views;
        if (auto const *rolling = std::get_if<std::reference_wrapper<corpus_t const>>(&source)) {
            corpus_t const &corpus = rolling->get();
            for (std::size_t candidate = 0; candidate != views.size(); ++candidate) {
                std::string_view const text = corpus.tokens[(token_index + candidate) % corpus.tokens.size()];
                views[candidate] = {text.data(), text.size()};
            }
            sz_sequence_from_string_views(views.data(), views.size(), &candidates);
            candidate_views = views;
        }
        else candidates = std::get<sz_sequence_t>(source);
        sz_stream_t const stream = runtime ? runtime->get().stream : nullptr;
        if (function(&engine, &candidates, scores.data(), scores.size(), 1, stream) != sz_success_k)
            throw std::runtime_error("The engine's round failed.");
        if (runtime && sz_stream_synchronize_best(runtime->get().capabilities, stream) != sz_success_k)
            throw std::runtime_error("The GPU round did not finish.");
        std::size_t bytes = 0, windows = 0;
        for (sz_string_view_t const &view : candidate_views) {
            sz_size_t const length = view.length;
            bytes += length;
            windows += width <= length ? length - width + 1 : 0;
        }
        return call_result_t(bytes, check(scores), windows);
    }

  private:
    void prepare_(auto init, std::string_view query) {
        sz_allocator_t allocator;
        if (runtime) allocator = runtime->get().unified;
        else if (sz_allocator_init_heap(&allocator) != sz_success_k)
            throw std::runtime_error("The heap allocator could not be initialized.");
        sz_string_view_t const view {query.data(), query.size()};
        sz_sequence_t queries {};
        sz_sequence_from_string_views(&view, 1, &queries);
        sz_stream_t const stream = runtime ? runtime->get().stream : nullptr;
        if (init(&engine, &queries, &width, 1, runtime ? std::get<sz_sequence_t>(source).count : 0, &allocator,
                 stream) != sz_success_k) {
            sz_overlap_engine_free(&engine, stream);
            throw std::runtime_error("The query forest could not be prepared.");
        }
    }
};

template <sz_kernel_overlap_engine_init_t init_, sz_kernel_overlap_scores_t scores_>
void bench_overlap_scores_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_lines();
    overlap_query_t const query(corpus, median_token_bytes(corpus));
    std::size_t const candidates = candidates_per_call(env, corpus);
    std::string const suffix = ":w" + std::to_string(query.width);
    bench_kernel_unary(
        env, corpus, fmt::format("sz_overlap_scores_{}{}", kit, suffix), "sz_overlap_scores_serial" + suffix,
        scores_from_sz {sz_overlap_engine_init_serial, sz_overlap_scores_serial, corpus, query, candidates},
        scores_from_sz {init_, scores_, corpus, query, candidates});
}

#pragma endregion Overlap

#pragma region Substrings

/** One vocabulary slice of the corpus, compiled, with the label every row over it carries. */
struct substrings_vocabulary_t {
    std::string label;
    substrings_dictionary_t dictionary;

    substrings_vocabulary_t(environment_t const &env, corpus_t const &corpus, substrings_slice_t slice,
                            sz_substrings_case_sensitivity_t sensitivity, sz_allocator_t const &allocator)
        : label(substrings_label(slice, sensitivity)), dictionary(env, corpus, slice, sensitivity, allocator) {}
};

/**
 *  @brief The vocabulary slices every substrings row runs over: the frequent and the rare cased
 *      words, the frequent words uncased, and substrings sampled from the corpus itself.
 *
 *  Drawn once per corpus, as each draw sorts every word of the corpus.
 */
inline std::deque<substrings_vocabulary_t> substrings_vocabularies(environment_t const &env, corpus_t const &corpus) {
    std::deque<substrings_vocabulary_t> vocabularies;
    std::pair<substrings_slice_t, sz_substrings_case_sensitivity_t> const slices[] = {
        {substrings_slice_t::frequent_k, sz_substrings_cased_k},
        {substrings_slice_t::rare_k, sz_substrings_cased_k},
        {substrings_slice_t::frequent_k, sz_substrings_uncased_k},
        {substrings_slice_t::sampled_k, sz_substrings_cased_k},
    };
    sz_allocator_t allocator;
    if (sz_allocator_init_heap(&allocator) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    for (auto const &[slice, sensitivity] : slices)
        vocabularies.emplace_back(env, corpus, slice, sensitivity, allocator);
    return vocabularies;
}

/** The suffix of a row over @p vocabulary under @p policy. */
inline std::string substrings_cover(substrings_vocabulary_t const &vocabulary, sz_substrings_overlap_policy_t policy) {
    return vocabulary.label + substrings_policy_name(policy);
}

/** Times one capability's counting, reporting, rewriting, and scoring over every vocabulary, under
 *  every overlap policy each verb accepts. */
template <sz_kernel_substrings_engine_init_t init_, sz_kernel_substrings_counts_t counts_,
          sz_kernel_substrings_find_t find_, sz_kernel_substrings_replace_t replace_,
          sz_kernel_substrings_bm25_scores_t bm25_>
void bench_substrings_kernels(environment_t &env, std::string_view kit) {
    sz_allocator_t allocator;
    if (sz_allocator_init_heap(&allocator) != sz_success_k)
        throw std::runtime_error("The heap allocator could not be initialized.");
    corpus_t const &corpus = env.corpora.multilingual_lines();
    substrings_corpus_t const staged(corpus);
    auto const name = [&](char const *verb, std::string_view of, std::string const &cover) {
        return fmt::format("sz_substrings_{}_{}{}", verb, of, cover);
    };
    std::deque<substrings_vocabulary_t> const vocabularies = substrings_vocabularies(env, corpus);
    for (substrings_vocabulary_t const &vocabulary : vocabularies) {
        substrings_dictionary_t const &dictionary = vocabulary.dictionary;
        if (dictionary.needles.empty()) continue;
        for (sz_substrings_overlap_policy_t const policy : substrings_policies_k) {
            substrings_engine_t engine(dictionary, policy, init_, allocator);
            std::string const cover = substrings_cover(vocabulary, policy);
            bench_kernel_unary(
                env, corpus, name("counts", kit, cover), name("counts", "serial", cover),
                substrings_counts_from_sz {sz_substrings_counts_serial, engine, staged, staged.haystacks()},
                substrings_counts_from_sz {counts_, engine, staged, staged.haystacks()});
            bench_kernel_unary(env, corpus, name("find", kit, cover), name("find", "serial", cover),
                               substrings_find_from_sz {sz_substrings_find_serial, engine, staged, staged.haystacks()},
                               substrings_find_from_sz {find_, engine, staged, staged.haystacks()});
        }
        for (sz_substrings_overlap_policy_t const policy : substrings_leftmost_policies_k) {
            substrings_engine_t engine(dictionary, policy, init_, allocator);
            std::string const cover = substrings_cover(vocabulary, policy);
            bench_kernel_unary(
                env, corpus, name("replace", kit, cover), name("replace", "serial", cover),
                substrings_replace_from_sz {sz_substrings_replace_serial, engine, staged, staged.haystacks(),
                                            dictionary.replacements()},
                substrings_replace_from_sz {replace_, engine, staged, staged.haystacks(), dictionary.replacements()});
        }
        substrings_engine_t engine(dictionary, sz_substrings_overlapping_k, init_, allocator);
        bench_kernel_unary(
            env, corpus, name("bm25_scores", kit, vocabulary.label), name("bm25_scores", "serial", vocabulary.label),
            substrings_bm25_from_sz {sz_substrings_bm25_scores_serial, engine, staged, staged.haystacks()},
            substrings_bm25_from_sz {bm25_, engine, staged, staged.haystacks()});
    }
}

#pragma endregion Substrings

#pragma region UTF8 Traverse

/** Counts the codepoints of each token; checksum = codepoint count. */
template <sz_kernel_utf8_count_t func_>
struct utf8_count_from_sz {
    corpus_t const &corpus;
    utf8_count_from_sz(corpus_t const &corpus_) : corpus(corpus_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = corpus.tokens[i];
        sz_size_t count = 0;
        func_(token.data(), token.size(), &count, nullptr);
        do_not_optimize(count);
        return {token.size(), static_cast<check_value_t>(count)};
    }
};

/** Locates the middle codepoint of each token; checksum = byte offset of the located codepoint. */
template <sz_kernel_utf8_seek_t func_>
struct utf8_seek_from_sz {
    corpus_t const &corpus;

    /** The codepoint to locate per token, counted outside the timed call. */
    std::vector<sz_size_t> targets;

    utf8_seek_from_sz(corpus_t const &corpus_) : corpus(corpus_) {
        targets.reserve(corpus.tokens.size());
        for (auto const &token : corpus.tokens) {
            sz_size_t count = 0;
            sz_utf8_count_serial(token.data(), token.size(), &count, nullptr);
            targets.push_back(count / 2);
        }
    }
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = corpus.tokens[i];
        sz_cptr_t located = nullptr;
        func_(token.data(), token.size(), targets[i], &located, nullptr);
        do_not_optimize(located);
        check_value_t offset = located ? static_cast<check_value_t>(located - token.data()) : (check_value_t)-1;
        // Throughput counts the bytes scanned to reach the Nth codepoint, not the whole token.
        std::size_t scanned = located ? static_cast<std::size_t>(located - token.data()) : token.size();
        return {scanned, offset};
    }
};

/** Transcodes each token UTF-8 → UTF-32 chunk by chunk; checksum = number of runes produced. */
template <sz_kernel_utf8_decode_t func_>
struct utf8_unpack_from_sz {
    corpus_t const &corpus;
    mutable std::vector<sz_rune_t> runes;
    utf8_unpack_from_sz(corpus_t const &corpus_) : corpus(corpus_) {
        std::size_t max_token = 1;
        for (auto const &token : corpus.tokens) max_token = std::max(max_token, token.size());
        runes.resize(max_token + 1);
    }
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = corpus.tokens[i];
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        std::size_t produced = 0;
        while (remaining) {
            sz_size_t unpacked = 0, consumed = 0;
            func_(cursor, remaining, runes.data(), runes.size(), &unpacked, &consumed, nullptr);
            produced += unpacked;
            cursor += consumed;
            remaining -= consumed;
        }
        do_not_optimize(runes.data());
        return {token.size(), static_cast<check_value_t>(produced)};
    }
};

/** Times one capability's codepoint counting over the multilingual slice. */
template <sz_kernel_utf8_count_t count_>
void bench_utf8_count_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_count_{}", kit), "sz_utf8_count_serial",
                       utf8_count_from_sz<sz_utf8_count_serial> {corpus}, utf8_count_from_sz<count_> {corpus});
}

/** Times one capability's Nth-codepoint seeking over the multilingual slice. */
template <sz_kernel_utf8_seek_t seek_>
void bench_utf8_seek_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_seek_{}", kit), "sz_utf8_seek_serial",
                       utf8_seek_from_sz<sz_utf8_seek_serial> {corpus}, utf8_seek_from_sz<seek_> {corpus});
}

/** Times one capability's UTF-8 → UTF-32 transcoding over the multilingual slice. */
template <sz_kernel_utf8_decode_t decode_>
void bench_utf8_decode_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_decode_{}", kit), "sz_utf8_decode_serial",
                       utf8_unpack_from_sz<sz_utf8_decode_serial> {corpus}, utf8_unpack_from_sz<decode_> {corpus});
}

#pragma endregion UTF8 Traverse

#pragma region UTF8 Scan

/** Enumerates every match of a codepoint class (newline, whitespace, delimiter) across each token
 *  via the multistep "find boundaries" API, resuming through the whole token in batches of
 *  @c sz_iterators_default_steps_k; the checksum is the total number of matches. */
template <sz_kernel_utf8_tokenizer_t find_func_>
struct utf8_enumerate_delimiters {
    corpus_t const &corpus;
    utf8_enumerate_delimiters(corpus_t const &corpus_) : corpus(corpus_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = corpus.tokens[i];
        sz_cptr_t text = token.data();
        sz_size_t len = token.size();
        sz_size_t offsets[sz_iterators_default_steps_k], lengths[sz_iterators_default_steps_k];
        sz_size_t pos = 0, total = 0;
        while (pos < len) {
            sz_size_t matches = 0, consumed = 0;
            find_func_(text + pos, len - pos, offsets, lengths, sz_iterators_default_steps_k, &matches, &consumed,
                       nullptr);
            total += matches;
            pos += consumed;
        }
        do_not_optimize(total);
        return {token.size(), static_cast<check_value_t>(total)};
    }
};

/** Times one capability's newline enumeration over the multilingual slice. */
template <sz_kernel_utf8_tokenizer_t newlines_>
void bench_utf8_newlines_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_newlines_{}", kit), "sz_utf8_newlines_serial",
                       utf8_enumerate_delimiters<sz_utf8_newlines_serial> {corpus},
                       utf8_enumerate_delimiters<newlines_> {corpus});
}

/** Times one capability's whitespace enumeration over the multilingual slice. */
template <sz_kernel_utf8_tokenizer_t whitespaces_>
void bench_utf8_whitespaces_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_whitespaces_{}", kit), "sz_utf8_whitespaces_serial",
                       utf8_enumerate_delimiters<sz_utf8_whitespaces_serial> {corpus},
                       utf8_enumerate_delimiters<whitespaces_> {corpus});
}

/** Times one capability's delimiter enumeration over the multilingual slice. */
template <sz_kernel_utf8_tokenizer_t delimiters_>
void bench_utf8_delimiters_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_delimiters_{}", kit), "sz_utf8_delimiters_serial",
                       utf8_enumerate_delimiters<sz_utf8_delimiters_serial> {corpus},
                       utf8_enumerate_delimiters<delimiters_> {corpus});
}

#pragma endregion UTF8 Scan

#pragma region UTF8 Segment

/** Segments each token into UAX-29 words, graphemes, sentences, or UAX-14 line breaks forward; the
 *  checksum is the number of segments. */
template <sz_kernel_utf8_segmenter_t func_>
struct utf8_word_forward_from_sz {
    corpus_t const &corpus;
    utf8_word_forward_from_sz(corpus_t const &corpus_) : corpus(corpus_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = corpus.tokens[i];
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        sz_size_t lengths[16];
        std::size_t words = 0;
        while (remaining) {
            sz_size_t produced = 0, consumed = 0;
            func_(cursor, remaining, lengths, 16, &produced, nullptr);
            words += static_cast<std::size_t>(produced);
            // Only a full batch leaves a suffix to segment.
            if (produced != 16) break;
            for (sz_size_t index = 0; index != produced; ++index) consumed += lengths[index];
            cursor += consumed;
            remaining -= consumed;
        }
        do_not_optimize(words);
        return {token.size(), static_cast<check_value_t>(words)};
    }
};

/** Times one capability's UAX-29 word segmentation over the multilingual slice. */
template <sz_kernel_utf8_segmenter_t wordbreaks_>
void bench_utf8_wordbreaks_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_wordbreaks_{}", kit), "sz_utf8_wordbreaks_serial",
                       utf8_word_forward_from_sz<sz_utf8_wordbreaks_serial> {corpus},
                       utf8_word_forward_from_sz<wordbreaks_> {corpus});
}

/** Times one capability's UAX-29 grapheme-cluster segmentation over the multilingual slice. */
template <sz_kernel_utf8_segmenter_t graphemes_>
void bench_utf8_graphemes_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_graphemes_{}", kit), "sz_utf8_graphemes_serial",
                       utf8_word_forward_from_sz<sz_utf8_graphemes_serial> {corpus},
                       utf8_word_forward_from_sz<graphemes_> {corpus});
}

/** Times one capability's UAX-29 sentence segmentation over the multilingual slice. */
template <sz_kernel_utf8_segmenter_t sentences_>
void bench_utf8_sentences_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_sentences_{}", kit), "sz_utf8_sentences_serial",
                       utf8_word_forward_from_sz<sz_utf8_sentences_serial> {corpus},
                       utf8_word_forward_from_sz<sentences_> {corpus});
}

/** Times one capability's UAX-14 line-break segmentation over the multilingual slice. */
template <sz_kernel_utf8_segmenter_t linebreaks_>
void bench_utf8_linebreaks_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_linebreaks_{}", kit), "sz_utf8_linebreaks_serial",
                       utf8_word_forward_from_sz<sz_utf8_linebreaks_serial> {corpus},
                       utf8_word_forward_from_sz<linebreaks_> {corpus});
}

#pragma endregion UTF8 Segment

#pragma region UTF8 Norm

/** The checksum a transforming adapter validates its output with: the library's fastest, as the
 *  timed call includes it, or the serial one in header-only builds, which have no kernel finder. */
inline sz_kernel_bytesum_t output_checksum_kernel() noexcept {
    sz_kernel_punned_t punned = nullptr;
    sz_capability_t capability = 0;
    sz_find_kernel_punned(sz_kernel_bytesum_k, sz::default_capabilities(), &punned, &capability);
    return punned ? reinterpret_cast<sz_kernel_bytesum_t>(punned) : &sz_bytesum_serial;
}

/** Wraps a hardware-specific UTF-8 normalization backend (transforms to NFC). */
template <sz_kernel_utf8_norm_t func_>
struct utf8_norm_from_sz {

    corpus_t const &corpus;
    mutable std::vector<char> output_buffer;
    sz_kernel_bytesum_t checksum_ = output_checksum_kernel();

    utf8_norm_from_sz(corpus_t const &corpus_) : corpus(corpus_) {
        // Pre-allocate worst-case buffer: 18x input size for the worst single-codepoint
        // compatibility decomposition (see `sz_utf8_norm_best` buffer-sizing docs).
        std::size_t max_token_size = 0;
        for (auto const &token : corpus.tokens) max_token_size = std::max(max_token_size, token.size());
        output_buffer.resize(max_token_size * 18 + 64); // Extra padding for safety
    }

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        // Ensure buffer is large enough
        if (output_buffer.size() < buffer.size() * 18) output_buffer.resize(buffer.size() * 18 + 64);

        sz_size_t result_length = 0;
        func_(buffer.data(), buffer.size(), sz_normal_form_nfc_k, output_buffer.data(), &result_length, nullptr);
        do_not_optimize(output_buffer.data());
        do_not_optimize(result_length);

        sz_u64_t checksum = 0;
        checksum_(output_buffer.data(), result_length, &checksum, nullptr);
        return {buffer.size(), static_cast<check_value_t>(checksum)};
    }
};

/** Wraps a hardware-specific UTF-8 normalization-violation backend (quick-check scan for NFC). */
template <sz_kernel_utf8_find_denormalized_t func_>
struct utf8_find_denormalized_from_sz {

    corpus_t const &corpus;

    utf8_find_denormalized_from_sz(corpus_t const &corpus_) : corpus(corpus_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        sz_cptr_t violation = nullptr;
        func_(buffer.data(), buffer.size(), sz_normal_form_nfc_k, &violation, nullptr);
        do_not_optimize(violation);

        // Encode the violation offset (or "no violation") as a validation checksum.
        check_value_t offset = violation ? static_cast<check_value_t>(violation - buffer.data())
                                         : static_cast<check_value_t>(buffer.size());
        return {buffer.size(), offset};
    }
};

/** Times one capability's NFC normalization over the multilingual slice. */
template <sz_kernel_utf8_norm_t norm_>
void bench_utf8_norm_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_norm_{}", kit), "sz_utf8_norm_serial",
                       utf8_norm_from_sz<sz_utf8_norm_serial> {corpus}, utf8_norm_from_sz<norm_> {corpus});
}

/** Times one capability's NFC quick-check scan over the multilingual slice. */
template <sz_kernel_utf8_find_denormalized_t find_denormalized_>
void bench_utf8_find_denormalized_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_find_denormalized_{}", kit),
                       "sz_utf8_find_denormalized_serial",
                       utf8_find_denormalized_from_sz<sz_utf8_find_denormalized_serial> {corpus},
                       utf8_find_denormalized_from_sz<find_denormalized_> {corpus});
}

#pragma endregion UTF8 Norm

#pragma region UTF8 Uncased

/** Wraps a hardware-specific UTF-8 case folding backend. */
template <sz_kernel_utf8_uncased_fold_t func_>
struct utf8_uncased_fold_from_sz {

    corpus_t const &corpus;
    mutable std::vector<char> output_buffer;
    sz_kernel_bytesum_t checksum_ = output_checksum_kernel();

    utf8_uncased_fold_from_sz(corpus_t const &corpus_) : corpus(corpus_) {
        // Pre-allocate worst-case buffer: 3x input size for worst-case expansion
        std::size_t max_token_size = 0;
        for (auto const &token : corpus.tokens) max_token_size = std::max(max_token_size, token.size());
        output_buffer.resize(max_token_size * 3 + 64); // Extra padding for safety
    }

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        return operator()(corpus.tokens[token_index]);
    }

    inline call_result_t operator()(std::string_view buffer) const noexcept {
        // Ensure buffer is large enough
        if (output_buffer.size() < buffer.size() * 3) output_buffer.resize(buffer.size() * 3 + 64);

        sz_size_t result_length = 0;
        func_(buffer.data(), buffer.size(), output_buffer.data(), &result_length, nullptr);
        do_not_optimize(output_buffer.data());
        do_not_optimize(result_length);

        sz_u64_t checksum = 0;
        checksum_(output_buffer.data(), result_length, &checksum, nullptr);
        return {buffer.size(), static_cast<check_value_t>(checksum)};
    }
};

/** Wraps a hardware-specific UTF-8 uncased find backend. */
template <sz_kernel_utf8_uncased_search_t func_>
struct utf8_uncased_search_from_sz {

    corpus_t const &corpus;

    utf8_uncased_search_from_sz(corpus_t const &corpus_) : corpus(corpus_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        std::string_view haystack = corpus.dataset;
        std::string_view needle = corpus.tokens[token_index];
        return operator()(haystack, needle);
    }

    inline call_result_t operator()(std::string_view haystack, std::string_view needle) const noexcept {
        sz_size_t match_length = 0;
        std::size_t count_matches = 0;
        sz_cptr_t h = haystack.data();
        sz_size_t haystack_length = haystack.size();
        sz_utf8_uncased_needle_t prepared;
        sz_utf8_uncased_needle_init_serial(needle.data(), needle.size(), &prepared, nullptr);

        // Count all uncased matches
        while (haystack_length >= needle.size()) {
            sz_cptr_t match = nullptr;
            func_(h, haystack_length, &prepared, &match, &match_length, nullptr);
            if (!match) break;
            ++count_matches;
            // Move past the match
            std::size_t offset = (match - h) + (match_length ? match_length : 1);
            h += offset;
            haystack_length -= offset;
        }

        do_not_optimize(count_matches);
        // Operations = haystack_bytes * needle_bytes (worst case comparisons)
        std::size_t count_operations = haystack.size() * needle.size();
        return {haystack.size(), static_cast<check_value_t>(count_matches), count_operations};
    }
};

/** Wraps a hardware-specific UTF-8 uncased ordering backend. */
template <sz_kernel_utf8_uncased_order_t func_>
struct utf8_uncased_order_from_sz {

    corpus_t const &corpus;

    utf8_uncased_order_from_sz(corpus_t const &corpus_) : corpus(corpus_) {}

    inline call_result_t operator()(std::size_t token_index) const noexcept {
        std::string_view a = corpus.tokens[token_index];
        std::string_view b = corpus.tokens[(token_index + 1) % corpus.tokens.size()];
        sz_ordering_t ordering = sz_equal_k;
        func_(a.data(), a.size(), b.data(), b.size(), &ordering, nullptr);
        do_not_optimize(ordering);
        // Worst case is a full case-insensitive scan of both operands.
        std::size_t count_operations = a.size() + b.size();
        // Fold the {-1,0,+1} verdict into a non-negative check value, which SIMD must match.
        return {a.size() + b.size(), static_cast<check_value_t>(ordering + 1), count_operations};
    }
};

/** Times one capability's case folding over the multilingual slice. */
template <sz_kernel_utf8_uncased_fold_t fold_>
void bench_utf8_uncased_fold_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_uncased_fold_{}", kit), "sz_utf8_uncased_fold_serial",
                       utf8_uncased_fold_from_sz<sz_utf8_uncased_fold_serial> {corpus},
                       utf8_uncased_fold_from_sz<fold_> {corpus});
}

/** Times one capability's uncased substring search, each line a needle in the whole slice. */
template <sz_kernel_utf8_uncased_search_t search_>
void bench_utf8_uncased_search_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_uncased_search_{}", kit), "sz_utf8_uncased_search_serial",
                       utf8_uncased_search_from_sz<sz_utf8_uncased_search_serial> {corpus},
                       utf8_uncased_search_from_sz<search_> {corpus});
}

/** Times one capability's uncased ordering of neighboring lines of the multilingual slice. */
template <sz_kernel_utf8_uncased_order_t order_>
void bench_utf8_uncased_order_kernels(environment_t &env, std::string_view kit) {
    corpus_t const &corpus = env.corpora.multilingual_slice();
    bench_kernel_unary(env, corpus, fmt::format("sz_utf8_uncased_order_{}", kit), "sz_utf8_uncased_order_serial",
                       utf8_uncased_order_from_sz<sz_utf8_uncased_order_serial> {corpus},
                       utf8_uncased_order_from_sz<order_> {corpus});
}

#pragma endregion UTF8 Uncased

} // namespace ashvardanian::stringzilla::bench

#endif // STRINGZILLA_BENCH_CROSS_HPP
