/**
 *  @file bench/find.cpp
 *  @author Ash Vardanian
 *  @date January 4, 2024
 *  @brief Benchmarks for bidirectional string search operations.
 *
 *  Times the search dispatch points over the English words, against LibC, POSIX, and the STL.
 *  Every capability's kernels are timed against the serial ones by the `cross_<arch>.cpp` files,
 *  through the adapters in `cross.hpp`.
 *
 *  Memory-bound: substring search is bandwidth-limited, so it reads the whole file by default and a
 *  larger haystack measures throughput truer; shrink the read with @c STRINGWARS_BYTES only when it
 *  has to be smaller.
 *
 *  Benchmarks include:
 *  - Substring search: find all inclusions of a token in the dataset - @b find & @b rfind.
 *  - Byte search: find a byte value in each word, line, or file - @b find_byte & @b rfind_byte.
 *  - Byteset search: any byte of a set in each line or file - @b find_byteset & @b rfind_byteset.
 *
 *  For substring search, the number of operations per second are reported as the number of
 *  character-level comparisons happening in the worst case in the naive algorithm, meaning O(N*M)
 *  for N characters in the haystack and M in the needle. In byteset search, the number of
 *  operations per second is computed the same way and the following character sets are tested
 *  against each scanned token:
 *
 *  - "\n\r\v\f": 4 tabs
 *  - "</>&'\"=[]": 9 html
 *  - "0123456789": 10 digits
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_cpu_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=words STRINGWARS_FILTER=find build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGZILLA_STRESS=1 STRINGZILLA_STRESS_TIME_LIMIT=120s STRINGZILLA_STRESS_DIR=logs
 *  build_release/stringzilla_cpu_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. This file is the sibling of `sequence.cpp`, `token.cpp`, and `memory.cpp`.
 */
#include <functional> // `std::boyer_moore_searcher`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

#pragma region Substring Search

/** Wraps the LibC functionality for finding the next occurrence of a NULL-terminated string into
 *  something similar to @c sz::matcher_find and compatible with @c sz::find_matches_view.
 *  The @p needle must be followed by a NUL byte. */
struct matcher_strstr_t {
    using size_type = std::size_t;
    std::string_view needle_;

    inline matcher_strstr_t(std::string_view needle = {}) noexcept : needle_(needle) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match_pointer = (char *)strstr(haystack.data(), needle_.data());
        do_not_optimize(match_pointer);
        if (!match_pointer) return std::string_view::npos; // No match found
        return (size_type)(match_pointer - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Like @c callable_for_substring_search, but hands LibC NUL-terminated copies of the tokens,
 *  made once, so the timed calls never allocate. */
auto callable_for_strstr(corpus_t const &corpus) {
    std::vector<std::string> needles(corpus.tokens.begin(), corpus.tokens.end());
    return [&corpus, needles = std::move(needles)](std::size_t token_index) -> call_result_t {
        std::string_view haystack = corpus.dataset;
        std::string_view needle = needles[token_index];
        sz::find_matches_view<std::string_view, matcher_strstr_t> matches(haystack, matcher_strstr_t(needle));
        std::size_t count_matches = matches.size();
        do_not_optimize(count_matches);
        return call_result_t {haystack.size(), count_matches, haystack.size() * needle.size()};
    };
}

#if defined(_GNU_SOURCE)

/** Wraps the LibC functionality for finding the next occurrence of a byte-string in a buffer into
 *  something similar to @c sz::matcher_find and compatible with @c sz::find_matches_view. */
struct matcher_memmem_t {
    using size_type = std::size_t;
    std::string_view needle_;

    inline matcher_memmem_t(std::string_view needle = {}) noexcept : needle_(needle) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match_pointer = (char *)memmem(haystack.data(), haystack.size(), needle_.data(), needle_.size());
        do_not_optimize(match_pointer);
        if (!match_pointer) return std::string_view::npos; // No match found
        return (size_type)(match_pointer - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};
#endif

#if __cpp_lib_boyer_moore_searcher

/**
 *  @brief Wraps the C++20 @b Boyer-Moore algorithms for finding the next occurrence of a string
 *      into something similar to @c sz::matcher_find and compatible with @c sz::find_matches_view.
 *  @tparam searcher_type_ Can be @c std::boyer_moore_searcher or
 *      @c std::boyer_moore_horspool_searcher. Both should be instantiated with the
 *      @c std::string_view::const_iterator type.
 */
template <typename searcher_type_>
struct matcher_from_std_search {
    using size_type = std::size_t;
    std::string_view needle_;
    searcher_type_ searcher_;

    inline matcher_from_std_search(std::string_view needle = {}) noexcept
        : needle_(needle), searcher_(needle.begin(), needle.end()) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match = std::search(haystack.begin(), haystack.end(), searcher_);
        do_not_optimize(match);
        if (match == haystack.end()) return std::string_view::npos; // No match found
        return (size_type)(match - haystack.begin());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

template <typename searcher_type_>
struct rmatcher_from_std_search {
    using size_type = std::size_t;
    std::string_view needle_;
    searcher_type_ searcher_;

    inline rmatcher_from_std_search(std::string_view needle = {}) noexcept
        : needle_(needle), searcher_(needle.rbegin(), needle.rend()) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match = std::search(haystack.rbegin(), haystack.rend(), searcher_);
        do_not_optimize(match);
        if (match == haystack.rend()) return std::string_view::npos; // No match found
        auto offset_from_end = match - haystack.rbegin();
        auto offset_from_start = haystack.size() - offset_from_end - needle_.size();
        return (size_type)offset_from_start;
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

#endif

/** Find all inclusions of each given token in the dataset, using various search backends. */
void bench_substring_search(environment_t const &env, corpus_t const &corpus) {

    // The "check value" for normal and reverse search is the same - simply the number of matches.
    auto base_call = callable_for_substring_search<sz::find_matches_view, matcher_from_sz_find<cpu_best<sz_find_best>>>(
        corpus);
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_find_best", base_call);
    print(base);
    std::optional<row_t> const base_reverse = bench_unary(
        env, corpus, "sz_rfind_best",
        callable_for_substring_search<sz::rfind_matches_view, matcher_from_sz_find<cpu_best<sz_rfind_best>>>(corpus));
    print(base_reverse);

    // Include LibC functionality
    // ! Despite taking string views, these functions assume null-terminated strings.
    print(bench_unary(env, corpus, "find<std::strstr>", base_call, //
                      callable_for_strstr(corpus)),
          baseline_of(base));

    // Include POSIX functionality
#if defined(_GNU_SOURCE)
    print(bench_unary(env, corpus, "find<memmem>", base_call, //
                      callable_for_substring_search<sz::find_matches_view, matcher_memmem_t>(corpus)),
          baseline_of(base));
#endif

    // Include STL functionality
#if __cpp_lib_boyer_moore_searcher
    using matcher_bm_t = matcher_from_std_search<std::boyer_moore_searcher<std::string_view::const_iterator>>;
    using matcher_bmh_t = matcher_from_std_search<std::boyer_moore_horspool_searcher<std::string_view::const_iterator>>;
    using rmatcher_bm_t = rmatcher_from_std_search<std::boyer_moore_searcher<std::string_view::const_reverse_iterator>>;
    using rmatcher_bmh_t =
        rmatcher_from_std_search<std::boyer_moore_horspool_searcher<std::string_view::const_reverse_iterator>>;
    print(bench_unary(env, corpus, "find<std::boyer_moore>", base_call,
                      callable_for_substring_search<sz::find_matches_view, matcher_bm_t>(corpus)),
          baseline_of(base));
    print(bench_unary(env, corpus, "rfind<std::boyer_moore>", base_call,
                      callable_for_substring_search<sz::rfind_matches_view, rmatcher_bm_t>(corpus)),
          baseline_of(base_reverse));
    print(bench_unary(env, corpus, "find<std::boyer_moore_horspool>", base_call,
                      callable_for_substring_search<sz::find_matches_view, matcher_bmh_t>(corpus)),
          baseline_of(base));
    print(bench_unary(env, corpus, "rfind<std::boyer_moore_horspool>", base_call,
                      callable_for_substring_search<sz::rfind_matches_view, rmatcher_bmh_t>(corpus)),
          baseline_of(base_reverse));
#endif
}

#pragma endregion Substring Search

#pragma region Byte Search

/** Wraps the LibC functionality for finding the next occurrence of a NULL-terminated string into
 *  something similar to @c sz::matcher_find and compatible with @c sz::find_matches_view. */
struct matcher_strchr_t {
    using size_type = std::size_t;
    char needle_;

    inline matcher_strchr_t(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match_pointer = (char *)strchr(haystack.data(), needle_);
        do_not_optimize(match_pointer);
        // Tokens aren't NUL-terminated, so `strchr` can run on into the following tokens
        if (!match_pointer || match_pointer >= haystack.data() + haystack.size()) return std::string_view::npos;
        return (size_type)(match_pointer - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Wraps the LibC functionality for finding the next occurrence of a byte-string in a buffer into
 *  something similar to @c sz::matcher_find and compatible with @c sz::find_matches_view. */
struct matcher_memchr_t {
    using size_type = std::size_t;
    char needle_;

    inline matcher_memchr_t(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match_pointer = (char *)std::memchr(haystack.data(), needle_, haystack.size());
        do_not_optimize(match_pointer);
        if (!match_pointer) return std::string_view::npos; // No match found
        return (size_type)(match_pointer - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Wraps the @c std::find algorithms for finding the next occurrence of a string into something
 *  similar to @c sz::matcher_find and compatible with @c sz::find_matches_view. */
struct matcher_from_std_find {
    using size_type = std::size_t;
    char needle_;

    inline matcher_from_std_find(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match = std::find(haystack.begin(), haystack.end(), needle_);
        do_not_optimize(match);
        if (match == haystack.end()) return std::string_view::npos;
        return (size_type)(match - haystack.begin());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief Find all inclusions of a byte value in each token: a word, a line, or the whole file.
 *  @warning Notice, the roles differ from @c bench_substring_search: each individual token is now
 *      treated as a haystack.
 */
void bench_byte_search(environment_t const &env, corpus_t const &corpus) {
    // The "check value" for normal and reverse search is the same - simply the number of matches.
    auto base_call =
        callable_for_byte_search<sz::find_matches_view, matcher_from_sz_find_byte<cpu_best<sz_find_byte_best>>>(corpus);
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_find_byte_best", base_call);
    print(base);
    print(bench_unary(
        env, corpus, "sz_rfind_byte_best",
        callable_for_byte_search<sz::rfind_matches_view, matcher_from_sz_find_byte<cpu_best<sz_rfind_byte_best>>>(
            corpus)));

    // Include LibC functionality
    print(bench_unary(env, corpus, "find_byte<std::strchr>", base_call, //
                      callable_for_byte_search<sz::find_matches_view, matcher_strchr_t>(corpus)),
          baseline_of(base));
    print(bench_unary(env, corpus, "find_byte<std::memchr>", base_call, //
                      callable_for_byte_search<sz::find_matches_view, matcher_memchr_t>(corpus)),
          baseline_of(base));

    // Include STL functionality
    print(bench_unary(env, corpus, "find_byte<std::find>", base_call, //
                      callable_for_byte_search<sz::find_matches_view, matcher_from_std_find>(corpus)),
          baseline_of(base));
}

#pragma endregion Byte Search

#pragma region Byteset Search

/** Wraps the LibC functionality for finding the next occurrence of a NULL-terminated string into
 *  something similar to @c sz::matcher_find and compatible with @c sz::find_matches_view. */
struct matcher_strcspn_t {
    using size_type = std::size_t;
    std::string_view needles_;

    inline matcher_strcspn_t(std::string_view needles) noexcept : needles_(needles) {}
    inline size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match = strcspn(haystack.data(), needles_.data());
        do_not_optimize(match);
        // Tokens aren't NUL-terminated, so `strcspn` can run on into the following tokens
        if (match >= haystack.size()) return std::string_view::npos;
        return match;
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Wraps the @c std::string_view::find_first_of algorithms for finding the next occurrence of a
 *  string into something similar to @c sz::matcher_find and compatible with the
 *  @c sz::find_matches_view range. */
struct matcher_std_string_first_of_t {
    using size_type = std::size_t;
    std::string_view needles_;

    inline matcher_std_string_first_of_t(std::string_view needles) noexcept : needles_(needles) {}
    inline size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept { return haystack.find_first_of(needles_); }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/** Wraps the @c std::string_view::find_last_of algorithms for finding the next occurrence of a
 *  string into something similar to @c sz::matcher_rfind and compatible with the
 *  @c sz::rfind_matches_view range. */
struct matcher_std_string_last_of_t {
    using size_type = std::size_t;
    std::string_view needles_;

    inline matcher_std_string_last_of_t(std::string_view needles) noexcept : needles_(needles) {}
    inline size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept { return haystack.find_last_of(needles_); }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief Find all inclusions of any byte from a set in each token: a word, a line, or a file.
 *  @warning Notice, the roles differ from @c bench_substring_search: each individual token is now
 *      treated as a haystack.
 */
void bench_byteset_search(environment_t const &env, corpus_t const &corpus) {

    // The "check value" for normal and reverse search is the same - simply the number of matches.
    using best_t = matcher_from_sz_find_byteset<cpu_best<sz_find_byteset_best>>;
    using rbest_t = matcher_from_sz_find_byteset<cpu_best<sz_rfind_byteset_best>>;
    auto base_call = callable_for_byteset_search<sz::find_matches_view, best_t>(corpus);
    std::optional<row_t> const base = bench_unary(env, corpus, "sz_find_byteset_best", base_call);
    print(base);
    std::optional<row_t> const base_reverse = bench_unary(
        env, corpus, "sz_rfind_byteset_best", callable_for_byteset_search<sz::rfind_matches_view, rbest_t>(corpus));
    print(base_reverse);

    // Include LibC functionality
    print(bench_unary(env, corpus, "find_byteset<std::strcspn>", base_call,
                      callable_for_byteset_search<sz::find_matches_view, matcher_strcspn_t>(corpus)),
          baseline_of(base));

    // Include STL functionality
    print(bench_unary(env, corpus, "find_byteset<std::string_view::find_first_of>", base_call,
                      callable_for_byteset_search<sz::find_matches_view, matcher_std_string_first_of_t>(corpus)),
          baseline_of(base));
    print(bench_unary(env, corpus, "rfind_byteset<std::string_view::find_last_of>", base_call,
                      callable_for_byteset_search<sz::rfind_matches_view, matcher_std_string_last_of_t>(corpus)),
          baseline_of(base_reverse));
}

#pragma endregion Byteset Search

void bench_find(environment_t &env) {
    corpus_t const &corpus = env.corpora.words();
    fmt::println("Starting search benchmarks...");
    bench_substring_search(env, corpus);
    bench_byte_search(env, corpus);
    bench_byteset_search(env, corpus);
}

} // namespace ashvardanian::stringzilla::bench
