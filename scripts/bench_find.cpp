/**
 *  @file   bench_find.cpp
 *  @brief  Benchmarks for bidirectional string search operations.
 *          The program accepts a file path to a dataset, tokenizes it, and benchmarks the search operations,
 *          validating the SIMD-accelerated backends against the serial baselines.
 *
 *  Benchmarks include:
 *  - Substring search: find all inclusions of a token in the dataset - @b find & @b rfind.
 *  - Byte search: find a specific byte value in each token (word, line, or file) - @b find_byte & @b rfind_byte.
 *  - Byteset search: find any byte value from a set in each token (line or file) - @b find_byteset & @b rfind_byteset.
 *
 *  For substring search, the number of operations per second are reported as the number of character-level comparisons
 *  happening in the worst case in the naive algorithm, meaning O(N*M) for N characters in the haystack and M in the
 *  needle. In byteset search, the number of operations per second is computed the same way and the following character
 *  sets are tested against each scanned token:
 *
 *  - "\n\r\v\f": 4 tabs
 *  - "</>&'\"=[]": 9 html
 *  - "0123456789": 10 digits
 *
 *  Instead of CLI arguments, for compatibility with @b StringWa.rs, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=words` : Tokenization model ("file", "lines", "words", or positive integer [1:200] for N-grams
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWa.rs, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_STRESS_DIR=/.tmp` : Output directory for stress-testing failures logs.
 *  - `STRINGWARS_STRESS_LIMIT=1` : Controls the number of failures we're willing to tolerate.
 *  - `STRINGWARS_STRESS_DURATION=10` : Stress-testing time limit (in seconds) per benchmark.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_find_cpp20
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=words build_release/stringzilla_bench_find_cpp20
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size inputs,
 *  like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes (exactly 1 cache line),
 *  your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGWARS_STRESS=1 STRINGWARS_STRESS_DURATION=120 STRINGWARS_STRESS_DIR=logs
 *  build_release/stringzilla_bench_find_cpp20
 *  @endcode
 *
 *  Unlike the full-blown StringWa.rs, it doesn't use any external frameworks like Criterion or Google Benchmark.
 *  This file is the sibling of `bench_sequence.cpp`, `bench_token.cpp`, and `bench_memory.cpp`.
 */
#include <cstring>    // `memmem`
#include <functional> // `std::boyer_moore_searcher`

#define SZ_USE_MISALIGNED_LOADS (1)
#include "bench.hpp"

using namespace ashvardanian::stringzilla::scripts;

#pragma region Substring Search

/**
 *  @brief  Wraps an individual hardware-specific search backend into something similar
 *          to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
template <sz_find_t find_func_>
struct matcher_from_sz_find {
    using size_type = std::size_t;
    std::string_view needle_;

    inline matcher_from_sz_find(std::string_view needle = {}) noexcept : needle_(needle) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = find_func_(haystack.data(), haystack.size(), needle_.data(), needle_.size());
        if (!ptr) return std::string_view::npos; // No match found
        return ptr - haystack.data();
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

static std::string strstr_needle_copy_ {}; //! Reuse the same memory for all needles, potentially causing allocations

/**
 *  @brief  Wraps the LibC functionality for finding the next occurrence of a NULL-terminated string
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_strstr_t {
    using size_type = std::size_t;

    inline matcher_strstr_t(std::string_view needle = {}) noexcept(false) { strstr_needle_copy_ = needle; }
    inline size_type needle_length() const noexcept { return strstr_needle_copy_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = (char *)strstr(haystack.data(), strstr_needle_copy_.c_str());
        do_not_optimize(ptr);
        if (!ptr) return std::string_view::npos; // No match found
        return (size_type)(ptr - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

#if defined(_GNU_SOURCE)
/**
 *  @brief  Wraps the LibC functionality for finding the next occurrence of a byte-string in a buffer
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_memmem_t {
    using size_type = std::size_t;
    std::string_view needle_;

    inline matcher_memmem_t(std::string_view needle = {}) noexcept : needle_(needle) {}
    inline size_type needle_length() const noexcept { return needle_.size(); }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = (char *)memmem(haystack.data(), haystack.size(), needle_.data(), needle_.size());
        do_not_optimize(ptr);
        if (!ptr) return std::string_view::npos; // No match found
        return (size_type)(ptr - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};
#endif

#if __cpp_lib_boyer_moore_searcher
/**
 *  @brief  Wraps the C++20 @b Boyer-Moore algorithms for finding the next occurrence of a string
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 *  @tparam searcher_type_ Can be `std::boyer_moore_searcher` or `std::boyer_moore_horspool_searcher`.
 *          Both should be instantiated with the `std::string_view::const_iterator` type.
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
        if (match == haystack.rend()) return std::string_view::npos; // No match found
        auto offset_from_end = match - haystack.rbegin();
        auto offset_from_start = haystack.size() - offset_from_end - needle_.size();
        return (size_type)offset_from_start;
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

#endif

template <template <typename, typename> class range_template_, typename matcher_type_>
auto callable_for_substring_search(environment_t const &env) {
    using matcher_t = matcher_type_;
    using matches_t = range_template_<std::string_view, matcher_t>;
    return [&env](std::size_t token_index) -> call_result_t {
        std::string_view haystack = env.dataset;
        std::string_view needle = env.tokens[token_index];
        matcher_t matcher(needle);
        matches_t matches(haystack, matcher);
        // Drain all matches to ensure the compiler doesn't optimize the search away
        std::size_t count_bytes = haystack.size();
        std::size_t count_matches = matches.size();
        std::size_t count_operations = count_bytes * needle.size();
        return call_result_t {count_bytes, count_matches, count_operations};
    };
}

/**
 *  @brief Find all inclusions of each given token in the dataset, using various search backends.
 */
void bench_substring_search(environment_t const &env) {

    // First, benchmark the serial function
    // The "check value" for normal and reverse search is the same - simply the number of matches.
    auto base_call = callable_for_substring_search<sz::range_matches, matcher_from_sz_find<sz_find_serial>>(env);
    bench_result_t base = bench_unary(env, "sz_find_serial", base_call).log();
    bench_result_t base_reverse =
        bench_unary(env, "sz_rfind_serial",
                    callable_for_substring_search<sz::range_rmatches, matcher_from_sz_find<sz_rfind_serial>>(env))
            .log();

    // Conditionally include SIMD-accelerated backends
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_find_skylake", base_call,
                callable_for_substring_search<sz::range_matches, matcher_from_sz_find<sz_find_skylake>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_skylake", base_call,
                callable_for_substring_search<sz::range_rmatches, matcher_from_sz_find<sz_rfind_skylake>>(env))
        .log(base_reverse);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_find_haswell", base_call,
                callable_for_substring_search<sz::range_matches, matcher_from_sz_find<sz_find_haswell>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_haswell", base_call,
                callable_for_substring_search<sz::range_rmatches, matcher_from_sz_find<sz_rfind_haswell>>(env))
        .log(base_reverse);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_find_sve", base_call,
                callable_for_substring_search<sz::range_matches, matcher_from_sz_find<sz_find_sve>>(env))
        .log(base);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_find_neon", base_call,
                callable_for_substring_search<sz::range_matches, matcher_from_sz_find<sz_find_neon>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_neon", base_call,
                callable_for_substring_search<sz::range_rmatches, matcher_from_sz_find<sz_rfind_neon>>(env))
        .log(base_reverse);
#endif

    // Include LibC functionality
    // ! Despite receiving string-views, following functions are assuming the strings are null-terminated.
    bench_unary(env, "find<std::strstr>", base_call, //
                callable_for_substring_search<sz::range_matches, matcher_strstr_t>(env))
        .log(base);

    // Include POSIX functionality
#if defined(_GNU_SOURCE)
    bench_unary(env, "find<memmem>", base_call, //
                callable_for_substring_search<sz::range_matches, matcher_memmem_t>(env))
        .log(base);
#endif

    // Include STL functionality
#if __cpp_lib_boyer_moore_searcher
    using matcher_bm_t = matcher_from_std_search<std::boyer_moore_searcher<std::string_view::const_iterator>>;
    using matcher_bmh_t = matcher_from_std_search<std::boyer_moore_horspool_searcher<std::string_view::const_iterator>>;
    using rmatcher_bm_t = rmatcher_from_std_search<std::boyer_moore_searcher<std::string_view::const_reverse_iterator>>;
    using rmatcher_bmh_t =
        rmatcher_from_std_search<std::boyer_moore_horspool_searcher<std::string_view::const_reverse_iterator>>;
    bench_unary(env, "find<std::boyer_moore>", base_call,
                callable_for_substring_search<sz::range_matches, matcher_bm_t>(env))
        .log(base);
    bench_unary(env, "rfind<std::boyer_moore>", base_call,
                callable_for_substring_search<sz::range_rmatches, rmatcher_bm_t>(env))
        .log(base_reverse);
    bench_unary(env, "find<std::boyer_moore_horspool>", base_call,
                callable_for_substring_search<sz::range_matches, matcher_bmh_t>(env))
        .log(base);
    bench_unary(env, "rfind<std::boyer_moore_horspool>", base_call,
                callable_for_substring_search<sz::range_rmatches, rmatcher_bmh_t>(env))
        .log(base_reverse);
#endif
}

#pragma endregion // Substring Search

#pragma region Byte Search

/**
 *  @brief  Wraps an individual hardware-specific search backend into something similar
 *          to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
template <sz_find_byte_t find_func_>
struct matcher_from_sz_find_byte {
    using size_type = std::size_t;
    char needle_;

    inline matcher_from_sz_find_byte(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = find_func_(haystack.data(), haystack.size(), &needle_);
        if (!ptr) return std::string_view::npos; // No match found
        return ptr - haystack.data();
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief  Wraps the LibC functionality for finding the next occurrence of a NULL-terminated string
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_strchr_t {
    using size_type = std::size_t;
    char needle_;

    inline matcher_strchr_t(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = (char *)strchr(haystack.data(), needle_);
        do_not_optimize(ptr);
        if (!ptr) return std::string_view::npos; // No match found
        return (size_type)(ptr - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief  Wraps the LibC functionality for finding the next occurrence of a byte-string in a buffer
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_memchr_t {
    using size_type = std::size_t;
    char needle_;

    inline matcher_memchr_t(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = (char *)std::memchr(haystack.data(), needle_, haystack.size());
        do_not_optimize(ptr);
        if (!ptr) return std::string_view::npos; // No match found
        return (size_type)(ptr - haystack.data());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief  Wraps the C++11 @b `std::find` algorithms for finding the next occurrence of a string
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_from_std_find {
    using size_type = std::size_t;
    char needle_;

    inline matcher_from_std_find(char needle) noexcept : needle_(needle) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match = std::find(haystack.begin(), haystack.end(), needle_);
        return (size_type)(match - haystack.begin());
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

template <template <typename, typename> class range_template_, typename matcher_type_>
auto callable_for_byte_search(environment_t const &env) {
    using matcher_t = matcher_type_;
    using matches_t = range_template_<std::string_view, matcher_t>;
    return [&env](std::size_t token_index) -> call_result_t {
        std::string_view haystack = env.tokens[token_index];
        std::size_t count_whitespaces = matches_t(haystack, matcher_t(' ')).size();
        std::size_t count_newlines = matches_t(haystack, matcher_t('\n')).size();
        std::size_t count_nulls = matches_t(haystack, matcher_t(0)).size();
        // As a checksum, mix the counts together
        std::size_t count_matches = count_whitespaces + count_newlines + count_nulls;
        std::size_t count_bytes = haystack.size() * 3; // We've traversed the input 3 times
        return call_result_t {count_bytes, count_matches};
    };
}

/**
 *  @brief Find all inclusions of a certain byte value in each token, be it a word, line, or the whole file.
 *  @warning Notice, the roles differ from `bench_substring_search`: each individual token is now treated as a haystack.
 */
void bench_byte_search(environment_t const &env) {
    // First, benchmark the serial function
    // The "check value" for normal and reverse search is the same - simply the number of matches.
    auto base_call = callable_for_byte_search<sz::range_matches, matcher_from_sz_find_byte<sz_find_byte_serial>>(env);
    bench_result_t base = bench_unary(env, "sz_find_byte_serial", base_call).log();
    bench_result_t base_reverse =
        bench_unary(env, "sz_rfind_byte_serial",
                    callable_for_byte_search<sz::range_rmatches, matcher_from_sz_find_byte<sz_rfind_byte_serial>>(env))
            .log();

    // Conditionally include SIMD-accelerated backends
#if SZ_USE_SKYLAKE
    bench_unary(env, "sz_find_byte_skylake", base_call,
                callable_for_byte_search<sz::range_matches, matcher_from_sz_find_byte<sz_find_byte_skylake>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_byte_skylake", base_call,
                callable_for_byte_search<sz::range_rmatches, matcher_from_sz_find_byte<sz_rfind_byte_skylake>>(env))
        .log(base_reverse);
#endif
#if SZ_USE_HASWELL
    bench_unary(env, "sz_find_byte_haswell", base_call,
                callable_for_byte_search<sz::range_matches, matcher_from_sz_find_byte<sz_find_byte_haswell>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_byte_haswell", base_call,
                callable_for_byte_search<sz::range_rmatches, matcher_from_sz_find_byte<sz_rfind_byte_haswell>>(env))
        .log(base_reverse);
#endif
#if SZ_USE_NEON
    bench_unary(env, "sz_find_byte_neon", base_call,
                callable_for_byte_search<sz::range_matches, matcher_from_sz_find_byte<sz_find_byte_neon>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_byte_neon", base_call,
                callable_for_byte_search<sz::range_rmatches, matcher_from_sz_find_byte<sz_rfind_byte_neon>>(env))
        .log(base_reverse);
#endif
#if SZ_USE_SVE
    bench_unary(env, "sz_find_byte_sve", base_call,
                callable_for_byte_search<sz::range_matches, matcher_from_sz_find_byte<sz_find_byte_sve>>(env))
        .log(base);
    bench_unary(env, "sz_rfind_byte_sve", base_call,
                callable_for_byte_search<sz::range_rmatches, matcher_from_sz_find_byte<sz_rfind_byte_sve>>(env))
        .log(base_reverse);
#endif

    // Include LibC functionality
    bench_unary(env, "find_byte<std::strchr>", base_call, //
                callable_for_byte_search<sz::range_matches, matcher_strchr_t>(env))
        .log(base);
    bench_unary(env, "find_byte<std::memchr>", base_call, //
                callable_for_byte_search<sz::range_matches, matcher_memchr_t>(env))
        .log(base);

    // Include STL functionality
    bench_unary(env, "find_byte<std::find>", base_call, //
                callable_for_byte_search<sz::range_matches, matcher_from_std_find>(env))
        .log(base);
}

#pragma endregion // Byte Search

#pragma region Byteset Search

/**
 *  @brief  Wraps an individual hardware-specific search backend into something similar
 *          to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
template <sz_find_byteset_t find_func_>
struct matcher_from_sz_find_byteset {
    using size_type = std::size_t;
    sz::byteset needles_; // Pick C++ alternative over `sz_byteset_t` for `constexp` constructor

    constexpr matcher_from_sz_find_byteset(sz::byteset needles) noexcept : needles_(needles) {}
    constexpr size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto ptr = find_func_(haystack.data(), haystack.size(), &needles_.raw());
        if (!ptr) return std::string_view::npos; // No match found
        return ptr - haystack.data();
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief  Wraps the LibC functionality for finding the next occurrence of a NULL-terminated string
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_strcspn_t {
    using size_type = std::size_t;
    std::string_view needles_;

    inline matcher_strcspn_t(std::string_view needles) noexcept : needles_(needles) {}
    inline size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept {
        auto match = strcspn(haystack.data(), needles_.data());
        if (match == haystack.size()) return std::string_view::npos; // No match found
        return match;
    }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief  Wraps the C++11 @b `std::string_view::find_first_of` algorithms for finding the next occurrence of a string
 *          into something similar to @b `sz::matcher_find` and compatible with @b `sz::range_matches`.
 */
struct matcher_std_string_first_of_t {
    using size_type = std::size_t;
    std::string_view needles_;

    inline matcher_std_string_first_of_t(std::string_view needles) noexcept : needles_(needles) {}
    inline size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept { return haystack.find_first_of(needles_); }
    constexpr size_type skip_length() const noexcept { return 1; }
};

/**
 *  @brief  Wraps the C++11 @b `std::string_view::find_last_of` algorithms for finding the next occurrence of a string
 *          into something similar to @b `sz::matcher_rfind` and compatible with @b `sz::range_rmatches`.
 */
struct matcher_std_string_last_of_t {
    using size_type = std::size_t;
    std::string_view needles_;

    inline matcher_std_string_last_of_t(std::string_view needles) noexcept : needles_(needles) {}
    inline size_type needle_length() const noexcept { return 1; }
    inline size_type operator()(std::string_view haystack) const noexcept { return haystack.find_last_of(needles_); }
    constexpr size_type skip_length() const noexcept { return 1; }
};

template <template <typename, typename> class range_template_, typename matcher_type_, typename byteset_type_>
auto callable_for_byteset_search(environment_t const &env) {
    using matcher_t = matcher_type_;
    using matches_t = range_template_<std::string_view, matcher_t>;
    return [&env](std::size_t token_index) -> call_result_t {
        std::string_view haystack = env.tokens[token_index];
        std::size_t count_tabs = matches_t(haystack, matcher_t(byteset_type_("\n\r\v\f", 4))).size();
        std::size_t count_html = matches_t(haystack, matcher_t(byteset_type_("</>&'\"=[]", 9))).size();
        std::size_t count_digits = matches_t(haystack, matcher_t(byteset_type_("0123456789", 10))).size();
        // As a checksum, mix the counts together
        std::size_t count_matches = count_tabs + count_html + count_digits;
        std::size_t count_bytes = haystack.size() * 3; // We've traversed the input 3 times
        return call_result_t {count_bytes, count_matches};
    };
}

/**
 *  @brief Find all inclusions of any byte from a set in each token, be it a word, line, or the whole file.
 *  @warning Notice, the roles differ from `bench_substring_search`: each individual token is now treated as a haystack.
 */
void bench_byteset_search(environment_t const &env) {

    // First, benchmark the serial function
    // The "check value" for normal and reverse search is the same - simply the number of matches.
    auto base_call =
        callable_for_byteset_search<sz::range_matches, matcher_from_sz_find_byteset<sz_find_byteset_serial>,
                                    sz::byteset>(env);
    bench_result_t base = bench_unary(env, "sz_find_byteset_serial", base_call).log();
    bench_result_t base_reverse =
        bench_unary(
            env, "sz_rfind_byteset_serial",
            callable_for_byteset_search<sz::range_rmatches, matcher_from_sz_find_byteset<sz_rfind_byteset_serial>,
                                        sz::byteset>(env))
            .log();

    // Conditionally include SIMD-accelerated backends
#if SZ_USE_HASWELL
    bench_unary( //
        env, "sz_find_byteset_haswell", base_call,
        callable_for_byteset_search<sz::range_matches, matcher_from_sz_find_byteset<sz_find_byteset_haswell>,
                                    sz::byteset>(env))
        .log(base);
    bench_unary( //
        env, "sz_rfind_byteset_haswell", base_call,
        callable_for_byteset_search<sz::range_rmatches, matcher_from_sz_find_byteset<sz_rfind_byteset_haswell>,
                                    sz::byteset>(env))
        .log(base_reverse);
#endif
#if SZ_USE_ICE
    bench_unary( //
        env, "sz_find_byteset_ice", base_call,
        callable_for_byteset_search<sz::range_matches, matcher_from_sz_find_byteset<sz_find_byteset_ice>, sz::byteset>(
            env))
        .log(base);
    bench_unary( //
        env, "sz_rfind_byteset_ice", base_call,
        callable_for_byteset_search<sz::range_rmatches, matcher_from_sz_find_byteset<sz_rfind_byteset_ice>,
                                    sz::byteset>(env))
        .log(base_reverse);
#endif
#if SZ_USE_NEON
    bench_unary(
        env, "sz_find_byteset_neon", base_call,
        callable_for_byteset_search<sz::range_matches, matcher_from_sz_find_byteset<sz_find_byteset_neon>, sz::byteset>(
            env))
        .log(base);
    bench_unary(env, "sz_rfind_byteset_neon", base_call,
                callable_for_byteset_search<sz::range_rmatches, matcher_from_sz_find_byteset<sz_rfind_byteset_neon>,
                                            sz::byteset>(env))
        .log(base_reverse);
#endif

    // Include LibC functionality
    bench_unary(env, "find_byteset<std::strcspn>", base_call,
                callable_for_byteset_search<sz::range_matches, matcher_strcspn_t, std::string_view>(env))
        .log(base);

    // Include STL functionality
    bench_unary(env, "find_byteset<std::string_view::find_first_of>", base_call,
                callable_for_byteset_search<sz::range_matches, matcher_std_string_first_of_t, std::string_view>(env))
        .log(base);
    bench_unary(env, "rfind_byteset<std::string_view::find_last_of>", base_call,
                callable_for_byteset_search<sz::range_rmatches, matcher_std_string_last_of_t, std::string_view>(env))
        .log(base_reverse);
}

#pragma endregion // Byteset Search

int main(int argc, char const **argv) {
    std::printf("Welcome to StringZilla!\n");

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "leipzig1M.txt",                   //
        environment_t::tokenization_t::words_k);

    std::printf("Starting search benchmarks...\n");
    bench_substring_search(env);
    bench_byte_search(env);
    bench_byteset_search(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}