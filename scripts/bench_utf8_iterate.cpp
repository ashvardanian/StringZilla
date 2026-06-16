/**
 *  @file scripts/bench_utf8_iterate.cpp
 *  @brief Benchmarks the @b `sz_utf8_*` iteration & segmentation family against the serial baselines.
 *         Every kernel is benchmarked across all available SIMD backends side-by-side, and each backend's
 *         result is validated (via a per-call checksum) against the serial reference — so this file doubles
 *         as a differential correctness harness for the UTF-8 traversal kernels.
 *
 *  Benchmarks include:
 *  - Codepoint counting - @b utf8_count.
 *  - Nth-codepoint location - @b utf8_find_nth (the BMI/PDEP "Nth set bit" kernel on x86).
 *  - Newline enumeration - @b utf8_find_newlines.
 *  - Whitespace enumeration - @b utf8_find_whitespaces (Unicode White_Space property).
 *  - UAX-29 word-boundary segmentation, forward & reverse - @b utf8_word_find_boundaries / @b _rfind_boundary.
 *  - UTF-8 -> UTF-32 transcoding - @b utf8_unpack_chunk.
 *
 *  Instead of CLI arguments, for compatibility with @b StringWars, the following environment variables are used:
 *  - `STRINGWARS_DATASET` : Path to the dataset file.
 *  - `STRINGWARS_TOKENS=lines` : Tokenization model ("file", "lines", "words", or [1:200] for N-grams).
 *  - `STRINGWARS_SEED=42` : Optional seed for shuffling reproducibility.
 *
 *  Unlike StringWars, the following additional environment variables are supported:
 *  - `STRINGWARS_DURATION=10` : Time limit (in seconds) per benchmark.
 *  - `STRINGWARS_STRESS=1` : Test SIMD-accelerated functions against the serial baselines.
 *  - `STRINGWARS_FILTER` : Regular Expression pattern to filter algorithm/backend names.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCHMARK=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench_utf8_iterate_cpp20
 *  STRINGWARS_DATASET=xlsum.csv STRINGWARS_TOKENS=lines \
 *      build_release/stringzilla_bench_utf8_iterate_cpp20
 *  @endcode
 *
 *  This file is the sibling of `bench_utf8_uncased.cpp`, `bench_find.cpp`, `bench_sequence.cpp`, and `bench_memory.cpp`.
 */
#include <vector>

#include "bench.hpp"
#include "test_stringzilla.hpp" // `log_environment`

using namespace ashvardanian::stringzilla::scripts;

// Each `SZ_IF_<ISA>(...)` expands to its argument when that backend is compiled in, else to nothing. This
// keeps every per-kernel registration block a flat, readable list instead of nested `#if` ladders.
#if SZ_USE_HASWELL
#define SZ_IF_HASWELL(...) __VA_ARGS__
#else
#define SZ_IF_HASWELL(...)
#endif
#if SZ_USE_ICELAKE
#define SZ_IF_ICELAKE(...) __VA_ARGS__
#else
#define SZ_IF_ICELAKE(...)
#endif
#if SZ_USE_NEON
#define SZ_IF_NEON(...) __VA_ARGS__
#else
#define SZ_IF_NEON(...)
#endif
#if SZ_USE_SVE2
#define SZ_IF_SVE2(...) __VA_ARGS__
#else
#define SZ_IF_SVE2(...)
#endif
#if SZ_USE_V128
#define SZ_IF_V128(...) __VA_ARGS__
#else
#define SZ_IF_V128(...)
#endif
#if SZ_USE_RVV
#define SZ_IF_RVV(...) __VA_ARGS__
#else
#define SZ_IF_RVV(...)
#endif
#if SZ_USE_POWERVSX
#define SZ_IF_POWERVSX(...) __VA_ARGS__
#else
#define SZ_IF_POWERVSX(...)
#endif
#if SZ_USE_LASX
#define SZ_IF_LASX(...) __VA_ARGS__
#else
#define SZ_IF_LASX(...)
#endif

#pragma region Wrappers

/** @brief  Counts the codepoints of each token; checksum = codepoint count. */
template <auto func_>
struct utf8_count_from_sz {
    environment_t const &env;
    utf8_count_from_sz(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_size_t count = func_(token.data(), token.size());
        do_not_optimize(count);
        return {token.size(), static_cast<check_value_t>(count)};
    }
};

/** @brief  Locates the middle codepoint of each token; checksum = byte offset of the located codepoint. */
template <auto func_>
struct utf8_find_nth_from_sz {
    environment_t const &env;
    std::vector<sz_size_t> targets; // Nth codepoint to locate per token, precomputed outside the timed call.
    utf8_find_nth_from_sz(environment_t const &env_) : env(env_) {
        targets.reserve(env.tokens.size());
        for (auto const &token : env.tokens) targets.push_back(sz_utf8_count_serial(token.data(), token.size()) / 2);
    }
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t located = func_(token.data(), token.size(), targets[i]);
        do_not_optimize(located);
        check_value_t offset = located ? static_cast<check_value_t>(located - token.data()) : (check_value_t)-1;
        return {token.size(), offset};
    }
};

/** @brief  Enumerates every delimiter (newline / whitespace) across each token via the multistep "find
 *          boundaries" API, resuming through the whole token in `sz_iterators_default_steps_k`-sized batches;
 *          checksum = total number of delimiters. */
template <sz_utf8_find_boundaries_t find_func_>
struct utf8_enumerate_delimiters {
    environment_t const &env;
    utf8_enumerate_delimiters(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t text = token.data();
        sz_size_t len = token.size();
        sz_size_t offsets[sz_iterators_default_steps_k], lengths[sz_iterators_default_steps_k];
        sz_size_t pos = 0, total = 0;
        while (pos < len) {
            sz_size_t consumed = 0;
            total += find_func_(text + pos, len - pos, offsets, lengths, sz_iterators_default_steps_k, &consumed);
            if (consumed == 0) break;
            pos += consumed;
        }
        do_not_optimize(total);
        return {token.size(), static_cast<check_value_t>(total)};
    }
};

/** @brief  Segments each token into UAX-29 words forward; checksum = number of words. */
template <auto func_>
struct utf8_word_forward_from_sz {
    environment_t const &env;
    utf8_word_forward_from_sz(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        sz_size_t starts[16], lengths[16];
        std::size_t words = 0;
        while (remaining) {
            sz_size_t consumed = 0;
            sz_size_t produced = func_(cursor, remaining, starts, lengths, 16, &consumed);
            words += static_cast<std::size_t>(produced);
            if (produced == 0 || consumed >= remaining) break; // Whole suffix segmented.
            cursor += consumed;                                // Resume from the first word that did not fit.
            remaining -= consumed;
        }
        do_not_optimize(words);
        return {token.size(), static_cast<check_value_t>(words)};
    }
};

/** @brief  Segments each token into UAX-29 words from the end backward; checksum = number of words. */
template <auto func_>
struct utf8_word_reverse_from_sz {
    environment_t const &env;
    utf8_word_reverse_from_sz(environment_t const &env_) : env(env_) {}
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t base = token.data();
        sz_size_t remaining = token.size();
        sz_size_t starts[16], lengths[16];
        std::size_t words = 0;
        while (remaining) {
            sz_size_t consumed = 0;
            sz_size_t produced = func_(base, remaining, starts, lengths, 16, &consumed);
            words += static_cast<std::size_t>(produced);
            if (produced == 0 || consumed == 0) break; // Whole prefix segmented.
            remaining = consumed;                      // Continue the reverse scan over the prefix [base, consumed).
        }
        do_not_optimize(words);
        return {token.size(), static_cast<check_value_t>(words)};
    }
};

/** @brief  Transcodes each token UTF-8 -> UTF-32 chunk by chunk; checksum = number of runes produced. */
template <auto func_>
struct utf8_unpack_from_sz {
    environment_t const &env;
    mutable std::vector<sz_rune_t> runes; // Reusable output buffer, sized to the largest token.
    utf8_unpack_from_sz(environment_t const &env_) : env(env_) {
        std::size_t max_token = 1;
        for (auto const &token : env.tokens) max_token = std::max(max_token, token.size());
        runes.resize(max_token + 1);
    }
    inline call_result_t operator()(std::size_t i) const noexcept {
        token_view_t token = env.tokens[i];
        sz_cptr_t cursor = token.data();
        sz_size_t remaining = token.size();
        std::size_t produced = 0;
        while (remaining) {
            sz_size_t unpacked = 0;
            sz_cptr_t consumed = func_(cursor, remaining, runes.data(), runes.size(), &unpacked);
            std::size_t step = static_cast<std::size_t>(consumed - cursor);
            if (step == 0) break; // Incomplete trailing sequence; nothing more to do.
            produced += unpacked;
            cursor += step;
            remaining -= step;
        }
        do_not_optimize(runes.data());
        return {token.size(), static_cast<check_value_t>(produced)};
    }
};

#pragma endregion

#pragma region Benchmarks

void bench_utf8_count(environment_t const &env) {
    auto base_v = utf8_count_from_sz<sz_utf8_count_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_count_serial", base_v).log();
    SZ_IF_HASWELL(
        bench_unary(env, "sz_utf8_count_haswell", base_v, utf8_count_from_sz<sz_utf8_count_haswell> {env}).log(base);)
    SZ_IF_ICELAKE(
        bench_unary(env, "sz_utf8_count_icelake", base_v, utf8_count_from_sz<sz_utf8_count_icelake> {env}).log(base);)
    SZ_IF_NEON(bench_unary(env, "sz_utf8_count_neon", base_v, utf8_count_from_sz<sz_utf8_count_neon> {env}).log(base);)
    SZ_IF_SVE2(bench_unary(env, "sz_utf8_count_sve2", base_v, utf8_count_from_sz<sz_utf8_count_sve2> {env}).log(base);)
    SZ_IF_V128(bench_unary(env, "sz_utf8_count_v128", base_v, utf8_count_from_sz<sz_utf8_count_v128> {env}).log(base);)
    SZ_IF_RVV(bench_unary(env, "sz_utf8_count_rvv", base_v, utf8_count_from_sz<sz_utf8_count_rvv> {env}).log(base);)
    SZ_IF_POWERVSX(
        bench_unary(env, "sz_utf8_count_powervsx", base_v, utf8_count_from_sz<sz_utf8_count_powervsx> {env}).log(base);)
    SZ_IF_LASX(bench_unary(env, "sz_utf8_count_lasx", base_v, utf8_count_from_sz<sz_utf8_count_lasx> {env}).log(base);)
}

void bench_utf8_find_nth(environment_t const &env) {
    auto base_v = utf8_find_nth_from_sz<sz_utf8_find_nth_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_find_nth_serial", base_v).log();
    SZ_IF_HASWELL(
        bench_unary(env, "sz_utf8_find_nth_haswell", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_haswell> {env})
            .log(base);)
    SZ_IF_ICELAKE(
        bench_unary(env, "sz_utf8_find_nth_icelake", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_icelake> {env})
            .log(base);)
    SZ_IF_NEON(bench_unary(env, "sz_utf8_find_nth_neon", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_neon> {env})
                   .log(base);)
    SZ_IF_SVE2(bench_unary(env, "sz_utf8_find_nth_sve2", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_sve2> {env})
                   .log(base);)
    SZ_IF_V128(bench_unary(env, "sz_utf8_find_nth_v128", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_v128> {env})
                   .log(base);)
    SZ_IF_RVV(
        bench_unary(env, "sz_utf8_find_nth_rvv", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_rvv> {env}).log(base);)
    SZ_IF_POWERVSX(
        bench_unary(env, "sz_utf8_find_nth_powervsx", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_powervsx> {env})
            .log(base);)
    SZ_IF_LASX(bench_unary(env, "sz_utf8_find_nth_lasx", base_v, utf8_find_nth_from_sz<sz_utf8_find_nth_lasx> {env})
                   .log(base);)
}

void bench_utf8_find_newlines(environment_t const &env) {
    auto base_v = utf8_enumerate_delimiters<sz_utf8_find_newlines_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_find_newlines_serial", base_v).log();
    SZ_IF_HASWELL(bench_unary(env, "sz_utf8_find_newlines_haswell", base_v,
                              utf8_enumerate_delimiters<sz_utf8_find_newlines_haswell> {env})
                      .log(base);)
    SZ_IF_ICELAKE(bench_unary(env, "sz_utf8_find_newlines_icelake", base_v,
                              utf8_enumerate_delimiters<sz_utf8_find_newlines_icelake> {env})
                      .log(base);)
    SZ_IF_NEON(bench_unary(env, "sz_utf8_find_newlines_neon", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_newlines_neon> {env})
                   .log(base);)
    SZ_IF_SVE2(bench_unary(env, "sz_utf8_find_newlines_sve2", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_newlines_sve2> {env})
                   .log(base);)
    SZ_IF_V128(bench_unary(env, "sz_utf8_find_newlines_v128", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_newlines_v128> {env})
                   .log(base);)
    SZ_IF_RVV(bench_unary(env, "sz_utf8_find_newlines_rvv", base_v,
                          utf8_enumerate_delimiters<sz_utf8_find_newlines_rvv> {env})
                  .log(base);)
    SZ_IF_POWERVSX(bench_unary(env, "sz_utf8_find_newlines_powervsx", base_v,
                               utf8_enumerate_delimiters<sz_utf8_find_newlines_powervsx> {env})
                       .log(base);)
    SZ_IF_LASX(bench_unary(env, "sz_utf8_find_newlines_lasx", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_newlines_lasx> {env})
                   .log(base);)
}

void bench_utf8_find_whitespaces(environment_t const &env) {
    auto base_v = utf8_enumerate_delimiters<sz_utf8_find_whitespaces_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_find_whitespaces_serial", base_v).log();
    SZ_IF_HASWELL(bench_unary(env, "sz_utf8_find_whitespaces_haswell", base_v,
                              utf8_enumerate_delimiters<sz_utf8_find_whitespaces_haswell> {env})
                      .log(base);)
    SZ_IF_ICELAKE(bench_unary(env, "sz_utf8_find_whitespaces_icelake", base_v,
                              utf8_enumerate_delimiters<sz_utf8_find_whitespaces_icelake> {env})
                      .log(base);)
    SZ_IF_NEON(bench_unary(env, "sz_utf8_find_whitespaces_neon", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_whitespaces_neon> {env})
                   .log(base);)
    SZ_IF_SVE2(bench_unary(env, "sz_utf8_find_whitespaces_sve2", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_whitespaces_sve2> {env})
                   .log(base);)
    SZ_IF_V128(bench_unary(env, "sz_utf8_find_whitespaces_v128", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_whitespaces_v128> {env})
                   .log(base);)
    SZ_IF_RVV(bench_unary(env, "sz_utf8_find_whitespaces_rvv", base_v,
                          utf8_enumerate_delimiters<sz_utf8_find_whitespaces_rvv> {env})
                  .log(base);)
    SZ_IF_POWERVSX(bench_unary(env, "sz_utf8_find_whitespaces_powervsx", base_v,
                               utf8_enumerate_delimiters<sz_utf8_find_whitespaces_powervsx> {env})
                       .log(base);)
    SZ_IF_LASX(bench_unary(env, "sz_utf8_find_whitespaces_lasx", base_v,
                           utf8_enumerate_delimiters<sz_utf8_find_whitespaces_lasx> {env})
                   .log(base);)
}

void bench_utf8_word_find_boundaries(environment_t const &env) {
    auto base_v = utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_word_find_boundaries_serial", base_v).log();
    SZ_IF_HASWELL(bench_unary(env, "sz_utf8_word_find_boundaries_haswell", base_v,
                              utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_haswell> {env})
                      .log(base);)
    SZ_IF_ICELAKE(bench_unary(env, "sz_utf8_word_find_boundaries_icelake", base_v,
                              utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_icelake> {env})
                      .log(base);)
    SZ_IF_NEON(bench_unary(env, "sz_utf8_word_find_boundaries_neon", base_v,
                           utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_neon> {env})
                   .log(base);)
    SZ_IF_SVE2(bench_unary(env, "sz_utf8_word_find_boundaries_sve2", base_v,
                           utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_sve2> {env})
                   .log(base);)
    SZ_IF_V128(bench_unary(env, "sz_utf8_word_find_boundaries_v128", base_v,
                           utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_v128> {env})
                   .log(base);)
    SZ_IF_RVV(bench_unary(env, "sz_utf8_word_find_boundaries_rvv", base_v,
                          utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_rvv> {env})
                  .log(base);)
    SZ_IF_POWERVSX(bench_unary(env, "sz_utf8_word_find_boundaries_powervsx", base_v,
                               utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_powervsx> {env})
                       .log(base);)
    SZ_IF_LASX(bench_unary(env, "sz_utf8_word_find_boundaries_lasx", base_v,
                           utf8_word_forward_from_sz<sz_utf8_word_find_boundaries_lasx> {env})
                   .log(base);)
}

void bench_utf8_word_rfind_boundaries(environment_t const &env) {
    auto base_v = utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_word_rfind_boundaries_serial", base_v).log();
    SZ_IF_HASWELL(bench_unary(env, "sz_utf8_word_rfind_boundaries_haswell", base_v,
                              utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_haswell> {env})
                      .log(base);)
    SZ_IF_ICELAKE(bench_unary(env, "sz_utf8_word_rfind_boundaries_icelake", base_v,
                              utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_icelake> {env})
                      .log(base);)
    SZ_IF_NEON(bench_unary(env, "sz_utf8_word_rfind_boundaries_neon", base_v,
                           utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_neon> {env})
                   .log(base);)
    SZ_IF_SVE2(bench_unary(env, "sz_utf8_word_rfind_boundaries_sve2", base_v,
                           utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_sve2> {env})
                   .log(base);)
    SZ_IF_V128(bench_unary(env, "sz_utf8_word_rfind_boundaries_v128", base_v,
                           utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_v128> {env})
                   .log(base);)
    SZ_IF_RVV(bench_unary(env, "sz_utf8_word_rfind_boundaries_rvv", base_v,
                          utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_rvv> {env})
                  .log(base);)
    SZ_IF_POWERVSX(bench_unary(env, "sz_utf8_word_rfind_boundaries_powervsx", base_v,
                               utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_powervsx> {env})
                       .log(base);)
    SZ_IF_LASX(bench_unary(env, "sz_utf8_word_rfind_boundaries_lasx", base_v,
                           utf8_word_reverse_from_sz<sz_utf8_word_rfind_boundaries_lasx> {env})
                   .log(base);)
}

void bench_utf8_unpack_chunk(environment_t const &env) {
    auto base_v = utf8_unpack_from_sz<sz_utf8_unpack_chunk_serial> {env};
    bench_result_t base = bench_unary(env, "sz_utf8_unpack_chunk_serial", base_v).log();
    SZ_IF_ICELAKE(bench_unary(env, "sz_utf8_unpack_chunk_icelake", base_v,
                              utf8_unpack_from_sz<sz_utf8_unpack_chunk_icelake> {env})
                      .log(base);)
    SZ_IF_NEON(
        bench_unary(env, "sz_utf8_unpack_chunk_neon", base_v, utf8_unpack_from_sz<sz_utf8_unpack_chunk_neon> {env})
            .log(base);)
    SZ_IF_V128(
        bench_unary(env, "sz_utf8_unpack_chunk_v128", base_v, utf8_unpack_from_sz<sz_utf8_unpack_chunk_v128> {env})
            .log(base);)
    SZ_IF_RVV(bench_unary(env, "sz_utf8_unpack_chunk_rvv", base_v, utf8_unpack_from_sz<sz_utf8_unpack_chunk_rvv> {env})
                  .log(base);)
    SZ_IF_POWERVSX(bench_unary(env, "sz_utf8_unpack_chunk_powervsx", base_v,
                               utf8_unpack_from_sz<sz_utf8_unpack_chunk_powervsx> {env})
                       .log(base);)
    SZ_IF_LASX(
        bench_unary(env, "sz_utf8_unpack_chunk_lasx", base_v, utf8_unpack_from_sz<sz_utf8_unpack_chunk_lasx> {env})
            .log(base);)
}

#pragma endregion

int main(int argc, char const **argv) {
    install_test_signal_handlers(); // Backtrace on SIGSEGV/SIGABRT + line-buffered stdout for crash localization.
    std::printf("Welcome to StringZilla UTF-8 Iteration Benchmarks!\n");
    if (auto code = log_environment(); code != 0) return code;

    std::printf("Building up the environment...\n");
    environment_t env = build_environment( //
        argc, argv,                        //
        "xlsum.csv",                       // Default to xlsum for multilingual coverage
        environment_t::tokenization_t::lines_k);

    std::printf("Starting UTF-8 iteration benchmarks...\n");

    bench_utf8_count(env);
    bench_utf8_find_nth(env);
    bench_utf8_find_newlines(env);
    bench_utf8_find_whitespaces(env);
    bench_utf8_word_find_boundaries(env);
    bench_utf8_word_rfind_boundaries(env);
    bench_utf8_unpack_chunk(env);

    std::printf("All benchmarks passed.\n");
    return 0;
}
