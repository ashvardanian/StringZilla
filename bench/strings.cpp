/**
 *  @file bench/strings.cpp
 *  @author Ash Vardanian
 *  @date September 28, 2024
 *  @brief Benchmarks random fills and the conversions of string collections: lookups, escaping and
 *      Unicode normalization.
 *
 *  Times the dispatch points over the English lines against their STL analogs and plain loops, a
 *  fill per line and a conversion per batch of lines, to mimic real-world scenarios dealing with
 *  strings of different lengths. Every capability's kernels are timed against the serial ones by
 *  the `cross_<arch>.cpp` files.
 *
 *  Memory-bound: the fill, lookup and escape kernels stream bytes, so it reads the whole file by
 *  default and a larger buffer measures throughput truer.
 *
 *  Here are a few build & run commands:
 *
 *  @code{.sh}
 *  cmake -D STRINGZILLA_BUILD_BENCH=1 -D CMAKE_BUILD_TYPE=Release -B build_release
 *  cmake --build build_release --config Release --target stringzilla_bench
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=lines STRINGWARS_FILTER='fill_random|strings' \
 *  build_release/stringzilla_bench
 *  @endcode
 *
 *  Alternatively, if you really want to stress-test a very specific function on a certain size
 *  inputs, like all Skylake-X and newer kernels on a boundary-condition input length of 64 bytes
 *  (exactly 1 cache line), your last command may look like:
 *
 *  @code{.sh}
 *  STRINGWARS_DATASET=leipzig1M.txt STRINGWARS_TOKENS=64 STRINGWARS_FILTER=skylake
 *  STRINGZILLA_STRESS=1 STRINGZILLA_STRESS_TIME_LIMIT=120s STRINGZILLA_STRESS_DIR=logs
 *  build_release/stringzilla_bench
 *  @endcode
 *
 *  Unlike the full-blown StringWars, it doesn't use any external frameworks like Criterion or
 *  Google Benchmark. This file is the sibling of `find.cpp`, `token.cpp`, and `sort.cpp`.
 */
#include <algorithm> // `std::transform`, `std::generate`
#include <random>    // `std::minstd_rand`

#include <fmt/format.h>

#include "cross.hpp"

namespace ashvardanian::stringzilla::bench {

#pragma region Random Fills

/**
 *  @brief The `std::` baseline for @c sz_fill_random_best, measuring generator throughput alone.
 *
 *  Like the kernels behind it, the same nonce replays the same bytes: each call seeds a
 *  @c std::minstd_rand from it, whose one word of state keeps the reseed out of the measurement
 *  where a Mersenne-Twister state fill would not.
 */
sz_status_t generate_like_sz(sz_ptr_t output, sz_size_t length, sz_u64_t nonce, sz_stream_t) {
    std::minstd_rand generator(static_cast<std::minstd_rand::result_type>(nonce));
    std::uniform_int_distribution<std::uint32_t> distribution(1, 255);
    std::generate(output, output + length, [&]() -> char { return static_cast<char>(distribution(generator)); });
    return sz_success_k;
}

/**
 *  @brief Benchmarks overwriting regions of output memory with reproducible random byte streams.
 *
 *  Multiple calls to the provided functions with the same arguments write the same bytes, so the
 *  dispatch point can be compared against the baseline generator.
 */
void bench_fill_random(environment_t const &env, corpus_t const &corpus) {
    dataset_copy_t output(corpus);
    sz_ptr_t o = output.data();

    std::optional<row_t> const random = bench_unary(env, corpus, "sz_fill_random_best",
                                                    fill_random_from_sz<cpu_best<sz_fill_random_best>> {corpus, o});
    print(random);

    // The generators differ, so the baseline is timed but never validated.
    print(bench_unary(env, corpus, "fill<std::random_device>", fill_random_from_sz<generate_like_sz> {corpus, o}),
          baseline_of(random));
}

#pragma endregion Random Fills

#pragma region Strings

/** Appends each line of the batch from a token index to one buffer it reuses, the way a baseline
 *  converts strings one at a time, and sums the bytes it wrote as @c strings_from_sz_t does. */
struct strings_from_std_t {
    corpus_t const &corpus;
    std::size_t batch;
    std::function<void(std::string &, std::string_view)> append;
    std::string output {};
    sz_kernel_bytesum_t checksum = output_checksum_kernel();

    call_result_t operator()(std::size_t token_index) {
        std::size_t const end = std::min(token_index + batch, corpus.tokens.size());
        std::size_t bytes = 0;
        output.clear();
        for (std::size_t index = token_index; index != end; ++index) {
            append(output, corpus.tokens[index]);
            bytes += corpus.tokens[index].size();
        }
        return {bytes, strings_check_value(checksum, output.data(), output.size())};
    }
};

/** Escapes @p text as an RFC 8259 string body one byte at a time, as a hand-written serializer
 *  would, matching @c sz_escape_json_k byte for byte. */
void escape_json_by_loop(std::string &output, std::string_view text) {
    char const *const digits = "0123456789abcdef";
    for (char const character : text) {
        switch (character) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default:
            if (static_cast<unsigned char>(character) >= 0x20) output += character;
            else output += "\\u00", output += digits[character >> 4], output += digits[character & 15];
        }
    }
}

/** Quotes @p text as an RFC 4180 field when it holds a comma, a quote, CR or LF, doubling quotes,
 *  matching @c sz_escape_csv_k byte for byte. */
void escape_csv_by_loop(std::string &output, std::string_view text) {
    if (text.find_first_of(",\"\r\n") == std::string_view::npos) {
        output += text;
        return;
    }
    output += '"';
    for (char const character : text) {
        if (character == '"') output += '"';
        output += character;
    }
    output += '"';
}

/**
 *  @brief Benchmarks look-up transformations of batches of lines into a reused tape.
 *
 *  Performs a simple cyclical rotation of the alphabet, to test the performance of the different
 *  "look-up table"-based transformations.
 */
void bench_strings_lookup(environment_t const &env, corpus_t const &corpus, std::size_t batch) {
    std::array<unsigned char, 256> const alphabet = rotated_alphabet();
    sz_cptr_t const lut = reinterpret_cast<sz_cptr_t>(alphabet.data());
    strings_input_t lines(corpus);
    strings_from_sz_t best {strings_lookup_call(cpu_best<sz_strings_lookup_best>, lut), lines, batch};
    auto const transform = [lut](std::string &output, std::string_view line) {
        std::size_t const start = output.size();
        output.resize(start + line.size());
        std::transform(line.begin(), line.end(), output.begin() + start,
                       [lut](char character) { return lut[static_cast<unsigned char>(character)]; });
    };

    std::optional<row_t> const base = bench_unary(env, corpus, "sz_strings_lookup_best", best);
    print(base);
    print(bench_unary(env, corpus, "lookup<std::transform>", best, strings_from_std_t {corpus, batch, transform}),
          baseline_of(base));
}

/** Benchmarks escaping batches of lines for JSON and CSV against a loop that escapes each line. */
void bench_strings_escape(environment_t const &env, corpus_t const &corpus, std::size_t batch) {
    strings_input_t lines(corpus);
    auto const bench_kind = [&](char const *name, sz_escape_kind_t kind,
                                void (*escape_by_loop)(std::string &, std::string_view)) {
        strings_from_sz_t best {strings_escape_call(cpu_best<sz_strings_escape_best>, kind), lines, batch};
        std::optional<row_t> const base = bench_unary(env, corpus, fmt::format("sz_strings_escape_{}_best", name),
                                                      best);
        print(base);
        print(bench_unary(env, corpus, fmt::format("escape_{}<loop>", name), best,
                          strings_from_std_t {corpus, batch, escape_by_loop}),
              baseline_of(base));
    };
    bench_kind("json", sz_escape_json_k, escape_json_by_loop);
    bench_kind("csv", sz_escape_csv_k, escape_csv_by_loop);
}

/** Benchmarks normalizing batches of lines to NFC and NFKC_Casefold against a loop that normalizes
 *  each line alone through @c sz_utf8_norm_best, into scratch sized for its worst case. */
void bench_strings_normalize(environment_t const &env, corpus_t const &corpus, std::size_t batch) {
    strings_input_t lines(corpus);
    auto const bench_form = [&](char const *name, sz_normal_form_t form) {
        strings_from_sz_t best {strings_normalize_call(cpu_best<sz_strings_normalize_best>, form), lines, batch};
        auto const normalize_alone = [form, scratch = std::string()](std::string &output,
                                                                     std::string_view line) mutable {
            scratch.resize(std::max(scratch.size(), line.size() * sz_utf8_norm_decomp_max_k));
            sz_size_t written = 0;
            if (sz_utf8_norm_best(line.data(), line.size(), form, scratch.data(), &written, sz::default_capabilities(),
                                  nullptr) != sz_success_k)
                throw std::runtime_error("A line would not normalize.");
            output.append(scratch.data(), written);
        };
        std::optional<row_t> const base = bench_unary(env, corpus, fmt::format("sz_strings_normalize_{}_best", name),
                                                      best);
        print(base);
        print(bench_unary(env, corpus, fmt::format("normalize_{}<loop>", name), best,
                          strings_from_std_t {corpus, batch, normalize_alone}),
              baseline_of(base));
    };
    bench_form("nfc", sz_normal_form_nfc_k);
    bench_form("nfkc_casefold", sz_normal_form_nfkc_casefold_k);
}

#pragma endregion Strings

void bench_strings(environment_t &env) {
    corpus_t const &corpus = env.corpora.lines();
    std::size_t const batch = candidates_per_call(env, corpus);
    fmt::println("Starting random-fill and string-collection benchmarks...");
    bench_fill_random(env, corpus);
    bench_strings_lookup(env, corpus, batch);
    bench_strings_escape(env, corpus, batch);
    bench_strings_normalize(env, corpus, batch);
}

} // namespace ashvardanian::stringzilla::bench
