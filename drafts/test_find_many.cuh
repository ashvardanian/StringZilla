/**
 *  @brief   Extensive @b stress-testing suite for StringCuZilla parallel operations, written in CUDA C++.
 *  @see     Stress-tests on real-world and synthetic data are integrated into the @b `scripts/bench*.cpp` benchmarks.
 *
 *  @file    test_stringzillas.cuh
 *  @author  Ash Vardanian
 */
#include <cstring> // `std::memcmp`
#include <thread>  // `std::thread::hardware_concurrency`

#define FU_ENABLE_NUMA 0
#include <fork_union.hpp> // Fork-join scoped thread pool

#include "stringzillas/find_many.hpp"

#if SZ_USE_CUDA
#include "stringzillas/find_many.cuh"
#endif

#if !SZ_IS_CPP17_
#error "This test requires C++17 or later."
#endif

#include "test_stringzilla.hpp" // `arrow_strings_view_t`

namespace ashvardanian {
namespace stringzillas {
namespace scripts {

namespace fu = fork_union;
using namespace stringzilla;
using namespace stringzilla::scripts;

struct find_many_baselines_t {
    using match_t = find_many_match_t;

    arrow_strings_tape_t needles_;

    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles) noexcept {
        return needles_.try_assign(needles.begin(), needles.end());
    }

    void reset() noexcept { needles_.reset(); }

    template <typename haystack_type_, typename needles_type_, typename match_callback_type_>
    bool one_haystack(haystack_type_ const &haystack, needles_type_ const &needles,
                      match_callback_type_ &&callback) const noexcept {

        // A wise man once said, `omp parallel for collapse(2) schedule(dynamic, 1)`...
        // But the compiler wasn't listening, and won't compile the cancellation point!
        // So we resort to a much less intricate solution:
        // - Manually slice the data per thread,
        // - Keep one atomic variable to signal cancellation,
        // - Use absolutely minimal OpenMP functionality just to assign N slices to N threads.
        std::atomic<bool> aborted {false};
        std::size_t const haystack_size = haystack.size();
        std::size_t const threads_count = std::thread::hardware_concurrency();
        std::size_t const start_offsets_per_thread = divide_round_up(haystack_size, threads_count);

#pragma omp parallel for schedule(static, 1)
        for (std::size_t thread_index = 0; thread_index != threads_count; ++thread_index) {
            std::size_t const start_offset = std::min(thread_index * start_offsets_per_thread, haystack_size);
            std::size_t const end_offset = std::min(start_offset + start_offsets_per_thread, haystack_size);

            // Check for matches in the current slice
            for (std::size_t match_offset = start_offset;
                 match_offset != end_offset && !aborted.load(std::memory_order_relaxed); ++match_offset) {
                for (std::size_t needle_index = 0; needle_index != needles.size(); ++needle_index) {
                    auto const &needle = needles[needle_index];
                    if (match_offset + needle.size() > haystack_size) continue;
                    auto const same = std::memcmp(haystack.data() + match_offset, needle.data(), needle.size()) == 0;
                    if (!same) continue;

                    // Create a match object
                    match_t match;
                    match.haystack_index = 0;
                    match.needle_index = needle_index;
                    match.haystack = {reinterpret_cast<byte_t const *>(haystack.data()), haystack.size()};
                    match.needle = {reinterpret_cast<byte_t const *>(haystack.data() + match_offset), needle.size()};
                    if (!callback(match)) {
                        aborted.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
            }
        }

        return !aborted.load(std::memory_order_relaxed);
    }

    template <typename haystacks_type_, typename needles_type_, typename match_callback_type_>
    void all_pairs(haystacks_type_ &&haystacks, needles_type_ &&needles,
                   match_callback_type_ &&callback) const noexcept {

        for (std::size_t haystack_index = 0; haystack_index != haystacks.size(); ++haystack_index) {
            auto const &haystack = haystacks[haystack_index];
            if (!one_haystack(haystack, needles, [&](match_t match) noexcept {
                    match.haystack_index = haystack_index;
                    return callback(match);
                }))
                return;
        }
    }

    template <typename haystacks_type_>
    status_t try_count(haystacks_type_ &&haystacks, span<size_t> counts) const noexcept {
        for (size_t &count : counts) count = 0;
        all_pairs(haystacks, needles_, [&](match_t const &match) noexcept {
            std::atomic_ref<size_t> count(counts[match.haystack_index]);
            count.fetch_add(1, std::memory_order_relaxed);
            return true;
        });
        return status_t::success_k;
    }

    template <typename haystacks_type_, typename output_matches_type_>
    status_t try_find(haystacks_type_ &&haystacks, span<size_t const> counts,
                      output_matches_type_ &&matches) const noexcept {

        sz_unused_(counts);
        std::atomic<size_t> count_found {0};
        std::size_t const count_allowed {matches.size()};
        all_pairs(haystacks, needles_, [&](match_t const &match) noexcept {
            size_t match_index = count_found.fetch_add(1, std::memory_order_relaxed);
            matches[match_index] = match;
            return match_index < count_allowed;
        });
        return status_t::success_k;
    }
};

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a @b fixed set of different representative ASCII and UTF-8 strings.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_on(std::vector<std::string> haystacks, std::vector<std::string> needles,
                       base_operator_ &&base_operator, simd_operator_ &&simd_operator, extra_args_ &&...extra_args) {

    using match_t = find_many_match_t;

    // First check with a batch-size of 1
    unified_vector<size_t> counts_base(1), counts_simd(1);
    unified_vector<match_t> matches_base(1), matches_simd(1);
    arrow_strings_tape_t haystacks_tape, needles_tape;
    needles_tape.try_assign(needles.data(), needles.data() + needles.size());

    // Construct the matchers
    status_t status_base = base_operator.try_build(needles_tape.view());
    status_t status_simd = simd_operator.try_build(needles_tape.view());
    sz_assert_(status_base == status_t::success_k);
    sz_assert_(status_simd == status_t::success_k);

    // Old C-style for-loops are much more debuggable than range-based loops!
    for (std::size_t haystack_idx = 0; haystack_idx != haystacks.size(); ++haystack_idx) {
        auto const &haystack = haystacks[haystack_idx];

        // Reset the tapes and results
        counts_base[0] = 0, counts_simd[0] = 0;
        matches_base.clear(), matches_simd.clear();
        haystacks_tape.try_assign(&haystack, &haystack + 1);

        // Count with both backends
        span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
        span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
        status_t status_count_base = base_operator.try_count(haystacks_tape.view(), counts_base_span);
        status_t status_count_simd = simd_operator.try_count(haystacks_tape.view(), counts_simd_span, extra_args...);
        sz_assert_(status_count_base == status_t::success_k);
        sz_assert_(status_count_simd == status_t::success_k);
        sz_assert_(counts_base[0] == counts_simd[0]);

        // Check the matches themselves
        matches_base.resize(std::accumulate(counts_base.begin(), counts_base.end(), 0));
        matches_simd.resize(std::accumulate(counts_simd.begin(), counts_simd.end(), 0));
        status_t status_matched_base = base_operator.try_find(haystacks_tape.view(), counts_base_span, matches_base);
        status_t status_matched_simd =
            simd_operator.try_find(haystacks_tape.view(), counts_simd_span, matches_simd, extra_args...);
        sz_assert_(status_matched_base == status_t::success_k);
        sz_assert_(status_matched_simd == status_t::success_k);

        // Check the contents and order of the matches
        std::sort(matches_base.begin(), matches_base.end(), match_t::less_globally);
        std::sort(matches_simd.begin(), matches_simd.end(), match_t::less_globally);
        for (std::size_t i = 0; i != matches_base.size(); ++i) {
            sz_assert_(matches_base[i].haystack.data() == matches_simd[i].haystack.data());
            sz_assert_(matches_base[i].needle.data() == matches_simd[i].needle.data());
            sz_assert_(matches_base[i].needle_index == matches_simd[i].needle_index);
        }
    }

    // Now test all the haystacks simultaneously
    {
        haystacks_tape.try_assign(haystacks.data(), haystacks.data() + haystacks.size());
        counts_base.resize(haystacks.size());
        counts_simd.resize(haystacks.size());

        // Count with both backends and compare all of the bounds
        span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
        span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
        status_t status_count_base = base_operator.try_count(haystacks_tape.view(), counts_base_span);
        status_t status_count_simd = simd_operator.try_count(haystacks_tape.view(), counts_simd_span, extra_args...);
        sz_assert_(status_count_base == status_t::success_k);
        sz_assert_(status_count_simd == status_t::success_k);
        sz_assert_(std::equal(counts_base.begin(), counts_base.end(), counts_simd.begin()));

        // Check the matches themselves
        matches_base.resize(std::accumulate(counts_base.begin(), counts_base.end(), 0));
        matches_simd.resize(std::accumulate(counts_simd.begin(), counts_simd.end(), 0));
        status_t status_matched_base = base_operator.try_find(haystacks_tape.view(), counts_base_span, matches_base);
        status_t status_matched_simd =
            simd_operator.try_find(haystacks_tape.view(), counts_simd_span, matches_simd, extra_args...);
        sz_assert_(status_matched_base == status_t::success_k);
        sz_assert_(status_matched_simd == status_t::success_k);

        // Check the contents and order of the matches
        std::sort(matches_base.begin(), matches_base.end(), match_t::less_globally);
        std::sort(matches_simd.begin(), matches_simd.end(), match_t::less_globally);
        for (std::size_t i = 0; i != matches_base.size(); ++i) {
            sz_assert_(matches_base[i].haystack.data() == matches_simd[i].haystack.data());
            sz_assert_(matches_base[i].needle.data() == matches_simd[i].needle.data());
            sz_assert_(matches_base[i].needle_index == matches_simd[i].needle_index);
        }
    }
}

/**
 *  @brief  Tests the correctness of the string class Levenshtein distance computation,
 *          as well as the similarity scoring functions for bioinformatics-like workloads
 *          on a @b fixed set of different representative ASCII and UTF-8 strings.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_fixed(base_operator_ &&base_operator, simd_operator_ &&simd_operator, extra_args_ &&...extra_args) {

    {
        std::vector<std::string> haystacks, needles;

        // Some vary basic variants:
        needles.emplace_back("his");
        needles.emplace_back("is");
        needles.emplace_back("she");
        needles.emplace_back("her");

        needles.emplace_back("école"), needles.emplace_back("école");                   // decomposed
        needles.emplace_back("Schön"), needles.emplace_back("Scho\u0308n");             // combining diaeresis
        needles.emplace_back("naïve"), needles.emplace_back("naive");                   // stripped diaeresis
        needles.emplace_back("façade"), needles.emplace_back("facade");                 // no cedilla
        needles.emplace_back("office"), needles.emplace_back("ofﬁce");                  // “fi” ligature
        needles.emplace_back("Straße"), needles.emplace_back("Strasse");                // ß vs ss
        needles.emplace_back("ABBA"), needles.emplace_back("\u0410\u0412\u0412\u0410"); // Latin vs Cyrillic
        needles.emplace_back("中国"), needles.emplace_back("中國");                     // simplified vs traditional
        needles.emplace_back("🙂"), needles.emplace_back("☺️");                          // emoji variants
        needles.emplace_back("€100"), needles.emplace_back("EUR 100"); // currency symbol vs abbreviation

        // Haystacks should contain arbitrary strings including those needles
        // in different positions, potentially interleaving
        haystacks.emplace_back("That is a test string"); // ? "only "is"
        haystacks.emplace_back("This is a test string"); // ? "his", 2x "is"
        haystacks.emplace_back("ahishers");              // textbook example
        haystacks.emplace_back("hishishersherishis");    // heavy overlap, prefix & suffix collisions
        haystacks.emplace_back("si siht si a tset gnirts; reh ton si ehs, tub sih ti si."); // no real matches
        haystacks.emplace_back("his\0is\r\nshe\0her");                                      // null-included

        // ~260 chars – dense English with overlapping words (“his”, “is”, “she”, “her”)
        haystacks.emplace_back(R"(
        In this historic thesis, the historian highlights his findings: this is the synthesis of data.
        She examined the theory, he shared her methodology. In this chapter, he lists his equipment:
        microscope, test kit, sensor. It is here that she erred: misalignment arises.
        )");

        // ~320 chars – multilingual snippet with needles in Latin, Arabic, Chinese, English
        haystacks.emplace_back(R"(
        The conference in 北京 attracted researchers from across the globe. His presentation “AI in Healthcare”
        was a hit—she received awards. الباحثون استعرضوا الأبحاث، واستشارت her colleagues. 这是一次重要的会议。
        She said: “This is only the beginning.” In her report, his name appears seventeen times.
        )");

        test_find_many_on(haystacks, needles, base_operator, simd_operator, extra_args...);
    }

    // Many of our algorithms depend on the idea that needles are shorter than the slices that each core may receive
    {
        std::vector<std::string> haystacks, needles;
        needles.emplace_back("is");
        needles.emplace_back("his");

        haystacks.emplace_back("this is his, that is his, those are his, these are his");
        haystacks.emplace_back("his is this, his is that, his are those, his are these");
        haystacks.emplace_back(R"(
        1 is this 2 is this 3 is this 4 is this 5 is this 6 is this 7 is this 8 is this
        9 is this 10 is this 11 is this 12 is this 13 is this 14 is this 15 is this 16 is this
        )");

        test_find_many_on(haystacks, needles, base_operator, simd_operator, extra_args...);
    }

    // Try even simpler alphabets
    {
        std::vector<std::string> haystacks, needles;
        needles.emplace_back("ab");
        needles.emplace_back("aba");

        haystacks.emplace_back("abababababababababababababababababababababababababababababababababab");
        haystacks.emplace_back("abbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabbaabba");

        test_find_many_on(haystacks, needles, base_operator, simd_operator, extra_args...);
    }

    // Try a combination of very short and very long needles
    {
        std::vector<std::string> haystacks, needles;
        needles.emplace_back("a");
        needles.emplace_back("b");
        needles.emplace_back("abracadabra");

        haystacks.emplace_back("abracadabra");
        haystacks.emplace_back("abracadabracadabra");

        test_find_many_on(haystacks, needles, base_operator, simd_operator, extra_args...);
    }
}

/**
 *  @brief Fuzzy test for multi-pattern exact search algorithms using randomly-generated haystacks and needles.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                    arrow_strings_tape_t const &haystacks_tape, arrow_strings_tape_t const &needles_tape,
                    extra_args_ &&...extra_args) {

    using match_t = find_many_match_t;
    unified_vector<match_t> results_base, results_simd;
    unified_vector<size_t> counts_base, counts_simd;

    counts_base.resize(haystacks_tape.size());
    counts_simd.resize(haystacks_tape.size());

    // Build the matchers
    sz_assert_(base_operator.try_build(needles_tape.view()) == status_t::success_k);
    sz_assert_(simd_operator.try_build(needles_tape.view()) == status_t::success_k);

    // Count the number of matches with both backends
    span<size_t> counts_base_span {counts_base.data(), counts_base.size()};
    span<size_t> counts_simd_span {counts_simd.data(), counts_simd.size()};
    status_t status_count_base = base_operator.try_count(haystacks_tape.view(), counts_base_span);
    status_t status_count_simd = simd_operator.try_count(haystacks_tape.view(), counts_simd_span, extra_args...);
    sz_assert_(status_count_base == status_t::success_k);
    sz_assert_(status_count_simd == status_t::success_k);
    size_t total_count_base = std::accumulate(counts_base.begin(), counts_base.end(), 0);
    size_t total_count_simd = std::accumulate(counts_simd.begin(), counts_simd.end(), 0);
    sz_assert_(total_count_base == total_count_simd);
    sz_assert_(std::equal(counts_base.begin(), counts_base.end(), counts_simd.begin()));

    // Compute with both backends
    results_base.resize(total_count_base);
    results_simd.resize(total_count_simd);
    size_t count_base = 0, count_simd = 0;
    status_t status_base = base_operator.try_find(haystacks_tape.view(), counts_base_span, results_base);
    status_t status_simd = simd_operator.try_find(haystacks_tape.view(), counts_simd_span, results_simd, extra_args...);
    sz_assert_(status_base == status_t::success_k);
    sz_assert_(status_simd == status_t::success_k);
    sz_assert_(count_base == count_simd);

    // Individually log the failed results
    std::sort(results_base.begin(), results_base.end(), match_t::less_globally);
    std::sort(results_simd.begin(), results_simd.end(), match_t::less_globally);
    for (std::size_t i = 0; i != results_base.size(); ++i) {
        sz_assert_(results_base[i].haystack_index == results_simd[i].haystack_index);
        sz_assert_(results_base[i].needle_index == results_simd[i].needle_index);
        sz_assert_(results_base[i].needle.data() == results_simd[i].needle.data());
    }

    base_operator.reset();
    simd_operator.reset();
}

/**
 *  @brief Fuzzy test for multi-pattern exact search algorithms using randomly-generated haystacks and needles.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_fuzzy(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                          fuzzy_config_t needles_config = {}, fuzzy_config_t haystacks_config = {},
                          std::size_t iterations = 10, extra_args_ &&...extra_args) {

    std::vector<std::string> haystacks_array, needles_array;
    arrow_strings_tape_t haystacks_tape, needles_tape;

    // Generate some random strings, using a small alphabet
    for (std::size_t iteration_idx = 0; iteration_idx < iterations; ++iteration_idx) {
        randomize_strings(haystacks_config, haystacks_array, haystacks_tape);
        randomize_strings(needles_config, needles_array, needles_tape, true);
        test_find_many(base_operator, simd_operator, haystacks_tape, needles_tape, extra_args...);
    }
}

/**
 *  @brief  Fuzzy test for multi-pattern exact search algorithms using randomly-generated haystacks,
 *          and using incrementally longer potentially-overlapping substrings as needles.
 */
template <typename base_operator_, typename simd_operator_, typename... extra_args_>
void test_find_many_prefixes(base_operator_ &&base_operator, simd_operator_ &&simd_operator,
                             fuzzy_config_t haystacks_config, std::size_t needle_length_limit,
                             std::size_t iterations = 10, extra_args_ &&...extra_args) {

    std::vector<std::string> haystacks_array;
    std::vector<std::string_view> needles_array;
    arrow_strings_tape_t haystacks_tape, needles_tape;

    for (std::size_t iteration_idx = 0; iteration_idx < iterations; ++iteration_idx) {
        randomize_strings(haystacks_config, haystacks_array, haystacks_tape);

        // Pick various substrings as needles from the first haystack
        needles_array.resize(std::min(haystacks_array[0].size(), needle_length_limit));
        for (std::size_t i = 0; i != needles_array.size(); ++i)
            needles_array[i] = std::string_view(haystacks_array[0]).substr(0, i + 1);
        needles_tape.try_assign(needles_array.data(), needles_array.data() + needles_array.size());

        test_find_many(base_operator, simd_operator, haystacks_tape, needles_tape, extra_args...);
    }
}

/**
 *  @brief  Tests the multi-pattern exact substring search algorithm
 *          against a baseline implementation for predefined and random inputs.
 */
void test_find_many_equivalence() {

    cpu_specs_t default_cpu_specs;
    fuzzy_config_t needles_short_config, needles_long_config, haystacks_config;
    haystacks_config.batch_size = default_cpu_specs.cores_total() * 4;
    haystacks_config.max_string_length = default_cpu_specs.l3_bytes;

    needles_short_config.min_string_length = 1;
    needles_short_config.max_string_length = 4;
    needles_short_config.batch_size =
        std::pow(needles_short_config.alphabet.size(), needles_short_config.max_string_length);

    needles_long_config.min_string_length = 3;
    needles_long_config.max_string_length = 6;
    needles_long_config.batch_size =
        std::pow(needles_long_config.alphabet.size(), needles_long_config.max_string_length);

#if SZ_USE_CUDA
    gpu_specs_t first_gpu_specs = *gpu_specs();
#endif

    // Single-threaded serial Aho-Corasick implementation
    test_find_many_fixed(find_many_baselines_t {}, find_many_u32_serial_t {});

    // Multi-threaded parallel Aho-Corasick implementation
    for (std::size_t threads : {2, 3, 4, 5}) {
        alignas(fu::default_alignment_k) fu::basic_pool_t pool;
        if (!pool.try_spawn(threads)) throw std::runtime_error("Failed to spawn thread pool.");
        static_assert(executor_like<fu::basic_pool_t>);
        test_find_many_fixed(find_many_baselines_t {}, find_many_u32_parallel_t {}, pool);
    }

    // Let's reuse a thread-pool to amortize the cost of spawning threads.
    alignas(fu::default_alignment_k) fu::basic_pool_t pool;
    if (!pool.try_spawn(std::thread::hardware_concurrency())) throw std::runtime_error("Failed to spawn thread pool.");
    static_assert(executor_like<fu::basic_pool_t>);

#if SZ_USE_CUDA
    test_find_many_fixed(find_many_baselines_t {}, find_many_u32_cuda_t {}, cuda_executor_t {});
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_cuda_t {}, needles_short_config, haystacks_config, 1,
                         cuda_executor_t {});
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_cuda_t {}, needles_long_config, haystacks_config, 1,
                         cuda_executor_t {});
    test_find_many_prefixes(find_many_baselines_t {}, find_many_u32_cuda_t {}, haystacks_config, 1024, 1,
                            cuda_executor_t {});
#endif

    // Fuzzy tests with random inputs
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_serial_t {}, needles_short_config, haystacks_config,
                         1);
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_serial_t {}, needles_long_config, haystacks_config, 1);
    test_find_many_prefixes(find_many_baselines_t {}, find_many_u32_serial_t {}, haystacks_config, 1024, 1);

    // Fuzzy tests with random inputs for multi-threaded CPU backend
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_parallel_t {}, needles_short_config, haystacks_config,
                         10, pool);
    test_find_many_fuzzy(find_many_baselines_t {}, find_many_u32_parallel_t {}, needles_long_config, haystacks_config,
                         10, pool);
    test_find_many_prefixes(find_many_baselines_t {}, find_many_u32_parallel_t {}, haystacks_config, 1024, 10, pool);
}

} // namespace scripts
} // namespace stringzillas
} // namespace ashvardanian
