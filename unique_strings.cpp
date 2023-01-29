#include <random>
#include <optional>
#include <span>
#include <string_view>
#include <vector>
#include <string>
#include <map>
#include <bit> // `std::bit_ceil`
#include <unordered_set>
#include <unordered_map>

#include <benchmark/benchmark.h>

namespace bm = benchmark;
static constexpr char alphabet[] = "abcdefghijklmnopABCDEFGHIJKLMNOP";
static constexpr std::size_t alphabet_size_k = 32;
static constexpr std::size_t string_length_k = 5;
static constexpr std::size_t strings_count_k = 1'000'000;

std::string random_string(std::size_t length = string_length_k) {
    static std::random_device r;
    static std::default_random_engine e(r());
    std::string result(length, 'a');
    for (auto &c : result)
        c = alphabet[e() % alphabet_size_k];
    return result;
}

std::vector<std::string> random_strings(std::size_t n, std::size_t length = string_length_k) {
    std::vector<std::string> result(n);
    for (auto &s : result)
        s = random_string(length);
    return result;
}

std::map<std::string, int> first_offsets_junior(std::vector<std::string> strings) {
    std::map<std::string, int> offsets;
    for (int idx = 0; idx < strings.size(); idx++)
        if (offsets.find(strings[idx]) == offsets.end())
            offsets[strings[idx]] = idx;
    return offsets;
}

std::unordered_map<std::string, std::size_t> first_offsets_middle(std::vector<std::string> const &strings) {
    std::unordered_map<std::string, std::size_t> offsets;
    for (std::size_t idx = 0; idx < strings.size(); idx++)
        if (!offsets.contains(strings[idx]))
            offsets[strings[idx]] = idx;
    return offsets;
}

std::unordered_map<std::string_view, std::size_t> first_offsets_senior(std::span<std::string> strings) {
    std::unordered_map<std::string_view, std::size_t> offsets;
    for (std::size_t idx = 0; idx != strings.size(); ++idx)
        offsets.try_emplace(std::string_view(strings[idx]), idx);
    return offsets;
}

class flat_unordered_set_t {
    using string_ptr_t = std::string const *;
    string_ptr_t first {};
    std::vector<string_ptr_t> hashed {};

  public:
    flat_unordered_set_t(std::span<std::string> strings)
        : first(strings.data()), hashed(std::bit_ceil<std::size_t>(strings.size() * 1.3)) {
        std::fill_n(hashed.data(), hashed.size(), nullptr);
    }
    void try_emplace(std::string const &string) {
        auto hash = std::hash<std::string_view> {}(string);
        auto slot = hash & (hashed.size() - 1);
        while (hashed[slot] && *hashed[slot] != string)
            slot = (slot + 1) & (hashed.size() - 1);
        if (!hashed[slot])
            hashed[slot] = &string;
    }
    std::size_t operator[](std::string_view string) const noexcept {
        auto hash = std::hash<std::string_view> {}(string);
        auto slot = hash & (hashed.size() - 1);
        while (hashed[slot] && *hashed[slot] != string)
            slot = (slot + 1) & (hashed.size() - 1);
        return hashed[slot] ? hashed[slot] - first : std::numeric_limits<std::size_t>::max();
    }
};

std::optional<flat_unordered_set_t> first_offsets_enthusiast(std::span<std::string> strings) try {
    if (strings.empty())
        return {};
    flat_unordered_set_t offsets {strings};
    for (auto const &string : strings)
        offsets.try_emplace(string);
    return {std::move(offsets)};
}
catch (...) {
    return {};
}

template <typename functor_at>
static void bench(bm::State &state, functor_at &&functor) {
    auto strings = random_strings(strings_count_k);
    for (auto _ : state) {
        bm::DoNotOptimize(functor(strings));
    }
    state.counters["strings/s"] = bm::Counter(strings_count_k * state.iterations(), bm::Counter::kIsRate);
}

BENCHMARK_CAPTURE(bench, junior, first_offsets_junior)->MinTime(10);
BENCHMARK_CAPTURE(bench, middle, first_offsets_middle)->MinTime(10);
BENCHMARK_CAPTURE(bench, senior, first_offsets_senior)->MinTime(10);
BENCHMARK_CAPTURE(bench, enthusiast, first_offsets_enthusiast)->MinTime(10);

BENCHMARK_MAIN();