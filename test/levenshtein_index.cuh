#include <stringzillas/levenshtein_index.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace szs = ashvardanian::stringzillas;
namespace sz = ashvardanian::stringzilla;

struct failing_allocator_state_t {
    bool fail = false;
};

template <typename value_type_>
struct failing_allocator_t {
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;

    failing_allocator_state_t *state = nullptr;

    failing_allocator_t() noexcept = default;
    explicit failing_allocator_t(failing_allocator_state_t *state_arg) noexcept : state(state_arg) {}
    template <typename other_type_>
    failing_allocator_t(failing_allocator_t<other_type_> const &other) noexcept : state(other.state) {}

    value_type *allocate(std::size_t count) const noexcept {
        if (state && state->fail) return nullptr;
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(value_type)) return nullptr;
        return static_cast<value_type *>(std::malloc(count * sizeof(value_type)));
    }
    void deallocate(value_type *address, std::size_t) const noexcept { std::free(address); }

    template <typename other_type_>
    bool operator==(failing_allocator_t<other_type_> const &other) const noexcept {
        return state == other.state;
    }
    template <typename other_type_>
    bool operator!=(failing_allocator_t<other_type_> const &other) const noexcept {
        return !(*this == other);
    }
};

template <typename string_type_>
static std::size_t reference_distance(string_type_ const &first, string_type_ const &second) {
    std::vector<std::size_t> previous(first.size() + 1), current(first.size() + 1);
    for (std::size_t column = 0; column <= first.size(); ++column) previous[column] = column;
    for (std::size_t row = 1; row <= second.size(); ++row) {
        current[0] = row;
        for (std::size_t column = 1; column <= first.size(); ++column)
            current[column] = std::min({previous[column] + 1, current[column - 1] + 1,
                                        previous[column - 1] + (first[column - 1] != second[row - 1])});
        previous.swap(current);
    }
    return previous.back();
}

static int test_levenshtein_index_unit_impl_() {
    std::vector<std::string> dictionary;
    for (std::size_t length = 0; length <= 8; ++length)
        for (std::size_t bits = 0; bits != (std::size_t(1) << length); ++bits) {
            std::string word(length, '\0');
            for (std::size_t index = 0; index != length; ++index) word[index] = "\0a"[(bits >> index) & 1];
            dictionary.push_back(std::move(word));
        }
    dictionary.push_back(dictionary[42]); // Duplicate values must retain distinct IDs.
    dictionary.push_back(std::string(70, 'x'));
    dictionary.push_back(std::string(70, 'x'));
    dictionary.back()[17] = 'y';
    dictionary.back()[53] = 'z';
    dictionary.push_back(std::string(68, 'x'));

    szs::levenshtein_index<> index;
    if (index.try_build(dictionary, 4, 128) != sz::status_t::success_k) return 1;
    szs::levenshtein_index<>::scratch_t scratch;
    szs::levenshtein_index<>::matches_t matches;
    std::size_t checks = 0;
    for (auto const &query : dictionary)
        for (std::uint8_t bound = 0; bound <= 4; ++bound) {
            if (index.find({query.data(), query.size()}, bound, scratch, matches) != sz::status_t::success_k) return 2;
            std::vector<std::pair<std::uint32_t, std::uint8_t>> actual, expected;
            for (auto const &match : matches) actual.emplace_back(match.id, match.distance);
            for (std::uint32_t id = 0; id != dictionary.size(); ++id) {
                std::size_t const score = reference_distance(dictionary[id], query);
                if (score <= bound) expected.emplace_back(id, static_cast<std::uint8_t>(score));
                ++checks;
            }
            std::sort(actual.begin(), actual.end());
            if (actual != expected) {
                std::cerr << "mismatch query_length=" << query.size() << " bound=" << unsigned(bound)
                          << " actual=" << actual.size() << " expected=" << expected.size() << '\n';
                return 3;
            }
        }

    // One immutable index must be safely searchable from independent worker scratch spaces.
    bool concurrent_ok[2] = {false, false};
    std::thread workers[2];
    for (std::size_t worker = 0; worker != 2; ++worker)
        workers[worker] = std::thread([&, worker] {
            szs::levenshtein_index<>::scratch_t worker_scratch;
            szs::levenshtein_index<>::matches_t worker_matches;
            auto const &query = dictionary[dictionary.size() - 1 - worker];
            concurrent_ok[worker] = index.find({query.data(), query.size()}, 4, worker_scratch, worker_matches) ==
                                        sz::status_t::success_k &&
                                    worker_matches.size() != 0;
            // Every query is present in the dictionary, so at least one exact match is required.
        });
    for (auto &worker : workers) worker.join();
    if (!concurrent_ok[0] || !concurrent_ok[1]) return 4;

    // An explicit deletion cutoff routes unusually long words through the same exact trie even for k<=2.
    szs::levenshtein_index<> fallback_index;
    if (fallback_index.try_build(dictionary, 2, 64) != sz::status_t::success_k) return 5;
    auto const &long_query = dictionary[dictionary.size() - 2];
    if (fallback_index.find({long_query.data(), long_query.size()}, 2, scratch, matches) !=
        sz::status_t::success_k)
        return 6;
    std::vector<std::pair<std::uint32_t, std::uint8_t>> actual_fallback, expected_fallback;
    for (auto const &match : matches) actual_fallback.emplace_back(match.id, match.distance);
    for (std::uint32_t id = 0; id != dictionary.size(); ++id) {
        std::size_t const score = reference_distance(dictionary[id], long_query);
        if (score <= 2) expected_fallback.emplace_back(id, static_cast<std::uint8_t>(score));
    }
    std::sort(actual_fallback.begin(), actual_fallback.end());
    if (actual_fallback != expected_fallback) return 7;

    // Automatic planning avoids quadratic deletion neighborhoods for uniformly long dictionaries.
    std::vector<std::string> long_dictionary = {std::string(100, 'a'), std::string(100, 'b')};
    szs::levenshtein_index<> automatic_index;
    if (automatic_index.try_build(long_dictionary, 2) != sz::status_t::success_k ||
        automatic_index.records_count() != 0)
        return 8;
    if (automatic_index.find({long_dictionary[0].data(), long_dictionary[0].size()}, 2, scratch, matches) !=
            sz::status_t::success_k ||
        matches.size() != 1 || matches[0].id != 0 || matches[0].distance != 0)
        return 9;

    // Exhaust the same machinery over a mixed-width Unicode alphabet, including duplicate IDs.
    char32_t const unicode_alphabet[] = {U'a', U'咖', U'🦖'};
    std::vector<std::u32string> unicode_dictionary;
    for (std::size_t length = 0, combinations = 1; length <= 5; ++length, combinations *= 3)
        for (std::size_t encoded = 0; encoded != combinations; ++encoded) {
            std::u32string word(length, U'\0');
            std::size_t digits = encoded;
            for (std::size_t position = 0; position != length; ++position, digits /= 3)
                word[position] = unicode_alphabet[digits % 3];
            unicode_dictionary.push_back(std::move(word));
        }
    unicode_dictionary.push_back(unicode_dictionary[42]);
    szs::basic_levenshtein_index<char32_t> unicode_index;
    if (unicode_index.try_build(unicode_dictionary, 4) != sz::status_t::success_k) return 10;
    szs::basic_levenshtein_index<char32_t>::scratch_t unicode_scratch;
    szs::basic_levenshtein_index<char32_t>::matches_t unicode_matches;
    for (auto const &query : unicode_dictionary)
        for (std::uint8_t bound = 0; bound <= 4; ++bound) {
            if (unicode_index.find({query.data(), query.size()}, bound, unicode_scratch, unicode_matches) !=
                sz::status_t::success_k)
                return 11;
            std::vector<std::pair<std::uint32_t, std::uint8_t>> actual, expected;
            for (auto const &match : unicode_matches) actual.emplace_back(match.id, match.distance);
            for (std::uint32_t id = 0; id != unicode_dictionary.size(); ++id) {
                std::size_t const score = reference_distance(unicode_dictionary[id], query);
                if (score <= bound) expected.emplace_back(id, static_cast<std::uint8_t>(score));
                ++checks;
            }
            std::sort(actual.begin(), actual.end());
            if (actual != expected) return 12;
        }

    // The UTF-8 facade validates once and measures edits in codepoints, unlike the byte index.
    std::vector<std::string> utf8_dictionary = {"caf\xC3\xA9", "cafe", "\xE5\x92\x96\xE5\x95\xA1",
                                                "\xE5\x92\x96\xE9\x9D\x9E", "\xF0\x9F\xA6\x96zilla",
                                                "caf\xC3\xA9"};
    szs::levenshtein_index_utf8<> utf8_index;
    if (utf8_index.try_build(utf8_dictionary, 4) != sz::status_t::success_k) return 13;
    szs::levenshtein_index_utf8<>::scratch_t utf8_scratch;
    szs::levenshtein_index_utf8<>::matches_t utf8_matches;
    std::string const coffee_query = "\xE5\x92\x96\xE5\x95\xA1";
    if (utf8_index.find({coffee_query.data(), coffee_query.size()}, 1, utf8_scratch, utf8_matches) !=
            sz::status_t::success_k ||
        utf8_matches.size() != 2)
        return 14;
    std::string const malformed = "\xF0\x9F";
    if (utf8_index.find({malformed.data(), malformed.size()}, 1, utf8_scratch, utf8_matches) !=
            sz::status_t::invalid_utf8_k ||
        utf8_matches.size() != 0)
        return 15;
    std::vector<std::string> malformed_dictionary = {"valid", malformed};
    if (utf8_index.try_build(malformed_dictionary, 2) != sz::status_t::invalid_utf8_k || utf8_index.size() != 6)
        return 16;

    // Exercise every trie path, including bounds above the small packed-state cases.
    std::vector<std::string> trie_dictionary;
    for (std::size_t id = 0; id != 32; ++id) {
        std::string word(18 + id % 9, 'a');
        for (std::size_t position = 0; position != word.size(); ++position)
            word[position] = "abcd"[(id * 7 + position * 3 + position / 5) % 4];
        trie_dictionary.push_back(std::move(word));
    }
    trie_dictionary.push_back(trie_dictionary[7]);
    szs::levenshtein_index<> trie_index;
    if (trie_index.try_build(trie_dictionary, 15, 0) != sz::status_t::success_k || trie_index.trie_bytes() == 0)
        return 17;
    std::vector<std::string> trie_queries = trie_dictionary;
    for (std::size_t id = 0; id != 12; ++id) {
        std::string query = trie_dictionary[id];
        query[id % query.size()] = 'z';
        query.insert(query.begin() + (id * 3) % query.size(), 'y');
        trie_queries.push_back(std::move(query));
    }
    for (auto const &query : trie_queries)
        for (std::uint8_t bound = 0; bound <= 15; ++bound) {
            if (trie_index.find({query.data(), query.size()}, bound, scratch, matches) != sz::status_t::success_k)
                return 18;
            std::vector<std::pair<std::uint32_t, std::uint8_t>> actual, expected;
            for (auto const &match : matches) actual.emplace_back(match.id, match.distance);
            for (std::uint32_t id = 0; id != trie_dictionary.size(); ++id) {
                std::size_t const score = reference_distance(trie_dictionary[id], query);
                if (score <= bound) expected.emplace_back(id, static_cast<std::uint8_t>(score));
                ++checks;
            }
            std::sort(actual.begin(), actual.end());
            if (actual != expected) return 19;
        }

    // A rejected rebuild must leave the previous immutable index available.
    if (trie_index.try_build(trie_dictionary, std::numeric_limits<std::uint8_t>::max()) !=
            sz::status_t::unexpected_dimensions_k ||
        trie_index.size() != trie_dictionary.size())
        return 20;
    if (trie_index.find({trie_dictionary[0].data(), trie_dictionary[0].size()}, 0, scratch, matches) !=
            sz::status_t::success_k ||
        matches.size() != 1 || matches[0].id != 0)
        return 21;

    // Search-time allocation failures must be reported, never translated into a missing match.
    failing_allocator_state_t failing_state;
    using failing_allocator_char_t = failing_allocator_t<char>;
    using failing_index_t = szs::levenshtein_index<failing_allocator_char_t>;
    failing_allocator_char_t failing_allocator {&failing_state};
    std::vector<std::string> failing_dictionary = {std::string(65, 'a')};
    failing_index_t failing_index {failing_allocator};
    if (failing_index.try_build(failing_dictionary, 2, 65) != sz::status_t::success_k) return 22;
    failing_index_t::scratch_t failing_scratch {failing_allocator};
    failing_index_t::matches_t failing_matches {failing_allocator};
    std::string non_candidate(65, 'b');
    if (failing_index.find({non_candidate.data(), non_candidate.size()}, 2, failing_scratch, failing_matches) !=
            sz::status_t::success_k ||
        failing_matches.size() != 0)
        return 23;
    failing_state.fail = true;
    if (failing_index.find({failing_dictionary[0].data(), failing_dictionary[0].size()}, 2, failing_scratch,
                           failing_matches) != sz::status_t::bad_alloc_k)
        return 24;
    failing_state.fail = false;
    if (failing_index.find({failing_dictionary[0].data(), failing_dictionary[0].size()}, 2, failing_scratch,
                           failing_matches) != sz::status_t::success_k ||
        failing_matches.size() != 1 || failing_matches[0].id != 0 || failing_matches[0].distance != 0)
        return 25;

    std::cout << "OK: " << checks << " exhaustive memberships, records=" << index.records_count()
              << " index_bytes=" << index.index_bytes() << '\n';
    return 0;
}

static void test_levenshtein_index_unit() { verify(test_levenshtein_index_unit_impl_() == 0); }
