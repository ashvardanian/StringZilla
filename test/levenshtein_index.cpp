#include <stringzillas/levenshtein_index.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace szs = ashvardanian::stringzillas;
namespace sz = ashvardanian::stringzilla;

static std::uint8_t reference_distance(std::string const &first, std::string const &second) {
    std::vector<std::uint16_t> previous(first.size() + 1), current(first.size() + 1);
    for (std::size_t column = 0; column <= first.size(); ++column)
        previous[column] = static_cast<std::uint16_t>(column);
    for (std::size_t row = 1; row <= second.size(); ++row) {
        current[0] = static_cast<std::uint16_t>(row);
        for (std::size_t column = 1; column <= first.size(); ++column)
            current[column] = std::min(
                {static_cast<std::uint16_t>(previous[column] + 1),
                 static_cast<std::uint16_t>(current[column - 1] + 1),
                 static_cast<std::uint16_t>(previous[column - 1] + (first[column - 1] != second[row - 1]))});
        previous.swap(current);
    }
    return static_cast<std::uint8_t>(previous[first.size()]);
}

static std::vector<std::string> binary_strings(std::size_t max_length) {
    std::vector<std::string> strings = {""};
    for (std::size_t length = 1; length <= max_length; ++length)
        for (std::size_t bits = 0; bits != (std::size_t(1) << length); ++bits) {
            std::string value(length, 'a');
            for (std::size_t position = 0; position != length; ++position)
                value[position] = ((bits >> position) & 1) ? 'b' : 'a';
            strings.push_back(std::move(value));
        }
    return strings;
}

static bool check_matches(szs::levenshtein_index<> const &index, std::vector<std::string> const &dictionary,
                          std::vector<std::string> const &queries, std::uint8_t max_bound) {
    szs::levenshtein_index<>::scratch_t scratch;
    szs::levenshtein_index<>::matches_t matches;
    for (std::string const &query : queries)
        for (std::uint8_t bound = 0; bound <= max_bound; ++bound) {
            if (index.find({query.data(), query.size()}, bound, scratch, matches) != sz::status_t::success_k)
                return false;
            std::vector<std::pair<std::uint32_t, std::uint8_t>> actual;
            for (auto const &match : matches) actual.emplace_back(match.id, match.distance);
            std::sort(actual.begin(), actual.end());

            std::vector<std::pair<std::uint32_t, std::uint8_t>> expected;
            for (std::uint32_t id = 0; id != dictionary.size(); ++id) {
                std::uint8_t const distance = reference_distance(dictionary[id], query);
                if (distance <= bound) expected.emplace_back(id, distance);
            }
            if (actual != expected) return false;
        }
    return true;
}

int main() {
    std::vector<std::string> dictionary = binary_strings(5);
    dictionary.push_back("book");
    dictionary.push_back("book");
    dictionary.push_back(std::string("a\0b", 3));
    dictionary.push_back(std::string(70, 'x'));
    std::vector<std::string> queries = binary_strings(5);
    queries.push_back("cook");
    queries.push_back(std::string("a\0c", 3));
    queries.push_back(std::string(69, 'x') + "y");

    // Bounds above two search the complete prefix tree.
    szs::levenshtein_index<> index;
    if (index.try_build(dictionary, 4) != sz::status_t::success_k ||
        !check_matches(index, dictionary, queries, 4))
        return 1;

    // A lower cutoff mixes deletion lookup for short strings with the tree for long strings.
    szs::levenshtein_index<> mixed_index;
    if (mixed_index.try_build(dictionary, 2, 4) != sz::status_t::success_k ||
        !check_matches(mixed_index, dictionary, queries, 2))
        return 2;

    szs::levenshtein_index<>::scratch_t scratch;
    szs::levenshtein_index<>::matches_t matches;
    std::string const query = "book";
    if (index.find({query.data(), query.size()}, 5, scratch, matches) != sz::status_t::unexpected_dimensions_k)
        return 3;

    // Failed rebuilds leave the previous immutable index intact.
    if (index.try_build(dictionary, std::numeric_limits<std::uint8_t>::max()) !=
            sz::status_t::unexpected_dimensions_k ||
        index.size() != dictionary.size())
        return 4;
    if (index.find({query.data(), query.size()}, 0, scratch, matches) != sz::status_t::success_k ||
        matches.size() != 2)
        return 5;
    return 0;
}
