#include <stringzillas/levenshtein_index.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace szs = ashvardanian::stringzillas;
namespace sz = ashvardanian::stringzilla;

int main() {
    std::vector<std::string> dictionary = {"book", "back", "book", "boon", "", std::string("a\0b", 3)};
    szs::levenshtein_index<> index;
    if (index.try_build(dictionary) != sz::status_t::success_k) return 1;

    szs::levenshtein_index<>::scratch_t scratch;
    szs::levenshtein_index<>::matches_t matches;
    std::string const query = "book";
    if (index.find({query.data(), query.size()}, 0, scratch, matches) != sz::status_t::success_k) return 2;
    std::vector<std::uint32_t> ids;
    for (auto const &match : matches) {
        if (match.distance != 0) return 3;
        ids.push_back(match.id);
    }
    std::sort(ids.begin(), ids.end());
    if (ids != std::vector<std::uint32_t>({0, 2})) return 4;

    std::string const missing = "cook";
    if (index.find({missing.data(), missing.size()}, 0, scratch, matches) != sz::status_t::success_k ||
        matches.size() != 0)
        return 5;
    if (index.find({query.data(), query.size()}, 1, scratch, matches) != sz::status_t::unexpected_dimensions_k ||
        matches.size() != 0)
        return 6;

    // Failed rebuilds leave the previous immutable index intact.
    if (index.try_build(dictionary, 1) != sz::status_t::unexpected_dimensions_k || index.size() != dictionary.size())
        return 7;
    if (index.find({query.data(), query.size()}, 0, scratch, matches) != sz::status_t::success_k ||
        matches.size() != 2)
        return 8;
    return 0;
}
