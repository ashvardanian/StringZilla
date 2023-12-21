#include <cassert>  // assertions
#include <cstring>  // `std::memcpy`
#include <iterator> // `std::distance`

#include <string>                      // Baseline
#include <string_view>                 // Baseline
#include <stringzilla/stringzilla.hpp> // Contender

namespace sz = av::sz;

void eval(std::string_view haystack_pattern, std::string_view needle_stl, std::size_t misalignment) {
    constexpr std::size_t max_repeats = 128;
    alignas(64) char haystack[max_repeats * haystack_pattern.size() + misalignment];

    for (std::size_t repeats = 0; repeats != 128; ++repeats) {
        std::memcpy(haystack + misalignment + repeats * haystack_pattern.size(), haystack_pattern.data(),
                    haystack_pattern.size());

        // Convert to string views
        auto haystack_stl = std::string_view(haystack + misalignment, repeats * haystack_pattern.size());
        auto haystack_sz = sz::string_view(haystack + misalignment, repeats * haystack_pattern.size());
        auto needle_sz = sz::string_view(needle_stl.data(), needle_stl.size());

        // Wrap into ranges
        auto matches_stl = sz::search_matches(haystack_stl, needle_stl);
        auto matches_sz = sz::search_matches(haystack_sz, needle_sz);
        auto begin_stl = matches_stl.begin();
        auto begin_sz = matches_sz.begin();
        auto end_stl = matches_stl.end();
        auto end_sz = matches_sz.end();
        auto count_stl = std::distance(begin_stl, end_stl);
        auto count_sz = std::distance(begin_sz, end_sz);

        // Compare results
        for (; begin_stl != end_stl && begin_sz != end_sz; ++begin_stl, ++begin_sz) {
            auto match_stl = *begin_stl;
            auto match_sz = *begin_sz;
            assert(match_stl.data() == match_sz.data());
        }

        // If one range is not finished, assert failure
        assert(count_stl == count_sz);
        assert(begin_stl == end_stl && begin_sz == end_sz);

        // Wrap into reverse-order ranges
        auto reverse_matches_stl = sz::reverse_search_matches(haystack_stl, needle_stl);
        auto reverse_matches_sz = sz::reverse_search_matches(haystack_sz, needle_sz);
        auto reverse_begin_stl = reverse_matches_stl.begin();
        auto reverse_begin_sz = reverse_matches_sz.begin();
        auto reverse_end_stl = reverse_matches_stl.end();
        auto reverse_end_sz = reverse_matches_sz.end();
        auto reverse_count_stl = std::distance(reverse_begin_stl, reverse_end_stl);
        auto reverse_count_sz = std::distance(reverse_begin_sz, reverse_end_sz);

        // Compare reverse-order results
        for (; reverse_begin_stl != reverse_end_stl && reverse_begin_sz != reverse_end_sz;
             ++reverse_begin_stl, ++reverse_begin_sz) {
            auto reverse_match_stl = *reverse_begin_stl;
            auto reverse_match_sz = *reverse_begin_sz;
            assert(reverse_match_stl.data() == reverse_match_sz.data());
        }

        // If one range is not finished, assert failure
        assert(reverse_count_stl == reverse_count_sz);
        assert(reverse_begin_stl == reverse_end_stl && reverse_begin_sz == reverse_end_sz);

        // Make sure number of elements is equal
        assert(count_stl == reverse_count_stl);
        assert(count_sz == reverse_count_sz);
    }
}

void eval(std::string_view haystack_pattern, std::string_view needle_stl) {
    eval(haystack_pattern, needle_stl, 0);
    eval(haystack_pattern, needle_stl, 1);
    eval(haystack_pattern, needle_stl, 2);
    eval(haystack_pattern, needle_stl, 3);
}

int main(int, char const **) {
    std::printf("Hi Ash! ... or is it someone else?!\n");

    std::string_view alphabet = "abcdefghijklmnopqrstuvwxyz";                                         // 26 characters
    std::string_view base64 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-";     // 64 characters
    std::string_view common = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=@$%"; // 68 characters

    // When haystack is only formed of needles:
    eval("a", "a");
    eval("ab", "ab");
    eval("abc", "abc");
    eval("abcd", "abcd");
    eval(alphabet, alphabet);
    eval(base64, base64);
    eval(common, common);

    // When haystack is formed of equidistant needles:
    eval("ab", "a");
    eval("abc", "a");
    eval("abcd", "a");

    // When matches occur in between pattern words:
    eval("ab", "ba");
    eval("abc", "ca");
    eval("abcd", "da");

    return 0;
}