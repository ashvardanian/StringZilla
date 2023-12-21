#include <cassert>  // assertions
#include <cstring>  // `std::memcpy`
#include <iterator> // `std::distance`

#define SZ_USE_X86_AVX2 0
#define SZ_USE_X86_AVX512 1
#define SZ_USE_ARM_NEON 0
#define SZ_USE_ARM_SVE 0

#include <string>                      // Baseline
#include <string_view>                 // Baseline
#include <stringzilla/stringzilla.hpp> // Contender

namespace sz = av::sz;

template <typename stl_matcher_, typename sz_matcher_>
void eval(std::string_view haystack_pattern, std::string_view needle_stl, std::size_t misalignment) {
    constexpr std::size_t max_repeats = 128;
    alignas(64) char haystack[misalignment + max_repeats * haystack_pattern.size()];

    for (std::size_t repeats = 0; repeats != 128; ++repeats) {
        std::size_t haystack_length = (repeats + 1) * haystack_pattern.size();
        std::memcpy(haystack + misalignment + repeats * haystack_pattern.size(), haystack_pattern.data(),
                    haystack_pattern.size());

        // Convert to string views
        auto haystack_stl = std::string_view(haystack + misalignment, haystack_length);
        auto haystack_sz = sz::string_view(haystack + misalignment, haystack_length);
        auto needle_sz = sz::string_view(needle_stl.data(), needle_stl.size());

        // Wrap into ranges
        auto matches_stl = stl_matcher_(haystack_stl, needle_stl);
        auto matches_sz = sz_matcher_(haystack_sz, needle_sz);
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
    }
}

void eval(std::string_view haystack_pattern, std::string_view needle_stl, std::size_t misalignment) {

    eval<                                                      //
        sz::range_matches<std::string_view, sz::matcher_find>, //
        sz::range_matches<sz::string_view, sz::matcher_find>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                               //
        sz::reverse_range_matches<std::string_view, sz::matcher_rfind>, //
        sz::reverse_range_matches<sz::string_view, sz::matcher_rfind>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                               //
        sz::range_matches<std::string_view, sz::matcher_find_first_of>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_of>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                                      //
        sz::reverse_range_matches<std::string_view, sz::matcher_find_last_of>, //
        sz::reverse_range_matches<sz::string_view, sz::matcher_find_last_of>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                                   //
        sz::range_matches<std::string_view, sz::matcher_find_first_not_of>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_not_of>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                                          //
        sz::reverse_range_matches<std::string_view, sz::matcher_find_last_not_of>, //
        sz::reverse_range_matches<sz::string_view, sz::matcher_find_last_not_of>>( //
        haystack_pattern, needle_stl, misalignment);
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