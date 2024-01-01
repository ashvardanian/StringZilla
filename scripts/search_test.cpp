#include <cassert>  // assertions
#include <cstdio>   // `std::printf`
#include <cstring>  // `std::memcpy`
#include <iterator> // `std::distance`
#include <vector>   // `std::vector`

#define SZ_USE_X86_AVX2 0
#define SZ_USE_X86_AVX512 0
#define SZ_USE_ARM_NEON 0
#define SZ_USE_ARM_SVE 0

#include <string>                      // Baseline
#include <string_view>                 // Baseline
#include <stringzilla/stringzilla.hpp> // Contender

namespace sz = ashvardanian::stringzilla;
using sz::literals::operator""_sz;

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
        auto matches_stl = stl_matcher_(haystack_stl, {needle_stl});
        auto matches_sz = sz_matcher_(haystack_sz, {needle_sz});
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

    eval<                                                        //
        sz::range_rmatches<std::string_view, sz::matcher_rfind>, //
        sz::range_rmatches<sz::string_view, sz::matcher_rfind>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                               //
        sz::range_matches<std::string_view, sz::matcher_find_first_of>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_of>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                               //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_of>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_of>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                                   //
        sz::range_matches<std::string_view, sz::matcher_find_first_not_of>, //
        sz::range_matches<sz::string_view, sz::matcher_find_first_not_of>>( //
        haystack_pattern, needle_stl, misalignment);

    eval<                                                                   //
        sz::range_rmatches<std::string_view, sz::matcher_find_last_not_of>, //
        sz::range_rmatches<sz::string_view, sz::matcher_find_last_not_of>>( //
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
    // eval("a", "a");
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

    // Check more advanced composite operations:
    assert("abbccc"_sz.split("bb").before.size() == 1);
    assert("abbccc"_sz.split("bb").match.size() == 2);
    assert("abbccc"_sz.split("bb").after.size() == 3);
    assert("abbccc"_sz.split("bb").before == "a");
    assert("abbccc"_sz.split("bb").match == "bb");
    assert("abbccc"_sz.split("bb").after == "ccc");

    assert(""_sz.find_all(".").size() == 0);
    assert("a.b.c.d"_sz.find_all(".").size() == 3);
    assert("a.,b.,c.,d"_sz.find_all(".,").size() == 3);
    assert("a.,b.,c.,d"_sz.rfind_all(".,").size() == 3);
    assert("a.b,c.d"_sz.find_all(sz::character_set(".,")).size() == 3);
    assert("a...b...c"_sz.rfind_all("..", true).size() == 4);

    auto finds = "a.b.c"_sz.find_all(sz::character_set("abcd")).template to<std::vector<std::string>>();
    assert(finds.size() == 3);
    assert(finds[0] == "a");

    auto rfinds = "a.b.c"_sz.rfind_all(sz::character_set("abcd")).template to<std::vector<std::string>>();
    assert(rfinds.size() == 3);
    assert(rfinds[0] == "c");

    auto splits = ".a..c."_sz.split_all(sz::character_set(".")).template to<std::vector<std::string>>();
    assert(splits.size() == 5);
    assert(splits[0] == "");
    assert(splits[1] == "a");
    assert(splits[4] == "");

    assert(""_sz.split_all(".").size() == 1);
    assert(""_sz.rsplit_all(".").size() == 1);
    assert("a.b.c.d"_sz.split_all(".").size() == 4);
    assert("a.b.c.d"_sz.rsplit_all(".").size() == 4);
    assert("a.b.,c,d"_sz.split_all(".,").size() == 2);
    assert("a.b,c.d"_sz.split_all(sz::character_set(".,")).size() == 4);

    auto rsplits = ".a..c."_sz.rsplit_all(sz::character_set(".")).template to<std::vector<std::string>>();
    assert(rsplits.size() == 5);
    assert(rsplits[0] == "");
    assert(rsplits[1] == "c");
    assert(rsplits[4] == "");

    // Compare STL and StringZilla strings append functionality.
    std::string stl_string;
    sz::string sz_string;
    for (std::size_t length = 1; length != 200; ++length) {
        stl_string.push_back('a');
        sz_string.push_back('a');
        assert(sz::string_view(stl_string) == sz::string_view(sz_string));
    }

    return 0;
}