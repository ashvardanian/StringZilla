#include <algorithm> // `std::transform`
#include <cassert>   // assertions
#include <cstdio>    // `std::printf`
#include <cstring>   // `std::memcpy`
#include <iterator>  // `std::distance`
#include <vector>    // `std::vector`

#define SZ_USE_X86_AVX2 0
#define SZ_USE_X86_AVX512 1
#define SZ_USE_ARM_NEON 0
#define SZ_USE_ARM_SVE 0

#include <string>                      // Baseline
#include <string_view>                 // Baseline
#include <stringzilla/stringzilla.hpp> // Contender


static void
test_util() {
    assert(sz_leading_zeros64(0x0000000000000000ull) == 64);

    assert(sz_leading_zeros64(0x0000000000000001ull) == 63);

    assert(sz_leading_zeros64(0x0000000000000002ull) == 62);
    assert(sz_leading_zeros64(0x0000000000000003ull) == 62);

    assert(sz_leading_zeros64(0x0000000000000004ull) == 61);
    assert(sz_leading_zeros64(0x0000000000000007ull) == 61);

    assert(sz_leading_zeros64(0x8000000000000000ull) == 0);
    assert(sz_leading_zeros64(0x8000000000000001ull) == 0);
    assert(sz_leading_zeros64(0xffffffffffffffffull) == 0);

    assert(sz_leading_zeros64(0x4000000000000000ull) == 1);

    assert(sz_size_log2i(1) == 0);
    assert(sz_size_log2i(2) == 1);

    assert(sz_size_log2i(3) == 2);
    assert(sz_size_log2i(4) == 2);
    assert(sz_size_log2i(5) == 3);

    assert(sz_size_log2i(7) == 3);
    assert(sz_size_log2i(8) == 3);
    assert(sz_size_log2i(9) == 4);

    assert(sz_size_bit_ceil(0) == 1);
    assert(sz_size_bit_ceil(1) == 1);

    assert(sz_size_bit_ceil(2) == 2);
    assert(sz_size_bit_ceil(3) == 4);
    assert(sz_size_bit_ceil(4) == 4);

    assert(sz_size_bit_ceil(77) == 128);
    assert(sz_size_bit_ceil(127) == 128);
    assert(sz_size_bit_ceil(128) == 128);

    assert(sz_size_bit_ceil(uint64_t(1e6)) == (1ull << 20));
    assert(sz_size_bit_ceil(uint64_t(2e6)) == (1ull << 21));
    assert(sz_size_bit_ceil(uint64_t(4e6)) == (1ull << 22));
    assert(sz_size_bit_ceil(uint64_t(8e6)) == (1ull << 23));

    assert(sz_size_bit_ceil(uint64_t(1.6e7)) == (1ull << 24));
    assert(sz_size_bit_ceil(uint64_t(3.2e7)) == (1ull << 25));
    assert(sz_size_bit_ceil(uint64_t(6.4e7)) == (1ull << 26));

    assert(sz_size_bit_ceil(uint64_t(1.28e8)) == (1ull << 27));
    assert(sz_size_bit_ceil(uint64_t(2.56e8)) == (1ull << 28));
    assert(sz_size_bit_ceil(uint64_t(5.12e8)) == (1ull << 29));

    assert(sz_size_bit_ceil(uint64_t(1e9)) == (1ull << 30));
    assert(sz_size_bit_ceil(uint64_t(2e9)) == (1ull << 31));
    assert(sz_size_bit_ceil(uint64_t(4e9)) == (1ull << 32));
    assert(sz_size_bit_ceil(uint64_t(8e9)) == (1ull << 33));

    assert(sz_size_bit_ceil(uint64_t(1.6e10)) == (1ull << 34));

    assert(sz_size_bit_ceil((1ull << 62))  == (1ull << 62));
    assert(sz_size_bit_ceil((1ull << 62) + 1)  == (1ull << 63));
    assert(sz_size_bit_ceil((1ull << 63))  == (1ull << 63));
}

namespace sz = ashvardanian::stringzilla;
using sz::literals::operator""_sz;

void
genstr(sz::string& out, size_t len, uint64_t seed) {
    auto chr = [&seed]() {
        seed = seed * 25214903917 + 11; // POSIX srand48 constants (shrug)
        return 'a' + seed % 36;
    };

    out.clear();
    for (auto i = 0; i < len; i++) {
        out.push_back(chr());
    }
}

static void
explicit_test_cases_run() {
    static const struct {
        const char* left;
        const char* right;
        size_t distance;
    } _explict_test_cases[] = {
        { "", "", 0 },
        { "", "abc", 3 },
        { "abc", "", 3 },
        { "abc", "ac", 1 },   // d,1
        { "abc", "a_bc", 1 }, // i,1
        { "abc", "adc", 1 },  // r,1
        { "ggbuzgjux{}l", "gbuzgjux{}l", 1 },  // prepend,1
    };

    auto cstr = [](const sz::string& s) {
        return &(sz::string_view(s))[0];
    };

    auto expect = [&cstr](const sz::string& l, const sz::string& r, size_t sz) {
        auto d = l.edit_distance(r);
        auto f = [&] {
            const char* ellipsis = l.length() > 22 || r.length() > 22 ? "..." : "";
            fprintf(stderr, "test failure: distance(\"%.22s%s\", \"%.22s%s\"); got %zd, expected %zd\n",
               cstr(l), ellipsis,
               cstr(r), ellipsis,
               d, sz);
            abort();
        };
        if (d != sz) {
            f();
        }
        // The distance relation commutes
        d = r.edit_distance(l);
        if (d != sz) {
            f();
        }
    };

    for (const auto tc: _explict_test_cases)
        expect(sz::string(tc.left), sz::string(tc.right), tc.distance);

    // Long string distances.
    const size_t LONG = size_t(19337);
    sz::string longstr;
    genstr(longstr, LONG, 071177);

    sz::string longstr2(longstr);
    expect(longstr, longstr2, 0); 

    for (auto i = 0; i < LONG; i += 17) {
        char buf[LONG + 1];
        // Insert at position i for a long string
        const char* longc = cstr(longstr);
        memcpy(buf, &longc[0], i);

        // Insert!
        buf[i] = longc[i];
        memcpy(buf + i + 1, &longc[i], LONG - i);

        sz::string inserted(sz::string_view(buf, LONG + 1));
        expect(inserted, longstr, 1);
    }
}

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurences of the `needle_stl`
 *  in a haystack formed of `haystack_pattern` repeated from one to `max_repeats` times.
 *
 *  @param misalignment The number of bytes to misalign the haystack within the cacheline.
 */
template <typename stl_matcher_, typename sz_matcher_>
void eval(std::string_view haystack_pattern, std::string_view needle_stl, std::size_t misalignment) {
    constexpr std::size_t max_repeats = 128;
    alignas(64) char haystack[misalignment + max_repeats * haystack_pattern.size()];
    std::vector<std::size_t> offsets_stl;
    std::vector<std::size_t> offsets_sz;

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

        // To simplify debugging, let's first export all the match offsets, and only then compare them
        std::transform(begin_stl, end_stl, std::back_inserter(offsets_stl),
                       [&](auto const &match) { return match.data() - haystack_stl.data(); });
        std::transform(begin_sz, end_sz, std::back_inserter(offsets_sz),
                       [&](auto const &match) { return match.data() - haystack_sz.data(); });
        auto print_all_matches = [&]() {
            std::printf("Breakdown of found matches:\n");
            std::printf("- STL (%zu): ", offsets_stl.size());
            for (auto offset : offsets_stl) std::printf("%zu ", offset);
            std::printf("\n");
            std::printf("- StringZilla (%zu): ", offsets_sz.size());
            for (auto offset : offsets_sz) std::printf("%zu ", offset);
            std::printf("\n");
        };

        // Compare results
        for (std::size_t match_idx = 0; begin_stl != end_stl && begin_sz != end_sz;
             ++begin_stl, ++begin_sz, ++match_idx) {
            auto match_stl = *begin_stl;
            auto match_sz = *begin_sz;
            if (match_stl.data() != match_sz.data()) {
                std::printf("Mismatch at index #%zu: %zu != %zu\n", match_idx, match_stl.data() - haystack_stl.data(),
                            match_sz.data() - haystack_sz.data());
                print_all_matches();
                assert(false);
            }
        }

        // If one range is not finished, assert failure
        if (count_stl != count_sz) {
            print_all_matches();
            assert(false);
        }
        assert(begin_stl == end_stl && begin_sz == end_sz);

        offsets_stl.clear();
        offsets_sz.clear();
    }
}

/**
 *  Evaluates the correctness of a "matcher", searching for all the occurences of the `needle_stl`,
 *  as a substring, as a set of allowed characters, or as a set of disallowed characters, in a haystack.
 */
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
    eval(haystack_pattern, needle_stl, 63);
    eval(haystack_pattern, needle_stl, 24);
    eval(haystack_pattern, needle_stl, 33);
}



int main(int argc, char const **argv) {

    test_util();
    explicit_test_cases_run();

    std::string_view alphabet = "abcdefghijklmnopqrstuvwxyz";                                         // 26 characters
    std::string_view base64 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-";     // 64 characters
    std::string_view common = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-=@$%"; // 68 characters

    assert(sz::string_view("a").find_first_of("az") == 0);
    assert(sz::string_view("a").find_last_of("az") == 0);
    assert(sz::string_view("a").find_first_of("xz") == sz::string_view::npos);
    assert(sz::string_view("a").find_last_of("xz") == sz::string_view::npos);

    assert(sz::string_view("a").find_first_not_of("xz") == 0);
    assert(sz::string_view("a").find_last_not_of("xz") == 0);
    assert(sz::string_view("a").find_first_not_of("az") == sz::string_view::npos);
    assert(sz::string_view("a").find_last_not_of("az") == sz::string_view::npos);

    // Comparing relative order of the strings
    assert("a"_sz.compare("a") == 0);
    assert("a"_sz.compare("ab") == -1);
    assert("ab"_sz.compare("a") == 1);
    assert("a"_sz.compare("a\0"_sz) == -1);
    assert("a\0"_sz.compare("a") == 1);
    assert("a\0"_sz.compare("a\0"_sz) == 0);
    assert("a"_sz == "a"_sz);
    assert("a"_sz != "a\0"_sz);
    assert("a\0"_sz == "a\0"_sz);
    assert(sz::string_view("aXbYaXbY").find_first_of("XY") == 1);
    assert(sz::string_view("axbYaxbY").find_first_of("Y") == 3);
    assert(sz::string_view("YbXaYbXa").find_last_of("XY") == 6);
    assert(sz::string_view("YbxaYbxa").find_last_of("Y") == 4);
    assert(sz::string_view(common).find_first_of("_") == sz::string_view::npos);
    assert(sz::string_view(common).find_first_of("+") == 62);
    assert(sz::string_view(common).find_first_of("=") == 64);

    // Make sure copy constructors work as expected:
    {
        std::vector<sz::string> strings;
        for (std::size_t alphabet_slice = 0; alphabet_slice != alphabet.size(); ++alphabet_slice)
            strings.push_back(alphabet.substr(0, alphabet_slice));
        std::vector<sz::string> copies {strings};
        assert(copies.size() == strings.size());
        for (size_t i = 0; i < copies.size(); i++) {
            assert(copies[i].size() == strings[i].size());
            assert(copies[i] == strings[i]);
            for (size_t j = 0; j < strings[i].size(); j++) {
                assert(copies[i][j] == strings[i][j]);
            }
        }
        std::vector<sz::string> assignments = strings;
        for (size_t i = 0; i < assignments.size(); i++) {
            assert(assignments[i].size() == strings[i].size());
            assert(assignments[i] == strings[i]);
            for (size_t j = 0; j < strings[i].size(); j++) {
                assert(assignments[i][j] == strings[i][j]);
            }
        }
        assert(std::equal(strings.begin(), strings.end(), copies.begin()));
        assert(std::equal(strings.begin(), strings.end(), assignments.begin()));
    }

    // When haystack is only formed of needles:
    eval("a", "a");
    eval("ab", "ab");
    eval("abc", "abc");
    eval("abcd", "abcd");
    eval(alphabet, alphabet);
    eval(base64, base64);
    eval(common, common);

    // When we are dealing with NULL characters inside the string
    eval("\0", "\0");
    eval("a\0", "a\0");
    eval("ab\0", "ab");
    eval("ab\0", "ab\0");
    eval("abc\0", "abc");
    eval("abc\0", "abc\0");
    eval("abcd\0", "abcd");

    // When haystack is formed of equidistant needles:
    eval("ab", "a");
    eval("abc", "a");
    eval("abcd", "a");

    // When matches occur in between pattern words:
    eval("ab", "ba");
    eval("abc", "ca");
    eval("abcd", "da");

    // Check more advanced composite operations:
    assert("abbccc"_sz.partition("bb").before.size() == 1);
    assert("abbccc"_sz.partition("bb").match.size() == 2);
    assert("abbccc"_sz.partition("bb").after.size() == 3);
    assert("abbccc"_sz.partition("bb").before == "a");
    assert("abbccc"_sz.partition("bb").match == "bb");
    assert("abbccc"_sz.partition("bb").after == "ccc");

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

    auto splits = ".a..c."_sz.split(sz::character_set(".")).template to<std::vector<std::string>>();
    assert(splits.size() == 5);
    assert(splits[0] == "");
    assert(splits[1] == "a");
    assert(splits[4] == "");

    assert(""_sz.split(".").size() == 1);
    assert(""_sz.rsplit(".").size() == 1);
    assert("a.b.c.d"_sz.split(".").size() == 4);
    assert("a.b.c.d"_sz.rsplit(".").size() == 4);
    assert("a.b.,c,d"_sz.split(".,").size() == 2);
    assert("a.b,c.d"_sz.split(sz::character_set(".,")).size() == 4);

    auto rsplits = ".a..c."_sz.rsplit(sz::character_set(".")).template to<std::vector<std::string>>();
    assert(rsplits.size() == 5);
    assert(rsplits[0] == "");
    assert(rsplits[1] == "c");
    assert(rsplits[4] == "");

    // Compare STL and StringZilla strings append functionality.
    char const alphabet_chars[] = "abcdefghijklmnopqrstuvwxyz";
    std::string stl_string;
    sz::string sz_string;
    for (std::size_t length = 1; length != 200; ++length) {
        char c = alphabet_chars[std::rand() % 26];
        stl_string.push_back(c);
        sz_string.push_back(c);
        assert(sz::string_view(stl_string) == sz::string_view(sz_string));
    }

    // Compare STL and StringZilla strings erase functionality.
    while (stl_string.length()) {
        std::size_t offset_to_erase = std::rand() % stl_string.length();
        std::size_t chars_to_erase = std::rand() % (stl_string.length() - offset_to_erase) + 1;
        stl_string.erase(offset_to_erase, chars_to_erase);
        sz_string.erase(offset_to_erase, chars_to_erase);
        assert(sz::string_view(stl_string) == sz::string_view(sz_string));
    }

    return 0;
}
