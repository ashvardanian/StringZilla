#include <gtest/gtest.h>
#include <string>
#include <stringzilla.h>

TEST(StringzillaSWAR, test_sz_count_char_swar) {
    std::string haystack = "daddadddaddddaddddda";
    std::string needle1 = "a";
    std::string needle2 = "d";
    ASSERT_EQ(sz_count_char_swar(haystack.data(), haystack.size(), needle1.data()), 5);
    ASSERT_EQ(sz_count_char_swar(haystack.data(), haystack.size(), needle2.data()), 15);
}

TEST(StringzillaSWAR, test_sz_find_Xchar_swar) {
    std::string haystack = "myneedleinhaystack";

    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "m") - haystack.data(), 0);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "y") - haystack.data(), 1);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "n") - haystack.data(), 2);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "e") - haystack.data(), 3);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "d") - haystack.data(), 5);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "l") - haystack.data(), 6);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "i") - haystack.data(), 8);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "h") - haystack.data(), 10);
    ASSERT_EQ(sz_find_1char_swar(haystack.data(), haystack.size(), "k") - haystack.data(), 17);

    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "my") - haystack.data(), 0);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "yn") - haystack.data(), 1);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "ne") - haystack.data(), 2);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "ee") - haystack.data(), 3);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "ed") - haystack.data(), 4);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "dl") - haystack.data(), 5);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "le") - haystack.data(), 6);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "ei") - haystack.data(), 7);
    ASSERT_EQ(sz_find_2char_swar(haystack.data(), haystack.size(), "in") - haystack.data(), 8);


    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "myn") - haystack.data(), 0);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "yne") - haystack.data(), 1);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "nee") - haystack.data(), 2);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "eed") - haystack.data(), 3);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "edl") - haystack.data(), 4);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "dle") - haystack.data(), 5);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "lei") - haystack.data(), 6);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "ein") - haystack.data(), 7);
    ASSERT_EQ(sz_find_3char_swar(haystack.data(), haystack.size(), "inh") - haystack.data(), 8);

    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "myne") - haystack.data(), 0);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "ynee") - haystack.data(), 1);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "need") - haystack.data(), 2);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "eedl") - haystack.data(), 3);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "edle") - haystack.data(), 4);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "dlei") - haystack.data(), 5);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "lein") - haystack.data(), 6);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "einh") - haystack.data(), 7);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "inha") - haystack.data(), 8);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "nhay") - haystack.data(), 9);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "hays") - haystack.data(), 10);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "ayst") - haystack.data(), 11);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "ysta") - haystack.data(), 12);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "stac") - haystack.data(), 13);
    ASSERT_EQ(sz_find_4char_swar(haystack.data(), haystack.size(), "tack") - haystack.data(), 14);
}

TEST(StringzillaSWAR, test_sz_rfind_1char_swar) {
    std::string haystack = "myneedleinhaystack";

    ASSERT_EQ(sz_rfind_1char_swar(haystack.data(), haystack.size(), "m") - haystack.data(), 0);
    ASSERT_EQ(sz_rfind_1char_swar(haystack.data(), haystack.size(), "d") - haystack.data(), 5);
    ASSERT_EQ(sz_rfind_1char_swar(haystack.data(), haystack.size(), "l") - haystack.data(), 6);
    ASSERT_EQ(sz_rfind_1char_swar(haystack.data(), haystack.size(), "e") - haystack.data(), 7);
    ASSERT_EQ(sz_rfind_1char_swar(haystack.data(), haystack.size(), "h") - haystack.data(), 10);
    ASSERT_EQ(sz_rfind_1char_swar(haystack.data(), haystack.size(), "a") - haystack.data(), 15);

    std::string alphabet = "abcdefghijklmnopqrstuvwxyz";

    for (int i = 24; i < alphabet.size(); i++) {
        std::string needle = std::string(1, alphabet[i]);
        EXPECT_EQ(sz_rfind_1char_swar(alphabet.data(), alphabet.size(), needle.data()) - alphabet.data(), i);
    }
}

TEST(StringzillaSWAR, test_sz_find_substring_swar) {
    std::string haystack = "myneedleinhaystack";
    std::string needle = "needle";
    sz_string_start_t ptr = sz_find_substring_swar(haystack.data(), haystack.size(), needle.data(), needle.size());
    ASSERT_TRUE(ptr);
    ASSERT_EQ(std::string(ptr, 6), "needle");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
