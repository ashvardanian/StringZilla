/**
 *  @file test/utf8_norm.cpp
 *  @author Ash Vardanian
 *  @date June 15, 2026
 *  @brief UTF-8 normalization (NFC/NFD/NFKC/NFKD) known-answer, differential and safety tests.
 */
#undef NDEBUG // ! A test's oracle must stay live in every build

#if defined(STRINGZILLA_DEBUG)
#undef STRINGZILLA_DEBUG
#endif
#define STRINGZILLA_DEBUG 1 // ! Enforce aggressive logging in this translation unit

/*  Make sure to include the StringZilla headers before anything else, to intercept missing
 *  `#include` directives and other issues. */
#include <stringzilla/stringzilla.h>   // Primary C API
#include <stringzilla/stringzilla.hpp> // C++ string class replacement

#include "cross.hpp" // `check_utf8_norm_unit_`, `check_utf8_norm_equivalence_`, `check_utf8_norm_safety_`

namespace ashvardanian::stringzilla::test {

/** The normalization dispatch points, in the shape of one backend's kernels. */
static utf8_norm_kernels_t const utf8_norm_dispatched {cpu_best<sz_utf8_norm_best>,
                                                       cpu_best<sz_utf8_find_denormalized_best>};

/**
 *  @brief Known-answer unit tests for UTF-8 normalization on simple, hand-verifiable inputs.
 *
 *  Exercises the normalizer and the normalization-violation finder through the dispatch points, and
 *  through the C++ @c sz::string_t and @c sz::string_view_t wrappers @c utf8_normalize,
 *  @c is_normalized and @c utf8_find_denormalized, so a regression that the serial-vs-SIMD
 *  agreement tests would miss - because both share a wrong constant - is still caught against an
 *  external ground truth.
 */
void test_utf8_norm_unit() {
    check_utf8_norm_unit_(utf8_norm_dispatched);

    // C++ binding round-trip: NFC → NFD → NFC should recover the original NFC string.
    char const cafe_nfc[] = "caf\xC3\xA9"; // U+00E9 (precomposed é), 5 bytes
    sz::string_t nfc_str;
    expect_true(sz::succeeded(nfc_str.assign(cafe_nfc)));
    expect_true(sz::succeeded(nfc_str.utf8_normalize(sz_normal_form_nfd_k)));
    expect_true(sz::succeeded(nfc_str.utf8_normalize(sz_normal_form_nfc_k)));
    expect_equal(nfc_str, cafe_nfc);

    // is_normalized: NFC string is normalized under NFC, not NFD (é decomposes in NFD).
    sz::string_view_t nfc_view {cafe_nfc};
    expect_true(nfc_view.is_normalized(sz_normal_form_nfc_k));
    expect_true(!nfc_view.is_normalized(sz_normal_form_nfd_k));

    // is_normalized on the owning type mirrors the view behaviour.
    sz::string_t nfc_own;
    expect_true(sz::succeeded(nfc_own.assign(cafe_nfc)));
    expect_true(nfc_own.is_normalized(sz_normal_form_nfc_k));
    expect_true(!nfc_own.is_normalized(sz_normal_form_nfd_k));

    // utf8_find_denormalized returns non-null for a non-normalized string.
    expect_true(nfc_view.utf8_find_denormalized(sz_normal_form_nfd_k) != STRINGZILLA_NULL_CHAR);
    expect_true(nfc_own.utf8_find_denormalized(sz_normal_form_nfd_k) != STRINGZILLA_NULL_CHAR);

    // NFKD of "ﬁ" (U+FB01 LATIN SMALL LIGATURE FI) decomposes to "fi".
    char const ligature_fi[] = "\xEF\xAC\x81"; // U+FB01
    sz::string_t fi_str;
    expect_true(sz::succeeded(fi_str.assign(ligature_fi)));
    expect_true(sz::succeeded(fi_str.utf8_normalize(sz_normal_form_nfkd_k)));
    expect_true(fi_str.contains('f'));
    expect_true(fi_str.contains('i'));
}

/** Malformed-input normalization safety probe through the dispatch points. */
void test_utf8_norm_safety() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_norm_safety_(generator, utf8_norm_dispatched);
}

/** The normalization differential fuzz of the dispatch points against the serial kernels. */
void test_utf8_norm_all() {
    split_mix_generator_t generator {case_stream_key()};
    check_utf8_norm_equivalence_(generator, utf8_norm_dispatched);
}

} // namespace ashvardanian::stringzilla::test
