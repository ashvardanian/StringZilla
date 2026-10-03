/**
 *  @brief  Compiles and exercises the public surface with the STL excluded.
 *  @file   test/avoid_stl.cpp
 *  @author Ash Vardanian
 *  @date   August 14, 2026
 *
 *  The `SZ_AVOID_STL=1` build drops STL interop - the `std::string` conversions, the `iterator_category`
 *  typedefs, and the `std::hash` specializations - but keeps every algorithm and every bounds check.
 *  This translation unit pins that contract: it names only what must survive, so a stray `#include`
 *  or an ungated `std::` reference fails the build instead of silently reviving the dependency.
 */
#undef NDEBUG // ! Enable all assertions for testing

#define SZ_AVOID_STL 1

#if defined(SZ_DEBUG)
#undef SZ_DEBUG
#endif
#define SZ_DEBUG 1 // ! Enforce aggressive logging in this translation unit

#include <stringzilla/stringzilla.hpp>

/**
 *  Compiling is only half the contract - the mode also has to keep the heavy headers out, or it
 *  silently stops paying for itself. Header guards are the only portable way to observe that from
 *  inside a translation unit, so we probe the three standard libraries we ship against.
 */
#if defined(_GLIBCXX_STRING) || defined(_LIBCPP_STRING) || defined(_STRING_)
#error "SZ_AVOID_STL=1 still reaches <string>"
#endif
#if defined(_GLIBCXX_MEMORY) || defined(_LIBCPP_MEMORY) || defined(_MEMORY_)
#error "SZ_AVOID_STL=1 still reaches <memory>"
#endif
#if defined(_GLIBCXX_ITERATOR) || defined(_LIBCPP_ITERATOR) || defined(_ITERATOR_)
#error "SZ_AVOID_STL=1 still reaches <iterator>"
#endif
#if defined(_GLIBCXX_VECTOR) || defined(_LIBCPP_VECTOR) || defined(_VECTOR_)
#error "SZ_AVOID_STL=1 still reaches <vector>"
#endif

#include <stdio.h>  // `printf`
#include <stdlib.h> // `EXIT_FAILURE`

namespace sz = ashvardanian::stringzilla;

#define verify(condition)                                                               \
    do {                                                                                \
        if (!(condition)) {                                                             \
            printf("Assertion failed: %s, in %s:%d\n", #condition, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

/** @brief Searching, slicing, and comparison stay available without the STL. */
static void test_slices_without_stl() {
    sz::string_view haystack("hello world");

    verify(haystack.size() == 11);
    verify(haystack.find("world") == 6);
    verify(haystack.rfind('o') == 7);
    verify(haystack.rfind('o', 6) == 4);
    verify(haystack.starts_with("hello"));
    verify(haystack.ends_with("world"));

    // `sub` clamps, `substr` checks - both survive, as does the comparison family built on them.
    verify(haystack.sub(0, 5) == "hello");
    verify(haystack.substr(6) == "world");
    verify(haystack.substr(0, 5) == "hello");
    verify(haystack.compare(0, 5, sz::string_view("hello")) == 0);
    verify(haystack.compare(sz::string_view("hello")) > 0);

    char destination[8] = {0};
    verify(haystack.copy(destination, 5, 0) == 5);
    verify(sz::string_view(destination, 5) == "hello");
}

/** @brief The owning string keeps its algorithms; only the `std::string` bridges are gone. */
static void test_owning_string_without_stl() {
    using allocator_t = sz::dummy_alloc<char>;
    sz::basic_string<char, allocator_t> refusing;

    // Short strings live in the inline buffer, so they need no allocator at all.
    verify(refusing.try_append("short") == true);
    verify(refusing == "short");

    // Past the inline capacity `dummy_alloc` refuses, and the failure is reported, not thrown.
    char long_text[256];
    for (sz_size_t index = 0; index + 1 != sizeof(long_text); ++index) long_text[index] = 'a';
    long_text[sizeof(long_text) - 1] = '\0';
    verify(refusing.try_append(long_text) == false);
    verify(refusing == "short");
}

/** @brief The tape stores variable-length strings through a caller-supplied allocator. */
static void test_arrow_tape_without_stl() {
    using tape_t = sz::arrow_strings_tape<char, sz_size_t, sz::dummy_alloc<char>>;
    tape_t tape;
    verify(tape.size() == 0);
}

int main() {
    test_slices_without_stl();
    test_owning_string_without_stl();
    test_arrow_tape_without_stl();
    printf("All tests passed... yay!\n");
    return 0;
}
