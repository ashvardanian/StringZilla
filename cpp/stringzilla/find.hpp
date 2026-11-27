/**
 *  @file cpp/stringzilla/find.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief Substring, byte and byte-set search over any string, and the lazy ranges of matches and
 *      splits, through the dispatch points of @c find.h.
 *
 *  Every verb takes @ref a_string, from StringZilla and STL strings to string literals. The default
 *  mask returns a plain result, while an explicit capability mask and stream return an @c expected.
 */
#ifndef STRINGZILLA_FIND_HPP_
#define STRINGZILLA_FIND_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include <cstddef> // `std::size_t`, `std::ptrdiff_t`

#include <ranges> // `std::ranges::enable_borrowed_range`

#include "stringzilla/find.h" // `sz_find_best`, `sz_rfind_best`, `sz_find_byteset_best`

#include "stringzilla/string.hpp"

#include "stringzilla/devices.hpp"

/** A string slice borrows its bytes, so the ranges may keep one past the call that made them. */
template <typename char_type_>
inline constexpr bool std::ranges::enable_borrowed_range<ashvardanian::stringzilla::string_slice<char_type_>> = true;

namespace ashvardanian {
namespace stringzilla {

#pragma region Search

/**
 *  @brief Finds the first inclusion of @p needle in @p haystack, past its first @p skip bytes.
 *  @return The offset of the match, @p skip for an empty @p needle, or @ref npos_of the haystack
 *      when there is none or @p skip passes the end.
 *  @sa sz_find_best
 */
template <a_string haystack_type_, a_string needle_type_>
expected<std::size_t> find(haystack_type_ const &haystack, needle_type_ const &needle, std::size_t skip,
                           sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack), pattern = view_of(needle);
    if (skip > text.size()) return {npos_of<haystack_type_>(), status_t::success_k};
    if (pattern.empty()) return {skip, status_t::success_k};
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_find_best(text.data() + skip, text.size() - skip, pattern.data(), pattern.size(),
                                            &match, capabilities, stream);
    return {match ? static_cast<std::size_t>(match - text.data()) : npos_of<haystack_type_>(),
            static_cast<status_t>(status)};
}

/** @copydoc find */
template <a_string haystack_type_, a_string needle_type_>
std::size_t find(haystack_type_ const &haystack, needle_type_ const &needle, std::size_t skip = 0) noexcept {
    return default_mask_value_(find(haystack, needle, skip, default_capabilities(), nullptr));
}

/**
 *  @brief Finds the first @p character in @p haystack, past its first @p skip bytes.
 *  @sa sz_find_byte_best
 */
template <a_string haystack_type_>
expected<std::size_t> find(haystack_type_ const &haystack, char character, std::size_t skip,
                           sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack);
    if (skip > text.size()) return {npos_of<haystack_type_>(), status_t::success_k};
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_find_byte_best(text.data() + skip, text.size() - skip, &character, &match,
                                                 capabilities, stream);
    return {match ? static_cast<std::size_t>(match - text.data()) : npos_of<haystack_type_>(),
            static_cast<status_t>(status)};
}

/** Finds the first @p character past the first @p skip bytes, over the default mask. */
template <a_string haystack_type_>
std::size_t find(haystack_type_ const &haystack, char character, std::size_t skip = 0) noexcept {
    return default_mask_value_(find(haystack, character, skip, default_capabilities(), nullptr));
}

/**
 *  @brief Finds the last inclusion of @p needle in @p haystack.
 *  @return The offset of the match, the haystack's length for an empty @p needle, or
 *      @ref npos_of the haystack.
 *  @sa sz_rfind_best
 */
template <a_string haystack_type_, a_string needle_type_>
expected<std::size_t> rfind(haystack_type_ const &haystack, needle_type_ const &needle, sz_capability_t capabilities,
                            sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack), pattern = view_of(needle);
    if (pattern.empty()) return {text.size(), status_t::success_k};
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_rfind_best(text.data(), text.size(), pattern.data(), pattern.size(), &match,
                                             capabilities, stream);
    return {match ? static_cast<std::size_t>(match - text.data()) : npos_of<haystack_type_>(),
            static_cast<status_t>(status)};
}

/** @copydoc rfind */
template <a_string haystack_type_, a_string needle_type_>
std::size_t rfind(haystack_type_ const &haystack, needle_type_ const &needle) noexcept {
    return default_mask_value_(rfind(haystack, needle, default_capabilities(), nullptr));
}

/**
 *  @brief Finds the last inclusion of @p needle starting at or before @p until, like
 *      <tt>std::string_view::rfind(needle, until)</tt>.
 *  @return For an empty @p needle, <tt>min(until, size())</tt>.
 */
template <a_string haystack_type_, a_string needle_type_>
std::size_t rfind(haystack_type_ const &haystack, needle_type_ const &needle, std::size_t until) noexcept {
    string_view_t const text = view_of(haystack), pattern = view_of(needle);
    bool const narrows = pattern.size() < text.size() && until < text.size() - pattern.size();
    std::size_t const found = rfind(narrows ? text.substr(0, until + pattern.size()) : text, pattern);
    return found == string_view_t::npos ? npos_of<haystack_type_>() : found;
}

/**
 *  @brief Finds the last @p character in @p haystack.
 *  @sa sz_rfind_byte_best
 */
template <a_string haystack_type_>
expected<std::size_t> rfind(haystack_type_ const &haystack, char character, sz_capability_t capabilities,
                            sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack);
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_rfind_byte_best(text.data(), text.size(), &character, &match, capabilities, stream);
    return {match ? static_cast<std::size_t>(match - text.data()) : npos_of<haystack_type_>(),
            static_cast<status_t>(status)};
}

/** Finds the last @p character in @p haystack, over the default mask. */
template <a_string haystack_type_>
std::size_t rfind(haystack_type_ const &haystack, char character) noexcept {
    return default_mask_value_(rfind(haystack, character, default_capabilities(), nullptr));
}

/** Finds the last @p character at or before @p until. */
template <a_string haystack_type_>
std::size_t rfind(haystack_type_ const &haystack, char character, std::size_t until) noexcept {
    string_view_t const text = view_of(haystack);
    std::size_t const found = rfind(until < text.size() ? text.substr(0, until + 1) : text, character);
    return found == string_view_t::npos ? npos_of<haystack_type_>() : found;
}

/**
 *  @brief Whether @p needle occurs in @p haystack.
 *  @return Always @c true for an @b empty @p needle, as <tt>std::string_view::find(v) != npos</tt> is.
 */
template <a_string haystack_type_, a_string needle_type_>
bool contains(haystack_type_ const &haystack, needle_type_ const &needle) noexcept {
    return find(view_of(haystack), needle) != string_view_t::npos;
}

/** Whether @p character occurs in @p haystack. */
template <a_string haystack_type_>
bool contains(haystack_type_ const &haystack, char character) noexcept {
    return find(view_of(haystack), character) != string_view_t::npos;
}

#pragma endregion Search

#pragma region Byte Sets

/** The set of the bytes of @p characters, or @p characters itself when it is already a set. */
template <typename characters_type_>
byteset_t byteset_of_(characters_type_ const &characters) noexcept {
    if constexpr (std::is_same_v<characters_type_, byteset_t>) return characters;
    else return view_of(characters).as_set();
}

/**
 *  @brief Finds the first byte of @p set in @p haystack, past its first @p skip bytes.
 *  @sa sz_find_byteset_best
 */
template <a_string haystack_type_>
expected<std::size_t> find_first_of(haystack_type_ const &haystack, byteset_t const &set, std::size_t skip,
                                    sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack);
    if (skip > text.size()) return {npos_of<haystack_type_>(), status_t::success_k};
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_find_byteset_best(text.data() + skip, text.size() - skip, &set.raw(), &match,
                                                    capabilities, stream);
    return {match ? static_cast<std::size_t>(match - text.data()) : npos_of<haystack_type_>(),
            static_cast<status_t>(status)};
}

/** Finds the first byte of @p characters, a set or a string of them, past the first @p skip. */
template <a_string haystack_type_, typename characters_type_>
std::size_t find_first_of(haystack_type_ const &haystack, characters_type_ const &characters,
                          std::size_t skip = 0) noexcept {
    return default_mask_value_(find_first_of(haystack, byteset_of_(characters), skip, default_capabilities(), nullptr));
}

/** Finds the first byte outside @p characters, a set or a string of them, past @p skip bytes. */
template <a_string haystack_type_, typename characters_type_>
std::size_t find_first_not_of(haystack_type_ const &haystack, characters_type_ const &characters,
                              std::size_t skip = 0) noexcept {
    return find_first_of(haystack, byteset_of_(characters).inverted(), skip);
}

/**
 *  @brief Finds the last byte of @p set in @p haystack at or before @p until.
 *  @sa sz_rfind_byteset_best
 */
template <a_string haystack_type_>
expected<std::size_t> find_last_of(haystack_type_ const &haystack, byteset_t const &set, std::size_t until,
                                   sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const text = view_of(haystack);
    std::size_t const length = until < text.size() ? until + 1 : text.size();
    sz_cptr_t match = nullptr;
    sz_status_t const status = sz_rfind_byteset_best(text.data(), length, &set.raw(), &match, capabilities, stream);
    return {match ? static_cast<std::size_t>(match - text.data()) : npos_of<haystack_type_>(),
            static_cast<status_t>(status)};
}

/** Finds the last byte of @p characters, a set or a string of them, at or before @p until. */
template <a_string haystack_type_, typename characters_type_>
std::size_t find_last_of(haystack_type_ const &haystack, characters_type_ const &characters,
                         std::size_t until = string_view_t::npos) noexcept {
    return default_mask_value_(find_last_of(haystack, byteset_of_(characters), until, default_capabilities(), nullptr));
}

/** Finds the last byte outside @p characters, a set or a string of them, at or before @p until. */
template <a_string haystack_type_, typename characters_type_>
std::size_t find_last_not_of(haystack_type_ const &haystack, characters_type_ const &characters,
                             std::size_t until = string_view_t::npos) noexcept {
    return find_last_of(haystack, byteset_of_(characters).inverted(), until);
}

/** Finds the first byte of @p set, like @ref find_first_of. */
template <a_string haystack_type_>
std::size_t find(haystack_type_ const &haystack, byteset_t const &set) noexcept {
    return find_first_of(haystack, set);
}

/** Finds the last byte of @p set, like @ref find_last_of. */
template <a_string haystack_type_>
std::size_t rfind(haystack_type_ const &haystack, byteset_t const &set) noexcept {
    return find_last_of(haystack, set);
}

/** Whether every byte of @p haystack belongs to @p set, @c true for an empty one. */
template <a_string haystack_type_>
bool contains_only(haystack_type_ const &haystack, byteset_t const &set) noexcept {
    return find_first_not_of(view_of(haystack), set) == string_view_t::npos;
}

/** Whether @p text is non-empty and all ASCII letters. */
template <a_string text_type_>
bool is_alpha(text_type_ const &text) noexcept {
    return !view_of(text).empty() && contains_only(text, ascii_letters_set());
}

/** Whether @p text is non-empty and all ASCII letters and digits. */
template <a_string text_type_>
bool is_alnum(text_type_ const &text) noexcept {
    return !view_of(text).empty() && contains_only(text, ascii_letters_set() | digits_set());
}

/** Whether @p text is all ASCII, @c true for an empty one. */
template <a_string text_type_>
bool is_ascii(text_type_ const &text) noexcept {
    return view_of(text).empty() || contains_only(text, ascii_controls_set() | ascii_printables_set());
}

/** Whether @p text is non-empty and all ASCII digits. */
template <a_string text_type_>
bool is_digit(text_type_ const &text) noexcept {
    return !view_of(text).empty() && contains_only(text, digits_set());
}

/** Whether @p text is non-empty and all ASCII lowercase letters. */
template <a_string text_type_>
bool is_lower(text_type_ const &text) noexcept {
    return !view_of(text).empty() && contains_only(text, ascii_lowercase_set());
}

/** Whether @p text is non-empty and all ASCII whitespace. */
template <a_string text_type_>
bool is_space(text_type_ const &text) noexcept {
    return !view_of(text).empty() && contains_only(text, whitespaces_set());
}

/** Whether @p text is non-empty and all ASCII uppercase letters. */
template <a_string text_type_>
bool is_upper(text_type_ const &text) noexcept {
    return !view_of(text).empty() && contains_only(text, ascii_uppercase_set());
}

/** Whether @p text is all printable ASCII, @c true for an empty one. */
template <a_string text_type_>
bool is_printable(text_type_ const &text) noexcept {
    return view_of(text).empty() || contains_only(text, ascii_printables_set());
}

/**
 *  @brief Drops the prefix made of bytes of @p set, like Python's <tt>str.lstrip</tt>.
 *  @note Similar to <tt>boost::algorithm::trim_left_if(str, is_any_of(set))</tt>.
 */
template <a_string text_type_>
string_view_t lstrip(text_type_ const &text, byteset_t const &set) noexcept {
    string_view_t const view = view_of(text);
    std::size_t const start = find_first_not_of(view, set);
    return start == string_view_t::npos ? string_view_t() : view.substr(start);
}

/**
 *  @brief Drops the suffix made of bytes of @p set, like Python's <tt>str.rstrip</tt>.
 *  @note Similar to <tt>boost::algorithm::trim_right_if(str, is_any_of(set))</tt>.
 */
template <a_string text_type_>
string_view_t rstrip(text_type_ const &text, byteset_t const &set) noexcept {
    string_view_t const view = view_of(text);
    std::size_t const last = find_last_not_of(view, set);
    return last == string_view_t::npos ? string_view_t() : view.substr(0, last + 1);
}

/**
 *  @brief Drops the bytes of @p set off both ends, like Python's <tt>str.strip</tt>.
 *  @note Similar to <tt>boost::algorithm::trim_if(str, is_any_of(set))</tt>.
 */
template <a_string text_type_>
string_view_t strip(text_type_ const &text, byteset_t const &set) noexcept {
    return rstrip(lstrip(text, set), set);
}

#pragma endregion Byte Sets

#pragma region Partitions

/** A result of split a string once, containing the string slice ::before,
 *  the ::match itself, and the slice ::after. */
template <typename string_>
struct string_partition_result {
    string_ before;
    string_ match;
    string_ after;
};

/** The three parts around a match of @p length bytes at @p position, or the whole text before an
 *  empty match when nothing matched. */
inline string_partition_result<string_view_t> partition_at_(string_view_t text, std::size_t position,
                                                            std::size_t length) noexcept {
    if (!length || position == string_view_t::npos) return {text, string_view_t(), string_view_t()};
    return {text.substr(0, position), text.substr(position, length), text.substr(position + length)};
}

/** Splits @p text around the first match of @p pattern, a string, a byte or a byte set, into the
 *  part before it, the match itself, and the part after it, like Python's <tt>str.partition</tt>. */
template <a_string text_type_, typename pattern_type_>
string_partition_result<string_view_t> partition(text_type_ const &text, pattern_type_ const &pattern) noexcept {
    string_view_t const view = view_of(text);
    if constexpr (std::is_same_v<pattern_type_, byteset_t>) return partition_at_(view, find_first_of(view, pattern), 1);
    else if constexpr (std::is_same_v<pattern_type_, char>) return partition_at_(view, find(view, pattern), 1);
    else return partition_at_(view, find(view, pattern), view_of(pattern).size());
}

/** Splits @p text around the last match of @p pattern, like Python's <tt>str.rpartition</tt>. */
template <a_string text_type_, typename pattern_type_>
string_partition_result<string_view_t> rpartition(text_type_ const &text, pattern_type_ const &pattern) noexcept {
    string_view_t const view = view_of(text);
    if constexpr (std::is_same_v<pattern_type_, byteset_t>) return partition_at_(view, find_last_of(view, pattern), 1);
    else if constexpr (std::is_same_v<pattern_type_, char>) return partition_at_(view, rfind(view, pattern), 1);
    else return partition_at_(view, rfind(view, pattern), view_of(pattern).size());
}

#pragma endregion Partitions

#pragma region Ranges of Search Matches

struct end_sentinel_t {};
struct include_overlaps_t {};
struct exclude_overlaps_t {};

inline static constexpr end_sentinel_t end_sentinel;
inline static constexpr include_overlaps_t include_overlaps;
inline static constexpr exclude_overlaps_t exclude_overlaps;

/** Finds the first inclusion of a needle, over @ref sz::find. */
template <typename string_type_, typename overlaps_type_ = include_overlaps_t>
struct matcher_find {
    using size_type = typename string_type_::size_type;
    string_type_ needle_;

    size_type needle_length() const noexcept { return needle_.length(); }
    size_type operator()(string_type_ haystack) const noexcept { return stringzilla::find(haystack, needle_); }
    size_type skip_length() const noexcept {
        // TODO: Apply Galil rule to match repetitive patterns in strictly linear time.
        // Floor at 1 so an empty needle still advances.
        return std::is_same_v<overlaps_type_, include_overlaps_t> ? 1 : sz_max_of_two(needle_.length(), size_type(1));
    }
};

/** Finds the last inclusion of a needle, over @ref sz::rfind. */
template <typename string_type_, typename overlaps_type_ = include_overlaps_t>
struct matcher_rfind {
    using size_type = typename string_type_::size_type;
    string_type_ needle_;

    size_type needle_length() const noexcept { return needle_.length(); }
    size_type operator()(string_type_ haystack) const noexcept { return stringzilla::rfind(haystack, needle_); }
    size_type skip_length() const noexcept {
        // TODO: Apply Galil rule to match repetitive patterns in strictly linear time.
        // Floor at 1 so an empty needle still advances.
        return std::is_same_v<overlaps_type_, include_overlaps_t> ? 1 : sz_max_of_two(needle_.length(), size_type(1));
    }
};

/** Finds the first byte of a set, over @ref sz::find_first_of. */
template <typename haystack_type_>
struct matcher_find_first_of {
    using size_type = typename haystack_type_::size_type;
    byteset_t set_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(haystack_type_ haystack) const noexcept { return stringzilla::find_first_of(haystack, set_); }
};

/** Finds the last byte of a set, over @ref sz::find_last_of. */
template <typename haystack_type_>
struct matcher_find_last_of {
    using size_type = typename haystack_type_::size_type;
    byteset_t set_;
    constexpr size_type needle_length() const noexcept { return 1; }
    constexpr size_type skip_length() const noexcept { return 1; }
    size_type operator()(haystack_type_ haystack) const noexcept { return stringzilla::find_last_of(haystack, set_); }
};

/** Whether a type carries a nested @c ::string_view_t typedef. */
template <typename type_>
concept a_type_with_string_view_member_ = requires { typename type_::string_view_t; };

/** Extracts the view type of a string-like type: the nested @c string_view_t typedef of
 *  StringZilla types, or the type itself for STL types like @c std::string_view. */
template <typename string_type_>
struct string_view_for {
    using type = string_type_;
};

template <typename string_type_>
    requires a_type_with_string_view_member_<string_type_>
struct string_view_for<string_type_> {
    using type = typename string_type_::string_view_t;
};

/**
 *  @brief A string a lazy range may keep past the call that made it: an lvalue, which the range
 *      borrows, or a view that borrows its bytes itself, like @c string_view_t or a pointer.
 *
 *  An owning temporary, like <tt>std::string("a b")</tt>, would die before the range is walked, so
 *  the range factories refuse it at the call site.
 */
template <typename type_>
concept a_borrowed_string = std::is_lvalue_reference_v<type_> || std::is_pointer_v<std::remove_cvref_t<type_>> ||
                            std::ranges::enable_borrowed_range<std::remove_cvref_t<type_>>;

/** The slice type a range over @p haystack_type_ walks and yields, shedding references and
 *  qualifiers first, as <tt>std::string_view const</tt> would lose its @c remove_prefix. */
template <typename haystack_type_>
using range_view_type = typename string_view_for<std::remove_cvref_t<haystack_type_>>::type;

/**
 *  @brief A range of string slices representing the matches of a substring search.
 *
 *  @note Lifetime semantics: Holds views only, as the factories accept only lvalues and borrowed
 *      views, so the caller keeps the bytes alive while the range is walked.
 *  @note For-loop optimized: Iterators are lightweight views with minimal register pressure, ideal
 *      for high-performance applications where cache efficiency and register allocation matter.
 *  @note Sentinel support: Supports sentinel-based iteration via <tt>operator==(end_sentinel_t)</tt> for
 *      efficient termination without constructing full end iterators.
 *  @note Compatible with C++20 ranges, @c std::string_view, and of course, StringZilla.
 *  @note Similar to a pair of @c boost::algorithm::find_iterator.
 */
template <typename string_type_, typename matcher_type_>
class find_matches_view {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;
    using string_view_type = typename string_view_for<string_type>::type;

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_view_type;
    using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
    using reference = string_view_type; // Needed for compatibility with STL container constructors.

    find_matches_view(string_type haystack, matcher_type needle) noexcept
        : matcher_(std::move(needle)), haystack_(std::move(haystack)) {}

    class iterator_t {
        matcher_type matcher_;

        /** From the current match to the end, null once exhausted. */
        string_view_type remaining_;

        void find_() noexcept {
            auto const position = matcher_(remaining_);
            if (position != string_view_type::npos) remaining_.remove_prefix(position);
            else remaining_ = string_view_type();
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view_type;
        using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
        using reference = string_view_type; // Needed for compatibility with STL container constructors.

        iterator_t(string_view_type haystack, matcher_type const &matcher) noexcept
            : matcher_(matcher), remaining_(haystack) {
            find_();
        }
        iterator_t(matcher_type const &matcher, end_sentinel_t) noexcept : matcher_(matcher), remaining_() {}

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept { return remaining_.substr(0, matcher_.needle_length()); }

        iterator_t &operator++() noexcept {
            // Only the zero-width match at the very end leaves nothing to skip.
            if (remaining_.empty()) remaining_ = string_view_type();
            else remaining_.remove_prefix(matcher_.skip_length()), find_();
            return *this;
        }

        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        // Assumes both iterators point to the same underlying string.
        bool operator!=(iterator_t const &other) const noexcept { return remaining_.data() != other.remaining_.data(); }
        bool operator==(iterator_t const &other) const noexcept { return remaining_.data() == other.remaining_.data(); }
        bool operator!=(end_sentinel_t) const noexcept { return remaining_.data() != nullptr; }
        bool operator==(end_sentinel_t) const noexcept { return remaining_.data() == nullptr; }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_), matcher_}; }
    iterator_t end() const noexcept { return {matcher_, end_sentinel_t {}}; }
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
    bool empty() const noexcept { return begin() == end_sentinel_t {}; }
    bool include_overlaps() const noexcept { return matcher_.skip_length() < matcher_.needle_length(); }

    /** Copies the matches into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the matches into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/**
 *  @brief A range of string slices representing the matches of a @b reverse-order substring search.
 *
 *  @note Lifetime semantics: Holds views only, as the factories accept only lvalues and borrowed
 *      views, so the caller keeps the bytes alive while the range is walked.
 *  @note For-loop optimized: Iterators are lightweight views with minimal register pressure, ideal
 *      for high-performance applications where cache efficiency and register allocation matter.
 *  @note Sentinel support: Supports sentinel-based iteration via <tt>operator==(end_sentinel_t)</tt> for
 *      efficient termination without constructing full end iterators.
 *  @note Compatible with C++20 ranges, @c std::string_view, and of course, StringZilla.
 *  @note Similar to a pair of @c boost::algorithm::find_iterator.
 */
template <typename string_type_, typename matcher_type_>
class rfind_matches_view {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;
    using string_view_type = typename string_view_for<string_type>::type;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_view_type;
    using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
    using reference = string_view_type; // Needed for compatibility with STL container constructors.

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    rfind_matches_view(string_type haystack, matcher_type needle) noexcept
        : matcher_(std::move(needle)), haystack_(std::move(haystack)) {}

    class iterator_t {
        matcher_type matcher_;

        /** From the start to the end of the current match, null once exhausted. */
        string_view_type remaining_;

        void rfind_() noexcept {
            auto const position = matcher_(remaining_);
            if (position != string_view_type::npos)
                remaining_.remove_suffix(remaining_.size() - position - matcher_.needle_length());
            else remaining_ = string_view_type();
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view_type;
        using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
        using reference = string_view_type; // Needed for compatibility with STL container constructors.

        iterator_t(string_view_type haystack, matcher_type const &matcher) noexcept
            : matcher_(matcher), remaining_(haystack) {
            rfind_();
        }
        iterator_t(matcher_type const &matcher, end_sentinel_t) noexcept : matcher_(matcher), remaining_() {}

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept {
            return remaining_.substr(remaining_.size() - matcher_.needle_length());
        }

        iterator_t &operator++() noexcept {
            // Only the zero-width match at the very start leaves nothing to skip.
            if (remaining_.empty()) remaining_ = string_view_type();
            else remaining_.remove_suffix(matcher_.skip_length()), rfind_();
            return *this;
        }

        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        // Assumes both iterators point to the same underlying string.
        // This has to be `.data() + .size()`, to be compatible with `std::string_view` on MSVC.
        bool operator!=(iterator_t const &other) const noexcept {
            return remaining_.data() + remaining_.size() != other.remaining_.data() + other.remaining_.size();
        }
        bool operator==(iterator_t const &other) const noexcept {
            return remaining_.data() + remaining_.size() == other.remaining_.data() + other.remaining_.size();
        }
        bool operator!=(end_sentinel_t) const noexcept { return remaining_.data() != nullptr; }
        bool operator==(end_sentinel_t) const noexcept { return remaining_.data() == nullptr; }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_), matcher_}; }
    iterator_t end() const noexcept { return {matcher_, end_sentinel_t {}}; }
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
    bool empty() const noexcept { return begin() == end_sentinel_t {}; }
    bool include_overlaps() const noexcept { return matcher_.skip_length() < matcher_.needle_length(); }

    /** Copies the matches into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the matches into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/**
 *  @brief A range of string slices for different splits of the data.
 *
 *  @note Lifetime semantics: Holds views only, as the factories accept only lvalues and borrowed
 *      views, so the caller keeps the bytes alive while the range is walked.
 *  @note For-loop optimized: Iterators are lightweight views with minimal register pressure, ideal
 *      for high-performance applications where cache efficiency and register allocation matter.
 *  @note Sentinel support: Supports sentinel-based iteration via <tt>operator==(end_sentinel_t)</tt> for
 *      efficient termination without constructing full end iterators.
 *  @note Compatible with C++20 ranges, @c std::string_view, and of course, StringZilla.
 *  @note Similar to a pair of @c boost::algorithm::split_iterator.
 *
 *  In some sense, represents the inverse operation to @c find_matches_view, as it reports not the
 *  search matches but the data between them. Meaning that for @c N search matches, there will be
 *  @c N+1 elements in the range, the last one empty when the text ends with a separator. Unlike
 *  ::find_matches_view, this range can't be empty, and an empty separator yields the whole text as
 *  its only segment. It also can't report overlapping intervals.
 */
template <typename string_type_, typename matcher_type_>
class find_splits_view {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;
    using string_view_type = typename string_view_for<string_type>::type;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_view_type;
    using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
    using reference = string_view_type; // Needed for compatibility with STL container constructors.

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    find_splits_view(string_type haystack, matcher_type needle) noexcept
        : matcher_(std::move(needle)), haystack_(std::move(haystack)) {}

    class iterator_t {
        matcher_type matcher_;

        /** The current segment, null once exhausted. */
        string_view_type segment_;

        /** The text after the separator ending @c segment_, null when no separator does. */
        string_view_type rest_;

        void split_(string_view_type text) noexcept {
            auto const position = matcher_(text);
            // Only an empty needle matches zero bytes, and it leaves the text whole.
            if (position == string_view_type::npos || !matcher_.needle_length())
                segment_ = text, rest_ = string_view_type();
            else segment_ = text.substr(0, position), rest_ = text.substr(position + matcher_.needle_length());
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view_type;
        using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
        using reference = string_view_type; // Needed for compatibility with STL container constructors.

        iterator_t(string_view_type haystack, matcher_type const &matcher) noexcept : matcher_(matcher) {
            // A null segment marks exhaustion, so a null text still yields its one empty segment.
            split_(haystack.data() ? haystack : string_view_type("", 0));
        }
        iterator_t(matcher_type const &matcher, end_sentinel_t) noexcept : matcher_(matcher), segment_(), rest_() {}

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept { return segment_; }

        iterator_t &operator++() noexcept {
            if (rest_.data()) split_(rest_);
            else segment_ = string_view_type();
            return *this;
        }

        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator_t const &other) const noexcept { return segment_.data() != other.segment_.data(); }
        bool operator==(iterator_t const &other) const noexcept { return segment_.data() == other.segment_.data(); }
        bool operator!=(end_sentinel_t) const noexcept { return segment_.data() != nullptr; }
        bool operator==(end_sentinel_t) const noexcept { return segment_.data() == nullptr; }
        bool is_last() const noexcept { return rest_.data() == nullptr; }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_), matcher_}; }
    iterator_t end() const noexcept { return {matcher_, end_sentinel_t {}}; }
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
    constexpr bool empty() const noexcept { return false; }

    /** Copies the matches into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the matches into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/**
 *  @brief A range of string slices for different splits of the data in @b reverse-order.
 *
 *  @note Lifetime semantics: Holds views only, as the factories accept only lvalues and borrowed
 *      views, so the caller keeps the bytes alive while the range is walked.
 *  @note For-loop optimized: Iterators are lightweight views with minimal register pressure, ideal
 *      for high-performance applications where cache efficiency and register allocation matter.
 *  @note Sentinel support: Supports sentinel-based iteration via <tt>operator==(end_sentinel_t)</tt> for
 *      efficient termination without constructing full end iterators.
 *  @note Compatible with C++20 ranges, @c std::string_view, and of course, StringZilla.
 *  @note Similar to a pair of @c boost::algorithm::split_iterator.
 *
 *  In some sense, represents the inverse operation to @c find_matches_view, as it reports not the
 *  search matches but the data between them. Meaning that for @c N search matches, there will be
 *  @c N+1 elements in the range, the last one empty when the text starts with a separator. Unlike
 *  ::find_matches_view, this range can't be empty, and an empty separator yields the whole text as
 *  its only segment. It also can't report overlapping intervals.
 */
template <typename string_type_, typename matcher_type_>
class rfind_splits_view {
  public:
    using string_type = string_type_;
    using matcher_type = matcher_type_;
    using string_view_type = typename string_view_for<string_type>::type;

    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using value_type = string_view_type;
    using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
    using reference = string_view_type; // Needed for compatibility with STL container constructors.

  private:
    matcher_type matcher_;
    string_type haystack_;

  public:
    rfind_splits_view(string_type haystack, matcher_type needle) noexcept
        : matcher_(std::move(needle)), haystack_(std::move(haystack)) {}

    class iterator_t {
        matcher_type matcher_;

        /** The current segment, null once exhausted. */
        string_view_type segment_;

        /** The text before the separator starting @c segment_, null when no separator does. */
        string_view_type rest_;

        void split_(string_view_type text) noexcept {
            auto const position = matcher_(text);
            // Only an empty needle matches zero bytes, and it leaves the text whole.
            if (position == string_view_type::npos || !matcher_.needle_length())
                segment_ = text, rest_ = string_view_type();
            else segment_ = text.substr(position + matcher_.needle_length()), rest_ = text.substr(0, position);
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = string_view_type;
        using pointer = string_view_type;   // Needed for compatibility with STL container constructors.
        using reference = string_view_type; // Needed for compatibility with STL container constructors.

        iterator_t(string_view_type haystack, matcher_type const &matcher) noexcept : matcher_(matcher) {
            // A null segment marks exhaustion, so a null text still yields its one empty segment.
            split_(haystack.data() ? haystack : string_view_type("", 0));
        }
        iterator_t(matcher_type const &matcher, end_sentinel_t) noexcept : matcher_(matcher), segment_(), rest_() {}

        pointer operator->() const noexcept = delete;
        value_type operator*() const noexcept { return segment_; }

        iterator_t &operator++() noexcept {
            if (rest_.data()) split_(rest_);
            else segment_ = string_view_type();
            return *this;
        }

        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        bool operator!=(iterator_t const &other) const noexcept { return segment_.data() != other.segment_.data(); }
        bool operator==(iterator_t const &other) const noexcept { return segment_.data() == other.segment_.data(); }
        bool operator!=(end_sentinel_t) const noexcept { return segment_.data() != nullptr; }
        bool operator==(end_sentinel_t) const noexcept { return segment_.data() == nullptr; }
        bool is_last() const noexcept { return rest_.data() == nullptr; }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_), matcher_}; }
    iterator_t end() const noexcept { return {matcher_, end_sentinel_t {}}; }
    size_type size() const noexcept { return static_cast<size_type>(ssize()); }
    difference_type ssize() const noexcept { return std::distance(begin(), end()); }
    constexpr bool empty() const noexcept { return false; }

    /** Copies the matches into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the matches into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/**
 *  @brief Finds all potentially @b overlapping inclusions of @p needle.
 *  @note An @b empty needle matches at every offset from 0 to <tt>size()</tt> inclusive.
 */
template <typename haystack_type_, typename needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
find_matches_view<range_view_type<haystack_type_>, matcher_find<range_view_type<haystack_type_>, include_overlaps_t>>
find_all(haystack_type_ &&haystack, needle_type_ &&needle, include_overlaps_t = {}) noexcept {
    using view_type = range_view_type<haystack_type_>;
    return {view_type(haystack), {view_type(needle)}};
}

/** Finds all potentially @b overlapping inclusions of @p needle, in @b reverse order. */
template <typename haystack_type_, typename needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
rfind_matches_view<range_view_type<haystack_type_>, matcher_rfind<range_view_type<haystack_type_>, include_overlaps_t>>
rfind_all(haystack_type_ &&haystack, needle_type_ &&needle, include_overlaps_t = {}) noexcept {
    using view_type = range_view_type<haystack_type_>;
    return {view_type(haystack), {view_type(needle)}};
}

/** Finds all @b non-overlapping inclusions of @p needle. */
template <typename haystack_type_, typename needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
find_matches_view<range_view_type<haystack_type_>, matcher_find<range_view_type<haystack_type_>, exclude_overlaps_t>>
find_all(haystack_type_ &&haystack, needle_type_ &&needle, exclude_overlaps_t) noexcept {
    using view_type = range_view_type<haystack_type_>;
    return {view_type(haystack), {view_type(needle)}};
}

/** Finds all @b non-overlapping inclusions of @p needle, in @b reverse order. */
template <typename haystack_type_, typename needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
rfind_matches_view<range_view_type<haystack_type_>, matcher_rfind<range_view_type<haystack_type_>, exclude_overlaps_t>>
rfind_all(haystack_type_ &&haystack, needle_type_ &&needle, exclude_overlaps_t) noexcept {
    using view_type = range_view_type<haystack_type_>;
    return {view_type(haystack), {view_type(needle)}};
}

/** Finds every byte of @p characters, a set or a string of them. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
find_matches_view<range_view_type<haystack_type_>, matcher_find_first_of<range_view_type<haystack_type_>>>
find_all_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters)}};
}

/** Finds every byte of @p characters, in @b reverse order. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
rfind_matches_view<range_view_type<haystack_type_>, matcher_find_last_of<range_view_type<haystack_type_>>>
rfind_all_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters)}};
}

/** Finds every byte outside @p characters. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
find_matches_view<range_view_type<haystack_type_>, matcher_find_first_of<range_view_type<haystack_type_>>>
find_all_other_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters).inverted()}};
}

/** Finds every byte outside @p characters, in @b reverse order. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
rfind_matches_view<range_view_type<haystack_type_>, matcher_find_last_of<range_view_type<haystack_type_>>>
rfind_all_other_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters).inverted()}};
}

/**
 *  @brief Splits around every @b non-overlapping inclusion of @p needle.
 *  @note An @b empty delimiter never splits, yielding the whole string as the only segment.
 */
template <typename haystack_type_, typename needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
find_splits_view<range_view_type<haystack_type_>, matcher_find<range_view_type<haystack_type_>, exclude_overlaps_t>>
split(haystack_type_ &&haystack, needle_type_ &&needle) noexcept {
    using view_type = range_view_type<haystack_type_>;
    return {view_type(haystack), {view_type(needle)}};
}

/** Splits around every @b non-overlapping inclusion of @p needle, in @b reverse order. */
template <typename haystack_type_, typename needle_type_>
    requires a_borrowed_string<haystack_type_ &&> && a_borrowed_string<needle_type_ &&>
rfind_splits_view<range_view_type<haystack_type_>, matcher_rfind<range_view_type<haystack_type_>, exclude_overlaps_t>>
rsplit(haystack_type_ &&haystack, needle_type_ &&needle) noexcept {
    using view_type = range_view_type<haystack_type_>;
    return {view_type(haystack), {view_type(needle)}};
}

/** Splits around every byte of @p characters, like <tt>split_characters(text, whitespaces_set())</tt>. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
find_splits_view<range_view_type<haystack_type_>, matcher_find_first_of<range_view_type<haystack_type_>>>
split_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters)}};
}

/** Splits around every byte of @p characters, in @b reverse order. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
rfind_splits_view<range_view_type<haystack_type_>, matcher_find_last_of<range_view_type<haystack_type_>>>
rsplit_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters)}};
}

/** Splits around every byte outside @p characters. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
find_splits_view<range_view_type<haystack_type_>, matcher_find_first_of<range_view_type<haystack_type_>>>
split_other_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters).inverted()}};
}

/** Splits around every byte outside @p characters, in @b reverse order. */
template <typename haystack_type_, typename characters_type_>
    requires a_borrowed_string<haystack_type_ &&>
rfind_splits_view<range_view_type<haystack_type_>, matcher_find_last_of<range_view_type<haystack_type_>>>
rsplit_other_characters(haystack_type_ &&haystack, characters_type_ const &characters) noexcept {
    return {range_view_type<haystack_type_>(haystack), {byteset_of_(characters).inverted()}};
}

/** Counts the potentially @b overlapping inclusions of @p needle, @c size()+1 for an empty one. */
template <a_string haystack_type_, a_string needle_type_>
std::size_t count(haystack_type_ const &haystack, needle_type_ const &needle, include_overlaps_t = {}) noexcept {
    string_view_t const text = view_of(haystack);
    return find_all(text, view_of(needle)).size();
}

/** Counts the @b non-overlapping inclusions of @p needle, like Python's <tt>str.count</tt>. */
template <a_string haystack_type_, a_string needle_type_>
std::size_t count(haystack_type_ const &haystack, needle_type_ const &needle, exclude_overlaps_t) noexcept {
    string_view_t const text = view_of(haystack);
    return find_all(text, view_of(needle), exclude_overlaps).size();
}

template <typename allocator_type_, typename pattern_type_>
status_t replace_all_(string<allocator_type_> &text, pattern_type_ pattern, string_view_t replacement) noexcept {
    using pointer = typename string<allocator_type_>::pointer;
    // Depending on the size of the pattern and the replacement, we may need to allocate more space.
    // There are 3 cases to consider:
    // 1. The pattern and the replacement are of the same length. Piece of cake!
    // 2. The pattern is longer than the replacement. We need to compact the strings.
    // 3. The pattern is shorter than the replacement. We may have to allocate more memory.
    using matcher_type =
        typename std::conditional<std::is_same_v<pattern_type_, byteset_t>, matcher_find_first_of<string_view_t>,
                                  matcher_find<string_view_t, exclude_overlaps_t>>::type;
    matcher_type matcher {pattern};
    string_view_t this_view = text.view();

    // Empty pattern: nothing to replace.
    if (matcher.needle_length() == 0) return status_t::success_k;

    // 1. The pattern and the replacement are of the same length.
    if (matcher.needle_length() == replacement.length()) {
        using matches_type = find_matches_view<string_view_t, matcher_type>;
        // Stops at the cheaper sentinel, not at the `end()` this range loop would build:
        //      for (string_view_t match : matches) { ... }
        matches_type matches(this_view, matcher);
        for (auto matches_iterator = matches.begin(); matches_iterator != end_sentinel_t {}; ++matches_iterator) {
            replacement.copy(const_cast<pointer>((*matches_iterator).data()));
        }
        return status_t::success_k;
    }

    // 2. The pattern is longer than the replacement. We need to compact the strings.
    else if (matcher.needle_length() > replacement.length()) {
        // Shorter replacements need no allocations: walking the split ranges, each gap between
        // matches moves once, right after `compacted_end`, the end of the last compacted area.
        using splits_type = find_splits_view<string_view_t, matcher_type>;
        splits_type splits(this_view, matcher);
        auto matches_iterator = splits.begin();
        auto compacted_end = (*matches_iterator).end();
        if (compacted_end == text.view().end()) return status_t::success_k; // No matches.

        ++matches_iterator; // Skip the first match.
        do {
            string_view_t match_view = *matches_iterator;
            replacement.copy(const_cast<pointer>(compacted_end));
            compacted_end += replacement.length();
            std::memmove(const_cast<pointer>(compacted_end), match_view.begin(), match_view.length());
            compacted_end += match_view.length();
            ++matches_iterator;
        } while (matches_iterator != end_sentinel_t {});

        return text.resize(static_cast<std::size_t>(compacted_end - text.view().begin()));
    }

    // 3. The pattern is shorter than the replacement. We may have to allocate more memory.
    else {
        using rmatcher_type =
            typename std::conditional<std::is_same_v<pattern_type_, byteset_t>, matcher_find_last_of<string_view_t>,
                                      matcher_rfind<string_view_t, exclude_overlaps_t>>::type;
        using rmatches_type = rfind_matches_view<string_view_t, rmatcher_type>;
        rmatches_type rmatches(this_view, rmatcher_type {pattern});

        // It's cheaper to iterate through the whole string once, counting the number of matches,
        // reserving memory once, than re-allocating and copying the string multiple times.
        auto matches_count = rmatches.size();
        if (matches_count == 0) return status_t::success_k; // No matches.

        // TODO: Resize without initializing the memory.
        auto replacement_delta_length = replacement.length() - matcher.needle_length();
        auto added_length = matches_count * replacement_delta_length;
        auto old_length = text.size();
        auto new_length = old_length + added_length;
        status_t const status = text.resize(new_length);
        if (failed(status)) return status;
        this_view = text.view().front(old_length);

        // Now iterate through splits similarly to the 2nd case, but in reverse order.
        using rsplits_type = rfind_splits_view<string_view_t, rmatcher_type>;
        rsplits_type splits(this_view, rmatcher_type {pattern});
        auto splits_iterator = splits.begin();

        // Put the compacted pointer to the end of the new string, and walk left.
        auto compacted_begin = this_view.data() + new_length;

        // By now we know that at least one match exists, which means the splits .
        do {
            string_view_t slice_view = *splits_iterator;
            compacted_begin -= slice_view.length();
            std::memmove(const_cast<pointer>(compacted_begin), slice_view.begin(), slice_view.length());
            compacted_begin -= replacement.length();
            replacement.copy(const_cast<pointer>(compacted_begin));
            ++splits_iterator;
        } while (!splits_iterator.is_last());

        return status_t::success_k;
    }
}

/**
 *  @brief Replaces @b (in-place) every non-overlapping inclusion of @p pattern with @p replacement.
 *  @note Similar to @c boost::algorithm::replace_all and Python's <tt>str.replace</tt>.
 */
template <typename allocator_type_>
status_t replace_all(string<allocator_type_> &text, string_view_t pattern, string_view_t replacement) noexcept {
    return replace_all_(text, pattern, replacement);
}

/** Replaces @b (in-place) every byte of @p pattern with @p replacement. */
template <typename allocator_type_>
status_t replace_all(string<allocator_type_> &text, byteset_t pattern, string_view_t replacement) noexcept {
    return replace_all_(text, pattern, replacement);
}

#pragma endregion Ranges of Search Matches

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_FIND_HPP_
