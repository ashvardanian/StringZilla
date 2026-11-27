/**
 *  @file cpp/stringzilla/utf8_runes.hpp
 *  @author Ash Vardanian
 *  @date December 19, 2023
 *  @brief UTF-8 rune counts, seeks and the lazy range of runes, through the dispatch points of
 *      @c utf8_runes.h, with the split and segment ranges the other UTF-8 families share.
 */
#ifndef STRINGZILLA_UTF8_RUNES_HPP_
#define STRINGZILLA_UTF8_RUNES_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/utf8_runes.h"      // `sz_utf8_count_best`, `sz_utf8_seek_best`, `sz_utf8_decode_best`
#include "stringzilla/utf8_wordbreaks.h" // `sz_iterators_default_steps_k`

#include "stringzilla/find.hpp" // `end_sentinel_t`, `a_borrowed_string`, `string_view_for`

namespace ashvardanian {
namespace stringzilla {

/**
 *  @brief Counts the UTF-8 runes, not bytes, of @p text.
 *  @sa sz_utf8_count_best
 */
template <a_string text_type_>
expected<std::size_t> utf8_count(text_type_ const &text, sz_capability_t capabilities, sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text);
    sz_size_t runes = 0;
    sz_status_t const status = sz_utf8_count_best(view.data(), view.size(), &runes, capabilities, stream);
    return {runes, static_cast<status_t>(status)};
}

/** @copydoc utf8_count */
template <a_string text_type_>
std::size_t utf8_count(text_type_ const &text) noexcept {
    return default_mask_value_(utf8_count(text, default_capabilities(), nullptr));
}

/**
 *  @brief The byte offset of rune @p runes of @p text, counting from zero.
 *  @return The offset, or @ref npos_of the text when it holds no more than @p runes runes.
 *  @sa sz_utf8_seek_best
 */
template <a_string text_type_>
expected<std::size_t> utf8_seek(text_type_ const &text, std::size_t runes, sz_capability_t capabilities,
                                sz_stream_t stream) noexcept {
    string_view_t const view = view_of(text);
    sz_cptr_t position = nullptr;
    sz_status_t const status = sz_utf8_seek_best(view.data(), view.size(), runes, &position, capabilities, stream);
    return {position ? static_cast<std::size_t>(position - view.data()) : npos_of<text_type_>(),
            static_cast<status_t>(status)};
}

/** @copydoc utf8_seek */
template <a_string text_type_>
std::size_t utf8_seek(text_type_ const &text, std::size_t runes) noexcept {
    return default_mask_value_(utf8_seek(text, runes, default_capabilities(), nullptr));
}

/**
 *  @brief A range view over UTF-8 characters (codepoints) in a string.
 *
 *  Iterates over UTF-32 codepoints decoded from UTF-8 bytes using efficient batched decoding. Each
 *  refill decodes up to @p steps_ codepoints in a single @c sz_utf8_decode_best call (the decoder
 *  fills the whole buffer regardless of script width), then yields them one by one. Ill-formed
 *  bytes, a truncated sequence at the end included, decode to U+FFFD.
 *
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 *  @tparam steps_ Codepoints buffered per decode call, the batch width, defaulting to the shared
 *      @c sz_iterators_default_steps_k constant.
 */
template <typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k>
class utf8_runes_view {
  public:
    using string_type = string_type_;
    using string_view_type = typename string_view_for<string_type>::type;
    using value_type = sz_rune_t;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    string_type haystack_;

  public:
    utf8_runes_view() noexcept = default;
    utf8_runes_view(string_type haystack) noexcept : haystack_(haystack) {}

    class iterator_t {

        /** The first byte not decoded yet. */
        char const *next_;
        char const *end_;

        sz_rune_t runes_[steps_];

        /** Runes in the batch, zero once exhausted, as any bytes left decode to at least one. */
        sz_size_t runes_count_;
        size_type runes_index_;

        void decode_() noexcept {
            sz_size_t consumed = 0;
            [[maybe_unused]] sz_status_t const status = sz_utf8_decode_best(next_, static_cast<size_type>(end_ - next_),
                                                                            runes_, steps_, &runes_count_, &consumed,
                                                                            default_capabilities(), nullptr);
            sz_assert_(status == sz_success_k);
            next_ += consumed;
            runes_index_ = 0;
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = sz_rune_t;
        using difference_type = std::ptrdiff_t;
        using pointer = sz_rune_t const *;
        using reference = sz_rune_t;

        iterator_t() noexcept : next_(nullptr), end_(nullptr), runes_count_(0), runes_index_(0) {}
        iterator_t(string_view_type text) noexcept : next_(text.data()), end_(text.data() + text.size()) { decode_(); }
        iterator_t(string_view_type text, end_sentinel_t) noexcept
            : next_(text.data() + text.size()), end_(next_), runes_count_(0), runes_index_(0) {}

        reference operator*() const noexcept { return runes_[runes_index_]; }
        pointer operator->() const noexcept { return &runes_[runes_index_]; }

        iterator_t &operator++() noexcept {
            if (++runes_index_ == runes_count_) decode_();
            return *this;
        }

        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        /**
         *  @brief Advance the iterator by @p n UTF-8 codepoints, decoding new batches as needed.
         *  @note Forward-only, so negative offsets are unsupported; skips bytes via the fast C API.
         */
        iterator_t &operator+=(size_type n) noexcept {
            size_type const buffered = runes_count_ - runes_index_;
            if (n < buffered) {
                runes_index_ += n;
                return *this;
            }
            // Past the batch, the remaining runes are counted from `next_`, the first one after it.
            sz_cptr_t target = nullptr;
            [[maybe_unused]] sz_status_t const status = sz_utf8_seek_best(
                next_, static_cast<size_type>(end_ - next_), n - buffered, &target, default_capabilities(), nullptr);
            sz_assert_(status == sz_success_k);
            next_ = target ? target : end_;
            decode_();
            return *this;
        }

        iterator_t operator+(size_type n) const noexcept {
            iterator_t shifted = *this;
            shifted += n;
            return shifted;
        }

        bool operator==(iterator_t const &other) const noexcept {
            return next_ == other.next_ && runes_count_ - runes_index_ == other.runes_count_ - other.runes_index_;
        }
        bool operator==(end_sentinel_t) const noexcept { return runes_count_ == 0; }
        bool operator!=(iterator_t const &other) const noexcept { return !(*this == other); }
        bool operator!=(end_sentinel_t sentinel) const noexcept { return !(*this == sentinel); }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_)}; }
    iterator_t end() const noexcept { return {string_view_type(haystack_), end_sentinel_t {}}; }
    end_sentinel_t end_sentinel() const noexcept { return {}; }

    /** Count UTF-8 characters in the string. */
    size_type size() const noexcept { return utf8_count(string_view_type(haystack_)); }

    difference_type ssize() const noexcept { return static_cast<difference_type>(size()); }
    bool empty() const noexcept { return string_view_type(haystack_).empty(); }

    /** Copies the characters into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the characters into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/** Which parts of the split a @c utf8_split_view yields at each separator the kernel reports. */
enum class split_parts_t {

    /** The default: the segments @b between separators, as in @c split_newlines,
     *  @c split_whitespaces, and @c split_delimiters. */
    between_k,

    /** The separator runs themselves: newlines, whitespace, or delimiter codepoints. */
    separators_k,

    /** Interleaved segment, separator, segment, and so on, so rejoining them yields the input. */
    both_k,
};

/** Whether a @ref utf8_split_view keeps or drops empty (zero-length) segments. */
enum class empty_segments_t { keep_k, skip_k };

/** A token kernel's dispatch point, like @c sz_utf8_newlines_best, which @c utf8_split_view
 *  drives over @c default_capabilities. */
using utf8_tokenizer_best_t = sz_status_t (*)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t, sz_size_t *,
                                              sz_size_t *, sz_capability_t, void *);

/** A tiling segmenter's dispatch point, like @c sz_utf8_graphemes_best, which
 *  @c utf8_segments_view drives over @c default_capabilities. */
using utf8_segmenter_best_t = sz_status_t (*)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t, sz_size_t *,
                                              sz_capability_t, void *);

/**
 *  @brief A range of string slices split on the delimiter codepoints that a @b token kernel
 *      reports to it.
 *
 *  The shared engine behind the between-segment views @c utf8_split_newlines_view,
 *  @c utf8_split_whitespaces_view, and @c utf8_split_delimiters_view, and the delimiter-run views
 *  @c utf8_newlines_view, @c utf8_whitespaces_view, and @c utf8_delimiters_view. The kernel reports
 *  the byte spans of the @b delimiters, and @p parts_ selects what the view yields:
 *
 *  - @c between_k: the N+1 segments between delimiters, the friendly lines, tokens, or fields;
 *  - @c separators_k: the N delimiter runs themselves, the newlines, whitespace, or delimiters;
 *  - @c both_k: the 2N+1 items interleaved, so concatenating them reconstructs the input exactly.
 *
 *  Each batch's separator endpoints are expanded once into a single boundary array, so every span
 *  is just <tt>bounds_[i] .. bounds_[i+1]</tt> and @c parts_ reduces to a compile-time @b (first,stride)
 *  over it: the iterator's hot path is one branchless formula shared by all three modes.
 *  @c empties_, a compile-time switch reachable via <tt>.skip_empty()</tt>, drops empty segments, and
 *  @c both_k is lossless only when empties are kept.
 *
 *  @tparam tokenizer_ A token kernel reporting delimiter spans, like @c sz_utf8_whitespaces_best.
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 *  @tparam steps_ Delimiters fetched per kernel call.
 *  @tparam parts_ Which parts to yield: between segments, the delimiters, or both interleaved.
 *  @tparam empties_ Whether empty segments are kept or skipped.
 */
template <utf8_tokenizer_best_t tokenizer_, typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k,
          split_parts_t parts_ = split_parts_t::between_k, empty_segments_t empties_ = empty_segments_t::keep_k>
class utf8_split_view {
  public:
    using string_type = string_type_;
    using string_view_type = typename string_view_for<string_type>::type;
    using value_type = string_view_type;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    string_type haystack_;

  public:
    utf8_split_view() noexcept = default;
    utf8_split_view(string_type haystack) noexcept : haystack_(haystack) {}

    class iterator_t {

        /**
         *  @brief Span selection as a compile-time @b (first,stride) over the boundary array.
         *
         *  The kernel reports separator spans, and their endpoints are the span boundaries:
         *  <tt>{start, sep0.start, sep0.end, sep1.start, ..., [end]}</tt>, with span @c i covering
         *  <tt>bounds_[i] .. bounds_[i+1]</tt>. So the three modes are just a starting boundary and a
         *  stride: @c between_k walks the even spans (the gaps), @c separators_k the odd ones,
         *  @c both_k every span, and @c operator* is one branchless formula for all.
         */
        static constexpr size_type first_k = parts_ == split_parts_t::separators_k ? 1 : 0;
        static constexpr size_type stride_k = parts_ == split_parts_t::both_k ? 1 : 2;

        /** End of the text. */
        char const *end_;

        /** Span boundaries of the current batch: its start, each separator's start and end, and
         *  the text end once the batch reaches it. */
        char const *bounds_[2 * steps_ + 2];

        /** Cursor: the current span is <tt>bounds_[index_] .. bounds_[index_ + 1]</tt>. */
        size_type index_;

        /** Spans in the batch, so <tt>bounds_[limit_]</tt> is its last boundary: the next batch start. */
        size_type limit_;

        /** Fetches the separators from @p start on, expanding their endpoints into @c bounds_. */
        void refill_(char const *start) noexcept {
            sz_size_t offsets[steps_], lengths[steps_], count = 0, consumed = 0;
            size_type const length = static_cast<size_type>(end_ - start);
            [[maybe_unused]] sz_status_t const status = tokenizer_(start, length, offsets, lengths, steps_, &count,
                                                                   &consumed, default_capabilities(), nullptr);
            sz_assert_(status == sz_success_k);
            bounds_[0] = start;
            for (size_type i = 0; i != count; ++i)
                bounds_[2 * i + 1] = start + offsets[i], bounds_[2 * i + 2] = start + offsets[i] + lengths[i];
            limit_ = 2 * count;
            // The end closes the trailing gap, empty after a trailing separator, keeping `both_k`
            // lossless, while a full batch stops at its last separator, where the next one resumes.
            if (consumed == length) bounds_[++limit_] = end_;
            index_ = first_k;
        }

        /** Land @c index_ on the next yieldable span, refilling and (when @c skip_k) skipping empty
         *  spans. */
        void settle_() noexcept {
            for (;;) {
                for (; index_ < limit_; index_ += stride_k)
                    if (empties_ == empty_segments_t::keep_k || bounds_[index_ + 1] != bounds_[index_]) return;
                if (bounds_[limit_] == end_) return;
                refill_(bounds_[limit_]);
            }
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = string_view_type;
        using difference_type = std::ptrdiff_t;
        using pointer = string_view_type;
        using reference = string_view_type;

        iterator_t() noexcept : end_(nullptr), index_(0), limit_(0) {}
        iterator_t(string_view_type text) noexcept : end_(text.data() + text.size()) {
            refill_(text.data()), settle_();
        }

        reference operator*() const noexcept {
            return string_view_type(bounds_[index_], static_cast<size_type>(bounds_[index_ + 1] - bounds_[index_]));
        }
        pointer operator->() const noexcept { return **this; }

        iterator_t &operator++() noexcept {
            index_ += stride_k, settle_();
            return *this;
        }
        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        bool at_end_() const noexcept { return index_ >= limit_; }
        bool operator==(iterator_t const &other) const noexcept {
            return at_end_() ? other.at_end_()
                             : (!other.at_end_() && bounds_[0] == other.bounds_[0] && index_ == other.index_);
        }
        bool operator!=(iterator_t const &other) const noexcept { return !(*this == other); }
        bool operator==(end_sentinel_t) const noexcept { return at_end_(); }
        bool operator!=(end_sentinel_t) const noexcept { return !at_end_(); }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_)}; }
    iterator_t end() const noexcept { return {}; }
    end_sentinel_t end_sentinel() const noexcept { return {}; }

    /** The same split with empty segments dropped (compile-time, branchless). */
    utf8_split_view<tokenizer_, string_type_, steps_, parts_, empty_segments_t::skip_k> skip_empty() const noexcept {
        return {haystack_};
    }

    /** The same split yielding segments @b and delimiters interleaved, losslessly, turning
     *  @c between_k into @c both_k. */
    utf8_split_view<tokenizer_, string_type_, steps_, split_parts_t::both_k, empties_> with_separators()
        const noexcept {
        return {haystack_};
    }

    /** Copies the items into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the items into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/**
 *  @brief A range of string slices that @b tile the input, as a segmentation kernel reports them.
 *
 *  The shared engine behind @c utf8_wordbreaks_view, @c utf8_graphemes_view,
 *  @c utf8_sentences_view, and @c utf8_linebreaks_view. The kernel returns the tiling units
 *  directly: every byte belongs to exactly one unit, so consecutive units are contiguous and no
 *  empty segments arise.
 *
 *  @tparam segmenter_ A segmenter returning tiling units, like @c sz_utf8_wordbreaks_best.
 *  @tparam string_type_ String type, like @c string_view_t, @c string_slice_t, or @c std::string.
 *  @tparam steps_ Units fetched per kernel call.
 */
template <utf8_segmenter_best_t segmenter_, typename string_type_, std::size_t steps_ = sz_iterators_default_steps_k>
class utf8_segments_view {
  public:
    using string_type = string_type_;
    using string_view_type = typename string_view_for<string_type>::type;
    using value_type = string_view_type;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

  private:
    string_type haystack_;

  public:
    utf8_segments_view() noexcept = default;
    utf8_segments_view(string_type haystack) noexcept : haystack_(haystack) {}

    class iterator_t {

        /** Start of the current unit, a boundary, and the text end once exhausted. */
        char const *suffix_;

        /** End of the original text (immutable). */
        char const *end_;

        /** Buffered unit lengths, each unit starting where the previous one ends. */
        sz_size_t lengths_[steps_];

        /** Number of buffered units (0 once exhausted). */
        sz_size_t count_;

        /** Index of the current unit within the buffer. */
        size_type index_;

        void fill_() noexcept {
            [[maybe_unused]] sz_status_t const status = segmenter_(suffix_, static_cast<size_type>(end_ - suffix_),
                                                                   lengths_, steps_, &count_, default_capabilities(),
                                                                   nullptr);
            sz_assert_(status == sz_success_k);
            index_ = 0;
        }

      public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = string_view_type;
        using difference_type = std::ptrdiff_t;
        using pointer = string_view_type;
        using reference = string_view_type;

        iterator_t() noexcept : suffix_(nullptr), end_(nullptr), count_(0), index_(0) {}
        iterator_t(string_view_type text) noexcept : suffix_(text.data()), end_(text.data() + text.size()) { fill_(); }

        reference operator*() const noexcept { return string_view_type(suffix_, lengths_[index_]); }
        pointer operator->() const noexcept { return **this; }

        iterator_t &operator++() noexcept {
            suffix_ += lengths_[index_];
            if (++index_ == count_) fill_(); // `count_` hits 0 at the end.
            return *this;
        }

        iterator_t operator++(int) noexcept {
            iterator_t temp = *this;
            ++(*this);
            return temp;
        }

        bool operator==(iterator_t const &other) const noexcept { return suffix_ == other.suffix_; }
        bool operator!=(iterator_t const &other) const noexcept { return suffix_ != other.suffix_; }
        bool operator!=(end_sentinel_t) const noexcept { return count_ != 0; }
        bool operator==(end_sentinel_t) const noexcept { return count_ == 0; }
    };

    iterator_t begin() const noexcept { return {string_view_type(haystack_)}; }
    end_sentinel_t end() const noexcept { return {}; }
    end_sentinel_t end_sentinel() const noexcept { return {}; }

    /** Copies the segments into a container. */
    template <typename container_>
    void to(container_ &container) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
    }

    /** Copies the segments into a consumed container, returning it at the end. */
    template <typename container_>
    container_ to(container_ &&container = {}) {
        for (auto it_ = begin(); it_ != end_sentinel_t {}; ++it_) container.push_back(*it_);
        return std::move(container);
    }
};

/** A lazy range of the UTF-32 runes decoded from @p text, an lvalue or a borrowed view. */
template <a_string text_type_>
    requires a_borrowed_string<text_type_ &&>
utf8_runes_view<string_view_t> utf8_runes(text_type_ &&text) noexcept {
    return {view_of(text)};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_UTF8_RUNES_HPP_
