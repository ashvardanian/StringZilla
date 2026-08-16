/**
 *  @brief Immutable exact Levenshtein dictionary retrieval.
 *  @file include/stringzillas/levenshtein_index.hpp
 *  @author Guillaume de Rouville
 */
#ifndef STRINGZILLAS_LEVENSHTEIN_INDEX_HPP_
#define STRINGZILLAS_LEVENSHTEIN_INDEX_HPP_

#include "stringzillas/types.hpp"
#include "stringzillas/levenshtein_index.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace ashvardanian {
namespace stringzillas {

using levenshtein_index_match_t = ::szs_levenshtein_index_match_t;

/**
 *  @brief Owns an immutable dictionary and returns matching IDs with exact distances.
 *
 *  The index can be shared by readers after construction. Each reader owns a @c scratch_t and output vector.
 *  Duplicate strings keep distinct IDs and result order is unspecified.
 */
template <typename symbol_type_ = char, typename allocator_type_ = std::allocator<char>>
class basic_levenshtein_index {
  public:
    using symbol_t = symbol_type_;
    using allocator_t = allocator_type_;
    using match_t = levenshtein_index_match_t;

    static_assert(std::is_integral<symbol_t>::value &&
                      !std::is_same<typename std::remove_cv<symbol_t>::type, bool>::value &&
                      sizeof(symbol_t) <= sizeof(u32_t),
                  "Levenshtein index symbols must be integral and at most 32 bits wide");

  private:
    static constexpr size_t min_prefix_bits_k = 20;
    static constexpr size_t max_prefix_bits_k = 25;
    static constexpr size_t packed_id_bits_k = 20;
    static constexpr u32_t packed_id_mask_k = (u32_t(1) << packed_id_bits_k) - 1;

    struct directory_word_t {
        u32_t rank = 0;
        u32_t bits_low = 0;
        u32_t bits_high = 0;
    };
    static_assert(sizeof(directory_word_t) == 12, "Ranked directory words must remain compact");

    template <typename value_type_>
    using rebound_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_type_>;
    template <typename value_type_>
    using vector_t = safe_vector<value_type_, rebound_allocator_t<value_type_>>;

    allocator_t alloc_ {};
    vector_t<symbol_t> tape_ {alloc_};
    vector_t<u64_t> offsets_ {alloc_};
    vector_t<u32_t> directory_ {alloc_};
    vector_t<directory_word_t> directory_words_ {alloc_};
    vector_t<u32_t> packed_records_ {alloc_};
    vector_t<u64_t> wide_records_ {alloc_};
    size_t max_word_length_ = 0;
    u8_t prefix_bits_ = min_prefix_bits_k;
    u8_t suffix_bits_ = 32 - min_prefix_bits_k;
    u32_t suffix_mask_ = (u32_t(1) << (32 - min_prefix_bits_k)) - 1;

    static u32_t fold_hash_(u64_t hash) noexcept {
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdull;
        hash ^= hash >> 33;
        return static_cast<u32_t>(hash ^ (hash >> 32));
    }

    static u32_t hash_word_(span<symbol_t const> word) noexcept {
        static constexpr u64_t base = 0x9E3779B185EBCA87ull;
        u64_t hash = 0;
        for (symbol_t symbol : word) {
            using unsigned_symbol_t = typename std::make_unsigned<symbol_t>::type;
            hash = hash * base + static_cast<u64_t>(static_cast<unsigned_symbol_t>(symbol)) + 1;
        }
        return fold_hash_(hash);
    }

    span<symbol_t const> word_(u32_t id) const noexcept {
        size_t const begin = static_cast<size_t>(offsets_[id]);
        size_t const end = static_cast<size_t>(offsets_[id + 1]);
        return {tape_.data() + begin, end - begin};
    }

    bool equals_(u32_t id, span<symbol_t const> query) const noexcept {
        span<symbol_t const> const candidate = word_(id);
        return candidate.size() == query.size() &&
               (candidate.empty() ||
                std::memcmp(candidate.data(), query.data(), query.size() * sizeof(symbol_t)) == 0);
    }

    status_t build_directory_() noexcept {
        prefix_bits_ = min_prefix_bits_k;
        while (prefix_bits_ != max_prefix_bits_k && wide_records_.size() > (size_t(1) << (prefix_bits_ - 3)))
            ++prefix_bits_;
        suffix_bits_ = static_cast<u8_t>(32 - prefix_bits_);
        suffix_mask_ = (u32_t(1) << suffix_bits_) - 1;
        size_t const prefix_buckets = size_t(1) << prefix_bits_;

        size_t nonempty_prefixes = 0;
        size_t cursor = 0;
        while (cursor != wide_records_.size()) {
            size_t const prefix = u32_t(wide_records_[cursor] >> 32) >> suffix_bits_;
            ++nonempty_prefixes;
            do
                ++cursor;
            while (cursor != wide_records_.size() &&
                   (u32_t(wide_records_[cursor] >> 32) >> suffix_bits_) == prefix);
        }

        size_t const directory_words = (prefix_buckets + 63) / 64;
        size_t const dense_bytes = (prefix_buckets + 1) * sizeof(u32_t);
        size_t const ranked_bytes = directory_words * sizeof(directory_word_t) +
                                    (nonempty_prefixes + 1) * sizeof(u32_t);
        if (ranked_bytes < dense_bytes) {
            if (directory_words_.try_resize(directory_words) != status_t::success_k ||
                directory_.try_resize(nonempty_prefixes + 1) != status_t::success_k)
                return status_t::bad_alloc_k;
            std::fill(directory_words_.begin(), directory_words_.end(), directory_word_t {});
            cursor = 0;
            size_t nonempty = 0;
            while (cursor != wide_records_.size()) {
                size_t const prefix = u32_t(wide_records_[cursor] >> 32) >> suffix_bits_;
                directory_word_t &word = directory_words_[prefix / 64];
                size_t const bit = prefix % 64;
                if (bit < 32) word.bits_low |= u32_t(1) << bit;
                else word.bits_high |= u32_t(1) << (bit - 32);
                directory_[nonempty++] = static_cast<u32_t>(cursor);
                do
                    ++cursor;
                while (cursor != wide_records_.size() &&
                       (u32_t(wide_records_[cursor] >> 32) >> suffix_bits_) == prefix);
            }
            directory_[nonempty] = static_cast<u32_t>(wide_records_.size());
            u32_t rank = 0;
            for (size_t word = 0; word != directory_words; ++word) {
                directory_words_[word].rank = rank;
                u64_t const bits = u64_t(directory_words_[word].bits_high) << 32 |
                                   directory_words_[word].bits_low;
                rank += static_cast<u32_t>(sz_u64_popcount(bits));
            }
        }
        else {
            if (directory_.try_resize(prefix_buckets + 1) != status_t::success_k) return status_t::bad_alloc_k;
            cursor = 0;
            for (size_t prefix = 0; prefix != prefix_buckets; ++prefix) {
                directory_[prefix] = static_cast<u32_t>(cursor);
                while (cursor != wide_records_.size() &&
                       (u32_t(wide_records_[cursor] >> 32) >> suffix_bits_) == prefix)
                    ++cursor;
            }
            directory_[prefix_buckets] = static_cast<u32_t>(wide_records_.size());
        }
        return status_t::success_k;
    }

    template <typename sequences_type_>
    status_t build_(sequences_type_ const &dictionary) noexcept {
        if (dictionary.size() > std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
        size_t tape_symbols = 0;
        max_word_length_ = 0;
        for (size_t id = 0; id != dictionary.size(); ++id) {
            size_t const length = dictionary[id].size();
            if (length > std::numeric_limits<size_t>::max() - tape_symbols) return status_t::overflow_risk_k;
            tape_symbols += length;
            max_word_length_ = sz_max_of_two(max_word_length_, length);
        }
        if (tape_.try_resize(tape_symbols) != status_t::success_k ||
            offsets_.try_resize(dictionary.size() + 1) != status_t::success_k ||
            wide_records_.try_reserve(dictionary.size()) != status_t::success_k)
            return status_t::bad_alloc_k;

        size_t tape_offset = 0;
        for (u32_t id = 0; id != dictionary.size(); ++id) {
            auto const item = dictionary[id];
            span<symbol_t const> const word {item.data(), item.size()};
            offsets_[id] = static_cast<u64_t>(tape_offset);
            if (!word.empty())
                std::memcpy(tape_.data() + tape_offset, word.data(), word.size() * sizeof(symbol_t));
            tape_offset += word.size();
            if (status_t status = wide_records_.try_push_back((u64_t(hash_word_(word)) << 32) | id);
                status != status_t::success_k)
                return status;
        }
        offsets_[dictionary.size()] = static_cast<u64_t>(tape_offset);
        std::sort(wide_records_.begin(), wide_records_.end());
        if (wide_records_.size()) {
            if (status_t status = build_directory_(); status != status_t::success_k) return status;
            if (dictionary.size() <= size_t(1) << packed_id_bits_k) {
                if (packed_records_.try_resize(wide_records_.size()) != status_t::success_k)
                    return status_t::bad_alloc_k;
                for (size_t index = 0; index != wide_records_.size(); ++index) {
                    u32_t const hash = static_cast<u32_t>(wide_records_[index] >> 32);
                    u32_t const id = static_cast<u32_t>(wide_records_[index]);
                    packed_records_[index] = ((hash & suffix_mask_) << packed_id_bits_k) | id;
                }
                wide_records_.reset();
            }
        }
        return status_t::success_k;
    }

    bool bucket_(u32_t hash, size_t &begin_offset, size_t &end_offset) const noexcept {
        size_t const prefix = hash >> suffix_bits_;
        if (directory_words_.size()) {
            size_t const word_index = prefix / 64;
            size_t const bit_index = prefix % 64;
            directory_word_t const &directory_word = directory_words_[word_index];
            u64_t const word = u64_t(directory_word.bits_high) << 32 | directory_word.bits_low;
            u64_t const bit = u64_t(1) << bit_index;
            if (!(word & bit)) return false;
            size_t const rank = directory_word.rank + sz_u64_popcount(word & (bit - 1));
            begin_offset = directory_[rank];
            end_offset = directory_[rank + 1];
        }
        else {
            begin_offset = directory_[prefix];
            end_offset = directory_[prefix + 1];
        }
        return true;
    }

  public:
    /** @brief Reserved reusable state for one reader. */
    class scratch_t {
      public:
        explicit scratch_t(allocator_t = {}) noexcept {}
    };

    using matches_t = vector_t<match_t>;

    explicit basic_levenshtein_index(allocator_t alloc = {}) noexcept
        : alloc_(alloc), tape_(alloc), offsets_(alloc), directory_(alloc), directory_words_(alloc),
          packed_records_(alloc), wide_records_(alloc) {}

    /** @brief Copies a dictionary for exact lookup. A failed build leaves the previous index unchanged. */
    template <typename sequences_type_>
    status_t try_build(sequences_type_ const &dictionary, u8_t max_distance = 0) noexcept {
        if (max_distance != 0) return status_t::unexpected_dimensions_k;
        basic_levenshtein_index candidate {alloc_};
        if (status_t status = candidate.build_(dictionary); status != status_t::success_k) return status;
        *this = std::move(candidate);
        return status_t::success_k;
    }

    /** @brief Finds every dictionary entry exactly equal to @p query. */
    status_t find(span<symbol_t const> query, u8_t bound, scratch_t &, matches_t &matches) const noexcept {
        if (status_t status = matches.try_resize(0); status != status_t::success_k) return status;
        if (bound != 0) return status_t::unexpected_dimensions_k;
        if (!size()) return status_t::success_k;
        u32_t const hash = hash_word_(query);
        size_t begin_offset = 0, end_offset = 0;
        if (!bucket_(hash, begin_offset, end_offset)) return status_t::success_k;
        if (!packed_records_.size()) {
            u64_t const key = u64_t(hash) << 32;
            u64_t const *record = std::lower_bound(wide_records_.begin() + begin_offset,
                                                   wide_records_.begin() + end_offset, key);
            u64_t const *const end = wide_records_.begin() + end_offset;
            for (; record != end && static_cast<u32_t>(*record >> 32) == hash; ++record) {
                u32_t const id = static_cast<u32_t>(*record);
                if (equals_(id, query))
                    if (status_t status = matches.try_push_back(match_t {id, 0, {}});
                        status != status_t::success_k)
                        return status;
            }
        }
        else {
            u32_t const suffix = hash & suffix_mask_;
            u32_t const key = suffix << packed_id_bits_k;
            u32_t const *record = std::lower_bound(packed_records_.begin() + begin_offset,
                                                   packed_records_.begin() + end_offset, key);
            u32_t const *const end = packed_records_.begin() + end_offset;
            for (; record != end && (*record >> packed_id_bits_k) == suffix; ++record) {
                u32_t const id = *record & packed_id_mask_k;
                if (equals_(id, query))
                    if (status_t status = matches.try_push_back(match_t {id, 0, {}});
                        status != status_t::success_k)
                        return status;
            }
        }
        return status_t::success_k;
    }

    size_t size() const noexcept { return offsets_.size() ? offsets_.size() - 1 : 0; }
    u8_t max_distance() const noexcept { return 0; }
    size_t max_word_length() const noexcept { return max_word_length_; }
    bool uses_packed_records() const noexcept { return packed_records_.size() != 0 || size() == 0; }
    size_t records_count() const noexcept {
        return packed_records_.size() ? packed_records_.size() : wide_records_.size();
    }
    size_t index_bytes() const noexcept {
        return directory_.size() * sizeof(u32_t) + directory_words_.size() * sizeof(directory_word_t) +
               packed_records_.size() * sizeof(u32_t) + wide_records_.size() * sizeof(u64_t);
    }
    size_t dictionary_bytes() const noexcept {
        return tape_.size() * sizeof(symbol_t) + offsets_.size() * sizeof(u64_t);
    }
};

template <typename allocator_type_ = std::allocator<char>>
using levenshtein_index = basic_levenshtein_index<char, allocator_type_>;

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_LEVENSHTEIN_INDEX_HPP_
