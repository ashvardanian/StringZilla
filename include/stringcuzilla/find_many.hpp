/**
 *  @brief  Hardware-accelerated multi-pattern exact substring search.
 *  @file   find_many.hpp
 *  @author Ash Vardanian
 *
 *  One of the most broadly used algorithms in string processing is the multi-pattern Aho-Corasick
 *  algorithm, that constructs a trie from the patterns, transforms it into a finite state machine,
 *  and then uses it to search for all patterns in the text in a single pass.
 *
 *  One of its biggest issues is the memory consumption, as one would often build a dense state transition
 *  table/matrix:
 *
 *  - with number of columns proportional to the size of the alphabet,
 *  - and number of rows proportional to the number of states, in worst case, the aggregate length
 *    of all needles, if none share prefixes.
 *
 *  Such dense representations simplify transition lookup down to a single memory access, but that access
 *  can be expensive if the memory doesn't fir into the CPU caches for really large vocabulary sizes.
 *
 *  Addressing this, we provide a sparse layout variants of the FSM, that uses predicated SIMD instructions
 *  to rapidly probe the transitions and find the next state. This allows us to use a much smaller state,
 *  fitting in L1/L2 caches much more frequently.
 *
 *  @section Use Cases
 *
 *  Before optimizing its relevant to understand the typical usecases for the algorithm. Typically,
 *  we would use `uint32_t` for the state indicies, and 256 state transitions for byte-level FSM.
 *
 *  | Use Case                      | Number of States        | Memory Usage            |
 *  |-------------------------------|-------------------------|-------------------------|
 *  | Malware/Intrusion Detection   | 10,000 – 1,000,000      | 10.24 MB – 1.024 GB     |
 *  | DNA/RNA Motif Scanning        | 100 – 100,000           | 0.1 MB – 102.4 MB       |
 *  | Keyword Filtering/Moderation  | 100 – 10,000            | 0.1 MB – 10.24 MB       |
 *  | Plagiarism/Code Similarity    | 1,000 – 100,000         | 1.024 MB – 102.4 MB     |
 *  | Product Catalog Matching      | 100,000 – 1,000,000     | 102.4 MB – 1.024 GB     |
 */
#ifndef STRINGZILLA_FIND_MANY_HPP_
#define STRINGZILLA_FIND_MANY_HPP_

#include "stringzilla/memory.h"  // `sz_move`
#include "stringzilla/types.hpp" // `status_t::status_t`

#include <memory>      // `std::allocator_traits` to re-bind the allocator
#include <type_traits> // `std::enable_if_t` for meta-programming
#include <limits>      // `std::numeric_limits` for numeric types
#include <iterator>    // `std::iterator_traits` for iterators

namespace ashvardanian {
namespace stringzilla {

#pragma region - Dictionary

struct find_many_match_t {

    span<char const> haystack {};
    /**
     *  @brief  The substring of the @p haystack that matched the needle.
     *          Can be used to infer the offset of the needle in the haystack.
     */
    span<char const> needle {};
    size_t haystack_index {};
    size_t needle_index {};
};

/**
 *  @brief Aho-Corasick dictionary for multi-pattern exact byte-level substring search.
 *  @note As FSM construction is almost never a bottleneck, we don't optimize it for speed.
 *  @tparam state_id_type_ The type of the state ID. Default is `sz_u32_t`.
 *  @tparam allocator_type_ The type of the allocator. Default is `dummy_alloc_t`.
 *
 *  Similar to the rest of the library, doesn't use `std::vector` or other STL containers
 *  and avoid `std::bad_alloc` and other exceptions in favor of `status_t::status_t` error codes
 *  and `try_`-prefixed functions.
 */
template <typename state_id_type_ = sz_u32_t, typename allocator_type_ = dummy_alloc_t>
struct aho_corasick_dictionary {

    using state_id_t = state_id_type_;
    using allocator_t = allocator_type_;
    using match_t = find_many_match_t;

    static constexpr state_id_t alphabet_size_k = 256;
    static constexpr state_id_t invalid_state_k = std::numeric_limits<state_id_t>::max();
    static constexpr size_t invalid_length_k = std::numeric_limits<size_t>::max();

    using state_transitions_t = safe_array<state_id_t, alphabet_size_k>;
    static_assert(std::is_unsigned_v<state_id_t>, "State ID should be unsigned");

    using size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;
    using state_id_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<state_id_t>;
    using state_transitions_allocator_t =
        typename std::allocator_traits<allocator_t>::template rebind_alloc<state_transitions_t>;

    /**
     *  @brief  State transitions for each state, at least `count_states_ * alphabet_size_k` in binary size.
     *          The transitions are being populated both during vocabulary construction and during search.
     */
    safe_vector<state_transitions_t, state_transitions_allocator_t> transitions_;

    /**
     *  @brief  The IDs of the output transitions per state.
     *
     *  During `try_insert`, contains exactly one entry per state, generally set to `invalid_state_k`.
     *  After `try_build`, contains at least as many entries as the number of unique needles provided,
     *  or potentially more, given how failure links are being merged, if needles have shared suffixes.
     */
    safe_vector<state_id_t, state_id_allocator_t> outputs_;

    /**
     *  @brief  Failure links for each state, exactly `count_states_` in effective size, potentially larger capacity.
     *          The failure links aren't very needed after the FSM construction, if we stick to a dense layout.
     */
    safe_vector<state_id_t, state_id_allocator_t> failures_;

    /**
     *  @brief  Number of states in the FSM, which should be smaller than the capacity of the transitions.
     *          The value grows on each successful `try_insert` call, and doesn't change even in `try_build`.
     */
    size_t count_states_ = 0;

    /**
     *  @brief  Contains number of needles ending at each state, exactly `count_states_` in size.
     *          We can use any `size_t`-like counter, but the `state_id_t` is probably the smallest safe type here.
     *
     *  This object is used to navigate into the `outputs_` array after the FSM construction. For any state `I`, the
     *  following slice defines all matches: `outputs_[outputs_offsets_[I], outputs_offsets_[I] + outputs_counts_[I]]`.
     */
    safe_vector<state_id_t, state_id_allocator_t> outputs_counts_;

    /**
     *  @brief  Contains number of merged needles & failure outputs ending before each state, `count_states_` in size.
     *          We can use any `size_t`-like counter, but the `state_id_t` is probably the smallest safe type here.
     *
     *  This object is used to navigate into the `outputs_` array after the FSM construction. It contains effectively
     *  the exclusive prefix sum of `outputs_counts_`. For any state `I`, the following slice defines all matches:
     *  `outputs_[outputs_offsets_[I], outputs_offsets_[I] + outputs_counts_[I]]`.
     */
    safe_vector<state_id_t, state_id_allocator_t> outputs_offsets_;

    /**
     *  @brief  Contains the lengths of needles.
     *          The array grows on each successful `try_insert` call, and doesn't change even in `try_build`.
     */
    safe_vector<size_t, size_allocator_t> needles_lengths_;

    allocator_t alloc_;

    aho_corasick_dictionary() = default;
    ~aho_corasick_dictionary() noexcept { reset(); }

    aho_corasick_dictionary(allocator_t alloc) noexcept
        : transitions_(alloc), failures_(alloc), outputs_(alloc), outputs_counts_(alloc), outputs_offsets_(alloc),
          needles_lengths_(alloc), alloc_(alloc) {}

    aho_corasick_dictionary(aho_corasick_dictionary const &) = delete;
    aho_corasick_dictionary(aho_corasick_dictionary &&) = delete;
    aho_corasick_dictionary &operator=(aho_corasick_dictionary const &) = delete;
    aho_corasick_dictionary &operator=(aho_corasick_dictionary &&) = delete;

    void clear() noexcept {
        transitions_.clear();
        failures_.clear();
        outputs_.clear();
        needles_lengths_.clear();
        outputs_counts_.clear();
        outputs_offsets_.clear();
        count_states_ = 0;
    }

    void reset() noexcept {
        transitions_.reset();
        failures_.reset();
        outputs_.reset();
        needles_lengths_.reset();
        outputs_counts_.reset();
        outputs_offsets_.reset();
        count_states_ = 0;
    }

    size_t size() const noexcept { return count_states_; }
    size_t capacity() const noexcept { return transitions_.size(); }

    status_t try_reserve(size_t new_capacity) noexcept {

        // Allocate new memory blocks.
        if (transitions_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        if (failures_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_counts_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;
        if (outputs_offsets_.try_resize(new_capacity) != status_t::success_k) return status_t::bad_alloc_k;

        // Initialize new states.
        size_t old_count = count_states_;
        for (size_t state = old_count; state < new_capacity; ++state) {
            for (size_t index = 0; index < alphabet_size_k; ++index) transitions_[state][index] = invalid_state_k;
            outputs_[state] = invalid_state_k;
            failures_[state] = 0;       // Default failure to root
            outputs_counts_[state] = 0; // Default count to zero
            outputs_offsets_[state] = invalid_state_k;
        }

        // The effective size doesn't change, but we now have a root!
        count_states_ = std::max<size_t>(old_count, 1);
        return status_t::success_k;
    }

    /**
     *  @brief Adds a single @p needle to the vocabulary, assigning it a unique @p needle_id.
     *  @note  Can't be called after `try_build`. Can't be called from multiple threads at the same time.
     */
    status_t try_insert(span<char const> needle) noexcept {
        if (!needle.size()) return status_t::success_k; // Don't care about empty needles.

        state_id_t const needle_id = static_cast<state_id_t>(needles_lengths_.size());
        if (needles_lengths_.try_reserve(sz_size_bit_ceil(needles_lengths_.size() + 1)) != status_t::success_k)
            return status_t::bad_alloc_k;

        state_id_t current_state = 0;
        for (size_t pos = 0; pos < needle.size(); ++pos) {
            unsigned char const symbol = static_cast<unsigned char>(needle[pos]);
            state_id_t *current_row = &transitions_[current_state][0];
            bool const has_root_state = transitions_.data() != nullptr;
            if (!has_root_state || current_row[symbol] == invalid_state_k) {
                if (count_states_ >= transitions_.size()) {
                    status_t reserve_status = try_reserve(sz_size_bit_ceil(transitions_.size() + 1 + !has_root_state));
                    if (reserve_status != status_t::success_k) return reserve_status;
                    current_row = &transitions_[current_state][0]; // Update the pointer to the row of state transitions
                }

                // Use the next available state ID
                state_id_t new_state = static_cast<state_id_t>(count_states_);
                current_row[symbol] = new_state;
                ++count_states_;
            }
            current_state = current_row[symbol];
        }

        outputs_[current_state] = needle_id;
        needles_lengths_.try_push_back(needle.size()); // ? Can't fail due to `try_reserve` above
        outputs_counts_[current_state] = 1; // ? This will snowball in `try_build` if needles have shared suffixes
        outputs_offsets_[current_state] = current_state;
        return status_t::success_k;
    }

    /**
     *  @brief  Construct the Finite State Machine (FSM) from the vocabulary. Can only be called @b once!
     *  @note   This function is not thread safe and allocates a significant amount of memory, so it can fail.
     */
    status_t try_build() noexcept {

        // Allocate a queue for Breadth-First Search (BFS) traversal.
        safe_vector<state_id_t, state_id_allocator_t> work_queue(alloc_);
        if (work_queue.try_resize(count_states_) != status_t::success_k) return status_t::bad_alloc_k;

        // We will construct nested dynamically growing arrays (yes, too many memory allocations, I know),
        // to expand and track all of the outputs for each state, merging the failure links.
        // We will later use `outputs_merged` to populate `outputs_`, `outputs_offsets_`, and `outputs_counts_`.
        using state_ids_vector_t = safe_vector<state_id_t, state_id_allocator_t>;
        using state_ids_vector_allocator_t =
            typename std::allocator_traits<allocator_t>::template rebind_alloc<state_ids_vector_t>;
        using state_ids_per_state_vector_t = safe_vector<state_ids_vector_t, state_ids_vector_allocator_t>;
        state_ids_per_state_vector_t outputs_merged(alloc_);
        if (outputs_merged.try_resize(count_states_) != status_t::success_k) return status_t::bad_alloc_k;

        // Populate the `outputs_merged` with the initial outputs.
        for (size_t state = 0; state < count_states_; ++state) {
            state_ids_vector_t &outputs = outputs_merged[state];
            if (outputs_[state] != invalid_state_k && outputs.try_push_back(outputs_[state]) != status_t::success_k)
                return status_t::bad_alloc_k;
        }

        // Reset all root transitions to point to itself - forming a loop.
        size_t queue_begin = 0, queue_end = 0;
        for (size_t symbol = 0; symbol < alphabet_size_k; ++symbol) {
            if (transitions_[0][symbol] == invalid_state_k) { transitions_[0][symbol] = 0; }
            else { failures_[transitions_[0][symbol]] = 0, work_queue[queue_end++] = transitions_[0][symbol]; }
        }

        while (queue_begin < queue_end) {
            state_id_t current_state = work_queue[queue_begin++];
            for (size_t symbol = 0; symbol < alphabet_size_k; ++symbol) {

                state_id_t next_state = transitions_[current_state][symbol];
                if (next_state != invalid_state_k) {

                    state_id_t failure_state = failures_[current_state];
                    while (transitions_[failure_state][symbol] == invalid_state_k)
                        failure_state = failures_[failure_state];
                    failures_[next_state] = transitions_[failure_state][symbol];

                    // Aggregate the outputs of the failure links
                    if (outputs_merged[next_state].try_append(outputs_merged[failures_[next_state]]) !=
                        status_t::success_k)
                        return status_t::bad_alloc_k;

                    if (outputs_[failures_[next_state]] != invalid_state_k && outputs_[next_state] == invalid_state_k)
                        outputs_[next_state] = outputs_[failures_[next_state]];
                    work_queue[queue_end++] = next_state;
                }
                else { transitions_[current_state][symbol] = transitions_[failures_[current_state]][symbol]; }
            }
        }

        // Re-populate the `outputs_` with a flattened version of `outputs_merged`.
        // Also populate the `outputs_counts_` with the number of needles ending at each state.
        size_t total_count = 0;
        for (size_t state = 0; state < count_states_; ++state) {
            state_ids_vector_t &outputs = outputs_merged[state];
            outputs_counts_[state] = static_cast<state_id_t>(outputs.size());
            outputs_offsets_[state] = static_cast<state_id_t>(total_count);
            total_count += outputs.size();
        }

        // Now in the second pass, perform the flattening of the `outputs_merged` into `outputs_`.
        if (outputs_.try_resize(total_count) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t state = 0; state < count_states_; ++state) {
            state_ids_vector_t &outputs = outputs_merged[state];
            for (size_t i = 0; i < outputs.size(); ++i) outputs_[outputs_offsets_[state] + i] = outputs[i];
        }

        return status_t::success_k;
    }

    /**
     *  @brief Find all occurrences of the needles in the @p haystack.
     *  @note This is a serial reference implementation only recommended for testing.
     *  @param haystack The input string to search in.
     *  @param callback The handler for a @b `match_t` match, returning `true` to continue.
     */
    template <typename callback_type_>
    void find(span<char const> haystack, callback_type_ &&callback) const noexcept {
        state_id_t current_state = 0;
        for (size_t pos = 0; pos < haystack.size(); ++pos) {
            unsigned char symbol = static_cast<unsigned char>(haystack[pos]);
            current_state = transitions_[current_state][symbol];

            size_t outputs_count = outputs_counts_[current_state];
            if (outputs_count == 0) continue;
            size_t outputs_offset = outputs_offsets_[current_state];
            for (size_t i = 0; i < outputs_count; ++i) { // In small vocabulary, this is generally just 1 iteration
                size_t needle_id = outputs_[outputs_offset + i];
                size_t match_length = needles_lengths_[needle_id];
                span<char const> match_span(&haystack[pos + 1 - match_length], match_length);
                match_t match {haystack, match_span, 0, needle_id};
                if (!callback(match)) break;
            }
        }
    }

    /**
     *  @brief Count the number of occurrences of all the needles in the @p haystack.
     *  @return The number of potentially-overlapping occurrences.
     */
    size_t count(span<char const> haystack) const noexcept {
        size_t count = 0;
        state_id_t current_state = 0;
        for (size_t pos = 0; pos < haystack.size(); ++pos) {
            unsigned char symbol = static_cast<unsigned char>(haystack[pos]);
            current_state = transitions_[current_state][symbol];
            count += outputs_counts_[current_state];
        }
        return count;
    }
};

#pragma endregion // Dictionary

#pragma region - Primary API

/**
 *  @brief Aho-Corasick-based @b single-threaded multi-pattern exact substring search.
 *  @tparam state_id_type_ The type of the state ID. Default is `sz_u32_t`.
 *  @tparam allocator_type_ The type of the allocator. Default is `dummy_alloc_t`.
 *  @tparam capability_ The capability of the dictionary. Default is `sz_cap_serial_k`.
 */
template <                                         //
    typename state_id_type_ = sz_u32_t,            //
    typename allocator_type_ = dummy_alloc_t,      //
    sz_capability_t capability_ = sz_cap_serial_k, //
    typename enable_ = void                        //
    >
struct find_many {
    using dictionary_t = aho_corasick_dictionary<state_id_type_, allocator_type_>;
    using state_id_t = typename dictionary_t::state_id_t;
    using allocator_t = typename dictionary_t::allocator_t;
    using match_t = typename dictionary_t::match_t;

    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles_strings) noexcept {
        for (auto const &needle : needles_strings) {
            status_t status = dict_.try_insert(needle);
            if (status != status_t::success_k) return status;
        }
        return dict_.try_build();
    }

    void reset() noexcept { dict_.reset(); }

    /**
     *  @brief Counts the number of occurrences of the needles in the haystack. Relevant for filtering and ranking.
     *  @param[in] haystacks The input strings to search in.
     *  @param[out] counts The output buffer for the counts of each needle.
     *  @return The total number of occurrences found.
     */
    template <typename haystacks_type_>
    size_t count(haystacks_type_ &&haystacks, span<size_t> counts) const noexcept {
        size_t count_total = 0;
        for (size_t i = 0; i < counts.size(); ++i) count_total += counts[i] = dict_.count(haystacks[i]);
        return count_total;
    }

    /**
     *  @brief Finds all occurrences of the needles in all the @p haystacks.
     *  @param haystacks The input strings to search in, with support for random access iterators.
     *  @param matches The output buffer for the matches, with support for random access iterators.
     *  @return The number of matches found across all the @p haystacks.
     *  @note The @p matches reference objects should be assignable from @b `match_t`.
     */
    template <typename haystacks_type_, typename output_matches_type_>
    size_t find(haystacks_type_ &&haystacks, output_matches_type_ &&matches) const noexcept {
        size_t count_found = 0, count_allowed = matches.size();
        for (auto it = haystacks.begin(); it != haystacks.end() && count_found != count_allowed; ++it)
            dict_.find(*it, [&](match_t match) {
                match.haystack_index = static_cast<size_t>(it - haystacks.begin());
                matches[count_found] = match;
                count_found++;
                return count_found < count_allowed;
            });
        return count_found;
    }

  private:
    dictionary_t dict_;
};

#pragma endregion // Primary API

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_FIND_MANY_HPP_
