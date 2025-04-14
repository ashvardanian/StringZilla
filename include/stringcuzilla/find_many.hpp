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

    using state_transitions_t = state_id_t[alphabet_size_k];

    /**
     *  @brief  State transitions for each state, at least `count_states_ * alphabet_size_k` in binary size.
     *  @note   The transitions are being populated both during vocabulary construction and during search.
     */
    span<state_transitions_t> transitions_;
    /**
     *  @brief  Output needle IDs for each state, at least `count_states_` in size.
     *  @note   They are being populated both during vocabulary construction and during search.
     */
    span<state_id_t> outputs_;
    /**
     *  @brief  Failure links for each state, at least `count_states_` in size.
     *  @note   The failure links aren't very needed after the FSM construction, if we stick to a dense layout.
     */
    span<state_id_t> failures_;
    /**
     *  @brief  Number of states in the FSM, which should be smaller than the capacity of the transitions.
     *          The value grows on each successful `try_insert` call, and doesn't change even in `try_build`.
     */
    size_t count_states_ = 0;
    /**
     *  @brief  Contains the lengths of the needles ending at each state, at least `count_states_` in size.
     *          The values grow on each successful `try_insert` call, and doesn't change even in `try_build`.
     */
    span<size_t> outputs_lengths_;
    /**
     *  @brief  Contains number of needles ending at each state, at least `count_states_` in size.
     *          The values grow on each successful `try_insert` call, and doesn't change even in `try_build`.
     */
    span<size_t> outputs_counts_;

    allocator_t alloc_;

    aho_corasick_dictionary() = default;
    ~aho_corasick_dictionary() noexcept { reset(); }

    aho_corasick_dictionary(aho_corasick_dictionary const &) = delete;
    aho_corasick_dictionary(aho_corasick_dictionary &&) = delete;
    aho_corasick_dictionary &operator=(aho_corasick_dictionary const &) = delete;
    aho_corasick_dictionary &operator=(aho_corasick_dictionary &&) = delete;

    void reset() noexcept {
        if (transitions_.data())
            alloc_.deallocate((char *)(transitions_.data()), transitions_.size() * sizeof(state_transitions_t));
        if (failures_.data()) alloc_.deallocate((char *)(failures_.data()), failures_.size() * sizeof(state_id_t));
        if (outputs_.data()) alloc_.deallocate((char *)(outputs_.data()), outputs_.size() * sizeof(state_id_t));
        if (outputs_lengths_.data())
            alloc_.deallocate((char *)(outputs_lengths_.data()), outputs_lengths_.size() * sizeof(size_t));
        transitions_ = {};
        failures_ = {};
        outputs_ = {};
        outputs_lengths_ = {};
        count_states_ = 0;
    }

    size_t size() const noexcept { return count_states_; }
    size_t capacity() const noexcept { return transitions_.size(); }

    status_t try_reserve(size_t new_capacity) noexcept {

        // Allocate new memory blocks.
        state_transitions_t *new_transitions =
            (state_transitions_t *)(alloc_.allocate(new_capacity * sizeof(state_transitions_t)));
        state_id_t *new_failures = (state_id_t *)(alloc_.allocate(new_capacity * sizeof(state_id_t)));
        state_id_t *new_outputs = (state_id_t *)(alloc_.allocate(new_capacity * sizeof(state_id_t)));
        size_t *new_lengths = (size_t *)(alloc_.allocate(new_capacity * sizeof(size_t)));
        if (!new_transitions || !new_failures || !new_outputs || !new_lengths) {
            if (new_transitions)
                alloc_.deallocate((char *)(new_transitions), new_capacity * sizeof(state_transitions_t));
            if (new_failures) alloc_.deallocate((char *)(new_failures), new_capacity * sizeof(state_id_t));
            if (new_outputs) alloc_.deallocate((char *)(new_outputs), new_capacity * sizeof(state_id_t));
            if (new_lengths) alloc_.deallocate((char *)(new_lengths), new_capacity * sizeof(size_t));
            return status_t::bad_alloc_k;
        }

        size_t old_count = count_states_;

        // Copy existing states data. Use memcpy for POD types if spans are contiguous.
        if (old_count > 0) {
            sz_copy((sz_ptr_t)new_transitions, (sz_cptr_t)transitions_.data(), old_count * sizeof(state_transitions_t));
            sz_copy((sz_ptr_t)new_outputs, (sz_cptr_t)outputs_.data(), old_count * sizeof(state_id_t));
            sz_copy((sz_ptr_t)new_lengths, (sz_cptr_t)outputs_lengths_.data(), old_count * sizeof(size_t));
            sz_copy((sz_ptr_t)new_failures, (sz_cptr_t)failures_.data(), old_count * sizeof(state_id_t));
        }

        // Initialize new states.
        for (size_t state = old_count; state < new_capacity; ++state) {
            for (size_t index = 0; index < alphabet_size_k; ++index) new_transitions[state][index] = invalid_state_k;
            new_outputs[state] = invalid_state_k;
            new_lengths[state] = invalid_length_k;
            new_failures[state] = 0; // Default failure to root
        }

        // Free old memory and update pointers.
        reset();
        transitions_ = {new_transitions, new_capacity};
        failures_ = {new_failures, new_capacity};
        outputs_ = {new_outputs, new_capacity};
        outputs_lengths_ = {new_lengths, new_capacity};
        count_states_ = std::max<size_t>(old_count, 1); // The effective size doesn't change, but we now have a root!
        return status_t::success_k;
    }

    /**
     *  @brief Adds a single @p needle to the vocabulary, assigning it a unique @p needle_id.
     */
    status_t try_insert(span<char const> needle, state_id_t needle_id) noexcept {
        if (!needle.size()) return status_t::success_k; // Don't care about empty needles.

        state_id_t current_state = 0;
        for (size_t pos = 0; pos < needle.size(); ++pos) {
            unsigned char const symbol = static_cast<unsigned char>(needle[pos]);
            state_id_t *current_row = transitions_[current_state];
            bool const has_root_state = transitions_.data() != nullptr;
            if (!has_root_state || current_row[symbol] == invalid_state_k) {
                if (count_states_ >= transitions_.size()) {
                    status_t reserve_status = try_reserve(sz_size_bit_ceil(transitions_.size() + 1 + !has_root_state));
                    if (reserve_status != status_t::success_k) return reserve_status;
                    current_row = transitions_[current_state]; // Update the pointer!
                }

                // Use the next available state ID
                state_id_t new_state = static_cast<state_id_t>(count_states_);
                current_row[symbol] = new_state;
                ++count_states_;
            }
            current_state = current_row[symbol];
        }

        outputs_[current_state] = needle_id;
        outputs_lengths_[current_state] = static_cast<size_t>(needle.size());
        return status_t::success_k;
    }

    status_t try_build() noexcept {

        // Allocate a queue for Breadth-First Search (BFS) traversal.
        size_t queue_capacity = count_states_;
        state_id_t *work_queue = (state_id_t *)(alloc_.allocate(queue_capacity * sizeof(state_id_t)));
        if (!work_queue) return status_t::bad_alloc_k;

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
                    if (outputs_[failures_[next_state]] != invalid_state_k && outputs_[next_state] == invalid_state_k)
                        outputs_[next_state] = outputs_[failures_[next_state]];
                    work_queue[queue_end++] = next_state;
                }
                else { transitions_[current_state][symbol] = transitions_[failures_[current_state]][symbol]; }
            }
        }
        alloc_.deallocate((char *)work_queue, queue_capacity * sizeof(state_id_t));
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
            if (outputs_[current_state] != invalid_state_k) {
                span<char const> match_span(&haystack[pos], haystack.size() - pos);
                match_t match {haystack, match_span, outputs_[current_state]};
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
            count += outputs_[current_state] != invalid_state_k;
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
        state_id_t needle_id = 0;
        for (auto const &needle : needles_strings) {
            status_t status = dict_.try_insert(needle, needle_id);
            if (status != status_t::success_k) return status;
            needle_id++;
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
            dict_.find(*it, [&](match_t const &match) {
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
