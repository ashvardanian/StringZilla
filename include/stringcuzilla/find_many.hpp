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
     */
    size_t count_ = 0;
    allocator_t alloc_;

    aho_corasick_dictionary() = default;
    ~aho_corasick_dictionary() noexcept { reset(); }

    void reset() noexcept {
        if (transitions_.data())
            alloc_.deallocate(reinterpret_cast<char *>(transitions_.data()),
                              transitions_.size() * sizeof(state_transitions_t));
        if (failures_.data())
            alloc_.deallocate(reinterpret_cast<char *>(failures_.data()), failures_.size() * sizeof(state_id_t));
        if (outputs_.data())
            alloc_.deallocate(reinterpret_cast<char *>(outputs_.data()), outputs_.size() * sizeof(state_id_t));
        transitions_ = {};
        failures_ = {};
        outputs_ = {};
        count_ = 0;
    }

    size_t size() const noexcept { return count_; }
    size_t capacity() const noexcept { return transitions_.size(); }

    status_t try_reserve(size_t new_capacity) noexcept {

        // Allocate new memory blocks.
        state_transitions_t *new_transitions =
            reinterpret_cast<state_transitions_t *>(alloc_.allocate(new_capacity * sizeof(state_transitions_t)));
        state_id_t *new_failures = reinterpret_cast<state_id_t *>(alloc_.allocate(new_capacity * sizeof(state_id_t)));
        state_id_t *new_outputs = reinterpret_cast<state_id_t *>(alloc_.allocate(new_capacity * sizeof(state_id_t)));
        if (!new_transitions || !new_failures || !new_outputs) {
            if (new_transitions)
                alloc_.deallocate(reinterpret_cast<char *>(new_transitions),
                                  new_capacity * sizeof(state_transitions_t));
            if (new_failures)
                alloc_.deallocate(reinterpret_cast<char *>(new_failures), new_capacity * sizeof(state_id_t));
            if (new_outputs)
                alloc_.deallocate(reinterpret_cast<char *>(new_outputs), new_capacity * sizeof(state_id_t));
            return status_t::bad_alloc_k;
        }

        // Copy existing states.
        for (size_t state = 0; state < count_; ++state) {
            for (size_t index = 0; index < alphabet_size_k; ++index)
                new_transitions[state][index] = transitions_[state][index];
            new_failures[state] = failures_[state];
            new_outputs[state] = outputs_[state];
        }

        // Initialize new states.
        for (size_t state = count_; state < new_capacity; ++state) {
            for (size_t index = 0; index < alphabet_size_k; ++index) new_transitions[state][index] = invalid_state_k;
            new_failures[state] = 0;
            new_outputs[state] = invalid_state_k;
        }

        // Free old memory and update pointers.
        size_t old_count = count_;
        reset();
        count_ = std::max<size_t>(old_count, 1); // The effective size doesn't change, but we now have a root!
        transitions_ = {new_transitions, new_capacity};
        failures_ = {new_failures, new_capacity};
        outputs_ = {new_outputs, new_capacity};
        return status_t::success_k;
    }

    status_t try_insert(span<char const> needle, state_id_t needle_id) noexcept {
        state_id_t current_state = 0;
        for (size_t pos = 0; pos < needle.size(); ++pos) {
            unsigned char const symbol = static_cast<unsigned char>(needle[pos]);
            state_id_t *current_row = transitions_[current_state];
            bool const has_root_state = transitions_.data() != nullptr;
            if (!has_root_state || current_row[symbol] == invalid_state_k) {
                if (count_ >= transitions_.size()) {
                    status_t reserve_status = try_reserve(sz_size_bit_ceil(transitions_.size() + 1 + !has_root_state));
                    if (reserve_status != status_t::success_k) return reserve_status;
                    current_row = transitions_[current_state]; // Update the pointer!
                }
                current_row[symbol] = static_cast<state_id_t>(count_);
                state_id_t new_state = static_cast<state_id_t>(count_);
                for (size_t index = 0; index < alphabet_size_k; ++index)
                    transitions_[new_state][index] = invalid_state_k;
                failures_[new_state] = 0;
                outputs_[new_state] = invalid_state_k;
                ++count_;
            }
            current_state = current_row[symbol];
        }
        outputs_[current_state] = needle_id;
        return status_t::success_k;
    }

    status_t try_build() noexcept {

        // Allocate a queue for Breadth-First Search (BFS) traversal.
        size_t queue_capacity = count_;
        size_t *work_queue = reinterpret_cast<size_t *>(alloc_.allocate(queue_capacity * sizeof(size_t)));
        if (!work_queue) return status_t::bad_alloc_k;

        // Reset all root transitions to point to itself - forming a loop.
        size_t queue_begin = 0, queue_end = 0;
        for (size_t symbol = 0; symbol < alphabet_size_k; ++symbol) {
            if (transitions_[0][symbol] == invalid_state_k) { transitions_[0][symbol] = 0; }
            else { failures_[transitions_[0][symbol]] = 0, work_queue[queue_end++] = transitions_[0][symbol]; }
        }

        while (queue_begin < queue_end) {
            size_t current_state = work_queue[queue_begin++];
            for (size_t symbol = 0; symbol < alphabet_size_k; ++symbol) {

                state_id_t next_state = transitions_[current_state][symbol];
                if (next_state != invalid_state_k) {

                    size_t failure_state = failures_[current_state];
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
        alloc_.deallocate(work_queue, queue_capacity * sizeof(size_t));
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
        for (auto const &needle : needles_strings) {
            status_t status = dict_.try_insert(needle, static_cast<state_id_t>(dict_.size()));
            if (status != status_t::success_k) return status;
        }
        return status_t::success_k;
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
