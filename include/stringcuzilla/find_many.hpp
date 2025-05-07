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
 *  | UTF-32 Mapping                | 10,000 – 100,000        | 10.24 MB – 102.4 MB     |
 *  | Malware/Intrusion Detection   | 10,000 – 1,000,000      | 10.24 MB – 1.024 GB     |
 *  | DNA/RNA Motif Scanning        | 100 – 100,000           | 0.1 MB – 102.4 MB       |
 *  | Keyword Filtering/Moderation  | 100 – 10,000            | 0.1 MB – 10.24 MB       |
 *  | Plagiarism/Code Similarity    | 1,000 – 100,000         | 1.024 MB – 102.4 MB     |
 *  | Product Catalog Matching      | 100,000 – 1,000,000     | 102.4 MB – 1.024 GB     |
 */
#ifndef STRINGCUZILLA_FIND_MANY_HPP_
#define STRINGCUZILLA_FIND_MANY_HPP_

#include "stringzilla/memory.h"    // `sz_move`
#include "stringzilla/types.hpp"   // `status_t::status_t`
#include "stringcuzilla/types.hpp" // `dummy_executor_t`

#include <memory>      // `std::allocator_traits` to re-bind the allocator
#include <type_traits> // `std::enable_if_t` for meta-programming
#include <limits>      // `std::numeric_limits` for numeric types
#include <iterator>    // `std::iterator_traits` for iterators

namespace ashvardanian {
namespace stringzilla {

#pragma region - Dictionary

/**
 *  @brief  Light-weight structure to hold the result of a match in many-to-many
 *          search with multiple haystacks and needles.
 */
struct find_many_match_t {

    span<char const> haystack {};
    /**
     *  @brief  The substring of the @p haystack that matched the needle.
     *          Can be used to infer the offset of the needle in the haystack.
     */
    span<char const> needle {};
    size_t haystack_index {};
    size_t needle_index {};

    /**
     *  @brief  Helper function discouraged outside of testing and debugging, used to sort match lists
     *          in many-to-many search, to compare the outputs of multiple algorithms.
     */
    inline static bool less_globally(find_many_match_t const &lhs, find_many_match_t const &rhs) noexcept {
        return lhs.needle.data() < rhs.needle.data() ||
               (lhs.needle.data() == rhs.needle.data() && lhs.needle.end() < rhs.needle.end());
    }
};

template <typename value_type_>
struct min_max_sum {
    value_type_ min = std::numeric_limits<value_type_>::max();
    value_type_ max = std::numeric_limits<value_type_>::min();
    value_type_ sum = 0;
    size_t count = 0;

    void add(value_type_ value) noexcept {
        if (value < min) min = value;
        if (value > max) max = value;
        sum += value;
        ++count;
    }

    template <typename other_value_type_ = value_type_>
    other_value_type_ mean() const noexcept {
        if (count == 0) return 0;
        return (other_value_type_)sum / count;
    }
};

/**
 *  @brief Metadata for the Aho-Corasick dictionary.
 */
struct aho_corasick_metadata_t {
    min_max_sum<size_t> transitions_per_state;
    min_max_sum<size_t> matches_per_terminal_state;
    min_max_sum<size_t> needle_lengths;
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

  private:
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

    /**
     *  @brief  The allocator state to be used both for the static FSM and for the dynamic data-structures
     *          on the Breadth-First Search (BFS) construction phase.
     */
    allocator_t alloc_;

  public:
    aho_corasick_dictionary() = default;
    ~aho_corasick_dictionary() noexcept { reset(); }

    aho_corasick_dictionary(allocator_t alloc) noexcept
        : transitions_(alloc), outputs_(alloc), failures_(alloc), count_states_(0), outputs_counts_(alloc),
          outputs_offsets_(alloc), needles_lengths_(alloc), alloc_(alloc) {}

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
    size_t max_needle_length() const noexcept {
        size_t max_length = 0;
        for (size_t length : needles_lengths_) max_length = std::max(max_length, length);
        return max_length;
    }

    allocator_t const &allocator() const noexcept { return alloc_; }

    /**
     *  @brief Returns the metadata for the Aho-Corasick dictionary.
     *  @note The metadata is not thread-safe and should be used only after `try_build`.
     */
    aho_corasick_metadata_t metadata() const noexcept {
        aho_corasick_metadata_t metadata;

        // Estimate the number of transitions per state.
        for (state_transitions_t const &row : transitions_) {
            size_t count_valid = 0;
            for (state_id_t const &state : row) count_valid += state != invalid_state_k;
            metadata.transitions_per_state.add(count_valid);
        }

        // Estimate the number of matches per terminal state and needle lengths.
        for (size_t count : outputs_counts_) metadata.matches_per_terminal_state.add(count);
        for (size_t length : needles_lengths_) metadata.needle_lengths.add(length);
        return metadata;
    }

    /**
     *  @brief Reserves space for the FSM, allocating memory for the state transitions.
     *  @param[in] new_capacity The new number of @b states to reserve, not needles!
     *
     *  @retval `status_t::success_k` The needle was successfully added.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     *  @retval `status_t::overflow_risk_k` Too many needles for the current state ID type.
     */
    status_t try_reserve(size_t new_capacity) noexcept {

        if (new_capacity > invalid_state_k) return status_t::overflow_risk_k;

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
     *  @note Can't be called after `try_build`. Can't be called from multiple threads at the same time.
     *
     *  @retval `status_t::success_k` The needle was successfully added.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     *  @retval `status_t::overflow_risk_k` Too many needles for the current state ID type.
     *  @retval `status_t::contains_duplicates_k` The needle is already in the vocabulary.
     */
    status_t try_insert(span<char const> needle) noexcept {
        if (!needle.size()) return status_t::success_k; // Don't care about empty needles.

        state_id_t const needle_id = static_cast<state_id_t>(needles_lengths_.size());
        if (needles_lengths_.try_reserve(sz_size_bit_ceil(needles_lengths_.size() + 1)) != status_t::success_k)
            return status_t::bad_alloc_k;

        state_id_t current_state = 0;
        for (size_t pos = 0; pos < needle.size(); ++pos) {
            sz_u8_t const symbol = static_cast<sz_u8_t>(needle[pos]);
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

        // If the terminal state's output is already set, the needle already exists.
        if (outputs_[current_state] != invalid_state_k) return status_t::contains_duplicates_k;

        // Populate the new state.
        outputs_[current_state] = needle_id;
        needles_lengths_.try_push_back(needle.size()); // ? Can't fail due to `try_reserve` above
        outputs_counts_[current_state] = 1; // ? This will snowball in `try_build` if needles have shared suffixes
        outputs_offsets_[current_state] = current_state;
        return status_t::success_k;
    }

    /**
     *  @brief Construct the Finite State Machine (FSM) from the vocabulary. Can only be called @b once!
     *  @note This function is not thread safe and allocates a significant amount of memory, so it can fail.
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
     *  @brief Find all occurrences of all needles in the @p haystack.
     *  @note This is a serial reference implementation only recommended for testing.
     *  @param[in] haystack The input string to search in.
     *  @param[in] callback The handler for a @b `match_t` match, returning `true` to continue.
     */
    template <typename callback_type_>
    void find(span<char const> haystack, callback_type_ &&callback) const noexcept {
        state_id_t current_state = 0;
        for (size_t haystack_offset = 0; haystack_offset < haystack.size(); ++haystack_offset) {
            sz_u8_t symbol = static_cast<sz_u8_t>(haystack[haystack_offset]);
            current_state = transitions_[current_state][symbol];

            size_t const outputs_count = outputs_counts_[current_state];
            if (outputs_count == 0) continue;
            size_t const outputs_offset = outputs_offsets_[current_state];

            // In a small & diverse vocabulary, the following loop generally does just 1 iteration
            for (size_t output_index = 0; output_index < outputs_count; ++output_index) {
                size_t needle_id = outputs_[outputs_offset + output_index];
                size_t match_length = needles_lengths_[needle_id];
                span<char const> match_span(&haystack[haystack_offset + 1 - match_length], match_length);
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
        for (size_t haystack_offset = 0; haystack_offset < haystack.size(); ++haystack_offset) {
            sz_u8_t symbol = static_cast<sz_u8_t>(haystack[haystack_offset]);
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

    find_many(allocator_t alloc = allocator_t()) noexcept : dict_(alloc) {}
    void reset() noexcept { dict_.reset(); }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM.
     *  @retval `status_t::success_k` The needle was successfully added.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     *  @retval `status_t::overflow_risk_k` Too many needles for the current state ID type.
     *  @retval `status_t::contains_duplicates_k` The needle is already in the vocabulary.
     *  @note Before reusing, please `reset` the FSM.
     */
    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles) noexcept {
        for (auto const &needle : needles) {
            status_t status = dict_.try_insert(needle);
            if (status != status_t::success_k) return status;
        }
        return dict_.try_build();
    }

    /**
     *  @brief Counts the number of occurrences of all needles in all @p haystacks. Relevant for filtering and ranking.
     *  @param[in] haystacks The input strings to search in.
     *  @param[in] counts The output buffer for the counts of all needles in each haystack.
     *  @return The total number of occurrences found.
     */
    template <typename haystacks_type_>
    status_t try_count(haystacks_type_ &&haystacks, span<size_t> counts) const noexcept {
        _sz_assert(counts.size() == haystacks.size());
        for (size_t i = 0; i < counts.size(); ++i) counts[i] = dict_.count(haystacks[i]);
        return status_t::success_k;
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks.
     *  @param[in] haystacks The input strings to search in, with support for random access iterators.
     *  @param[in] matches The output buffer for the matches, with support for random access iterators.
     *  @param[out] matches_count The number of matches found.
     *  @return The number of matches found across all the @p haystacks.
     *  @note The @p matches reference objects should be assignable from @b `match_t`.
     */
    template <typename haystacks_type_, typename output_matches_type_>
    status_t try_find(haystacks_type_ &&haystacks, output_matches_type_ &&matches,
                      size_t &matches_count) const noexcept {
        size_t count_found = 0, count_allowed = matches.size();
        for (auto it = haystacks.begin(); it != haystacks.end() && count_found != count_allowed; ++it)
            dict_.find(*it, [&](match_t match) noexcept {
                match.haystack_index = static_cast<size_t>(it - haystacks.begin());
                matches[count_found] = match;
                count_found++;
                return count_found < count_allowed;
            });
        matches_count = count_found;
        return status_t::success_k;
    }

  private:
    dictionary_t dict_;
};

#pragma endregion // Primary API

#pragma region - Parallel Backend

/**
 *  @brief  Aho-Corasick-based @b multi-threaded multi-pattern exact substring search with.
 *  @note   Construction of the FSM is not parallelized, as it is not generally a bottleneck.
 *
 *  Implements 2 levels of parallelism: "core per input" for small haystacks and "all cores on each input"
 *  for very large ones.
 */
template <typename state_id_type_, typename allocator_type_, typename enable_>
struct find_many<state_id_type_, allocator_type_, sz_caps_sp_k, enable_> {

    using dictionary_t = aho_corasick_dictionary<state_id_type_, allocator_type_>;
    using state_id_t = typename dictionary_t::state_id_t;
    using allocator_t = typename dictionary_t::allocator_t;
    using match_t = typename dictionary_t::match_t;

    using size_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<size_t>;

    find_many(allocator_t alloc = allocator_t()) noexcept : dict_(alloc) {}
    void reset() noexcept { dict_.reset(); }

    /**
     *  @brief Indexes all of the @p needles strings into the FSM.
     *  @retval `status_t::success_k` The needle was successfully added.
     *  @retval `status_t::bad_alloc_k` Memory allocation failed.
     *  @retval `status_t::overflow_risk_k` Too many needles for the current state ID type.
     *  @retval `status_t::contains_duplicates_k` The needle is already in the vocabulary.
     *  @note Before reusing, please `reset` the FSM.
     */
    template <typename needles_type_>
    status_t try_build(needles_type_ &&needles) noexcept {
        for (auto const &needle : needles)
            if (status_t status = dict_.try_insert(needle); status != status_t::success_k) return status;
        return dict_.try_build();
    }

    /**
     *  @brief Counts the number of occurrences of all needles in all @p haystacks. Relevant for filtering and ranking.
     *  @param[in] haystacks The input strings to search in.
     *  @param[in] counts The output buffer for the counts of all needles in each haystack.
     *  @param[in] executor The executor to use for parallelization.
     *  @param[in] specs The CPU specifications on the current system to pick the right multi-threading strategy.
     *  @return The total number of occurrences found.
     */
    template <typename haystacks_type_, typename executor_type_ = dummy_executor_t>
#if _SZ_IS_CPP20
        requires executor_like<executor_type_>
#endif
    status_t try_count(haystacks_type_ &&haystacks, span<size_t> counts, executor_type_ &&executor = {},
                       cpu_specs_t const &specs = {}) const noexcept {

        _sz_assert(counts.size() == haystacks.size());
        size_t const cache_line_width = specs.cache_line_width;

        using haystacks_t = typename std::remove_reference_t<haystacks_type_>;
        using haystack_t = typename haystacks_t::value_type;

        // On small strings, individually compute the counts
        executor.for_each_dynamic(counts.size(), [&](size_t haystack_index) noexcept {
            haystack_t const &haystack = haystacks[haystack_index];
            size_t haystack_length = haystack.size();
            if (haystack_length > specs.l2_bytes) return;
            counts[haystack_index] = dict_.count(haystack);
        });

        // On longer strings, throw all cores on each haystack
        for (size_t haystack_index = 0; haystack_index < counts.size(); ++haystack_index) {
            haystack_t const &haystack = haystacks[haystack_index];
            size_t const haystack_length = haystack.size();
            // The shorter strings have already been processed
            if (haystack_length <= specs.l2_bytes) continue;

            std::atomic<size_t> count_across_cores = 0;
            size_t const cores_total = executor.thread_count();
            executor.for_each_thread([&](size_t core_index) noexcept {
                size_t count_partial = count_matches_in_one_part(haystack, core_index, cores_total, cache_line_width);
                count_across_cores.fetch_add(count_partial, std::memory_order_relaxed);
            });
            counts[haystack_index] = count_across_cores;
        }

        return status_t::success_k;
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks.
     *  @param[in] haystacks The input strings to search in, with support for random access iterators.
     *  @param[in] matches The output buffer for the matches, with support for random access iterators.
     *  @param[out] matches_count The number of matches found.
     *  @return The number of matches found across all the @p haystacks.
     *  @note The @p matches reference objects should be assignable from @b `match_t`.
     *
     *  The core problem of all such algorithm is the overlapping matches between the slices of text
     *  processed by individual threads. One approach around it is to pass in a callback, and fire it
     *  concurrently from different threads, leaving synchronization to a user... generally resorting
     *  to mutexes, atomics, and other expensive primitives! We can do better!
     *
     *  A common approach to parallelizing such algorithms is to use a little memory for
     */
    template <typename haystacks_type_, typename output_matches_type_, typename executor_type_ = dummy_executor_t>
#if _SZ_IS_CPP20
        requires executor_like<executor_type_>
#endif
    status_t try_find(haystacks_type_ &&haystacks, output_matches_type_ &&matches, size_t &matches_count,
                      executor_type_ &&executor = {}, cpu_specs_t const &specs = {}) const noexcept {

        safe_vector<size_t, size_allocator_t> counts_per_haystack(dict_.allocator());
        if (counts_per_haystack.try_resize(haystacks.size()) != status_t::success_k) return status_t::bad_alloc_k;
        status_t count_status = try_count(haystacks, counts_per_haystack, executor, specs);
        if (count_status != status_t::success_k) return count_status;
        return try_find(haystacks, counts_per_haystack, matches, matches_count, executor, specs);
    }

    /**
     *  @brief Finds all occurrences of all needles in all the @p haystacks.
     *  @param[in] haystacks The input strings to search in, with support for random access iterators.
     *  @param[in] counts The input counts for the number of matches in each haystack.
     *  @param[in] matches The output buffer for the matches, with support for random access iterators.
     *  @param[out] matches_count The number of matches found.
     *  @return The number of matches found across all the @p haystacks.
     *  @note The @p matches reference objects should be assignable from @b `match_t`.
     *
     *  A common approach to parallelizing such algorithms is to us a little memory for
     */
    template <typename haystacks_type_, typename output_matches_type_, typename executor_type_ = dummy_executor_t>
#if _SZ_IS_CPP20
        requires executor_like<executor_type_>
#endif
    status_t try_find(haystacks_type_ &&haystacks, span<size_t const> counts, output_matches_type_ &&matches,
                      size_t &matches_count, executor_type_ &&executor = {},
                      cpu_specs_t const &specs = {}) const noexcept {

        _sz_assert(counts.size() == haystacks.size());
        size_t const cores_total = specs.cores_total();
        size_t const cache_line_width = specs.cache_line_width;

        using haystacks_t = typename std::remove_reference_t<haystacks_type_>;
        using haystack_t = typename haystacks_t::value_type;
        using char_t = typename haystack_t::value_type;

        // Calculate the exclusive prefix sum of the counts to navigate into the `matches` array
        safe_vector<size_t, size_allocator_t> offsets_per_haystack(dict_.allocator());
        if (offsets_per_haystack.try_resize(counts.size()) != status_t::success_k) return status_t::bad_alloc_k;
        offsets_per_haystack[0] = 0;
        for (size_t i = 1; i < counts.size(); ++i)
            offsets_per_haystack[i] = offsets_per_haystack[i - 1] + counts[i - 1];

        // Process the small haystacks, outputting their matches individually without any synchronization
        executor.for_each_dynamic(counts.size(), [&](size_t haystack_index) noexcept {
            haystack_t const &haystack = haystacks[haystack_index];
            size_t haystack_length = haystack.size();
            if (haystack_length > specs.l2_bytes) return;
            size_t matches_found = 0;
            dict_.find({haystack.data(), haystack_length}, [&](match_t match) noexcept {
                match.haystack_index = haystack_index;
                matches[offsets_per_haystack[haystack_index] + matches_found] = match;
                ++matches_found;
                return true;
            });
            _sz_assert(counts[haystack_index] == matches_found);
        });

        // On longer strings, throw all cores on each haystack, but between the threads we need additional
        // memory to track the number of matches within a core-specific slice of the haystack.
        safe_vector<size_t, size_allocator_t> counts_per_core(dict_.allocator());
        if (counts_per_core.try_resize(cores_total) != status_t::success_k) return status_t::bad_alloc_k;
        for (size_t haystack_index = 0; haystack_index < counts.size(); ++haystack_index) {
            haystack_t const &haystack = haystacks[haystack_index];
            char_t const *haystack_begin = haystack.data();
            size_t const haystack_length = haystack.size();
            // The shorter strings have already been processed
            if (haystack_length <= specs.l2_bytes) continue;

            // First, on each core, estimate the number of matches in the haystack
            size_t const cores_total = executor.thread_count();
            executor.for_each_thread([&](size_t core_index) noexcept {
                counts_per_core[core_index] =
                    count_matches_in_one_part(haystack, core_index, cores_total, cache_line_width);
            });

            // Now that we know the number of matches to expect per slice, we can convert the counts
            // into offsets using inclusive prefix sum
            {
                for (size_t core_index = 1; core_index < cores_total; ++core_index)
                    counts_per_core[core_index] += counts_per_core[core_index - 1];
            }

            // We shouldn't even consider needles longer than the haystack
            size_t const max_needle_length = std::min(dict_.max_needle_length(), haystack_length);

            // On each core, pick an overlapping slice and go through all of the matches in it,
            // that start before the end of the private slice.
            size_t const bytes_per_core_optimal =
                round_up_to_multiple(divide_round_up(haystack_length, cores_total), cache_line_width);
            size_t const count_matches_before_this_haystack = offsets_per_haystack[haystack_index];
            executor.for_each_thread([&](size_t core_index) noexcept {
                size_t const count_matches_before_this_core = core_index ? counts_per_core[core_index - 1] : 0;
                size_t const count_matches_expected_on_this_core =
                    counts_per_core[core_index] - count_matches_before_this_core;

                // The last core may have a smaller slice, so we need to be careful
                size_t const bytes_for_core =
                    std::min(bytes_per_core_optimal, haystack_length - core_index * bytes_per_core_optimal);
                char_t const *optimal_start = haystack_begin + core_index * bytes_per_core_optimal;
                char_t const *optimal_end = optimal_start + bytes_for_core;
                char_t const *overlapping_end =
                    std::min(optimal_start + bytes_for_core + max_needle_length - 1, haystack_begin + haystack_length);

                // Iterate through the matches in the overlapping region
                size_t count_matches_found_on_this_core = 0;
                dict_.find({optimal_start, overlapping_end}, [&](match_t match) noexcept {
                    bool blongs_to_this_core = match.needle.begin() < optimal_end;
                    if (!blongs_to_this_core) return true;
                    match.haystack_index = haystack_index;
                    matches[count_matches_before_this_haystack + count_matches_before_this_core +
                            count_matches_found_on_this_core] = match;
                    count_matches_found_on_this_core++;
                    return true;
                });
                _sz_assert(count_matches_found_on_this_core == count_matches_expected_on_this_core);
            });
        }

        // Aggregate the results
        matches_count = 0;
        for (size_t count : counts) matches_count += count;
        return status_t::success_k;
    }

  private:
    dictionary_t dict_;

    /**
     *  @brief  Helper method implementing the core logic of the parallel `try_count` and part of `try_find`.
     *          For a given single input haystack, assumes all of the cores are processing it in parallel,
     *          and this method is called from each core with its own index to count the number of potentially
     *          overlapping matches.
     */
    template <typename char_type_>
    size_t count_matches_in_one_part(span<char_type_ const> haystack, size_t core_index, size_t cores_total,
                                     size_t cache_line_width) const noexcept {

        using char_t = char_type_;
        char_t const *haystack_begin = haystack.data();
        size_t const haystack_length = haystack.size();
        char_t const *const haystack_end = haystack_begin + haystack_length;
        size_t const bytes_per_core_optimal =
            round_up_to_multiple(divide_round_up(haystack_length, cores_total), cache_line_width);

        // We shouldn't even consider needles longer than the haystack
        size_t const max_needle_length = std::min(dict_.max_needle_length(), haystack_length);

        // We may have a case of a thread receiving no data at all
        char_t const *optimal_start = haystack_begin + core_index * bytes_per_core_optimal;
        if (optimal_start >= haystack_end) return 0;

        // First, each core will process its own slice excluding the overlapping regions
        char_t const *optimal_end = std::min(optimal_start + bytes_per_core_optimal, haystack_end);
        size_t const count_matches_non_overlapping = dict_.count({optimal_start, optimal_end});

        // Now, each thread will take care of the subsequent overlapping regions,
        // but we must be careful for cases when the core-specific slice is shorter
        // than the longest needle! It's a very unlikely case in practice, but we
        // still may want an optimization for it down the road.
        char_t const *overlapping_start;
        char_t const *overlapping_end;
        if (optimal_start + max_needle_length >= optimal_end) {
            // Our needles are longer than a slice for the core
            overlapping_start = optimal_start;
            overlapping_end = std::min(optimal_start + max_needle_length, haystack_end);
        }
        else {
            overlapping_start = std::max(optimal_end - max_needle_length + 1, haystack_begin);
            overlapping_end = std::min(optimal_end + max_needle_length - 1, haystack_end);
        }

        // Count the matches that start in one core's slice and end in another
        size_t count_matches_overlapping = 0;
        dict_.find({overlapping_start, overlapping_end}, [&](match_t match) noexcept {
            bool is_boundary = match.needle.begin() < optimal_end && match.needle.end() > optimal_end;
            count_matches_overlapping += is_boundary;
            return true;
        });

        // Now, finally, aggregate the results
        return count_matches_non_overlapping + count_matches_overlapping;
    }
};

#pragma endregion // Parallel Backend

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGCUZILLA_FIND_MANY_HPP_
