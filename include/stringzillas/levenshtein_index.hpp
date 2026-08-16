/**
 *  @brief Immutable exact Levenshtein dictionary retrieval for small edit bounds.
 *  @file include/stringzillas/levenshtein_index.hpp
 *  @author Guillaume de Rouville
 *
 *  Small bounds use the deletion lookup table. Larger bounds and long strings use a compact prefix tree.
 *  Both paths return candidates only after their exact distance has been checked.
 */
#ifndef STRINGZILLAS_LEVENSHTEIN_INDEX_HPP_
#define STRINGZILLAS_LEVENSHTEIN_INDEX_HPP_

#include "stringzillas/types.hpp"
#include "stringzilla/utf8_runes/serial.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace ashvardanian {
namespace stringzillas {

struct levenshtein_index_match_t {
    u32_t id;
    u8_t distance;
};

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
    static constexpr size_t automatic_deletion_max_word_length_k = std::numeric_limits<size_t>::max();

    static_assert(std::is_integral<symbol_t>::value &&
                      !std::is_same<typename std::remove_cv<symbol_t>::type, bool>::value &&
                      sizeof(symbol_t) <= sizeof(u32_t),
                  "Levenshtein index symbols must be integral and at most 32 bits wide");

  private:
    static constexpr size_t min_prefix_bits_k = 20;
    static constexpr size_t max_prefix_bits_k = 25;
    static constexpr size_t packed_id_bits_k = 20;
    static constexpr u32_t packed_id_mask_k = (u32_t(1) << packed_id_bits_k) - 1;
    static constexpr u8_t rejected_distance_k = 3;

    struct trie_node_t {
        u32_t first_edge = 0;
        u32_t edges_count = 0;
        u32_t first_terminal = 0;
        u32_t terminals_count = 0;
    };
    struct trie_edge_t {
        u64_t label_offset = 0;
        u32_t child = 0;
        u32_t label_length = 0;
    };
    struct trie_frame_t {
        u32_t node = 0;
        u32_t next_edge = 0;
        u32_t end_edge = 0;
        u32_t depth = 0;
    };
    struct dfa_frame_t {
        u64_t state = 0;
        u32_t node = 0;
        u32_t next_edge = 0;
        u32_t end_edge = 0;
        u32_t depth = 0;
    };
    struct dfa_transition_t {
        u64_t state = 0;
        u64_t next_state = 0;
        u32_t generation = 0;
        u32_t depth = 0;
        u32_t symbol = 0;
        u8_t min_distance = 0;
        u8_t terminal_distance = 0;
    };
    struct dfa_step_t {
        u64_t state = 0;
        u8_t min_distance = 0;
        u8_t terminal_distance = 0;
    };
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

    // Persistent state has two indexes over one owned dictionary. Short words contribute deletion records, while the
    // prefix tree covers long words and every word needed for bounds above two. Only scratch_t changes during search.
    allocator_t alloc_ {};
    vector_t<symbol_t> tape_ {alloc_};
    vector_t<u64_t> offsets_ {alloc_};
    vector_t<u32_t> directory_ {alloc_};
    vector_t<directory_word_t> directory_words_ {alloc_};
    vector_t<u32_t> packed_records_ {alloc_};
    vector_t<u64_t> wide_records_ {alloc_};
    vector_t<trie_node_t> trie_nodes_ {alloc_};
    vector_t<trie_edge_t> trie_edges_ {alloc_};
    vector_t<u32_t> trie_terminals_ {alloc_};
    u8_t max_distance_ = 0;
    size_t max_word_length_ = 0;
    size_t deletion_max_word_length_ = 64;
    size_t fallback_words_count_ = 0;
    u8_t prefix_bits_ = min_prefix_bits_k;
    u8_t suffix_bits_ = 32 - min_prefix_bits_k;
    u32_t suffix_mask_ = (u32_t(1) << (32 - min_prefix_bits_k)) - 1;

    static u32_t fold_hash_(u64_t hash) noexcept {
        hash ^= hash >> 33;
        hash *= 0xff51afd7ed558ccdull;
        hash ^= hash >> 33;
        return static_cast<u32_t>(hash ^ (hash >> 32));
    }

    static bool checked_add_(size_t &total, size_t increment) noexcept {
        if (increment > std::numeric_limits<size_t>::max() - total) return false;
        total += increment;
        return true;
    }

    static bool residuals_upper_bound_(size_t length, u8_t distance, size_t &count) noexcept {
        count = 1;
        if (distance >= 1 && !checked_add_(count, length)) return false;
        if (distance >= 2) {
            if (length && length - 1 > std::numeric_limits<size_t>::max() / length) return false;
            if (!checked_add_(count, length * (length - 1) / 2)) return false;
        }
        return true;
    }

    template <typename scratch_type_>
    static status_t generate_residuals_(span<symbol_t const> word, u8_t distance, scratch_type_ &scratch) noexcept {
        // Hash the original word and every distinct result of removing one or two symbols. Equal residuals are only a
        // candidate filter. verify_ checks the owned word before any match is returned.
        size_t upper_bound = 0;
        if (!residuals_upper_bound_(word.size(), distance, upper_bound)) return status_t::overflow_risk_k;
        if (status_t status = scratch.residuals.try_resize(0); status != status_t::success_k) return status;
        if (status_t status = scratch.residuals.try_reserve(upper_bound); status != status_t::success_k) return status;
        if (status_t status = scratch.prefixes.try_resize(word.size() + 1); status != status_t::success_k) return status;
        if (status_t status = scratch.powers.try_resize(word.size() + 1); status != status_t::success_k) return status;

        static constexpr u64_t base = 0x9E3779B185EBCA87ull;
        scratch.prefixes[0] = 0;
        scratch.powers[0] = 1;
        for (size_t index = 0; index != word.size(); ++index) {
            using unsigned_symbol_t = typename std::make_unsigned<symbol_t>::type;
            scratch.prefixes[index + 1] = scratch.prefixes[index] * base +
                                           static_cast<u64_t>(static_cast<unsigned_symbol_t>(word[index])) + 1;
            scratch.powers[index + 1] = scratch.powers[index] * base;
        }
        auto const substring_hash = [&](size_t begin, size_t end) noexcept {
            return scratch.prefixes[end] - scratch.prefixes[begin] * scratch.powers[end - begin];
        };
        if (status_t status = scratch.residuals.try_push_back(fold_hash_(scratch.prefixes[word.size()]));
            status != status_t::success_k)
            return status;
        if (distance >= 1)
            for (size_t first = 0; first != word.size(); ++first) {
                u64_t const without_first = scratch.prefixes[first] * scratch.powers[word.size() - first - 1] +
                                            substring_hash(first + 1, word.size());
                if (status_t status = scratch.residuals.try_push_back(fold_hash_(without_first));
                    status != status_t::success_k)
                    return status;
                if (distance >= 2)
                    for (size_t second = first + 1; second != word.size(); ++second) {
                        size_t const middle_length = second - first - 1;
                        size_t const suffix_length = word.size() - second - 1;
                        u64_t hash = scratch.prefixes[first] * scratch.powers[middle_length] +
                                     substring_hash(first + 1, second);
                        hash = hash * scratch.powers[suffix_length] + substring_hash(second + 1, word.size());
                        if (status_t status = scratch.residuals.try_push_back(fold_hash_(hash));
                            status != status_t::success_k)
                            return status;
                    }
            }
        std::sort(scratch.residuals.begin(), scratch.residuals.end());
        auto const unique_end = std::unique(scratch.residuals.begin(), scratch.residuals.end());
        return scratch.residuals.try_resize(static_cast<size_t>(unique_end - scratch.residuals.begin()));
    }

    span<symbol_t const> word_(u32_t id) const noexcept {
        size_t const begin = static_cast<size_t>(offsets_[id]);
        size_t const end = static_cast<size_t>(offsets_[id + 1]);
        return {tape_.data() + begin, end - begin};
    }

    static u8_t distance_within_one_(span<symbol_t const> first, span<symbol_t const> second) noexcept {
        if (first.size() == second.size() &&
            (first.empty() || std::memcmp(first.data(), second.data(), first.size() * sizeof(symbol_t)) == 0))
            return 0;
        if (first.size() + 1 < second.size() || second.size() + 1 < first.size()) return rejected_distance_k;
        size_t first_index = 0, second_index = 0;
        bool edited = false;
        while (first_index != first.size() && second_index != second.size()) {
            if (first[first_index] == second[second_index]) ++first_index, ++second_index;
            else if (edited) return rejected_distance_k;
            else {
                edited = true;
                if (first.size() > second.size()) ++first_index;
                else if (second.size() > first.size()) ++second_index;
                else ++first_index, ++second_index;
            }
        }
        size_t const tail = first.size() - first_index + second.size() - second_index;
        return static_cast<u8_t>((edited ? 1 : 0) + tail <= 1 ? 1 : rejected_distance_k);
    }

    template <typename scratch_type_>
    static u8_t distance_within_two_(span<symbol_t const> first, span<symbol_t const> second,
                                     scratch_type_ &scratch) noexcept {
        // Myers fits byte patterns through 64 symbols in one machine word. Wider symbols and longer patterns use the
        // same narrow dynamic-programming band, clipped at three because callers only need 0, 1, 2, or rejected.
        if (first.size() > second.size()) std::swap(first, second);
        if (second.size() - first.size() > 2) return rejected_distance_k;
        if (first.empty()) return second.size() <= 2 ? static_cast<u8_t>(second.size()) : rejected_distance_k;

        if constexpr (sizeof(symbol_t) == 1) {
            if (first.size() <= 64) {
                u64_t equality[256] = {};
                for (size_t index = 0; index != first.size(); ++index)
                    equality[static_cast<u8_t>(first[index])] |= u64_t(1) << index;
                u64_t positive = ~u64_t(0), negative = 0;
                u64_t const top = u64_t(1) << (first.size() - 1);
                size_t score = first.size();
                for (size_t index = 0; index != second.size(); ++index) {
                    u64_t const matches = equality[static_cast<u8_t>(second[index])];
                    u64_t const carry_in = matches | negative;
                    u64_t const differences = (((carry_in & positive) + positive) ^ positive) | carry_in;
                    u64_t const horizontal_positive = negative | ~(differences | positive);
                    u64_t const horizontal_negative = positive & differences;
                    score += (horizontal_positive & top) != 0;
                    score -= (horizontal_negative & top) != 0;
                    u64_t const shifted_positive = (horizontal_positive << 1) | 1;
                    u64_t const shifted_negative = horizontal_negative << 1;
                    positive = shifted_negative | ~(differences | shifted_positive);
                    negative = shifted_positive & differences;
                    if (score > 2 + second.size() - index - 1) return rejected_distance_k;
                }
                return score <= 2 ? static_cast<u8_t>(score) : rejected_distance_k;
            }
        }

        size_t const width = first.size() + 1;
        if (scratch.dp_rows.try_resize(width * 2) != status_t::success_k) return rejected_distance_k;
        u8_t *previous = scratch.dp_rows.data();
        u8_t *current = previous + width;
        std::fill(previous, previous + width, rejected_distance_k);
        for (size_t column = 0; column <= sz_min_of_two(first.size(), size_t(2)); ++column)
            previous[column] = static_cast<u8_t>(column);
        for (size_t row = 1; row <= second.size(); ++row) {
            std::fill(current, current + width, rejected_distance_k);
            if (row <= 2) current[0] = static_cast<u8_t>(row);
            size_t const from = row > 2 ? row - 2 : 1;
            size_t const to = sz_min_of_two(first.size(), row + 2);
            for (size_t column = from; column <= to; ++column) {
                unsigned const value = std::min(
                    {unsigned(previous[column]) + 1, unsigned(current[column - 1]) + 1,
                     unsigned(previous[column - 1]) + (first[column - 1] != second[row - 1])});
                current[column] = static_cast<u8_t>(sz_min_of_two(value, unsigned(rejected_distance_k)));
            }
            std::swap(previous, current);
        }
        return previous[first.size()] <= 2 ? previous[first.size()] : rejected_distance_k;
    }

    template <typename scratch_type_>
    u8_t verify_(u32_t id, span<symbol_t const> query, u8_t bound, scratch_type_ &scratch) const noexcept {
        span<symbol_t const> const candidate = word_(id);
        if (bound == 0)
            return candidate.size() == query.size() &&
                           (candidate.empty() ||
                            std::memcmp(candidate.data(), query.data(), query.size() * sizeof(symbol_t)) == 0)
                       ? 0
                       : rejected_distance_k;
        if (bound == 1) return distance_within_one_(candidate, query);
        return distance_within_two_(candidate, query, scratch);
    }

    status_t build_trie_(bool fallback_only) noexcept {
        // Sort IDs, form a prefix tree from adjacent common prefixes, then compress every run without a branch into
        // one edge that points back into the owned dictionary tape.
        size_t const words_count = fallback_only ? fallback_words_count_ : size();
        vector_t<u32_t> order {alloc_}, parents {alloc_}, depths {alloc_}, representatives {alloc_},
            word_nodes {alloc_}, child_cursors {alloc_}, terminal_cursors {alloc_}, stack {alloc_};
        if (order.try_reserve(words_count) != status_t::success_k ||
            word_nodes.try_resize(size()) != status_t::success_k)
            return status_t::bad_alloc_k;
        for (u32_t id = 0; id != size(); ++id) {
            if (fallback_only && word_(id).size() <= deletion_max_word_length_) continue;
            if (order.try_push_back(id) != status_t::success_k) return status_t::bad_alloc_k;
        }
        std::sort(order.begin(), order.end(), [&](u32_t first_id, u32_t second_id) {
            span<symbol_t const> const first = word_(first_id), second = word_(second_id);
            size_t const shared = sz_min_of_two(first.size(), second.size());
            size_t common = 0;
            while (common != shared && first[common] == second[common]) ++common;
            if (common != shared) return first[common] < second[common];
            if (first.size() != second.size()) return first.size() < second.size();
            return first_id < second_id;
        });

        if (parents.try_push_back(0) != status_t::success_k || depths.try_push_back(0) != status_t::success_k ||
            representatives.try_push_back(0) != status_t::success_k || stack.try_push_back(0) != status_t::success_k)
            return status_t::bad_alloc_k;
        span<symbol_t const> previous;
        u32_t previous_id = 0;
        for (u32_t id : order) {
            span<symbol_t const> const word = word_(id);
            if (word.size() > std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
            size_t common = 0, common_limit = sz_min_of_two(previous.size(), word.size());
            while (common != common_limit && previous[common] == word[common]) ++common;
            u32_t previous_child = stack.back();
            while (depths[stack.back()] > common) {
                previous_child = stack.back();
                if (stack.try_resize(stack.size() - 1) != status_t::success_k) return status_t::bad_alloc_k;
            }
            if (depths[stack.back()] < common) {
                if (parents.size() == std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
                u32_t const branch = static_cast<u32_t>(parents.size());
                if (parents.try_push_back(stack.back()) != status_t::success_k ||
                    depths.try_push_back(static_cast<u32_t>(common)) != status_t::success_k ||
                    representatives.try_push_back(previous_id) != status_t::success_k ||
                    stack.try_push_back(branch) != status_t::success_k)
                    return status_t::bad_alloc_k;
                parents[previous_child] = branch;
            }
            if (word.size() != common) {
                if (parents.size() == std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
                u32_t const leaf = static_cast<u32_t>(parents.size());
                if (parents.try_push_back(stack.back()) != status_t::success_k ||
                    depths.try_push_back(static_cast<u32_t>(word.size())) != status_t::success_k ||
                    representatives.try_push_back(id) != status_t::success_k ||
                    stack.try_push_back(leaf) != status_t::success_k)
                    return status_t::bad_alloc_k;
            }
            word_nodes[id] = stack.back();
            previous = word;
            previous_id = id;
        }

        size_t const nodes_count = parents.size();
        if (trie_nodes_.try_resize(nodes_count) != status_t::success_k ||
            trie_edges_.try_resize(nodes_count ? nodes_count - 1 : 0) != status_t::success_k ||
            trie_terminals_.try_resize(words_count) != status_t::success_k ||
            child_cursors.try_resize(nodes_count) != status_t::success_k ||
            terminal_cursors.try_resize(nodes_count) != status_t::success_k)
            return status_t::bad_alloc_k;
        for (size_t node = 1; node != nodes_count; ++node) ++trie_nodes_[parents[node]].edges_count;
        for (u32_t id : order) ++trie_nodes_[word_nodes[id]].terminals_count;
        u32_t edge_offset = 0, terminal_offset = 0;
        for (size_t node = 0; node != nodes_count; ++node) {
            trie_nodes_[node].first_edge = edge_offset;
            trie_nodes_[node].first_terminal = terminal_offset;
            child_cursors[node] = edge_offset;
            terminal_cursors[node] = terminal_offset;
            edge_offset += trie_nodes_[node].edges_count;
            terminal_offset += trie_nodes_[node].terminals_count;
        }
        for (u32_t node = 1; node != nodes_count; ++node) {
            u32_t const parent = parents[node];
            u32_t const representative = representatives[node];
            trie_edges_[child_cursors[parent]++] =
                trie_edge_t {offsets_[representative] + depths[parent], node, depths[node] - depths[parent]};
        }
        for (u32_t id : order)
            trie_terminals_[terminal_cursors[word_nodes[id]]++] = id;
        return status_t::success_k;
    }

    template <typename scratch_type_>
    status_t find_trie_dfa_(span<symbol_t const> query, u8_t bound, bool fallback_only, scratch_type_ &scratch,
                            vector_t<match_t> &matches) const noexcept {
        // Queries through 15 symbols fit a complete clipped distance row into one 64-bit value. Cache each row and
        // next-symbol transition reached while walking the tree. Cache exhaustion only recomputes a transition.
        static constexpr size_t transitions_capacity = size_t(1) << 15;
        static constexpr size_t transitions_mask = transitions_capacity - 1;
        if (scratch.dfa_transitions.size() != transitions_capacity) {
            if (scratch.dfa_transitions.try_resize(transitions_capacity) != status_t::success_k)
                return status_t::bad_alloc_k;
            for (auto &entry : scratch.dfa_transitions) entry.generation = 0;
            scratch.dfa_generation = 0;
        }
        if (++scratch.dfa_generation == 0) {
            for (auto &entry : scratch.dfa_transitions) entry.generation = 0;
            scratch.dfa_generation = 1;
        }
        if (scratch.dfa_frames.try_resize(0) != status_t::success_k ||
            scratch.dfa_frames.try_reserve(max_word_length_ + 1) != status_t::success_k)
            return status_t::bad_alloc_k;
        u8_t const cap = bound + 1;

        auto const emit_terminals = [&](u32_t node_id, u8_t distance) noexcept -> status_t {
            trie_node_t const &node = trie_nodes_[node_id];
            for (size_t offset = node.first_terminal; offset != node.first_terminal + node.terminals_count; ++offset) {
                u32_t const id = trie_terminals_[offset];
                if (fallback_only && word_(id).size() <= deletion_max_word_length_) continue;
                if (status_t status = matches.try_push_back(match_t {id, distance});
                    status != status_t::success_k)
                    return status;
            }
            return status_t::success_k;
        };
        auto const compute_transition = [&](u64_t state, symbol_t symbol) noexcept {
            u64_t next = (state & 0xF) < cap ? (state & 0xF) + 1 : cap;
            u8_t left = static_cast<u8_t>(next & 0xF);
            u8_t diagonal = static_cast<u8_t>(state & 0xF);
            u8_t min_distance = left;
            for (size_t column = 1; column <= query.size(); ++column) {
                u8_t const above = static_cast<u8_t>((state >> (column * 4)) & 0xF);
                unsigned const value = std::min({unsigned(above) + 1, unsigned(left) + 1,
                                                 unsigned(diagonal) + (query[column - 1] != symbol),
                                                 unsigned(cap)});
                left = static_cast<u8_t>(value);
                min_distance = sz_min_of_two(min_distance, left);
                diagonal = above;
                next |= u64_t(left) << (column * 4);
            }
            return dfa_step_t {next, min_distance, left};
        };
        auto const transition = [&](u64_t state, symbol_t symbol) noexcept {
            u64_t mixed = state ^ (static_cast<u64_t>(symbol) * 0x9E3779B185EBCA87ull);
            mixed ^= mixed >> 33;
            mixed *= 0xff51afd7ed558ccdull;
            mixed ^= mixed >> 33;
            size_t slot = static_cast<size_t>(mixed) & transitions_mask;
            for (size_t probe = 0; probe != transitions_capacity; ++probe) {
                dfa_transition_t &entry = scratch.dfa_transitions[slot];
                if (entry.generation != scratch.dfa_generation) {
                    entry.state = state;
                    entry.symbol = static_cast<u32_t>(symbol);
                    entry.depth = 0;
                    dfa_step_t const step = compute_transition(state, symbol);
                    entry.next_state = step.state;
                    entry.min_distance = step.min_distance;
                    entry.terminal_distance = step.terminal_distance;
                    entry.generation = scratch.dfa_generation;
                    return step;
                }
                if (entry.state == state && entry.symbol == static_cast<u32_t>(symbol) && entry.depth == 0)
                    return dfa_step_t {entry.next_state, entry.min_distance, entry.terminal_distance};
                slot = (slot + 1) & transitions_mask;
            }
            return compute_transition(state, symbol);
        };

        u64_t initial_state = 0;
        for (size_t column = 0; column <= query.size(); ++column)
            initial_state |= u64_t(sz_min_of_two(column, size_t(cap))) << (column * 4);
        u8_t const initial_distance = static_cast<u8_t>((initial_state >> (query.size() * 4)) & 0xF);
        if (initial_distance <= bound && trie_nodes_[0].terminals_count)
            if (status_t status = emit_terminals(0, initial_distance); status != status_t::success_k) return status;

        trie_node_t const &root = trie_nodes_[0];
        if (scratch.dfa_frames.try_push_back(
                dfa_frame_t {initial_state, 0, root.first_edge, root.first_edge + root.edges_count, 0}) !=
            status_t::success_k)
            return status_t::bad_alloc_k;
        while (scratch.dfa_frames.size()) {
            dfa_frame_t &frame = scratch.dfa_frames.back();
            if (frame.next_edge == frame.end_edge) {
                scratch.dfa_frames.try_resize(scratch.dfa_frames.size() - 1);
                continue;
            }
            trie_edge_t const edge = trie_edges_[frame.next_edge++];
            u64_t state = frame.state;
            dfa_step_t step;
            bool alive = true;
            for (size_t offset = 0; offset != edge.label_length; ++offset) {
                step = transition(state, tape_[edge.label_offset + offset]);
                if (step.min_distance > bound) {
                    alive = false;
                    break;
                }
                state = step.state;
            }
            if (!alive) continue;
            trie_node_t const &child = trie_nodes_[edge.child];
            if (step.terminal_distance <= bound && child.terminals_count)
                if (status_t status = emit_terminals(edge.child, step.terminal_distance);
                    status != status_t::success_k)
                    return status;
            if (step.min_distance <= bound) {
                if (scratch.dfa_frames.try_push_back(
                        dfa_frame_t {state, edge.child, child.first_edge, child.first_edge + child.edges_count,
                                     frame.depth + edge.label_length}) !=
                    status_t::success_k)
                    return status_t::bad_alloc_k;
            }
        }
        return status_t::success_k;
    }

    template <typename scratch_type_>
    status_t find_trie_banded_dfa_(span<symbol_t const> query, u8_t bound, bool fallback_only, scratch_type_ &scratch,
                                   vector_t<match_t> &matches) const noexcept {
        // Longer queries cannot pack the complete row. For bounds through seven, pack only the moving window that can
        // still reach an accepted answer. Tree depth is part of the key because it positions that window.
        static constexpr size_t transitions_capacity = size_t(1) << 15;
        static constexpr size_t transitions_mask = transitions_capacity - 1;
        if (scratch.dfa_transitions.size() != transitions_capacity) {
            if (scratch.dfa_transitions.try_resize(transitions_capacity) != status_t::success_k)
                return status_t::bad_alloc_k;
            for (auto &entry : scratch.dfa_transitions) entry.generation = 0;
            scratch.dfa_generation = 0;
        }
        if (++scratch.dfa_generation == 0) {
            for (auto &entry : scratch.dfa_transitions) entry.generation = 0;
            scratch.dfa_generation = 1;
        }
        if (scratch.dfa_frames.try_resize(0) != status_t::success_k ||
            scratch.dfa_frames.try_reserve(max_word_length_ + 1) != status_t::success_k)
            return status_t::bad_alloc_k;
        u8_t const cap = bound + 1;

        auto const emit_terminals = [&](u32_t node_id, u8_t distance) noexcept -> status_t {
            trie_node_t const &node = trie_nodes_[node_id];
            for (size_t offset = node.first_terminal; offset != node.first_terminal + node.terminals_count; ++offset) {
                u32_t const id = trie_terminals_[offset];
                if (fallback_only && word_(id).size() <= deletion_max_word_length_) continue;
                if (status_t status = matches.try_push_back(match_t {id, distance});
                    status != status_t::success_k)
                    return status;
            }
            return status_t::success_k;
        };
        auto const compute_transition = [&](u64_t state, u32_t previous_depth, symbol_t symbol) noexcept {
            size_t const current_depth = size_t(previous_depth) + 1;
            size_t const previous_from = previous_depth > bound ? previous_depth - bound : 0;
            size_t const previous_to = sz_min_of_two(query.size(), size_t(previous_depth) + bound);
            size_t const current_from = current_depth > bound ? current_depth - bound : 0;
            size_t const current_to = sz_min_of_two(query.size(), current_depth + bound);
            auto const previous_at = [&](size_t column) noexcept -> u8_t {
                return column >= previous_from && column <= previous_to
                           ? static_cast<u8_t>((state >> ((column - previous_from) * 4)) & 0xF)
                           : cap;
            };
            u64_t next = 0;
            u8_t left = cap;
            u8_t min_distance = cap;
            u8_t terminal_distance = cap;
            for (size_t column = current_from; column <= current_to; ++column) {
                u8_t value;
                if (column == 0) value = static_cast<u8_t>(sz_min_of_two(current_depth, size_t(cap)));
                else {
                    unsigned const deletion = unsigned(previous_at(column)) + 1;
                    unsigned const insertion = column > current_from ? unsigned(left) + 1 : cap;
                    unsigned const substitution = unsigned(previous_at(column - 1)) + (query[column - 1] != symbol);
                    value = static_cast<u8_t>(
                        sz_min_of_two(std::min({deletion, insertion, substitution}), unsigned(cap)));
                }
                next |= u64_t(value) << ((column - current_from) * 4);
                left = value;
                min_distance = sz_min_of_two(min_distance, value);
                if (column == query.size()) terminal_distance = value;
            }
            return dfa_step_t {next, min_distance, terminal_distance};
        };
        auto const transition = [&](u64_t state, u32_t depth, symbol_t symbol) noexcept {
            u64_t mixed = state ^ (u64_t(depth) * 0xD6E8FEB86659FD93ull) ^
                          (static_cast<u64_t>(symbol) * 0x9E3779B185EBCA87ull);
            mixed ^= mixed >> 33;
            mixed *= 0xff51afd7ed558ccdull;
            mixed ^= mixed >> 33;
            size_t slot = static_cast<size_t>(mixed) & transitions_mask;
            for (size_t probe = 0; probe != transitions_capacity; ++probe) {
                dfa_transition_t &entry = scratch.dfa_transitions[slot];
                if (entry.generation != scratch.dfa_generation) {
                    entry.state = state;
                    entry.depth = depth;
                    entry.symbol = static_cast<u32_t>(symbol);
                    dfa_step_t const step = compute_transition(state, depth, symbol);
                    entry.next_state = step.state;
                    entry.min_distance = step.min_distance;
                    entry.terminal_distance = step.terminal_distance;
                    entry.generation = scratch.dfa_generation;
                    return step;
                }
                if (entry.state == state && entry.depth == depth && entry.symbol == static_cast<u32_t>(symbol))
                    return dfa_step_t {entry.next_state, entry.min_distance, entry.terminal_distance};
                slot = (slot + 1) & transitions_mask;
            }
            return compute_transition(state, depth, symbol);
        };

        size_t const root_to = sz_min_of_two(query.size(), size_t(bound));
        u64_t initial_state = 0;
        for (size_t column = 0; column <= root_to; ++column) initial_state |= u64_t(column) << (column * 4);
        if (query.size() <= bound && trie_nodes_[0].terminals_count)
            if (status_t status = emit_terminals(0, static_cast<u8_t>(query.size()));
                status != status_t::success_k)
                return status;

        trie_node_t const &root = trie_nodes_[0];
        if (scratch.dfa_frames.try_push_back(
                dfa_frame_t {initial_state, 0, root.first_edge, root.first_edge + root.edges_count, 0}) !=
            status_t::success_k)
            return status_t::bad_alloc_k;
        while (scratch.dfa_frames.size()) {
            dfa_frame_t &frame = scratch.dfa_frames.back();
            if (frame.next_edge == frame.end_edge) {
                scratch.dfa_frames.try_resize(scratch.dfa_frames.size() - 1);
                continue;
            }
            trie_edge_t const edge = trie_edges_[frame.next_edge++];
            u64_t state = frame.state;
            dfa_step_t step;
            bool alive = true;
            for (u32_t offset = 0; offset != edge.label_length; ++offset) {
                step = transition(state, frame.depth + offset, tape_[edge.label_offset + offset]);
                if (step.min_distance > bound) {
                    alive = false;
                    break;
                }
                state = step.state;
            }
            if (!alive) continue;
            trie_node_t const &child = trie_nodes_[edge.child];
            if (step.terminal_distance <= bound && child.terminals_count)
                if (status_t status = emit_terminals(edge.child, step.terminal_distance);
                    status != status_t::success_k)
                    return status;
            if (step.min_distance <= bound) {
                if (scratch.dfa_frames.try_push_back(
                        dfa_frame_t {state, edge.child, child.first_edge, child.first_edge + child.edges_count,
                                     frame.depth + edge.label_length}) !=
                    status_t::success_k)
                    return status_t::bad_alloc_k;
            }
        }
        return status_t::success_k;
    }

    template <typename scratch_type_>
    status_t find_trie_(span<symbol_t const> query, u8_t bound, bool fallback_only, scratch_type_ &scratch,
                        vector_t<match_t> &matches) const noexcept {
        if (!trie_nodes_.size()) return status_t::success_k;
        if (query.size() <= 15 && bound <= 14) return find_trie_dfa_(query, bound, fallback_only, scratch, matches);
        if (bound <= 7) return find_trie_banded_dfa_(query, bound, fallback_only, scratch, matches);
        // General fallback for larger bounds. Keep one narrow row per tree depth so siblings can reuse their parent's
        // row. Values are clipped at bound + 1 and a branch stops when its complete live window is rejected.
        size_t const stride = size_t(2) * bound + 3;
        if (max_word_length_ + 1 > std::numeric_limits<size_t>::max() / stride)
            return status_t::overflow_risk_k;
        if (scratch.trie_rows.try_resize((max_word_length_ + 1) * stride) != status_t::success_k ||
            scratch.trie_frames.try_resize(0) != status_t::success_k ||
            scratch.trie_frames.try_reserve(max_word_length_ + 1) != status_t::success_k)
            return status_t::bad_alloc_k;
        u16_t const cap = static_cast<u16_t>(bound) + 1;
        u16_t *root_row = scratch.trie_rows.data();
        size_t const root_to = sz_min_of_two(query.size(), size_t(bound));
        for (size_t column = 0; column <= root_to; ++column) root_row[column] = static_cast<u16_t>(column);

        auto const emit_terminals = [&](u32_t node_id, u16_t distance) noexcept -> status_t {
            trie_node_t const &node = trie_nodes_[node_id];
            for (size_t offset = node.first_terminal; offset != node.first_terminal + node.terminals_count; ++offset) {
                u32_t const id = trie_terminals_[offset];
                if (fallback_only && word_(id).size() <= deletion_max_word_length_) continue;
                if (status_t status = matches.try_push_back(match_t {id, static_cast<u8_t>(distance)});
                    status != status_t::success_k)
                    return status;
            }
            return status_t::success_k;
        };
        if (query.size() <= bound)
            if (status_t status = emit_terminals(0, static_cast<u16_t>(query.size()));
                status != status_t::success_k)
                return status;

        trie_node_t const &root = trie_nodes_[0];
        if (scratch.trie_frames.try_push_back(
                trie_frame_t {0, root.first_edge, root.first_edge + root.edges_count, 0}) != status_t::success_k)
            return status_t::bad_alloc_k;
        while (scratch.trie_frames.size()) {
            trie_frame_t &frame = scratch.trie_frames.back();
            if (frame.next_edge == frame.end_edge) {
                scratch.trie_frames.try_resize(scratch.trie_frames.size() - 1);
                continue;
            }
            trie_edge_t const edge = trie_edges_[frame.next_edge++];
            u16_t row_min = cap;
            u16_t terminal_distance = cap;
            bool alive = true;
            for (u32_t edge_offset = 0; edge_offset != edge.label_length; ++edge_offset) {
                size_t const previous_depth = frame.depth + edge_offset;
                size_t const current_depth = previous_depth + 1;
                size_t const previous_from = previous_depth > bound ? previous_depth - bound : 0;
                size_t const previous_to = sz_min_of_two(query.size(), previous_depth + bound);
                size_t const current_from = current_depth > bound ? current_depth - bound : 0;
                size_t const current_to = sz_min_of_two(query.size(), current_depth + bound);
                u16_t const *previous_row = scratch.trie_rows.data() + previous_depth * stride;
                u16_t *current_row = scratch.trie_rows.data() + current_depth * stride;
                auto const previous_at = [&](size_t column) noexcept -> u16_t {
                    return column >= previous_from && column <= previous_to ? previous_row[column - previous_from]
                                                                            : cap;
                };
                row_min = cap;
                terminal_distance = cap;
                for (size_t column = current_from; column <= current_to; ++column) {
                    u16_t value;
                    if (column == 0) value = static_cast<u16_t>(sz_min_of_two(current_depth, size_t(cap)));
                    else {
                        unsigned const deletion = unsigned(previous_at(column)) + 1;
                        unsigned const insertion =
                            column > current_from ? unsigned(current_row[column - current_from - 1]) + 1 : cap;
                        unsigned const substitution =
                            unsigned(previous_at(column - 1)) +
                            (query[column - 1] != tape_[edge.label_offset + edge_offset]);
                        value = static_cast<u16_t>(sz_min_of_two(
                            std::min({deletion, insertion, substitution}), unsigned(cap)));
                    }
                    current_row[column - current_from] = value;
                    row_min = sz_min_of_two(row_min, value);
                    if (column == query.size()) terminal_distance = value;
                }
                if (row_min > bound) {
                    alive = false;
                    break;
                }
            }
            if (!alive) continue;
            if (terminal_distance <= bound)
                if (status_t status = emit_terminals(edge.child, terminal_distance);
                    status != status_t::success_k)
                    return status;
            if (row_min <= bound) {
                trie_node_t const &child = trie_nodes_[edge.child];
                if (scratch.trie_frames.try_push_back(trie_frame_t {
                        edge.child, child.first_edge, child.first_edge + child.edges_count,
                        static_cast<u32_t>(frame.depth + edge.label_length)}) != status_t::success_k)
                    return status_t::bad_alloc_k;
            }
        }
        return status_t::success_k;
    }

    status_t build_directory_() noexcept {
        // The high hash bits select one sorted record range. Choose a dense offset array or a ranked bitmap from their
        // actual byte sizes. Small dictionaries then pack the remaining hash bits and ID into four-byte records.
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

    template <typename sequences_type_, typename build_scratch_type_>
    status_t build_(sequences_type_ const &dictionary, u8_t max_distance, size_t deletion_max_word_length,
                    build_scratch_type_ &scratch) noexcept {
        if (max_distance == std::numeric_limits<u8_t>::max()) return status_t::unexpected_dimensions_k;
        if (dictionary.size() > std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
        u8_t const indexed_distance = sz_min_of_two(max_distance, u8_t(2));
        if (deletion_max_word_length == automatic_deletion_max_word_length_k) {
            // Deletion records grow quadratically with word length at k=2. Keep them only when the complete dictionary
            // stays below a simple average budget. Otherwise the prefix tree becomes the main path.
            static constexpr size_t residuals_per_word_budget = 80;
            size_t residuals_budget = std::numeric_limits<size_t>::max();
            if (dictionary.size() <= residuals_budget / residuals_per_word_budget)
                residuals_budget = dictionary.size() * residuals_per_word_budget;
            size_t residuals_upper_bound = 0;
            bool residuals_budget_exceeded = false;
            for (size_t id = 0; id != dictionary.size(); ++id) {
                size_t word_records = 0;
                if (!residuals_upper_bound_(dictionary[id].size(), indexed_distance, word_records) ||
                    word_records > residuals_budget - residuals_upper_bound) {
                    residuals_budget_exceeded = true;
                    break;
                }
                residuals_upper_bound += word_records;
            }
            deletion_max_word_length = !residuals_budget_exceeded
                                           ? automatic_deletion_max_word_length_k
                                           : size_t(0);
        }
        deletion_max_word_length_ = deletion_max_word_length;
        fallback_words_count_ = 0;
        size_t tape_symbols = 0, records_upper_bound = 0;
        max_word_length_ = 0;
        for (size_t id = 0; id != dictionary.size(); ++id) {
            size_t const length = dictionary[id].size();
            if (!checked_add_(tape_symbols, length)) return status_t::overflow_risk_k;
            if (length <= deletion_max_word_length_) {
                size_t word_records = 0;
                if (!residuals_upper_bound_(length, indexed_distance, word_records) ||
                    !checked_add_(records_upper_bound, word_records))
                    return status_t::overflow_risk_k;
            }
            else ++fallback_words_count_;
            max_word_length_ = sz_max_of_two(max_word_length_, length);
        }
        if (records_upper_bound > std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
        if (tape_.try_resize(tape_symbols) != status_t::success_k ||
            offsets_.try_resize(dictionary.size() + 1) != status_t::success_k ||
            wide_records_.try_reserve(records_upper_bound) != status_t::success_k)
            return status_t::bad_alloc_k;

        size_t tape_offset = 0;
        for (u32_t id = 0; id != dictionary.size(); ++id) {
            auto const item = dictionary[id];
            span<symbol_t const> const word {item.data(), item.size()};
            offsets_[id] = static_cast<u64_t>(tape_offset);
            if (!word.empty())
                std::memcpy(tape_.data() + tape_offset, word.data(), word.size() * sizeof(symbol_t));
            tape_offset += word.size();
            if (word.size() <= deletion_max_word_length_) {
                if (status_t status = generate_residuals_(word, indexed_distance, scratch);
                    status != status_t::success_k)
                    return status;
                for (u32_t hash : scratch.residuals)
                    if (status_t status = wide_records_.try_push_back((u64_t(hash) << 32) | id);
                        status != status_t::success_k)
                        return status;
            }
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
        if (max_distance > 2 || fallback_words_count_)
            if (status_t status = build_trie_(max_distance <= 2); status != status_t::success_k) return status;
        max_distance_ = max_distance;
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
        friend class basic_levenshtein_index;
        vector_t<u32_t> generations;
        vector_t<u32_t> residuals;
        vector_t<u64_t> prefixes;
        vector_t<u64_t> powers;
        vector_t<u8_t> dp_rows;
        vector_t<u16_t> trie_rows;
        vector_t<trie_frame_t> trie_frames;
        vector_t<dfa_transition_t> dfa_transitions;
        vector_t<dfa_frame_t> dfa_frames;
        u32_t generation = 0;
        u32_t dfa_generation = 0;

      public:
        explicit scratch_t(allocator_t alloc = {}) noexcept
            : generations(alloc), residuals(alloc), prefixes(alloc), powers(alloc), dp_rows(alloc), trie_rows(alloc),
              trie_frames(alloc), dfa_transitions(alloc), dfa_frames(alloc) {}
    };

    using matches_t = vector_t<match_t>;

    explicit basic_levenshtein_index(allocator_t alloc = {}) noexcept
        : alloc_(alloc), tape_(alloc), offsets_(alloc), directory_(alloc), directory_words_(alloc),
          packed_records_(alloc), wide_records_(alloc), trie_nodes_(alloc), trie_edges_(alloc), trie_terminals_(alloc) {}

    /** @brief Copies and indexes a dictionary, choosing the deletion table or prefix tree from its expected size. */
    template <typename sequences_type_>
    status_t try_build(sequences_type_ const &dictionary, u8_t max_distance,
                       size_t deletion_max_word_length = automatic_deletion_max_word_length_k) noexcept {
        basic_levenshtein_index candidate {alloc_};
        scratch_t scratch {alloc_};
        if (status_t status = candidate.build_(dictionary, max_distance, deletion_max_word_length, scratch);
            status != status_t::success_k)
            return status;
        *this = std::move(candidate);
        return status_t::success_k;
    }

    /** @brief Finds every dictionary entry whose exact distance from @p query is at most @p bound. */
    status_t find(span<symbol_t const> query, u8_t bound, scratch_t &scratch, matches_t &matches) const noexcept {
        if (status_t status = matches.try_resize(0); status != status_t::success_k) return status;
        if (bound > max_distance_) return status_t::unexpected_dimensions_k;
        if (bound > 2) return find_trie_(query, bound, false, scratch, matches);
        if (!packed_records_.size() && !wide_records_.size())
            return fallback_words_count_ ? find_trie_(query, bound, true, scratch, matches) : status_t::success_k;
        if (scratch.generations.size() != size()) {
            if (status_t status = scratch.generations.try_resize(size()); status != status_t::success_k) return status;
            std::fill(scratch.generations.begin(), scratch.generations.end(), u32_t(0));
            scratch.generation = 0;
        }
        if (++scratch.generation == 0) {
            std::fill(scratch.generations.begin(), scratch.generations.end(), u32_t(0));
            scratch.generation = 1;
        }
        if (status_t status = generate_residuals_(query, bound, scratch); status != status_t::success_k) return status;

        for (u32_t hash : scratch.residuals) {
            size_t begin_offset = 0, end_offset = 0;
            if (!bucket_(hash, begin_offset, end_offset)) continue;
            if (!packed_records_.size()) {
                u64_t const key = u64_t(hash) << 32;
                u64_t const *record = std::lower_bound(wide_records_.begin() + begin_offset,
                                                       wide_records_.begin() + end_offset, key);
                u64_t const *const end = wide_records_.begin() + end_offset;
                for (; record != end && static_cast<u32_t>(*record >> 32) == hash; ++record) {
                    u32_t const id = static_cast<u32_t>(*record);
                    if (scratch.generations[id] == scratch.generation) continue;
                    scratch.generations[id] = scratch.generation;
                    u8_t const distance = verify_(id, query, bound, scratch);
                    if (distance <= bound)
                        if (status_t status = matches.try_push_back(match_t {id, distance});
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
                    if (scratch.generations[id] == scratch.generation) continue;
                    scratch.generations[id] = scratch.generation;
                    u8_t const distance = verify_(id, query, bound, scratch);
                    if (distance <= bound)
                        if (status_t status = matches.try_push_back(match_t {id, distance});
                            status != status_t::success_k)
                            return status;
                }
            }
        }
        if (fallback_words_count_)
            return find_trie_(query, bound, true, scratch, matches);
        return status_t::success_k;
    }

    size_t size() const noexcept { return offsets_.size() ? offsets_.size() - 1 : 0; }
    u8_t max_distance() const noexcept { return max_distance_; }
    size_t max_word_length() const noexcept { return max_word_length_; }
    size_t deletion_max_word_length() const noexcept { return deletion_max_word_length_; }
    bool uses_packed_records() const noexcept { return packed_records_.size() != 0 || size() == 0; }
    size_t records_count() const noexcept {
        return packed_records_.size() ? packed_records_.size() : wide_records_.size();
    }
    size_t index_bytes() const noexcept {
        return directory_.size() * sizeof(u32_t) + directory_words_.size() * sizeof(directory_word_t) +
               packed_records_.size() * sizeof(u32_t) + wide_records_.size() * sizeof(u64_t) + trie_bytes();
    }
    size_t dictionary_bytes() const noexcept {
        return tape_.size() * sizeof(symbol_t) + offsets_.size() * sizeof(u64_t);
    }
    size_t trie_bytes() const noexcept {
        return trie_nodes_.size() * sizeof(trie_node_t) + trie_edges_.size() * sizeof(trie_edge_t) +
               trie_terminals_.size() * sizeof(u32_t);
    }
};

template <typename allocator_type_ = std::allocator<char>>
using levenshtein_index = basic_levenshtein_index<char, allocator_type_>;

/** @brief Exact immutable Levenshtein retrieval over validated UTF-8 codepoints rather than encoded bytes. */
template <typename allocator_type_ = std::allocator<char>>
class levenshtein_index_utf8 {
  public:
    using allocator_t = allocator_type_;
    using rune_t = sz_rune_t;
    using index_t = basic_levenshtein_index<rune_t, allocator_t>;
    using match_t = typename index_t::match_t;
    using matches_t = typename index_t::matches_t;
    static constexpr size_t automatic_deletion_max_word_length_k = index_t::automatic_deletion_max_word_length_k;

  private:
    template <typename value_type_>
    using rebound_allocator_t = typename std::allocator_traits<allocator_t>::template rebind_alloc<value_type_>;
    template <typename value_type_>
    using vector_t = safe_vector<value_type_, rebound_allocator_t<value_type_>>;

    struct decoded_dictionary_t {
        struct word_t {
            rune_t const *data_ = nullptr;
            size_t size_ = 0;
            rune_t const *data() const noexcept { return data_; }
            size_t size() const noexcept { return size_; }
        };

        vector_t<rune_t> tape;
        vector_t<u64_t> offsets;

        explicit decoded_dictionary_t(allocator_t alloc) noexcept : tape(alloc), offsets(alloc) {}
        size_t size() const noexcept { return offsets.size() ? offsets.size() - 1 : 0; }
        word_t operator[](size_t id) const noexcept {
            size_t const begin = static_cast<size_t>(offsets[id]);
            size_t const end = static_cast<size_t>(offsets[id + 1]);
            return {tape.data() + begin, end - begin};
        }
    };

    allocator_t alloc_ {};
    index_t index_ {alloc_};

    static status_t decode_(span<char const> source, vector_t<rune_t> &destination) noexcept {
        if (status_t status = destination.try_resize(0); status != status_t::success_k) return status;
        if (status_t status = destination.try_reserve(source.size()); status != status_t::success_k) return status;
        if (source.empty()) return status_t::success_k;
        char const *cursor = source.data();
        char const *const end = cursor + source.size();
        while (cursor != end) {
            rune_t rune = 0;
            rune_length_t const length = sz_rune_decode(cursor, end, &rune);
            if (length == sz_rune_invalid_k) return status_t::invalid_utf8_k;
            if (status_t status = destination.try_push_back(rune); status != status_t::success_k) return status;
            cursor += length;
        }
        return status_t::success_k;
    }

  public:
    /** @brief Reusable query memory owned by one UTF-8 reader. */
    class scratch_t {
        friend class levenshtein_index_utf8;
        typename index_t::scratch_t index_scratch_;
        vector_t<rune_t> query_runes_;

      public:
        explicit scratch_t(allocator_t alloc = {}) noexcept : index_scratch_(alloc), query_runes_(alloc) {}
    };

    explicit levenshtein_index_utf8(allocator_t alloc = {}) noexcept : alloc_(alloc), index_(alloc) {}

    /** @brief Builds from validated UTF-8 strings and measures edits in Unicode codepoints. */
    template <typename sequences_type_>
    status_t try_build(sequences_type_ const &dictionary, u8_t max_distance,
                       size_t deletion_max_word_length = automatic_deletion_max_word_length_k) noexcept {
        if (dictionary.size() > std::numeric_limits<u32_t>::max()) return status_t::overflow_risk_k;
        decoded_dictionary_t decoded {alloc_};
        if (status_t status = decoded.offsets.try_resize(dictionary.size() + 1); status != status_t::success_k)
            return status;
        for (size_t id = 0; id != dictionary.size(); ++id) {
            auto const item = dictionary[id];
            if (decoded.tape.size() > std::numeric_limits<u64_t>::max()) return status_t::overflow_risk_k;
            decoded.offsets[id] = static_cast<u64_t>(decoded.tape.size());
            if (!item.size()) continue;
            char const *cursor = item.data();
            char const *const end = cursor + item.size();
            while (cursor != end) {
                rune_t rune = 0;
                rune_length_t const length = sz_rune_decode(cursor, end, &rune);
                if (length == sz_rune_invalid_k) return status_t::invalid_utf8_k;
                if (status_t status = decoded.tape.try_push_back(rune); status != status_t::success_k) return status;
                cursor += length;
            }
        }
        decoded.offsets[dictionary.size()] = static_cast<u64_t>(decoded.tape.size());
        return index_.try_build(decoded, max_distance, deletion_max_word_length);
    }

    /** @brief Finds all matches for a validated UTF-8 query and returns exact codepoint distances. */
    status_t find(span<char const> query, u8_t bound, scratch_t &scratch, matches_t &matches) const noexcept {
        if (status_t status = matches.try_resize(0); status != status_t::success_k) return status;
        if (status_t status = decode_(query, scratch.query_runes_); status != status_t::success_k) return status;
        return index_.find({scratch.query_runes_.data(), scratch.query_runes_.size()}, bound, scratch.index_scratch_,
                           matches);
    }

    size_t size() const noexcept { return index_.size(); }
    u8_t max_distance() const noexcept { return index_.max_distance(); }
    size_t max_word_length() const noexcept { return index_.max_word_length(); }
    size_t deletion_max_word_length() const noexcept { return index_.deletion_max_word_length(); }
    bool uses_packed_records() const noexcept { return index_.uses_packed_records(); }
    size_t records_count() const noexcept { return index_.records_count(); }
    size_t index_bytes() const noexcept { return index_.index_bytes(); }
    size_t dictionary_bytes() const noexcept { return index_.dictionary_bytes(); }
    size_t trie_bytes() const noexcept { return index_.trie_bytes(); }
};

} // namespace stringzillas
} // namespace ashvardanian

#endif // STRINGZILLAS_LEVENSHTEIN_INDEX_HPP_
