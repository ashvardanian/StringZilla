/**
 *  @file c/stringzillas/levenshtein_index.cuh
 *  @brief Exact immutable Levenshtein dictionary-index C shim.
 *  @author Guillaume de Rouville
 */
#ifndef STRINGZILLAS_SZS_LEVENSHTEIN_INDEX_CUH_
#define STRINGZILLAS_SZS_LEVENSHTEIN_INDEX_CUH_

#include "stringzillas.cuh"

#include <stringzillas/levenshtein_index.hpp>

#include <atomic>
#include <limits>
#include <new>
#include <type_traits>

/** Adapts StringZilla's C allocator to the allocator shape used by the C++ index. */
template <typename value_type_>
class levenshtein_index_allocator_t {
    template <typename>
    friend class levenshtein_index_allocator_t;

    sz_memory_allocator_t alloc_ {};

  public:
    using value_type = value_type_;
    using propagate_on_container_move_assignment = std::true_type;

    template <typename other_type_>
    struct rebind {
        using other = levenshtein_index_allocator_t<other_type_>;
    };

    levenshtein_index_allocator_t() noexcept { sz_memory_allocator_init_default(&alloc_); }
    explicit levenshtein_index_allocator_t(sz_memory_allocator_t const *alloc) noexcept {
        if (alloc) alloc_ = *alloc;
        else sz_memory_allocator_init_default(&alloc_);
    }
    template <typename other_type_>
    levenshtein_index_allocator_t(levenshtein_index_allocator_t<other_type_> const &other) noexcept
        : alloc_(other.alloc_) {}

    value_type *allocate(std::size_t count) noexcept {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(value_type)) return nullptr;
        return reinterpret_cast<value_type *>(alloc_.allocate(count * sizeof(value_type), alloc_.handle));
    }
    void deallocate(value_type *address, std::size_t count) noexcept {
        alloc_.free(address, count * sizeof(value_type), alloc_.handle);
    }

    template <typename other_type_>
    bool operator==(levenshtein_index_allocator_t<other_type_> const &other) const noexcept {
        return alloc_.allocate == other.alloc_.allocate && alloc_.free == other.alloc_.free &&
               alloc_.handle == other.alloc_.handle;
    }
    template <typename other_type_>
    bool operator!=(levenshtein_index_allocator_t<other_type_> const &other) const noexcept {
        return !(*this == other);
    }
};

using levenshtein_index_alloc_t = levenshtein_index_allocator_t<char>;
using levenshtein_index_t = szs::levenshtein_index<levenshtein_index_alloc_t>;
using levenshtein_index_utf8_t = szs::levenshtein_index_utf8<levenshtein_index_alloc_t>;

template <typename index_type_>
struct levenshtein_index_reader_t {
    using alloc_t = typename index_type_::allocator_t;
    typename index_type_::scratch_t scratch;
    typename index_type_::matches_t matches;

    explicit levenshtein_index_reader_t(alloc_t alloc = {}) noexcept : scratch(alloc), matches(alloc) {}
};

template <typename index_type_>
struct levenshtein_index_engine_t {
    using alloc_t = typename index_type_::allocator_t;
    using reader_t = levenshtein_index_reader_t<index_type_>;
    using readers_alloc_t = typename std::allocator_traits<alloc_t>::template rebind_alloc<reader_t>;

    alloc_t alloc;
    index_type_ index;
    szs::safe_vector<reader_t, readers_alloc_t> readers;
    sz_capability_t capabilities;

    levenshtein_index_engine_t(alloc_t alloc_arg, sz_capability_t capabilities_arg) noexcept
        : alloc(alloc_arg), index(alloc_arg), readers(readers_alloc_t {alloc_arg}), capabilities(capabilities_arg) {}

    sz::status_t prepare_readers(std::size_t count) noexcept {
        while (readers.size() < count) {
            if (sz::status_t status = readers.try_push_back(reader_t {alloc}); status != sz::status_t::success_k)
                return status;
        }
        return sz::status_t::success_k;
    }
};

using levenshtein_index_engine_bytes_t = levenshtein_index_engine_t<levenshtein_index_t>;
using levenshtein_index_engine_utf8_t = levenshtein_index_engine_t<levenshtein_index_utf8_t>;

template <typename index_type_, typename dictionary_type_, typename index_handle_type_>
sz_status_t szs_levenshtein_index_init_(                                         //
    dictionary_type_ const &dictionary, sz_size_t max_distance,                  //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,            //
    index_handle_type_ *index_punned, char const **error_message) noexcept {

    using engine_t = levenshtein_index_engine_t<index_type_>;
    sz_assert_(index_punned != nullptr && *index_punned == nullptr && "Index must be uninitialized");
    if (max_distance >= std::numeric_limits<sz::u8_t>::max())
        return propagate_error(sz::status_t::unexpected_dimensions_k, error_message,
                               "Maximum Levenshtein distance must fit below 255");
    capabilities = static_cast<sz_capability_t>(capabilities & sz_caps_sp_k);
    if (!(capabilities & sz_cap_serial_k))
        return propagate_error(sz::status_t::device_code_mismatch_k, error_message,
                               "Levenshtein index requires the serial CPU capability");

    levenshtein_index_alloc_t cpp_alloc {alloc};
    auto *engine = new (std::nothrow) engine_t {cpp_alloc, capabilities};
    if (!engine)
        return propagate_error(sz::status_t::bad_alloc_k, error_message,
                               "Failed to allocate Levenshtein dictionary index");
    sz::status_t const status = engine->index.try_build(dictionary, static_cast<sz::u8_t>(max_distance));
    if (status != sz::status_t::success_k) {
        delete engine;
        return propagate_error(status, error_message, "Failed to build Levenshtein dictionary index");
    }
    *index_punned = reinterpret_cast<index_handle_type_>(engine);
    return propagate_error(sz::status_t::success_k, error_message);
}

template <typename engine_type_, typename queries_type_, typename executor_type_>
sz::status_t szs_levenshtein_index_find_with_(                                 //
    engine_type_ &engine, queries_type_ const &queries, sz_size_t bound,        //
    sz_u64_t *query_indices, sz_u32_t *dictionary_indices, sz_u8_t *distances,  //
    sz_size_t matches_capacity, sz_size_t &matches_found,                       //
    executor_type_ &executor) noexcept {

    matches_found = 0;
    if (bound >= std::numeric_limits<sz::u8_t>::max() || bound > engine.index.max_distance())
        return sz::status_t::unexpected_dimensions_k;
    if (matches_capacity && (!query_indices || !dictionary_indices || !distances))
        return sz::status_t::unexpected_dimensions_k;
    if (queries.size() && engine.index.size() > std::numeric_limits<sz_size_t>::max() / queries.size())
        return sz::status_t::overflow_risk_k;
    if (sz::status_t status = engine.prepare_readers(executor.threads_count()); status != sz::status_t::success_k)
        return status;

    // Keep the default one-core scope free of atomics and scheduler callbacks. Besides being the common C and
    // Python default, this is the clean baseline against indexed libraries whose public APIs are serial.
    if (executor.threads_count() == 1) {
        auto &reader = engine.readers[0];
        for (sz_size_t query_idx = 0; query_idx != queries.size(); ++query_idx) {
            auto const query = queries[query_idx];
            if (sz::status_t status = engine.index.find({query.data(), query.size()}, static_cast<sz::u8_t>(bound),
                                                         reader.scratch, reader.matches);
                status != sz::status_t::success_k) {
                matches_found = 0;
                return status;
            }
            sz_size_t const count = reader.matches.size();
            sz_size_t const output_offset = matches_found;
            matches_found += count;
            if (output_offset > matches_capacity || count > matches_capacity - output_offset) continue;
            for (sz_size_t match_idx = 0; match_idx != count; ++match_idx) {
                auto const &match = reader.matches[match_idx];
                query_indices[output_offset + match_idx] = static_cast<sz_u64_t>(query_idx);
                dictionary_indices[output_offset + match_idx] = match.id;
                distances[output_offset + match_idx] = match.distance;
            }
        }
        return matches_found > matches_capacity ? sz::status_t::unexpected_dimensions_k : sz::status_t::success_k;
    }

    std::atomic<sz_size_t> next_output {0};
    szs::atomic_status_t shared_status;
    executor.for_n_dynamic(queries.size(), [&](typename executor_type_::prong_t prong) noexcept {
        if (static_cast<sz::status_t>(shared_status) != sz::status_t::success_k) return;
        auto &reader = engine.readers[prong.thread];
        auto const query = queries[prong.task];
        sz::status_t const status = engine.index.find({query.data(), query.size()}, static_cast<sz::u8_t>(bound),
                                                       reader.scratch, reader.matches);
        if (status != sz::status_t::success_k) {
            shared_status = status;
            return;
        }

        sz_size_t const count = reader.matches.size();
        sz_size_t const output_offset = next_output.fetch_add(count, std::memory_order_relaxed);
        if (output_offset > matches_capacity || count > matches_capacity - output_offset) return;
        for (sz_size_t match_idx = 0; match_idx != count; ++match_idx) {
            auto const &match = reader.matches[match_idx];
            query_indices[output_offset + match_idx] = static_cast<sz_u64_t>(prong.task);
            dictionary_indices[output_offset + match_idx] = match.id;
            distances[output_offset + match_idx] = match.distance;
        }
    });

    if (static_cast<sz::status_t>(shared_status) != sz::status_t::success_k) return shared_status;
    matches_found = next_output.load(std::memory_order_relaxed);
    return matches_found > matches_capacity ? sz::status_t::unexpected_dimensions_k : sz::status_t::success_k;
}

template <typename engine_type_, typename queries_type_>
sz_status_t szs_levenshtein_index_find_(                                      //
    engine_type_ &engine, device_scope_t &device, queries_type_ const &queries, //
    sz_size_t bound, sz_u64_t *query_indices, sz_u32_t *dictionary_indices,     //
    sz_u8_t *distances, sz_size_t matches_capacity, sz_size_t *matches_found,   //
    char const **error_message) noexcept {

    sz_assert_(matches_found != nullptr && "Match count output must not be null");
    *matches_found = 0;
    sz::status_t const status = std::visit(
        [&](auto &scope) -> sz::status_t {
            using scope_t = std::decay_t<decltype(scope)>;
            if constexpr (!is_cpu_scope<scope_t>()) return sz::status_t::device_code_mismatch_k;
            else {
                constexpr bool is_parallel_k = std::is_same<scope_t, cpu_scope_t>::value;
                if (is_parallel_k && !(engine.capabilities & sz_cap_parallel_k))
                    return sz::status_t::device_code_mismatch_k;
                auto &&executor = get_executor(scope);
                return szs_levenshtein_index_find_with_(engine, queries, bound, query_indices,
                                                        dictionary_indices, distances, matches_capacity,
                                                        *matches_found, executor);
            }
        },
        device.variants);
    if (status == sz::status_t::unexpected_dimensions_k && *matches_found > matches_capacity)
        return propagate_error(status, error_message, "Levenshtein match output is too small");
    return propagate_error(status, error_message);
}

extern "C" {

#define SZS_LEVENSHTEIN_INDEX_INIT_(function_name, handle_type, index_type, dictionary_type, wrapper_type)            \
    SZ_API_RUNTIME sz_status_t function_name(dictionary_type const *dictionary, sz_size_t max_distance,              \
                                             sz_memory_allocator_t const *alloc, sz_capability_t capabilities,        \
                                             handle_type *index, char const **error_message) {                        \
        sz_assert_(dictionary != nullptr && "Dictionary must not be null");                                           \
        return szs_levenshtein_index_init_<index_type>(wrapper_type {dictionary}, max_distance, alloc, capabilities,  \
                                                       index, error_message);                                         \
    }

SZS_LEVENSHTEIN_INDEX_INIT_(szs_levenshtein_index_init, szs_levenshtein_index_t, levenshtein_index_t,
                            sz_sequence_t, sz_sequence_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_INIT_(szs_levenshtein_index_init_u32tape, szs_levenshtein_index_t, levenshtein_index_t,
                            sz_sequence_u32tape_t, sz_sequence_u32tape_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_INIT_(szs_levenshtein_index_init_u64tape, szs_levenshtein_index_t, levenshtein_index_t,
                            sz_sequence_u64tape_t, sz_sequence_u64tape_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_INIT_(szs_levenshtein_index_utf8_init, szs_levenshtein_index_utf8_t,
                            levenshtein_index_utf8_t, sz_sequence_t, sz_sequence_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_INIT_(szs_levenshtein_index_utf8_init_u32tape, szs_levenshtein_index_utf8_t,
                            levenshtein_index_utf8_t, sz_sequence_u32tape_t,
                            sz_sequence_u32tape_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_INIT_(szs_levenshtein_index_utf8_init_u64tape, szs_levenshtein_index_utf8_t,
                            levenshtein_index_utf8_t, sz_sequence_u64tape_t,
                            sz_sequence_u64tape_as_cpp_container_t)

#undef SZS_LEVENSHTEIN_INDEX_INIT_

#define SZS_LEVENSHTEIN_INDEX_FIND_(function_name, handle_type, engine_type, queries_type, wrapper_type)              \
    SZ_API_RUNTIME sz_status_t function_name(                                                                         \
        handle_type index, szs_device_scope_t device, queries_type const *queries, sz_size_t bound,                  \
        sz_u64_t *query_indices, sz_u32_t *dictionary_indices, sz_u8_t *distances, sz_size_t matches_capacity,       \
        sz_size_t *matches_found, char const **error_message) {                                                       \
        sz_assert_(index != nullptr && "Index must be initialized");                                                  \
        sz_assert_(device != nullptr && "Device scope must be initialized");                                          \
        sz_assert_(queries != nullptr && "Queries must not be null");                                                 \
        auto &engine = *reinterpret_cast<engine_type *>(index);                                                       \
        auto &scope = *reinterpret_cast<device_scope_t *>(device);                                                    \
        return szs_levenshtein_index_find_(engine, scope, wrapper_type {queries}, bound, query_indices,               \
                                           dictionary_indices, distances, matches_capacity, matches_found,           \
                                           error_message);                                                            \
    }

SZS_LEVENSHTEIN_INDEX_FIND_(szs_levenshtein_index_find, szs_levenshtein_index_t,
                            levenshtein_index_engine_bytes_t, sz_sequence_t, sz_sequence_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_FIND_(szs_levenshtein_index_find_u32tape, szs_levenshtein_index_t,
                            levenshtein_index_engine_bytes_t, sz_sequence_u32tape_t,
                            sz_sequence_u32tape_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_FIND_(szs_levenshtein_index_find_u64tape, szs_levenshtein_index_t,
                            levenshtein_index_engine_bytes_t, sz_sequence_u64tape_t,
                            sz_sequence_u64tape_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_FIND_(szs_levenshtein_index_utf8_find, szs_levenshtein_index_utf8_t,
                            levenshtein_index_engine_utf8_t, sz_sequence_t, sz_sequence_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_FIND_(szs_levenshtein_index_utf8_find_u32tape, szs_levenshtein_index_utf8_t,
                            levenshtein_index_engine_utf8_t, sz_sequence_u32tape_t,
                            sz_sequence_u32tape_as_cpp_container_t)
SZS_LEVENSHTEIN_INDEX_FIND_(szs_levenshtein_index_utf8_find_u64tape, szs_levenshtein_index_utf8_t,
                            levenshtein_index_engine_utf8_t, sz_sequence_u64tape_t,
                            sz_sequence_u64tape_as_cpp_container_t)

#undef SZS_LEVENSHTEIN_INDEX_FIND_

SZ_API_RUNTIME void szs_levenshtein_index_free(szs_levenshtein_index_t index) {
    sz_assert_(index != nullptr && "Index must be initialized");
    delete reinterpret_cast<levenshtein_index_engine_bytes_t *>(index);
}

SZ_API_RUNTIME void szs_levenshtein_index_utf8_free(szs_levenshtein_index_utf8_t index) {
    sz_assert_(index != nullptr && "Index must be initialized");
    delete reinterpret_cast<levenshtein_index_engine_utf8_t *>(index);
}

} // extern "C"

#endif // STRINGZILLAS_SZS_LEVENSHTEIN_INDEX_CUH_
