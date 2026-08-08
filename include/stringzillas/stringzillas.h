/**
 *  @brief StringZillas is a collection of advanced string algorithms, designed to be used in Big Data applications.
 *         It is generally faster than LibC, and has a broader & cleaner interface for safer @b length-bounded strings.
 *         On modern CPUs it uses AVX2, AVX-512, NEON, SVE, & SVE2 @b SIMD instructions & provides SWAR for older CPUs.
 *         On @b CUDA-capable GPUs it also provides C++ kernels for bulk processing.
 *
 *  Unlike traditional StringZilla interfaces, all of the functions:
 *  - operators are stateful, and should be reused between calls;
 *  - receive large collections of string inputs instead of just one or two strings;
 *  - receive executors or thread pools to parallelize the work, targeting a fraction of CPU cores or a GPU;
 *  - support overriding a default memory allocator with a custom one, wrapped into `sz_memory_allocator_t`.
 *
 *  Under the hood, a ton of C++ templates are instantiated to handle different types of inputs, like:
 *  - `sz_sequence_t` - for a C-style `std::vector<std::string_view>`-like structure.
 *  - `sz_sequence_u32tape_t`, `sz_sequence_u64tape_t` - for Apache Arrow-like tapes with 32-bit and 64-bit offsets.
 *
 *  Those templates also reuse the same pre-configured operators for different thread-pool & executor types,
 *  hardware capability levels.
 *
 *  @file include/stringzillas/stringzillas.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLAS_H_
#define STRINGZILLAS_H_

#include <stringzilla/stringzilla.h> // `sz_sequence_t` and other types

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Get StringZillas major version number.
 *  @sa sz_version_major
 */
SZ_API_RUNTIME int szs_version_major(void);

/**
 *  @brief Get StringZillas minor version number.
 *  @sa sz_version_minor
 */
SZ_API_RUNTIME int szs_version_minor(void);

/**
 *  @brief Get StringZillas patch version number.
 *  @sa sz_version_patch
 */
SZ_API_RUNTIME int szs_version_patch(void);

/**
 *  @brief What this binary @b ships: StringZilla's CPU kernels, the Fork Union pool, and the GPU tiers
 *         the CUDA toolkit was able to compile.
 *  @sa sz_capabilities_comptime
 */
SZ_API_RUNTIME sz_capability_t szs_capabilities_comptime(void);

/**
 *  @brief What this machine @b offers: the CPU's ISA, its usable core count, and the first GPU's tier.
 *  @note A GPU that cannot be enumerated - for any driver reason at all - simply contributes no GPU
 *        bits. It never speaks for the CPU, so the answer degrades instead of collapsing.
 *  @sa sz_capabilities_runtime
 */
SZ_API_RUNTIME sz_capability_t szs_capabilities_runtime(void);

/**
 *  @brief Get hardware capabilities mask for current system, as the intersection of what this binary
 *         ships and what this machine offers.
 *  @sa sz_capabilities
 */
SZ_API_RUNTIME sz_capability_t szs_capabilities(void);

/**
 *  @brief Apache Arrow-like tape for non-NULL strings with 32-bit offsets.
 *  @sa `sz_sequence_u64tape_t` for larger collections.
 *  @note Like Apache Arrow, we take (N+1) offsets for (N) strings, where `lengths[i] = offsets[i] - offsets[i-1]`.
 */
typedef struct sz_sequence_u32tape_t {
    sz_cptr_t data;
    sz_u32_t const *offsets;
    sz_size_t count;
} sz_sequence_u32tape_t;

/**
 *  @brief Apache Arrow-like tape for non-NULL strings with 64-bit offsets.
 *  @sa `sz_sequence_u32tape_t` for smaller space-efficient collections.
 *  @note Like Apache Arrow, we take (N+1) offsets for (N) strings, where `lengths[i] = offsets[i] - offsets[i-1]`.
 */
typedef struct sz_sequence_u64tape_t {
    sz_cptr_t data;
    sz_u64_t const *offsets;
    sz_size_t count;
} sz_sequence_u64tape_t;

/**
 *  @brief Prepares the default allocator for unified memory management.
 *  @param[out] error_message Optional output pointer for detailed error information.
 *  @note When compiled on CUDA-capable systems, this function will use `cudaMallocManaged`.
 */
SZ_API_RUNTIME sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc, char const **error_message);

/**
 *  Doesn't aim to provide the same level of granularity as the C++ API.
 *  It expects that the C functions will be called in bulk, generally,
 *  by just a single caller, either targeting:
 *
 *  - a single CPU core,
 *  - a fraction of CPU cores through some global thread pool,
 *  - a single GPU device.
 *
 *  Set `cpu_cores` to 0 to target all available CPU cores, to -1 to avoid CPUs, to 1 to use only calling thread.
 *  Set `gpu_device` to -1 to avoid GPUs, or to a positive device ID to target a specific GPU.
 */
typedef void *szs_device_scope_t;

/**
 * @brief Initialize device scope with system defaults.
 * @param[out] scope Pointer to device scope handle.
 * @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_device_scope_init_default(szs_device_scope_t *scope, char const **error_message);

/**
 * @brief Initialize device scope for CPU parallel execution.
 * @param[in] cpu_cores Number of CPU cores to use, or zero for all cores.
 * @param[out] scope Pointer to device scope handle.
 * @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_device_scope_init_cpu_cores(sz_size_t cpu_cores, szs_device_scope_t *scope,
                                                           char const **error_message);

/**
 * @brief Initialize device scope for GPU execution.
 * @param[in] gpu_device GPU device index to target.
 * @param[out] scope Pointer to device scope handle.
 * @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_device_scope_init_gpu_device(sz_size_t gpu_device, szs_device_scope_t *scope,
                                                            char const **error_message);

/**
 * @brief Query configured CPU cores count.
 * @param[in] scope Device scope handle.
 * @param[out] cpu_cores Number of CPU cores configured.
 * @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_device_scope_get_cpu_cores(szs_device_scope_t scope, sz_size_t *cpu_cores,
                                                          char const **error_message);

/**
 * @brief Query configured GPU device ID.
 * @param[in] scope Device scope handle.
 * @param[out] gpu_device GPU device index.
 * @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_device_scope_get_gpu_device(szs_device_scope_t scope, sz_size_t *gpu_device,
                                                           char const **error_message);

/**
 * @brief Get device scope hardware capabilities.
 * @param[in] scope Device scope handle.
 * @param[out] capabilities Hardware capabilities mask.
 * @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_device_scope_get_capabilities(szs_device_scope_t scope, sz_capability_t *capabilities,
                                                             char const **error_message);

/**
 * @brief Free device scope resources.
 * @param[in] scope Device scope handle to free.
 */
SZ_API_RUNTIME void szs_device_scope_free(szs_device_scope_t scope);

/*  APIs for computing edit-distances between binary and UTF-8 strings as a cross-product matrix.
 *  Each call scores every `queries[query_index]` against every `candidates[candidate_index]` and writes the result
 *  to `results[query_index * results_row_stride + candidate_index]`. Passing `candidates == NULL` requests symmetric
 *  self-similarity of `queries` (the lower triangle is computed and mirrored, `rows == columns`).
 *  Supports `sz_sequence_t`, `sz_sequence_u32tape_t`, and `sz_sequence_u64tape_t` inputs.
 */
typedef void *szs_levenshtein_distances_t;
typedef void *szs_levenshtein_distances_utf8_t;

/**
 *  @brief Initialize Levenshtein distance engine with affine gap costs.
 *
 *  Creates an engine for computing edit distances between binary sequences using
 *  the Wagner-Fischer dynamic programming algorithm with configurable costs.
 *
 *  @param[in] match Cost for character matches (typically negative or zero).
 *  @param[in] mismatch Cost for character mismatches (typically positive).
 *  @param[in] open Cost for opening a gap (typically positive).
 *  @param[in] extend Cost for extending an existing gap (typically smaller than open).
 *  @param[in] alloc Memory allocator (NULL for default).
 *  @param[in] capabilities Hardware capabilities mask.
 *  @param[out] engine Pointer to initialized engine handle.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_init(                                         //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_t *engine, char const **error_message);

/**
 *  @brief Compute the cross-product matrix of Levenshtein distances between two sequence collections.
 *  @param[in] engine Initialized distance engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence collection (matrix rows).
 *  @param[in] candidates Candidate sequence collection (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output distance matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances(              //
    szs_levenshtein_distances_t engine, szs_device_scope_t device, //
    sz_sequence_t const *queries, sz_sequence_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride,              //
    char const **error_message);

/**
 *  @brief Compute the cross-product matrix of Levenshtein distances for 32-bit tape format.
 *  @param[in] engine Initialized distance engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output distance matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_u32tape(                      //
    szs_levenshtein_distances_t engine, szs_device_scope_t device,                 //
    sz_sequence_u32tape_t const *queries, sz_sequence_u32tape_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride,                              //
    char const **error_message);

/**
 *  @brief Compute the cross-product matrix of Levenshtein distances for 64-bit tape format.
 *  @param[in] engine Initialized distance engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output distance matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_u64tape(                      //
    szs_levenshtein_distances_t engine, szs_device_scope_t device,                 //
    sz_sequence_u64tape_t const *queries, sz_sequence_u64tape_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride,                              //
    char const **error_message);

/**
 *  @brief Free Levenshtein distance engine resources.
 *  @param[in] engine Engine handle to free.
 */
SZ_API_RUNTIME void szs_levenshtein_distances_free(szs_levenshtein_distances_t engine);

/**
 *  @brief Initialize UTF-8 aware Levenshtein distance engine.
 *
 *  Creates an engine for computing edit distances between UTF-8 encoded strings
 *  using character-level comparison instead of byte-level.
 *
 *  @param[in] match Cost for character matches (typically negative or zero).
 *  @param[in] mismatch Cost for character mismatches (typically positive).
 *  @param[in] open Cost for opening a gap (typically positive).
 *  @param[in] extend Cost for extending an existing gap (typically smaller than open).
 *  @param[in] alloc Memory allocator (NULL for default).
 *  @param[in] capabilities Hardware capabilities mask.
 *  @param[out] engine Pointer to initialized engine handle.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_utf8_init(                                    //
    sz_error_cost_t match, sz_error_cost_t mismatch, sz_error_cost_t open, sz_error_cost_t extend, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,                              //
    szs_levenshtein_distances_utf8_t *engine, char const **error_message);

/**
 *  @brief Compute the cross-product matrix of UTF-8 aware Levenshtein distances between two collections.
 *  @param[in] engine Initialized UTF-8 distance engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence collection (matrix rows).
 *  @param[in] candidates Candidate sequence collection (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output distance matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_utf8(              //
    szs_levenshtein_distances_utf8_t engine, szs_device_scope_t device, //
    sz_sequence_t const *queries, sz_sequence_t const *candidates,      //
    sz_size_t *results, sz_size_t results_row_stride,                   //
    char const **error_message);

/**
 *  @brief Compute the cross-product matrix of UTF-8 aware distances for 32-bit tape format.
 *  @param[in] engine Initialized UTF-8 distance engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output distance matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_utf8_u32tape(                 //
    szs_levenshtein_distances_utf8_t engine, szs_device_scope_t device,            //
    sz_sequence_u32tape_t const *queries, sz_sequence_u32tape_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride,                              //
    char const **error_message);

/**
 *  @brief Compute the cross-product matrix of UTF-8 aware distances for 64-bit tape format.
 *  @param[in] engine Initialized UTF-8 distance engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output distance matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_levenshtein_distances_utf8_u64tape(                 //
    szs_levenshtein_distances_utf8_t engine, szs_device_scope_t device,            //
    sz_sequence_u64tape_t const *queries, sz_sequence_u64tape_t const *candidates, //
    sz_size_t *results, sz_size_t results_row_stride,                              //
    char const **error_message);

/**
 *  @brief Free UTF-8 Levenshtein distance engine resources.
 *  @param[in] engine Engine handle to free.
 */
SZ_API_RUNTIME void szs_levenshtein_distances_utf8_free(szs_levenshtein_distances_utf8_t engine);

/*  APIs for computing similarity scores between two string collections as a cross-product matrix.
 *  Each call scores every `queries[query_index]` against every `candidates[candidate_index]` and writes the result
 *  to `results[query_index * results_row_stride + candidate_index]`. Passing `candidates == NULL` requests symmetric
 *  self-similarity of `queries` (the lower triangle is computed and mirrored, `rows == columns`).
 *  Supports `sz_sequence_t`, `sz_sequence_u32tape_t`, and `sz_sequence_u64tape_t` inputs.
 */

typedef void *szs_needleman_wunsch_scores_t;
typedef void *szs_smith_waterman_scores_t;

/**
 *  @brief Initialize Needleman-Wunsch global alignment scorer.
 *
 *  Creates an engine for computing global alignment scores between sequences using
 *  the Needleman-Wunsch algorithm with a compact, class-based substitution matrix and gap costs.
 *
 *  @param[in] byte_to_class Array of 256 bytes mapping each input byte to one of 32 character classes.
 *  @param[in] class_substitution_costs Row-major 32x32 matrix of signed costs between character classes.
 *  @param[in] open Cost for opening a gap (typically positive).
 *  @param[in] extend Cost for extending an existing gap (typically smaller than open).
 *  @param[in] alloc Memory allocator (NULL for default).
 *  @param[in] capabilities Hardware capabilities mask.
 *  @param[out] engine Pointer to initialized engine handle.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_needleman_wunsch_scores_init(                       //
    sz_u8_t const *byte_to_class, sz_error_cost_t const *class_substitution_costs, //
    sz_error_cost_t open, sz_error_cost_t extend,                                  //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,              //
    szs_needleman_wunsch_scores_t *engine, char const **error_message);

/**
 *  @brief Calculate the cross-product matrix of Needleman-Wunsch global alignment scores.
 *  @param[in] engine Initialized global alignment engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence collection (matrix rows).
 *  @param[in] candidates Candidate sequence collection (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output score matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_needleman_wunsch_scores(              //
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device, //
    sz_sequence_t const *queries, sz_sequence_t const *candidates,   //
    sz_ssize_t *results, sz_size_t results_row_stride,               //
    char const **error_message);

/**
 *  @brief Calculate the cross-product matrix of global alignment scores for 32-bit tape format.
 *  @param[in] engine Initialized global alignment engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output score matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_needleman_wunsch_scores_u32tape(                    //
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device,               //
    sz_sequence_u32tape_t const *queries, sz_sequence_u32tape_t const *candidates, //
    sz_ssize_t *results, sz_size_t results_row_stride,                             //
    char const **error_message);

/**
 *  @brief Calculate the cross-product matrix of global alignment scores for 64-bit tape format.
 *  @param[in] engine Initialized global alignment engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output score matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_needleman_wunsch_scores_u64tape(                    //
    szs_needleman_wunsch_scores_t engine, szs_device_scope_t device,               //
    sz_sequence_u64tape_t const *queries, sz_sequence_u64tape_t const *candidates, //
    sz_ssize_t *results, sz_size_t results_row_stride,                             //
    char const **error_message);

/**
 *  @brief Free Needleman-Wunsch scorer resources.
 *  @param[in] engine Engine handle to free.
 */
SZ_API_RUNTIME void szs_needleman_wunsch_scores_free(szs_needleman_wunsch_scores_t engine);

/**
 *  @brief Initialize Smith-Waterman local alignment scorer.
 *
 *  Creates an engine for computing local alignment scores between sequences using
 *  the Smith-Waterman algorithm with a compact, class-based substitution matrix and gap costs.
 *
 *  @param[in] byte_to_class Array of 256 bytes mapping each input byte to one of 32 character classes.
 *  @param[in] class_substitution_costs Row-major 32x32 matrix of signed costs between character classes.
 *  @param[in] open Cost for opening a gap (typically positive).
 *  @param[in] extend Cost for extending an existing gap (typically smaller than open).
 *  @param[in] alloc Memory allocator (NULL for default).
 *  @param[in] capabilities Hardware capabilities mask.
 *  @param[out] engine Pointer to initialized engine handle.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_smith_waterman_scores_init(                         //
    sz_u8_t const *byte_to_class, sz_error_cost_t const *class_substitution_costs, //
    sz_error_cost_t open, sz_error_cost_t extend,                                  //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,              //
    szs_smith_waterman_scores_t *engine, char const **error_message);

/**
 *  @brief Calculate the cross-product matrix of Smith-Waterman local alignment scores.
 *  @param[in] engine Initialized local alignment engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence collection (matrix rows).
 *  @param[in] candidates Candidate sequence collection (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output score matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_smith_waterman_scores(              //
    szs_smith_waterman_scores_t engine, szs_device_scope_t device, //
    sz_sequence_t const *queries, sz_sequence_t const *candidates, //
    sz_ssize_t *results, sz_size_t results_row_stride,             //
    char const **error_message);

/**
 *  @brief Calculate the cross-product matrix of local alignment scores for 32-bit tape format.
 *  @param[in] engine Initialized local alignment engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output score matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_smith_waterman_scores_u32tape(                      //
    szs_smith_waterman_scores_t engine, szs_device_scope_t device,                 //
    sz_sequence_u32tape_t const *queries, sz_sequence_u32tape_t const *candidates, //
    sz_ssize_t *results, sz_size_t results_row_stride,                             //
    char const **error_message);

/**
 *  @brief Calculate the cross-product matrix of local alignment scores for 64-bit tape format.
 *  @param[in] engine Initialized local alignment engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] queries Query sequence tape (matrix rows).
 *  @param[in] candidates Candidate sequence tape (matrix columns); NULL requests symmetric self-similarity of @p queries.
 *  @param[out] results Output score matrix; cell `(query_index, candidate_index)` is at `results[query_index * results_row_stride + candidate_index]`.
 *  @param[in] results_row_stride Number of elements between consecutive query rows (>= candidate count).
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_smith_waterman_scores_u64tape(                      //
    szs_smith_waterman_scores_t engine, szs_device_scope_t device,                 //
    sz_sequence_u64tape_t const *queries, sz_sequence_u64tape_t const *candidates, //
    sz_ssize_t *results, sz_size_t results_row_stride,                             //
    char const **error_message);

/**
 *  @brief Free Smith-Waterman scorer resources.
 *  @param[in] engine Engine handle to free.
 */
SZ_API_RUNTIME void szs_smith_waterman_scores_free(szs_smith_waterman_scores_t engine);

/**
 *  APIs for computing fingerprints, Min-Hashes, and Count-Min-Sketches of binary and UTF-8 strings.
 *  Supports `sz_sequence_t`, `sz_sequence_u32tape_t`, and `sz_sequence_u64tape_t` inputs.
 *
 *  @section Speed Considerations
 *
 *  For each window width you should aim for a multiple of 64 dimensions. Rolling hashes with identical window widths
 *  will share the same memory access pattern and can be effectively parallelized. For each platform, different minimum
 *  dimensions are recommended:
 *
 *  - on AVX-512 capable CPUs, take at least 8 hash-functions of each width,
 *  - on AVX-512 capable CPUs with a physical 512-bit path, take 16 or more, to increase register utilization,
 *  - on Nvidia GPUs, take at least 32 hash-functions of each width, to activate all 32 threads in a warp.
 *  - on AMD GPUs, take at least 64 hash-functions of each width, to activate all 64 threads in a wave.
 *
 *  Assuming 64 is the smallest size saturating all platforms - its a great default.
 *
 *  Having too many dimensions is also a problem, as we'll end up with a ton of redundant compute.
 *  For short Tweet-sized strings of 256 characters, 64 dimensions of each of [3, 5, 7, 9] seem like a good default.
 *  For web packets, around 1 KB memory pages, 64 dimensions of each of [3, 4, 5, 7, 9, 11, 15, 31] are a good default.
 *  For longer strings, like 4 KB memory pages, we can aim for 128 dimensions of the same widths.
 */
typedef void *szs_fingerprints_t;
typedef void *szs_fingerprints_utf8_t;

/**
 *  @brief Initialize fingerprinting engine for Min-Hash computation.
 *
 *  Creates an engine for computing rolling hash fingerprints using multiple
 *  configurable window sizes and dimensions for efficient similarity detection.
 *
 *  @param[in] dimensions Total dimensions per fingerprint, ideally 1024 or a (64 * window_widths_count) multiple.
 *  @param[in] alphabet_size Size of the alphabet (256 for binary, 128 for ASCII, 4 for DNA, 22 for protein).
 *  @param[in] window_widths Array of window widths (NULL for defaults like [3, 4, 5, 7, 9, 11, 15, 31]).
 *  @param[in] window_widths_count Number of window widths in array (0 for defaults).
 *  @param[in] seed Reproducibility seed; every value derives independent per-dimension multipliers.
 *  @param[in] alloc Memory allocator (NULL for default).
 *  @param[in] capabilities Hardware capabilities mask.
 *  @param[out] engine Pointer to initialized engine handle.
 *  @param[out] error_message Optional output pointer for detailed error information.
 *  @note If alphabet_size is 0, defaults to 256. If window_widths is NULL, uses default widths.
 */
SZ_API_RUNTIME sz_status_t szs_fingerprints_init(                                 //
    sz_size_t dimensions, sz_size_t alphabet_size,                                //
    sz_size_t const *window_widths, sz_size_t window_widths_count, sz_u64_t seed, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,             //
    szs_fingerprints_t *engine, char const **error_message);

/**
 *  @brief Compute Min-Hash fingerprints for sequences.
 *  @param[in] engine Initialized fingerprinting engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] texts Input sequence collection.
 *  @param[out] min_hashes Output Min-Hash array.
 *  @param[in] min_hashes_stride Stride between hash results in bytes.
 *  @param[out] min_counts Output Count-Min-Sketch array.
 *  @param[in] min_counts_stride Stride between count results in bytes.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_fingerprints_sequence(     //
    szs_fingerprints_t engine, szs_device_scope_t device, //
    sz_sequence_t const *texts,                           //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,    //
    sz_u32_t *min_counts, sz_size_t min_counts_stride,    //
    char const **error_message);

/**
 *  @brief Compute Min-Hash fingerprints for 64-bit tape format.
 *  @param[in] engine Initialized fingerprinting engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] texts Input sequence tape.
 *  @param[out] min_hashes Output Min-Hash array.
 *  @param[in] min_hashes_stride Stride between hash results in bytes.
 *  @param[out] min_counts Output Count-Min-Sketch array.
 *  @param[in] min_counts_stride Stride between count results in bytes.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_fingerprints_u64tape(      //
    szs_fingerprints_t engine, szs_device_scope_t device, //
    sz_sequence_u64tape_t const *texts,                   //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,    //
    sz_u32_t *min_counts, sz_size_t min_counts_stride,    //
    char const **error_message);

/**
 *  @brief Compute Min-Hash fingerprints for 32-bit tape format.
 *  @param[in] engine Initialized fingerprinting engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] texts Input sequence tape.
 *  @param[out] min_hashes Output Min-Hash array.
 *  @param[in] min_hashes_stride Stride between hash results in bytes.
 *  @param[out] min_counts Output Count-Min-Sketch array.
 *  @param[in] min_counts_stride Stride between count results in bytes.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_fingerprints_u32tape(      //
    szs_fingerprints_t engine, szs_device_scope_t device, //
    sz_sequence_u32tape_t const *texts,                   //
    sz_u32_t *min_hashes, sz_size_t min_hashes_stride,    //
    sz_u32_t *min_counts, sz_size_t min_counts_stride,    //
    char const **error_message);

/**
 *  @brief Free fingerprinting engine resources.
 *  @param[in] engine Engine handle to free.
 */
SZ_API_RUNTIME void szs_fingerprints_free(szs_fingerprints_t engine);

/**
 *  @brief Multi-pattern search: one compiled dictionary of needles applied to many haystacks in one pass.
 *
 *  Compiles a needle set into an Aho-Corasick automaton once, then reuses it across every later call, so
 *  the build cost is paid per dictionary rather than per haystack. Every needle is tested against every
 *  haystack in a single walk, and all matches are reported, including overlapping ones.
 *
 *  Supports `sz_sequence_t`, `sz_sequence_u32tape_t`, and `sz_sequence_u64tape_t` inputs.
 *
 *  @section Case Sensitivity
 *
 *  `szs_find_many_cased_k` matches raw bytes and accepts any needle, including malformed UTF-8.
 *
 *  `szs_find_many_uncased_k` applies full Unicode case folding as defined by `CaseFolding.txt` status codes
 *  `C` and `F`, the same contract `sz_utf8_uncased_find` implements, so the two agree. Needles must be
 *  well-formed UTF-8 and are rejected with `sz_invalid_utf8_k` otherwise. Folding applies no normalization,
 *  so a precomposed character does not match its decomposed spelling; compose `sz_utf8_norm` ahead of the
 *  search when canonical equivalence is wanted.
 */
typedef void *szs_find_many_t;

/** @brief Whether a dictionary matches needles byte-for-byte or folds both sides to a shared case first. */
typedef enum szs_find_many_case_sensitivity_t {
    szs_find_many_cased_k = 0,   /**< Byte-exact matching; needles may be arbitrary bytes. */
    szs_find_many_uncased_k = 1, /**< Full Unicode case folding; needles must be valid UTF-8. */
} szs_find_many_case_sensitivity_t;

/**
 *  @brief One reported match, locating it by haystack, by needle, and by byte span.
 *
 *  Under case folding a needle's own byte length is not the length of every match - needle "k" matches both
 *  the 1-byte "k" and the 3-byte Kelvin sign - so the span is carried per match rather than looked up.
 */
typedef struct szs_find_many_match_t {
    sz_size_t haystack_index; /**< Which haystack the match was found in. */
    sz_size_t needle_index;   /**< Which needle matched. */
    sz_size_t byte_offset;    /**< Offset of the match within its haystack, in bytes. */
    sz_size_t byte_length;    /**< Length of the matched span, in bytes. */
} szs_find_many_match_t;

/**
 *  @brief Initialize a multi-pattern search engine from a set of needles.
 *  @param[in] needles Needle collection to compile into the automaton.
 *  @param[in] case_sensitivity Byte-exact or case-folded matching.
 *  @param[in] alloc Memory allocator (NULL for default).
 *  @param[in] capabilities Hardware capabilities mask.
 *  @param[out] engine Pointer to initialized engine handle.
 *  @param[out] error_message Optional output pointer for detailed error information.
 *  @retval `sz_invalid_utf8_k` A needle is not well-formed UTF-8 while folding is requested.
 *  @retval `sz_unexpected_dimensions_k` A needle is empty.
 */
SZ_API_RUNTIME sz_status_t szs_find_many_init(                                    //
    sz_sequence_t const *needles, szs_find_many_case_sensitivity_t case_sensitivity, //
    sz_memory_allocator_t const *alloc, sz_capability_t capabilities,             //
    szs_find_many_t *engine, char const **error_message);

/**
 *  @brief Count matches of every needle in every haystack.
 *  @param[in] engine Initialized search engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] haystacks Input haystack collection.
 *  @param[out] counts Output array of per-haystack match counts, one entry per haystack.
 *  @param[out] error_message Optional output pointer for detailed error information.
 */
SZ_API_RUNTIME sz_status_t szs_find_many_count(        //
    szs_find_many_t engine, szs_device_scope_t device, //
    sz_sequence_t const *haystacks, sz_size_t *counts, //
    char const **error_message);

/**
 *  @brief Locate matches of every needle in every haystack.
 *  @param[in] engine Initialized search engine.
 *  @param[in] device Device scope for execution.
 *  @param[in] haystacks Input haystack collection.
 *  @param[out] matches Output match array, filled in ascending haystack order.
 *  @param[in] matches_capacity Number of entries @p matches can hold.
 *  @param[out] matches_found Number of matches written.
 *  @param[out] error_message Optional output pointer for detailed error information.
 *  @retval `sz_unexpected_dimensions_k` @p matches_capacity is too small to hold every match.
 */
SZ_API_RUNTIME sz_status_t szs_find_many(                                                 //
    szs_find_many_t engine, szs_device_scope_t device,                                    //
    sz_sequence_t const *haystacks,                                                       //
    szs_find_many_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message);

/** @copydoc szs_find_many_count */
SZ_API_RUNTIME sz_status_t szs_find_many_count_u32tape(        //
    szs_find_many_t engine, szs_device_scope_t device,         //
    sz_sequence_u32tape_t const *haystacks, sz_size_t *counts, //
    char const **error_message);

/** @copydoc szs_find_many_count */
SZ_API_RUNTIME sz_status_t szs_find_many_count_u64tape(        //
    szs_find_many_t engine, szs_device_scope_t device,         //
    sz_sequence_u64tape_t const *haystacks, sz_size_t *counts, //
    char const **error_message);

/** @copydoc szs_find_many */
SZ_API_RUNTIME sz_status_t szs_find_many_u32tape(                                         //
    szs_find_many_t engine, szs_device_scope_t device,                                    //
    sz_sequence_u32tape_t const *haystacks,                                               //
    szs_find_many_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message);

/** @copydoc szs_find_many */
SZ_API_RUNTIME sz_status_t szs_find_many_u64tape(                                         //
    szs_find_many_t engine, szs_device_scope_t device,                                    //
    sz_sequence_u64tape_t const *haystacks,                                               //
    szs_find_many_match_t *matches, sz_size_t matches_capacity, sz_size_t *matches_found, //
    char const **error_message);

/**
 *  @brief Free multi-pattern search engine resources.
 *  @param[in] engine Engine handle to free.
 */
SZ_API_RUNTIME void szs_find_many_free(szs_find_many_t engine);

/**
 *  @brief Allocates memory using unified memory allocator.
 *  @param[in] size_bytes Number of bytes to allocate.
 *  @return Pointer to allocated memory, or NULL on failure.
 *
 *  Uses CUDA unified memory when available, falls back to malloc otherwise.
 *  Allocated memory can be accessed from both CPU and GPU when CUDA is available.
 */
SZ_API_RUNTIME void *szs_unified_alloc(sz_size_t size_bytes);

/**
 *  @brief Deallocates memory allocated by szs_unified_alloc.
 *  @param[in] ptr Pointer to memory to deallocate.
 *  @param[in] size_bytes Size of the allocation (for compatibility, may be ignored).
 */
SZ_API_RUNTIME void szs_unified_free(void *ptr, sz_size_t size_bytes);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // STRINGZILLAS_H_
