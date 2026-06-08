/**
 *  @file include/stringzillas/types.h
 *  @brief Shared plain-C PODs for the StringZillas engines & device scopes.
 *  @author Ash Vardanian
 *
 *  StringZillas is the parallel CPU + CUDA layer, mirroring the C core's flat dispatch-table
 *  architecture. This header declares the tagged engine PODs that each opaque `szs_*_t` handle points
 *  at, the device-scope POD behind `szs_device_scope_t`, and the compact substitution-cost LUT.
 *
 *  Keep every field tersely commented. The resolved backend is stored as a `sz_capability_t` (the same
 *  enum the C core dispatches on), not a parallel type.
 */
#ifndef STRINGZILLAS_TYPES_H_
#define STRINGZILLAS_TYPES_H_

#include <stringzilla/types.h> // `sz_error_cost_t`, `sz_capability_t`, `sz_memory_allocator_t`, similarity enums

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Affine gap-cost pair; linear gaps are encoded as @c open == @c extend. */
typedef struct szs_gap_costs_t {
    sz_error_cost_t open;   ///< Cost charged to open a new gap.
    sz_error_cost_t extend; ///< Cost charged to extend an existing gap.
} szs_gap_costs_t;

/** @brief Uniform match/mismatch costs for Levenshtein-style distances (no substitution table). */
typedef struct szs_uniform_costs_t {
    sz_error_cost_t match;    ///< Cost for a character match (typically 0).
    sz_error_cost_t mismatch; ///< Cost for a character mismatch (typically 1).
} szs_uniform_costs_t;

/** @brief Maximum number of equivalence classes in the small substitution LUT (protein ~25, DNA+IUPAC ~15). */
#define SZ_SUBS_MAX_CLASSES 32

/**
 *  @brief Compact substitution cost model for Needleman-Wunsch & Smith-Waterman.
 *
 *  A small per-class table (~1.3KB) that fits in L1 / SMEM / a single AVX-512 register row, enabling
 *  `vpermb`-based register permutes on CPU and a kernel-arg-staged-into-SMEM path on GPU (instead of a
 *  divergent constant-memory lookup). Each byte is mapped to a class via @c byte_to_class, then the
 *  cost is @c costs[class_a][class_b].
 */
typedef struct sz_substitution_costs_t {
    sz_u8_t alphabet_size;      ///< Number of live classes in use (<= SZ_SUBS_MAX_CLASSES).
    sz_u8_t unknown_class;      ///< Class index assigned to bytes outside the alphabet.
    sz_error_cost_t gap_open;   ///< Whole cost model travels as one object: gap-open cost.
    sz_error_cost_t gap_extend; ///< Gap-extend cost (== gap_open for linear gaps).
    sz_error_cost_t costs[SZ_SUBS_MAX_CLASSES][SZ_SUBS_MAX_CLASSES]; ///< 1KB class x class cost matrix (i8 cells).
    sz_u8_t byte_to_class[256]; ///< 256B remap from raw byte to class index (replaces the `A - 65` hack).
} sz_substitution_costs_t;

/** @brief How the input collections behind a `szs_sequence_view_t` are encoded. */
typedef enum szs_inputs_kind_t {
    szs_inputs_sequence_k = 0, ///< `sz_sequence_t const *` (vector of string views).
    szs_inputs_u32tape_k,      ///< `sz_sequence_u32tape_t const *` (Arrow-like 32-bit offset tape).
    szs_inputs_u64tape_k,      ///< `sz_sequence_u64tape_t const *` (Arrow-like 64-bit offset tape).
} szs_inputs_kind_t;

/*  Tape structs are defined in `stringzillas.h` (included after this header); forward-declare the tags so
 *  the typed view below can hold pointers to them without a typedef redefinition (illegal in C99). */
struct sz_sequence_u32tape_t;
struct sz_sequence_u64tape_t;

/** @brief One input collection, in any of the three supported encodings (tagged by `szs_sequence_view_t::kind`). */
typedef union szs_sequence_any_t {
    sz_sequence_t const *sequence;               ///< Vector of string views.
    struct sz_sequence_u32tape_t const *u32tape; ///< Arrow-like 32-bit offset tape.
    struct sz_sequence_u64tape_t const *u64tape; ///< Arrow-like 64-bit offset tape.
} szs_sequence_any_t;

/**
 *  @brief Typed pair of input collections handed to a dispatch entry, replacing the old
 *         `void const *a, void const *b, szs_inputs_kind_t kind` triple.
 *  @note Single-input operations (fingerprints) populate @c first only; @c second is unused.
 */
typedef struct szs_sequence_view_t {
    szs_inputs_kind_t kind;    ///< Tags the active union member of @c first / @c second.
    szs_sequence_any_t first;  ///< First collection (the `a` / `texts` operand).
    szs_sequence_any_t second; ///< Second collection (the `b` operand); unused for single-input ops.
} szs_sequence_view_t;

/** @brief Which execution surface a device scope targets. */
typedef enum szs_scope_kind_t {
    szs_scope_default_k = 0, ///< Pick GPU if present, else CPU.
    szs_scope_cpu_k,         ///< Force CPU execution.
    szs_scope_gpu_k,         ///< Force GPU execution.
} szs_scope_kind_t;

/** @brief CPU execution scope; @c pool is an opaque fork_union thread pool (NULL => run serially). */
typedef struct szs_cpu_scope_t {
    void *pool;          ///< Opaque `fu_pool_t *`; NULL means the calling thread executes serially.
    sz_size_t cpu_cores; ///< Number of logical cores to spawn (0 => all available).
} szs_cpu_scope_t;

#if SZ_USE_CUDA
/** @brief GPU execution scope; @c stream is an opaque CUDA stream handle. */
typedef struct szs_gpu_scope_t {
    int device_id; ///< CUDA device ordinal to target.
    void *stream;  ///< Opaque `CUstream` / `cudaStream_t`; NULL => default stream.
} szs_gpu_scope_t;
#endif

/**
 *  @brief Plain-C POD behind the opaque `szs_device_scope_t` handle.
 *  @note The per-call entry switches on @c kind (default => GPU-if-present-else-CPU), replacing the
 *        old `holds_alternative` ladder.
 */
typedef struct szs_device_scope_impl_t {
    szs_scope_kind_t kind; ///< Tag selecting the active union member below.
    szs_cpu_scope_t cpu;   ///< CPU scope state (used when kind != gpu).
#if SZ_USE_CUDA
    szs_gpu_scope_t gpu; ///< GPU scope state (used when kind == gpu).
#endif
    sz_capability_t capabilities; ///< Hardware capabilities resolved for this scope.
} szs_device_scope_impl_t;

/**
 *  @brief Plain-C POD behind `szs_levenshtein_distances_t` & `szs_levenshtein_distances_utf8_t`.
 *  @note Uniform-cost model (no substitution table) to preserve the SIMD equality fast-path. The byte-vs-rune
 *        choice is owned by the dispatch entry, not stored here.
 */
typedef struct szs_levenshtein_engine_t {
    szs_uniform_costs_t substitution; ///< Match/mismatch costs.
    szs_gap_costs_t gaps;             ///< Gap open/extend costs.
    sz_similarity_gaps_t gap_mode;    ///< Linear vs affine gap recurrence.
    sz_capability_t capabilities;     ///< Hardware capabilities the engine may use.
    sz_memory_allocator_t alloc;      ///< Allocator for per-call scratch buffers.
} szs_levenshtein_engine_t;

/**
 *  @brief Plain-C POD behind `szs_needleman_wunsch_scores_t` (global, max-score alignment).
 *  @note Locality (global) & objective (maximize) are fixed by the operation and live in the entry, not here.
 */
typedef struct szs_needleman_wunsch_engine_t {
    sz_substitution_costs_t substitution; ///< Compact class-based substitution LUT.
    szs_gap_costs_t gaps;                 ///< Gap open/extend costs.
    sz_similarity_gaps_t gap_mode;        ///< Linear vs affine gap recurrence.
    sz_capability_t capabilities;         ///< Hardware capabilities the engine may use.
    sz_memory_allocator_t alloc;          ///< Allocator for per-call scratch buffers.
} szs_needleman_wunsch_engine_t;

/**
 *  @brief Plain-C POD behind `szs_smith_waterman_scores_t` (local, max-score alignment).
 *  @note Same field layout as `szs_needleman_wunsch_engine_t`, but a distinct type so the local-vs-global
 *        recurrence is selected by the operation's entry rather than a stored flag.
 */
typedef struct szs_smith_waterman_engine_t {
    sz_substitution_costs_t substitution; ///< Compact class-based substitution LUT.
    szs_gap_costs_t gaps;                 ///< Gap open/extend costs.
    sz_similarity_gaps_t gap_mode;        ///< Linear vs affine gap recurrence.
    sz_capability_t capabilities;         ///< Hardware capabilities the engine may use.
    sz_memory_allocator_t alloc;          ///< Allocator for per-call scratch buffers.
} szs_smith_waterman_engine_t;

/** @brief Maximum number of distinct rolling-hash window widths a fingerprint engine can carry. */
#define SZS_FINGERPRINTS_MAX_WINDOWS 32

/**
 *  @brief Plain-C POD behind `szs_fingerprints_t` & `szs_fingerprints_utf8_t`.
 *  @note Fingerprints have no gap/width/locality/objective axes; the SIMD backend is haswell/skylake.
 */
typedef struct szs_fingerprints_engine_t {
    sz_size_t dimensions;    ///< Total Min-Hash dimensions per fingerprint.
    sz_size_t alphabet_size; ///< Alphabet cardinality (256 binary, 4 DNA, 22 protein, ...).
    sz_size_t window_widths[SZS_FINGERPRINTS_MAX_WINDOWS]; ///< Rolling-hash window widths.
    sz_size_t window_widths_count;                         ///< Number of live entries in @c window_widths.
    sz_capability_t capabilities;                          ///< Hardware capabilities the engine may use.
    sz_memory_allocator_t alloc;                           ///< Allocator for per-call scratch buffers.
} szs_fingerprints_engine_t;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLAS_TYPES_H_
