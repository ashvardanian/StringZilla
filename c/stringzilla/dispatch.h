/**
 *  @file c/stringzilla/dispatch.h
 *  @brief Shared dispatch table for the per-domain StringZilla shims.
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *
 *  The compiled StringZilla library is split into one translation unit per domain
 *  (`compare.c`, `memory.c`, `hash.c`, `find.c`, `sort.c`, `intersect.c`, `utf8_codepoints.c`,
 *  `utf8_tokens.c`, `utf8_words.c`, `utf8_graphemes.c`, `utf8_sentences.c`, `utf8_linewraps.c`,
 *  `utf8_uncased_fold.c`, `utf8_uncased.c`), so that touching one domain only recompiles
 *  that domain. Each TU includes only its own domain header, fills its slice of the shared
 *  `sz_dispatch_table` via `sz_dispatch_<domain>_update_`, and defines the `SZ_DYNAMIC` public
 *  wrappers that call through the table. The thin `runtime.c` owns the table definition and the
 *  one-time initialization.
 */
#ifndef STRINGZILLA_DISPATCH_H_
#define STRINGZILLA_DISPATCH_H_

// Overwrite `SZ_DYNAMIC_DISPATCH` before including StringZilla.
#ifdef SZ_DYNAMIC_DISPATCH
#undef SZ_DYNAMIC_DISPATCH
#endif
#define SZ_DYNAMIC_DISPATCH 1
#include <stringzilla/types.h> // Function-pointer typedefs, `sz_capability_t`, `SZ_USE_*`

// The dispatch table and per-domain updaters are shared across translation units,
// but must stay internal to the shared object to preserve the exported ABI.
#if defined(_MSC_VER)
#define SZ_DISPATCH_INTERNAL
#else
#define SZ_DISPATCH_INTERNAL __attribute__((visibility("hidden")))
#endif

typedef struct sz_implementations_t {
    sz_equal_t equal;
    sz_order_t order;

    sz_copy_t copy;
    sz_move_t move;
    sz_fill_t fill;
    sz_lookup_t lookup;

    sz_bytesum_t bytesum;
    sz_hash_t hash;
    sz_hash_multiseed_t hash_multiseed;
    sz_hash_state_init_t hash_state_init;
    sz_hash_state_update_t hash_state_update;
    sz_hash_state_digest_t hash_state_digest;
    sz_fill_random_t fill_random;

    sz_sha256_state_init_t sha256_state_init;
    sz_sha256_state_update_t sha256_state_update;
    sz_sha256_state_digest_t sha256_state_digest;

    sz_find_byte_t find_byte;
    sz_find_byte_t rfind_byte;
    sz_find_t find;
    sz_find_t rfind;
    sz_find_byteset_t find_byteset;
    sz_find_byteset_t rfind_byteset;

    sz_utf8_count_t utf8_count;
    sz_utf8_find_nth_t utf8_find_nth;
    sz_utf8_segmenter_t utf8_newlines;
    sz_utf8_segmenter_t utf8_whitespaces;
    sz_utf8_unpack_chunk_t utf8_unpack_chunk;

    sz_utf8_norm_t utf8_norm;
    sz_utf8_norm_violation_t utf8_norm_violation;
    sz_utf8_uncased_fold_t utf8_uncased_fold;
    sz_utf8_uncased_find_t utf8_uncased_find;

    sz_utf8_segmenter_t utf8_words;
    sz_utf8_segmenter_t utf8_graphemes;
    sz_utf8_segmenter_t utf8_sentences;
    sz_utf8_segmenter_t utf8_linewraps;
    sz_utf8_uncased_order_t utf8_uncased_order;

    sz_sequence_argsort_t sequence_argsort;
    sz_sequence_argsort_t sequence_argsort_utf8_uncased;
    sz_sequence_intersect_t sequence_intersect;

} sz_implementations_t;

/**
 *  @brief The global "virtual table" of supported backends, defined in `stringzilla.c`
 *         and populated by the per-domain updaters below.
 */
extern SZ_DISPATCH_INTERNAL sz_implementations_t sz_dispatch_table;

/*  Each updater fills only its own fields, defaulting to the serial backend and then
 *  overriding for the most capable enabled SIMD generation matching @p caps.
 */
SZ_DISPATCH_INTERNAL void sz_dispatch_compare_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_memory_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_hash_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_find_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_sort_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_intersect_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_codepoints_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_delimiters_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_words_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_graphemes_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_sentences_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_lines_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_uncased_fold_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_norm_update_(sz_capability_t caps);
SZ_DISPATCH_INTERNAL void sz_dispatch_utf8_uncased_update_(sz_capability_t caps);

#endif // STRINGZILLA_DISPATCH_H_
