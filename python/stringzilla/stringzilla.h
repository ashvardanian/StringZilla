/**
 *  @brief The shared types and forward declarations for the per-domain files.
 *  @file python/stringzilla/stringzilla.h
 *  @author Ash Vardanian
 *
 *  The `stringzilla` extension is split into one translation unit per domain - `memory.c`, `hash.c`,
 *  `cipher.c`, `find.c`, `compare.c`, `sort.c`, `intersect.c`, the `utf8_` files - alongside the CPython
 *  object-model files `file.c`, `str.c`, `strs.c` and the `shared.c` plumbing. This header carries
 *  everything more than one of those files touches: the `File`/`Str`/`Strs` struct layouts, the full
 *  `PyTypeObject` forward-declaration set the module-init file needs to build its type-registration table,
 *  and the per-interpreter free-list state. Structs read by a single translation unit live in that file
 *  instead.
 *
 *  The sibling `stringzillas` extension has its own private header, `stringzillas.h`. The two must not be
 *  mixed: both define `SZ_METHOD_FLAGS`, to different values, and `stringzillas` shadows this extension's
 *  capsule-exported function names as file-local pointers.
 *
 *  Not installed; private to this extension's build.
 */
#ifndef STRINGZILLA_PYTHON_STRINGZILLA_H_
#define STRINGZILLA_PYTHON_STRINGZILLA_H_

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>    // `O_RDNLY`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `stat`
#include <sys/types.h>
#endif

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <limits.h> // `SSIZE_MAX`
#include <unistd.h> // `ssize_t`
#endif

// It seems like some Python versions forget to include a header, so we should:
// https://github.com/ashvardanian/StringZilla/actions/runs/7706636733/job/21002535521
#ifndef SSIZE_MAX
#define SSIZE_MAX (SIZE_MAX / 2)
#endif

// Undefine _POSIX_C_SOURCE to avoid redefinition warning with Python headers
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#include <Python.h> // Core CPython interfaces

#include <errno.h>  // `errno`
#include <stdio.h>  // `fopen`
#include <stdlib.h> // `rand`, `srand`
#include <time.h>   // `time`

#include <stringzilla/stringzilla.h>

/// @brief  Fast-call + keywords, the calling convention every `Str_like_*`/`Strs_*` method uses.
#define SZ_METHOD_FLAGS METH_FASTCALL | METH_KEYWORDS

// strs.c
/** @brief  Reads the start pointer of the @p i -th element of a `Strs` handed as a raw `void const *`. */
extern sz_cptr_t Strs_get_start_(void const *handle, sz_size_t i);

/** @brief  Reads the byte length of the @p i -th element of a `Strs` handed as a raw `void const *`. */
extern sz_size_t Strs_get_length_(void const *handle, sz_size_t i);

/** @brief  Helper function to export a `Strs` or similar sequence objects into a `sz_sequence_t`. */
extern sz_bool_t sz_py_export_strings_as_sequence(PyObject *object, sz_sequence_t *sequence);

/** @brief  Helper function to export a `Strs` object into `sz_sequence_u32tape_t` components. */
extern sz_bool_t sz_py_export_strings_as_u32tape(PyObject *object, sz_cptr_t *data, sz_u32_t const **offsets,
                                                 sz_size_t *count);

/** @brief  Helper function to export a `Strs` object into `sz_sequence_u64tape_t` components. */
extern sz_bool_t sz_py_export_strings_as_u64tape(PyObject *object, sz_cptr_t *data, sz_u64_t const **offsets,
                                                 sz_size_t *count);

/** @brief  Helper function to replace the memory allocator in a `Strs` object. */
extern sz_bool_t sz_py_replace_strings_allocator(PyObject *object, sz_memory_allocator_t *allocator);

// shared.c
/** @brief  Whether a Python buffer-like object is writable; sets a `TypeError` and returns false if not. */
extern sz_bool_t sz_py_is_mutable(PyObject *object);

/** @brief  Helper function to export a Python string-like object into a `sz_string_view_t`. */
extern sz_bool_t sz_py_export_string_like(PyObject *object, sz_cptr_t *start, sz_size_t *length);

/**
 *  @brief  Parses an optional `start`/`end` index argument with CPython slice semantics.
 *  @return 1 on success (result written to @p result_out), 0 if a Python exception was set.
 */
extern int sz_py_export_optional_index(PyObject *index_obj, Py_ssize_t default_index, Py_ssize_t *result_out);

extern PyTypeObject FileType;
extern PyTypeObject StrType;
extern PyTypeObject StrsType;
extern PyTypeObject FindSplitsType;
extern PyTypeObject Utf8SplitNewlinesType;
extern PyTypeObject Utf8NewlinesType;
extern PyTypeObject Utf8SplitWhitespacesType;
extern PyTypeObject Utf8WhitespacesType;
extern PyTypeObject Utf8SplitDelimitersType;
extern PyTypeObject Utf8DelimitersType;
extern PyTypeObject Utf8WordbreaksType;
extern PyTypeObject Utf8GraphemesType;
extern PyTypeObject Utf8SentencesType;
extern PyTypeObject Utf8LinebreaksType;
extern PyTypeObject Utf8CodepointsType;
extern PyTypeObject Utf8UncasedMatchesType;
extern PyTypeObject HasherType;
extern PyTypeObject Sha256Type;
extern PyTypeObject Sha256sType;
extern PyTypeObject Aes256CtrKeyType;
extern PyTypeObject Aes256GcmKeyType;
extern PyTypeObject Aes256GcmEncryptorType;
extern PyTypeObject Aes256GcmDecryptorType;

extern struct PyModuleDef stringzilla_module;

/**
 *  @brief  Describes an on-disk file mapped into RAM, which is different from Python's
 *          native `mmap` module, as it exposes the address of the mapping in memory.
 */
typedef struct {
    PyObject ob_base;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int file_descriptor;
#endif
    sz_string_view_t memory;
} File;

/**
 *  @brief  Type-punned StringZilla-string, that points to a slice of an existing Python `str`
 *          or a `File`.
 *
 *  When a slice is constructed, the `parent` object's reference count is being incremented to preserve lifetime.
 *  It usage in Python would look like:
 *
 *      - Str() # Empty string
 *      - Str("some-string") # Full-range slice of a Python `str`
 *      - Str(File("some-path.txt")) # Full-range view of a persisted file
 *      - Str(File("some-path.txt"), from=0, to=sys.maxsize)
 */
typedef struct {
    PyObject ob_base;

    PyObject *parent;
    sz_string_view_t memory;
} Str;

/**
 *  @brief  Iterator for finding UAX segment boundaries in UTF-8 text - words, grapheme clusters,
 *          sentences, and line-break opportunities.
 *
 *  Streams segments by refilling a small inline buffer with @c kernel, yielding one segment per
 *  @c __next__. The buffer lives in the iterator itself - no extra allocation. @c start advances past
 *  each batch as it is consumed.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj; //< For reference counting

    sz_cptr_t start; //< Start of the not-yet-segmented suffix (a boundary); advances forward as batches drain.
    sz_cptr_t end;   //< End of the original text; immutable.

    sz_utf8_segmenter_t kernel; //< Segmenting kernel this iterator was constructed with.

    /// @brief  Inline batch of segment offsets relative to @c start, refilled on demand.
    sz_size_t batch_starts[sz_iterators_default_steps_k];
    sz_size_t batch_lengths[sz_iterators_default_steps_k];
    sz_size_t batch_count; //< Number of segments currently buffered.
    sz_size_t batch_index; //< Index of the next segment to yield from the buffer.

} Utf8Boundaries;

/**
 *  @brief  Variable length Python object similar to `Tuple[Union[Str, str]]`,
 *          for faster sorting, shuffling, joins, and lookups.
 */
typedef struct {
    PyObject ob_base;

    enum {
        STRS_U32_TAPE_VIEW = 0,
        STRS_U64_TAPE_VIEW = 1,
        STRS_U32_TAPE = 2,
        STRS_U64_TAPE = 3,
        STRS_FRAGMENTED = 4,
    } layout;

    union {
        /**
         *  U32 tape view - references existing Arrow array data, owns nothing.
         *  The layout is identical to Apache Arrow format: N+1 offsets for N strings.
         *  https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
         */
        struct u32_tape_view_t {
            sz_size_t count;
            sz_cptr_t data;    // Points to existing data (not owned)
            sz_u32_t *offsets; // Points to existing offsets (not owned)
            PyObject *parent;  // Parent Arrow array or other object
        } u32_tape_view;

        /**
         *  U32 tape - owns both offsets and data with custom allocator.
         */
        struct u32_tape_t {
            sz_size_t count;
            sz_cptr_t data;    // Owned data
            sz_u32_t *offsets; // Owned offsets (N+1 for N strings)
            sz_memory_allocator_t allocator;
        } u32_tape;

        /**
         *  U64 tape view - references existing Arrow array data, owns nothing.
         *  The layout is identical to Apache Arrow format: N+1 offsets for N strings.
         *  https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
         */
        struct u64_tape_view_t {
            sz_size_t count;
            sz_cptr_t data;    // Points to existing data (not owned)
            sz_u64_t *offsets; // Points to existing offsets (not owned)
            PyObject *parent;  // Parent Arrow array or other object
        } u64_tape_view;

        /**
         *  U64 tape - owns both offsets and data with custom allocator.
         */
        struct u64_tape_t {
            sz_size_t count;
            sz_cptr_t data;    // Owned data
            sz_u64_t *offsets; // Owned offsets (N+1 for N strings)
            sz_memory_allocator_t allocator;
        } u64_tape;

        /**
         *  Reordered subviews - owns only the array of individual spans.
         *  Each span points to data in the parent object.
         */
        struct fragmented_t {
            sz_size_t count;
            sz_string_view_t *spans; // Owned array of spans
            PyObject *parent;        // Parent object (Str, Strs, or other)
            sz_memory_allocator_t allocator;
        } fragmented;

    } data;

} Strs;

/**
 *  @brief  Per-interpreter module state, holding the intrusive free-lists for `Str` and `Strs`.
 *
 *  Both objects churn heavily: nearly every `split`/iterate element and every slice mints a fresh
 *  fixed-size header that is torn down moments later. Instead of round-tripping each header through
 *  `PyObject_Malloc`/`PyObject_Free`, dealloc parks the dead header on a singly-linked free-list and
 *  the allocation helper pops it back. The link is threaded through the dead object's own storage
 *  (`Str::parent`, `Strs::data`), so the state needs only a head pointer and a counter per type - no
 *  array. Living in module state keeps it per-interpreter, but on a free-threaded build the module
 *  state is shared by every thread in the interpreter, so all four fields are guarded by
 *  `freelist_lock`: without it, concurrent `alloc_`/`dealloc` calls race on the same linked list and
 *  can hand out one header to two live objects at once.
 */
enum { sz_freelist_capacity_k = 64 }; //< Headers retained per interpreter, per type.

typedef struct {
    Str *str_freelist_head;        //< Intrusive list of dead `Str` headers (link via `parent`).
    sz_size_t str_freelist_count;  //< Cached `Str` headers, never exceeds `sz_freelist_capacity_k`.
    Strs *strs_freelist_head;      //< Intrusive list of dead `Strs` headers (link via `data`).
    sz_size_t strs_freelist_count; //< Cached `Strs` headers, never exceeds `sz_freelist_capacity_k`.
#if defined(Py_GIL_DISABLED)
    PyMutex freelist_lock; //< Guards the four fields above; zero-initialized, so starts unlocked.
#endif
} stringzilla_state_t;

// shared.c
/** @brief Reach the per-interpreter free-list state, or @c NULL before registration / during teardown. */
extern stringzilla_state_t *stringzilla_state_(void);

/** @brief Allocate a blank @c Str header, reusing a cached one from the free-list when available. */
extern Str *Str_alloc_(void);

/** @brief Allocate a blank @c Strs header, reusing a cached one from the free-list when available. */
extern Strs *Strs_alloc_(void);

/**
 *  @brief  Allocates an empty `Strs` in the fragmented layout. Consolidates the count-zero
 *          initialization otherwise inlined across slicing, splitting, and reordering paths.
 *  @return A new empty `Strs`, or `NULL` with a Python exception set on allocation failure.
 */
extern Strs *strs_make_empty_fragmented_(void);

/** @brief  Helper function to wrap the current exception with a custom prefix message. */
extern void wrap_current_exception(sz_cptr_t comment);

/** @brief The dead `Strs` header's `data` union storage doubles as the intrusive `next` link. */
extern Strs **Strs_freelist_next_(Strs *node);

// strs.c
/** @brief Number of live elements in a `Strs` collection, regardless of layout. */
extern Py_ssize_t Strs_len(Strs *self);

// On a free-threaded build the module state above is shared by every thread in the interpreter, so
// the free-list head/count pairs need a lock around each read-modify-write; on a GIL build the GIL
// already serializes these calls, so the lock/unlock compile away to nothing.
#if defined(Py_GIL_DISABLED)
#define sz_freelist_lock_(state) PyMutex_Lock(&(state)->freelist_lock)
#define sz_freelist_unlock_(state) PyMutex_Unlock(&(state)->freelist_lock)
#else
#define sz_freelist_lock_(state) \
    do {                         \
    } while (0)
#define sz_freelist_unlock_(state) \
    do {                           \
    } while (0)
#endif

/**
 *  @brief  Cross-domain function/docstring declarations.
 *
 *  Every `Str_like_*`/`Strs_*` method and its `doc_*` docstring constant named by the `Str_methods[]`,
 *  `Strs_methods[]`, and `stringzilla_methods[]` `PyMethodDef` tables in `str.c`, `strs.c`, and
 *  `stringzilla.c`. Each symbol is visible here so those tables can reach the domain file that owns it.
 */

// memory.c
extern char const doc_translate[];
extern PyObject *Str_like_translate(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);

// hash.c
extern char const doc_like_hash[], doc_hash_multiseed[], doc_fill_random[], doc_random[], doc_like_bytesum[],
    doc_like_sha256[], doc_hmac_sha256[];
extern Py_hash_t Str_hash(Str *self);
extern PyObject *Str_like_hash(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple);
extern PyObject *Str_like_hash_multiseed(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple);
extern PyObject *Str_like_fill_random(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple);
extern PyObject *module_random(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple);
extern PyObject *Str_like_bytesum(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple);
extern PyObject *Str_like_sha256(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *hmac_sha256(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple);

// cipher.c
/** @brief  Exception type raised when an AEAD tag fails to authenticate a ciphertext. */
extern PyObject *AuthenticationErrorType;
extern char const doc_AuthenticationError[];

// compare.c
extern char const doc_like_equal[];
extern PyObject *Str_like_equal(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Str_richcompare(PyObject *self, PyObject *other, int op);

// find.c
extern char const doc_contains[], doc_find[], doc_index[], doc_rfind[], doc_rindex[], doc_partition[], doc_rpartition[],
    doc_count[], doc_startswith[], doc_endswith[], doc_find_first_of[], doc_find_first_not_of[], doc_find_last_of[],
    doc_find_last_not_of[], doc_count_byteset[], doc_split[], doc_rsplit[], doc_split_byteset[], doc_rsplit_byteset[],
    doc_split_iter[], doc_rsplit_iter[], doc_split_byteset_iter[], doc_rsplit_byteset_iter[], doc_lstrip[],
    doc_rstrip[], doc_strip[], doc_splitlines[];
extern int Str_in(Str *self, PyObject *needle_obj);
extern PyObject *Str_like_contains(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *Str_like_find(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple);
extern PyObject *Str_like_index(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Str_like_rfind(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Str_like_rindex(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *Str_like_partition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *Str_like_rpartition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple);
extern PyObject *Str_like_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Str_like_startswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple);
extern PyObject *Str_like_endswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *Str_like_find_first_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *Str_like_find_first_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);
extern PyObject *Str_like_find_last_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple);
extern PyObject *Str_like_find_last_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple);
extern PyObject *Str_like_count_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *Str_like_split(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Str_like_rsplit(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *Str_like_split_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *Str_like_rsplit_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple);
extern PyObject *Str_like_split_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple);
extern PyObject *Str_like_rsplit_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple);
extern PyObject *Str_like_split_byteset_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);
extern PyObject *Str_like_rsplit_byteset_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                              PyObject *args_names_tuple);
extern PyObject *Str_like_lstrip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *Str_like_rstrip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *Str_like_strip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Str_like_splitlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple);

// sort.c
extern char const doc_argsort[];
extern sz_status_t Strs_run_argsort_(sz_bool_t uncased, sz_sequence_t const *sequence, sz_sorted_idx_t *order,
                                     sz_size_t top, sz_bool_t reverse);
extern PyObject *Strs_argsort(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple);

// intersect.c
extern char const doc_Strs_intersect[];
extern PyObject *Strs_intersect(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);

// utf8_runes.c
extern char const doc_utf8_count[], doc_utf8_codepoints[];
extern PyObject *Str_like_utf8_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_codepoints(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple);

// utf8_tokens.c
extern char const doc_utf8_split_newlines[], doc_utf8_newlines[], doc_utf8_split_whitespaces[], doc_utf8_whitespaces[],
    doc_utf8_split_delimiters[], doc_utf8_delimiters[];
extern PyObject *Str_like_utf8_split_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                              PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_split_whitespaces(PyObject *self, PyObject *const *args,
                                                 Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_whitespaces(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_split_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                                PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple);

// utf8_boundaries.c
/** @brief  Builds a @c Utf8Boundaries iterator of @p type over @p text_obj, segmenting it with @p kernel. */
extern PyObject *Utf8Boundaries_make_(PyTypeObject *type, PyObject *text_obj, sz_utf8_segmenter_t kernel);
/** @brief  Yields the next segment as a @c Str view, refilling the inline batch when it drains. */
extern PyObject *Utf8Boundaries_next_(Utf8Boundaries *self);
/** @brief  Releases the reference to the segmented text and frees the iterator. */
extern void Utf8Boundaries_dealloc_(Utf8Boundaries *self);
/** @brief  Returns the iterator itself, as required by the Python iterator protocol. */
extern PyObject *Utf8Boundaries_iter_(PyObject *self);

// utf8_wordbreaks.c
extern char const doc_utf8_wordbreaks[];
extern PyObject *Str_like_utf8_wordbreaks(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple);

// utf8_graphemes.c
extern char const doc_utf8_graphemes[];
extern PyObject *Str_like_utf8_graphemes(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple);

// utf8_sentences.c
extern char const doc_utf8_sentences[];
extern PyObject *Str_like_utf8_sentences(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple);

// utf8_linebreaks.c
extern char const doc_utf8_linebreaks[];
extern PyObject *Str_like_utf8_linebreaks(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple);

// utf8_uncased_fold.c
extern char const doc_utf8_uncased_fold[];
extern PyObject *Str_like_utf8_uncased_fold(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);

// utf8_uncased.c
extern char const doc_utf8_uncased_search[], doc_utf8_uncased_order[], doc_utf8_uncased_matches[];
extern PyObject *Str_like_utf8_uncased_search(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                              PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_uncased_order(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_uncased_matches(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple);

// utf8_norm.c
extern char const doc_utf8_norm[], doc_utf8_find_denormalized[];
extern PyObject *Str_like_utf8_norm(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *Str_like_utf8_find_denormalized(PyObject *self, PyObject *const *args,
                                                 Py_ssize_t positional_args_count, PyObject *args_names_tuple);

// str.c
extern char const doc_offset_within[], doc_write_to[], doc_decode[];
extern PyObject *Str_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
extern PyObject *Str_like_decode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *Str_offset_within(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *Str_write_to(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple);

#endif // STRINGZILLA_PYTHON_STRINGZILLA_H_
