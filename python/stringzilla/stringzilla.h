/**
 *  @file python/stringzilla/stringzilla.h
 *  @author Ash Vardanian
 *  @date September 2, 2023
 *  @brief The shared types and forward declarations for the per-domain files.
 *
 *  The @c stringzilla extension is split into one translation unit per domain - `strings.c`,
 *  `hash.c`, `cipher.c`, `find.c`, `compare.c`, `sort.c`, `intersect.c`, `levenshtein.c`,
 *  `overlap.c`, `substrings.c`, and the `utf8_*.c` files - alongside the CPython object-model files
 *  `file.c`, `string.c` and the `shared.c` plumbing. This header carries everything more than one
 *  of those files touches: the @c File, @c String, and @c Strings struct layouts, the full
 *  @c PyTypeObject forward-declaration set the module-init file needs to build its
 *  type-registration table, and the per-interpreter free-list state. Structs read by a single
 *  translation unit live in that file instead.
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

/**
 *  @brief Some Python versions forget to include a header, so define the limit here.
 *
 *  @see CI failure: https://github.com/ashvardanian/StringZilla/actions/runs/7706636733/job/21002535521
 */
#ifndef SSIZE_MAX
#define SSIZE_MAX (SIZE_MAX / 2)
#endif

/* Undefine @c _POSIX_C_SOURCE to avoid a redefinition warning with Python headers. */
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif
#include <Python.h> // Core CPython interfaces

#include <errno.h>  // `errno`
#include <stdio.h>  // `fopen`
#include <stdlib.h> // `rand`, `srand`
#include <time.h>   // `time`

#include <stringzilla/stringzilla.h>

/** Fast-call with keywords, the convention every `String_like_*` and `Strings_*` method uses. */
#define STRINGZILLA_METHOD_FLAGS METH_FASTCALL | METH_KEYWORDS

/* `strings.c` */

/** Views the tape of a @c Strings in place, or returns false for any other object. */
extern sz_bool_t sz_py_export_strings_as_view(PyObject *object, sz_strings_view_t *strings);

/** Exports the bytes, the 32-bit offsets and the count of a @c Strings tape with 32-bit offsets. */
extern sz_bool_t sz_py_export_strings_as_u32tape(PyObject *object, sz_cptr_t *data, sz_u32_t const **offsets,
                                                 sz_size_t *count);

/** Exports the bytes, the 64-bit offsets and the count of a @c Strings tape with 64-bit offsets. */
extern sz_bool_t sz_py_export_strings_as_u64tape(PyObject *object, sz_cptr_t *data, sz_u64_t const **offsets,
                                                 sz_size_t *count);

/** Gathers a @c Strings into a tape of blocks from @p allocator, unless it already owns one. */
extern sz_bool_t sz_py_replace_strings_allocator(PyObject *object, sz_allocator_t *allocator);

/**
 *  @brief Views a @c Strings in place, or any other iterable of strings through @p views of the
 *      items a @p snapshot tuple pins, which the caller releases once done reading @p strings.
 *  @return 0, or -1 with a Python exception set that names @p context and nothing left to release.
 */
extern int sz_py_export_strings_like(PyObject *sequence, char const *context, sz_strings_view_t *strings,
                                     PyObject **snapshot, sz_string_view_t **views);

/* `shared.c` */

/** Overwrites @p length bytes at @p start with zeros, in a way a compiler may not elide. */
extern void sz_py_wipe_bytes(void *start, sz_size_t length);

/** Whether a buffer-like object is writable; sets a @c TypeError and returns false if not. */
extern sz_bool_t sz_py_is_mutable(PyObject *object);

/** Helper function to export a Python string-like object into a @c sz_string_view_t. */
extern sz_bool_t sz_py_export_string_like(PyObject *object, sz_cptr_t *start, sz_size_t *length);

/**
 *  @brief Parses an optional @c start or @c end index argument with CPython slice semantics.
 *  @return 1 with the result written to @p result_out, or 0 with a Python exception set.
 */
extern int sz_py_export_optional_index(PyObject *index_obj, Py_ssize_t default_index, Py_ssize_t *result_out);

/**
 *  @brief Parses an optional non-negative size argument, like a @c bound.
 *  @return 1 with the result written to @p result_out, or 0 with a Python exception set.
 */
extern int sz_py_export_optional_size(PyObject *size_object, char const *name, sz_size_t default_size,
                                      sz_size_t *result_out);

extern PyTypeObject FileType;
extern PyTypeObject StringType;
extern PyTypeObject StringsType;
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
extern PyTypeObject LevenshteinEngineType;
extern PyTypeObject OverlapEngineType;
extern PyTypeObject SubstringsEngineType;
extern PyTypeObject RegexEngineType;
extern PyTypeObject SubdocumentsEngineType;

extern struct PyModuleDef stringzilla_module;

/** Describes an on-disk file mapped into RAM, which is different from Python's native @c mmap
 *  module, as it exposes the address of the mapping in memory. */
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
 *  @brief Type-punned StringZilla-string, that points to a slice of an existing Python @c str or a
 *      @c File.
 *
 *  When a slice is constructed, the @c parent object's reference count is being incremented to
 *  preserve lifetime. It usage in Python would look like:
 *
 *      - String() # Empty string
 *      - String("some-string") # Full-range slice of a Python @c str
 *      - String(File("some-path.txt")) # Full-range view of a persisted file
 *      - String(File("some-path.txt"), from=0, to=sys.maxsize)
 */
typedef struct {
    PyObject ob_base;

    PyObject *parent;
    sz_string_view_t memory;
} String;

/** The shape of every tiling segmenter's dispatch point, like @c sz_utf8_wordbreaks_best. */
typedef sz_status_t (*sz_py_segmenter_t)(sz_cptr_t text, sz_size_t length, sz_size_t *lengths, sz_size_t capacity,
                                         sz_size_t *count, sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Iterator for finding UAX segment boundaries in UTF-8 text - words, grapheme clusters,
 *      sentences, and line-break opportunities.
 *
 *  Streams segments by refilling a small inline buffer of lengths with @c segmenter, yielding one
 *  segment per @c __next__. Segments tile the text, so the next one always starts at @c start.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj;

    sz_cptr_t start;
    sz_cptr_t end;

    sz_py_segmenter_t segmenter;
    sz_capability_t capabilities;

    sz_size_t batch_lengths[sz_iterators_default_steps_k];
    sz_size_t batch_count;
    sz_size_t batch_index;

} Utf8Boundaries;

/**
 *  @brief An immutable collection of strings on one tape, in the Apache Arrow @c String or
 *      @c LargeString layout, which kernels, engines and Arrow read in place.
 *
 *  The tape either borrows the blocks of @c parent, like an Arrow array or the @c Strings it was
 *  sliced from, with offsets that may start above zero, or owns blocks @c allocator frees.
 *
 *  @see Arrow layout: https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
 */
typedef struct {
    PyObject ob_base;

    /** 32-bit offsets while the payload fits 4 GiB, 64-bit beyond, the count, and the bytes. */
    sz_strings_span_t tape;

    /** The object whose blocks @c tape borrows, or null when @c allocator owns them. */
    PyObject *parent;

    /** Frees both blocks of @c tape when nothing is borrowed. */
    sz_allocator_t allocator;
} Strings;

/** The view of string @p index of @p tape. */
static inline sz_string_view_t sz_py_tape_string_(sz_strings_span_t const *tape, sz_size_t index) {
    sz_string_view_t string;
    if (tape->layout == sz_strings_u64_tape_k) {
        sz_u64_t const *offsets = tape->data.u64_tape.offsets + index;
        string.start = tape->data.u64_tape.data + offsets[0], string.length = (sz_size_t)(offsets[1] - offsets[0]);
    }
    else {
        sz_u32_t const *offsets = tape->data.u32_tape.offsets + index;
        string.start = tape->data.u32_tape.data + offsets[0], string.length = offsets[1] - offsets[0];
    }
    return string;
}

/** Bytes from the start of the payload of @p tape to the end of its last string. */
static inline sz_size_t sz_py_tape_payload_(sz_strings_span_t const *tape) {
    if (tape->layout == sz_strings_u64_tape_k)
        return tape->data.u64_tape.offsets ? (sz_size_t)tape->data.u64_tape.offsets[tape->count] : 0;
    return tape->data.u32_tape.offsets ? tape->data.u32_tape.offsets[tape->count] : 0;
}

/* A lock some threads take with the GIL released, so a waiter never blocks a holder that needs the
 * GIL. @c PyMutex is zero-initialized by @c tp_alloc and never torn down; before 3.13 it is a
 * @c PyThread_type_lock that its owner opens before first use and closes when it dies. */
#if PY_VERSION_HEX >= 0x030D0000

typedef PyMutex sz_py_mutex_t;

static inline int sz_py_mutex_open_(sz_py_mutex_t *mutex) {
    sz_unused_(mutex);
    return 0;
}

static inline void sz_py_mutex_close_(sz_py_mutex_t *mutex) { sz_unused_(mutex); }
static inline void sz_py_mutex_lock_(sz_py_mutex_t *mutex) { PyMutex_Lock(mutex); }
static inline void sz_py_mutex_unlock_(sz_py_mutex_t *mutex) { PyMutex_Unlock(mutex); }

#else

typedef PyThread_type_lock sz_py_mutex_t;

static inline int sz_py_mutex_open_(sz_py_mutex_t *mutex) {
    if (!*mutex) *mutex = PyThread_allocate_lock();
    if (*mutex) return 0;
    PyErr_NoMemory();
    return -1;
}

static inline void sz_py_mutex_close_(sz_py_mutex_t *mutex) {
    if (*mutex) PyThread_free_lock(*mutex);
}

static inline void sz_py_mutex_lock_(sz_py_mutex_t *mutex) { PyThread_acquire_lock(*mutex, WAIT_LOCK); }
static inline void sz_py_mutex_unlock_(sz_py_mutex_t *mutex) { PyThread_release_lock(*mutex); }

#endif

/** Headers retained per interpreter, per type. */
enum { sz_freelist_capacity_k = 64 };

/**
 *  @brief Per-interpreter module state, holding the intrusive free-lists for @c String and
 *      @c Strings.
 *
 *  Both objects churn heavily: nearly every @c split or iterate element and every slice mints a
 *  fresh fixed-size header that is torn down moments later. Instead of round-tripping each header
 *  through @c PyObject_Malloc and @c PyObject_Free, dealloc parks the dead header on a
 *  singly-linked free-list and the allocation helper pops it back. The link is threaded through the
 *  dead object's own @c parent, so the state needs only a head pointer and a counter per type,
 *  never above @c sz_freelist_capacity_k. Living in module state keeps it per-interpreter, but on a
 *  free-threaded build the module state is shared by every thread in the interpreter, so all four
 *  fields are guarded by @c freelist_mutex: without it, concurrent @c alloc_ and @c dealloc calls
 *  race on the same linked list and can hand out one header to two live objects at once.
 */
typedef struct {
    String *string_freelist_head;
    sz_size_t string_freelist_count;
    Strings *strings_freelist_head;
    sz_size_t strings_freelist_count;

    /** Zero-initialized with the module state, so it starts unlocked. */
#if defined(Py_GIL_DISABLED)
    sz_py_mutex_t freelist_mutex;
#endif
} stringzilla_state_t;

/* On a free-threaded build the module state is shared by every thread in the interpreter, so the
 * free-list head and count pairs need a lock around each read-modify-write; on a GIL build the GIL
 * already serializes these calls, so the lock is not taken at all. */
#if defined(Py_GIL_DISABLED)
static inline void sz_freelist_lock_(stringzilla_state_t *state) { sz_py_mutex_lock_(&state->freelist_mutex); }
static inline void sz_freelist_unlock_(stringzilla_state_t *state) { sz_py_mutex_unlock_(&state->freelist_mutex); }
#else
static inline void sz_freelist_lock_(stringzilla_state_t *state) { sz_unused_(state); }
static inline void sz_freelist_unlock_(stringzilla_state_t *state) { sz_unused_(state); }
#endif

/* `shared.c` */

/** Reach the per-interpreter free-list state, or @c NULL before registration / during teardown. */
extern stringzilla_state_t *stringzilla_state_(void);

/** Allocate a blank @c String header, reusing a cached one from the free-list when available. */
extern String *String_alloc_(void);

/** Allocate a blank @c Strings header, reusing a cached one from the free-list when available. */
extern Strings *Strings_alloc_(void);

/** Helper function to wrap the current exception with a custom prefix message. */
extern void wrap_current_exception(sz_cptr_t comment);

/* `strings.c` */

/** Wraps a tape that a verb wrote with blocks from @p allocator into a new @c Strings owning them,
 *  or frees the tape and raises the failed @p status, naming @p context. */
extern PyObject *Strings_from_tape_(sz_status_t status, sz_strings_span_t *tape, sz_allocator_t *allocator,
                                    char const *context);

/** Gathers `strings[order[i]]`, or the first @p count strings without @p order, into a new
 *  @c Strings owning its tape, or raises naming @p context. */
extern PyObject *Strings_gather_(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                                 char const *context);

/*  Cross-domain function and docstring declarations: every `String_like_*` and `Strings_*` method
 *  and its `doc_*` docstring constant named by the `String_methods[]`, `Strings_methods[]`, and
 *  `stringzilla_methods[]` @c PyMethodDef tables in `string.c`, `strings.c`, and `stringzilla.c`.
 *  Each symbol is visible here so those tables can reach the domain file that owns it. */

/* `strings.c` */
extern char const doc_translate[], doc_Strings_translate[], doc_Strings_escape[], doc_Strings_unescape[],
    doc_Strings_normalize[], doc_Strings_to_json[], doc_to_json[], doc_to_csv[], doc_String_join[];

/** Maps @p length bytes of @p source through @p lut into @p target, which may be @p source. */
extern sz_status_t sz_py_lookup(sz_ptr_t target, sz_cptr_t source, sz_size_t length, char const *lut,
                                sz_capability_t capabilities);
extern PyObject *String_like_translate(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple);
extern PyObject *Strings_translate(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *Strings_escape(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *Strings_unescape(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple);
extern PyObject *Strings_normalize(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *Strings_to_json(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);
extern PyObject *module_to_json(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *module_to_csv(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple);
extern PyObject *String_join(String *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple);

/* `hash.c` */
extern char const doc_like_hash[], doc_hash_multiseed[], doc_fill_random[], doc_random[], doc_like_bytesum[],
    doc_like_sha256[], doc_hmac_sha256[], doc_Strings_hashes[];
extern Py_hash_t String_hash(String *self);
extern PyObject *Strings_hashes(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple);
extern PyObject *String_like_hash(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple);
extern PyObject *String_like_hash_multiseed(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);
extern PyObject *String_like_fill_random(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple);
extern PyObject *module_random(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple);
extern PyObject *String_like_bytesum(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple);
extern PyObject *String_like_sha256(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *hmac_sha256(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple);

/* `cipher.c` */

/** Exception type raised when an AEAD tag fails to authenticate a ciphertext. */
extern PyObject *AuthenticationErrorType;
extern char const doc_AuthenticationError[];

/* `compare.c` */
extern char const doc_like_equal[];
extern PyObject *String_like_equal(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *String_richcompare(PyObject *self, PyObject *other, int op);

/* `find.c` */
extern char const doc_contains[], doc_find[], doc_index[], doc_rfind[], doc_rindex[], doc_partition[], doc_rpartition[],
    doc_count[], doc_startswith[], doc_endswith[], doc_find_first_of[], doc_find_first_not_of[], doc_find_last_of[],
    doc_find_last_not_of[], doc_count_byteset[], doc_split[], doc_rsplit[], doc_split_byteset[], doc_rsplit_byteset[],
    doc_split_iter[], doc_rsplit_iter[], doc_split_byteset_iter[], doc_rsplit_byteset_iter[], doc_lstrip[],
    doc_rstrip[], doc_strip[], doc_splitlines[], doc_Strings_find[];
extern int String_in(String *self, PyObject *needle_obj);
extern PyObject *Strings_find(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple);
extern PyObject *String_like_contains(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple);
extern PyObject *String_like_find(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple);
extern PyObject *String_like_index(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *String_like_rfind(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *String_like_rindex(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *String_like_partition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple);
extern PyObject *String_like_rpartition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *String_like_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *String_like_startswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *String_like_endswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple);
extern PyObject *String_like_find_first_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple);
extern PyObject *String_like_find_first_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple);
extern PyObject *String_like_find_last_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple);
extern PyObject *String_like_find_last_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                              PyObject *args_names_tuple);
extern PyObject *String_like_count_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple);
extern PyObject *String_like_split(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *String_like_rsplit(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *String_like_split_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple);
extern PyObject *String_like_rsplit_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);
extern PyObject *String_like_split_iterator(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);
extern PyObject *String_like_rsplit_iterator(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);
extern PyObject *String_like_split_byteset_iterator(PyObject *self, PyObject *const *args,
                                                    Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *String_like_rsplit_byteset_iterator(PyObject *self, PyObject *const *args,
                                                     Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *String_like_lstrip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *String_like_rstrip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *String_like_strip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);
extern PyObject *String_like_splitlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);

/* `sort.c` */
extern char const doc_argsort[];
extern sz_status_t Strings_run_argsort_(sz_bool_t uncased, sz_strings_view_t const *strings, sz_size_t *order,
                                        sz_size_t top, sz_bool_t reverse, sz_capability_t capabilities);
extern PyObject *Strings_argsort(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);

/* `intersect.c` */
extern char const doc_Strings_intersect[];
extern PyObject *Strings_intersect(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple);

/* `utf8_runes.c` */
extern char const doc_utf8_count[], doc_utf8_codepoints[], doc_Strings_utf8_count[];
extern PyObject *String_like_utf8_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple);
extern PyObject *Strings_utf8_count(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *String_like_utf8_codepoints(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);

/* `utf8_tokens.c` */
extern char const doc_utf8_split_newlines[], doc_utf8_newlines[], doc_utf8_split_whitespaces[], doc_utf8_whitespaces[],
    doc_utf8_split_delimiters[], doc_utf8_delimiters[];
extern PyObject *String_like_utf8_split_newlines(PyObject *self, PyObject *const *args,
                                                 Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *String_like_utf8_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                           PyObject *args_names_tuple);
extern PyObject *String_like_utf8_split_whitespaces(PyObject *self, PyObject *const *args,
                                                    Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *String_like_utf8_whitespaces(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                              PyObject *args_names_tuple);
extern PyObject *String_like_utf8_split_delimiters(PyObject *self, PyObject *const *args,
                                                   Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *String_like_utf8_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);

/* `utf8_boundaries.c` */

/** Builds a @c Utf8Boundaries iterator of @p type, segmented by @p segmenter, over @p self when it
 *  is a @c String, or else the one positional argument of a `String_like_*` call named @p name. */
extern PyObject *Utf8Boundaries_make_(PyTypeObject *type, char const *name, sz_py_segmenter_t segmenter, PyObject *self,
                                      PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple);

/** Yields the next segment as a @c String view, refilling the inline batch when it drains. */
extern PyObject *Utf8Boundaries_next_(Utf8Boundaries *self);

/** Releases the reference to the segmented text and frees the iterator. */
extern void Utf8Boundaries_dealloc_(Utf8Boundaries *self);

/** Returns the iterator itself, as required by the Python iterator protocol. */
extern PyObject *Utf8Boundaries_iter_(PyObject *self);

/* `utf8_wordbreaks.c` */
extern char const doc_utf8_wordbreaks[];
extern PyObject *String_like_utf8_wordbreaks(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);

/* `utf8_graphemes.c` */
extern char const doc_utf8_graphemes[];
extern PyObject *String_like_utf8_graphemes(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);

/* `utf8_sentences.c` */
extern char const doc_utf8_sentences[];
extern PyObject *String_like_utf8_sentences(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                            PyObject *args_names_tuple);

/* `utf8_linebreaks.c` */
extern char const doc_utf8_linebreaks[];
extern PyObject *String_like_utf8_linebreaks(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple);

/* `utf8_uncased_fold.c` */
extern char const doc_utf8_uncased_fold[];
extern PyObject *String_like_utf8_uncased_fold(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple);

/* `utf8_uncased.c` */
extern char const doc_utf8_uncased_search[], doc_utf8_uncased_order[], doc_utf8_uncased_matches[];
extern PyObject *String_like_utf8_uncased_search(PyObject *self, PyObject *const *args,
                                                 Py_ssize_t positional_args_count, PyObject *args_names_tuple);
extern PyObject *String_like_utf8_uncased_order(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                                PyObject *args_names_tuple);
extern PyObject *String_like_utf8_uncased_matches(PyObject *self, PyObject *const *args,
                                                  Py_ssize_t positional_args_count, PyObject *args_names_tuple);

/* `utf8_norm.c` */
extern char const doc_utf8_norm[], doc_utf8_find_denormalized[];

/** Reads a form name `sz.utf8_norm` accepts, or raises an exception naming @p context. */
extern int sz_py_export_normal_form(PyObject *form_object, char const *context, sz_normal_form_t *form);
extern PyObject *String_like_utf8_norm(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple);
extern PyObject *String_like_utf8_find_denormalized(PyObject *self, PyObject *const *args,
                                                    Py_ssize_t positional_args_count, PyObject *args_names_tuple);

/* `stringzilla.c` */

/**
 *  @brief Reads the `capabilities=` keyword of a call that runs kernels.
 *  @param[in] capabilities_object A @c Capability mask, or @c NULL or None for the enabled mask.
 *  @param[out] capabilities The mask to dispatch with, kept to CPU capabilities this machine has.
 *  @return 0 on success, or -1 with a Python exception set.
 */
extern int sz_py_export_capabilities(PyObject *capabilities_object, sz_capability_t *capabilities);

/** Raises the Python exception @p status names, blaming @p context; never called on success. */
extern void sz_py_raise_status(sz_status_t status, char const *context);

/** Views the @c Strings @p object in place, or raises a @c TypeError naming @p name. Device engines
 *  read only tapes in memory their device reaches, as from @c Strings.copy, and refuse the rest. */
extern int sz_py_export_strings(PyObject *object, char const *name, sz_strings_view_t *strings);

/** Reads a device stream handle carried as an integer, or @c NULL for the default stream. */
extern int sz_py_export_stream(PyObject *stream_object, sz_stream_t *stream);

/**
 *  @brief Reads how an engine is built: its `capabilities=` and `stream=` keywords.
 *  @param[in] capabilities_object A producer's mask, or @c NULL or None for the CPU's enabled one.
 *  @param[out] capabilities The mask to build the engine with.
 *  @param[out] stream A stream of the device that mask names, @c NULL when absent.
 *  @return 0 on success, or -1 with a Python exception set, refusing CPU capabilities this CPU
 *      cannot run and a CPU engine given a stream.
 */
extern int sz_py_export_engine_placement(PyObject *capabilities_object, PyObject *stream_object,
                                         sz_capability_t *capabilities, sz_stream_t *stream);

/**
 *  @brief Binds @p object as a writable output of @p rank axes, each at least @p extents wide.
 *  @param[out] strides Entries from one index of each axis to the next, in items rather than bytes.
 *  @return 0 with @p view bound, or -1 with an exception set and nothing left to release.
 *  @note Checks no contiguity, so a verb taking no stride for an axis checks that axis itself.
 */
extern int sz_py_export_output_buffer(PyObject *object, char const *name, Py_ssize_t itemsize, int rank,
                                      sz_size_t const *extents, Py_buffer *view, sz_size_t *strides);

/** Binds @p object as a contiguous buffer of at least @p count items, read-only for a @p request of
 *  @c PyBUF_CONTIG_RO and writable for @c PyBUF_CONTIG, like the `out=` of `Strings.argsort`. */
extern int sz_py_export_contiguous_buffer(PyObject *object, char const *name, int request, Py_ssize_t itemsize,
                                          sz_size_t count, Py_buffer *view);

/* `string.c` */
extern char const doc_offset_within[], doc_write_to[], doc_decode[];
extern PyObject *String_new(PyTypeObject *type, PyObject *args, PyObject *kwds);
extern PyObject *String_like_decode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple);
extern PyObject *String_offset_within(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple);
extern PyObject *String_write_to(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple);

#endif // STRINGZILLA_PYTHON_STRINGZILLA_H_
