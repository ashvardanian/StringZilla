/**
 *  @file       stringzilla.c
 *  @brief      Very light-weight CPython wrapper for StringZilla, with support for memory-mapping,
 *              native Python strings, Apache Arrow collections, and more.
 *  @author     Ash Vardanian
 *  @date       July 10, 2023
 *  @copyright  Copyright (c) 2023
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids `PyArg_ParseTupleAndKeywords` calls.
 *  - Reimplements all of the `str` functionality in C as a `Str` type.
 *  - Provides a highly generic `Strs` class for handling collections of strings, Arrow-style or not.
 *
 *  Pandas doesn't provide a C API, and even in the 2.0 the Apache Arrow representation is opt-in, not default.
 *  PyCapsule protocol in conjunction with @b `__arrow_c_array__` dunder methods can be used to extract strings.
 *  @see https://arrow.apache.org/docs/python/generated/pyarrow.array.html
 *
 *  This module exports C functions via `PyCapsule` of `PyAPI` for use by other extensions (like `stringzillas-cpus`):
 *  - `sz_py_export_string_like`.
 *  - `sz_py_export_strings_as_sequence`.
 *  - `sz_py_export_strings_as_u32tape`.
 *  - `sz_py_export_strings_as_u64tape`.
 *  - `sz_py_replace_strings_allocator`.
 */
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

/**
 *  @brief Arrow C Data Interface structure for an array schema.
 *  @see https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
 */
struct ArrowSchema {
    char const *format;
    char const *name;
    char const *metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema **children;
    struct ArrowSchema *dictionary;
    void (*release)(struct ArrowSchema *);
    void *private_data;
};

/**
 *  @brief Arrow C Data Interface structure for an array content.
 *  @see https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
 */
struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    void const **buffers;
    struct ArrowArray **children;
    struct ArrowArray *dictionary;
    void (*release)(struct ArrowArray *);
    void *private_data;
};

typedef struct PyAPI {
    sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
    sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
    sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *);
} PyAPI;

#pragma region Forward Declarations

static PyTypeObject FileType;
static PyTypeObject StrType;
static PyTypeObject StrsType;
static PyTypeObject SplitIteratorType;

static sz_string_view_t temporary_memory = {NULL, 0};

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
 *  @brief  String-splitting separator.
 *
 *  Allows lazy evaluation of the `split` and `rsplit`, and can be used to create a `Strs` object.
 *  which might be more memory-friendly, than greedily invoking `str.split`.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj;      //< For reference counting
    PyObject *separator_obj; //< For reference counting

    sz_string_view_t text;
    sz_string_view_t separator;
    sz_find_t finder;

    /// @brief  How many bytes to skip after each successful find.
    ///         Generally equal to `needle_length`, or 1 for character sets.
    sz_size_t match_length;

    /// @brief  Should we include the separator in the resulting slices?
    sz_bool_t include_match;

    /// @brief  Should we enumerate the slices in normal or reverse order?
    sz_bool_t is_reverse;

    /// @brief  Upper limit for the number of splits to report. Monotonically decreases during iteration.
    sz_size_t max_parts;

    /// @brief  Indicates that we've already reported the tail of the split, and should return NULL next.
    sz_bool_t reached_tail;

} SplitIterator;

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

#pragma endregion

#pragma region Helpers

static sz_ptr_t temporary_memory_allocate(sz_size_t size, sz_string_view_t *existing) {
    if (existing->length < size) {
        sz_cptr_t new_start = realloc(existing->start, size);
        if (!new_start) {
            PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the Levenshtein matrix");
            return NULL;
        }
        existing->start = new_start;
        existing->length = size;
    }
    return existing->start;
}

static void temporary_memory_free(sz_ptr_t start, sz_size_t size, sz_string_view_t *existing) {}

static sz_cptr_t Strs_get_start_(void const *handle, sz_size_t i) {
    Strs *strs = (Strs *)handle;
    switch (strs->layout) {
    case STRS_U32_TAPE: return strs->data.u32_tape.data + strs->data.u32_tape.offsets[i];
    case STRS_U32_TAPE_VIEW: return strs->data.u32_tape_view.data + strs->data.u32_tape_view.offsets[i];
    case STRS_U64_TAPE: return strs->data.u64_tape.data + strs->data.u64_tape.offsets[i];
    case STRS_U64_TAPE_VIEW: return strs->data.u64_tape_view.data + strs->data.u64_tape_view.offsets[i];
    case STRS_FRAGMENTED: return strs->data.fragmented.spans[i].start;
    }
    return NULL;
}

static sz_size_t Strs_get_length_(void const *handle, sz_size_t i) {
    Strs *strs = (Strs *)handle;
    switch (strs->layout) {
    case STRS_U32_TAPE: return strs->data.u32_tape.offsets[i + 1] - strs->data.u32_tape.offsets[i];
    case STRS_U32_TAPE_VIEW: return strs->data.u32_tape_view.offsets[i + 1] - strs->data.u32_tape_view.offsets[i];
    case STRS_U64_TAPE: return strs->data.u64_tape.offsets[i + 1] - strs->data.u64_tape.offsets[i];
    case STRS_U64_TAPE_VIEW: return strs->data.u64_tape_view.offsets[i + 1] - strs->data.u64_tape_view.offsets[i];
    case STRS_FRAGMENTED: return strs->data.fragmented.spans[i].length;
    }
    return 0;
}

void reverse_offsets(sz_sorted_idx_t *array, sz_size_t length) {
    sz_size_t i, j;
    // Swap array[i] and array[j]
    for (i = 0, j = length - 1; i < j; i++, j--) {
        sz_sorted_idx_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void reverse_haystacks(sz_string_view_t *array, sz_size_t length) {
    sz_size_t i, j;
    // Swap array[i] and array[j]
    for (i = 0, j = length - 1; i < j; i++, j--) {
        sz_string_view_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void permute(sz_string_view_t *array, sz_sorted_idx_t *order, sz_size_t length) {
    for (sz_size_t i = 0; i < length; ++i) {
        if (i == order[i]) continue;
        sz_string_view_t temp = array[i];
        sz_size_t k = i, j;
        while (i != (j = (sz_size_t)order[k])) {
            array[k] = array[j];
            order[k] = k;
            k = j;
        }
        array[k] = temp;
        order[k] = k;
    }
}

/**
 *  @brief  Helper function to export a Python string-like object into a `sz_string_view_t`.
 *          On failure, sets a Python exception and returns 0.
 */
SZ_DYNAMIC sz_bool_t sz_py_export_string_like(PyObject *object, sz_cptr_t *start, sz_size_t *length) {
    if (PyUnicode_Check(object)) {
        // Handle Python `str` object
        Py_ssize_t signed_length;
        *start = PyUnicode_AsUTF8AndSize(object, &signed_length);
        *length = (sz_size_t)signed_length;
        return 1;
    }
    else if (PyBytes_Check(object)) {
        // Handle Python `bytes` object
        // https://docs.python.org/3/c-api/bytes.html
        Py_ssize_t signed_length;
        if (PyBytes_AsStringAndSize(object, (sz_ptr_t *)start, &signed_length) == -1) {
            PyErr_SetString(PyExc_ValueError, "Couldn't access `bytes` buffer internals");
            return 0;
        }
        *length = (sz_size_t)signed_length;
        return 1;
    }
    else if (PyByteArray_Check(object)) {
        // Handle Python mutable `bytearray` object
        // https://docs.python.org/3/c-api/bytearray.html
        *start = PyByteArray_AS_STRING(object);
        *length = PyByteArray_GET_SIZE(object);
        return 1;
    }
    else if (PyObject_TypeCheck(object, &StrType)) {
        Str *str = (Str *)object;
        *start = str->memory.start;
        *length = str->memory.length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &FileType)) {
        File *file = (File *)object;
        *start = file->memory.start;
        *length = file->memory.length;
        return 1;
    }
    else if (PyMemoryView_Check(object)) {
        // Handle Python `memoryview` object
        // https://docs.python.org/3/c-api/memoryview.html
        // https://docs.python.org/3/c-api/buffer.html#c.Py_buffer
        Py_buffer *view = PyMemoryView_GET_BUFFER(object);
        // Make sure we are dealing with single-byte integral representations
        if (view->itemsize != 1) {
            PyErr_SetString(PyExc_ValueError, "Only single-byte integral types are supported");
            return 0;
        }
        // Let's make sure the data is contiguous.
        // This can be a bit trickier for high-dimensional arrays, but CPython has a built-in function for that.
        // The flag 'C' stands for C-style-contiguous, which means that the last dimension is contiguous.
        // The flag 'F' stands for Fortran-style-contiguous, which means that the first dimension is contiguous.
        // The flag 'A' stands for any-contiguous, which only means there are no gaps between elements.
        // For byte-level processing that's all we need.
        if (!PyBuffer_IsContiguous(view, 'A')) {
            PyErr_SetString(PyExc_ValueError, "The array must be contiguous");
            return 0;
        }

        *start = (sz_cptr_t)view->buf;
        *length = (sz_size_t)view->len;
        return 1;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument layout");
        return 0;
    }
}

sz_cptr_t sz_py_strs_sequence_member_start_if_fragmented(void const *sequence_punned, sz_size_t index) {
    Strs *strs = (Strs *)sequence_punned;
    sz_assert_(strs->layout == STRS_FRAGMENTED && "Expected a reordered Strs layout");
    if (index < 0 || index >= strs->data.fragmented.count) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds");
        return NULL;
    }
    return strs->data.fragmented.spans[index].start;
}

sz_size_t sz_py_strs_sequence_member_length_if_fragmented(void const *sequence_punned, sz_size_t index) {
    Strs *strs = (Strs *)sequence_punned;
    sz_assert_(strs->layout == STRS_FRAGMENTED && "Expected a reordered Strs layout");
    if (index < 0 || index >= strs->data.fragmented.count) {
        PyErr_SetString(PyExc_IndexError, "Index out of bounds");
        return 0;
    }
    return strs->data.fragmented.spans[index].length;
}

/**
 *  @brief  Helper function to export a `Strs` or similar sequence objects into a `sz_sequence_t`.
 */
SZ_DYNAMIC sz_bool_t sz_py_export_strings_as_sequence(PyObject *object, sz_sequence_t *sequence) {
    if (!sequence) return sz_false_k;

    if (PyObject_TypeCheck(object, &StrsType)) {
        Strs *strs = (Strs *)object;
        sz_assert_(strs->layout == STRS_FRAGMENTED && "View as tapes!");

        sequence->handle = strs;
        sequence->count = strs->data.fragmented.count;
        sequence->get_start = sz_py_strs_sequence_member_start_if_fragmented;
        sequence->get_length = sz_py_strs_sequence_member_length_if_fragmented;
        return sz_true_k;
    }

    return sz_false_k;
}

/**
 *  @brief  Helper function to export a `Strs` object into `sz_sequence_u32tape_t` components.
 */
SZ_DYNAMIC sz_bool_t sz_py_export_strings_as_u32tape(PyObject *object, sz_cptr_t *data, sz_u32_t const **offsets,
                                                     sz_size_t *count) {

    if (!data || !offsets || !count) return sz_false_k;
    if (!PyObject_TypeCheck(object, &StrsType)) return sz_false_k;
    Strs *strs = (Strs *)object;

    if (strs->layout == STRS_U32_TAPE) {
        *data = strs->data.u32_tape.data;
        *offsets = strs->data.u32_tape.offsets;
        *count = strs->data.u32_tape.count;
        return sz_true_k;
    }
    else if (strs->layout == STRS_U32_TAPE_VIEW) {
        *data = strs->data.u32_tape_view.data;
        *offsets = strs->data.u32_tape_view.offsets;
        *count = strs->data.u32_tape_view.count;
        return sz_true_k;
    }
    else { return sz_false_k; }
}

/**
 *  @brief  Helper function to export a `Strs` object into `sz_sequence_u64tape_t` components.
 */
SZ_DYNAMIC sz_bool_t sz_py_export_strings_as_u64tape(PyObject *object, sz_cptr_t *data, sz_u64_t const **offsets,
                                                     sz_size_t *count) {

    if (!data || !offsets || !count) return sz_false_k;
    if (!PyObject_TypeCheck(object, &StrsType)) return sz_false_k;
    Strs *strs = (Strs *)object;

    if (strs->layout == STRS_U64_TAPE) {
        *data = strs->data.u64_tape.data;
        *offsets = strs->data.u64_tape.offsets;
        *count = strs->data.u64_tape.count;
        return sz_true_k;
    }
    else if (strs->layout == STRS_U64_TAPE_VIEW) {
        *data = strs->data.u64_tape_view.data;
        *offsets = strs->data.u64_tape_view.offsets;
        *count = strs->data.u64_tape_view.count;
        return sz_true_k;
    }
    else { return sz_false_k; }
}

static sz_bool_t sz_py_replace_u32_tape_allocator(Strs *strs, sz_memory_allocator_t *old_allocator,
                                                  sz_memory_allocator_t *allocator) {
    struct u32_tape_t *data = &strs->data.u32_tape;
    sz_assert_(data->offsets && "Expected offsets to be allocated");

    sz_size_t const string_data_size = (sz_size_t)data->offsets[data->count];
    sz_size_t const offsets_size = (data->count + 1) * sizeof(sz_u32_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data =
        string_data_size ? (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle) : (sz_ptr_t)NULL;
    if (string_data_size && !new_string_data) return sz_false_k;
    memcpy(new_string_data, data->data, string_data_size);

    // Allocate new offsets array
    sz_u32_t *new_offsets =
        offsets_size ? (sz_u32_t *)allocator->allocate(offsets_size, allocator->handle) : (sz_u32_t *)NULL;
    if (offsets_size && !new_offsets) {
        if (string_data_size) allocator->free(new_string_data, string_data_size, allocator->handle);
        return sz_false_k;
    }
    memcpy(new_offsets, data->offsets, offsets_size);

    // Free old memory with old allocator (tapes always own their data)
    old_allocator->free(data->data, string_data_size, old_allocator->handle);
    old_allocator->free(data->offsets, offsets_size, old_allocator->handle);

    // Update pointers and allocator
    data->data = new_string_data;
    data->offsets = new_offsets;
    data->allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_u64_tape_allocator(Strs *strs, sz_memory_allocator_t *old_allocator,
                                                  sz_memory_allocator_t *allocator) {
    struct u64_tape_t *data = &strs->data.u64_tape;
    sz_assert_(data->offsets && "Expected offsets to be allocated");

    sz_size_t string_data_size = (sz_size_t)data->offsets[data->count];
    sz_size_t offsets_size = (data->count + 1) * sizeof(sz_u64_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data =
        string_data_size ? (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle) : (sz_ptr_t)NULL;
    if (string_data_size && !new_string_data) return sz_false_k;
    memcpy(new_string_data, data->data, string_data_size);

    // Allocate new offsets array
    sz_u64_t *new_offsets =
        offsets_size ? (sz_u64_t *)allocator->allocate(offsets_size, allocator->handle) : (sz_u64_t *)NULL;
    if (offsets_size && !new_offsets) {
        if (string_data_size) allocator->free(new_string_data, string_data_size, allocator->handle);
        return sz_false_k;
    }
    memcpy(new_offsets, data->offsets, offsets_size);

    // Free old memory with old allocator (tapes always own their data)
    old_allocator->free(data->data, string_data_size, old_allocator->handle);
    old_allocator->free(data->offsets, offsets_size, old_allocator->handle);

    // Update pointers and allocator
    data->data = new_string_data;
    data->offsets = new_offsets;
    data->allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_u32_tape_view_allocator(Strs *strs, sz_memory_allocator_t *allocator) {
    // Convert view to tape by copying the data
    struct u32_tape_view_t *view = &strs->data.u32_tape_view;
    sz_size_t const string_data_size = (sz_size_t)view->offsets[view->count];
    sz_size_t const offsets_size = (view->count + 1) * sizeof(sz_u32_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data = NULL;
    if (string_data_size > 0) {
        new_string_data = (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle);
        if (!new_string_data) return sz_false_k;
        memcpy(new_string_data, view->data, string_data_size);
    }

    // Allocate new offsets array
    sz_u32_t *new_offsets = NULL;
    if (offsets_size > 0) {
        new_offsets = (sz_u32_t *)allocator->allocate(offsets_size, allocator->handle);
        if (!new_offsets) {
            if (string_data_size > 0) allocator->free(new_string_data, string_data_size, allocator->handle);
            return sz_false_k;
        }
        memcpy(new_offsets, view->offsets, offsets_size);
    }

    // Release parent reference if any
    Py_XDECREF(view->parent);

    // Convert to tape layout
    strs->layout = STRS_U32_TAPE;
    strs->data.u32_tape.count = view->count;
    strs->data.u32_tape.data = new_string_data;
    strs->data.u32_tape.offsets = new_offsets;
    strs->data.u32_tape.allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_u64_tape_view_allocator(Strs *strs, sz_memory_allocator_t *allocator) {
    // Convert view to tape by copying the data
    struct u64_tape_view_t *view = &strs->data.u64_tape_view;
    sz_size_t const string_data_size = (sz_size_t)view->offsets[view->count];
    sz_size_t const offsets_size = (view->count + 1) * sizeof(sz_u64_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data = NULL;
    if (string_data_size > 0) {
        new_string_data = (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle);
        if (!new_string_data) return sz_false_k;
        memcpy(new_string_data, view->data, string_data_size);
    }

    // Allocate new offsets array
    sz_u64_t *new_offsets = NULL;
    if (offsets_size > 0) {
        new_offsets = (sz_u64_t *)allocator->allocate(offsets_size, allocator->handle);
        if (!new_offsets) {
            if (string_data_size > 0) allocator->free(new_string_data, string_data_size, allocator->handle);
            return sz_false_k;
        }
        memcpy(new_offsets, view->offsets, offsets_size);
    }

    // Release parent reference if any
    Py_XDECREF(view->parent);

    // Convert to tape layout
    strs->layout = STRS_U64_TAPE;
    strs->data.u64_tape.count = view->count;
    strs->data.u64_tape.data = new_string_data;
    strs->data.u64_tape.offsets = new_offsets;
    strs->data.u64_tape.allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_fragmented_allocator(Strs *strs, sz_memory_allocator_t *old_allocator,
                                                    sz_memory_allocator_t *allocator) {
    struct fragmented_t *fragmented = &strs->data.fragmented;
    sz_assert_(fragmented->spans && "Expected spans to be allocated");

    // Calculate total size needed for consolidated tape
    sz_size_t total_bytes = 0;
    for (sz_size_t i = 0; i < fragmented->count; i++) total_bytes += fragmented->spans[i].length;

    // Choose 32-bit or 64-bit tape based on size
    sz_bool_t use_64bit = total_bytes >= UINT32_MAX;

    // Skip allocation if there's no data to allocate (empty strings case)
    if (total_bytes == 0) {
        // Convert to empty tape layout
        old_allocator->free(fragmented->spans, fragmented->count * sizeof(sz_string_view_t), old_allocator->handle);
        Py_XDECREF(fragmented->parent);

        strs->layout = STRS_U32_TAPE;
        strs->data.u32_tape.count = fragmented->count;
        strs->data.u32_tape.data = NULL;
        strs->data.u32_tape.offsets = NULL;
        strs->data.u32_tape.allocator = *allocator;
        return sz_true_k;
    }

    // Allocate consolidated data buffer and offsets array
    sz_ptr_t new_data = (sz_ptr_t)allocator->allocate(total_bytes, allocator->handle);
    if (!new_data) return sz_false_k;

    if (use_64bit) {
        sz_u64_t *new_offsets =
            (sz_u64_t *)allocator->allocate((fragmented->count + 1) * sizeof(sz_u64_t), allocator->handle);
        if (!new_offsets) {
            allocator->free(new_data, total_bytes, allocator->handle);
            return sz_false_k;
        }

        // Copy fragmented data into consolidated buffer
        sz_size_t current_offset = 0;
        new_offsets[0] = 0;
        for (sz_size_t i = 0; i < fragmented->count; i++) {
            sz_size_t len = fragmented->spans[i].length;
            if (len > 0) { memcpy(new_data + current_offset, fragmented->spans[i].start, len); }
            current_offset += len;
            new_offsets[i + 1] = current_offset;
        }

        // Free old fragmented data and convert to 64-bit tape
        old_allocator->free(fragmented->spans, fragmented->count * sizeof(sz_string_view_t), old_allocator->handle);
        Py_XDECREF(fragmented->parent);

        strs->layout = STRS_U64_TAPE;
        strs->data.u64_tape.count = fragmented->count;
        strs->data.u64_tape.data = new_data;
        strs->data.u64_tape.offsets = new_offsets;
        strs->data.u64_tape.allocator = *allocator;
    }
    else {
        sz_u32_t *new_offsets =
            (sz_u32_t *)allocator->allocate((fragmented->count + 1) * sizeof(sz_u32_t), allocator->handle);
        if (!new_offsets) {
            allocator->free(new_data, total_bytes, allocator->handle);
            return sz_false_k;
        }

        // Copy fragmented data into consolidated buffer
        sz_size_t current_offset = 0;
        new_offsets[0] = 0;
        for (sz_size_t i = 0; i < fragmented->count; i++) {
            sz_size_t len = fragmented->spans[i].length;
            if (len > 0) { memcpy(new_data + current_offset, fragmented->spans[i].start, len); }
            current_offset += len;
            // Ensure we don't overflow 32-bit offset
            if (current_offset > UINT32_MAX) {
                allocator->free(new_data, total_bytes, allocator->handle);
                allocator->free(new_offsets, (fragmented->count + 1) * sizeof(sz_u32_t), allocator->handle);
                return sz_false_k;
            }
            new_offsets[i + 1] = (sz_u32_t)current_offset;
        }

        // Free old fragmented data and convert to 32-bit tape
        old_allocator->free(fragmented->spans, fragmented->count * sizeof(sz_string_view_t), old_allocator->handle);
        Py_XDECREF(fragmented->parent);

        strs->layout = STRS_U32_TAPE;
        strs->data.u32_tape.count = fragmented->count;
        strs->data.u32_tape.data = new_data;
        strs->data.u32_tape.offsets = new_offsets;
        strs->data.u32_tape.allocator = *allocator;
    }
    return sz_true_k;
}

/**
 *  @brief  Helper function to replace the memory allocator in a `Strs` object.
 *          This reallocates existing string data using the new allocator.
 *
 *  This may change the layout of the `Strs` layout:
 *  - `STRS_U32_TAPE_VIEW` becomes `STRS_U32_TAPE`.
 *  - `STRS_U64_TAPE_VIEW` becomes `STRS_U64_TAPE`.
 *  - `STRS_U32_TAPE` remains, if the allocator is different.
 *  - `STRS_U64_TAPE` remains, if the allocator is different.
 *  - `STRS_FRAGMENTED` becomes a `STRS_U32_TAPE` or `STRS_U64_TAPE` depending on the content size.
 */
SZ_DYNAMIC sz_bool_t sz_py_replace_strings_allocator(PyObject *object, sz_memory_allocator_t *allocator) {
    if (!object || !allocator) return sz_false_k;
    if (!PyObject_TypeCheck(object, &StrsType)) return sz_false_k;

    Strs *strs = (Strs *)object;

    // Get the current allocator based on layout
    sz_memory_allocator_t old_allocator;
    switch (strs->layout) {
    case STRS_U32_TAPE: old_allocator = strs->data.u32_tape.allocator; break;
    case STRS_U64_TAPE: old_allocator = strs->data.u64_tape.allocator; break;
    case STRS_FRAGMENTED: old_allocator = strs->data.fragmented.allocator; break;
    case STRS_U32_TAPE_VIEW:
    case STRS_U64_TAPE_VIEW:
    default:
        // Views don't own memory, use default allocator for comparison
        sz_memory_allocator_init_default(&old_allocator);
        break;
    }

    // Check if the allocators are the same - no need to reallocate
    if (sz_memory_allocator_equal(&old_allocator, allocator)) return sz_true_k;

    // Handle different Strs layouts using dedicated functions
    switch (strs->layout) {
    case STRS_U32_TAPE: return sz_py_replace_u32_tape_allocator(strs, &old_allocator, allocator);
    case STRS_U64_TAPE: return sz_py_replace_u64_tape_allocator(strs, &old_allocator, allocator);
    case STRS_U32_TAPE_VIEW: return sz_py_replace_u32_tape_view_allocator(strs, allocator);
    case STRS_U64_TAPE_VIEW: return sz_py_replace_u64_tape_view_allocator(strs, allocator);
    case STRS_FRAGMENTED: return sz_py_replace_fragmented_allocator(strs, &old_allocator, allocator);
    }

    return sz_false_k; // Should never reach here
}

/**
 *  @brief  Helper function to wrap the current exception with a custom prefix message.
 *          A example is augmenting the argument parsing error with the name of the variable
 *          that didn't pass the validation.
 */
void wrap_current_exception(sz_cptr_t comment) {
    // ? Prior to Python 3.12 we need to fetch and restore the exception state using
    // ? `PyErr_Fetch` and `PyErr_Restore` to avoid overwriting the current exception.
    // ? After Python 3.12 we can use `PyErr_GetRaisedException` and `PyErr_SetRaisedException`.
    sz_unused_(comment);
}

typedef void (*get_string_at_offset_t)(Strs *, Py_ssize_t, Py_ssize_t, PyObject **, sz_cptr_t *, sz_size_t *);

void str_at_offset_u32_tape(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                            PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u32_t start_offset = strs->data.u32_tape.offsets[i];
    sz_u32_t end_offset = strs->data.u32_tape.offsets[i + 1];
    *start = strs->data.u32_tape.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs; // Tapes own their data
}

void str_at_offset_u32_tape_view(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                 PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u32_t start_offset = strs->data.u32_tape_view.offsets[i];
    sz_u32_t end_offset = strs->data.u32_tape_view.offsets[i + 1];
    *start = strs->data.u32_tape_view.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs->data.u32_tape_view.parent;
}

void str_at_offset_u64_tape(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                            PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u64_t start_offset = strs->data.u64_tape.offsets[i];
    sz_u64_t end_offset = strs->data.u64_tape.offsets[i + 1];
    *start = strs->data.u64_tape.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs; // Tapes own their data
}

void str_at_offset_u64_tape_view(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                 PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u64_t start_offset = strs->data.u64_tape_view.offsets[i];
    sz_u64_t end_offset = strs->data.u64_tape_view.offsets[i + 1];
    *start = strs->data.u64_tape_view.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs->data.u64_tape_view.parent;
}

void str_at_offset_fragmented(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                              PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    *start = strs->data.fragmented.spans[i].start;
    *length = strs->data.fragmented.spans[i].length;
    *memory_owner = strs->data.fragmented.parent;
}

get_string_at_offset_t str_at_offset_getter(Strs *strs) {
    switch (strs->layout) {
    case STRS_U32_TAPE: return str_at_offset_u32_tape;
    case STRS_U32_TAPE_VIEW: return str_at_offset_u32_tape_view;
    case STRS_U64_TAPE: return str_at_offset_u64_tape;
    case STRS_U64_TAPE_VIEW: return str_at_offset_u64_tape_view;
    case STRS_FRAGMENTED: return str_at_offset_fragmented;
    default:
        // Unsupported layout
        PyErr_SetString(PyExc_TypeError, "Unsupported layout for conversion");
        return NULL;
    }
}

#pragma endregion

#pragma region Memory Mapping File

static void File_dealloc(File *self) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    if (self->memory.start) {
        UnmapViewOfFile(self->memory.start);
        self->memory.start = NULL;
    }
    if (self->mapping_handle) {
        CloseHandle(self->mapping_handle);
        self->mapping_handle = NULL;
    }
    if (self->file_handle) {
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
    }
#else
    if (self->memory.start) {
        munmap(self->memory.start, self->memory.length);
        self->memory.start = NULL;
        self->memory.length = 0;
    }
    if (self->file_descriptor != 0) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
    }
#endif
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *File_new(PyTypeObject *type, PyObject *positional_args, PyObject *named_args) {
    File *self;
    self = (File *)type->tp_alloc(type, 0);
    if (self == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't allocate the file handle!");
        return NULL;
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    self->file_handle = NULL;
    self->mapping_handle = NULL;
#else
    self->file_descriptor = 0;
#endif
    self->memory.start = NULL;
    self->memory.length = 0;
    return (PyObject *)self;
}

static int File_init(File *self, PyObject *positional_args, PyObject *named_args) {
    sz_cptr_t path;
    if (!PyArg_ParseTuple(positional_args, "s", &path)) return -1;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    DWORD path_attributes = GetFileAttributes(path);
    if (path_attributes == INVALID_FILE_ATTRIBUTES) {
        PyErr_SetString(PyExc_OSError, "Couldn't get file attributes!");
        return -1;
    }
    if (path_attributes & FILE_ATTRIBUTE_DIRECTORY) {
        PyErr_SetString(PyExc_ValueError, "The provided path is a directory, not a normal file!");
        return -1;
    }
    self->file_handle = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (self->file_handle == INVALID_HANDLE_VALUE) {
        PyErr_SetString(PyExc_OSError, "Couldn't map the file!");
        return -1;
    }

    self->mapping_handle = CreateFileMapping(self->file_handle, 0, PAGE_READONLY, 0, 0, 0);
    if (self->mapping_handle == 0) {
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_OSError, "Couldn't map the file!");
        return -1;
    }

    sz_ptr_t file = (sz_ptr_t)MapViewOfFile(self->mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (file == 0) {
        CloseHandle(self->mapping_handle);
        self->mapping_handle = NULL;
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_OSError, "Couldn't map the file!");
        return -1;
    }
    self->memory.start = file;
    self->memory.length = GetFileSize(self->file_handle, 0);
#else
    self->file_descriptor = open(path, O_RDONLY);
    if (self->file_descriptor == -1) {
        PyErr_Format(PyExc_OSError, "Couldn't open the file at '%s': %s", path, strerror(errno));
        return -1;
    }
    // No permissions are required on the file itself to get it's properties from the existing descriptor.
    // https://linux.die.net/man/2/fstat
    struct stat sb;
    if (fstat(self->file_descriptor, &sb) != 0) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_Format(PyExc_OSError, "Can't retrieve file size at '%s': %s", path, strerror(errno));
        return -1;
    }
    // Check if it's a regular file
    if (!S_ISREG(sb.st_mode)) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_Format(PyExc_ValueError, "The provided path is not a normal file at '%s'", path);
        return -1;
    }
    sz_size_t file_size = sb.st_size;
    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, self->file_descriptor, 0);
    if (map == MAP_FAILED) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_Format(PyExc_OSError, "Couldn't map the file at '%s': %s", path, strerror(errno));
        return -1;
    }
    self->memory.start = map;
    self->memory.length = file_size;
#endif

    return 0;
}

static PyMethodDef File_methods[] = { //
    {NULL, NULL, 0, NULL}};

static char const doc_File[] = //
    "File(path, mode='r')\\n"
    "\\n"
    "Memory-mapped file class that exposes the memory range for low-level access.\\n"
    "Provides efficient read-only access to file contents without loading into memory.\\n"
    "\\n"
    "Args:\\n"
    "  path (str): Path to the file to memory-map.\\n"
    "  mode (str): File access mode (default: 'r' for read-only).\\n"
    "\\n"
    "Example:\\n"
    "  >>> f = sz.File('data.txt')\\n"
    "  >>> content = str(f)  # Access file contents as string";

static PyTypeObject FileType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "stringzilla.File",
    .tp_doc = doc_File,
    .tp_basicsize = sizeof(File),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = File_methods,
    .tp_new = (newfunc)File_new,
    .tp_init = (initproc)File_init,
    .tp_dealloc = (destructor)File_dealloc,
};

#pragma endregion

#pragma region Str

static int Str_init(Str *self, PyObject *args, PyObject *kwargs) {

    // Parse all arguments into PyObjects first
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return -1;
    }
    PyObject *parent_obj = nargs >= 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *from_obj = nargs >= 2 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *to_obj = nargs >= 3 ? PyTuple_GET_ITEM(args, 2) : NULL;

    // Parse keyword arguments, if provided, and ensure no duplicates
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "parent") == 0 && !parent_obj) { parent_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "from") == 0 && !from_obj) { from_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "to") == 0 && !to_obj) { to_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return -1;
    }

    // Now, layout-check and cast each argument
    Py_ssize_t from = 0, to = PY_SSIZE_T_MAX;
    if (from_obj) {
        from = PyLong_AsSsize_t(from_obj);
        if (from == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The `from` argument must be an integer");
            return -1;
        }
    }
    if (to_obj) {
        to = PyLong_AsSsize_t(to_obj);
        if (to == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The `to` argument must be an integer");
            return -1;
        }
    }

    // Handle empty string
    if (parent_obj == NULL) {
        self->memory.start = NULL;
        self->memory.length = 0;
    }
    // Increment the reference count of the parent
    else if (sz_py_export_string_like(parent_obj, &self->memory.start, &self->memory.length)) {
        self->parent = parent_obj;
        Py_INCREF(parent_obj);
    }
    else {
        wrap_current_exception("Unsupported parent type");
        return -1;
    }

    // Apply slicing
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(self->memory.length, from, to, &normalized_offset, &normalized_length);
    self->memory.start = ((sz_ptr_t)self->memory.start) + normalized_offset;
    self->memory.length = normalized_length;
    return 0;
}

static PyObject *Str_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Str *self;
    self = (Str *)type->tp_alloc(type, 0);
    if (!self) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't allocate a Str handle!");
        return NULL;
    }

    self->parent = NULL;
    self->memory.start = NULL;
    self->memory.length = 0;
    return (PyObject *)self;
}

static void Str_dealloc(Str *self) {
    if (self->parent) { Py_XDECREF(self->parent); }
    else if (self->memory.start) { free(self->memory.start); }
    self->parent = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Str_str(Str *self) { return PyUnicode_FromStringAndSize(self->memory.start, self->memory.length); }

static PyObject *Str_repr(Str *self) {
    // Interestingly, known-length string formatting only works in Python 3.12 and later.
    // https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_FromFormat
    if (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12)
        return PyUnicode_FromFormat("sz.Str('%.*s')", (int)self->memory.length, self->memory.start);
    else {
        // Use a simpler formatting rule for older versions
        PyObject *str_obj = PyUnicode_FromStringAndSize(self->memory.start, self->memory.length);
        PyObject *result = PyUnicode_FromFormat("sz.Str('%U')", str_obj);
        Py_DECREF(str_obj);
        return result;
    }
}

static Py_hash_t Str_hash(Str *self) { return (Py_hash_t)sz_hash(self->memory.start, self->memory.length, 0); }

static char const doc_like_hash[] = //
    "Compute the hash value of the string.\n"
    "\n"
    "This function can be called as a method on a Str object or as a standalone function.\n"
    "Args:\n"
    "  text (Str or str or bytes): The string to hash (positional-only when standalone).\n"
    "  seed (int, optional): The seed value for hashing. Defaults to 0. Can be positional or keyword.\n"
    "Returns:\n"
    "  int: The hash value as an unsigned 64-bit integer. This differs from Python's\n"
    "       built-in `hash()` which returns a `Py_hash_t` and may be platform-dependent.\n"
    "Raises:\n"
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.\n"
    "Signature:\n"
    "  >>> def hash(text, seed=0, /) -> int: ...";

static PyObject *Str_like_hash(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *text_obj = NULL;
    PyObject *seed_obj = NULL;
    sz_string_view_t text;
    sz_u64_t seed = 0;

    // Check if this is a method call on a Str instance
    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 0 : 1;
    Py_ssize_t const expected_max = expected_min + 1;

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, is_member ? "hash() takes 0 or 1 positional arguments"
                                                   : "hash() takes 1 or 2 positional arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        text_obj = self;
        if (positional_args_count >= 1) seed_obj = args[0];
    }
    else {
        if (positional_args_count >= 1) text_obj = args[0];
        if (positional_args_count >= 2) seed_obj = args[1];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) {
                if (seed_obj) {
                    PyErr_SetString(PyExc_TypeError, "seed specified twice");
                    return NULL;
                }
                seed_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    // Validate and convert text
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Parse seed
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) return NULL;
    }

    sz_u64_t result = sz_hash(text.start, text.length, seed);
    return PyLong_FromUnsignedLongLong((unsigned long long)result);
}

static char const doc_like_bytesum[] = //
    "Compute the checksum of individual byte values in a string.\n"
    "\n"
    "This function can be called as a method on a Str object or as a standalone function.\n"
    "Args:\n"
    "  text (Str or str or bytes): The string to hash.\n"
    "Returns:\n"
    "  int: The checksum of individual byte values in a string.\n"
    "Raises:\n"
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.";

static PyObject *Str_like_bytesum(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 1 || args_names_tuple) {
        PyErr_SetString(PyExc_TypeError, "bytesum() expects exactly one positional argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    sz_string_view_t text;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    sz_u64_t result = sz_bytesum(text.start, text.length);
    return PyLong_FromUnsignedLongLong((unsigned long long)result);
}

static char const doc_like_equal[] = //
    "Check if two strings are equal.\n"
    "\n"
    "This function can be called as a method on a Str object or as a standalone function.\n"
    "Args:\n"
    "  first (Str or str or bytes): The first string object.\n"
    "  second (Str or str or bytes): The second string object.\n"
    "Returns:\n"
    "  bool: True if the strings are equal, False otherwise.\n"
    "Raises:\n"
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.";

static PyObject *Str_like_equal(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 1 || args_names_tuple) {
        PyErr_SetString(PyExc_TypeError, "equals() expects exactly two positional arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *other_obj = args[is_member];
    sz_string_view_t text, other;

    // Validate and convert tje texts
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length) || //
        !sz_py_export_string_like(other_obj, &other.start, &other.length)) {
        wrap_current_exception("The arguments must be string-like");
        return NULL;
    }

    if (text.length != other.length) { Py_RETURN_FALSE; }
    sz_bool_t result = sz_equal(text.start, other.start, text.length);
    if (result != sz_true_k) { Py_RETURN_FALSE; }
    Py_RETURN_TRUE;
}

static PyObject *Str_get_address(Str *self, void *closure) { return PyLong_FromSize_t((sz_size_t)self->memory.start); }
static PyObject *Str_get_nbytes(Str *self, void *closure) { return PyLong_FromSize_t(self->memory.length); }

static Py_ssize_t Str_len(Str *self) { return self->memory.length; }

static PyObject *Str_getitem(Str *self, Py_ssize_t i) {

    // Negative indexing
    if (i < 0) i += self->memory.length;

    if (i < 0 || (sz_size_t)i >= self->memory.length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    // Assuming the underlying data is UTF-8 encoded
    return PyUnicode_FromStringAndSize(self->memory.start + i, 1);
}

static PyObject *Str_subscript(Str *self, PyObject *key) {
    if (PySlice_Check(key)) {
        // Sanity checks
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
        if (PySlice_AdjustIndices(self->memory.length, &start, &stop, step) < 0) return NULL;
        if (step != 1) {
            PyErr_SetString(PyExc_IndexError, "Efficient step is not supported");
            return NULL;
        }

        // Create a new `Str` object
        Str *self_slice = (Str *)StrType.tp_alloc(&StrType, 0);
        if (self_slice == NULL && PyErr_NoMemory()) return NULL;

        // Set its properties based on the slice
        self_slice->memory.start = self->memory.start + start;
        self_slice->memory.length = stop - start;
        self_slice->parent = (PyObject *)self; // Set parent to keep it alive

        // Increment the reference count of the parent
        Py_INCREF(self);
        return (PyObject *)self_slice;
    }
    else if (PyLong_Check(key)) { return Str_getitem(self, PyLong_AsSsize_t(key)); }
    else {
        PyErr_SetString(PyExc_TypeError, "Str indices must be integers or slices");
        return NULL;
    }
}

static int Str_getbuffer(Str *self, Py_buffer *view, int flags) {
    if (view == NULL) {
        PyErr_SetString(PyExc_ValueError, "NULL view in getbuffer");
        return -1;
    }

    static Py_ssize_t itemsize[1] = {1};
    view->obj = (PyObject *)self;
    view->buf = self->memory.start;
    view->len = self->memory.length;
    view->readonly = 1;
    view->itemsize = sizeof(char);
    view->format = "c"; // https://docs.python.org/3/library/struct.html#format-characters
    view->ndim = 1;
    view->shape = (Py_ssize_t *)&self->memory.length; // 1-D array, so shape is just a pointer to the length
    view->strides = itemsize;                         // strides in a 1-D array is just the item size
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(self);
    return 0;
}

static void Str_releasebuffer(PyObject *_, Py_buffer *view) {
    //! This function MUST NOT decrement view->obj, since that is done automatically
    //! in PyBuffer_Release() (this scheme is useful for breaking reference cycles).
    //! https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

/**
 *  @brief  Will be called by the `PySequence_Contains` to check presence of a substring.
 *  @return 1 if the string is present, 0 if it is not, -1 in case of error.
 *  @see    Docs: https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
 */
static int Str_in(Str *self, PyObject *needle_obj) {

    sz_string_view_t needle;
    if (!sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Unsupported needle layout");
        return -1;
    }

    return sz_find(self->memory.start, self->memory.length, needle.start, needle.length) != NULL;
}

static PyObject *Strs_get_tape(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_offsets_are_large(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_tape_address(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_offsets_address(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_tape_nbytes(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_offsets_nbytes(Str *self, void *closure) { return NULL; }

static Py_ssize_t Strs_len(Strs *self) {
    switch (self->layout) {
    case STRS_U32_TAPE: return self->data.u32_tape.count;
    case STRS_U32_TAPE_VIEW: return self->data.u32_tape_view.count;
    case STRS_U64_TAPE: return self->data.u64_tape.count;
    case STRS_U64_TAPE_VIEW: return self->data.u64_tape_view.count;
    case STRS_FRAGMENTED: return self->data.fragmented.count;
    default: return 0;
    }
}

static PyObject *Strs_getitem(Strs *self, Py_ssize_t i) {

    // Check for negative index and convert to positive
    Py_ssize_t count = Strs_len(self);
    if (i < 0) i += count;
    if (i < 0 || i >= count) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    PyObject *memory_owner = NULL;
    sz_cptr_t start = NULL;
    sz_size_t length = 0;
    getter(self, i, count, &memory_owner, &start, &length);

    // Create a new `Str` object
    Str *view_copy = (Str *)StrType.tp_alloc(&StrType, 0);
    if (view_copy == NULL && PyErr_NoMemory()) return NULL;

    view_copy->memory.start = start;
    view_copy->memory.length = length;
    view_copy->parent = memory_owner;
    Py_XINCREF(memory_owner);
    return view_copy;
}

/**
 *  This returns a `Strs` object of a potentially different layout:
 *  - `STRS_U32_TAPE_VIEW` input yields a `STRS_U32_TAPE_VIEW` for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_U64_TAPE_VIEW` input yields a `STRS_U64_TAPE_VIEW` for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_U32_TAPE` input yields a `STRS_U32_TAPE_VIEW`  for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_U64_TAPE` input yields a `STRS_U64_TAPE_VIEW`  for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_FRAGMENTED` input yields a `STRS_FRAGMENTED` output.
 */
static PyObject *Strs_subscript(Strs *self, PyObject *key) {

    if (PyLong_Check(key)) { return Strs_getitem(self, PyLong_AsSsize_t(key)); }

    if (!PySlice_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Strs indices must be integers or slices");
        return NULL;
    }

    // Sanity checks
    Py_ssize_t count = Strs_len(self);
    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
    Py_ssize_t result_count = PySlice_AdjustIndices(count, &start, &stop, step);
    if (result_count < 0) return NULL;

    // Create a new `Strs` object
    Strs *result = (Strs *)StrsType.tp_alloc(&StrsType, 0);
    if (result == NULL && PyErr_NoMemory()) return NULL;

    if (result_count == 0) {
        result->layout = STRS_FRAGMENTED;
        result->data.fragmented.count = 0;
        result->data.fragmented.spans = NULL;
        result->data.fragmented.parent = NULL;
        sz_memory_allocator_init_default(&result->data.fragmented.allocator);
        return (PyObject *)result;
    }

    // If a step is requested, we have to create a new `FRAGMENTED` instance of `Strs`,
    // even if the original one was a tape layout.
    if (step != 1) {
        sz_string_view_t *new_spans = (sz_string_view_t *)malloc(result_count * sizeof(sz_string_view_t));
        if (new_spans == NULL) {
            Py_XDECREF(result);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for fragmented spans");
            return NULL;
        }

        get_string_at_offset_t getter = str_at_offset_getter(self);
        result->layout = STRS_FRAGMENTED;
        result->data.fragmented.count = result_count;
        result->data.fragmented.spans = new_spans;
        result->data.fragmented.parent = NULL;
        sz_memory_allocator_init_default(&result->data.fragmented.allocator);

        // Populate the new fragmented array using `get_string_at_offset`
        sz_size_t j = 0;
        if (step > 0)
            for (Py_ssize_t i = start; i < stop; i += step, ++j) {
                getter(self, i, count, &result->data.fragmented.parent, &new_spans[j].start, &new_spans[j].length);
            }
        else
            for (Py_ssize_t i = start; i > stop; i += step, ++j) {
                getter(self, i, count, &result->data.fragmented.parent, &new_spans[j].start, &new_spans[j].length);
            }

        // Ensure the parent string isn't prematurely deallocated by this view.
        Py_XINCREF(result->data.fragmented.parent);
        return (PyObject *)result;
    }

    // For step=1, follow the docstring behavior:
    switch (self->layout) {

    case STRS_U32_TAPE_VIEW: {
        // STRS_U32_TAPE_VIEW input yields STRS_U32_TAPE_VIEW for step=1
        result->layout = STRS_U32_TAPE_VIEW;
        result->data.u32_tape_view.count = result_count;
        result->data.u32_tape_view.data = self->data.u32_tape_view.data + self->data.u32_tape_view.offsets[start];
        result->data.u32_tape_view.offsets = self->data.u32_tape_view.offsets + start;
        result->data.u32_tape_view.parent = self->data.u32_tape_view.parent;
        Py_INCREF(result->data.u32_tape_view.parent);
        break;
    }

    case STRS_U64_TAPE_VIEW: {
        // STRS_U64_TAPE_VIEW input yields STRS_U64_TAPE_VIEW for step=1
        result->layout = STRS_U64_TAPE_VIEW;
        result->data.u64_tape_view.count = result_count;
        result->data.u64_tape_view.data = self->data.u64_tape_view.data + self->data.u64_tape_view.offsets[start];
        result->data.u64_tape_view.offsets = self->data.u64_tape_view.offsets + start;
        result->data.u64_tape_view.parent = self->data.u64_tape_view.parent;
        Py_INCREF(result->data.u64_tape_view.parent);
        break;
    }

    case STRS_U32_TAPE: {
        // STRS_U32_TAPE input yields STRS_U32_TAPE_VIEW for step=1
        result->layout = STRS_U32_TAPE_VIEW;
        result->data.u32_tape_view.count = result_count;
        result->data.u32_tape_view.data = self->data.u32_tape.data + self->data.u32_tape.offsets[start];
        result->data.u32_tape_view.offsets = self->data.u32_tape.offsets + start;
        result->data.u32_tape_view.parent = (PyObject *)self;
        Py_INCREF((PyObject *)self);
        break;
    }

    case STRS_U64_TAPE: {
        // STRS_U64_TAPE input yields STRS_U64_TAPE_VIEW for step=1
        result->layout = STRS_U64_TAPE_VIEW;
        result->data.u64_tape_view.count = result_count;
        result->data.u64_tape_view.data = self->data.u64_tape.data + self->data.u64_tape.offsets[start];
        result->data.u64_tape_view.offsets = self->data.u64_tape.offsets + start;
        result->data.u64_tape_view.parent = (PyObject *)self;
        Py_INCREF((PyObject *)self);
        break;
    }

    case STRS_FRAGMENTED: {
        // STRS_FRAGMENTED input yields STRS_FRAGMENTED output
        result->layout = STRS_FRAGMENTED;
        result->data.fragmented.count = result_count;
        result->data.fragmented.parent = self->data.fragmented.parent;
        sz_memory_allocator_init_default(&result->data.fragmented.allocator);

        result->data.fragmented.spans = malloc(sizeof(sz_string_view_t) * result_count);
        if (result->data.fragmented.spans == NULL && PyErr_NoMemory()) {
            Py_XDECREF(result);
            return NULL;
        }
        sz_copy(result->data.fragmented.spans, self->data.fragmented.spans + start,
                sizeof(sz_string_view_t) * result_count);
        Py_INCREF(result->data.fragmented.parent);
        break;
    }

    default:
        // Unsupported layout
        PyErr_SetString(PyExc_TypeError, "Unsupported layout for conversion");
        Py_XDECREF(result);
        return NULL;
    }

    return (PyObject *)result;
}

/**
 *  @brief  Will be called by the `PySequence_Contains` to check the presence of a string in array.
 *  @return 1 if the string is present, 0 if it is not, -1 in case of error.
 *  @see    Docs: https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
 */
static int Strs_in(Str *self, PyObject *needle_obj) {

    // Validate and convert `needle`
    sz_string_view_t needle;
    if (!sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("The needle argument must be string-like");
        return -1;
    }

    // Depending on the layout, we will need to use different logic
    Py_ssize_t count = Strs_len(self);
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return -1;
    }

    // Time for a full-scan
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *parent = NULL;
        sz_cptr_t start = NULL;
        sz_size_t length = 0;
        getter(self, i, count, &parent, &start, &length);
        if (length == needle.length && sz_equal(start, needle.start, needle.length) == sz_true_k) return 1;
    }

    return 0;
}

static PyObject *Str_richcompare(PyObject *self, PyObject *other, int op) {

    sz_cptr_t a_start = NULL, b_start = NULL;
    sz_size_t a_length = 0, b_length = 0;
    if (!sz_py_export_string_like(self, &a_start, &a_length) || !sz_py_export_string_like(other, &b_start, &b_length))
        Py_RETURN_NOTIMPLEMENTED;

    int order = (int)sz_order(a_start, a_length, b_start, b_length);
    switch (op) {
    case Py_LT: return PyBool_FromLong(order < 0);
    case Py_LE: return PyBool_FromLong(order <= 0);
    case Py_EQ: return PyBool_FromLong(order == 0);
    case Py_NE: return PyBool_FromLong(order != 0);
    case Py_GT: return PyBool_FromLong(order > 0);
    case Py_GE: return PyBool_FromLong(order >= 0);
    default: Py_RETURN_NOTIMPLEMENTED;
    }
}

static PyObject *Strs_richcompare(PyObject *self, PyObject *other, int op) {

    Strs *a = (Strs *)self;
    Py_ssize_t a_length = Strs_len(a);
    get_string_at_offset_t a_getter = str_at_offset_getter(a);
    if (!a_getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    // If the other object is also a Strs, we can compare them much faster,
    // avoiding the CPython API entirely
    if (PyObject_TypeCheck(other, &StrsType)) {
        Strs *b = (Strs *)other;

        // Check if lengths are equal
        Py_ssize_t b_length = Strs_len(b);
        if (a_length != b_length) {
            if (op == Py_EQ) { Py_RETURN_FALSE; }
            if (op == Py_NE) { Py_RETURN_TRUE; }
        }

        // The second array may have a different layout
        get_string_at_offset_t b_getter = str_at_offset_getter(b);
        if (!b_getter) {
            PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
            return NULL;
        }

        // Check each item for equality
        Py_ssize_t min_length = sz_min_of_two(a_length, b_length);
        for (Py_ssize_t i = 0; i < min_length; i++) {
            PyObject *ai_parent = NULL, *bi_parent = NULL;
            sz_cptr_t ai_start = NULL, *bi_start = NULL;
            sz_size_t ai_length = 0, bi_length = 0;
            a_getter(a, i, a_length, &ai_parent, &ai_start, &ai_length);
            b_getter(b, i, b_length, &bi_parent, &bi_start, &bi_length);

            // When dealing with arrays, early exists make sense only in some cases
            int order = (int)sz_order(ai_start, ai_length, bi_start, bi_length);
            switch (op) {
            case Py_LT:
            case Py_LE:
                if (order > 0) { Py_RETURN_FALSE; }
                break;
            case Py_EQ:
                if (order != 0) { Py_RETURN_FALSE; }
                break;
            case Py_NE:
                if (order == 0) { Py_RETURN_TRUE; }
                break;
            case Py_GT:
            case Py_GE:
                if (order < 0) { Py_RETURN_FALSE; }
                break;
            default: break;
            }
        }

        // Prefixes are identical, compare lengths
        switch (op) {
        case Py_LT: return PyBool_FromLong(a_length < b_length);
        case Py_LE: return PyBool_FromLong(a_length <= b_length);
        case Py_EQ: return PyBool_FromLong(a_length == b_length);
        case Py_NE: return PyBool_FromLong(a_length != b_length);
        case Py_GT: return PyBool_FromLong(a_length > b_length);
        case Py_GE: return PyBool_FromLong(a_length >= b_length);
        default: Py_RETURN_NOTIMPLEMENTED;
        }
    }

    // The second argument is a sequence, but not a `Strs` object,
    // so we need to iterate through it.
    PyObject *other_iter = PyObject_GetIter(other);
    if (!other_iter) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "The second argument is not iterable");
        return NULL;
    }

    // We may not even know the length of the second sequence, so
    // let's just iterate as far as we can.
    Py_ssize_t i = 0;
    PyObject *other_item;
    for (; (other_item = PyIter_Next(other_iter)); ++i) {
        // Check if the second array is longer than the first
        if (a_length <= i) {
            Py_DECREF(other_item);
            Py_DECREF(other_iter);
            switch (op) {
            case Py_LT: Py_RETURN_TRUE;
            case Py_LE: Py_RETURN_TRUE;
            case Py_EQ: Py_RETURN_FALSE;
            case Py_NE: Py_RETURN_TRUE;
            case Py_GT: Py_RETURN_FALSE;
            case Py_GE: Py_RETURN_FALSE;
            default: Py_RETURN_NOTIMPLEMENTED;
            }
        }

        // Try unpacking the element from the second sequence
        sz_string_view_t bi;
        if (!sz_py_export_string_like(other_item, &bi.start, &bi.length)) {
            Py_DECREF(other_item);
            Py_DECREF(other_iter);
            wrap_current_exception("The second container must contain string-like objects");
            return NULL;
        }

        // Both sequences aren't exhausted yet
        PyObject *ai_parent = NULL;
        sz_cptr_t ai_start = NULL;
        sz_size_t ai_length = 0;
        a_getter(a, i, a_length, &ai_parent, &ai_start, &ai_length);

        // When dealing with arrays, early exists make sense only in some cases
        int order = (int)sz_order(ai_start, ai_length, bi.start, bi.length);
        switch (op) {
        case Py_LT:
        case Py_LE:
            if (order > 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_FALSE;
            }
            break;
        case Py_EQ:
            if (order != 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_FALSE;
            }
            break;
        case Py_NE:
            if (order == 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_TRUE;
            }
            break;
        case Py_GT:
        case Py_GE:
            if (order < 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_FALSE;
            }
            break;
        default: break;
        }
    }

    // The prefixes are equal and the second sequence is exhausted, but the first one may not be
    switch (op) {
    case Py_LT: return PyBool_FromLong(i < a_length);
    case Py_LE: Py_RETURN_TRUE;
    case Py_EQ: return PyBool_FromLong(i == a_length);
    case Py_NE: return PyBool_FromLong(i != a_length);
    case Py_GT: Py_RETURN_FALSE;
    case Py_GE: return PyBool_FromLong(i == a_length);
    default: Py_RETURN_NOTIMPLEMENTED;
    }
}

static char const doc_decode[] = //
    "Decode the bytes into a Unicode string with a given encoding.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  encoding (str, optional): The encoding to use (default is 'utf-8').\n"
    "  errors (str, optional): Error handling scheme (default is 'strict').\n"
    "Returns:\n"
    "  str: The decoded Unicode string.\n"
    "Raises:\n"
    "  UnicodeDecodeError: If decoding fails.";

static PyObject *Str_decode(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *encoding_obj = positional_args_count > !is_member + 0 ? args[!is_member + 0] : NULL;
    PyObject *errors_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "encoding") == 0 && !encoding_obj) { encoding_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "errors") == 0 && !errors_obj) { errors_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
        }
    }

    // Convert `encoding` and `errors` to `NULL` if they are `None`
    if (encoding_obj == Py_None) encoding_obj = NULL;
    if (errors_obj == Py_None) errors_obj = NULL;

    sz_string_view_t text, encoding, errors;
    if ((!sz_py_export_string_like(text_obj, &text.start, &text.length)) ||
        (encoding_obj && !sz_py_export_string_like(encoding_obj, &encoding.start, &encoding.length)) ||
        (errors_obj && !sz_py_export_string_like(errors_obj, &errors.start, &errors.length))) {
        wrap_current_exception("text, encoding, and errors must be string-like");
        return NULL;
    }

    if (encoding_obj == NULL) encoding = (sz_string_view_t) {"utf-8", 5};
    if (errors_obj == NULL) errors = (sz_string_view_t) {"strict", 6};

    // Python docs: https://docs.python.org/3/library/stdtypes.html#bytes.decode
    // CPython docs: https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_Decode
    return PyUnicode_Decode(text.start, text.length, encoding.start, errors.start);
}

static char const doc_write_to[] = //
    "Write the string to a file.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  filename (str): The file path to write to.\n"
    "Returns:\n"
    "  None.";

static PyObject *Str_write_to(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count != !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *path_obj = args[!is_member + 0];

    // Parse keyword arguments
    if (args_names_tuple) {
        PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument");
        return NULL;
    }

    sz_string_view_t text;
    sz_string_view_t path;

    // Validate and convert `text` and `path`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length) ||
        !sz_py_export_string_like(path_obj, &path.start, &path.length)) {
        wrap_current_exception("Text and path must be string-like");
        return NULL;
    }

    // There is a chance, the path isn't NULL-terminated, so copy it to a new buffer.
    // Many OSes have fairly low limit for the maximum path length.
    // On Windows its 260, but up to __around__ 32,767 characters are supported in extended API.
    // But it's better to be safe than sorry and use malloc :)
    //
    // https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
    // https://doc.owncloud.com/server/next/admin_manual/troubleshooting/path_filename_length.html
    sz_ptr_t path_buffer = (sz_ptr_t)malloc(path.length + 1);
    if (path_buffer == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the path");
        return NULL;
    }
    sz_copy(path_buffer, path.start, path.length);
    path_buffer[path.length] = '\0';

    // Unlock the Global Interpreter Lock (GIL) to allow other threads to run
    // while the current thread is waiting for the file to be written.
    PyThreadState *gil_state = PyEval_SaveThread();
    FILE *file_pointer = fopen(path_buffer, "wb");
    if (file_pointer == NULL) {
        PyEval_RestoreThread(gil_state);
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path_buffer);
        free(path_buffer);
        PyEval_RestoreThread(gil_state);
        return NULL;
    }

    setbuf(file_pointer, NULL); // Set the stream to unbuffered
    int status = fwrite(text.start, 1, text.length, file_pointer);
    PyEval_RestoreThread(gil_state);
    if (status != (Py_ssize_t)text.length) {
        PyErr_SetFromErrnoWithFilename(PyExc_OSError, path_buffer);
        free(path_buffer);
        fclose(file_pointer);
        return NULL;
    }

    free(path_buffer);
    fclose(file_pointer);
    Py_RETURN_NONE;
}

static char const doc_offset_within[] = //
    "Return the raw byte offset of this StringZilla string within a larger StringZilla string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The substring.\n"
    "  larger (Str): The larger string to search within.\n"
    "Returns:\n"
    "  int: The byte offset where 'self' is found within 'larger', or -1 if not found.";

static PyObject *Str_offset_within(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count != !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *slice_obj = is_member ? self : args[0];
    PyObject *text_obj = args[!is_member + 0];

    // Parse keyword arguments
    if (args_names_tuple) {
        PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument");
        return NULL;
    }

    sz_string_view_t text;
    sz_string_view_t slice;

    // Validate and convert `text` and `slice`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length) ||
        !sz_py_export_string_like(slice_obj, &slice.start, &slice.length)) {
        wrap_current_exception("Text and slice must be string-like");
        return NULL;
    }

    if (slice.start < text.start || slice.start + slice.length > text.start + text.length) {
        PyErr_SetString(PyExc_ValueError, "The slice is not within the text bounds");
        return NULL;
    }

    return PyLong_FromSize_t((sz_size_t)(slice.start - text.start));
}

/**
 *  @brief  Implementation function for all search-like operations, parameterized by a function callback.
 *  @return 1 on success, 0 on failure.
 */
static int Str_find_implementation_( //
    PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count, PyObject *args_names_tuple,
    sz_find_t finder, sz_bool_t is_reverse, Py_ssize_t *offset_out, sz_string_view_t *haystack_out,
    sz_string_view_t *needle_out) {

    // Fast path variables
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // needle is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return 0;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return 0;
    }

    // Fast positional argument extraction
    if (is_member) {
        haystack_obj = self;
        if (positional_args_count >= 1) needle_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
    }
    else {
        if (positional_args_count >= 1) haystack_obj = args[0];
        if (positional_args_count >= 2) needle_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return 0;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return 0;
                }
                end_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return 0;
            }
        }
    }

    sz_string_view_t haystack;
    sz_string_view_t needle;
    Py_ssize_t start, end;

    // Validate and convert `haystack` and `needle`
    if (!sz_py_export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Haystack and needle must be string-like");
        return 0;
    }

    // Validate and convert `start`
    if (start_obj) {
        start = PyLong_AsSsize_t(start_obj);
        if (start == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The start argument must be an integer");
            return 0;
        }
    }
    else { start = 0; }

    // Validate and convert `end`
    if (end_obj) {
        end = PyLong_AsSsize_t(end_obj);
        if (end == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The end argument must be an integer");
            return 0;
        }
    }
    else { end = PY_SSIZE_T_MAX; }

    // Limit the `haystack` range
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    // If the needle length is zero, the result is start index in normal order or end index in reverse order
    if (needle.length == 0) {
        *offset_out = !is_reverse ? normalized_offset : (normalized_offset + normalized_length);
        *haystack_out = haystack;
        *needle_out = needle;
        return 1;
    }

    // Perform contains operation
    sz_cptr_t match = finder(haystack.start, haystack.length, needle.start, needle.length);
    if (match == NULL) { *offset_out = -1; }
    else { *offset_out = (Py_ssize_t)(match - haystack.start + normalized_offset); }

    *haystack_out = haystack;
    *needle_out = needle;
    return 1;
}

static char const doc_contains[] = //
    "Check if a string contains a substring.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  substring (str): The substring to search for.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  bool: True if the substring is found, False otherwise.";

static PyObject *Str_contains(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find, sz_false_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) { Py_RETURN_FALSE; }
    else { Py_RETURN_TRUE; }
}

static char const doc_find[] = //
    "Find the first occurrence of a substring.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  substring (str): The substring to find.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  int: The index of the first occurrence, or -1 if not found.";

static PyObject *Str_find(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find, sz_false_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_index[] = //
    "Find the first occurrence of a substring or raise an error if not found.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  substring (str): The substring to find.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  int: The index of the first occurrence.\n"
    "Raises:\n"
    "  ValueError: If the substring is not found.";

static PyObject *Str_index(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                           PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find, sz_false_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_rfind[] = //
    "Find the last occurrence of a substring.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  substring (str): The substring to find.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  int: The index of the last occurrence, or -1 if not found.";

static PyObject *Str_rfind(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                           PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind, sz_true_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_rindex[] = //
    "Find the last occurrence of a substring or raise an error if not found.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  substring (str): The substring to find.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  int: The index of the last occurrence.\n"
    "Raises:\n"
    "  ValueError: If the substring is not found.";

static PyObject *Str_rindex(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind, sz_true_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_partition_implementation_(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple, sz_find_t finder, sz_bool_t is_reverse) {
    Py_ssize_t separator_index;
    sz_string_view_t text;
    sz_string_view_t separator;
    PyObject *result_tuple;

    // Use `Str_find_implementation_` to get the index of the separator
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, finder, is_reverse,
                                  &separator_index, &text, &separator))
        return NULL;

    // If the separator length is zero, we must raise a `ValueError`
    if (separator.length == 0) {
        PyErr_SetString(PyExc_ValueError, "empty separator");
        return NULL;
    }

    // If separator is not found, return a tuple (self, "", "")
    if (separator_index == -1) {
        PyObject *empty_str1 = Str_new(&StrType, Py_None, Py_None);
        PyObject *empty_str2 = Str_new(&StrType, Py_None, Py_None);

        result_tuple = PyTuple_New(3);
        Py_INCREF(self);
        PyTuple_SET_ITEM(result_tuple, 0, self);
        PyTuple_SET_ITEM(result_tuple, 1, empty_str1);
        PyTuple_SET_ITEM(result_tuple, 2, empty_str2);
        return result_tuple;
    }

    // Create the three parts manually
    Str *before = Str_new(&StrType, NULL, NULL);
    Str *middle = Str_new(&StrType, NULL, NULL);
    Str *after = Str_new(&StrType, NULL, NULL);

    before->parent = self, before->memory.start = text.start, before->memory.length = separator_index;
    middle->parent = self, middle->memory.start = text.start + separator_index,
    middle->memory.length = separator.length;
    after->parent = self, after->memory.start = text.start + separator_index + separator.length,
    after->memory.length = text.length - separator_index - separator.length;

    // All parts reference the same parent
    Py_INCREF(self);
    Py_INCREF(self);
    Py_INCREF(self);

    // Build the result tuple
    result_tuple = PyTuple_New(3);
    PyTuple_SET_ITEM(result_tuple, 0, before);
    PyTuple_SET_ITEM(result_tuple, 1, middle);
    PyTuple_SET_ITEM(result_tuple, 2, after);

    return result_tuple;
}

static char const doc_partition[] = //
    "Split the string into a 3-tuple around the first occurrence of a separator.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to partition by.\n"
    "Returns:\n"
    "  tuple: A 3-tuple (head, separator, tail). If the separator is not found, returns (self, '', '').";

static PyObject *Str_partition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {
    return Str_partition_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find, sz_false_k);
}

static char const doc_rpartition[] = //
    "Split the string into a 3-tuple around the last occurrence of a separator.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to partition by.\n"
    "Returns:\n"
    "  tuple: A 3-tuple (head, separator, tail). If the separator is not found, returns ('', '', self).";

static PyObject *Str_rpartition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    return Str_partition_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind, sz_true_k);
}

static char const doc_count[] = //
    "Count the occurrences of a substring.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  substring (str): The substring to count.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "  allowoverlap (bool, optional): Count overlapping occurrences (default is False).\n"
    "Returns:\n"
    "  int: The number of occurrences of the substring.";

static PyObject *Str_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                           PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *allowoverlap_obj = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // needle is required
    Py_ssize_t const expected_max = expected_min + 3;  // + start + end + allowoverlap

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        haystack_obj = self;
        if (positional_args_count >= 1) needle_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
        if (positional_args_count >= 4) allowoverlap_obj = args[3];
    }
    else {
        if (positional_args_count >= 1) haystack_obj = args[0];
        if (positional_args_count >= 2) needle_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
        if (positional_args_count >= 5) allowoverlap_obj = args[4];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "allowoverlap") == 0) {
                if (allowoverlap_obj) {
                    PyErr_SetString(PyExc_TypeError, "allowoverlap specified twice");
                    return NULL;
                }
                allowoverlap_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    sz_string_view_t haystack;
    sz_string_view_t needle;
    Py_ssize_t start = start_obj ? PyLong_AsSsize_t(start_obj) : 0;
    Py_ssize_t end = end_obj ? PyLong_AsSsize_t(end_obj) : PY_SSIZE_T_MAX;
    int allowoverlap = allowoverlap_obj ? PyObject_IsTrue(allowoverlap_obj) : 0;

    if (!sz_py_export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Haystack and needle must be string-like");
        return NULL;
    }

    if ((start == -1 || end == -1 || allowoverlap == -1) && PyErr_Occurred()) return NULL;

    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    sz_size_t count = 0;
    if (needle.length == 0 || haystack.length == 0 || haystack.length < needle.length) { count = 0; }
    else if (allowoverlap) {
        while (haystack.length) {
            sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + found;
            haystack.length -= offset + found;
        }
    }
    else {
        while (haystack.length) {
            sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + needle.length;
            haystack.length -= offset + needle.length * found;
        }
    }

    return PyLong_FromSize_t(count);
}

static char const doc_startswith[] = //
    "Check if a string starts with a given prefix.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  prefix (str): The prefix to check.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  bool: True if the string starts with the prefix, False otherwise.";

static PyObject *Str_startswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *str_obj = NULL;
    PyObject *prefix_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // prefix is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        str_obj = self;
        if (positional_args_count >= 1) prefix_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
    }
    else {
        if (positional_args_count >= 1) str_obj = args[0];
        if (positional_args_count >= 2) prefix_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str, prefix;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length) ||
        !sz_py_export_string_like(prefix_obj, &prefix.start, &prefix.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && (sz_size_t)(end - start) < str.length) { str.length = (sz_size_t)(end - start); }

    if (str.length < prefix.length) { Py_RETURN_FALSE; }
    else if (strncmp(str.start, prefix.start, prefix.length) == 0) { Py_RETURN_TRUE; }
    else { Py_RETURN_FALSE; }
}

static char const doc_endswith[] = //
    "Check if a string ends with a given suffix.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  suffix (str): The suffix to check.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  bool: True if the string ends with the suffix, False otherwise.";

static PyObject *Str_endswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *str_obj = NULL;
    PyObject *suffix_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // suffix is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        str_obj = self;
        if (positional_args_count >= 1) suffix_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
    }
    else {
        if (positional_args_count >= 1) str_obj = args[0];
        if (positional_args_count >= 2) suffix_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str, suffix;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length) ||
        !sz_py_export_string_like(suffix_obj, &suffix.start, &suffix.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && (sz_size_t)(end - start) < str.length) { str.length = (sz_size_t)(end - start); }

    if (str.length < suffix.length) { Py_RETURN_FALSE; }
    else if (strncmp(str.start + (str.length - suffix.length), suffix.start, suffix.length) == 0) { Py_RETURN_TRUE; }
    else { Py_RETURN_FALSE; }
}

static char const doc_translate[] = //
    "Perform transformation of a string using a look-up table.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  table (str or dict): A 256-character string or a dictionary mapping bytes to bytes.\n"
    "  inplace (bool, optional): If True, the string is modified in place (default is False).\n"
    "\n"
    "  start (int, optional): The starting index for translation (default is 0).\n"
    "  end (int, optional): The ending index for translation (default is the string length).\n"
    "Returns:\n"
    "  Union[None, str, bytes]: If inplace is False, a new string is returned, otherwise None.\n"
    "Raises:\n"
    "  ValueError: If the table is not 256 bytes long.\n"
    "  TypeError: If the table is not a string or dictionary.";

static PyObject *Str_translate(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member + 1 || positional_args_count > !is_member + 4) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : args[0];
    PyObject *look_up_table_obj = args[!is_member];
    PyObject *inplace_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *start_obj = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;
    PyObject *end_obj = positional_args_count > !is_member + 3 ? args[!is_member + 3] : NULL;

    // Optional keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "inplace") == 0 && !inplace_obj) { inplace_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "start") == 0 && !start_obj) { start_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0 && !end_obj) { end_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
        }
    }

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }

    sz_string_view_t look_up_table_str;
    SZ_ALIGN64 char look_up_table[256];
    if (PyDict_Check(look_up_table_obj)) {

        // If any character is not defined, it will be replaced with itself:
        for (int i = 0; i < 256; i++) look_up_table[i] = (char)i;

        // Process the dictionary into the look-up table
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(look_up_table_obj, &pos, &key, &value)) {
            if (!PyUnicode_Check(key) || PyUnicode_GetLength(key) != 1 || !PyUnicode_Check(value) ||
                PyUnicode_GetLength(value) != 1) {
                PyErr_SetString(PyExc_TypeError, "Keys and values must be single characters");
                return NULL;
            }

            char key_char = PyUnicode_AsUTF8(key)[0];
            char value_char = PyUnicode_AsUTF8(value)[0];
            look_up_table[(unsigned char)key_char] = value_char;
        }
    }
    else if (sz_py_export_string_like(look_up_table_obj, &look_up_table_str.start, &look_up_table_str.length)) {
        if (look_up_table_str.length != 256) {
            PyErr_SetString(PyExc_ValueError, "The look-up table must be exactly 256 bytes long");
            return NULL;
        }
        sz_copy(&look_up_table[0], look_up_table_str.start, look_up_table_str.length);
    }
    else {
        wrap_current_exception("The look-up table must be string-like or a dictionary");
        return NULL;
    }

    int is_inplace = inplace_obj ? PyObject_IsTrue(inplace_obj) : 0;
    if (is_inplace == -1) {
        PyErr_SetString(PyExc_TypeError, "The inplace argument must be a boolean");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && (sz_size_t)(end - start) < str.length) { str.length = (sz_size_t)(end - start); }

    // Perform the translation using the look-up table
    if (is_inplace) {
        sz_lookup(str.start, str.length, str.start, look_up_table);
        Py_RETURN_NONE;
    }
    // Allocate a string of the same size, get it's raw pointer and transform the data into it
    else {

        // For binary inputs return bytes, for unicode return str
        if (PyUnicode_Check(str_obj)) {
            // Create a new Unicode object
            PyObject *new_unicode_obj = PyUnicode_New(str.length, PyUnicode_MAX_CHAR_VALUE(str_obj));
            if (!new_unicode_obj) {
                PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for new Unicode string");
                return NULL;
            }

            sz_ptr_t new_buffer = (sz_ptr_t)PyUnicode_DATA(new_unicode_obj);
            sz_lookup(new_buffer, str.length, str.start, look_up_table);
            return new_unicode_obj;
        }
        else {
            PyObject *new_bytes_obj = PyBytes_FromStringAndSize(NULL, str.length);
            if (!new_bytes_obj) {
                PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for new string");
                return NULL;
            }

            // Get the buffer and perform the transformation
            sz_ptr_t new_buffer = (sz_ptr_t)PyBytes_AS_STRING(new_bytes_obj);
            sz_lookup(new_buffer, str.length, str.start, look_up_table);
            return new_bytes_obj;
        }
    }
}

static char const doc_find_first_of[] = //
    "Find the index of the first occurrence of any character from another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to search for.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the first matching character, or -1 if none found.";

static PyObject *Str_find_first_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find_byte_from, sz_false_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_find_first_not_of[] = //
    "Find the index of the first character not in another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to exclude.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the first non-matching character, or -1 if all match.";

static PyObject *Str_find_first_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find_byte_not_from,
                                  sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_find_last_of[] = //
    "Find the index of the last occurrence of any character from another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to search for.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the last matching character, or -1 if none found.";

static PyObject *Str_find_last_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind_byte_from, sz_true_k,
                                  &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_find_last_not_of[] = //
    "Find the index of the last character not in another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to exclude.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the last non-matching character, or -1 if all match.";

static PyObject *Str_find_last_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind_byte_not_from,
                                  sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

/**
 *  @brief  Given parsed split settings, constructs an iterator that would produce that split.
 */
static SplitIterator *Str_split_iter_(PyObject *text_obj, PyObject *separator_obj,                   //
                                      sz_string_view_t const text, sz_string_view_t const separator, //
                                      int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length,
                                      sz_bool_t is_reverse) {

    // Create a new `SplitIterator` object
    SplitIterator *result_obj = (SplitIterator *)SplitIteratorType.tp_alloc(&SplitIteratorType, 0);
    if (result_obj == NULL && PyErr_NoMemory()) return NULL;

    // Set its properties based on the slice
    result_obj->text_obj = text_obj;
    result_obj->separator_obj = separator_obj;
    result_obj->text = text;
    result_obj->separator = separator;
    result_obj->finder = finder;

    result_obj->match_length = match_length;
    result_obj->include_match = keepseparator;
    result_obj->is_reverse = is_reverse;
    result_obj->max_parts = (sz_size_t)maxsplit + 1;
    result_obj->reached_tail = 0;

    // Increment the reference count of the parent
    Py_INCREF(result_obj->text_obj);
    Py_XINCREF(result_obj->separator_obj);
    return result_obj;
}

/**
 *  @brief  Implements the normal order split logic for both string-delimiters and character sets.
 *          Produces a `Strs` object with `REORDERED_SUBVIEWS` layout.
 */
static Strs *Str_split_(PyObject *parent_string, sz_string_view_t const text, sz_string_view_t const separator,
                        int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length) {
    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) return NULL;

    // Use reordered subviews layout with the haystack as parent
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.parent = parent_string;
    sz_memory_allocator_init_default(&result->data.fragmented.allocator);

    // Collect split positions first
    sz_string_view_t *spans = NULL;
    sz_size_t spans_capacity = 4;
    sz_size_t spans_count = 0;

    spans = (sz_string_view_t *)malloc(spans_capacity * sizeof(sz_string_view_t));
    if (!spans) {
        Py_XDECREF(result);
        PyErr_NoMemory();
        return NULL;
    }

    sz_cptr_t current_start = text.start;
    sz_size_t remaining_length = text.length;
    sz_size_t splits_made = 0;
    sz_size_t max_splits = (maxsplit < 0) ? SIZE_MAX : (sz_size_t)maxsplit;

    while (remaining_length > 0 && splits_made < max_splits) {
        sz_cptr_t match = finder(current_start, remaining_length, separator.start, separator.length);

        if (match) {
            // Add the part before the separator
            sz_size_t part_length = match - current_start;

            // Reallocate spans array if needed
            if (spans_count >= spans_capacity) {
                spans_capacity *= 2;
                sz_string_view_t *new_spans =
                    (sz_string_view_t *)realloc(spans, spans_capacity * sizeof(sz_string_view_t));
                if (!new_spans) {
                    free(spans);
                    Py_XDECREF(result);
                    PyErr_NoMemory();
                    return NULL;
                }
                spans = new_spans;
            }

            spans[spans_count].start = current_start;
            spans[spans_count].length = keepseparator ? part_length + match_length : part_length;
            spans_count++;

            // Move past the separator
            current_start = match + match_length;
            remaining_length = text.length - (current_start - text.start);
            splits_made++;
        }
        else { break; }
    }

    // Add the final part (everything remaining)
    if (spans_count >= spans_capacity) {
        spans_capacity++;
        sz_string_view_t *new_spans = (sz_string_view_t *)realloc(spans, spans_capacity * sizeof(sz_string_view_t));
        if (!new_spans) {
            free(spans);
            Py_XDECREF(result);
            PyErr_NoMemory();
            return NULL;
        }
        spans = new_spans;
    }

    spans[spans_count].start = current_start;
    spans[spans_count].length = remaining_length;
    spans_count++;

    // Set up the result
    result->data.fragmented.spans = spans;
    result->data.fragmented.count = spans_count;
    Py_INCREF(parent_string);

    return result;
}

/**
 *  @brief  Implements the reverse order split logic for both string-delimiters and character sets.
 *          Produces a `Strs` object with `REORDERED_SUBVIEWS` layout.
 */
static Strs *Str_rsplit_(PyObject *parent_string, sz_string_view_t const text, sz_string_view_t const separator,
                         int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length) {
    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) return NULL;

    // Use reordered subviews layout with the haystack as parent
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.parent = parent_string;
    sz_memory_allocator_init_default(&result->data.fragmented.allocator);
    result->data.fragmented.spans = NULL;
    result->data.fragmented.count = 0;

    // Keep track of the memory usage
    sz_string_view_t *parts = NULL;
    sz_size_t parts_capacity = 4;
    sz_size_t parts_count = 0;

    parts = (sz_string_view_t *)malloc(parts_capacity * sizeof(sz_string_view_t));
    if (!parts) {
        Py_XDECREF(result);
        PyErr_NoMemory();
        return NULL;
    }

    sz_bool_t reached_tail = 0;
    sz_size_t total_skipped = 0;
    sz_size_t max_parts = (maxsplit < 0) ? SIZE_MAX : ((sz_size_t)maxsplit + 1);

    while (!reached_tail) {
        sz_cptr_t match = parts_count + 1 < max_parts
                              ? finder(text.start, text.length - total_skipped, separator.start, separator.length)
                              : NULL;

        // Determine the next part
        sz_string_view_t part;
        if (match) {
            part.start = match + match_length * !keepseparator;
            part.length = text.start + text.length - total_skipped - part.start;
            total_skipped = text.start + text.length - match;
        }
        else {
            part.start = text.start;
            part.length = text.length - total_skipped;
            reached_tail = 1;
        }

        // Reallocate parts array if needed
        if (parts_count >= parts_capacity) {
            parts_capacity *= 2;
            sz_string_view_t *new_parts = (sz_string_view_t *)realloc(parts, parts_capacity * sizeof(sz_string_view_t));
            if (!new_parts) {
                free(parts);
                Py_XDECREF(result);
                PyErr_NoMemory();
                return NULL;
            }
            parts = new_parts;
        }

        // Populate the parts array
        parts[parts_count] = part;
        parts_count++;
    }

    // Python does this weird thing, where the `rsplit` results appear in the same order as `split`
    // so we need to reverse the order of elements in the `parts` array.
    for (sz_size_t i = 0; i < parts_count / 2; i++) {
        sz_string_view_t temp = parts[i];
        parts[i] = parts[parts_count - i - 1];
        parts[parts_count - i - 1] = temp;
    }

    result->data.fragmented.spans = parts;
    result->data.fragmented.count = parts_count;
    Py_INCREF(parent_string);
    return result;
}

/**
 *  @brief  Proxy routing requests like `Str.split`, `Str.rsplit`, `Str.split_byteset` and `Str.rsplit_byteset`
 *          to `Str_split_` and `Str_rsplit_` implementations, parsing function arguments.
 */
static PyObject *Str_split_with_known_callback(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple,               //
                                               sz_find_t finder, sz_size_t match_length, //
                                               sz_bool_t is_reverse, sz_bool_t is_lazy_iterator) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t expected_min_args = !is_member;
    Py_ssize_t expected_max_args = !is_member + 3;
    if (positional_args_count < expected_min_args || positional_args_count > expected_max_args) {
        PyErr_SetString(PyExc_TypeError, "sz.split() received unsupported number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *separator_obj = positional_args_count > !is_member + 0 ? args[!is_member + 0] : NULL;
    PyObject *maxsplit_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *keepseparator_obj = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "separator") == 0 && !separator_obj) { separator_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0 && !maxsplit_obj) { maxsplit_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "keepseparator") == 0 && !keepseparator_obj) {
                keepseparator_obj = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
        }
    }

    sz_string_view_t text;
    sz_string_view_t separator;
    int keepseparator;
    Py_ssize_t maxsplit;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `separator`
    if (separator_obj) {
        if (!sz_py_export_string_like(separator_obj, &separator.start, &separator.length)) {
            wrap_current_exception("The separator argument must be string-like");
            return NULL;
        }
        // Raise a `ValueError` if it's length is zero, like the native `str.split`
        if (separator.length == 0) {
            PyErr_SetString(PyExc_ValueError, "The separator argument must not be empty");
            return NULL;
        }
        if (match_length == 0) match_length = separator.length;
    }
    else {
        separator.start = " ";
        match_length = separator.length = 1;
    }

    // Validate and convert `keepseparator`
    if (keepseparator_obj) {
        keepseparator = PyObject_IsTrue(keepseparator_obj);
        if (keepseparator == -1) {
            PyErr_SetString(PyExc_TypeError, "The keepseparator argument must be a boolean");
            return NULL;
        }
    }
    else { keepseparator = 0; }

    // Validate and convert `maxsplit`
    if (maxsplit_obj) {
        maxsplit = PyLong_AsSsize_t(maxsplit_obj);
        if (maxsplit == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The maxsplit argument must be an integer");
            return NULL;
        }
    }
    else { maxsplit = PY_SSIZE_T_MAX; }

    // Dispatch the right backend
    if (is_lazy_iterator)
        return Str_split_iter_(text_obj, separator_obj, text, separator, //
                               keepseparator, maxsplit, finder, match_length, is_reverse);
    else
        return !is_reverse ? Str_split_(text_obj, text, separator, keepseparator, maxsplit, finder, match_length)
                           : Str_rsplit_(text_obj, text, separator, keepseparator, maxsplit, finder, match_length);
}

static char const doc_split[] = //
    "Split a string by a separator.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include the separator in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the separator.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_split(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                           PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_find, 0, sz_false_k,
                                         sz_false_k);
}

static char const doc_rsplit[] = //
    "Split a string by a separator starting from the end.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include the separator in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the separator.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_rsplit(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_rfind, 0, sz_true_k,
                                         sz_false_k);
}

static char const doc_split_byteset[] = //
    "Split a string by a set of character separators.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the character set.";

static PyObject *Str_split_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_find_byte_from, 1,
                                         sz_false_k, sz_false_k);
}

static char const doc_rsplit_byteset[] = //
    "Split a string by a set of character separators in reverse order.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the character set.";

static PyObject *Str_rsplit_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_rfind_byte_from, 1,
                                         sz_true_k, sz_false_k);
}

static char const doc_split_iter[] = //
    "Create an iterator for splitting a string by a separator.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  keepseparator (bool, optional): Include separator in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_split_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_find, 0, sz_false_k,
                                         sz_true_k);
}

static char const doc_rsplit_iter[] = //
    "Create an iterator for splitting a string by a separator in reverse order.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  keepseparator (bool, optional): Include separator in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings in reverse.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_rsplit_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_rfind, 0, sz_true_k,
                                         sz_true_k);
}

static char const doc_split_byteset_iter[] = //
    "Create an iterator for splitting a string by a set of character separators.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings.";

static PyObject *Str_split_byteset_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_find_byte_from, 1,
                                         sz_false_k, sz_true_k);
}

static char const doc_rsplit_byteset_iter[] = //
    "Create an iterator for splitting a string by a set of character separators in reverse order.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings in reverse.";

static PyObject *Str_rsplit_byteset_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, &sz_rfind_byte_from, 1,
                                         sz_true_k, sz_true_k);
}

static char const doc_splitlines[] = //
    "Split a string by line breaks.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  keeplinebreaks (bool, optional): Include line breaks in the results (default is False).\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "Returns:\n"
    "  Strs: A list of strings split by line breaks.";

static PyObject *Str_splitlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 2) {
        PyErr_SetString(PyExc_TypeError, "splitlines() requires at least 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *keeplinebreaks_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *maxsplit_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "keeplinebreaks") == 0 && !keeplinebreaks_obj) {
                keeplinebreaks_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0 && !maxsplit_obj) { maxsplit_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_string_view_t text;
    int keeplinebreaks;
    Py_ssize_t maxsplit = PY_SSIZE_T_MAX; // Default value for maxsplit

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `keeplinebreaks`
    if (keeplinebreaks_obj) {
        keeplinebreaks = PyObject_IsTrue(keeplinebreaks_obj);
        if (keeplinebreaks == -1) {
            wrap_current_exception("The keeplinebreaks argument must be a boolean");
            return NULL;
        }
    }
    else { keeplinebreaks = 0; }

    // Validate and convert `maxsplit`
    if (maxsplit_obj) {
        maxsplit = PyLong_AsSsize_t(maxsplit_obj);
        if (maxsplit == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The maxsplit argument must be an integer");
            return NULL;
        }
    }

    // The Unicode standard defines a number of characters that conforming applications
    // should recognize as line terminators:
    //
    //      LF:    Line Feed, U+000A                            - 1 byte (\n)
    //      VT:    Vertical Tab, U+000B                         - 1 byte (\v)
    //      FF:    Form Feed, U+000C                            - 1 byte (\f)
    //      CR:    Carriage Return, U+000D                      - 1 byte (\r)
    //      NEL:   Next Line, U+0085                            - 1 byte (\x85)
    //      LS:    Line Separator, U+2028                       - 2 bytes
    //      PS:    Paragraph Separator, U+2029                  - 2 bytes
    //      CR+LF: CR (U+000D) followed by LF (U+000A)          - 2 bytes
    //
    // The Python standard is different, it also includes:
    //
    //     FS:    File Separator, U+001C                       - 1 byte (\x1C)
    //     GS:    Group Separator, U+001D                      - 1 byte (\x1D)
    //     RS:    Record Separator, U+001E                     - 1 byte (\x1E)
    //
    // We avoid all 2-byte sequences and only consider 1-byte delimiters.
    // CPython docs: https://docs.python.org/3/library/stdtypes.html#str.splitlines
    sz_string_view_t separator;
    separator.start = "\x0A\x0B\x0C\x0D\x85\x1C\x1D\x1E";
    separator.length = 8;
    return Str_split_(text_obj, text, separator, keeplinebreaks, maxsplit, &sz_find_byte_from, 1);
}

static PyObject *Str_concat(PyObject *self, PyObject *other) {
    struct sz_string_view_t self_str, other_str;

    // Validate and convert `self` and `other`
    if (!sz_py_export_string_like(self, &self_str.start, &self_str.length) ||
        !sz_py_export_string_like(other, &other_str.start, &other_str.length)) {
        wrap_current_exception("Both operands must be string-like");
        return NULL;
    }

    // Allocate a new Str instance
    Str *result_str = PyObject_New(Str, &StrType);
    if (result_str == NULL) { return NULL; }

    // Calculate the total length of the new string
    result_str->parent = NULL;
    result_str->memory.length = self_str.length + other_str.length;

    // Allocate memory for the new string
    result_str->memory.start = malloc(result_str->memory.length);
    if (result_str->memory.start == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for string concatenation");
        return NULL;
    }

    // Perform the string concatenation
    sz_copy(result_str->memory.start, self_str.start, self_str.length);
    sz_copy(result_str->memory.start + self_str.length, other_str.start, other_str.length);

    return (PyObject *)result_str;
}

static PySequenceMethods Str_as_sequence = {
    .sq_length = Str_len,   //
    .sq_item = Str_getitem, //
    .sq_contains = Str_in,  //
};

static PyMappingMethods Str_as_mapping = {
    .mp_length = Str_len,          //
    .mp_subscript = Str_subscript, // Is used to implement slices in Python
};

static PyBufferProcs Str_as_buffer = {
    .bf_getbuffer = Str_getbuffer,
    .bf_releasebuffer = Str_releasebuffer,
};

static PyNumberMethods Str_as_number = {
    .nb_add = Str_concat,
};

static PyGetSetDef Str_getsetters[] = {
    // Compatibility with PyArrow
    {"address", (getter)Str_get_address, NULL, "Get the memory address of the first byte of the string", NULL},
    {"nbytes", (getter)Str_get_nbytes, NULL, "Get the length of the string in bytes", NULL},
    {NULL} // Sentinel
};

#define SZ_METHOD_FLAGS METH_FASTCALL | METH_KEYWORDS

static PyMethodDef Str_methods[] = {
    {"contains", (PyCFunction)Str_contains, SZ_METHOD_FLAGS, doc_contains},
    {"count", (PyCFunction)Str_count, SZ_METHOD_FLAGS, doc_count},
    {"splitlines", (PyCFunction)Str_splitlines, SZ_METHOD_FLAGS, doc_splitlines},
    {"startswith", (PyCFunction)Str_startswith, SZ_METHOD_FLAGS, doc_startswith},
    {"endswith", (PyCFunction)Str_endswith, SZ_METHOD_FLAGS, doc_endswith},
    {"translate", (PyCFunction)Str_translate, SZ_METHOD_FLAGS, doc_translate},
    {"decode", (PyCFunction)Str_decode, SZ_METHOD_FLAGS, doc_decode},

    // Bidirectional operations
    {"find", (PyCFunction)Str_find, SZ_METHOD_FLAGS, doc_find},
    {"index", (PyCFunction)Str_index, SZ_METHOD_FLAGS, doc_index},
    {"partition", (PyCFunction)Str_partition, SZ_METHOD_FLAGS, doc_partition},
    {"split", (PyCFunction)Str_split, SZ_METHOD_FLAGS, doc_split},
    {"rfind", (PyCFunction)Str_rfind, SZ_METHOD_FLAGS, doc_rfind},
    {"rindex", (PyCFunction)Str_rindex, SZ_METHOD_FLAGS, doc_rindex},
    {"rpartition", (PyCFunction)Str_rpartition, SZ_METHOD_FLAGS, doc_rpartition},
    {"rsplit", (PyCFunction)Str_rsplit, SZ_METHOD_FLAGS, doc_rsplit},

    // Character search extensions
    {"find_first_of", (PyCFunction)Str_find_first_of, SZ_METHOD_FLAGS, doc_find_first_of},
    {"find_last_of", (PyCFunction)Str_find_last_of, SZ_METHOD_FLAGS, doc_find_last_of},
    {"find_first_not_of", (PyCFunction)Str_find_first_not_of, SZ_METHOD_FLAGS, doc_find_first_not_of},
    {"find_last_not_of", (PyCFunction)Str_find_last_not_of, SZ_METHOD_FLAGS, doc_find_last_not_of},
    {"split_byteset", (PyCFunction)Str_split_byteset, SZ_METHOD_FLAGS, doc_split_byteset},
    {"rsplit_byteset", (PyCFunction)Str_rsplit_byteset, SZ_METHOD_FLAGS, doc_rsplit_byteset},

    // Lazily evaluated iterators
    {"split_iter", (PyCFunction)Str_split_iter, SZ_METHOD_FLAGS, doc_split_iter},
    {"rsplit_iter", (PyCFunction)Str_rsplit_iter, SZ_METHOD_FLAGS, doc_rsplit_iter},
    {"split_byteset_iter", (PyCFunction)Str_split_byteset_iter, SZ_METHOD_FLAGS, doc_split_byteset_iter},
    {"rsplit_byteset_iter", (PyCFunction)Str_rsplit_byteset_iter, SZ_METHOD_FLAGS, doc_rsplit_byteset_iter},

    // Dealing with larger-than-memory datasets
    {"offset_within", (PyCFunction)Str_offset_within, SZ_METHOD_FLAGS, doc_offset_within},
    {"write_to", (PyCFunction)Str_write_to, SZ_METHOD_FLAGS, doc_write_to},

    {NULL, NULL, 0, NULL} // Sentinel
};

static char const doc_Str[] = //
    "Str(source)\\n"
    "\\n"
    "Immutable byte-string/slice class with SIMD and SWAR-accelerated operations.\\n"
    "Provides high-performance byte-string operations using modern CPU instructions.\\n"
    "\\n"
    "Args:\\n"
    "  source (str, bytes, bytearray, or buffer): Source data to wrap.\\n"
    "\\n"
    "Methods:\\n"
    "  - find(), rfind(): Fast substring search with SIMD acceleration\\n"
    "  - count(): Count occurrences with optional overlap support\\n"
    "  - split(), rsplit(): String splitting with various separators\\n"
    "  - contains(): Fast membership testing\\n"
    "  - translate(): Byte-level translations with lookup tables\\n"
    "\\n"
    "Example:\\n"
    "  >>> s = sz.Str('hello world')\\n"
    "  >>> s.find('world')  # Returns 6\\n"
    "  >>> s.count('l')     # Returns 3";

static PyTypeObject StrType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "stringzilla.Str",
    .tp_doc = doc_Str,
    .tp_basicsize = sizeof(Str),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Str_new,
    .tp_init = Str_init,
    .tp_dealloc = Str_dealloc,
    .tp_hash = Str_hash,
    .tp_richcompare = Str_richcompare,
    .tp_repr = (reprfunc)Str_repr,
    .tp_str = Str_str,
    .tp_methods = Str_methods,
    .tp_as_sequence = &Str_as_sequence,
    .tp_as_mapping = &Str_as_mapping,
    .tp_as_buffer = &Str_as_buffer,
    .tp_as_number = &Str_as_number,
    .tp_getset = Str_getsetters,
};

#pragma endregion

#pragma region Split Iterator

static PyObject *SplitIteratorType_next(SplitIterator *self) {
    // No more data to split
    if (self->reached_tail) return NULL;

    // Create a new `Str` object
    Str *result_obj = (Str *)StrType.tp_alloc(&StrType, 0);
    if (result_obj == NULL && PyErr_NoMemory()) return NULL;

    sz_string_view_t result_memory;

    // Find the next needle
    sz_cptr_t found =
        self->max_parts > 1 //
            ? self->finder(self->text.start, self->text.length, self->separator.start, self->separator.length)
            : NULL;

    // We've reached the end of the string
    if (found == NULL) {
        result_memory.start = self->text.start;
        result_memory.length = self->text.length;
        self->text.length = 0;
        self->reached_tail = 1;
        self->max_parts = 0;
    }
    else {
        if (self->is_reverse) {
            result_memory.start = found + self->match_length * !self->include_match;
            result_memory.length = self->text.start + self->text.length - result_memory.start;
            self->text.length = found - self->text.start;
        }
        else {
            result_memory.start = self->text.start;
            result_memory.length = found - self->text.start;
            self->text.start = found + self->match_length;
            self->text.length -= result_memory.length + self->match_length;
            result_memory.length += self->match_length * self->include_match;
        }
        self->max_parts--;
    }

    // Set its properties based on the slice
    result_obj->memory = result_memory;
    result_obj->parent = self->text_obj;

    // Increment the reference count of the parent
    Py_INCREF(self->text_obj);
    return (PyObject *)result_obj;
}

static void SplitIteratorType_dealloc(SplitIterator *self) {
    Py_XDECREF(self->text_obj);
    Py_XDECREF(self->separator_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SplitIteratorType_iter(PyObject *self) {
    Py_INCREF(self); // Iterator should return itself in __iter__.
    return self;
}

static char const doc_SplitIterator[] = //
    "SplitIterator(string, separator, ...)\\n"
    "\\n"
    "Text-splitting iterator for efficient string processing.\\n"
    "Provides lazy evaluation of string splits without materializing all results.\\n"
    "\\n"
    "Created by:\\n"
    "  - Str.split_iter()\\n"
    "  - Str.rsplit_iter()\\n"
    "  - Str.split_byteset_iter()\\n"
    "  - Str.rsplit_byteset_iter()\\n"
    "\\n"
    "Features:\\n"
    "  - Memory-efficient: yields results one at a time\\n"
    "  - Forward and reverse iteration support\\n"
    "  - Character set and string separator support\\n"
    "\\n"
    "Example:\\n"
    "  >>> s = sz.Str('a,b,c,d')\\n"
    "  >>> for part in s.split_iter(','):\\n"
    "  ...     print(part)";

static PyTypeObject SplitIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.SplitIterator",
    .tp_basicsize = sizeof(SplitIterator),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)SplitIteratorType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_SplitIterator,
    .tp_iter = SplitIteratorType_iter,
    .tp_iternext = (iternextfunc)SplitIteratorType_next,
};

#pragma endregion

#pragma region Strs

/**
 *  @brief Shuffles the parts of a `Strs` object.
 *
 *  This accepts a `Strs` object and potentially produces a new `Strs` object of a different layout:
 *  - `STRS_U32_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U32_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_FRAGMENTED` returns a copy of itself, with the parts shuffled.
 */
static PyObject *Strs_shuffled(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {

    // Check for positional arguments
    PyObject *seed_obj = positional_args_count == 1 ? args[0] : NULL;
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "shuffle() takes at most 1 positional argument");
        return NULL;
    }

    // Check for keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_obj) { seed_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Fisher-Yates Shuffle Algorithm
    unsigned int seed = (unsigned int)time(NULL);
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_obj);
    }

    // Determine the amount of memory needed
    sz_size_t substrings_count = 0;
    get_string_at_offset_t substring_getter = NULL;
    PyObject *parent_to_increment = NULL;
    sz_memory_allocator_t allocator;

    switch (self->layout) {
    case STRS_U32_TAPE:
        substring_getter = str_at_offset_u32_tape;
        substrings_count = self->data.u32_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u32_tape.allocator;
        break;
    case STRS_U32_TAPE_VIEW:
        substring_getter = str_at_offset_u32_tape_view;
        substrings_count = self->data.u32_tape_view.count;
        parent_to_increment = self->data.u32_tape_view.parent;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_U64_TAPE:
        substring_getter = str_at_offset_u64_tape;
        substrings_count = self->data.u64_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u64_tape.allocator;
        break;
    case STRS_U64_TAPE_VIEW:
        substring_getter = str_at_offset_u64_tape_view;
        substrings_count = self->data.u64_tape_view.count;
        parent_to_increment = self->data.u64_tape_view.parent;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_FRAGMENTED:
        substring_getter = str_at_offset_fragmented;
        substrings_count = self->data.fragmented.count;
        parent_to_increment = self->data.fragmented.parent;
        allocator = self->data.fragmented.allocator;
        break;
    }

    sz_string_view_t *new_spans =
        (sz_string_view_t *)allocator.allocate(substrings_count * sizeof(sz_string_view_t), allocator.handle);
    if (new_spans == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for reordered slices");
        return NULL;
    }

    // Populate the new reordered array using get_string_at_offset
    for (sz_size_t i = 0; i < substrings_count; ++i) {
        PyObject *unused_parent;
        sz_cptr_t start;
        sz_size_t length;
        substring_getter(self, (Py_ssize_t)i, substrings_count, &unused_parent, &start, &length);
        new_spans[i].start = start;
        new_spans[i].length = length;
    }

    // Create a new Strs object for the reordered layout
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) {
        allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_NoMemory();
        return NULL;
    }

    srand(seed);
    for (sz_size_t i = substrings_count - 1; i > 0; --i) {
        sz_size_t j = rand() % (i + 1);
        // Swap parts[i] and parts[j]
        sz_string_view_t temp = new_spans[i];
        new_spans[i] = new_spans[j];
        new_spans[j] = temp;
    }

    // Set up the new reordered object
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = substrings_count;
    result->data.fragmented.spans = new_spans;
    result->data.fragmented.parent = parent_to_increment;
    result->data.fragmented.allocator = allocator;
    Py_INCREF(parent_to_increment); // Keep the original as parent

    return result;
}

/**
 *  @brief Sorts the parts of a `Strs` object.
 *
 *  This accepts a `Strs` object and potentially produces a new `Strs` object of a different layout:
 *  - `STRS_U32_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U32_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_FRAGMENTED` returns a copy of itself, with the parts sorted.
 */
static PyObject *Strs_sorted(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    PyObject *reverse_obj = NULL; // Default is not reversed

    // Check for positional arguments
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "sort() takes at most 1 positional argument");
        return NULL;
    }
    else if (positional_args_count == 1) { reverse_obj = args[0]; }

    // Check for keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0 && !reverse_obj) { reverse_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_bool_t reverse = 0; // Default is False
    if (reverse_obj) {
        if (!PyBool_Check(reverse_obj)) {
            PyErr_SetString(PyExc_TypeError, "The reverse must be a boolean");
            return NULL;
        }
        reverse = PyObject_IsTrue(reverse_obj);
    }

    // Determine the amount of memory needed
    sz_size_t substrings_count = 0;
    get_string_at_offset_t substring_getter = NULL;
    PyObject *parent_to_increment = NULL;
    sz_memory_allocator_t allocator;

    switch (self->layout) {
    case STRS_U32_TAPE:
        substring_getter = str_at_offset_u32_tape;
        substrings_count = self->data.u32_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u32_tape.allocator;
        break;
    case STRS_U32_TAPE_VIEW:
        substring_getter = str_at_offset_u32_tape_view;
        substrings_count = self->data.u32_tape_view.count;
        parent_to_increment = (PyObject *)self;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_U64_TAPE:
        substring_getter = str_at_offset_u64_tape;
        substrings_count = self->data.u64_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u64_tape.allocator;
        break;
    case STRS_U64_TAPE_VIEW:
        substring_getter = str_at_offset_u64_tape_view;
        substrings_count = self->data.u64_tape_view.count;
        parent_to_increment = (PyObject *)self;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_FRAGMENTED:
        substring_getter = str_at_offset_fragmented;
        substrings_count = self->data.fragmented.count;
        parent_to_increment = self->data.fragmented.parent;
        allocator = self->data.fragmented.allocator;
        break;
    }

    sz_string_view_t *new_spans =
        (sz_string_view_t *)allocator.allocate(substrings_count * sizeof(sz_string_view_t), allocator.handle);
    if (new_spans == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for reordered slices");
        return NULL;
    }

    // Populate the new reordered array using get_string_at_offset
    for (sz_size_t i = 0; i < substrings_count; ++i) {
        PyObject *unused_parent;
        sz_cptr_t start;
        sz_size_t length;
        substring_getter((Strs *)self, (Py_ssize_t)i, substrings_count, &unused_parent, &start, &length);
        new_spans[i].start = start;
        new_spans[i].length = length;
    }

    // Determine memory needed for sorting
    sz_size_t const memory_needed = sizeof(sz_sorted_idx_t) * substrings_count;
    if (temporary_memory.length < memory_needed) {
        void *new_memory = realloc(temporary_memory.start, memory_needed);
        if (!new_memory) {
            allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
            PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the sorting operation");
            return NULL;
        }
        temporary_memory.start = new_memory;
        temporary_memory.length = memory_needed;
    }
    if (!temporary_memory.start) {
        allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the sorting operation");
        return NULL;
    }
    sz_sorted_idx_t *order = (sz_sorted_idx_t *)temporary_memory.start;

    // Call our sorting algorithm
    sz_sequence_t sequence;
    sz_fill(&sequence, sizeof(sequence), 0);
    sequence.count = substrings_count;
    sequence.handle = (void *)self;
    sequence.get_start = Strs_get_start_;
    sequence.get_length = Strs_get_length_;
    sz_status_t status = sz_sequence_argsort(&sequence, NULL, order);
    sz_unused_(status);

    // Apply the sorting algorithm here, considering the `reverse` value
    if (reverse) reverse_offsets(order, substrings_count);

    // Apply the new order to create sorted spans
    sz_string_view_t *sorted_spans =
        (sz_string_view_t *)allocator.allocate(substrings_count * sizeof(sz_string_view_t), allocator.handle);
    if (sorted_spans == NULL) {
        allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for sorted slices");
        return NULL;
    }

    // Apply the permutation
    for (sz_size_t i = 0; i < substrings_count; ++i) sorted_spans[i] = new_spans[order[i]];

    // Free the temporary spans array
    allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);

    // Create a new Strs object for the sorted layout
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) {
        allocator.free(sorted_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_NoMemory();
        return NULL;
    }

    // Set up the new sorted object
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = substrings_count;
    result->data.fragmented.spans = sorted_spans;
    result->data.fragmented.parent = parent_to_increment;
    result->data.fragmented.allocator = allocator;
    Py_INCREF(parent_to_increment); // Keep the original as parent

    return (PyObject *)result;
}

/**
 *  @brief Returns the tuple permuting a `Strs` object into a sorted order.
 */
static PyObject *Strs_argsort(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    PyObject *reverse_obj = NULL; // Default is not reversed

    // Check for positional arguments
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "order() takes at most 1 positional argument");
        return NULL;
    }
    else if (positional_args_count == 1) { reverse_obj = args[0]; }

    // Check for keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0 && !reverse_obj) { reverse_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_bool_t reverse = 0; // Default is False
    if (reverse_obj) {
        if (!PyBool_Check(reverse_obj)) {
            PyErr_SetString(PyExc_TypeError, "The reverse must be a boolean");
            return NULL;
        }
        reverse = PyObject_IsTrue(reverse_obj);
    }

    // Determine the amount of memory needed
    sz_size_t const count = Strs_len(self);
    sz_size_t const memory_needed = sizeof(sz_sorted_idx_t) * count;
    if (temporary_memory.length < memory_needed) {
        void *new_memory = realloc(temporary_memory.start, memory_needed);
        if (!new_memory) {
            PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the sorting operation");
            return 0;
        }
        temporary_memory.start = new_memory;
        temporary_memory.length = memory_needed;
    }
    if (!temporary_memory.start) {
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the sorting operation");
        return 0;
    }
    sz_sorted_idx_t *order = (sz_sorted_idx_t *)temporary_memory.start;

    // Call our sorting algorithm
    sz_sequence_t sequence;
    sz_fill(&sequence, sizeof(sequence), 0);
    sequence.count = count;
    sequence.handle = self;
    sequence.get_start = Strs_get_start_;
    sequence.get_length = Strs_get_length_;
    sz_status_t status = sz_sequence_argsort(&sequence, NULL, order);
    sz_unused_(status);

    // Apply the sorting algorithm here, considering the `reverse` value
    if (reverse) reverse_offsets(order, count);

    // Here, instead of applying the order, we want to return the copy of the
    // order as a NumPy array of 64-bit unsigned integers.
    //
    //      npy_intp numpy_size = count;
    //      PyObject *array = PyArray_SimpleNew(1, &numpy_size, NPY_UINT64);
    //      if (!array) {
    //          PyErr_SetString(PyExc_RuntimeError, "Failed to create a NumPy array");
    //          return NULL;
    //      }
    //      sz_sorted_idx_t *numpy_data_ptr = (sz_sorted_idx_t *)PyArray_DATA((PyArrayObject *)array);
    //      sz_copy(numpy_data_ptr, order, count * sizeof(sz_sorted_idx_t));
    //
    // There are compilation issues with NumPy.
    // Here is an example for `cp312-musllinux_s390x`: https://x.com/ashvardanian/status/1757880762278531447?s=20
    // So instead of NumPy, let's produce a tuple of integers.
    PyObject *tuple = PyTuple_New(count);
    if (!tuple) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create a tuple");
        return NULL;
    }
    for (sz_size_t i = 0; i < count; ++i) {
        PyObject *index = PyLong_FromUnsignedLong(order[i]);
        if (!index) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create a tuple element");
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, index);
    }
    return tuple;
}

static PyObject *Strs_sample(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    PyObject *sample_size_obj = NULL;
    PyObject *seed_obj = NULL;

    // Check for positional arguments
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "sample() takes 1 positional argument and 1 keyword argument");
        return NULL;
    }
    else if (positional_args_count == 1) { sample_size_obj = args[0]; }

    // Parse keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_obj) { seed_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Translate the seed and the sample size to C types
    sz_size_t sample_size = 0;
    if (sample_size_obj) {
        if (!PyLong_Check(sample_size_obj)) {
            PyErr_SetString(PyExc_TypeError, "The sample size must be an integer");
            return NULL;
        }
        sample_size = PyLong_AsSize_t(sample_size_obj);
    }
    unsigned int seed = (unsigned int)time(NULL); // Default seed
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_obj);
    }

    // Create a new `Strs` object
    Strs *result = (Strs *)StrsType.tp_alloc(&StrsType, 0);
    if (result == NULL && PyErr_NoMemory()) return NULL;

    // Initialize the memory allocator with default malloc wrapper
    sz_memory_allocator_init_default(&result->data.fragmented.allocator);

    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = 0;
    result->data.fragmented.spans = NULL;
    result->data.fragmented.parent = NULL;
    if (sample_size == 0) { return (PyObject *)result; }

    // Now create a new Strs object with the sampled strings
    sz_string_view_t *result_spans = malloc(sample_size * sizeof(sz_string_view_t));
    if (!result_spans) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the sample");
        return NULL;
    }

    // Introspect the Strs object to know the from which will be sampling
    Py_ssize_t count = Strs_len(self);
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    // Randomly sample the strings
    srand(seed);
    PyObject *parent_string;
    for (Py_ssize_t i = 0; i < (Py_ssize_t)sample_size; i++) {
        sz_size_t index = rand() % count;
        getter(self, index, count, &parent_string, &result_spans[i].start, &result_spans[i].length);
    }

    // Update the `Strs` object
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = sample_size;
    result->data.fragmented.spans = result_spans;
    result->data.fragmented.parent = parent_string;
    // Hold a reference to the parent backing buffer while this view is alive
    Py_XINCREF(result->data.fragmented.parent);
    return result;
}

static PyObject *Strs_get_layout(Strs *self, void *Py_UNUSED(closure)) {
    char buffer[1024];

    switch (self->layout) {
    case STRS_U32_TAPE_VIEW:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U32_TAPE_VIEW, count=%zu, data=%p, offsets=%p, parent=%p]",
                 self->data.u32_tape_view.count, (void *)self->data.u32_tape_view.data,
                 (void *)self->data.u32_tape_view.offsets, (void *)self->data.u32_tape_view.parent);
        break;

    case STRS_U64_TAPE_VIEW:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U64_TAPE_VIEW, count=%zu, data=%p, offsets=%p, parent=%p]",
                 self->data.u64_tape_view.count, (void *)self->data.u64_tape_view.data,
                 (void *)self->data.u64_tape_view.offsets, (void *)self->data.u64_tape_view.parent);
        break;

    case STRS_U32_TAPE:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U32_TAPE, count=%zu, data=%p, offsets=%p]",
                 self->data.u32_tape.count, (void *)self->data.u32_tape.data, (void *)self->data.u32_tape.offsets);
        break;

    case STRS_U64_TAPE:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U64_TAPE, count=%zu, data=%p, offsets=%p]",
                 self->data.u64_tape.count, (void *)self->data.u64_tape.data, (void *)self->data.u64_tape.offsets);
        break;

    case STRS_FRAGMENTED:
        snprintf(buffer, sizeof(buffer), "Strs[layout=FRAGMENTED, count=%zu, spans=%p, parent=%p]",
                 self->data.fragmented.count, (void *)self->data.fragmented.spans,
                 (void *)self->data.fragmented.parent);
        break;

    default: snprintf(buffer, sizeof(buffer), "Strs[layout=UNKNOWN(%d)]", self->layout); break;
    }

    return PyUnicode_FromString(buffer);
}

/**
 *  @brief Exports a string to a UTF-8 buffer, escaping single quotes.
 *  @param[in] cstr The input string to export.
 *  @param[in] cstr_length The length of the input string.
 *  @param[out] buffer The output buffer to write to.
 *  @param[in] buffer_length The size of the output buffer.
 *  @param[out] did_fit Populated with 1 if the string is fully exported, 0 if it didn't fit, -1 if invalid UTF-8.
 *  @return Pointer to the end of the written data in the buffer, or buffer position where error occurred.
 */
sz_cptr_t export_escaped_unquoted_to_utf8_buffer(sz_cptr_t cstr, sz_size_t cstr_length,    //
                                                 sz_ptr_t buffer, sz_size_t buffer_length, //
                                                 int *did_fit) {
    sz_cptr_t const cstr_end = cstr + cstr_length;
    sz_ptr_t buffer_ptr = buffer;
    *did_fit = 1;

    // First pass: calculate required buffer size and validate UTF-8
    sz_size_t required_bytes = 2; // Opening and closing quotes
    sz_cptr_t scan_ptr = cstr;
    while (scan_ptr < cstr_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(scan_ptr, &rune, &rune_length);

        // Check for invalid UTF-8
        if (rune_length == sz_utf8_invalid_k) {
            *did_fit = -1; // Signal UTF-8 error
            return buffer_ptr;
        }

        if (rune_length == 1 && *scan_ptr == '\'') { required_bytes += 2; } // Escaped quote: \'
        else { required_bytes += rune_length; }                             // Normal rune
        scan_ptr += rune_length;
    }

    // Check if we have enough buffer space
    if (required_bytes > buffer_length) {
        *did_fit = 0;
        return buffer_ptr;
    }

    // Second pass: actually write to buffer
    *(buffer_ptr++) = '\''; // Opening quote

    while (cstr < cstr_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(cstr, &rune, &rune_length);

        if (rune_length == 1 && *cstr == '\'') {
            *(buffer_ptr++) = '\\';
            *(buffer_ptr++) = '\'';
        }
        else {
            sz_copy(buffer_ptr, cstr, rune_length);
            buffer_ptr += rune_length;
        }
        cstr += rune_length;
    }

    *(buffer_ptr++) = '\''; // Closing quote
    return buffer_ptr;
}

/**
 *  @brief Exports a binary string to a buffer in Python bytes representation (b'\\x..').
 *  @param[in] data The binary data to export.
 *  @param[in] data_length The length of the binary data.
 *  @param[out] buffer The output buffer to write to.
 *  @param[in] buffer_length The size of the output buffer.
 *  @param[out] did_fit Populated with 1 if the data is fully exported, 0 if it didn't fit.
 *  @return Pointer to the end of the written data in the buffer.
 */
sz_cptr_t export_escaped_unquoted_to_binary_buffer(sz_cptr_t data, sz_size_t data_length,    //
                                                   sz_ptr_t buffer, sz_size_t buffer_length, //
                                                   int *did_fit) {
    sz_ptr_t buffer_ptr = buffer;
    *did_fit = 1;

    // First pass: calculate required buffer size
    // Format: b'\x00\x01...'  -> 3 bytes prefix + 4 bytes per byte + 1 byte suffix
    sz_size_t required_bytes = 3 + (data_length * 4) + 1;

    // Check if we have enough buffer space
    if (required_bytes > buffer_length) {
        *did_fit = 0;
        return buffer_ptr;
    }

    // Second pass: write to buffer
    *(buffer_ptr++) = 'b';
    *(buffer_ptr++) = '\'';

    // Export each byte as \x followed by two hex digits
    static const char hex_chars[] = "0123456789abcdef";
    for (sz_size_t i = 0; i < data_length; i++) {
        unsigned char byte = (unsigned char)data[i];
        *(buffer_ptr++) = '\\';
        *(buffer_ptr++) = 'x';
        *(buffer_ptr++) = hex_chars[byte >> 4];
        *(buffer_ptr++) = hex_chars[byte & 0x0f];
    }

    *(buffer_ptr++) = '\'';
    return buffer_ptr;
}

/**
 *  @brief  Formats an array of strings, similar to the `repr` method of Python lists.
 *          Will output an object that looks like `sz.Str(['item1', 'item2... ])`, potentially
 *          dropping the last few entries.
 */
static PyObject *Strs_repr(Strs *self) {
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    char repr_buffer[1024];
    sz_ptr_t repr_buffer_ptr = &repr_buffer[0];
    sz_cptr_t const repr_buffer_end = repr_buffer_ptr + 1024;

    // Start of the array
    sz_copy(repr_buffer_ptr, "sz.Strs([", 9);
    repr_buffer_ptr += 9;

    sz_size_t count = Strs_len(self);
    PyObject *parent_string;

    // In the worst case, we must have enough space for `...', ...])`
    // That's extra 11 bytes of content.
    sz_cptr_t non_fitting_array_tail = "... ])";
    int const non_fitting_array_tail_length = 6;

    // If the whole string doesn't fit, even before the `non_fitting_array_tail` tail,
    // we need to add `, '` separator of 3 bytes.
    for (sz_size_t i = 0; i < count && repr_buffer_ptr + (non_fitting_array_tail_length + 3) < repr_buffer_end; i++) {
        sz_cptr_t cstr_start = NULL;
        sz_size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);

        if (i > 0) { *(repr_buffer_ptr++) = ',', *(repr_buffer_ptr++) = ' '; }

        // Check if the string contains valid UTF-8
        int did_fit;
        repr_buffer_ptr = sz_runes_valid(cstr_start, cstr_length)
                              ? export_escaped_unquoted_to_utf8_buffer(
                                    cstr_start, cstr_length, repr_buffer_ptr,
                                    repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length, &did_fit)
                              : export_escaped_unquoted_to_binary_buffer(
                                    cstr_start, cstr_length, repr_buffer_ptr,
                                    repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length, &did_fit);

        // If it didn't fit, let's put an ellipsis
        if (!did_fit) {
            sz_copy(repr_buffer_ptr, non_fitting_array_tail, non_fitting_array_tail_length);
            repr_buffer_ptr += non_fitting_array_tail_length;
            return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
        }
    }

    // Close the array
    *(repr_buffer_ptr++) = ']', *(repr_buffer_ptr++) = ')';
    return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
}

/**
 *  @brief  Array to string conversion method, that concatenates all the strings in the array.
 *          Will output an object that looks like `['item1', 'item2', 'item3']`, containing all
 *          the strings.
 */
static PyObject *Strs_str(Strs *self) {
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    // Aggregate the total length of all the slices and count the number of bytes we need to allocate:
    sz_size_t count = Strs_len(self);
    PyObject *parent_string;
    sz_size_t total_bytes = 2; // opening and closing square brackets
    for (sz_size_t i = 0; i < count; i++) {
        sz_cptr_t cstr_start = NULL;
        sz_size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);

        if (i != 0) total_bytes += 2; // For the preceding comma and space

        // Check if string is valid UTF-8 to determine format
        if (sz_runes_valid(cstr_start, cstr_length)) {
            // Valid UTF-8: format as '...' with escaped quotes
            total_bytes += 2;           // Opening and closing quotes
            total_bytes += cstr_length; // Base string length

            // Count the number of single quotes that need escaping
            sz_cptr_t scan_ptr = cstr_start;
            sz_size_t scan_length = cstr_length;
            while (scan_length) {
                char quote = '\'';
                sz_cptr_t next_quote = sz_find_byte(scan_ptr, scan_length, &quote);
                if (next_quote == NULL) break;
                total_bytes++; // Extra byte for escaping
                scan_length -= next_quote - scan_ptr + 1;
                scan_ptr = next_quote + 1;
            }
        }
        else {
            // Invalid UTF-8: format as b'\x...'
            total_bytes += 3;               // "b'" prefix
            total_bytes += cstr_length * 4; // Each byte becomes \xNN (4 chars)
            total_bytes += 1;               // Closing quote
        }
    }

    // Now allocate the memory for the concatenated string
    sz_ptr_t const result_buffer = malloc(total_bytes);
    if (!result_buffer) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the concatenated string");
        return NULL;
    }

    // Copy the strings into the result buffer
    sz_ptr_t result_ptr = result_buffer;
    *result_ptr++ = '[';
    for (sz_size_t i = 0; i < count; i++) {
        if (i != 0) {
            *result_ptr++ = ',';
            *result_ptr++ = ' ';
        }
        sz_cptr_t cstr_start = NULL;
        sz_size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);
        int did_fit;
        // Check if the string contains valid UTF-8 and export appropriately
        result_ptr =
            sz_runes_valid(cstr_start, cstr_length)
                ? export_escaped_unquoted_to_utf8_buffer(cstr_start, cstr_length, result_ptr,
                                                         total_bytes - (result_ptr - result_buffer), &did_fit)
                : export_escaped_unquoted_to_binary_buffer(cstr_start, cstr_length, result_ptr,
                                                           total_bytes - (result_ptr - result_buffer), &did_fit);

        // Note: If did_fit is 0, we have a buffer size calculation error, but we continue for robustness
    }

    *result_ptr++ = ']';
    sz_size_t actual_bytes = result_ptr - result_buffer;
    PyObject *result = PyUnicode_FromStringAndSize(result_buffer, actual_bytes);
    free(result_buffer);
    return result;
}

static PySequenceMethods Strs_as_sequence = {
    .sq_length = Strs_len,   //
    .sq_item = Strs_getitem, //
    .sq_contains = Strs_in,  //
};

static PyMappingMethods Strs_as_mapping = {
    .mp_length = Strs_len,          //
    .mp_subscript = Strs_subscript, // Is used to implement slices in Python
};

static PyGetSetDef Strs_getsetters[] = {
    // Compatibility with PyArrow
    {"tape", (getter)Strs_get_tape, NULL, "In-place transforms the string representation to match Apache Arrow", NULL},
    {"tape_address", (getter)Strs_get_tape_address, NULL, "Address of the first byte of the first string", NULL},
    {"tape_nbytes", (getter)Strs_get_tape_nbytes, NULL, "Length of the entire tape of strings in bytes", NULL},
    {"offsets_address", (getter)Strs_get_offsets_address, NULL, "Address of the first byte of offsets array", NULL},
    {"offsets_nbytes", (getter)Strs_get_offsets_nbytes, NULL, "Get teh length of offsets array in bytes", NULL},
    {"offsets_are_large", (getter)Strs_get_offsets_are_large, NULL,
     "Checks if 64-bit addressing should be used to convert to Arrow", NULL},
    {"__layout__", (getter)Strs_get_layout, NULL, "Debug information about the internal layout", NULL},
    {NULL} // Sentinel
};

// The efficient `Strs_init` path initializing from PyArrow array capsules.
static int Strs_init_from_pyarrow(Strs *self, PyObject *sequence_obj, int view) {
    // Handle Arrow array
    PyObject *capsules = PyObject_CallMethod(sequence_obj, "__arrow_c_array__", NULL);
    if (!capsules || !PyTuple_Check(capsules) || PyTuple_Size(capsules) != 2) {
        Py_XDECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "__arrow_c_array__ must return a tuple of 2 capsules");
        return -1;
    }

    PyObject *schema_capsule = PyTuple_GET_ITEM(capsules, 0);
    PyObject *array_capsule = PyTuple_GET_ITEM(capsules, 1);

    if (!PyCapsule_CheckExact(schema_capsule) || !PyCapsule_CheckExact(array_capsule)) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "Expected PyCapsule objects from __arrow_c_array__");
        return -1;
    }

    struct ArrowSchema *schema = (struct ArrowSchema *)PyCapsule_GetPointer(schema_capsule, "arrow_schema");
    struct ArrowArray *array = (struct ArrowArray *)PyCapsule_GetPointer(array_capsule, "arrow_array");

    if (!schema || !array) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "Failed to extract Arrow C structures");
        return -1;
    }

    // Validate string array layout
    if (!schema->format || (strcmp(schema->format, "u") != 0 && strcmp(schema->format, "U") != 0 &&
                            strcmp(schema->format, "z") != 0 && strcmp(schema->format, "Z") != 0)) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "Arrow array must be string layout");
        return -1;
    }

    if (array->n_buffers != 3) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "String Arrow array must have 3 buffers");
        return -1;
    }

    // Determine if 32-bit or 64-bit offsets
    int use_64bit = (strcmp(schema->format, "U") == 0 || strcmp(schema->format, "Z") == 0);
    void const **buffers = (void const **)array->buffers;
    sz_u8_t const *validity = (sz_u8_t const *)buffers[0]; // May be NULL
    sz_cptr_t data_buffer = (sz_cptr_t)buffers[2];
    sz_size_t length = array->length;

    // Zero-copy mode for Arrow arrays
    if (view) {
        if (use_64bit) {
            sz_i64_t const *offsets_64 = (sz_i64_t const *)buffers[1];
            self->layout = STRS_U64_TAPE_VIEW;
            self->data.u64_tape_view.count = length;
            self->data.u64_tape_view.parent = capsules;
            self->data.u64_tape_view.data = data_buffer;
            self->data.u64_tape_view.offsets = (sz_u64_t *)offsets_64;
            Py_INCREF(capsules);
        }
        else {
            sz_i32_t const *offsets_32 = (sz_i32_t const *)buffers[1];
            self->layout = STRS_U32_TAPE_VIEW;
            self->data.u32_tape_view.count = length;
            self->data.u32_tape_view.parent = capsules;
            self->data.u32_tape_view.data = data_buffer;
            self->data.u32_tape_view.offsets = (sz_u32_t *)offsets_32;
            Py_INCREF(capsules);
        }
    }
    // Copy mode for Arrow arrays
    else {
        // Copy mode for Arrow arrays - use allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        if (use_64bit) {
            sz_i64_t const *offsets_64 = (sz_i64_t const *)buffers[1];
            sz_size_t total_bytes = offsets_64[length] - offsets_64[0];

            // Allocate new buffer and offsets using the allocator
            sz_ptr_t new_data =
                total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle) : (sz_ptr_t)NULL;
            sz_u64_t *new_offsets = (sz_u64_t *)allocator.allocate((length + 1) * sizeof(sz_u64_t), allocator.handle);
            int const failed_to_allocate_data = total_bytes && !new_data;
            if (failed_to_allocate_data || !new_offsets) {
                if (new_data) allocator.free(new_data, total_bytes, allocator.handle);
                if (new_offsets) allocator.free(new_offsets, (length + 1) * sizeof(sz_u64_t), allocator.handle);
                PyErr_NoMemory();
                return -1;
            }

            // Copy data and adjust offsets (Apache Arrow format)
            sz_size_t actual_bytes = offsets_64[length] - offsets_64[0];
            if (actual_bytes > 0) sz_copy(new_data, data_buffer + offsets_64[0], actual_bytes);
            new_offsets[0] = 0; // First offset is always 0
            for (sz_size_t i = 0; i < length; i++) {
                // Handle null values by checking validity bitmap
                if (validity && !(validity[i / 8] & (1 << (i % 8)))) { new_offsets[i + 1] = new_offsets[i]; }
                else { new_offsets[i + 1] = offsets_64[i + 1] - offsets_64[0]; }
            }

            self->layout = STRS_U64_TAPE;
            self->data.u64_tape.count = length;
            self->data.u64_tape.data = new_data;
            self->data.u64_tape.offsets = new_offsets;
            self->data.u64_tape.allocator = allocator;
        }
        else {
            sz_i32_t const *offsets_32 = (sz_i32_t const *)buffers[1];
            sz_size_t total_bytes = offsets_32[length] - offsets_32[0];

            // Allocate new buffer and offsets using the allocator
            sz_ptr_t new_data =
                total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle) : (sz_ptr_t)NULL;
            sz_u32_t *new_offsets = (sz_u32_t *)allocator.allocate((length + 1) * sizeof(sz_u32_t), allocator.handle);
            int const failed_to_allocate_data = total_bytes && !new_data;
            if (failed_to_allocate_data || !new_offsets) {
                if (new_data) allocator.free(new_data, total_bytes, allocator.handle);
                if (new_offsets) allocator.free(new_offsets, (length + 1) * sizeof(sz_u32_t), allocator.handle);
                PyErr_NoMemory();
                return -1;
            }

            // Copy data and adjust offsets (Apache Arrow format)
            sz_size_t actual_bytes = offsets_32[length] - offsets_32[0];
            if (actual_bytes > 0) sz_copy(new_data, data_buffer + offsets_32[0], actual_bytes);
            new_offsets[0] = 0; // First offset is always 0
            for (sz_size_t i = 0; i < length; i++) {
                // Handle null values by checking validity bitmap
                if (validity && !(validity[i / 8] & (1 << (i % 8)))) { new_offsets[i + 1] = new_offsets[i]; }
                else { new_offsets[i + 1] = offsets_32[i + 1] - offsets_32[0]; }
            }

            self->layout = STRS_U32_TAPE;
            self->data.u32_tape.count = length;
            self->data.u32_tape.data = new_data;
            self->data.u32_tape.offsets = new_offsets;
            self->data.u32_tape.allocator = allocator;
        }
    }

    Py_DECREF(capsules);
    return 0;
}

// The less efficient `Strs_init` path initializing from a Pythonic tuple of strings.
static int Strs_init_from_tuple(Strs *self, PyObject *sequence_obj, int view) {
    Py_ssize_t count = PyTuple_GET_SIZE(sequence_obj);

    // Empty tuple, create empty Strs
    if (count == 0) {
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        self->data.fragmented.parent = NULL;
        sz_memory_allocator_init_default(&self->data.fragmented.allocator);
        return 0;
    }

    // Zero-copy mode for Python sequences - use reordered layout for memory-scattered strings
    if (view) {
        // Initialize allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        sz_string_view_t *parts =
            (sz_string_view_t *)allocator.allocate(count * sizeof(sz_string_view_t), allocator.handle);
        if (!parts) {
            PyErr_NoMemory();
            return -1;
        }

        // Create views directly to Python string objects
        for (sz_size_t i = 0; i < (sz_size_t)count; i++) {
            PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
            sz_cptr_t item_start;
            sz_size_t item_length;
            if (!sz_py_export_string_like(item, &item_start, &item_length)) {
                allocator.free(parts, count * sizeof(sz_string_view_t), allocator.handle);
                PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", i);
                return -1;
            }
            parts[i].start = item_start;
            parts[i].length = item_length;
        }

        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = count;
        self->data.fragmented.spans = parts;
        self->data.fragmented.allocator = allocator;
        self->data.fragmented.parent = sequence_obj; // Keep sequence alive
        Py_INCREF(sequence_obj);
    }
    // Allocate a new tape to fit all of the items
    else {
        // Estimate the overall size of strings in bytes
        sz_size_t total_bytes = 0;
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
            sz_cptr_t item_start;
            sz_size_t item_length;
            if (!sz_py_export_string_like(item, &item_start, &item_length)) {
                PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", i);
                return -1;
            }
            total_bytes += item_length;
        }

        int use_64bit = (total_bytes >= UINT32_MAX);

        // Initialize allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        // Allocate data buffer using allocator
        sz_ptr_t data_buffer =
            total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle) : (sz_ptr_t)NULL;
        int const failed_to_allocate_data = total_bytes && !data_buffer;
        if (failed_to_allocate_data) {
            PyErr_NoMemory();
            return -1;
        }

        if (use_64bit) {
            // Apache Arrow format: N+1 offsets for N strings
            sz_u64_t *offsets = (sz_u64_t *)allocator.allocate((count + 1) * sizeof(sz_u64_t), allocator.handle);
            if (!offsets) {
                if (data_buffer) allocator.free(data_buffer, total_bytes, allocator.handle);
                PyErr_NoMemory();
                return -1;
            }

            sz_size_t offset = 0;
            offsets[0] = 0; // First offset is always 0
            for (Py_ssize_t i = 0; i < count; i++) {
                PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
                sz_cptr_t item_start;
                sz_size_t item_length;
                sz_py_export_string_like(item, &item_start, &item_length);

                sz_copy(data_buffer + offset, item_start, item_length);
                offset += item_length;
                offsets[i + 1] = offset; // Apache Arrow format: offset after this string
            }

            self->layout = STRS_U64_TAPE;
            self->data.u64_tape.count = count;
            self->data.u64_tape.data = data_buffer;
            self->data.u64_tape.offsets = offsets;
            self->data.u64_tape.allocator = allocator;
        }
        else {
            // Apache Arrow format: N+1 offsets for N strings
            sz_u32_t *offsets = (sz_u32_t *)allocator.allocate((count + 1) * sizeof(sz_u32_t), allocator.handle);
            if (!offsets) {
                if (data_buffer) allocator.free(data_buffer, total_bytes, allocator.handle);
                PyErr_NoMemory();
                return -1;
            }

            sz_size_t offset = 0;
            offsets[0] = 0; // First offset is always 0
            for (Py_ssize_t i = 0; i < count; i++) {
                PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
                sz_cptr_t item_start;
                sz_size_t item_length;
                sz_py_export_string_like(item, &item_start, &item_length);

                sz_copy(data_buffer + offset, item_start, item_length);
                offset += item_length;
                offsets[i + 1] = offset; // Apache Arrow format: offset after this string
            }

            self->layout = STRS_U32_TAPE;
            self->data.u32_tape.count = count;
            self->data.u32_tape.data = data_buffer;
            self->data.u32_tape.offsets = offsets;
            self->data.u32_tape.allocator = allocator;
        }
    }

    return 0;
}

// The inefficient `Strs_init` path initializing from a Pythonic list of strings.
static int Strs_init_from_list(Strs *self, PyObject *sequence_obj, int view) {
    Py_ssize_t count = PyList_GET_SIZE(sequence_obj);

    // Handle empty list
    if (count == 0) {
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        sz_memory_allocator_init_default(&self->data.fragmented.allocator);
        self->data.fragmented.parent = NULL;
        return 0;
    }

    // Zero-copy mode for Python sequences - use reordered layout for memory-scattered strings
    if (view) {
        // Initialize allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        sz_string_view_t *parts =
            (sz_string_view_t *)allocator.allocate(count * sizeof(sz_string_view_t), allocator.handle);
        if (!parts) {
            PyErr_NoMemory();
            return -1;
        }

        // Build views directly to the string data
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PyList_GET_ITEM(sequence_obj, i);

            // Export string data directly (no copying, just span)
            sz_cptr_t item_start;
            sz_size_t item_length;
            if (!sz_py_export_string_like(item, &item_start, &item_length)) {
                allocator.free(parts, count * sizeof(sz_string_view_t), allocator.handle);
                PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", i);
                return -1;
            }

            parts[i].start = item_start;
            parts[i].length = item_length;
        }

        // Setup reordered layout with parent list to keep strings alive
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = count;
        self->data.fragmented.spans = parts;
        self->data.fragmented.allocator = allocator;
        self->data.fragmented.parent = sequence_obj; // Keep list alive
        Py_INCREF(sequence_obj);
        return 0;
    }
    // Allocate a new tape to fit all of the items
    else {

        // First pass: calculate total size needed
        sz_size_t total_bytes = 0;
        int use_64bit = 0;

        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PyList_GET_ITEM(sequence_obj, i);
            sz_cptr_t item_start;
            sz_size_t item_length;
            if (!sz_py_export_string_like(item, &item_start, &item_length)) {
                PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", i);
                return -1;
            }

            // Check if we need 64-bit offsets
            if (total_bytes + item_length > UINT32_MAX) { use_64bit = 1; }
            total_bytes += item_length;
        }

        // Initialize allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        // Allocate buffers based on calculated sizes
        sz_ptr_t data_buffer =
            total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle) : (sz_ptr_t)NULL;

        // Apache Arrow format: N+1 offsets for N strings
        void *offsets;
        if (use_64bit) { offsets = allocator.allocate((count + 1) * sizeof(sz_u64_t), allocator.handle); }
        else { offsets = allocator.allocate((count + 1) * sizeof(sz_u32_t), allocator.handle); }

        int const failed_to_allocate_data = total_bytes && !data_buffer;
        if (failed_to_allocate_data || !offsets) {
            if (data_buffer) allocator.free(data_buffer, total_bytes, allocator.handle);
            if (offsets) {
                sz_size_t offsets_size = use_64bit ? (count + 1) * sizeof(sz_u64_t) : (count + 1) * sizeof(sz_u32_t);
                allocator.free(offsets, offsets_size, allocator.handle);
            }
            PyErr_NoMemory();
            return -1;
        }

        // Second pass: copy data and build offsets (Apache Arrow format)
        sz_size_t current_offset = 0;
        // Set first offset to 0
        if (use_64bit) { ((sz_u64_t *)offsets)[0] = 0; }
        else { ((sz_u32_t *)offsets)[0] = 0; }

        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PyList_GET_ITEM(sequence_obj, i);
            sz_cptr_t item_start;
            sz_size_t item_length;

            // We already validated this in first pass, so this should not fail
            sz_py_export_string_like(item, &item_start, &item_length);

            // Copy the string data
            memcpy(data_buffer + current_offset, item_start, item_length);
            current_offset += item_length;

            // Store offset (Apache Arrow format: offset after this string)
            if (use_64bit) { ((sz_u64_t *)offsets)[i + 1] = current_offset; }
            else { ((sz_u32_t *)offsets)[i + 1] = current_offset; }
        }

        // Setup the consecutive layout (32-bit or 64-bit)
        if (use_64bit) {
            self->layout = STRS_U64_TAPE;
            self->data.u64_tape.count = count;
            self->data.u64_tape.data = data_buffer;
            self->data.u64_tape.offsets = (sz_u64_t *)offsets;
            self->data.u64_tape.allocator = allocator;
        }
        else {
            self->layout = STRS_U32_TAPE;
            self->data.u32_tape.count = count;
            self->data.u32_tape.data = data_buffer;
            self->data.u32_tape.offsets = (sz_u32_t *)offsets;
            self->data.u32_tape.allocator = allocator;
        }

        return 0;
    }
}

// The inefficient `Strs_init` path initializing from a Pythonic iterable of strings.
static int Strs_init_from_iterable(Strs *self, PyObject *sequence_obj, int view) {
    // Get an iterator from the object
    PyObject *iterator = PyObject_GetIter(sequence_obj);
    if (!iterator) {
        PyErr_SetString(PyExc_TypeError, "Object is not iterable");
        return -1;
    }

    if (view) {
        // View mode is not supported for iterators because we can't safely keep references
        // to all the individual string objects without significant overhead
        Py_DECREF(iterator);
        PyErr_SetString(PyExc_ValueError, "View mode (view=True) is not supported for iterators. "
                                          "Use view=False to create a copy, or convert to a list/tuple first.");
        return -1;
    }

    // Initialize allocator for memory management
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_default(&allocator);

    // Incrementally allocate a new tape to fit all of the items
    sz_size_t data_capacity = 4096;
    sz_size_t offsets_capacity = 16;
    sz_size_t count = 0;
    sz_size_t total_bytes = 0;
    int use_64bit = 0; // Start with 32-bit

    sz_ptr_t data_buffer = (sz_ptr_t)allocator.allocate(data_capacity, allocator.handle);
    void *offsets = allocator.allocate(offsets_capacity * sizeof(sz_u32_t), allocator.handle); // Start with 32-bit

    if (!data_buffer || !offsets) {
        if (data_buffer) allocator.free(data_buffer, data_capacity, allocator.handle);
        if (offsets) allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
        Py_DECREF(iterator);
        PyErr_NoMemory();
        return -1;
    }

    // Set initial offset to 0 (Apache Arrow format: N+1 offsets for N strings)
    if (use_64bit) { ((sz_u64_t *)offsets)[0] = 0; }
    else { ((sz_u32_t *)offsets)[0] = 0; }

    // Iterate through all items
    PyObject *item;
    while ((item = PyIter_Next(iterator))) {
        sz_cptr_t item_start;
        sz_size_t item_length;
        if (!sz_py_export_string_like(item, &item_start, &item_length)) {
            Py_DECREF(item);
            allocator.free(data_buffer, data_capacity, allocator.handle);
            allocator.free(offsets, offsets_capacity * (use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t)),
                           allocator.handle);
            Py_DECREF(iterator);
            PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", count);
            return -1;
        }

        // Check if adding this string would exceed UINT32_MAX and switch to 64-bit
        if (!use_64bit && total_bytes + item_length > UINT32_MAX) {
            // Convert offsets from 32-bit to 64-bit
            sz_size_t new_offsets_size = offsets_capacity * sizeof(sz_u64_t);
            sz_u64_t *new_offsets = (sz_u64_t *)allocator.allocate(new_offsets_size, allocator.handle);
            if (!new_offsets) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
                Py_DECREF(iterator);
                PyErr_NoMemory();
                return -1;
            }

            // Copy existing 32-bit offsets to 64-bit (including initial 0 and all current offsets)
            sz_u32_t *old_offsets = (sz_u32_t *)offsets;
            for (sz_size_t i = 0; i <= count; i++) { new_offsets[i] = old_offsets[i]; }

            allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
            offsets = new_offsets;
            use_64bit = 1;
        }

        // Grow data buffer if needed (doubling strategy)
        while (total_bytes + item_length > data_capacity) {
            sz_size_t new_capacity = data_capacity * 2;
            sz_ptr_t new_buffer = (sz_ptr_t)allocator.allocate(new_capacity, allocator.handle);
            if (!new_buffer) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * (use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t)),
                               allocator.handle);
                Py_DECREF(iterator);
                PyErr_NoMemory();
                return -1;
            }
            memcpy(new_buffer, data_buffer, total_bytes);
            allocator.free(data_buffer, data_capacity, allocator.handle);
            data_buffer = new_buffer;
            data_capacity = new_capacity;
        }

        // Grow offsets array if needed (doubling strategy)
        // Need space for count+2 offsets total (0, 1, ..., count+1)
        if (count + 1 >= offsets_capacity) {
            sz_size_t new_capacity = offsets_capacity * 2;
            sz_size_t element_size = use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t);
            if (new_capacity > SIZE_MAX / element_size) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
                Py_DECREF(iterator);
                PyErr_SetString(PyExc_MemoryError, "Too many strings");
                return -1;
            }

            void *new_offsets = allocator.allocate(new_capacity * element_size, allocator.handle);
            if (!new_offsets) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
                Py_DECREF(iterator);
                PyErr_NoMemory();
                return -1;
            }
            memcpy(new_offsets, offsets, (count + 1) * element_size);
            allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
            offsets = new_offsets;
            offsets_capacity = new_capacity;
        }

        // Copy the string data
        memcpy(data_buffer + total_bytes, item_start, item_length);
        total_bytes += item_length;
        count++;

        // Store next offset (end of the string we just added)
        if (use_64bit) { ((sz_u64_t *)offsets)[count] = total_bytes; }
        else { ((sz_u32_t *)offsets)[count] = total_bytes; }

        Py_DECREF(item);
    }

    Py_DECREF(iterator);

    // Check for errors during iteration
    if (PyErr_Occurred()) {
        allocator.free(data_buffer, data_capacity, allocator.handle);
        allocator.free(offsets, offsets_capacity * (use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t)), allocator.handle);
        return -1;
    }

    // Handle empty iterator
    if (count == 0) {
        allocator.free(data_buffer, data_capacity, allocator.handle);
        allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        self->data.fragmented.allocator = allocator;
        self->data.fragmented.parent = NULL;
        return 0;
    }

    // Shrink buffers to actual size
    sz_ptr_t final_buffer = (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle);
    if (final_buffer) {
        memcpy(final_buffer, data_buffer, total_bytes);
        allocator.free(data_buffer, data_capacity, allocator.handle);
        data_buffer = final_buffer;
    }

    sz_size_t element_size = use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t);
    sz_size_t final_offsets_size = (count + 1) * element_size;
    void *final_offsets = allocator.allocate(final_offsets_size, allocator.handle);
    if (final_offsets) {
        memcpy(final_offsets, offsets, final_offsets_size);
        allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
        offsets = final_offsets;
    }

    // Setup the consecutive layout (32-bit or 64-bit)
    if (use_64bit) {
        self->layout = STRS_U64_TAPE;
        self->data.u64_tape.count = count;
        self->data.u64_tape.data = data_buffer;
        self->data.u64_tape.offsets = (sz_u64_t *)offsets;
        self->data.u64_tape.allocator = allocator;
    }
    else {
        self->layout = STRS_U32_TAPE;
        self->data.u32_tape.count = count;
        self->data.u32_tape.data = data_buffer;
        self->data.u32_tape.offsets = (sz_u32_t *)offsets;
        self->data.u32_tape.allocator = allocator;
    }

    return 0;
}

static int Strs_init(Strs *self, PyObject *args, PyObject *kwargs) {

    // Manual argument parsing for performance
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 2) {
        PyErr_SetString(PyExc_TypeError,
                        "Strs() takes at most 2 arguments: sequence of strings and a boolean indicator");
        return -1;
    }

    PyObject *sequence_obj = nargs >= 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *view_obj = nargs >= 2 ? PyTuple_GET_ITEM(args, 1) : NULL;
    int view = 0; // Default to copy mode

    // Parse keyword arguments if provided
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "sequence") == 0 && !sequence_obj) { sequence_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "view") == 0 && !view_obj) { view_obj = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    // Parse view flag
    if (view_obj) {
        view = PyObject_IsTrue(view_obj);
        if (view == -1) return -1;
    }

    // If no sequence provided, create empty Strs
    if (!sequence_obj) {
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        sz_memory_allocator_init_default(&self->data.fragmented.allocator);
        self->data.fragmented.parent = NULL;
        return 0;
    }

    // Check if it's an Arrow array (has `__arrow_c_array__` method)
    PyObject *arrow_method = PyObject_GetAttrString(sequence_obj, "__arrow_c_array__");
    if (arrow_method) {
        Py_DECREF(arrow_method);
        return Strs_init_from_pyarrow(self, sequence_obj, view);
    }

    // Handle more traditional Python sequences
    PyErr_Clear(); // Clear the attribute error from checking for `__arrow_c_array__`

    if (PyTuple_Check(sequence_obj)) { return Strs_init_from_tuple(self, sequence_obj, view); }
    else if (PyList_Check(sequence_obj)) { return Strs_init_from_list(self, sequence_obj, view); }
    else if (PyObject_HasAttrString(sequence_obj, "__iter__")) {
        return Strs_init_from_iterable(self, sequence_obj, view);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Strs() argument must be a tuple, list, or iterable");
        return -1;
    }

    return 0;
}

static void Strs_dealloc(Strs *self) {
    switch (self->layout) {
    case STRS_U32_TAPE:
        // Free owned data and offsets
        if (self->data.u32_tape.data) {
            sz_size_t data_size = self->data.u32_tape.offsets[self->data.u32_tape.count];
            self->data.u32_tape.allocator.free((sz_ptr_t)self->data.u32_tape.data, data_size,
                                               self->data.u32_tape.allocator.handle);
        }
        if (self->data.u32_tape.offsets) {
            sz_size_t offsets_size = (self->data.u32_tape.count + 1) * sizeof(sz_u32_t);
            self->data.u32_tape.allocator.free(self->data.u32_tape.offsets, offsets_size,
                                               self->data.u32_tape.allocator.handle);
        }
        break;

    case STRS_U64_TAPE:
        // Free owned data and offsets
        if (self->data.u64_tape.data) {
            sz_size_t data_size = self->data.u64_tape.offsets[self->data.u64_tape.count];
            self->data.u64_tape.allocator.free((sz_ptr_t)self->data.u64_tape.data, data_size,
                                               self->data.u64_tape.allocator.handle);
        }
        if (self->data.u64_tape.offsets) {
            sz_size_t offsets_size = (self->data.u64_tape.count + 1) * sizeof(sz_u64_t);
            self->data.u64_tape.allocator.free(self->data.u64_tape.offsets, offsets_size,
                                               self->data.u64_tape.allocator.handle);
        }
        break;

    case STRS_U32_TAPE_VIEW:
        // Views don't own data, just release parent reference
        Py_XDECREF(self->data.u32_tape_view.parent);
        break;

    case STRS_U64_TAPE_VIEW:
        // Views don't own data, just release parent reference
        Py_XDECREF(self->data.u64_tape_view.parent);
        break;

    case STRS_FRAGMENTED:
        // Free owned spans array and release parent reference
        if (self->data.fragmented.spans) {
            sz_size_t spans_size = self->data.fragmented.count * sizeof(sz_string_view_t);
            self->data.fragmented.allocator.free(self->data.fragmented.spans, spans_size,
                                                 self->data.fragmented.allocator.handle);
        }
        Py_XDECREF(self->data.fragmented.parent);
        break;
    }

    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyMethodDef Strs_methods[] = {
    {"shuffled", Strs_shuffled, SZ_METHOD_FLAGS, "Shuffle the elements of the Strs object."},        //
    {"sorted", Strs_sorted, SZ_METHOD_FLAGS, "Sort (in-place) the elements of the Strs object."},    //
    {"argsort", Strs_argsort, SZ_METHOD_FLAGS, "Provides the permutation to achieve sorted order."}, //
    {"sample", Strs_sample, SZ_METHOD_FLAGS, "Provides a random sample of a given size."},           //
    // {"to_pylist", Strs_to_pylist, SZ_METHOD_FLAGS, "Exports string-views to a native list of native strings."}, //
    {NULL, NULL, 0, NULL} // Sentinel
};

static char const doc_Strs[] = //
    "Strs(sequence, view=False)\\n"
    "\\n"
    "Space-efficient container for large collections of strings and their slices.\\n"
    "Optimized for memory efficiency and bulk operations on string collections.\\n"
    "\\n"
    "Args:\\n"
    "  sequence (list | tuple | generator | pyarrow.Array): Collection of strings to store.\\n"
    "  view (bool): If True, create a view into the original data instead of copying it.\\n"
    "\\n"
    "Features:\\n"
    "  - Memory-efficient storage with shared backing buffers\\n"
    "  - Zero-copy slicing and indexing operations\\n"
    "  - Bulk operations: sort(), shuffle(), sample()\\n"
    "  - Arrow integration: from_arrow() for zero-copy imports\\n"
    "  - Fast comparison operations with native containers\\n"
    "\\n"
    "Methods:\\n"
    "  - sort(): In-place sorting with custom comparison\\n"
    "  - argsort(): Get indices for sorted order\\n"
    "  - shuffle(): Randomize element order\\n"
    "  - sample(): Get random subset of elements\\n"
    "\\n"
    "Example:\\n"
    "  >>> strs = sz.Strs(['apple', 'banana', 'cherry'])\\n"
    "  >>> strs.sort()\\n"
    "  >>> list(strs)  # ['apple', 'banana', 'cherry']";

static PyTypeObject StrsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Strs",
    .tp_doc = doc_Strs,
    .tp_basicsize = sizeof(Strs),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Strs_init,
    .tp_dealloc = (destructor)Strs_dealloc,
    .tp_methods = Strs_methods,
    .tp_as_sequence = &Strs_as_sequence,
    .tp_as_mapping = &Strs_as_mapping,
    .tp_getset = Strs_getsetters,
    .tp_richcompare = Strs_richcompare,
    .tp_repr = (reprfunc)Strs_repr,
    .tp_str = (reprfunc)Strs_str,
};

#pragma endregion

static int parse_and_intersect_capabilities(PyObject *caps_obj, sz_capability_t *result) {
    if (!caps_obj) {
        PyErr_SetString(PyExc_TypeError, "capabilities must be a tuple or list of strings");
        return -1;
    }
    PyObject *seq = PySequence_Fast(caps_obj, "capabilities must be a tuple or list of strings");
    if (!seq) return -1;

    sz_capability_t requested_caps = 0;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    PyObject **items = PySequence_Fast_ITEMS(seq);

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = items[i];
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "capabilities must be strings");
            Py_DECREF(seq);
            return -1;
        }
        char const *cap_str = PyUnicode_AsUTF8(item);
        if (!cap_str) {
            Py_DECREF(seq);
            return -1;
        }
        sz_capability_t flag = sz_capability_from_string_implementation_(cap_str);
        if (flag == sz_caps_none_k) {
            PyErr_Format(PyExc_ValueError, "Unknown capability: %s", cap_str);
            Py_DECREF(seq);
            return -1;
        }
        requested_caps |= flag;
    }
    Py_DECREF(seq);

    // Intersect with hardware capabilities for safety
    *result = requested_caps & sz_capabilities();
    if (*result == 0) { *result = sz_cap_serial_k; }
    return 0;
}

static char const doc_reset_capabilities[] = //
    "reset_capabilities(names) -> None\n\n"
    "Sets the active SIMD/backend capabilities for this module and updates the\n"
    "runtime dispatch table. The provided names are intersected with hardware\n"
    "capabilities; if the result is empty, falls back to 'serial'.\n\n"
    "Side effects: updates stringzilla.__capabilities__ and __capabilities_str__.";

static PyObject *module_reset_capabilities(PyObject *self, PyObject *args) {
    PyObject *caps_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &caps_obj)) return NULL;

    sz_capability_t caps = 0;
    if (parse_and_intersect_capabilities(caps_obj, &caps) != 0) return NULL;

    // Update the dispatch table
    sz_dispatch_table_update(caps);

    // Recompute and set module-level capability exports
    sz_cptr_t cap_strings[SZ_CAPABILITIES_COUNT];
    sz_size_t cap_count = sz_capabilities_to_strings_implementation_(caps, cap_strings, SZ_CAPABILITIES_COUNT);
    PyObject *caps_tuple = PyTuple_New(cap_count);
    if (!caps_tuple) return NULL;
    for (sz_size_t i = 0; i < cap_count; i++) {
        PyObject *cap_str = PyUnicode_FromString(cap_strings[i]);
        if (!cap_str) {
            Py_DECREF(caps_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(caps_tuple, i, cap_str);
    }
    if (PyObject_SetAttrString(self, "__capabilities__", caps_tuple) != 0) {
        Py_DECREF(caps_tuple);
        return NULL;
    }
    Py_DECREF(caps_tuple);

    sz_cptr_t caps_str = sz_capabilities_to_string(caps);
    if (PyObject_SetAttrString(self, "__capabilities_str__", PyUnicode_FromString(caps_str)) != 0) { return NULL; }

    Py_RETURN_NONE;
}

static void stringzilla_cleanup(PyObject *m) {
    if (temporary_memory.start) free(temporary_memory.start);
    temporary_memory.start = NULL;
    temporary_memory.length = 0;
}

static PyMethodDef stringzilla_methods[] = {
    // Basic `str`, `bytes`, and `bytearray`-like functionality
    {"contains", (PyCFunction)Str_contains, SZ_METHOD_FLAGS, doc_contains},
    {"count", (PyCFunction)Str_count, SZ_METHOD_FLAGS, doc_count},
    {"splitlines", (PyCFunction)Str_splitlines, SZ_METHOD_FLAGS, doc_splitlines},
    {"startswith", (PyCFunction)Str_startswith, SZ_METHOD_FLAGS, doc_startswith},
    {"endswith", (PyCFunction)Str_endswith, SZ_METHOD_FLAGS, doc_endswith},
    {"translate", (PyCFunction)Str_translate, SZ_METHOD_FLAGS, doc_translate},
    {"decode", (PyCFunction)Str_decode, SZ_METHOD_FLAGS, doc_decode},
    {"equal", (PyCFunction)Str_like_equal, SZ_METHOD_FLAGS, doc_like_equal},

    // Bidirectional operations
    {"find", (PyCFunction)Str_find, SZ_METHOD_FLAGS, doc_find},
    {"index", (PyCFunction)Str_index, SZ_METHOD_FLAGS, doc_index},
    {"partition", (PyCFunction)Str_partition, SZ_METHOD_FLAGS, doc_partition},
    {"split", (PyCFunction)Str_split, SZ_METHOD_FLAGS, doc_split},
    {"rfind", (PyCFunction)Str_rfind, SZ_METHOD_FLAGS, doc_rfind},
    {"rindex", (PyCFunction)Str_rindex, SZ_METHOD_FLAGS, doc_rindex},
    {"rpartition", (PyCFunction)Str_rpartition, SZ_METHOD_FLAGS, doc_rpartition},
    {"rsplit", (PyCFunction)Str_rsplit, SZ_METHOD_FLAGS, doc_rsplit},

    // Character search extensions
    {"find_first_of", (PyCFunction)Str_find_first_of, SZ_METHOD_FLAGS, doc_find_first_of},
    {"find_last_of", (PyCFunction)Str_find_last_of, SZ_METHOD_FLAGS, doc_find_last_of},
    {"find_first_not_of", (PyCFunction)Str_find_first_not_of, SZ_METHOD_FLAGS, doc_find_first_not_of},
    {"find_last_not_of", (PyCFunction)Str_find_last_not_of, SZ_METHOD_FLAGS, doc_find_last_not_of},
    {"split_byteset", (PyCFunction)Str_split_byteset, SZ_METHOD_FLAGS, doc_split_byteset},
    {"rsplit_byteset", (PyCFunction)Str_rsplit_byteset, SZ_METHOD_FLAGS, doc_rsplit_byteset},

    // Lazily evaluated iterators
    {"split_iter", (PyCFunction)Str_split_iter, SZ_METHOD_FLAGS, doc_split_iter},
    {"rsplit_iter", (PyCFunction)Str_rsplit_iter, SZ_METHOD_FLAGS, doc_rsplit_iter},
    {"split_byteset_iter", (PyCFunction)Str_split_byteset_iter, SZ_METHOD_FLAGS, doc_split_byteset_iter},
    {"rsplit_byteset_iter", (PyCFunction)Str_rsplit_byteset_iter, SZ_METHOD_FLAGS, doc_rsplit_byteset_iter},

    // Dealing with larger-than-memory datasets
    {"offset_within", (PyCFunction)Str_offset_within, SZ_METHOD_FLAGS, doc_offset_within},
    {"write_to", (PyCFunction)Str_write_to, SZ_METHOD_FLAGS, doc_write_to},

    // Global unary extensions
    {"hash", (PyCFunction)Str_like_hash, SZ_METHOD_FLAGS, doc_like_hash},
    {"bytesum", (PyCFunction)Str_like_bytesum, SZ_METHOD_FLAGS, doc_like_bytesum},

    // Updating module capabilities
    {"reset_capabilities", (PyCFunction)module_reset_capabilities, METH_VARARGS, doc_reset_capabilities},

    {NULL, NULL, 0, NULL}};

static PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU",
    -1,
    stringzilla_methods,
    NULL,
    NULL,
    NULL,
    stringzilla_cleanup,
};

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;

    if (PyType_Ready(&StrType) < 0) return NULL;
    if (PyType_Ready(&FileType) < 0) return NULL;
    if (PyType_Ready(&StrsType) < 0) return NULL;
    if (PyType_Ready(&SplitIteratorType) < 0) return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL) return NULL;

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", sz_version_major(), sz_version_minor(), sz_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Define SIMD capabilities as a tuple
    {
        sz_capability_t caps = sz_capabilities();

        // Get capability strings using the new function
        sz_cptr_t cap_strings[SZ_CAPABILITIES_COUNT];
        sz_size_t cap_count = sz_capabilities_to_strings_implementation_(caps, cap_strings, SZ_CAPABILITIES_COUNT);

        // Create a Python tuple with the capabilities
        PyObject *caps_tuple = PyTuple_New(cap_count);
        if (!caps_tuple) {
            Py_XDECREF(m);
            return NULL;
        }

        for (sz_size_t i = 0; i < cap_count; i++) {
            PyObject *cap_str = PyUnicode_FromString(cap_strings[i]);
            if (!cap_str) {
                Py_DECREF(caps_tuple);
                Py_XDECREF(m);
                return NULL;
            }
            PyTuple_SET_ITEM(caps_tuple, i, cap_str);
        }

        if (PyModule_AddObject(m, "__capabilities__", caps_tuple) < 0) {
            Py_DECREF(caps_tuple);
            Py_XDECREF(m);
            return NULL;
        }

        // Also keep the old comma-separated string version for backward compatibility
        sz_cptr_t caps_str = sz_capabilities_to_string(caps);
        PyModule_AddStringConstant(m, "__capabilities_str__", caps_str);
    }

    Py_INCREF(&StrType);
    if (PyModule_AddObject(m, "Str", (PyObject *)&StrType) < 0) {
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&FileType);
    if (PyModule_AddObject(m, "File", (PyObject *)&FileType) < 0) {
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&StrsType);
    if (PyModule_AddObject(m, "Strs", (PyObject *)&StrsType) < 0) {
        Py_XDECREF(&StrsType);
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&SplitIteratorType);
    if (PyModule_AddObject(m, "SplitIterator", (PyObject *)&SplitIteratorType) < 0) {
        Py_XDECREF(&SplitIteratorType);
        Py_XDECREF(&StrsType);
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    // Export C API functions as a single capsule structure for StringZillas
    static PyAPI sz_py_api = {
        .sz_py_export_string_like = sz_py_export_string_like,
        .sz_py_export_strings_as_sequence = sz_py_export_strings_as_sequence,
        .sz_py_export_strings_as_u32tape = sz_py_export_strings_as_u32tape,
        .sz_py_export_strings_as_u64tape = sz_py_export_strings_as_u64tape,
        .sz_py_replace_strings_allocator = sz_py_replace_strings_allocator,
    };
    if (PyModule_AddObject(m, "_sz_py_api", PyCapsule_New(&sz_py_api, "_sz_py_api", NULL)) < 0) {
        Py_XDECREF(&SplitIteratorType);
        Py_XDECREF(&StrsType);
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    // Initialize temporary_memory, if needed
    temporary_memory.start = malloc(4096);
    temporary_memory.length = 4096 * (temporary_memory.start != NULL);
    return m;
}
