/**
 *  @file       lib.c
 *  @brief      Very light-weight CPython wrapper for StringZilla, with support for memory-mapping,
 *              native Python strings, Apache Arrow collections, and more.
 *  @author     Ash Vardanian
 *  @date       July 10, 2023
 *  @copyright  Copyright (c) 2023
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids `PyArg_ParseTupleAndKeywords` calls.
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

#include <Python.h> // Core CPython interfaces

#include <errno.h>  // `errno`
#include <stdio.h>  // `fopen`
#include <stdlib.h> // `rand`, `srand`
#include <string.h> // `memset`, `memcpy`
#include <time.h>   // `time`

#include <stringzilla/stringzilla.h>

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
    sz_cptr_t start;
    sz_size_t length;
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
    sz_cptr_t start;
    sz_size_t length;
} Str;

/**
 *  @brief  String-splitting separator.
 *
 *  Allows lazy evaluation of the `split` and `rsplit`, and can be used to create a `Strs` object.
 *  which might be more memory-friendly, than greedily invoking `str.split`.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_object;      //< For reference counting
    PyObject *separator_object; //< For reference counting

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
        STRS_CONSECUTIVE_32,
        STRS_CONSECUTIVE_64,
        STRS_REORDERED,
        STRS_MULTI_SOURCE,
    } type;

    union {
        /**
         *  Simple structure resembling Apache Arrow arrays of variable length strings.
         *  When you split a `Str`, that is under 4 GB in size, this is used for space-efficiency.
         *  The `end_offsets` contains `count`-many integers marking the end offset of part at a given
         *  index. The length of consecutive elements can be determined as the difference in consecutive
         *  offsets. The starting offset of the first element is zero bytes after the `start`.
         *  Every chunk will include a separator of length `separator_length` at the end, except for the
         *  last one.
         *
         *  The layout isn't exactly identical to Arrow, as we have an optional separator and we have one less offset.
         *  https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
         */
        struct consecutive_slices_32bit_t {
            size_t count;
            size_t separator_length;
            PyObject *parent_string;
            char const *start;
            uint32_t *end_offsets;
        } consecutive_32bit;

        /**
         *  Simple structure resembling Apache Arrow arrays of variable length strings.
         *  When you split a `Str`, over 4 GB long, this structure is used to indicate chunk offsets.
         *  The `end_offsets` contains `count`-many integers marking the end offset of part at a given
         *  index. The length of consecutive elements can be determined as the difference in consecutive
         *  offsets. The starting offset of the first element is zero bytes after the `start`.
         *  Every chunk will include a separator of length `separator_length` at the end, except for the
         *  last one.
         *
         *  The layout isn't exactly identical to Arrow, as we have an optional separator and we have one less offset.
         *  https://arrow.apache.org/docs/format/Columnar.html#variable-size-binary-layout
         */
        struct consecutive_slices_64bit_t {
            size_t count;
            size_t separator_length;
            PyObject *parent_string;
            char const *start;
            uint64_t *end_offsets;
        } consecutive_64bit;

        /**
         *  Once you sort, shuffle, or reorganize slices making up a larger string, this structure
         *  cn be used for space-efficient lookups.
         */
        struct reordered_slices_t {
            size_t count;
            PyObject *parent_string;
            sz_string_view_t *parts;
        } reordered;

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

static sz_cptr_t parts_get_start(sz_sequence_t *seq, sz_size_t i) {
    return ((sz_string_view_t const *)seq->handle)[i].start;
}

static sz_size_t parts_get_length(sz_sequence_t *seq, sz_size_t i) {
    return ((sz_string_view_t const *)seq->handle)[i].length;
}

void reverse_offsets(sz_sorted_idx_t *array, size_t length) {
    size_t i, j;
    // Swap array[i] and array[j]
    for (i = 0, j = length - 1; i < j; i++, j--) {
        sz_sorted_idx_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void reverse_haystacks(sz_string_view_t *array, size_t length) {
    size_t i, j;
    // Swap array[i] and array[j]
    for (i = 0, j = length - 1; i < j; i++, j--) {
        sz_string_view_t temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

void apply_order(sz_string_view_t *array, sz_sorted_idx_t *order, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (i == order[i]) continue;
        sz_string_view_t temp = array[i];
        size_t k = i, j;
        while (i != (j = (size_t)order[k])) {
            array[k] = array[j];
            order[k] = k;
            k = j;
        }
        array[k] = temp;
        order[k] = k;
    }
}

sz_bool_t export_string_like(PyObject *object, sz_cptr_t **start, sz_size_t *length) {
    if (PyUnicode_Check(object)) {
        // Handle Python str
        Py_ssize_t signed_length;
        *start = PyUnicode_AsUTF8AndSize(object, &signed_length);
        *length = (size_t)signed_length;
        return 1;
    }
    else if (PyBytes_Check(object)) {
        // Handle Python str
        Py_ssize_t signed_length;
        if (PyBytes_AsStringAndSize(object, (char **)start, &signed_length) == -1) {
            PyErr_SetString(PyExc_TypeError, "Mapping bytes failed");
            return 0;
        }
        *length = (size_t)signed_length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &StrType)) {
        Str *str = (Str *)object;
        *start = str->start;
        *length = str->length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &FileType)) {
        File *file = (File *)object;
        *start = file->start;
        *length = file->length;
        return 1;
    }
    return 0;
}

typedef void (*get_string_at_offset_t)(Strs *, Py_ssize_t, Py_ssize_t, PyObject **, char const **, size_t *);

void str_at_offset_consecutive_32bit(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                     PyObject **parent_string, char const **start, size_t *length) {
    uint32_t start_offset = (i == 0) ? 0 : strs->data.consecutive_32bit.end_offsets[i - 1];
    uint32_t end_offset = strs->data.consecutive_32bit.end_offsets[i] - //
                          strs->data.consecutive_32bit.separator_length * (i + 1 != count);
    *start = strs->data.consecutive_32bit.start + start_offset;
    *length = end_offset - start_offset;
    *parent_string = strs->data.consecutive_32bit.parent_string;
}

void str_at_offset_consecutive_64bit(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                     PyObject **parent_string, char const **start, size_t *length) {
    uint64_t start_offset = (i == 0) ? 0 : strs->data.consecutive_64bit.end_offsets[i - 1];
    uint64_t end_offset = strs->data.consecutive_64bit.end_offsets[i] - //
                          strs->data.consecutive_64bit.separator_length * (i + 1 != count);
    *start = strs->data.consecutive_64bit.start + start_offset;
    *length = end_offset - start_offset;
    *parent_string = strs->data.consecutive_64bit.parent_string;
}

void str_at_offset_reordered(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                             PyObject **parent_string, char const **start, size_t *length) {
    *start = strs->data.reordered.parts[i].start;
    *length = strs->data.reordered.parts[i].length;
    *parent_string = strs->data.reordered.parent_string;
}

get_string_at_offset_t str_at_offset_getter(Strs *strs) {
    switch (strs->type) {
    case STRS_CONSECUTIVE_32: return str_at_offset_consecutive_32bit;
    case STRS_CONSECUTIVE_64: return str_at_offset_consecutive_64bit;
    case STRS_REORDERED: return str_at_offset_reordered;
    default:
        // Unsupported type
        PyErr_SetString(PyExc_TypeError, "Unsupported type for conversion");
        return NULL;
    }
}

sz_bool_t prepare_strings_for_reordering(Strs *strs) {

    // Allocate memory for reordered slices
    size_t count = 0;
    void *old_buffer = NULL;
    get_string_at_offset_t getter = NULL;
    PyObject *parent_string = NULL;
    switch (strs->type) {
    case STRS_CONSECUTIVE_32:
        count = strs->data.consecutive_32bit.count;
        old_buffer = strs->data.consecutive_32bit.end_offsets;
        parent_string = strs->data.consecutive_32bit.parent_string;
        getter = str_at_offset_consecutive_32bit;
        break;
    case STRS_CONSECUTIVE_64:
        count = strs->data.consecutive_64bit.count;
        old_buffer = strs->data.consecutive_64bit.end_offsets;
        parent_string = strs->data.consecutive_64bit.parent_string;
        getter = str_at_offset_consecutive_64bit;
        break;
    // Already in reordered form
    case STRS_REORDERED: return 1;
    case STRS_MULTI_SOURCE: return 1;
    default:
        // Unsupported type
        PyErr_SetString(PyExc_TypeError, "Unsupported type for conversion");
        return 0;
    }

    sz_string_view_t *new_parts = (sz_string_view_t *)malloc(count * sizeof(sz_string_view_t));
    if (new_parts == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for reordered slices");
        return 0;
    }

    // Populate the new reordered array using get_string_at_offset
    for (size_t i = 0; i < count; ++i) {
        PyObject *parent_string;
        char const *start;
        size_t length;
        getter(strs, (Py_ssize_t)i, count, &parent_string, &start, &length);
        new_parts[i].start = start;
        new_parts[i].length = length;
    }

    // Release previous used memory.
    if (old_buffer) free(old_buffer);

    // Update the Strs object
    strs->type = STRS_REORDERED;
    strs->data.reordered.count = count;
    strs->data.reordered.parts = new_parts;
    strs->data.reordered.parent_string = parent_string;
    return 1;
}

sz_bool_t prepare_strings_for_extension(Strs *strs, size_t new_parents, size_t new_parts) { return 1; }

#pragma endregion

#pragma region Memory Mapping File

static void File_dealloc(File *self) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    if (self->start) {
        UnmapViewOfFile(self->start);
        self->start = NULL;
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
    if (self->start) {
        munmap(self->start, self->length);
        self->start = NULL;
        self->length = 0;
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
    self->start = NULL;
    self->length = 0;
    return (PyObject *)self;
}

static int File_init(File *self, PyObject *positional_args, PyObject *named_args) {
    char const *path;
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

    char *file = (char *)MapViewOfFile(self->mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (file == 0) {
        CloseHandle(self->mapping_handle);
        self->mapping_handle = NULL;
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_OSError, "Couldn't map the file!");
        return -1;
    }
    self->start = file;
    self->length = GetFileSize(self->file_handle, 0);
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
    size_t file_size = sb.st_size;
    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, self->file_descriptor, 0);
    if (map == MAP_FAILED) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_Format(PyExc_OSError, "Couldn't map the file at '%s': %s", path, strerror(errno));
        return -1;
    }
    self->start = map;
    self->length = file_size;
#endif

    return 0;
}

static PyMethodDef File_methods[] = { //
    {NULL, NULL, 0, NULL}};

static PyTypeObject FileType = {
    PyObject_HEAD_INIT(NULL).tp_name = "stringzilla.File",
    .tp_doc = "Memory mapped file class, that exposes the memory range for low-level access",
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
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "parent") == 0) {
                if (parent_obj) {
                    PyErr_SetString(PyExc_TypeError, "Received `parent` both as positional and keyword argument");
                    return -1;
                }
                parent_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "from") == 0) {
                if (from_obj) {
                    PyErr_SetString(PyExc_TypeError, "Received `from` both as positional and keyword argument");
                    return -1;
                }
                from_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "to") == 0) {
                if (to_obj) {
                    PyErr_SetString(PyExc_TypeError, "Received `to` both as positional and keyword argument");
                    return -1;
                }
                to_obj = value;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "Invalid keyword argument");
                return -1;
            }
        }
    }

    // Now, type-check and cast each argument
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
        self->start = NULL;
        self->length = 0;
    }
    // Increment the reference count of the parent
    else if (export_string_like(parent_obj, &self->start, &self->length)) {
        self->parent = parent_obj;
        Py_INCREF(parent_obj);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported parent type");
        return -1;
    }

    // Apply slicing
    size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(self->length, from, to, &normalized_offset, &normalized_length);
    self->start = ((char *)self->start) + normalized_offset;
    self->length = normalized_length;
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
    self->start = NULL;
    self->length = 0;
    return (PyObject *)self;
}

static void Str_dealloc(Str *self) {
    if (self->parent) { Py_XDECREF(self->parent); }
    else if (self->start) { free(self->start); }
    self->parent = NULL;
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Str_str(Str *self) { return PyUnicode_FromStringAndSize(self->start, self->length); }

static PyObject *Str_repr(Str *self) {
    // Interestingly, known-length string formatting only works in Python 3.12 and later.
    // https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_FromFormat
    if (PY_MAJOR_VERSION >= 3 && PY_MINOR_VERSION >= 12)
        return PyUnicode_FromFormat("sz.Str('%.*s')", (int)self->length, self->start);
    else {
        // Use a simpler formatting rule for older versions
        PyObject *str_obj = PyUnicode_FromStringAndSize(self->start, self->length);
        PyObject *result = PyUnicode_FromFormat("sz.Str('%U')", str_obj);
        Py_DECREF(str_obj);
        return result;
    }
}

static Py_hash_t Str_hash(Str *self) { return (Py_hash_t)sz_hash(self->start, self->length); }

static PyObject *Str_like_hash(PyObject *self, PyObject *args, PyObject *kwargs) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 1 || kwargs) {
        PyErr_SetString(PyExc_TypeError, "hash() expects exactly one positional argument");
        return NULL;
    }

    PyObject *text_object = is_member ? self : PyTuple_GET_ITEM(args, 0);
    sz_string_view_t text;

    // Validate and convert `text`
    if (!export_string_like(text_object, &text.start, &text.length)) {
        PyErr_SetString(PyExc_TypeError, "The text argument must be string-like");
        return NULL;
    }

    sz_u64_t result = sz_hash(text.start, text.length);
    return PyLong_FromSize_t((size_t)result);
}

static PyObject *Str_get_address(Str *self, void *closure) { return PyLong_FromSize_t((sz_size_t)self->start); }
static PyObject *Str_get_nbytes(Str *self, void *closure) { return PyLong_FromSize_t(self->length); }

static Py_ssize_t Str_len(Str *self) { return self->length; }

static PyObject *Str_getitem(Str *self, Py_ssize_t i) {

    // Negative indexing
    if (i < 0) i += self->length;

    if (i < 0 || (size_t)i >= self->length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    // Assuming the underlying data is UTF-8 encoded
    return PyUnicode_FromStringAndSize(self->start + i, 1);
}

static PyObject *Str_subscript(Str *self, PyObject *key) {
    if (PySlice_Check(key)) {
        // Sanity checks
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
        if (PySlice_AdjustIndices(self->length, &start, &stop, step) < 0) return NULL;
        if (step != 1) {
            PyErr_SetString(PyExc_IndexError, "Efficient step is not supported");
            return NULL;
        }

        // Create a new `Str` object
        Str *self_slice = (Str *)StrType.tp_alloc(&StrType, 0);
        if (self_slice == NULL && PyErr_NoMemory()) return NULL;

        // Set its properties based on the slice
        self_slice->start = self->start + start;
        self_slice->length = stop - start;
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
    view->buf = self->start;
    view->len = self->length;
    view->readonly = 1;
    view->itemsize = sizeof(char);
    view->format = "c"; // https://docs.python.org/3/library/struct.html#format-characters
    view->ndim = 1;
    view->shape = (Py_ssize_t *)&self->length; // 1-D array, so shape is just a pointer to the length
    view->strides = itemsize;                  // strides in a 1-D array is just the item size
    view->suboffsets = NULL;
    view->internal = NULL;

    Py_INCREF(self);
    return 0;
}

static void Str_releasebuffer(PyObject *_, Py_buffer *view) {
    // This function MUST NOT decrement view->obj, since that is done automatically
    // in PyBuffer_Release() (this scheme is useful for breaking reference cycles).
    // https://docs.python.org/3/c-api/typeobj.html#c.PyBufferProcs.bf_releasebuffer
}

/**
 *  @brief  Will be called by the `PySequence_Contains` to check presence of a substring.
 *  @return 1 if the string is present, 0 if it is not, -1 in case of error.
 *  @see    Docs: https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
 */
static int Str_in(Str *self, PyObject *needle_obj) {

    sz_string_view_t needle;
    if (!export_string_like(needle_obj, &needle.start, &needle.length)) {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument type");
        return -1;
    }

    return sz_find(self->start, self->length, needle.start, needle.length) != NULL;
}

static PyObject *Strs_get_tape(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_offsets_are_large(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_tape_address(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_offsets_address(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_tape_nbytes(Str *self, void *closure) { return NULL; }
static PyObject *Strs_get_offsets_nbytes(Str *self, void *closure) { return NULL; }

static Py_ssize_t Strs_len(Strs *self) {
    switch (self->type) {
    case STRS_CONSECUTIVE_32: return self->data.consecutive_32bit.count;
    case STRS_CONSECUTIVE_64: return self->data.consecutive_64bit.count;
    case STRS_REORDERED: return self->data.reordered.count;
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

    PyObject *parent = NULL;
    char const *start = NULL;
    size_t length = 0;
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }
    else
        getter(self, i, count, &parent, &start, &length);

    // Create a new `Str` object
    Str *view_copy = (Str *)StrType.tp_alloc(&StrType, 0);
    if (view_copy == NULL && PyErr_NoMemory()) return NULL;

    view_copy->start = start;
    view_copy->length = length;
    view_copy->parent = parent;
    Py_INCREF(parent);
    return view_copy;
}

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
    // REVIEW(alexbowe): Does this raise the appropriate Error on the Python side?
    if (result == NULL && PyErr_NoMemory()) return NULL;
    if (result_count == 0) {
        result->type = STRS_REORDERED;
        result->data.reordered.count = 0;
        result->data.reordered.parts = NULL;
        result->data.reordered.parent_string = NULL;
        return (PyObject *)result;
    }

    // If a step is requested, we have to create a new `REORDERED` Strs object,
    // even if the original one was `CONSECUTIVE`.
    if (step != 1) {
        sz_string_view_t *new_parts = (sz_string_view_t *)malloc(result_count * sizeof(sz_string_view_t));
        if (new_parts == NULL) {
            Py_XDECREF(result);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for reordered slices");
            return 0;
        }

        get_string_at_offset_t getter = str_at_offset_getter(self);
        result->type = STRS_REORDERED;
        result->data.reordered.count = result_count;
        result->data.reordered.parts = new_parts;
        result->data.reordered.parent_string = NULL;

        // Populate the new reordered array using get_string_at_offset
        size_t j = 0;
        if (step > 0)
            for (Py_ssize_t i = start; i < stop; i += step, ++j) {
                getter(self, i, count, &result->data.reordered.parent_string, &new_parts[j].start,
                       &new_parts[j].length);
            }
        else
            for (Py_ssize_t i = start; i > stop; i += step, ++j) {
                getter(self, i, count, &result->data.reordered.parent_string, &new_parts[j].start,
                       &new_parts[j].length);
            }

        return (PyObject *)result;
    }

    // Depending on the layout, the procedure will be different, but by now we know that:
    // - `start` and `stop` are valid indices
    // - `step` is 1
    // - `result_count` is positive
    // - the resulting object will have the same type as the original one
    result->type = self->type;
    switch (self->type) {

    case STRS_CONSECUTIVE_32: {
        typedef struct consecutive_slices_32bit_t consecutive_slices_t;
        consecutive_slices_t *from = &self->data.consecutive_32bit;
        consecutive_slices_t *to = &result->data.consecutive_32bit;
        to->count = result_count;

        // Allocate memory for the end offsets
        to->separator_length = from->separator_length;
        to->end_offsets = malloc(sizeof(uint32_t) * result_count);
        if (to->end_offsets == NULL && PyErr_NoMemory()) {
            Py_XDECREF(result);
            return NULL;
        }

        // Now populate the offsets
        size_t element_length;
        str_at_offset_consecutive_32bit(self, start, count, &to->parent_string, &to->start, &element_length);
        to->end_offsets[0] = element_length;
        for (Py_ssize_t i = 1; i < result_count; ++i) {
            to->end_offsets[i - 1] += from->separator_length;
            PyObject *element_parent = NULL;
            char const *element_start = NULL;
            str_at_offset_consecutive_32bit(self, start, count, &element_parent, &element_start, &element_length);
            to->end_offsets[i] = element_length + to->end_offsets[i - 1];
        }
        Py_INCREF(to->parent_string);
        break;
    }

    case STRS_CONSECUTIVE_64: {
        typedef struct consecutive_slices_64bit_t consecutive_slices_t;
        consecutive_slices_t *from = &self->data.consecutive_64bit;
        consecutive_slices_t *to = &result->data.consecutive_64bit;
        to->count = result_count;

        // Allocate memory for the end offsets
        to->separator_length = from->separator_length;
        to->end_offsets = malloc(sizeof(uint64_t) * result_count);
        if (to->end_offsets == NULL && PyErr_NoMemory()) {
            Py_XDECREF(result);
            return NULL;
        }

        // Now populate the offsets
        size_t element_length;
        str_at_offset_consecutive_64bit(self, start, count, &to->parent_string, &to->start, &element_length);
        to->end_offsets[0] = element_length;
        for (Py_ssize_t i = 1; i < result_count; ++i) {
            to->end_offsets[i - 1] += from->separator_length;
            PyObject *element_parent = NULL;
            char const *element_start = NULL;
            str_at_offset_consecutive_64bit(self, start, count, &element_parent, &element_start, &element_length);
            to->end_offsets[i] = element_length + to->end_offsets[i - 1];
        }
        Py_INCREF(to->parent_string);
        break;
    }

    case STRS_REORDERED: {
        struct reordered_slices_t *from = &self->data.reordered;
        struct reordered_slices_t *to = &result->data.reordered;
        to->count = result_count;
        to->parent_string = from->parent_string;

        to->parts = malloc(sizeof(sz_string_view_t) * to->count);
        if (to->parts == NULL && PyErr_NoMemory()) {
            Py_XDECREF(result);
            return NULL;
        }
        memcpy(to->parts, from->parts + start, sizeof(sz_string_view_t) * to->count);
        Py_INCREF(to->parent_string);
        break;
    }
    default:
        // Unsupported type
        PyErr_SetString(PyExc_TypeError, "Unsupported type for conversion");
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
    if (!export_string_like(needle_obj, &needle.start, &needle.length)) {
        PyErr_SetString(PyExc_TypeError, "The needle argument must be string-like");
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
        char const *start = NULL;
        size_t length = 0;
        getter(self, i, count, &parent, &start, &length);
        if (length == needle.length && sz_equal(start, needle.start, needle.length) == sz_true_k) return 1;
    }

    return 0;
}

static PyObject *Str_richcompare(PyObject *self, PyObject *other, int op) {

    sz_cptr_t a_start = NULL, b_start = NULL;
    sz_size_t a_length = 0, b_length = 0;
    if (!export_string_like(self, &a_start, &a_length) || !export_string_like(other, &b_start, &b_length))
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
            char const *ai_start = NULL, *bi_start = NULL;
            size_t ai_length = 0, bi_length = 0;
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
        if (!export_string_like(other_item, &bi.start, &bi.length)) {
            Py_DECREF(other_item);
            Py_DECREF(other_iter);
            PyErr_SetString(PyExc_TypeError, "The second container must contain string-like objects");
            return NULL;
        }

        // Both sequences aren't exhausted yet
        PyObject *ai_parent = NULL;
        char const *ai_start = NULL;
        size_t ai_length = 0;
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

static PyObject *Str_decode(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *encoding_obj = nargs > !is_member + 0 ? PyTuple_GET_ITEM(args, !is_member + 0) : NULL;
    PyObject *errors_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "encoding") == 0) { encoding_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "errors") == 0) { errors_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
    }

    // Convert `encoding` and `errors` to `NULL` if they are `None`
    if (encoding_obj == Py_None) encoding_obj = NULL;
    if (errors_obj == Py_None) errors_obj = NULL;

    sz_string_view_t text, encoding, errors;
    if ((!export_string_like(text_obj, &text.start, &text.length)) ||
        (encoding_obj && !export_string_like(encoding_obj, &encoding.start, &encoding.length)) ||
        (errors_obj && !export_string_like(errors_obj, &errors.start, &errors.length))) {
        PyErr_Format(PyExc_TypeError, "text, encoding, and errors must be string-like");
        return NULL;
    }

    if (encoding_obj == NULL) encoding = (sz_string_view_t) {"utf-8", 5};
    if (errors_obj == NULL) errors = (sz_string_view_t) {"strict", 6};

    // Python docs: https://docs.python.org/3/library/stdtypes.html#bytes.decode
    // CPython docs: https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_Decode
    return PyUnicode_Decode(text.start, text.length, encoding.start, errors.start);
}

/**
 *  @brief  Saves a StringZilla string to disk.
 */
static PyObject *Str_write_to(PyObject *self, PyObject *args, PyObject *kwargs) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs != !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_object = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *path_obj = PyTuple_GET_ITEM(args, !is_member + 0);

    // Parse keyword arguments
    if (kwargs) {
        PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument");
        return NULL;
    }

    sz_string_view_t text;
    sz_string_view_t path;

    // Validate and convert `text` and `path`
    if (!export_string_like(text_object, &text.start, &text.length) ||
        !export_string_like(path_obj, &path.start, &path.length)) {
        PyErr_SetString(PyExc_TypeError, "Text and path must be string-like");
        return NULL;
    }

    // There is a chance, the path isn't NULL-terminated, so copy it to a new buffer.
    // Many OSes have fairly low limit for the maximum path length.
    // On Windows its 260, but up to __around__ 32,767 characters are supported in extended API.
    // But it's better to be safe than sorry and use malloc :)
    //
    // https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=registry
    // https://doc.owncloud.com/server/next/admin_manual/troubleshooting/path_filename_length.html
    char *path_buffer = (char *)malloc(path.length + 1);
    if (path_buffer == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the path");
        return NULL;
    }
    memcpy(path_buffer, path.start, path.length);
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

/**
 *  @brief  Given a native StringZilla string, suggests it's offset within another native StringZilla string.
 *          Very practical when dealing with large files.
 *  @return Unsigned integer on success.
 */
static PyObject *Str_offset_within(PyObject *self, PyObject *args, PyObject *kwargs) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs != !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *slice_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *text_object = PyTuple_GET_ITEM(args, !is_member + 0);

    // Parse keyword arguments
    if (kwargs) {
        PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument");
        return NULL;
    }

    sz_string_view_t text;
    sz_string_view_t slice;

    // Validate and convert `text` and `slice`
    if (!export_string_like(text_object, &text.start, &text.length) ||
        !export_string_like(slice_obj, &slice.start, &slice.length)) {
        PyErr_SetString(PyExc_TypeError, "Text and slice must be string-like");
        return NULL;
    }

    if (slice.start < text.start || slice.start + slice.length > text.start + text.length) {
        PyErr_SetString(PyExc_ValueError, "The slice is not within the text bounds");
        return NULL;
    }

    return PyLong_FromSize_t((size_t)(slice.start - text.start));
}

/**
 *  @brief  Implementation function for all search-like operations, parameterized by a function callback.
 *  @return 1 on success, 0 on failure.
 */
static int _Str_find_implementation_( //
    PyObject *self, PyObject *args, PyObject *kwargs, sz_find_t finder, sz_bool_t is_reverse, Py_ssize_t *offset_out,
    sz_string_view_t *haystack_out, sz_string_view_t *needle_out) {

    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 3) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return 0;
    }

    PyObject *haystack_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *needle_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *start_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *end_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    // Parse keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) { start_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) { end_obj = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return 0;
            }
        }
    }

    sz_string_view_t haystack;
    sz_string_view_t needle;
    Py_ssize_t start, end;

    // Validate and convert `haystack` and `needle`
    if (!export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !export_string_like(needle_obj, &needle.start, &needle.length)) {
        PyErr_SetString(PyExc_TypeError, "Haystack and needle must be string-like");
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
    size_t normalized_offset, normalized_length;
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

static PyObject *Str_contains(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find, sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) { Py_RETURN_FALSE; }
    else { Py_RETURN_TRUE; }
}

static PyObject *Str_find(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find, sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_index(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find, sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_rfind(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_rfind, sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_rindex(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_rfind, sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *_Str_partition_implementation(PyObject *self, PyObject *args, PyObject *kwargs, sz_find_t finder,
                                               sz_bool_t is_reverse) {
    Py_ssize_t separator_index;
    sz_string_view_t text;
    sz_string_view_t separator;
    PyObject *result_tuple;

    // Use _Str_find_implementation_ to get the index of the separator
    if (!_Str_find_implementation_(self, args, kwargs, finder, is_reverse, &separator_index, &text, &separator))
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

    before->parent = self, before->start = text.start, before->length = separator_index;
    middle->parent = self, middle->start = text.start + separator_index, middle->length = separator.length;
    after->parent = self, after->start = text.start + separator_index + separator.length,
    after->length = text.length - separator_index - separator.length;

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

static PyObject *Str_partition(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_partition_implementation(self, args, kwargs, &sz_find, sz_false_k);
}

static PyObject *Str_rpartition(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_partition_implementation(self, args, kwargs, &sz_rfind, sz_true_k);
}

static PyObject *Str_count(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 4) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *haystack_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *needle_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *start_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *end_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;
    PyObject *allowoverlap_obj = nargs > !is_member + 3 ? PyTuple_GET_ITEM(args, !is_member + 3) : NULL;

    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) { start_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) { end_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "allowoverlap") == 0) { allowoverlap_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
    }

    sz_string_view_t haystack;
    sz_string_view_t needle;
    Py_ssize_t start = start_obj ? PyLong_AsSsize_t(start_obj) : 0;
    Py_ssize_t end = end_obj ? PyLong_AsSsize_t(end_obj) : PY_SSIZE_T_MAX;
    int allowoverlap = allowoverlap_obj ? PyObject_IsTrue(allowoverlap_obj) : 0;

    if (!export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !export_string_like(needle_obj, &needle.start, &needle.length))
        return PyErr_Format(PyExc_TypeError, "Haystack and needle must be string-like"), NULL;

    if ((start == -1 || end == -1 || allowoverlap == -1) && PyErr_Occurred()) return NULL;

    size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    size_t count = 0;
    if (needle.length == 0 || haystack.length == 0 || haystack.length < needle.length) { count = 0; }
    else if (allowoverlap) {
        while (haystack.length) {
            sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? ptr - haystack.start : haystack.length;
            count += found;
            haystack.start += offset + found;
            haystack.length -= offset + found;
        }
    }
    else {
        while (haystack.length) {
            sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? ptr - haystack.start : haystack.length;
            count += found;
            haystack.start += offset + needle.length;
            haystack.length -= offset + needle.length * found;
        }
    }

    return PyLong_FromSize_t(count);
}

static PyObject *_Str_edit_distance(PyObject *self, PyObject *args, PyObject *kwargs, sz_edit_distance_t function) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *str2_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *bound_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "bound") == 0) {
                if (bound_obj) {
                    PyErr_Format(PyExc_TypeError, "Received bound both as positional and keyword argument");
                    return NULL;
                }
                bound_obj = value;
            }
    }

    Py_ssize_t bound = 0; // Default value for bound
    if (bound_obj && ((bound = PyLong_AsSsize_t(bound_obj)) < 0)) {
        PyErr_Format(PyExc_ValueError, "Bound must be a non-negative integer");
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        PyErr_Format(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Allocate memory for the Levenshtein matrix
    sz_memory_allocator_t reusing_allocator;
    reusing_allocator.allocate = &temporary_memory_allocate;
    reusing_allocator.free = &temporary_memory_free;
    reusing_allocator.handle = &temporary_memory;

    sz_size_t distance =
        function(str1.start, str1.length, str2.start, str2.length, (sz_size_t)bound, &reusing_allocator);

    // Check for memory allocation issues
    if (distance == SZ_SIZE_MAX) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromSize_t(distance);
}

static PyObject *Str_edit_distance(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_edit_distance(self, args, kwargs, &sz_edit_distance);
}

static PyObject *Str_edit_distance_unicode(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_edit_distance(self, args, kwargs, &sz_edit_distance_utf8);
}

static PyObject *_Str_hamming_distance(PyObject *self, PyObject *args, PyObject *kwargs,
                                       sz_hamming_distance_t function) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *str2_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *bound_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "bound") == 0) {
                if (bound_obj) {
                    PyErr_Format(PyExc_TypeError, "Received bound both as positional and keyword argument");
                    return NULL;
                }
                bound_obj = value;
            }
    }

    Py_ssize_t bound = 0; // Default value for bound
    if (bound_obj && ((bound = PyLong_AsSsize_t(bound_obj)) < 0)) {
        PyErr_Format(PyExc_ValueError, "Bound must be a non-negative integer");
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        PyErr_Format(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    sz_size_t distance = function(str1.start, str1.length, str2.start, str2.length, (sz_size_t)bound);

    // Check for memory allocation issues
    if (distance == SZ_SIZE_MAX) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromSize_t(distance);
}

static PyObject *Str_hamming_distance(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_hamming_distance(self, args, kwargs, &sz_hamming_distance);
}

static PyObject *Str_hamming_distance_unicode(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_hamming_distance(self, args, kwargs, &sz_hamming_distance_utf8);
}

static PyObject *Str_alignment_score(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *str2_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *substitutions_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *gap_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "gap_score") == 0) {
                if (gap_obj) {
                    PyErr_Format(PyExc_TypeError, "Received the `gap_score` both as positional and keyword argument");
                    return NULL;
                }
                gap_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "substitution_matrix") == 0) {
                if (substitutions_obj) {
                    PyErr_Format(PyExc_TypeError,
                                 "Received the `substitution_matrix` both as positional and keyword argument");
                    return NULL;
                }
                substitutions_obj = value;
            }
    }

    Py_ssize_t gap = 1; // Default value for gap costs
    if (gap_obj && (gap = PyLong_AsSsize_t(gap_obj)) && (gap >= 128 || gap <= -128)) {
        PyErr_Format(PyExc_ValueError, "The `gap_score` must fit into an 8-bit signed integer");
        return NULL;
    }

    // Now extract the substitution matrix from the `substitutions_obj`.
    // It must conform to the buffer protocol, and contain a continuous 256x256 matrix of 8-bit signed integers.
    sz_error_cost_t const *substitutions;

    // Ensure the substitution matrix object is provided
    if (!substitutions_obj) {
        PyErr_Format(PyExc_TypeError, "No substitution matrix provided");
        return NULL;
    }

    // Request a buffer view
    Py_buffer substitutions_view;
    if (PyObject_GetBuffer(substitutions_obj, &substitutions_view, PyBUF_FULL)) {
        PyErr_Format(PyExc_TypeError, "Failed to get buffer from substitution matrix");
        return NULL;
    }

    // Validate the buffer
    if (substitutions_view.ndim != 2 || substitutions_view.shape[0] != 256 || substitutions_view.shape[1] != 256 ||
        substitutions_view.itemsize != sizeof(sz_error_cost_t)) {
        PyErr_Format(PyExc_ValueError, "Substitution matrix must be a 256x256 matrix of 8-bit signed integers");
        PyBuffer_Release(&substitutions_view);
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        PyErr_Format(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Assign the buffer's data to substitutions
    substitutions = (sz_error_cost_t const *)substitutions_view.buf;

    // Allocate memory for the Levenshtein matrix
    sz_memory_allocator_t reusing_allocator;
    reusing_allocator.allocate = &temporary_memory_allocate;
    reusing_allocator.free = &temporary_memory_free;
    reusing_allocator.handle = &temporary_memory;

    sz_ssize_t score = sz_alignment_score(str1.start, str1.length, str2.start, str2.length, substitutions,
                                          (sz_error_cost_t)gap, &reusing_allocator);

    // Don't forget to release the buffer view
    PyBuffer_Release(&substitutions_view);

    // Check for memory allocation issues
    if (score == SZ_SSIZE_MAX) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromSsize_t(score);
}

static PyObject *Str_startswith(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 3) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *prefix_obj = PyTuple_GET_ITEM(args, !is_member);
    PyObject *start_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *end_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

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
    if (!export_string_like(str_obj, &str.start, &str.length) ||
        !export_string_like(prefix_obj, &prefix.start, &prefix.length)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && end - start < str.length) { str.length = end - start; }

    if (str.length < prefix.length) { Py_RETURN_FALSE; }
    else if (strncmp(str.start, prefix.start, prefix.length) == 0) { Py_RETURN_TRUE; }
    else { Py_RETURN_FALSE; }
}

static PyObject *Str_endswith(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 3) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *suffix_obj = PyTuple_GET_ITEM(args, !is_member);
    PyObject *start_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *end_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

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
    if (!export_string_like(str_obj, &str.start, &str.length) ||
        !export_string_like(suffix_obj, &suffix.start, &suffix.length)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && end - start < str.length) { str.length = end - start; }

    if (str.length < suffix.length) { Py_RETURN_FALSE; }
    else if (strncmp(str.start + (str.length - suffix.length), suffix.start, suffix.length) == 0) { Py_RETURN_TRUE; }
    else { Py_RETURN_FALSE; }
}

static PyObject *Str_find_first_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find_char_from, sz_false_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_find_first_not_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find_char_not_from, sz_false_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_find_last_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_rfind_char_from, sz_true_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_find_last_not_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_rfind_char_not_from, sz_true_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

/**
 *  @brief  Given parsed split settings, constructs an iterator that would produce that split.
 */
static SplitIterator *Str_split_iter_(PyObject *text_object, PyObject *separator_object,             //
                                      sz_string_view_t const text, sz_string_view_t const separator, //
                                      int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length,
                                      sz_bool_t is_reverse) {

    // Create a new `SplitIterator` object
    SplitIterator *result_obj = (SplitIterator *)SplitIteratorType.tp_alloc(&SplitIteratorType, 0);
    if (result_obj == NULL && PyErr_NoMemory()) return NULL;

    // Set its properties based on the slice
    result_obj->text_object = text_object;
    result_obj->separator_object = separator_object;
    result_obj->text = text;
    result_obj->separator = separator;
    result_obj->finder = finder;

    result_obj->match_length = match_length;
    result_obj->include_match = keepseparator;
    result_obj->is_reverse = is_reverse;
    result_obj->max_parts = (sz_size_t)maxsplit + 1;
    result_obj->reached_tail = 0;

    // Increment the reference count of the parent
    Py_INCREF(result_obj->text_object);
    Py_XINCREF(result_obj->separator_object);
    return result_obj;
}

/**
 *  @brief  Implements the normal order split logic for both string-delimiters and character sets.
 *          Produuces one of the consecutive layouts - `STRS_CONSECUTIVE_64` or `STRS_CONSECUTIVE_32`.
 */
static Strs *Str_split_(PyObject *parent_string, sz_string_view_t const text, sz_string_view_t const separator,
                        int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length) {
    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) return NULL;

    // Initialize Strs object based on the splitting logic
    void *offsets_endings = NULL;
    size_t offsets_capacity = 0;
    size_t offsets_count = 0;
    size_t bytes_per_offset;
    if (text.length >= UINT32_MAX) {
        bytes_per_offset = 8;
        result->type = STRS_CONSECUTIVE_64;
        result->data.consecutive_64bit.start = text.start;
        result->data.consecutive_64bit.parent_string = parent_string;
        result->data.consecutive_64bit.separator_length = !keepseparator * match_length;
    }
    else {
        bytes_per_offset = 4;
        result->type = STRS_CONSECUTIVE_32;
        result->data.consecutive_32bit.start = text.start;
        result->data.consecutive_32bit.parent_string = parent_string;
        result->data.consecutive_32bit.separator_length = !keepseparator * match_length;
    }

    sz_bool_t reached_tail = 0;
    sz_size_t total_skipped = 0;
    sz_size_t max_parts = (sz_size_t)maxsplit + 1;
    while (!reached_tail) {

        sz_cptr_t match =
            offsets_count + 1 < max_parts
                ? finder(text.start + total_skipped, text.length - total_skipped, separator.start, separator.length)
                : NULL;

        sz_size_t part_end_offset;
        if (match) {
            part_end_offset = (match - text.start) + match_length;
            total_skipped = part_end_offset;
        }
        else {
            part_end_offset = text.length;
            total_skipped = text.length;
            reached_tail = 1;
        }

        // Reallocate offsets array if needed
        if (offsets_count >= offsets_capacity) {
            offsets_capacity = (offsets_capacity + 1) * 2;
            void *new_offsets = realloc(offsets_endings, offsets_capacity * bytes_per_offset);
            if (!new_offsets) {
                if (offsets_endings) free(offsets_endings);
            }
            offsets_endings = new_offsets;
        }

        // If the memory allocation has failed - discard the response
        if (!offsets_endings) {
            Py_XDECREF(result);
            PyErr_NoMemory();
            return NULL;
        }

        // Export the offset
        if (bytes_per_offset == 8) { ((uint64_t *)offsets_endings)[offsets_count] = (uint64_t)part_end_offset; }
        else { ((uint32_t *)offsets_endings)[offsets_count] = (uint32_t)part_end_offset; }
        offsets_count++;
    }

    // Populate the Strs object with the offsets
    if (bytes_per_offset == 8) {
        result->data.consecutive_64bit.end_offsets = offsets_endings;
        result->data.consecutive_64bit.count = offsets_count;
    }
    else {
        result->data.consecutive_32bit.end_offsets = offsets_endings;
        result->data.consecutive_32bit.count = offsets_count;
    }

    Py_INCREF(parent_string);
    return result;
}

/**
 *  @brief  Implements the reverse order split logic for both string-delimiters and character sets.
 *          Unlike the `Str_split_` can't use consecutive layouts and produces a `REAORDERED` one.
 */
static Strs *Str_rsplit_(PyObject *parent_string, sz_string_view_t const text, sz_string_view_t const separator,
                         int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length) {
    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) return NULL;

    // Initialize Strs object based on the splitting logic
    result->type = STRS_REORDERED;
    result->data.reordered.parent_string = parent_string;
    result->data.reordered.parts = NULL;
    result->data.reordered.count = 0;

    // Keep track of the memory usage
    sz_string_view_t *parts = NULL;
    sz_size_t parts_capacity = 0;
    sz_size_t parts_count = 0;

    sz_bool_t reached_tail = 0;
    sz_size_t total_skipped = 0;
    sz_size_t max_parts = (sz_size_t)maxsplit + 1;
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
            parts_capacity = (parts_capacity + 1) * 2;
            sz_string_view_t *new_parts = (sz_string_view_t *)realloc(parts, parts_capacity * sizeof(sz_string_view_t));
            if (!new_parts) {
                if (parts) free(parts);
            }
            parts = new_parts;
        }

        // If the memory allocation has failed - discard the response
        if (!parts) {
            Py_XDECREF(result);
            PyErr_NoMemory();
            return NULL;
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

    result->data.reordered.parts = parts;
    result->data.reordered.count = parts_count;
    Py_INCREF(parent_string);
    return result;
}

/**
 *  @brief  Proxy routing requests like `Str.split`, `Str.rsplit`, `Str.split_charset` and `Str.rsplit_charset`
 *          to `Str_split_` and `Str_rsplit_` implementations, parsing function arguments.
 */
static PyObject *Str_split_with_known_callback(PyObject *self, PyObject *args, PyObject *kwargs, //
                                               sz_find_t finder, sz_size_t match_length,         //
                                               sz_bool_t is_reverse, sz_bool_t is_lazy_iterator) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 3) {
        PyErr_SetString(PyExc_TypeError, "sz.split() received unsupported number of arguments");
        return NULL;
    }

    PyObject *text_object = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *separator_object = nargs > !is_member + 0 ? PyTuple_GET_ITEM(args, !is_member + 0) : NULL;
    PyObject *maxsplit_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *keepseparator_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "separator") == 0) { separator_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0) { maxsplit_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "keepseparator") == 0) { keepseparator_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
        }
    }

    sz_string_view_t text;
    sz_string_view_t separator;
    int keepseparator;
    Py_ssize_t maxsplit;

    // Validate and convert `text`
    if (!export_string_like(text_object, &text.start, &text.length)) {
        PyErr_SetString(PyExc_TypeError, "The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `separator`
    if (separator_object) {
        if (!export_string_like(separator_object, &separator.start, &separator.length)) {
            PyErr_SetString(PyExc_TypeError, "The separator argument must be string-like");
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
        return Str_split_iter_(text_object, separator_object, text, separator, //
                               keepseparator, maxsplit, finder, match_length, is_reverse);
    else
        return !is_reverse ? Str_split_(text_object, text, separator, keepseparator, maxsplit, finder, match_length)
                           : Str_rsplit_(text_object, text, separator, keepseparator, maxsplit, finder, match_length);
}

static PyObject *Str_split(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find, 0, sz_false_k, sz_false_k);
}

static PyObject *Str_rsplit(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind, 0, sz_true_k, sz_false_k);
}

static PyObject *Str_split_charset(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find_char_from, 1, sz_false_k, sz_false_k);
}

static PyObject *Str_rsplit_charset(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind_char_from, 1, sz_true_k, sz_false_k);
}

static PyObject *Str_split_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find, 0, sz_false_k, sz_true_k);
}

static PyObject *Str_rsplit_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind, 0, sz_true_k, sz_true_k);
}

static PyObject *Str_split_charset_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find_char_from, 1, sz_false_k, sz_true_k);
}

static PyObject *Str_rsplit_charset_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind_char_from, 1, sz_true_k, sz_true_k);
}

static PyObject *Str_splitlines(PyObject *self, PyObject *args, PyObject *kwargs) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 2) {
        PyErr_SetString(PyExc_TypeError, "splitlines() requires at least 1 argument");
        return NULL;
    }

    PyObject *text_object = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *keeplinebreaks_obj = nargs > !is_member ? PyTuple_GET_ITEM(args, !is_member) : NULL;
    PyObject *maxsplit_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "keeplinebreaks") == 0) { keeplinebreaks_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0) { maxsplit_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_string_view_t text;
    int keeplinebreaks;
    Py_ssize_t maxsplit = PY_SSIZE_T_MAX; // Default value for maxsplit

    // Validate and convert `text`
    if (!export_string_like(text_object, &text.start, &text.length)) {
        PyErr_SetString(PyExc_TypeError, "The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `keeplinebreaks`
    if (keeplinebreaks_obj) {
        keeplinebreaks = PyObject_IsTrue(keeplinebreaks_obj);
        if (keeplinebreaks == -1) {
            PyErr_SetString(PyExc_TypeError, "The keeplinebreaks argument must be a boolean");
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
    return Str_split_(text_object, text, separator, keeplinebreaks, maxsplit, &sz_find_char_from, 1);
}

static PyObject *Str_concat(PyObject *self, PyObject *other) {
    struct sz_string_view_t self_str, other_str;

    // Validate and convert `self`
    if (!export_string_like(self, &self_str.start, &self_str.length)) {
        PyErr_SetString(PyExc_TypeError, "The self object must be string-like");
        return NULL;
    }

    // Validate and convert `other`
    if (!export_string_like(other, &other_str.start, &other_str.length)) {
        PyErr_SetString(PyExc_TypeError, "The other object must be string-like");
        return NULL;
    }

    // Allocate a new Str instance
    Str *result_str = PyObject_New(Str, &StrType);
    if (result_str == NULL) { return NULL; }

    // Calculate the total length of the new string
    result_str->parent = NULL;
    result_str->length = self_str.length + other_str.length;

    // Allocate memory for the new string
    result_str->start = malloc(result_str->length);
    if (result_str->start == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for string concatenation");
        return NULL;
    }

    // Perform the string concatenation
    memcpy(result_str->start, self_str.start, self_str.length);
    memcpy(result_str->start + self_str.length, other_str.start, other_str.length);

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

#define SZ_METHOD_FLAGS METH_VARARGS | METH_KEYWORDS

static PyMethodDef Str_methods[] = {
    // Basic `str`, `bytes`, and `bytearray`-like functionality
    {"contains", Str_contains, SZ_METHOD_FLAGS, "Check if a string contains a substring."},
    {"count", Str_count, SZ_METHOD_FLAGS, "Count the occurrences of a substring."},
    {"splitlines", Str_splitlines, SZ_METHOD_FLAGS, "Split a string by line breaks."},
    {"startswith", Str_startswith, SZ_METHOD_FLAGS, "Check if a string starts with a given prefix."},
    {"endswith", Str_endswith, SZ_METHOD_FLAGS, "Check if a string ends with a given suffix."},
    {"decode", Str_decode, SZ_METHOD_FLAGS, "Decode the bytes into `str` with a given encoding"},

    // Bidirectional operations
    {"find", Str_find, SZ_METHOD_FLAGS, "Find the first occurrence of a substring."},
    {"index", Str_index, SZ_METHOD_FLAGS, "Find the first occurrence of a substring or raise error if missing."},
    {"partition", Str_partition, SZ_METHOD_FLAGS, "Splits string into 3-tuple: before, first match, after."},
    {"split", Str_split, SZ_METHOD_FLAGS, "Split a string by a separator."},
    {"rfind", Str_rfind, SZ_METHOD_FLAGS, "Find the last occurrence of a substring."},
    {"rindex", Str_rindex, SZ_METHOD_FLAGS, "Find the last occurrence of a substring or raise error if missing."},
    {"rpartition", Str_rpartition, SZ_METHOD_FLAGS, "Splits string into 3-tuple: before, last match, after."},
    {"rsplit", Str_rsplit, SZ_METHOD_FLAGS, "Split a string by a separator in reverse order."},

    // Edit distance extensions
    {"hamming_distance", Str_hamming_distance, SZ_METHOD_FLAGS,
     "Hamming distance between two strings, as the number of replaced bytes, and difference in length."},
    {"hamming_distance_unicode", Str_hamming_distance_unicode, SZ_METHOD_FLAGS,
     "Hamming distance between two strings, as the number of replaced unicode characters, and difference in length."},
    {"edit_distance", Str_edit_distance, SZ_METHOD_FLAGS,
     "Levenshtein distance between two strings, as the number of inserted, deleted, and replaced bytes."},
    {"edit_distance_unicode", Str_edit_distance_unicode, SZ_METHOD_FLAGS,
     "Levenshtein distance between two strings, as the number of inserted, deleted, and replaced unicode characters."},
    {"alignment_score", Str_alignment_score, SZ_METHOD_FLAGS,
     "Needleman-Wunsch alignment score given a substitution cost matrix."},

    // Character search extensions
    {"find_first_of", Str_find_first_of, SZ_METHOD_FLAGS,
     "Finds the first occurrence of a character from another string."},
    {"find_last_of", Str_find_last_of, SZ_METHOD_FLAGS,
     "Finds the last occurrence of a character from another string."},
    {"find_first_not_of", Str_find_first_not_of, SZ_METHOD_FLAGS,
     "Finds the first occurrence of a character not present in another string."},
    {"find_last_not_of", Str_find_last_not_of, SZ_METHOD_FLAGS,
     "Finds the last occurrence of a character not present in another string."},
    {"split_charset", Str_split_charset, SZ_METHOD_FLAGS, "Split a string by a set of character separators."},
    {"rsplit_charset", Str_rsplit_charset, SZ_METHOD_FLAGS,
     "Split a string by a set of character separators in reverse order."},

    // Lazily evaluated iterators
    {"split_iter", Str_split_iter, SZ_METHOD_FLAGS, "Create an iterator for splitting a string by a separator."},
    {"rsplit_iter", Str_rsplit_iter, SZ_METHOD_FLAGS,
     "Create an iterator for splitting a string by a separator in reverse order."},
    {"split_charset_iter", Str_split_charset_iter, SZ_METHOD_FLAGS,
     "Create an iterator for splitting a string by a set of character separators."},
    {"rsplit_charset_iter", Str_rsplit_charset_iter, SZ_METHOD_FLAGS,
     "Create an iterator for splitting a string by a set of character separators in reverse order."},

    // Dealing with larger-than-memory datasets
    {"offset_within", Str_offset_within, SZ_METHOD_FLAGS,
     "Return the raw byte offset of one binary string within another."},
    {"write_to", Str_write_to, SZ_METHOD_FLAGS, "Return the raw byte offset of one binary string within another."},

    {NULL, NULL, 0, NULL}};

static PyTypeObject StrType = {
    PyObject_HEAD_INIT(NULL).tp_name = "stringzilla.Str",
    .tp_doc = "Immutable string/slice class with SIMD and SWAR-accelerated operations",
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

    sz_cptr_t result_start;
    sz_size_t result_length;

    // Find the next needle
    sz_cptr_t found =
        self->max_parts > 1 //
            ? self->finder(self->text.start, self->text.length, self->separator.start, self->separator.length)
            : NULL;

    // We've reached the end of the string
    if (found == NULL) {
        result_start = self->text.start;
        result_length = self->text.length;
        self->text.length = 0;
        self->reached_tail = 1;
        self->max_parts = 0;
    }
    else {
        if (self->is_reverse) {
            result_start = found + self->match_length * !self->include_match;
            result_length = self->text.start + self->text.length - result_start;
            self->text.length = found - self->text.start;
        }
        else {
            result_start = self->text.start;
            result_length = found - self->text.start;
            self->text.start = found + self->match_length;
            self->text.length -= result_length + self->match_length;
            result_length += self->match_length * self->include_match;
        }
        self->max_parts--;
    }

    // Set its properties based on the slice
    result_obj->start = result_start;
    result_obj->length = result_length;
    result_obj->parent = self->text_object;

    // Increment the reference count of the parent
    Py_INCREF(self->text_object);
    return (PyObject *)result_obj;
}

static void SplitIteratorType_dealloc(SplitIterator *self) {
    Py_XDECREF(self->text_object);
    Py_XDECREF(self->separator_object);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SplitIteratorType_iter(PyObject *self) {
    Py_INCREF(self); // Iterator should return itself in __iter__.
    return self;
}

static PyTypeObject SplitIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.SplitIterator",
    .tp_basicsize = sizeof(SplitIterator),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)SplitIteratorType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Text-splitting iterator",
    .tp_iter = SplitIteratorType_iter,
    .tp_iternext = (iternextfunc)SplitIteratorType_next,
};

#pragma endregion

#pragma region Strs

static PyObject *Strs_shuffle(Strs *self, PyObject *args, PyObject *kwargs) {
    unsigned int seed = time(NULL); // Default seed

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "shuffle() takes at most 1 positional argument");
        return NULL;
    }
    else if (nargs == 1) {
        PyObject *seed_obj = PyTuple_GET_ITEM(args, 0);
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_obj);
    }

    // Check for keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) {
                if (nargs == 1) {
                    PyErr_SetString(PyExc_TypeError, "Received seed both as positional and keyword argument");
                    return NULL;
                }
                if (!PyLong_Check(value)) {
                    PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
                    return NULL;
                }
                seed = PyLong_AsUnsignedLong(value);
            }
            else {
                PyErr_Format(PyExc_TypeError, "Received an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    // Change the layout
    if (!prepare_strings_for_reordering(self)) {
        PyErr_Format(PyExc_TypeError, "Failed to prepare the sequence for shuffling");
        return NULL;
    }

    // Get the parts and their count
    struct reordered_slices_t *reordered = &self->data.reordered;
    sz_string_view_t *parts = reordered->parts;
    size_t count = reordered->count;

    // Fisher-Yates Shuffle Algorithm
    srand(seed);
    for (size_t i = count - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        // Swap parts[i] and parts[j]
        sz_string_view_t temp = parts[i];
        parts[i] = parts[j];
        parts[j] = temp;
    }

    Py_RETURN_NONE;
}

static sz_bool_t Strs_sort_(Strs *self, sz_string_view_t **parts_output, sz_sorted_idx_t **order_output,
                            sz_size_t *count_output) {
    // Change the layout
    if (!prepare_strings_for_reordering(self)) {
        PyErr_Format(PyExc_TypeError, "Failed to prepare the sequence for sorting");
        return 0;
    }

    // Get the parts and their count
    // The only possible `self->type` by now is the `STRS_REORDERED`
    sz_string_view_t *parts = self->data.reordered.parts;
    size_t count = self->data.reordered.count;

    // Allocate temporary memory to store the ordering offsets
    size_t memory_needed = sizeof(sz_sorted_idx_t) * count;
    if (temporary_memory.length < memory_needed) {
        temporary_memory.start = realloc(temporary_memory.start, memory_needed);
        temporary_memory.length = memory_needed;
    }
    if (!temporary_memory.start) {
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the Levenshtein matrix");
        return 0;
    }

    // Call our sorting algorithm
    sz_sequence_t sequence;
    memset(&sequence, 0, sizeof(sequence));
    sequence.order = (sz_sorted_idx_t *)temporary_memory.start;
    sequence.count = count;
    sequence.handle = parts;
    sequence.get_start = parts_get_start;
    sequence.get_length = parts_get_length;
    for (sz_sorted_idx_t i = 0; i != sequence.count; ++i) sequence.order[i] = i;
    sz_sort(&sequence);

    // Export results
    *parts_output = parts;
    *order_output = sequence.order;
    *count_output = sequence.count;
    return 1;
}

static PyObject *Strs_sort(Strs *self, PyObject *args, PyObject *kwargs) {
    PyObject *reverse_obj = NULL; // Default is not reversed

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "sort() takes at most 1 positional argument");
        return NULL;
    }
    else if (nargs == 1) { reverse_obj = PyTuple_GET_ITEM(args, 0); }

    // Check for keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0) {
                if (reverse_obj) {
                    PyErr_SetString(PyExc_TypeError, "Received reverse both as positional and keyword argument");
                    return NULL;
                }
                reverse_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "Received an unexpected keyword argument '%U'", key);
                return NULL;
            }
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

    sz_string_view_t *parts = NULL;
    sz_size_t *order = NULL;
    sz_size_t count = 0;
    if (!Strs_sort_(self, &parts, &order, &count)) return NULL;

    // Apply the sorting algorithm here, considering the `reverse` value
    if (reverse) reverse_offsets(order, count);

    // Apply the new order.
    apply_order(parts, order, count);

    Py_RETURN_NONE;
}

static PyObject *Strs_order(Strs *self, PyObject *args, PyObject *kwargs) {
    PyObject *reverse_obj = NULL; // Default is not reversed

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "order() takes at most 1 positional argument");
        return NULL;
    }
    else if (nargs == 1) { reverse_obj = PyTuple_GET_ITEM(args, 0); }

    // Check for keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0) {
                if (reverse_obj) {
                    PyErr_SetString(PyExc_TypeError, "Received reverse both as positional and keyword argument");
                    return NULL;
                }
                reverse_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "Received an unexpected keyword argument '%U'", key);
                return NULL;
            }
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

    sz_string_view_t *parts = NULL;
    sz_sorted_idx_t *order = NULL;
    sz_size_t count = 0;
    if (!Strs_sort_(self, &parts, &order, &count)) return NULL;

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
    //      memcpy(numpy_data_ptr, order, count * sizeof(sz_sorted_idx_t));
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

static PyObject *Strs_sample(Strs *self, PyObject *args, PyObject *kwargs) {
    PyObject *seed_obj = NULL;
    PyObject *sample_size_obj = NULL;

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "sample() takes 1 positional argument and 1 keyword argument");
        return NULL;
    }
    else if (nargs == 1) { sample_size_obj = PyTuple_GET_ITEM(args, 0); }

    // Parse keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) { seed_obj = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return 0;
            }
        }
    }

    // Translate the seed and the sample size to C types
    size_t sample_size = 0;
    if (sample_size_obj) {
        if (!PyLong_Check(sample_size_obj)) {
            PyErr_SetString(PyExc_TypeError, "The sample size must be an integer");
            return NULL;
        }
        sample_size = PyLong_AsSize_t(sample_size_obj);
    }
    unsigned int seed = time(NULL); // Default seed
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

    result->type = STRS_REORDERED;
    result->data.reordered.count = 0;
    result->data.reordered.parts = NULL;
    result->data.reordered.parent_string = NULL;
    if (sample_size == 0) { return (PyObject *)result; }

    // Now create a new Strs object with the sampled strings
    sz_string_view_t *result_parts = malloc(sample_size * sizeof(sz_string_view_t));
    if (!result_parts) {
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
    for (Py_ssize_t i = 0; i < sample_size; i++) {
        size_t index = rand() % count;
        getter(self, index, count, &parent_string, &result_parts[i].start, &result_parts[i].length);
    }

    // Update the Strs object
    result->type = STRS_REORDERED;
    result->data.reordered.count = sample_size;
    result->data.reordered.parts = result_parts;
    result->data.reordered.parent_string = parent_string;
    return result;
}

/**
 *  @brief Exports a string to a UTF-8 buffer, escaping single quotes.
 *  @param[out] did_fit Populated with 1 if the string is fully exported, 0 if it didn't fit.
 */
char const *export_escaped_unquoted_to_utf8_buffer(char const *cstr, size_t cstr_length, //
                                                   char *buffer, size_t buffer_length,   //
                                                   int *did_fit) {
    char const *const cstr_end = cstr + cstr_length;
    char *const buffer_end = buffer + buffer_length;
    *did_fit = 1;

    while (cstr < cstr_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        _sz_extract_utf8_rune(cstr, &rune, &rune_length);
        if (rune_length == 1 && buffer + 2 < buffer_end) {
            if (*cstr == '\'') {
                *(buffer++) = '\\';
                *(buffer++) = '\'';
                cstr++;
            }
            else if (*cstr == '\'') {
                *(buffer++) = '\\';
                *(buffer++) = '\'';
                cstr++;
            }
            else { *(buffer++) = *(cstr++); }
        }
        else if (buffer + rune_length < buffer_end) {
            memcpy(buffer, cstr, rune_length);
            buffer += rune_length;
            cstr += rune_length;
        }
        else {
            *did_fit = 0;
            break;
        }
    }

    return buffer;
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
    char *repr_buffer_ptr = &repr_buffer[0];
    char const *const repr_buffer_end = repr_buffer_ptr + 1024;

    // Start of the array
    memcpy(repr_buffer_ptr, "sz.Strs([", 9);
    repr_buffer_ptr += 9;

    size_t count = Strs_len(self);
    PyObject *parent_string;

    // In the worst case, we must have enough space for `...', ...])`
    // That's extra 11 bytes of content.
    char const *non_fitting_array_tail = "... ])";
    int const non_fitting_array_tail_length = 6;

    // If the whole string doesn't fit, even before the `non_fitting_array_tail` tail,
    // we need to add `, '` separator of 3 bytes.
    for (size_t i = 0; i < count && repr_buffer_ptr + (non_fitting_array_tail_length + 3) < repr_buffer_end; i++) {
        char const *cstr_start = NULL;
        size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);

        if (i > 0) { *(repr_buffer_ptr++) = ',', *(repr_buffer_ptr++) = ' '; }
        *(repr_buffer_ptr++) = '\'';

        int did_fit;
        repr_buffer_ptr = export_escaped_unquoted_to_utf8_buffer(
            cstr_start, cstr_length, repr_buffer_ptr, repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length,
            &did_fit);
        // If it didn't fit, let's put an ellipsis
        if (!did_fit) {
            memcpy(repr_buffer_ptr, non_fitting_array_tail, non_fitting_array_tail_length);
            repr_buffer_ptr += non_fitting_array_tail_length;
            return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
        }
        else
            *(repr_buffer_ptr++) = '\''; // Close the string
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
    size_t count = Strs_len(self);
    PyObject *parent_string;
    size_t total_bytes = 2; // opening and closing square brackets
    for (size_t i = 0; i < count; i++) {
        char const *cstr_start = NULL;
        size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);
        total_bytes += cstr_length;
        total_bytes += 2;             // For the single quotes
        if (i != 0) total_bytes += 2; // For the preceding comma and space

        // Count the number of single quotes in the string
        while (cstr_length) {
            char quote = '\'';
            sz_cptr_t next_quote = sz_find_byte(cstr_start, cstr_length, &quote);
            if (next_quote == NULL) break;
            total_bytes++;
            cstr_length -= next_quote - cstr_start;
            cstr_start = next_quote + 1;
        }
    }

    // Now allocate the memory for the concatenated string
    char *const result_buffer = malloc(total_bytes);
    if (!result_buffer) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the concatenated string");
        return NULL;
    }

    // Copy the strings into the result buffer
    char *result_ptr = result_buffer;
    *result_ptr++ = '[';
    for (size_t i = 0; i < count; i++) {
        if (i != 0) {
            *result_ptr++ = ',';
            *result_ptr++ = ' ';
        }
        char const *cstr_start = NULL;
        size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);
        *result_ptr++ = '\'';
        int did_fit;
        result_ptr = export_escaped_unquoted_to_utf8_buffer(cstr_start, cstr_length, result_ptr, total_bytes, &did_fit);
        *result_ptr++ = '\'';
    }

    *result_ptr++ = ']';
    return PyUnicode_FromStringAndSize(result_buffer, total_bytes);
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
    {NULL} // Sentinel
};

static PyMethodDef Strs_methods[] = {
    {"shuffle", Strs_shuffle, SZ_METHOD_FLAGS, "Shuffle (in-place) the elements of the Strs object."}, //
    {"sort", Strs_sort, SZ_METHOD_FLAGS, "Sort (in-place) the elements of the Strs object."},          //
    {"order", Strs_order, SZ_METHOD_FLAGS, "Provides the indexes to achieve sorted order."},           //
    {"sample", Strs_sample, SZ_METHOD_FLAGS, "Provides a random sample of a given size."},             //
    // {"to_pylist", Strs_to_pylist, SZ_METHOD_FLAGS, "Exports string-views to a native list of native strings."},
    // //
    {NULL, NULL, 0, NULL}};

static PyTypeObject StrsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Strs",
    .tp_doc = "Space-efficient container for large collections of strings and their slices",
    .tp_basicsize = sizeof(Strs),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_methods = Strs_methods,
    .tp_as_sequence = &Strs_as_sequence,
    .tp_as_mapping = &Strs_as_mapping,
    .tp_getset = Strs_getsetters,
    .tp_richcompare = Strs_richcompare,
    .tp_repr = (reprfunc)Strs_repr,
    .tp_str = (reprfunc)Strs_str,
};

#pragma endregion

static void stringzilla_cleanup(PyObject *m) {
    if (temporary_memory.start) free(temporary_memory.start);
    temporary_memory.start = NULL;
    temporary_memory.length = 0;
}

static PyMethodDef stringzilla_methods[] = {
    // Basic `str`, `bytes`, and `bytearray`-like functionality
    {"contains", Str_contains, SZ_METHOD_FLAGS, "Check if a string contains a substring."},
    {"count", Str_count, SZ_METHOD_FLAGS, "Count the occurrences of a substring."},
    {"splitlines", Str_splitlines, SZ_METHOD_FLAGS, "Split a string by line breaks."},
    {"startswith", Str_startswith, SZ_METHOD_FLAGS, "Check if a string starts with a given prefix."},
    {"endswith", Str_endswith, SZ_METHOD_FLAGS, "Check if a string ends with a given suffix."},
    {"decode", Str_decode, SZ_METHOD_FLAGS, "Decode the bytes into `str` with a given encoding"},

    // Bidirectional operations
    {"find", Str_find, SZ_METHOD_FLAGS, "Find the first occurrence of a substring."},
    {"index", Str_index, SZ_METHOD_FLAGS, "Find the first occurrence of a substring or raise error if missing."},
    {"partition", Str_partition, SZ_METHOD_FLAGS, "Splits string into 3-tuple: before, first match, after."},
    {"split", Str_split, SZ_METHOD_FLAGS, "Split a string by a separator."},
    {"rfind", Str_rfind, SZ_METHOD_FLAGS, "Find the last occurrence of a substring."},
    {"rindex", Str_rindex, SZ_METHOD_FLAGS, "Find the last occurrence of a substring or raise error if missing."},
    {"rpartition", Str_rpartition, SZ_METHOD_FLAGS, "Splits string into 3-tuple: before, last match, after."},
    {"rsplit", Str_rsplit, SZ_METHOD_FLAGS, "Split a string by a separator in reverse order."},

    // Edit distance extensions
    {"hamming_distance", Str_hamming_distance, SZ_METHOD_FLAGS,
     "Hamming distance between two strings, as the number of replaced bytes, and difference in length."},
    {"hamming_distance_unicode", Str_hamming_distance_unicode, SZ_METHOD_FLAGS,
     "Hamming distance between two strings, as the number of replaced unicode characters, and difference in "
     "length."},
    {"edit_distance", Str_edit_distance, SZ_METHOD_FLAGS,
     "Levenshtein distance between two strings, as the number of inserted, deleted, and replaced bytes."},
    {"edit_distance_unicode", Str_edit_distance_unicode, SZ_METHOD_FLAGS,
     "Levenshtein distance between two strings, as the number of inserted, deleted, and replaced unicode "
     "characters."},
    {"alignment_score", Str_alignment_score, SZ_METHOD_FLAGS,
     "Needleman-Wunsch alignment score given a substitution cost matrix."},

    // Character search extensions
    {"find_first_of", Str_find_first_of, SZ_METHOD_FLAGS,
     "Finds the first occurrence of a character from another string."},
    {"find_last_of", Str_find_last_of, SZ_METHOD_FLAGS,
     "Finds the last occurrence of a character from another string."},
    {"find_first_not_of", Str_find_first_not_of, SZ_METHOD_FLAGS,
     "Finds the first occurrence of a character not present in another string."},
    {"find_last_not_of", Str_find_last_not_of, SZ_METHOD_FLAGS,
     "Finds the last occurrence of a character not present in another string."},
    {"split_charset", Str_split_charset, SZ_METHOD_FLAGS, "Split a string by a set of character separators."},
    {"rsplit_charset", Str_rsplit_charset, SZ_METHOD_FLAGS,
     "Split a string by a set of character separators in reverse order."},

    // Lazily evaluated iterators
    {"split_iter", Str_split_iter, SZ_METHOD_FLAGS, "Create an iterator for splitting a string by a separator."},
    {"rsplit_iter", Str_rsplit_iter, SZ_METHOD_FLAGS,
     "Create an iterator for splitting a string by a separator in reverse order."},
    {"split_charset_iter", Str_split_charset_iter, SZ_METHOD_FLAGS,
     "Create an iterator for splitting a string by a set of character separators."},
    {"rsplit_charset_iter", Str_rsplit_charset_iter, SZ_METHOD_FLAGS,
     "Create an iterator for splitting a string by a set of character separators in reverse order."},

    // Dealing with larger-than-memory datasets
    {"offset_within", Str_offset_within, SZ_METHOD_FLAGS,
     "Return the raw byte offset of one binary string within another."},
    {"write_to", Str_write_to, SZ_METHOD_FLAGS, "Return the raw byte offset of one binary string within another."},

    // Global unary extensions
    {"hash", Str_like_hash, SZ_METHOD_FLAGS, "Hash a string or a byte-array."},

    {NULL, NULL, 0, NULL}};

static PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "SIMD-accelerated string search, sort, hashes, fingerprints, & edit distances",
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
        sprintf(version_str, "%d.%d.%d", STRINGZILLA_VERSION_MAJOR, STRINGZILLA_VERSION_MINOR,
                STRINGZILLA_VERSION_PATCH);
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Define SIMD capabilities
    {
        sz_capability_t caps = sz_capabilities();
        char caps_str[512];
        char const *serial = (caps & sz_cap_serial_k) ? "serial," : "";
        char const *neon = (caps & sz_cap_arm_neon_k) ? "neon," : "";
        char const *sve = (caps & sz_cap_arm_sve_k) ? "sve," : "";
        char const *avx2 = (caps & sz_cap_x86_avx2_k) ? "avx2," : "";
        char const *avx512f = (caps & sz_cap_x86_avx512f_k) ? "avx512f," : "";
        char const *avx512vl = (caps & sz_cap_x86_avx512vl_k) ? "avx512vl," : "";
        char const *avx512bw = (caps & sz_cap_x86_avx512bw_k) ? "avx512bw," : "";
        char const *avx512vbmi = (caps & sz_cap_x86_avx512vbmi_k) ? "avx512vbmi," : "";
        char const *gfni = (caps & sz_cap_x86_gfni_k) ? "gfni," : "";
        char const *avx512vbmi2 = (caps & sz_cap_x86_avx512vbmi2_k) ? "avx512vbmi2," : "";
        sprintf(caps_str, "%s%s%s%s%s%s%s%s%s%s", serial, neon, sve, avx2, avx512f, avx512vl, avx512bw, avx512vbmi,
                avx512vbmi2, gfni);
        PyModule_AddStringConstant(m, "__capabilities__", caps_str);
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

    // Initialize temporary_memory, if needed
    temporary_memory.start = malloc(4096);
    temporary_memory.length = 4096 * (temporary_memory.start != NULL);
    return m;
}
