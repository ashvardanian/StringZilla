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
#include <unistd.h> // `ssize_t`
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>            // Core CPython interfaces
#include <numpy/arrayobject.h> // NumPy

#include <string.h> // `memset`, `memcpy`

#include <stringzilla.h>

#pragma region Forward Declarations

static PyTypeObject FileType;
static PyTypeObject StrType;
static PyTypeObject StrsType;

static sz_string_view_t temporary_memory = {NULL, 0};

/**
 *  @brief  Describes an on-disk file mapped into RAM, which is different from Python's
 *          native `mmap` module, as it exposes the address of the mapping in memory.
 */
typedef struct {
    PyObject_HEAD
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        HANDLE file_handle;
    HANDLE mapping_handle;
#else
        int file_descriptor;
#endif
    sz_string_start_t start;
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
 *      - Str(File("some-path.txt"), from=0, to=sys.maxint)
 */
typedef struct {
    PyObject_HEAD //
        PyObject *parent;
    sz_string_start_t start;
    sz_size_t length;
} Str;

/**
 *  @brief  Variable length Python object similar to `Tuple[Union[Str, str]]`,
 *          for faster sorting, shuffling, joins, and lookups.
 */
typedef struct {
    PyObject_HEAD

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
         */
        struct consecutive_slices_32bit_t {
            size_t count;
            size_t separator_length;
            PyObject *parent;
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
         */
        struct consecutive_slices_64bit_t {
            size_t count;
            size_t separator_length;
            PyObject *parent;
            char const *start;
            uint64_t *end_offsets;
        } consecutive_64bit;

        /**
         *  Once you sort, shuffle, or reorganize slices making up a larger string, this structure
         *  cn be used for space-efficient lookups.
         */
        struct reordered_slices_t {
            size_t count;
            PyObject *parent;
            sz_string_view_t *parts;
        } reordered;

    } data;

} Strs;

#pragma endregion

#pragma region Helpers

inline static sz_string_start_t parts_get_start(sz_sequence_t *seq, sz_size_t i) {
    return ((sz_string_view_t const *)seq->handle)[i].start;
}

inline static sz_size_t parts_get_length(sz_sequence_t *seq, sz_size_t i) {
    return ((sz_string_view_t const *)seq->handle)[i].length;
}

void reverse_offsets(sz_size_t *array, size_t length) {
    size_t i, j;
    // Swap array[i] and array[j]
    for (i = 0, j = length - 1; i < j; i++, j--) {
        sz_size_t temp = array[i];
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

void apply_order(sz_string_view_t *array, sz_u64_t *order, size_t length) {
    for (sz_u64_t i = 0; i < length; ++i) {
        if (i == order[i]) continue;
        sz_string_view_t temp = array[i];
        sz_u64_t k = i, j;
        while (i != (j = order[k])) {
            array[k] = array[j];
            order[k] = k;
            k = j;
        }
        array[k] = temp;
        order[k] = k;
    }
}

void slice(size_t length, ssize_t start, ssize_t end, size_t *normalized_offset, size_t *normalized_length) {

    // clang-format off
    // Normalize negative indices
    if (start < 0) start += length;
    if (end < 0) end += length;

    // Clamp indices to a valid range
    if (start < 0) start = 0;
    if (end < 0) end = 0;
    if (start > (ssize_t)length) start = length;
    if (end > (ssize_t)length) end = length;

    // Ensure start <= end
    if (start > end) start = end;
    // clang-format on

    *normalized_offset = start;
    *normalized_length = end - start;
}

sz_bool_t export_string_like(PyObject *object, sz_string_start_t **start, sz_size_t *length) {
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

void str_at_offset_consecutive_32bit(
    Strs *strs, Py_ssize_t i, Py_ssize_t count, PyObject **parent, char const **start, size_t *length) {
    uint32_t start_offset = (i == 0) ? 0 : strs->data.consecutive_32bit.end_offsets[i - 1];
    uint32_t end_offset = strs->data.consecutive_32bit.end_offsets[i];
    *start = strs->data.consecutive_32bit.start + start_offset;
    *length = end_offset - start_offset - strs->data.consecutive_32bit.separator_length * (i + 1 != count);
    *parent = strs->data.consecutive_32bit.parent;
}

void str_at_offset_consecutive_64bit(
    Strs *strs, Py_ssize_t i, Py_ssize_t count, PyObject **parent, char const **start, size_t *length) {
    uint64_t start_offset = (i == 0) ? 0 : strs->data.consecutive_64bit.end_offsets[i - 1];
    uint64_t end_offset = strs->data.consecutive_64bit.end_offsets[i];
    *start = strs->data.consecutive_64bit.start + start_offset;
    *length = end_offset - start_offset - strs->data.consecutive_64bit.separator_length * (i + 1 != count);
    *parent = strs->data.consecutive_64bit.parent;
}

void str_at_offset_reordered(
    Strs *strs, Py_ssize_t i, Py_ssize_t count, PyObject **parent, char const **start, size_t *length) {
    *start = strs->data.reordered.parts[i].start;
    *length = strs->data.reordered.parts[i].length;
    *parent = strs->data.reordered.parent;
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
    PyObject *parent = NULL;
    switch (strs->type) {
    case STRS_CONSECUTIVE_32:
        count = strs->data.consecutive_32bit.count;
        old_buffer = strs->data.consecutive_32bit.end_offsets;
        parent = strs->data.consecutive_32bit.parent;
        getter = str_at_offset_consecutive_32bit;
        break;
    case STRS_CONSECUTIVE_64:
        count = strs->data.consecutive_64bit.count;
        old_buffer = strs->data.consecutive_64bit.end_offsets;
        parent = strs->data.consecutive_64bit.parent;
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
        PyObject *parent;
        char const *start;
        size_t length;
        getter(strs, (Py_ssize_t)i, count, &parent, &start, &length);
        new_parts[i].start = start;
        new_parts[i].length = length;
    }

    // Release previous used memory.
    if (old_buffer) free(old_buffer);

    // Update the Strs object
    strs->type = STRS_REORDERED;
    strs->data.reordered.count = count;
    strs->data.reordered.parts = new_parts;
    strs->data.reordered.parent = parent;
    return 1;
}

sz_bool_t prepare_strings_for_extension(Strs *strs, size_t new_parents, size_t new_parts) { return 1; }

#pragma endregion

#pragma region MemoryMappingFile

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
    if (self == NULL) return NULL;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    self->file_handle = NULL;
    self->mapping_handle = NULL;
#else
    self->file_descriptor = 0;
#endif
    self->start = NULL;
    self->length = 0;
}

static int File_init(File *self, PyObject *positional_args, PyObject *named_args) {
    const char *path;
    if (!PyArg_ParseTuple(positional_args, "s", &path)) return -1;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    self->file_handle = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
    if (self->file_handle == INVALID_HANDLE_VALUE) {
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }

    self->mapping_handle = CreateFileMapping(self->file_handle, 0, PAGE_READONLY, 0, 0, 0);
    if (self->mapping_handle == 0) {
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }

    char *file = (char *)MapViewOfFile(self->mapping_handle, FILE_MAP_READ, 0, 0, 0);
    if (file == 0) {
        CloseHandle(self->mapping_handle);
        self->mapping_handle = NULL;
        CloseHandle(self->file_handle);
        self->file_handle = NULL;
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
        return -1;
    }
    self->start = file;
    self->length = GetFileSize(self->file_handle, 0);
#else
    struct stat sb;
    self->file_descriptor = open(path, O_RDONLY);
    if (fstat(self->file_descriptor, &sb) != 0) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_SetString(PyExc_RuntimeError, "Can't retrieve file size!");
        return -1;
    }
    size_t file_size = sb.st_size;
    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, self->file_descriptor, 0);
    if (map == MAP_FAILED) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
        PyErr_SetString(PyExc_RuntimeError, "Couldn't map the file!");
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
    slice(self->length, from, to, &normalized_offset, &normalized_length);
    self->start = ((char *)self->start) + normalized_offset;
    self->length = normalized_length;
    return 0;
}

static PyObject *Str_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Str *self;
    self = (Str *)type->tp_alloc(type, 0);
    if (!self) return NULL;

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

static Py_hash_t Str_hash(Str *self) { return (Py_hash_t)sz_hash_crc32(self->start, self->length); }

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

static int Str_in(Str *self, PyObject *arg) {

    sz_string_view_t needle_struct;
    if (!export_string_like(arg, &needle_struct.start, &needle_struct.length)) {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument type");
        return -1;
    }

    return sz_find_substring(self->start, self->length, needle_struct.start, needle_struct.length) != NULL;
}

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
    if (PySlice_Check(key)) {
        // Sanity checks
        Py_ssize_t count = Strs_len(self);
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
        if (PySlice_AdjustIndices(count, &start, &stop, step) < 0) return NULL;
        if (step != 1) {
            PyErr_SetString(PyExc_IndexError, "Efficient step is not supported");
            return NULL;
        }

        // Create a new `Strs` object
        Strs *self_slice = (Strs *)StrsType.tp_alloc(&StrsType, 0);
        if (self_slice == NULL && PyErr_NoMemory()) return NULL;

        // Depending on the layout, the procedure will be different.
        self_slice->type = self->type;
        switch (self->type) {
        case STRS_CONSECUTIVE_32: {
            struct consecutive_slices_32bit_t *from = &self->data.consecutive_32bit;
            struct consecutive_slices_32bit_t *to = &self_slice->data.consecutive_32bit;
            to->count = stop - start;
            to->separator_length = from->separator_length;
            to->parent = from->parent;

            size_t first_length;
            str_at_offset_consecutive_32bit(self, start, count, &to->parent, &to->start, &first_length);
            uint32_t first_offset = to->start - from->start;
            to->end_offsets = malloc(sizeof(uint32_t) * to->count);
            if (to->end_offsets == NULL && PyErr_NoMemory()) {
                Py_XDECREF(self_slice);
                return NULL;
            }
            for (size_t i = 0; i != to->count; ++i) to->end_offsets[i] = from->end_offsets[i] - first_offset;
            Py_INCREF(to->parent);
            break;
        }
        case STRS_CONSECUTIVE_64: {
            struct consecutive_slices_64bit_t *from = &self->data.consecutive_64bit;
            struct consecutive_slices_64bit_t *to = &self_slice->data.consecutive_64bit;
            to->count = stop - start;
            to->separator_length = from->separator_length;
            to->parent = from->parent;

            size_t first_length;
            str_at_offset_consecutive_64bit(self, start, count, &to->parent, &to->start, &first_length);
            uint64_t first_offset = to->start - from->start;
            to->end_offsets = malloc(sizeof(uint64_t) * to->count);
            if (to->end_offsets == NULL && PyErr_NoMemory()) {
                Py_XDECREF(self_slice);
                return NULL;
            }
            for (size_t i = 0; i != to->count; ++i) to->end_offsets[i] = from->end_offsets[i] - first_offset;
            Py_INCREF(to->parent);
            break;
        }
        case STRS_REORDERED: {
            struct reordered_slices_t *from = &self->data.reordered;
            struct reordered_slices_t *to = &self_slice->data.reordered;
            to->count = stop - start;
            to->parent = from->parent;

            to->parts = malloc(sizeof(sz_string_view_t) * to->count);
            if (to->parts == NULL && PyErr_NoMemory()) {
                Py_XDECREF(self_slice);
                return NULL;
            }
            memcpy(to->parts, from->parts + start, sizeof(sz_string_view_t) * to->count);
            Py_INCREF(to->parent);
            break;
        }
        default:
            // Unsupported type
            PyErr_SetString(PyExc_TypeError, "Unsupported type for conversion");
            return NULL;
        }

        return (PyObject *)self_slice;
    }
    else if (PyLong_Check(key)) { return Strs_getitem(self, PyLong_AsSsize_t(key)); }
    else {
        PyErr_SetString(PyExc_TypeError, "Strs indices must be integers or slices");
        return NULL;
    }
}

// Will be called by the `PySequence_Contains`
static int Strs_contains(Str *self, PyObject *arg) { return 0; }

static PyObject *Str_richcompare(PyObject *self, PyObject *other, int op) {

    sz_string_start_t a_start = NULL, b_start = NULL;
    sz_size_t a_length = 0, b_length = 0;
    if (!export_string_like(self, &a_start, &a_length) || !export_string_like(other, &b_start, &b_length))
        Py_RETURN_NOTIMPLEMENTED;

    // Perform byte-wise comparison up to the minimum length
    sz_size_t min_length = a_length < b_length ? a_length : b_length;
    int order = memcmp(a_start, b_start, min_length);

    // If the strings are equal up to `min_length`, then the shorter string is smaller
    if (order == 0) order = (a_length > b_length) - (a_length < b_length);

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

/**
 *  @return 1 on success, 0 on failure.
 */
static int Str_find_( //
    PyObject *self,
    PyObject *args,
    PyObject *kwargs,
    Py_ssize_t *offset_out,
    sz_string_view_t *haystack_out,
    sz_string_view_t *needle_out) {

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
    slice(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    // Perform contains operation
    sz_string_start_t match = sz_find_substring(haystack.start, haystack.length, needle.start, needle.length);
    if (match == NULL) { *offset_out = -1; }
    else { *offset_out = (Py_ssize_t)(match - haystack.start); }

    *haystack_out = haystack;
    *needle_out = needle;
    return 1;
}

static PyObject *Str_find(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_(self, args, kwargs, &signed_offset, &text, &separator)) return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_index(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_(self, args, kwargs, &signed_offset, &text, &separator)) return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_contains(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_(self, args, kwargs, &signed_offset, &text, &separator)) return NULL;
    if (signed_offset == -1) { Py_RETURN_FALSE; }
    else { Py_RETURN_TRUE; }
}

static PyObject *Str_partition(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t separator_index;
    sz_string_view_t text;
    sz_string_view_t separator;
    PyObject *result_tuple;

    // Use Str_find_ to get the index of the separator
    if (!Str_find_(self, args, kwargs, &separator_index, &text, &separator)) return NULL;

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
    slice(haystack.length, start, end, &normalized_offset, &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    size_t count = 0;
    if (needle.length == 0 || haystack.length == 0 || haystack.length < needle.length) { count = 0; }
    else if (needle.length == 1) { count = sz_count_char(haystack.start, haystack.length, needle.start); }
    else if (allowoverlap) {
        while (haystack.length) {
            sz_string_start_t ptr = sz_find_substring(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? ptr - haystack.start : haystack.length;
            count += found;
            haystack.start += offset + found;
            haystack.length -= offset + found;
        }
    }
    else {
        while (haystack.length) {
            sz_string_start_t ptr = sz_find_substring(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? ptr - haystack.start : haystack.length;
            count += found;
            haystack.start += offset + needle.length;
            haystack.length -= offset + needle.length * found;
        }
    }

    return PyLong_FromSize_t(count);
}

static PyObject *Str_levenstein(PyObject *self, PyObject *args, PyObject *kwargs) {
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

    int bound = 255; // Default value for bound
    if (bound_obj && ((bound = PyLong_AsLong(bound_obj)) > 255 || bound < 0)) {
        PyErr_Format(PyExc_ValueError, "Bound must be an integer between 0 and 255");
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        PyErr_Format(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Allocate memory for the Levenstein matrix
    size_t memory_needed = sz_levenstein_memory_needed(str1.length, str2.length);
    if (temporary_memory.length < memory_needed) {
        temporary_memory.start = realloc(temporary_memory.start, memory_needed);
        temporary_memory.length = memory_needed;
    }
    if (!temporary_memory.start) {
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the Levenshtein matrix");
        return NULL;
    }

    levenstein_distance_t small_bound = (levenstein_distance_t)bound;
    levenstein_distance_t distance =
        sz_levenstein(str1.start, str1.length, str2.start, str2.length, small_bound, temporary_memory.start);

    return PyLong_FromLong(distance);
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

static Strs *Str_split_(
    PyObject *parent, sz_string_view_t text, sz_string_view_t separator, int keepseparator, Py_ssize_t maxsplit) {

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
        result->data.consecutive_64bit.parent = parent;
        result->data.consecutive_64bit.separator_length = !keepseparator * separator.length;
    }
    else {
        bytes_per_offset = 4;
        result->type = STRS_CONSECUTIVE_32;
        result->data.consecutive_32bit.start = text.start;
        result->data.consecutive_32bit.parent = parent;
        result->data.consecutive_32bit.separator_length = !keepseparator * separator.length;
    }

    // Iterate through string, keeping track of the
    sz_size_t last_start = 0;
    while (last_start <= text.length && offsets_count < maxsplit) {
        sz_string_start_t match =
            sz_find_substring(text.start + last_start, text.length - last_start, separator.start, separator.length);
        sz_size_t offset_in_remaining = match ? match - text.start - last_start : text.length - last_start;

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
        size_t will_continue = match != NULL;
        size_t next_offset = last_start + offset_in_remaining + separator.length * will_continue;
        if (text.length >= UINT32_MAX) { ((uint64_t *)offsets_endings)[offsets_count++] = (uint64_t)next_offset; }
        else { ((uint32_t *)offsets_endings)[offsets_count++] = (uint32_t)next_offset; }

        // Next time we want to start
        last_start = last_start + offset_in_remaining + separator.length;
    }

    // Populate the Strs object with the offsets
    if (text.length >= UINT32_MAX) {
        result->data.consecutive_64bit.end_offsets = offsets_endings;
        result->data.consecutive_64bit.count = offsets_count;
    }
    else {
        result->data.consecutive_32bit.end_offsets = offsets_endings;
        result->data.consecutive_32bit.count = offsets_count;
    }

    Py_INCREF(parent);
    return result;
}

static PyObject *Str_split(PyObject *self, PyObject *args, PyObject *kwargs) {

    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 3) {
        PyErr_SetString(PyExc_TypeError, "sz.split() received unsupported number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *separator_obj = nargs > !is_member + 0 ? PyTuple_GET_ITEM(args, !is_member + 0) : NULL;
    PyObject *maxsplit_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *keepseparator_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "separator") == 0) { separator_obj = value; }
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
    if (!export_string_like(text_obj, &text.start, &text.length)) {
        PyErr_SetString(PyExc_TypeError, "The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `separator`
    if (separator_obj) {
        Py_ssize_t len;
        if (!export_string_like(separator_obj, &separator.start, &len)) {
            PyErr_SetString(PyExc_TypeError, "The separator argument must be string-like");
            return NULL;
        }
        separator.length = (size_t)len;
    }
    else {
        separator.start = " ";
        separator.length = 1;
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

    return Str_split_(text_obj, text, separator, keepseparator, maxsplit);
}

static PyObject *Str_splitlines(PyObject *self, PyObject *args, PyObject *kwargs) {

    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 2) {
        PyErr_SetString(PyExc_TypeError, "splitlines() requires at least 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
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
    if (!export_string_like(text_obj, &text.start, &text.length)) {
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

    // TODO: Support arbitrary newline characters:
    // https://docs.python.org/3/library/stdtypes.html#str.splitlines
    // \n, \r, \r\n, \v or \x0b, \f or \x0c, \x1c, \x1d, \x1e, \x85, \u2028, \u2029
    // https://github.com/ashvardanian/StringZilla/issues/29
    sz_string_view_t separator;
    separator.start = "\n";
    separator.length = 1;
    return Str_split_(text_obj, text, separator, keeplinebreaks, maxsplit);
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

#define sz_method_flags_m METH_VARARGS | METH_KEYWORDS

static PyMethodDef Str_methods[] = {
    //
    {"find", Str_find, sz_method_flags_m, "Find the first occurrence of a substring."},
    {"index", Str_index, sz_method_flags_m, "Find the first occurrence of a substring or raise error if missing."},
    {"contains", Str_contains, sz_method_flags_m, "Check if a string contains a substring."},
    {"partition", Str_partition, sz_method_flags_m, "Splits string into 3-tuple: before, match, after."},
    {"count", Str_count, sz_method_flags_m, "Count the occurrences of a substring."},
    {"split", Str_split, sz_method_flags_m, "Split a string by a separator."},
    {"splitlines", Str_splitlines, sz_method_flags_m, "Split a string by line breaks."},
    {"startswith", Str_startswith, sz_method_flags_m, "Check if a string starts with a given prefix."},
    {"endswith", Str_endswith, sz_method_flags_m, "Check if a string ends with a given suffix."},
    {"levenstein", Str_levenstein, sz_method_flags_m, "Calculate the Levenshtein distance between two strings."},
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
    .tp_str = Str_str,
    .tp_methods = Str_methods,
    .tp_as_sequence = &Str_as_sequence,
    .tp_as_mapping = &Str_as_mapping,
    .tp_as_buffer = &Str_as_buffer,
    .tp_as_number = &Str_as_number,
};

#pragma endregion

#pragma regions Strs

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
    for (size_t i = count - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        // Swap parts[i] and parts[j]
        sz_string_view_t temp = parts[i];
        parts[i] = parts[j];
        parts[j] = temp;
    }

    Py_RETURN_NONE;
}

static sz_bool_t Strs_sort_(Strs *self,
                            sz_string_view_t **parts_output,
                            sz_size_t **order_output,
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
    size_t memory_needed = sizeof(sz_size_t) * count;
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
    sz_sort_config_t sort_config;
    memset(&sequence, 0, sizeof(sequence));
    memset(&sort_config, 0, sizeof(sort_config));
    sequence.order = (sz_size_t *)temporary_memory.start;
    sequence.count = count;
    sequence.handle = parts;
    sequence.get_start = parts_get_start;
    sequence.get_length = parts_get_length;
    for (sz_size_t i = 0; i != sequence.count; ++i) sequence.order[i] = i;
    sz_sort(&sequence, &sort_config);

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
    sz_size_t *order = NULL;
    sz_size_t count = 0;
    if (!Strs_sort_(self, &parts, &order, &count)) return NULL;

    // Apply the sorting algorithm here, considering the `reverse` value
    if (reverse) reverse_offsets(order, count);

    // Here, instead of applying the order, we want to return the copy of the
    // order as a NumPy array of 64-bit unsigned integers.
    npy_intp numpy_size = count;
    PyObject *array = PyArray_SimpleNew(1, &numpy_size, NPY_UINT64);
    if (!array) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create a NumPy array");
        return NULL;
    }

    // Copy the data from the order array to the newly created NumPy array
    sz_size_t *numpy_data_ptr = (sz_size_t *)PyArray_DATA((PyArrayObject *)array);
    memcpy(numpy_data_ptr, order, count * sizeof(sz_size_t));
    return array;
}

static PySequenceMethods Strs_as_sequence = {
    .sq_length = Strs_len,        //
    .sq_item = Strs_getitem,      //
    .sq_contains = Strs_contains, //
};

static PyMappingMethods Strs_as_mapping = {
    .mp_length = Strs_len,          //
    .mp_subscript = Strs_subscript, // Is used to implement slices in Python
};

static PyMethodDef Strs_methods[] = {
    {"shuffle", Strs_shuffle, sz_method_flags_m, "Shuffle the elements of the Strs object."},  //
    {"sort", Strs_sort, sz_method_flags_m, "Sort the elements of the Strs object."},           //
    {"order", Strs_order, sz_method_flags_m, "Provides the indexes to achieve sorted order."}, //
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
};

#pragma endregion

static void stringzilla_cleanup(PyObject *m) {
    if (temporary_memory.start) free(temporary_memory.start);
    temporary_memory.start = NULL;
    temporary_memory.length = 0;
}

static PyMethodDef stringzilla_methods[] = {
    {"find", Str_find, sz_method_flags_m, "Find the first occurrence of a substring."},
    {"index", Str_index, sz_method_flags_m, "Find the first occurrence of a substring or raise error if missing."},
    {"contains", Str_contains, sz_method_flags_m, "Check if a string contains a substring."},
    {"partition", Str_partition, sz_method_flags_m, "Splits string into 3-tuple: before, match, after."},
    {"count", Str_count, sz_method_flags_m, "Count the occurrences of a substring."},
    {"split", Str_split, sz_method_flags_m, "Split a string by a separator."},
    {"splitlines", Str_splitlines, sz_method_flags_m, "Split a string by line breaks."},
    {"startswith", Str_startswith, sz_method_flags_m, "Check if a string starts with a given prefix."},
    {"endswith", Str_endswith, sz_method_flags_m, "Check if a string ends with a given suffix."},
    {"levenstein", Str_levenstein, sz_method_flags_m, "Calculate the Levenshtein distance between two strings."},
    {NULL, NULL, 0, NULL}};

static PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "Crunch 100+ GB Strings in Python with ease",
    -1,
    stringzilla_methods,
    NULL,
    NULL,
    NULL,
    stringzilla_cleanup,
};

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;

    import_array();

    if (PyType_Ready(&StrType) < 0) return NULL;
    if (PyType_Ready(&FileType) < 0) return NULL;
    if (PyType_Ready(&StrsType) < 0) return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL) return NULL;

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

    // Initialize temporary_memory, if needed
    temporary_memory.start = malloc(4096);
    temporary_memory.length = 4096 * (temporary_memory.start != NULL);
    return m;
}
