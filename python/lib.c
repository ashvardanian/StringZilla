/**
 *  @brief  Very light-weight CPython wrapper for StringZilla, with support for memory-mapping,
 *          native Python strings, Apache Arrow collections, and more.
 */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/stat.h> // `stat`
#include <sys/mman.h> // `mmap`
#include <fcntl.h>    // `O_RDNLY`
#endif

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <unistd.h> // `ssize_t`
#endif

#include <Python.h>

#include <stringzilla.h>

#pragma region Forward Declarations

static PyTypeObject FileType;
static PyTypeObject StrType;

struct {
    void *ptr;
    size_t len;
} temporary_memory = {NULL, 0};

/**
 *  @brief  Describes an on-disk file mapped into RAM, which is different from Python's
 *          native `mmap` module, as it exposes the address of the mapping in memory.
 */
typedef struct {
    PyObject_HEAD;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    HANDLE file_handle;
    HANDLE mapping_handle;
#else
    int file_descriptor;
#endif
    void *start;
    size_t length;
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
    PyObject_HEAD;
    PyObject *parent;
    char const *start;
    size_t length;
} Str;

#pragma endregion

#pragma region Helpers

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

int export_string_like(PyObject *object, char const **start, size_t *length) {
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

#pragma endregion

#pragma region Global Functions

static Py_ssize_t str_find_vectorcall_(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    // Initialize defaults
    Py_ssize_t start = 0;
    Py_ssize_t end = PY_SSIZE_T_MAX;

    // Parse positional arguments: haystack and needle
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *haystack_obj = args[0];
    PyObject *needle_obj = args[1];
    struct strzl_haystack_t haystack;
    struct strzl_needle_t needle;
    needle.anomaly_offset = 0;
    if (!export_string_like(haystack_obj, &haystack.ptr, &haystack.len) ||
        !export_string_like(needle_obj, &needle.ptr, &needle.len)) {
        PyErr_SetString(PyExc_TypeError, "Haystack and needle must be string-like");
        return NULL;
    }

    // Parse additional positional arguments
    if (nargs > 2)
        start = PyLong_AsSsize_t(args[2]);
    if (nargs > 3)
        end = PyLong_AsSsize_t(args[3]);

    // Parse keyword arguments
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];
            if (PyUnicode_CompareWithASCIIString(key, "start") == 0)
                start = PyLong_AsSsize_t(value);
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0)
                end = PyLong_AsSsize_t(value);
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    // Limit the haystack range
    size_t normalized_offset, normalized_length;
    slice(haystack.len, start, end, &normalized_offset, &normalized_length);
    haystack.ptr += normalized_offset;
    haystack.len = normalized_length;

    // Perform contains operation
    size_t offset = strzl_neon_find_substr(haystack, needle);
    if (offset == haystack.len)
        return -1;
    return (Py_ssize_t)offset;
}

static PyObject *str_find_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t signed_offset = str_find_vectorcall_(NULL, args, nargsf, kwnames);
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *str_contains_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t signed_offset = str_find_vectorcall_(NULL, args, nargsf, kwnames);
    if (signed_offset == -1) {
        Py_RETURN_FALSE;
    }
    else {
        Py_RETURN_TRUE;
    }
}

static PyObject *str_count_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    // Initialize defaults
    Py_ssize_t start = 0;
    Py_ssize_t end = PY_SSIZE_T_MAX;
    int allow_overlap = 0;

    // Parse positional arguments: haystack and needle
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *haystack_obj = args[0];
    PyObject *needle_obj = args[1];

    struct strzl_haystack_t haystack;
    struct strzl_needle_t needle;
    needle.anomaly_offset = 0;
    if (!export_string_like(haystack_obj, &haystack.ptr, &haystack.len) ||
        !export_string_like(needle_obj, &needle.ptr, &needle.len)) {
        PyErr_SetString(PyExc_TypeError, "Haystack and needle must be string-like");
        return NULL;
    }

    // Parse additional positional arguments
    if (nargs > 2)
        start = PyLong_AsSsize_t(args[2]);
    if (nargs > 3)
        end = PyLong_AsSsize_t(args[3]);

    // Parse keyword arguments
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];
            if (PyUnicode_CompareWithASCIIString(key, "start") == 0)
                start = PyLong_AsSsize_t(value);
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0)
                end = PyLong_AsSsize_t(value);
            else if (PyUnicode_CompareWithASCIIString(key, "allowoverlap") == 0)
                allow_overlap = PyObject_IsTrue(value);
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    // Limit the haystack range
    size_t normalized_offset, normalized_length;
    slice(haystack.len, start, end, &normalized_offset, &normalized_length);
    haystack.ptr += normalized_offset;
    haystack.len = normalized_length;

    // Perform counting operation
    size_t count = 0;
    if (needle.len == 1) {
        count = strzl_naive_count_char(haystack, *needle.ptr);
    }
    else {
        // Your existing logic for count_substr can be embedded here
        if (allow_overlap) {
            while (haystack.len) {
                size_t offset = strzl_neon_find_substr(haystack, needle);
                int found = offset != haystack.len;
                count += found;
                haystack.ptr += offset + found;
                haystack.len -= offset + found;
            }
        }
        else {
            while (haystack.len) {
                size_t offset = strzl_neon_find_substr(haystack, needle);
                int found = offset != haystack.len;
                count += found;
                haystack.ptr += offset + needle.len;
                haystack.len -= offset + needle.len * found;
            }
        }
    }

    return PyLong_FromSize_t(count);
}

static PyObject *str_levenstein_vectorcall(PyObject *_, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    Py_ssize_t nargs = PyVectorcall_NARGS(nargsf);

    // Validate the number of arguments
    if (nargs < 2 || nargs > 3) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = args[0];
    PyObject *str2_obj = args[1];

    struct strzl_haystack_t str1, str2;
    if (!export_string_like(str1_obj, &str1.ptr, &str1.len) || !export_string_like(str2_obj, &str2.ptr, &str2.len)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be string-like");
        return NULL;
    }

    // Initialize bound argument
    int bound = 255;

    // Check if `bound` is given as a positional argument
    if (nargs == 3) {
        bound = PyLong_AsLong(args[2]);
        if (bound > 255 || bound < 0) {
            PyErr_SetString(PyExc_ValueError, "Bound must be an integer between 0 and 255");
            return NULL;
        }
    }

    // Parse keyword arguments
    if (kwnames != NULL) {
        for (Py_ssize_t i = 0; i < PyTuple_Size(kwnames); ++i) {
            PyObject *key = PyTuple_GetItem(kwnames, i);
            PyObject *value = args[nargs + i];
            if (PyUnicode_CompareWithASCIIString(key, "bound") == 0) {
                if (nargs == 3) {
                    PyErr_SetString(PyExc_TypeError, "Received bound both as positional and keyword argument");
                    return NULL;
                }
                bound = PyLong_AsLong(value);
                if (bound > 255 || bound < 0) {
                    PyErr_SetString(PyExc_ValueError, "Bound must be an integer between 0 and 255");
                    return NULL;
                }
            }
        }
    }

    // Initialize or reallocate the Levenshtein distance matrix
    size_t memory_needed = strzl_levenstein_memory_needed(str1.len, str2.len);
    if (temporary_memory.len < memory_needed) {
        temporary_memory.ptr = realloc(temporary_memory.ptr, memory_needed);
        temporary_memory.len = memory_needed;
    }
    if (temporary_memory.ptr == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the Levenshtein matrix");
        return NULL;
    }

    levenstein_distance_t distance = strzl_levenstein( //
        str1.ptr,
        str1.len,
        str2.ptr,
        str2.len,
        (levenstein_distance_t)bound,
        temporary_memory.ptr);
    return PyLong_FromLong(distance);
}

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
    if (self == NULL)
        return NULL;

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
    if (!PyArg_ParseTuple(positional_args, "s", &path))
        return -1;

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

    // PyBufferProcs *tp_as_buffer;

    // reprfunc tp_repr;
    // PyNumberMethods *tp_as_number;
    // PySequenceMethods *tp_as_sequence;
    // PyMappingMethods *tp_as_mapping;
    // ternaryfunc tp_call;
    // reprfunc tp_str;
    // getattrofunc tp_getattro;
    // setattrofunc tp_setattro;
};

#pragma endregion

#pragma region Str

static int Str_init(Str *self, PyObject *positional_args, PyObject *named_args) {
    PyObject *parent = NULL;
    Py_ssize_t from = 0;
    Py_ssize_t to = PY_SSIZE_T_MAX;

    // The `named_args` would be `NULL`
    if (named_args) {
        static char *names[] = {"parent", "from", "to", NULL};
        if (!PyArg_ParseTupleAndKeywords(positional_args, named_args, "|Onn", names, &parent, &from, &to))
            return -1;
    }
    else if (!PyArg_ParseTuple(positional_args, "|Onn", &parent, &from, &to))
        return -1;

    // Handle empty string
    if (parent == NULL) {
        self->start = NULL;
        self->length = 0;
    }
    // Increment the reference count of the parent
    else if (export_string_like(parent, &self->start, &self->length)) {
        self->parent = parent;
        Py_INCREF(parent);
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
    if (!self)
        return NULL;

    self->parent = NULL;
    self->start = NULL;
    self->length = 0;
    return (PyObject *)self;
}

static void Str_dealloc(Str *self) {
    if (self->parent)
        Py_XDECREF(self->parent);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Str_str(Str *self) { return PyUnicode_FromStringAndSize(self->start, self->length); }

static Py_ssize_t Str_len(Str *self) { return self->length; }

static Py_hash_t Str_hash(Str *self) { return (Py_hash_t)strzl_hash_crc32_native(self->start, self->length); }

static PyObject *Str_getitem(Str *self, Py_ssize_t i) {

    // Negative indexing
    if (i < 0)
        i += self->length;

    if (i < 0 || (size_t)i >= self->length) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    // Assuming the underlying data is UTF-8 encoded
    return PyUnicode_FromStringAndSize(self->start + i, 1);
}

static PyObject *Str_subscript(Str *self, PyObject *key) {
    if (PySlice_Check(key)) {
        Py_ssize_t start, stop, step;
        if (PySlice_Unpack(key, &start, &stop, &step) < 0)
            return NULL;
        if (PySlice_AdjustIndices(self->length, &start, &stop, step) < 0)
            return NULL;

        if (step != 1) {
            PyErr_SetString(PyExc_IndexError, "Efficient step is not supported");
            return NULL;
        }

        // Create a new `Str` object
        Str *self_slice = (Str *)StrType.tp_alloc(&StrType, 0);
        if (self_slice == NULL && PyErr_NoMemory())
            return NULL;

        // Set its properties based on the slice
        self_slice->start = self->start + start;
        self_slice->length = stop - start;
        self_slice->parent = (PyObject *)self; // Set parent to keep it alive

        // Increment the reference count of the parent
        Py_INCREF(self);
        return (PyObject *)self_slice;
    }
    else if (PyLong_Check(key)) {
        return Str_getitem(self, PyLong_AsSsize_t(key));
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Str indices must be integers or slices");
        return NULL;
    }
}

// Will be called by the `PySequence_Contains`
static int Str_contains(Str *self, PyObject *arg) {

    struct strzl_needle_t needle_struct;
    needle_struct.anomaly_offset = 0;
    if (!export_string_like(arg, &needle_struct.ptr, &needle_struct.len)) {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument type");
        return -1;
    }

    struct strzl_haystack_t haystack;
    haystack.ptr = self->start;
    haystack.len = self->length;
    size_t position = strzl_neon_find_substr(haystack, needle_struct);
    return position != haystack.len;
}

static PyObject *Str_getslice(Str *self, PyObject *args) {
    PyObject *start_obj = NULL, *end_obj = NULL;
    ssize_t start = 0, end = self->length; // Default values

    if (!PyArg_ParseTuple(args, "|OO", &start_obj, &end_obj))
        return NULL;

    if (start_obj != NULL && start_obj != Py_None) {
        if (!PyLong_Check(start_obj)) {
            PyErr_SetString(PyExc_TypeError, "Start index must be an integer or None");
            return NULL;
        }
        start = PyLong_AsSsize_t(start_obj);
    }

    if (end_obj != NULL && end_obj != Py_None) {
        if (!PyLong_Check(end_obj)) {
            PyErr_SetString(PyExc_TypeError, "End index must be an integer or None");
            return NULL;
        }
        end = PyLong_AsSsize_t(end_obj);
    }

    size_t normalized_offset, normalized_length;
    slice(self->length, start, end, &normalized_offset, &normalized_length);

    if (normalized_length == 0)
        return PyUnicode_FromString("");

    // Create a new Str object
    Str *new_str = (Str *)PyObject_New(Str, &StrType);
    if (new_str == NULL)
        return NULL;

    // Set the parent to the original Str object and increment its reference count
    new_str->parent = (PyObject *)self;
    Py_INCREF(self);

    // Set the start and length to point to the slice
    new_str->start = self->start + normalized_offset;
    new_str->length = normalized_length;
    return (PyObject *)new_str;
}

static PyObject *Str_richcompare(PyObject *self, PyObject *other, int op) {

    char const *a_start, *b_start;
    size_t a_length, b_length;
    if (!export_string_like(self, &a_start, &a_length) || !export_string_like(other, &b_start, &b_length))
        Py_RETURN_NOTIMPLEMENTED;

    // Perform byte-wise comparison up to the minimum length
    size_t min_length = a_length < b_length ? a_length : b_length;
    int cmp_result = memcmp(a_start, b_start, min_length);

    // If the strings are equal up to `min_length`, then the shorter string is smaller
    if (cmp_result == 0)
        cmp_result = (a_length > b_length) - (a_length < b_length);

    switch (op) {
    case Py_LT: return PyBool_FromLong(cmp_result < 0);
    case Py_LE: return PyBool_FromLong(cmp_result <= 0);
    case Py_EQ: return PyBool_FromLong(cmp_result == 0);
    case Py_NE: return PyBool_FromLong(cmp_result != 0);
    case Py_GT: return PyBool_FromLong(cmp_result > 0);
    case Py_GE: return PyBool_FromLong(cmp_result >= 0);
    default: Py_RETURN_NOTIMPLEMENTED;
    }
}

static PySequenceMethods Str_as_sequence = {
    .sq_length = Str_len,        //
    .sq_item = Str_getitem,      //
    .sq_contains = Str_contains, //
};

static PyMappingMethods Str_as_mapping = {
    .mp_length = Str_len,          //
    .mp_subscript = Str_subscript, // Is used to implement slices in Python
};

static PyMethodDef Str_methods[] = { //
    // {"contains", (PyCFunction)..., METH_NOARGS, "Convert to Python `str`"},
    // {"find", (PyCFunction)..., METH_NOARGS, "Get length"},
    // {"__getitem__", (PyCFunction)..., METH_O, "Indexing"},
    {NULL, NULL, 0, NULL}};

static PyTypeObject StrType = {
    PyObject_HEAD_INIT(NULL).tp_name = "stringzilla.Str",
    .tp_doc = "Immutable string/slice class with SIMD and SWAR-accelerated operations",
    .tp_basicsize = sizeof(Str),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = Str_methods,
    .tp_new = Str_new,
    .tp_init = Str_init,
    .tp_dealloc = Str_dealloc,
    .tp_as_sequence = &Str_as_sequence,
    .tp_as_mapping = &Str_as_mapping,
    .tp_hash = Str_hash, // String hashing functions
    .tp_richcompare = Str_richcompare,
    .tp_str = Str_str,
    // .tp_as_buffer = (PyBufferProcs *)NULL, // Functions to access object as input/output buffer
};

#pragma endregion

static PyMethodDef stringzilla_methods[] = { //
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
    NULL,
};

PyObject *register_vectorcall(PyObject *module, char const *name, vectorcallfunc vectorcall) {

    PyCFunctionObject *vectorcall_object = (PyCFunctionObject *)PyObject_Malloc(sizeof(PyCFunctionObject));
    if (vectorcall_object == NULL)
        return NULL;

    PyObject_Init(vectorcall_object, &PyCFunction_Type);
    vectorcall_object->m_ml = NULL; // No regular `PyMethodDef`
    vectorcall_object->vectorcall = vectorcall;

    // Add the 'find' function to the module
    if (PyModule_AddObject(module, name, vectorcall_object) < 0) {
        Py_XDECREF(vectorcall_object);
        return NULL;
    }
    return vectorcall_object;
}

void cleanup_module(void) {
    free(temporary_memory.ptr);
    temporary_memory.ptr = NULL;
    temporary_memory.len = 0;
}

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;

    if (PyType_Ready(&StrType) < 0)
        return NULL;

    if (PyType_Ready(&FileType) < 0)
        return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL)
        return NULL;

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

    // Initialize temporary_memory, if needed
    // For example, allocate an initial chunk
    temporary_memory.ptr = malloc(4096);
    temporary_memory.len = 4096 * (temporary_memory.ptr != NULL);
    atexit(cleanup_module);

    // Register the vectorized functions
    PyObject *vectorized_find = register_vectorcall(m, "find", str_find_vectorcall);
    PyObject *vectorized_contains = register_vectorcall(m, "contains", str_contains_vectorcall);
    PyObject *vectorized_count = register_vectorcall(m, "count", str_count_vectorcall);
    PyObject *vectorized_levenstein = register_vectorcall(m, "levenstein", str_levenstein_vectorcall);

    PyObject *vectorized_split = register_vectorcall(m, "split", str_find_vectorcall);
    PyObject *vectorized_sort = register_vectorcall(m, "sort", str_find_vectorcall);
    PyObject *vectorized_shuffle = register_vectorcall(m, "shuffle", str_find_vectorcall);

    if (!vectorized_find || !vectorized_count ||          //
        !vectorized_contains || !vectorized_levenstein || //
        !vectorized_split || !vectorized_sort || !vectorized_shuffle) {
        goto cleanup;
    }

    return m;

cleanup:
    if (vectorized_find)
        Py_XDECREF(vectorized_find);
    if (vectorized_contains)
        Py_XDECREF(vectorized_contains);
    if (vectorized_count)
        Py_XDECREF(vectorized_count);
    if (vectorized_levenstein)
        Py_XDECREF(vectorized_levenstein);
    if (vectorized_split)
        Py_XDECREF(vectorized_split);
    if (vectorized_sort)
        Py_XDECREF(vectorized_sort);
    if (vectorized_shuffle)
        Py_XDECREF(vectorized_shuffle);

    Py_XDECREF(&FileType);
    Py_XDECREF(&StrType);
    Py_XDECREF(m);
    PyErr_NoMemory();
    return NULL;
}
