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

#pragma region Helpers

void slice(size_t length, ssize_t start, ssize_t end, size_t *normalized_offset, size_t *normalized_length) {

    // clang-format off
    // Normalize negative indices
    if (start < 0) start += length;
    if (end < 0) end += length;

    // Clamp indices to a valid range
    if (start < 0) start = 0;
    if (end < 0) end = 0;
    if (start > length) start = length;
    if (end > length) end = length;

    // Ensure start <= end
    if (start > end) start = end;
    // clang-format on

    *normalized_offset = start;
    *normalized_length = end - start;
}

#pragma endregion

#pragma region MemoryMappingFile

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
    void *ptr;
    size_t len;
} MemoryMappedFile;

static void MemoryMappedFile_dealloc(MemoryMappedFile *self) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    if (self->ptr) {
        UnmapViewOfFile(self->ptr);
        self->ptr = NULL;
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
    if (self->ptr) {
        munmap(self->ptr, self->len);
        self->ptr = NULL;
        self->len = 0;
    }
    if (self->file_descriptor != 0) {
        close(self->file_descriptor);
        self->file_descriptor = 0;
    }
#endif
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *MemoryMappedFile_new(PyTypeObject *type, PyObject *positional_args, PyObject *named_args) {
    MemoryMappedFile *self;
    self = (MemoryMappedFile *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ptr = NULL;
        self->len = 0;
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        self->file_handle = NULL;
        self->mapping_handle = NULL;
#else
        self->file_descriptor = 0;
#endif
    }
    return (PyObject *)self;
}

static int MemoryMappedFile_init(MemoryMappedFile *self, PyObject *positional_args, PyObject *named_args) {
    const char *path;
    if (!PyArg_ParseTuple(positional_args, "s", &path)) {
        return -1;
    }

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
    self->ptr = file;
    self->len = GetFileSize(self->file_handle, 0);
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
    self->ptr = map;
    self->len = file_size;
#endif

    return 0;
}

static PyMethodDef MemoryMappedFile_methods[] = {
    // Your method definitions here
    {NULL} /* Sentinel */
};

static PyTypeObject MemoryMappedFileType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.MemoryMappedFile",
    .tp_doc = "MemoryMappedFile objects",
    .tp_basicsize = sizeof(MemoryMappedFile),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = MemoryMappedFile_new,
    .tp_init = (initproc)MemoryMappedFile_init,
    .tp_dealloc = (destructor)MemoryMappedFile_dealloc,
    .tp_methods = MemoryMappedFile_methods,
};

#pragma endregion

#pragma region Str

/**
 *  @brief  Type-punned StringZilla-string, that points to a slice of an existing Python `str`
 *          or a `MemoryMappedFile`.
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
    void *start;
    size_t length;
} Str;

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

    self->parent = parent;
    if (PyUnicode_Check(parent)) {
        // Handle Python str
        self->start = PyUnicode_DATA(parent);
        self->length = PyUnicode_GET_DATA_SIZE(parent);
        Py_INCREF(parent); // Increment the reference count of the parent
    }
    else if (PyObject_TypeCheck(parent, &MemoryMappedFileType)) {
        // Handle MemoryMappedFile
        MemoryMappedFile *file = (MemoryMappedFile *)parent;
        self->start = file->ptr;
        self->length = file->len;
        Py_INCREF(parent); // Increment the reference count of the parent
    }
    else if (parent == NULL) {
        // Handle empty string
        self->start = NULL;
        self->length = 0;
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

static PyTypeObject StrType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Str",
    .tp_doc = "Stringzilla Str objects",
    .tp_basicsize = sizeof(Str),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Str_new,
    .tp_dealloc = (destructor)Str_dealloc,
};

#pragma endregion

static PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "Crunch 100+ GB Strings in Python with ease",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL,
};

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;

    if (PyType_Ready(&StrType) < 0)
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

    Py_INCREF(&MemoryMappedFileType);
    if (PyModule_AddObject(m, "MemoryMappedFile", (PyObject *)&MemoryMappedFileType) < 0) {
        Py_XDECREF(&MemoryMappedFileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    return m;
}