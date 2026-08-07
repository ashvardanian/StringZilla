/**
 *  @brief The `File` type - a memory-mapped file exposing its address range.
 *  @file python/stringzilla/file.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief  Releases the memory mapping and OS handles held by a `File`, leaving it in the blank state
 *          produced by `File_new`. Shared by `File_dealloc` and the `File_init` re-initialization guard.
 */
static void File_release_(File *self) {
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
}

static void File_dealloc(File *self) {
    File_release_(self);
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

    // Re-initialization: release any mapping/handles from a prior __init__ before opening a new file
    File_release_(self);

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
    if (file_size == 0) {
        // mmap rejects zero-length mappings (EINVAL); expose an empty file as an empty view
        self->memory.start = NULL;
        self->memory.length = 0;
        return 0;
    }
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

static PyMethodDef File_methods[] = {//
                                     {NULL, NULL, 0, NULL}};

static char const doc_File[] =                                                               //
    "File(path, mode='r')\n"                                                                 //
    "\n"                                                                                     //
    "Memory-mapped file class that exposes the memory range for low-level access.\n"         //
    "Provides efficient read-only access to file contents without loading into memory.\n"    //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "  path (str): Path to the file to memory-map.\n"                                        //
    "  mode (str): File access mode (default: 'r' for read-only).\n"                         //
    "\n"                                                                                     //
    "Example:\n"                                                                             //
    "  >>> import os, tempfile, pathlib\n"                                                   //
    "  >>> path = os.path.join(tempfile.mkdtemp(), 'log.txt')\n"                             //
    "  >>> _ = pathlib.Path(path).write_text('error: disk full\\nok\\nerror: timeout\\n')\n" //
    "  >>> # Memory-map and scan in place - no copy into RAM, SIMD-accelerated:\n"           //
    "  >>> sz.Str(sz.File(path)).count('error')\n"                                           //
    "  2";

PyTypeObject FileType = {
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
