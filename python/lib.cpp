/**
 *  @brief
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

/**
 *  @brief  Type-punned StringZilla-string, that may either be an immutable in-memory string,
 *          similar to Python's native `str`, or a memory-mapped immutable file from disk,
 *          or a slice of one of those classes or the Python's native `str` and `bytes` classes.
 *
 *  When a slice is being used, the `parent` object's reference count is being incremented.
 *  When an in-memory string is used - we avoid the second memory allocation and allocate the `HEAD`,
 *  the length, and the content region in a single continuous chunk.
 */
typedef struct {
    PyObject_HEAD;

    typedef enum {
        in_memory_k,
        on_disk_k,
        slice_k,
    } variant;

    typedef struct {
        size_t length;
    } in_memory_t;

    typedef struct {
        void *start;
        size_t length;
        int file_descriptor;
    } on_disk_t;

    typedef struct {
        PyObject *parent;
        void *start;
        size_t length;
    } slice_t;
} strzl_t;

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

    if (PyType_Ready(&PySpanType) < 0)
        return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&PySpanType);
    PyModule_AddObject(m, "Span", (PyObject *)&PySpanType);

    return m;
}
