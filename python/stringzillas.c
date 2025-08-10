/**
 *  @file       stringzillas.c
 *  @brief      Very light-weight CPython wrapper for StringZillas advanced algorithms,
 *              supporting bulk operations for edit distances, sequence alignment, and fingerprinting.
 *  @author     Ash Vardanian
 *  @date       December 15, 2024
 *  @copyright  Copyright (c) 2024
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids `PyArg_ParseTupleAndKeywords` calls.
 *  - Uses manual argument parsing for performance on hot paths.
 *  - Returns & accepts NumPy arrays when available, avoiding memory-scattered Python lists.
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

#include <errno.h>  // `errno`
#include <stdio.h>  // `fopen`
#include <stdlib.h> // `rand`, `srand`
#include <time.h>   // `time`

#include <Python.h>            // CPython API
#include <numpy/arrayobject.h> // NumPy C API

#include <stringzillas/stringzillas.h>

#pragma region Forward Declarations

// Try to import NumPy, and fail if it's not available
static int numpy_available = 0;
static PyObject *numpy_module = NULL;

static PyTypeObject DeviceScopeType;
static PyTypeObject LevenshteinDistancesType;
static PyTypeObject LevenshteinDistancesUTF8Type;
static PyTypeObject NeedlemanWunschType;
static PyTypeObject SmithWatermanType;
static PyTypeObject FingerprintsType;
static PyTypeObject FingerprintsUTF8Type;

// Function pointers for stringzilla functions imported from capsules
static sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *) = NULL;
static sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *) = NULL;
static sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *) = NULL;
static sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *) = NULL;

// Default device scope that can be safely reused across calls
// The underlying implementation is stateless and thread-safe
static sz_device_scope_t default_device_scope = NULL;

typedef struct PyAPI {
    sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
    sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
} PyAPI;

// Method flags
#define SZ_METHOD_FLAGS METH_VARARGS | METH_KEYWORDS

#pragma endregion

#pragma region DeviceScope

/**
 *  @brief  Device scope for controlling execution context (CPU cores or GPU device).
 */
typedef struct {
    PyObject ob_base;
    sz_device_scope_t handle;
} DeviceScope;

static void DeviceScope_dealloc(DeviceScope *self) {
    if (self->handle) {
        sz_device_scope_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *DeviceScope_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    DeviceScope *self = (DeviceScope *)type->tp_alloc(type, 0);
    if (self != NULL) { self->handle = NULL; }
    return (PyObject *)self;
}

static int DeviceScope_init(DeviceScope *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_Size(args);
    PyObject *cpu_cores_obj = NULL;
    PyObject *gpu_device_obj = NULL;

    // Manual argument parsing - check positional args first
    if (nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "DeviceScope() takes at most 2 arguments");
        return -1;
    }

    if (nargs >= 1) cpu_cores_obj = PyTuple_GET_ITEM(args, 0);
    if (nargs >= 2) gpu_device_obj = PyTuple_GET_ITEM(args, 1);

    // Parse keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "cpu_cores") == 0) {
                if (cpu_cores_obj) {
                    PyErr_SetString(PyExc_TypeError, "cpu_cores specified twice");
                    return -1;
                }
                cpu_cores_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "gpu_device") == 0) {
                if (gpu_device_obj) {
                    PyErr_SetString(PyExc_TypeError, "gpu_device specified twice");
                    return -1;
                }
                gpu_device_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_status_t status;

    if (cpu_cores_obj != NULL && gpu_device_obj != NULL) {
        PyErr_SetString(PyExc_ValueError, "Cannot specify both cpu_cores and gpu_device");
        return -1;
    }
    else if (cpu_cores_obj != NULL) {
        if (!PyLong_Check(cpu_cores_obj)) {
            PyErr_SetString(PyExc_TypeError, "cpu_cores must be an integer");
            return -1;
        }
        sz_size_t cpu_cores = PyLong_AsSize_t(cpu_cores_obj);
        if (cpu_cores == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = sz_device_scope_init_cpu_cores(cpu_cores, &self->handle);
    }
    else if (gpu_device_obj != NULL) {
        if (!PyLong_Check(gpu_device_obj)) {
            PyErr_SetString(PyExc_TypeError, "gpu_device must be an integer");
            return -1;
        }
        sz_size_t gpu_device = PyLong_AsSize_t(gpu_device_obj);
        if (gpu_device == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = sz_device_scope_init_gpu_device(gpu_device, &self->handle);
    }
    else { status = sz_device_scope_init_default(&self->handle); }

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize device scope");
        return -1;
    }

    return 0;
}

static char const doc_DeviceScope[] = //
    "DeviceScope(cpu_cores=None, gpu_device=None)\n"
    "\n"
    "Context for controlling execution on CPU cores or GPU devices.\n"
    "\n"
    "Args:\n"
    "  cpu_cores (int, optional): Number of CPU cores to use (0 for all, 1 for single-threaded).\n"
    "  gpu_device (int, optional): GPU device ID to target.\n"
    "\n"
    "Note: Cannot specify both cpu_cores and gpu_device.";

static PyTypeObject DeviceScopeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.DeviceScope",
    .tp_doc = doc_DeviceScope,
    .tp_basicsize = sizeof(DeviceScope),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = DeviceScope_new,
    .tp_init = (initproc)DeviceScope_init,
    .tp_dealloc = (destructor)DeviceScope_dealloc,
};

#pragma endregion

#pragma region LevenshteinDistances

/**
 *  @brief  Levenshtein distance computation engine for binary strings.
 */
typedef struct {
    PyObject ob_base;
    sz_levenshtein_distances_t handle;
} LevenshteinDistances;

static void LevenshteinDistances_dealloc(LevenshteinDistances *self) {
    if (self->handle) {
        sz_levenshtein_distances_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *LevenshteinDistances_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    LevenshteinDistances *self = (LevenshteinDistances *)type->tp_alloc(type, 0);
    if (self != NULL) { self->handle = NULL; }
    return (PyObject *)self;
}

static int LevenshteinDistances_init(LevenshteinDistances *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_Size(args);
    sz_error_cost_t match = 0, mismatch = 1, open = 1, extend = 1;

    // Manual argument parsing - fast path for positional args
    if (nargs >= 1) {
        PyObject *obj = PyTuple_GET_ITEM(args, 0);
        if (PyLong_Check(obj)) {
            long val = PyLong_AsLong(obj);
            if (val < -128 || val > 127) {
                PyErr_SetString(PyExc_ValueError, "match cost must fit in 8-bit signed integer");
                return -1;
            }
            match = (sz_error_cost_t)val;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "match cost must be an integer");
            return -1;
        }
    }

    if (nargs >= 2) {
        PyObject *obj = PyTuple_GET_ITEM(args, 1);
        if (PyLong_Check(obj)) {
            long val = PyLong_AsLong(obj);
            if (val < -128 || val > 127) {
                PyErr_SetString(PyExc_ValueError, "mismatch cost must fit in 8-bit signed integer");
                return -1;
            }
            mismatch = (sz_error_cost_t)val;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "mismatch cost must be an integer");
            return -1;
        }
    }

    if (nargs >= 3) {
        PyObject *obj = PyTuple_GET_ITEM(args, 2);
        if (PyLong_Check(obj)) {
            long val = PyLong_AsLong(obj);
            if (val < -128 || val > 127) {
                PyErr_SetString(PyExc_ValueError, "open cost must fit in 8-bit signed integer");
                return -1;
            }
            open = (sz_error_cost_t)val;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "open cost must be an integer");
            return -1;
        }
    }

    if (nargs >= 4) {
        PyObject *obj = PyTuple_GET_ITEM(args, 3);
        if (PyLong_Check(obj)) {
            long val = PyLong_AsLong(obj);
            if (val < -128 || val > 127) {
                PyErr_SetString(PyExc_ValueError, "extend cost must fit in 8-bit signed integer");
                return -1;
            }
            extend = (sz_error_cost_t)val;
        }
        else {
            PyErr_SetString(PyExc_TypeError, "extend cost must be an integer");
            return -1;
        }
    }

    if (nargs > 4) {
        PyErr_SetString(PyExc_TypeError, "LevenshteinDistances() takes at most 4 arguments");
        return -1;
    }

    // Parse keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "match") == 0) {
                if (nargs >= 1) {
                    PyErr_SetString(PyExc_TypeError, "match specified twice");
                    return -1;
                }
                if (!PyLong_Check(value)) {
                    PyErr_SetString(PyExc_TypeError, "match must be an integer");
                    return -1;
                }
                long val = PyLong_AsLong(value);
                if (val < -128 || val > 127) {
                    PyErr_SetString(PyExc_ValueError, "match cost must fit in 8-bit signed integer");
                    return -1;
                }
                match = (sz_error_cost_t)val;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "mismatch") == 0) {
                if (nargs >= 2) {
                    PyErr_SetString(PyExc_TypeError, "mismatch specified twice");
                    return -1;
                }
                if (!PyLong_Check(value)) {
                    PyErr_SetString(PyExc_TypeError, "mismatch must be an integer");
                    return -1;
                }
                long val = PyLong_AsLong(value);
                if (val < -128 || val > 127) {
                    PyErr_SetString(PyExc_ValueError, "mismatch cost must fit in 8-bit signed integer");
                    return -1;
                }
                mismatch = (sz_error_cost_t)val;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "open") == 0) {
                if (nargs >= 3) {
                    PyErr_SetString(PyExc_TypeError, "open specified twice");
                    return -1;
                }
                if (!PyLong_Check(value)) {
                    PyErr_SetString(PyExc_TypeError, "open must be an integer");
                    return -1;
                }
                long val = PyLong_AsLong(value);
                if (val < -128 || val > 127) {
                    PyErr_SetString(PyExc_ValueError, "open cost must fit in 8-bit signed integer");
                    return -1;
                }
                open = (sz_error_cost_t)val;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "extend") == 0) {
                if (nargs >= 4) {
                    PyErr_SetString(PyExc_TypeError, "extend specified twice");
                    return -1;
                }
                if (!PyLong_Check(value)) {
                    PyErr_SetString(PyExc_TypeError, "extend must be an integer");
                    return -1;
                }
                long val = PyLong_AsLong(value);
                if (val < -128 || val > 127) {
                    PyErr_SetString(PyExc_ValueError, "extend cost must fit in 8-bit signed integer");
                    return -1;
                }
                extend = (sz_error_cost_t)val;
            }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_status_t status =
        sz_levenshtein_distances_init(match, mismatch, open, extend, NULL, szs_capabilities(), &self->handle);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Levenshtein distances engine");
        return -1;
    }

    return 0;
}

static PyObject *LevenshteinDistances_call(LevenshteinDistances *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_Size(args);
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    // Manual argument parsing for hot path
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "LevenshteinDistances() requires at least 2 arguments");
        return NULL;
    }

    if (nargs > 4) {
        PyErr_SetString(PyExc_TypeError, "LevenshteinDistances() takes at most 4 arguments");
        return NULL;
    }

    a_obj = PyTuple_GET_ITEM(args, 0);
    b_obj = PyTuple_GET_ITEM(args, 1);
    if (nargs >= 3) device_obj = PyTuple_GET_ITEM(args, 2);
    if (nargs >= 4) out_obj = PyTuple_GET_ITEM(args, 3);

    // Parse keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "device") == 0) {
                if (device_obj) {
                    PyErr_SetString(PyExc_TypeError, "device specified twice");
                    return NULL;
                }
                device_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
                if (out_obj) {
                    PyErr_SetString(PyExc_TypeError, "out specified twice");
                    return NULL;
                }
                out_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    sz_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t kernel_input_size = 0;
    void *kernel_a_texts_punned = NULL;
    void *kernel_b_texts_punned = NULL;
    sz_size_t *kernel_results = NULL;
    sz_size_t kernel_results_stride = sizeof(sz_size_t);
    sz_status_t (*kernel_punned)(sz_levenshtein_distances_t, sz_device_scope_t, void *, void *, sz_size_t *,
                                 sz_size_t) = NULL;

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t a_u32tape, b_u32tape;
    sz_bool_t a_is_u32tape = sz_py_export_strings_as_u32tape( //
        a_obj, &a_u32tape.data, &a_u32tape.offsets, &a_u32tape.count);
    sz_bool_t b_is_u32tape = sz_py_export_strings_as_u32tape( //
        b_obj, &b_u32tape.data, &b_u32tape.offsets, &b_u32tape.count);
    if (a_is_u32tape && b_is_u32tape) {
        if (a_u32tape.count != b_u32tape.count) {
            PyErr_SetString(PyExc_ValueError, "Input sequences must have the same length");
            return NULL;
        }

        kernel_input_size = a_u32tape.count;
        kernel_punned = sz_levenshtein_distances_u32tape;
        kernel_a_texts_punned = &a_u32tape;
        kernel_b_texts_punned = &b_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t a_u64tape, b_u64tape;
    sz_bool_t a_is_u64tape = !a_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                  a_obj, &a_u64tape.data, &a_u64tape.offsets, &a_u64tape.count);
    sz_bool_t b_is_u64tape = !b_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                  b_obj, &b_u64tape.data, &b_u64tape.offsets, &b_u64tape.count);
    if (a_is_u64tape && b_is_u64tape) {
        if (a_u64tape.count != b_u64tape.count) {
            PyErr_SetString(PyExc_ValueError, "Input sequences must have the same length");
            return NULL;
        }
        kernel_input_size = a_u64tape.count;
        kernel_punned = sz_levenshtein_distances_u64tape;
        kernel_a_texts_punned = &a_u64tape;
        kernel_b_texts_punned = &b_u64tape;
    }

    // Handle sequence inputs
    sz_sequence_t a_seq, b_seq;
    sz_bool_t a_is_sequence = !a_is_u32tape && !a_is_u64tape && sz_py_export_strings_as_sequence(a_obj, &a_seq);
    sz_bool_t b_is_sequence = !b_is_u32tape && !b_is_u64tape && sz_py_export_strings_as_sequence(b_obj, &b_seq);
    if (a_is_sequence && b_is_sequence) {
        if (a_seq.count != b_seq.count) {
            PyErr_SetString(PyExc_ValueError, "Input sequences must have the same length");
            return NULL;
        }
        kernel_input_size = a_seq.count;
        kernel_punned = sz_levenshtein_distances_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Unsupported input types for Levenshtein distances. "
                     "u32tape: a=%d b=%d, u64tape: a=%d b=%d, seq: a=%d b=%d",
                     a_is_u32tape, b_is_u32tape, a_is_u64tape, b_is_u64tape, a_is_sequence, b_is_sequence);
        return NULL;
    }

    // Make sure the `out` argument is valid NumPy array and extract `kernel_results` and `kernel_results_stride`
    // or create a new results array.
    PyObject *results_array = NULL;
    if (!out_obj || out_obj == Py_None) {
        // Create a new NumPy array for results
        npy_intp numpy_size = kernel_input_size;
        results_array = PyArray_SimpleNew(1, &numpy_size, NPY_UINT64);
        if (!results_array) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array for results");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_stride = sizeof(sz_size_t);
    }
    else {
        // Validate existing NumPy array
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out argument must be a NumPy array");
            goto cleanup;
        }
        PyArrayObject *array = (PyArrayObject *)out_obj;
        if (PyArray_NDIM(array) != 1) {
            PyErr_SetString(PyExc_ValueError, "out array must be 1-dimensional");
            goto cleanup;
        }
        if (PyArray_SIZE(array) < (npy_intp)kernel_input_size) {
            PyErr_SetString(PyExc_ValueError, "out array is too small for results");
            goto cleanup;
        }
        if (PyArray_TYPE(array) != NPY_UINT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have uint64 dtype");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA(array);
        kernel_results_stride = PyArray_STRIDE(array, 0);
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    sz_status_t status = kernel_punned(               //
        self->handle, device_handle,                  //
        kernel_a_texts_punned, kernel_b_texts_punned, //
        kernel_results, kernel_results_stride);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Levenshtein distance computation failed");
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_LevenshteinDistances[] = //
    "LevenshteinDistances(match=0, mismatch=1, open=1, extend=1)\n"
    "\n"
    "Compute Levenshtein edit distances between pairs of binary strings.\n"
    "\n"
    "Args:\n"
    "  match (int): Cost for matching characters (default: 0).\n"
    "  mismatch (int): Cost for mismatched characters (default: 1).\n"
    "  open (int): Cost for opening a gap (default: 1).\n"
    "  extend (int): Cost for extending a gap (default: 1).\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.";

static PyTypeObject LevenshteinDistancesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.LevenshteinDistances",
    .tp_doc = doc_LevenshteinDistances,
    .tp_basicsize = sizeof(LevenshteinDistances),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = LevenshteinDistances_new,
    .tp_init = (initproc)LevenshteinDistances_init,
    .tp_dealloc = (destructor)LevenshteinDistances_dealloc,
    .tp_call = (ternaryfunc)LevenshteinDistances_call,
};

#pragma endregion

#pragma region LevenshteinDistancesUTF8

typedef struct {
    PyObject ob_base;
    sz_levenshtein_distances_utf8_t handle;
} LevenshteinDistancesUTF8;

static PyObject *LevenshteinDistancesUTF8_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LevenshteinDistancesUTF8 *self = (LevenshteinDistancesUTF8 *)type->tp_alloc(type, 0);
    if (self != NULL) { self->handle = NULL; }
    return (PyObject *)self;
}

static void LevenshteinDistancesUTF8_dealloc(LevenshteinDistancesUTF8 *self) {
    if (self->handle) { sz_levenshtein_distances_utf8_free(self->handle); }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int LevenshteinDistancesUTF8_init(LevenshteinDistancesUTF8 *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"match", "mismatch", "gap_open", "gap_extend", NULL};
    sz_error_cost_t match = 0, mismatch = 1, open = 1, extend = 1;

    if (args) {
        Py_ssize_t n_args = PyTuple_Size(args);
        for (Py_ssize_t i = 0; i < n_args; i++) {
            PyObject *arg = PyTuple_GetItem(args, i);
            int val = PyLong_AsLong(arg);
            if (PyErr_Occurred()) return -1;
            if (i == 0) { match = (sz_error_cost_t)val; }
            else if (i == 1) { mismatch = (sz_error_cost_t)val; }
            else if (i == 2) { open = (sz_error_cost_t)val; }
            else if (i == 3) { extend = (sz_error_cost_t)val; }
        }
    }

    if (kwds) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwds, &pos, &key, &value)) {
            int val = PyLong_AsLong(value);
            if (PyErr_Occurred()) return -1;

            if (PyUnicode_CompareWithASCIIString(key, "match") == 0) { match = (sz_error_cost_t)val; }
            else if (PyUnicode_CompareWithASCIIString(key, "mismatch") == 0) { mismatch = (sz_error_cost_t)val; }
            else if (PyUnicode_CompareWithASCIIString(key, "gap_open") == 0) { open = (sz_error_cost_t)val; }
            else if (PyUnicode_CompareWithASCIIString(key, "gap_extend") == 0) { extend = (sz_error_cost_t)val; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_status_t status =
        sz_levenshtein_distances_utf8_init(match, mismatch, open, extend, NULL, szs_capabilities(), &self->handle);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize UTF-8 Levenshtein distances engine");
        return -1;
    }
    return 0;
}

static PyObject *LevenshteinDistancesUTF8_call(LevenshteinDistancesUTF8 *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t nargs = PyTuple_Size(args);
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    // Manual argument parsing for hot path
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "LevenshteinDistancesUTF8() requires at least 2 arguments");
        return NULL;
    }

    if (nargs > 4) {
        PyErr_SetString(PyExc_TypeError, "LevenshteinDistancesUTF8() takes at most 4 arguments");
        return NULL;
    }

    a_obj = PyTuple_GET_ITEM(args, 0);
    b_obj = PyTuple_GET_ITEM(args, 1);
    if (nargs >= 3) device_obj = PyTuple_GET_ITEM(args, 2);
    if (nargs >= 4) out_obj = PyTuple_GET_ITEM(args, 3);

    // Parse keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "device") == 0) {
                if (device_obj) {
                    PyErr_SetString(PyExc_TypeError, "device specified twice");
                    return NULL;
                }
                device_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
                if (out_obj) {
                    PyErr_SetString(PyExc_TypeError, "out specified twice");
                    return NULL;
                }
                out_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    sz_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t kernel_input_size = 0;
    void *kernel_a_texts_punned = NULL;
    void *kernel_b_texts_punned = NULL;
    sz_size_t *kernel_results = NULL;
    sz_size_t kernel_results_stride = sizeof(sz_size_t);
    sz_status_t (*kernel_punned)(sz_levenshtein_distances_t, sz_device_scope_t, void *, void *, sz_size_t *,
                                 sz_size_t) = NULL;

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t a_u32tape, b_u32tape;
    sz_bool_t a_is_u32tape = sz_py_export_strings_as_u32tape( //
        a_obj, &a_u32tape.data, &a_u32tape.offsets, &a_u32tape.count);
    sz_bool_t b_is_u32tape = sz_py_export_strings_as_u32tape( //
        b_obj, &b_u32tape.data, &b_u32tape.offsets, &b_u32tape.count);
    if (a_is_u32tape && b_is_u32tape) {
        if (a_u32tape.count != b_u32tape.count) {
            PyErr_SetString(PyExc_ValueError, "Input sequences must have the same length");
            return NULL;
        }

        kernel_input_size = a_u32tape.count;
        kernel_punned = sz_levenshtein_distances_utf8_u32tape;
        kernel_a_texts_punned = &a_u32tape;
        kernel_b_texts_punned = &b_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t a_u64tape, b_u64tape;
    sz_bool_t a_is_u64tape = !a_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                  a_obj, &a_u64tape.data, &a_u64tape.offsets, &a_u64tape.count);
    sz_bool_t b_is_u64tape = !b_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                  b_obj, &b_u64tape.data, &b_u64tape.offsets, &b_u64tape.count);
    if (a_is_u64tape && b_is_u64tape) {
        if (a_u64tape.count != b_u64tape.count) {
            PyErr_SetString(PyExc_ValueError, "Input sequences must have the same length");
            return NULL;
        }
        kernel_input_size = a_u64tape.count;
        kernel_punned = sz_levenshtein_distances_utf8_u64tape;
        kernel_a_texts_punned = &a_u64tape;
        kernel_b_texts_punned = &b_u64tape;
    }

    // Handle sequence inputs
    sz_sequence_t a_seq, b_seq;
    sz_bool_t a_is_sequence = !a_is_u32tape && !a_is_u64tape && sz_py_export_strings_as_sequence(a_obj, &a_seq);
    sz_bool_t b_is_sequence = !b_is_u32tape && !b_is_u64tape && sz_py_export_strings_as_sequence(b_obj, &b_seq);
    if (a_is_sequence && b_is_sequence) {
        if (a_seq.count != b_seq.count) {
            PyErr_SetString(PyExc_ValueError, "Input sequences must have the same length");
            return NULL;
        }
        kernel_input_size = a_seq.count;
        kernel_punned = sz_levenshtein_distances_utf8_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Unsupported input types for Levenshtein distances. "
                     "u32tape: a=%d b=%d, u64tape: a=%d b=%d, seq: a=%d b=%d",
                     a_is_u32tape, b_is_u32tape, a_is_u64tape, b_is_u64tape, a_is_sequence, b_is_sequence);
        return NULL;
    }

    // Make sure the `out` argument is valid NumPy array and extract `kernel_results` and `kernel_results_stride`
    // or create a new results array.
    PyObject *results_array = NULL;
    if (!out_obj || out_obj == Py_None) {
        // Create a new NumPy array for results
        npy_intp numpy_size = kernel_input_size;
        results_array = PyArray_SimpleNew(1, &numpy_size, NPY_UINT64);
        if (!results_array) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array for results");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_stride = sizeof(sz_size_t);
    }
    else {
        // Validate existing NumPy array
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out argument must be a NumPy array");
            goto cleanup;
        }
        PyArrayObject *array = (PyArrayObject *)out_obj;
        if (PyArray_NDIM(array) != 1) {
            PyErr_SetString(PyExc_ValueError, "out array must be 1-dimensional");
            goto cleanup;
        }
        if (PyArray_SIZE(array) < (npy_intp)kernel_input_size) {
            PyErr_SetString(PyExc_ValueError, "out array is too small for results");
            goto cleanup;
        }
        if (PyArray_TYPE(array) != NPY_UINT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have uint64 dtype");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA(array);
        kernel_results_stride = PyArray_STRIDE(array, 0);
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    sz_status_t status = kernel_punned(               //
        self->handle, device_handle,                  //
        kernel_a_texts_punned, kernel_b_texts_punned, //
        kernel_results, kernel_results_stride);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Levenshtein distance computation failed");
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_LevenshteinDistancesUTF8[] = //
    "LevenshteinDistancesUTF8(match=0, mismatch=1, gap_open=1, gap_extend=1)\n"
    "\n"
    "Vectorized UTF-8 Levenshtein distance calculator.\n"
    "Computes edit distances between pairs of UTF-8 encoded strings.\n"
    "\n"
    "Args:\n"
    "  match (int): Cost of matching characters (default 0).\n"
    "  mismatch (int): Cost of mismatched characters (default 1).\n"
    "  gap_open (int): Cost of opening a gap (default 1).\n"
    "  gap_extend (int): Cost of extending a gap (default 1).\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of UTF-8 strings.\n"
    "  b (sequence): Second sequence of UTF-8 strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.";

static PyTypeObject LevenshteinDistancesUTF8Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.LevenshteinDistancesUTF8",
    .tp_doc = doc_LevenshteinDistancesUTF8,
    .tp_basicsize = sizeof(LevenshteinDistancesUTF8),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = LevenshteinDistancesUTF8_new,
    .tp_init = (initproc)LevenshteinDistancesUTF8_init,
    .tp_dealloc = (destructor)LevenshteinDistancesUTF8_dealloc,
    .tp_call = (ternaryfunc)LevenshteinDistancesUTF8_call,
};

#pragma endregion

static void stringzillas_cleanup(PyObject *m) {
    sz_unused_(m);
    if (default_device_scope) {
        sz_device_scope_free(default_device_scope);
        default_device_scope = NULL;
    }
}

static PyMethodDef stringzillas_methods[] = {{NULL, NULL, 0, NULL}};

static PyModuleDef stringzillas_module = {
    PyModuleDef_HEAD_INIT,
    "stringzillas",
    "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU",
    -1,
    stringzillas_methods,
    NULL,
    NULL,
    NULL,
    stringzillas_cleanup,
};

PyMODINIT_FUNC PyInit_stringzillas(void) {
    PyObject *m;

    // Try to import NumPy
#if defined(NPY_VERSION)
    import_array();
    numpy_available = 1;
#else
    // Try to import numpy module dynamically
    numpy_module = PyImport_ImportModule("numpy");
    if (numpy_module) { numpy_available = 1; }
    else {
        PyErr_Clear(); // Clear the import error
        PyErr_SetString(PyExc_ImportError, "NumPy is required but not available");
        return NULL;
    }
#endif

    // Try to import StringZilla and get the C API functions
    PyObject *stringzilla_module = PyImport_ImportModule("stringzilla");
    if (!stringzilla_module) {
        PyErr_SetString(PyExc_ImportError, "StringZilla module is required but not available");
        return NULL;
    }

    // Import the C API struct from the single capsule
    PyObject *capsule = PyObject_GetAttrString(stringzilla_module, "_sz_py_api");
    if (!capsule || !PyCapsule_CheckExact(capsule)) {
        Py_XDECREF(capsule);
        Py_DECREF(stringzilla_module);
        PyErr_SetString(PyExc_ImportError, "Failed to import StringZilla C API capsule");
        return NULL;
    }

    // Get the PyAPI struct from the capsule
    PyAPI *api = (PyAPI *)PyCapsule_GetPointer(capsule, "_sz_py_api");
    if (!api) {
        Py_DECREF(capsule);
        Py_DECREF(stringzilla_module);
        PyErr_SetString(PyExc_ImportError, "Failed to get StringZilla C API pointer from capsule");
        return NULL;
    }

    // Extract the function pointers from the struct
    sz_py_export_string_like = api->sz_py_export_string_like;
    sz_py_export_strings_as_sequence = api->sz_py_export_strings_as_sequence;
    sz_py_export_strings_as_u32tape = api->sz_py_export_strings_as_u32tape;
    sz_py_export_strings_as_u64tape = api->sz_py_export_strings_as_u64tape;

    Py_DECREF(capsule);
    Py_DECREF(stringzilla_module);

    // Check that all functions were loaded
    if (!sz_py_export_string_like || !sz_py_export_strings_as_sequence || !sz_py_export_strings_as_u32tape ||
        !sz_py_export_strings_as_u64tape) {
        PyErr_SetString(PyExc_ImportError, "Failed to import required StringZilla C API functions");
        return NULL;
    }

    // Initialize the default device scope for reuse
    sz_status_t status = sz_device_scope_init_default(&default_device_scope);
    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize default device scope");
        return NULL;
    }

    if (PyType_Ready(&DeviceScopeType) < 0) return NULL;
    if (PyType_Ready(&LevenshteinDistancesType) < 0) return NULL;
    // if (PyType_Ready(&LevenshteinDistancesUTF8Type) < 0) return NULL;

    m = PyModule_Create(&stringzillas_module);
    if (m == NULL) return NULL;

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", szs_version_major(), szs_version_minor(), szs_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Define SIMD capabilities
    {
        sz_capability_t caps = szs_capabilities();
        sz_cptr_t caps_str = sz_capabilities_to_string_implementation_(caps);
        PyModule_AddStringConstant(m, "__capabilities__", caps_str);
    }

    Py_INCREF(&DeviceScopeType);
    if (PyModule_AddObject(m, "DeviceScope", (PyObject *)&DeviceScopeType) < 0) {
        Py_XDECREF(&DeviceScopeType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&LevenshteinDistancesType);
    if (PyModule_AddObject(m, "LevenshteinDistances", (PyObject *)&LevenshteinDistancesType) < 0) {
        Py_XDECREF(&LevenshteinDistancesType);
        Py_XDECREF(&DeviceScopeType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&LevenshteinDistancesUTF8Type);
    if (PyModule_AddObject(m, "LevenshteinDistancesUTF8", (PyObject *)&LevenshteinDistancesUTF8Type) < 0) {
        Py_XDECREF(&LevenshteinDistancesUTF8Type);
        Py_XDECREF(&LevenshteinDistancesType);
        Py_XDECREF(&DeviceScopeType);
        Py_XDECREF(m);
        return NULL;
    }

    return m;
}