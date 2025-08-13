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

/**
 * @brief Creates a Python tuple from capabilities mask.
 * @param[in] caps Capabilities mask
 * @return New reference to Python tuple, or NULL on error
 */
static PyObject *capabilities_to_tuple(sz_capability_t caps) {
    char const *cap_strings[SZ_CAPABILITIES_COUNT];
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
    return caps_tuple;
}

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
static sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *) = NULL;

// Default device scope that can be safely reused across calls
// The underlying implementation is stateless and thread-safe
static sz_device_scope_t default_device_scope = NULL;
// Static variable to store hardware capabilities
static sz_capability_t default_hardware_capabilities = 0;
// Static unified memory allocator for GPU compatibility
static sz_memory_allocator_t unified_allocator;

typedef struct PyAPI {
    sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
    sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
    sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *);
} PyAPI;

// Method flags
#define SZ_METHOD_FLAGS METH_VARARGS | METH_KEYWORDS

/**
 *  @brief  Helper function to automatically swap a Strs object's allocator to unified memory.
 *          This ensures GPU compatibility for string operations.
 *  @param[in] strs_obj The Strs object to swap allocator for
 *  @return sz_true_k on success, sz_false_k on failure
 */
static sz_bool_t try_swap_to_unified_allocator(PyObject *strs_obj) {
    if (!strs_obj || !sz_py_replace_strings_allocator) return sz_false_k;

    // Try to swap to unified allocator - this will be a no-op if already using it
    sz_bool_t success = sz_py_replace_strings_allocator(strs_obj, &unified_allocator);

    if (!success) {
        // Set Python error to inform user of the failure
        PyErr_SetString(PyExc_RuntimeError, "Failed to allocate unified memory for GPU compatibility. "
                                            "Consider reducing input size or freeing memory.");
    }
    return success;
}

#pragma endregion

#pragma region Metadata

/**
 *  @brief Parse capabilities from a Python tuple of strings and intersect with hardware capabilities.
 *  @param[in] caps_tuple Python tuple containing capability strings (e.g., ('serial', 'haswell')).
 *  @param[out] result Output capability mask after intersection with hardware capabilities.
 *  @return 0 on success, -1 on error (with Python exception set).
 */
static int parse_and_intersect_capabilities(PyObject *caps_tuple, sz_capability_t *result) {
    if (!PyTuple_Check(caps_tuple)) {
        PyErr_SetString(PyExc_TypeError, "capabilities must be a tuple of strings");
        return -1;
    }

    sz_capability_t requested_caps = 0;
    Py_ssize_t n = PyTuple_Size(caps_tuple);

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(caps_tuple, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "capabilities must be a tuple of strings");
            return -1;
        }

        char const *cap_str = PyUnicode_AsUTF8(item);
        if (!cap_str) return -1;

        // Map string to capability flag
        if (strcmp(cap_str, "serial") == 0) { requested_caps |= sz_cap_serial_k; }
        else if (strcmp(cap_str, "parallel") == 0) { requested_caps |= sz_cap_parallel_k; }
        else if (strcmp(cap_str, "haswell") == 0) { requested_caps |= sz_cap_haswell_k; }
        else if (strcmp(cap_str, "skylake") == 0) { requested_caps |= sz_cap_skylake_k; }
        else if (strcmp(cap_str, "ice") == 0) { requested_caps |= sz_cap_ice_k; }
        else if (strcmp(cap_str, "neon") == 0) { requested_caps |= sz_cap_neon_k; }
        else if (strcmp(cap_str, "neon_aes") == 0) { requested_caps |= sz_cap_neon_aes_k; }
        else if (strcmp(cap_str, "sve") == 0) { requested_caps |= sz_cap_sve_k; }
        else if (strcmp(cap_str, "sve2") == 0) { requested_caps |= sz_cap_sve2_k; }
        else if (strcmp(cap_str, "sve2_aes") == 0) { requested_caps |= sz_cap_sve2_aes_k; }
        else if (strcmp(cap_str, "cuda") == 0) { requested_caps |= sz_cap_cuda_k; }
        else if (strcmp(cap_str, "kepler") == 0) { requested_caps |= sz_cap_kepler_k; }
        else if (strcmp(cap_str, "hopper") == 0) { requested_caps |= sz_cap_hopper_k; }
        else if (strcmp(cap_str, "any") == 0) { requested_caps |= sz_cap_any_k; }
        else {
            PyErr_Format(PyExc_ValueError, "Unknown capability: %s", cap_str);
            return -1;
        }
    }

    // Intersect with hardware capabilities
    *result = requested_caps & default_hardware_capabilities;

    // If no capabilities match, fall back to serial
    if (*result == 0) { *result = sz_cap_serial_k; }

    return 0;
}

#pragma endregion

#pragma region DeviceScope

/**
 *  @brief  Device scope for controlling execution context (CPU cores or GPU device).
 */
typedef struct {
    PyObject ob_base;
    sz_device_scope_t handle;
    char description[32];
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
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
    }
    return (PyObject *)self;
}

static int DeviceScope_init(DeviceScope *self, PyObject *args, PyObject *kwargs) {
    sz_size_t cpu_cores = 0;
    sz_size_t gpu_device = 0;
    int has_cpu_cores = 0;
    int has_gpu_device = 0;
    PyObject *cpu_cores_obj = NULL;
    PyObject *gpu_device_obj = NULL;

    static char *kwlist[] = {"cpu_cores", "gpu_device", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &cpu_cores_obj, &gpu_device_obj)) { return -1; }

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
        cpu_cores = PyLong_AsSize_t(cpu_cores_obj);
        if (cpu_cores == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = sz_device_scope_init_cpu_cores(cpu_cores, &self->handle);
        snprintf(self->description, sizeof(self->description), "CPUs:%zu", cpu_cores);
    }
    else if (gpu_device_obj != NULL) {
        if (!PyLong_Check(gpu_device_obj)) {
            PyErr_SetString(PyExc_TypeError, "gpu_device must be an integer");
            return -1;
        }
        gpu_device = PyLong_AsSize_t(gpu_device_obj);
        if (gpu_device == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = sz_device_scope_init_gpu_device(gpu_device, &self->handle);
        snprintf(self->description, sizeof(self->description), "GPU:%zu", gpu_device);
    }
    else {
        status = sz_device_scope_init_default(&self->handle);
        snprintf(self->description, sizeof(self->description), "default");
    }

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize device scope");
        return -1;
    }

    return 0;
}

static PyObject *DeviceScope_repr(DeviceScope *self) {
    return PyUnicode_FromFormat("DeviceScope(%s)", self->description);
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
    .tp_repr = (reprfunc)DeviceScope_repr,
};

#pragma endregion

#pragma region LevenshteinDistances

/**
 *  @brief  Levenshtein distance computation engine for binary strings.
 */
typedef struct {
    PyObject ob_base;
    sz_levenshtein_distances_t handle;
    char description[32];
    sz_capability_t capabilities;
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
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
    }
    return (PyObject *)self;
}

static int LevenshteinDistances_init(LevenshteinDistances *self, PyObject *args, PyObject *kwargs) {
    int match = 0, mismatch = 1, open = 1, extend = 1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    static char *kwlist[] = {"match", "mismatch", "open", "extend", "capabilities", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiO", kwlist, &match, &mismatch, &open, &extend,
                                     &capabilities_tuple)) {
        return -1;
    }

    // Validate range of values
    if (match < -128 || match > 127) {
        PyErr_SetString(PyExc_ValueError, "match cost must fit in 8-bit signed integer");
        return -1;
    }
    if (mismatch < -128 || mismatch > 127) {
        PyErr_SetString(PyExc_ValueError, "mismatch cost must fit in 8-bit signed integer");
        return -1;
    }
    if (open < -128 || open > 127) {
        PyErr_SetString(PyExc_ValueError, "open cost must fit in 8-bit signed integer");
        return -1;
    }
    if (extend < -128 || extend > 127) {
        PyErr_SetString(PyExc_ValueError, "extend cost must fit in 8-bit signed integer");
        return -1;
    }

    // Parse capabilities if provided
    if (capabilities_tuple) {
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) { return -1; }
    }

    sz_status_t status =
        sz_levenshtein_distances_init(match, mismatch, open, extend, NULL, capabilities, &self->handle);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Levenshtein distances engine");
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "%d,%d,%d,%d", match, mismatch, open, extend);
    self->capabilities = capabilities;
    return 0;
}

static PyObject *LevenshteinDistances_repr(LevenshteinDistances *self) {
    return PyUnicode_FromFormat("LevenshteinDistances(match,mismatch,open,extend=%s)", self->description);
}

static PyObject *LevenshteinDistances_get_capabilities(LevenshteinDistances *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *LevenshteinDistances_call(LevenshteinDistances *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    static char *kwlist[] = {"a", "b", "device", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) {
        return NULL;
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

    // Try to swap allocators to unified memory for GPU compatibility
    if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) { return NULL; }

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
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "Levenshtein failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "Levenshtein failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "Levenshtein failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "Levenshtein failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "Levenshtein failed: input/output size mismatch"; break;
        case sz_missing_gpu_k: error_msg = "Levenshtein failed: GPU support is missing in the library"; break;
        case sz_status_unknown_k: error_msg = "Levenshtein failed: unknown error"; break;
        default: error_msg = "Levenshtein failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_LevenshteinDistances[] = //
    "LevenshteinDistances(match=0, mismatch=1, open=1, extend=1, capabilities=None)\n"
    "\n"
    "Compute Levenshtein edit distances between pairs of binary strings.\n"
    "\n"
    "Args:\n"
    "  match (int): Cost for matching characters (default: 0).\n"
    "  mismatch (int): Cost for mismatched characters (default: 1).\n"
    "  open (int): Cost for opening a gap (default: 1).\n"
    "  extend (int): Cost for extending a gap (default: 1).\n"
    "  capabilities (Tuple[str], optional): Hardware capabilities to use.\n"
    "                                       Will be intersected with detected capabilities.\n"
    "                                       Examples: ('serial',), ('haswell', 'parallel')\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.";

static PyGetSetDef LevenshteinDistances_getsetters[] = {
    {"__capabilities__", (getter)LevenshteinDistances_get_capabilities, NULL,
     "Hardware capabilities used by this engine", NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject LevenshteinDistancesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.LevenshteinDistances",
    .tp_doc = doc_LevenshteinDistances,
    .tp_basicsize = sizeof(LevenshteinDistances),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = LevenshteinDistances_new,
    .tp_init = (initproc)LevenshteinDistances_init,
    .tp_dealloc = (destructor)LevenshteinDistances_dealloc,
    .tp_call = (ternaryfunc)LevenshteinDistances_call,
    .tp_repr = (reprfunc)LevenshteinDistances_repr,
    .tp_getset = LevenshteinDistances_getsetters,
};

#pragma endregion

#pragma region LevenshteinDistancesUTF8

typedef struct {
    PyObject ob_base;
    sz_levenshtein_distances_utf8_t handle;
    char description[32];
    sz_capability_t capabilities;
} LevenshteinDistancesUTF8;

static PyObject *LevenshteinDistancesUTF8_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LevenshteinDistancesUTF8 *self = (LevenshteinDistancesUTF8 *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
    }
    return (PyObject *)self;
}

static void LevenshteinDistancesUTF8_dealloc(LevenshteinDistancesUTF8 *self) {
    if (self->handle) { sz_levenshtein_distances_utf8_free(self->handle); }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int LevenshteinDistancesUTF8_init(LevenshteinDistancesUTF8 *self, PyObject *args, PyObject *kwargs) {
    int match = 0, mismatch = 1, open = 1, extend = 1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    static char *kwlist[] = {"match", "mismatch", "open", "extend", "capabilities", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiO", kwlist, &match, &mismatch, &open, &extend,
                                     &capabilities_tuple)) {
        return -1;
    }

    // Validate range of values
    if (match < -128 || match > 127) {
        PyErr_SetString(PyExc_ValueError, "match cost must fit in 8-bit signed integer");
        return -1;
    }
    if (mismatch < -128 || mismatch > 127) {
        PyErr_SetString(PyExc_ValueError, "mismatch cost must fit in 8-bit signed integer");
        return -1;
    }
    if (open < -128 || open > 127) {
        PyErr_SetString(PyExc_ValueError, "open cost must fit in 8-bit signed integer");
        return -1;
    }
    if (extend < -128 || extend > 127) {
        PyErr_SetString(PyExc_ValueError, "extend cost must fit in 8-bit signed integer");
        return -1;
    }

    // Parse capabilities if provided
    if (capabilities_tuple) {
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) { return -1; }
    }

    sz_status_t status =
        sz_levenshtein_distances_utf8_init(match, mismatch, open, extend, NULL, capabilities, &self->handle);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize UTF-8 Levenshtein distances engine");
        return -1;
    }
    snprintf(self->description, sizeof(self->description), "%d,%d,%d,%d", match, mismatch, open, extend);
    self->capabilities = capabilities;
    return 0;
}

static PyObject *LevenshteinDistancesUTF8_repr(LevenshteinDistancesUTF8 *self) {
    return PyUnicode_FromFormat("LevenshteinDistancesUTF8(match,mismatch,open,extend=%s)", self->description);
}

static PyObject *LevenshteinDistancesUTF8_get_capabilities(LevenshteinDistancesUTF8 *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *LevenshteinDistancesUTF8_call(LevenshteinDistancesUTF8 *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    static char *kwlist[] = {"a", "b", "device", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) {
        return NULL;
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

    // Try to swap allocators to unified memory for GPU compatibility
    if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) { return NULL; }

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
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "Levenshtein failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "Levenshtein failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "Levenshtein failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "Levenshtein failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "Levenshtein failed: input/output size mismatch"; break;
        case sz_missing_gpu_k: error_msg = "Levenshtein failed: GPU support is missing in the library"; break;
        case sz_status_unknown_k: error_msg = "Levenshtein failed: unknown error"; break;
        default: error_msg = "Levenshtein failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_LevenshteinDistancesUTF8[] = //
    "LevenshteinDistancesUTF8(match=0, mismatch=1, open=1, extend=1, capabilities=None)\n"
    "\n"
    "Vectorized UTF-8 Levenshtein distance calculator with affine gap penalties.\n"
    "Computes edit distances between pairs of UTF-8 encoded strings.\n"
    "\n"
    "Args:\n"
    "  match (int): Cost of matching characters (default 0).\n"
    "  mismatch (int): Cost of mismatched characters (default 1).\n"
    "  open (int): Cost of opening a gap (default 1).\n"
    "  extend (int): Cost of extending a gap (default 1).\n"
    "  capabilities (Tuple[str], optional): Hardware capabilities to use.\n"
    "                                       Will be intersected with detected capabilities.\n"
    "                                       Examples: ('serial',), ('haswell', 'parallel')\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of UTF-8 strings.\n"
    "  b (sequence): Second sequence of UTF-8 strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.";

static PyGetSetDef LevenshteinDistancesUTF8_getsetters[] = {
    {"__capabilities__", (getter)LevenshteinDistancesUTF8_get_capabilities, NULL,
     "Hardware capabilities used by this engine", NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject LevenshteinDistancesUTF8Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.LevenshteinDistancesUTF8",
    .tp_doc = doc_LevenshteinDistancesUTF8,
    .tp_basicsize = sizeof(LevenshteinDistancesUTF8),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = LevenshteinDistancesUTF8_new,
    .tp_init = (initproc)LevenshteinDistancesUTF8_init,
    .tp_dealloc = (destructor)LevenshteinDistancesUTF8_dealloc,
    .tp_call = (ternaryfunc)LevenshteinDistancesUTF8_call,
    .tp_repr = (reprfunc)LevenshteinDistancesUTF8_repr,
    .tp_getset = LevenshteinDistancesUTF8_getsetters,
};

#pragma endregion

#pragma region NeedlemanWunsch

/**
 *  @brief  Needleman-Wunsch global alignment scoring engine.
 */
typedef struct {
    PyObject ob_base;
    sz_needleman_wunsch_scores_t handle;
    char description[32];
    sz_capability_t capabilities;
} NeedlemanWunsch;

static void NeedlemanWunsch_dealloc(NeedlemanWunsch *self) {
    if (self->handle) {
        sz_needleman_wunsch_scores_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *NeedlemanWunsch_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    NeedlemanWunsch *self = (NeedlemanWunsch *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
    }
    return (PyObject *)self;
}

static int NeedlemanWunsch_init(NeedlemanWunsch *self, PyObject *args, PyObject *kwargs) {
    PyObject *substitution_matrix_obj = NULL;
    sz_error_cost_t open = -1, extend = -1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    // Parse arguments: substitution_matrix, open, extend, capabilities
    static char *kwlist[] = {"substitution_matrix", "open", "extend", "capabilities", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iiO", kwlist, &substitution_matrix_obj, &open, &extend,
                                     &capabilities_tuple)) {
        return -1;
    }

    // Validate substitution matrix (should be a 256x256 numpy array)
    if (!numpy_available || !PyArray_Check(substitution_matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "substitution_matrix must be a NumPy array");
        return -1;
    }

    PyArrayObject *subs_array = (PyArrayObject *)substitution_matrix_obj;
    if (PyArray_NDIM(subs_array) != 2 || PyArray_DIM(subs_array, 0) != 256 || PyArray_DIM(subs_array, 1) != 256) {
        PyErr_SetString(PyExc_ValueError, "substitution_matrix must be a 256x256 array");
        return -1;
    }

    if (PyArray_TYPE(subs_array) != NPY_INT8) {
        PyErr_SetString(PyExc_TypeError, "substitution_matrix must have int8 dtype");
        return -1;
    }

    // Parse capabilities if provided
    if (capabilities_tuple) {
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) { return -1; }
    }

    // Initialize the engine
    sz_error_cost_t *subs_data = (sz_error_cost_t *)PyArray_DATA(subs_array);

    // Create a simple checksum of the substitution matrix for the description
    uint32_t subs_checksum = 0;
    for (int i = 0; i < 256; i += 16) {                    // Sample every 16th element
        subs_checksum += (uint32_t)subs_data[i * 256 + i]; // Diagonal elements
    }

    sz_status_t status = sz_needleman_wunsch_scores_init(subs_data, open, extend, NULL, capabilities, &self->handle);

    if (status != sz_success_k) {
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "NeedlemanWunsch failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "NeedlemanWunsch failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "NeedlemanWunsch failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "NeedlemanWunsch failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "NeedlemanWunsch failed: input/output size mismatch"; break;
        case sz_missing_gpu_k: error_msg = "NeedlemanWunsch failed: GPU support is missing in the library"; break;
        case sz_status_unknown_k: error_msg = "NeedlemanWunsch failed: unknown error"; break;
        default: error_msg = "NeedlemanWunsch failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "%X,%d,%d", subs_checksum & 0xFFFF, open, extend);
    self->capabilities = capabilities;
    return 0;
}

static PyObject *NeedlemanWunsch_repr(NeedlemanWunsch *self) {
    return PyUnicode_FromFormat("NeedlemanWunsch(subs_checksum,open,extend=%s)", self->description);
}

static PyObject *NeedlemanWunsch_get_capabilities(NeedlemanWunsch *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *NeedlemanWunsch_call(NeedlemanWunsch *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    static char *kwlist[] = {"a", "b", "device", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) {
        return NULL;
    }

    // Get device handle
    sz_device_scope_t device_handle = default_device_scope;
    if (device_obj && device_obj != Py_None) {
        if (!PyObject_IsInstance(device_obj, (PyObject *)&DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_handle = ((DeviceScope *)device_obj)->handle;
    }

    sz_size_t kernel_input_size = 0;
    void const *kernel_a_texts_punned = NULL;
    void const *kernel_b_texts_punned = NULL;
    sz_status_t (*kernel_punned)(sz_needleman_wunsch_scores_t, sz_device_scope_t, void const *, void const *,
                                 sz_ssize_t *, sz_size_t) = NULL;
    // Try to swap allocators to unified memory for GPU compatibility
    if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) { return NULL; }

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
        kernel_punned = sz_needleman_wunsch_scores_u32tape;
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
        kernel_punned = sz_needleman_wunsch_scores_u64tape;
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
        kernel_punned = sz_needleman_wunsch_scores_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Unsupported input types for NeedlemanWunsch. "
                     "u32tape: a=%d b=%d, u64tape: a=%d b=%d, seq: a=%d b=%d",
                     a_is_u32tape, b_is_u32tape, a_is_u64tape, b_is_u64tape, a_is_sequence, b_is_sequence);
        return NULL;
    }

    // Make sure the `out` argument is valid NumPy array and extract results info
    PyObject *results_array = NULL;
    sz_ssize_t *kernel_results = NULL;
    sz_size_t kernel_results_stride = sizeof(sz_ssize_t);

    if (!out_obj || out_obj == Py_None) {
        // Create a new NumPy array for results (signed integers for scores)
        npy_intp numpy_size = kernel_input_size;
        results_array = PyArray_SimpleNew(1, &numpy_size, NPY_INT64);
        if (!results_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate results array");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_stride = PyArray_STRIDE((PyArrayObject *)results_array, 0);
    }
    else {
        // Use provided array
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out must be a NumPy array");
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
        if (PyArray_TYPE(array) != NPY_INT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have int64 dtype");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA(array);
        kernel_results_stride = PyArray_STRIDE(array, 0);
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    sz_status_t status = kernel_punned(               //
        self->handle, device_handle,                  //
        kernel_a_texts_punned, kernel_b_texts_punned, //
        kernel_results, kernel_results_stride);

    if (status != sz_success_k) {
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "NeedlemanWunsch failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "NeedlemanWunsch failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "NeedlemanWunsch failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "NeedlemanWunsch failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "NeedlemanWunsch failed: input/output size mismatch"; break;
        case sz_missing_gpu_k: error_msg = "NeedlemanWunsch failed: GPU support is missing in the library"; break;
        case sz_status_unknown_k: error_msg = "NeedlemanWunsch failed: unknown error"; break;
        default: error_msg = "NeedlemanWunsch failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_NeedlemanWunsch[] = //
    "NeedlemanWunsch(substitution_matrix, open=-1, extend=-1, capabilities=None)\n"
    "\n"
    "Needleman-Wunsch global alignment scoring engine.\n"
    "\n"
    "Args:\n"
    "  substitution_matrix (np.ndarray): 256x256 int8 substitution matrix.\n"
    "  open (int): Cost for opening a gap (default: -1).\n"
    "  extend (int): Cost for extending a gap (default: -1).\n"
    "  capabilities (Tuple[str], optional): Hardware capabilities to use.\n"
    "                                       Will be intersected with detected capabilities.\n"
    "                                       Examples: ('serial',), ('haswell', 'parallel')\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.";

static PyTypeObject NeedlemanWunschType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.NeedlemanWunsch",
    .tp_doc = doc_NeedlemanWunsch,
    .tp_basicsize = sizeof(NeedlemanWunsch),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NeedlemanWunsch_new,
    .tp_init = (initproc)NeedlemanWunsch_init,
    .tp_dealloc = (destructor)NeedlemanWunsch_dealloc,
    .tp_call = (ternaryfunc)NeedlemanWunsch_call,
};

#pragma endregion

#pragma region SmithWaterman

/**
 *  @brief  Smith-Waterman local alignment scoring engine.
 */
typedef struct {
    PyObject ob_base;
    sz_smith_waterman_scores_t handle;
} SmithWaterman;

static void SmithWaterman_dealloc(SmithWaterman *self) {
    if (self->handle) {
        sz_smith_waterman_scores_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SmithWaterman_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    SmithWaterman *self = (SmithWaterman *)type->tp_alloc(type, 0);
    if (self != NULL) { self->handle = NULL; }
    return (PyObject *)self;
}

static int SmithWaterman_init(SmithWaterman *self, PyObject *args, PyObject *kwargs) {
    PyObject *substitution_matrix_obj = NULL;
    sz_error_cost_t open = -1, extend = -1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    // Parse arguments: substitution_matrix, open, extend, capabilities
    static char *kwlist[] = {"substitution_matrix", "open", "extend", "capabilities", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iiO", kwlist, &substitution_matrix_obj, &open, &extend,
                                     &capabilities_tuple)) {
        return -1;
    }

    // Validate substitution matrix (should be a 256x256 numpy array)
    if (!numpy_available || !PyArray_Check(substitution_matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "substitution_matrix must be a NumPy array");
        return -1;
    }

    PyArrayObject *subs_array = (PyArrayObject *)substitution_matrix_obj;
    if (PyArray_NDIM(subs_array) != 2 || PyArray_DIM(subs_array, 0) != 256 || PyArray_DIM(subs_array, 1) != 256) {
        PyErr_SetString(PyExc_ValueError, "substitution_matrix must be a 256x256 array");
        return -1;
    }

    if (PyArray_TYPE(subs_array) != NPY_INT8) {
        PyErr_SetString(PyExc_TypeError, "substitution_matrix must have int8 dtype");
        return -1;
    }

    // Parse capabilities if provided
    if (capabilities_tuple) {
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) { return -1; }
    }

    // Initialize the engine
    sz_error_cost_t *subs_data = (sz_error_cost_t *)PyArray_DATA(subs_array);
    sz_status_t status = sz_smith_waterman_scores_init(subs_data, open, extend, NULL, capabilities, &self->handle);

    if (status != sz_success_k) {
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "SmithWaterman failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "SmithWaterman failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "SmithWaterman failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "SmithWaterman failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "SmithWaterman failed: input/output size mismatch"; break;
        case sz_missing_gpu_k: error_msg = "SmithWaterman failed: GPU support is missing in the library"; break;
        case sz_status_unknown_k: error_msg = "SmithWaterman failed: unknown error"; break;
        default: error_msg = "SmithWaterman failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        return -1;
    }

    return 0;
}

static PyObject *SmithWaterman_call(SmithWaterman *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    static char *kwlist[] = {"a", "b", "device", "out", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) {
        return NULL;
    }

    // Get device handle
    sz_device_scope_t device_handle = default_device_scope;
    if (device_obj && device_obj != Py_None) {
        if (!PyObject_IsInstance(device_obj, (PyObject *)&DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_handle = ((DeviceScope *)device_obj)->handle;
    }

    sz_size_t kernel_input_size = 0;
    void const *kernel_a_texts_punned = NULL;
    void const *kernel_b_texts_punned = NULL;
    sz_status_t (*kernel_punned)(sz_smith_waterman_scores_t, sz_device_scope_t, void const *, void const *,
                                 sz_ssize_t *, sz_size_t) = NULL;
    // Try to swap allocators to unified memory for GPU compatibility
    if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) { return NULL; }

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
        kernel_punned = sz_smith_waterman_scores_u32tape;
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
        kernel_punned = sz_smith_waterman_scores_u64tape;
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
        kernel_punned = sz_smith_waterman_scores_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Unsupported input types for SmithWaterman. "
                     "u32tape: a=%d b=%d, u64tape: a=%d b=%d, seq: a=%d b=%d",
                     a_is_u32tape, b_is_u32tape, a_is_u64tape, b_is_u64tape, a_is_sequence, b_is_sequence);
        return NULL;
    }

    // Make sure the `out` argument is valid NumPy array and extract results info
    PyObject *results_array = NULL;
    sz_ssize_t *kernel_results = NULL;
    sz_size_t kernel_results_stride = sizeof(sz_ssize_t);

    if (!out_obj || out_obj == Py_None) {
        // Create a new NumPy array for results (signed integers for scores)
        npy_intp numpy_size = kernel_input_size;
        results_array = PyArray_SimpleNew(1, &numpy_size, NPY_INT64);
        if (!results_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate results array");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_stride = PyArray_STRIDE((PyArrayObject *)results_array, 0);
    }
    else {
        // Use provided array
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out must be a NumPy array");
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
        if (PyArray_TYPE(array) != NPY_INT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have int64 dtype");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA(array);
        kernel_results_stride = PyArray_STRIDE(array, 0);
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    sz_status_t status = kernel_punned(               //
        self->handle, device_handle,                  //
        kernel_a_texts_punned, kernel_b_texts_punned, //
        kernel_results, kernel_results_stride);

    if (status != sz_success_k) {
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "SmithWaterman failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "SmithWaterman failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "SmithWaterman failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "SmithWaterman failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "SmithWaterman failed: input/output size mismatch"; break;
        case sz_missing_gpu_k: error_msg = "SmithWaterman failed: GPU support is missing in the library"; break;
        case sz_status_unknown_k: error_msg = "SmithWaterman failed: unknown error"; break;
        default: error_msg = "SmithWaterman failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_SmithWaterman[] = //
    "SmithWaterman(substitution_matrix, open=-1, extend=-1, capabilities=None)\n"
    "\n"
    "Smith-Waterman local alignment scoring engine.\n"
    "\n"
    "Args:\n"
    "  substitution_matrix (np.ndarray): 256x256 int8 substitution matrix.\n"
    "  open (int): Cost for opening a gap (default: -1).\n"
    "  extend (int): Cost for extending a gap (default: -1).\n"
    "  capabilities (Tuple[str], optional): Hardware capabilities to use.\n"
    "                                       Will be intersected with detected capabilities.\n"
    "                                       Examples: ('serial',), ('haswell', 'parallel')\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.";

static PyTypeObject SmithWatermanType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.SmithWaterman",
    .tp_doc = doc_SmithWaterman,
    .tp_basicsize = sizeof(SmithWaterman),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = SmithWaterman_new,
    .tp_init = (initproc)SmithWaterman_init,
    .tp_dealloc = (destructor)SmithWaterman_dealloc,
    .tp_call = (ternaryfunc)SmithWaterman_call,
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
    sz_py_replace_strings_allocator = api->sz_py_replace_strings_allocator;

    Py_DECREF(capsule);
    Py_DECREF(stringzilla_module);

    // Check that all functions were loaded
    if (!sz_py_export_string_like || !sz_py_export_strings_as_sequence || !sz_py_export_strings_as_u32tape ||
        !sz_py_export_strings_as_u64tape || !sz_py_replace_strings_allocator) {
        PyErr_SetString(PyExc_ImportError, "Failed to import required StringZilla C API functions");
        return NULL;
    }

    // Initialize the unified memory allocator for GPU compatibility
    sz_status_t alloc_status = sz_memory_allocator_init_unified(&unified_allocator);
    if (alloc_status != sz_success_k) sz_memory_allocator_init_default(&unified_allocator);

    // Initialize the default device scope for reuse
    sz_status_t status = sz_device_scope_init_default(&default_device_scope);
    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize default device scope");
        return NULL;
    }

    if (PyType_Ready(&DeviceScopeType) < 0) return NULL;
    if (PyType_Ready(&LevenshteinDistancesType) < 0) return NULL;
    if (PyType_Ready(&LevenshteinDistancesUTF8Type) < 0) return NULL;
    if (PyType_Ready(&NeedlemanWunschType) < 0) return NULL;
    if (PyType_Ready(&SmithWatermanType) < 0) return NULL;

    m = PyModule_Create(&stringzillas_module);
    if (m == NULL) return NULL;

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", szs_version_major(), szs_version_minor(), szs_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Initialize hardware capabilities for capability intersection
    default_hardware_capabilities = szs_capabilities();

    // Define SIMD capabilities as a tuple
    {
        sz_capability_t caps = default_hardware_capabilities;

        // Get capability strings using the new function
        char const *cap_strings[SZ_CAPABILITIES_COUNT];
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
        sz_cptr_t caps_str = sz_capabilities_to_string_implementation_(caps);
        PyModule_AddStringConstant(m, "__capabilities_str__", caps_str);
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

    Py_INCREF(&NeedlemanWunschType);
    if (PyModule_AddObject(m, "NeedlemanWunsch", (PyObject *)&NeedlemanWunschType) < 0) {
        Py_XDECREF(&NeedlemanWunschType);
        Py_XDECREF(&LevenshteinDistancesUTF8Type);
        Py_XDECREF(&LevenshteinDistancesType);
        Py_XDECREF(&DeviceScopeType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&SmithWatermanType);
    if (PyModule_AddObject(m, "SmithWaterman", (PyObject *)&SmithWatermanType) < 0) {
        Py_XDECREF(&SmithWatermanType);
        Py_XDECREF(&NeedlemanWunschType);
        Py_XDECREF(&LevenshteinDistancesUTF8Type);
        Py_XDECREF(&LevenshteinDistancesType);
        Py_XDECREF(&DeviceScopeType);
        Py_XDECREF(m);
        return NULL;
    }

    return m;
}