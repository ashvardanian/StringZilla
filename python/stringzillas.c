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
 *  @brief Creates a Python tuple from capabilities mask.
 *  @param[in] caps Capabilities mask
 *  @return New reference to Python tuple, or NULL on error
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

// Function pointers for stringzilla functions imported from capsules
static sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *) = NULL;
static sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *) = NULL;
static sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *) = NULL;
static sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *) = NULL;
static sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *) = NULL;

// Default device scope that can be safely reused across calls
// The underlying implementation is stateless and thread-safe
static szs_device_scope_t default_device_scope = NULL;
// Static variable to store hardware capabilities
static sz_capability_t default_hardware_capabilities = 0;
// Static unified memory allocator for GPU compatibility
static sz_memory_allocator_t unified_allocator;
// Default CPU-side allocator for buffer-based flows
static sz_memory_allocator_t default_allocator;

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
 *  @brief Helper function to automatically swap a Strs object's allocator to unified memory for GPU kernels.
 *  @param[in] strs_obj The Strs object to swap allocator for
 *  @return sz_true_k on success, sz_false_k on failure
 *  @note Sets Pythonic error on failure.
 */
static inline sz_bool_t try_swap_to_unified_allocator(PyObject *strs_obj) {
    if (!strs_obj || !sz_py_replace_strings_allocator) return sz_false_k;

    // Try to swap to unified allocator - this will be a no-op if already using it
    sz_bool_t success = sz_py_replace_strings_allocator(strs_obj, &unified_allocator);

    if (!success) {
        // Always fatal: GPU kernels require unified/device-accessible memory
        PyErr_SetString(PyExc_RuntimeError,
                        "Device memory mismatch: GPU kernels require unified/device-accessible memory. "
                        "Consider reducing input size, freeing memory, or using CPU capabilities.");
        return sz_false_k;
    }
    return sz_true_k;
}

/**
 *  @brief Helper function to determine if unified memory is required based on capabilities and device scope.
 *  @param[in] capabilities The capabilities bitmask of the current engine.
 */
static inline sz_bool_t requires_unified_memory(sz_capability_t capabilities) {
    return (capabilities & sz_cap_cuda_k) != 0;
}

#pragma endregion

#pragma region DeviceScope

/**
 *  @brief  Device scope for controlling execution context (CPU cores or GPU device).
 */
typedef struct {
    PyObject ob_base;
    szs_device_scope_t handle;
    char description[32];
} DeviceScope;

static void DeviceScope_dealloc(DeviceScope *self) {
    if (self->handle) {
        szs_device_scope_free(self->handle);
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
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &cpu_cores_obj, &gpu_device_obj)) return -1;

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
        status = szs_device_scope_init_cpu_cores(cpu_cores, &self->handle);
        if (cpu_cores == 1) { snprintf(self->description, sizeof(self->description), "default"); }
        else if (cpu_cores == 0) { snprintf(self->description, sizeof(self->description), "CPUs:all"); }
        else { snprintf(self->description, sizeof(self->description), "CPUs:%zu", cpu_cores); }
    }
    else if (gpu_device_obj != NULL) {
        if (!PyLong_Check(gpu_device_obj)) {
            PyErr_SetString(PyExc_TypeError, "gpu_device must be an integer");
            return -1;
        }
        gpu_device = PyLong_AsSize_t(gpu_device_obj);
        if (gpu_device == (sz_size_t)-1 && PyErr_Occurred()) { return -1; }
        status = szs_device_scope_init_gpu_device(gpu_device, &self->handle);
        snprintf(self->description, sizeof(self->description), "GPU:%zu", gpu_device);
    }
    else {
        status = szs_device_scope_init_default(&self->handle);
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
    "  cpu_cores (int, optional): Number of CPU cores to use, or zero for all cores.\n"
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

#pragma region Metadata

/**
 *  @brief Parse capabilities from a Python tuple of strings and intersect with hardware capabilities.
 *  @param[in] caps_tuple Python tuple containing capability strings (e.g., ('serial', 'haswell')).
 *  @param[out] result Output capability mask after intersection with hardware capabilities.
 *  @return 0 on success, -1 on error (with Python exception set).
 */
static int parse_and_intersect_capabilities(PyObject *caps_obj, sz_capability_t *result) {
    // Handle `DeviceScope` objects
    if (PyObject_IsInstance(caps_obj, (PyObject *)&DeviceScopeType)) {
        DeviceScope *device_scope = (DeviceScope *)caps_obj;

        // Try to get GPU device
        sz_size_t gpu_device;
        if (szs_device_scope_get_gpu_device(device_scope->handle, &gpu_device) == sz_success_k) {
            if (default_hardware_capabilities & sz_caps_cuda_k) {
                *result = sz_caps_cuda_k & default_hardware_capabilities;
                return 0;
            }
            else {
                PyErr_SetString(PyExc_RuntimeError, "GPU DeviceScope requested but CUDA not available");
                return -1;
            }
        }

        // Try to get CPU cores first
        sz_size_t cpu_cores;
        if (szs_device_scope_get_cpu_cores(device_scope->handle, &cpu_cores) == sz_success_k) {
            *result = sz_caps_cpus_k & default_hardware_capabilities;
            return 0;
        }

        // Default scope - use all available capabilities
        *result = default_hardware_capabilities;
        return 0;
    }

    // Handle tuple of capability strings (original behavior)
    if (!PyTuple_Check(caps_obj)) {
        PyErr_SetString(PyExc_TypeError, "capabilities must be a tuple of strings or a DeviceScope object");
        return -1;
    }

    sz_capability_t requested_caps = 0;
    Py_ssize_t n = PyTuple_Size(caps_obj);

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(caps_obj, i);
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

#pragma region LevenshteinDistances

/**
 *  @brief  Levenshtein distance computation engine for binary strings.
 */
typedef struct {
    PyObject ob_base;
    szs_levenshtein_distances_t handle;
    char description[32];
    sz_capability_t capabilities;
} LevenshteinDistances;

static void LevenshteinDistances_dealloc(LevenshteinDistances *self) {
    if (self->handle) {
        szs_levenshtein_distances_free(self->handle);
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
                                     &capabilities_tuple))
        return -1;

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
        szs_levenshtein_distances_init(match, mismatch, open, extend, NULL, capabilities, &self->handle);

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
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) return NULL;

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    szs_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t kernel_input_size = 0;
    void *kernel_a_texts_punned = NULL;
    void *kernel_b_texts_punned = NULL;
    sz_size_t *kernel_results = NULL;
    sz_size_t kernel_results_stride = sizeof(sz_size_t);
    sz_status_t (*kernel_punned)(szs_levenshtein_distances_t, szs_device_scope_t, void *, void *, sz_size_t *,
                                 sz_size_t) = NULL;

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    if (requires_unified_memory(self->capabilities))
        if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) return NULL;

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
        kernel_punned = szs_levenshtein_distances_u32tape;
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
        kernel_punned = szs_levenshtein_distances_u64tape;
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
        kernel_punned = szs_levenshtein_distances_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs objects, got %s and %s. "
                     "Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(a_obj)->tp_name, Py_TYPE(b_obj)->tp_name);
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
        case sz_missing_gpu_k:
            error_msg = "Levenshtein failed: CUDA backend requested but no GPU device scope provided. "
                        "Pass device=stringzillas.DeviceScope(gpu_device=0) or use serial/CPU capabilities.";
            break;
        case sz_device_code_mismatch_k:
            error_msg = "Levenshtein failed: device-code mismatch between backend and executor. "
                        "Use a GPU DeviceScope with CUDA backends or select CPU capabilities.";
            break;
        case sz_device_memory_mismatch_k:
            error_msg = "Levenshtein failed: device-memory mismatch (unified/device-accessible memory required).";
            break;
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
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"
    "                                       or a DeviceScope for automatic capability inference.\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.\n"
    "\n"
    "Examples:\n"
    "  ```python\n"
    "  # Minimal CPU example with auto-inferred capabilities\n"
    "  import stringzilla as sz, stringzillas as szs\n"
    "  engine = szs.LevenshteinDistances()\n"
    "  strings_a = sz.Strs(['hello', 'world'])\n"
    "  strings_b = sz.Strs(['hallo', 'word'])\n"
    "  distances = engine(strings_a, strings_b)\n"
    "  \n"
    "  # GPU example with custom costs and auto-inferred capabilities\n"
    "  gpu_scope = szs.DeviceScope(gpu_device=0)\n"
    "  engine = szs.LevenshteinDistances(match=0, mismatch=2, open=3, extend=1, capabilities=gpu_scope)\n"
    "  distances = engine(strings_a, strings_b, device=gpu_scope)\n"
    "  ```";

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
    szs_levenshtein_distances_utf8_t handle;
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
    if (self->handle) { szs_levenshtein_distances_utf8_free(self->handle); }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int LevenshteinDistancesUTF8_init(LevenshteinDistancesUTF8 *self, PyObject *args, PyObject *kwargs) {
    int match = 0, mismatch = 1, open = 1, extend = 1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    static char *kwlist[] = {"match", "mismatch", "open", "extend", "capabilities", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|iiiiO", kwlist, &match, &mismatch, &open, &extend,
                                     &capabilities_tuple))
        return -1;

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
        szs_levenshtein_distances_utf8_init(match, mismatch, open, extend, NULL, capabilities, &self->handle);

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
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) return NULL;

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    szs_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t kernel_input_size = 0;
    void *kernel_a_texts_punned = NULL;
    void *kernel_b_texts_punned = NULL;
    sz_size_t *kernel_results = NULL;
    sz_size_t kernel_results_stride = sizeof(sz_size_t);
    sz_status_t (*kernel_punned)(szs_levenshtein_distances_t, szs_device_scope_t, void *, void *, sz_size_t *,
                                 sz_size_t) = NULL;

    // Swap allocators when engine supports CUDA
    if (requires_unified_memory(self->capabilities))
        if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) return NULL;

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
        kernel_punned = szs_levenshtein_distances_utf8_u32tape;
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
        kernel_punned = szs_levenshtein_distances_utf8_u64tape;
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
        kernel_punned = szs_levenshtein_distances_utf8_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs objects, got %s and %s. "
                     "Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(a_obj)->tp_name, Py_TYPE(b_obj)->tp_name);
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
        case sz_missing_gpu_k:
            error_msg = "Levenshtein failed: CUDA backend requested but no GPU device scope provided. "
                        "Pass device=stringzillas.DeviceScope(gpu_device=0) or use serial/CPU capabilities.";
            break;
        case sz_device_code_mismatch_k:
            error_msg = "Levenshtein failed: device-code mismatch between backend and executor. "
                        "Use a GPU DeviceScope with CUDA backends or select CPU capabilities.";
            break;
        case sz_device_memory_mismatch_k:
            error_msg = "Levenshtein failed: device-memory mismatch (unified/device-accessible memory required).";
            break;
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
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"
    "                                       or a DeviceScope for automatic capability inference.\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of UTF-8 strings.\n"
    "  b (sequence): Second sequence of UTF-8 strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.\n"
    "\n"
    "Examples:\n"
    "  ```python\n"
    "  # Minimal CPU example with Unicode strings\n"
    "  import stringzilla as sz, stringzillas as szs\n"
    "  engine = szs.LevenshteinDistancesUTF8()\n"
    "  strings_a = sz.Strs(['café', 'naïve'])\n"
    "  strings_b = sz.Strs(['caffe', 'naive'])\n"
    "  distances = engine(strings_a, strings_b)\n"
    "  \n"
    "  # GPU example with high mismatch penalty\n"
    "  gpu_scope = szs.DeviceScope(gpu_device=0)\n"
    "  engine = szs.LevenshteinDistancesUTF8(mismatch=5, capabilities=gpu_scope)\n"
    "  distances = engine(strings_a, strings_b, device=gpu_scope)\n"
    "  ```";

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
    szs_needleman_wunsch_scores_t handle;
    char description[32];
    sz_capability_t capabilities;
} NeedlemanWunsch;

static void NeedlemanWunsch_dealloc(NeedlemanWunsch *self) {
    if (self->handle) {
        szs_needleman_wunsch_scores_free(self->handle);
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
                                     &capabilities_tuple))
        return -1;

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
    sz_u32_t subs_checksum = 0;
    for (int i = 0; i < 256; i += 16)                      // Sample every 16th element
        subs_checksum += (sz_u32_t)subs_data[i * 256 + i]; // Diagonal elements

    sz_status_t status = szs_needleman_wunsch_scores_init(subs_data, open, extend, NULL, capabilities, &self->handle);
    if (status != sz_success_k) {
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "NeedlemanWunsch failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "NeedlemanWunsch failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "NeedlemanWunsch failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "NeedlemanWunsch failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "NeedlemanWunsch failed: input/output size mismatch"; break;
        case sz_missing_gpu_k:
            error_msg = "NeedlemanWunsch failed: CUDA backend requested but no GPU device scope provided. "
                        "Pass device=stringzillas.DeviceScope(gpu_device=0) or use serial/CPU capabilities.";
            break;

        case sz_device_code_mismatch_k:
            error_msg = "NeedlemanWunsch failed: device-code mismatch between backend and executor. "
                        "Use a GPU DeviceScope with CUDA backends or select CPU capabilities.";
            break;
        case sz_device_memory_mismatch_k:
            error_msg = "NeedlemanWunsch failed: device-memory mismatch (unified/device-accessible memory required).";
            break;
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
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) return NULL;

    // Get device handle
    szs_device_scope_t device_handle = default_device_scope;
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
    sz_status_t (*kernel_punned)(szs_needleman_wunsch_scores_t, szs_device_scope_t, void const *, void const *,
                                 sz_ssize_t *, sz_size_t) = NULL;

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    if (requires_unified_memory(self->capabilities))
        if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) return NULL;

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
        kernel_punned = szs_needleman_wunsch_scores_u32tape;
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
        kernel_punned = szs_needleman_wunsch_scores_u64tape;
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
        kernel_punned = szs_needleman_wunsch_scores_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs objects, got %s and %s. "
                     "Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(a_obj)->tp_name, Py_TYPE(b_obj)->tp_name);
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
        case sz_missing_gpu_k:
            error_msg = "NeedlemanWunsch failed: CUDA backend requested but no GPU device scope provided. "
                        "Pass device=stringzillas.DeviceScope(gpu_device=0) or use serial/CPU capabilities.";
            break;
        case sz_device_code_mismatch_k:
            error_msg = "NeedlemanWunsch failed: device-code mismatch between backend and executor. "
                        "Use a GPU DeviceScope with CUDA backends or select CPU capabilities.";
            break;
        case sz_device_memory_mismatch_k:
            error_msg = "NeedlemanWunsch failed: device-memory mismatch (unified/device-accessible memory required).";
            break;
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
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"
    "                                       or a DeviceScope for automatic capability inference.\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.\n"
    "\n"
    "Examples:\n"
    "  ```python\n"
    "  # Minimal CPU example with BLOSUM62 matrix\n"
    "  import numpy as np, stringzilla as sz, stringzillas as szs\n"
    "  matrix = np.zeros((256, 256), dtype=np.int8)\n"
    "  engine = szs.NeedlemanWunsch(substitution_matrix=matrix)\n"
    "  proteins_a = sz.Strs(['ACGT', 'TGCA'])\n"
    "  proteins_b = sz.Strs(['ACCT', 'TGAA'])\n"
    "  scores = engine(proteins_a, proteins_b)\n"
    "  \n"
    "  # GPU example with custom gap penalties\n"
    "  gpu_scope = szs.DeviceScope(gpu_device=0)\n"
    "  engine = szs.NeedlemanWunsch(substitution_matrix=matrix, open=-2, extend=-1, capabilities=gpu_scope)\n"
    "  scores = engine(proteins_a, proteins_b, device=gpu_scope)\n"
    "  ```";

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
    szs_smith_waterman_scores_t handle;
    char description[32];
    sz_capability_t capabilities;
} SmithWaterman;

static void SmithWaterman_dealloc(SmithWaterman *self) {
    if (self->handle) {
        szs_smith_waterman_scores_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SmithWaterman_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    SmithWaterman *self = (SmithWaterman *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
    }
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
                                     &capabilities_tuple))
        return -1;

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
    sz_status_t status = szs_smith_waterman_scores_init(subs_data, open, extend, NULL, capabilities, &self->handle);

    if (status != sz_success_k) {
        char const *error_msg;
        switch (status) {
        case sz_bad_alloc_k: error_msg = "SmithWaterman failed: memory allocation failed"; break;
        case sz_invalid_utf8_k: error_msg = "SmithWaterman failed: invalid UTF-8 input"; break;
        case sz_contains_duplicates_k: error_msg = "SmithWaterman failed: contains duplicates"; break;
        case sz_overflow_risk_k: error_msg = "SmithWaterman failed: overflow risk"; break;
        case sz_unexpected_dimensions_k: error_msg = "SmithWaterman failed: input/output size mismatch"; break;
        case sz_missing_gpu_k:
            error_msg = "SmithWaterman failed: CUDA backend requested but no GPU device scope provided. "
                        "Pass device=stringzillas.DeviceScope(gpu_device=0) or use serial/CPU capabilities.";
            break;
        case sz_device_code_mismatch_k:
            error_msg = "SmithWaterman failed: device-code mismatch between backend and executor. "
                        "Use a GPU DeviceScope with CUDA backends or select CPU capabilities.";
            break;
        case sz_device_memory_mismatch_k:
            error_msg = "SmithWaterman failed: device-memory mismatch (unified/device-accessible memory required).";
            break;
        case sz_status_unknown_k: error_msg = "SmithWaterman failed: unknown error"; break;
        default: error_msg = "SmithWaterman failed: unexpected error"; break;
        }
        PyErr_Format(PyExc_RuntimeError, "%s (status code: %d)", error_msg, (int)status);
        return -1;
    }

    // Create a simple checksum of the substitution matrix for the description
    sz_u32_t subs_checksum = 0;
    for (int i = 0; i < 256; i += 16)                      // Sample every 16th element
        subs_checksum += (sz_u32_t)subs_data[i * 256 + i]; // Diagonal elements

    snprintf(self->description, sizeof(self->description), "%X,%d,%d", subs_checksum & 0xFFFF, open, extend);
    self->capabilities = capabilities;
    return 0;
}

static PyObject *SmithWaterman_call(SmithWaterman *self, PyObject *args, PyObject *kwargs) {
    PyObject *a_obj = NULL, *b_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    static char *kwlist[] = {"a", "b", "device", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &a_obj, &b_obj, &device_obj, &out_obj)) return NULL;

    // Get device handle
    szs_device_scope_t device_handle = default_device_scope;
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
    sz_status_t (*kernel_punned)(szs_smith_waterman_scores_t, szs_device_scope_t, void const *, void const *,
                                 sz_ssize_t *, sz_size_t) = NULL;

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    if (requires_unified_memory(self->capabilities))
        if (!try_swap_to_unified_allocator(a_obj) || !try_swap_to_unified_allocator(b_obj)) return NULL;

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
        kernel_punned = szs_smith_waterman_scores_u32tape;
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
        kernel_punned = szs_smith_waterman_scores_u64tape;
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
        kernel_punned = szs_smith_waterman_scores_sequence;
        kernel_a_texts_punned = &a_seq;
        kernel_b_texts_punned = &b_seq;
    }

    // If no valid input types were found, raise an error
    if (!kernel_punned) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs objects, got %s and %s. "
                     "Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(a_obj)->tp_name, Py_TYPE(b_obj)->tp_name);
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
        case sz_missing_gpu_k:
            error_msg = "SmithWaterman failed: CUDA backend requested but no GPU device scope provided. "
                        "Pass device=stringzillas.DeviceScope(gpu_device=0) or use serial/CPU capabilities.";
            break;
        case sz_device_code_mismatch_k:
            error_msg = "SmithWaterman failed: device-code mismatch between backend and executor. "
                        "Use a GPU DeviceScope with CUDA backends or select CPU capabilities.";
            break;
        case sz_device_memory_mismatch_k:
            error_msg = "SmithWaterman failed: device-memory mismatch (unified/device-accessible memory required).";
            break;
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

static PyObject *SmithWaterman_repr(SmithWaterman *self) {
    return PyUnicode_FromFormat("SmithWaterman(subs_checksum,open,extend=%s)", self->description);
}

static PyObject *SmithWaterman_get_capabilities(SmithWaterman *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyGetSetDef SmithWaterman_getsetters[] = {
    {"__capabilities__", (getter)SmithWaterman_get_capabilities, NULL, "Hardware capabilities used by this engine",
     NULL},
    {NULL} /* Sentinel */
};

static char const doc_SmithWaterman[] = //
    "SmithWaterman(substitution_matrix, open=-1, extend=-1, capabilities=None)\n"
    "\n"
    "Smith-Waterman local alignment scoring engine.\n"
    "\n"
    "Args:\n"
    "  substitution_matrix (np.ndarray): 256x256 int8 substitution matrix.\n"
    "  open (int): Cost for opening a gap (default: -1).\n"
    "  extend (int): Cost for extending a gap (default: -1).\n"
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"
    "                                       or a DeviceScope for automatic capability inference.\n"
    "\n"
    "Call with:\n"
    "  a (sequence): First sequence of strings.\n"
    "  b (sequence): Second sequence of strings.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "  out (array, optional): Output buffer for results.\n"
    "\n"
    "Examples:\n"
    "  ```python\n"
    "  # Minimal CPU example for local alignment\n"
    "  import numpy as np, stringzilla as sz, stringzillas as szs\n"
    "  matrix = np.eye(256, dtype=np.int8)  # Identity matrix\n"
    "  engine = szs.SmithWaterman(substitution_matrix=matrix)\n"
    "  seqs_a = sz.Strs(['ACGTACGT', 'TGCATGCA'])\n"
    "  seqs_b = sz.Strs(['CGTACGTA', 'GCATGCAT'])\n"
    "  scores = engine(seqs_a, seqs_b)\n"
    "  \n"
    "  # GPU example with different gap costs\n"
    "  gpu_scope = szs.DeviceScope(gpu_device=0)\n"
    "  engine = szs.SmithWaterman(substitution_matrix=matrix, open=-3, extend=-1, capabilities=gpu_scope)\n"
    "  scores = engine(seqs_a, seqs_b, device=gpu_scope)\n"
    "  ```";

static PyTypeObject SmithWatermanType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.SmithWaterman",
    .tp_doc = doc_SmithWaterman,
    .tp_basicsize = sizeof(SmithWaterman),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = SmithWaterman_new,
    .tp_init = (initproc)SmithWaterman_init,
    .tp_dealloc = (destructor)SmithWaterman_dealloc,
    .tp_call = (ternaryfunc)SmithWaterman_call,
    .tp_repr = (reprfunc)SmithWaterman_repr,
    .tp_getset = SmithWaterman_getsetters,
};

#pragma endregion

#pragma region Fingerprints

/**
 *  @brief  Fingerprinting engine for binary strings.
 */
typedef struct {
    PyObject ob_base;
    szs_fingerprints_t handle;
    char description[64];
    sz_capability_t capabilities;
    sz_size_t ndim;
} Fingerprints;

static void Fingerprints_dealloc(Fingerprints *self) {
    if (self->handle) {
        szs_fingerprints_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Fingerprints_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Fingerprints *self = (Fingerprints *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
        self->ndim = 0;
    }
    return (PyObject *)self;
}

static int Fingerprints_init(Fingerprints *self, PyObject *args, PyObject *kwargs) {
    sz_size_t ndim;
    PyObject *window_widths_obj = NULL;
    sz_size_t alphabet_size = 256;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    static char *kwlist[] = {"ndim", "window_widths", "alphabet_size", "capabilities", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|OnO", kwlist, &ndim, &window_widths_obj, &alphabet_size,
                                     &capabilities_tuple))
        return -1;

    // Parse capabilities if provided
    if (capabilities_tuple)
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) return -1;

    sz_size_t *window_widths = NULL;
    sz_size_t window_widths_count = 0;

    // Parse window_widths if provided - require NumPy array of uint64
    if (window_widths_obj && window_widths_obj != Py_None) {
        if (!PyArray_Check(window_widths_obj)) {
            PyErr_SetString(PyExc_TypeError, "window_widths must be a numpy array of uint64");
            return -1;
        }

        PyArrayObject *arr = (PyArrayObject *)window_widths_obj;

        // Check dtype is uint64
        if (PyArray_TYPE(arr) != NPY_UINT64) {
            PyErr_SetString(PyExc_TypeError, "window_widths must have dtype uint64");
            return -1;
        }

        // Check that it's 1D
        if (PyArray_NDIM(arr) != 1) {
            PyErr_SetString(PyExc_ValueError, "window_widths must be a 1D array");
            return -1;
        }

        // Check that it's contiguous (no strides)
        if (!PyArray_IS_C_CONTIGUOUS(arr)) {
            PyErr_SetString(PyExc_ValueError, "window_widths must be a contiguous C-style array (no strides)");
            return -1;
        }

        window_widths_count = PyArray_SIZE(arr);
        window_widths = (sz_size_t *)PyArray_DATA(arr);
    }

    sz_status_t status = szs_fingerprints_init(ndim, alphabet_size, window_widths, window_widths_count, NULL,
                                               capabilities, &self->handle);

    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize Fingerprints engine");
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "ndim=%zu,window_widths=%zu,alphabet_size=%zu", ndim,
             window_widths_count, alphabet_size);
    self->capabilities = capabilities;
    self->ndim = ndim;
    return 0;
}

static PyObject *Fingerprints_repr(Fingerprints *self) {
    return PyUnicode_FromFormat("Fingerprints(%s)", self->description);
}

static PyObject *Fingerprints_get_capabilities(Fingerprints *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *Fingerprints_call(Fingerprints *self, PyObject *args, PyObject *kwargs) {

    PyObject *texts_obj = NULL, *device_obj = NULL, *out_obj = NULL;
    static char *kwlist[] = {"texts", "device", "out", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &texts_obj, &device_obj, &out_obj)) return NULL;

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    szs_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;

    // Handle empty input - return tuple of empty arrays
    if (PySequence_Check(texts_obj) && PySequence_Size(texts_obj) == 0) {
        npy_intp dims[2] = {0, self->ndim};
        PyArrayObject *empty_hashes = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
        PyArrayObject *empty_counts = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);

        if (!empty_hashes || !empty_counts) {
            Py_XDECREF(empty_hashes);
            Py_XDECREF(empty_counts);
            return PyErr_NoMemory();
        }

        PyObject *result_tuple = PyTuple_New(2);
        if (!result_tuple) {
            Py_DECREF(empty_hashes);
            Py_DECREF(empty_counts);
            return NULL;
        }

        PyTuple_SET_ITEM(result_tuple, 0, (PyObject *)empty_hashes);
        PyTuple_SET_ITEM(result_tuple, 1, (PyObject *)empty_counts);
        return result_tuple;
    }

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    sz_bool_t need_unified = requires_unified_memory(self->capabilities);
    if (need_unified)
        if (!try_swap_to_unified_allocator(texts_obj)) return NULL;

    sz_size_t kernel_input_size = 0;
    void *kernel_texts_punned = NULL;
    sz_status_t (*kernel_punned)(szs_fingerprints_t, szs_device_scope_t, void *, sz_u32_t *, sz_size_t, sz_u32_t *,
                                 sz_size_t) = NULL;

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t texts_u32tape;
    sz_bool_t texts_is_u32tape = sz_py_export_strings_as_u32tape( //
        texts_obj, &texts_u32tape.data, &texts_u32tape.offsets, &texts_u32tape.count);
    if (texts_is_u32tape) {
        kernel_input_size = texts_u32tape.count;
        kernel_punned = szs_fingerprints_u32tape;
        kernel_texts_punned = &texts_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t texts_u64tape;
    sz_bool_t texts_is_u64tape =
        !texts_is_u32tape && sz_py_export_strings_as_u64tape( //
                                 texts_obj, &texts_u64tape.data, &texts_u64tape.offsets, &texts_u64tape.count);
    if (texts_is_u64tape) {
        kernel_input_size = texts_u64tape.count;
        kernel_punned = szs_fingerprints_u64tape;
        kernel_texts_punned = &texts_u64tape;
    }

    // Handle generic sequence inputs
    sz_sequence_t texts_seq;
    sz_bool_t texts_is_sequence =
        !texts_is_u32tape && !texts_is_u64tape && sz_py_export_strings_as_sequence(texts_obj, &texts_seq);
    if (texts_is_sequence) {
        kernel_input_size = texts_seq.count;
        kernel_punned = szs_fingerprints_sequence;
        kernel_texts_punned = &texts_seq;
    }

    if (kernel_punned == NULL) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs object, got %s. Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(texts_obj)->tp_name);
        return NULL;
    }

    // Create NumPy outputs up front and copy into them (CPU or GPU)
    npy_intp dims[2] = {kernel_input_size, self->ndim};
    PyArrayObject *hashes_array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
    PyArrayObject *counts_array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_UINT32);
    if (!hashes_array || !counts_array) {
        Py_XDECREF(hashes_array);
        Py_XDECREF(counts_array);
        return PyErr_NoMemory();
    }

    // Determine bytes to write; if zero, we'll just return the empty arrays
    sz_memory_allocator_t *out_alloc = need_unified ? &unified_allocator : &default_allocator;
    sz_size_t const total_elements = kernel_input_size * self->ndim;
    sz_size_t const total_bytes = total_elements * sizeof(sz_u32_t);

    if (total_bytes > 0) {
        sz_u32_t *buf_hashes = (sz_u32_t *)out_alloc->allocate(total_bytes, out_alloc->handle);
        sz_u32_t *buf_counts = (sz_u32_t *)out_alloc->allocate(total_bytes, out_alloc->handle);
        if (!buf_hashes || !buf_counts) {
            if (buf_hashes) out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
            if (buf_counts) out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
            Py_DECREF(hashes_array);
            Py_DECREF(counts_array);
            return PyErr_NoMemory();
        }

        sz_status_t status = kernel_punned(self->handle, device_handle, kernel_texts_punned, buf_hashes,
                                           self->ndim * sizeof(sz_u32_t), buf_counts, self->ndim * sizeof(sz_u32_t));
        if (status != sz_success_k) {
            out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
            out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
            Py_DECREF(hashes_array);
            Py_DECREF(counts_array);
            PyErr_SetString(PyExc_RuntimeError, "Fingerprinting computation failed");
            return NULL;
        }

        memcpy(PyArray_DATA(hashes_array), buf_hashes, total_bytes);
        memcpy(PyArray_DATA(counts_array), buf_counts, total_bytes);
        out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
        out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
    }

    PyObject *result_tuple = PyTuple_New(2);
    if (!result_tuple) {
        Py_DECREF(hashes_array);
        Py_DECREF(counts_array);
        return NULL;
    }

    PyTuple_SET_ITEM(result_tuple, 0, (PyObject *)hashes_array);
    PyTuple_SET_ITEM(result_tuple, 1, (PyObject *)counts_array);

    return result_tuple;
}

static char const doc_Fingerprints[] = //
    "Fingerprints(ndim, window_widths=None, alphabet_size=256, capabilities=None)\n"
    "\n"
    "Compute MinHash fingerprints for binary strings.\n"
    "\n"
    "Args:\n"
    "  ndim (int): Number of dimensions per fingerprint.\n"
    "  window_widths (numpy.array, optional): 1D uint64 contiguous array of window widths. Uses defaults if None.\n"
    "  alphabet_size (int, optional): Alphabet size, default 256 for binary strings.\n"
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"
    "                                       Can be explicit capabilities like ('serial', 'parallel', 'cuda')\n"
    "                                       or a DeviceScope for automatic capability inference.\n"
    "\n"
    "Call with:\n"
    "  texts (sequence): Sequence of strings to fingerprint.\n"
    "  device (DeviceScope, optional): Device execution context.\n"
    "\n"
    "Returns:\n"
    "  tuple: (hashes_matrix, counts_matrix) - Two numpy uint32 matrices of shape (num_texts, ndim).\n"
    "\n"
    "Examples:\n"
    "  ```python\n"
    "  # Minimal CPU example with auto-inferred capabilities\n"
    "  import stringzilla as sz, stringzillas as szs\n"
    "  engine = szs.Fingerprints(ndim=128)\n"
    "  docs = sz.Strs(['document one', 'document two', 'document three'])\n"
    "  hashes, counts = engine(docs)\n"
    "  \n"
    "  # GPU example with custom dimensions\n"
    "  gpu_scope = szs.DeviceScope(gpu_device=0)\n"
    "  engine = szs.Fingerprints(ndim=256, capabilities=gpu_scope)\n"
    "  hashes, counts = engine(docs, device=gpu_scope)\n"
    "  ```";

static PyGetSetDef Fingerprints_getsetters[] = {
    {"capabilities", (getter)Fingerprints_get_capabilities, NULL, "computational capabilities", NULL},
    {NULL} /* Sentinel */
};

static PyTypeObject FingerprintsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.Fingerprints",
    .tp_doc = doc_Fingerprints,
    .tp_basicsize = sizeof(Fingerprints),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Fingerprints_new,
    .tp_init = (initproc)Fingerprints_init,
    .tp_getset = Fingerprints_getsetters,
    .tp_repr = (reprfunc)Fingerprints_repr,
    .tp_dealloc = (destructor)Fingerprints_dealloc,
    .tp_call = (ternaryfunc)Fingerprints_call,
};

#pragma endregion

static void stringzillas_cleanup(PyObject *m) {
    sz_unused_(m);
    if (default_device_scope) {
        szs_device_scope_free(default_device_scope);
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
    // Initialize default CPU allocator
    sz_memory_allocator_init_default(&default_allocator);

    // Initialize the default device scope for reuse
    sz_status_t status = szs_device_scope_init_default(&default_device_scope);
    if (status != sz_success_k) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize default device scope");
        return NULL;
    }

    if (PyType_Ready(&DeviceScopeType) < 0) return NULL;
    if (PyType_Ready(&LevenshteinDistancesType) < 0) return NULL;
    if (PyType_Ready(&LevenshteinDistancesUTF8Type) < 0) return NULL;
    if (PyType_Ready(&NeedlemanWunschType) < 0) return NULL;
    if (PyType_Ready(&SmithWatermanType) < 0) return NULL;
    if (PyType_Ready(&FingerprintsType) < 0) return NULL;

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
        // Create a Python tuple with the capabilities
        sz_capability_t caps = default_hardware_capabilities;
        PyObject *caps_tuple = capabilities_to_tuple(caps);
        if (!caps_tuple) {
            Py_XDECREF(m);
            return NULL;
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

    Py_INCREF(&FingerprintsType);
    if (PyModule_AddObject(m, "Fingerprints", (PyObject *)&FingerprintsType) < 0) {
        Py_XDECREF(&FingerprintsType);
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
