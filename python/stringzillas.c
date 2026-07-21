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

/**
 * @brief Set appropriate Python exception based on StringZilla status code and error detail.
 * @param[in] status The StringZilla status code
 * @param[in] error_detail Detailed error message from StringZilla (never NULL)
 * @param[in] context Context string describing the operation (e.g., "Levenshtein initialization")
 */
static void set_stringzilla_error(sz_status_t status, char const *error_detail, char const *context) {
    switch (status) {
    case sz_bad_alloc_k: PyErr_Format(PyExc_MemoryError, "%s: %s", context, error_detail); break;
    case sz_invalid_utf8_k: PyErr_Format(PyExc_ValueError, "%s: %s", context, error_detail); break;
    case sz_overflow_risk_k: PyErr_Format(PyExc_OverflowError, "%s: %s", context, error_detail); break;
    case sz_unexpected_dimensions_k: PyErr_Format(PyExc_ValueError, "%s: %s", context, error_detail); break;
    case sz_missing_gpu_k:
    case sz_device_code_mismatch_k:
    case sz_device_memory_mismatch_k:
    default: PyErr_Format(PyExc_RuntimeError, "%s: %s", context, error_detail); break;
    }
}

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
SZ_HELPER_AUTO sz_bool_t try_swap_to_unified_allocator(PyObject *strs_obj) {
    if (!strs_obj || !sz_py_replace_strings_allocator) return sz_false_k;

    // Try to swap to unified allocator - this will be a no-op if already using it
    sz_bool_t success = sz_py_replace_strings_allocator(strs_obj, &unified_allocator);

    if (!success) {
        // Always fatal: GPU kernels require unified/device-accessible memory
        PyErr_SetString( //
            PyExc_RuntimeError,
            "Device memory mismatch: GPU kernels require unified/device-accessible memory. " //
            "Consider reducing input size, freeing memory, or using CPU capabilities.");
        return sz_false_k;
    }
    return sz_true_k;
}

/**
 *  @brief Helper function to determine if unified memory is required based on capabilities and device scope.
 *  @param[in] capabilities The capabilities bitmask of the current engine.
 */
SZ_HELPER_AUTO sz_bool_t requires_unified_memory(sz_capability_t capabilities) {
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
    PyObject *cpu_cores_obj = NULL;
    PyObject *gpu_device_obj = NULL;

    static char *kwlist[] = {"cpu_cores", "gpu_device", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|OO", kwlist, &cpu_cores_obj, &gpu_device_obj)) return -1;

    sz_status_t status;
    char const *error_detail = NULL;

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
        status = szs_device_scope_init_cpu_cores(cpu_cores, &self->handle, &error_detail);
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
        status = szs_device_scope_init_gpu_device(gpu_device, &self->handle, &error_detail);
        snprintf(self->description, sizeof(self->description), "GPU:%zu", gpu_device);
    }
    else {
        status = szs_device_scope_init_default(&self->handle, &error_detail);
        snprintf(self->description, sizeof(self->description), "default");
    }

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "DeviceScope initialization");
        return -1;
    }

    return 0;
}

static PyObject *DeviceScope_repr(DeviceScope *self) {
    return PyUnicode_FromFormat("DeviceScope(%s)", self->description);
}

static char const doc_DeviceScope[] =                                                   //
    "DeviceScope(cpu_cores=None, gpu_device=None)\n"                                    //
    "\n"                                                                                //
    "Context for controlling execution on CPU cores or GPU devices.\n"                  //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "  cpu_cores (int, optional): Number of CPU cores to use, or zero for all cores.\n" //
    "  gpu_device (int, optional): GPU device ID to target.\n"                          //
    "\n"                                                                                //
    "Note: Cannot specify both cpu_cores and gpu_device.\n"                             //
    "\n"                                                                                //
    "Examples:\n"                                                                       //
    "  >>> import stringzillas as szs\n"                                                //
    "  >>> scope = szs.DeviceScope(cpu_cores=4)  # restrict engines to 4 CPU cores";

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
        char const *error_detail_gpu = NULL;
        if (szs_device_scope_get_gpu_device(device_scope->handle, &gpu_device, &error_detail_gpu) == sz_success_k) {
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
        char const *error_detail_cpu = NULL;
        if (szs_device_scope_get_cpu_cores(device_scope->handle, &cpu_cores, &error_detail_cpu) == sz_success_k) {
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

        sz_capability_t flag = sz_capability_from_string_implementation_(cap_str);
        if (flag == sz_caps_none_k) {
            PyErr_Format(PyExc_ValueError, "Unknown capability: %s", cap_str);
            return -1;
        }
        requested_caps |= flag;
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
    vectorcallfunc vectorcall;
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

static PyObject *LevenshteinDistances_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                                 PyObject *kwnames);

static PyObject *LevenshteinDistances_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    LevenshteinDistances *self = (LevenshteinDistances *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->vectorcall = (vectorcallfunc)LevenshteinDistances_vectorcall;
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

    // Manual positional + keyword parse (no `PyArg_ParseTupleAndKeywords`, no generic binder).
    char const *const callable_name = Py_TYPE(self)->tp_name;
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count > 5) {
        PyErr_Format(PyExc_TypeError, "%s takes at most 5 arguments (%zd given)", callable_name, positional_count);
        return -1;
    }
    PyObject *match_obj = positional_count > 0 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *mismatch_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *open_obj = positional_count > 2 ? PyTuple_GET_ITEM(args, 2) : NULL;
    PyObject *extend_obj = positional_count > 3 ? PyTuple_GET_ITEM(args, 3) : NULL;
    capabilities_tuple = positional_count > 4 ? PyTuple_GET_ITEM(args, 4) : NULL;
    if (kwargs != NULL) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "match") == 0) {
                if (match_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'match'", callable_name);
                    return -1;
                }
                match_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "mismatch") == 0) {
                if (mismatch_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'mismatch'", callable_name);
                    return -1;
                }
                mismatch_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "open") == 0) {
                if (open_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'open'", callable_name);
                    return -1;
                }
                open_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "extend") == 0) {
                if (extend_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'extend'", callable_name);
                    return -1;
                }
                extend_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                if (capabilities_tuple) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'capabilities'", callable_name);
                    return -1;
                }
                capabilities_tuple = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }
    if (match_obj) {
        match = (int)PyLong_AsLong(match_obj);
        if (PyErr_Occurred()) return -1;
    }
    if (mismatch_obj) {
        mismatch = (int)PyLong_AsLong(mismatch_obj);
        if (PyErr_Occurred()) return -1;
    }
    if (open_obj) {
        open = (int)PyLong_AsLong(open_obj);
        if (PyErr_Occurred()) return -1;
    }
    if (extend_obj) {
        extend = (int)PyLong_AsLong(extend_obj);
        if (PyErr_Occurred()) return -1;
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

    char const *error_detail = NULL;
    sz_status_t status = szs_levenshtein_distances_init(match, mismatch, open, extend, NULL, capabilities,
                                                        &self->handle, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Levenshtein distances initialization");
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "%d,%d,%d,%d", match, mismatch, open, extend);
    self->capabilities = capabilities;
    return 0;
}

/**
 *  @brief Manual fast parser for the cross-product `__call__(queries, candidates=None, device=None, out=None)`.
 *
 *  Mirrors the keyword-walking convention used throughout the sibling `stringzilla` C module: the engines are
 *  invoked through the vectorcall fast path, so positionals arrive in the flat `args` array and keywords are
 *  named by the trailing `kwnames` tuple (their values continue past the positionals in `args`). Rejects
 *  unknown keys and positional/keyword collisions. Returns 0 on success, -1 with a raised exception otherwise.
 *  Outputs are left untouched on failure.
 */
static int parse_cross_product_call_args(                                               //
    char const *callable_name, PyObject *const *args, size_t nargsf, PyObject *kwnames, //
    PyObject **queries_out, PyObject **candidates_out, PyObject **device_out, PyObject **out_out) {

    Py_ssize_t const positional_args_count = PyVectorcall_NARGS(nargsf);
    if (positional_args_count < 1 || positional_args_count > 4) {
        PyErr_Format(PyExc_TypeError, "%s() takes 1 to 4 positional arguments, got %zd", callable_name,
                     positional_args_count);
        return -1;
    }

    PyObject *queries_obj = args[0];
    PyObject *candidates_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *device_obj = positional_args_count > 2 ? args[2] : NULL;
    PyObject *out_obj = positional_args_count > 3 ? args[3] : NULL;

    if (kwnames != NULL) {
        Py_ssize_t const keyword_count = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t keyword_index = 0; keyword_index < keyword_count; ++keyword_index) {
            PyObject *const key = PyTuple_GET_ITEM(kwnames, keyword_index);
            PyObject *const value = args[positional_args_count + keyword_index];
            if (PyUnicode_CompareWithASCIIString(key, "queries") == 0) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'queries'", callable_name);
                return -1;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "candidates") == 0) {
                if (positional_args_count > 1) {
                    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'candidates'", callable_name);
                    return -1;
                }
                candidates_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "device") == 0) {
                if (positional_args_count > 2) {
                    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'device'", callable_name);
                    return -1;
                }
                device_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
                if (positional_args_count > 3) {
                    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'out'", callable_name);
                    return -1;
                }
                out_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }

    *queries_out = queries_obj;
    *candidates_out = candidates_obj;
    *device_out = device_obj;
    *out_out = out_obj;
    return 0;
}

static PyObject *LevenshteinDistances_repr(LevenshteinDistances *self) {
    return PyUnicode_FromFormat("LevenshteinDistances(match,mismatch,open,extend=%s)", self->description);
}

static PyObject *LevenshteinDistances_get_capabilities(LevenshteinDistances *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *LevenshteinDistances_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                                 PyObject *kwnames) {
    LevenshteinDistances *self = (LevenshteinDistances *)callable;
    PyObject *queries_obj = NULL, *candidates_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    if (parse_cross_product_call_args("LevenshteinDistances.__call__", args, nargsf, kwnames, &queries_obj,
                                      &candidates_obj, &device_obj, &out_obj) != 0)
        return NULL;

    // Treat an explicit `None` for `candidates` as "compute symmetric self-similarity of queries".
    if (candidates_obj == Py_None) candidates_obj = NULL;
    sz_bool_t is_self_similarity = (candidates_obj == NULL) ? sz_true_k : sz_false_k;

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    szs_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t queries_count = 0;
    sz_size_t candidates_count = 0;
    void *kernel_queries_punned = NULL;
    void *kernel_candidates_punned = NULL;
    sz_size_t *kernel_results = NULL;
    sz_size_t kernel_results_row_stride = 0;
    sz_status_t (*kernel_punned)(szs_levenshtein_distances_t, szs_device_scope_t, void *, void *, sz_size_t *,
                                 sz_size_t, char const **) = NULL;

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    if (requires_unified_memory(self->capabilities)) {
        if (!try_swap_to_unified_allocator(queries_obj)) return NULL;
        if (candidates_obj && !try_swap_to_unified_allocator(candidates_obj)) return NULL;
    }

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t queries_u32tape, candidates_u32tape;
    sz_bool_t queries_is_u32tape = sz_py_export_strings_as_u32tape( //
        queries_obj, &queries_u32tape.data, &queries_u32tape.offsets, &queries_u32tape.count);
    sz_bool_t candidates_is_u32tape = candidates_obj && sz_py_export_strings_as_u32tape( //
                                                            candidates_obj, &candidates_u32tape.data,
                                                            &candidates_u32tape.offsets, &candidates_u32tape.count);
    if (queries_is_u32tape && (is_self_similarity || candidates_is_u32tape)) {
        queries_count = queries_u32tape.count;
        candidates_count = is_self_similarity ? queries_u32tape.count : candidates_u32tape.count;
        kernel_punned = szs_levenshtein_distances_u32tape;
        kernel_queries_punned = &queries_u32tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t queries_u64tape, candidates_u64tape;
    sz_bool_t queries_is_u64tape = !queries_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                              queries_obj, &queries_u64tape.data,
                                                              &queries_u64tape.offsets, &queries_u64tape.count);
    sz_bool_t candidates_is_u64tape = candidates_obj && !candidates_is_u32tape &&
                                      sz_py_export_strings_as_u64tape( //
                                          candidates_obj, &candidates_u64tape.data, &candidates_u64tape.offsets,
                                          &candidates_u64tape.count);
    if (!kernel_punned && queries_is_u64tape && (is_self_similarity || candidates_is_u64tape)) {
        queries_count = queries_u64tape.count;
        candidates_count = is_self_similarity ? queries_u64tape.count : candidates_u64tape.count;
        kernel_punned = szs_levenshtein_distances_u64tape;
        kernel_queries_punned = &queries_u64tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u64tape;
    }

    // Handle sequence inputs
    sz_sequence_t queries_seq, candidates_seq;
    sz_bool_t queries_is_sequence = !queries_is_u32tape && !queries_is_u64tape &&
                                    sz_py_export_strings_as_sequence(queries_obj, &queries_seq);
    sz_bool_t candidates_is_sequence = candidates_obj && !candidates_is_u32tape && !candidates_is_u64tape &&
                                       sz_py_export_strings_as_sequence(candidates_obj, &candidates_seq);
    if (!kernel_punned && queries_is_sequence && (is_self_similarity || candidates_is_sequence)) {
        queries_count = queries_seq.count;
        candidates_count = is_self_similarity ? queries_seq.count : candidates_seq.count;
        kernel_punned = szs_levenshtein_distances;
        kernel_queries_punned = &queries_seq;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_seq;
    }

    // No homogeneous kernel matched. This happens when one side is empty: an empty Strs is always
    // FRAGMENTED and cannot pair with a non-empty side's tape above. If both sides are recognized Strs
    // and either is empty, the cross product is an empty matrix -- record the counts and fall through
    // with a NULL kernel (the call below is skipped and the empty matrix returned).
    if (!kernel_punned) {
        sz_bool_t queries_recognized = queries_is_u32tape || queries_is_u64tape || queries_is_sequence;
        sz_bool_t candidates_recognized = is_self_similarity || candidates_is_u32tape || candidates_is_u64tape ||
                                          candidates_is_sequence;
        sz_size_t queries_any_count = queries_is_u32tape    ? queries_u32tape.count
                                      : queries_is_u64tape  ? queries_u64tape.count
                                      : queries_is_sequence ? queries_seq.count
                                                            : 0;
        sz_size_t candidates_any_count = is_self_similarity       ? queries_any_count
                                         : candidates_is_u32tape  ? candidates_u32tape.count
                                         : candidates_is_u64tape  ? candidates_u64tape.count
                                         : candidates_is_sequence ? candidates_seq.count
                                                                  : 0;
        if (!(queries_recognized && candidates_recognized && (queries_any_count == 0 || candidates_any_count == 0))) {
            PyErr_Format( //
                PyExc_TypeError,
                "Expected stringzilla.Strs objects, got %s and %s. " //
                "Convert using: stringzilla.Strs(your_string_list)",
                Py_TYPE(queries_obj)->tp_name, candidates_obj ? Py_TYPE(candidates_obj)->tp_name : "None");
            return NULL;
        }
        queries_count = queries_any_count;
        candidates_count = candidates_any_count;
    }

    // Allocate a fresh 2-D matrix or validate the provided `out` array, deriving the row stride in ELEMENTS.
    PyObject *results_array = NULL;
    if (!out_obj || out_obj == Py_None) {
        npy_intp results_shape[2] = {(npy_intp)queries_count, (npy_intp)candidates_count};
        results_array = PyArray_SimpleNew(2, results_shape, NPY_UINT64);
        if (!results_array) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array for results");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_row_stride = candidates_count;
    }
    else {
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out argument must be a NumPy array");
            goto cleanup;
        }
        PyArrayObject *array = (PyArrayObject *)out_obj;
        if (PyArray_NDIM(array) != 2) {
            PyErr_SetString(PyExc_ValueError, "out array must be 2-dimensional");
            goto cleanup;
        }
        if (PyArray_DIM(array, 0) < (npy_intp)queries_count || PyArray_DIM(array, 1) < (npy_intp)candidates_count) {
            PyErr_SetString(PyExc_ValueError, "out array is too small for results");
            goto cleanup;
        }
        if (PyArray_TYPE(array) != NPY_UINT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have uint64 dtype");
            goto cleanup;
        }
        // Row stride is expressed in elements; the C ABI does not accept padded columns within a row.
        if (PyArray_STRIDE(array, 1) != (npy_intp)sizeof(sz_size_t)) {
            PyErr_SetString(PyExc_ValueError, "out array rows must be contiguous (unit stride along columns)");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA(array);
        kernel_results_row_stride = (sz_size_t)(PyArray_STRIDE(array, 0) / (npy_intp)sizeof(sz_size_t));
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    char const *error_detail = NULL;
    sz_status_t status = sz_success_k; // An empty cross product (zero-row/col matrix) needs no kernel
    if (kernel_punned)
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Levenshtein distances computation");
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_LevenshteinDistances[] =                                                               //
    "LevenshteinDistances(match=0, mismatch=1, open=1, extend=1, capabilities=None)\n"                       //
    "\n"                                                                                                     //
    "Compute the cross-product matrix of Levenshtein edit distances between two string collections.\n"       //
    "\n"                                                                                                     //
    "Args:\n"                                                                                                //
    "  match (int): Cost for matching characters (default: 0).\n"                                            //
    "  mismatch (int): Cost for mismatched characters (default: 1).\n"                                       //
    "  open (int): Cost for opening a gap (default: 1).\n"                                                   //
    "  extend (int): Cost for extending a gap (default: 1).\n"                                               //
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"                  //
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"      //
    "                                       or a DeviceScope for automatic capability inference.\n"          //
    "\n"                                                                                                     //
    "Call with:\n"                                                                                           //
    "  queries (sequence): Query strings forming the matrix rows.\n"                                         //
    "  candidates (sequence, optional): Candidate strings forming the matrix columns. When omitted\n"        //
    "                                   (or None), computes the symmetric self-similarity of queries.\n"     //
    "  device (DeviceScope, optional): Device execution context.\n"                                          //
    "  out (np.ndarray, optional): 2-D uint64 output buffer of shape (len(queries), len(candidates)).\n"     //
    "\n"                                                                                                     //
    "Returns:\n"                                                                                             //
    "  np.ndarray: 2-D uint64 matrix where result[query_index, candidate_index] is the distance\n"           //
    "              between queries[query_index] and candidates[candidate_index].\n"                          //
    "\n"                                                                                                     //
    "Examples:\n"                                                                                            //
    "  >>> # Minimal CPU example with auto-inferred capabilities\n"                                          //
    "  >>> import stringzilla as sz, stringzillas as szs\n"                                                  //
    "  >>> engine = szs.LevenshteinDistances()\n"                                                            //
    "  >>> strings_a = sz.Strs(['hello', 'world'])\n"                                                        //
    "  >>> strings_b = sz.Strs(['hallo', 'word'])\n"                                                         //
    "  >>> distances = engine(strings_a, strings_b)\n"                                                       //
    "  >>> # GPU example with custom costs; falls back to CPU when CUDA is unavailable\n"                    //
    "  >>> scope = szs.DeviceScope(gpu_device=0) if 'cuda' in szs.__capabilities__ else szs.DeviceScope()\n" //
    "  >>> match, mismatch, gap_open, gap_extend = 0, 2, 3, 1\n"                                             //
    "  >>> engine = szs.LevenshteinDistances(match, mismatch, gap_open, gap_extend, scope)\n"                //
    "  >>> distances = engine(strings_a, strings_b, device=scope)";
static char const doc_capabilities[] =                                          //
    "Hardware backends and SIMD capabilities this engine selects at runtime.\n" //
    "\n"                                                                        //
    "Returns:\n"                                                                //
    "  str: The detected CPU/GPU features driving kernel dispatch.";

static PyGetSetDef LevenshteinDistances_getsetters[] = {
    {"__capabilities__", (getter)LevenshteinDistances_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                             /* Sentinel */
};

static PyTypeObject LevenshteinDistancesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.LevenshteinDistances",
    .tp_doc = doc_LevenshteinDistances,
    .tp_basicsize = sizeof(LevenshteinDistances),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(LevenshteinDistances, vectorcall),
    .tp_new = LevenshteinDistances_new,
    .tp_init = (initproc)LevenshteinDistances_init,
    .tp_dealloc = (destructor)LevenshteinDistances_dealloc,
    .tp_call = PyVectorcall_Call,
    .tp_repr = (reprfunc)LevenshteinDistances_repr,
    .tp_getset = LevenshteinDistances_getsetters,
};

#pragma endregion

#pragma region LevenshteinDistancesUTF8

typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
    szs_levenshtein_distances_utf8_t handle;
    char description[32];
    sz_capability_t capabilities;
} LevenshteinDistancesUTF8;

static PyObject *LevenshteinDistancesUTF8_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames);

static PyObject *LevenshteinDistancesUTF8_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    LevenshteinDistancesUTF8 *self = (LevenshteinDistancesUTF8 *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->vectorcall = (vectorcallfunc)LevenshteinDistancesUTF8_vectorcall;
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

    // Manual positional + keyword parse (no `PyArg_ParseTupleAndKeywords`, no generic binder).
    char const *const callable_name = Py_TYPE(self)->tp_name;
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count > 5) {
        PyErr_Format(PyExc_TypeError, "%s takes at most 5 arguments (%zd given)", callable_name, positional_count);
        return -1;
    }
    PyObject *match_obj = positional_count > 0 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *mismatch_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *open_obj = positional_count > 2 ? PyTuple_GET_ITEM(args, 2) : NULL;
    PyObject *extend_obj = positional_count > 3 ? PyTuple_GET_ITEM(args, 3) : NULL;
    capabilities_tuple = positional_count > 4 ? PyTuple_GET_ITEM(args, 4) : NULL;
    if (kwargs != NULL) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "match") == 0) {
                if (match_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'match'", callable_name);
                    return -1;
                }
                match_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "mismatch") == 0) {
                if (mismatch_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'mismatch'", callable_name);
                    return -1;
                }
                mismatch_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "open") == 0) {
                if (open_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'open'", callable_name);
                    return -1;
                }
                open_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "extend") == 0) {
                if (extend_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'extend'", callable_name);
                    return -1;
                }
                extend_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                if (capabilities_tuple) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'capabilities'", callable_name);
                    return -1;
                }
                capabilities_tuple = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }
    if (match_obj) {
        match = (int)PyLong_AsLong(match_obj);
        if (PyErr_Occurred()) return -1;
    }
    if (mismatch_obj) {
        mismatch = (int)PyLong_AsLong(mismatch_obj);
        if (PyErr_Occurred()) return -1;
    }
    if (open_obj) {
        open = (int)PyLong_AsLong(open_obj);
        if (PyErr_Occurred()) return -1;
    }
    if (extend_obj) {
        extend = (int)PyLong_AsLong(extend_obj);
        if (PyErr_Occurred()) return -1;
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

    char const *error_detail = NULL;
    sz_status_t status = szs_levenshtein_distances_utf8_init(match, mismatch, open, extend, NULL, capabilities,
                                                             &self->handle, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "UTF-8 Levenshtein distances initialization");
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

static PyObject *LevenshteinDistancesUTF8_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                                     PyObject *kwnames) {
    LevenshteinDistancesUTF8 *self = (LevenshteinDistancesUTF8 *)callable;
    PyObject *queries_obj = NULL, *candidates_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    if (parse_cross_product_call_args("LevenshteinDistancesUTF8.__call__", args, nargsf, kwnames, &queries_obj,
                                      &candidates_obj, &device_obj, &out_obj) != 0)
        return NULL;

    // Treat an explicit `None` for `candidates` as "compute symmetric self-similarity of queries".
    if (candidates_obj == Py_None) candidates_obj = NULL;
    sz_bool_t is_self_similarity = (candidates_obj == NULL) ? sz_true_k : sz_false_k;

    DeviceScope *device_scope = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    szs_device_scope_t device_handle = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t queries_count = 0;
    sz_size_t candidates_count = 0;
    void *kernel_queries_punned = NULL;
    void *kernel_candidates_punned = NULL;
    sz_size_t *kernel_results = NULL;
    sz_size_t kernel_results_row_stride = 0;
    sz_status_t (*kernel_punned)(szs_levenshtein_distances_utf8_t, szs_device_scope_t, void *, void *, sz_size_t *,
                                 sz_size_t, char const **) = NULL;

    // Swap allocators when engine supports CUDA
    if (requires_unified_memory(self->capabilities)) {
        if (!try_swap_to_unified_allocator(queries_obj)) return NULL;
        if (candidates_obj && !try_swap_to_unified_allocator(candidates_obj)) return NULL;
    }

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t queries_u32tape, candidates_u32tape;
    sz_bool_t queries_is_u32tape = sz_py_export_strings_as_u32tape( //
        queries_obj, &queries_u32tape.data, &queries_u32tape.offsets, &queries_u32tape.count);
    sz_bool_t candidates_is_u32tape = candidates_obj && sz_py_export_strings_as_u32tape( //
                                                            candidates_obj, &candidates_u32tape.data,
                                                            &candidates_u32tape.offsets, &candidates_u32tape.count);
    if (queries_is_u32tape && (is_self_similarity || candidates_is_u32tape)) {
        queries_count = queries_u32tape.count;
        candidates_count = is_self_similarity ? queries_u32tape.count : candidates_u32tape.count;
        kernel_punned = szs_levenshtein_distances_utf8_u32tape;
        kernel_queries_punned = &queries_u32tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t queries_u64tape, candidates_u64tape;
    sz_bool_t queries_is_u64tape = !queries_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                              queries_obj, &queries_u64tape.data,
                                                              &queries_u64tape.offsets, &queries_u64tape.count);
    sz_bool_t candidates_is_u64tape = candidates_obj && !candidates_is_u32tape &&
                                      sz_py_export_strings_as_u64tape( //
                                          candidates_obj, &candidates_u64tape.data, &candidates_u64tape.offsets,
                                          &candidates_u64tape.count);
    if (!kernel_punned && queries_is_u64tape && (is_self_similarity || candidates_is_u64tape)) {
        queries_count = queries_u64tape.count;
        candidates_count = is_self_similarity ? queries_u64tape.count : candidates_u64tape.count;
        kernel_punned = szs_levenshtein_distances_utf8_u64tape;
        kernel_queries_punned = &queries_u64tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u64tape;
    }

    // Handle sequence inputs
    sz_sequence_t queries_seq, candidates_seq;
    sz_bool_t queries_is_sequence = !queries_is_u32tape && !queries_is_u64tape &&
                                    sz_py_export_strings_as_sequence(queries_obj, &queries_seq);
    sz_bool_t candidates_is_sequence = candidates_obj && !candidates_is_u32tape && !candidates_is_u64tape &&
                                       sz_py_export_strings_as_sequence(candidates_obj, &candidates_seq);
    if (!kernel_punned && queries_is_sequence && (is_self_similarity || candidates_is_sequence)) {
        queries_count = queries_seq.count;
        candidates_count = is_self_similarity ? queries_seq.count : candidates_seq.count;
        kernel_punned = szs_levenshtein_distances_utf8;
        kernel_queries_punned = &queries_seq;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_seq;
    }

    // No homogeneous kernel matched. This happens when one side is empty: an empty Strs is always
    // FRAGMENTED and cannot pair with a non-empty side's tape above. If both sides are recognized Strs
    // and either is empty, the cross product is an empty matrix -- record the counts and fall through
    // with a NULL kernel (the call below is skipped and the empty matrix returned).
    if (!kernel_punned) {
        sz_bool_t queries_recognized = queries_is_u32tape || queries_is_u64tape || queries_is_sequence;
        sz_bool_t candidates_recognized = is_self_similarity || candidates_is_u32tape || candidates_is_u64tape ||
                                          candidates_is_sequence;
        sz_size_t queries_any_count = queries_is_u32tape    ? queries_u32tape.count
                                      : queries_is_u64tape  ? queries_u64tape.count
                                      : queries_is_sequence ? queries_seq.count
                                                            : 0;
        sz_size_t candidates_any_count = is_self_similarity       ? queries_any_count
                                         : candidates_is_u32tape  ? candidates_u32tape.count
                                         : candidates_is_u64tape  ? candidates_u64tape.count
                                         : candidates_is_sequence ? candidates_seq.count
                                                                  : 0;
        if (!(queries_recognized && candidates_recognized && (queries_any_count == 0 || candidates_any_count == 0))) {
            PyErr_Format( //
                PyExc_TypeError,
                "Expected stringzilla.Strs objects, got %s and %s. " //
                "Convert using: stringzilla.Strs(your_string_list)",
                Py_TYPE(queries_obj)->tp_name, candidates_obj ? Py_TYPE(candidates_obj)->tp_name : "None");
            return NULL;
        }
        queries_count = queries_any_count;
        candidates_count = candidates_any_count;
    }

    // Allocate a fresh 2-D matrix or validate the provided `out` array, deriving the row stride in ELEMENTS.
    PyObject *results_array = NULL;
    if (!out_obj || out_obj == Py_None) {
        npy_intp results_shape[2] = {(npy_intp)queries_count, (npy_intp)candidates_count};
        results_array = PyArray_SimpleNew(2, results_shape, NPY_UINT64);
        if (!results_array) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array for results");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_row_stride = candidates_count;
    }
    else {
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out argument must be a NumPy array");
            goto cleanup;
        }
        PyArrayObject *array = (PyArrayObject *)out_obj;
        if (PyArray_NDIM(array) != 2) {
            PyErr_SetString(PyExc_ValueError, "out array must be 2-dimensional");
            goto cleanup;
        }
        if (PyArray_DIM(array, 0) < (npy_intp)queries_count || PyArray_DIM(array, 1) < (npy_intp)candidates_count) {
            PyErr_SetString(PyExc_ValueError, "out array is too small for results");
            goto cleanup;
        }
        if (PyArray_TYPE(array) != NPY_UINT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have uint64 dtype");
            goto cleanup;
        }
        if (PyArray_STRIDE(array, 1) != (npy_intp)sizeof(sz_size_t)) {
            PyErr_SetString(PyExc_ValueError, "out array rows must be contiguous (unit stride along columns)");
            goto cleanup;
        }
        kernel_results = (sz_size_t *)PyArray_DATA(array);
        kernel_results_row_stride = (sz_size_t)(PyArray_STRIDE(array, 0) / (npy_intp)sizeof(sz_size_t));
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    char const *error_detail = NULL;
    sz_status_t status = sz_success_k; // An empty cross product (zero-row/col matrix) needs no kernel
    if (kernel_punned)
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Levenshtein distances computation");
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_LevenshteinDistancesUTF8[] =                                                           //
    "LevenshteinDistancesUTF8(match=0, mismatch=1, open=1, extend=1, capabilities=None)\n"                   //
    "\n"                                                                                                     //
    "Vectorized UTF-8 Levenshtein distance calculator with affine gap penalties.\n"                          //
    "Computes the cross-product matrix of edit distances between two UTF-8 string collections.\n"            //
    "\n"                                                                                                     //
    "Args:\n"                                                                                                //
    "  match (int): Cost of matching characters (default 0).\n"                                              //
    "  mismatch (int): Cost of mismatched characters (default 1).\n"                                         //
    "  open (int): Cost of opening a gap (default 1).\n"                                                     //
    "  extend (int): Cost of extending a gap (default 1).\n"                                                 //
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"                  //
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"      //
    "                                       or a DeviceScope for automatic capability inference.\n"          //
    "\n"                                                                                                     //
    "Call with:\n"                                                                                           //
    "  queries (sequence): Query UTF-8 strings forming the matrix rows.\n"                                   //
    "  candidates (sequence, optional): Candidate UTF-8 strings forming the matrix columns. When\n"          //
    "                                   omitted (or None), computes symmetric self-similarity of queries.\n" //
    "  device (DeviceScope, optional): Device execution context.\n"                                          //
    "  out (np.ndarray, optional): 2-D uint64 output buffer of shape (len(queries), len(candidates)).\n"     //
    "\n"                                                                                                     //
    "Returns:\n"                                                                                             //
    "  np.ndarray: 2-D uint64 matrix where result[query_index, candidate_index] is the distance\n"           //
    "              between queries[query_index] and candidates[candidate_index].\n"                          //
    "\n"                                                                                                     //
    "Examples:\n"                                                                                            //
    "  >>> # Minimal CPU example with Unicode strings\n"                                                     //
    "  >>> import stringzilla as sz, stringzillas as szs\n"                                                  //
    "  >>> engine = szs.LevenshteinDistancesUTF8()\n"                                                        //
    "  >>> strings_a = sz.Strs(['café', 'naïve'])\n"                                                         //
    "  >>> strings_b = sz.Strs(['caffe', 'naive'])\n"                                                        //
    "  >>> distances = engine(strings_a, strings_b)\n"                                                       //
    "  >>> # GPU example with high mismatch penalty; falls back to CPU when CUDA is unavailable\n"           //
    "  >>> scope = szs.DeviceScope(gpu_device=0) if 'cuda' in szs.__capabilities__ else szs.DeviceScope()\n" //
    "  >>> engine = szs.LevenshteinDistancesUTF8(mismatch=5, capabilities=scope)\n"                          //
    "  >>> distances = engine(strings_a, strings_b, device=scope)";
static PyGetSetDef LevenshteinDistancesUTF8_getsetters[] = {
    {"__capabilities__", (getter)LevenshteinDistancesUTF8_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL} /* Sentinel */
};

static PyTypeObject LevenshteinDistancesUTF8Type = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.LevenshteinDistancesUTF8",
    .tp_doc = doc_LevenshteinDistancesUTF8,
    .tp_basicsize = sizeof(LevenshteinDistancesUTF8),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(LevenshteinDistancesUTF8, vectorcall),
    .tp_new = LevenshteinDistancesUTF8_new,
    .tp_init = (initproc)LevenshteinDistancesUTF8_init,
    .tp_dealloc = (destructor)LevenshteinDistancesUTF8_dealloc,
    .tp_call = PyVectorcall_Call,
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
    vectorcallfunc vectorcall;
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

static PyObject *NeedlemanWunsch_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                            PyObject *kwnames);

static PyObject *NeedlemanWunsch_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    NeedlemanWunsch *self = (NeedlemanWunsch *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->vectorcall = (vectorcallfunc)NeedlemanWunsch_vectorcall;
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
    }
    return (PyObject *)self;
}

static int NeedlemanWunsch_init(NeedlemanWunsch *self, PyObject *args, PyObject *kwargs) {
    PyObject *byte_to_class_obj = NULL;
    PyObject *class_substitution_costs_obj = NULL;
    sz_error_cost_t open = -1, extend = -1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    // Manual positional + keyword parse (no `PyArg_ParseTupleAndKeywords`, no generic binder).
    // Arguments: byte_to_class, class_substitution_costs, open, extend, capabilities.
    char const *const callable_name = Py_TYPE(self)->tp_name;
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count > 5) {
        PyErr_Format(PyExc_TypeError, "%s takes at most 5 arguments (%zd given)", callable_name, positional_count);
        return -1;
    }
    byte_to_class_obj = positional_count > 0 ? PyTuple_GET_ITEM(args, 0) : NULL;
    class_substitution_costs_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *open_obj = positional_count > 2 ? PyTuple_GET_ITEM(args, 2) : NULL;
    PyObject *extend_obj = positional_count > 3 ? PyTuple_GET_ITEM(args, 3) : NULL;
    capabilities_tuple = positional_count > 4 ? PyTuple_GET_ITEM(args, 4) : NULL;
    if (kwargs != NULL) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "byte_to_class") == 0) {
                if (byte_to_class_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'byte_to_class'", callable_name);
                    return -1;
                }
                byte_to_class_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "class_substitution_costs") == 0) {
                if (class_substitution_costs_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'class_substitution_costs'",
                                 callable_name);
                    return -1;
                }
                class_substitution_costs_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "open") == 0) {
                if (open_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'open'", callable_name);
                    return -1;
                }
                open_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "extend") == 0) {
                if (extend_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'extend'", callable_name);
                    return -1;
                }
                extend_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                if (capabilities_tuple) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'capabilities'", callable_name);
                    return -1;
                }
                capabilities_tuple = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }
    if (open_obj) {
        long open_value = PyLong_AsLong(open_obj);
        if (PyErr_Occurred()) return -1;
        open = (sz_error_cost_t)open_value;
    }
    if (extend_obj) {
        long extend_value = PyLong_AsLong(extend_obj);
        if (PyErr_Occurred()) return -1;
        extend = (sz_error_cost_t)extend_value;
    }
    if (byte_to_class_obj == NULL || class_substitution_costs_obj == NULL) {
        PyErr_Format(PyExc_TypeError, "%s requires 'byte_to_class' and 'class_substitution_costs'",
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    // Validate byte-to-class map (should be a 256-element uint8 numpy array)
    if (!numpy_available || !PyArray_Check(byte_to_class_obj)) {
        PyErr_SetString(PyExc_TypeError, "byte_to_class must be a NumPy array");
        return -1;
    }
    PyArrayObject *byte_to_class_array = (PyArrayObject *)byte_to_class_obj;
    if (PyArray_NDIM(byte_to_class_array) != 1 || PyArray_DIM(byte_to_class_array, 0) != 256) {
        PyErr_SetString(PyExc_ValueError, "byte_to_class must be a 256-element array");
        return -1;
    }
    if (PyArray_TYPE(byte_to_class_array) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "byte_to_class must have uint8 dtype");
        return -1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(byte_to_class_array)) {
        PyErr_SetString(PyExc_ValueError,
                        "byte_to_class must be a C-contiguous array. Use np.ascontiguousarray() to convert.");
        return -1;
    }

    // Validate class substitution costs (should be a 32x32 int8 numpy array)
    if (!PyArray_Check(class_substitution_costs_obj)) {
        PyErr_SetString(PyExc_TypeError, "class_substitution_costs must be a NumPy array");
        return -1;
    }
    PyArrayObject *class_costs_array = (PyArrayObject *)class_substitution_costs_obj;
    if (PyArray_NDIM(class_costs_array) != 2 || PyArray_DIM(class_costs_array, 0) != 32 ||
        PyArray_DIM(class_costs_array, 1) != 32) {
        PyErr_SetString(PyExc_ValueError, "class_substitution_costs must be a 32x32 array");
        return -1;
    }
    if (PyArray_TYPE(class_costs_array) != NPY_INT8) {
        PyErr_SetString(PyExc_TypeError, "class_substitution_costs must have int8 dtype");
        return -1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(class_costs_array)) {
        PyErr_SetString(
            PyExc_ValueError,
            "class_substitution_costs must be a C-contiguous array. Use np.ascontiguousarray() to convert.");
        return -1;
    }

    // Parse capabilities if provided
    if (capabilities_tuple) {
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) { return -1; }
    }

    // Initialize the engine
    sz_u8_t *byte_to_class_data = (sz_u8_t *)PyArray_DATA(byte_to_class_array);
    sz_error_cost_t *class_costs_data = (sz_error_cost_t *)PyArray_DATA(class_costs_array);

    // Create a simple checksum of the class cost matrix for the description
    sz_u32_t subs_checksum = 0;
    for (int i = 0; i < 32; ++i) subs_checksum += (sz_u32_t)class_costs_data[i * 32 + i]; // Diagonal elements

    char const *error_detail = NULL;
    sz_status_t status = szs_needleman_wunsch_scores_init(byte_to_class_data, class_costs_data, open, extend, NULL,
                                                          capabilities, &self->handle, &error_detail);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "NeedlemanWunsch initialization");
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "%X,%d,%d", subs_checksum & 0xFFFF, open, extend);
    self->capabilities = capabilities;
    return 0;
}

static PyObject *NeedlemanWunsch_repr(NeedlemanWunsch *self) {
    return PyUnicode_FromFormat("NeedlemanWunschScores(subs_checksum,open,extend=%s)", self->description);
}

static PyObject *NeedlemanWunsch_get_capabilities(NeedlemanWunsch *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyObject *NeedlemanWunsch_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                            PyObject *kwnames) {
    NeedlemanWunsch *self = (NeedlemanWunsch *)callable;
    PyObject *queries_obj = NULL, *candidates_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    if (parse_cross_product_call_args("NeedlemanWunsch.__call__", args, nargsf, kwnames, &queries_obj, &candidates_obj,
                                      &device_obj, &out_obj) != 0)
        return NULL;

    // Treat an explicit `None` for `candidates` as "compute symmetric self-similarity of queries".
    if (candidates_obj == Py_None) candidates_obj = NULL;
    sz_bool_t is_self_similarity = (candidates_obj == NULL) ? sz_true_k : sz_false_k;

    // Get device handle
    szs_device_scope_t device_handle = default_device_scope;
    if (device_obj && device_obj != Py_None) {
        if (!PyObject_IsInstance(device_obj, (PyObject *)&DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_handle = ((DeviceScope *)device_obj)->handle;
    }

    sz_size_t queries_count = 0;
    sz_size_t candidates_count = 0;
    void const *kernel_queries_punned = NULL;
    void const *kernel_candidates_punned = NULL;
    sz_status_t (*kernel_punned)(szs_needleman_wunsch_scores_t, szs_device_scope_t, void const *, void const *,
                                 sz_ssize_t *, sz_size_t, char const **) = NULL;

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    if (requires_unified_memory(self->capabilities)) {
        if (!try_swap_to_unified_allocator(queries_obj)) return NULL;
        if (candidates_obj && !try_swap_to_unified_allocator(candidates_obj)) return NULL;
    }

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t queries_u32tape, candidates_u32tape;
    sz_bool_t queries_is_u32tape = sz_py_export_strings_as_u32tape( //
        queries_obj, &queries_u32tape.data, &queries_u32tape.offsets, &queries_u32tape.count);
    sz_bool_t candidates_is_u32tape = candidates_obj && sz_py_export_strings_as_u32tape( //
                                                            candidates_obj, &candidates_u32tape.data,
                                                            &candidates_u32tape.offsets, &candidates_u32tape.count);
    if (queries_is_u32tape && (is_self_similarity || candidates_is_u32tape)) {
        queries_count = queries_u32tape.count;
        candidates_count = is_self_similarity ? queries_u32tape.count : candidates_u32tape.count;
        kernel_punned = szs_needleman_wunsch_scores_u32tape;
        kernel_queries_punned = &queries_u32tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t queries_u64tape, candidates_u64tape;
    sz_bool_t queries_is_u64tape = !queries_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                              queries_obj, &queries_u64tape.data,
                                                              &queries_u64tape.offsets, &queries_u64tape.count);
    sz_bool_t candidates_is_u64tape = candidates_obj && !candidates_is_u32tape &&
                                      sz_py_export_strings_as_u64tape( //
                                          candidates_obj, &candidates_u64tape.data, &candidates_u64tape.offsets,
                                          &candidates_u64tape.count);
    if (!kernel_punned && queries_is_u64tape && (is_self_similarity || candidates_is_u64tape)) {
        queries_count = queries_u64tape.count;
        candidates_count = is_self_similarity ? queries_u64tape.count : candidates_u64tape.count;
        kernel_punned = szs_needleman_wunsch_scores_u64tape;
        kernel_queries_punned = &queries_u64tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u64tape;
    }

    // Handle sequence inputs
    sz_sequence_t queries_seq, candidates_seq;
    sz_bool_t queries_is_sequence = !queries_is_u32tape && !queries_is_u64tape &&
                                    sz_py_export_strings_as_sequence(queries_obj, &queries_seq);
    sz_bool_t candidates_is_sequence = candidates_obj && !candidates_is_u32tape && !candidates_is_u64tape &&
                                       sz_py_export_strings_as_sequence(candidates_obj, &candidates_seq);
    if (!kernel_punned && queries_is_sequence && (is_self_similarity || candidates_is_sequence)) {
        queries_count = queries_seq.count;
        candidates_count = is_self_similarity ? queries_seq.count : candidates_seq.count;
        kernel_punned = szs_needleman_wunsch_scores;
        kernel_queries_punned = &queries_seq;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_seq;
    }

    // No homogeneous kernel matched. This happens when one side is empty: an empty Strs is always
    // FRAGMENTED and cannot pair with a non-empty side's tape above. If both sides are recognized Strs
    // and either is empty, the cross product is an empty matrix -- record the counts and fall through
    // with a NULL kernel (the call below is skipped and the empty matrix returned).
    if (!kernel_punned) {
        sz_bool_t queries_recognized = queries_is_u32tape || queries_is_u64tape || queries_is_sequence;
        sz_bool_t candidates_recognized = is_self_similarity || candidates_is_u32tape || candidates_is_u64tape ||
                                          candidates_is_sequence;
        sz_size_t queries_any_count = queries_is_u32tape    ? queries_u32tape.count
                                      : queries_is_u64tape  ? queries_u64tape.count
                                      : queries_is_sequence ? queries_seq.count
                                                            : 0;
        sz_size_t candidates_any_count = is_self_similarity       ? queries_any_count
                                         : candidates_is_u32tape  ? candidates_u32tape.count
                                         : candidates_is_u64tape  ? candidates_u64tape.count
                                         : candidates_is_sequence ? candidates_seq.count
                                                                  : 0;
        if (!(queries_recognized && candidates_recognized && (queries_any_count == 0 || candidates_any_count == 0))) {
            PyErr_Format( //
                PyExc_TypeError,
                "Expected stringzilla.Strs objects, got %s and %s. " //
                "Convert using: stringzilla.Strs(your_string_list)",
                Py_TYPE(queries_obj)->tp_name, candidates_obj ? Py_TYPE(candidates_obj)->tp_name : "None");
            return NULL;
        }
        queries_count = queries_any_count;
        candidates_count = candidates_any_count;
    }

    // Allocate a fresh 2-D matrix or validate the provided `out` array, deriving the row stride in ELEMENTS.
    PyObject *results_array = NULL;
    sz_ssize_t *kernel_results = NULL;
    sz_size_t kernel_results_row_stride = 0;

    if (!out_obj || out_obj == Py_None) {
        npy_intp results_shape[2] = {(npy_intp)queries_count, (npy_intp)candidates_count};
        results_array = PyArray_SimpleNew(2, results_shape, NPY_INT64);
        if (!results_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate results array");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_row_stride = candidates_count;
    }
    else {
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out must be a NumPy array");
            goto cleanup;
        }
        PyArrayObject *array = (PyArrayObject *)out_obj;
        if (PyArray_NDIM(array) != 2) {
            PyErr_SetString(PyExc_ValueError, "out array must be 2-dimensional");
            goto cleanup;
        }
        if (PyArray_DIM(array, 0) < (npy_intp)queries_count || PyArray_DIM(array, 1) < (npy_intp)candidates_count) {
            PyErr_SetString(PyExc_ValueError, "out array is too small for results");
            goto cleanup;
        }
        if (PyArray_TYPE(array) != NPY_INT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have int64 dtype");
            goto cleanup;
        }
        if (PyArray_STRIDE(array, 1) != (npy_intp)sizeof(sz_ssize_t)) {
            PyErr_SetString(PyExc_ValueError, "out array rows must be contiguous (unit stride along columns)");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA(array);
        kernel_results_row_stride = (sz_size_t)(PyArray_STRIDE(array, 0) / (npy_intp)sizeof(sz_ssize_t));
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    char const *error_detail = NULL;
    sz_status_t status = sz_success_k; // An empty cross product (zero-row/col matrix) needs no kernel
    if (kernel_punned)
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "NeedlemanWunsch computation");
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static char const doc_NeedlemanWunsch[] =                                                                     //
    "NeedlemanWunschScores(byte_to_class, class_substitution_costs, open=-1, extend=-1, capabilities=None)\n" //
    "\n"                                                                                                      //
    "Needleman-Wunsch global alignment scoring engine.\n"                                                     //
    "Computes the cross-product matrix of alignment scores between two string collections.\n"                 //
    "\n"                                                                                                      //
    "Args:\n"                                                                                                 //
    "  byte_to_class (np.ndarray): 256-element uint8 map from each byte to one of 32 classes.\n"              //
    "  class_substitution_costs (np.ndarray): 32x32 int8 matrix of costs between classes.\n"                  //
    "  open (int): Cost for opening a gap (default: -1).\n"                                                   //
    "  extend (int): Cost for extending a gap (default: -1).\n"                                               //
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"                   //
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"       //
    "                                       or a DeviceScope for automatic capability inference.\n"           //
    "\n"                                                                                                      //
    "Call with:\n"                                                                                            //
    "  queries (sequence): Query strings forming the matrix rows.\n"                                          //
    "  candidates (sequence, optional): Candidate strings forming the matrix columns. When omitted\n"         //
    "                                   (or None), computes the symmetric self-similarity of queries.\n"      //
    "  device (DeviceScope, optional): Device execution context.\n"                                           //
    "  out (np.ndarray, optional): 2-D int64 output buffer of shape (len(queries), len(candidates)).\n"       //
    "\n"                                                                                                      //
    "Returns:\n"                                                                                              //
    "  np.ndarray: 2-D int64 matrix where result[query_index, candidate_index] is the score\n"                //
    "              between queries[query_index] and candidates[candidate_index].\n"                           //
    "\n"                                                                                                      //
    "Examples:\n"                                                                                             //
    "  >>> # Minimal CPU example mapping every byte to its own class modulo 32\n"                             //
    "  >>> import numpy as np, stringzilla as sz, stringzillas as szs\n"                                      //
    "  >>> classes = (np.arange(256) % 32).astype(np.uint8)\n"                                                //
    "  >>> costs = np.zeros((32, 32), dtype=np.int8)\n"                                                       //
    "  >>> engine = szs.NeedlemanWunschScores(classes, costs)\n"                                              //
    "  >>> proteins_a = sz.Strs(['ACGT', 'TGCA'])\n"                                                          //
    "  >>> proteins_b = sz.Strs(['ACCT', 'TGAA'])\n"                                                          //
    "  >>> scores = engine(proteins_a, proteins_b)\n"                                                         //
    "  >>> # GPU example with custom gap penalties; falls back to CPU when CUDA is unavailable\n"             //
    "  >>> scope = szs.DeviceScope(gpu_device=0) if 'cuda' in szs.__capabilities__ else szs.DeviceScope()\n"  //
    "  >>> gap_open, gap_extend = -2, -1\n"                                                                   //
    "  >>> engine = szs.NeedlemanWunschScores(classes, costs, gap_open, gap_extend, scope)\n"                 //
    "  >>> scores = engine(proteins_a, proteins_b, device=scope)";
static PyGetSetDef NeedlemanWunsch_getsetters[] = {
    {"__capabilities__", (getter)NeedlemanWunsch_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                        /* Sentinel */
};

static PyTypeObject NeedlemanWunschType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.NeedlemanWunschScores",
    .tp_doc = doc_NeedlemanWunsch,
    .tp_basicsize = sizeof(NeedlemanWunsch),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(NeedlemanWunsch, vectorcall),
    .tp_new = NeedlemanWunsch_new,
    .tp_init = (initproc)NeedlemanWunsch_init,
    .tp_dealloc = (destructor)NeedlemanWunsch_dealloc,
    .tp_call = PyVectorcall_Call,
    .tp_repr = (reprfunc)NeedlemanWunsch_repr,
    .tp_getset = NeedlemanWunsch_getsetters,
};

#pragma endregion

#pragma region SmithWaterman

/**
 *  @brief  Smith-Waterman local alignment scoring engine.
 */
typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
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

static PyObject *SmithWaterman_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames);

static PyObject *SmithWaterman_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    SmithWaterman *self = (SmithWaterman *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->vectorcall = (vectorcallfunc)SmithWaterman_vectorcall;
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
    }
    return (PyObject *)self;
}

static int SmithWaterman_init(SmithWaterman *self, PyObject *args, PyObject *kwargs) {
    PyObject *byte_to_class_obj = NULL;
    PyObject *class_substitution_costs_obj = NULL;
    sz_error_cost_t open = -1, extend = -1;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    // Manual positional + keyword parse (no `PyArg_ParseTupleAndKeywords`, no generic binder).
    // Arguments: byte_to_class, class_substitution_costs, open, extend, capabilities.
    char const *const callable_name = Py_TYPE(self)->tp_name;
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count > 5) {
        PyErr_Format(PyExc_TypeError, "%s takes at most 5 arguments (%zd given)", callable_name, positional_count);
        return -1;
    }
    byte_to_class_obj = positional_count > 0 ? PyTuple_GET_ITEM(args, 0) : NULL;
    class_substitution_costs_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *open_obj = positional_count > 2 ? PyTuple_GET_ITEM(args, 2) : NULL;
    PyObject *extend_obj = positional_count > 3 ? PyTuple_GET_ITEM(args, 3) : NULL;
    capabilities_tuple = positional_count > 4 ? PyTuple_GET_ITEM(args, 4) : NULL;
    if (kwargs != NULL) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "byte_to_class") == 0) {
                if (byte_to_class_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'byte_to_class'", callable_name);
                    return -1;
                }
                byte_to_class_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "class_substitution_costs") == 0) {
                if (class_substitution_costs_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'class_substitution_costs'",
                                 callable_name);
                    return -1;
                }
                class_substitution_costs_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "open") == 0) {
                if (open_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'open'", callable_name);
                    return -1;
                }
                open_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "extend") == 0) {
                if (extend_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'extend'", callable_name);
                    return -1;
                }
                extend_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                if (capabilities_tuple) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'capabilities'", callable_name);
                    return -1;
                }
                capabilities_tuple = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }
    if (open_obj) {
        long open_value = PyLong_AsLong(open_obj);
        if (PyErr_Occurred()) return -1;
        open = (sz_error_cost_t)open_value;
    }
    if (extend_obj) {
        long extend_value = PyLong_AsLong(extend_obj);
        if (PyErr_Occurred()) return -1;
        extend = (sz_error_cost_t)extend_value;
    }
    if (byte_to_class_obj == NULL || class_substitution_costs_obj == NULL) {
        PyErr_Format(PyExc_TypeError, "%s requires 'byte_to_class' and 'class_substitution_costs'",
                     Py_TYPE(self)->tp_name);
        return -1;
    }

    // Validate byte-to-class map (should be a 256-element uint8 numpy array)
    if (!numpy_available || !PyArray_Check(byte_to_class_obj)) {
        PyErr_SetString(PyExc_TypeError, "byte_to_class must be a NumPy array");
        return -1;
    }
    PyArrayObject *byte_to_class_array = (PyArrayObject *)byte_to_class_obj;
    if (PyArray_NDIM(byte_to_class_array) != 1 || PyArray_DIM(byte_to_class_array, 0) != 256) {
        PyErr_SetString(PyExc_ValueError, "byte_to_class must be a 256-element array");
        return -1;
    }
    if (PyArray_TYPE(byte_to_class_array) != NPY_UINT8) {
        PyErr_SetString(PyExc_TypeError, "byte_to_class must have uint8 dtype");
        return -1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(byte_to_class_array)) {
        PyErr_SetString(PyExc_ValueError,
                        "byte_to_class must be a C-contiguous array. Use np.ascontiguousarray() to convert.");
        return -1;
    }

    // Validate class substitution costs (should be a 32x32 int8 numpy array)
    if (!PyArray_Check(class_substitution_costs_obj)) {
        PyErr_SetString(PyExc_TypeError, "class_substitution_costs must be a NumPy array");
        return -1;
    }
    PyArrayObject *class_costs_array = (PyArrayObject *)class_substitution_costs_obj;
    if (PyArray_NDIM(class_costs_array) != 2 || PyArray_DIM(class_costs_array, 0) != 32 ||
        PyArray_DIM(class_costs_array, 1) != 32) {
        PyErr_SetString(PyExc_ValueError, "class_substitution_costs must be a 32x32 array");
        return -1;
    }
    if (PyArray_TYPE(class_costs_array) != NPY_INT8) {
        PyErr_SetString(PyExc_TypeError, "class_substitution_costs must have int8 dtype");
        return -1;
    }
    if (!PyArray_IS_C_CONTIGUOUS(class_costs_array)) {
        PyErr_SetString(
            PyExc_ValueError,
            "class_substitution_costs must be a C-contiguous array. Use np.ascontiguousarray() to convert.");
        return -1;
    }

    // Parse capabilities if provided
    if (capabilities_tuple) {
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) { return -1; }
    }

    // Initialize the engine
    sz_u8_t *byte_to_class_data = (sz_u8_t *)PyArray_DATA(byte_to_class_array);
    sz_error_cost_t *class_costs_data = (sz_error_cost_t *)PyArray_DATA(class_costs_array);
    char const *error_detail = NULL;
    sz_status_t status = szs_smith_waterman_scores_init(byte_to_class_data, class_costs_data, open, extend, NULL,
                                                        capabilities, &self->handle, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "SmithWaterman initialization");
        return -1;
    }

    // Create a simple checksum of the class cost matrix for the description
    sz_u32_t subs_checksum = 0;
    for (int i = 0; i < 32; ++i) subs_checksum += (sz_u32_t)class_costs_data[i * 32 + i]; // Diagonal elements

    snprintf(self->description, sizeof(self->description), "%X,%d,%d", subs_checksum & 0xFFFF, open, extend);
    self->capabilities = capabilities;
    return 0;
}

static PyObject *SmithWaterman_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf, PyObject *kwnames) {
    SmithWaterman *self = (SmithWaterman *)callable;
    PyObject *queries_obj = NULL, *candidates_obj = NULL, *device_obj = NULL, *out_obj = NULL;

    if (parse_cross_product_call_args("SmithWaterman.__call__", args, nargsf, kwnames, &queries_obj, &candidates_obj,
                                      &device_obj, &out_obj) != 0)
        return NULL;

    // Treat an explicit `None` for `candidates` as "compute symmetric self-similarity of queries".
    if (candidates_obj == Py_None) candidates_obj = NULL;
    sz_bool_t is_self_similarity = (candidates_obj == NULL) ? sz_true_k : sz_false_k;

    // Get device handle
    szs_device_scope_t device_handle = default_device_scope;
    if (device_obj && device_obj != Py_None) {
        if (!PyObject_IsInstance(device_obj, (PyObject *)&DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_handle = ((DeviceScope *)device_obj)->handle;
    }

    sz_size_t queries_count = 0;
    sz_size_t candidates_count = 0;
    void const *kernel_queries_punned = NULL;
    void const *kernel_candidates_punned = NULL;
    sz_status_t (*kernel_punned)(szs_smith_waterman_scores_t, szs_device_scope_t, void const *, void const *,
                                 sz_ssize_t *, sz_size_t, char const **) = NULL;

    // Swap allocators only when using CUDA with a GPU device (inputs must be unified)
    if (requires_unified_memory(self->capabilities)) {
        if (!try_swap_to_unified_allocator(queries_obj)) return NULL;
        if (candidates_obj && !try_swap_to_unified_allocator(candidates_obj)) return NULL;
    }

    // Handle 32-bit tape inputs
    sz_sequence_u32tape_t queries_u32tape, candidates_u32tape;
    sz_bool_t queries_is_u32tape = sz_py_export_strings_as_u32tape( //
        queries_obj, &queries_u32tape.data, &queries_u32tape.offsets, &queries_u32tape.count);
    sz_bool_t candidates_is_u32tape = candidates_obj && sz_py_export_strings_as_u32tape( //
                                                            candidates_obj, &candidates_u32tape.data,
                                                            &candidates_u32tape.offsets, &candidates_u32tape.count);
    if (queries_is_u32tape && (is_self_similarity || candidates_is_u32tape)) {
        queries_count = queries_u32tape.count;
        candidates_count = is_self_similarity ? queries_u32tape.count : candidates_u32tape.count;
        kernel_punned = szs_smith_waterman_scores_u32tape;
        kernel_queries_punned = &queries_u32tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u32tape;
    }

    // Handle 64-bit tape inputs
    sz_sequence_u64tape_t queries_u64tape, candidates_u64tape;
    sz_bool_t queries_is_u64tape = !queries_is_u32tape && sz_py_export_strings_as_u64tape( //
                                                              queries_obj, &queries_u64tape.data,
                                                              &queries_u64tape.offsets, &queries_u64tape.count);
    sz_bool_t candidates_is_u64tape = candidates_obj && !candidates_is_u32tape &&
                                      sz_py_export_strings_as_u64tape( //
                                          candidates_obj, &candidates_u64tape.data, &candidates_u64tape.offsets,
                                          &candidates_u64tape.count);
    if (!kernel_punned && queries_is_u64tape && (is_self_similarity || candidates_is_u64tape)) {
        queries_count = queries_u64tape.count;
        candidates_count = is_self_similarity ? queries_u64tape.count : candidates_u64tape.count;
        kernel_punned = szs_smith_waterman_scores_u64tape;
        kernel_queries_punned = &queries_u64tape;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_u64tape;
    }

    // Handle sequence inputs
    sz_sequence_t queries_seq, candidates_seq;
    sz_bool_t queries_is_sequence = !queries_is_u32tape && !queries_is_u64tape &&
                                    sz_py_export_strings_as_sequence(queries_obj, &queries_seq);
    sz_bool_t candidates_is_sequence = candidates_obj && !candidates_is_u32tape && !candidates_is_u64tape &&
                                       sz_py_export_strings_as_sequence(candidates_obj, &candidates_seq);
    if (!kernel_punned && queries_is_sequence && (is_self_similarity || candidates_is_sequence)) {
        queries_count = queries_seq.count;
        candidates_count = is_self_similarity ? queries_seq.count : candidates_seq.count;
        kernel_punned = szs_smith_waterman_scores;
        kernel_queries_punned = &queries_seq;
        kernel_candidates_punned = is_self_similarity ? NULL : &candidates_seq;
    }

    // No homogeneous kernel matched. This happens when one side is empty: an empty Strs is always
    // FRAGMENTED and cannot pair with a non-empty side's tape above. If both sides are recognized Strs
    // and either is empty, the cross product is an empty matrix -- record the counts and fall through
    // with a NULL kernel (the call below is skipped and the empty matrix returned).
    if (!kernel_punned) {
        sz_bool_t queries_recognized = queries_is_u32tape || queries_is_u64tape || queries_is_sequence;
        sz_bool_t candidates_recognized = is_self_similarity || candidates_is_u32tape || candidates_is_u64tape ||
                                          candidates_is_sequence;
        sz_size_t queries_any_count = queries_is_u32tape    ? queries_u32tape.count
                                      : queries_is_u64tape  ? queries_u64tape.count
                                      : queries_is_sequence ? queries_seq.count
                                                            : 0;
        sz_size_t candidates_any_count = is_self_similarity       ? queries_any_count
                                         : candidates_is_u32tape  ? candidates_u32tape.count
                                         : candidates_is_u64tape  ? candidates_u64tape.count
                                         : candidates_is_sequence ? candidates_seq.count
                                                                  : 0;
        if (!(queries_recognized && candidates_recognized && (queries_any_count == 0 || candidates_any_count == 0))) {
            PyErr_Format( //
                PyExc_TypeError,
                "Expected stringzilla.Strs objects, got %s and %s. " //
                "Convert using: stringzilla.Strs(your_string_list)",
                Py_TYPE(queries_obj)->tp_name, candidates_obj ? Py_TYPE(candidates_obj)->tp_name : "None");
            return NULL;
        }
        queries_count = queries_any_count;
        candidates_count = candidates_any_count;
    }

    // Allocate a fresh 2-D matrix or validate the provided `out` array, deriving the row stride in ELEMENTS.
    PyObject *results_array = NULL;
    sz_ssize_t *kernel_results = NULL;
    sz_size_t kernel_results_row_stride = 0;

    if (!out_obj || out_obj == Py_None) {
        npy_intp results_shape[2] = {(npy_intp)queries_count, (npy_intp)candidates_count};
        results_array = PyArray_SimpleNew(2, results_shape, NPY_INT64);
        if (!results_array) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate results array");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA((PyArrayObject *)results_array);
        kernel_results_row_stride = candidates_count;
    }
    else {
        if (!PyArray_Check(out_obj)) {
            PyErr_SetString(PyExc_TypeError, "out must be a NumPy array");
            goto cleanup;
        }
        PyArrayObject *array = (PyArrayObject *)out_obj;
        if (PyArray_NDIM(array) != 2) {
            PyErr_SetString(PyExc_ValueError, "out array must be 2-dimensional");
            goto cleanup;
        }
        if (PyArray_DIM(array, 0) < (npy_intp)queries_count || PyArray_DIM(array, 1) < (npy_intp)candidates_count) {
            PyErr_SetString(PyExc_ValueError, "out array is too small for results");
            goto cleanup;
        }
        if (PyArray_TYPE(array) != NPY_INT64) {
            PyErr_SetString(PyExc_TypeError, "out array must have int64 dtype");
            goto cleanup;
        }
        if (PyArray_STRIDE(array, 1) != (npy_intp)sizeof(sz_ssize_t)) {
            PyErr_SetString(PyExc_ValueError, "out array rows must be contiguous (unit stride along columns)");
            goto cleanup;
        }
        kernel_results = (sz_ssize_t *)PyArray_DATA(array);
        kernel_results_row_stride = (sz_size_t)(PyArray_STRIDE(array, 0) / (npy_intp)sizeof(sz_ssize_t));
        results_array = out_obj;
        Py_INCREF(results_array);
    }

    char const *error_detail = NULL;
    sz_status_t status = sz_success_k; // An empty cross product (zero-row/col matrix) needs no kernel
    if (kernel_punned)
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "SmithWaterman computation");
        goto cleanup;
    }
    return results_array;

cleanup:
    Py_XDECREF(results_array);
    return NULL;
}

static PyObject *SmithWaterman_repr(SmithWaterman *self) {
    return PyUnicode_FromFormat("SmithWatermanScores(subs_checksum,open,extend=%s)", self->description);
}

static PyObject *SmithWaterman_get_capabilities(SmithWaterman *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static PyGetSetDef SmithWaterman_getsetters[] = {
    {"__capabilities__", (getter)SmithWaterman_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                      /* Sentinel */
};

static char const doc_SmithWaterman[] =                                                                      //
    "SmithWatermanScores(byte_to_class, class_substitution_costs, open=-1, extend=-1, capabilities=None)\n"  //
    "\n"                                                                                                     //
    "Smith-Waterman local alignment scoring engine.\n"                                                       //
    "Computes the cross-product matrix of local alignment scores between two string collections.\n"          //
    "\n"                                                                                                     //
    "Args:\n"                                                                                                //
    "  byte_to_class (np.ndarray): 256-element uint8 map from each byte to one of 32 classes.\n"             //
    "  class_substitution_costs (np.ndarray): 32x32 int8 matrix of costs between classes.\n"                 //
    "  open (int): Cost for opening a gap (default: -1).\n"                                                  //
    "  extend (int): Cost for extending a gap (default: -1).\n"                                              //
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"                  //
    "                                       Can be explicit capabilities like ('serial', 'parallel')\n"      //
    "                                       or a DeviceScope for automatic capability inference.\n"          //
    "\n"                                                                                                     //
    "Call with:\n"                                                                                           //
    "  queries (sequence): Query strings forming the matrix rows.\n"                                         //
    "  candidates (sequence, optional): Candidate strings forming the matrix columns. When omitted\n"        //
    "                                   (or None), computes the symmetric self-similarity of queries.\n"     //
    "  device (DeviceScope, optional): Device execution context.\n"                                          //
    "  out (np.ndarray, optional): 2-D int64 output buffer of shape (len(queries), len(candidates)).\n"      //
    "\n"                                                                                                     //
    "Returns:\n"                                                                                             //
    "  np.ndarray: 2-D int64 matrix where result[query_index, candidate_index] is the score\n"               //
    "              between queries[query_index] and candidates[candidate_index].\n"                          //
    "\n"                                                                                                     //
    "Examples:\n"                                                                                            //
    "  >>> # Minimal CPU example for local alignment\n"                                                      //
    "  >>> import numpy as np, stringzilla as sz, stringzillas as szs\n"                                     //
    "  >>> classes = (np.arange(256) % 32).astype(np.uint8)\n"                                               //
    "  >>> costs = np.eye(32, dtype=np.int8)  # Identity matrix\n"                                           //
    "  >>> engine = szs.SmithWatermanScores(classes, costs)\n"                                               //
    "  >>> seqs_a = sz.Strs(['ACGTACGT', 'TGCATGCA'])\n"                                                     //
    "  >>> seqs_b = sz.Strs(['CGTACGTA', 'GCATGCAT'])\n"                                                     //
    "  >>> scores = engine(seqs_a, seqs_b)\n"                                                                //
    "  >>> # GPU example with different gap costs; falls back to CPU when CUDA is unavailable\n"             //
    "  >>> scope = szs.DeviceScope(gpu_device=0) if 'cuda' in szs.__capabilities__ else szs.DeviceScope()\n" //
    "  >>> gap_open, gap_extend = -3, -1\n"                                                                  //
    "  >>> engine = szs.SmithWatermanScores(classes, costs, gap_open, gap_extend, scope)\n"                  //
    "  >>> scores = engine(seqs_a, seqs_b, device=scope)";
static PyTypeObject SmithWatermanType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.SmithWatermanScores",
    .tp_doc = doc_SmithWaterman,
    .tp_basicsize = sizeof(SmithWaterman),
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,
    .tp_vectorcall_offset = offsetof(SmithWaterman, vectorcall),
    .tp_new = SmithWaterman_new,
    .tp_init = (initproc)SmithWaterman_init,
    .tp_dealloc = (destructor)SmithWaterman_dealloc,
    .tp_call = PyVectorcall_Call,
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
    unsigned long long seed = 0;
    PyObject *capabilities_tuple = NULL;
    sz_capability_t capabilities = default_hardware_capabilities;

    static char *kwlist[] = {"ndim", "window_widths", "alphabet_size", "seed", "capabilities", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "n|OnKO", kwlist, &ndim, &window_widths_obj, &alphabet_size, &seed,
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

    char const *error_detail = NULL;
    sz_status_t status = szs_fingerprints_init(ndim, alphabet_size, window_widths, window_widths_count, (sz_u64_t)seed,
                                               NULL, capabilities, &self->handle, &error_detail);

    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Fingerprints initialization");
        return -1;
    }

    snprintf(self->description, sizeof(self->description), "ndim=%zu,window_widths=%zu,alphabet_size=%zu,seed=%llu",
             ndim, window_widths_count, alphabet_size, seed);
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
                                 sz_size_t, char const **) = NULL;

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
    sz_bool_t texts_is_u64tape = !texts_is_u32tape &&
                                 sz_py_export_strings_as_u64tape( //
                                     texts_obj, &texts_u64tape.data, &texts_u64tape.offsets, &texts_u64tape.count);
    if (texts_is_u64tape) {
        kernel_input_size = texts_u64tape.count;
        kernel_punned = szs_fingerprints_u64tape;
        kernel_texts_punned = &texts_u64tape;
    }

    // Handle generic sequence inputs
    sz_sequence_t texts_seq;
    sz_bool_t texts_is_sequence = !texts_is_u32tape && !texts_is_u64tape &&
                                  sz_py_export_strings_as_sequence(texts_obj, &texts_seq);
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

        char const *error_detail = NULL;
        sz_status_t status = kernel_punned(self->handle, device_handle, kernel_texts_punned, buf_hashes,
                                           self->ndim * sizeof(sz_u32_t), buf_counts, self->ndim * sizeof(sz_u32_t),
                                           &error_detail);
        if (status != sz_success_k) {
            out_alloc->free(buf_hashes, total_bytes, out_alloc->handle);
            out_alloc->free(buf_counts, total_bytes, out_alloc->handle);
            Py_DECREF(hashes_array);
            Py_DECREF(counts_array);
            set_stringzilla_error(status, error_detail, "Fingerprints computation");
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

static char const doc_Fingerprints[] =                                                                               //
    "Fingerprints(ndim, window_widths=None, alphabet_size=256, seed=0, capabilities=None)\n"                         //
    "\n"                                                                                                             //
    "Compute MinHash fingerprints for binary strings.\n"                                                             //
    "\n"                                                                                                             //
    "Args:\n"                                                                                                        //
    "  ndim (int): Number of dimensions per fingerprint.\n"                                                          //
    "  window_widths (numpy.array, optional): 1D uint64 contiguous array of window widths. Uses defaults if None.\n" //
    "  alphabet_size (int, optional): Alphabet size, default 256 for binary strings.\n"                              //
    "  seed (int, optional): Reproducibility seed; every value derives independent per-dimension multipliers and\n"  //
    "                        moduli for MinHash independence (0 is the default seed, not a special mode).\n"         //
    "  capabilities (Tuple[str] or DeviceScope, optional): Hardware capabilities to use.\n"                          //
    "                                       Can be explicit capabilities like ('serial', 'parallel', 'cuda')\n"      //
    "                                       or a DeviceScope for automatic capability inference.\n"                  //
    "\n"                                                                                                             //
    "Call with:\n"                                                                                                   //
    "  texts (sequence): Sequence of strings to fingerprint.\n"                                                      //
    "  device (DeviceScope, optional): Device execution context.\n"                                                  //
    "\n"                                                                                                             //
    "Returns:\n"                                                                                                     //
    "  tuple: (hashes_matrix, counts_matrix) - Two numpy uint32 matrices of shape (num_texts, ndim).\n"              //
    "\n"                                                                                                             //
    "Examples:\n"                                                                                                    //
    "  >>> # Minimal CPU example with auto-inferred capabilities\n"                                                  //
    "  >>> import stringzilla as sz, stringzillas as szs\n"                                                          //
    "  >>> engine = szs.Fingerprints(ndim=128)\n"                                                                    //
    "  >>> docs = sz.Strs(['document one', 'document two', 'document three'])\n"                                     //
    "  >>> hashes, counts = engine(docs)\n"                                                                          //
    "  >>> # GPU example with custom dimensions; falls back to CPU when CUDA is unavailable\n"                       //
    "  >>> scope = szs.DeviceScope(gpu_device=0) if 'cuda' in szs.__capabilities__ else szs.DeviceScope()\n"         //
    "  >>> engine = szs.Fingerprints(ndim=256, capabilities=scope)\n"                                                //
    "  >>> hashes, counts = engine(docs, device=scope)";
static PyGetSetDef Fingerprints_getsetters[] = {
    {"capabilities", (getter)Fingerprints_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                 /* Sentinel */
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

static char const doc_reset_capabilities[] =                                            //
    "reset_capabilities(names) -> None\n\n"                                             //
    "Sets the active SIMD/backend capabilities for this module and updates the\n"       //
    "default hardware capabilities. The provided names are intersected with hardware\n" //
    "capabilities; if the result is empty, falls back to 'serial'.\n\n"                 //
    "Side effects: updates stringzillas.__capabilities__ and __capabilities_str__.\n"   //
    "\n"                                                                                //
    "Examples:\n"                                                                       //
    "  >>> import stringzillas as szs\n"                                                //
    "  >>> szs.reset_capabilities(('serial',))  # restrict dispatch to the scalar backend";

static PyObject *module_reset_capabilities(PyObject *self, PyObject *args) {
    PyObject *caps_obj = NULL;
    if (!PyArg_ParseTuple(args, "O", &caps_obj)) return NULL;

    sz_capability_t caps = 0;
    if (parse_and_intersect_capabilities(caps_obj, &caps) != 0) return NULL;

    // Update the default hardware capabilities
    default_hardware_capabilities = caps;

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

    sz_cptr_t caps_str = sz_capabilities_to_string_implementation_(caps);
    if (PyObject_SetAttrString(self, "__capabilities_str__", PyUnicode_FromString(caps_str)) != 0) { return NULL; }

    Py_RETURN_NONE;
}

static char const doc_to_device[] =                                                  //
    "to_device(strs: sz.Strs) -> sz.Strs\n\n"                                        //
    "Converts a Strs object to use unified/device-accessible memory allocator.\n"    //
    "This function forces the allocator swap that would normally happen during\n"    //
    "GPU kernel execution. Useful for testing slice handling after re-allocation.\n" //
    "\n"                                                                             //
    "Examples:\n"                                                                    //
    "  >>> import stringzilla as sz, stringzillas as szs\n"                          //
    "  >>> strs = sz.Strs(['alpha', 'beta'])\n"                                      //
    "  >>> device_strs = szs.to_device(strs) if 'cuda' in szs.__capabilities__ else strs";

static PyObject *module_to_device(PyObject *self, PyObject *strs_obj) {
    if (!try_swap_to_unified_allocator(strs_obj)) return NULL;

    Py_INCREF(strs_obj);
    return strs_obj;
}

static void stringzillas_cleanup(PyObject *m) {
    sz_unused_(m);
    if (default_device_scope) {
        szs_device_scope_free(default_device_scope);
        default_device_scope = NULL;
    }
}

static PyMethodDef stringzillas_methods[] = {
    {"reset_capabilities", (PyCFunction)module_reset_capabilities, METH_VARARGS, doc_reset_capabilities},
    {"to_device", (PyCFunction)module_to_device, METH_O, doc_to_device},
    {NULL, NULL, 0, NULL}};

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
    sz_unused_(numpy_module);
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
    char const *alloc_error = NULL;
    sz_status_t alloc_status = sz_memory_allocator_init_unified(&unified_allocator, &alloc_error);
    if (alloc_status != sz_success_k) sz_memory_allocator_init_default(&unified_allocator);
    // Initialize default CPU allocator
    sz_memory_allocator_init_default(&default_allocator);

    // Initialize the default device scope for reuse
    char const *error_detail = NULL;
    sz_status_t status = szs_device_scope_init_default(&default_device_scope, &error_detail);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Default DeviceScope initialization");
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
    if (PyModule_AddObject(m, "NeedlemanWunschScores", (PyObject *)&NeedlemanWunschType) < 0) {
        Py_XDECREF(&NeedlemanWunschType);
        Py_XDECREF(&LevenshteinDistancesUTF8Type);
        Py_XDECREF(&LevenshteinDistancesType);
        Py_XDECREF(&DeviceScopeType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&SmithWatermanType);
    if (PyModule_AddObject(m, "SmithWatermanScores", (PyObject *)&SmithWatermanType) < 0) {
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
