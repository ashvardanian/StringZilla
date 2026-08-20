/**
 *  @brief Module-init glue: `PyModuleDef`, type registration, and `PyInit_stringzillas`.
 *  @file python/stringzillas/stringzillas.c
 *  @author Ash Vardanian
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids `PyArg_ParseTupleAndKeywords` calls.
 *  - Uses manual argument parsing for performance on hot paths.
 *  - Returns & accepts NumPy arrays when available, avoiding memory-scattered Python lists.
 */
#define SZS_NUMPY_API_OWNER_ 1
#include "stringzillas.h"

sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *) = NULL;
sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *) = NULL;
sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *) = NULL;
sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *) = NULL;
sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *) = NULL;

// Try to import NumPy, and fail if it's not available
int numpy_available = 0;
static PyObject *numpy_module = NULL;

szs_device_scope_t default_device_scope = NULL;
sz_capability_t comptime_capabilities_ = 0;
sz_capability_t runtime_capabilities_ = 0;
sz_capability_t active_capabilities_ = 0;
sz_memory_allocator_t unified_allocator;
sz_memory_allocator_t default_allocator;

char const doc_capabilities[] =                                                 //
    "Hardware backends and SIMD capabilities this engine selects at runtime.\n" //
    "\n"                                                                        //
    "Returns:\n"                                                                //
    "  str: The detected CPU/GPU features driving kernel dispatch.";

void set_stringzilla_error(sz_status_t status, char const *error_detail, char const *context) {
    switch (status) {
    case sz_bad_alloc_k: PyErr_Format(PyExc_MemoryError, "%s: %s", context, error_detail); break;
    case sz_invalid_utf8_k: PyErr_Format(PyExc_ValueError, "%s: %s", context, error_detail); break;
    case sz_overflow_risk_k: PyErr_Format(PyExc_OverflowError, "%s: %s", context, error_detail); break;
    case sz_unexpected_dimensions_k: PyErr_Format(PyExc_ValueError, "%s: %s", context, error_detail); break;
    // A backstop: the binding stages every buffer it hands an engine under a CUDA capability, so a reachable
    // one of these means the binding has a bug rather than the caller.
    case sz_device_memory_mismatch_k: PyErr_Format(PyExc_BufferError, "%s: %s", context, error_detail); break;
    case sz_missing_gpu_k:
    case sz_device_code_mismatch_k:
    default: PyErr_Format(PyExc_RuntimeError, "%s: %s", context, error_detail); break;
    }
}

PyObject *capabilities_to_tuple(sz_capability_t caps) {
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

int parse_and_intersect_capabilities(PyObject *caps_obj, sz_capability_t *result) {
    sz_capability_t const ceiling = (sz_capability_t)(comptime_capabilities_ & runtime_capabilities_);

    // Handle `DeviceScope` objects
    int const is_device_scope = PyObject_IsInstance(caps_obj, (PyObject *)&DeviceScopeType);
    if (is_device_scope < 0) return -1;
    if (is_device_scope) {
        DeviceScope *device_scope = (DeviceScope *)caps_obj;

        // Try to get GPU device
        sz_size_t gpu_device;
        char const *error_detail_gpu = NULL;
        if (szs_device_scope_get_gpu_device(device_scope->handle, &gpu_device, &error_detail_gpu) == sz_success_k) {
            if (ceiling & sz_caps_cuda_k) {
                *result = sz_caps_cuda_k & ceiling;
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
            *result = sz_caps_cpus_k & ceiling;
            return 0;
        }

        // Default scope - use all available capabilities
        *result = ceiling;
        return 0;
    }

    // Any sequence of names, matching the StringZilla module rather than demanding a tuple. Snapshotted into
    // a tuple rather than borrowed through `PySequence_Fast`, which returns the caller's list itself and
    // leaves the walk indexing into storage another thread can resize.
    PyObject *seq = PySequence_Tuple(caps_obj);
    if (!seq) {
        PyErr_SetString(PyExc_TypeError, "capabilities must be a sequence of strings or a DeviceScope object");
        return -1;
    }

    sz_capability_t requested_caps = 0;
    Py_ssize_t n = PyTuple_GET_SIZE(seq);

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyTuple_GET_ITEM(seq, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "capabilities must be strings");
            Py_DECREF(seq);
            return -1;
        }

        char const *cap_str = PyUnicode_AsUTF8(item);
        if (!cap_str) {
            Py_DECREF(seq);
            return -1;
        }

        sz_capability_t flag = sz_capability_from_string_implementation_(cap_str);
        if (flag == sz_caps_none_k) {
            PyErr_Format(PyExc_ValueError, "Unknown capability: %s", cap_str);
            Py_DECREF(seq);
            return -1;
        }
        requested_caps |= flag;
    }
    Py_DECREF(seq);

    // An empty request or one entirely outside this build's reach is a configuration error; silently
    // degrading to serial would hide it, and the `DeviceScope` arm above already raises for the same case.
    *result = requested_caps & ceiling;
    if (*result == 0) {
        char available[256];
        sz_capabilities_to_string_implementation_(ceiling, available, sizeof(available));
        PyErr_Format(PyExc_ValueError, "No requested capability is available here; available: %s", available);
        return -1;
    }

    return 0;
}

static char const doc_reset_capabilities[] =                                            //
    "reset_capabilities(names) -> None\n\n"                                             //
    "Sets the active SIMD/backend capabilities for this module and updates the\n"       //
    "default hardware capabilities. The provided names are intersected with hardware\n" //
    "capabilities; raises ValueError if none of them is available.\n\n"                 //
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

    // Only the active subset moves; the probed sets stay put, so `('any',)` widens back out.
    active_capabilities_ = caps;

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

    char caps_str[256];
    sz_capabilities_to_string_implementation_(caps, caps_str, sizeof(caps_str));
    if (PyObject_SetAttrString(self, "__capabilities_str__", PyUnicode_FromString(caps_str)) != 0) { return NULL; }

    Py_RETURN_NONE;
}

static char const doc_to_device[] =                                                     //
    "to_device(strs: sz.Strs) -> sz.Strs\n\n"                                           //
    "Move a Strs onto device-accessible unified memory, in place.\n"                    //
    "\n"                                                                                //
    "Engines do this themselves for the inputs of a GPU-scoped call, so it is only\n"   //
    "worth calling ahead of time to pay the swap once for a collection reused across\n" //
    "many calls, or to hold a slice's identity across the swap.\n"                      //
    "\n"                                                                                //
    "Examples:\n"                                                                       //
    "  >>> import stringzilla as sz, stringzillas as szs\n"                             //
    "  >>> strs = sz.Strs(['alpha', 'beta'])\n"                                         //
    "  >>> device_strs = szs.to_device(strs) if 'cuda' in szs.__capabilities__ else strs";

static PyObject *module_to_device(PyObject *self, PyObject *strs_obj) {
    if (!try_swap_to_unified_allocator(strs_obj)) return NULL;

    Py_INCREF(strs_obj);
    return strs_obj;
}

static char const doc_unified_array[] =                                               //
    "unified_array(shape, dtype=numpy.float32) -> numpy.ndarray\n\n"                  //
    "Allocate a NumPy array backed by device-accessible unified memory.\n"            //
    "\n"                                                                              //
    "A GPU scope refuses host buffers, so the arrays a caller supplies there - an\n"  //
    "engine's `out=`, or `Substrings.score_bm25`'s weights - come from here when\n"   //
    "CuPy or Torch is not in play. A CPU scope needs none of this.\n"                 //
    "\n"                                                                              //
    "Args:\n"                                                                         //
    "  shape (int or tuple): Length of a vector, or (rows, columns) of a matrix.\n"   //
    "  dtype (numpy.dtype, optional): Element type; float32 by default.\n"            //
    "\n"                                                                              //
    "Returns:\n"                                                                      //
    "  numpy.ndarray: Uninitialized, wrapping unified memory freed with the array.\n" //
    "\n"                                                                              //
    "Examples:\n"                                                                     //
    "  >>> import numpy as np, stringzillas as szs\n"                                 //
    "  >>> weights = szs.unified_array(2, dtype=np.float32)\n"                        //
    "  >>> weights[:] = 1.0";

static PyObject *module_unified_array(PyObject *self, PyObject *args, PyObject *kwargs) {
    sz_unused_(self);
    static char *kwlist[] = {"shape", "dtype", NULL};
    PyObject *shape_obj = NULL;
    PyObject *dtype_obj = NULL;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|O", kwlist, &shape_obj, &dtype_obj)) return NULL;
    if (!numpy_available) {
        PyErr_SetString(PyExc_RuntimeError, "NumPy is required to allocate unified arrays");
        return NULL;
    }

    // One length or a (rows, columns) pair, matching what the engines accept as a vector or a matrix.
    npy_intp shape[2] = {0, 0};
    int dimensions = 0;
    if (PyIndex_Check(shape_obj)) {
        Py_ssize_t const length = PyNumber_AsSsize_t(shape_obj, PyExc_OverflowError);
        if (length == -1 && PyErr_Occurred()) return NULL;
        if (length < 0) {
            PyErr_SetString(PyExc_ValueError, "Negative dimensions are not allowed");
            return NULL;
        }
        shape[0] = (npy_intp)length, dimensions = 1;
    }
    else if (PyTuple_Check(shape_obj)) {
        dimensions = (int)PyTuple_GET_SIZE(shape_obj);
        if (dimensions < 1 || dimensions > 2) {
            PyErr_SetString(PyExc_ValueError, "Only 1-D and 2-D unified arrays are supported");
            return NULL;
        }
        for (int axis = 0; axis < dimensions; ++axis) {
            Py_ssize_t const length = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape_obj, axis), PyExc_OverflowError);
            if (length == -1 && PyErr_Occurred()) return NULL;
            if (length < 0) {
                PyErr_SetString(PyExc_ValueError, "Negative dimensions are not allowed");
                return NULL;
            }
            shape[axis] = (npy_intp)length;
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "shape must be an integer or a tuple of one or two integers");
        return NULL;
    }

    // `PyArray_Descr::elsize` is private in NumPy 2, so the width comes from the accessor rather than the struct.
    PyArray_Descr *descr = NULL;
    if (dtype_obj == NULL || dtype_obj == Py_None) { descr = PyArray_DescrFromType(NPY_FLOAT32); }
    else if (!PyArray_DescrConverter(dtype_obj, &descr)) { return NULL; }
    if (!descr) return NULL;

    int const type_number = descr->type_num;
    sz_size_t const element_bytes = (sz_size_t)PyDataType_ELSIZE(descr);
    Py_DECREF(descr);
    if (element_bytes == 0) {
        PyErr_SetString(PyExc_TypeError, "dtype must have a fixed, non-zero element width");
        return NULL;
    }

    return new_unified_array(dimensions, shape, type_number, element_bytes);
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
    {"unified_array", (PyCFunction)module_unified_array, METH_VARARGS | METH_KEYWORDS, doc_unified_array},
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
    if (PyType_Ready(&SubstringsType) < 0) return NULL;

    m = PyModule_Create(&stringzillas_module);
    if (m == NULL) return NULL;

#ifdef Py_GIL_DISABLED
    // Declare that this module is safe for free-threaded Python. Every engine serializes its own
    // scratch buffers through the per-object `PyMutex` in `SZS_LOCK_`, and `DeviceScope` hands out
    // its executor by value, so concurrent calls never share mutable kernel state.
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", szs_version_major(), szs_version_minor(), szs_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    comptime_capabilities_ = szs_capabilities_comptime();
    runtime_capabilities_ = szs_capabilities_runtime();
    active_capabilities_ = (sz_capability_t)(comptime_capabilities_ & runtime_capabilities_);

    // Define SIMD capabilities as a tuple
    {
        // Create a Python tuple with the capabilities
        sz_capability_t caps = active_capabilities_;
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
        char caps_str[256];
        sz_capabilities_to_string_implementation_(caps, caps_str, sizeof(caps_str));
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

    Py_INCREF(&SubstringsType);
    if (PyModule_AddObject(m, "Substrings", (PyObject *)&SubstringsType) < 0) {
        Py_XDECREF(&SubstringsType);
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
