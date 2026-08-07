/**
 *  @brief Batch edit-distance and sequence-alignment engines.
 *  @file python/stringzillas/similarities.c
 *  @author Ash Vardanian
 */
#include "stringzillas.h"

/**
 *  @brief  Levenshtein distance computation engine for binary strings.
 */
typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
    szs_levenshtein_distances_t handle;
    char description[32];
    sz_capability_t capabilities;
    SZS_LOCK_FIELD_
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
    sz_capability_t capabilities = active_capabilities_;

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
    if (kernel_punned) {
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    }

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

static PyGetSetDef LevenshteinDistances_getsetters[] = {
    {"__capabilities__", (getter)LevenshteinDistances_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                             /* Sentinel */
};

PyTypeObject LevenshteinDistancesType = {
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

typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
    szs_levenshtein_distances_utf8_t handle;
    char description[32];
    sz_capability_t capabilities;
    SZS_LOCK_FIELD_
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
    sz_capability_t capabilities = active_capabilities_;

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
    if (kernel_punned) {
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    }

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

PyTypeObject LevenshteinDistancesUTF8Type = {
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

/**
 *  @brief  Needleman-Wunsch global alignment scoring engine.
 */
typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
    szs_needleman_wunsch_scores_t handle;
    char description[32];
    sz_capability_t capabilities;
    SZS_LOCK_FIELD_
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
    sz_capability_t capabilities = active_capabilities_;

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
    if (kernel_punned) {
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    }

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

PyTypeObject NeedlemanWunschType = {
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

/**
 *  @brief  Smith-Waterman local alignment scoring engine.
 */
typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
    szs_smith_waterman_scores_t handle;
    char description[32];
    sz_capability_t capabilities;
    SZS_LOCK_FIELD_
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
    sz_capability_t capabilities = active_capabilities_;

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
    if (kernel_punned) {
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        status = kernel_punned(                              //
            self->handle, device_handle,                     //
            kernel_queries_punned, kernel_candidates_punned, //
            kernel_results, kernel_results_row_stride, &error_detail);
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    }

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
PyTypeObject SmithWatermanType = {
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
