/**
 *  @brief Immutable exact Levenshtein dictionary retrieval for Python.
 *  @file python/stringzillas/levenshtein_index.c
 *  @author Guillaume de Rouville
 */
#include "stringzillas.h"

typedef struct {
    PyObject ob_base;
    vectorcallfunc vectorcall;
    szs_levenshtein_index_t handle;
    sz_size_t max_distance;
    sz_capability_t capabilities;
    int utf8;
    SZS_LOCK_FIELD_
} LevenshteinIndex;

typedef sz_status_t (*levenshtein_index_find_kernel_t)( //
    void *, szs_device_scope_t, void const *, sz_size_t, //
    sz_u64_t *, sz_u32_t *, sz_u8_t *, sz_size_t, sz_size_t *, char const **);

typedef struct {
    PyObject *items;
    sz_string_view_t *views;
    sz_sequence_t sequence;
    sz_sequence_u32tape_t u32tape;
    sz_sequence_u64tape_t u64tape;
    void const *input;
    int layout;
    sz_size_t count;
} levenshtein_index_queries_t;

static void levenshtein_index_queries_free(levenshtein_index_queries_t *queries) {
    PyMem_Free(queries->views);
    Py_XDECREF(queries->items);
}

static int levenshtein_index_queries_export(PyObject *object, levenshtein_index_queries_t *queries) {
    memset(queries, 0, sizeof(*queries));
    if (sz_py_export_strings_as_u32tape(object, &queries->u32tape.data, &queries->u32tape.offsets,
                                        &queries->u32tape.count)) {
        queries->input = &queries->u32tape;
        queries->layout = 32;
        queries->count = queries->u32tape.count;
        return 0;
    }
    if (sz_py_export_strings_as_u64tape(object, &queries->u64tape.data, &queries->u64tape.offsets,
                                        &queries->u64tape.count)) {
        queries->input = &queries->u64tape;
        queries->layout = 64;
        queries->count = queries->u64tape.count;
        return 0;
    }
    if (sz_py_export_strings_as_sequence(object, &queries->sequence)) {
        queries->input = &queries->sequence;
        queries->layout = 1;
        queries->count = queries->sequence.count;
        return 0;
    }

    queries->items = PySequence_Fast(object, "queries must be an iterable of string-like objects");
    if (!queries->items) return -1;
    Py_ssize_t const count = PySequence_Fast_GET_SIZE(queries->items);
    if ((size_t)count > SIZE_MAX / sizeof(sz_string_view_t)) {
        PyErr_NoMemory();
        return -1;
    }
    queries->views = (sz_string_view_t *)PyMem_Malloc((size_t)count * sizeof(sz_string_view_t));
    if (!queries->views && count) {
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t index = 0; index != count; ++index)
        if (!sz_py_export_string_like(PySequence_Fast_GET_ITEM(queries->items, index),
                                      &queries->views[index].start, &queries->views[index].length)) {
            PyErr_Format(PyExc_TypeError, "query item %zd is not string-like", index);
            return -1;
        }
    sz_sequence_from_string_views(queries->views, (sz_size_t)count, &queries->sequence);
    queries->input = &queries->sequence;
    queries->layout = 1;
    queries->count = (sz_size_t)count;
    return 0;
}

static void LevenshteinIndex_dealloc(LevenshteinIndex *self) {
    if (self->handle) {
        if (self->utf8) szs_levenshtein_index_utf8_free(self->handle);
        else szs_levenshtein_index_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *LevenshteinIndex_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                             PyObject *kwnames);

static PyObject *LevenshteinIndex_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    LevenshteinIndex *self = (LevenshteinIndex *)type->tp_alloc(type, 0);
    if (self) {
        self->vectorcall = (vectorcallfunc)LevenshteinIndex_vectorcall;
        self->handle = NULL;
        self->max_distance = 0;
        self->capabilities = 0;
        self->utf8 = type == &LevenshteinIndexUTF8Type;
    }
    return (PyObject *)self;
}

static int LevenshteinIndex_init(LevenshteinIndex *self, PyObject *args, PyObject *kwargs) {
    char const *const callable_name = Py_TYPE(self)->tp_name;
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count > 3) {
        PyErr_Format(PyExc_TypeError, "%s takes at most 3 arguments (%zd given)", callable_name, positional_count);
        return -1;
    }
    PyObject *dictionary_obj = positional_count > 0 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *max_distance_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *capabilities_obj = positional_count > 2 ? PyTuple_GET_ITEM(args, 2) : NULL;
    if (kwargs) {
        Py_ssize_t cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "dictionary") == 0) {
                if (dictionary_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'dictionary'", callable_name);
                    return -1;
                }
                dictionary_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "max_distance") == 0) {
                if (max_distance_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'max_distance'", callable_name);
                    return -1;
                }
                max_distance_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                if (capabilities_obj) {
                    PyErr_Format(PyExc_TypeError, "%s got multiple values for argument 'capabilities'", callable_name);
                    return -1;
                }
                capabilities_obj = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }
    if (self->handle) {
        PyErr_SetString(PyExc_RuntimeError, "LevenshteinIndex is already initialized");
        return -1;
    }
    if (!dictionary_obj) {
        PyErr_Format(PyExc_TypeError, "%s missing required argument 'dictionary'", callable_name);
        return -1;
    }

    sz_size_t max_distance = 2;
    if (max_distance_obj && max_distance_obj != Py_None) {
        max_distance = PyLong_AsSize_t(max_distance_obj);
        if (PyErr_Occurred()) return -1;
    }
    sz_capability_t capabilities = active_capabilities_;
    if (capabilities_obj && capabilities_obj != Py_None &&
        parse_and_intersect_capabilities(capabilities_obj, &capabilities) != 0)
        return -1;
    capabilities = (sz_capability_t)(capabilities & sz_caps_sp_k);

    levenshtein_index_queries_t dictionary;
    if (levenshtein_index_queries_export(dictionary_obj, &dictionary) != 0) {
        levenshtein_index_queries_free(&dictionary);
        return -1;
    }
    char const *error_detail = NULL;
    sz_status_t status;
    if (dictionary.layout == 32)
        status = self->utf8 ? szs_levenshtein_index_utf8_init_u32tape(
                                  &dictionary.u32tape, max_distance, NULL, capabilities, &self->handle, &error_detail)
                            : szs_levenshtein_index_init_u32tape(
                                  &dictionary.u32tape, max_distance, NULL, capabilities, &self->handle, &error_detail);
    else if (dictionary.layout == 64)
        status = self->utf8 ? szs_levenshtein_index_utf8_init_u64tape(
                                  &dictionary.u64tape, max_distance, NULL, capabilities, &self->handle, &error_detail)
                            : szs_levenshtein_index_init_u64tape(
                                  &dictionary.u64tape, max_distance, NULL, capabilities, &self->handle, &error_detail);
    else
        status = self->utf8 ? szs_levenshtein_index_utf8_init(
                                  &dictionary.sequence, max_distance, NULL, capabilities, &self->handle, &error_detail)
                            : szs_levenshtein_index_init(
                                  &dictionary.sequence, max_distance, NULL, capabilities, &self->handle, &error_detail);
    levenshtein_index_queries_free(&dictionary);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "LevenshteinIndex construction");
        return -1;
    }
    self->max_distance = max_distance;
    self->capabilities = capabilities;
    return 0;
}

static PyObject *LevenshteinIndex_repr(LevenshteinIndex *self) {
    return PyUnicode_FromFormat("%s(max_distance=%zu)", self->utf8 ? "LevenshteinIndexUTF8" : "LevenshteinIndex",
                                self->max_distance);
}

static PyObject *LevenshteinIndex_get_capabilities(LevenshteinIndex *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

static int parse_levenshtein_index_call_args(                                  //
    char const *callable_name, PyObject *const *args, size_t nargsf,           //
    PyObject *kwnames, PyObject **queries_out, PyObject **bound_out, PyObject **device_out) {
    Py_ssize_t const positional_count = PyVectorcall_NARGS(nargsf);
    if (positional_count < 1 || positional_count > 3) {
        PyErr_Format(PyExc_TypeError, "%s() takes 1 to 3 positional arguments, got %zd", callable_name,
                     positional_count);
        return -1;
    }
    PyObject *queries = args[0];
    PyObject *bound = positional_count > 1 ? args[1] : NULL;
    PyObject *device = positional_count > 2 ? args[2] : NULL;
    if (kwnames) {
        Py_ssize_t const keyword_count = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t index = 0; index != keyword_count; ++index) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, index);
            PyObject *value = args[positional_count + index];
            if (PyUnicode_CompareWithASCIIString(key, "queries") == 0) {
                PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'queries'", callable_name);
                return -1;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "bound") == 0) {
                if (positional_count > 1) {
                    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'bound'", callable_name);
                    return -1;
                }
                bound = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "device") == 0) {
                if (positional_count > 2) {
                    PyErr_Format(PyExc_TypeError, "%s() got multiple values for argument 'device'", callable_name);
                    return -1;
                }
                device = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", callable_name, key);
                return -1;
            }
        }
    }
    *queries_out = queries;
    *bound_out = bound;
    *device_out = device;
    return 0;
}

static PyObject *LevenshteinIndex_vectorcall(PyObject *callable, PyObject *const *args, size_t nargsf,
                                             PyObject *kwnames) {
    LevenshteinIndex *self = (LevenshteinIndex *)callable;
    PyObject *queries_obj = NULL, *bound_obj = NULL, *device_obj = NULL;
    if (parse_levenshtein_index_call_args("LevenshteinIndex.__call__", args, nargsf, kwnames, &queries_obj,
                                          &bound_obj, &device_obj) != 0)
        return NULL;

    sz_size_t bound = self->max_distance;
    if (bound_obj && bound_obj != Py_None) {
        bound = PyLong_AsSize_t(bound_obj);
        if (PyErr_Occurred()) return NULL;
    }
    DeviceScope *device_scope = NULL;
    if (device_obj && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return NULL;
        }
        device_scope = (DeviceScope *)device_obj;
    }

    levenshtein_index_queries_t queries;
    if (levenshtein_index_queries_export(queries_obj, &queries) != 0) {
        levenshtein_index_queries_free(&queries);
        return NULL;
    }

    levenshtein_index_find_kernel_t kernel;
    if (self->utf8)
        kernel = queries.layout == 32 ? (levenshtein_index_find_kernel_t)szs_levenshtein_index_utf8_find_u32tape
                 : queries.layout == 64 ? (levenshtein_index_find_kernel_t)szs_levenshtein_index_utf8_find_u64tape
                                        : (levenshtein_index_find_kernel_t)szs_levenshtein_index_utf8_find;
    else
        kernel = queries.layout == 32 ? (levenshtein_index_find_kernel_t)szs_levenshtein_index_find_u32tape
                 : queries.layout == 64 ? (levenshtein_index_find_kernel_t)szs_levenshtein_index_find_u64tape
                                        : (levenshtein_index_find_kernel_t)szs_levenshtein_index_find;

    PyObject *query_indices = NULL;
    PyObject *dictionary_indices = NULL;
    PyObject *distances = NULL;
    if (queries.count > (sz_size_t)NPY_MAX_INTP / 8) {
        PyErr_SetString(PyExc_OverflowError, "too many queries for a NumPy result");
        goto cleanup;
    }
    sz_size_t capacity = queries.count * 8;
    npy_intp shape[1] = {(npy_intp)capacity};
    query_indices = PyArray_SimpleNew(1, shape, NPY_UINT64);
    dictionary_indices = PyArray_SimpleNew(1, shape, NPY_UINT32);
    distances = PyArray_SimpleNew(1, shape, NPY_UINT8);
    if (!query_indices || !dictionary_indices || !distances) goto cleanup;

    szs_device_scope_t device = device_scope ? device_scope->handle : default_device_scope;
    sz_size_t matches_found = 0;
    char const *error_detail = NULL;
    if (device_scope) SZS_LOCK_(&device_scope->lock);
    SZS_LOCK_(&self->lock);
    sz_status_t status = kernel(self->handle, device, queries.input, bound,
                                (sz_u64_t *)PyArray_DATA((PyArrayObject *)query_indices),
                                (sz_u32_t *)PyArray_DATA((PyArrayObject *)dictionary_indices),
                                (sz_u8_t *)PyArray_DATA((PyArrayObject *)distances), capacity,
                                &matches_found, &error_detail);
    SZS_UNLOCK_(&self->lock);
    if (device_scope) SZS_UNLOCK_(&device_scope->lock);

    if (status == sz_unexpected_dimensions_k && matches_found > capacity) {
        Py_DECREF(query_indices);
        Py_DECREF(dictionary_indices);
        Py_DECREF(distances);
        if (matches_found > (sz_size_t)NPY_MAX_INTP) {
            PyErr_SetString(PyExc_OverflowError, "too many matches for a NumPy result");
            goto cleanup;
        }
        shape[0] = (npy_intp)matches_found;
        query_indices = PyArray_SimpleNew(1, shape, NPY_UINT64);
        dictionary_indices = PyArray_SimpleNew(1, shape, NPY_UINT32);
        distances = PyArray_SimpleNew(1, shape, NPY_UINT8);
        if (!query_indices || !dictionary_indices || !distances) goto cleanup;
        capacity = matches_found;
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        status = kernel(self->handle, device, queries.input, bound,
                        (sz_u64_t *)PyArray_DATA((PyArrayObject *)query_indices),
                        (sz_u32_t *)PyArray_DATA((PyArrayObject *)dictionary_indices),
                        (sz_u8_t *)PyArray_DATA((PyArrayObject *)distances), capacity,
                        &matches_found, &error_detail);
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    }
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "LevenshteinIndex search");
        goto cleanup;
    }

    npy_intp final_length = (npy_intp)matches_found;
    PyArray_Dims final_shape = {&final_length, 1};
    PyObject *resized = PyArray_Resize((PyArrayObject *)query_indices, &final_shape, 0, NPY_CORDER);
    if (!resized) goto cleanup;
    Py_DECREF(resized);
    resized = PyArray_Resize((PyArrayObject *)dictionary_indices, &final_shape, 0, NPY_CORDER);
    if (!resized) goto cleanup;
    Py_DECREF(resized);
    resized = PyArray_Resize((PyArrayObject *)distances, &final_shape, 0, NPY_CORDER);
    if (!resized) goto cleanup;
    Py_DECREF(resized);
    PyObject *result = PyTuple_Pack(3, query_indices, dictionary_indices, distances);
    Py_DECREF(query_indices);
    Py_DECREF(dictionary_indices);
    Py_DECREF(distances);
    levenshtein_index_queries_free(&queries);
    return result;

cleanup:
    Py_XDECREF(query_indices);
    Py_XDECREF(dictionary_indices);
    Py_XDECREF(distances);
    levenshtein_index_queries_free(&queries);
    return NULL;
}

static char const doc_LevenshteinIndex[] =
    "LevenshteinIndex(dictionary, max_distance=2, capabilities=None)\n\n"
    "Build an exact byte-level index for a dictionary that will be searched many times. Calling it with "
    "``(queries, bound=None, device=None)`` returns three compact arrays containing the query IDs, dictionary "
    "IDs, and exact distances for every match. Duplicate dictionary values keep separate IDs.";

static char const doc_LevenshteinIndexUTF8[] =
    "LevenshteinIndexUTF8(dictionary, max_distance=2, capabilities=None)\n\n"
    "Unicode counterpart of LevenshteinIndex. Inputs must be valid UTF-8 and edits are counted between Unicode "
    "characters rather than encoded bytes.";

static PyGetSetDef LevenshteinIndex_getsetters[] = {
    {"__capabilities__", (getter)LevenshteinIndex_get_capabilities, NULL, doc_capabilities, NULL},
    {NULL}
};

#define SZS_LEVENSHTEIN_INDEX_TYPE_(name, doc)                                    \
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = name,                                \
    .tp_doc = doc,                                                                 \
    .tp_basicsize = sizeof(LevenshteinIndex),                                      \
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_VECTORCALL,                   \
    .tp_vectorcall_offset = offsetof(LevenshteinIndex, vectorcall),                \
    .tp_new = LevenshteinIndex_new,                                                \
    .tp_init = (initproc)LevenshteinIndex_init,                                    \
    .tp_dealloc = (destructor)LevenshteinIndex_dealloc,                            \
    .tp_call = PyVectorcall_Call,                                                   \
    .tp_repr = (reprfunc)LevenshteinIndex_repr,                                    \
    .tp_getset = LevenshteinIndex_getsetters

PyTypeObject LevenshteinIndexType = {
    SZS_LEVENSHTEIN_INDEX_TYPE_("stringzillas.LevenshteinIndex", doc_LevenshteinIndex)};
PyTypeObject LevenshteinIndexUTF8Type = {
    SZS_LEVENSHTEIN_INDEX_TYPE_("stringzillas.LevenshteinIndexUTF8", doc_LevenshteinIndexUTF8)};

#undef SZS_LEVENSHTEIN_INDEX_TYPE_
