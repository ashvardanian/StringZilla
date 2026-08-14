/**
 *  @brief Multi-pattern Aho-Corasick search engine over a whole dictionary of needles.
 *  @file python/stringzillas/substrings.c
 *  @author Ash Vardanian
 */
#include "stringzillas.h"

/**
 *  @brief  Multi-pattern search engine, compiled once from a needle set and reused across calls.
 *
 *  Unlike the other engines here, construction is a device operation - the automaton's tier split is sized
 *  against the cache the walk reads through, and a CUDA automaton is uploaded to a device - so `__init__`
 *  takes a scope of its own. The device buffers are managed memory, so a later call may name a different
 *  GPU; only a CPU scope on a GPU-built engine is refused, and the engine reports that itself.
 */
typedef struct {
    PyObject ob_base;
    szs_substrings_t handle;
    char description[64];
    sz_capability_t capabilities;
    sz_size_t needles_count; //? Fixed once built; every per-needle array is validated against it.
    SZS_LOCK_FIELD_
} Substrings;

#pragma region Argument Parsing

/** @brief Narrows a policy name to the C enumeration, so callers never spell an integer. */
static int parse_overlap_policy(PyObject *policy_obj, szs_substrings_overlap_policy_t *result) {
    if (policy_obj == NULL || policy_obj == Py_None) {
        *result = szs_substrings_overlapping_k;
        return 0;
    }
    char const *name = PyUnicode_AsUTF8(policy_obj);
    if (!name) {
        PyErr_SetString(PyExc_TypeError, "overlap policy must be a string");
        return -1;
    }
    if (strcmp(name, "overlapping") == 0) { *result = szs_substrings_overlapping_k; }
    else if (strcmp(name, "leftmost-longest") == 0) { *result = szs_substrings_leftmost_longest_k; }
    else if (strcmp(name, "leftmost-first") == 0) { *result = szs_substrings_leftmost_first_k; }
    else {
        PyErr_Format(PyExc_ValueError,
                     "Unknown overlap policy '%s', expected 'overlapping', 'leftmost-longest', or 'leftmost-first'",
                     name);
        return -1;
    }
    return 0;
}

/** @brief Narrows a sensitivity name to the C enumeration. */
static int parse_case_sensitivity(PyObject *sensitivity_obj, szs_substrings_case_sensitivity_t *result) {
    if (sensitivity_obj == NULL || sensitivity_obj == Py_None) {
        *result = szs_substrings_cased_k;
        return 0;
    }
    char const *name = PyUnicode_AsUTF8(sensitivity_obj);
    if (!name) {
        PyErr_SetString(PyExc_TypeError, "case_sensitivity must be a string");
        return -1;
    }
    if (strcmp(name, "cased") == 0) { *result = szs_substrings_cased_k; }
    else if (strcmp(name, "uncased") == 0) { *result = szs_substrings_uncased_k; }
    else {
        PyErr_Format(PyExc_ValueError, "Unknown case sensitivity '%s', expected 'cased' or 'uncased'", name);
        return -1;
    }
    return 0;
}

/** @brief Resolves an optional `DeviceScope` argument to its handle, or the module default. */
static int parse_device_scope(PyObject *device_obj, DeviceScope **scope_out, szs_device_scope_t *handle_out) {
    *scope_out = NULL;
    if (device_obj != NULL && device_obj != Py_None) {
        if (!PyObject_TypeCheck(device_obj, &DeviceScopeType)) {
            PyErr_SetString(PyExc_TypeError, "device must be a DeviceScope instance");
            return -1;
        }
        *scope_out = (DeviceScope *)device_obj;
    }
    *handle_out = *scope_out ? (*scope_out)->handle : default_device_scope;
    return 0;
}

/**
 *  @brief  Points @p kernels at whichever haystack shape @p texts_obj carries, or fails with a type error.
 *
 *  The probe order matches every other engine here: the two tape layouts hand their offsets straight to the
 *  kernel, and the callback-addressed sequence is the fallback a reordered `Strs` takes.
 */
typedef struct {
    void *punned;
    sz_size_t count;
    int is_u32tape;
    int is_u64tape;
    sz_sequence_u32tape_t u32tape;
    sz_sequence_u64tape_t u64tape;
    sz_sequence_t sequence;
} haystacks_view_t;

/**
 *  @brief Views a `Strs` as whichever shape the engine takes, re-pinning it to unified memory first when the
 *         engine holds CUDA capabilities.
 *
 *  The sequence exporter accepts every layout, so it doubles as the type test - and it has to run before the
 *  swap, which reallocates the tape a pointer-capturing view would otherwise be left pointing into.
 */
static int parse_haystacks(PyObject *texts_obj, sz_capability_t capabilities, haystacks_view_t *view) {
    if (!sz_py_export_strings_as_sequence(texts_obj, &view->sequence)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs object, got %s. Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(texts_obj)->tp_name);
        return -1;
    }
    if (requires_unified_memory(capabilities))
        if (!try_swap_to_unified_allocator(texts_obj)) return -1;

    view->is_u32tape = sz_py_export_strings_as_u32tape( //
        texts_obj, &view->u32tape.data, &view->u32tape.offsets, &view->u32tape.count);
    if (view->is_u32tape) {
        view->punned = &view->u32tape;
        view->count = view->u32tape.count;
        return 0;
    }
    view->is_u64tape = sz_py_export_strings_as_u64tape( //
        texts_obj, &view->u64tape.data, &view->u64tape.offsets, &view->u64tape.count);
    if (view->is_u64tape) {
        view->punned = &view->u64tape;
        view->count = view->u64tape.count;
        return 0;
    }
    view->punned = &view->sequence;
    view->count = view->sequence.count;
    return 0;
}

#pragma endregion Argument Parsing

#pragma region Lifetime

static void Substrings_dealloc(Substrings *self) {
    if (self->handle) {
        szs_substrings_free(self->handle);
        self->handle = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Substrings_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    Substrings *self = (Substrings *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->handle = NULL;
        self->description[0] = '\0';
        self->capabilities = 0;
        self->needles_count = 0;
    }
    return (PyObject *)self;
}

static int Substrings_init(Substrings *self, PyObject *args, PyObject *kwargs) {
    PyObject *needles_obj = NULL, *sensitivity_obj = NULL, *device_obj = NULL, *capabilities_tuple = NULL;
    sz_capability_t capabilities = active_capabilities_;

    static char *kwlist[] = {"needles", "case_sensitivity", "device", "capabilities", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OOO", kwlist, &needles_obj, &sensitivity_obj, &device_obj,
                                     &capabilities_tuple))
        return -1;

    if (capabilities_tuple)
        if (parse_and_intersect_capabilities(capabilities_tuple, &capabilities) != 0) return -1;

    szs_substrings_case_sensitivity_t sensitivity;
    if (parse_case_sensitivity(sensitivity_obj, &sensitivity) != 0) return -1;

    DeviceScope *device_scope = NULL;
    szs_device_scope_t device_handle = NULL;
    if (parse_device_scope(device_obj, &device_scope, &device_handle) != 0) return -1;

    // The mask alone picks the backend, so a CPU scope must not leave the CUDA bit standing: the engine would
    // be built for a device the operations then refuse to run on.
    if (device_handle) {
        sz_capability_t scope_capabilities = 0;
        char const *scope_error = NULL;
        sz_status_t const scope_status =
            szs_device_scope_get_capabilities(device_handle, &scope_capabilities, &scope_error);
        if (scope_status != sz_success_k) {
            set_stringzilla_error(scope_status, scope_error, "Device scope capabilities");
            return -1;
        }
        capabilities &= scope_capabilities;
    }

    // Needles reach the engine as a callback-addressed sequence whatever layout they arrived in - the
    // automaton has no tape overload, since it walks the needles once at construction and never again.
    sz_sequence_t needles_sequence;
    if (!sz_py_export_strings_as_sequence(needles_obj, &needles_sequence)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs object, got %s. Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(needles_obj)->tp_name);
        return -1;
    }
    if (requires_unified_memory(capabilities))
        if (!try_swap_to_unified_allocator(needles_obj)) return -1;

    char const *error_detail = NULL;
    sz_status_t status = szs_substrings_init(NULL, capabilities, &self->handle, &error_detail);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Substrings construction");
        return -1;
    }

    // Indexing is where the automaton is tiered against a device, so it takes the scope's lock like every
    // operation does. A failure leaves `needles_count` at zero and the handle for `dealloc` to release.
    if (device_scope) SZS_LOCK_(&device_scope->lock);
    status = szs_substrings_index(self->handle, &needles_sequence, sensitivity, device_handle, &error_detail);
    if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Substrings indexing");
        return -1;
    }

    self->capabilities = capabilities;
    self->needles_count = needles_sequence.count;
    snprintf(self->description, sizeof(self->description), "%zu needles, %s", (size_t)self->needles_count,
             sensitivity == szs_substrings_uncased_k ? "uncased" : "cased");
    return 0;
}

static PyObject *Substrings_repr(Substrings *self) {
    return PyUnicode_FromFormat("stringzillas.Substrings(%s)", self->description);
}

static PyObject *Substrings_get_capabilities(Substrings *self, void *closure) {
    return capabilities_to_tuple(self->capabilities);
}

#pragma endregion Lifetime

#pragma region Operations

static PyObject *Substrings_count(Substrings *self, PyObject *args, PyObject *kwargs) {
    PyObject *haystacks_obj = NULL, *device_obj = NULL, *policy_obj = NULL;
    static char *kwlist[] = {"haystacks", "device", "policy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &haystacks_obj, &device_obj, &policy_obj))
        return NULL;

    szs_substrings_overlap_policy_t policy;
    if (parse_overlap_policy(policy_obj, &policy) != 0) return NULL;
    DeviceScope *device_scope = NULL;
    szs_device_scope_t device_handle = NULL;
    if (parse_device_scope(device_obj, &device_scope, &device_handle) != 0) return NULL;

    sz_bool_t const need_unified = requires_unified_memory(self->capabilities);
    haystacks_view_t haystacks;
    if (parse_haystacks(haystacks_obj, self->capabilities, &haystacks) != 0) return NULL;

    npy_intp dims[1] = {(npy_intp)haystacks.count};
    PyArrayObject *counts_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!counts_array) return PyErr_NoMemory();

    sz_memory_allocator_t *out_alloc = need_unified ? &unified_allocator : &default_allocator;
    sz_size_t const counts_bytes = haystacks.count * sizeof(sz_size_t);
    sz_size_t matches_total = 0;
    if (counts_bytes > 0) {
        sz_size_t *counts = (sz_size_t *)out_alloc->allocate(counts_bytes, out_alloc->handle);
        if (!counts) {
            Py_DECREF(counts_array);
            return PyErr_NoMemory();
        }

        char const *error_detail = NULL;
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        sz_status_t status =
            haystacks.is_u32tape
                ? szs_substrings_count_u32tape(self->handle, device_handle, &haystacks.u32tape, policy, counts,
                                               &matches_total, &error_detail)
                : (haystacks.is_u64tape
                       ? szs_substrings_count_u64tape(self->handle, device_handle, &haystacks.u64tape, policy, counts,
                                                      &matches_total, &error_detail)
                       : szs_substrings_count(self->handle, device_handle, &haystacks.sequence, policy, counts,
                                              &matches_total, &error_detail));
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
        if (status != sz_success_k) {
            out_alloc->free(counts, counts_bytes, out_alloc->handle);
            Py_DECREF(counts_array);
            set_stringzilla_error(status, error_detail, "Substrings counting");
            return NULL;
        }

        memcpy(PyArray_DATA(counts_array), counts, counts_bytes);
        out_alloc->free(counts, counts_bytes, out_alloc->handle);
    }

    PyObject *result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(counts_array);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, (PyObject *)counts_array);
    PyTuple_SET_ITEM(result, 1, PyLong_FromSize_t((size_t)matches_total));
    return result;
}

static PyObject *Substrings_find(Substrings *self, PyObject *args, PyObject *kwargs) {
    PyObject *haystacks_obj = NULL, *device_obj = NULL, *policy_obj = NULL;
    static char *kwlist[] = {"haystacks", "device", "policy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|OO", kwlist, &haystacks_obj, &device_obj, &policy_obj))
        return NULL;

    szs_substrings_overlap_policy_t policy;
    if (parse_overlap_policy(policy_obj, &policy) != 0) return NULL;
    DeviceScope *device_scope = NULL;
    szs_device_scope_t device_handle = NULL;
    if (parse_device_scope(device_obj, &device_scope, &device_handle) != 0) return NULL;

    sz_bool_t const need_unified = requires_unified_memory(self->capabilities);
    haystacks_view_t haystacks;
    if (parse_haystacks(haystacks_obj, self->capabilities, &haystacks) != 0) return NULL;

    // A zero capacity is the engine's own size query: it reports what it wanted and writes nothing, so the
    // buffer below is sized from the walk rather than from a guess the caller would have to make.
    char const *error_detail = NULL;
    sz_size_t matches_found = 0;
    if (device_scope) SZS_LOCK_(&device_scope->lock);
    SZS_LOCK_(&self->lock);
    sz_status_t status =
        haystacks.is_u32tape
            ? szs_substrings_find_u32tape(self->handle, device_handle, &haystacks.u32tape, policy, NULL, 0,
                                          &matches_found, &error_detail)
            : (haystacks.is_u64tape ? szs_substrings_find_u64tape(self->handle, device_handle, &haystacks.u64tape,
                                                                  policy, NULL, 0, &matches_found, &error_detail)
                                    : szs_substrings_find(self->handle, device_handle, &haystacks.sequence, policy,
                                                          NULL, 0, &matches_found, &error_detail));
    SZS_UNLOCK_(&self->lock);
    if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    if (status != sz_success_k && status != sz_unexpected_dimensions_k) {
        set_stringzilla_error(status, error_detail, "Substrings sizing");
        return NULL;
    }

    npy_intp dims[1] = {(npy_intp)matches_found};
    PyArrayObject *haystack_indices = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    PyArrayObject *needle_indices = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    PyArrayObject *byte_offsets = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    PyArrayObject *byte_lengths = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!haystack_indices || !needle_indices || !byte_offsets || !byte_lengths) {
        Py_XDECREF(haystack_indices);
        Py_XDECREF(needle_indices);
        Py_XDECREF(byte_offsets);
        Py_XDECREF(byte_lengths);
        return PyErr_NoMemory();
    }

    sz_memory_allocator_t *out_alloc = need_unified ? &unified_allocator : &default_allocator;
    sz_size_t const matches_bytes = matches_found * sizeof(szs_substrings_match_t);
    if (matches_bytes > 0) {
        szs_substrings_match_t *matches =
            (szs_substrings_match_t *)out_alloc->allocate(matches_bytes, out_alloc->handle);
        if (!matches) {
            Py_DECREF(haystack_indices);
            Py_DECREF(needle_indices);
            Py_DECREF(byte_offsets);
            Py_DECREF(byte_lengths);
            return PyErr_NoMemory();
        }

        sz_size_t written = 0;
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        status = haystacks.is_u32tape
                     ? szs_substrings_find_u32tape(self->handle, device_handle, &haystacks.u32tape, policy, matches,
                                                   matches_found, &written, &error_detail)
                     : (haystacks.is_u64tape
                            ? szs_substrings_find_u64tape(self->handle, device_handle, &haystacks.u64tape, policy,
                                                          matches, matches_found, &written, &error_detail)
                            : szs_substrings_find(self->handle, device_handle, &haystacks.sequence, policy, matches,
                                                  matches_found, &written, &error_detail));
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
        if (status != sz_success_k) {
            out_alloc->free(matches, matches_bytes, out_alloc->handle);
            Py_DECREF(haystack_indices);
            Py_DECREF(needle_indices);
            Py_DECREF(byte_offsets);
            Py_DECREF(byte_lengths);
            set_stringzilla_error(status, error_detail, "Substrings search");
            return NULL;
        }

        // One column per field, so a caller wanting only the needle identities takes that array alone.
        sz_u64_t *haystack_data = (sz_u64_t *)PyArray_DATA(haystack_indices);
        sz_u64_t *needle_data = (sz_u64_t *)PyArray_DATA(needle_indices);
        sz_u64_t *offset_data = (sz_u64_t *)PyArray_DATA(byte_offsets);
        sz_u64_t *length_data = (sz_u64_t *)PyArray_DATA(byte_lengths);
        for (sz_size_t index = 0; index < written; ++index) {
            haystack_data[index] = (sz_u64_t)matches[index].haystack_index;
            needle_data[index] = (sz_u64_t)matches[index].needle_index;
            offset_data[index] = (sz_u64_t)matches[index].byte_offset;
            length_data[index] = (sz_u64_t)matches[index].byte_length;
        }
        out_alloc->free(matches, matches_bytes, out_alloc->handle);
    }

    PyObject *result = PyTuple_New(4);
    if (!result) {
        Py_DECREF(haystack_indices);
        Py_DECREF(needle_indices);
        Py_DECREF(byte_offsets);
        Py_DECREF(byte_lengths);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, (PyObject *)haystack_indices);
    PyTuple_SET_ITEM(result, 1, (PyObject *)needle_indices);
    PyTuple_SET_ITEM(result, 2, (PyObject *)byte_offsets);
    PyTuple_SET_ITEM(result, 3, (PyObject *)byte_lengths);
    return result;
}

static PyObject *Substrings_score_bm25(Substrings *self, PyObject *args, PyObject *kwargs) {
    PyObject *haystacks_obj = NULL, *weights_obj = NULL, *device_obj = NULL, *lengths_obj = NULL;
    // The corpus mean is a property of the corpus, not a tunable, so it has no default and stays required;
    // `length_normalization=0.0` is how a caller opts out of needing one, and the engine refuses the pair
    // that asks to normalize without a mean.
    double average_document_length = 0.0;
    double term_frequency_saturation = 1.2, length_normalization = 0.75;
    static char *kwlist[] = {"haystacks",
                             "needle_weights",
                             "average_document_length",
                             "device",
                             "document_lengths",
                             "term_frequency_saturation",
                             "length_normalization",
                             NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOd|OOdd", kwlist, &haystacks_obj, &weights_obj,
                                     &average_document_length, &device_obj, &lengths_obj,
                                     &term_frequency_saturation, &length_normalization))
        return NULL;

    DeviceScope *device_scope = NULL;
    szs_device_scope_t device_handle = NULL;
    if (parse_device_scope(device_obj, &device_scope, &device_handle) != 0) return NULL;

    if (!numpy_available || !PyArray_Check(weights_obj)) {
        PyErr_SetString(PyExc_TypeError, "needle_weights must be a NumPy array of float32, one entry per needle");
        return NULL;
    }
    PyArrayObject *weights_array = (PyArrayObject *)weights_obj;
    if (PyArray_TYPE(weights_array) != NPY_FLOAT32 || PyArray_NDIM(weights_array) != 1 ||
        !PyArray_ISCARRAY_RO(weights_array)) {
        PyErr_SetString(PyExc_TypeError, "needle_weights must be a contiguous 1-D float32 array");
        return NULL;
    }
    if ((sz_size_t)PyArray_DIM(weights_array, 0) != self->needles_count) {
        PyErr_Format(PyExc_ValueError, "Expected one weight per needle: %zu given, %zu needed",
                     (size_t)PyArray_DIM(weights_array, 0), (size_t)self->needles_count);
        return NULL;
    }

    sz_f32_t const *document_lengths = NULL;
    if (lengths_obj != NULL && lengths_obj != Py_None) {
        if (!PyArray_Check(lengths_obj) || PyArray_TYPE((PyArrayObject *)lengths_obj) != NPY_FLOAT32 ||
            !PyArray_ISCARRAY_RO((PyArrayObject *)lengths_obj)) {
            PyErr_SetString(PyExc_TypeError, "document_lengths must be a contiguous 1-D float32 array");
            return NULL;
        }
        document_lengths = (sz_f32_t const *)PyArray_DATA((PyArrayObject *)lengths_obj);
    }

    sz_bool_t const need_unified = requires_unified_memory(self->capabilities);
    haystacks_view_t haystacks;
    if (parse_haystacks(haystacks_obj, self->capabilities, &haystacks) != 0) return NULL;

    npy_intp dims[1] = {(npy_intp)haystacks.count};
    PyArrayObject *scores_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!scores_array) return PyErr_NoMemory();

    szs_substrings_bm25_t parameters;
    parameters.term_frequency_saturation = (sz_f32_t)term_frequency_saturation;
    parameters.length_normalization = (sz_f32_t)length_normalization;
    parameters.average_document_length = (sz_f32_t)average_document_length;

    sz_memory_allocator_t *out_alloc = need_unified ? &unified_allocator : &default_allocator;
    sz_size_t const scores_bytes = haystacks.count * sizeof(sz_f32_t);
    if (scores_bytes > 0) {
        sz_f32_t *scores = (sz_f32_t *)out_alloc->allocate(scores_bytes, out_alloc->handle);
        if (!scores) {
            Py_DECREF(scores_array);
            return PyErr_NoMemory();
        }

        sz_f32_t const *weights = (sz_f32_t const *)PyArray_DATA(weights_array);
        char const *error_detail = NULL;
        if (device_scope) SZS_LOCK_(&device_scope->lock);
        SZS_LOCK_(&self->lock);
        sz_status_t status =
            haystacks.is_u32tape
                ? szs_substrings_score_bm25_u32tape(self->handle, device_handle, &haystacks.u32tape, document_lengths,
                                                    parameters, weights, scores, &error_detail)
                : (haystacks.is_u64tape ? szs_substrings_score_bm25_u64tape(self->handle, device_handle,
                                                                            &haystacks.u64tape, document_lengths,
                                                                            parameters, weights, scores, &error_detail)
                                        : szs_substrings_score_bm25(self->handle, device_handle, &haystacks.sequence,
                                                                    document_lengths, parameters, weights, scores,
                                                                    &error_detail));
        SZS_UNLOCK_(&self->lock);
        if (device_scope) SZS_UNLOCK_(&device_scope->lock);
        if (status != sz_success_k) {
            out_alloc->free(scores, scores_bytes, out_alloc->handle);
            Py_DECREF(scores_array);
            set_stringzilla_error(status, error_detail, "Substrings BM25 scoring");
            return NULL;
        }

        memcpy(PyArray_DATA(scores_array), scores, scores_bytes);
        out_alloc->free(scores, scores_bytes, out_alloc->handle);
    }

    return (PyObject *)scores_array;
}

/** @brief What a rewrite of nothing returns: empty bytes beside the one boundary an empty tape carries. */
static PyObject *empty_rewrite_result(void) {
    npy_intp dims[1] = {1};
    PyArrayObject *offsets_array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_UINT64);
    if (!offsets_array) return PyErr_NoMemory();
    *(sz_u64_t *)PyArray_DATA(offsets_array) = 0;

    PyObject *rewritten = PyBytes_FromStringAndSize(NULL, 0);
    if (!rewritten) {
        Py_DECREF(offsets_array);
        return NULL;
    }
    PyObject *result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(rewritten);
        Py_DECREF(offsets_array);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, rewritten);
    PyTuple_SET_ITEM(result, 1, (PyObject *)offsets_array);
    return result;
}

static PyObject *Substrings_replace(Substrings *self, PyObject *args, PyObject *kwargs) {
    PyObject *haystacks_obj = NULL, *replacements_obj = NULL, *device_obj = NULL, *policy_obj = NULL;
    static char *kwlist[] = {"haystacks", "replacements", "device", "policy", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|OO", kwlist, &haystacks_obj, &replacements_obj, &device_obj,
                                     &policy_obj))
        return NULL;

    // A rewrite needs a cover whose matches share no bytes, so the default is the leftmost one rather than
    // the overlapping walk the search entry points default to. An overlapping rewrite is not a function, so
    // naming one is an argument error here rather than an engine refusal further down.
    szs_substrings_overlap_policy_t policy = szs_substrings_leftmost_longest_k;
    if (policy_obj != NULL && policy_obj != Py_None)
        if (parse_overlap_policy(policy_obj, &policy) != 0) return NULL;
    if (policy == szs_substrings_overlapping_k) {
        PyErr_SetString(PyExc_ValueError,
                        "Rewriting needs a cover whose matches share no bytes: "
                        "policy must be 'leftmost-longest' or 'leftmost-first'");
        return NULL;
    }

    DeviceScope *device_scope = NULL;
    szs_device_scope_t device_handle = NULL;
    if (parse_device_scope(device_obj, &device_scope, &device_handle) != 0) return NULL;

    sz_sequence_t replacements_sequence;
    if (!sz_py_export_strings_as_sequence(replacements_obj, &replacements_sequence)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs object, got %s. Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(replacements_obj)->tp_name);
        return NULL;
    }

    sz_bool_t const need_unified = requires_unified_memory(self->capabilities);
    haystacks_view_t haystacks;
    if (parse_haystacks(haystacks_obj, self->capabilities, &haystacks) != 0) return NULL;

    // An empty batch is a legal batch with nothing to rewrite, and its layout is not observable - an empty
    // `Strs` is fragmented until an allocator swap turns it into a tape whose offsets are still NULL - so it
    // is answered here rather than being sorted into a layout that then reads a trailing offset that is not
    // there.
    if (haystacks.count == 0) return empty_rewrite_result();

    if (haystacks.is_u32tape == 0 && haystacks.is_u64tape == 0) {
        PyErr_SetString(PyExc_TypeError,
                        "Rewriting takes a tape in and produces a tape out, so a reordered Strs is not accepted");
        return NULL;
    }

    // The bound is arithmetic over the needle set, so one call sizes the tape and the rewrite below cannot
    // then be refused for capacity.
    sz_size_t input_bytes = haystacks.is_u32tape
                                ? (sz_size_t)haystacks.u32tape.offsets[haystacks.u32tape.count]
                                : (sz_size_t)haystacks.u64tape.offsets[haystacks.u64tape.count];
    sz_size_t output_bound = 0;
    char const *error_detail = NULL;
    SZS_LOCK_(&self->lock);
    sz_status_t status =
        szs_substrings_replace_bound(self->handle, &replacements_sequence, input_bytes, &output_bound, &error_detail);
    SZS_UNLOCK_(&self->lock);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Substrings rewrite bound");
        return NULL;
    }

    npy_intp offsets_dims[1] = {(npy_intp)(haystacks.count + 1)};
    PyArrayObject *offsets_array = (PyArrayObject *)PyArray_SimpleNew(1, offsets_dims, NPY_UINT64);
    if (!offsets_array) return PyErr_NoMemory();

    sz_memory_allocator_t *out_alloc = need_unified ? &unified_allocator : &default_allocator;
    sz_ptr_t output_data = (sz_ptr_t)out_alloc->allocate(output_bound ? output_bound : 1, out_alloc->handle);
    if (!output_data) {
        Py_DECREF(offsets_array);
        return PyErr_NoMemory();
    }

    sz_size_t output_bytes_written = 0;
    if (device_scope) SZS_LOCK_(&device_scope->lock);
    SZS_LOCK_(&self->lock);
    status = haystacks.is_u32tape
                 ? szs_substrings_replace_u32tape(self->handle, device_handle, &haystacks.u32tape, policy,
                                                  &replacements_sequence, output_data, output_bound,
                                                  (sz_u64_t *)PyArray_DATA(offsets_array), &output_bytes_written,
                                                  &error_detail)
                 : szs_substrings_replace_u64tape(self->handle, device_handle, &haystacks.u64tape, policy,
                                                  &replacements_sequence, output_data, output_bound,
                                                  (sz_u64_t *)PyArray_DATA(offsets_array), &output_bytes_written,
                                                  &error_detail);
    SZS_UNLOCK_(&self->lock);
    if (device_scope) SZS_UNLOCK_(&device_scope->lock);
    if (status != sz_success_k) {
        out_alloc->free(output_data, output_bound ? output_bound : 1, out_alloc->handle);
        Py_DECREF(offsets_array);
        set_stringzilla_error(status, error_detail, "Substrings rewriting");
        return NULL;
    }

    PyObject *rewritten = PyBytes_FromStringAndSize(output_data, (Py_ssize_t)output_bytes_written);
    out_alloc->free(output_data, output_bound ? output_bound : 1, out_alloc->handle);
    if (!rewritten) {
        Py_DECREF(offsets_array);
        return NULL;
    }

    PyObject *result = PyTuple_New(2);
    if (!result) {
        Py_DECREF(rewritten);
        Py_DECREF(offsets_array);
        return NULL;
    }
    PyTuple_SET_ITEM(result, 0, rewritten);
    PyTuple_SET_ITEM(result, 1, (PyObject *)offsets_array);
    return result;
}

static PyObject *Substrings_replace_bound(Substrings *self, PyObject *args, PyObject *kwargs) {
    PyObject *replacements_obj = NULL;
    sz_size_t input_bytes = 0;
    static char *kwlist[] = {"replacements", "input_bytes", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "On", kwlist, &replacements_obj, &input_bytes)) return NULL;

    sz_sequence_t replacements_sequence;
    if (!sz_py_export_strings_as_sequence(replacements_obj, &replacements_sequence)) {
        PyErr_Format(PyExc_TypeError,
                     "Expected stringzilla.Strs object, got %s. Convert using: stringzilla.Strs(your_string_list)",
                     Py_TYPE(replacements_obj)->tp_name);
        return NULL;
    }

    sz_size_t output_bound = 0;
    char const *error_detail = NULL;
    SZS_LOCK_(&self->lock);
    sz_status_t status =
        szs_substrings_replace_bound(self->handle, &replacements_sequence, input_bytes, &output_bound, &error_detail);
    SZS_UNLOCK_(&self->lock);
    if (status != sz_success_k) {
        set_stringzilla_error(status, error_detail, "Substrings rewrite bound");
        return NULL;
    }
    return PyLong_FromSize_t((size_t)output_bound);
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_count[] = //
    "count(haystacks, device=None, policy='overlapping')\n"
    "\n"
    "Count matches of every needle in every haystack.\n"
    "\n"
    "Args:\n"
    "  haystacks (Strs): Haystack collection to scan.\n"
    "  device (DeviceScope, optional): Execution scope. Defaults to the module scope.\n"
    "  policy (str, optional): 'overlapping', 'leftmost-longest', or 'leftmost-first'.\n"
    "\n"
    "Returns:\n"
    "  tuple[numpy.ndarray, int]: Per-haystack counts as uint64, and the corpus-wide total.\n"
    "\n"
    "Examples:\n"
    "  >>> import stringzilla as sz, stringzillas as szs\n"
    "  >>> engine = szs.Substrings(sz.Strs(['he', 'she']))\n"
    "  >>> counts, total = engine.count(sz.Strs(['hershey', 'nothing']))\n"
    "  >>> total\n"
    "  3";

static char const doc_find[] = //
    "find(haystacks, device=None, policy='overlapping')\n"
    "\n"
    "Locate every match, resolved under `policy`.\n"
    "\n"
    "Args:\n"
    "  haystacks (Strs): Haystack collection to scan.\n"
    "  device (DeviceScope, optional): Execution scope. Defaults to the module scope.\n"
    "  policy (str, optional): 'overlapping', 'leftmost-longest', or 'leftmost-first'.\n"
    "\n"
    "Returns:\n"
    "  tuple: Four uint64 arrays - haystack indices, needle indices, byte offsets, byte lengths.\n"
    "\n"
    "Examples:\n"
    "  >>> import stringzilla as sz, stringzillas as szs\n"
    "  >>> engine = szs.Substrings(sz.Strs(['he', 'she']))\n"
    "  >>> haystack_ix, needle_ix, offsets, lengths = engine.find(sz.Strs(['hershey']))\n"
    "  >>> len(offsets)\n"
    "  3";

static char const doc_score_bm25[] = //
    "score_bm25(haystacks, needle_weights, average_document_length, device=None, document_lengths=None, "
    "term_frequency_saturation=1.2, length_normalization=0.75)\n"
    "\n"
    "Score every haystack against the dictionary, which is the query.\n"
    "\n"
    "Term frequencies are raw overlapping counts, which is classic BM25, so no overlap policy applies.\n"
    "\n"
    "Args:\n"
    "  haystacks (Strs): Documents to score.\n"
    "  needle_weights (numpy.ndarray): One float32 IDF or boost per needle.\n"
    "  average_document_length (float): Corpus-wide mean of the lengths below, in the same unit. A\n"
    "    property of the corpus rather than a tunable, so it has no default; pass 0.0 only alongside\n"
    "    length_normalization=0.0, since normalizing without a mean is refused.\n"
    "  device (DeviceScope, optional): Execution scope. Defaults to the module scope.\n"
    "  document_lengths (numpy.ndarray, optional): One float32 length per haystack; byte lengths when None.\n"
    "  term_frequency_saturation (float, optional): The literature's k1, how slowly repeated occurrences\n"
    "    stop adding score.\n"
    "  length_normalization (float, optional): The literature's b, in [0, 1]; 0 ignores document length\n"
    "    and every length input with it, 1 normalizes fully.\n"
    "\n"
    "Returns:\n"
    "  numpy.ndarray: One float32 score per haystack.\n"
    "\n"
    "Examples:\n"
    "  >>> import numpy as np, stringzilla as sz, stringzillas as szs\n"
    "  >>> engine = szs.Substrings(sz.Strs(['cat', 'dog']))\n"
    "  >>> weights = np.ones(2, dtype=np.float32)\n"
    "  >>> scores = engine.score_bm25(sz.Strs(['cat and dog', 'nothing here']), weights, 10.0)\n"
    "  >>> float(scores[1])\n"
    "  0.0";

static char const doc_replace[] = //
    "replace(haystacks, replacements, device=None, policy='leftmost-longest')\n"
    "\n"
    "Rewrite every haystack, substituting each match with its needle's replacement.\n"
    "\n"
    "Tape in, tape out: a rewrite's product is itself a tape, so a reordered `Strs` is refused. The output\n"
    "buffer is sized from `replace_bound`, so the call cannot be refused for capacity.\n"
    "\n"
    "Args:\n"
    "  haystacks (Strs): Haystacks to rewrite, in a tape layout.\n"
    "  replacements (Strs): One replacement per needle.\n"
    "  device (DeviceScope, optional): Execution scope. Defaults to the module scope.\n"
    "  policy (str, optional): 'leftmost-longest' or 'leftmost-first'; overlapping is refused.\n"
    "\n"
    "Returns:\n"
    "  tuple[bytes, numpy.ndarray]: The rewritten tape and its uint64 offsets, one per haystack plus one.";

static char const doc_replace_bound[] = //
    "replace_bound(replacements, input_bytes)\n"
    "\n"
    "Bound the bytes a rewrite can produce, from the dictionary and replacements alone.\n"
    "\n"
    "Needs no haystacks, no walk, and no device: the answer is arithmetic over the needle set.\n"
    "\n"
    "Returns:\n"
    "  int: Bytes an output tape must hold to accept any such rewrite.";

static PyMethodDef Substrings_methods[] = {
    {"count", (PyCFunction)Substrings_count, SZ_METHOD_FLAGS, doc_count},
    {"find", (PyCFunction)Substrings_find, SZ_METHOD_FLAGS, doc_find},
    {"score_bm25", (PyCFunction)Substrings_score_bm25, SZ_METHOD_FLAGS, doc_score_bm25},
    {"replace", (PyCFunction)Substrings_replace, SZ_METHOD_FLAGS, doc_replace},
    {"replace_bound", (PyCFunction)Substrings_replace_bound, SZ_METHOD_FLAGS, doc_replace_bound},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static char const doc_Substrings[] = //
    "Substrings(needles, case_sensitivity='cased', device=None, capabilities=None)\n"
    "\n"
    "Match a whole dictionary of needles against a whole collection of haystacks in one pass.\n"
    "\n"
    "The needle set is compiled once into an Aho-Corasick automaton and reused across every later call, so\n"
    "the dictionary is paid for once rather than per haystack. Matches are reported under a caller-chosen\n"
    "overlap policy, and the same automaton also scores haystacks with BM25 and rewrites them.\n"
    "\n"
    "Construction is a device operation: the automaton's tier split is sized against the cache the walk\n"
    "reads through, and a CUDA automaton is uploaded to a device, so `device` belongs here as well as on\n"
    "each call.\n"
    "\n"
    "Args:\n"
    "  needles (Strs): Needle collection to compile. Non-empty, and valid UTF-8 when folding.\n"
    "  case_sensitivity (str, optional): 'cased' matches bytes, 'uncased' folds both sides.\n"
    "  device (DeviceScope, optional): Scope the automaton is built for and uploaded to.\n"
    "  capabilities (tuple, optional): Hardware capabilities to restrict the engine to.\n"
    "\n"
    "Examples:\n"
    "  >>> import stringzilla as sz, stringzillas as szs\n"
    "  >>> engine = szs.Substrings(sz.Strs(['he', 'she', 'his', 'hers']))\n"
    "  >>> counts, total = engine.count(sz.Strs(['ushers', 'nothing']))\n"
    "  >>> total\n"
    "  3";

static PyGetSetDef Substrings_getsetters[] = {
    {"__capabilities__", (getter)Substrings_get_capabilities, NULL, doc_capabilities, NULL}, //
    {NULL}                                                                                   /* Sentinel */
};

PyTypeObject SubstringsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzillas.Substrings",
    .tp_doc = doc_Substrings,
    .tp_basicsize = sizeof(Substrings),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Substrings_new,
    .tp_init = (initproc)Substrings_init,
    .tp_methods = Substrings_methods,
    .tp_getset = Substrings_getsetters,
    .tp_repr = (reprfunc)Substrings_repr,
    .tp_dealloc = (destructor)Substrings_dealloc,
};

#pragma endregion Type Registration
