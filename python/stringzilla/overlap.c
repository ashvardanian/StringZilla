/**
 *  @file python/stringzilla/overlap.c
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief Window overlap between one prepared batch of queries and many collections of candidates.
 */
#include "stringzilla.h"

/** A forest of query trees built once, probed by a fresh collection of candidates each round. */
typedef struct {
    PyObject ob_base;

    /** Owned; freed in @c tp_dealloc, rebuilt by a second @c __init__. */
    sz_overlap_engine_t engine;

    /** Guards the round scratch a call grows. */
    sz_engine_lock_field_
} OverlapEngine;

#pragma region Construction

/**
 *  @brief Copies the window widths out of a Python sequence of integers.
 *  @return An array of @p count widths from @c malloc, or @c NULL with a Python exception set.
 */
static sz_size_t *parse_window_widths_(PyObject *widths_obj, sz_size_t *count) {
    PyObject *const widths_tuple = PySequence_Tuple(widths_obj);
    if (!widths_tuple) {
        PyErr_SetString(PyExc_TypeError, "widths must be a sequence of integers");
        return NULL;
    }
    Py_ssize_t const widths_count = PyTuple_GET_SIZE(widths_tuple);
    sz_size_t *const widths = (sz_size_t *)malloc((widths_count ? (sz_size_t)widths_count : 1) * sizeof(sz_size_t));
    if (!widths) {
        Py_DECREF(widths_tuple);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t width_index = 0; width_index < widths_count; ++width_index) {
        PyObject *const item = PyTuple_GET_ITEM(widths_tuple, width_index);
        if (!PyLong_Check(item)) {
            Py_DECREF(widths_tuple);
            free(widths);
            PyErr_SetString(PyExc_TypeError, "widths must be a sequence of integers");
            return NULL;
        }
        widths[width_index] = (sz_size_t)PyLong_AsSize_t(item);
        if (PyErr_Occurred()) {
            Py_DECREF(widths_tuple);
            free(widths);
            return NULL;
        }
    }
    Py_DECREF(widths_tuple);
    *count = (sz_size_t)widths_count;
    return widths;
}

static void OverlapEngine_dealloc(OverlapEngine *self) {
    sz_overlap_engine_free(&self->engine);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int OverlapEngine_init(OverlapEngine *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count < 1 || positional_count > 2) {
        PyErr_Format(PyExc_TypeError, "OverlapEngine() takes 1 to 2 positional arguments, got %zd", positional_count);
        return -1;
    }
    PyObject *const queries_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *widths_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *candidates_budget_object = NULL, *capabilities_object = NULL, *device_object = NULL,
             *stream_object = NULL;
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "widths") == 0) {
                if (widths_obj) {
                    PyErr_SetString(PyExc_TypeError, "OverlapEngine() got multiple values for argument 'widths'");
                    return -1;
                }
                widths_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "candidates_budget") == 0) {
                candidates_budget_object = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "device") == 0) { device_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "OverlapEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }
    if (!widths_obj) {
        PyErr_SetString(PyExc_TypeError, "OverlapEngine() needs the window widths to score at");
        return -1;
    }

    sz_sequence_t queries;
    sz_size_t candidates_budget = 0;
    sz_capability_t capabilities;
    sz_size_t ordinal;
    void *stream;
    if (sz_py_export_strings(queries_obj, "queries", &queries) != 0) return -1;
    if (candidates_budget_object && candidates_budget_object != Py_None) {
        candidates_budget = (sz_size_t)PyLong_AsSize_t(candidates_budget_object);
        if (PyErr_Occurred()) return -1;
    }
    if (sz_py_export_engine_placement(device_object, capabilities_object, stream_object, &capabilities, &ordinal,
                                      &stream) != 0)
        return -1;
    sz_size_t widths_count = 0;
    sz_size_t *const widths = parse_window_widths_(widths_obj, &widths_count);
    if (!widths) return -1;

    sz_engine_lock_(self);
    sz_overlap_engine_free(&self->engine);
    sz_status_t const status = sz_overlap_engine_init(&self->engine, &queries, widths, widths_count, candidates_budget,
                                                      capabilities, ordinal, STRINGZILLA_NULL, stream);
    sz_engine_unlock_(self);
    free(widths);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "OverlapEngine()");
        return -1;
    }
    return 0;
}

#pragma endregion Construction

#pragma region Operations

static char const doc_OverlapEngine_scores[] =                                                     //
    "scores(candidates, out, *, stream=None) -> None\n"                                            //
    "\n"                                                                                           //
    "Score every prepared query against every candidate, at every width, into `out`.\n"            //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  candidates (Strs): Texts whose windows probe the forest.\n"                                 //
    "  out (buffer): Writable 3-D buffer of 32-bit floats, at least\n"                             //
    "    (len(queries), len(candidates), len(widths)), contiguous along its width axis.\n"         //
    "  stream (int, optional): A stream of the engine's device as an integer, or None for the\n"   //
    "    default. A device engine enqueues there and returns, so `out` must be device-reachable\n" //
    "    and read only after the caller joins the stream; a host engine ignores it.\n"             //
    "Example:\n"                                                                                   //
    "  >>> engine = sz.OverlapEngine(sz.Strs(['abcdef']), [3])\n"                                  //
    "  >>> out = memoryview(bytearray(4)).cast('f', (1, 1, 1))\n"                                  //
    "  >>> engine.scores(sz.Strs(['abcdef']), out)\n"                                              //
    "  >>> out[0, 0, 0]\n"                                                                         //
    "  1.0";

static PyObject *OverlapEngine_scores(OverlapEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "scores() takes 1 to 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const candidates_obj = args[0];
    PyObject *out_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
            if (out_obj) {
                PyErr_SetString(PyExc_TypeError, "scores() got multiple values for argument 'out'");
                return NULL;
            }
            out_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "scores() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_obj) {
        PyErr_SetString(PyExc_TypeError, "scores() needs an `out` buffer to write into");
        return NULL;
    }
    if (!self->engine.memory) {
        PyErr_SetString(PyExc_ValueError, "OverlapEngine holds no prepared queries");
        return NULL;
    }

    sz_sequence_t candidates;
    void *stream = NULL;
    if (sz_py_export_strings(candidates_obj, "candidates", &candidates) != 0) return NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;

    sz_size_t const extents[3] = {self->engine.count, candidates.count, self->engine.widths_count};
    sz_size_t strides[3];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_obj, "out", (Py_ssize_t)sizeof(sz_f32_t), 3, extents, &out_view, strides) != 0)
        return NULL;
    if (strides[2] != 1) {
        PyErr_SetString(PyExc_ValueError, "out must be contiguous along its width axis");
        PyBuffer_Release(&out_view);
        return NULL;
    }

    sz_f32_t *const scores = (sz_f32_t *)out_view.buf;
    sz_engine_lock_(self);
    if (!(self->engine.capability & sz_cap_devices_k)) stream = NULL;
    sz_status_t const status = sz_overlap_scores(&self->engine, &candidates, scores, strides[0], strides[1], stream);
    sz_engine_unlock_(self);
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "scores()");
        return NULL;
    }
    Py_RETURN_NONE;
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_OverlapEngine[] =                                                            //
    "OverlapEngine(queries, widths, *, candidates_budget=0, device=None, capabilities=None,\n"     //
    "              stream=None)\n"                                                                 //
    "\n"                                                                                           //
    "Hash a batch of queries into one window forest and probe it with many candidates.\n"          //
    "\n"                                                                                           //
    "A window is a fixed-width byte n-gram, and a score is the share of a candidate's windows\n"   //
    "that the query also spells, in [0, 1]. The score is asymmetric in the two sides.\n"           //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  queries (Strs): Texts whose windows are sorted into the forest.\n"                          //
    "  widths (sequence of int): Window widths in bytes, the last axis of every output.\n"         //
    "  candidates_budget (int, optional): The most candidates one round may carry on a device,\n"  //
    "    ignored on the CPU.\n"                                                                    //
    "  device (Device, optional): Where the engine runs, defaulting to Device.cpu().\n"            //
    "  capabilities (Capability, optional): A narrowing of the device's capabilities_enabled(),\n" //
    "    which is the default.\n"                                                                  //
    "  stream (int, optional): A stream of that device as an integer, or None for the default.\n"  //
    "Example:\n"                                                                                   //
    "  >>> engine = sz.OverlapEngine(sz.Strs(['abcdef']), [3, 4])\n"                               //
    "  >>> out = memoryview(bytearray(8)).cast('f', (1, 1, 2))\n"                                  //
    "  >>> engine.scores(sz.Strs(['abcdefg']), out)\n"                                             //
    "  >>> 0.0 <= out[0, 0, 0] <= 1.0\n"                                                           //
    "  True";

static PyMethodDef OverlapEngine_methods[] = {
    {"scores", (PyCFunction)OverlapEngine_scores, STRINGZILLA_METHOD_FLAGS, doc_OverlapEngine_scores},
    {NULL, NULL, 0, NULL},
};

PyTypeObject OverlapEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.OverlapEngine",
    .tp_doc = doc_OverlapEngine,
    .tp_basicsize = sizeof(OverlapEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)OverlapEngine_init,
    .tp_dealloc = (destructor)OverlapEngine_dealloc,
    .tp_methods = OverlapEngine_methods,
};

#pragma endregion Type Registration
