/**
 *  @file python/stringzilla/ngrams.c
 *  @author Ash Vardanian
 *  @date January 27, 2024
 *  @brief N-gram overlap of one prepared batch of queries with many collections of candidates, and
 *      MinHash sketches of any texts over the same N-grams.
 */
#include "stringzilla.h"

/** Hash tables of a batch of queries' N-grams built once, probed by a fresh collection of
 *  candidates each round, beside the sketch state every sketching call reuses. */
typedef struct {
    PyObject ob_base;

    /** Owned; freed in @c tp_dealloc, rebuilt by a second @c __init__. */
    sz_ngrams_engine_t engine;

    /** Guards the round scratch a call grows. */
    sz_py_mutex_t mutex;
} NgramsEngine;

#pragma region Construction

/**
 *  @brief Copies the window widths out of a Python sequence of integers.
 *  @return An array of @p count widths from @c malloc, or @c NULL with a Python exception set.
 */
static sz_size_t *parse_window_widths_(PyObject *widths_object, sz_size_t *count) {
    PyObject *const widths_tuple = PySequence_Tuple(widths_object);
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

static void NgramsEngine_dealloc(NgramsEngine *self) {
    sz_ngrams_engine_free(&self->engine, STRINGZILLA_NULL);
    sz_py_mutex_close_(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int NgramsEngine_init(NgramsEngine *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count < 1 || positional_count > 2) {
        PyErr_Format(PyExc_TypeError, "NgramsEngine() takes 1 to 2 positional arguments, got %zd", positional_count);
        return -1;
    }
    PyObject *const queries_object = PyTuple_GET_ITEM(args, 0);
    PyObject *widths_object = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *candidates_budget_object = NULL, *capabilities_object = NULL, *stream_object = NULL;
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "widths") == 0) {
                if (widths_object) {
                    PyErr_SetString(PyExc_TypeError, "NgramsEngine() got multiple values for argument 'widths'");
                    return -1;
                }
                widths_object = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "candidates_budget") == 0) {
                candidates_budget_object = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "NgramsEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }
    if (!widths_object) {
        PyErr_SetString(PyExc_TypeError, "NgramsEngine() needs the window widths");
        return -1;
    }

    sz_strings_view_t queries;
    sz_size_t candidates_budget = 0;
    sz_capability_t capabilities;
    sz_stream_t stream;
    if (sz_py_export_strings(queries_object, "queries", &queries) != 0) return -1;
    if (candidates_budget_object && candidates_budget_object != Py_None) {
        candidates_budget = (sz_size_t)PyLong_AsSize_t(candidates_budget_object);
        if (PyErr_Occurred()) return -1;
    }
    if (sz_py_export_engine_placement(capabilities_object, stream_object, &capabilities, &stream) != 0) return -1;
    sz_size_t widths_count = 0;
    sz_size_t *const widths = parse_window_widths_(widths_object, &widths_count);
    if (!widths) return -1;
    if (sz_py_mutex_open_(&self->mutex) != 0) {
        free(widths);
        return -1;
    }

    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    sz_ngrams_engine_free(&self->engine, STRINGZILLA_NULL);
    status = sz_ngrams_engine_init(&self->engine, &queries, widths, widths_count, candidates_budget, capabilities,
                                   STRINGZILLA_NULL, stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    free(widths);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "NgramsEngine()");
        return -1;
    }
    return 0;
}

#pragma endregion Construction

#pragma region Operations

static char const doc_NgramsEngine_overlaps[] =                                                      //
    "overlaps(candidates, out, *, radius=None, stream=None) -> None\n"                               //
    "\n"                                                                                             //
    "Score every prepared query against every candidate: the share of the query's distinct\n"        //
    "N-grams that the candidate holds, per width, in [0, 1].\n"                                      //
    "\n"                                                                                             //
    "The score is asymmetric, so a candidate holding the whole query scores 1.0 however much\n"      //
    "more it holds. The share of query q, candidate c and width w lands at out[q, c, w].\n"          //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  candidates (Strings): Texts scored against the prepared queries.\n"                           //
    "  out (buffer): Writable 3-D buffer of 32-bit floats, at least\n"                               //
    "    (len(queries), len(candidates), len(widths)), contiguous along its width axis.\n"           //
    "  radius (float, optional): The furthest distance a pair may lie at and still score exactly,\n" //
    "    a distance being the misses over the query's distinct N-grams, summed over every width.\n"  //
    "    A pair past it scores 0.0 at every width, and None or 1.0 keep every score exact.\n"        //
    "  stream (int, optional): A stream of the engine's device as an integer, or None for the\n"     //
    "    default. A device engine joins it before returning; a host engine ignores it.\n"            //
    "Example:\n"                                                                                     //
    "  >>> engine = sz.NgramsEngine(sz.Strings(['abcdef']), [3])\n"                                  //
    "  >>> out = memoryview(bytearray(4)).cast('f', (1, 1, 1))\n"                                    //
    "  >>> engine.overlaps(sz.Strings(['abcdef']), out)\n"                                           //
    "  >>> out[0, 0, 0]\n"                                                                           //
    "  1.0";

static PyObject *NgramsEngine_overlaps(NgramsEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "overlaps() takes 1 to 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const candidates_object = args[0];
    PyObject *out_object = positional_args_count > 1 ? args[1] : NULL;
    PyObject *radius_object = NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
            if (out_object) {
                PyErr_SetString(PyExc_TypeError, "overlaps() got multiple values for argument 'out'");
                return NULL;
            }
            out_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "radius") == 0) { radius_object = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "overlaps() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_object) {
        PyErr_SetString(PyExc_TypeError, "overlaps() needs an `out` buffer to write into");
        return NULL;
    }
    if (!self->engine.count) {
        PyErr_SetString(PyExc_ValueError, "NgramsEngine holds no prepared queries");
        return NULL;
    }

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t candidates;
    sz_stream_t stream = NULL;
    sz_f32_t radius = 1.0f;
    if (radius_object && radius_object != Py_None) {
        double const share = PyFloat_AsDouble(radius_object);
        if (PyErr_Occurred()) return NULL;
        if (!(share >= 0.0)) {
            PyErr_SetString(PyExc_ValueError, "radius must be a share of at least 0.0");
            return NULL;
        }
        radius = (sz_f32_t)share;
    }
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(candidates_object, "candidates", &candidates) != 0) return NULL;

    sz_size_t const extents[3] = {self->engine.count, candidates.count, self->engine.widths_count};
    sz_size_t strides[3];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_object, "out", (Py_ssize_t)sizeof(sz_f32_t), 3, extents, &out_view, strides) !=
        0)
        return NULL;
    if (strides[2] != sizeof(sz_f32_t)) {
        PyErr_SetString(PyExc_ValueError, "out must be contiguous along its width axis");
        PyBuffer_Release(&out_view);
        return NULL;
    }

    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its shape.
    if (self->engine.capability != capability || self->engine.count != extents[0] ||
        self->engine.widths_count != extents[2])
        status = sz_unexpected_dimensions_k;
    else {
        status = sz_ngrams_overlaps(&self->engine, &candidates, radius, (sz_f32_t *)out_view.buf, strides[0],
                                    strides[1], stream);
        if (status == sz_success_k && (capability & sz_cap_gpus_k))
            status = sz_stream_synchronize_best(capability, stream);
    }
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "overlaps()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_NgramsEngine_sketches[] =                                                     //
    "sketches(texts, out, *, stream=None) -> None\n"                                                //
    "\n"                                                                                            //
    "One MinHash sketch per text, summarizing the set of its N-grams at the engine's widths.\n"     //
    "\n"                                                                                            //
    "The share of positions where two sketches differ estimates one minus the Jaccard\n"            //
    "similarity of their N-gram sets, and more slots give a finer estimate. Only sketches of\n"     //
    "one slot count compare. Sketch i fills row out[i], and the column count of `out` is the\n"     //
    "slot count.\n"                                                                                 //
    "\n"                                                                                            //
    "Args:\n"                                                                                       //
    "  texts (Strings): Texts to sketch.\n"                                                         //
    "  out (buffer): Writable 2-D buffer of unsigned bytes, at least len(texts) rows, contiguous\n" //
    "    along its column axis; its column count is the number of slots, at least 1.\n"             //
    "  stream (int, optional): A stream of the engine's device as an integer, or None for the\n"    //
    "    default. A device engine joins it before returning; a host engine ignores it.\n"           //
    "Example:\n"                                                                                    //
    "  >>> engine = sz.NgramsEngine(sz.Strings([]), [4])\n"                                         //
    "  >>> out = memoryview(bytearray(2 * 512)).cast('B', (2, 512))\n"                              //
    "  >>> engine.sketches(sz.Strings(['hello world', 'hello world']), out)\n"                      //
    "  >>> rows = out.tolist()\n"                                                                   //
    "  >>> rows[0] == rows[1]\n"                                                                    //
    "  True";

static PyObject *NgramsEngine_sketches(NgramsEngine *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "sketches() takes 1 to 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const texts_object = args[0];
    PyObject *out_object = positional_args_count > 1 ? args[1] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
            if (out_object) {
                PyErr_SetString(PyExc_TypeError, "sketches() got multiple values for argument 'out'");
                return NULL;
            }
            out_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "sketches() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_object) {
        PyErr_SetString(PyExc_TypeError, "sketches() needs an `out` buffer to write into");
        return NULL;
    }
    if (!self->engine.widths_count) {
        PyErr_SetString(PyExc_ValueError, "NgramsEngine was never initialized");
        return NULL;
    }

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t texts;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(texts_object, "texts", &texts) != 0) return NULL;

    sz_size_t const extents[2] = {texts.count, 1};
    sz_size_t strides[2];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_object, "out", (Py_ssize_t)sizeof(sz_u8_t), 2, extents, &out_view, strides) != 0)
        return NULL;
    if (strides[1] != 1) {
        PyErr_SetString(PyExc_ValueError, "out must be contiguous along its column axis");
        PyBuffer_Release(&out_view);
        return NULL;
    }
    sz_size_t const slots = (sz_size_t)out_view.shape[1];

    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    if (self->engine.capability != capability) status = sz_unexpected_dimensions_k;
    else {
        status = sz_ngrams_sketches(&self->engine, &texts, slots, (sz_u8_t *)out_view.buf, strides[0], stream);
        if (status == sz_success_k && (capability & sz_cap_gpus_k))
            status = sz_stream_synchronize_best(capability, stream);
    }
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "sketches()");
        return NULL;
    }
    Py_RETURN_NONE;
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_NgramsEngine[] =                                                               //
    "NgramsEngine(queries, widths, *, candidates_budget=0, capabilities=None, stream=None)\n"        //
    "\n"                                                                                             //
    "Score how much of each prepared query every candidate holds, and summarize any texts into\n"    //
    "MinHash sketches, both over the N-grams of the engine's widths.\n"                              //
    "\n"                                                                                             //
    "An N-gram is a run of a fixed number of bytes, and the engine works at one or more widths\n"    //
    "at once. Every query's distinct N-grams are hashed into one table at construction, and an\n"    //
    "engine of no queries only sketches. A GPU engine's methods take texts from `Strings.copy`\n"    //
    "and output buffers its device reaches.\n"                                                       //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  queries (Strings): Texts whose distinct N-grams fill the tables, possibly none.\n"            //
    "  widths (sequence of int): N-gram widths in bytes, the last axis of every overlap output.\n"   //
    "  candidates_budget (int, optional): The most candidates one round may carry on a device,\n"    //
    "    ignored on the CPU.\n"                                                                      //
    "  capabilities (Capability, optional): A producer's mask, like capabilities_enabled_cuda(0),\n" //
    "    defaulting to capabilities_enabled_cpu().\n"                                                //
    "  stream (int, optional): A stream of that vendor as an integer, naming the device, or None\n"  //
    "    for the default.\n"                                                                         //
    "Example:\n"                                                                                     //
    "  >>> engine = sz.NgramsEngine(sz.Strings(['abcdef']), [3, 4])\n"                               //
    "  >>> out = memoryview(bytearray(8)).cast('f', (1, 1, 2))\n"                                    //
    "  >>> engine.overlaps(sz.Strings(['abcdefg']), out)\n"                                          //
    "  >>> 0.0 <= out[0, 0, 0] <= 1.0\n"                                                             //
    "  True";

static PyMethodDef NgramsEngine_methods[] = {
    {"overlaps", (PyCFunction)NgramsEngine_overlaps, STRINGZILLA_METHOD_FLAGS, doc_NgramsEngine_overlaps},
    {"sketches", (PyCFunction)NgramsEngine_sketches, STRINGZILLA_METHOD_FLAGS, doc_NgramsEngine_sketches},
    {NULL, NULL, 0, NULL},
};

PyTypeObject NgramsEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.NgramsEngine",
    .tp_doc = doc_NgramsEngine,
    .tp_basicsize = sizeof(NgramsEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)NgramsEngine_init,
    .tp_dealloc = (destructor)NgramsEngine_dealloc,
    .tp_methods = NgramsEngine_methods,
};

#pragma endregion Type Registration
