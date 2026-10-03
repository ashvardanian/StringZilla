/**
 *  @file python/stringzilla/levenshtein.c
 *  @author Ash Vardanian
 *  @date September 6, 2023
 *  @brief Cross-product Levenshtein edit distances over one prepared batch of queries.
 */
#include "stringzilla.h"

/** A batch of queries prepared once, scored against a fresh collection of candidates each round. */
typedef struct {
    PyObject ob_base;

    /** Owned; freed in @c tp_dealloc, rebuilt by a second @c __init__. */
    sz_levenshtein_engine_t engine;

    /** Guards the round scratch a call grows. */
    sz_py_mutex_t mutex;
} LevenshteinEngine;

#pragma region Construction

/** Reads the alphabet name a batch counts its distances in, defaulting to bytes. */
static int parse_levenshtein_symbol_(PyObject *symbol_obj, sz_levenshtein_symbol_t *result) {
    if (!symbol_obj || symbol_obj == Py_None) {
        *result = sz_levenshtein_bytes_k;
        return 0;
    }
    if (!PyUnicode_Check(symbol_obj)) {
        PyErr_SetString(PyExc_TypeError, "symbol must be a string");
        return -1;
    }
    if (PyUnicode_CompareWithASCIIString(symbol_obj, "bytes") == 0) {
        *result = sz_levenshtein_bytes_k;
        return 0;
    }
    if (PyUnicode_CompareWithASCIIString(symbol_obj, "runes") == 0) {
        *result = sz_levenshtein_runes_k;
        return 0;
    }
    PyErr_SetString(PyExc_ValueError, "Unknown symbol, expected 'bytes' or 'runes'");
    return -1;
}

static void LevenshteinEngine_dealloc(LevenshteinEngine *self) {
    sz_levenshtein_engine_free(&self->engine, STRINGZILLA_NULL);
    sz_py_mutex_close_(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int LevenshteinEngine_init(LevenshteinEngine *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count < 1 || positional_count > 2) {
        PyErr_Format(PyExc_TypeError, "LevenshteinEngine() takes 1 to 2 positional arguments, got %zd",
                     positional_count);
        return -1;
    }
    PyObject *const queries_obj = PyTuple_GET_ITEM(args, 0);
    PyObject *symbol_obj = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *capabilities_object = NULL, *stream_object = NULL;
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "symbol") == 0) {
                if (symbol_obj) {
                    PyErr_SetString(PyExc_TypeError, "LevenshteinEngine() got multiple values for argument 'symbol'");
                    return -1;
                }
                symbol_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "LevenshteinEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_sequence_t queries;
    sz_levenshtein_symbol_t symbol;
    sz_capability_t capabilities;
    void *stream;
    if (sz_py_export_strings(queries_obj, "queries", &queries) != 0) return -1;
    if (parse_levenshtein_symbol_(symbol_obj, &symbol) != 0) return -1;
    if (sz_py_export_engine_placement(capabilities_object, stream_object, &capabilities, &stream) != 0) return -1;
    if (sz_py_mutex_open_(&self->mutex) != 0) return -1;

    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    sz_levenshtein_engine_free(&self->engine, STRINGZILLA_NULL);
    status = sz_levenshtein_engine_init(&self->engine, &queries, symbol, capabilities, STRINGZILLA_NULL, stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    if (status != sz_success_k) {
        sz_py_raise_status(status, "LevenshteinEngine()");
        return -1;
    }
    return 0;
}

#pragma endregion Construction

#pragma region Operations

static char const doc_LevenshteinEngine_distances[] =                                              //
    "distances(candidates, out, *, stream=None) -> None\n"                                         //
    "\n"                                                                                           //
    "Score every prepared query against every candidate, into `out`.\n"                            //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  candidates (Strs): Texts forming the matrix columns, from `Strs.copy` on a GPU.\n"          //
    "  out (buffer): Writable 2-D buffer of pointer-width unsigned integers like numpy.uintp,\n"   //
    "    at least (len(queries), len(candidates)), contiguous along its candidate axis.\n"         //
    "  stream (int, optional): A stream of the engine's device as an integer, or None for the\n"   //
    "    default. A device engine enqueues there and returns, so `out` must be device-reachable\n" //
    "    and read only after the caller joins the stream; a host engine ignores it.\n"             //
    "Example:\n"                                                                                   //
    "  >>> engine = sz.LevenshteinEngine(sz.Strs(['hello', 'world']))\n"                           //
    "  >>> out = memoryview(bytearray(32)).cast('Q', (2, 2))\n"                                    //
    "  >>> engine.distances(sz.Strs(['hallo', 'word']), out)\n"                                    //
    "  >>> out[0, 0]\n"                                                                            //
    "  1";

static PyObject *LevenshteinEngine_distances(LevenshteinEngine *self, PyObject *const *args,
                                             Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "distances() takes 1 to 2 positional arguments, got %zd", positional_args_count);
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
                PyErr_SetString(PyExc_TypeError, "distances() got multiple values for argument 'out'");
                return NULL;
            }
            out_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "distances() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_obj) {
        PyErr_SetString(PyExc_TypeError, "distances() needs an `out` buffer to write into");
        return NULL;
    }
    if (!self->engine.memory) {
        PyErr_SetString(PyExc_ValueError, "LevenshteinEngine holds no prepared queries");
        return NULL;
    }

    sz_capability_t const capability = self->engine.capability;
    sz_sequence_t candidates;
    void *stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_engine_strings(candidates_obj, "candidates", capability, stream, &candidates) != 0) return NULL;

    sz_size_t const extents[2] = {self->engine.count, candidates.count};
    sz_size_t strides[2];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_obj, "out", (Py_ssize_t)sizeof(sz_size_t), 2, extents, &out_view, strides) != 0)
        return NULL;
    if (strides[1] != 1) {
        PyErr_SetString(PyExc_ValueError, "out must be contiguous along its candidate axis");
        PyBuffer_Release(&out_view);
        return NULL;
    }

    sz_size_t *const distances = (sz_size_t *)out_view.buf;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its shape.
    if (self->engine.capability != capability || self->engine.count != extents[0]) status = sz_unexpected_dimensions_k;
    else status = sz_levenshtein_distances(&self->engine, &candidates, distances, strides[0], stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "distances()");
        return NULL;
    }
    Py_RETURN_NONE;
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_LevenshteinEngine[] =                                                          //
    "LevenshteinEngine(queries, symbol='bytes', *, capabilities=None, stream=None)\n"                //
    "\n"                                                                                             //
    "Prepare a batch of queries once and score it against many collections of candidates.\n"         //
    "\n"                                                                                             //
    "Unit-cost Myers bit-parallel edit distance: every query is packed into match masks at\n"        //
    "construction, and each round streams the candidates past them.\n"                               //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  queries (Strs): Patterns every candidate is scored against.\n"                                //
    "  symbol (str, optional): 'bytes' counts byte edits, 'runes' counts UTF-8 rune edits.\n"        //
    "  capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),\n" //
    "    defaulting to cpu_capabilities_enabled().\n"                                                //
    "  stream (int, optional): A stream of that vendor as an integer, naming the device, or None\n"  //
    "    for the default.\n"                                                                         //
    "Example:\n"                                                                                     //
    "  >>> engine = sz.LevenshteinEngine(sz.Strs(['hello']), symbol='bytes')\n"                      //
    "  >>> out = memoryview(bytearray(8)).cast('Q', (1, 1))\n"                                       //
    "  >>> engine.distances(sz.Strs(['hallo']), out)\n"                                              //
    "  >>> out[0, 0]\n"                                                                              //
    "  1";

static PyMethodDef LevenshteinEngine_methods[] = {
    {"distances", (PyCFunction)LevenshteinEngine_distances, STRINGZILLA_METHOD_FLAGS, doc_LevenshteinEngine_distances},
    {NULL, NULL, 0, NULL},
};

PyTypeObject LevenshteinEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.LevenshteinEngine",
    .tp_doc = doc_LevenshteinEngine,
    .tp_basicsize = sizeof(LevenshteinEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)LevenshteinEngine_init,
    .tp_dealloc = (destructor)LevenshteinEngine_dealloc,
    .tp_methods = LevenshteinEngine_methods,
};

#pragma endregion Type Registration
