/**
 *  @brief Cross-product Levenshtein edit distances over one prepared batch of queries.
 *  @file python/stringzilla/levenshtein.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/** @brief A batch of queries prepared once, scored against a fresh collection of candidates each round. */
typedef struct {
    PyObject ob_base;
    sz_levenshtein_engine_t engine; //< Owned; freed in `tp_dealloc`, rebuilt by a second `__init__`.
    sz_engine_lock_field_           //< Guards the round scratch a call grows.
} LevenshteinEngine;

#pragma region Construction

/** @brief Reads the alphabet name a batch counts its distances in, defaulting to bytes. */
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
    sz_levenshtein_engine_free(&self->engine);
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
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "symbol") != 0) {
                PyErr_Format(PyExc_TypeError, "LevenshteinEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
            if (symbol_obj) {
                PyErr_SetString(PyExc_TypeError, "LevenshteinEngine() got multiple values for argument 'symbol'");
                return -1;
            }
            symbol_obj = value;
        }
    }

    sz_sequence_t queries;
    sz_levenshtein_symbol_t symbol;
    if (sz_py_export_strings(queries_obj, "queries", &queries) != 0) return -1;
    if (parse_levenshtein_symbol_(symbol_obj, &symbol) != 0) return -1;

    sz_engine_lock_(self);
    sz_levenshtein_engine_free(&self->engine);
    sz_status_t const status = sz_levenshtein_engine_init_cpu(&queries, symbol, SZ_NULL, &self->engine);
    sz_engine_unlock_(self);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "LevenshteinEngine()");
        return -1;
    }
    return 0;
}

static char const doc_LevenshteinEngine_on_gpu[] =                                                 //
    "on_gpu(queries, symbol='bytes', stream=0) -> LevenshteinEngine\n"                             //
    "\n"                                                                                           //
    "Prepare the same batch on a device, on a stream the caller owns.\n"                           //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  queries (Strs): Patterns every candidate is scored against.\n"                              //
    "  symbol (str, optional): 'bytes' or 'runes'.\n"                                              //
    "  stream (int, optional): A `cudaStream_t` as an integer, or 0 for the default stream.\n"     //
    "Returns:\n"                                                                                   //
    "  LevenshteinEngine: A device engine whose rounds enqueue and return; join before reading.\n" //
    "Example:\n"                                                                                   //
    "  >>> engine = sz.LevenshteinEngine.on_gpu(sz.Strs(['hello']))  # doctest: +SKIP";

static PyObject *LevenshteinEngine_on_gpu(PyObject *type_obj, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "on_gpu() takes 1 to 3 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const queries_obj = args[0];
    PyObject *symbol_obj = positional_args_count > 1 ? args[1] : NULL;
    PyObject *stream_obj = positional_args_count > 2 ? args[2] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "symbol") == 0) {
            if (symbol_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'symbol'");
                return NULL;
            }
            symbol_obj = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) {
            if (stream_obj) {
                PyErr_SetString(PyExc_TypeError, "on_gpu() got multiple values for argument 'stream'");
                return NULL;
            }
            stream_obj = value;
        }
        else {
            PyErr_Format(PyExc_TypeError, "on_gpu() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }

    sz_sequence_t queries;
    sz_levenshtein_symbol_t symbol;
    void *stream = NULL;
    if (sz_py_export_strings(queries_obj, "queries", &queries) != 0) return NULL;
    if (parse_levenshtein_symbol_(symbol_obj, &symbol) != 0) return NULL;
    if (sz_py_export_stream(stream_obj, &stream) != 0) return NULL;

    PyTypeObject *const type = (PyTypeObject *)type_obj;
    LevenshteinEngine *const self = (LevenshteinEngine *)type->tp_alloc(type, 0);
    if (!self) return NULL;
    sz_status_t const status = sz_levenshtein_engine_init_gpu(&queries, symbol, SZ_NULL, stream, &self->engine);
    if (status != sz_success_k) {
        Py_DECREF(self);
        sz_py_raise_status(status, "LevenshteinEngine.on_gpu()");
        return NULL;
    }
    return (PyObject *)self;
}

#pragma endregion Construction

#pragma region Operations

static char const doc_LevenshteinEngine_distances[] =                                              //
    "distances(candidates, out) -> None\n"                                                         //
    "\n"                                                                                           //
    "Score every prepared query against every candidate, into `out`.\n"                            //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  candidates (Strs): Texts forming the matrix columns.\n"                                     //
    "  out (buffer): Writable 2-D buffer of pointer-width unsigned integers (e.g. numpy.uintp),\n" //
    "    at least (len(queries), len(candidates)), contiguous along its candidate axis.\n"         //
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
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") != 0) {
            PyErr_Format(PyExc_TypeError, "distances() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
        if (out_obj) {
            PyErr_SetString(PyExc_TypeError, "distances() got multiple values for argument 'out'");
            return NULL;
        }
        out_obj = value;
    }
    if (!out_obj) {
        PyErr_SetString(PyExc_TypeError, "distances() needs an `out` buffer to write into");
        return NULL;
    }
    if (!self->engine.memory) {
        PyErr_SetString(PyExc_ValueError, "LevenshteinEngine holds no prepared queries");
        return NULL;
    }

    sz_sequence_t candidates;
    if (sz_py_export_strings(candidates_obj, "candidates", &candidates) != 0) return NULL;

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

    sz_engine_lock_(self);
    sz_status_t const status = sz_levenshtein_distances(&self->engine, &candidates, (sz_size_t *)out_view.buf,
                                                        strides[0]);
    sz_engine_unlock_(self);
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "distances()");
        return NULL;
    }
    Py_RETURN_NONE;
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_LevenshteinEngine[] =                                                   //
    "LevenshteinEngine(queries, symbol='bytes')\n"                                            //
    "\n"                                                                                      //
    "Prepare a batch of queries once and score it against many collections of candidates.\n"  //
    "\n"                                                                                      //
    "Unit-cost Myers bit-parallel edit distance: every query is packed into match masks at\n" //
    "construction, and each round streams the candidates past them.\n"                        //
    "\n"                                                                                      //
    "Args:\n"                                                                                 //
    "  queries (Strs): Patterns every candidate is scored against.\n"                         //
    "  symbol (str, optional): 'bytes' counts byte edits, 'runes' counts UTF-8 rune edits.\n" //
    "Example:\n"                                                                              //
    "  >>> engine = sz.LevenshteinEngine(sz.Strs(['hello']), symbol='bytes')\n"               //
    "  >>> out = memoryview(bytearray(8)).cast('Q', (1, 1))\n"                                //
    "  >>> engine.distances(sz.Strs(['hallo']), out)\n"                                       //
    "  >>> out[0, 0]\n"                                                                       //
    "  1";

static PyMethodDef LevenshteinEngine_methods[] = {
    {"on_gpu", (PyCFunction)LevenshteinEngine_on_gpu, SZ_METHOD_FLAGS | METH_CLASS, doc_LevenshteinEngine_on_gpu},
    {"distances", (PyCFunction)LevenshteinEngine_distances, SZ_METHOD_FLAGS, doc_LevenshteinEngine_distances},
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
