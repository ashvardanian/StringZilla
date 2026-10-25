/**
 *  @file python/stringzilla/subdocuments.c
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief JSONPath and XPath extraction over one compiled vocabulary of paths.
 */
#include "stringzilla.h"

/** A vocabulary of paths compiled once, pulling what they select out of
 *  many collections of documents. */
typedef struct {
    PyObject ob_base;

    /** Owned; freed in @c tp_dealloc, rebuilt by a second @c __init__. */
    sz_subdocuments_engine_t engine;

    /** Guards the depth stack every call grows. */
    sz_py_mutex_t mutex;
} SubdocumentsEngine;

/** Raises for @p status, naming the path or document at @p malformed_index
 *  when it broke its grammar. */
static void SubdocumentsEngine_raise_(sz_status_t status, char const *context, char const *subject,
                                      sz_size_t malformed_index) {
    if (status == sz_invalid_syntax_k)
        PyErr_Format(PyExc_ValueError, "%s: %s %zu breaks its grammar or leaves the supported subset", context, subject,
                     (size_t)malformed_index);
    else sz_py_raise_status(status, context);
}

#pragma region Construction

/** Reads the grammar of every document, which also picks the path language. */
static int parse_format_(PyObject *format_object, sz_subdocuments_format_t *result) {
    if (!format_object || format_object == Py_None) {
        *result = sz_subdocuments_json_k;
        return 0;
    }
    if (!PyUnicode_Check(format_object)) {
        PyErr_SetString(PyExc_TypeError, "format must be a string");
        return -1;
    }
    if (PyUnicode_CompareWithASCIIString(format_object, "json") == 0) {
        *result = sz_subdocuments_json_k;
        return 0;
    }
    if (PyUnicode_CompareWithASCIIString(format_object, "xml") == 0) {
        *result = sz_subdocuments_xml_k;
        return 0;
    }
    PyErr_SetString(PyExc_ValueError, "Unknown format, expected 'json' or 'xml'");
    return -1;
}

static void SubdocumentsEngine_dealloc(SubdocumentsEngine *self) {
    sz_subdocuments_engine_free(&self->engine, STRINGZILLA_NULL);
    sz_py_mutex_close_(&self->mutex);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int SubdocumentsEngine_init(SubdocumentsEngine *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_count = args ? PyTuple_GET_SIZE(args) : 0;
    if (positional_count < 1 || positional_count > 2) {
        PyErr_Format(PyExc_TypeError, "SubdocumentsEngine() takes 1 to 2 positional arguments, got %zd",
                     positional_count);
        return -1;
    }
    PyObject *const paths_object = PyTuple_GET_ITEM(args, 0);
    PyObject *format_object = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    PyObject *capabilities_object = NULL, *stream_object = NULL;
    if (kwargs) {
        Py_ssize_t keyword_cursor = 0;
        PyObject *key = NULL, *value = NULL;
        while (PyDict_Next(kwargs, &keyword_cursor, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "format") == 0) {
                if (format_object) {
                    PyErr_SetString(PyExc_TypeError, "SubdocumentsEngine() got multiple values for argument 'format'");
                    return -1;
                }
                format_object = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "SubdocumentsEngine() got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    sz_strings_view_t paths;
    sz_subdocuments_format_t format;
    sz_capability_t capabilities;
    sz_stream_t stream;
    sz_size_t malformed_index = 0;
    if (sz_py_export_strings(paths_object, "paths", &paths) != 0) return -1;
    if (parse_format_(format_object, &format) != 0) return -1;
    if (sz_py_export_engine_placement(capabilities_object, stream_object, &capabilities, &stream) != 0) return -1;
    if (sz_py_mutex_open_(&self->mutex) != 0) return -1;

    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    sz_subdocuments_engine_free(&self->engine, STRINGZILLA_NULL);
    status = sz_subdocuments_engine_init(&self->engine, &paths, format, &malformed_index, capabilities,
                                         STRINGZILLA_NULL, stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    if (status != sz_success_k) {
        SubdocumentsEngine_raise_(status, "SubdocumentsEngine()", "path", malformed_index);
        return -1;
    }
    return 0;
}

#pragma endregion Construction

#pragma region Operations

/** Refuses a verb called on an object whose vocabulary was never compiled. */
static int SubdocumentsEngine_ready_(SubdocumentsEngine *self) {
    if (self->engine.memory) return 0;
    PyErr_SetString(PyExc_ValueError, "SubdocumentsEngine holds no compiled vocabulary");
    return -1;
}

static char const doc_SubdocumentsEngine_counts[] =                                               //
    "counts(documents, out, *, stream=None) -> None\n"                                            //
    "\n"                                                                                          //
    "Count the nodes of every document that some path selects, once per path, into `out`.\n"      //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  documents (Strings): Documents to walk, from `Strings.copy` on a GPU.\n"                   //
    "  out (buffer): Writable 1-D buffer of pointer-width unsigned integers, one per document.\n" //
    "  stream (int, optional): A stream of the engine's device as an integer, or None.\n"         //
    "    A device engine enqueues there and returns, so `out` must be device-reachable\n"         //
    "    and read only after the caller joins the stream; a host engine ignores it.\n"            //
    "Raises:\n"                                                                                   //
    "  ValueError: Naming the first malformed document.\n"                                        //
    "Example:\n"                                                                                  //
    "  >>> import struct\n"                                                                       //
    "  >>> engine = sz.SubdocumentsEngine(sz.Strings(['$..name']))\n"                             //
    "  >>> out = memoryview(bytearray(struct.calcsize('N'))).cast('N')\n"                         //
    "  >>> engine.counts(sz.Strings(['{\"name\": \"a\", \"b\": {\"name\": \"c\"}}']), out)\n"     //
    "  >>> out[0]\n"                                                                              //
    "  2";

static PyObject *SubdocumentsEngine_counts(SubdocumentsEngine *self, PyObject *const *args,
                                           Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "counts() takes 1 to 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const documents_object = args[0];
    PyObject *out_object = positional_args_count > 1 ? args[1] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "out") == 0) {
            if (out_object) {
                PyErr_SetString(PyExc_TypeError, "counts() got multiple values for argument 'out'");
                return NULL;
            }
            out_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "counts() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!out_object) {
        PyErr_SetString(PyExc_TypeError, "counts() needs an `out` buffer to write into");
        return NULL;
    }
    if (SubdocumentsEngine_ready_(self) != 0) return NULL;

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t documents;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(documents_object, "documents", &documents) != 0) return NULL;

    sz_size_t const extents[1] = {documents.count};
    sz_size_t strides[1];
    Py_buffer out_view;
    if (sz_py_export_output_buffer(out_object, "out", (Py_ssize_t)sizeof(sz_size_t), 1, extents, &out_view, strides) !=
        0)
        return NULL;

    sz_size_t *const counts = (sz_size_t *)out_view.buf;
    sz_size_t malformed_index = 0;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its placement.
    if (self->engine.capability != capability) status = sz_unexpected_dimensions_k;
    else status = sz_subdocuments_counts(&self->engine, &documents, counts, strides[0], &malformed_index, stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&out_view);
    if (status != sz_success_k) {
        SubdocumentsEngine_raise_(status, "counts()", "document", malformed_index);
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_SubdocumentsEngine_find[] =                                                          //
    "find(documents, matches, offsets, *, stream=None) -> None\n"                                          //
    "\n"                                                                                                   //
    "Locate every selected node, writing one row per match and one boundary per document.\n"               //
    "\n"                                                                                                   //
    "Matches come by document, then start offset, then path, so a container precedes what it holds.\n"     //
    "A capacity too small is not an error: `offsets` is filled either way, which is what sizes the next\n" //
    "call, and `Strings.matched` gathers the spans the rows locate.\n"                                     //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  documents (Strings): Documents to walk, from `Strings.copy` on a GPU.\n"                            //
    "  matches (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned\n"        //
    "    integers, shaped `(capacity, 4)` - document index, path index, byte offset, byte length.\n"       //
    "    None makes the call a pure size query.\n"                                                         //
    "  offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding\n"     //
    "    len(documents) + 1 boundaries into `matches`, the last being the total.\n"                        //
    "  stream (int, optional): A stream of the engine's device as an integer, or None.\n"                  //
    "    A device engine enqueues there and returns, so both buffers must be device-reachable\n"           //
    "    and read only after the caller joins the stream; a host engine ignores it.\n"                     //
    "Raises:\n"                                                                                            //
    "  ValueError: Naming the first malformed document.\n"                                                 //
    "Example:\n"                                                                                           //
    "  >>> import struct\n"                                                                                //
    "  >>> engine = sz.SubdocumentsEngine(sz.Strings(['/a/b/text()']), format='xml')\n"                    //
    "  >>> matches = memoryview(bytearray(2 * 4 * struct.calcsize('N'))).cast('N', (2, 4))\n"              //
    "  >>> offsets = memoryview(bytearray(2 * struct.calcsize('N'))).cast('N')\n"                          //
    "  >>> engine.find(sz.Strings(['<a><b>x</b><b>y</b></a>']), matches, offsets)\n"                       //
    "  >>> offsets[1]\n"                                                                                   //
    "  2";

static PyObject *SubdocumentsEngine_find(SubdocumentsEngine *self, PyObject *const *args,
                                         Py_ssize_t positional_args_count, PyObject *args_names_tuple) {
    if (positional_args_count < 1 || positional_args_count > 3) {
        PyErr_Format(PyExc_TypeError, "find() takes 1 to 3 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *const documents_object = args[0];
    PyObject *matches_object = positional_args_count > 1 ? args[1] : NULL;
    PyObject *offsets_object = positional_args_count > 2 ? args[2] : NULL;
    PyObject *stream_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject *const value = args[positional_args_count + keyword_index];
        if (PyUnicode_CompareWithASCIIString(key, "matches") == 0) {
            if (matches_object) {
                PyErr_SetString(PyExc_TypeError, "find() got multiple values for argument 'matches'");
                return NULL;
            }
            matches_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "offsets") == 0) {
            if (offsets_object) {
                PyErr_SetString(PyExc_TypeError, "find() got multiple values for argument 'offsets'");
                return NULL;
            }
            offsets_object = value;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "stream") == 0) { stream_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "find() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (!offsets_object) {
        PyErr_SetString(PyExc_TypeError, "find() needs an `offsets` buffer to write into");
        return NULL;
    }
    if (SubdocumentsEngine_ready_(self) != 0) return NULL;

    sz_capability_t const capability = self->engine.capability;
    sz_strings_view_t documents;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    if (!(capability & sz_cap_gpus_k)) stream = NULL;
    if (sz_py_export_strings(documents_object, "documents", &documents) != 0) return NULL;

    // One match is four pointer-width fields back to back, so rows must be packed like the columns.
    sz_match_t *matches = STRINGZILLA_NULL;
    sz_size_t matches_capacity = 0;
    Py_buffer matches_view;
    sz_bool_t have_matches = sz_false_k;
    if (matches_object && matches_object != Py_None) {
        sz_size_t const matches_extents[2] = {0, 4};
        sz_size_t matches_strides[2];
        if (sz_py_export_output_buffer(matches_object, "matches", (Py_ssize_t)sizeof(sz_size_t), 2, matches_extents,
                                       &matches_view, matches_strides) != 0)
            return NULL;
        if (matches_strides[0] != 4 || matches_strides[1] != 1) {
            PyErr_SetString(PyExc_ValueError, "matches rows must be contiguous, four fields wide");
            PyBuffer_Release(&matches_view);
            return NULL;
        }
        matches = (sz_match_t *)matches_view.buf;
        matches_capacity = (sz_size_t)matches_view.shape[0];
        have_matches = sz_true_k;
    }

    sz_size_t const offsets_extents[1] = {documents.count + 1};
    sz_size_t offsets_strides[1];
    Py_buffer offsets_view;
    if (sz_py_export_output_buffer(offsets_object, "offsets", (Py_ssize_t)sizeof(sz_size_t), 1, offsets_extents,
                                   &offsets_view, offsets_strides) != 0) {
        if (have_matches) PyBuffer_Release(&matches_view);
        return NULL;
    }
    if (offsets_strides[0] != 1) {
        PyErr_SetString(PyExc_ValueError, "offsets must be contiguous");
        PyBuffer_Release(&offsets_view);
        if (have_matches) PyBuffer_Release(&matches_view);
        return NULL;
    }

    sz_size_t *const offsets = (sz_size_t *)offsets_view.buf;
    sz_size_t malformed_index = 0;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    sz_py_mutex_lock_(&self->mutex);
    // Another thread may have rebuilt the engine since the arguments were bound to its placement.
    if (self->engine.capability != capability) status = sz_unexpected_dimensions_k;
    else
        status = sz_subdocuments_find(&self->engine, &documents, matches, matches_capacity, offsets, &malformed_index,
                                      stream);
    sz_py_mutex_unlock_(&self->mutex);
    Py_END_ALLOW_THREADS;
    PyBuffer_Release(&offsets_view);
    if (have_matches) PyBuffer_Release(&matches_view);
    if (status != sz_success_k) {
        SubdocumentsEngine_raise_(status, "find()", "document", malformed_index);
        return NULL;
    }
    Py_RETURN_NONE;
}

#pragma endregion Operations

#pragma region Type Registration

static char const doc_SubdocumentsEngine[] =                                                          //
    "SubdocumentsEngine(paths, format='json', *, capabilities=None, stream=None)\n"                   //
    "\n"                                                                                              //
    "Compile JSONPath or XPath queries once and pull what they select out of many documents.\n"       //
    "\n"                                                                                              //
    "Every document is walked once, whatever the path count, and no tree is built. A match spans\n"   //
    "the node's exact bytes: in JSON a string with its quotes, a number or literal, or a container\n" //
    "through its closing bracket; in XML an element, an attribute's value, or a text node.\n"         //
    "\n"                                                                                              //
    "Args:\n"                                                                                         //
    "  paths (Strings): RFC 9535 JSONPath queries, or for XML, XPath 1.0 location paths along\n"      //
    "    forward axes.\n"                                                                             //
    "  format (str, optional): 'json' or 'xml', the grammar of every document.\n"                     //
    "  capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),\n"  //
    "    defaulting to cpu_capabilities_enabled().\n"                                                 //
    "  stream (int, optional): A stream of that vendor as an integer, naming the device, or None\n"   //
    "    for the default.\n"                                                                          //
    "Raises:\n"                                                                                       //
    "  ValueError: Naming the first path that breaks its grammar or leaves the supported subset.\n"   //
    "Example:\n"                                                                                      //
    "  >>> import struct\n"                                                                           //
    "  >>> engine = sz.SubdocumentsEngine(sz.Strings(['$.a[?@ > 1]']))\n"                             //
    "  >>> out = memoryview(bytearray(struct.calcsize('N'))).cast('N')\n"                             //
    "  >>> engine.counts(sz.Strings(['{\"a\": [1, 2, 3]}']), out)\n"                                  //
    "  >>> out[0]\n"                                                                                  //
    "  2";

static PyMethodDef SubdocumentsEngine_methods[] = {
    {"counts", (PyCFunction)SubdocumentsEngine_counts, STRINGZILLA_METHOD_FLAGS, doc_SubdocumentsEngine_counts},
    {"find", (PyCFunction)SubdocumentsEngine_find, STRINGZILLA_METHOD_FLAGS, doc_SubdocumentsEngine_find},
    {NULL, NULL, 0, NULL},
};

PyTypeObject SubdocumentsEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.SubdocumentsEngine",
    .tp_doc = doc_SubdocumentsEngine,
    .tp_basicsize = sizeof(SubdocumentsEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)SubdocumentsEngine_init,
    .tp_dealloc = (destructor)SubdocumentsEngine_dealloc,
    .tp_methods = SubdocumentsEngine_methods,
};

#pragma endregion Type Registration
