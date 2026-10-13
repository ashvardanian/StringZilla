/**
 *  @file python/stringzilla/utf8_runes.c
 *  @author Ash Vardanian
 *  @date November 24, 2025
 *  @brief Codepoint-level UTF-8 machinery — counting, seeking, decoding, and rune iteration.
 */
#include "stringzilla.h"

/**
 *  @brief Iterator yielding Unicode code points (as Python @c int) decoded from UTF-8 text.
 *
 *  Streams code points by refilling a small inline buffer with @c sz_utf8_decode_best, which fills
 *  the whole buffer (or drains the input) per call regardless of script width, and substitutes
 *  U+FFFD for ill-formed bytes. The buffer lives in the iterator itself, so nothing is allocated,
 *  and @c cursor advances by the bytes consumed on each refill. Mirrors the @c Utf8Boundaries
 *  batched model, but buffers decoded runes rather than segment lengths.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj;

    sz_cptr_t cursor;
    sz_cptr_t end;
    sz_capability_t capabilities;

    sz_rune_t batch_runes[sz_iterators_default_steps_k];
    sz_size_t batch_count;
    sz_size_t batch_index;

} Utf8Codepoints;

char const doc_utf8_count[] =                                                                             //
    "Count the number of UTF-8 characters in a string.\n"                                                 //
    "\n"                                                                                                  //
    "Unlike len() which returns bytes, this counts actual Unicode characters,\n"                          //
    "handling multi-byte UTF-8 sequences correctly.\n"                                                    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (String or str or bytes): The string object.\n"                                               //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: Number of UTF-8 characters in the string.\n"                                                  //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.utf8_count('hello')  # 5 ASCII chars = 5\n"                                                 //
    "  5\n"                                                                                               //
    "  >>> sz.utf8_count('\xc3\xa9')  # 1 e-acute char = 1\n"                                             //
    "  1";

PyObject *String_like_utf8_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StringType);
    Py_ssize_t min_args = !is_member;
    Py_ssize_t max_args = !is_member;
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_count() takes exactly %zd argument(s)", min_args);
        return NULL;
    }

    PyObject *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
        if (PyUnicode_CompareWithASCIIString(key, "capabilities") != 0) {
            PyErr_Format(PyExc_TypeError, "utf8_count() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
        capabilities_object = args[positional_args_count + i];
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    PyObject *text_obj = is_member ? self : args[0];
    sz_string_view_t text;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    sz_size_t count = 0;
    sz_status_t const status = sz_utf8_count_best(text.start, text.length, &count, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "utf8_count()");
        return NULL;
    }
    return PyLong_FromSize_t(count);
}

char const doc_Strings_utf8_count[] =                                                                     //
    "utf8_count(*, out=None, capabilities=None) -> tuple | buffer\n"                                      //
    "\n"                                                                                                  //
    "Count the codepoints of every string, like `len(text.decode())` over each, validating every one\n"   //
    "as UTF-8 in the same pass over the whole tape.\n"                                                    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  out (buffer, optional): Writable, C-contiguous buffer of pointer-width unsigned integers like\n"   //
    "    numpy.uintp to receive the counts with zero allocation. Defaults to None.\n"                     //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  tuple[int, ...]: The codepoints of each string, or `out` itself when an `out` buffer is given.\n"  //
    "Raises:\n"                                                                                           //
    "  ValueError: Naming the index of the first string holding ill-formed UTF-8.\n"                      //
    "Example:\n"                                                                                          //
    "  >>> sz.Strings(['hello', 'caf\\u00e9', '']).utf8_count()\n"                                        //
    "  (5, 4, 0)";

PyObject *Strings_utf8_count(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    if (positional_args_count)
        return PyErr_Format(PyExc_TypeError, "utf8_count() takes no positional arguments, got %zd",
                            positional_args_count);
    PyObject *out_object = NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "out") == 0            ? &out_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "utf8_count() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "utf8_count() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_size_t const count = self->tape.count;
    Py_buffer out_view;
    if (out_object == Py_None) out_object = NULL;
    if (out_object && sz_py_export_contiguous_buffer(out_object, "out", PyBUF_CONTIG, (Py_ssize_t)sizeof(sz_size_t),
                                                     count, &out_view) != 0)
        return NULL;
    sz_size_t *runes = out_object ? (sz_size_t *)out_view.buf : PyMem_New(sz_size_t, count);
    if (!runes) return PyErr_NoMemory();

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_size_t malformed_index = count;
    sz_status_t const status = sz_utf8_count_strings_best(&strings, NULL, count, runes, &malformed_index, capabilities,
                                                          NULL);
    PyObject *result = NULL;
    if (status == sz_invalid_utf8_k)
        PyErr_Format(PyExc_ValueError, "utf8_count(): the string at index %zu holds ill-formed UTF-8", malformed_index);
    else if (status != sz_success_k) sz_py_raise_status(status, "utf8_count()");
    else if (out_object) result = Py_NewRef(out_object);
    else result = PyTuple_New((Py_ssize_t)count);
    for (sz_size_t index = 0; result && !out_object && index != count; ++index) {
        PyObject *const runes_count = PyLong_FromSize_t(runes[index]);
        if (runes_count) PyTuple_SET_ITEM(result, (Py_ssize_t)index, runes_count);
        else Py_CLEAR(result);
    }
    if (out_object) PyBuffer_Release(&out_view);
    else PyMem_Free(runes);
    return result;
}

char const doc_utf8_codepoints[] =                                                                       //
    "utf8_codepoints(string, /, *, capabilities=None)\n"                                                 //
    "\n"                                                                                                 //
    "Return an iterator yielding Unicode code points as int, decoded from UTF-8.\n"                      //
    "Ill-formed bytes decode to U+FFFD, the replacement character, so iteration is\n"                    //
    "total and never raises on malformed input.\n"                                                       //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "    string: The input UTF-8 string to decode into code points.\n"                                   //
    "    capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.\n" //
    "\n"                                                                                                 //
    "Returns:\n"                                                                                         //
    "    Iterator yielding int code points, one per Unicode scalar value.\n\n"                           //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "  >>> list(sz.utf8_codepoints('AB'))\n"                                                             //
    "  [65, 66]";

PyObject *String_like_utf8_codepoints(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *kwnames) {
    if (positional_args_count != 1) {
        PyErr_SetString(PyExc_TypeError, "utf8_codepoints() requires exactly one positional argument");
        return NULL;
    }
    PyObject *capabilities_object = NULL;
    Py_ssize_t const kwnames_count = kwnames ? PyTuple_GET_SIZE(kwnames) : 0;
    for (Py_ssize_t i = 0; i < kwnames_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_CompareWithASCIIString(key, "capabilities") != 0) {
            PyErr_Format(PyExc_TypeError, "utf8_codepoints() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
        capabilities_object = args[positional_args_count + i];
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    PyObject *text_obj = args[0];
    sz_string_view_t text_view;
    if (PyObject_TypeCheck(text_obj, &StringType)) {
        String *str_obj = (String *)text_obj;
        text_view = str_obj->memory;
    }
    else if (PyUnicode_Check(text_obj)) {
        Py_ssize_t signed_length;
        text_view.start = PyUnicode_AsUTF8AndSize(text_obj, &signed_length);
        if (!text_view.start) return NULL;
        text_view.length = (sz_size_t)signed_length;
    }
    else if (PyBytes_Check(text_obj)) {
        text_view.start = PyBytes_AS_STRING(text_obj);
        text_view.length = (sz_size_t)PyBytes_GET_SIZE(text_obj);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Expected str, bytes, or String");
        return NULL;
    }

    Utf8Codepoints *iter = PyObject_New(Utf8Codepoints, &Utf8CodepointsType);
    if (!iter) return PyErr_NoMemory();

    iter->text_obj = text_obj;
    Py_INCREF(text_obj);
    iter->cursor = text_view.start;
    iter->end = text_view.start + text_view.length;
    iter->capabilities = capabilities;
    iter->batch_count = 0;
    iter->batch_index = 0;

    sz_unused_(self);
    return (PyObject *)iter;
}

static PyObject *Utf8CodepointsType_next(Utf8Codepoints *self) {
    if (self->batch_index == self->batch_count) {
        if (self->cursor == self->end) return NULL;
        sz_size_t unpacked = 0, bytes_consumed = 0;
        sz_status_t const status = sz_utf8_decode_best(self->cursor, (sz_size_t)(self->end - self->cursor),
                                                       self->batch_runes, sz_iterators_default_steps_k, &unpacked,
                                                       &bytes_consumed, self->capabilities, NULL);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "__next__()");
            return NULL;
        }
        self->cursor += bytes_consumed;
        self->batch_count = unpacked;
        self->batch_index = 0;
    }

    sz_rune_t rune = self->batch_runes[self->batch_index++];
    return PyLong_FromUnsignedLong((unsigned long)rune);
}

static void Utf8CodepointsType_dealloc(Utf8Codepoints *self) {
    Py_XDECREF(self->text_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Utf8CodepointsType_iter(PyObject *self) {
    Py_INCREF(self);
    return self;
}

static char const doc_Utf8Codepoints[] =                                       //
    "Utf8Codepoints(string)\n"                                                 //
    "\n"                                                                       //
    "UTF-8 aware code point iterator yielding Unicode scalar values as int.\n" //
    "Ill-formed bytes decode to U+FFFD, so iteration is total.\n"              //
    "\n"                                                                       //
    "Created by:\n"                                                            //
    "  - String.utf8_codepoints()\n"                                           //
    "  - sz.utf8_codepoints()\n"                                               //
    "\n"                                                                       //
    "Example:\n"                                                               //
    "  >>> list(sz.utf8_codepoints('AB'))\n"                                   //
    "  [65, 66]";

PyTypeObject Utf8CodepointsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Codepoints",
    .tp_basicsize = sizeof(Utf8Codepoints),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Utf8CodepointsType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_Utf8Codepoints,
    .tp_iter = Utf8CodepointsType_iter,
    .tp_iternext = (iternextfunc)Utf8CodepointsType_next,
};
