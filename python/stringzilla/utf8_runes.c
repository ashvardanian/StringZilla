/**
 *  @brief Codepoint-level UTF-8 machinery — counting, seeking, decoding, and rune iteration.
 *  @file python/stringzilla/utf8_runes.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief  Iterator yielding Unicode code points (as Python @c int) decoded from UTF-8 text.
 *
 *  Streams code points by refilling a small inline buffer with @c sz_utf8_decode, which fills the whole
 *  buffer (or drains the input) per call regardless of script width, and substitutes U+FFFD for ill-formed bytes.
 *  The buffer lives in the iterator itself - no extra allocation. @c cursor advances by the bytes consumed on each
 *  refill. Mirrors the @c Utf8Boundaries batched model, but buffers decoded runes rather than (start, length) pairs.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj; //< For reference counting

    sz_cptr_t cursor; //< Resume cursor into the text; advances by the bytes consumed per refill.
    sz_cptr_t end;    //< End of the text (immutable).

    /// @brief  Inline batch of decoded UTF-32 code points, refilled on demand.
    sz_rune_t batch_runes[sz_iterators_default_steps_k];
    sz_size_t batch_count; //< Number of code points currently buffered.
    sz_size_t batch_index; //< Index of the next code point to yield from the buffer.

} Utf8Codepoints;

char const doc_utf8_count[] =                                                    //
    "Count the number of UTF-8 characters in a string.\n"                        //
    "\n"                                                                         //
    "Unlike len() which returns bytes, this counts actual Unicode characters,\n" //
    "handling multi-byte UTF-8 sequences correctly.\n"                           //
    "\n"                                                                         //
    "Args:\n"                                                                    //
    "  text (Str or str or bytes): The string object.\n"                         //
    "Returns:\n"                                                                 //
    "  int: Number of UTF-8 characters in the string.\n"                         //
    "\n"                                                                         //
    "Example:\n"                                                                 //
    "  >>> sz.utf8_count('hello')  # 5 ASCII chars = 5\n"                        //
    "  5\n"                                                                      //
    "  >>> sz.utf8_count('\xc3\xa9')  # 1 char (e-acute) = 1\n"                  //
    "  1";

PyObject *Str_like_utf8_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t min_args = !is_member;
    Py_ssize_t max_args = !is_member;
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_count() takes exactly %zd argument(s)", min_args);
        return NULL;
    }

    // No keyword arguments expected
    if (args_names_tuple && PyTuple_GET_SIZE(args_names_tuple) > 0) {
        PyErr_SetString(PyExc_TypeError, "utf8_count() takes no keyword arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    sz_string_view_t text;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    sz_size_t count = sz_utf8_count(text.start, text.length);
    return PyLong_FromSize_t(count);
}

char const doc_utf8_codepoints[] =                                                     //
    "utf8_codepoints(string)\n"                                                        //
    "\n"                                                                               //
    "Return an iterator yielding Unicode code points (as int) decoded from UTF-8.\n"   //
    "Ill-formed bytes decode to U+FFFD (the replacement character), so iteration is\n" //
    "total and never raises on malformed input.\n"                                     //
    "\n"                                                                               //
    "Args:\n"                                                                          //
    "    string: The input UTF-8 string to decode into code points.\n"                 //
    "\n"                                                                               //
    "Returns:\n"                                                                       //
    "    Iterator yielding int code points, one per Unicode scalar value.\n\n"         //
    "\n"                                                                               //
    "Example:\n"                                                                       //
    "  >>> list(sz.utf8_codepoints('AB'))\n"                                           //
    "  [65, 66]";

PyObject *Str_like_utf8_codepoints(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *kwnames) {
    if (positional_args_count != 1 || (kwnames && PyTuple_GET_SIZE(kwnames) != 0)) {
        PyErr_SetString(PyExc_TypeError, "utf8_codepoints() requires exactly one positional argument");
        return NULL;
    }

    PyObject *text_obj = args[0];
    sz_string_view_t text_view;
    if (PyObject_TypeCheck(text_obj, &StrType)) {
        Str *str_obj = (Str *)text_obj;
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
        PyErr_SetString(PyExc_TypeError, "Expected str, bytes, or Str");
        return NULL;
    }

    Utf8Codepoints *iter = PyObject_New(Utf8Codepoints, &Utf8CodepointsType);
    if (!iter) return PyErr_NoMemory();

    iter->text_obj = text_obj;
    Py_INCREF(text_obj);
    iter->cursor = text_view.start;
    iter->end = text_view.start + text_view.length;
    iter->batch_count = 0;
    iter->batch_index = 0;

    sz_unused_(self);
    return (PyObject *)iter;
}

static PyObject *Utf8CodepointsType_next(Utf8Codepoints *self) {
    // Refill the inline batch when drained. `sz_utf8_decode` fills the whole buffer (or drains the input) per
    // call and substitutes U+FFFD for ill-formed bytes, so every buffered value is a valid Unicode scalar value.
    if (self->batch_index >= self->batch_count) {
        if (self->cursor >= self->end) return NULL;
        sz_size_t unpacked = 0;
        sz_cptr_t next = sz_utf8_decode(self->cursor, (sz_size_t)(self->end - self->cursor), self->batch_runes,
                                        sz_iterators_default_steps_k, &unpacked);
        // A well-formed but truncated trailing sequence yields nothing and does not advance; we own the whole text,
        // so finalize it as one U+FFFD (its maximal subpart) rather than silently dropping it.
        if (unpacked == 0 && next < self->end) {
            self->batch_runes[0] = (sz_rune_t)sz_rune_replacement_k;
            unpacked = 1;
            next = self->end;
        }
        self->cursor = next;
        self->batch_count = unpacked;
        self->batch_index = 0;
        if (self->batch_count == 0) return NULL;
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
    "  - Str.utf8_codepoints()\n"                                              //
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
