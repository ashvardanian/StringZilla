/**
 *  @file python/stringzilla/utf8_tokens.c
 *  @author Ash Vardanian
 *  @date November 24, 2025
 *  @brief Newline, whitespace, and delimiter segmentation of UTF-8 text.
 */
#include "stringzilla.h"

/** The shape of every token kernel's dispatch point, like @c sz_utf8_newlines_best. */
typedef sz_status_t (*sz_py_tokenizer_t)(sz_cptr_t text, sz_size_t length, sz_size_t *offsets, sz_size_t *lengths,
                                         sz_size_t capacity, sz_size_t *count, sz_size_t *bytes_consumed,
                                         sz_capability_t capabilities, sz_stream_t stream);

/**
 *  @brief Iterator splitting a UTF-8 string on the separators a token kernel reports.
 *
 *  One shared layout behind every `utf8_split_*` and bare-separator iterator. The kernel's
 *  separator endpoints are the span boundaries `{suffix, s0.start, s0.end, ..., [end]}`; span @c i
 *  is `bounds[i] .. bounds[i+1]`, and @c parts selects which spans via a @b (first,stride) walk:
 *  `0` for the segments between separators - lines, tokens, or fields - `1` for the separators
 *  themselves, and `2` for both interleaved losslessly. The constructor fills the first batch and
 *  the iterator ends once a batch reaching @c end drains.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj;
    sz_py_tokenizer_t tokenizer;
    sz_capability_t capabilities;

    /** 0 for the parts between separators, 1 for the separators, 2 for both interleaved. */
    int parts;
    sz_bool_t skip_empty;

    /** Where the next batch starts: the end of the last separator buffered. */
    sz_cptr_t suffix;
    sz_cptr_t end;

    sz_cptr_t bounds[2 * sz_iterators_default_steps_k + 2];
    sz_size_t spans;

    /** The next span to yield is `bounds[index] .. bounds[index + 1]`. */
    sz_size_t index;

} Utf8Split;

/** Allocates a @c Utf8Split and fills its first batch. */
static PyObject *Utf8Split_make_(PyTypeObject *type, PyObject *text_obj, sz_string_view_t text,
                                 sz_py_tokenizer_t tokenizer, sz_capability_t capabilities, int parts, int skip_empty);

/** Shared body for the six `utf8_split_*` and bare-separator factories. Parses @c skip_empty, plus
 *  @c with_separators for the `split_*` variants, and @c capabilities, and allocates a @c Utf8Split
 *  of @p type that runs @p tokenizer over @p base_parts. */
static PyObject *Str_like_utf8_split_(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple, PyTypeObject *type, sz_py_tokenizer_t tokenizer,
                                      int base_parts, int allow_with_separators) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t min_args = !is_member;
    Py_ssize_t max_args = !is_member + (allow_with_separators ? 2 : 1);
    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "this splitter requires %zd to %zd arguments", min_args, max_args);
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *skip_empty_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *with_separators_obj = (allow_with_separators && positional_args_count > !is_member + 1)
                                        ? args[!is_member + 1]
                                        : NULL;
    PyObject *capabilities_object = NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "skip_empty") == 0 && !skip_empty_obj) { skip_empty_obj = value; }
            else if (allow_with_separators && PyUnicode_CompareWithASCIIString(key, "with_separators") == 0 &&
                     !with_separators_obj) {
                with_separators_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    int skip_empty = 0, with_separators = 0;
    if (skip_empty_obj) {
        skip_empty = PyObject_IsTrue(skip_empty_obj);
        if (skip_empty == -1) {
            wrap_current_exception("The skip_empty argument must be a boolean");
            return NULL;
        }
    }
    if (with_separators_obj) {
        with_separators = PyObject_IsTrue(with_separators_obj);
        if (with_separators == -1) {
            wrap_current_exception("The with_separators argument must be a boolean");
            return NULL;
        }
    }

    int parts = with_separators ? 2 : base_parts;
    return Utf8Split_make_(type, text_obj, text, tokenizer, capabilities, parts, skip_empty);
}

char const doc_utf8_split_newlines[] =                                                                    //
    "Create an iterator over the lines, the content between Unicode newlines.\n"                          //
    "\n"                                                                                                  //
    "Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.\n"                    //
    "Unlike splitlines(), this returns an iterator for memory-efficient processing.\n"                    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  skip_empty (bool, optional): Skip empty lines, defaulting to False.\n"                             //
    "  with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.\n"    //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding the lines as Str objects.\n"                                        //
    "\n"                                                                                                  //
    "Recognized newlines:\n"                                                                              //
    "  LF (\\n), VT (\\v), FF (\\f), CR (\\r), NEL (U+0085),\n"                                           //
    "  LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\\r\\n)\n"                            //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sum(1 for _ in sz.Str('a\\nb').utf8_split_newlines())\n"                                       //
    "  2";
PyObject *Str_like_utf8_split_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8SplitNewlinesType,
                                sz_utf8_newlines_best, 0, 1);
}

char const doc_utf8_newlines[] =                                                                          //
    "Create an iterator over the Unicode newline separators, one per codepoint, CR+LF kept together.\n"   //
    "\n"                                                                                                  //
    "Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.\n"                    //
    "Yields each newline separator; utf8_split_newlines() yields the lines between them.\n"               //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding the newline separators as Str objects.\n"                           //
    "\n"                                                                                                  //
    "Recognized newlines:\n"                                                                              //
    "  LF (\\n), VT (\\v), FF (\\f), CR (\\r), NEL (U+0085),\n"                                           //
    "  LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\\r\\n)\n"                            //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sum(1 for _ in sz.Str('a\\nb').utf8_newlines())\n"                                             //
    "  1";
PyObject *Str_like_utf8_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8NewlinesType,
                                sz_utf8_newlines_best, 1, 0);
}

char const doc_utf8_split_whitespaces[] =                                                                 //
    "Create an iterator over the tokens, the content between Unicode whitespace.\n"                       //
    "\n"                                                                                                  //
    "Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.\n"                         //
    "Splits on each whitespace codepoint; pass skip_empty=True for str.split()-style tokens.\n"           //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.\n"    //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding the non-whitespace tokens as Str objects.\n"                        //
    "\n"                                                                                                  //
    "Recognized whitespace:\n"                                                                            //
    "  ASCII: TAB, LF, VT, FF, CR, SPACE\n"                                                               //
    "  Latin-1: NEXT LINE, NO-BREAK SPACE\n"                                                              //
    "  General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.\n"                                         //
    "  CJK: IDEOGRAPHIC SPACE (U+3000)\n"                                                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sum(1 for _ in sz.Str('foo bar baz').utf8_split_whitespaces())\n"                              //
    "  3";
PyObject *Str_like_utf8_split_whitespaces(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8SplitWhitespacesType,
                                sz_utf8_whitespaces_best, 0, 1);
}

char const doc_utf8_whitespaces[] =                                                                       //
    "Create an iterator over Unicode whitespace separators, one per codepoint, CR+LF kept together.\n"    //
    "\n"                                                                                                  //
    "Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.\n"                         //
    "Yields each whitespace codepoint; utf8_split_whitespaces() yields the tokens between them.\n"        //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding the whitespace separators as Str objects.\n"                        //
    "\n"                                                                                                  //
    "Recognized whitespace:\n"                                                                            //
    "  ASCII: TAB, LF, VT, FF, CR, SPACE\n"                                                               //
    "  Latin-1: NEXT LINE, NO-BREAK SPACE\n"                                                              //
    "  General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.\n"                                         //
    "  CJK: IDEOGRAPHIC SPACE (U+3000)\n"                                                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sum(1 for _ in sz.Str('foo bar').utf8_whitespaces())\n"                                        //
    "  1";
PyObject *Str_like_utf8_whitespaces(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8WhitespacesType,
                                sz_utf8_whitespaces_best, 1, 0);
}

char const doc_utf8_split_delimiters[] =                                                                  //
    "Create an iterator over the fields, the content between Unicode delimiters.\n"                       //
    "\n"                                                                                                  //
    "Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and\n"                       //
    "separator/whitespace (Z*) codepoint - the superset of utf8_split_whitespaces().\n"                   //
    "Splits on delimiters, one separator per codepoint, and yields the segments between them.\n"          //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.\n"    //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding the non-delimiter segments as Str objects.\n"                       //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(str(t) for t in sz.Str('Hi, world').utf8_split_delimiters(skip_empty=True))\n"            //
    "  ['Hi', 'world']";
PyObject *Str_like_utf8_split_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8SplitDelimitersType,
                                sz_utf8_delimiters_best, 0, 1);
}

char const doc_utf8_delimiters[] =                                                                        //
    "Create an iterator over Unicode delimiter separators, one per codepoint, CR+LF kept together.\n"     //
    "\n"                                                                                                  //
    "Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and\n"                       //
    "separator/whitespace (Z*) codepoint. Yields each delimiter codepoint;\n"                             //
    "utf8_split_delimiters() yields the fields between them.\n"                                           //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding the delimiter separators as Str objects.\n"                         //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(str(d) for d in sz.Str('a.b').utf8_delimiters())\n"                                       //
    "  ['.']";
PyObject *Str_like_utf8_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8DelimitersType,
                                sz_utf8_delimiters_best, 1, 0);
}

/** Refills @c bounds with the next batch of separators from @c suffix, closing it with @c end once
 *  the batch reaches it; the @b (first,stride) walk of @c parts restarts at its first span. */
static sz_status_t Utf8Split_refill_(Utf8Split *self) {
    sz_size_t const region = (sz_size_t)(self->end - self->suffix);
    sz_size_t offsets[sz_iterators_default_steps_k];
    sz_size_t lengths[sz_iterators_default_steps_k];
    sz_size_t consumed = 0, separators = 0;
    sz_status_t const status = self->tokenizer(self->suffix, region, offsets, lengths, sz_iterators_default_steps_k,
                                               &separators, &consumed, self->capabilities, NULL);
    if (status != sz_success_k) return status;
    self->bounds[0] = self->suffix;
    for (sz_size_t separator = 0; separator < separators; ++separator) {
        self->bounds[2 * separator + 1] = self->suffix + offsets[separator];
        self->bounds[2 * separator + 2] = self->suffix + offsets[separator] + lengths[separator];
    }
    sz_size_t boundaries = 2 * separators + 1;
    if (consumed == region) self->bounds[boundaries++] = self->end;
    self->spans = boundaries - 1;
    // Separators sit on the odd spans.
    self->index = self->parts == 1;
    self->suffix += consumed;
    return sz_success_k;
}

static PyObject *Utf8SplitType_next(Utf8Split *self) {
    sz_size_t const stride = self->parts == 2 ? 1 : 2;
    for (;;) {
        if (self->skip_empty)
            while (self->index < self->spans && self->bounds[self->index + 1] == self->bounds[self->index])
                self->index += stride;
        if (self->index < self->spans) break;
        if (self->suffix == self->end) return NULL;
        sz_status_t const status = Utf8Split_refill_(self);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "__next__()");
            return NULL;
        }
    }

    sz_size_t const i = self->index;
    self->index += stride;
    sz_cptr_t const segment_start = self->bounds[i];
    sz_size_t const segment_length = (sz_size_t)(self->bounds[i + 1] - self->bounds[i]);

    Str *result_obj = Str_alloc_();
    if (result_obj == NULL) return PyErr_NoMemory();

    result_obj->memory.start = segment_start;
    result_obj->memory.length = segment_length;
    result_obj->parent = self->text_obj;
    Py_INCREF(self->text_obj);
    return (PyObject *)result_obj;
}

static void Utf8SplitType_dealloc(Utf8Split *self) {
    Py_XDECREF(self->text_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Utf8SplitType_iter(PyObject *self) {
    Py_INCREF(self); // Iterator should return itself in __iter__.
    return self;
}

/** Allocates a @c Utf8Split of @p type over @p text and fills its first batch, running @p tokenizer
 *  for the spans @p parts selects. */
static PyObject *Utf8Split_make_(PyTypeObject *type, PyObject *text_obj, sz_string_view_t text,
                                 sz_py_tokenizer_t tokenizer, sz_capability_t capabilities, int parts, int skip_empty) {
    Utf8Split *result_obj = (Utf8Split *)type->tp_alloc(type, 0);
    if (result_obj == NULL) return NULL;
    result_obj->text_obj = text_obj;
    Py_INCREF(text_obj);
    result_obj->tokenizer = tokenizer;
    result_obj->capabilities = capabilities;
    result_obj->parts = parts;
    result_obj->skip_empty = skip_empty ? sz_true_k : sz_false_k;
    result_obj->suffix = text.start;
    result_obj->end = text.start + text.length;
    sz_status_t const status = Utf8Split_refill_(result_obj);
    if (status != sz_success_k) {
        Py_DECREF(result_obj);
        sz_py_raise_status(status, "utf8_split()");
        return NULL;
    }
    return (PyObject *)result_obj;
}

PyTypeObject Utf8SplitNewlinesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8SplitNewlines",
    .tp_basicsize = sizeof(Utf8Split),
    .tp_dealloc = (destructor)Utf8SplitType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = Utf8SplitType_iter,
    .tp_iternext = (iternextfunc)Utf8SplitType_next,
};
PyTypeObject Utf8NewlinesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Newlines",
    .tp_basicsize = sizeof(Utf8Split),
    .tp_dealloc = (destructor)Utf8SplitType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = Utf8SplitType_iter,
    .tp_iternext = (iternextfunc)Utf8SplitType_next,
};
PyTypeObject Utf8SplitWhitespacesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8SplitWhitespaces",
    .tp_basicsize = sizeof(Utf8Split),
    .tp_dealloc = (destructor)Utf8SplitType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = Utf8SplitType_iter,
    .tp_iternext = (iternextfunc)Utf8SplitType_next,
};
PyTypeObject Utf8WhitespacesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Whitespaces",
    .tp_basicsize = sizeof(Utf8Split),
    .tp_dealloc = (destructor)Utf8SplitType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = Utf8SplitType_iter,
    .tp_iternext = (iternextfunc)Utf8SplitType_next,
};
PyTypeObject Utf8SplitDelimitersType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8SplitDelimiters",
    .tp_basicsize = sizeof(Utf8Split),
    .tp_dealloc = (destructor)Utf8SplitType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = Utf8SplitType_iter,
    .tp_iternext = (iternextfunc)Utf8SplitType_next,
};
PyTypeObject Utf8DelimitersType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8Delimiters",
    .tp_basicsize = sizeof(Utf8Split),
    .tp_dealloc = (destructor)Utf8SplitType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_iter = Utf8SplitType_iter,
    .tp_iternext = (iternextfunc)Utf8SplitType_next,
};
