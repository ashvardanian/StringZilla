/**
 *  @brief Newline, whitespace, and delimiter segmentation of UTF-8 text.
 *  @file python/stringzilla/utf8_tokens.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief  Iterator splitting a UTF-8 string on the separators a segmenter kernel reports.
 *
 *  One shared layout behind every `utf8_split_*` / bare-separator iterator. The kernel's separator endpoints are
 *  the span boundaries `{0, s0.start, s0.end, ..., [region]}`; span `i` is `bounds[i] .. bounds[i+1]`, and `parts`
 *  selects which spans via a `(first, stride)` walk: `0` between-segments (lines/tokens/fields), `1` the separators
 *  themselves, `2` both interleaved (lossless). Batches refill the inline `bounds` buffer on demand.
 *
 *  Termination: `suffix == end` after a drain, or `spans == 0`.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj; //< For reference counting.

    sz_utf8_segmenter_t kernel; //< The segmenter reporting separator spans (newlines / whitespace / delimiters).
    int parts;                  //< 0 = between, 1 = separators, 2 = both interleaved.

    sz_cptr_t origin; //< Fixed text base; `bounds` are byte offsets from here.
    sz_cptr_t suffix; //< Base of the current batch (scans forward by `consumed` each refill).
    sz_cptr_t end;    //< End of original text (immutable).

    sz_bool_t skip_empty; //< Should we skip empty (zero-length) spans?
    sz_bool_t primed;     //< Whether the first batch has been filled (lazy on first `__next__`).

    /// @brief  Inline span boundaries for the current batch, relative to `origin`.
    sz_size_t bounds[2 * sz_iterators_default_steps_k + 2];
    sz_size_t spans; //< Number of yieldable spans; `spans == 0` is the end sentinel.
    sz_size_t index; //< Current boundary cursor (span is `bounds[index] .. bounds[index + 1]`).

} Utf8Split;

/** @brief  Allocates and lazily primes a `Utf8Split`. */
static PyObject *Utf8Split_make_(PyTypeObject *type, PyObject *text_obj, sz_string_view_t text,
                                 sz_utf8_segmenter_t kernel, int parts, int skip_empty);

/**
 *  @brief  Shared body for the six `utf8_split_*` / bare-separator factories. Parses `skip_empty` (and, for the
 *          `split_*` variants, `with_separators`) and allocates a `Utf8Split` of `type` for `kernel` + `base_parts`.
 */
static PyObject *Str_like_utf8_split_(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple, PyTypeObject *type, sz_utf8_segmenter_t kernel,
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
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

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
    return Utf8Split_make_(type, text_obj, text, kernel, parts, skip_empty);
}

char const doc_utf8_split_newlines[] =                                                                            //
    "Create an iterator over the content between Unicode newlines (the lines).\n"                                 //
    "\n"                                                                                                          //
    "Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.\n"                            //
    "Unlike splitlines(), this returns an iterator for memory-efficient processing.\n"                            //
    "\n"                                                                                                          //
    "Args:\n"                                                                                                     //
    "  text (Str or str or bytes): The string object.\n"                                                          //
    "  skip_empty (bool, optional): Skip empty lines (default is False).\n"                                       //
    "  with_separators (bool, optional): Interleave the newline separators too, losslessly (default is False).\n" //
    "Returns:\n"                                                                                                  //
    "  iterator: An iterator yielding the lines as Str objects.\n"                                                //
    "\n"                                                                                                          //
    "Recognized newlines:\n"                                                                                      //
    "  LF (\\n), VT (\\v), FF (\\f), CR (\\r), NEL (U+0085),\n"                                                   //
    "  LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\\r\\n)\n"                                    //
    "\n"                                                                                                          //
    "Example:\n"                                                                                                  //
    "  >>> sum(1 for _ in sz.Str('a\\nb').utf8_split_newlines())\n"                                               //
    "  2";
PyObject *Str_like_utf8_split_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8SplitNewlinesType,
                                sz_utf8_newlines, 0, 1);
}

char const doc_utf8_newlines[] =                                                                                    //
    "Create an iterator over the Unicode newline separators themselves (one per codepoint, CR+LF kept together).\n" //
    "\n"                                                                                                            //
    "Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.\n"                              //
    "Yields each newline separator; utf8_split_newlines() yields the lines between them.\n"                         //
    "\n"                                                                                                            //
    "Args:\n"                                                                                                       //
    "  text (Str or str or bytes): The string object.\n"                                                            //
    "  skip_empty (bool, optional): Skip empty segments (default is False).\n"                                      //
    "Returns:\n"                                                                                                    //
    "  iterator: An iterator yielding the newline separators as Str objects.\n"                                     //
    "\n"                                                                                                            //
    "Recognized newlines:\n"                                                                                        //
    "  LF (\\n), VT (\\v), FF (\\f), CR (\\r), NEL (U+0085),\n"                                                     //
    "  LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\\r\\n)\n"                                      //
    "\n"                                                                                                            //
    "Example:\n"                                                                                                    //
    "  >>> sum(1 for _ in sz.Str('a\\nb').utf8_newlines())\n"                                                       //
    "  1";
PyObject *Str_like_utf8_newlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8NewlinesType,
                                sz_utf8_newlines, 1, 0);
}

char const doc_utf8_split_whitespaces[] =                                                                            //
    "Create an iterator over the content between Unicode whitespace (the tokens).\n"                                 //
    "\n"                                                                                                             //
    "Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.\n"                                    //
    "Splits on whitespace, one separator per codepoint; pass skip_empty=True for str.split()-style tokens.\n"        //
    "\n"                                                                                                             //
    "Args:\n"                                                                                                        //
    "  text (Str or str or bytes): The string object.\n"                                                             //
    "  skip_empty (bool, optional): Skip empty segments (default is False).\n"                                       //
    "  with_separators (bool, optional): Interleave the whitespace separators too, losslessly (default is False).\n" //
    "Returns:\n"                                                                                                     //
    "  iterator: An iterator yielding the non-whitespace tokens as Str objects.\n"                                   //
    "\n"                                                                                                             //
    "Recognized whitespace:\n"                                                                                       //
    "  ASCII: TAB, LF, VT, FF, CR, SPACE\n"                                                                          //
    "  Latin-1: NEXT LINE, NO-BREAK SPACE\n"                                                                         //
    "  General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.\n"                                                    //
    "  CJK: IDEOGRAPHIC SPACE (U+3000)\n"                                                                            //
    "\n"                                                                                                             //
    "Example:\n"                                                                                                     //
    "  >>> sum(1 for _ in sz.Str('foo bar baz').utf8_split_whitespaces())\n"                                         //
    "  3";
PyObject *Str_like_utf8_split_whitespaces(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8SplitWhitespacesType,
                                sz_utf8_whitespaces, 0, 1);
}

char const doc_utf8_whitespaces[] = //
    "Create an iterator over the Unicode whitespace separators themselves (one per codepoint, CR+LF kept together).\n" //
    "\n"                                                                                           //
    "Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.\n"                  //
    "Yields each whitespace codepoint; utf8_split_whitespaces() yields the tokens between them.\n" //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  text (Str or str or bytes): The string object.\n"                                           //
    "  skip_empty (bool, optional): Skip empty segments (default is False).\n"                     //
    "Returns:\n"                                                                                   //
    "  iterator: An iterator yielding the whitespace separators as Str objects.\n"                 //
    "\n"                                                                                           //
    "Recognized whitespace:\n"                                                                     //
    "  ASCII: TAB, LF, VT, FF, CR, SPACE\n"                                                        //
    "  Latin-1: NEXT LINE, NO-BREAK SPACE\n"                                                       //
    "  General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.\n"                                  //
    "  CJK: IDEOGRAPHIC SPACE (U+3000)\n"                                                          //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "  >>> sum(1 for _ in sz.Str('foo bar').utf8_whitespaces())\n"                                 //
    "  1";
PyObject *Str_like_utf8_whitespaces(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8WhitespacesType,
                                sz_utf8_whitespaces, 1, 0);
}

char const doc_utf8_split_delimiters[] =                                                                            //
    "Create an iterator over the content between Unicode delimiters (the fields).\n"                                //
    "\n"                                                                                                            //
    "Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and\n"                                 //
    "separator/whitespace (Z*) codepoint - the superset of utf8_split_whitespaces().\n"                             //
    "Splits on delimiters (one separator per codepoint) and yields the segments between them.\n"                    //
    "\n"                                                                                                            //
    "Args:\n"                                                                                                       //
    "  text (Str or str or bytes): The string object.\n"                                                            //
    "  skip_empty (bool, optional): Skip empty segments (default is False).\n"                                      //
    "  with_separators (bool, optional): Interleave the delimiter separators too, losslessly (default is False).\n" //
    "Returns:\n"                                                                                                    //
    "  iterator: An iterator yielding the non-delimiter segments as Str objects.\n"                                 //
    "\n"                                                                                                            //
    "Example:\n"                                                                                                    //
    "  >>> list(str(t) for t in sz.Str('Hi, world').utf8_split_delimiters(skip_empty=True))\n"                      //
    "  ['Hi', 'world']";
PyObject *Str_like_utf8_split_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                         PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8SplitDelimitersType,
                                sz_utf8_delimiters, 0, 1);
}

char const doc_utf8_delimiters[] =                                                                                    //
    "Create an iterator over the Unicode delimiter separators themselves (one per codepoint, CR+LF kept together).\n" //
    "\n"                                                                                                              //
    "Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and\n"                                   //
    "separator/whitespace (Z*) codepoint. Yields each delimiter codepoint;\n"                                         //
    "utf8_split_delimiters() yields the fields between them.\n"                                                       //
    "\n"                                                                                                              //
    "Args:\n"                                                                                                         //
    "  text (Str or str or bytes): The string object.\n"                                                              //
    "  skip_empty (bool, optional): Skip empty segments (default is False).\n"                                        //
    "Returns:\n"                                                                                                      //
    "  iterator: An iterator yielding the delimiter separators as Str objects.\n"                                     //
    "\n"                                                                                                              //
    "Example:\n"                                                                                                      //
    "  >>> list(str(d) for d in sz.Str('a.b').utf8_delimiters())\n"                                                   //
    "  ['.']";
PyObject *Str_like_utf8_delimiters(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                   PyObject *args_names_tuple) {
    return Str_like_utf8_split_(self, args, positional_args_count, args_names_tuple, &Utf8DelimitersType,
                                sz_utf8_delimiters, 1, 0);
}

/**
 *  @brief  Refill the inline batch from `suffix`: fetch a separator batch and expand its endpoints into `bounds`.
 *          Span `i` is `bounds[i] .. bounds[i+1]`; `_settle_` / `_next` walk them by `parts`'s (first, stride).
 */
static void Utf8Split_refill_(Utf8Split *self) {
    sz_size_t base = (sz_size_t)(self->suffix - self->origin);
    sz_size_t region = (sz_size_t)(self->end - self->suffix);
    sz_size_t offsets[sz_iterators_default_steps_k];
    sz_size_t lengths[sz_iterators_default_steps_k];
    sz_size_t consumed = 0;
    sz_size_t separators = self->kernel(self->suffix, region, offsets, lengths, sz_iterators_default_steps_k,
                                        &consumed);
    sz_assert_(separators <= sz_iterators_default_steps_k &&
               "segmenter reported more spans than the requested capacity");
    sz_assert_(consumed <= region && "segmenter consumed past the region end");
    sz_assert_((consumed > 0 || region == 0) && "segmenter made no progress (the iterator would loop forever)");
    self->bounds[0] = base;
    for (sz_size_t separator = 0; separator < separators; ++separator) {
        sz_assert_(offsets[separator] + lengths[separator] <= region && "separator span runs past the region end");
        sz_assert_((separator == 0 || offsets[separator] >= offsets[separator - 1] + lengths[separator - 1]) &&
                   "separator spans are out of order or overlap");
        self->bounds[2 * separator + 1] = base + offsets[separator];
        self->bounds[2 * separator + 2] = base + offsets[separator] + lengths[separator];
    }
    sz_size_t boundaries = 2 * separators + 1;
    if (consumed == region) self->bounds[boundaries++] = base + region; // end-of-text: closing boundary
    self->spans = boundaries - 1;
    self->index = self->parts == 1 ? 1 : 0; // `separators` starts on boundary 1
    self->suffix += consumed;
    self->primed = sz_true_k;
}

/**
 *  @brief  Position `index` on the next yieldable span, refilling and (when `skip_empty`) skipping empty spans.
 *          Leaves `spans == 0` as the end sentinel.
 */
static void Utf8Split_settle_(Utf8Split *self) {
    sz_size_t stride = self->parts == 2 ? 1 : 2;
    if (!self->primed) Utf8Split_refill_(self);
    for (;;) {
        if (self->skip_empty)
            while (self->index < self->spans && self->bounds[self->index + 1] == self->bounds[self->index])
                self->index += stride;
        if (self->index < self->spans || self->spans == 0) return;
        if (self->suffix == self->end) {
            self->spans = 0;
            return;
        }
        Utf8Split_refill_(self);
    }
}

static PyObject *Utf8SplitType_next(Utf8Split *self) {
    sz_size_t stride = self->parts == 2 ? 1 : 2;
    Utf8Split_settle_(self);
    if (self->spans == 0) return NULL;

    sz_size_t i = self->index;
    self->index += stride;
    sz_cptr_t segment_start = self->origin + self->bounds[i];
    sz_size_t segment_length = self->bounds[i + 1] - self->bounds[i];

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

/** @brief  Allocates and lazily primes a `Utf8Split` of `type` over `text` for `kernel` + `parts`. */
static PyObject *Utf8Split_make_(PyTypeObject *type, PyObject *text_obj, sz_string_view_t text,
                                 sz_utf8_segmenter_t kernel, int parts, int skip_empty) {
    Utf8Split *result_obj = (Utf8Split *)type->tp_alloc(type, 0);
    if (result_obj == NULL) return PyErr_NoMemory();
    result_obj->text_obj = text_obj;
    result_obj->kernel = kernel;
    result_obj->parts = parts;
    result_obj->origin = text.start;
    result_obj->suffix = text.start;
    result_obj->end = text.start + text.length;
    result_obj->skip_empty = skip_empty ? sz_true_k : sz_false_k;
    result_obj->spans = 0;
    result_obj->index = 0;
    result_obj->primed = sz_false_k;
    Py_INCREF(text_obj);
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
