/**
 *  @brief Case-insensitive UTF-8 search, ordering, and match iteration.
 *  @file python/stringzilla/utf8_uncased.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief  Iterator that yields all uncased matches of a needle in a haystack.
 *          Uses `sz_utf8_uncased_search` for Unicode-aware case folding.
 */
typedef struct {
    PyObject ob_base;

    PyObject *haystack_obj; //< Reference for garbage collection
    PyObject *needle_obj;   //< Reference for garbage collection (needle bytes must remain valid)

    sz_cptr_t current;      //< Current search position in haystack
    sz_cptr_t haystack_end; //< End boundary of haystack

    sz_string_view_t needle; //< Needle view (bytes and length)

    /// @brief  Reusable metadata for repeated searches with the same needle.
    sz_utf8_uncased_needle_metadata_t metadata;

    /// @brief  Whether to allow overlapping matches.
    sz_bool_t include_overlapping;

} Utf8UncasedMatches;

char const doc_utf8_uncased_search[] =                                                  //
    "Find substring using Unicode uncased matching.\n"                                  //
    "\n"                                                                                //
    "Performs a uncased search using Unicode case folding rules,\n"                     //
    "correctly handling one-to-many expansions (e.g., 'ß' matches 'SS').\n"             //
    "\n"                                                                                //
    "IMPORTANT - Type-dependent behavior:\n"                                            //
    "  - str input:   start/end are CODEPOINT offsets, returns CODEPOINT offset\n"      //
    "  - bytes input: start/end are BYTE offsets, returns BYTE offset\n"                //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "    haystack (Str or str or bytes): The string to search in.\n"                    //
    "    needle (Str or str or bytes): The substring to find.\n"                        //
    "    start (int, optional): Starting index (default: 0).\n"                         //
    "    end (int, optional): Ending index (default: length).\n"                        //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n" //
    "\n"                                                                                //
    "Returns:\n"                                                                        //
    "    int: Index of the first match, or -1 if not found.\n"                          //
    "\n"                                                                                //
    "Example:\n"                                                                        //
    "    >>> sz.utf8_uncased_search('Hello World', 'WORLD')  # str: codepoint offset\n" //
    "    6\n"                                                                           //
    "    >>> sz.utf8_uncased_search('Straße', 'STRASSE')  # 'ß' = 1 codepoint\n"        //
    "    0\n"                                                                           //
    "    >>> sz.utf8_uncased_search(b'Stra\\xc3\\x9fe', b'STRASSE')  # 'ß' = 2 bytes\n" //
    "    0";

PyObject *Str_like_utf8_uncased_search(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Argument objects
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    int validate = 0;

    // Argument count validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // needle required
    Py_ssize_t const expected_max = expected_min + 3;  // + start + end + validate

    if (total_args < expected_min || total_args > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    // Extract positional arguments
    if (is_member) {
        haystack_obj = self;
        if (positional_args_count >= 1) needle_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
    }
    else {
        if (positional_args_count >= 1) haystack_obj = args[0];
        if (positional_args_count >= 2) needle_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
    }

    // Parse keyword arguments
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *val = args[positional_args_count + i];

        if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
            if (start_obj) {
                PyErr_SetString(PyExc_TypeError, "start specified twice");
                return NULL;
            }
            start_obj = val;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
            if (end_obj) {
                PyErr_SetString(PyExc_TypeError, "end specified twice");
                return NULL;
            }
            end_obj = val;
        }
        else if (PyUnicode_CompareWithASCIIString(key, "validate") == 0) {
            validate = PyObject_IsTrue(val);
            if (validate < 0) return NULL;
        }
        else {
            PyErr_Format(PyExc_TypeError, "utf8_uncased_search() got unexpected keyword argument '%U'", key);
            return NULL;
        }
    }

    // Determine if input is Unicode (str) or bytes - affects offset semantics
    int const is_unicode = PyUnicode_Check(haystack_obj);

    // Extract string views (UTF-8 bytes)
    sz_string_view_t haystack_full, needle;
    if (!sz_py_export_string_like(haystack_obj, &haystack_full.start, &haystack_full.length)) {
        wrap_current_exception("First argument (haystack) must be string-like");
        return NULL;
    }
    if (!sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Second argument (needle) must be string-like");
        return NULL;
    }

    // Parse start/end (these are codepoint offsets for str, byte offsets for bytes)
    Py_ssize_t start, end;
    if (!sz_py_export_optional_index(start_obj, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }
    if (!sz_py_export_optional_index(end_obj, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    // Convert offsets and prepare search range
    sz_size_t byte_offset_start = 0;
    sz_size_t byte_length = haystack_full.length;
    sz_size_t codepoint_offset_start = 0; // Only used for str return value
    sz_bool_t window_valid = sz_true_k;   // A degenerate [start, end) window can't hold even an empty needle

    if (is_unicode) {
        // For str: start/end are codepoint offsets, convert to byte offsets
        sz_size_t total_codepoints = sz_utf8_count(haystack_full.start, haystack_full.length);

        // Clamp codepoint offsets with CPython slice semantics (negatives count from the end)
        sz_ssize_t signed_start = start, signed_end = end;
        if (signed_start < 0) signed_start += (sz_ssize_t)total_codepoints;
        if (signed_end < 0) signed_end += (sz_ssize_t)total_codepoints;
        sz_size_t codepoint_start = signed_start < 0 ? 0 : (sz_size_t)signed_start;
        sz_size_t codepoint_end =
            signed_end < 0 ? 0 : ((sz_size_t)signed_end > total_codepoints ? total_codepoints : (sz_size_t)signed_end);
        window_valid = codepoint_start <= codepoint_end ? sz_true_k : sz_false_k;
        if (codepoint_start > codepoint_end) codepoint_start = codepoint_end;

        codepoint_offset_start = codepoint_start;

        // Convert codepoint start to byte offset
        if (codepoint_start > 0) {
            sz_cptr_t start_ptr = sz_utf8_seek(haystack_full.start, haystack_full.length, codepoint_start);
            byte_offset_start = start_ptr ? (sz_size_t)(start_ptr - haystack_full.start) : haystack_full.length;
        }

        // Convert codepoint end to byte offset
        sz_size_t byte_offset_end = haystack_full.length;
        if (codepoint_end < total_codepoints) {
            sz_cptr_t end_ptr = sz_utf8_seek(haystack_full.start, haystack_full.length, codepoint_end);
            byte_offset_end = end_ptr ? (sz_size_t)(end_ptr - haystack_full.start) : haystack_full.length;
        }

        byte_length = (byte_offset_end > byte_offset_start) ? (byte_offset_end - byte_offset_start) : 0;
    }
    else {
        // For bytes: start/end are byte offsets, use directly
        window_valid = sz_ssize_clamp_interval_checked(haystack_full.length, start, end, &byte_offset_start,
                                                       &byte_length);
    }

    // Prepare the search haystack
    sz_string_view_t haystack;
    haystack.start = haystack_full.start + byte_offset_start;
    haystack.length = byte_length;

    // Empty needle matches at the window start, unless the window is degenerate (out-of-range/inverted)
    if (needle.length == 0) {
        if (!window_valid) { return PyLong_FromSsize_t(-1); }
        return PyLong_FromSsize_t((Py_ssize_t)(is_unicode ? codepoint_offset_start : byte_offset_start));
    }
    // Empty haystack (after slicing) can't contain non-empty needle
    if (haystack.length == 0) { return PyLong_FromSsize_t(-1); }

    // Validate UTF-8 input only if requested
    if (validate) {
        if (sz_utf8_find_malformed(haystack.start, haystack.length) != SZ_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "Haystack is not valid UTF-8");
            return NULL;
        }
        if (sz_utf8_find_malformed(needle.start, needle.length) != SZ_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "Needle is not valid UTF-8");
            return NULL;
        }
    }

    sz_size_t matched_length = 0;
    sz_utf8_uncased_needle_metadata_t needle_metadata = {0}; // Zero-init triggers analysis
    sz_cptr_t result = sz_utf8_uncased_search(haystack.start, haystack.length, needle.start, needle.length,
                                              &needle_metadata, &matched_length);

    if (result == NULL) { return PyLong_FromSsize_t(-1); }

    // Compute and return the appropriate offset type
    sz_size_t result_byte_offset = (sz_size_t)(result - haystack_full.start);

    if (is_unicode) {
        // For str: return codepoint offset
        sz_size_t result_codepoint_offset = sz_utf8_count(haystack_full.start, result_byte_offset);
        return PyLong_FromSsize_t((Py_ssize_t)result_codepoint_offset);
    }
    else {
        // For bytes: return byte offset
        return PyLong_FromSsize_t((Py_ssize_t)result_byte_offset);
    }
}

char const doc_utf8_uncased_order[] =                                                   //
    "Compare two UTF-8 strings uncasedly.\n"                                            //
    "\n"                                                                                //
    "Performs lexicographical comparison using Unicode case folding,\n"                 //
    "correctly handling one-to-many expansions (e.g., 'Straße' equals 'STRASSE').\n"    //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "    a (Str or str or bytes): First string to compare.\n"                           //
    "    b (Str or str or bytes): Second string to compare.\n"                          //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n" //
    "\n"                                                                                //
    "Returns:\n"                                                                        //
    "    int: Negative if a < b, zero if equal, positive if a > b.\n"                   //
    "\n"                                                                                //
    "Example:\n"                                                                        //
    "    >>> sz.utf8_uncased_order('hello', 'HELLO')\n"                                 //
    "    0\n"                                                                           //
    "    >>> sz.utf8_uncased_order('apple', 'BANANA')\n"                                //
    "    -1";

PyObject *Str_like_utf8_uncased_order(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs_expected = is_member ? 1 : 2; // b if method, a+b if function
    int validate = 0;                              // Default: no validation

    if (positional_args_count != nargs_expected) {
        PyErr_Format(PyExc_TypeError, "utf8_uncased_order() takes exactly %zd positional argument(s)", nargs_expected);
        return NULL;
    }

    // Parse optional 'validate' keyword argument
    if (args_names_tuple) {
        Py_ssize_t nkwargs = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < nkwargs; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            if (PyUnicode_CompareWithASCIIString(key, "validate") == 0) {
                PyObject *val = args[positional_args_count + i];
                validate = PyObject_IsTrue(val);
                if (validate < 0) return NULL;
            }
            else {
                PyErr_Format(PyExc_TypeError, "utf8_uncased_order() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    PyObject *a_obj = is_member ? self : args[0];
    PyObject *b_obj = is_member ? args[0] : args[1];

    sz_string_view_t a, b;
    if (!sz_py_export_string_like(a_obj, &a.start, &a.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }
    if (!sz_py_export_string_like(b_obj, &b.start, &b.length)) {
        wrap_current_exception("Second argument must be string-like");
        return NULL;
    }

    // Validate UTF-8 input only if requested
    if (validate) {
        if (sz_utf8_find_malformed(a.start, a.length) != SZ_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "First argument is not valid UTF-8");
            return NULL;
        }
        if (sz_utf8_find_malformed(b.start, b.length) != SZ_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "Second argument is not valid UTF-8");
            return NULL;
        }
    }

    sz_ordering_t order = sz_utf8_uncased_order(a.start, a.length, b.start, b.length);
    return PyLong_FromLong((long)order);
}

char const doc_utf8_uncased_matches[] =                                                      //
    "utf8_uncased_matches(haystack, needle, /, include_overlapping=False)\n"                 //
    "\n"                                                                                     //
    "Iterate over all uncased matches of needle in haystack.\n"                              //
    "\n"                                                                                     //
    "This function uses Unicode case folding for proper handling of\n"                       //
    "international text. The matched region length may differ from the\n"                    //
    "needle length due to case folding expansions (e.g., 'ß' matches 'SS').\n"               //
    "\n"                                                                                     //
    "Args:\n"                                                                                //
    "    haystack (Str or str or bytes): The string to search in.\n"                         //
    "    needle (Str or str or bytes): The pattern to find.\n"                               //
    "    include_overlapping (bool, optional): Allow overlapping matches (default False).\n" //
    "\n"                                                                                     //
    "Yields:\n"                                                                              //
    "    Str: Each matched region as a view into the original haystack.\n"                   //
    "\n"                                                                                     //
    "Examples:\n"                                                                            //
    "    >>> list(sz.utf8_uncased_matches('Hello HELLO hello', 'hello'))\n"                  //
    "    [sz.Str('Hello'), sz.Str('HELLO'), sz.Str('hello')]\n"                              //
    "    >>> list(sz.utf8_uncased_matches('Straße STRASSE', 'strasse'))\n"                   //
    "    [sz.Str('Straße'), sz.Str('STRASSE')]";

PyObject *Str_like_utf8_uncased_matches(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                        PyObject *kwnames) {
    // Check if called as member or module function
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    int min_args = is_member ? 1 : 2;
    int max_args = is_member ? 2 : 3;

    if (positional_args_count < min_args || positional_args_count > max_args) {
        PyErr_Format(PyExc_TypeError, "utf8_uncased_matches() requires %d to %d positional arguments, got %zd",
                     min_args, max_args, positional_args_count);
        return NULL;
    }

    PyObject *haystack_obj = is_member ? self : args[0];
    PyObject *needle_obj = is_member ? args[0] : args[1];
    int include_overlapping = 0;

    // Parse keyword arguments
    if (kwnames) {
        Py_ssize_t n_kwnames = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < n_kwnames; ++i) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "include_overlapping") == 0) {
                include_overlapping = PyObject_IsTrue(value);
            }
            else {
                PyErr_Format(PyExc_TypeError, "utf8_uncased_matches() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    // Check positional include_overlapping argument
    if (positional_args_count > max_args - 1) include_overlapping = PyObject_IsTrue(args[is_member ? 1 : 2]);

    // Extract haystack and needle views
    sz_string_view_t haystack_view, needle_view;
    if (!sz_py_export_string_like(haystack_obj, &haystack_view.start, &haystack_view.length) ||
        !sz_py_export_string_like(needle_obj, &needle_view.start, &needle_view.length)) {
        return NULL; // Exception already set by helper
    }

    // Handle edge case: empty needle yields nothing
    if (needle_view.length == 0) {
        // Return an empty iterator by setting current = end
        Utf8UncasedMatches *iter = PyObject_New(Utf8UncasedMatches, &Utf8UncasedMatchesType);
        if (!iter) return PyErr_NoMemory();

        iter->haystack_obj = haystack_obj;
        Py_INCREF(haystack_obj);
        iter->needle_obj = needle_obj;
        Py_INCREF(needle_obj);
        iter->current = haystack_view.start + haystack_view.length; // Start at end = empty iterator
        iter->haystack_end = haystack_view.start + haystack_view.length;
        iter->needle = needle_view;
        memset(&iter->metadata, 0, sizeof(iter->metadata));
        iter->include_overlapping = sz_false_k;

        return (PyObject *)iter;
    }

    // Allocate iterator
    Utf8UncasedMatches *iter = PyObject_New(Utf8UncasedMatches, &Utf8UncasedMatchesType);
    if (!iter) return PyErr_NoMemory();

    iter->haystack_obj = haystack_obj;
    Py_INCREF(haystack_obj);
    iter->needle_obj = needle_obj;
    Py_INCREF(needle_obj);
    iter->current = haystack_view.start;
    iter->haystack_end = haystack_view.start + haystack_view.length;
    iter->needle = needle_view;
    memset(&iter->metadata, 0, sizeof(iter->metadata));
    iter->include_overlapping = include_overlapping ? sz_true_k : sz_false_k;

    return (PyObject *)iter;
}

static PyObject *Utf8UncasedMatchesType_next(Utf8UncasedMatches *self) {
    // Check if we've reached the end
    sz_size_t remaining = (sz_size_t)(self->haystack_end - self->current);
    if (remaining == 0) return NULL;

    // Search for next match
    sz_size_t matched_length = 0;
    sz_cptr_t match = sz_utf8_uncased_search(self->current, remaining, self->needle.start, self->needle.length,
                                             &self->metadata, &matched_length);

    if (!match) return NULL;

    // Create a new `Str` object for the matched region
    Str *result_obj = Str_alloc_();
    if (result_obj == NULL) return PyErr_NoMemory();

    result_obj->memory.start = match;
    result_obj->memory.length = matched_length;
    result_obj->parent = self->haystack_obj;
    Py_INCREF(self->haystack_obj);

    // Advance position for next search
    if (self->include_overlapping) {
        // Move forward by one UTF-8 codepoint to allow overlapping matches
        sz_size_t pos = 0;
        sz_utf8_next_rune_(match, matched_length, &pos);
        self->current = match + (pos > 0 ? pos : 1);
    }
    else {
        // Move past the entire matched region (non-overlapping)
        self->current = match + matched_length;
    }

    return (PyObject *)result_obj;
}

static void Utf8UncasedMatchesType_dealloc(Utf8UncasedMatches *self) {
    Py_XDECREF(self->haystack_obj);
    Py_XDECREF(self->needle_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *Utf8UncasedMatchesType_iter(PyObject *self) {
    Py_INCREF(self);
    return self;
}

static char const doc_Utf8UncasedMatches[] =                                    //
    "Utf8UncasedMatches(haystack, needle, ...)\n"                               //
    "\n"                                                                        //
    "Iterator yielding all uncased matches of needle in haystack.\n"            //
    "Uses Unicode case folding for proper handling of international text.\n"    //
    "\n"                                                                        //
    "Created by:\n"                                                             //
    "  - Str.utf8_uncased_matches()\n"                                          //
    "  - sz.utf8_uncased_matches()\n"                                           //
    "\n"                                                                        //
    "Each iteration yields a Str view of the matched region in the haystack.\n" //
    "The matched length may differ from needle length due to case folding\n"    //
    "expansions (e.g., German 'ß' matches 'SS').\n"                             //
    "\n"                                                                        //
    "Example:\n"                                                                //
    "  >>> len(list(sz.utf8_uncased_matches('aAaA', 'a')))\n"                   //
    "  4";

PyTypeObject Utf8UncasedMatchesType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Utf8UncasedMatches",
    .tp_basicsize = sizeof(Utf8UncasedMatches),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)Utf8UncasedMatchesType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_Utf8UncasedMatches,
    .tp_iter = Utf8UncasedMatchesType_iter,
    .tp_iternext = (iternextfunc)Utf8UncasedMatchesType_next,
};
