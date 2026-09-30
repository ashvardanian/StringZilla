/**
 *  @file python/stringzilla/utf8_uncased.c
 *  @author Ash Vardanian
 *  @date November 28, 2025
 *  @brief Case-insensitive UTF-8 search, ordering, and match iteration.
 */
#include "stringzilla.h"

/** Iterator that yields all uncased matches of a needle in a haystack, prepared once at
 *  construction for @c sz_utf8_uncased_search_best. */
typedef struct {
    PyObject ob_base;

    PyObject *haystack_obj;

    /** Keeps alive the needle bytes that @c needle points at. */
    PyObject *needle_obj;

    /** Where the next search starts, or NULL once exhausted. */
    sz_cptr_t current;
    sz_cptr_t haystack_end;

    sz_utf8_uncased_needle_t needle;
    sz_bool_t include_overlapping;
    sz_capability_t capabilities;

} Utf8UncasedMatches;

char const doc_utf8_uncased_search[] =                                                                   //
    "Find substring using Unicode uncased matching.\n"                                                   //
    "\n"                                                                                                 //
    "Performs a uncased search using Unicode case folding rules,\n"                                      //
    "correctly handling one-to-many expansions, like 'ß' matching 'SS'.\n"                               //
    "\n"                                                                                                 //
    "IMPORTANT - Type-dependent behavior:\n"                                                             //
    "  - str input:   start/end are CODEPOINT offsets, returns CODEPOINT offset\n"                       //
    "  - bytes input: start/end are BYTE offsets, returns BYTE offset\n"                                 //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "    haystack (Str or str or bytes): The string to search in.\n"                                     //
    "    needle (Str or str or bytes): The substring to find.\n"                                         //
    "    start (int, optional): Starting index, defaulting to 0.\n"                                      //
    "    end (int, optional): Ending index, defaulting to length.\n"                                     //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n"                  //
    "    capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.\n" //
    "\n"                                                                                                 //
    "Returns:\n"                                                                                         //
    "    int: Index of the first match, or -1 if not found.\n"                                           //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "    >>> sz.utf8_uncased_search('Hello World', 'WORLD')  # str: codepoint offset\n"                  //
    "    6\n"                                                                                            //
    "    >>> sz.utf8_uncased_search('Straße', 'STRASSE')  # 'ß' = 1 codepoint\n"                         //
    "    0\n"                                                                                            //
    "    >>> sz.utf8_uncased_search(b'Stra\\xc3\\x9fe', b'STRASSE')  # 'ß' = 2 bytes\n"                  //
    "    0";

PyObject *Str_like_utf8_uncased_search(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Argument objects
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *capabilities_object = NULL;
    int validate = 0;

    // Argument count validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // needle required
    Py_ssize_t const expected_max = expected_min + 4;  // + start + end + validate + capabilities

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
        else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = val; }
        else {
            PyErr_Format(PyExc_TypeError, "utf8_uncased_search() got unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

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
    sz_status_t status = sz_success_k;

    if (is_unicode) {
        // For str: start/end are codepoint offsets, convert to byte offsets
        sz_size_t total_codepoints = 0;
        status = sz_utf8_count_best(haystack_full.start, haystack_full.length, &total_codepoints, capabilities, NULL);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "utf8_uncased_search()");
            return NULL;
        }

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

        // Convert codepoint offsets to byte offsets, a null seek landing past the last rune
        sz_cptr_t start_ptr = haystack_full.start, end_ptr = NULL;
        if (codepoint_start > 0)
            status = sz_utf8_seek_best(haystack_full.start, haystack_full.length, codepoint_start, &start_ptr,
                                       capabilities, NULL);
        if (codepoint_end < total_codepoints && status == sz_success_k)
            status = sz_utf8_seek_best(haystack_full.start, haystack_full.length, codepoint_end, &end_ptr, capabilities,
                                       NULL);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "utf8_uncased_search()");
            return NULL;
        }
        byte_offset_start = start_ptr ? (sz_size_t)(start_ptr - haystack_full.start) : haystack_full.length;
        sz_size_t byte_offset_end = end_ptr ? (sz_size_t)(end_ptr - haystack_full.start) : haystack_full.length;

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
        if (sz_utf8_find_malformed(haystack.start, haystack.length) != STRINGZILLA_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "Haystack is not valid UTF-8");
            return NULL;
        }
        if (sz_utf8_find_malformed(needle.start, needle.length) != STRINGZILLA_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "Needle is not valid UTF-8");
            return NULL;
        }
    }

    sz_size_t match_length = 0;
    sz_utf8_uncased_needle_t prepared;
    sz_cptr_t result = NULL;
    status = sz_utf8_uncased_needle_init_best(needle.start, needle.length, &prepared, capabilities, NULL);
    if (status == sz_success_k)
        status = sz_utf8_uncased_search_best(haystack.start, haystack.length, &prepared, &result, &match_length,
                                             capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "utf8_uncased_search()");
        return NULL;
    }

    if (result == NULL) { return PyLong_FromSsize_t(-1); }

    // Compute and return the appropriate offset type
    sz_size_t result_byte_offset = (sz_size_t)(result - haystack_full.start);

    if (is_unicode) {
        // For str: return codepoint offset
        sz_size_t result_codepoint_offset = 0;
        status = sz_utf8_count_best(haystack_full.start, result_byte_offset, &result_codepoint_offset, capabilities,
                                    NULL);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "utf8_uncased_search()");
            return NULL;
        }
        return PyLong_FromSsize_t((Py_ssize_t)result_codepoint_offset);
    }
    else {
        // For bytes: return byte offset
        return PyLong_FromSsize_t((Py_ssize_t)result_byte_offset);
    }
}

char const doc_utf8_uncased_order[] =                                                                    //
    "Compare two UTF-8 strings uncasedly.\n"                                                             //
    "\n"                                                                                                 //
    "Performs lexicographical comparison using Unicode case folding, correctly handling\n"               //
    "one-to-many expansions, like 'Straße' equaling 'STRASSE'.\n"                                        //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "    a (Str or str or bytes): First string to compare.\n"                                            //
    "    b (Str or str or bytes): Second string to compare.\n"                                           //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n"                  //
    "    capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.\n" //
    "\n"                                                                                                 //
    "Returns:\n"                                                                                         //
    "    int: Negative if a < b, zero if equal, positive if a > b.\n"                                    //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "    >>> sz.utf8_uncased_order('hello', 'HELLO')\n"                                                  //
    "    0\n"                                                                                            //
    "    >>> sz.utf8_uncased_order('apple', 'BANANA')\n"                                                 //
    "    -1";

PyObject *Str_like_utf8_uncased_order(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs_expected = is_member ? 1 : 2; // b if method, a+b if function
    int validate = 0;                              // Default: no validation
    PyObject *capabilities_object = NULL;

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
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                capabilities_object = args[positional_args_count + i];
            }
            else {
                PyErr_Format(PyExc_TypeError, "utf8_uncased_order() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

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
        if (sz_utf8_find_malformed(a.start, a.length) != STRINGZILLA_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "First argument is not valid UTF-8");
            return NULL;
        }
        if (sz_utf8_find_malformed(b.start, b.length) != STRINGZILLA_NULL_CHAR) {
            PyErr_SetString(PyExc_ValueError, "Second argument is not valid UTF-8");
            return NULL;
        }
    }

    sz_ordering_t order = sz_equal_k;
    sz_status_t const status = sz_utf8_uncased_order_best(a.start, a.length, b.start, b.length, &order, capabilities,
                                                          NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "utf8_uncased_order()");
        return NULL;
    }
    return PyLong_FromLong((long)order);
}

char const doc_utf8_uncased_matches[] =                                                                  //
    "utf8_uncased_matches(haystack, needle, /, include_overlapping=False, *, capabilities=None)\n"       //
    "\n"                                                                                                 //
    "Iterate over all uncased matches of needle in haystack.\n"                                          //
    "\n"                                                                                                 //
    "This function uses Unicode case folding for proper handling of\n"                                   //
    "international text. The matched region length may differ from the\n"                                //
    "needle length due to case folding expansions, like 'ß' matching 'SS'.\n"                            //
    "An empty needle matches at every codepoint boundary, the end included.\n"                           //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "    haystack (Str or str or bytes): The string to search in.\n"                                     //
    "    needle (Str or str or bytes): The pattern to find.\n"                                           //
    "    include_overlapping (bool, optional): Allow overlapping matches, defaulting to False.\n"        //
    "    capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.\n" //
    "\n"                                                                                                 //
    "Yields:\n"                                                                                          //
    "    Str: Each matched region as a view into the original haystack.\n"                               //
    "\n"                                                                                                 //
    "Examples:\n"                                                                                        //
    "    >>> list(sz.utf8_uncased_matches('Hello HELLO hello', 'hello'))\n"                              //
    "    [sz.Str('Hello'), sz.Str('HELLO'), sz.Str('hello')]\n"                                          //
    "    >>> list(sz.utf8_uncased_matches('Straße STRASSE', 'strasse'))\n"                               //
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
    PyObject *overlapping_object = positional_args_count == max_args ? args[max_args - 1] : NULL;
    PyObject *capabilities_object = NULL;

    // Parse keyword arguments
    if (kwnames) {
        Py_ssize_t n_kwnames = PyTuple_GET_SIZE(kwnames);
        for (Py_ssize_t i = 0; i < n_kwnames; ++i) {
            PyObject *key = PyTuple_GET_ITEM(kwnames, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "include_overlapping") == 0) { overlapping_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "utf8_uncased_matches() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    int const include_overlapping = overlapping_object ? PyObject_IsTrue(overlapping_object) : 0;
    if (include_overlapping < 0) return NULL;

    // Extract haystack and needle views
    sz_string_view_t haystack_view, needle_view;
    if (!sz_py_export_string_like(haystack_obj, &haystack_view.start, &haystack_view.length) ||
        !sz_py_export_string_like(needle_obj, &needle_view.start, &needle_view.length)) {
        return NULL; // Exception already set by helper
    }

    sz_utf8_uncased_needle_t needle;
    sz_status_t const status = sz_utf8_uncased_needle_init_best(needle_view.start, needle_view.length, &needle,
                                                                capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "utf8_uncased_matches()");
        return NULL;
    }

    Utf8UncasedMatches *iter = PyObject_New(Utf8UncasedMatches, &Utf8UncasedMatchesType);
    if (!iter) return PyErr_NoMemory();

    iter->haystack_obj = haystack_obj;
    Py_INCREF(haystack_obj);
    iter->needle_obj = needle_obj;
    Py_INCREF(needle_obj);
    iter->current = haystack_view.start;
    iter->haystack_end = haystack_view.start + haystack_view.length;
    iter->needle = needle;
    iter->include_overlapping = include_overlapping ? sz_true_k : sz_false_k;
    iter->capabilities = capabilities;

    return (PyObject *)iter;
}

static PyObject *Utf8UncasedMatchesType_next(Utf8UncasedMatches *self) {
    if (!self->current) return NULL;

    sz_size_t match_length = 0;
    sz_cptr_t match = NULL;
    sz_status_t const status = sz_utf8_uncased_search_best(
        self->current, (sz_size_t)(self->haystack_end - self->current), &self->needle, &match, &match_length,
        self->capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "__next__()");
        return NULL;
    }
    if (!match) {
        self->current = NULL;
        return NULL;
    }

    Str *result_obj = Str_alloc_();
    if (result_obj == NULL) return PyErr_NoMemory();

    result_obj->memory.start = match;
    result_obj->memory.length = match_length;
    result_obj->parent = self->haystack_obj;
    Py_INCREF(self->haystack_obj);

    // A zero-width match steps one codepoint too, so an empty needle stops after matching the end.
    if (self->include_overlapping || match_length == 0) {
        sz_size_t const remaining = (sz_size_t)(self->haystack_end - match);
        sz_size_t const step = sz_utf8_lead_length_(*(sz_u8_t const *)match);
        self->current = remaining ? match + (step < remaining ? step : remaining) : NULL;
    }
    else { self->current = match + match_length; }

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
    "expansions, like German 'ß' matching 'SS'.\n"                              //
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
