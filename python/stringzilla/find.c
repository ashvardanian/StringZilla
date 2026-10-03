/**
 *  @file python/stringzilla/find.c
 *  @author Ash Vardanian
 *  @date September 16, 2023
 *  @brief Substring and byteset search, partitioning, counting, and splitting.
 */
#include "stringzilla.h"

/** The shape of @c sz_find_best, which every search this file parameterizes on shares. */
typedef sz_status_t (*sz_py_finder_t)(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                      sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                      void *stream);

/** Finds the first byte of @p haystack present in @p needle, as @c sz_find_byte_from does. */
static sz_status_t sz_py_find_byte_from_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                         void *stream) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_find_byteset_best(haystack, haystack_length, &set, match, capabilities, stream);
}

/** Finds the first byte of @p haystack absent from @p needle, as @c sz_find_byte_not_from does. */
static sz_status_t sz_py_find_byte_not_from_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                             void *stream) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_find_byteset_best(haystack, haystack_length, &set, match, capabilities, stream);
}

/** Finds the last byte of @p haystack present in @p needle, as @c sz_rfind_byte_from does. */
static sz_status_t sz_py_rfind_byte_from_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                          void *stream) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    return sz_rfind_byteset_best(haystack, haystack_length, &set, match, capabilities, stream);
}

/** Finds the last byte of @p haystack absent from @p needle, as @c sz_rfind_byte_not_from does. */
static sz_status_t sz_py_rfind_byte_not_from_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length, sz_cptr_t *match, sz_capability_t capabilities,
                                              void *stream) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; needle_length; ++needle, --needle_length) sz_byteset_add(&set, *needle);
    sz_byteset_invert(&set);
    return sz_rfind_byteset_best(haystack, haystack_length, &set, match, capabilities, stream);
}

typedef struct sz_py_split_t sz_py_split_t;

/** Finds the next separator in the remainder of @p split, from its front or from its back. */
typedef sz_status_t (*sz_py_split_finder_t)(sz_py_split_t const *split, sz_cptr_t *match);

/**
 *  @brief A split in progress, shared by the lazy @c FindSplits and the eager @c split family.
 *
 *  `[start, end)` is the remainder not split yet: forward splits cut parts off its front, reverse
 *  ones off its back. A substring finder looks for @c separator, a byteset finder for @c set.
 */
struct sz_py_split_t {
    sz_cptr_t start;
    sz_cptr_t end;
    sz_py_split_finder_t finder;
    sz_string_view_t separator;
    sz_byteset_t set;
    sz_capability_t capabilities;

    /** Bytes a separator spans: the separator length, or 1 for a byteset. */
    sz_size_t match_length;

    /** Parts left to yield, 0 once the last one is out. */
    sz_size_t max_parts;

    sz_bool_t include_match;
    sz_bool_t is_reverse;
    sz_bool_t skip_empty;
};

static sz_status_t sz_py_split_find_(sz_py_split_t const *split, sz_cptr_t *match) {
    return sz_find_best(split->start, (sz_size_t)(split->end - split->start), split->separator.start,
                        split->separator.length, match, split->capabilities, NULL);
}

static sz_status_t sz_py_split_rfind_(sz_py_split_t const *split, sz_cptr_t *match) {
    return sz_rfind_best(split->start, (sz_size_t)(split->end - split->start), split->separator.start,
                         split->separator.length, match, split->capabilities, NULL);
}

static sz_status_t sz_py_split_find_byteset_(sz_py_split_t const *split, sz_cptr_t *match) {
    return sz_find_byteset_best(split->start, (sz_size_t)(split->end - split->start), &split->set, match,
                                split->capabilities, NULL);
}

static sz_status_t sz_py_split_rfind_byteset_(sz_py_split_t const *split, sz_cptr_t *match) {
    return sz_rfind_byteset_best(split->start, (sz_size_t)(split->end - split->start), &split->set, match,
                                 split->capabilities, NULL);
}

/**
 *  @brief Cuts the next part off the remainder of @p split.
 *  @return 1 with @p part set, 0 once every part is out, or -1 with a Python exception set.
 */
static int sz_py_split_next_(sz_py_split_t *split, sz_string_view_t *part) {
    do {
        if (split->max_parts == 0) return 0;
        sz_cptr_t match = NULL;
        if (split->max_parts != 1) {
            sz_status_t const status = split->finder(split, &match);
            if (status != sz_success_k) {
                sz_py_raise_status(status, "split()");
                return -1;
            }
        }
        if (!match) {
            part->start = split->start;
            part->length = (sz_size_t)(split->end - split->start);
            split->max_parts = 0;
        }
        else if (split->is_reverse) {
            part->start = match + split->match_length * !split->include_match;
            part->length = (sz_size_t)(split->end - part->start);
            split->end = match;
            split->max_parts--;
        }
        else {
            part->start = split->start;
            part->length = (sz_size_t)(match - split->start) + split->match_length * split->include_match;
            split->start = match + split->match_length;
            split->max_parts--;
        }
    } while (split->skip_empty && part->length == 0);
    return 1;
}

/**
 *  @brief String-splitting iterator.
 *
 *  Allows lazy evaluation of @c split and @c rsplit, which might be more memory-friendly than
 *  greedily invoking `str.split`.
 */
typedef struct {
    PyObject ob_base;

    PyObject *text_obj;

    /** Keeps the bytes of a substring separator alive. */
    PyObject *separator_obj;

    sz_py_split_t split;

} FindSplits;

/**
 *  @brief Called by @c PySequence_Contains to check the presence of a substring.
 *  @return 1 if the string is present, 0 if it is not, -1 in case of error.
 *  @see Docs: https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
 */
int Str_in(Str *self, PyObject *needle_obj) {

    sz_string_view_t needle;
    if (!sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Unsupported needle layout");
        return -1;
    }

    if (needle.length == 0) return 1; // CPython: the empty string is a substring of every string
    sz_cptr_t match = NULL;
    sz_status_t const status = sz_find_best(self->memory.start, self->memory.length, needle.start, needle.length,
                                            &match, sz_cap_cpus_k, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "in");
        return -1;
    }
    return match != NULL;
}

/**
 *  @brief Implementation of all search-like operations, parameterized by a function callback.
 *  @return 1 on success, 0 on failure.
 */
static int Str_find_implementation_( //
    PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count, PyObject *args_names_tuple,
    sz_py_finder_t finder, sz_bool_t is_reverse, sz_bool_t needle_is_byteset, Py_ssize_t *offset_out,
    sz_string_view_t *haystack_out, sz_string_view_t *needle_out) {

    // Fast path variables
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *capabilities_object = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // needle is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (positional_args_count < expected_min || positional_args_count + args_names_count > expected_max + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return 0;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return 0;
    }

    // Fast positional argument extraction
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

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return 0;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return 0;
                }
                end_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return 0;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return 0;

    sz_string_view_t haystack;
    sz_string_view_t needle;
    Py_ssize_t start, end;

    // Validate and convert `haystack` and `needle`
    if (!sz_py_export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Haystack and needle must be string-like");
        return 0;
    }

    // Validate and convert `start`/`end` (None → default, out-of-ssize_t → clamped)
    if (!sz_py_export_optional_index(start_obj, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "The start argument must be an integer");
        return 0;
    }
    if (!sz_py_export_optional_index(end_obj, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "The end argument must be an integer");
        return 0;
    }

    // Limit the `haystack` range
    sz_size_t normalized_offset, normalized_length;
    sz_bool_t const window_valid = sz_ssize_clamp_interval_checked(haystack.length, start, end, &normalized_offset,
                                                                   &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    // Empty needle: match CPython `str.find`/`str.rfind`. A forward match sits at the window start, a
    // reverse match at its end; a degenerate window reports "not found" so that `start = index + 1`
    // scans terminate instead of looping on a clamped offset. Bytesets fall through to the finder,
    // which already reports "no member" (`*_of`) or "every member" (`*_not_of`) for an empty set.
    if (needle.length == 0 && !needle_is_byteset) {
        *offset_out = window_valid
                          ? (Py_ssize_t)(is_reverse ? normalized_offset + normalized_length : normalized_offset)
                          : -1;
        *haystack_out = haystack;
        *needle_out = needle;
        return 1;
    }

    // Perform contains operation
    sz_cptr_t match = NULL;
    sz_status_t const status = finder(haystack.start, haystack.length, needle.start, needle.length, &match,
                                      capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "search");
        return 0;
    }
    if (match == NULL) { *offset_out = -1; }
    else { *offset_out = (Py_ssize_t)(match - haystack.start + normalized_offset); }

    *haystack_out = haystack;
    *needle_out = needle;
    return 1;
}

char const doc_contains[] =                                                                               //
    "Check if a string contains a substring.\n"                                                           //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  substring (str): The substring to search for.\n"                                                   //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  bool: True if the substring is found, False otherwise.\n"                                          //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').contains('ell')\n"                                                             //
    "  True";

PyObject *Str_like_contains(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find_best, sz_false_k,
                                  sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) { Py_RETURN_FALSE; }
    else { Py_RETURN_TRUE; }
}

char const doc_find[] =                                                                                   //
    "Find the first occurrence of a substring.\n"                                                         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  substring (str): The substring to find.\n"                                                         //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: The index of the first occurrence, or -1 if not found.\n"                                     //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').find('l')\n"                                                                   //
    "  2";

PyObject *Str_like_find(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                        PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find_best, sz_false_k,
                                  sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_index[] =                                                                                  //
    "Find the first occurrence of a substring or raise an error if not found.\n"                          //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  substring (str): The substring to find.\n"                                                         //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: The index of the first occurrence.\n"                                                         //
    "Raises:\n"                                                                                           //
    "  ValueError: If the substring is not found.\n"                                                      //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').index('l')\n"                                                                  //
    "  2";

PyObject *Str_like_index(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find_best, sz_false_k,
                                  sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_rfind[] =                                                                                  //
    "Find the last occurrence of a substring.\n"                                                          //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  substring (str): The substring to find.\n"                                                         //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: The index of the last occurrence, or -1 if not found.\n"                                      //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').rfind('l')\n"                                                                  //
    "  3";

PyObject *Str_like_rfind(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind_best, sz_true_k,
                                  sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_rindex[] =                                                                                 //
    "Find the last occurrence of a substring or raise an error if not found.\n"                           //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  substring (str): The substring to find.\n"                                                         //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: The index of the last occurrence.\n"                                                          //
    "Raises:\n"                                                                                           //
    "  ValueError: If the substring is not found.\n"                                                      //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').rindex('l')\n"                                                                 //
    "  3";

PyObject *Str_like_rindex(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind_best, sz_true_k,
                                  sz_false_k, &signed_offset, &text, &separator))
        return NULL;
    if (signed_offset == -1) {
        PyErr_SetString(PyExc_ValueError, "substring not found");
        return NULL;
    }
    return PyLong_FromSsize_t(signed_offset);
}

static PyObject *Str_partition_implementation_(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple, sz_py_finder_t finder,
                                               sz_bool_t is_reverse) {
    Py_ssize_t separator_index;
    sz_string_view_t text;
    sz_string_view_t separator;
    PyObject *result_tuple;

    // Use `Str_find_implementation_` to get the index of the separator
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, finder, is_reverse, sz_false_k,
                                  &separator_index, &text, &separator))
        return NULL;

    // If the separator length is zero, we must raise a `ValueError`
    if (separator.length == 0) {
        PyErr_SetString(PyExc_ValueError, "empty separator");
        return NULL;
    }

    // If separator is not found, return a tuple (self, "", "")
    if (separator_index == -1) {
        PyObject *empty_str1 = Str_new(&StrType, Py_None, Py_None);
        PyObject *empty_str2 = Str_new(&StrType, Py_None, Py_None);
        if (!empty_str1 || !empty_str2) {
            Py_XDECREF(empty_str1);
            Py_XDECREF(empty_str2);
            return NULL;
        }

        result_tuple = PyTuple_New(3);
        if (!result_tuple) {
            Py_DECREF(empty_str1);
            Py_DECREF(empty_str2);
            return NULL;
        }
        Py_INCREF(self);
        PyTuple_SET_ITEM(result_tuple, 0, self);
        PyTuple_SET_ITEM(result_tuple, 1, empty_str1);
        PyTuple_SET_ITEM(result_tuple, 2, empty_str2);
        return result_tuple;
    }

    // Create the three parts manually
    Str *before = Str_new(&StrType, NULL, NULL);
    Str *middle = Str_new(&StrType, NULL, NULL);
    Str *after = Str_new(&StrType, NULL, NULL);
    if (!before || !middle || !after) {
        Py_XDECREF(before);
        Py_XDECREF(middle);
        Py_XDECREF(after);
        return NULL;
    }

    before->parent = self, before->memory.start = text.start, before->memory.length = separator_index;
    middle->parent = self, middle->memory.start = text.start + separator_index,
    middle->memory.length = separator.length;
    after->parent = self, after->memory.start = text.start + separator_index + separator.length,
    after->memory.length = text.length - separator_index - separator.length;

    // All parts reference the same parent
    Py_INCREF(self);
    Py_INCREF(self);
    Py_INCREF(self);

    // Build the result tuple
    result_tuple = PyTuple_New(3);
    if (!result_tuple) {
        Py_DECREF(before);
        Py_DECREF(middle);
        Py_DECREF(after);
        return NULL;
    }
    PyTuple_SET_ITEM(result_tuple, 0, before);
    PyTuple_SET_ITEM(result_tuple, 1, middle);
    PyTuple_SET_ITEM(result_tuple, 2, after);

    return result_tuple;
}

char const doc_partition[] =                                                                              //
    "Split the string into a 3-tuple around the first occurrence of a separator.\n"                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separator (str): The separator to partition by.\n"                                                 //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  tuple: A 3-tuple (head, separator, tail), or `(self, '', '')` if the separator is not found.\n"    //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> tuple(map(str, sz.Str('a=b=c').partition('=')))\n"                                             //
    "  ('a', '=', 'b=c')";

PyObject *Str_like_partition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    return Str_partition_implementation_(self, args, positional_args_count, args_names_tuple, &sz_find_best,
                                         sz_false_k);
}

char const doc_rpartition[] =                                                                              //
    "Split the string into a 3-tuple around the last occurrence of a separator.\n"                         //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  text (Str or str or bytes): The string object.\n"                                                   //
    "  separator (str): The separator to partition by.\n"                                                  //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n"  //
    "Returns:\n"                                                                                           //
    "  tuple: A 3-tuple (head, separator, tail). If the separator is not found, returns ('', '', self).\n" //
    "\n"                                                                                                   //
    "Example:\n"                                                                                           //
    "  >>> tuple(map(str, sz.Str('a=b=c').rpartition('=')))\n"                                             //
    "  ('a=b', '=', 'c')";

PyObject *Str_like_rpartition(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    return Str_partition_implementation_(self, args, positional_args_count, args_names_tuple, &sz_rfind_best,
                                         sz_true_k);
}

char const doc_count[] =                                                                                  //
    "Count the occurrences of a substring.\n"                                                             //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  substring (str): The substring to count.\n"                                                        //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  allowoverlap (bool, optional): Count overlapping occurrences, defaulting to False.\n"              //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: The number of occurrences of the substring.\n"                                                //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('banana').count('a')\n"                                                                 //
    "  3";

PyObject *Str_like_count(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *allowoverlap_obj = NULL;
    PyObject *capabilities_object = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // needle is required
    Py_ssize_t const expected_max = expected_min + 3;  // + start + end + allowoverlap

    if (positional_args_count < expected_min || positional_args_count + args_names_count > expected_max + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        haystack_obj = self;
        if (positional_args_count >= 1) needle_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
        if (positional_args_count >= 4) allowoverlap_obj = args[3];
    }
    else {
        if (positional_args_count >= 1) haystack_obj = args[0];
        if (positional_args_count >= 2) needle_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
        if (positional_args_count >= 5) allowoverlap_obj = args[4];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "allowoverlap") == 0) {
                if (allowoverlap_obj) {
                    PyErr_SetString(PyExc_TypeError, "allowoverlap specified twice");
                    return NULL;
                }
                allowoverlap_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t haystack;
    sz_string_view_t needle;
    Py_ssize_t start, end;
    if (!sz_py_export_optional_index(start_obj, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "The start argument must be an integer");
        return NULL;
    }
    if (!sz_py_export_optional_index(end_obj, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "The end argument must be an integer");
        return NULL;
    }
    int const allowoverlap = allowoverlap_obj ? PyObject_IsTrue(allowoverlap_obj) : 0;
    if (allowoverlap < 0) return NULL;

    if (!sz_py_export_string_like(haystack_obj, &haystack.start, &haystack.length) ||
        !sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("Haystack and needle must be string-like");
        return NULL;
    }

    sz_size_t normalized_offset, normalized_length;
    sz_bool_t const window_valid = sz_ssize_clamp_interval_checked(haystack.length, start, end, &normalized_offset,
                                                                   &normalized_length);
    haystack.start += normalized_offset;
    haystack.length = normalized_length;

    sz_size_t count = 0;
    // CPython counts the empty needle at every gap: `len + 1` positions within a valid window.
    if (needle.length == 0) { count = window_valid ? normalized_length + 1 : 0; }
    else if (haystack.length == 0 || haystack.length < needle.length) { count = 0; }
    else if (allowoverlap) {
        while (haystack.length) {
            sz_cptr_t ptr = NULL;
            sz_status_t const status = sz_find_best(haystack.start, haystack.length, needle.start, needle.length, &ptr,
                                                    capabilities, NULL);
            if (status != sz_success_k) {
                sz_py_raise_status(status, "count()");
                return NULL;
            }
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + found;
            haystack.length -= offset + found;
        }
    }
    else {
        while (haystack.length) {
            sz_cptr_t ptr = NULL;
            sz_status_t const status = sz_find_best(haystack.start, haystack.length, needle.start, needle.length, &ptr,
                                                    capabilities, NULL);
            if (status != sz_success_k) {
                sz_py_raise_status(status, "count()");
                return NULL;
            }
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + needle.length;
            haystack.length -= offset + needle.length * found;
        }
    }

    return PyLong_FromSize_t(count);
}

char const doc_startswith[] =                                                                             //
    "Check if a string starts with a given prefix.\n"                                                     //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  prefix (str): The prefix to check.\n"                                                              //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  bool: True if the string starts with the prefix, False otherwise.\n"                               //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').startswith('he')\n"                                                            //
    "  True";

PyObject *Str_like_startswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *str_obj = NULL;
    PyObject *prefix_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *capabilities_object = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // prefix is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (positional_args_count < expected_min || positional_args_count + args_names_count > expected_max + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        str_obj = self;
        if (positional_args_count >= 1) prefix_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
    }
    else {
        if (positional_args_count >= 1) str_obj = args[0];
        if (positional_args_count >= 2) prefix_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (!sz_py_export_optional_index(start_obj, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (!sz_py_export_optional_index(end_obj, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str, prefix;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length) ||
        !sz_py_export_string_like(prefix_obj, &prefix.start, &prefix.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Clamp [start, end) into the string (handles negative / out-of-range / inverted safely)
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(str.length, start, end, &normalized_offset, &normalized_length);
    str.start += normalized_offset;
    str.length = normalized_length;

    if (str.length < prefix.length) { Py_RETURN_FALSE; }
    sz_bool_t equal = sz_false_k; // Binary-safe, NUL-tolerant
    sz_status_t const status = sz_equal_best(str.start, prefix.start, prefix.length, &equal, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "startswith()");
        return NULL;
    }
    return PyBool_FromLong(equal == sz_true_k);
}

char const doc_endswith[] =                                                                               //
    "Check if a string ends with a given suffix.\n"                                                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  suffix (str): The suffix to check.\n"                                                              //
    "  start (int, optional): The starting index, defaulting to 0.\n"                                     //
    "  end (int, optional): The ending index, defaulting to the string length.\n"                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  bool: True if the string ends with the suffix, False otherwise.\n"                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').endswith('lo')\n"                                                              //
    "  True";

PyObject *Str_like_endswith(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *str_obj = NULL;
    PyObject *suffix_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *capabilities_object = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // suffix is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (positional_args_count < expected_min || positional_args_count + args_names_count > expected_max + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
    if (is_member) {
        str_obj = self;
        if (positional_args_count >= 1) suffix_obj = args[0];
        if (positional_args_count >= 2) start_obj = args[1];
        if (positional_args_count >= 3) end_obj = args[2];
    }
    else {
        if (positional_args_count >= 1) str_obj = args[0];
        if (positional_args_count >= 2) suffix_obj = args[1];
        if (positional_args_count >= 3) start_obj = args[2];
        if (positional_args_count >= 4) end_obj = args[3];
    }

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else {
                PyErr_Format(PyExc_TypeError, "unexpected keyword argument: %S", key);
                return NULL;
            }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (!sz_py_export_optional_index(start_obj, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (!sz_py_export_optional_index(end_obj, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str, suffix;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length) ||
        !sz_py_export_string_like(suffix_obj, &suffix.start, &suffix.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Clamp [start, end) into the string (handles negative / out-of-range / inverted safely)
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(str.length, start, end, &normalized_offset, &normalized_length);
    str.start += normalized_offset;
    str.length = normalized_length;

    if (str.length < suffix.length) { Py_RETURN_FALSE; }
    sz_bool_t equal = sz_false_k;
    sz_status_t const status = sz_equal_best(str.start + (str.length - suffix.length), suffix.start, suffix.length,
                                             &equal, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "endswith()");
        return NULL;
    }
    return PyBool_FromLong(equal == sz_true_k);
}

char const doc_find_first_of[] =                                                                          //
    "Find the index of the first occurrence of any character from another string.\n"                      //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str): A string containing characters to search for.\n"                                      //
    "  start (int, optional): Starting index, defaulting to 0.\n"                                         //
    "  end (int, optional): Ending index, defaulting to the string length.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: Index of the first matching character, or -1 if none found.\n"                                //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').find_first_of('aeiou')\n"                                                      //
    "  1";

PyObject *Str_like_find_first_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_py_find_byte_from_,
                                  sz_false_k, sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_find_first_not_of[] =                                                                      //
    "Find the index of the first character not in another string.\n"                                      //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str): A string containing characters to exclude.\n"                                         //
    "  start (int, optional): Starting index, defaulting to 0.\n"                                         //
    "  end (int, optional): Ending index, defaulting to the string length.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: Index of the first non-matching character, or -1 if all match.\n"                             //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').find_first_not_of('he')\n"                                                     //
    "  2";

PyObject *Str_like_find_first_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_py_find_byte_not_from_,
                                  sz_false_k, sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_find_last_of[] =                                                                           //
    "Find the index of the last occurrence of any character from another string.\n"                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str): A string containing characters to search for.\n"                                      //
    "  start (int, optional): Starting index, defaulting to 0.\n"                                         //
    "  end (int, optional): Ending index, defaulting to the string length.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: Index of the last matching character, or -1 if none found.\n"                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').find_last_of('aeiou')\n"                                                       //
    "  4";

PyObject *Str_like_find_last_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_py_rfind_byte_from_,
                                  sz_true_k, sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_find_last_not_of[] =                                                                       //
    "Find the index of the last character not in another string.\n"                                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str): A string containing characters to exclude.\n"                                         //
    "  start (int, optional): Starting index, defaulting to 0.\n"                                         //
    "  end (int, optional): Ending index, defaulting to the string length.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: Index of the last non-matching character, or -1 if all match.\n"                              //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello').find_last_not_of('lo')\n"                                                      //
    "  1";

PyObject *Str_like_find_last_not_of(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!Str_find_implementation_(self, args, positional_args_count, args_names_tuple, &sz_py_rfind_byte_not_from_,
                                  sz_true_k, sz_true_k, &signed_offset, &text, &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

char const doc_count_byteset[] =                                                                          //
    "Count the occurrences of any character from a set of characters.\n"                                  //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str): A string containing characters to count.\n"                                           //
    "  start (int, optional): Starting index, defaulting to 0.\n"                                         //
    "  end (int, optional): Ending index, defaulting to the string length.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  int: The number of occurrences of any character from the set.\n"                                   //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hello world').count_byteset('lo')\n"                                                   //
    "  5";

PyObject *Str_like_count_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    // Fast path variables
    PyObject *haystack_obj = NULL;
    PyObject *needle_obj = NULL;
    PyObject *start_obj = NULL;
    PyObject *end_obj = NULL;
    PyObject *capabilities_object = NULL;

    int const is_member = self != NULL && PyObject_TypeCheck(self, &StrType);

    // Fast argument validation
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_Size(args_names_tuple) : 0;
    Py_ssize_t const total_args = positional_args_count + args_names_count;
    Py_ssize_t const expected_min = is_member ? 1 : 2; // chars is required
    Py_ssize_t const expected_max = expected_min + 2;  // + start + end

    if (total_args < expected_min || total_args > expected_max + 1) {
        PyErr_SetString(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    if (positional_args_count > expected_max) {
        PyErr_SetString(PyExc_TypeError, "Too many positional arguments");
        return NULL;
    }

    // Fast positional argument extraction
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

    // Fast keyword argument parsing
    if (args_names_count > 0) {
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *const key = PyTuple_GetItem(args_names_tuple, i);
            PyObject *const value = args[positional_args_count + i];

            if (PyUnicode_CompareWithASCIIString(key, "start") == 0) {
                if (start_obj) {
                    PyErr_SetString(PyExc_TypeError, "start specified twice");
                    return NULL;
                }
                start_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0) {
                if (end_obj) {
                    PyErr_SetString(PyExc_TypeError, "end specified twice");
                    return NULL;
                }
                end_obj = value;
            }
            else if (!is_member && PyUnicode_CompareWithASCIIString(key, "text") == 0) {
                if (haystack_obj) {
                    PyErr_SetString(PyExc_TypeError, "text specified twice");
                    return NULL;
                }
                haystack_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "chars") == 0) {
                if (needle_obj) {
                    PyErr_SetString(PyExc_TypeError, "chars specified twice");
                    return NULL;
                }
                needle_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else {
                PyErr_SetString(PyExc_TypeError, "Unknown keyword argument");
                return NULL;
            }
        }
    }

    // Validate required arguments
    if (!haystack_obj || !needle_obj) {
        PyErr_SetString(PyExc_TypeError, "Required arguments missing");
        return NULL;
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    // Parse string objects
    sz_string_view_t haystack_view;
    sz_string_view_t needle_view;
    if (!sz_py_export_string_like(haystack_obj, &haystack_view.start, &haystack_view.length) ||
        !sz_py_export_string_like(needle_obj, &needle_view.start, &needle_view.length)) {
        wrap_current_exception("Haystack and needle must be string-like");
        return NULL;
    }

    // Parse slice bounds
    Py_ssize_t start_idx = start_obj ? PyLong_AsSsize_t(start_obj) : 0;
    Py_ssize_t end_idx = end_obj ? PyLong_AsSsize_t(end_obj) : (Py_ssize_t)PY_SSIZE_T_MAX;
    if ((start_idx == -1 || end_idx == -1) && PyErr_Occurred()) return NULL;

    // Normalize slice indices
    if (end_idx == PY_SSIZE_T_MAX) end_idx = (Py_ssize_t)haystack_view.length;
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(haystack_view.length, start_idx, end_idx, &normalized_offset, &normalized_length);
    haystack_view.start += normalized_offset;
    haystack_view.length = normalized_length;

    // Handle empty cases
    if (needle_view.length == 0 || haystack_view.length == 0) return PyLong_FromSsize_t(0);

    sz_byteset_t set;
    sz_byteset_init(&set);
    for (sz_size_t i = 0; i != needle_view.length; ++i) sz_byteset_add(&set, needle_view.start[i]);

    sz_size_t count = 0;
    sz_cptr_t current_pos = haystack_view.start;
    sz_size_t remaining_length = haystack_view.length;

    while (remaining_length > 0) {
        sz_cptr_t found = NULL;
        sz_status_t const status = sz_find_byteset_best(current_pos, remaining_length, &set, &found, capabilities,
                                                        NULL);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "count_byteset()");
            return NULL;
        }
        if (found == NULL) break;

        count++;
        // Move past the found character
        sz_size_t offset = (sz_size_t)(found - current_pos + 1);
        if (offset > remaining_length) break;
        current_pos = found + 1;
        remaining_length -= offset;
    }

    return PyLong_FromSize_t(count);
}

/** Collects every part of @p split into a @c Strs viewing @p parent_string, in source order even
 *  when splitting from the back, as CPython's @c rsplit reports them. */
static Strs *Str_split_(PyObject *parent_string, sz_py_split_t *split) {
    sz_size_t spans_capacity = 4, spans_count = 0;
    sz_string_view_t *spans = (sz_string_view_t *)malloc(spans_capacity * sizeof(sz_string_view_t));
    if (!spans) return (Strs *)PyErr_NoMemory();

    for (;;) {
        sz_string_view_t part;
        int const step = sz_py_split_next_(split, &part);
        if (step < 0) {
            free(spans);
            return NULL;
        }
        if (step == 0) break;
        if (spans_count == spans_capacity) {
            spans_capacity *= 2;
            sz_string_view_t *new_spans = (sz_string_view_t *)realloc(spans, spans_capacity * sizeof(sz_string_view_t));
            if (!new_spans) {
                free(spans);
                return (Strs *)PyErr_NoMemory();
            }
            spans = new_spans;
        }
        spans[spans_count++] = part;
    }

    if (split->is_reverse)
        for (sz_size_t i = 0; i < spans_count / 2; i++) {
            sz_string_view_t const temp = spans[i];
            spans[i] = spans[spans_count - i - 1];
            spans[spans_count - i - 1] = temp;
        }

    Strs *result = strs_make_empty_fragmented_();
    if (!result) {
        free(spans);
        return NULL;
    }
    result->data.fragmented.parent = parent_string;
    Py_INCREF(parent_string);
    result->data.fragmented.spans = spans;
    result->data.fragmented.count = spans_count;
    return result;
}

/** Proxy parsing the function arguments of `Str.split`, `Str.rsplit`, `Str.split_byteset`, and
 *  `Str.rsplit_byteset`, then splitting eagerly with @c Str_split_ or lazily as @c FindSplits. */
static PyObject *Str_split_with_known_callback(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                               PyObject *args_names_tuple,                        //
                                               sz_py_split_finder_t finder, sz_bool_t is_byteset, //
                                               sz_bool_t is_reverse, sz_bool_t is_lazy_iterator) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t expected_min_args = !is_member;
    Py_ssize_t expected_max_args = !is_member + 4;
    if (positional_args_count < expected_min_args || positional_args_count > expected_max_args) {
        PyErr_SetString(PyExc_TypeError, "sz.split() received unsupported number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *separator_obj = positional_args_count > !is_member + 0 ? args[!is_member + 0] : NULL;
    PyObject *maxsplit_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *keepseparator_obj = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;
    PyObject *skip_empty_obj = positional_args_count > !is_member + 3 ? args[!is_member + 3] : NULL;
    PyObject *capabilities_object = NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "separator") == 0 && !separator_obj) { separator_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0 && !maxsplit_obj) { maxsplit_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "keepseparator") == 0 && !keepseparator_obj) {
                keepseparator_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "skip_empty") == 0 && !skip_empty_obj) {
                skip_empty_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text;
    sz_string_view_t separator = {" ", 1};
    int keepseparator;
    int skip_empty = 0;
    Py_ssize_t maxsplit;

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `separator`
    if (separator_obj && !sz_py_export_string_like(separator_obj, &separator.start, &separator.length)) {
        wrap_current_exception("The separator argument must be string-like");
        return NULL;
    }

    // Validate and convert `keepseparator`
    if (keepseparator_obj) {
        keepseparator = PyObject_IsTrue(keepseparator_obj);
        if (keepseparator == -1) {
            PyErr_SetString(PyExc_TypeError, "The keepseparator argument must be a boolean");
            return NULL;
        }
    }
    else { keepseparator = 0; }

    // Validate and convert `maxsplit`
    if (maxsplit_obj) {
        maxsplit = PyLong_AsSsize_t(maxsplit_obj);
        if (maxsplit == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The maxsplit argument must be an integer");
            return NULL;
        }
    }
    else { maxsplit = PY_SSIZE_T_MAX; }

    // Validate and convert `skip_empty`
    if (skip_empty_obj) {
        skip_empty = PyObject_IsTrue(skip_empty_obj);
        if (skip_empty == -1) {
            PyErr_SetString(PyExc_TypeError, "The skip_empty argument must be a boolean");
            return NULL;
        }
    }

    // An empty separator never matches, leaving the whole text as one part.
    if (separator.length == 0) maxsplit = 0;
    sz_py_split_t split = {
        .start = text.start,
        .end = text.start + text.length,
        .finder = finder,
        .separator = separator,
        .capabilities = capabilities,
        .match_length = is_byteset ? 1 : separator.length,
        .max_parts = maxsplit < 0 ? SIZE_MAX : (sz_size_t)maxsplit + 1,
        .include_match = keepseparator ? sz_true_k : sz_false_k,
        .is_reverse = is_reverse,
        .skip_empty = skip_empty ? sz_true_k : sz_false_k,
    };
    if (is_byteset)
        for (sz_size_t i = 0; i != separator.length; ++i) sz_byteset_add(&split.set, separator.start[i]);
    if (!is_lazy_iterator) return (PyObject *)Str_split_(text_obj, &split);

    FindSplits *iterator = (FindSplits *)FindSplitsType.tp_alloc(&FindSplitsType, 0);
    if (!iterator) return NULL;
    iterator->text_obj = text_obj;
    Py_INCREF(text_obj);
    iterator->separator_obj = separator_obj;
    Py_XINCREF(separator_obj);
    iterator->split = split;
    return (PyObject *)iterator;
}

char const doc_split[] =                                                                                  //
    "Split a string by a separator.\n"                                                                    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separator (str): The separator to split by; an empty one leaves the text whole.\n"                 //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include the separator in results, defaulting to False.\n"          //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A list of strings split by the separator.\n"                                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(map(str, sz.Str('a,b,c').split(',')))\n"                                                  //
    "  ['a', 'b', 'c']";

PyObject *Str_like_split(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, sz_py_split_find_,
                                         sz_false_k, sz_false_k, sz_false_k);
}

char const doc_rsplit[] =                                                                                 //
    "Split a string by a separator starting from the end.\n"                                              //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separator (str): The separator to split by; an empty one leaves the text whole.\n"                 //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include the separator in results, defaulting to False.\n"          //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A list of strings split by the separator.\n"                                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(map(str, sz.Str('a,b,c').rsplit(',')))\n"                                                 //
    "  ['a', 'b', 'c']";

PyObject *Str_like_rsplit(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, sz_py_split_rfind_,
                                         sz_false_k, sz_true_k, sz_false_k);
}

char const doc_split_byteset[] =                                                                          //
    "Split a string by a set of character separators.\n"                                                  //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separators (str): A string containing separator characters.\n"                                     //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include separators in results, defaulting to False.\n"             //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A list of strings split by the character set.\n"                                             //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(map(str, sz.Str('a,b;c').split_byteset(',;')))\n"                                         //
    "  ['a', 'b', 'c']";

PyObject *Str_like_split_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, sz_py_split_find_byteset_,
                                         sz_true_k, sz_false_k, sz_false_k);
}

char const doc_rsplit_byteset[] =                                                                         //
    "Split a string by a set of character separators in reverse order.\n"                                 //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separators (str): A string containing separator characters.\n"                                     //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include separators in results, defaulting to False.\n"             //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A list of strings split by the character set.\n"                                             //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(map(str, sz.Str('a,b;c').rsplit_byteset(',;')))\n"                                        //
    "  ['a', 'b', 'c']";

PyObject *Str_like_rsplit_byteset(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple,
                                         sz_py_split_rfind_byteset_, sz_true_k, sz_true_k, sz_false_k);
}

char const doc_split_iter[] =                                                                             //
    "Create an iterator for splitting a string by a separator.\n"                                         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separator (str): The separator to split by; an empty one leaves the text whole.\n"                 //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include separator in results, defaulting to False.\n"              //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding split substrings.\n"                                                //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> # Stream parts lazily instead of materializing a list (that is what split() is for):\n"        //
    "  >>> sum(1 for _ in sz.Str('a,b,c').split_iter(','))\n"                                             //
    "  3";

PyObject *Str_like_split_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, sz_py_split_find_,
                                         sz_false_k, sz_false_k, sz_true_k);
}

char const doc_rsplit_iter[] =                                                                            //
    "Create an iterator for splitting a string by a separator in reverse order.\n"                        //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separator (str): The separator to split by; an empty one leaves the text whole.\n"                 //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include separator in results, defaulting to False.\n"              //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding split substrings in reverse.\n"                                     //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> # Iterates from the end; the first yielded part is the last field:\n"                          //
    "  >>> str(next(iter(sz.Str('a/b/c').rsplit_iter('/'))))\n"                                           //
    "  'c'";

PyObject *Str_like_rsplit_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, sz_py_split_rfind_,
                                         sz_false_k, sz_true_k, sz_true_k);
}

char const doc_split_byteset_iter[] =                                                                     //
    "Create an iterator for splitting a string by a set of character separators.\n"                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separators (str): A string containing separator characters.\n"                                     //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include separators in results, defaulting to False.\n"             //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding split substrings.\n"                                                //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> # Splits on ANY byte in the set, streamed lazily:\n"                                           //
    "  >>> str(next(iter(sz.Str('a,b;c').split_byteset_iter(',;'))))\n"                                   //
    "  'a'";

PyObject *Str_like_split_byteset_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                      PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple, sz_py_split_find_byteset_,
                                         sz_true_k, sz_false_k, sz_true_k);
}

char const doc_rsplit_byteset_iter[] =                                                                    //
    "Create an iterator for splitting a string by a set of character separators in reverse order.\n"      //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  separators (str): A string containing separator characters.\n"                                     //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  keepseparator (bool, optional): Include separators in results, defaulting to False.\n"             //
    "  skip_empty (bool, optional): Skip empty segments, defaulting to False.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  iterator: An iterator yielding split substrings in reverse.\n"                                     //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> # Reverse byteset split; first yielded part is the last field:\n"                              //
    "  >>> str(next(iter(sz.Str('a,b;c').rsplit_byteset_iter(',;'))))\n"                                  //
    "  'c'";

PyObject *Str_like_rsplit_byteset_iter(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                       PyObject *args_names_tuple) {
    return Str_split_with_known_callback(self, args, positional_args_count, args_names_tuple,
                                         sz_py_split_rfind_byteset_, sz_true_k, sz_true_k, sz_true_k);
}

char const doc_splitlines[] =                                                                             //
    "Split a string by line breaks.\n"                                                                    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  keeplinebreaks (bool, optional): Include line breaks in the results, defaulting to False.\n"       //
    "  maxsplit (int, optional): Maximum number of splits, defaulting to no limit.\n"                     //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strs: A list of strings split by line breaks.\n"                                                   //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> list(map(str, sz.Str('a\\nb\\nc').splitlines()))\n"                                            //
    "  ['a', 'b', 'c']";

PyObject *Str_like_splitlines(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member || positional_args_count > !is_member + 2) {
        PyErr_SetString(PyExc_TypeError, "splitlines() requires at least 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *keeplinebreaks_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *maxsplit_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *capabilities_object = NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "keeplinebreaks") == 0 && !keeplinebreaks_obj) {
                keeplinebreaks_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0 && !maxsplit_obj) { maxsplit_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text;
    int keeplinebreaks;
    Py_ssize_t maxsplit = PY_SSIZE_T_MAX; // Default value for maxsplit

    // Validate and convert `text`
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `keeplinebreaks`
    if (keeplinebreaks_obj) {
        keeplinebreaks = PyObject_IsTrue(keeplinebreaks_obj);
        if (keeplinebreaks == -1) {
            wrap_current_exception("The keeplinebreaks argument must be a boolean");
            return NULL;
        }
    }
    else { keeplinebreaks = 0; }

    // Validate and convert `maxsplit`
    if (maxsplit_obj) {
        maxsplit = PyLong_AsSsize_t(maxsplit_obj);
        if (maxsplit == -1 && PyErr_Occurred()) {
            PyErr_SetString(PyExc_TypeError, "The maxsplit argument must be an integer");
            return NULL;
        }
    }

    // The Unicode standard defines a number of characters that conforming applications
    // should recognize as line terminators:
    //
    //      LF:    Line Feed, U+000A                            - 1 byte (\n)
    //      VT:    Vertical Tab, U+000B                         - 1 byte (\v)
    //      FF:    Form Feed, U+000C                            - 1 byte (\f)
    //      CR:    Carriage Return, U+000D                      - 1 byte (\r)
    //      NEL:   Next Line, U+0085                            - 1 byte (\x85)
    //      LS:    Line Separator, U+2028                       - 2 bytes
    //      PS:    Paragraph Separator, U+2029                  - 2 bytes
    //      CR+LF: CR (U+000D) followed by LF (U+000A)          - 2 bytes
    //
    // The Python standard is different, it also includes:
    //
    //     FS:    File Separator, U+001C                       - 1 byte (\x1C)
    //     GS:    Group Separator, U+001D                      - 1 byte (\x1D)
    //     RS:    Record Separator, U+001E                     - 1 byte (\x1E)
    //
    // We avoid all 2-byte sequences and only consider 1-byte delimiters.
    // CPython docs: https://docs.python.org/3/library/stdtypes.html#str.splitlines
    sz_py_split_t split = {
        .start = text.start,
        .end = text.start + text.length,
        .finder = sz_py_split_find_byteset_,
        .capabilities = capabilities,
        .match_length = 1,
        .max_parts = maxsplit < 0 ? SIZE_MAX : (sz_size_t)maxsplit + 1,
        .include_match = keeplinebreaks ? sz_true_k : sz_false_k,
    };
    for (char const *terminator = "\x0A\x0B\x0C\x0D\x85\x1C\x1D\x1E"; *terminator; ++terminator)
        sz_byteset_add(&split.set, *terminator);
    Strs *result = Str_split_(text_obj, &split);

    // Unlike a plain split, CPython `splitlines` yields no trailing empty line after a final terminator,
    // and `[]` for an empty input. Drop that single spurious trailing segment (interior blank lines stay).
    if (result && result->data.fragmented.count > 0) {
        sz_size_t const last = result->data.fragmented.count - 1;
        if ((text.length == 0 || sz_byteset_contains(&split.set, text.start[text.length - 1])) &&
            result->data.fragmented.spans[last].length == 0)
            result->data.fragmented.count = last;
    }
    return (PyObject *)result;
}

char const doc_lstrip[] =                                                                                 //
    "Remove leading characters from a string.\n"                                                          //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str, optional): Characters to remove, defaulting to whitespace.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Str: A new string with leading characters removed.\n"                                              //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('  hi').lstrip() == 'hi'\n"                                                             //
    "  True";

PyObject *Str_like_lstrip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    // Check arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t expected_min_args = !is_member;
    Py_ssize_t expected_max_args = !is_member + 1;
    if (positional_args_count < expected_min_args || positional_args_count > expected_max_args) {
        PyErr_SetString(PyExc_TypeError, "lstrip() takes at most 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *chars_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *capabilities_object = NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "chars") == 0 && !chars_obj) { chars_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text;
    sz_string_view_t chars;

    // Validate and convert text
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Default to whitespace if chars is not provided
    char const *default_chars = " \t\n\r\f\v";
    if (chars_obj) {
        if (!sz_py_export_string_like(chars_obj, &chars.start, &chars.length)) {
            wrap_current_exception("The chars argument must be string-like");
            return NULL;
        }
    }
    else {
        chars.start = default_chars;
        chars.length = 6;
    }

    // Create byteset from chars
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (sz_size_t i = 0; i < chars.length; ++i) sz_byteset_add(&set, chars.start[i]);
    sz_byteset_invert(&set);

    // Find first character not in the set (i.e., not to be stripped)
    sz_cptr_t new_start = NULL;
    sz_status_t const status = sz_find_byteset_best(text.start, text.length, &set, &new_start, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "lstrip()");
        return NULL;
    }
    if (!new_start) {
        // Return empty string
        Str *result = Str_alloc_();
        if (result == NULL) return PyErr_NoMemory();
        result->memory.start = NULL;
        result->memory.length = 0;
        result->parent = NULL;
        return (PyObject *)result;
    }

    // Create a new Str object for the result
    sz_size_t new_length = text.length - (new_start - text.start);
    Str *result = Str_alloc_();
    if (result == NULL) return PyErr_NoMemory();
    result->memory.start = new_start;
    result->memory.length = new_length;
    result->parent = text_obj;
    Py_INCREF(text_obj); // Keep the parent alive
    return (PyObject *)result;
}

char const doc_rstrip[] =                                                                                 //
    "Remove trailing characters from a string.\n"                                                         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str, optional): Characters to remove, defaulting to whitespace.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Str: A new string with trailing characters removed.\n"                                             //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('hi  ').rstrip() == 'hi'\n"                                                             //
    "  True";

PyObject *Str_like_rstrip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    // Check arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t expected_min_args = !is_member;
    Py_ssize_t expected_max_args = !is_member + 1;
    if (positional_args_count < expected_min_args || positional_args_count > expected_max_args) {
        PyErr_SetString(PyExc_TypeError, "rstrip() takes at most 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *chars_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *capabilities_object = NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "chars") == 0 && !chars_obj) { chars_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text;
    sz_string_view_t chars;

    // Validate and convert text
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Default to whitespace if chars is not provided
    char const *default_chars = " \t\n\r\f\v";
    if (chars_obj) {
        if (!sz_py_export_string_like(chars_obj, &chars.start, &chars.length)) {
            wrap_current_exception("The chars argument must be string-like");
            return NULL;
        }
    }
    else {
        chars.start = default_chars;
        chars.length = 6;
    }

    // Create byteset from chars
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (sz_size_t i = 0; i < chars.length; ++i) sz_byteset_add(&set, chars.start[i]);
    sz_byteset_invert(&set);

    // Find last character not in the set (i.e., not to be stripped)
    sz_cptr_t new_end = NULL;
    sz_status_t const status = sz_rfind_byteset_best(text.start, text.length, &set, &new_end, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "rstrip()");
        return NULL;
    }
    if (!new_end) {
        // Return empty string
        Str *result = Str_alloc_();
        if (result == NULL) return PyErr_NoMemory();
        result->memory.start = NULL;
        result->memory.length = 0;
        result->parent = NULL;
        return (PyObject *)result;
    }

    // Create a new Str object for the result
    sz_size_t new_length = new_end - text.start + 1;
    Str *result = Str_alloc_();
    if (result == NULL) return PyErr_NoMemory();
    result->memory.start = text.start;
    result->memory.length = new_length;
    result->parent = text_obj;
    Py_INCREF(text_obj); // Keep the parent alive
    return (PyObject *)result;
}

char const doc_strip[] =                                                                                  //
    "Remove leading and trailing characters from a string.\n"                                             //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (Str or str or bytes): The string object.\n"                                                  //
    "  chars (str, optional): Characters to remove, defaulting to whitespace.\n"                          //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Str: A new string with leading and trailing characters removed.\n"                                 //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.Str('  hi  ').strip() == 'hi'\n"                                                            //
    "  True";

PyObject *Str_like_strip(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    // Check arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t expected_min_args = !is_member;
    Py_ssize_t expected_max_args = !is_member + 1;
    if (positional_args_count < expected_min_args || positional_args_count > expected_max_args) {
        PyErr_SetString(PyExc_TypeError, "strip() takes at most 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *chars_obj = positional_args_count > !is_member ? args[!is_member] : NULL;
    PyObject *capabilities_object = NULL;

    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "chars") == 0 && !chars_obj) { chars_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text;
    sz_string_view_t chars;

    // Validate and convert text
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Default to whitespace if chars is not provided
    char const *default_chars = " \t\n\r\f\v";
    if (chars_obj) {
        if (!sz_py_export_string_like(chars_obj, &chars.start, &chars.length)) {
            wrap_current_exception("The chars argument must be string-like");
            return NULL;
        }
    }
    else {
        chars.start = default_chars;
        chars.length = 6;
    }

    // Create byteset from chars
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (sz_size_t i = 0; i < chars.length; ++i) sz_byteset_add(&set, chars.start[i]);
    sz_byteset_invert(&set);

    // Find first character not in the set (i.e., not to be stripped)
    sz_cptr_t new_start = NULL;
    sz_status_t status = sz_find_byteset_best(text.start, text.length, &set, &new_start, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "strip()");
        return NULL;
    }
    if (!new_start) {
        // Return empty string
        Str *result = Str_alloc_();
        if (result == NULL) return PyErr_NoMemory();
        result->memory.start = NULL;
        result->memory.length = 0;
        result->parent = NULL;
        return (PyObject *)result;
    }

    // Find last character not in the set from the new start position
    sz_size_t remaining_length = text.length - (new_start - text.start);
    sz_cptr_t new_end = NULL;
    status = sz_rfind_byteset_best(new_start, remaining_length, &set, &new_end, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "strip()");
        return NULL;
    }
    if (!new_end) {
        // Return empty string
        Str *result = Str_alloc_();
        if (result == NULL) return PyErr_NoMemory();
        result->memory.start = NULL;
        result->memory.length = 0;
        result->parent = NULL;
        return (PyObject *)result;
    }

    // Create a new Str object for the result
    sz_size_t new_length = new_end - new_start + 1;
    Str *result = Str_alloc_();
    if (result == NULL) return PyErr_NoMemory();
    result->memory.start = new_start;
    result->memory.length = new_length;
    result->parent = text_obj;
    Py_INCREF(text_obj); // Keep the parent alive
    return (PyObject *)result;
}

static PyObject *FindSplitsType_next(FindSplits *self) {
    sz_string_view_t result_memory;
    if (sz_py_split_next_(&self->split, &result_memory) <= 0) return NULL;

    // Create a new `Str` object
    Str *result_obj = Str_alloc_();
    if (result_obj == NULL) return PyErr_NoMemory();

    // Set its properties based on the slice
    result_obj->memory = result_memory;
    result_obj->parent = self->text_obj;

    // Increment the reference count of the parent
    Py_INCREF(self->text_obj);
    return (PyObject *)result_obj;
}

static void FindSplitsType_dealloc(FindSplits *self) {
    Py_XDECREF(self->text_obj);
    Py_XDECREF(self->separator_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *FindSplitsType_iter(PyObject *self) {
    Py_INCREF(self); // Iterator should return itself in __iter__.
    return self;
}

static char const doc_FindSplits[] =                                                      //
    "FindSplits(string, separator, ...)\n"                                                //
    "\n"                                                                                  //
    "Text-splitting iterator for efficient string processing.\n"                          //
    "Provides lazy evaluation of string splits without materializing all results.\n"      //
    "\n"                                                                                  //
    "Created by:\n"                                                                       //
    "  - Str.split_iter()\n"                                                              //
    "  - Str.rsplit_iter()\n"                                                             //
    "  - Str.split_byteset_iter()\n"                                                      //
    "  - Str.rsplit_byteset_iter()\n"                                                     //
    "\n"                                                                                  //
    "Features:\n"                                                                         //
    "  - Memory-efficient: yields results one at a time\n"                                //
    "  - Forward and reverse iteration support\n"                                         //
    "  - Character set and string separator support\n"                                    //
    "\n"                                                                                  //
    "Example:\n"                                                                          //
    "  >>> # split_iter streams parts lazily - prefer it over split() on large inputs:\n" //
    "  >>> for field in sz.Str('2024-01-15').split_iter('-'):\n"                          //
    "  ...     print(field)\n"                                                            //
    "  2024\n"                                                                            //
    "  01\n"                                                                              //
    "  15";

PyTypeObject FindSplitsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.FindSplits",
    .tp_basicsize = sizeof(FindSplits),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)FindSplitsType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = doc_FindSplits,
    .tp_iter = FindSplitsType_iter,
    .tp_iternext = (iternextfunc)FindSplitsType_next,
};
