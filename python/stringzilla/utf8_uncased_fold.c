/**
 *  @brief Unicode case folding of UTF-8 text.
 *  @file python/stringzilla/utf8_uncased_fold.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_utf8_uncased_fold[] =                                                    //
    "Apply Unicode case folding to a UTF-8 string.\n"                                   //
    "\n"                                                                                //
    "Case folding normalizes text for uncased comparisons,\n"                           //
    "handling one-to-many expansions (e.g., German sharp S to 'ss').\n"                 //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "    text (Str or str or bytes): The input UTF-8 string.\n"                         //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n" //
    "\n"                                                                                //
    "Returns:\n"                                                                        //
    "    bytes: The case-folded UTF-8 string.\n"                                        //
    "\n"                                                                                //
    "Example:\n"                                                                        //
    "    >>> sz.utf8_uncased_fold('HELLO')\n"                                           //
    "    b'hello'\n"                                                                    //
    "    >>> sz.utf8_uncased_fold('Stra\\u00dfe')  # German sharp S\n"                  //
    "    b'strasse'";

PyObject *Str_like_utf8_uncased_fold(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                     PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs_expected = !is_member; // 0 if method, 1 if module function
    int validate = 0;                       // Default: no validation

    if (positional_args_count != nargs_expected) {
        PyErr_Format(PyExc_TypeError, "utf8_uncased_fold() takes exactly %zd positional argument(s)", nargs_expected);
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
                PyErr_Format(PyExc_TypeError, "utf8_uncased_fold() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    PyObject *str_obj = is_member ? self : args[0];

    sz_string_view_t str;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }

    // Validate UTF-8 input only if requested
    if (validate && sz_utf8_find_malformed(str.start, str.length) != SZ_NULL_CHAR) {
        PyErr_SetString(PyExc_ValueError, "Input is not valid UTF-8");
        return NULL;
    }

    // Allocate buffer with 3x capacity for maximum expansion (e.g., some Greek characters)
    sz_size_t max_result_length = str.length * 3;
    if (max_result_length == 0) { return PyBytes_FromStringAndSize("", 0); }

    PyObject *result_bytes = PyBytes_FromStringAndSize(NULL, max_result_length);
    if (!result_bytes) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for case-folded string");
        return NULL;
    }

    sz_ptr_t destination = (sz_ptr_t)PyBytes_AS_STRING(result_bytes);
    sz_size_t actual_length = sz_utf8_uncased_fold(str.start, str.length, destination);

    // Resize to actual length if smaller than allocated
    if (actual_length < max_result_length) {
        if (_PyBytes_Resize(&result_bytes, actual_length) < 0) {
            Py_XDECREF(result_bytes);
            return NULL;
        }
    }

    return result_bytes;
}
