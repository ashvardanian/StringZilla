/**
 *  @brief Unicode normalization of UTF-8 text — NFC, NFD, NFKC, and NFKD.
 *  @file python/stringzilla/utf8_norm.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_utf8_norm[] =                                                            //
    "Normalize a UTF-8 string to a Unicode normalization form.\n"                       //
    "\n"                                                                                //
    "Transforms the input into NFD, NFC, NFKD, or NFKC, mirroring\n"                    //
    "`unicodedata.normalize` but returning raw UTF-8 bytes.\n"                          //
    "\n"                                                                                //
    "Args:\n"                                                                           //
    "    text (Str or str or bytes): The input UTF-8 string.\n"                         //
    "    form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD'.\n"                            //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n" //
    "\n"                                                                                //
    "Returns:\n"                                                                        //
    "    bytes: The normalized UTF-8 string.\n"                                         //
    "\n"                                                                                //
    "Example:\n"                                                                        //
    "    >>> sz.utf8_norm('caf\\u00e9', 'NFD')  # decompose\n"                          //
    "    b'cafe\\xcc\\x81'\n"                                                           //
    "    >>> sz.utf8_norm('\\ufb01', 'NFKD')  # fi ligature\n"                          //
    "    b'fi'";

PyObject *Str_like_utf8_norm(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs_expected = !is_member + 1; // +1 for the `form` argument
    int validate = 0;                           // Default: no validation
    PyObject *form_obj = NULL;

    if (positional_args_count < nargs_expected - 1 || positional_args_count > nargs_expected) {
        PyErr_Format(PyExc_TypeError, "utf8_norm() takes %zd positional argument(s) (%zd given)", nargs_expected,
                     positional_args_count);
        return NULL;
    }

    // The `form` argument is always the last positional: args[nargs_expected - 1]
    // But it may also be supplied as a keyword if the user passes only `text` positionally.
    // We accept it either way.
    if (positional_args_count == nargs_expected) { form_obj = args[nargs_expected - 1]; }

    // Parse keyword arguments: 'form' and 'validate'
    if (args_names_tuple) {
        Py_ssize_t nkwargs = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < nkwargs; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            if (PyUnicode_CompareWithASCIIString(key, "form") == 0) {
                if (form_obj != NULL) {
                    PyErr_SetString(PyExc_TypeError, "utf8_norm() got multiple values for argument 'form'");
                    return NULL;
                }
                form_obj = args[positional_args_count + i];
            }
            else if (PyUnicode_CompareWithASCIIString(key, "validate") == 0) {
                PyObject *val = args[positional_args_count + i];
                validate = PyObject_IsTrue(val);
                if (validate < 0) return NULL;
            }
            else {
                PyErr_Format(PyExc_TypeError, "utf8_norm() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    if (form_obj == NULL) {
        PyErr_SetString(PyExc_TypeError, "utf8_norm() missing required argument: 'form'");
        return NULL;
    }

    // Parse the `form` string into sz_normal_form_t
    sz_normal_form_t form;
    Py_ssize_t form_len = 0;
    char const *form_str = PyUnicode_AsUTF8AndSize(form_obj, &form_len);
    if (!form_str) {
        PyErr_SetString(PyExc_TypeError, "utf8_norm() argument 'form' must be a string");
        return NULL;
    }
    if (strncmp(form_str, "NFD", (sz_size_t)form_len) == 0 && form_len == 3) { form = sz_normal_form_nfd_k; }
    else if (strncmp(form_str, "NFC", (sz_size_t)form_len) == 0 && form_len == 3) { form = sz_normal_form_nfc_k; }
    else if (strncmp(form_str, "NFKD", (sz_size_t)form_len) == 0 && form_len == 4) { form = sz_normal_form_nfkd_k; }
    else if (strncmp(form_str, "NFKC", (sz_size_t)form_len) == 0 && form_len == 4) { form = sz_normal_form_nfkc_k; }
    else {
        PyErr_Format(PyExc_ValueError, "utf8_norm() unknown form '%s': expected 'NFC', 'NFD', 'NFKC', or 'NFKD'",
                     form_str);
        return NULL;
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

    // Worst-case: each byte can expand by 18x (compatibility decomposition of single codepoints)
    sz_size_t max_result_length = str.length * 18;
    if (max_result_length == 0) { return PyBytes_FromStringAndSize("", 0); }

    PyObject *result_bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)max_result_length);
    if (!result_bytes) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for normalized string");
        return NULL;
    }

    sz_ptr_t destination = (sz_ptr_t)PyBytes_AS_STRING(result_bytes);
    sz_size_t actual_length = sz_utf8_norm(str.start, str.length, form, destination);

    // Resize down to the actual output length
    if (actual_length < max_result_length) {
        if (_PyBytes_Resize(&result_bytes, (Py_ssize_t)actual_length) < 0) {
            Py_XDECREF(result_bytes);
            return NULL;
        }
    }

    return result_bytes;
}

char const doc_utf8_find_denormalized[] =                                                          //
    "Return the byte offset of the first normalization violation, or None.\n"                      //
    "\n"                                                                                           //
    "Scans the UTF-8 string and returns the byte offset of the first codepoint\n"                  //
    "that breaks the given normalization form (first non-Yes Quick-Check result\n"                 //
    "or canonical-ordering violation). Returns None when the string is already\n"                  //
    "fully normalized.\n"                                                                          //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "    text (Str or str or bytes): The input UTF-8 string.\n"                                    //
    "    form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD'.\n"                                       //
    "\n"                                                                                           //
    "Returns:\n"                                                                                   //
    "    int | None: Byte offset of first violation, or None if already normalized.\n"             //
    "\n"                                                                                           //
    "Example:\n"                                                                                   //
    "    >>> sz.utf8_find_denormalized('cafe\\u0301', 'NFC')  # 'e' + combining acute\n"           //
    "    3\n"                                                                                      //
    "    >>> sz.utf8_find_denormalized('caf\\u00e9', 'NFC') is None  # precomposed, already NFC\n" //
    "    True";

PyObject *Str_like_utf8_find_denormalized(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                          PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs_expected = !is_member + 1; // +1 for `form`
    PyObject *form_obj = NULL;

    if (positional_args_count < nargs_expected - 1 || positional_args_count > nargs_expected) {
        PyErr_Format(PyExc_TypeError, "utf8_find_denormalized() takes %zd positional argument(s) (%zd given)",
                     nargs_expected, positional_args_count);
        return NULL;
    }

    if (positional_args_count == nargs_expected) { form_obj = args[nargs_expected - 1]; }

    // Parse keyword arguments
    if (args_names_tuple) {
        Py_ssize_t nkwargs = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < nkwargs; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            if (PyUnicode_CompareWithASCIIString(key, "form") == 0) {
                if (form_obj != NULL) {
                    PyErr_SetString(PyExc_TypeError,
                                    "utf8_find_denormalized() got multiple values for argument 'form'");
                    return NULL;
                }
                form_obj = args[positional_args_count + i];
            }
            else {
                PyErr_Format(PyExc_TypeError, "utf8_find_denormalized() got unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    if (form_obj == NULL) {
        PyErr_SetString(PyExc_TypeError, "utf8_find_denormalized() missing required argument: 'form'");
        return NULL;
    }

    // Parse the `form` string into sz_normal_form_t
    sz_normal_form_t form;
    Py_ssize_t form_len = 0;
    char const *form_str = PyUnicode_AsUTF8AndSize(form_obj, &form_len);
    if (!form_str) {
        PyErr_SetString(PyExc_TypeError, "utf8_find_denormalized() argument 'form' must be a string");
        return NULL;
    }
    if (strncmp(form_str, "NFD", (sz_size_t)form_len) == 0 && form_len == 3) { form = sz_normal_form_nfd_k; }
    else if (strncmp(form_str, "NFC", (sz_size_t)form_len) == 0 && form_len == 3) { form = sz_normal_form_nfc_k; }
    else if (strncmp(form_str, "NFKD", (sz_size_t)form_len) == 0 && form_len == 4) { form = sz_normal_form_nfkd_k; }
    else if (strncmp(form_str, "NFKC", (sz_size_t)form_len) == 0 && form_len == 4) { form = sz_normal_form_nfkc_k; }
    else {
        PyErr_Format(PyExc_ValueError,
                     "utf8_find_denormalized() unknown form '%s': expected 'NFC', 'NFD', 'NFKC', or 'NFKD'", form_str);
        return NULL;
    }

    PyObject *str_obj = is_member ? self : args[0];

    sz_string_view_t str;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }

    sz_cptr_t violation = sz_utf8_find_denormalized(str.start, str.length, form);
    if (violation == SZ_NULL_CHAR) { Py_RETURN_NONE; }

    sz_size_t offset = (sz_size_t)(violation - str.start);
    return PyLong_FromSize_t(offset);
}
