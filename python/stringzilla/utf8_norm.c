/**
 *  @file python/stringzilla/utf8_norm.c
 *  @author Ash Vardanian
 *  @date June 15, 2026
 *  @brief Unicode normalization of UTF-8 text — NFC, NFD, NFKC, NFKD and NFKC_Casefold.
 */
#include "stringzilla.h"

int sz_py_export_normal_form(PyObject *form_object, char const *context, sz_normal_form_t *form) {
    static struct {
        char const *name;
        sz_normal_form_t form;
    } const forms[] = {
        {"NFC", sz_normal_form_nfc_k},
        {"NFD", sz_normal_form_nfd_k},
        {"NFKC", sz_normal_form_nfkc_k},
        {"NFKD", sz_normal_form_nfkd_k},
        {"NFKC_Casefold", sz_normal_form_nfkc_casefold_k},
    };
    if (!PyUnicode_Check(form_object)) {
        PyErr_Format(PyExc_TypeError, "%s argument 'form' must be a string", context);
        return -1;
    }
    for (sz_size_t index = 0; index != sizeof(forms) / sizeof(forms[0]); ++index)
        if (PyUnicode_CompareWithASCIIString(form_object, forms[index].name) == 0) {
            *form = forms[index].form;
            return 0;
        }
    PyErr_Format(PyExc_ValueError, "%s unknown form '%U': expected 'NFC', 'NFD', 'NFKC', 'NFKD', or 'NFKC_Casefold'",
                 context, form_object);
    return -1;
}

char const doc_utf8_norm[] =                                                                             //
    "Normalize a UTF-8 string to a Unicode normalization form.\n"                                        //
    "\n"                                                                                                 //
    "Transforms the input into NFD, NFC, NFKD, or NFKC, mirroring `unicodedata.normalize` but\n"         //
    "returning raw UTF-8 bytes, or into NFKC_Casefold, the Unicode toNFKC_Casefold: NFKC with full\n"    //
    "case folding and no default-ignorable codepoints.\n"                                                //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "    text (String or str or bytes): The input UTF-8 string.\n"                                       //
    "    form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD', 'NFKC_Casefold'.\n"                            //
    "    validate (bool): If True, validate UTF-8 before processing. Default: False.\n"                  //
    "    capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.\n" //
    "\n"                                                                                                 //
    "Returns:\n"                                                                                         //
    "    bytes: The normalized UTF-8 string.\n"                                                          //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "    >>> sz.utf8_norm('caf\\u00e9', 'NFD')  # decompose\n"                                           //
    "    b'cafe\\xcc\\x81'\n"                                                                            //
    "    >>> sz.utf8_norm('\\ufb01', 'NFKD')  # fi ligature\n"                                           //
    "    b'fi'\n"                                                                                        //
    "    >>> sz.utf8_norm('Stra\\u00dfe\\u00ad', 'NFKC_Casefold')  # folded, soft hyphen dropped\n"      //
    "    b'strasse'";

PyObject *String_like_utf8_norm(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StringType);
    Py_ssize_t nargs_expected = !is_member + 1; // +1 for the `form` argument
    int validate = 0;                           // Default: no validation
    PyObject *form_obj = NULL, *capabilities_object = NULL;

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
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                capabilities_object = args[positional_args_count + i];
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
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_normal_form_t form;
    if (sz_py_export_normal_form(form_obj, "utf8_norm()", &form) != 0) return NULL;

    PyObject *str_obj = is_member ? self : args[0];

    sz_string_view_t str;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }

    // Validate UTF-8 input only if requested
    if (validate && sz_utf8_find_malformed(str.start, str.length) != STRINGZILLA_NULL_CHAR) {
        PyErr_SetString(PyExc_ValueError, "Input is not valid UTF-8");
        return NULL;
    }

    // Worst-case: each byte can expand by 18x (compatibility decomposition of single codepoints)
    sz_size_t max_result_length = str.length * 18;
    if (max_result_length == 0) { return PyBytes_FromStringAndSize("", 0); }

    PyObject *result_bytes = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)max_result_length);
    if (!result_bytes) return NULL;

    sz_ptr_t destination = (sz_ptr_t)PyBytes_AS_STRING(result_bytes);
    sz_size_t actual_length = 0;
    sz_status_t const status = sz_utf8_norm_best(str.start, str.length, form, destination, &actual_length, capabilities,
                                                 NULL);
    if (status != sz_success_k) {
        Py_DECREF(result_bytes);
        sz_py_raise_status(status, "utf8_norm()");
        return NULL;
    }

    // Resize down to the actual output length
    if (actual_length < max_result_length) {
        if (_PyBytes_Resize(&result_bytes, (Py_ssize_t)actual_length) < 0) {
            Py_XDECREF(result_bytes);
            return NULL;
        }
    }

    return result_bytes;
}

char const doc_utf8_find_denormalized[] =                                                                //
    "Return the byte offset of the first normalization violation, or None.\n"                            //
    "\n"                                                                                                 //
    "Scans the UTF-8 string and returns the byte offset of the first codepoint\n"                        //
    "that breaks the given normalization form (first non-Yes Quick-Check result\n"                       //
    "or canonical-ordering violation). Returns None when the string is already\n"                        //
    "fully normalized.\n"                                                                                //
    "\n"                                                                                                 //
    "Args:\n"                                                                                            //
    "    text (String or str or bytes): The input UTF-8 string.\n"                                       //
    "    form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD', 'NFKC_Casefold'.\n"                            //
    "    capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.\n" //
    "\n"                                                                                                 //
    "Returns:\n"                                                                                         //
    "    int | None: Byte offset of first violation, or None if already normalized.\n"                   //
    "\n"                                                                                                 //
    "Example:\n"                                                                                         //
    "    >>> sz.utf8_find_denormalized('cafe\\u0301', 'NFC')  # 'e' + combining acute\n"                 //
    "    3\n"                                                                                            //
    "    >>> sz.utf8_find_denormalized('caf\\u00e9', 'NFC') is None  # precomposed, already NFC\n"       //
    "    True";

PyObject *String_like_utf8_find_denormalized(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                             PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StringType);
    Py_ssize_t nargs_expected = !is_member + 1; // +1 for `form`
    PyObject *form_obj = NULL, *capabilities_object = NULL;

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
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) {
                capabilities_object = args[positional_args_count + i];
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
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_normal_form_t form;
    if (sz_py_export_normal_form(form_obj, "utf8_find_denormalized()", &form) != 0) return NULL;

    PyObject *str_obj = is_member ? self : args[0];

    sz_string_view_t str;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("Argument must be string-like");
        return NULL;
    }

    sz_cptr_t violation = STRINGZILLA_NULL_CHAR;
    sz_status_t const status = sz_utf8_find_denormalized_best(str.start, str.length, form, &violation, capabilities,
                                                              NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "utf8_find_denormalized()");
        return NULL;
    }
    if (violation == STRINGZILLA_NULL_CHAR) { Py_RETURN_NONE; }

    sz_size_t offset = (sz_size_t)(violation - str.start);
    return PyLong_FromSize_t(offset);
}
