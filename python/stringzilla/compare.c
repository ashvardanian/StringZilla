/**
 *  @file python/stringzilla/compare.c
 *  @author Ash Vardanian
 *  @date September 16, 2023
 *  @brief Byte-level equality and ordering.
 */
#include "stringzilla.h"

char const doc_like_equal[] =                                                                             //
    "Check if two strings are equal.\n"                                                                   //
    "\n"                                                                                                  //
    "This function can be called as a method on a Str object or as a standalone function.\n"              //
    "Args:\n"                                                                                             //
    "  first (Str or str or bytes): The first string object.\n"                                           //
    "  second (Str or str or bytes): The second string object.\n"                                         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  bool: True if the strings are equal, False otherwise.\n"                                           //
    "Note:\n"                                                                                             //
    "  Comparison is byte-level: a `str` and a `bytes` with identical bytes compare equal.\n"             //
    "Raises:\n"                                                                                           //
    "  TypeError: If the argument is not string-like or incorrect number of arguments is provided.\n"     //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.equal('abc', 'abc')\n"                                                                      //
    "  True";

PyObject *Str_like_equal(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member + 1 || positional_args_count > !is_member + 1) {
        PyErr_SetString(PyExc_TypeError, "equal() expects exactly two positional arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : args[0];
    PyObject *other_obj = args[!is_member]; // Second operand: args[0] as a method, args[1] as a function
    PyObject *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
        if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object)
            capabilities_object = args[positional_args_count + i];
        else {
            PyErr_Format(PyExc_TypeError, "equal() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text, other;

    // Validate and convert the texts
    if (!sz_py_export_string_like(text_obj, &text.start, &text.length) || //
        !sz_py_export_string_like(other_obj, &other.start, &other.length)) {
        wrap_current_exception("The arguments must be string-like");
        return NULL;
    }

    if (text.length != other.length) { Py_RETURN_FALSE; }
    sz_bool_t result = sz_false_k;
    sz_status_t const status = sz_equal_best(text.start, other.start, text.length, &result, capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "equal()");
        return NULL;
    }
    if (result != sz_true_k) { Py_RETURN_FALSE; }
    Py_RETURN_TRUE;
}

PyObject *Str_richcompare(PyObject *self, PyObject *other, int op) {

    sz_cptr_t a_start = NULL, b_start = NULL;
    sz_size_t a_length = 0, b_length = 0;
    if (!sz_py_export_string_like(self, &a_start, &a_length) || !sz_py_export_string_like(other, &b_start, &b_length))
        Py_RETURN_NOTIMPLEMENTED;

    sz_ordering_t ordering = sz_equal_k;
    sz_status_t const status = sz_order_best(a_start, a_length, b_start, b_length, &ordering, sz_cap_cpus_k, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "Str comparison");
        return NULL;
    }
    int const order = (int)ordering;
    switch (op) {
    case Py_LT: return PyBool_FromLong(order < 0);
    case Py_LE: return PyBool_FromLong(order <= 0);
    case Py_EQ: return PyBool_FromLong(order == 0);
    case Py_NE: return PyBool_FromLong(order != 0);
    case Py_GT: return PyBool_FromLong(order > 0);
    case Py_GE: return PyBool_FromLong(order >= 0);
    default: Py_RETURN_NOTIMPLEMENTED;
    }
}
