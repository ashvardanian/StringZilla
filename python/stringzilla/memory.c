/**
 *  @brief Byte-level memory kernels — copies, moves, fills, and lookup transforms.
 *  @file python/stringzilla/memory.c
 *  @author Ash Vardanian
 *
 *  Of those kernels, only the lookup transform is exposed to Python.
 */
#include "stringzilla.h"

char const doc_translate[] =                                                                     //
    "Perform transformation of a string using a look-up table.\n"                                //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  text (Str or str or bytes): The string object.\n"                                         //
    "  table (str or dict): A 256-character string or a dictionary mapping bytes to bytes.\n"    //
    "  inplace (bool, optional): If True, the string is modified in place (default is False).\n" //
    "\n"                                                                                         //
    "  start (int, optional): The starting index for translation (default is 0).\n"              //
    "  end (int, optional): The ending index for translation (default is the string length).\n"  //
    "  out (writable buffer, keyword-only): Storage for at least the selected byte count.\n"     //
    "Returns:\n"                                                                                 //
    "  The supplied out object, None for inplace=True, or a new str/bytes object.\n"             //
    "Raises:\n"                                                                                  //
    "  ValueError: For an invalid size, partial overlap, or out with inplace=True.\n"            //
    "  TypeError: If the table is invalid or out does not have byte-sized elements.\n"           //
    "  BufferError: If out is read-only or not C-contiguous.\n"                                  //
    "\n"                                                                                         //
    "Example:\n"                                                                                 //
    "  >>> sz.Str('abc').translate({'a': 'A'}) == b'Abc'\n"                                      //
    "  True";

static PyObject *Str_translate_into_(PyObject *str_obj, PyObject *out_obj, char const *look_up_table, Py_ssize_t start,
                                     Py_ssize_t end) {
    PyObject *result = NULL;
    Py_buffer out_view;
    if (PyObject_GetBuffer(out_obj, &out_view, PyBUF_CONTIG) != 0) return NULL;
    if (out_view.itemsize != 1) {
        PyErr_SetString(PyExc_TypeError, "out must be a contiguous buffer of bytes");
        goto cleanup;
    }

    // Exporters may resize backing storage or release a memoryview, including a Str's parent.
    PyObject *owner = str_obj;
    while (PyObject_TypeCheck(owner, &StrType) && ((Str *)owner)->parent) owner = ((Str *)owner)->parent;
    if (PyMemoryView_Check(owner)) {
        Py_buffer source_view;
        if (PyObject_GetBuffer(owner, &source_view, PyBUF_FULL_RO) != 0) goto cleanup;
        PyBuffer_Release(&source_view);
    }
    sz_string_view_t str, backing;
    if (!sz_py_export_string_like(str_obj, &str.start, &str.length)) goto cleanup;
    if (owner != str_obj) {
        if (!sz_py_export_string_like(owner, &backing.start, &backing.length)) goto cleanup;
        uintptr_t offset = (uintptr_t)str.start - (uintptr_t)backing.start;
        if (offset > backing.length || str.length > backing.length - offset) {
            PyErr_SetString(PyExc_ValueError, "input view no longer matches its backing buffer");
            goto cleanup;
        }
    }
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(str.length, start, end, &normalized_offset, &normalized_length);
    if (normalized_offset) str.start += normalized_offset;
    str.length = normalized_length;
    if ((sz_size_t)out_view.len < str.length) {
        PyErr_Format(PyExc_ValueError, "out buffer holds %zd bytes, need %zu", out_view.len, str.length);
        goto cleanup;
    }
    uintptr_t source_address = (uintptr_t)str.start;
    uintptr_t output_address = (uintptr_t)out_view.buf;
    if (source_address != output_address &&
        ((source_address < output_address && output_address - source_address < str.length) ||
         (output_address < source_address && source_address - output_address < str.length))) {
        PyErr_SetString(PyExc_ValueError, "out must not partially overlap the input");
        goto cleanup;
    }
    sz_lookup((sz_ptr_t)out_view.buf, str.length, str.start, look_up_table);
    Py_INCREF(out_obj);
    result = out_obj;
cleanup:
    PyBuffer_Release(&out_view);
    return result;
}

PyObject *Str_like_translate(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    if (positional_args_count < !is_member + 1 || positional_args_count > !is_member + 4) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : args[0];
    PyObject *look_up_table_obj = args[!is_member];
    PyObject *inplace_obj = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *start_obj = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;
    PyObject *end_obj = positional_args_count > !is_member + 3 ? args[!is_member + 3] : NULL;
    PyObject *out_obj = NULL;

    // Optional keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "inplace") == 0 && !inplace_obj) { inplace_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "start") == 0 && !start_obj) { start_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0 && !end_obj) { end_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "out") == 0 && !out_obj) { out_obj = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return NULL;
            }
        }
    }

    // Optional start and end arguments
    Py_ssize_t start, end;
    if (!sz_py_export_optional_index(start_obj, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }
    if (!sz_py_export_optional_index(end_obj, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    int is_inplace = inplace_obj ? PyObject_IsTrue(inplace_obj) : 0;
    if (is_inplace == -1) {
        PyErr_SetString(PyExc_TypeError, "The inplace argument must be a boolean");
        return NULL;
    }

    int const have_out = out_obj && out_obj != Py_None;
    sz_string_view_t str;
    if (!have_out && !sz_py_export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }

    sz_string_view_t look_up_table_str;
    sz_align_(64) char look_up_table[256];
    if (PyDict_Check(look_up_table_obj)) {

        // If any character is not defined, it will be replaced with itself:
        for (int i = 0; i < 256; i++) look_up_table[i] = (char)i;

        // Process the dictionary into the look-up table
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(look_up_table_obj, &pos, &key, &value)) {
            if (!PyUnicode_Check(key) || PyUnicode_GetLength(key) != 1 || !PyUnicode_Check(value) ||
                PyUnicode_GetLength(value) != 1) {
                PyErr_SetString(PyExc_TypeError, "Keys and values must be single characters");
                return NULL;
            }

            char key_char = PyUnicode_AsUTF8(key)[0];
            char value_char = PyUnicode_AsUTF8(value)[0];
            look_up_table[(unsigned char)key_char] = value_char;
        }
    }
    else if (sz_py_export_string_like(look_up_table_obj, &look_up_table_str.start, &look_up_table_str.length)) {
        if (look_up_table_str.length != 256) {
            PyErr_SetString(PyExc_ValueError, "The look-up table must be exactly 256 bytes long");
            return NULL;
        }
        sz_copy(&look_up_table[0], look_up_table_str.start, look_up_table_str.length);
    }
    else {
        wrap_current_exception("The look-up table must be string-like or a dictionary");
        return NULL;
    }

    if (have_out) {
        if (is_inplace) {
            PyErr_SetString(PyExc_ValueError, "out and inplace=True cannot be combined");
            return NULL;
        }
        return Str_translate_into_(str_obj, out_obj, look_up_table, start, end);
    }

    // Clamp [start, end) into the string (handles negative / out-of-range / inverted safely)
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(str.length, start, end, &normalized_offset, &normalized_length);
    str.start += normalized_offset;
    str.length = normalized_length;

    // Perform the translation using the look-up table
    if (is_inplace) {
        if (sz_py_is_mutable(str_obj) == sz_false_k) return NULL;
        sz_lookup(str.start, str.length, str.start, look_up_table);
        Py_RETURN_NONE;
    }
    // Allocate a string of the same size, get it's raw pointer and transform the data into it
    else {

        // For binary inputs return bytes, for unicode return str
        if (PyUnicode_Check(str_obj)) {
            // Create a new Unicode object
            PyObject *new_unicode_obj = PyUnicode_New(str.length, PyUnicode_MAX_CHAR_VALUE(str_obj));
            if (!new_unicode_obj) {
                PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for new Unicode string");
                return NULL;
            }

            sz_ptr_t new_buffer = (sz_ptr_t)PyUnicode_DATA(new_unicode_obj);
            sz_lookup(new_buffer, str.length, str.start, look_up_table);
            return new_unicode_obj;
        }
        else {
            PyObject *new_bytes_obj = PyBytes_FromStringAndSize(NULL, str.length);
            if (!new_bytes_obj) {
                PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for new string");
                return NULL;
            }

            // Get the buffer and perform the transformation
            sz_ptr_t new_buffer = (sz_ptr_t)PyBytes_AS_STRING(new_bytes_obj);
            sz_lookup(new_buffer, str.length, str.start, look_up_table);
            return new_bytes_obj;
        }
    }
}
