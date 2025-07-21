/**
 *  @file       stringzillas.c
 *  @brief      Very light-weight CPython wrapper for StringZilla, with support for memory-mapping,
 *              native Python strings, Apache Arrow collections, and more.
 *  @author     Ash Vardanian
 *  @date       July 10, 2023
 *  @copyright  Copyright (c) 2023
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids `PyArg_ParseTupleAndKeywords` calls.
 */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>    // `O_RDNLY`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `stat`
#include <sys/types.h>
#endif

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <limits.h> // `SSIZE_MAX`
#include <unistd.h> // `ssize_t`
#endif

// It seems like some Python versions forget to include a header, so we should:
// https://github.com/ashvardanian/StringZilla/actions/runs/7706636733/job/21002535521
#ifndef SSIZE_MAX
#define SSIZE_MAX (SIZE_MAX / 2)
#endif

#include <Python.h> // Core CPython interfaces

#include <errno.h>  // `errno`
#include <stdio.h>  // `fopen`
#include <stdlib.h> // `rand`, `srand`
#include <time.h>   // `time`

#include <stringzilla/stringzillas.h>

#pragma region Forward Declarations

static PyTypeObject SimilaritiesEngineType;
static PyTypeObject FingerprintsEngineType;

/**
 *  @brief  Type-punned StringZilla-string, that points to a slice of an existing Python `str`
 *          or a `File`.
 *
 *  When a slice is constructed, the `parent` object's reference count is being incremented to preserve lifetime.
 *  It usage in Python would look like:
 *
 *      - Str() # Empty string
 *      - Str("some-string") # Full-range slice of a Python `str`
 *      - Str(File("some-path.txt")) # Full-range view of a persisted file
 *      - Str(File("some-path.txt"), from=0, to=sys.maxsize)
 */
typedef struct {
    PyObject ob_base;

    PyObject *parent;
    sz_string_view_t memory;
} Str;

static PyObject *_Str_levenshtein_distance(PyObject *self, PyObject *args, PyObject *kwargs,
                                           sz_levenshtein_distance_t function) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *str2_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *bound_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "bound") == 0 && !bound_obj) { bound_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
    }

    sz_size_t bound = SZ_SIZE_MAX; // Default value for bound
    if (bound_obj && ((bound = (sz_size_t)PyLong_AsSize_t(bound_obj)) == (sz_size_t)(-1))) {
        PyErr_Format(PyExc_ValueError, "Bound must be a non-negative integer");
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Allocate memory for the Levenshtein matrix
    sz_memory_allocator_t reusing_allocator;
    reusing_allocator.allocate = &temporary_memory_allocate;
    reusing_allocator.free = &temporary_memory_free;
    reusing_allocator.handle = &temporary_memory;

    sz_size_t distance;
    sz_status_t status =
        function(str1.start, str1.length, str2.start, str2.length, bound, &reusing_allocator, &distance);

    // Check for memory allocation issues
    if (status != sz_success_k) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromSize_t(distance);
}

static char const doc_levenshtein_distance[] = //
    "Compute the Levenshtein edit distance between two strings.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The first string.\n"
    "  other (str): The second string to compare.\n"
    "  bound (int, optional): Optional maximum distance to compute (default is no bound).\n"
    "Returns:\n"
    "  int: The edit distance (number of insertions, deletions, substitutions).";

static PyObject *Str_levenshtein_distance(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_levenshtein_distance(self, args, kwargs, &sz_levenshtein_distance);
}

static char const doc_levenshtein_distance_unicode[] = //
    "Compute the Levenshtein edit distance between two Unicode strings.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The first string.\n"
    "  other (str): The second string to compare.\n"
    "  bound (int, optional): Optional maximum distance to compute (default is no bound).\n"
    "Returns:\n"
    "  int: The edit distance in Unicode characters.";

static PyObject *Str_levenshtein_distance_unicode(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_levenshtein_distance(self, args, kwargs, &sz_levenshtein_distance_utf8);
}

static PyObject *_Str_hamming_distance(PyObject *self, PyObject *args, PyObject *kwargs,
                                       sz_hamming_distance_t function) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *str2_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *bound_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "bound") == 0 && !bound_obj) { bound_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
    }

    Py_ssize_t bound = 0; // Default value for bound
    if (bound_obj && ((bound = PyLong_AsSsize_t(bound_obj)) < 0)) {
        PyErr_Format(PyExc_ValueError, "Bound must be a non-negative integer");
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    sz_size_t distance;
    sz_status_t status = function(str1.start, str1.length, str2.start, str2.length, (sz_size_t)bound, &distance);

    // Check for memory allocation issues
    if (status != sz_success_k) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromSize_t(distance);
}

static char const doc_hamming_distance[] = //
    "Compute the Hamming distance between two strings.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The first string.\n"
    "  other (str): The second string to compare.\n"
    "  bound (int, optional): Optional maximum distance to compute (default is no bound).\n"
    "Returns:\n"
    "  int: The Hamming distance, including differing bytes and length difference.";

static PyObject *Str_hamming_distance(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_hamming_distance(self, args, kwargs, &sz_hamming_distance);
}

static char const doc_hamming_distance_unicode[] = //
    "Compute the Hamming distance between two Unicode strings.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The first string.\n"
    "  other (str): The second string to compare.\n"
    "  bound (int, optional): Optional maximum distance to compute (default is no bound).\n"
    "Returns:\n"
    "  int: The Hamming distance, including differing Unicode characters and length difference.";

static PyObject *Str_hamming_distance_unicode(PyObject *self, PyObject *args, PyObject *kwargs) {
    return _Str_hamming_distance(self, args, kwargs, &sz_hamming_distance_utf8);
}

static char const doc_needleman_wunsch_score[] = //
    "Compute the Needleman-Wunsch alignment score between two strings.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The first string.\n"
    "  other (str): The second string to align.\n"
    "  substitution_matrix (numpy.ndarray): A 256x256 substitution cost matrix.\n"
    "  gap_score (int): The score for introducing a gap.\n"
    "  bound (int, optional): Optional maximum score to compute (default is no bound).\n"
    "Returns:\n"
    "  int: The alignment score.";

static PyObject *Str_needleman_wunsch_score(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 2) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str1_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *str2_obj = PyTuple_GET_ITEM(args, !is_member + 0);
    PyObject *substitution_matrix_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *gap_score_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "gap_score") == 0 && !gap_score_obj) { gap_score_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "substitution_matrix") == 0 && !substitution_matrix_obj) {
                substitution_matrix_obj = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
    }

    Py_ssize_t gap = 1; // Default value for gap costs
    if (gap_score_obj && (gap = PyLong_AsSsize_t(gap_score_obj)) && (gap >= 128 || gap <= -128)) {
        PyErr_Format(PyExc_ValueError, "The `gap_score` must fit into an 8-bit signed integer");
        return NULL;
    }

    // Now extract the substitution matrix from the `substitution_matrix_obj`.
    // It must conform to the buffer protocol, and contain a continuous 256x256 matrix of 8-bit signed integers.
    sz_error_cost_t const *substitutions;

    // Ensure the substitution matrix object is provided
    if (!substitution_matrix_obj) {
        PyErr_Format(PyExc_TypeError, "No substitution matrix provided");
        return NULL;
    }

    // Request a buffer view
    Py_buffer substitutions_view;
    if (PyObject_GetBuffer(substitution_matrix_obj, &substitutions_view, PyBUF_FULL)) {
        PyErr_Format(PyExc_TypeError, "Failed to get buffer from substitution matrix");
        return NULL;
    }

    // Validate the buffer
    if (substitutions_view.ndim != 2 || substitutions_view.shape[0] != 256 || substitutions_view.shape[1] != 256 ||
        substitutions_view.itemsize != sizeof(sz_error_cost_t)) {
        PyErr_Format(PyExc_ValueError, "Substitution matrix must be a 256x256 matrix of 8-bit signed integers");
        PyBuffer_Release(&substitutions_view);
        return NULL;
    }

    sz_string_view_t str1, str2;
    if (!export_string_like(str1_obj, &str1.start, &str1.length) ||
        !export_string_like(str2_obj, &str2.start, &str2.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Assign the buffer's data to substitutions
    substitutions = (sz_error_cost_t const *)substitutions_view.buf;

    // Allocate memory for the Levenshtein matrix
    sz_memory_allocator_t reusing_allocator;
    reusing_allocator.allocate = &temporary_memory_allocate;
    reusing_allocator.free = &temporary_memory_free;
    reusing_allocator.handle = &temporary_memory;

    sz_ssize_t score;
    sz_status_t status = sz_needleman_wunsch_score(str1.start, str1.length, str2.start, str2.length, substitutions,
                                                   (sz_error_cost_t)gap, &reusing_allocator, &score);

    // Don't forget to release the buffer view
    PyBuffer_Release(&substitutions_view);

    // Check for memory allocation issues
    if (status != sz_success_k) {
        PyErr_NoMemory();
        return NULL;
    }

    return PyLong_FromSsize_t(score);
}

static char const doc_startswith[] = //
    "Check if a string starts with a given prefix.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  prefix (str): The prefix to check.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  bool: True if the string starts with the prefix, False otherwise.";

static PyObject *Str_startswith(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 3) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *prefix_obj = PyTuple_GET_ITEM(args, !is_member);
    PyObject *start_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *end_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str, prefix;
    if (!export_string_like(str_obj, &str.start, &str.length) ||
        !export_string_like(prefix_obj, &prefix.start, &prefix.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && (sz_size_t)(end - start) < str.length) { str.length = (sz_size_t)(end - start); }

    if (str.length < prefix.length) { Py_RETURN_FALSE; }
    else if (strncmp(str.start, prefix.start, prefix.length) == 0) { Py_RETURN_TRUE; }
    else { Py_RETURN_FALSE; }
}

static char const doc_endswith[] = //
    "Check if a string ends with a given suffix.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  suffix (str): The suffix to check.\n"
    "  start (int, optional): The starting index (default is 0).\n"
    "  end (int, optional): The ending index (default is the string length).\n"
    "Returns:\n"
    "  bool: True if the string ends with the suffix, False otherwise.";

static PyObject *Str_endswith(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 3) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *suffix_obj = PyTuple_GET_ITEM(args, !is_member);
    PyObject *start_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *end_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str, suffix;
    if (!export_string_like(str_obj, &str.start, &str.length) ||
        !export_string_like(suffix_obj, &suffix.start, &suffix.length)) {
        wrap_current_exception("Both arguments must be string-like");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && (sz_size_t)(end - start) < str.length) { str.length = (sz_size_t)(end - start); }

    if (str.length < suffix.length) { Py_RETURN_FALSE; }
    else if (strncmp(str.start + (str.length - suffix.length), suffix.start, suffix.length) == 0) { Py_RETURN_TRUE; }
    else { Py_RETURN_FALSE; }
}

static char const doc_translate[] = //
    "Perform transformation of a string using a look-up table.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  table (str or dict): A 256-character string or a dictionary mapping bytes to bytes.\n"
    "  inplace (bool, optional): If True, the string is modified in place (default is False).\n"
    "\n"
    "  start (int, optional): The starting index for translation (default is 0).\n"
    "  end (int, optional): The ending index for translation (default is the string length).\n"
    "Returns:\n"
    "  Union[None, str, bytes]: If inplace is False, a new string is returned, otherwise None.\n"
    "Raises:\n"
    "  ValueError: If the table is not 256 bytes long.\n"
    "  TypeError: If the table is not a string or dictionary.";

static PyObject *Str_translate(PyObject *self, PyObject *args, PyObject *kwargs) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member + 1 || nargs > !is_member + 4) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *str_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *look_up_table_obj = PyTuple_GET_ITEM(args, !is_member);
    PyObject *inplace_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *start_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;
    PyObject *end_obj = nargs > !is_member + 3 ? PyTuple_GET_ITEM(args, !is_member + 3) : NULL;

    // Optional keyword arguments
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value))
            if (PyUnicode_CompareWithASCIIString(key, "inplace") == 0 && !inplace_obj) { inplace_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "start") == 0 && !start_obj) { start_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0 && !end_obj) { end_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
    }

    // Optional start and end arguments
    Py_ssize_t start = 0, end = PY_SSIZE_T_MAX;

    if (start_obj && ((start = PyLong_AsSsize_t(start_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }

    if (end_obj && ((end = PyLong_AsSsize_t(end_obj)) == -1 && PyErr_Occurred())) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t str;
    if (!export_string_like(str_obj, &str.start, &str.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }

    sz_string_view_t look_up_table_str;
    SZ_ALIGN64 char look_up_table[256];
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
    else if (export_string_like(look_up_table_obj, &look_up_table_str.start, &look_up_table_str.length)) {
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

    int is_inplace = inplace_obj ? PyObject_IsTrue(inplace_obj) : 0;
    if (is_inplace == -1) {
        PyErr_SetString(PyExc_TypeError, "The inplace argument must be a boolean");
        return NULL;
    }

    // Apply start and end arguments
    str.start += start;
    str.length -= start;
    if (end != PY_SSIZE_T_MAX && (sz_size_t)(end - start) < str.length) { str.length = (sz_size_t)(end - start); }

    // Perform the translation using the look-up table
    if (is_inplace) {
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

static char const doc_find_first_of[] = //
    "Find the index of the first occurrence of any character from another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to search for.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the first matching character, or -1 if none found.";

static PyObject *Str_find_first_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find_byte_from, sz_false_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_find_first_not_of[] = //
    "Find the index of the first character not in another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to exclude.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the first non-matching character, or -1 if all match.";

static PyObject *Str_find_first_not_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_find_byte_not_from, sz_false_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_find_last_of[] = //
    "Find the index of the last occurrence of any character from another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to search for.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the last matching character, or -1 if none found.";

static PyObject *Str_find_last_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_rfind_byte_from, sz_true_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

static char const doc_find_last_not_of[] = //
    "Find the index of the last character not in another string.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  chars (str): A string containing characters to exclude.\n"
    "  start (int, optional): Starting index (default is 0).\n"
    "  end (int, optional): Ending index (default is the string length).\n"
    "Returns:\n"
    "  int: Index of the last non-matching character, or -1 if all match.";

static PyObject *Str_find_last_not_of(PyObject *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t signed_offset;
    sz_string_view_t text;
    sz_string_view_t separator;
    if (!_Str_find_implementation_(self, args, kwargs, &sz_rfind_byte_not_from, sz_true_k, &signed_offset, &text,
                                   &separator))
        return NULL;
    return PyLong_FromSsize_t(signed_offset);
}

/**
 *  @brief  Given parsed split settings, constructs an iterator that would produce that split.
 */
static SplitIterator *Str_split_iter_(PyObject *text_obj, PyObject *separator_obj,                   //
                                      sz_string_view_t const text, sz_string_view_t const separator, //
                                      int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length,
                                      sz_bool_t is_reverse) {

    // Create a new `SplitIterator` object
    SplitIterator *result_obj = (SplitIterator *)SplitIteratorType.tp_alloc(&SplitIteratorType, 0);
    if (result_obj == NULL && PyErr_NoMemory()) return NULL;

    // Set its properties based on the slice
    result_obj->text_obj = text_obj;
    result_obj->separator_obj = separator_obj;
    result_obj->text = text;
    result_obj->separator = separator;
    result_obj->finder = finder;

    result_obj->match_length = match_length;
    result_obj->include_match = keepseparator;
    result_obj->is_reverse = is_reverse;
    result_obj->max_parts = (sz_size_t)maxsplit + 1;
    result_obj->reached_tail = 0;

    // Increment the reference count of the parent
    Py_INCREF(result_obj->text_obj);
    Py_XINCREF(result_obj->separator_obj);
    return result_obj;
}

/**
 *  @brief  Implements the normal order split logic for both string-delimiters and character sets.
 *          Produces one of the consecutive layouts - `STRS_CONSECUTIVE_64` or `STRS_CONSECUTIVE_32`.
 */
static Strs *Str_split_(PyObject *parent_string, sz_string_view_t const text, sz_string_view_t const separator,
                        int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length) {
    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) return NULL;

    // Initialize Strs object based on the splitting logic
    void *offsets_endings = NULL;
    size_t offsets_capacity = 0;
    size_t offsets_count = 0;
    size_t bytes_per_offset;
    if (text.length >= UINT32_MAX) {
        bytes_per_offset = 8;
        result->type = STRS_CONSECUTIVE_64;
        result->data.consecutive_64bit.start = text.start;
        result->data.consecutive_64bit.parent_string = parent_string;
        result->data.consecutive_64bit.separator_length = !keepseparator * match_length;
    }
    else {
        bytes_per_offset = 4;
        result->type = STRS_CONSECUTIVE_32;
        result->data.consecutive_32bit.start = text.start;
        result->data.consecutive_32bit.parent_string = parent_string;
        result->data.consecutive_32bit.separator_length = !keepseparator * match_length;
    }

    sz_bool_t reached_tail = 0;
    sz_size_t total_skipped = 0;
    sz_size_t max_parts = (sz_size_t)maxsplit + 1;
    while (!reached_tail) {

        sz_cptr_t match =
            offsets_count + 1 < max_parts
                ? finder(text.start + total_skipped, text.length - total_skipped, separator.start, separator.length)
                : NULL;

        sz_size_t part_end_offset;
        if (match) {
            part_end_offset = (match - text.start) + match_length;
            total_skipped = part_end_offset;
        }
        else {
            part_end_offset = text.length;
            total_skipped = text.length;
            reached_tail = 1;
        }

        // Reallocate offsets array if needed
        if (offsets_count >= offsets_capacity) {
            offsets_capacity = (offsets_capacity + 1) * 2;
            void *new_offsets = realloc(offsets_endings, offsets_capacity * bytes_per_offset);
            if (!new_offsets) {
                if (offsets_endings) free(offsets_endings);
            }
            offsets_endings = new_offsets;
        }

        // If the memory allocation has failed - discard the response
        if (!offsets_endings) {
            Py_XDECREF(result);
            PyErr_NoMemory();
            return NULL;
        }

        // Export the offset
        if (bytes_per_offset == 8) { ((uint64_t *)offsets_endings)[offsets_count] = (uint64_t)part_end_offset; }
        else { ((uint32_t *)offsets_endings)[offsets_count] = (uint32_t)part_end_offset; }
        offsets_count++;
    }

    // Populate the Strs object with the offsets
    if (bytes_per_offset == 8) {
        result->data.consecutive_64bit.end_offsets = offsets_endings;
        result->data.consecutive_64bit.count = offsets_count;
    }
    else {
        result->data.consecutive_32bit.end_offsets = offsets_endings;
        result->data.consecutive_32bit.count = offsets_count;
    }

    Py_INCREF(parent_string);
    return result;
}

/**
 *  @brief  Implements the reverse order split logic for both string-delimiters and character sets.
 *          Unlike the `Str_split_` can't use consecutive layouts and produces a `REORDERED` one.
 */
static Strs *Str_rsplit_(PyObject *parent_string, sz_string_view_t const text, sz_string_view_t const separator,
                         int keepseparator, Py_ssize_t maxsplit, sz_find_t finder, sz_size_t match_length) {
    // Create Strs object
    Strs *result = (Strs *)PyObject_New(Strs, &StrsType);
    if (!result) return NULL;

    // Initialize Strs object based on the splitting logic
    result->type = STRS_REORDERED;
    result->data.reordered.parent_string = parent_string;
    result->data.reordered.parts = NULL;
    result->data.reordered.count = 0;

    // Keep track of the memory usage
    sz_string_view_t *parts = NULL;
    sz_size_t parts_capacity = 0;
    sz_size_t parts_count = 0;

    sz_bool_t reached_tail = 0;
    sz_size_t total_skipped = 0;
    sz_size_t max_parts = (sz_size_t)maxsplit + 1;
    while (!reached_tail) {

        sz_cptr_t match = parts_count + 1 < max_parts
                              ? finder(text.start, text.length - total_skipped, separator.start, separator.length)
                              : NULL;

        // Determine the next part
        sz_string_view_t part;
        if (match) {
            part.start = match + match_length * !keepseparator;
            part.length = text.start + text.length - total_skipped - part.start;
            total_skipped = text.start + text.length - match;
        }
        else {
            part.start = text.start;
            part.length = text.length - total_skipped;
            reached_tail = 1;
        }

        // Reallocate parts array if needed
        if (parts_count >= parts_capacity) {
            parts_capacity = (parts_capacity + 1) * 2;
            sz_string_view_t *new_parts = (sz_string_view_t *)realloc(parts, parts_capacity * sizeof(sz_string_view_t));
            if (!new_parts) {
                if (parts) free(parts);
            }
            parts = new_parts;
        }

        // If the memory allocation has failed - discard the response
        if (!parts) {
            Py_XDECREF(result);
            PyErr_NoMemory();
            return NULL;
        }

        // Populate the parts array
        parts[parts_count] = part;
        parts_count++;
    }

    // Python does this weird thing, where the `rsplit` results appear in the same order as `split`
    // so we need to reverse the order of elements in the `parts` array.
    for (sz_size_t i = 0; i < parts_count / 2; i++) {
        sz_string_view_t temp = parts[i];
        parts[i] = parts[parts_count - i - 1];
        parts[parts_count - i - 1] = temp;
    }

    result->data.reordered.parts = parts;
    result->data.reordered.count = parts_count;
    Py_INCREF(parent_string);
    return result;
}

/**
 *  @brief  Proxy routing requests like `Str.split`, `Str.rsplit`, `Str.split_byteset` and `Str.rsplit_byteset`
 *          to `Str_split_` and `Str_rsplit_` implementations, parsing function arguments.
 */
static PyObject *Str_split_with_known_callback(PyObject *self, PyObject *args, PyObject *kwargs, //
                                               sz_find_t finder, sz_size_t match_length,         //
                                               sz_bool_t is_reverse, sz_bool_t is_lazy_iterator) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 3) {
        PyErr_SetString(PyExc_TypeError, "sz.split() received unsupported number of arguments");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *separator_obj = nargs > !is_member + 0 ? PyTuple_GET_ITEM(args, !is_member + 0) : NULL;
    PyObject *maxsplit_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;
    PyObject *keepseparator_obj = nargs > !is_member + 2 ? PyTuple_GET_ITEM(args, !is_member + 2) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "separator") == 0 && !separator_obj) { separator_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0 && !maxsplit_obj) { maxsplit_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "keepseparator") == 0 && !keepseparator_obj) {
                keepseparator_obj = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key))
                return NULL;
        }
    }

    sz_string_view_t text;
    sz_string_view_t separator;
    int keepseparator;
    Py_ssize_t maxsplit;

    // Validate and convert `text`
    if (!export_string_like(text_obj, &text.start, &text.length)) {
        wrap_current_exception("The text argument must be string-like");
        return NULL;
    }

    // Validate and convert `separator`
    if (separator_obj) {
        if (!export_string_like(separator_obj, &separator.start, &separator.length)) {
            wrap_current_exception("The separator argument must be string-like");
            return NULL;
        }
        // Raise a `ValueError` if it's length is zero, like the native `str.split`
        if (separator.length == 0) {
            PyErr_SetString(PyExc_ValueError, "The separator argument must not be empty");
            return NULL;
        }
        if (match_length == 0) match_length = separator.length;
    }
    else {
        separator.start = " ";
        match_length = separator.length = 1;
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

    // Dispatch the right backend
    if (is_lazy_iterator)
        return Str_split_iter_(text_obj, separator_obj, text, separator, //
                               keepseparator, maxsplit, finder, match_length, is_reverse);
    else
        return !is_reverse ? Str_split_(text_obj, text, separator, keepseparator, maxsplit, finder, match_length)
                           : Str_rsplit_(text_obj, text, separator, keepseparator, maxsplit, finder, match_length);
}

static char const doc_split[] = //
    "Split a string by a separator.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include the separator in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the separator.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_split(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find, 0, sz_false_k, sz_false_k);
}

static char const doc_rsplit[] = //
    "Split a string by a separator starting from the end.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include the separator in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the separator.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_rsplit(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind, 0, sz_true_k, sz_false_k);
}

static char const doc_split_byteset[] = //
    "Split a string by a set of character separators.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the character set.";

static PyObject *Str_split_byteset(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find_byte_from, 1, sz_false_k, sz_false_k);
}

static char const doc_rsplit_byteset[] = //
    "Split a string by a set of character separators in reverse order.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  Strs: A list of strings split by the character set.";

static PyObject *Str_rsplit_byteset(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind_byte_from, 1, sz_true_k, sz_false_k);
}

static char const doc_split_iter[] = //
    "Create an iterator for splitting a string by a separator.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  keepseparator (bool, optional): Include separator in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_split_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find, 0, sz_false_k, sz_true_k);
}

static char const doc_rsplit_iter[] = //
    "Create an iterator for splitting a string by a separator in reverse order.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separator (str): The separator to split by (cannot be empty).\n"
    "  keepseparator (bool, optional): Include separator in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings in reverse.\n"
    "Raises:\n"
    "  ValueError: If the separator is an empty string.";

static PyObject *Str_rsplit_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind, 0, sz_true_k, sz_true_k);
}

static char const doc_split_byteset_iter[] = //
    "Create an iterator for splitting a string by a set of character separators.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings.";

static PyObject *Str_split_byteset_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_find_byte_from, 1, sz_false_k, sz_true_k);
}

static char const doc_rsplit_byteset_iter[] = //
    "Create an iterator for splitting a string by a set of character separators in reverse order.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  separators (str): A string containing separator characters.\n"
    "  keepseparator (bool, optional): Include separators in results (default is False).\n"
    "Returns:\n"
    "  iterator: An iterator yielding split substrings in reverse.";

static PyObject *Str_rsplit_byteset_iter(PyObject *self, PyObject *args, PyObject *kwargs) {
    return Str_split_with_known_callback(self, args, kwargs, &sz_rfind_byte_from, 1, sz_true_k, sz_true_k);
}

static char const doc_splitlines[] = //
    "Split a string by line breaks.\n"
    "\n"
    "Args:\n"
    "  text (Str or str or bytes): The string object.\n"
    "  keeplinebreaks (bool, optional): Include line breaks in the results (default is False).\n"
    "  maxsplit (int, optional): Maximum number of splits (default is no limit).\n"
    "Returns:\n"
    "  Strs: A list of strings split by line breaks.";

static PyObject *Str_splitlines(PyObject *self, PyObject *args, PyObject *kwargs) {
    // Check minimum arguments
    int is_member = self != NULL && PyObject_TypeCheck(self, &StrType);
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs < !is_member || nargs > !is_member + 2) {
        PyErr_SetString(PyExc_TypeError, "splitlines() requires at least 1 argument");
        return NULL;
    }

    PyObject *text_obj = is_member ? self : PyTuple_GET_ITEM(args, 0);
    PyObject *keeplinebreaks_obj = nargs > !is_member ? PyTuple_GET_ITEM(args, !is_member) : NULL;
    PyObject *maxsplit_obj = nargs > !is_member + 1 ? PyTuple_GET_ITEM(args, !is_member + 1) : NULL;

    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "keeplinebreaks") == 0 && !keeplinebreaks_obj) {
                keeplinebreaks_obj = value;
            }
            else if (PyUnicode_CompareWithASCIIString(key, "maxsplit") == 0 && !maxsplit_obj) { maxsplit_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_string_view_t text;
    int keeplinebreaks;
    Py_ssize_t maxsplit = PY_SSIZE_T_MAX; // Default value for maxsplit

    // Validate and convert `text`
    if (!export_string_like(text_obj, &text.start, &text.length)) {
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
    sz_string_view_t separator;
    separator.start = "\x0A\x0B\x0C\x0D\x85\x1C\x1D\x1E";
    separator.length = 8;
    return Str_split_(text_obj, text, separator, keeplinebreaks, maxsplit, &sz_find_byte_from, 1);
}

static PyObject *Str_concat(PyObject *self, PyObject *other) {
    struct sz_string_view_t self_str, other_str;

    // Validate and convert `self` and `other`
    if (!export_string_like(self, &self_str.start, &self_str.length) ||
        !export_string_like(other, &other_str.start, &other_str.length)) {
        wrap_current_exception("Both operands must be string-like");
        return NULL;
    }

    // Allocate a new Str instance
    Str *result_str = PyObject_New(Str, &StrType);
    if (result_str == NULL) { return NULL; }

    // Calculate the total length of the new string
    result_str->parent = NULL;
    result_str->memory.length = self_str.length + other_str.length;

    // Allocate memory for the new string
    result_str->memory.start = malloc(result_str->memory.length);
    if (result_str->memory.start == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for string concatenation");
        return NULL;
    }

    // Perform the string concatenation
    sz_copy(result_str->memory.start, self_str.start, self_str.length);
    sz_copy(result_str->memory.start + self_str.length, other_str.start, other_str.length);

    return (PyObject *)result_str;
}

static PySequenceMethods Str_as_sequence = {
    .sq_length = Str_len,   //
    .sq_item = Str_getitem, //
    .sq_contains = Str_in,  //
};

static PyMappingMethods Str_as_mapping = {
    .mp_length = Str_len,          //
    .mp_subscript = Str_subscript, // Is used to implement slices in Python
};

static PyBufferProcs Str_as_buffer = {
    .bf_getbuffer = Str_getbuffer,
    .bf_releasebuffer = Str_releasebuffer,
};

static PyNumberMethods Str_as_number = {
    .nb_add = Str_concat,
};

static PyGetSetDef Str_getsetters[] = {
    // Compatibility with PyArrow
    {"address", (getter)Str_get_address, NULL, "Get the memory address of the first byte of the string", NULL},
    {"nbytes", (getter)Str_get_nbytes, NULL, "Get the length of the string in bytes", NULL},
    {NULL} // Sentinel
};

#define SZ_METHOD_FLAGS METH_VARARGS | METH_KEYWORDS

static PyMethodDef Str_methods[] = {
    {"contains", (PyCFunction)Str_contains, SZ_METHOD_FLAGS, doc_contains},
    {"count", (PyCFunction)Str_count, SZ_METHOD_FLAGS, doc_count},
    {"splitlines", (PyCFunction)Str_splitlines, SZ_METHOD_FLAGS, doc_splitlines},
    {"startswith", (PyCFunction)Str_startswith, SZ_METHOD_FLAGS, doc_startswith},
    {"endswith", (PyCFunction)Str_endswith, SZ_METHOD_FLAGS, doc_endswith},
    {"translate", (PyCFunction)Str_translate, SZ_METHOD_FLAGS, doc_translate},
    {"decode", (PyCFunction)Str_decode, SZ_METHOD_FLAGS, doc_decode},

    // Bidirectional operations
    {"find", (PyCFunction)Str_find, SZ_METHOD_FLAGS, doc_find},
    {"index", (PyCFunction)Str_index, SZ_METHOD_FLAGS, doc_index},
    {"partition", (PyCFunction)Str_partition, SZ_METHOD_FLAGS, doc_partition},
    {"split", (PyCFunction)Str_split, SZ_METHOD_FLAGS, doc_split},
    {"rfind", (PyCFunction)Str_rfind, SZ_METHOD_FLAGS, doc_rfind},
    {"rindex", (PyCFunction)Str_rindex, SZ_METHOD_FLAGS, doc_rindex},
    {"rpartition", (PyCFunction)Str_rpartition, SZ_METHOD_FLAGS, doc_rpartition},
    {"rsplit", (PyCFunction)Str_rsplit, SZ_METHOD_FLAGS, doc_rsplit},

    // Edit distance extensions
    {"hamming_distance", (PyCFunction)Str_hamming_distance, SZ_METHOD_FLAGS, doc_hamming_distance},
    {"hamming_distance_unicode", (PyCFunction)Str_hamming_distance_unicode, SZ_METHOD_FLAGS,
     doc_hamming_distance_unicode},
    {"levenshtein_distance", (PyCFunction)Str_levenshtein_distance, SZ_METHOD_FLAGS, doc_levenshtein_distance},
    {"levenshtein_distance_unicode", (PyCFunction)Str_levenshtein_distance_unicode, SZ_METHOD_FLAGS,
     doc_levenshtein_distance_unicode},
    {"needleman_wunsch_score", (PyCFunction)Str_needleman_wunsch_score, SZ_METHOD_FLAGS, doc_needleman_wunsch_score},

    // Character search extensions
    {"find_first_of", (PyCFunction)Str_find_first_of, SZ_METHOD_FLAGS, doc_find_first_of},
    {"find_last_of", (PyCFunction)Str_find_last_of, SZ_METHOD_FLAGS, doc_find_last_of},
    {"find_first_not_of", (PyCFunction)Str_find_first_not_of, SZ_METHOD_FLAGS, doc_find_first_not_of},
    {"find_last_not_of", (PyCFunction)Str_find_last_not_of, SZ_METHOD_FLAGS, doc_find_last_not_of},
    {"split_byteset", (PyCFunction)Str_split_byteset, SZ_METHOD_FLAGS, doc_split_byteset},
    {"rsplit_byteset", (PyCFunction)Str_rsplit_byteset, SZ_METHOD_FLAGS, doc_rsplit_byteset},

    // Lazily evaluated iterators
    {"split_iter", (PyCFunction)Str_split_iter, SZ_METHOD_FLAGS, doc_split_iter},
    {"rsplit_iter", (PyCFunction)Str_rsplit_iter, SZ_METHOD_FLAGS, doc_rsplit_iter},
    {"split_byteset_iter", (PyCFunction)Str_split_byteset_iter, SZ_METHOD_FLAGS, doc_split_byteset_iter},
    {"rsplit_byteset_iter", (PyCFunction)Str_rsplit_byteset_iter, SZ_METHOD_FLAGS, doc_rsplit_byteset_iter},

    // Dealing with larger-than-memory datasets
    {"offset_within", (PyCFunction)Str_offset_within, SZ_METHOD_FLAGS, doc_offset_within},
    {"write_to", (PyCFunction)Str_write_to, SZ_METHOD_FLAGS, doc_write_to},

    {NULL, NULL, 0, NULL} // Sentinel
};

static PyTypeObject StrType = {
    PyVarObject_HEAD_INIT(NULL, 0) //
        .tp_name = "stringzilla.Str",
    .tp_doc = "Immutable string/slice class with SIMD and SWAR-accelerated operations",
    .tp_basicsize = sizeof(Str),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Str_new,
    .tp_init = Str_init,
    .tp_dealloc = Str_dealloc,
    .tp_hash = Str_hash,
    .tp_richcompare = Str_richcompare,
    .tp_repr = (reprfunc)Str_repr,
    .tp_str = Str_str,
    .tp_methods = Str_methods,
    .tp_as_sequence = &Str_as_sequence,
    .tp_as_mapping = &Str_as_mapping,
    .tp_as_buffer = &Str_as_buffer,
    .tp_as_number = &Str_as_number,
    .tp_getset = Str_getsetters,
};

#pragma endregion

#pragma region Split Iterator

static PyObject *SplitIteratorType_next(SplitIterator *self) {
    // No more data to split
    if (self->reached_tail) return NULL;

    // Create a new `Str` object
    Str *result_obj = (Str *)StrType.tp_alloc(&StrType, 0);
    if (result_obj == NULL && PyErr_NoMemory()) return NULL;

    sz_string_view_t result_memory;

    // Find the next needle
    sz_cptr_t found =
        self->max_parts > 1 //
            ? self->finder(self->text.start, self->text.length, self->separator.start, self->separator.length)
            : NULL;

    // We've reached the end of the string
    if (found == NULL) {
        result_memory.start = self->text.start;
        result_memory.length = self->text.length;
        self->text.length = 0;
        self->reached_tail = 1;
        self->max_parts = 0;
    }
    else {
        if (self->is_reverse) {
            result_memory.start = found + self->match_length * !self->include_match;
            result_memory.length = self->text.start + self->text.length - result_memory.start;
            self->text.length = found - self->text.start;
        }
        else {
            result_memory.start = self->text.start;
            result_memory.length = found - self->text.start;
            self->text.start = found + self->match_length;
            self->text.length -= result_memory.length + self->match_length;
            result_memory.length += self->match_length * self->include_match;
        }
        self->max_parts--;
    }

    // Set its properties based on the slice
    result_obj->memory = result_memory;
    result_obj->parent = self->text_obj;

    // Increment the reference count of the parent
    Py_INCREF(self->text_obj);
    return (PyObject *)result_obj;
}

static void SplitIteratorType_dealloc(SplitIterator *self) {
    Py_XDECREF(self->text_obj);
    Py_XDECREF(self->separator_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *SplitIteratorType_iter(PyObject *self) {
    Py_INCREF(self); // Iterator should return itself in __iter__.
    return self;
}

static PyTypeObject SplitIteratorType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.SplitIterator",
    .tp_basicsize = sizeof(SplitIterator),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)SplitIteratorType_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Text-splitting iterator",
    .tp_iter = SplitIteratorType_iter,
    .tp_iternext = (iternextfunc)SplitIteratorType_next,
};

#pragma endregion

#pragma region Strs

static PyObject *Strs_shuffle(Strs *self, PyObject *args, PyObject *kwargs) {

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    PyObject *seed_obj = nargs == 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "shuffle() takes at most 1 positional argument");
        return NULL;
    }

    // Check for keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_obj) { seed_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Change the layout
    if (!prepare_strings_for_reordering(self)) {
        PyErr_Format(PyExc_TypeError, "Failed to prepare the sequence for shuffling");
        return NULL;
    }

    // Get the parts and their count
    struct reordered_slices_t *reordered = &self->data.reordered;
    sz_string_view_t *parts = reordered->parts;
    size_t count = reordered->count;

    // Fisher-Yates Shuffle Algorithm
    unsigned int seed = seed_obj ? PyLong_AsUnsignedLong(seed_obj) : time(NULL);
    srand(seed);
    for (size_t i = count - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        // Swap parts[i] and parts[j]
        sz_string_view_t temp = parts[i];
        parts[i] = parts[j];
        parts[j] = temp;
    }

    Py_RETURN_NONE;
}

static sz_bool_t Strs_argsort_(Strs *self, sz_string_view_t **parts_output, sz_sorted_idx_t **order_output,
                               sz_size_t *count_output) {
    // Change the layout
    if (!prepare_strings_for_reordering(self)) {
        PyErr_Format(PyExc_TypeError, "Failed to prepare the sequence for sorting");
        return 0;
    }

    // Get the parts and their count
    // The only possible `self->type` by now is the `STRS_REORDERED`
    sz_string_view_t *parts = self->data.reordered.parts;
    size_t count = self->data.reordered.count;

    // Allocate temporary memory to store the ordering offsets
    size_t memory_needed = sizeof(sz_sorted_idx_t) * count;
    if (temporary_memory.length < memory_needed) {
        temporary_memory.start = realloc(temporary_memory.start, memory_needed);
        temporary_memory.length = memory_needed;
    }
    if (!temporary_memory.start) {
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the Levenshtein matrix");
        return 0;
    }

    // Call our sorting algorithm
    sz_sequence_t sequence;
    sz_fill(&sequence, sizeof(sequence), 0);
    sequence.count = count;
    sequence.handle = parts;
    sequence.get_start = parts_get_start;
    sequence.get_length = parts_get_length;
    sz_status_t status = sz_sequence_argsort(&sequence, NULL, (sz_sorted_idx_t *)temporary_memory.start);

    // Export results
    *parts_output = parts;
    *order_output = (sz_sorted_idx_t *)temporary_memory.start;
    *count_output = sequence.count;
    return 1;
}

static PyObject *Strs_sort(Strs *self, PyObject *args, PyObject *kwargs) {
    PyObject *reverse_obj = NULL; // Default is not reversed

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "sort() takes at most 1 positional argument");
        return NULL;
    }
    else if (nargs == 1) { reverse_obj = PyTuple_GET_ITEM(args, 0); }

    // Check for keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0 && !reverse_obj) { reverse_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_bool_t reverse = 0; // Default is False
    if (reverse_obj) {
        if (!PyBool_Check(reverse_obj)) {
            PyErr_SetString(PyExc_TypeError, "The reverse must be a boolean");
            return NULL;
        }
        reverse = PyObject_IsTrue(reverse_obj);
    }

    sz_string_view_t *parts = NULL;
    sz_size_t *order = NULL;
    sz_size_t count = 0;
    if (!Strs_argsort_(self, &parts, &order, &count)) return NULL;

    // Apply the sorting algorithm here, considering the `reverse` value
    if (reverse) reverse_offsets(order, count);

    // Apply the new order.
    permute(parts, order, count);

    Py_RETURN_NONE;
}

static PyObject *Strs_argsort(Strs *self, PyObject *args, PyObject *kwargs) {
    PyObject *reverse_obj = NULL; // Default is not reversed

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "order() takes at most 1 positional argument");
        return NULL;
    }
    else if (nargs == 1) { reverse_obj = PyTuple_GET_ITEM(args, 0); }

    // Check for keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0 && !reverse_obj) { reverse_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    sz_bool_t reverse = 0; // Default is False
    if (reverse_obj) {
        if (!PyBool_Check(reverse_obj)) {
            PyErr_SetString(PyExc_TypeError, "The reverse must be a boolean");
            return NULL;
        }
        reverse = PyObject_IsTrue(reverse_obj);
    }

    sz_string_view_t *parts = NULL;
    sz_sorted_idx_t *order = NULL;
    sz_size_t count = 0;
    if (!Strs_argsort_(self, &parts, &order, &count)) return NULL;

    // Apply the sorting algorithm here, considering the `reverse` value
    if (reverse) reverse_offsets(order, count);

    // Here, instead of applying the order, we want to return the copy of the
    // order as a NumPy array of 64-bit unsigned integers.
    //
    //      npy_intp numpy_size = count;
    //      PyObject *array = PyArray_SimpleNew(1, &numpy_size, NPY_UINT64);
    //      if (!array) {
    //          PyErr_SetString(PyExc_RuntimeError, "Failed to create a NumPy array");
    //          return NULL;
    //      }
    //      sz_sorted_idx_t *numpy_data_ptr = (sz_sorted_idx_t *)PyArray_DATA((PyArrayObject *)array);
    //      sz_copy(numpy_data_ptr, order, count * sizeof(sz_sorted_idx_t));
    //
    // There are compilation issues with NumPy.
    // Here is an example for `cp312-musllinux_s390x`: https://x.com/ashvardanian/status/1757880762278531447?s=20
    // So instead of NumPy, let's produce a tuple of integers.
    PyObject *tuple = PyTuple_New(count);
    if (!tuple) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create a tuple");
        return NULL;
    }
    for (sz_size_t i = 0; i < count; ++i) {
        PyObject *index = PyLong_FromUnsignedLong(order[i]);
        if (!index) {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create a tuple element");
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, index);
    }
    return tuple;
}

static PyObject *Strs_sample(Strs *self, PyObject *args, PyObject *kwargs) {
    PyObject *sample_size_obj = NULL;
    PyObject *seed_obj = NULL;

    // Check for positional arguments
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 1) {
        PyErr_SetString(PyExc_TypeError, "sample() takes 1 positional argument and 1 keyword argument");
        return NULL;
    }
    else if (nargs == 1) { sample_size_obj = PyTuple_GET_ITEM(args, 0); }

    // Parse keyword arguments
    if (kwargs) {
        PyObject *key, *value;
        Py_ssize_t pos = 0;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_obj) { seed_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Translate the seed and the sample size to C types
    size_t sample_size = 0;
    if (sample_size_obj) {
        if (!PyLong_Check(sample_size_obj)) {
            PyErr_SetString(PyExc_TypeError, "The sample size must be an integer");
            return NULL;
        }
        sample_size = PyLong_AsSize_t(sample_size_obj);
    }
    unsigned int seed = time(NULL); // Default seed
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_obj);
    }

    // Create a new `Strs` object
    Strs *result = (Strs *)StrsType.tp_alloc(&StrsType, 0);
    if (result == NULL && PyErr_NoMemory()) return NULL;

    result->type = STRS_REORDERED;
    result->data.reordered.count = 0;
    result->data.reordered.parts = NULL;
    result->data.reordered.parent_string = NULL;
    if (sample_size == 0) { return (PyObject *)result; }

    // Now create a new Strs object with the sampled strings
    sz_string_view_t *result_parts = malloc(sample_size * sizeof(sz_string_view_t));
    if (!result_parts) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the sample");
        return NULL;
    }

    // Introspect the Strs object to know the from which will be sampling
    Py_ssize_t count = Strs_len(self);
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    // Randomly sample the strings
    srand(seed);
    PyObject *parent_string;
    for (Py_ssize_t i = 0; i < sample_size; i++) {
        size_t index = rand() % count;
        getter(self, index, count, &parent_string, &result_parts[i].start, &result_parts[i].length);
    }

    // Update the Strs object
    result->type = STRS_REORDERED;
    result->data.reordered.count = sample_size;
    result->data.reordered.parts = result_parts;
    result->data.reordered.parent_string = parent_string;
    return result;
}

/**
 *  @brief Exports a string to a UTF-8 buffer, escaping single quotes.
 *  @param[out] did_fit Populated with 1 if the string is fully exported, 0 if it didn't fit.
 */
char const *export_escaped_unquoted_to_utf8_buffer(char const *cstr, size_t cstr_length, //
                                                   char *buffer, size_t buffer_length,   //
                                                   int *did_fit) {
    char const *const cstr_end = cstr + cstr_length;
    char *const buffer_end = buffer + buffer_length;
    *did_fit = 1;

    while (cstr < cstr_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(cstr, &rune, &rune_length);
        if (rune_length == 1 && buffer + 2 < buffer_end) {
            if (*cstr == '\'') {
                *(buffer++) = '\\';
                *(buffer++) = '\'';
                cstr++;
            }
            else if (*cstr == '\'') {
                *(buffer++) = '\\';
                *(buffer++) = '\'';
                cstr++;
            }
            else { *(buffer++) = *(cstr++); }
        }
        else if (buffer + rune_length < buffer_end) {
            sz_copy(buffer, cstr, rune_length);
            buffer += rune_length;
            cstr += rune_length;
        }
        else {
            *did_fit = 0;
            break;
        }
    }

    return buffer;
}

/**
 *  @brief  Formats an array of strings, similar to the `repr` method of Python lists.
 *          Will output an object that looks like `sz.Str(['item1', 'item2... ])`, potentially
 *          dropping the last few entries.
 */
static PyObject *Strs_repr(Strs *self) {
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    char repr_buffer[1024];
    char *repr_buffer_ptr = &repr_buffer[0];
    char const *const repr_buffer_end = repr_buffer_ptr + 1024;

    // Start of the array
    sz_copy(repr_buffer_ptr, "sz.Strs([", 9);
    repr_buffer_ptr += 9;

    size_t count = Strs_len(self);
    PyObject *parent_string;

    // In the worst case, we must have enough space for `...', ...])`
    // That's extra 11 bytes of content.
    char const *non_fitting_array_tail = "... ])";
    int const non_fitting_array_tail_length = 6;

    // If the whole string doesn't fit, even before the `non_fitting_array_tail` tail,
    // we need to add `, '` separator of 3 bytes.
    for (size_t i = 0; i < count && repr_buffer_ptr + (non_fitting_array_tail_length + 3) < repr_buffer_end; i++) {
        char const *cstr_start = NULL;
        size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);

        if (i > 0) { *(repr_buffer_ptr++) = ',', *(repr_buffer_ptr++) = ' '; }
        *(repr_buffer_ptr++) = '\'';

        int did_fit;
        repr_buffer_ptr = export_escaped_unquoted_to_utf8_buffer(
            cstr_start, cstr_length, repr_buffer_ptr, repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length,
            &did_fit);
        // If it didn't fit, let's put an ellipsis
        if (!did_fit) {
            sz_copy(repr_buffer_ptr, non_fitting_array_tail, non_fitting_array_tail_length);
            repr_buffer_ptr += non_fitting_array_tail_length;
            return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
        }
        else
            *(repr_buffer_ptr++) = '\''; // Close the string
    }

    // Close the array
    *(repr_buffer_ptr++) = ']', *(repr_buffer_ptr++) = ')';
    return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
}

/**
 *  @brief  Array to string conversion method, that concatenates all the strings in the array.
 *          Will output an object that looks like `['item1', 'item2', 'item3']`, containing all
 *          the strings.
 */
static PyObject *Strs_str(Strs *self) {
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    // Aggregate the total length of all the slices and count the number of bytes we need to allocate:
    size_t count = Strs_len(self);
    PyObject *parent_string;
    size_t total_bytes = 2; // opening and closing square brackets
    for (size_t i = 0; i < count; i++) {
        char const *cstr_start = NULL;
        size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);
        total_bytes += cstr_length;
        total_bytes += 2;             // For the single quotes
        if (i != 0) total_bytes += 2; // For the preceding comma and space

        // Count the number of single quotes in the string
        while (cstr_length) {
            char quote = '\'';
            sz_cptr_t next_quote = sz_find_byte(cstr_start, cstr_length, &quote);
            if (next_quote == NULL) break;
            total_bytes++;
            cstr_length -= next_quote - cstr_start;
            cstr_start = next_quote + 1;
        }
    }

    // Now allocate the memory for the concatenated string
    char *const result_buffer = malloc(total_bytes);
    if (!result_buffer) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the concatenated string");
        return NULL;
    }

    // Copy the strings into the result buffer
    char *result_ptr = result_buffer;
    *result_ptr++ = '[';
    for (size_t i = 0; i < count; i++) {
        if (i != 0) {
            *result_ptr++ = ',';
            *result_ptr++ = ' ';
        }
        char const *cstr_start = NULL;
        size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);
        *result_ptr++ = '\'';
        int did_fit;
        result_ptr = export_escaped_unquoted_to_utf8_buffer(cstr_start, cstr_length, result_ptr, total_bytes, &did_fit);
        *result_ptr++ = '\'';
    }

    *result_ptr++ = ']';
    return PyUnicode_FromStringAndSize(result_buffer, total_bytes);
}

static PySequenceMethods Strs_as_sequence = {
    .sq_length = Strs_len,   //
    .sq_item = Strs_getitem, //
    .sq_contains = Strs_in,  //
};

static PyMappingMethods Strs_as_mapping = {
    .mp_length = Strs_len,          //
    .mp_subscript = Strs_subscript, // Is used to implement slices in Python
};

static PyGetSetDef Strs_getsetters[] = {
    // Compatibility with PyArrow
    {"tape", (getter)Strs_get_tape, NULL, "In-place transforms the string representation to match Apache Arrow", NULL},
    {"tape_address", (getter)Strs_get_tape_address, NULL, "Address of the first byte of the first string", NULL},
    {"tape_nbytes", (getter)Strs_get_tape_nbytes, NULL, "Length of the entire tape of strings in bytes", NULL},
    {"offsets_address", (getter)Strs_get_offsets_address, NULL, "Address of the first byte of offsets array", NULL},
    {"offsets_nbytes", (getter)Strs_get_offsets_nbytes, NULL, "Get teh length of offsets array in bytes", NULL},
    {"offsets_are_large", (getter)Strs_get_offsets_are_large, NULL,
     "Checks if 64-bit addressing should be used to convert to Arrow", NULL},
    {NULL} // Sentinel
};

static PyMethodDef Strs_methods[] = {
    {"shuffle", Strs_shuffle, SZ_METHOD_FLAGS, "Shuffle (in-place) the elements of the Strs object."}, //
    {"sort", Strs_sort, SZ_METHOD_FLAGS, "Sort (in-place) the elements of the Strs object."},          //
    {"argsort", Strs_argsort, SZ_METHOD_FLAGS, "Provides the permutation to achieve sorted order."},   //
    {"sample", Strs_sample, SZ_METHOD_FLAGS, "Provides a random sample of a given size."},             //
    // {"to_pylist", Strs_to_pylist, SZ_METHOD_FLAGS, "Exports string-views to a native list of native strings."}, //
    {NULL, NULL, 0, NULL} // Sentinel
};

static PyTypeObject StrsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Strs",
    .tp_doc = "Space-efficient container for large collections of strings and their slices",
    .tp_basicsize = sizeof(Strs),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_methods = Strs_methods,
    .tp_as_sequence = &Strs_as_sequence,
    .tp_as_mapping = &Strs_as_mapping,
    .tp_getset = Strs_getsetters,
    .tp_richcompare = Strs_richcompare,
    .tp_repr = (reprfunc)Strs_repr,
    .tp_str = (reprfunc)Strs_str,
};

#pragma endregion

static void stringzilla_cleanup(PyObject *m) {
    if (temporary_memory.start) free(temporary_memory.start);
    temporary_memory.start = NULL;
    temporary_memory.length = 0;
}

static PyMethodDef stringzilla_methods[] = {
    // Basic `str`, `bytes`, and `bytearray`-like functionality
    {"contains", Str_contains, SZ_METHOD_FLAGS, doc_contains},
    {"count", Str_count, SZ_METHOD_FLAGS, doc_count},
    {"splitlines", Str_splitlines, SZ_METHOD_FLAGS, doc_splitlines},
    {"startswith", Str_startswith, SZ_METHOD_FLAGS, doc_startswith},
    {"endswith", Str_endswith, SZ_METHOD_FLAGS, doc_endswith},
    {"translate", Str_translate, SZ_METHOD_FLAGS, doc_translate},
    {"decode", Str_decode, SZ_METHOD_FLAGS, doc_decode},
    {"equal", Str_like_equal, SZ_METHOD_FLAGS, doc_like_equal},

    // Bidirectional operations
    {"find", Str_find, SZ_METHOD_FLAGS, doc_find},
    {"index", Str_index, SZ_METHOD_FLAGS, doc_index},
    {"partition", Str_partition, SZ_METHOD_FLAGS, doc_partition},
    {"split", Str_split, SZ_METHOD_FLAGS, doc_split},
    {"rfind", Str_rfind, SZ_METHOD_FLAGS, doc_rfind},
    {"rindex", Str_rindex, SZ_METHOD_FLAGS, doc_rindex},
    {"rpartition", Str_rpartition, SZ_METHOD_FLAGS, doc_rpartition},
    {"rsplit", Str_rsplit, SZ_METHOD_FLAGS, doc_rsplit},

    // Edit distance extensions
    {"hamming_distance", Str_hamming_distance, SZ_METHOD_FLAGS, doc_hamming_distance},
    {"hamming_distance_unicode", Str_hamming_distance_unicode, SZ_METHOD_FLAGS, doc_hamming_distance_unicode},
    {"levenshtein_distance", Str_levenshtein_distance, SZ_METHOD_FLAGS, doc_levenshtein_distance},
    {"levenshtein_distance_unicode", Str_levenshtein_distance_unicode, SZ_METHOD_FLAGS,
     doc_levenshtein_distance_unicode},
    {"needleman_wunsch_score", Str_needleman_wunsch_score, SZ_METHOD_FLAGS, doc_needleman_wunsch_score},

    // Character search extensions
    {"find_first_of", Str_find_first_of, SZ_METHOD_FLAGS, doc_find_first_of},
    {"find_last_of", Str_find_last_of, SZ_METHOD_FLAGS, doc_find_last_of},
    {"find_first_not_of", Str_find_first_not_of, SZ_METHOD_FLAGS, doc_find_first_not_of},
    {"find_last_not_of", Str_find_last_not_of, SZ_METHOD_FLAGS, doc_find_last_not_of},
    {"split_byteset", Str_split_byteset, SZ_METHOD_FLAGS, doc_split_byteset},
    {"rsplit_byteset", Str_rsplit_byteset, SZ_METHOD_FLAGS, doc_rsplit_byteset},

    // Lazily evaluated iterators
    {"split_iter", Str_split_iter, SZ_METHOD_FLAGS, doc_split_iter},
    {"rsplit_iter", Str_rsplit_iter, SZ_METHOD_FLAGS, doc_rsplit_iter},
    {"split_byteset_iter", Str_split_byteset_iter, SZ_METHOD_FLAGS, doc_split_byteset_iter},
    {"rsplit_byteset_iter", Str_rsplit_byteset_iter, SZ_METHOD_FLAGS, doc_rsplit_byteset_iter},

    // Dealing with larger-than-memory datasets
    {"offset_within", Str_offset_within, SZ_METHOD_FLAGS, doc_offset_within},
    {"write_to", Str_write_to, SZ_METHOD_FLAGS, doc_write_to},

    // Global unary extensions
    {"hash", Str_like_hash, SZ_METHOD_FLAGS, doc_like_hash},
    {"bytesum", Str_like_bytesum, SZ_METHOD_FLAGS, doc_like_bytesum},

    {NULL, NULL, 0, NULL}};

static PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "SIMD-accelerated string search, sort, hashes, fingerprints, & edit distances",
    -1,
    stringzilla_methods,
    NULL,
    NULL,
    NULL,
    stringzilla_cleanup,
};

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;

    if (PyType_Ready(&StrType) < 0) return NULL;
    if (PyType_Ready(&FileType) < 0) return NULL;
    if (PyType_Ready(&StrsType) < 0) return NULL;
    if (PyType_Ready(&SplitIteratorType) < 0) return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL) return NULL;

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", sz_version_major(), sz_version_minor(), sz_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Define SIMD capabilities
    {
        sz_capability_t caps = sz_capabilities();
        sz_cptr_t caps_str = sz_capabilities_to_string(caps);
        PyModule_AddStringConstant(m, "__capabilities__", caps_str);
    }

    Py_INCREF(&StrType);
    if (PyModule_AddObject(m, "Str", (PyObject *)&StrType) < 0) {
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&FileType);
    if (PyModule_AddObject(m, "File", (PyObject *)&FileType) < 0) {
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&StrsType);
    if (PyModule_AddObject(m, "Strs", (PyObject *)&StrsType) < 0) {
        Py_XDECREF(&StrsType);
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    Py_INCREF(&SplitIteratorType);
    if (PyModule_AddObject(m, "SplitIterator", (PyObject *)&SplitIteratorType) < 0) {
        Py_XDECREF(&SplitIteratorType);
        Py_XDECREF(&StrsType);
        Py_XDECREF(&FileType);
        Py_XDECREF(&StrType);
        Py_XDECREF(m);
        return NULL;
    }

    // Initialize temporary_memory, if needed
    temporary_memory.start = malloc(4096);
    temporary_memory.length = 4096 * (temporary_memory.start != NULL);
    return m;
}
