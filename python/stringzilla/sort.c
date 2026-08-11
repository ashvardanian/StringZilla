/**
 *  @brief Argument-sorting of string sequences.
 *  @file python/stringzilla/sort.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/** @brief Dispatches to the byte-wise or Unicode case-folded argsort backend. */
sz_status_t Strs_run_argsort_(sz_bool_t uncased, sz_sequence_t const *sequence, sz_sorted_idx_t *order, sz_size_t top,
                              sz_bool_t reverse) {
    return uncased ? sz_sequence_argsort_uncased(sequence, NULL, order, top, reverse)
                   : sz_sequence_argsort(sequence, NULL, order, top, reverse);
}

char const doc_argsort[] =                                                                         //
    "argsort(*, reverse=False, uncased=False, top=None, out=None) -> tuple[int, ...] | buffer\n"   //
    "\n"                                                                                           //
    "Return the stable permutation of indices that sorts the Strs.\n"                              //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  reverse (bool, optional): Sort in descending order. Defaults to False.\n"                   //
    "  uncased (bool, optional): Order by Unicode case-folding. Defaults to False.\n"              //
    "  top (int, optional): Keep only the `top` leading indices. Defaults to None (all).\n"        //
    "  out (buffer, optional): Writable, C-contiguous buffer of pointer-width unsigned integers "  //
    "(e.g. numpy.uintp) to receive the indices with zero allocation. "                             //
    "Defaults to None.\n"                                                                          //
    "Returns:\n"                                                                                   //
    "  tuple[int, ...]: The sorting permutation, or `out` itself when an `out` buffer is given.\n" //
    "Example:\n"                                                                                   //
    "  >>> sz.Strs(['banana', 'apple', 'cherry']).argsort()\n"                                     //
    "  (1, 0, 2)";

/**
 *  @brief Returns the tuple of indices permuting a `Strs` object into sorted order.
 *         With `top=k`, returns just the `k` leading indices (top-K).
 */
PyObject *Strs_argsort(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                       PyObject *args_names_tuple) {

    sz_bool_t reverse = sz_false_k, uncased = sz_false_k;
    sz_size_t top = 0;
    PyObject *out_obj = NULL;
    if (positional_args_count != 0) {
        PyErr_SetString(PyExc_TypeError, "argsort() takes no positional arguments");
        return NULL;
    }

    PyObject *reverse_obj = NULL, *uncased_obj = NULL, *top_obj = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *value = args[positional_args_count + i];
        if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0) { reverse_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "uncased") == 0) { uncased_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "top") == 0) { top_obj = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "out") == 0) { out_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "argsort() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (reverse_obj) {
        if (!PyBool_Check(reverse_obj)) {
            PyErr_SetString(PyExc_TypeError, "argsort(): reverse must be a bool");
            return NULL;
        }
        reverse = (sz_bool_t)(PyObject_IsTrue(reverse_obj) != 0);
    }
    if (uncased_obj) {
        if (!PyBool_Check(uncased_obj)) {
            PyErr_SetString(PyExc_TypeError, "argsort(): uncased must be a bool");
            return NULL;
        }
        uncased = (sz_bool_t)(PyObject_IsTrue(uncased_obj) != 0);
    }
    if (top_obj && top_obj != Py_None) {
        if (!PyLong_Check(top_obj)) {
            PyErr_SetString(PyExc_TypeError, "argsort(): top must be an int or None");
            return NULL;
        }
        Py_ssize_t top_value = PyLong_AsSsize_t(top_obj);
        if (top_value == -1 && PyErr_Occurred()) return NULL;
        if (top_value < 0) {
            PyErr_SetString(PyExc_ValueError, "argsort(): top must be non-negative");
            return NULL;
        }
        top = (sz_size_t)top_value;
    }

    sz_size_t const count = Strs_len(self);
    // With `top` set, only the leading `top` indices are ordered, so we expose just those.
    sz_size_t const result_count = (top != 0 && top < count) ? top : count;

    // Optionally bind a caller-provided output buffer of `sz_sorted_idx_t` for a zero-allocation path.
    Py_buffer out_view;
    sz_bool_t have_out = sz_false_k;
    if (out_obj && out_obj != Py_None) {
        // PyBUF_CONTIG = writable + C-contiguous; rejects strided/read-only targets up front.
        if (PyObject_GetBuffer(out_obj, &out_view, PyBUF_CONTIG) != 0) return NULL;
        if (out_view.itemsize != (Py_ssize_t)sizeof(sz_sorted_idx_t)) {
            PyErr_Format(PyExc_TypeError,
                         "out must be a contiguous buffer of %zu-byte unsigned integers (e.g. numpy.uintp)",
                         sizeof(sz_sorted_idx_t));
            PyBuffer_Release(&out_view);
            return NULL;
        }
        if ((sz_size_t)(out_view.len / out_view.itemsize) < result_count) {
            PyErr_Format(PyExc_ValueError, "out buffer holds %zd indices, need %zu",
                         (Py_ssize_t)(out_view.len / out_view.itemsize), result_count);
            PyBuffer_Release(&out_view);
            return NULL;
        }
        have_out = sz_true_k;
    }

    // The backend writes a full permutation of `[0, count)` even for top-k, so the `order` buffer must
    // hold `count` indices. For a full sort the result *is* that permutation, so we can sort straight into
    // the caller's buffer; for top-k we sort into a scratch allocation and copy out only the leading `top`,
    // never writing past `result_count` of the caller's buffer.
    sz_bool_t const sort_into_out = have_out && result_count == count;
    sz_sorted_idx_t *order = sort_into_out ? (sz_sorted_idx_t *)out_view.buf
                                           : (sz_sorted_idx_t *)malloc(sizeof(sz_sorted_idx_t) * count);
    if (!order && count) {
        if (have_out) PyBuffer_Release(&out_view);
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the sorting operation");
        return NULL;
    }

    // Call our sorting algorithm (`reverse` and `uncased` are handled natively).
    sz_sequence_t sequence;
    sz_fill(&sequence, sizeof(sequence), 0);
    sequence.count = count;
    sequence.handle = self;
    sequence.get_start = Strs_get_start_;
    sequence.get_length = Strs_get_length_;
    sz_status_t status = Strs_run_argsort_(uncased, &sequence, order, top, reverse);
    sz_unused_(status);

    // Caller buffer path: indices are already in (or now copied into) `out`; hand the buffer back.
    if (have_out) {
        if (!sort_into_out) {
            sz_copy((sz_ptr_t)out_view.buf, (sz_cptr_t)order, result_count * sizeof(sz_sorted_idx_t));
            free(order);
        }
        PyBuffer_Release(&out_view);
        Py_INCREF(out_obj);
        return out_obj;
    }

    // No caller buffer: materialize a tuple of Python ints.
    PyObject *tuple = PyTuple_New(result_count);
    if (!tuple) {
        free(order);
        return NULL;
    }
    for (sz_size_t i = 0; i < result_count; ++i) {
        PyObject *index = PyLong_FromSize_t(order[i]);
        if (!index) {
            free(order);
            Py_DECREF(tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(tuple, i, index);
    }
    free(order);
    return tuple;
}
