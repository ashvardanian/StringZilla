/**
 *  @brief Set intersections over string sequences.
 *  @file python/stringzilla/intersect.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

char const doc_Strs_intersect[] =                                                                            //
    "intersect(other, *, seed=0) -> tuple[tuple[int, ...], tuple[int, ...]]\n"                               //
    "\n"                                                                                                     //
    "Return the positions of strings present in both collections.\n"                                         //
    "Each distinct shared value is matched exactly once, even if either side has duplicates.\n"              //
    "\n"                                                                                                     //
    "Args:\n"                                                                                                //
    "  other (Strs): The collection to intersect with.\n"                                                    //
    "  seed (int, optional): Seed for the hash table to avoid attacks. Defaults to 0.\n"                     //
    "Returns:\n"                                                                                             //
    "  tuple[tuple[int, ...], tuple[int, ...]]: Parallel position tuples - `result[0][i]` in\n"              //
    "  this collection and `result[1][i]` in `other` point to equal strings.\n"                              //
    "Example:\n"                                                                                             //
    "  >>> ours, theirs = sz.Strs(['banana', 'apple', 'cherry']).intersect(sz.Strs(['cherry', 'banana']))\n" //
    "  >>> sorted(ours)\n"                                                                                   //
    "  [0, 2]";

/**
 *  @brief Returns the positions of strings shared between two `Strs` objects.
 *         Duplicates within either side are tolerated: each distinct shared value is matched exactly once.
 */
PyObject *Strs_intersect(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    if (positional_args_count != 1) {
        PyErr_SetString(PyExc_TypeError, "intersect() takes exactly one positional argument: the other Strs");
        return NULL;
    }
    PyObject *other_obj = args[0];
    if (!PyObject_TypeCheck(other_obj, &StrsType)) {
        PyErr_SetString(PyExc_TypeError, "intersect(): the other collection must be a Strs");
        return NULL;
    }
    Strs *other = (Strs *)other_obj;

    sz_u64_t seed = 0;
    PyObject *seed_obj = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *value = args[positional_args_count + i];
        if (PyUnicode_CompareWithASCIIString(key, "seed") == 0) { seed_obj = value; }
        else {
            PyErr_Format(PyExc_TypeError, "intersect() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "intersect(): seed must be an int");
            return NULL;
        }
        seed = PyLong_AsUnsignedLongLong(seed_obj);
        if (PyErr_Occurred()) return NULL; // Reject negative / oversized seed
    }

    sz_size_t const first_count = Strs_len(self);
    sz_size_t const second_count = Strs_len(other);
    sz_size_t const max_size = sz_min_of_two(first_count, second_count);
    if (max_size == 0) return Py_BuildValue("(()())");

    // Both outputs must fit `min(first_count, second_count)` indices; one allocation covers both halves.
    sz_sorted_idx_t *positions = (sz_sorted_idx_t *)malloc(2 * max_size * sizeof(sz_sorted_idx_t));
    if (!positions) {
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the intersection positions");
        return NULL;
    }

    sz_sequence_t first_sequence, second_sequence;
    sz_fill(&first_sequence, sizeof(first_sequence), 0);
    first_sequence.count = first_count;
    first_sequence.handle = self;
    first_sequence.get_start = Strs_get_start_;
    first_sequence.get_length = Strs_get_length_;
    sz_fill(&second_sequence, sizeof(second_sequence), 0);
    second_sequence.count = second_count;
    second_sequence.handle = other;
    second_sequence.get_start = Strs_get_start_;
    second_sequence.get_length = Strs_get_length_;

    sz_size_t intersection_size = 0;
    sz_status_t status = sz_sequence_intersect(&first_sequence, &second_sequence, NULL, seed, &intersection_size,
                                               positions, positions + max_size);
    if (status != sz_success_k) {
        free(positions);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for the intersection operation");
        return NULL;
    }

    PyObject *first_tuple = PyTuple_New(intersection_size);
    PyObject *second_tuple = first_tuple ? PyTuple_New(intersection_size) : NULL;
    if (!first_tuple || !second_tuple) {
        free(positions);
        Py_XDECREF(first_tuple);
        return NULL;
    }
    for (sz_size_t i = 0; i < intersection_size; ++i) {
        PyObject *first_index = PyLong_FromSize_t(positions[i]);
        PyObject *second_index = first_index ? PyLong_FromSize_t(positions[max_size + i]) : NULL;
        if (!first_index || !second_index) {
            free(positions);
            Py_XDECREF(first_index);
            Py_DECREF(first_tuple);
            Py_DECREF(second_tuple);
            return NULL;
        }
        PyTuple_SET_ITEM(first_tuple, i, first_index);
        PyTuple_SET_ITEM(second_tuple, i, second_index);
    }
    free(positions);

    PyObject *result = PyTuple_Pack(2, first_tuple, second_tuple);
    Py_DECREF(first_tuple);
    Py_DECREF(second_tuple);
    return result;
}
