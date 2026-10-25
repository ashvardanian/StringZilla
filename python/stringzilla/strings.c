/**
 *  @file python/stringzilla/strings.c
 *  @author Ash Vardanian
 *  @date October 11, 2024
 *  @brief The @c Strings collection, one tape that is borrowed or owned, and every verb over it.
 *
 *  Every @c Strings verb here is one call over the whole collection into a new tape, and the
 *  dumpers escape each column once, then join the framed rows with one more call into a single
 *  @c bytes object.
 */
#include "stringzilla.h"

/**
 *  @brief Arrow C Data Interface structure for an array schema.
 *  @see https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
 */
struct arrow_schema_t {
    char const *format;
    char const *name;
    char const *metadata;
    int64_t flags;
    int64_t n_children;
    struct arrow_schema_t **children;
    struct arrow_schema_t *dictionary;
    void (*release)(struct arrow_schema_t *);
    void *private_data;
};

/**
 *  @brief Arrow C Data Interface structure for an array content.
 *  @see https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
 */
struct arrow_array_t {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    void const **buffers;
    struct arrow_array_t **children;
    struct arrow_array_t *dictionary;
    void (*release)(struct arrow_array_t *);
    void *private_data;
};

sz_bool_t sz_py_export_strings_as_view(PyObject *object, sz_strings_view_t *strings) {
    if (!strings || !PyObject_TypeCheck(object, &StringsType)) return sz_false_k;
    sz_strings_from_span(&((Strings *)object)->tape, strings);
    return sz_true_k;
}

sz_bool_t sz_py_export_strings_as_u32tape(PyObject *object, sz_cptr_t *data, sz_u32_t const **offsets,
                                          sz_size_t *count) {
    if (!data || !offsets || !count || !PyObject_TypeCheck(object, &StringsType)) return sz_false_k;
    sz_strings_span_t const *tape = &((Strings *)object)->tape;
    if (tape->layout != sz_strings_u32_tape_k) return sz_false_k;
    *data = tape->data.u32_tape.data, *offsets = tape->data.u32_tape.offsets, *count = tape->count;
    return sz_true_k;
}

sz_bool_t sz_py_export_strings_as_u64tape(PyObject *object, sz_cptr_t *data, sz_u64_t const **offsets,
                                          sz_size_t *count) {
    if (!data || !offsets || !count || !PyObject_TypeCheck(object, &StringsType)) return sz_false_k;
    sz_strings_span_t const *tape = &((Strings *)object)->tape;
    if (tape->layout != sz_strings_u64_tape_k) return sz_false_k;
    *data = tape->data.u64_tape.data, *offsets = tape->data.u64_tape.offsets, *count = tape->count;
    return sz_true_k;
}

/** Frees the blocks @p strings owns, or drops its reference to the object it borrows them from. */
static void Strings_release_(Strings *strings) {
    if (!strings->parent) sz_strings_span_free(&strings->tape, &strings->allocator, NULL);
    Py_CLEAR(strings->parent);
    strings->tape = (sz_strings_span_t) {0};
}

/** The object holding the blocks of @p strings: the one it borrows them from, or itself. */
static PyObject *Strings_owner_(Strings *strings) { return strings->parent ? strings->parent : (PyObject *)strings; }

sz_bool_t sz_py_replace_strings_allocator(PyObject *object, sz_allocator_t *allocator) {
    if (!allocator || !PyObject_TypeCheck(object, &StringsType)) return sz_false_k;
    Strings *strings = (Strings *)object;
    if (!strings->parent && sz_allocator_equal(&strings->allocator, allocator)) return sz_true_k;
    sz_strings_view_t view;
    sz_strings_from_span(&strings->tape, &view);
    sz_strings_span_t tape = {0};
    if (sz_strings_to_tape_best(&view, NULL, view.count, allocator, &tape, sz_cap_cpus_k, NULL) != sz_success_k)
        return sz_false_k;
    Strings_release_(strings);
    strings->tape = tape, strings->allocator = *allocator;
    return sz_true_k;
}

PyObject *Strings_from_tape_(sz_status_t status, sz_strings_span_t *tape, sz_allocator_t *allocator,
                             char const *context) {
    Strings *result = NULL;
    if (status != sz_success_k) sz_py_raise_status(status, context);
    else result = Strings_alloc_();
    if (!result) {
        sz_strings_span_free(tape, allocator, NULL);
        return NULL;
    }
    result->tape = *tape, result->allocator = *allocator;
    return (PyObject *)result;
}

PyObject *Strings_gather_(sz_strings_view_t const *strings, sz_size_t const *order, sz_size_t count,
                          char const *context) {
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_to_tape_best(strings, order, count, &allocator, &tape, sz_cap_cpus_k, NULL);
    return Strings_from_tape_(status, &tape, &allocator, context);
}

static PyObject *Strings_get_offsets_are_large(Strings *self, void *closure) {
    return PyBool_FromLong(self->tape.layout == sz_strings_u64_tape_k);
}

static PyObject *Strings_get_tape_address(Strings *self, void *closure) {
    sz_strings_span_t const *tape = &self->tape;
    return PyLong_FromSize_t(
        (sz_size_t)(tape->layout == sz_strings_u64_tape_k ? tape->data.u64_tape.data : tape->data.u32_tape.data));
}

static PyObject *Strings_get_offsets_address(Strings *self, void *closure) {
    sz_strings_span_t const *tape = &self->tape;
    return PyLong_FromSize_t(tape->layout == sz_strings_u64_tape_k ? (sz_size_t)tape->data.u64_tape.offsets
                                                                   : (sz_size_t)tape->data.u32_tape.offsets);
}

static PyObject *Strings_get_tape_nbytes(Strings *self, void *closure) {
    return PyLong_FromSize_t(sz_py_tape_payload_(&self->tape));
}

static PyObject *Strings_get_offsets_nbytes(Strings *self, void *closure) {
    sz_size_t const offset_size = self->tape.layout == sz_strings_u64_tape_k ? sizeof(sz_u64_t) : sizeof(sz_u32_t);
    return PyLong_FromSize_t((self->tape.count + 1) * offset_size);
}

static Py_ssize_t Strings_length(Strings *self) { return (Py_ssize_t)self->tape.count; }

static PyObject *Strings_getitem(Strings *self, Py_ssize_t i) {
    Py_ssize_t const count = Strings_length(self);
    if (i < 0) i += count;
    if (i < 0 || i >= count) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }
    String *result = String_alloc_();
    if (!result) return PyErr_NoMemory();
    result->memory = sz_py_tape_string_(&self->tape, (sz_size_t)i);
    result->parent = Strings_owner_(self);
    Py_INCREF(result->parent);
    return (PyObject *)result;
}

/** Indexes, borrows a slice with a step of one, and gathers any other slice into a new tape. */
static PyObject *Strings_subscript(Strings *self, PyObject *key) {

    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        if (index == -1 && PyErr_Occurred()) return NULL; // Propagate OverflowError instead of leaking it
        return Strings_getitem(self, index);
    }

    if (!PySlice_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Strings indices must be integers or slices");
        return NULL;
    }

    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
    Py_ssize_t const count = PySlice_AdjustIndices(Strings_length(self), &start, &stop, step);

    if (step != 1) {
        sz_size_t *order = PyMem_New(sz_size_t, count);
        if (!order) return PyErr_NoMemory();
        for (Py_ssize_t i = 0; i != count; ++i) order[i] = (sz_size_t)(start + i * step);
        sz_strings_view_t strings;
        sz_strings_from_span(&self->tape, &strings);
        PyObject *result = Strings_gather_(&strings, order, (sz_size_t)count, "slicing");
        PyMem_Free(order);
        return result;
    }

    Strings *result = Strings_alloc_();
    if (!result) return NULL;
    result->tape = self->tape;
    result->tape.count = (sz_size_t)count;
    if (self->tape.layout == sz_strings_u64_tape_k) result->tape.data.u64_tape.offsets += start;
    else result->tape.data.u32_tape.offsets += start;
    result->parent = Strings_owner_(self);
    Py_INCREF(result->parent);
    return (PyObject *)result;
}

/**
 *  @brief Will be called by the @c PySequence_Contains to check the presence of a string in array.
 *  @return 1 if the string is present, 0 if it is not, -1 in case of error.
 *  @see Docs: https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
 */
static int Strings_in(Strings *self, PyObject *needle_object) {

    // Validate and convert `needle`
    sz_string_view_t needle;
    if (!sz_py_export_string_like(needle_object, &needle.start, &needle.length)) {
        wrap_current_exception("The needle argument must be string-like");
        return -1;
    }

    // Time for a full-scan
    for (sz_size_t i = 0; i < self->tape.count; ++i) {
        sz_string_view_t const string = sz_py_tape_string_(&self->tape, i);
        if (string.length != needle.length) continue;
        sz_bool_t equal = sz_false_k;
        sz_status_t const status = sz_equal_best(string.start, needle.start, needle.length, &equal, sz_cap_cpus_k,
                                                 NULL);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "in");
            return -1;
        }
        if (equal == sz_true_k) return 1;
    }

    return 0;
}

static PyObject *Strings_richcompare(PyObject *self, PyObject *other, int op) {

    Strings *a = (Strings *)self;
    Py_ssize_t a_length = Strings_length(a);

    // If the other object is also a Strings, we can compare them much faster,
    // avoiding the CPython API entirely
    if (PyObject_TypeCheck(other, &StringsType)) {
        Strings *b = (Strings *)other;

        // Check if lengths are equal
        Py_ssize_t b_length = Strings_length(b);
        if (a_length != b_length) {
            if (op == Py_EQ) { Py_RETURN_FALSE; }
            if (op == Py_NE) { Py_RETURN_TRUE; }
        }

        // Check each item for equality
        Py_ssize_t min_length = sz_min_of_two(a_length, b_length);
        for (Py_ssize_t i = 0; i < min_length; i++) {
            sz_string_view_t const ai = sz_py_tape_string_(&a->tape, (sz_size_t)i);
            sz_string_view_t const bi = sz_py_tape_string_(&b->tape, (sz_size_t)i);

            // When dealing with arrays, early exists make sense only in some cases
            sz_ordering_t ordering = sz_equal_k;
            sz_status_t const status = sz_order_best(ai.start, ai.length, bi.start, bi.length, &ordering, sz_cap_cpus_k,
                                                     NULL);
            if (status != sz_success_k) {
                sz_py_raise_status(status, "Strings comparison");
                return NULL;
            }
            int const order = (int)ordering;
            switch (op) {
            case Py_LT:
            case Py_LE:
                if (order > 0) { Py_RETURN_FALSE; }
                break;
            case Py_EQ:
                if (order != 0) { Py_RETURN_FALSE; }
                break;
            case Py_NE:
                if (order == 0) { Py_RETURN_TRUE; }
                break;
            case Py_GT:
            case Py_GE:
                if (order < 0) { Py_RETURN_FALSE; }
                break;
            default: break;
            }
        }

        // Prefixes are identical, compare lengths
        switch (op) {
        case Py_LT: return PyBool_FromLong(a_length < b_length);
        case Py_LE: return PyBool_FromLong(a_length <= b_length);
        case Py_EQ: return PyBool_FromLong(a_length == b_length);
        case Py_NE: return PyBool_FromLong(a_length != b_length);
        case Py_GT: return PyBool_FromLong(a_length > b_length);
        case Py_GE: return PyBool_FromLong(a_length >= b_length);
        default: Py_RETURN_NOTIMPLEMENTED;
        }
    }

    // The second argument is a sequence, but not a `Strings` object,
    // so we need to iterate through it.
    PyObject *other_iterator = PyObject_GetIter(other);
    if (!other_iterator) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "The second argument is not iterable");
        return NULL;
    }

    // We may not even know the length of the second sequence, so
    // let's just iterate as far as we can.
    Py_ssize_t i = 0;
    PyObject *other_item;
    for (; (other_item = PyIter_Next(other_iterator)); ++i) {
        // Check if the second array is longer than the first
        if (a_length <= i) {
            Py_DECREF(other_item);
            Py_DECREF(other_iterator);
            switch (op) {
            case Py_LT: Py_RETURN_TRUE;
            case Py_LE: Py_RETURN_TRUE;
            case Py_EQ: Py_RETURN_FALSE;
            case Py_NE: Py_RETURN_TRUE;
            case Py_GT: Py_RETURN_FALSE;
            case Py_GE: Py_RETURN_FALSE;
            default: Py_RETURN_NOTIMPLEMENTED;
            }
        }

        // Try unpacking the element from the second sequence
        sz_string_view_t bi;
        if (!sz_py_export_string_like(other_item, &bi.start, &bi.length)) {
            Py_DECREF(other_item);
            Py_DECREF(other_iterator);
            wrap_current_exception("The second container must contain string-like objects");
            return NULL;
        }

        // Both sequences aren't exhausted yet
        sz_string_view_t const ai = sz_py_tape_string_(&a->tape, (sz_size_t)i);

        // When dealing with arrays, early exists make sense only in some cases
        sz_ordering_t ordering = sz_equal_k;
        sz_status_t const status = sz_order_best(ai.start, ai.length, bi.start, bi.length, &ordering, sz_cap_cpus_k,
                                                 NULL);
        if (status != sz_success_k) {
            Py_DECREF(other_item);
            Py_DECREF(other_iterator);
            sz_py_raise_status(status, "Strings comparison");
            return NULL;
        }
        int const order = (int)ordering;
        switch (op) {
        case Py_LT:
        case Py_LE:
            if (order > 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iterator);
                Py_RETURN_FALSE;
            }
            break;
        case Py_EQ:
            if (order != 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iterator);
                Py_RETURN_FALSE;
            }
            break;
        case Py_NE:
            if (order == 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iterator);
                Py_RETURN_TRUE;
            }
            break;
        case Py_GT:
        case Py_GE:
            if (order < 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iterator);
                Py_RETURN_FALSE;
            }
            break;
        default: break;
        }
    }

    // The prefixes are equal and the second sequence is exhausted, but the first one may not be
    switch (op) {
    case Py_LT: return PyBool_FromLong(i < a_length);
    case Py_LE: Py_RETURN_TRUE;
    case Py_EQ: return PyBool_FromLong(i == a_length);
    case Py_NE: return PyBool_FromLong(i != a_length);
    case Py_GT: Py_RETURN_FALSE;
    case Py_GE: return PyBool_FromLong(i == a_length);
    default: Py_RETURN_NOTIMPLEMENTED;
    }
}

static PyObject *Strings_shuffled(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                  PyObject *args_names_tuple) {

    // Check for positional arguments
    PyObject *seed_object = positional_args_count == 1 ? args[0] : NULL;
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "shuffle() takes at most 1 positional argument");
        return NULL;
    }

    // Check for keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_object) { seed_object = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    unsigned int seed = (unsigned int)time(NULL);
    if (seed_object) {
        if (!PyLong_Check(seed_object)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_object);
        if (PyErr_Occurred()) return NULL; // Reject negative / oversized seed
    }

    sz_size_t const count = self->tape.count;
    sz_size_t *order = PyMem_New(sz_size_t, count);
    if (!order) return PyErr_NoMemory();
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // Fisher-Yates Shuffle Algorithm
    srand(seed);
    for (sz_size_t i = count; i > 1; --i) {
        sz_size_t const j = (sz_size_t)rand() % i;
        sz_size_t const temporary = order[i - 1];
        order[i - 1] = order[j];
        order[j] = temporary;
    }

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    PyObject *result = Strings_gather_(&strings, order, count, "shuffled()");
    PyMem_Free(order);
    return result;
}

static char const doc_sorted[] =                                                                          //
    "sorted(*, reverse=False, uncased=False, top=None, capabilities=None) -> Strings\n"                   //
    "\n"                                                                                                  //
    "Return a new, stably sorted Strings, like the built-in `sorted`; the original is unchanged.\n"       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  reverse (bool, optional): Sort in descending order. Defaults to False.\n"                          //
    "  uncased (bool, optional): Order by Unicode case-folding. Defaults to False.\n"                     //
    "  top (int, optional): Keep only the `top` smallest elements, or largest if reversed. "              //
    "Defaults to None, keeping all.\n"                                                                    //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strings: A new, sorted collection on a tape of its own.\n"                                         //
    "Example:\n"                                                                                          //
    "  >>> list(map(str, sz.Strings(['banana', 'apple', 'cherry']).sorted()))\n"                          //
    "  ['apple', 'banana', 'cherry']";

static PyObject *Strings_sorted(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    // Parse the keyword-only options: `reverse` and `uncased` as bools, `top` as a count or None.
    sz_bool_t reverse = sz_false_k, uncased = sz_false_k;
    sz_size_t top = 0;
    if (positional_args_count != 0) {
        PyErr_SetString(PyExc_TypeError, "sorted() takes no positional arguments");
        return NULL;
    }

    PyObject *reverse_object = NULL, *uncased_object = NULL, *top_object = NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t i = 0; i < args_names_count; ++i) {
        PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
        PyObject *value = args[positional_args_count + i];
        if (PyUnicode_CompareWithASCIIString(key, "reverse") == 0) { reverse_object = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "uncased") == 0) { uncased_object = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "top") == 0) { top_object = value; }
        else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0) { capabilities_object = value; }
        else {
            PyErr_Format(PyExc_TypeError, "sorted() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    if (reverse_object) {
        if (!PyBool_Check(reverse_object)) {
            PyErr_SetString(PyExc_TypeError, "sorted(): reverse must be a bool");
            return NULL;
        }
        reverse = (sz_bool_t)(PyObject_IsTrue(reverse_object) != 0);
    }
    if (uncased_object) {
        if (!PyBool_Check(uncased_object)) {
            PyErr_SetString(PyExc_TypeError, "sorted(): uncased must be a bool");
            return NULL;
        }
        uncased = (sz_bool_t)(PyObject_IsTrue(uncased_object) != 0);
    }
    if (top_object && top_object != Py_None) {
        if (!PyLong_Check(top_object)) {
            PyErr_SetString(PyExc_TypeError, "sorted(): top must be an int or None");
            return NULL;
        }
        Py_ssize_t top_value = PyLong_AsSsize_t(top_object);
        if (top_value == -1 && PyErr_Occurred()) return NULL;
        if (top_value < 0) {
            PyErr_SetString(PyExc_ValueError, "sorted(): top must be non-negative");
            return NULL;
        }
        top = (sz_size_t)top_value;
    }

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_size_t *order = PyMem_New(sz_size_t, strings.count);
    if (!order) return PyErr_NoMemory();

    // Skip an empty collection: the kernels report its zero-byte scratch as a failed allocation.
    sz_status_t status = strings.count ? Strings_run_argsort_(uncased, &strings, order, top, reverse, capabilities)
                                       : sz_success_k;
    if (status != sz_success_k) {
        PyMem_Free(order);
        sz_py_raise_status(status, "sorted()");
        return NULL;
    }

    // With `top` set, only the leading `top` elements are ordered, so the result keeps just those.
    sz_size_t const result_count = (top != 0 && top < strings.count) ? top : strings.count;
    PyObject *result = Strings_gather_(&strings, order, result_count, "sorted()");
    PyMem_Free(order);
    return result;
}

static PyObject *Strings_sample(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    PyObject *sample_size_object = NULL;
    PyObject *seed_object = NULL;

    // Check for positional arguments
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "sample() takes 1 positional argument and 1 keyword argument");
        return NULL;
    }
    else if (positional_args_count == 1) { sample_size_object = args[0]; }

    // Parse keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_object) { seed_object = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Translate the seed and the sample size to C types
    sz_size_t sample_size = 0;
    if (sample_size_object) {
        if (!PyLong_Check(sample_size_object)) {
            PyErr_SetString(PyExc_TypeError, "The sample size must be an integer");
            return NULL;
        }
        sample_size = PyLong_AsSize_t(sample_size_object);
        if (sample_size == (sz_size_t)-1 && PyErr_Occurred()) return NULL; // Reject negative / oversized
    }
    unsigned int seed = (unsigned int)time(NULL); // Default seed
    if (seed_object) {
        if (!PyLong_Check(seed_object)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_object);
        if (PyErr_Occurred()) return NULL; // Reject negative / oversized seed
    }

    // An empty source has nothing to draw from, whatever the requested size
    sz_size_t const count = self->tape.count;
    if (!count) sample_size = 0;
    sz_size_t *order = PyMem_New(sz_size_t, sample_size);
    if (!order) return PyErr_NoMemory();

    // Randomly sample the strings (with replacement)
    srand(seed);
    for (sz_size_t i = 0; i != sample_size; ++i) order[i] = (sz_size_t)rand() % count;

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    PyObject *result = Strings_gather_(&strings, order, sample_size, "sample()");
    PyMem_Free(order);
    return result;
}

static PyObject *Strings_matched(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                 PyObject *args_names_tuple) {
    if (positional_args_count != 1 || (args_names_tuple && PyTuple_GET_SIZE(args_names_tuple))) {
        PyErr_SetString(PyExc_TypeError, "matched() takes exactly one positional argument");
        return NULL;
    }
    Py_buffer matches_view;
    if (sz_py_export_contiguous_buffer(args[0], "matches", PyBUF_CONTIG_RO, (Py_ssize_t)sizeof(sz_size_t), 0,
                                       &matches_view) != 0)
        return NULL;
    if (matches_view.ndim != 2 || matches_view.shape[1] != 4) {
        PyErr_SetString(PyExc_ValueError, "matched(): matches must be shaped (count, 4)");
        PyBuffer_Release(&matches_view);
        return NULL;
    }

    // A row past its haystack would read beyond the tape, so every row is checked first.
    sz_match_t const *const matches = (sz_match_t const *)matches_view.buf;
    sz_size_t const count = (sz_size_t)matches_view.shape[0];
    for (sz_size_t index = 0; index != count; ++index) {
        sz_match_t const match = matches[index];
        sz_size_t const length = match.haystack_index < self->tape.count
                                     ? sz_py_tape_string_(&self->tape, match.haystack_index).length
                                     : 0;
        if (match.haystack_index >= self->tape.count || match.byte_offset > length ||
            match.byte_length > length - match.byte_offset) {
            PyErr_Format(PyExc_IndexError, "matched(): match %zu falls outside its haystack", (size_t)index);
            PyBuffer_Release(&matches_view);
            return NULL;
        }
    }

    sz_string_view_t *views = PyMem_New(sz_string_view_t, count);
    if (!views) {
        PyBuffer_Release(&matches_view);
        return PyErr_NoMemory();
    }
    sz_strings_view_t haystacks, spans;
    sz_strings_from_span(&self->tape, &haystacks);
    sz_strings_from_matches(&haystacks, matches, count, views, &spans);
    PyBuffer_Release(&matches_view);
    PyObject *result = Strings_gather_(&spans, NULL, count, "matched()");
    PyMem_Free(views);
    return result;
}

/**
 *  @brief Exports a string to a UTF-8 buffer, escaping single quotes.
 *  @param[in] cstr The input string to export.
 *  @param[in] cstr_length The length of the input string.
 *  @param[out] buffer The output buffer to write to.
 *  @param[in] buffer_length The size of the output buffer.
 *  @param[out] did_fit Set to 1 if the string is fully exported, 0 if it didn't fit, or -1 if it is
 *      not valid UTF-8.
 *  @return Pointer to the end of the written data in the buffer, or to the position of the error.
 */
static sz_cptr_t export_escaped_unquoted_to_utf8_buffer(sz_cptr_t cstr, sz_size_t cstr_length,    //
                                                        sz_ptr_t buffer, sz_size_t buffer_length, //
                                                        int *did_fit) {
    sz_cptr_t const cstr_end = cstr + cstr_length;
    sz_ptr_t buffer_ptr = buffer;
    *did_fit = 1;

    // Validate UTF-8 first
    if (sz_utf8_find_malformed(cstr, cstr_length) != STRINGZILLA_NULL_CHAR) {
        *did_fit = -1; // Signal UTF-8 error
        return buffer_ptr;
    }

    // First pass: calculate required buffer size (input already validated)
    sz_size_t required_bytes = 2; // Opening and closing quotes
    sz_cptr_t scan_ptr = cstr;
    while (scan_ptr < cstr_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(scan_ptr, cstr_end, &rune);

        if (rune_length == 1 && *scan_ptr == '\'') { required_bytes += 2; } // Escaped quote: \'
        else { required_bytes += rune_length; }                             // Normal rune
        scan_ptr += rune_length;
    }

    // Check if we have enough buffer space
    if (required_bytes > buffer_length) {
        *did_fit = 0;
        return buffer_ptr;
    }

    // Second pass: actually write to buffer
    *(buffer_ptr++) = '\''; // Opening quote

    while (cstr < cstr_end) {
        sz_rune_t rune;
        sz_rune_length_t rune_length = sz_rune_decode(cstr, cstr_end, &rune);

        if (rune_length == 1 && *cstr == '\'') {
            *(buffer_ptr++) = '\\';
            *(buffer_ptr++) = '\'';
        }
        else {
            // At most 4 bytes, too few to be worth a dispatched copy
            for (sz_size_t byte_index = 0; byte_index != rune_length; ++byte_index) *(buffer_ptr++) = cstr[byte_index];
        }
        cstr += rune_length;
    }

    *(buffer_ptr++) = '\''; // Closing quote
    return buffer_ptr;
}

/**
 *  @brief Exports a binary string to a buffer in Python bytes representation (b'\\x..').
 *  @param[in] data The binary data to export.
 *  @param[in] data_length The length of the binary data.
 *  @param[out] buffer The output buffer to write to.
 *  @param[in] buffer_length The size of the output buffer.
 *  @param[out] did_fit Populated with 1 if the data is fully exported, 0 if it didn't fit.
 *  @return Pointer to the end of the written data in the buffer.
 */
static sz_cptr_t export_escaped_unquoted_to_binary_buffer(sz_cptr_t data, sz_size_t data_length,    //
                                                          sz_ptr_t buffer, sz_size_t buffer_length, //
                                                          int *did_fit) {
    sz_ptr_t buffer_ptr = buffer;
    *did_fit = 1;

    // First pass: calculate required buffer size
    // Format: b'\x00\x01...'  → 3 bytes prefix + 4 bytes per byte + 1 byte suffix
    sz_size_t required_bytes = 3 + (data_length * 4) + 1;

    // Check if we have enough buffer space
    if (required_bytes > buffer_length) {
        *did_fit = 0;
        return buffer_ptr;
    }

    // Second pass: write to buffer
    *(buffer_ptr++) = 'b';
    *(buffer_ptr++) = '\'';

    // Export each byte as \x followed by two hex digits
    static const char hex_chars[] = "0123456789abcdef";
    for (sz_size_t i = 0; i < data_length; i++) {
        unsigned char byte = (unsigned char)data[i];
        *(buffer_ptr++) = '\\';
        *(buffer_ptr++) = 'x';
        *(buffer_ptr++) = hex_chars[byte >> 4];
        *(buffer_ptr++) = hex_chars[byte & 0x0f];
    }

    *(buffer_ptr++) = '\'';
    return buffer_ptr;
}

/** Formats an array of strings, similar to the @c repr method of Python lists. Outputs an object
 *  that looks like `sz.Strings(['item1', 'item2... ])`, possibly dropping the last few entries. */
static PyObject *Strings_repr(Strings *self) {
    char repr_buffer[1024];
    sz_ptr_t repr_buffer_ptr = &repr_buffer[0];
    sz_cptr_t const repr_buffer_end = repr_buffer_ptr + 1024;

    // Start of the array
    memcpy(repr_buffer_ptr, "sz.Strings([", 12);
    repr_buffer_ptr += 12;

    sz_size_t count = self->tape.count;

    // In the worst case, we must have enough space for `...', ...])`
    // That's extra 11 bytes of content.
    sz_cptr_t non_fitting_array_tail = "... ])";
    int const non_fitting_array_tail_length = 6;

    // If the whole string doesn't fit, even before the `non_fitting_array_tail` tail,
    // we need to add `, '` separator of 3 bytes.
    for (sz_size_t i = 0; i < count && repr_buffer_ptr + (non_fitting_array_tail_length + 3) < repr_buffer_end; i++) {
        sz_string_view_t const string = sz_py_tape_string_(&self->tape, i);

        if (i > 0) { *(repr_buffer_ptr++) = ',', *(repr_buffer_ptr++) = ' '; }

        // Check if the string contains valid UTF-8
        int did_fit;
        repr_buffer_ptr = sz_utf8_find_malformed(string.start, string.length) == STRINGZILLA_NULL_CHAR
                              ? export_escaped_unquoted_to_utf8_buffer(
                                    string.start, string.length, repr_buffer_ptr,
                                    repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length, &did_fit)
                              : export_escaped_unquoted_to_binary_buffer(
                                    string.start, string.length, repr_buffer_ptr,
                                    repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length, &did_fit);

        // If it didn't fit, let's put an ellipsis
        if (!did_fit) {
            memcpy(repr_buffer_ptr, non_fitting_array_tail, non_fitting_array_tail_length);
            repr_buffer_ptr += non_fitting_array_tail_length;
            return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
        }
    }

    // Close the array
    *(repr_buffer_ptr++) = ']', *(repr_buffer_ptr++) = ')';
    return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
}

/** Array to string conversion method, that concatenates all the strings in the array. Will output
 *  an object that looks like `['item1', 'item2', 'item3']`, containing all the strings. */
static PyObject *Strings_to_unicode(Strings *self) {
    // Sum the lengths of all the slices to size the one allocation:
    sz_size_t count = self->tape.count;
    sz_size_t total_bytes = 2; // opening and closing square brackets
    for (sz_size_t i = 0; i < count; i++) {
        sz_string_view_t const string = sz_py_tape_string_(&self->tape, i);

        if (i != 0) total_bytes += 2; // For the preceding comma and space

        // Check if string is valid UTF-8 to determine format
        if (sz_utf8_find_malformed(string.start, string.length) == STRINGZILLA_NULL_CHAR) {
            // Valid UTF-8: format as '...' with escaped quotes
            total_bytes += 2;             // Opening and closing quotes
            total_bytes += string.length; // Base string length

            // Count the number of single quotes that need escaping
            sz_cptr_t scan_ptr = string.start;
            sz_size_t scan_length = string.length;
            while (scan_length) {
                char quote = '\'';
                sz_cptr_t next_quote = NULL;
                sz_status_t const status = sz_find_byte_best(scan_ptr, scan_length, &quote, &next_quote, sz_cap_cpus_k,
                                                             NULL);
                if (status != sz_success_k) {
                    sz_py_raise_status(status, "str()");
                    return NULL;
                }
                if (next_quote == NULL) break;
                total_bytes++; // Extra byte for escaping
                scan_length -= next_quote - scan_ptr + 1;
                scan_ptr = next_quote + 1;
            }
        }
        else {
            // Invalid UTF-8: format as b'\x...'
            total_bytes += 3;                 // "b'" prefix
            total_bytes += string.length * 4; // Each byte becomes \xNN (4 chars)
            total_bytes += 1;                 // Closing quote
        }
    }

    // Now allocate the memory for the concatenated string
    sz_ptr_t const result_buffer = malloc(total_bytes);
    if (!result_buffer) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the concatenated string");
        return NULL;
    }

    // Copy the strings into the result buffer
    sz_ptr_t result_ptr = result_buffer;
    *result_ptr++ = '[';
    for (sz_size_t i = 0; i < count; i++) {
        if (i != 0) {
            *result_ptr++ = ',';
            *result_ptr++ = ' ';
        }
        sz_string_view_t const string = sz_py_tape_string_(&self->tape, i);
        int did_fit;
        // Check if the string contains valid UTF-8 and export appropriately
        result_ptr = sz_utf8_find_malformed(string.start, string.length) == STRINGZILLA_NULL_CHAR
                         ? export_escaped_unquoted_to_utf8_buffer(string.start, string.length, result_ptr,
                                                                  total_bytes - (result_ptr - result_buffer), &did_fit)
                         : export_escaped_unquoted_to_binary_buffer(string.start, string.length, result_ptr,
                                                                    total_bytes - (result_ptr - result_buffer),
                                                                    &did_fit);

        // A zero `did_fit` means the buffer was sized wrong; the result is still returned.
    }

    *result_ptr++ = ']';
    sz_size_t actual_bytes = result_ptr - result_buffer;
    PyObject *result = PyUnicode_FromStringAndSize(result_buffer, actual_bytes);
    free(result_buffer);
    return result;
}

static PySequenceMethods Strings_as_sequence = {
    .sq_length = Strings_length, //
    .sq_item = Strings_getitem,  //
    .sq_contains = Strings_in,   //
};

static PyMappingMethods Strings_as_mapping = {
    .mp_length = Strings_length,       //
    .mp_subscript = Strings_subscript, // Is used to implement slices in Python
};

static char const doc_Strings_tape_address[] = //
    "Memory address of the first byte of the tape, which the offsets count from, as an integer.";

static char const doc_Strings_tape_nbytes[] = //
    "Bytes from the start of the tape to the end of its last string.";

static char const doc_Strings_offsets_address[] = //
    "Memory address of the first offset, as an integer.";

static char const doc_Strings_offsets_nbytes[] = //
    "Length of the offsets array in bytes, one offset per string and one past the last.";

static char const doc_Strings_offsets_are_large[] = //
    "True for 64-bit offsets, the Arrow `large_string` layout, and False for the 32-bit `string` one.";

static PyGetSetDef Strings_getsetters[] = {
    // Compatibility with PyArrow
    {"tape_address", (getter)Strings_get_tape_address, NULL, doc_Strings_tape_address, NULL},                //
    {"tape_nbytes", (getter)Strings_get_tape_nbytes, NULL, doc_Strings_tape_nbytes, NULL},                   //
    {"offsets_address", (getter)Strings_get_offsets_address, NULL, doc_Strings_offsets_address, NULL},       //
    {"offsets_nbytes", (getter)Strings_get_offsets_nbytes, NULL, doc_Strings_offsets_nbytes, NULL},          //
    {"offsets_are_large", (getter)Strings_get_offsets_are_large, NULL, doc_Strings_offsets_are_large, NULL}, //
    {NULL}                                                                                                   // Sentinel
};

/** Borrows the tape of an Apache Arrow string or binary array, pinned by its owning capsule. */
static int Strings_init_from_arrow_(Strings *self, PyObject *arrow_method) {
    PyObject *const capsules = PyObject_CallNoArgs(arrow_method);
    if (!capsules) return -1;
    if (!PyTuple_Check(capsules) || PyTuple_GET_SIZE(capsules) != 2 ||
        !PyCapsule_CheckExact(PyTuple_GET_ITEM(capsules, 0)) || !PyCapsule_CheckExact(PyTuple_GET_ITEM(capsules, 1))) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "__arrow_c_array__ must return a tuple of 2 capsules");
        return -1;
    }

    PyObject *const array_capsule = PyTuple_GET_ITEM(capsules, 1);
    struct arrow_schema_t const *schema = PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 0), "arrow_schema");
    struct arrow_array_t const *array = schema ? PyCapsule_GetPointer(array_capsule, "arrow_array") : NULL;
    if (!array) {
        Py_DECREF(capsules);
        return -1;
    }

    // Strings and binaries, with 32-bit offsets in lowercase formats and 64-bit ones in uppercase
    char const *const format = schema->format ? schema->format : "";
    sz_bool_t const large = (sz_bool_t)(strcmp(format, "U") == 0 || strcmp(format, "Z") == 0);
    if ((!large && strcmp(format, "u") != 0 && strcmp(format, "z") != 0) || array->n_buffers != 3) {
        Py_DECREF(capsules);
        PyErr_Format(PyExc_ValueError, "Arrow array must hold strings or binaries, not format '%s'", format);
        return -1;
    }

    // Null entries read as whatever their offsets span, which Arrow writers leave empty
    void const *const *buffers = array->buffers;
    self->tape.layout = large ? sz_strings_u64_tape_k : sz_strings_u32_tape_k;
    self->tape.count = (sz_size_t)array->length;
    if (large)
        self->tape.data.u64_tape.data = (sz_ptr_t)buffers[2],
        self->tape.data.u64_tape.offsets = (sz_u64_t *)buffers[1] + array->offset;
    else
        self->tape.data.u32_tape.data = (sz_ptr_t)buffers[2],
        self->tape.data.u32_tape.offsets = (sz_u32_t *)buffers[1] + array->offset;
    self->parent = array_capsule;
    Py_INCREF(array_capsule);
    Py_DECREF(capsules);
    return 0;
}

int sz_py_export_strings_like(PyObject *sequence, char const *context, sz_strings_view_t *strings, PyObject **snapshot,
                              sz_string_view_t **views) {
    *snapshot = NULL, *views = NULL;
    if (sz_py_export_strings_as_view(sequence, strings)) return 0;
    *snapshot = PySequence_Tuple(sequence);
    if (!*snapshot) return -1;
    sz_size_t const count = (sz_size_t)PyTuple_GET_SIZE(*snapshot);
    *views = PyMem_New(sz_string_view_t, count);
    if (!*views) {
        Py_CLEAR(*snapshot);
        PyErr_NoMemory();
        return -1;
    }
    for (sz_size_t index = 0; index != count; ++index) {
        PyObject *const item = PyTuple_GET_ITEM(*snapshot, (Py_ssize_t)index);
        sz_string_view_t *const view = *views + index;
        if (sz_py_export_string_like(item, &view->start, &view->length)) continue;
        // A `str` that cannot be UTF-8 keeps its `UnicodeEncodeError`
        if (!PyUnicode_Check(item))
            PyErr_Format(PyExc_TypeError, "%s item %zu is a %s, not a string", context, index, Py_TYPE(item)->tp_name);
        PyMem_Free(*views), *views = NULL;
        Py_CLEAR(*snapshot);
        return -1;
    }
    sz_strings_from_string_views(*views, count, strings);
    return 0;
}

/** Copies the strings of any iterable into one tape. */
static int Strings_init_from_iterable_(Strings *self, PyObject *iterable) {
    sz_strings_view_t strings;
    PyObject *snapshot;
    sz_string_view_t *views;
    if (sz_py_export_strings_like(iterable, "Strings()", &strings, &snapshot, &views) != 0) return -1;
    sz_status_t status = sz_allocator_init_heap(&self->allocator);
    if (status == sz_success_k)
        status = sz_strings_to_tape_best(&strings, NULL, strings.count, &self->allocator, &self->tape, sz_cap_cpus_k,
                                         NULL);
    PyMem_Free(views);
    Py_XDECREF(snapshot);
    if (status == sz_success_k) return 0;
    sz_py_raise_status(status, "Strings()");
    return -1;
}

static int Strings_init(Strings *self, PyObject *args, PyObject *kwargs) {
    Py_ssize_t const positional_args_count = PyTuple_GET_SIZE(args);
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "Strings() takes at most 1 argument: a sequence of strings");
        return -1;
    }
    PyObject *sequence_object = positional_args_count ? PyTuple_GET_ITEM(args, 0) : NULL;
    if (kwargs) {
        Py_ssize_t position = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &position, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "sequence") == 0 && !sequence_object) { sequence_object = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    // A second `__init__` rebinds the collection, so whatever the first one bound is released
    Strings_release_(self);
    PyObject *const arrow_method = sequence_object ? PyObject_GetAttrString(sequence_object, "__arrow_c_array__")
                                                   : NULL;
    if (!arrow_method) {
        PyErr_Clear();
        // Without a sequence, the empty tuple of positional arguments stands for one
        return Strings_init_from_iterable_(self, sequence_object ? sequence_object : args);
    }
    int const status = Strings_init_from_arrow_(self, arrow_method);
    Py_DECREF(arrow_method);
    return status;
}

/** The unified allocator of one vendor and the stream naming the device its blocks belong to, which
 *  the host gathers a @c Strings.copy through, since host kernels pass no stream of their own. */
typedef struct sz_py_placement_t {

    /** The allocator of the copy's capabilities, which the copy keeps to free its tape. */
    sz_allocator_t unified;

    /** The stream every block is made on and returned behind. */
    sz_stream_t stream;
} sz_py_placement_t;

static void *sz_py_placement_allocate_(sz_size_t bytes, void *handle, sz_stream_t stream) {
    sz_py_placement_t const *placement = (sz_py_placement_t const *)handle;
    sz_unused_(stream);
    return placement->unified.allocate(bytes, placement->unified.handle, placement->stream);
}

static void *sz_py_placement_reallocate_(void *pointer, sz_size_t old_bytes, sz_size_t new_bytes, void *handle,
                                         sz_stream_t stream) {
    sz_py_placement_t const *placement = (sz_py_placement_t const *)handle;
    sz_unused_(stream);
    return placement->unified.reallocate(pointer, old_bytes, new_bytes, placement->unified.handle, placement->stream);
}

static void sz_py_placement_free_(void *pointer, sz_size_t bytes, void *handle, sz_stream_t stream) {
    sz_py_placement_t const *placement = (sz_py_placement_t const *)handle;
    sz_unused_(stream);
    placement->unified.free(pointer, bytes, placement->unified.handle, placement->stream);
}

static char const doc_Strings_copy[] =                                                                 //
    "copy(capabilities, stream=None) -> Strings\n"                                                     //
    "\n"                                                                                               //
    "Copy the strings into a tape in the unified memory of `capabilities`, which engines of those\n"   //
    "capabilities read in place and the host reads as a normal read-only Strings.\n"                   //
    "\n"                                                                                               //
    "Args:\n"                                                                                          //
    "  capabilities (int): A producer's mask, like sz.cuda_capabilities_enabled(0).\n"                 //
    "  stream (int, optional): A stream of that vendor as an integer, naming the device the tape is\n" //
    "    made on, or None for the default.\n"                                                          //
    "Returns:\n"                                                                                       //
    "  Strings: A tape owning its unified memory, freed with it.\n"                                    //
    "Example:\n"                                                                                       //
    "  >>> tape = sz.Strings(['kitten', 'sitting']).copy(sz.cpu_capabilities_enabled())\n"             //
    "  >>> [str(text) for text in tape]\n"                                                             //
    "  ['kitten', 'sitting']";

static PyObject *Strings_copy(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                              PyObject *args_names_tuple) {
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "copy() takes at most 2 positional arguments, got %zd", positional_args_count);
        return NULL;
    }
    PyObject *capabilities_object = positional_args_count > 0 ? args[0] : NULL;
    PyObject *stream_object = positional_args_count > 1 ? args[1] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                          : PyUnicode_CompareWithASCIIString(key, "stream") == 0     ? &stream_object
                                                                                     : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "copy() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "copy() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!capabilities_object) {
        PyErr_SetString(PyExc_TypeError, "copy() missing required argument 'capabilities'");
        return NULL;
    }
    unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities_object);
    if (PyErr_Occurred()) return NULL;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;

    sz_py_placement_t placement;
    placement.stream = stream;
    sz_allocator_t placed;
    placed.allocate = sz_py_placement_allocate_, placed.reallocate = sz_py_placement_reallocate_;
    placed.free = sz_py_placement_free_, placed.handle = &placement;
    sz_strings_view_t source;
    sz_strings_from_span(&self->tape, &source);
    sz_strings_span_t tape = {0};
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    status = sz_allocator_init_unified_best(&placement.unified, (sz_capability_t)bits);
    // The host gathers, as the strings sit in host memory, which a device does not read.
    if (status == sz_success_k)
        status = sz_strings_to_tape_best(&source, NULL, source.count, &placed, &tape, sz_cap_cpus_k, NULL);
    Py_END_ALLOW_THREADS;
    return Strings_from_tape_(status, &tape, &placement.unified, "copy()");
}

static void Strings_dealloc(Strings *self) {
    Strings_release_(self);

    // Park the dead header on the free-list instead of freeing it, threading the `next` link
    // through the now-unused `parent`; free it for real when the cache is full or absent.
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->strings_freelist_count < sz_freelist_capacity_k) {
        self->parent = (PyObject *)state->strings_freelist_head;
        state->strings_freelist_head = self;
        state->strings_freelist_count++;
        sz_freelist_unlock_(state);
        return;
    }
    if (state) sz_freelist_unlock_(state);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static char const doc_Strings_shuffled[] =                                                         //
    "shuffled(seed=None) -> Strings\n"                                                             //
    "\n"                                                                                           //
    "Return a new Strings with the elements randomly permuted, like `random.sample` drawing all\n" //
    "of them; the original is unchanged.\n"                                                        //
    "\n"                                                                                           //
    "Args:\n"                                                                                      //
    "  seed (int, optional): Seed for reproducible shuffling. Defaults to a random seed.\n"        //
    "Returns:\n"                                                                                   //
    "  Strings: A new shuffled collection on a tape of its own.\n"                                 //
    "Example:\n"                                                                                   //
    "  >>> sorted(map(str, sz.Strings(['a', 'b', 'c']).shuffled(seed=42))) == ['a', 'b', 'c']\n"   //
    "  True";

static char const doc_Strings_sample[] =                                                   //
    "sample(size, seed=None) -> Strings\n"                                                 //
    "\n"                                                                                   //
    "Return a new Strings with `size` elements drawn at random with replacement, like\n"   //
    "`random.choices(strings, k=size)`.\n"                                                 //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  size (int): Number of elements to draw.\n"                                          //
    "  seed (int, optional): Seed for reproducible sampling. Defaults to a random seed.\n" //
    "Returns:\n"                                                                           //
    "  Strings: A new collection of the sampled elements on a tape of its own.\n"          //
    "Example:\n"                                                                           //
    "  >>> list(map(bytes, sz.Strings(['a', 'a', 'a']).sample(2)))\n"                      //
    "  [b'a', b'a']";

static char const doc_Strings_matched[] =                                                         //
    "matched(matches) -> Strings\n"                                                               //
    "\n"                                                                                          //
    "Return a new Strings holding the bytes each match row spans, in row order, gathering what\n" //
    "a `find` of `SubstringsEngine` or `SubdocumentsEngine` located in this collection.\n"        //
    "\n"                                                                                          //
    "Args:\n"                                                                                     //
    "  matches (buffer): Contiguous 2-D buffer of pointer-width unsigned integers shaped\n"       //
    "    `(count, 4)` - haystack index, needle index, byte offset, byte length.\n"                //
    "Returns:\n"                                                                                  //
    "  Strings: A new collection of the matched spans on a tape of its own.\n"                    //
    "Raises:\n"                                                                                   //
    "  IndexError: When a row reaches outside this collection.\n"                                 //
    "Example:\n"                                                                                  //
    "  >>> import struct\n"                                                                       //
    "  >>> documents = sz.Strings(['{\"a\": [1, \"x\"]}'])\n"                                     //
    "  >>> engine = sz.SubdocumentsEngine(sz.Strings(['$.a[*]']))\n"                              //
    "  >>> matches = memoryview(bytearray(2 * 4 * struct.calcsize('N'))).cast('N', (2, 4))\n"     //
    "  >>> offsets = memoryview(bytearray(2 * struct.calcsize('N'))).cast('N')\n"                 //
    "  >>> engine.find(documents, matches, offsets)\n"                                            //
    "  >>> list(map(bytes, documents.matched(matches)))\n"                                        //
    "  [b'1', b'\"x\"']";

static PyMethodDef Strings_methods[] = {
    {"shuffled", Strings_shuffled, STRINGZILLA_METHOD_FLAGS, doc_Strings_shuffled}, //
    {"sorted", Strings_sorted, STRINGZILLA_METHOD_FLAGS, doc_sorted},               //
    {"argsort", Strings_argsort, STRINGZILLA_METHOD_FLAGS, doc_argsort},            //
    {"find", Strings_find, STRINGZILLA_METHOD_FLAGS, doc_Strings_find},
    {"hashes", Strings_hashes, STRINGZILLA_METHOD_FLAGS, doc_Strings_hashes},
    {"utf8_count", Strings_utf8_count, STRINGZILLA_METHOD_FLAGS, doc_Strings_utf8_count},
    {"sample", Strings_sample, STRINGZILLA_METHOD_FLAGS, doc_Strings_sample},          //
    {"matched", Strings_matched, STRINGZILLA_METHOD_FLAGS, doc_Strings_matched},       //
    {"intersect", Strings_intersect, STRINGZILLA_METHOD_FLAGS, doc_Strings_intersect}, //
    {"copy", Strings_copy, STRINGZILLA_METHOD_FLAGS, doc_Strings_copy},                //
    {"translate", Strings_translate, STRINGZILLA_METHOD_FLAGS, doc_Strings_translate}, //
    {"escape", Strings_escape, STRINGZILLA_METHOD_FLAGS, doc_Strings_escape},          //
    {"unescape", Strings_unescape, STRINGZILLA_METHOD_FLAGS, doc_Strings_unescape},    //
    {"normalize", Strings_normalize, STRINGZILLA_METHOD_FLAGS, doc_Strings_normalize}, //
    {"to_json", Strings_to_json, STRINGZILLA_METHOD_FLAGS, doc_Strings_to_json},       //
    {NULL, NULL, 0, NULL}                                                              // Sentinel
};

static char const doc_Strings[] =                                                                    //
    "Strings(sequence=())\n"                                                                         //
    "\n"                                                                                             //
    "An immutable sequence of byte strings on one tape - their bytes back to back and an array of\n" //
    "offsets, the Apache Arrow `string` or `large_string` layout - like a `tuple` of `bytes` that\n" //
    "costs two allocations rather than one per element.\n"                                           //
    "\n"                                                                                             //
    "Args:\n"                                                                                        //
    "  sequence (iterable | pyarrow.Array, optional): The strings to hold. An Arrow string or\n"     //
    "    binary array is borrowed in place; any other iterable of `str` or bytes-like objects\n"     //
    "    is copied once.\n"                                                                          //
    "\n"                                                                                             //
    "Slicing with a step of one borrows the tape, while any other step, `sorted`, `shuffled` and\n"  //
    "`sample` gather into a new one, and `argsort` returns the order without copying, as `find`,\n"  //
    "`hashes` and `utf8_count` return one number per string in a single pass over the tape.\n"       //
    "\n"                                                                                             //
    "Example:\n"                                                                                     //
    "  >>> names = sz.Strings(['banana', 'apple', 'cherry', 'date'])\n"                              //
    "  >>> [str(x) for x in names.sorted()]\n"                                                       //
    "  ['apple', 'banana', 'cherry', 'date']\n"                                                      //
    "  >>> [str(x) for x in names.sorted(top=2)]  # partial top-k is cheaper than a full sort\n"     //
    "  ['apple', 'banana']";

PyTypeObject StringsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Strings",
    .tp_doc = doc_Strings,
    .tp_basicsize = sizeof(Strings),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Strings_init,
    .tp_dealloc = (destructor)Strings_dealloc,
    .tp_methods = Strings_methods,
    .tp_as_sequence = &Strings_as_sequence,
    .tp_as_mapping = &Strings_as_mapping,
    .tp_getset = Strings_getsetters,
    .tp_richcompare = Strings_richcompare,
    .tp_repr = (reprfunc)Strings_repr,
    .tp_str = (reprfunc)Strings_to_unicode,
};

sz_status_t sz_py_lookup(sz_ptr_t target, sz_cptr_t source, sz_size_t length, char const *lut,
                         sz_capability_t capabilities) {
    // A one-string tape over the caller's buffers, in the offsets width the kernel writes, so it
    // allocates nothing; in place, the offsets are shared too, making it the very tape it reads.
    sz_u32_t narrow_source[2] = {0, (sz_u32_t)length}, narrow_target[2] = {0, (sz_u32_t)length};
    sz_u64_t wide_source[2] = {0, length}, wide_target[2] = {0, length};
    sz_bool_t const in_place = (sz_bool_t)(source == target);
    sz_strings_view_t strings;
    sz_strings_span_t tape;
    strings.count = tape.count = 1;
    if (length <= 0xFFFFFFFFull) {
        strings.layout = tape.layout = sz_strings_u32_tape_k;
        strings.data.u32_tape.data = source, strings.data.u32_tape.offsets = narrow_source;
        tape.data.u32_tape.data = target, tape.data.u32_tape.offsets = in_place ? narrow_source : narrow_target;
    }
    else {
        strings.layout = tape.layout = sz_strings_u64_tape_k;
        strings.data.u64_tape.data = source, strings.data.u64_tape.offsets = wide_source;
        tape.data.u64_tape.data = target, tape.data.u64_tape.offsets = in_place ? wide_source : wide_target;
    }
    return sz_strings_lookup_best(&strings, NULL, 1, lut, NULL, &tape, capabilities, NULL);
}

/** Reads a 256-byte string or a dict of characters below U+0100, one byte each, into @p lut. */
static int sz_py_export_lookup_table_(PyObject *table_object, char *lut) {
    sz_string_view_t table;
    if (PyDict_Check(table_object)) {
        // Every byte the dict leaves out maps to itself
        for (int i = 0; i < 256; i++) lut[i] = (char)i;
        PyObject *key, *value;
        Py_ssize_t position = 0;
        while (PyDict_Next(table_object, &position, &key, &value)) {
            if (!PyUnicode_Check(key) || PyUnicode_GetLength(key) != 1 || !PyUnicode_Check(value) ||
                PyUnicode_GetLength(value) != 1) {
                PyErr_SetString(PyExc_TypeError, "Keys and values must be single characters");
                return -1;
            }
            Py_UCS4 const key_char = PyUnicode_READ_CHAR(key, 0), value_char = PyUnicode_READ_CHAR(value, 0);
            if (key_char > 255 || value_char > 255) {
                PyErr_SetString(PyExc_ValueError, "Keys and values must be characters below U+0100, naming one byte");
                return -1;
            }
            lut[key_char] = (char)value_char;
        }
        return 0;
    }
    if (!sz_py_export_string_like(table_object, &table.start, &table.length)) {
        wrap_current_exception("The look-up table must be string-like or a dictionary");
        return -1;
    }
    if (table.length != 256) {
        PyErr_SetString(PyExc_ValueError, "The look-up table must be exactly 256 bytes long");
        return -1;
    }
    memcpy(lut, table.start, 256);
    return 0;
}

/** Appends @p string at @p target, returning the byte past it; an empty one may have no start. */
static sz_ptr_t sz_py_append_(sz_ptr_t target, sz_string_view_t string) {
    if (string.length) memcpy(target, string.start, string.length);
    return target + string.length;
}

#pragma region Translation

char const doc_translate[] =                                                                              //
    "Perform transformation of a string using a look-up table.\n"                                         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  text (String or str or bytes): The string object.\n"                                               //
    "  table (str or dict): A 256-byte string, or a dict mapping characters below U+0100 to such.\n"      //
    "  inplace (bool, optional): If True, the string is modified in place, defaulting to False.\n"        //
    "\n"                                                                                                  //
    "  start (int, optional): The starting index for translation, defaulting to 0.\n"                     //
    "  end (int, optional): The ending index for translation, defaulting to the string length.\n"         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Union[None, str, bytes]: If inplace is False, a translated copy of the [start, end) slice is\n"    //
    "    returned, otherwise None.\n"                                                                     //
    "Raises:\n"                                                                                           //
    "  ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.\n"       //
    "  UnicodeDecodeError: If a str input translates into malformed UTF-8.\n"                             //
    "  TypeError: If the table is not a string or dictionary.\n"                                          //
    "\n"                                                                                                  //
    "Example:\n"                                                                                          //
    "  >>> sz.String('abc').translate({'a': 'A'}) == b'Abc'\n"                                            //
    "  True";

PyObject *String_like_translate(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                                PyObject *args_names_tuple) {
    int is_member = self != NULL && PyObject_TypeCheck(self, &StringType);
    if (positional_args_count < !is_member + 1 || positional_args_count > !is_member + 4) {
        PyErr_Format(PyExc_TypeError, "Invalid number of arguments");
        return NULL;
    }

    PyObject *text_object = is_member ? self : args[0];
    PyObject *table_object = args[!is_member];
    PyObject *inplace_object = positional_args_count > !is_member + 1 ? args[!is_member + 1] : NULL;
    PyObject *start_object = positional_args_count > !is_member + 2 ? args[!is_member + 2] : NULL;
    PyObject *end_object = positional_args_count > !is_member + 3 ? args[!is_member + 3] : NULL;
    PyObject *capabilities_object = NULL;

    // Optional keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "inplace") == 0 && !inplace_object) { inplace_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "start") == 0 && !start_object) { start_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "end") == 0 && !end_object) { end_object = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 && !capabilities_object) {
                capabilities_object = value;
            }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) return NULL;
        }
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    // Optional start and end arguments
    Py_ssize_t start, end;
    if (!sz_py_export_optional_index(start_object, 0, &start)) {
        PyErr_SetString(PyExc_TypeError, "start must be an integer");
        return NULL;
    }
    if (!sz_py_export_optional_index(end_object, PY_SSIZE_T_MAX, &end)) {
        PyErr_SetString(PyExc_TypeError, "end must be an integer");
        return NULL;
    }

    sz_string_view_t text;
    if (!sz_py_export_string_like(text_object, &text.start, &text.length)) {
        wrap_current_exception("First argument must be string-like");
        return NULL;
    }

    sz_align_(64) char look_up_table[256];
    if (sz_py_export_lookup_table_(table_object, look_up_table) != 0) return NULL;

    int is_inplace = inplace_object ? PyObject_IsTrue(inplace_object) : 0;
    if (is_inplace == -1) {
        PyErr_SetString(PyExc_TypeError, "The inplace argument must be a boolean");
        return NULL;
    }

    // Clamp [start, end) into the string, so negative, out-of-range or inverted windows fit
    sz_size_t normalized_offset, normalized_length;
    sz_ssize_clamp_interval(text.length, start, end, &normalized_offset, &normalized_length);
    text.start += normalized_offset;
    text.length = normalized_length;

    if (is_inplace) {
        if (sz_py_is_mutable(text_object) == sz_false_k) return NULL;
        sz_status_t const status = sz_py_lookup((sz_ptr_t)text.start, text.start, text.length, look_up_table,
                                                capabilities);
        if (status != sz_success_k) {
            sz_py_raise_status(status, "translate()");
            return NULL;
        }
        Py_RETURN_NONE;
    }

    // The table maps UTF-8 bytes, which a `str` does not store, so its result is decoded back
    if (PyUnicode_Check(text_object)) {
        sz_ptr_t translated = (sz_ptr_t)PyMem_Malloc(text.length);
        if (!translated) return PyErr_NoMemory();
        sz_status_t const status = sz_py_lookup(translated, text.start, text.length, look_up_table, capabilities);
        if (status != sz_success_k) {
            PyMem_Free(translated);
            sz_py_raise_status(status, "translate()");
            return NULL;
        }
        PyObject *translated_text = PyUnicode_DecodeUTF8(translated, (Py_ssize_t)text.length, "strict");
        PyMem_Free(translated);
        return translated_text;
    }
    PyObject *translated_bytes = PyBytes_FromStringAndSize(NULL, text.length);
    if (!translated_bytes) return NULL;
    sz_status_t const status = sz_py_lookup(PyBytes_AS_STRING(translated_bytes), text.start, text.length, look_up_table,
                                            capabilities);
    if (status != sz_success_k) {
        Py_DECREF(translated_bytes);
        sz_py_raise_status(status, "translate()");
        return NULL;
    }
    return translated_bytes;
}

char const doc_Strings_translate[] =                                                                      //
    "translate(table, *, capabilities=None) -> Strings\n"                                                 //
    "\n"                                                                                                  //
    "Map every byte of every string through a look-up table, like `bytes.translate` over each.\n"         //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  table (str or bytes or dict): A 256-byte table, or a dict mapping characters below U+0100.\n"      //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strings: A new collection, each string as long as its source.\n"                                   //
    "Raises:\n"                                                                                           //
    "  ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.\n"       //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strings(['cab', 'bee']).translate({'a': 'A', 'b': 'B'})]\n"          //
    "  ['cAB', 'Bee']";

PyObject *Strings_translate(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "translate() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *table_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "table") == 0          ? &table_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "translate() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "translate() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!table_object) {
        PyErr_SetString(PyExc_TypeError, "translate() missing required argument 'table'");
        return NULL;
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_align_(64) char look_up_table[256];
    if (sz_py_export_lookup_table_(table_object, look_up_table) != 0) return NULL;

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_lookup_best(&strings, NULL, strings.count, look_up_table, &allocator, &tape, capabilities,
                                        NULL);
    return Strings_from_tape_(status, &tape, &allocator, "translate()");
}

#pragma endregion Translation

#pragma region Escaping

/** Reads the name of an escape kind, @c json when absent, raising errors that name @p context. */
static int sz_py_export_escape_kind_(PyObject *kind_object, char const *context, sz_escape_kind_t *kind) {
    static struct {
        char const *name;
        sz_escape_kind_t kind;
    } const kinds[] = {
        {"json", sz_escape_json_k}, {"json_ascii", sz_escape_json_ascii_k},
        {"hex", sz_escape_hex_k},   {"html", sz_escape_html_k},
        {"url", sz_escape_url_k},   {"console", sz_escape_console_k},
        {"csv", sz_escape_csv_k},
    };
    *kind = sz_escape_json_k;
    if (!kind_object) return 0;
    if (!PyUnicode_Check(kind_object)) {
        PyErr_Format(PyExc_TypeError, "%s argument 'kind' must be a str", context);
        return -1;
    }
    for (sz_size_t index = 0; index != sizeof(kinds) / sizeof(kinds[0]); ++index)
        if (PyUnicode_CompareWithASCIIString(kind_object, kinds[index].name) == 0) {
            *kind = kinds[index].kind;
            return 0;
        }
    PyErr_Format(PyExc_ValueError,
                 "%s unknown kind '%U': expected 'json', 'json_ascii', 'hex', 'html', 'url', 'console' or 'csv'",
                 context, kind_object);
    return -1;
}

char const doc_Strings_escape[] =                                                                         //
    "escape(kind='json', *, capabilities=None) -> Strings\n"                                              //
    "\n"                                                                                                  //
    "Escape every string for a text format, in one pass over the whole collection.\n"                     //
    "\n"                                                                                                  //
    "Kinds:\n"                                                                                            //
    "  json: A JSON string body, as `json.dumps(text, ensure_ascii=False)` writes it unquoted.\n"         //
    "  json_ascii: The same with every non-ASCII codepoint as `\\uXXXX`, like `json.dumps(text)`.\n"      //
    "  hex: Two lowercase digits per byte, as `bytes.hex` writes them.\n"                                 //
    "  html: `&`, `<`, `>`, `\"` and `'` as entities, like `html.escape`, with `&#39;` for `'`.\n"        //
    "  url: Percent-encoding, as `urllib.parse.quote(text, safe='')` writes it.\n"                        //
    "  console: Controls, ill-formed UTF-8 and bidirectional overrides as `\\n`, `\\xHH` or `\\uXXXX`.\n" //
    "  csv: A field quoted only around a comma, a quote, CR or LF, as `csv.writer` writes it.\n"          //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  kind (str, optional): One of the kinds above, defaulting to 'json'.\n"                             //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strings: A new collection of the escaped strings.\n"                                               //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strings(['<b>', 'Tom & Jerry']).escape('html')]\n"                   //
    "  ['&lt;b&gt;', 'Tom &amp; Jerry']";

PyObject *Strings_escape(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "escape() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *kind_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "kind") == 0           ? &kind_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "escape() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "escape() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_escape_kind_t kind;
    sz_capability_t capabilities;
    if (sz_py_export_escape_kind_(kind_object, "escape()", &kind) != 0) return NULL;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_escape_best(&strings, NULL, strings.count, kind, &allocator, &tape, capabilities, NULL);
    return Strings_from_tape_(status, &tape, &allocator, "escape()");
}

char const doc_Strings_unescape[] =                                                                       //
    "unescape(kind='json', *, capabilities=None) -> Strings\n"                                            //
    "\n"                                                                                                  //
    "Reverse `escape` for every string, in one pass over the whole collection. The json kinds\n"          //
    "decode what `json.loads` decodes in a string body, html what `html.unescape` decodes of the\n"       //
    "five XML entities and numeric references, url what `urllib.parse.unquote_to_bytes` decodes,\n"       //
    "and csv one field as `csv.reader` reads it.\n"                                                       //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  kind (str, optional): One of the kinds `escape` takes, defaulting to 'json'.\n"                    //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strings: A new collection of the unescaped strings.\n"                                             //
    "Raises:\n"                                                                                           //
    "  ValueError: Naming the index of the first string holding a malformed escape.\n"                    //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strings(['a%20b', '100%25']).unescape('url')]\n"                     //
    "  ['a b', '100%']";

PyObject *Strings_unescape(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                           PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "unescape() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *kind_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "kind") == 0           ? &kind_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "unescape() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "unescape() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_escape_kind_t kind;
    sz_capability_t capabilities;
    if (sz_py_export_escape_kind_(kind_object, "unescape()", &kind) != 0) return NULL;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_size_t malformed_index = 0;
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_unescape_best(&strings, NULL, strings.count, kind, &allocator, &tape, &malformed_index,
                                          capabilities, NULL);
    if (status != sz_invalid_utf8_k) return Strings_from_tape_(status, &tape, &allocator, "unescape()");
    sz_strings_span_free(&tape, &allocator, NULL);
    PyErr_Format(PyExc_ValueError, "unescape(): the string at index %zu holds a malformed escape", malformed_index);
    return NULL;
}

#pragma endregion Escaping

#pragma region Normalization

char const doc_Strings_normalize[] =                                                                      //
    "normalize(form='NFC', *, capabilities=None) -> Strings\n"                                            //
    "\n"                                                                                                  //
    "Bring every string into a Unicode normalization form, in one pass over the whole collection,\n"      //
    "like `unicodedata.normalize` over each. 'NFKC_Casefold' is the Unicode toNFKC_Casefold, the\n"       //
    "keys search engines match on: NFKC with full case folding and no default-ignorable codepoints.\n"    //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  form (str, optional): 'NFC', 'NFD', 'NFKC', 'NFKD' or 'NFKC_Casefold', defaulting to 'NFC'.\n"     //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  Strings: A new collection of the normalized strings.\n"                                            //
    "Example:\n"                                                                                          //
    "  >>> [str(text) for text in sz.Strings(['\\ufb01', 'Stra\\u00dfe']).normalize('NFKC_Casefold')]\n"  //
    "  ['fi', 'strasse']";

PyObject *Strings_normalize(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                            PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "normalize() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *form_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "form") == 0           ? &form_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "normalize() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "normalize() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    sz_normal_form_t form = sz_normal_form_nfc_k;
    sz_capability_t capabilities;
    if (form_object && sz_py_export_normal_form(form_object, "normalize()", &form) != 0) return NULL;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_normalize_best(&strings, NULL, strings.count, form, &allocator, &tape, capabilities, NULL);
    return Strings_from_tape_(status, &tape, &allocator, "normalize()");
}

#pragma endregion Normalization

#pragma region Dumping

/**
 *  @brief Joins @p rows rows of @p columns, framed by @p frames and separated by @p between, into
 *      one new @c bytes object between @p opening and @p closing, measured first and written once.
 *  @return The bytes, or null with a Python exception set that names @p context.
 */
static PyObject *sz_py_join_bytes_(sz_strings_view_t const *columns, sz_size_t columns_count, sz_size_t rows,
                                   sz_string_view_t const *frames, sz_string_view_t between, sz_string_view_t opening,
                                   sz_string_view_t closing, sz_capability_t capabilities, char const *context) {
    sz_size_t length = 0;
    sz_status_t status = sz_strings_join_best(columns, columns_count, rows, frames, between, NULL, &length,
                                              capabilities, NULL);
    if (status == sz_success_k && length > (sz_size_t)PY_SSIZE_T_MAX - opening.length - closing.length)
        status = sz_bad_alloc_k;
    if (status != sz_success_k) {
        sz_py_raise_status(status, context);
        return NULL;
    }
    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)(opening.length + length + closing.length));
    if (!result) return NULL;
    sz_ptr_t const target = sz_py_append_(PyBytes_AS_STRING(result), opening);
    sz_py_append_(target + length, closing);
    // Only measuring can fail, so writing the measured bytes succeeds
    sz_strings_join_best(columns, columns_count, rows, frames, between, target, &length, capabilities, NULL);
    return result;
}

char const doc_Strings_to_json[] =                                                                         //
    "to_json(ensure_ascii=True, *, capabilities=None) -> bytes\n"                                          //
    "\n"                                                                                                   //
    "Dump the strings as one compact JSON array, the bytes of\n"                                           //
    "`json.dumps(strings, ensure_ascii=ensure_ascii, separators=(',', ':')).encode()`.\n"                  //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  ensure_ascii (bool, optional): Write every non-ASCII codepoint as `\\uXXXX`, defaulting to True.\n" //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n"  //
    "Returns:\n"                                                                                           //
    "  bytes: The UTF-8 JSON text.\n"                                                                      //
    "Example:\n"                                                                                           //
    "  >>> sz.Strings(['Ada', 'Grace']).to_json()\n"                                                       //
    "  b'[\"Ada\",\"Grace\"]'";

PyObject *Strings_to_json(Strings *self, PyObject *const *args, Py_ssize_t positional_args_count,
                          PyObject *args_names_tuple) {
    if (positional_args_count > 1)
        return PyErr_Format(PyExc_TypeError, "to_json() takes at most 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *ascii_object = positional_args_count ? args[0] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "ensure_ascii") == 0   ? &ascii_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "to_json() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "to_json() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    int const ensure_ascii = ascii_object ? PyObject_IsTrue(ascii_object) : 1;
    if (ensure_ascii < 0) return NULL;
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_strings_view_t strings;
    sz_strings_from_span(&self->tape, &strings);
    sz_allocator_t allocator;
    sz_strings_span_t tape = {0};
    sz_status_t status = sz_allocator_init_heap(&allocator);
    if (status == sz_success_k)
        status = sz_strings_escape_best(&strings, NULL, strings.count,
                                        ensure_ascii ? sz_escape_json_ascii_k : sz_escape_json_k, &allocator, &tape,
                                        capabilities, NULL);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "to_json()");
        return NULL;
    }

    sz_string_view_t const frames[2] = {{"\"", 1}, {"\"", 1}}, comma = {",", 1}, opening = {"[", 1}, closing = {"]", 1};
    sz_strings_view_t escaped;
    sz_strings_from_span(&tape, &escaped);
    PyObject *result = sz_py_join_bytes_(&escaped, 1, escaped.count, frames, comma, opening, closing, capabilities,
                                         "to_json()");
    sz_strings_span_free(&tape, &allocator, NULL);
    return result;
}

/** Escapes the strings of one column, a @c Strings read in place or any other sequence of strings,
 *  into @p tape, or returns -1 with a Python exception set that names @p context. */
static int sz_py_escape_column_(PyObject *column, char const *context, sz_escape_kind_t kind, sz_allocator_t *allocator,
                                sz_strings_span_t *tape, sz_capability_t capabilities) {
    sz_strings_view_t strings;
    PyObject *snapshot;
    sz_string_view_t *views;
    if (sz_py_export_strings_like(column, context, &strings, &snapshot, &views) != 0) return -1;
    sz_status_t const status = sz_strings_escape_best(&strings, NULL, strings.count, kind, allocator, tape,
                                                      capabilities, NULL);
    PyMem_Free(views);
    Py_XDECREF(snapshot);
    if (status == sz_success_k) return 0;
    sz_py_raise_status(status, context);
    return -1;
}

/** Frees the @p count tapes of @c sz_py_escape_columns_ and the array holding them. */
static void sz_py_free_tapes_(sz_strings_span_t *tapes, Py_ssize_t count, sz_allocator_t *allocator) {
    for (Py_ssize_t index = 0; index != count; ++index) sz_strings_span_free(&tapes[index], allocator, NULL);
    PyMem_Free(tapes);
}

/**
 *  @brief Escapes the names of the mapping @p columns into the first of @p tapes, and every column
 *      into the tapes after it, checking that all columns hold equally many strings.
 *  @param[out] tapes Receives `1 + columns` tapes, freed by @c sz_py_free_tapes_.
 *  @return The number of columns, or -1 with a Python exception set and nothing left to free.
 */
static Py_ssize_t sz_py_escape_columns_(PyObject *columns, char const *context, sz_escape_kind_t kind,
                                        sz_allocator_t *allocator, sz_capability_t capabilities,
                                        sz_strings_span_t **tapes) {
    PyObject *const names = PyMapping_Keys(columns);
    if (!names) return -1;
    Py_ssize_t const columns_count = PyList_GET_SIZE(names);
    // Zeroed tapes hold no blocks, so a failure frees all of them, escaped or not
    sz_strings_span_t *const escaped = (sz_strings_span_t *)PyMem_Calloc((size_t)columns_count + 1,
                                                                         sizeof(sz_strings_span_t));
    if (!escaped) {
        Py_DECREF(names);
        PyErr_NoMemory();
        return -1;
    }
    int failed = sz_py_escape_column_(names, context, kind, allocator, &escaped[0], capabilities);
    for (Py_ssize_t column_index = 1; !failed && column_index <= columns_count; ++column_index) {
        PyObject *const name = PyList_GET_ITEM(names, column_index - 1);
        PyObject *const column = PyObject_GetItem(columns, name);
        failed = !column ||
                 sz_py_escape_column_(column, context, kind, allocator, &escaped[column_index], capabilities);
        Py_XDECREF(column);
        if (!failed && escaped[column_index].count != escaped[1].count) {
            PyErr_Format(PyExc_ValueError, "%s column %R holds %zu strings, while the first one holds %zu", context,
                         name, escaped[column_index].count, escaped[1].count);
            failed = 1;
        }
    }
    Py_DECREF(names);
    if (failed) {
        sz_py_free_tapes_(escaped, columns_count + 1, allocator);
        return -1;
    }
    *tapes = escaped;
    return columns_count;
}

char const doc_to_json[] =                                                                                 //
    "to_json(columns, ensure_ascii=True, *, capabilities=None) -> bytes\n"                                 //
    "\n"                                                                                                   //
    "Dump columns of strings as a compact JSON array of records, one object per row, the bytes of\n"       //
    "`json.dumps([dict(zip(columns, row)) for row in zip(*columns.values())], separators=(',', ':'))`.\n"  //
    "Each column is escaped in one pass, and the rows are written straight into the result.\n"             //
    "\n"                                                                                                   //
    "Args:\n"                                                                                              //
    "  columns (Mapping[str, Strings | Sequence[str]]): Columns of equal length, keyed by name.\n"         //
    "  ensure_ascii (bool, optional): Write every non-ASCII codepoint as `\\uXXXX`, defaulting to True.\n" //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n"  //
    "Returns:\n"                                                                                           //
    "  bytes: The UTF-8 JSON text.\n"                                                                      //
    "Raises:\n"                                                                                            //
    "  ValueError: If the columns differ in length.\n"                                                     //
    "Example:\n"                                                                                           //
    "  >>> sz.to_json({'name': sz.Strings(['Ada', 'Linus']), 'city': ['London', 'Helsinki']})\n"           //
    "  b'[{\"name\":\"Ada\",\"city\":\"London\"},{\"name\":\"Linus\",\"city\":\"Helsinki\"}]'";

PyObject *module_to_json(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                         PyObject *args_names_tuple) {
    sz_unused_(self);
    if (positional_args_count > 2)
        return PyErr_Format(PyExc_TypeError, "to_json() takes at most 2 positional arguments, got %zd",
                            positional_args_count);
    PyObject *columns = positional_args_count > 0 ? args[0] : NULL;
    PyObject *ascii_object = positional_args_count > 1 ? args[1] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "columns") == 0        ? &columns
                          : PyUnicode_CompareWithASCIIString(key, "ensure_ascii") == 0 ? &ascii_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                                                                                       : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "to_json() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "to_json() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!columns) {
        PyErr_SetString(PyExc_TypeError, "to_json() missing required argument 'columns'");
        return NULL;
    }
    int const ensure_ascii = ascii_object ? PyObject_IsTrue(ascii_object) : 1;
    if (ensure_ascii < 0) return NULL;
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_allocator_t allocator;
    sz_status_t const allocator_status = sz_allocator_init_heap(&allocator);
    if (allocator_status != sz_success_k) {
        sz_py_raise_status(allocator_status, "to_json()");
        return NULL;
    }

    sz_strings_span_t *tapes;
    Py_ssize_t const columns_count = sz_py_escape_columns_(columns, "to_json()",
                                                           ensure_ascii ? sz_escape_json_ascii_k : sz_escape_json_k,
                                                           &allocator, capabilities, &tapes);
    if (columns_count < 0) return NULL;

    // A record reads {"name":"Ada","city":"London"}: every name sits in the frame before its cell
    sz_size_t const rows = columns_count ? tapes[1].count : 0, cells = (sz_size_t)columns_count;
    sz_strings_view_t *const escaped = (sz_strings_view_t *)PyMem_Malloc(cells * sizeof(sz_strings_view_t) +
                                                                         (cells + 1) * sizeof(sz_string_view_t) +
                                                                         cells * 6 + sz_py_tape_payload_(&tapes[0]));
    PyObject *result = escaped ? NULL : PyErr_NoMemory();
    if (escaped) {
        sz_string_view_t *const frames = (sz_string_view_t *)(escaped + cells);
        sz_ptr_t cursor = (sz_ptr_t)(frames + cells + 1);
        for (sz_size_t cell = 0; cell != cells; ++cell) {
            sz_strings_from_span(&tapes[cell + 1], &escaped[cell]);
            frames[cell].start = cursor;
            cursor = sz_py_append_(cursor, cell ? (sz_string_view_t) {"\",\"", 3} : (sz_string_view_t) {"{\"", 2});
            cursor = sz_py_append_(cursor, sz_py_tape_string_(&tapes[0], cell));
            cursor = sz_py_append_(cursor, (sz_string_view_t) {"\":\"", 3});
            frames[cell].length = (sz_size_t)(cursor - frames[cell].start);
        }
        frames[cells].start = "\"}", frames[cells].length = 2;
        sz_string_view_t const comma = {",", 1}, opening = {"[", 1}, closing = {"]", 1};
        result = sz_py_join_bytes_(escaped, cells, rows, frames, comma, opening, closing, capabilities, "to_json()");
        PyMem_Free(escaped);
    }
    sz_py_free_tapes_(tapes, columns_count + 1, &allocator);
    return result;
}

/** A CSV column with its name above its escaped cells, as the header is line zero. */
typedef struct sz_py_csv_column_t {
    sz_string_view_t name;
    sz_strings_span_t const *cells;
} sz_py_csv_column_t;

/** Reads the fields of the @c sz_py_csv_column_t @p handle on @p indices lines. */
static void sz_py_csv_column_(void const *handle, sz_size_t const *indices, sz_size_t count, sz_string_view_t *views) {
    sz_py_csv_column_t const *column = (sz_py_csv_column_t const *)handle;
    for (sz_size_t index = 0; index != count; ++index)
        views[index] = indices[index] ? sz_py_tape_string_(column->cells, indices[index] - 1) : column->name;
}

/** Reads the fields of the only column, quoting an empty one as `csv.writer` does, so that its line
 *  is not mistaken for one of no fields. */
static void sz_py_csv_lone_column_(void const *handle, sz_size_t const *indices, sz_size_t count,
                                   sz_string_view_t *views) {
    sz_py_csv_column_(handle, indices, count, views);
    for (sz_size_t index = 0; index != count; ++index)
        if (!views[index].length) views[index].start = "\"\"", views[index].length = 2;
}

char const doc_to_csv[] =                                                                                 //
    "to_csv(columns, lineterminator='\\n', *, capabilities=None) -> bytes\n"                              //
    "\n"                                                                                                  //
    "Dump columns of strings as CSV with a header row of their names, the bytes `csv.writer` writes\n"    //
    "with its default dialect, quoting a field only around a comma, a quote, CR or LF. Each column\n"     //
    "is escaped in one pass, and the rows are written straight into the result.\n"                        //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  columns (Mapping[str, Strings | Sequence[str]]): Columns of equal length, keyed by name.\n"        //
    "  lineterminator (str, optional): '\\n' or '\\r\\n' after every row, defaulting to '\\n'.\n"         //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  bytes: The UTF-8 CSV text.\n"                                                                      //
    "Raises:\n"                                                                                           //
    "  ValueError: If the columns differ in length, or the terminator is another string.\n"               //
    "Example:\n"                                                                                          //
    "  >>> sz.to_csv({'name': ['Ada', 'Grace \"Amazing\" Hopper'], 'city': ['London', 'New York']})\n"    //
    "  b'name,city\\nAda,London\\n\"Grace \"\"Amazing\"\" Hopper\",New York\\n'";

PyObject *module_to_csv(PyObject *self, PyObject *const *args, Py_ssize_t positional_args_count,
                        PyObject *args_names_tuple) {
    sz_unused_(self);
    if (positional_args_count > 2)
        return PyErr_Format(PyExc_TypeError, "to_csv() takes at most 2 positional arguments, got %zd",
                            positional_args_count);
    PyObject *columns = positional_args_count > 0 ? args[0] : NULL;
    PyObject *terminator_object = positional_args_count > 1 ? args[1] : NULL, *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "columns") == 0          ? &columns
                          : PyUnicode_CompareWithASCIIString(key, "lineterminator") == 0 ? &terminator_object
                          : PyUnicode_CompareWithASCIIString(key, "capabilities") == 0   ? &capabilities_object
                                                                                         : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "to_csv() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "to_csv() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!columns) {
        PyErr_SetString(PyExc_TypeError, "to_csv() missing required argument 'columns'");
        return NULL;
    }
    // Only CR and LF end a row, as every field holding either is quoted
    sz_string_view_t terminator = {"\n", 1};
    if (terminator_object && PyUnicode_Check(terminator_object) &&
        PyUnicode_CompareWithASCIIString(terminator_object, "\r\n") == 0)
        terminator.start = "\r\n", terminator.length = 2;
    else if (terminator_object &&
             (!PyUnicode_Check(terminator_object) || PyUnicode_CompareWithASCIIString(terminator_object, "\n") != 0)) {
        PyErr_SetString(PyExc_ValueError, "to_csv() lineterminator must be '\\n' or '\\r\\n'");
        return NULL;
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_allocator_t allocator;
    sz_status_t const allocator_status = sz_allocator_init_heap(&allocator);
    if (allocator_status != sz_success_k) {
        sz_py_raise_status(allocator_status, "to_csv()");
        return NULL;
    }

    sz_strings_span_t *tapes;
    Py_ssize_t const columns_count = sz_py_escape_columns_(columns, "to_csv()", sz_escape_csv_k, &allocator,
                                                           capabilities, &tapes);
    if (columns_count < 0) return NULL;

    sz_size_t const rows = columns_count ? tapes[1].count : 0, cells = (sz_size_t)columns_count;
    sz_py_csv_column_t *const handles = (sz_py_csv_column_t *)PyMem_Malloc(
        cells * (sizeof(sz_py_csv_column_t) + sizeof(sz_strings_view_t)) + (cells + 1) * sizeof(sz_string_view_t));
    PyObject *result = handles ? NULL : PyErr_NoMemory();
    if (handles) {
        sz_strings_view_t *const escaped = (sz_strings_view_t *)(handles + cells);
        sz_string_view_t *const frames = (sz_string_view_t *)(escaped + cells);
        for (sz_size_t cell = 0; cell != cells; ++cell) {
            handles[cell].name = sz_py_tape_string_(&tapes[0], cell), handles[cell].cells = &tapes[cell + 1];
            escaped[cell].layout = sz_strings_accessor_k, escaped[cell].count = rows + 1;
            escaped[cell].data.accessor.get = cells == 1 ? sz_py_csv_lone_column_ : sz_py_csv_column_;
            escaped[cell].data.accessor.handle = &handles[cell];
            frames[cell].start = ",", frames[cell].length = cell != 0;
        }
        frames[cells] = terminator;
        sz_string_view_t const nothing = {"", 0};
        result = sz_py_join_bytes_(escaped, cells, rows + 1, frames, nothing, nothing, nothing, capabilities,
                                   "to_csv()");
        PyMem_Free(handles);
    }
    sz_py_free_tapes_(tapes, columns_count + 1, &allocator);
    return result;
}

char const doc_String_join[] =                                                                            //
    "join(strings, /, *, capabilities=None) -> String\n"                                                  //
    "\n"                                                                                                  //
    "Concatenate the strings with this one between every two, like `str.join`, measuring the result\n"    //
    "first and writing it once.\n"                                                                        //
    "\n"                                                                                                  //
    "Args:\n"                                                                                             //
    "  strings (Strings or Iterable[str | bytes]): The strings to join; a `Strings` is read in place.\n"  //
    "  capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.\n" //
    "Returns:\n"                                                                                          //
    "  String: The joined bytes.\n"                                                                       //
    "Example:\n"                                                                                          //
    "  >>> str(sz.String(', ').join(['Ada', 'Grace', 'Linus']))\n"                                        //
    "  'Ada, Grace, Linus'";

PyObject *String_join(String *self, PyObject *const *args, Py_ssize_t positional_args_count,
                      PyObject *args_names_tuple) {
    if (positional_args_count != 1)
        return PyErr_Format(PyExc_TypeError, "join() takes exactly 1 positional argument, got %zd",
                            positional_args_count);
    PyObject *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        if (PyUnicode_CompareWithASCIIString(key, "capabilities") != 0)
            return PyErr_Format(PyExc_TypeError, "join() got an unexpected keyword argument '%U'", key);
        capabilities_object = args[positional_args_count + keyword_index];
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;
    sz_strings_view_t strings;
    PyObject *snapshot;
    sz_string_view_t *views;
    if (sz_py_export_strings_like(args[0], "join()", &strings, &snapshot, &views) != 0) return NULL;

    sz_string_view_t const frames[2] = {{"", 0}, {"", 0}}, nothing = {"", 0};
    PyObject *const joined = sz_py_join_bytes_(&strings, 1, strings.count, frames, self->memory, nothing, nothing,
                                               capabilities, "join()");
    PyMem_Free(views);
    Py_XDECREF(snapshot);
    String *const result = joined ? String_alloc_() : NULL;
    if (!result) {
        Py_XDECREF(joined);
        return NULL;
    }
    result->parent = joined, result->memory.start = PyBytes_AS_STRING(joined);
    result->memory.length = (sz_size_t)PyBytes_GET_SIZE(joined);
    return (PyObject *)result;
}

#pragma endregion Dumping
