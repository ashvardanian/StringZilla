/**
 *  @brief The `Str`/`Strs` free-lists and the argument-export helpers.
 *  @file python/stringzilla/shared.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/** @brief Reach the per-interpreter free-list state, or @c NULL before registration / during teardown. */
stringzilla_state_t *stringzilla_state_(void) {
    PyObject *module = PyState_FindModule(&stringzilla_module);
    return module ? (stringzilla_state_t *)PyModule_GetState(module) : NULL;
}

/** @brief The dead @c Strs header's @c data union storage doubles as the intrusive @c next link. */
Strs **Strs_freelist_next_(Strs *node) { return (Strs **)&node->data; }

/** @brief Allocate a blank @c Str header, reusing a cached one from the free-list when available. */
Str *Str_alloc_(void) {
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->str_freelist_head) {
        Str *self = state->str_freelist_head;
        state->str_freelist_head = (Str *)self->parent; // Unlink (the dead `parent` held `next`).
        state->str_freelist_count--;
        sz_freelist_unlock_(state);
        _Py_NewReference((PyObject *)self); // Refcount back to 1; non-GC type, so no retracking.
        self->parent = NULL;
        self->memory.start = NULL;
        self->memory.length = 0;
        return self;
    }
    if (state) sz_freelist_unlock_(state);
    return (Str *)StrType.tp_alloc(&StrType, 0); // Fresh header (zero-initialized); NULL propagates on OOM.
}

/** @brief Allocate a blank @c Strs header, reusing a cached one from the free-list when available. */
Strs *Strs_alloc_(void) {
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->strs_freelist_head) {
        Strs *self = state->strs_freelist_head;
        state->strs_freelist_head = *Strs_freelist_next_(self); // Unlink (the dead `data` held `next`).
        state->strs_freelist_count--;
        sz_freelist_unlock_(state);
        _Py_NewReference((PyObject *)self); // Refcount back to 1; non-GC type, so no retracking.
        // Restore `tp_alloc`'s zero-init so a caller's early error path deallocs safely (the dealloc
        // `switch` reads `layout`/`data`); `STRS_U32_TAPE_VIEW` with a NULL parent is a no-op to free.
        self->layout = STRS_U32_TAPE_VIEW;
        memset(&self->data, 0, sizeof(self->data));
        return self;
    }
    if (state) sz_freelist_unlock_(state);
    return (Strs *)StrsType.tp_alloc(&StrsType, 0); // Fresh header (zero-initialized); NULL propagates on OOM.
}

/**
 *  @brief  Allocates an empty `Strs` in the fragmented layout. Consolidates the count-zero
 *          initialization otherwise inlined across slicing, splitting, and reordering paths.
 *  @return A new empty `Strs`, or `NULL` with a Python exception set on allocation failure.
 */
Strs *strs_make_empty_fragmented_(void) {
    Strs *result = Strs_alloc_();
    if (result == NULL) return (Strs *)PyErr_NoMemory();
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = 0;
    result->data.fragmented.spans = NULL;
    result->data.fragmented.parent = NULL;
    sz_memory_allocator_init_default(&result->data.fragmented.allocator);
    return result;
}

/**
 *  @brief  Helper function to check if a Python object represents a mutable buffer.
 *          Returns sz_true_k if the object is mutable (can be written to), sz_false_k if immutable.
 *          Sets a Python exception if immutable.
 */
sz_bool_t sz_py_is_mutable(PyObject *object) {
    if (PyUnicode_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "str objects are immutable (use bytearray instead)");
        return sz_false_k;
    }
    else if (PyBytes_Check(object)) {
        PyErr_SetString(PyExc_TypeError, "bytes objects are immutable (use bytearray instead)");
        return sz_false_k;
    }
    else if (PyMemoryView_Check(object)) {
        Py_buffer *view = PyMemoryView_GET_BUFFER(object);
        if (view->readonly) {
            PyErr_SetString(PyExc_TypeError, "memoryview is read-only");
            return sz_false_k;
        }
    }
    // Everything else is optimistically considered mutable
    return sz_true_k;
}

/**
 *  @brief  Parses an optional `start`/`end` index argument with CPython slice semantics.
 *          A `NULL` or `None` object yields @p default_index; an out-of-`ssize_t` value is clamped to
 *          the representable range (via `PyNumber_AsSsize_t` with a `NULL` exception) instead of raising;
 *          a non-integer leaves a `TypeError` set.
 *  @return 1 on success (result written to @p result_out), 0 if a Python exception was set.
 */
int sz_py_export_optional_index(PyObject *index_obj, Py_ssize_t default_index, Py_ssize_t *result_out) {
    if (index_obj == NULL || index_obj == Py_None) {
        *result_out = default_index;
        return 1;
    }
    *result_out = PyNumber_AsSsize_t(index_obj, NULL);
    return !(*result_out == -1 && PyErr_Occurred());
}

/**
 *  @brief  Helper function to export a Python string-like object into a `sz_string_view_t`.
 *          On failure, sets a Python exception and returns 0.
 */
SZ_API_RUNTIME sz_bool_t sz_py_export_string_like(PyObject *object, sz_cptr_t *start, sz_size_t *length) {
    if (PyUnicode_Check(object)) {
        // Handle Python `str` object
        Py_ssize_t signed_length;
        *start = PyUnicode_AsUTF8AndSize(object, &signed_length);
        *length = (sz_size_t)signed_length;
        return 1;
    }
    else if (PyBytes_Check(object)) {
        // Handle Python `bytes` object
        // https://docs.python.org/3/c-api/bytes.html
        Py_ssize_t signed_length;
        if (PyBytes_AsStringAndSize(object, (sz_ptr_t *)start, &signed_length) == -1) {
            PyErr_SetString(PyExc_ValueError, "Couldn't access `bytes` buffer internals");
            return 0;
        }
        *length = (sz_size_t)signed_length;
        return 1;
    }
    else if (PyByteArray_Check(object)) {
        // Handle Python mutable `bytearray` object
        // https://docs.python.org/3/c-api/bytearray.html
        *start = PyByteArray_AS_STRING(object);
        *length = PyByteArray_GET_SIZE(object);
        return 1;
    }
    else if (PyObject_TypeCheck(object, &StrType)) {
        Str *str = (Str *)object;
        *start = str->memory.start;
        *length = str->memory.length;
        return 1;
    }
    else if (PyObject_TypeCheck(object, &FileType)) {
        File *file = (File *)object;
        *start = file->memory.start;
        *length = file->memory.length;
        return 1;
    }
    else if (PyMemoryView_Check(object)) {
        // Handle Python `memoryview` object
        // https://docs.python.org/3/c-api/memoryview.html
        // https://docs.python.org/3/c-api/buffer.html#c.Py_buffer
        Py_buffer *view = PyMemoryView_GET_BUFFER(object);
        // Make sure we are dealing with single-byte integral representations
        if (view->itemsize != 1) {
            PyErr_SetString(PyExc_ValueError, "Only single-byte integral types are supported");
            return 0;
        }
        // Let's make sure the data is contiguous.
        // This can be a bit trickier for high-dimensional arrays, but CPython has a built-in function for that.
        // The flag 'C' stands for C-style-contiguous, which means that the last dimension is contiguous.
        // The flag 'F' stands for Fortran-style-contiguous, which means that the first dimension is contiguous.
        // The flag 'A' stands for any-contiguous, which only means there are no gaps between elements.
        // For byte-level processing that's all we need.
        if (!PyBuffer_IsContiguous(view, 'A')) {
            PyErr_SetString(PyExc_ValueError, "The array must be contiguous");
            return 0;
        }

        *start = (sz_cptr_t)view->buf;
        *length = (sz_size_t)view->len;
        return 1;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Unsupported argument layout");
        return 0;
    }
}

/**
 *  @brief  Helper function to wrap the current exception with a custom prefix message.
 *          A example is augmenting the argument parsing error with the name of the variable
 *          that didn't pass the validation.
 */
void wrap_current_exception(sz_cptr_t comment) {
    // ? Prior to Python 3.12 we need to fetch and restore the exception state using
    // ? `PyErr_Fetch` and `PyErr_Restore` to avoid overwriting the current exception.
    // ? After Python 3.12 we can use `PyErr_GetRaisedException` and `PyErr_SetRaisedException`.
    sz_unused_(comment);
}
