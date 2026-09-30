/**
 *  @file python/stringzilla/utf8_boundaries.c
 *  @author Ash Vardanian
 *  @date August 7, 2026
 *  @brief The shared iterator machinery behind the four UAX segmenters.
 */
#include "stringzilla.h"

PyObject *Utf8Boundaries_make_(PyTypeObject *type, char const *name, sz_py_segmenter_t segmenter, PyObject *const *args,
                               Py_ssize_t positional_args_count, PyObject *args_names_tuple) {

    if (positional_args_count != 1) {
        PyErr_Format(PyExc_TypeError, "%s() takes one positional argument, got %zd", name, positional_args_count);
        return NULL;
    }
    PyObject *const text_obj = args[0];
    PyObject *capabilities_object = NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        if (PyUnicode_CompareWithASCIIString(key, "capabilities") != 0) {
            PyErr_Format(PyExc_TypeError, "%s() got an unexpected keyword argument '%U'", name, key);
            return NULL;
        }
        capabilities_object = args[positional_args_count + keyword_index];
    }
    sz_capability_t capabilities;
    if (sz_py_export_capabilities(capabilities_object, &capabilities) != 0) return NULL;

    sz_string_view_t text_view;
    if (PyObject_TypeCheck(text_obj, &StrType)) {
        Str *str_obj = (Str *)text_obj;
        text_view = str_obj->memory;
    }
    else if (PyUnicode_Check(text_obj)) {
        Py_ssize_t signed_length;
        text_view.start = PyUnicode_AsUTF8AndSize(text_obj, &signed_length);
        if (!text_view.start) return NULL;
        text_view.length = (sz_size_t)signed_length;
    }
    else if (PyBytes_Check(text_obj)) {
        text_view.start = PyBytes_AS_STRING(text_obj);
        text_view.length = (sz_size_t)PyBytes_GET_SIZE(text_obj);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Expected str, bytes, or Str");
        return NULL;
    }

    Utf8Boundaries *iter = PyObject_New(Utf8Boundaries, type);
    if (!iter) return PyErr_NoMemory();

    iter->text_obj = text_obj;
    Py_INCREF(text_obj);
    iter->start = text_view.start;
    iter->end = text_view.start + text_view.length;
    iter->segmenter = segmenter;
    iter->capabilities = capabilities;
    iter->batch_count = 0;
    iter->batch_index = 0;

    return (PyObject *)iter;
}

PyObject *Utf8Boundaries_next_(Utf8Boundaries *self) {
    // Refill the inline batch when drained. UAX segmentation never yields zero-length segments, so the
    // `skip_empty` option is a no-op here and needs no special handling. Batch offsets are always relative to
    // `self->start` (the not-yet-segmented suffix start); forward only, so `start` advances past each batch.
    if (self->batch_index >= self->batch_count) {
        if (self->start >= self->end) return NULL;
        sz_size_t consumed = 0;
        sz_status_t const status = self->segmenter(
            self->start, (sz_size_t)(self->end - self->start), self->batch_starts, self->batch_lengths,
            sz_iterators_default_steps_k, &self->batch_count, &consumed, self->capabilities, NULL);
        self->batch_index = 0;
        if (status != sz_success_k) {
            self->batch_count = 0;
            sz_py_raise_status(status, "__next__()");
            return NULL;
        }
        if (self->batch_count == 0) return NULL;
    }

    sz_size_t i = self->batch_index++;
    sz_cptr_t segment_start = self->start + self->batch_starts[i];
    sz_size_t segment_len = self->batch_lengths[i];

    // Once the batch is drained, move the suffix start to the last buffered segment's end (a UAX boundary) so
    // the next refill resumes there.
    if (self->batch_index >= self->batch_count) {
        sz_size_t last = self->batch_count - 1;
        self->start += self->batch_starts[last] + self->batch_lengths[last]; // last segment's end
    }

    Str *result_obj = Str_alloc_();
    if (result_obj == NULL) return PyErr_NoMemory();

    result_obj->memory.start = segment_start;
    result_obj->memory.length = segment_len;
    result_obj->parent = self->text_obj;
    Py_INCREF(self->text_obj);

    return (PyObject *)result_obj;
}

void Utf8Boundaries_dealloc_(Utf8Boundaries *self) {
    Py_XDECREF(self->text_obj);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *Utf8Boundaries_iter_(PyObject *self) {
    Py_INCREF(self);
    return self;
}
