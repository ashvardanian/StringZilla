/**
 *  @brief The shared iterator machinery behind the four UAX segmenters.
 *  @file python/stringzilla/utf8_boundaries.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

PyObject *Utf8Boundaries_make_(PyTypeObject *type, PyObject *text_obj, sz_utf8_segmenter_t kernel) {

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
    iter->kernel = kernel;
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
        self->batch_count = self->kernel(self->start, (sz_size_t)(self->end - self->start), self->batch_starts,
                                         self->batch_lengths, sz_iterators_default_steps_k, &consumed);
        self->batch_index = 0;
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
