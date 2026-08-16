/**
 *  @brief The `Strs` collection - tape and fragmented layouts, batch operations.
 *  @file python/stringzilla/strs.c
 *  @author Ash Vardanian
 */
#include "stringzilla.h"

/**
 *  @brief Arrow C Data Interface structure for an array schema.
 *  @see https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
 */
struct ArrowSchema {
    char const *format;
    char const *name;
    char const *metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema **children;
    struct ArrowSchema *dictionary;
    void (*release)(struct ArrowSchema *);
    void *private_data;
};

/**
 *  @brief Arrow C Data Interface structure for an array content.
 *  @see https://arrow.apache.org/docs/format/CDataInterface.html#structure-definitions
 */
struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    void const **buffers;
    struct ArrowArray **children;
    struct ArrowArray *dictionary;
    void (*release)(struct ArrowArray *);
    void *private_data;
};

sz_cptr_t Strs_get_start_(void const *handle, sz_size_t i) {
    Strs *strs = (Strs *)handle;
    switch (strs->layout) {
    case STRS_U32_TAPE: return strs->data.u32_tape.data + strs->data.u32_tape.offsets[i];
    case STRS_U32_TAPE_VIEW: return strs->data.u32_tape_view.data + strs->data.u32_tape_view.offsets[i];
    case STRS_U64_TAPE: return strs->data.u64_tape.data + strs->data.u64_tape.offsets[i];
    case STRS_U64_TAPE_VIEW: return strs->data.u64_tape_view.data + strs->data.u64_tape_view.offsets[i];
    case STRS_FRAGMENTED: return strs->data.fragmented.spans[i].start;
    }
    return NULL;
}

sz_size_t Strs_get_length_(void const *handle, sz_size_t i) {
    Strs *strs = (Strs *)handle;
    switch (strs->layout) {
    case STRS_U32_TAPE: return strs->data.u32_tape.offsets[i + 1] - strs->data.u32_tape.offsets[i];
    case STRS_U32_TAPE_VIEW: return strs->data.u32_tape_view.offsets[i + 1] - strs->data.u32_tape_view.offsets[i];
    case STRS_U64_TAPE: return strs->data.u64_tape.offsets[i + 1] - strs->data.u64_tape.offsets[i];
    case STRS_U64_TAPE_VIEW: return strs->data.u64_tape_view.offsets[i + 1] - strs->data.u64_tape_view.offsets[i];
    case STRS_FRAGMENTED: return strs->data.fragmented.spans[i].length;
    }
    return 0;
}

/**
 *  @brief  Helper function to export a `Strs` or similar sequence objects into a `sz_sequence_t`.
 */
sz_bool_t sz_py_export_strings_as_sequence(PyObject *object, sz_sequence_t *sequence) {
    if (!sequence) return sz_false_k;

    if (PyObject_TypeCheck(object, &StrsType)) {
        Strs *strs = (Strs *)object;

        // Every layout, not just the reordered one. A caller holding a tape normally prefers the tape
        // exporters, which hand the kernel its offsets directly - but an entry point with no tape overload,
        // like the multi-pattern engine's needle set, has nowhere else to go and must get a sequence here.
        sequence->handle = strs;
        sequence->count = (sz_size_t)Strs_len(strs);
        sequence->get_start = Strs_get_start_;
        sequence->get_length = Strs_get_length_;
        return sz_true_k;
    }

    return sz_false_k;
}

/**
 *  @brief  Helper function to export a `Strs` object into `sz_sequence_u32tape_t` components.
 */
sz_bool_t sz_py_export_strings_as_u32tape(PyObject *object, sz_cptr_t *data, sz_u32_t const **offsets,
                                          sz_size_t *count) {

    if (!data || !offsets || !count) return sz_false_k;
    if (!PyObject_TypeCheck(object, &StrsType)) return sz_false_k;
    Strs *strs = (Strs *)object;

    if (strs->layout == STRS_U32_TAPE) {
        *data = strs->data.u32_tape.data;
        *offsets = strs->data.u32_tape.offsets;
        *count = strs->data.u32_tape.count;
        return sz_true_k;
    }
    else if (strs->layout == STRS_U32_TAPE_VIEW) {
        *data = strs->data.u32_tape_view.data;
        *offsets = strs->data.u32_tape_view.offsets;
        *count = strs->data.u32_tape_view.count;
        return sz_true_k;
    }
    else { return sz_false_k; }
}

/**
 *  @brief  Helper function to export a `Strs` object into `sz_sequence_u64tape_t` components.
 */
sz_bool_t sz_py_export_strings_as_u64tape(PyObject *object, sz_cptr_t *data, sz_u64_t const **offsets,
                                          sz_size_t *count) {

    if (!data || !offsets || !count) return sz_false_k;
    if (!PyObject_TypeCheck(object, &StrsType)) return sz_false_k;
    Strs *strs = (Strs *)object;

    if (strs->layout == STRS_U64_TAPE) {
        *data = strs->data.u64_tape.data;
        *offsets = strs->data.u64_tape.offsets;
        *count = strs->data.u64_tape.count;
        return sz_true_k;
    }
    else if (strs->layout == STRS_U64_TAPE_VIEW) {
        *data = strs->data.u64_tape_view.data;
        *offsets = strs->data.u64_tape_view.offsets;
        *count = strs->data.u64_tape_view.count;
        return sz_true_k;
    }
    else { return sz_false_k; }
}

static sz_bool_t sz_py_replace_u32_tape_allocator(Strs *strs, sz_memory_allocator_t *old_allocator,
                                                  sz_memory_allocator_t *allocator) {
    struct u32_tape_t *data = &strs->data.u32_tape;
    sz_assert_(data->offsets && "Expected offsets to be allocated");

    sz_size_t const string_data_size = (sz_size_t)data->offsets[data->count];
    sz_size_t const offsets_size = (data->count + 1) * sizeof(sz_u32_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data = string_data_size ? (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle)
                                                : (sz_ptr_t)NULL;
    if (string_data_size && !new_string_data) return sz_false_k;
    memcpy(new_string_data, data->data, string_data_size);

    // Allocate new offsets array
    sz_u32_t *new_offsets = offsets_size ? (sz_u32_t *)allocator->allocate(offsets_size, allocator->handle)
                                         : (sz_u32_t *)NULL;
    if (offsets_size && !new_offsets) {
        if (string_data_size) allocator->free(new_string_data, string_data_size, allocator->handle);
        return sz_false_k;
    }
    memcpy(new_offsets, data->offsets, offsets_size);

    // Free old memory with old allocator (tapes always own their data)
    old_allocator->free(data->data, string_data_size, old_allocator->handle);
    old_allocator->free(data->offsets, offsets_size, old_allocator->handle);

    // Update pointers and allocator
    data->data = new_string_data;
    data->offsets = new_offsets;
    data->allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_u64_tape_allocator(Strs *strs, sz_memory_allocator_t *old_allocator,
                                                  sz_memory_allocator_t *allocator) {
    struct u64_tape_t *data = &strs->data.u64_tape;
    sz_assert_(data->offsets && "Expected offsets to be allocated");

    sz_size_t string_data_size = (sz_size_t)data->offsets[data->count];
    sz_size_t offsets_size = (data->count + 1) * sizeof(sz_u64_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data = string_data_size ? (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle)
                                                : (sz_ptr_t)NULL;
    if (string_data_size && !new_string_data) return sz_false_k;
    memcpy(new_string_data, data->data, string_data_size);

    // Allocate new offsets array
    sz_u64_t *new_offsets = offsets_size ? (sz_u64_t *)allocator->allocate(offsets_size, allocator->handle)
                                         : (sz_u64_t *)NULL;
    if (offsets_size && !new_offsets) {
        if (string_data_size) allocator->free(new_string_data, string_data_size, allocator->handle);
        return sz_false_k;
    }
    memcpy(new_offsets, data->offsets, offsets_size);

    // Free old memory with old allocator (tapes always own their data)
    old_allocator->free(data->data, string_data_size, old_allocator->handle);
    old_allocator->free(data->offsets, offsets_size, old_allocator->handle);

    // Update pointers and allocator
    data->data = new_string_data;
    data->offsets = new_offsets;
    data->allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_u32_tape_view_allocator(Strs *strs, sz_memory_allocator_t *allocator) {
    // Convert view to tape by copying the data
    struct u32_tape_view_t *view = &strs->data.u32_tape_view;
    sz_u32_t const slice_start_offset = view->offsets[0];
    sz_size_t const string_data_size = (sz_size_t)(view->offsets[view->count] - slice_start_offset);
    sz_size_t const offsets_size = (view->count + 1) * sizeof(sz_u32_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data = NULL;
    if (string_data_size > 0) {
        new_string_data = (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle);
        if (!new_string_data) return sz_false_k;
        memcpy(new_string_data, view->data + slice_start_offset, string_data_size);
    }

    // Allocate new offsets array and adjust to be relative to slice start
    sz_u32_t *new_offsets = NULL;
    if (offsets_size > 0) {
        new_offsets = (sz_u32_t *)allocator->allocate(offsets_size, allocator->handle);
        if (!new_offsets) {
            if (string_data_size > 0) allocator->free(new_string_data, string_data_size, allocator->handle);
            return sz_false_k;
        }
        for (sz_size_t i = 0; i <= view->count; ++i) new_offsets[i] = view->offsets[i] - slice_start_offset;
    }

    // Release parent reference if any
    Py_XDECREF(view->parent);

    // Convert to tape layout
    strs->layout = STRS_U32_TAPE;
    strs->data.u32_tape.count = view->count;
    strs->data.u32_tape.data = new_string_data;
    strs->data.u32_tape.offsets = new_offsets;
    strs->data.u32_tape.allocator = *allocator;
    return sz_true_k;
}

static sz_bool_t sz_py_replace_u64_tape_view_allocator(Strs *strs, sz_memory_allocator_t *allocator) {
    // Convert view to tape by copying the data
    struct u64_tape_view_t *view = &strs->data.u64_tape_view;
    sz_u64_t const slice_start_offset = view->offsets[0];
    sz_size_t const string_data_size = (sz_size_t)(view->offsets[view->count] - slice_start_offset);
    sz_size_t const offsets_size = (view->count + 1) * sizeof(sz_u64_t);

    // Allocate new string data with new allocator
    sz_ptr_t new_string_data = NULL;
    if (string_data_size > 0) {
        new_string_data = (sz_ptr_t)allocator->allocate(string_data_size, allocator->handle);
        if (!new_string_data) return sz_false_k;
        memcpy(new_string_data, view->data + slice_start_offset, string_data_size);
    }

    // Allocate new offsets array and adjust to be relative to slice start
    sz_u64_t *new_offsets = NULL;
    if (offsets_size > 0) {
        new_offsets = (sz_u64_t *)allocator->allocate(offsets_size, allocator->handle);
        if (!new_offsets) {
            if (string_data_size > 0) allocator->free(new_string_data, string_data_size, allocator->handle);
            return sz_false_k;
        }
        for (sz_size_t i = 0; i <= view->count; ++i) new_offsets[i] = view->offsets[i] - slice_start_offset;
    }

    // Release parent reference if any
    Py_XDECREF(view->parent);

    // Convert to tape layout
    strs->layout = STRS_U64_TAPE;
    strs->data.u64_tape.count = view->count;
    strs->data.u64_tape.data = new_string_data;
    strs->data.u64_tape.offsets = new_offsets;
    strs->data.u64_tape.allocator = *allocator;
    return sz_true_k;
}

/** @brief  Consolidates a fragmented `Strs` into a single tape under a new allocator. */
static sz_bool_t sz_py_replace_fragmented_allocator(Strs *strs, sz_memory_allocator_t *old_allocator,
                                                    sz_memory_allocator_t *allocator) {
    struct fragmented_t *fragmented = &strs->data.fragmented;
    sz_assert_(fragmented->spans && "Expected spans to be allocated");

    // Calculate total size needed for consolidated tape
    sz_size_t total_bytes = 0;
    for (sz_size_t i = 0; i < fragmented->count; i++) total_bytes += fragmented->spans[i].length;

    // Choose 32-bit or 64-bit tape based on size
    sz_bool_t use_64bit = total_bytes >= UINT32_MAX;

    // Skip allocation if there's no data to allocate (empty strings case)
    if (total_bytes == 0) {
        // Convert to empty tape layout
        old_allocator->free(fragmented->spans, fragmented->count * sizeof(sz_string_view_t), old_allocator->handle);
        Py_XDECREF(fragmented->parent);

        strs->layout = STRS_U32_TAPE;
        strs->data.u32_tape.count = fragmented->count;
        strs->data.u32_tape.data = NULL;
        strs->data.u32_tape.offsets = NULL;
        strs->data.u32_tape.allocator = *allocator;
        return sz_true_k;
    }

    // Allocate consolidated data buffer and offsets array
    sz_ptr_t new_data = (sz_ptr_t)allocator->allocate(total_bytes, allocator->handle);
    if (!new_data) return sz_false_k;

    if (use_64bit) {
        sz_u64_t *new_offsets = (sz_u64_t *)allocator->allocate((fragmented->count + 1) * sizeof(sz_u64_t),
                                                                allocator->handle);
        if (!new_offsets) {
            allocator->free(new_data, total_bytes, allocator->handle);
            return sz_false_k;
        }

        // Copy fragmented data into consolidated buffer
        sz_size_t current_offset = 0;
        new_offsets[0] = 0;
        for (sz_size_t i = 0; i < fragmented->count; i++) {
            sz_size_t len = fragmented->spans[i].length;
            if (len > 0) { memcpy(new_data + current_offset, fragmented->spans[i].start, len); }
            current_offset += len;
            new_offsets[i + 1] = current_offset;
        }

        // Free old fragmented data and convert to 64-bit tape
        old_allocator->free(fragmented->spans, fragmented->count * sizeof(sz_string_view_t), old_allocator->handle);
        Py_XDECREF(fragmented->parent);

        strs->layout = STRS_U64_TAPE;
        strs->data.u64_tape.count = fragmented->count;
        strs->data.u64_tape.data = new_data;
        strs->data.u64_tape.offsets = new_offsets;
        strs->data.u64_tape.allocator = *allocator;
    }
    else {
        sz_u32_t *new_offsets = (sz_u32_t *)allocator->allocate((fragmented->count + 1) * sizeof(sz_u32_t),
                                                                allocator->handle);
        if (!new_offsets) {
            allocator->free(new_data, total_bytes, allocator->handle);
            return sz_false_k;
        }

        // Copy fragmented data into consolidated buffer
        sz_size_t current_offset = 0;
        new_offsets[0] = 0;
        for (sz_size_t i = 0; i < fragmented->count; i++) {
            sz_size_t len = fragmented->spans[i].length;
            if (len > 0) { memcpy(new_data + current_offset, fragmented->spans[i].start, len); }
            current_offset += len;
            // Ensure we don't overflow 32-bit offset
            if (current_offset > UINT32_MAX) {
                allocator->free(new_data, total_bytes, allocator->handle);
                allocator->free(new_offsets, (fragmented->count + 1) * sizeof(sz_u32_t), allocator->handle);
                return sz_false_k;
            }
            new_offsets[i + 1] = (sz_u32_t)current_offset;
        }

        // Free old fragmented data and convert to 32-bit tape
        old_allocator->free(fragmented->spans, fragmented->count * sizeof(sz_string_view_t), old_allocator->handle);
        Py_XDECREF(fragmented->parent);

        strs->layout = STRS_U32_TAPE;
        strs->data.u32_tape.count = fragmented->count;
        strs->data.u32_tape.data = new_data;
        strs->data.u32_tape.offsets = new_offsets;
        strs->data.u32_tape.allocator = *allocator;
    }
    return sz_true_k;
}

/**
 *  @brief  Helper function to replace the memory allocator in a `Strs` object.
 *          This reallocates existing string data using the new allocator.
 *
 *  This may change the layout of the `Strs` layout:
 *  - `STRS_U32_TAPE_VIEW` becomes `STRS_U32_TAPE`.
 *  - `STRS_U64_TAPE_VIEW` becomes `STRS_U64_TAPE`.
 *  - `STRS_U32_TAPE` remains, if the allocator is different.
 *  - `STRS_U64_TAPE` remains, if the allocator is different.
 *  - `STRS_FRAGMENTED` becomes a `STRS_U32_TAPE` or `STRS_U64_TAPE` depending on the content size.
 */
sz_bool_t sz_py_replace_strings_allocator(PyObject *object, sz_memory_allocator_t *allocator) {
    if (!object || !allocator) return sz_false_k;
    if (!PyObject_TypeCheck(object, &StrsType)) return sz_false_k;

    Strs *strs = (Strs *)object;

    // Get the current allocator based on layout
    sz_memory_allocator_t old_allocator;
    switch (strs->layout) {
    case STRS_U32_TAPE: old_allocator = strs->data.u32_tape.allocator; break;
    case STRS_U64_TAPE: old_allocator = strs->data.u64_tape.allocator; break;
    case STRS_FRAGMENTED: old_allocator = strs->data.fragmented.allocator; break;
    case STRS_U32_TAPE_VIEW:
    case STRS_U64_TAPE_VIEW:
        // Traverse parent chain until we find an allocator
        {
            Strs *up = strs;
            while (up && (up->layout == STRS_U32_TAPE_VIEW || up->layout == STRS_U64_TAPE_VIEW)) {
                PyObject *parent = (up->layout == STRS_U32_TAPE_VIEW) ? up->data.u32_tape_view.parent
                                                                      : up->data.u64_tape_view.parent;
                if (!parent || !PyObject_TypeCheck(parent, &StrsType)) break;
                up = (Strs *)parent;
            }

            // Extract allocator from the owning layout we found
            if (up && up->layout == STRS_U32_TAPE) { old_allocator = up->data.u32_tape.allocator; }
            else if (up && up->layout == STRS_U64_TAPE) { old_allocator = up->data.u64_tape.allocator; }
            else if (up && up->layout == STRS_FRAGMENTED) { old_allocator = up->data.fragmented.allocator; }
            else { sz_memory_allocator_init_default(&old_allocator); } // Final fallback
        }
        break;
    default: sz_memory_allocator_init_default(&old_allocator); break;
    }

    // Check if the allocators are the same - no need to reallocate
    if (sz_memory_allocator_equal(&old_allocator, allocator)) return sz_true_k;

    // Handle different Strs layouts using dedicated functions
    switch (strs->layout) {
    case STRS_U32_TAPE: return sz_py_replace_u32_tape_allocator(strs, &old_allocator, allocator);
    case STRS_U64_TAPE: return sz_py_replace_u64_tape_allocator(strs, &old_allocator, allocator);
    case STRS_U32_TAPE_VIEW: return sz_py_replace_u32_tape_view_allocator(strs, allocator);
    case STRS_U64_TAPE_VIEW: return sz_py_replace_u64_tape_view_allocator(strs, allocator);
    case STRS_FRAGMENTED: return sz_py_replace_fragmented_allocator(strs, &old_allocator, allocator);
    }

    return sz_false_k; // Should never reach here
}

typedef void (*get_string_at_offset_t)(Strs *, Py_ssize_t, Py_ssize_t, PyObject **, sz_cptr_t *, sz_size_t *);

static void str_at_offset_u32_tape(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                   PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u32_t start_offset = strs->data.u32_tape.offsets[i];
    sz_u32_t end_offset = strs->data.u32_tape.offsets[i + 1];
    *start = strs->data.u32_tape.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs; // Tapes own their data
}

static void str_at_offset_u32_tape_view(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                        PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u32_t start_offset = strs->data.u32_tape_view.offsets[i];
    sz_u32_t end_offset = strs->data.u32_tape_view.offsets[i + 1];
    *start = strs->data.u32_tape_view.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs->data.u32_tape_view.parent;
}

static void str_at_offset_u64_tape(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                   PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u64_t start_offset = strs->data.u64_tape.offsets[i];
    sz_u64_t end_offset = strs->data.u64_tape.offsets[i + 1];
    *start = strs->data.u64_tape.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs; // Tapes own their data
}

static void str_at_offset_u64_tape_view(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                        PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    // Apache Arrow format: offsets[i] to offsets[i+1] defines string i
    sz_u64_t start_offset = strs->data.u64_tape_view.offsets[i];
    sz_u64_t end_offset = strs->data.u64_tape_view.offsets[i + 1];
    *start = strs->data.u64_tape_view.data + start_offset;
    *length = end_offset - start_offset;
    *memory_owner = strs->data.u64_tape_view.parent;
}

static void str_at_offset_fragmented(Strs *strs, Py_ssize_t i, Py_ssize_t count, //
                                     PyObject **memory_owner, sz_cptr_t *start, sz_size_t *length) {
    *start = strs->data.fragmented.spans[i].start;
    *length = strs->data.fragmented.spans[i].length;
    *memory_owner = strs->data.fragmented.parent;
}

static get_string_at_offset_t str_at_offset_getter(Strs *strs) {
    switch (strs->layout) {
    case STRS_U32_TAPE: return str_at_offset_u32_tape;
    case STRS_U32_TAPE_VIEW: return str_at_offset_u32_tape_view;
    case STRS_U64_TAPE: return str_at_offset_u64_tape;
    case STRS_U64_TAPE_VIEW: return str_at_offset_u64_tape_view;
    case STRS_FRAGMENTED: return str_at_offset_fragmented;
    default:
        // Unsupported layout
        PyErr_SetString(PyExc_TypeError, "Unsupported layout for conversion");
        return NULL;
    }
}

/**
 *  @brief  Ensures the Strs is in a tape layout (not fragmented).
 *          Converts FRAGMENTED to TAPE if necessary.
 *  @return 1 on success, 0 on failure (sets Python exception).
 */
static int Strs_ensure_tape_layout(Strs *self) {
    if (self->layout != STRS_FRAGMENTED) return 1; // Already in tape layout

    // Get the default allocator
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_default(&allocator);

    // Convert fragmented to tape
    if (!sz_py_replace_fragmented_allocator(self, &self->data.fragmented.allocator, &allocator)) {
        PyErr_SetString(PyExc_MemoryError, "Failed to convert fragmented layout to tape");
        return 0;
    }
    return 1;
}

static PyObject *Strs_get_tape(Strs *self, void *closure) {
    // Ensure we're in tape layout
    if (!Strs_ensure_tape_layout(self)) return NULL;
    // Return self to allow chaining: strs.tape.tape_address
    Py_INCREF(self);
    return (PyObject *)self;
}

static PyObject *Strs_get_offsets_are_large(Strs *self, void *closure) {
    if (!Strs_ensure_tape_layout(self)) return NULL;

    switch (self->layout) {
    case STRS_U32_TAPE:
    case STRS_U32_TAPE_VIEW: Py_RETURN_FALSE;
    case STRS_U64_TAPE:
    case STRS_U64_TAPE_VIEW: Py_RETURN_TRUE;
    default: PyErr_SetString(PyExc_RuntimeError, "Unknown Strs layout"); return NULL;
    }
}

static PyObject *Strs_get_tape_address(Strs *self, void *closure) {
    if (!Strs_ensure_tape_layout(self)) return NULL;

    sz_cptr_t tape_ptr = NULL;
    switch (self->layout) {
    case STRS_U32_TAPE_VIEW: tape_ptr = self->data.u32_tape_view.data; break;
    case STRS_U32_TAPE: tape_ptr = self->data.u32_tape.data; break;
    case STRS_U64_TAPE_VIEW: tape_ptr = self->data.u64_tape_view.data; break;
    case STRS_U64_TAPE: tape_ptr = self->data.u64_tape.data; break;
    default: PyErr_SetString(PyExc_RuntimeError, "Unknown Strs layout"); return NULL;
    }

    return PyLong_FromSize_t((sz_size_t)tape_ptr);
}

static PyObject *Strs_get_offsets_address(Strs *self, void *closure) {
    if (!Strs_ensure_tape_layout(self)) return NULL;

    void *offsets_ptr = NULL;
    switch (self->layout) {
    case STRS_U32_TAPE_VIEW: offsets_ptr = self->data.u32_tape_view.offsets; break;
    case STRS_U32_TAPE: offsets_ptr = self->data.u32_tape.offsets; break;
    case STRS_U64_TAPE_VIEW: offsets_ptr = self->data.u64_tape_view.offsets; break;
    case STRS_U64_TAPE: offsets_ptr = self->data.u64_tape.offsets; break;
    default: PyErr_SetString(PyExc_RuntimeError, "Unknown Strs layout"); return NULL;
    }

    return PyLong_FromSize_t((sz_size_t)offsets_ptr);
}

static PyObject *Strs_get_tape_nbytes(Strs *self, void *closure) {
    if (!Strs_ensure_tape_layout(self)) return NULL;

    sz_size_t tape_nbytes = 0;
    switch (self->layout) {
    case STRS_U32_TAPE_VIEW: {
        sz_size_t count = self->data.u32_tape_view.count;
        sz_u32_t *offsets = self->data.u32_tape_view.offsets;
        // The tape size is the last offset (offsets[count])
        tape_nbytes = (count > 0) ? offsets[count] : 0;
        break;
    }
    case STRS_U32_TAPE: {
        sz_size_t count = self->data.u32_tape.count;
        sz_u32_t *offsets = self->data.u32_tape.offsets;
        tape_nbytes = (count > 0) ? offsets[count] : 0;
        break;
    }
    case STRS_U64_TAPE_VIEW: {
        sz_size_t count = self->data.u64_tape_view.count;
        sz_u64_t *offsets = self->data.u64_tape_view.offsets;
        tape_nbytes = (count > 0) ? offsets[count] : 0;
        break;
    }
    case STRS_U64_TAPE: {
        sz_size_t count = self->data.u64_tape.count;
        sz_u64_t *offsets = self->data.u64_tape.offsets;
        tape_nbytes = (count > 0) ? offsets[count] : 0;
        break;
    }
    default: PyErr_SetString(PyExc_RuntimeError, "Unknown Strs layout"); return NULL;
    }

    return PyLong_FromSize_t(tape_nbytes);
}

static PyObject *Strs_get_offsets_nbytes(Strs *self, void *closure) {
    if (!Strs_ensure_tape_layout(self)) return NULL;

    sz_size_t count = 0;
    sz_size_t offset_size = 0;

    switch (self->layout) {
    case STRS_U32_TAPE_VIEW:
        count = self->data.u32_tape_view.count;
        offset_size = sizeof(sz_u32_t);
        break;
    case STRS_U32_TAPE:
        count = self->data.u32_tape.count;
        offset_size = sizeof(sz_u32_t);
        break;
    case STRS_U64_TAPE_VIEW:
        count = self->data.u64_tape_view.count;
        offset_size = sizeof(sz_u64_t);
        break;
    case STRS_U64_TAPE:
        count = self->data.u64_tape.count;
        offset_size = sizeof(sz_u64_t);
        break;
    default: PyErr_SetString(PyExc_RuntimeError, "Unknown Strs layout"); return NULL;
    }

    // Arrow format uses N+1 offsets for N strings
    sz_size_t offsets_nbytes = (count + 1) * offset_size;
    return PyLong_FromSize_t(offsets_nbytes);
}

Py_ssize_t Strs_len(Strs *self) {
    switch (self->layout) {
    case STRS_U32_TAPE: return self->data.u32_tape.count;
    case STRS_U32_TAPE_VIEW: return self->data.u32_tape_view.count;
    case STRS_U64_TAPE: return self->data.u64_tape.count;
    case STRS_U64_TAPE_VIEW: return self->data.u64_tape_view.count;
    case STRS_FRAGMENTED: return self->data.fragmented.count;
    default: return 0;
    }
}

static PyObject *Strs_getitem(Strs *self, Py_ssize_t i) {

    // Check for negative index and convert to positive
    Py_ssize_t count = Strs_len(self);
    if (i < 0) i += count;
    if (i < 0 || i >= count) {
        PyErr_SetString(PyExc_IndexError, "Index out of range");
        return NULL;
    }

    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    PyObject *memory_owner = NULL;
    sz_cptr_t start = NULL;
    sz_size_t length = 0;
    getter(self, i, count, &memory_owner, &start, &length);

    // Create a new `Str` object
    Str *view_copy = Str_alloc_();
    if (view_copy == NULL) return PyErr_NoMemory();

    view_copy->memory.start = start;
    view_copy->memory.length = length;
    view_copy->parent = memory_owner;
    Py_XINCREF(memory_owner);
    return view_copy;
}

/**
 *  This returns a `Strs` object of a potentially different layout:
 *  - `STRS_U32_TAPE_VIEW` input yields a `STRS_U32_TAPE_VIEW` for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_U64_TAPE_VIEW` input yields a `STRS_U64_TAPE_VIEW` for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_U32_TAPE` input yields a `STRS_U32_TAPE_VIEW`  for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_U64_TAPE` input yields a `STRS_U64_TAPE_VIEW`  for `step=1`, `STRS_FRAGMENTED` otherwise.
 *  - `STRS_FRAGMENTED` input yields a `STRS_FRAGMENTED` output.
 */
static PyObject *Strs_subscript(Strs *self, PyObject *key) {

    if (PyLong_Check(key)) {
        Py_ssize_t index = PyLong_AsSsize_t(key);
        if (index == -1 && PyErr_Occurred()) return NULL; // Propagate OverflowError instead of leaking it
        return Strs_getitem(self, index);
    }

    if (!PySlice_Check(key)) {
        PyErr_SetString(PyExc_TypeError, "Strs indices must be integers or slices");
        return NULL;
    }

    // Sanity checks
    Py_ssize_t count = Strs_len(self);
    Py_ssize_t start, stop, step;
    if (PySlice_Unpack(key, &start, &stop, &step) < 0) return NULL;
    Py_ssize_t result_count = PySlice_AdjustIndices(count, &start, &stop, step);
    if (result_count < 0) return NULL;

    // Create a new `Strs` object
    Strs *result = Strs_alloc_();
    if (result == NULL) return PyErr_NoMemory();

    if (result_count == 0) {
        result->layout = STRS_FRAGMENTED;
        result->data.fragmented.count = 0;
        result->data.fragmented.spans = NULL;
        result->data.fragmented.parent = NULL;
        sz_memory_allocator_init_default(&result->data.fragmented.allocator);
        return (PyObject *)result;
    }

    // If a step is requested, we have to create a new `FRAGMENTED` instance of `Strs`,
    // even if the original one was a tape layout.
    if (step != 1) {
        sz_string_view_t *new_spans = (sz_string_view_t *)malloc(result_count * sizeof(sz_string_view_t));
        if (new_spans == NULL) {
            Py_XDECREF(result);
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for fragmented spans");
            return NULL;
        }

        get_string_at_offset_t getter = str_at_offset_getter(self);
        result->layout = STRS_FRAGMENTED;
        result->data.fragmented.count = result_count;
        result->data.fragmented.spans = new_spans;
        result->data.fragmented.parent = NULL;
        sz_memory_allocator_init_default(&result->data.fragmented.allocator);

        // Populate the new fragmented array using `get_string_at_offset`
        sz_size_t j = 0;
        if (step > 0)
            for (Py_ssize_t i = start; i < stop; i += step, ++j) {
                getter(self, i, count, &result->data.fragmented.parent, &new_spans[j].start, &new_spans[j].length);
            }
        else
            for (Py_ssize_t i = start; i > stop; i += step, ++j) {
                getter(self, i, count, &result->data.fragmented.parent, &new_spans[j].start, &new_spans[j].length);
            }

        // Ensure the parent string isn't prematurely deallocated by this view.
        Py_XINCREF(result->data.fragmented.parent);
        return (PyObject *)result;
    }

    // For step=1, follow the docstring behavior:
    switch (self->layout) {

    case STRS_U32_TAPE_VIEW: {
        // STRS_U32_TAPE_VIEW input yields STRS_U32_TAPE_VIEW for step=1
        result->layout = STRS_U32_TAPE_VIEW;
        result->data.u32_tape_view.count = result_count;
        result->data.u32_tape_view.data = self->data.u32_tape_view.data;
        result->data.u32_tape_view.offsets = self->data.u32_tape_view.offsets + start;
        result->data.u32_tape_view.parent = self->data.u32_tape_view.parent;
        Py_INCREF(result->data.u32_tape_view.parent);
        break;
    }

    case STRS_U64_TAPE_VIEW: {
        // STRS_U64_TAPE_VIEW input yields STRS_U64_TAPE_VIEW for step=1
        result->layout = STRS_U64_TAPE_VIEW;
        result->data.u64_tape_view.count = result_count;
        result->data.u64_tape_view.data = self->data.u64_tape_view.data;
        result->data.u64_tape_view.offsets = self->data.u64_tape_view.offsets + start;
        result->data.u64_tape_view.parent = self->data.u64_tape_view.parent;
        Py_INCREF(result->data.u64_tape_view.parent);
        break;
    }

    case STRS_U32_TAPE: {
        // STRS_U32_TAPE input yields STRS_U32_TAPE_VIEW for step=1
        result->layout = STRS_U32_TAPE_VIEW;
        result->data.u32_tape_view.count = result_count;
        result->data.u32_tape_view.data = self->data.u32_tape.data;
        result->data.u32_tape_view.offsets = self->data.u32_tape.offsets + start;
        result->data.u32_tape_view.parent = (PyObject *)self;
        Py_INCREF((PyObject *)self);
        break;
    }

    case STRS_U64_TAPE: {
        // STRS_U64_TAPE input yields STRS_U64_TAPE_VIEW for step=1
        result->layout = STRS_U64_TAPE_VIEW;
        result->data.u64_tape_view.count = result_count;
        result->data.u64_tape_view.data = self->data.u64_tape.data;
        result->data.u64_tape_view.offsets = self->data.u64_tape.offsets + start;
        result->data.u64_tape_view.parent = (PyObject *)self;
        Py_INCREF((PyObject *)self);
        break;
    }

    case STRS_FRAGMENTED: {
        // STRS_FRAGMENTED input yields STRS_FRAGMENTED output
        result->layout = STRS_FRAGMENTED;
        result->data.fragmented.count = result_count;
        result->data.fragmented.parent = self->data.fragmented.parent;
        Py_XINCREF(result->data.fragmented.parent);
        sz_memory_allocator_init_default(&result->data.fragmented.allocator);

        result->data.fragmented.spans = malloc(sizeof(sz_string_view_t) * result_count);
        if (result->data.fragmented.spans == NULL) {
            PyErr_NoMemory();
            Py_XDECREF(result);
            return NULL;
        }
        sz_copy(result->data.fragmented.spans, self->data.fragmented.spans + start,
                sizeof(sz_string_view_t) * result_count);
        break;
    }

    default:
        // Unsupported layout
        PyErr_SetString(PyExc_TypeError, "Unsupported layout for conversion");
        Py_XDECREF(result);
        return NULL;
    }

    return (PyObject *)result;
}

/**
 *  @brief  Will be called by the `PySequence_Contains` to check the presence of a string in array.
 *  @return 1 if the string is present, 0 if it is not, -1 in case of error.
 *  @see    Docs: https://docs.python.org/3/c-api/sequence.html#c.PySequence_Contains
 */
static int Strs_in(Str *self, PyObject *needle_obj) {

    // Validate and convert `needle`
    sz_string_view_t needle;
    if (!sz_py_export_string_like(needle_obj, &needle.start, &needle.length)) {
        wrap_current_exception("The needle argument must be string-like");
        return -1;
    }

    // Depending on the layout, we will need to use different logic
    Py_ssize_t count = Strs_len(self);
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return -1;
    }

    // Time for a full-scan
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *parent = NULL;
        sz_cptr_t start = NULL;
        sz_size_t length = 0;
        getter(self, i, count, &parent, &start, &length);
        if (length == needle.length && sz_equal(start, needle.start, needle.length) == sz_true_k) return 1;
    }

    return 0;
}

static PyObject *Strs_richcompare(PyObject *self, PyObject *other, int op) {

    Strs *a = (Strs *)self;
    Py_ssize_t a_length = Strs_len(a);
    get_string_at_offset_t a_getter = str_at_offset_getter(a);
    if (!a_getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }

    // If the other object is also a Strs, we can compare them much faster,
    // avoiding the CPython API entirely
    if (PyObject_TypeCheck(other, &StrsType)) {
        Strs *b = (Strs *)other;

        // Check if lengths are equal
        Py_ssize_t b_length = Strs_len(b);
        if (a_length != b_length) {
            if (op == Py_EQ) { Py_RETURN_FALSE; }
            if (op == Py_NE) { Py_RETURN_TRUE; }
        }

        // The second array may have a different layout
        get_string_at_offset_t b_getter = str_at_offset_getter(b);
        if (!b_getter) {
            PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
            return NULL;
        }

        // Check each item for equality
        Py_ssize_t min_length = sz_min_of_two(a_length, b_length);
        for (Py_ssize_t i = 0; i < min_length; i++) {
            PyObject *ai_parent = NULL, *bi_parent = NULL;
            sz_cptr_t ai_start = NULL, *bi_start = NULL;
            sz_size_t ai_length = 0, bi_length = 0;
            a_getter(a, i, a_length, &ai_parent, &ai_start, &ai_length);
            b_getter(b, i, b_length, &bi_parent, &bi_start, &bi_length);

            // When dealing with arrays, early exists make sense only in some cases
            int order = (int)sz_order(ai_start, ai_length, bi_start, bi_length);
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

    // The second argument is a sequence, but not a `Strs` object,
    // so we need to iterate through it.
    PyObject *other_iter = PyObject_GetIter(other);
    if (!other_iter) {
        PyErr_Clear();
        PyErr_SetString(PyExc_TypeError, "The second argument is not iterable");
        return NULL;
    }

    // We may not even know the length of the second sequence, so
    // let's just iterate as far as we can.
    Py_ssize_t i = 0;
    PyObject *other_item;
    for (; (other_item = PyIter_Next(other_iter)); ++i) {
        // Check if the second array is longer than the first
        if (a_length <= i) {
            Py_DECREF(other_item);
            Py_DECREF(other_iter);
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
            Py_DECREF(other_iter);
            wrap_current_exception("The second container must contain string-like objects");
            return NULL;
        }

        // Both sequences aren't exhausted yet
        PyObject *ai_parent = NULL;
        sz_cptr_t ai_start = NULL;
        sz_size_t ai_length = 0;
        a_getter(a, i, a_length, &ai_parent, &ai_start, &ai_length);

        // When dealing with arrays, early exists make sense only in some cases
        int order = (int)sz_order(ai_start, ai_length, bi.start, bi.length);
        switch (op) {
        case Py_LT:
        case Py_LE:
            if (order > 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_FALSE;
            }
            break;
        case Py_EQ:
            if (order != 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_FALSE;
            }
            break;
        case Py_NE:
            if (order == 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
                Py_RETURN_TRUE;
            }
            break;
        case Py_GT:
        case Py_GE:
            if (order < 0) {
                Py_DECREF(other_item);
                Py_DECREF(other_iter);
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

/**
 *  @brief Shuffles the parts of a `Strs` object.
 *
 *  This accepts a `Strs` object and potentially produces a new `Strs` object of a different layout:
 *  - `STRS_U32_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U32_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_FRAGMENTED` returns a copy of itself, with the parts shuffled.
 */
static PyObject *Strs_shuffled(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                               PyObject *args_names_tuple) {

    // Check for positional arguments
    PyObject *seed_obj = positional_args_count == 1 ? args[0] : NULL;
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
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_obj) { seed_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Fisher-Yates Shuffle Algorithm
    unsigned int seed = (unsigned int)time(NULL);
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_obj);
        if (PyErr_Occurred()) return NULL; // Reject negative / oversized seed
    }

    // Determine the amount of memory needed
    sz_size_t substrings_count = 0;
    get_string_at_offset_t substring_getter = NULL;
    PyObject *parent_to_increment = NULL;
    sz_memory_allocator_t allocator;

    switch (self->layout) {
    case STRS_U32_TAPE:
        substring_getter = str_at_offset_u32_tape;
        substrings_count = self->data.u32_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u32_tape.allocator;
        break;
    case STRS_U32_TAPE_VIEW:
        substring_getter = str_at_offset_u32_tape_view;
        substrings_count = self->data.u32_tape_view.count;
        parent_to_increment = self->data.u32_tape_view.parent;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_U64_TAPE:
        substring_getter = str_at_offset_u64_tape;
        substrings_count = self->data.u64_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u64_tape.allocator;
        break;
    case STRS_U64_TAPE_VIEW:
        substring_getter = str_at_offset_u64_tape_view;
        substrings_count = self->data.u64_tape_view.count;
        parent_to_increment = self->data.u64_tape_view.parent;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_FRAGMENTED:
        substring_getter = str_at_offset_fragmented;
        substrings_count = self->data.fragmented.count;
        parent_to_increment = self->data.fragmented.parent;
        allocator = self->data.fragmented.allocator;
        break;
    }

    // An empty container has nothing to reorder; allocate(0) would spuriously fail (and count-1 would underflow)
    if (substrings_count == 0) return (PyObject *)strs_make_empty_fragmented_();

    sz_string_view_t *new_spans = (sz_string_view_t *)allocator.allocate(substrings_count * sizeof(sz_string_view_t),
                                                                         allocator.handle);
    if (new_spans == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for reordered slices");
        return NULL;
    }

    // Populate the new reordered array using get_string_at_offset
    for (sz_size_t i = 0; i < substrings_count; ++i) {
        PyObject *unused_parent;
        sz_cptr_t start;
        sz_size_t length;
        substring_getter(self, (Py_ssize_t)i, substrings_count, &unused_parent, &start, &length);
        new_spans[i].start = start;
        new_spans[i].length = length;
    }

    // Create a new Strs object for the reordered layout
    Strs *result = Strs_alloc_();
    if (!result) {
        allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_NoMemory();
        return NULL;
    }

    srand(seed);
    for (sz_size_t i = substrings_count - 1; i > 0; --i) {
        sz_size_t j = rand() % (i + 1);
        // Swap parts[i] and parts[j]
        sz_string_view_t temp = new_spans[i];
        new_spans[i] = new_spans[j];
        new_spans[j] = temp;
    }

    // Set up the new reordered object
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = substrings_count;
    result->data.fragmented.spans = new_spans;
    result->data.fragmented.parent = parent_to_increment;
    result->data.fragmented.allocator = allocator;
    Py_XINCREF(parent_to_increment); // Keep the original as parent

    return result;
}

static char const doc_sorted[] =                                                               //
    "sorted(*, reverse=False, uncased=False, top=None) -> Strs\n"                              //
    "\n"                                                                                       //
    "Return a new, stably sorted Strs; the original is unchanged.\n"                           //
    "\n"                                                                                       //
    "Args:\n"                                                                                  //
    "  reverse (bool, optional): Sort in descending order. Defaults to False.\n"               //
    "  uncased (bool, optional): Order by Unicode case-folding. Defaults to False.\n"          //
    "  top (int, optional): Keep only the `top` smallest (or largest, if reversed) elements. " //
    "Defaults to None (all).\n"                                                                //
    "Returns:\n"                                                                               //
    "  Strs: A new, sorted collection.\n"                                                      //
    "Example:\n"                                                                               //
    "  >>> list(map(str, sz.Strs(['banana', 'apple', 'cherry']).sorted()))\n"                  //
    "  ['apple', 'banana', 'cherry']";

/**
 *  @brief Sorts the parts of a `Strs` object.
 *
 *  This accepts a `Strs` object and potentially produces a new `Strs` object of a different layout:
 *  - `STRS_U32_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE_VIEW` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U32_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_U64_TAPE` becomes `STRS_FRAGMENTED`, and keeps a link to the old as a parent.
 *  - `STRS_FRAGMENTED` returns a copy of itself, with the parts sorted.
 */
static PyObject *Strs_sorted(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    // Parse the keyword-only options: `reverse` (bool), `uncased` (bool), `top` (non-negative int or None).
    sz_bool_t reverse = sz_false_k, uncased = sz_false_k;
    sz_size_t top = 0;
    if (positional_args_count != 0) {
        PyErr_SetString(PyExc_TypeError, "sorted() takes no positional arguments");
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
        else {
            PyErr_Format(PyExc_TypeError, "sorted() got an unexpected keyword argument '%U'", key);
            return NULL;
        }
    }
    if (reverse_obj) {
        if (!PyBool_Check(reverse_obj)) {
            PyErr_SetString(PyExc_TypeError, "sorted(): reverse must be a bool");
            return NULL;
        }
        reverse = (sz_bool_t)(PyObject_IsTrue(reverse_obj) != 0);
    }
    if (uncased_obj) {
        if (!PyBool_Check(uncased_obj)) {
            PyErr_SetString(PyExc_TypeError, "sorted(): uncased must be a bool");
            return NULL;
        }
        uncased = (sz_bool_t)(PyObject_IsTrue(uncased_obj) != 0);
    }
    if (top_obj && top_obj != Py_None) {
        if (!PyLong_Check(top_obj)) {
            PyErr_SetString(PyExc_TypeError, "sorted(): top must be an int or None");
            return NULL;
        }
        Py_ssize_t top_value = PyLong_AsSsize_t(top_obj);
        if (top_value == -1 && PyErr_Occurred()) return NULL;
        if (top_value < 0) {
            PyErr_SetString(PyExc_ValueError, "sorted(): top must be non-negative");
            return NULL;
        }
        top = (sz_size_t)top_value;
    }

    // Determine the amount of memory needed
    sz_size_t substrings_count = 0;
    get_string_at_offset_t substring_getter = NULL;
    PyObject *parent_to_increment = NULL;
    sz_memory_allocator_t allocator;

    switch (self->layout) {
    case STRS_U32_TAPE:
        substring_getter = str_at_offset_u32_tape;
        substrings_count = self->data.u32_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u32_tape.allocator;
        break;
    case STRS_U32_TAPE_VIEW:
        substring_getter = str_at_offset_u32_tape_view;
        substrings_count = self->data.u32_tape_view.count;
        parent_to_increment = (PyObject *)self;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_U64_TAPE:
        substring_getter = str_at_offset_u64_tape;
        substrings_count = self->data.u64_tape.count;
        parent_to_increment = (PyObject *)self;
        allocator = self->data.u64_tape.allocator;
        break;
    case STRS_U64_TAPE_VIEW:
        substring_getter = str_at_offset_u64_tape_view;
        substrings_count = self->data.u64_tape_view.count;
        parent_to_increment = (PyObject *)self;
        sz_memory_allocator_init_default(&allocator);
        break;
    case STRS_FRAGMENTED:
        substring_getter = str_at_offset_fragmented;
        substrings_count = self->data.fragmented.count;
        parent_to_increment = self->data.fragmented.parent;
        allocator = self->data.fragmented.allocator;
        break;
    }

    // An empty container has nothing to reorder; allocate(0) would spuriously fail (and count-1 would underflow)
    if (substrings_count == 0) return (PyObject *)strs_make_empty_fragmented_();

    sz_string_view_t *new_spans = (sz_string_view_t *)allocator.allocate(substrings_count * sizeof(sz_string_view_t),
                                                                         allocator.handle);
    if (new_spans == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for reordered slices");
        return NULL;
    }

    // Populate the new reordered array using get_string_at_offset
    for (sz_size_t i = 0; i < substrings_count; ++i) {
        PyObject *unused_parent;
        sz_cptr_t start;
        sz_size_t length;
        substring_getter((Strs *)self, (Py_ssize_t)i, substrings_count, &unused_parent, &start, &length);
        new_spans[i].start = start;
        new_spans[i].length = length;
    }

    sz_sorted_idx_t *order = (sz_sorted_idx_t *)malloc(sizeof(sz_sorted_idx_t) * substrings_count);
    if (!order) {
        allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_Format(PyExc_MemoryError, "Unable to allocate memory for the sorting operation");
        return NULL;
    }

    // Call our sorting algorithm (`reverse` and `uncased` are handled natively).
    sz_sequence_t sequence;
    sz_fill(&sequence, sizeof(sequence), 0);
    sequence.count = substrings_count;
    sequence.handle = (void *)self;
    sequence.get_start = Strs_get_start_;
    sequence.get_length = Strs_get_length_;
    sz_status_t status = Strs_run_argsort_(uncased, &sequence, order, top, reverse);
    sz_unused_(status);

    // With `top` set, only the leading `top` elements are ordered, so the result keeps just those.
    sz_size_t const result_count = (top != 0 && top < substrings_count) ? top : substrings_count;

    // Apply the new order to create sorted spans
    sz_string_view_t *sorted_spans = (sz_string_view_t *)allocator.allocate(result_count * sizeof(sz_string_view_t),
                                                                            allocator.handle);
    if (sorted_spans == NULL && result_count) {
        free(order);
        allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate memory for sorted slices");
        return NULL;
    }

    // Apply the permutation
    for (sz_size_t i = 0; i < result_count; ++i) sorted_spans[i] = new_spans[order[i]];
    free(order);

    // Free the temporary spans array
    allocator.free(new_spans, substrings_count * sizeof(sz_string_view_t), allocator.handle);

    // Create a new Strs object for the sorted layout
    Strs *result = Strs_alloc_();
    if (!result) {
        allocator.free(sorted_spans, result_count * sizeof(sz_string_view_t), allocator.handle);
        PyErr_NoMemory();
        return NULL;
    }

    // Set up the new sorted object
    result->layout = STRS_FRAGMENTED;
    result->data.fragmented.count = result_count;
    result->data.fragmented.spans = sorted_spans;
    result->data.fragmented.parent = parent_to_increment;
    result->data.fragmented.allocator = allocator;
    Py_XINCREF(parent_to_increment); // Keep the original as parent

    return (PyObject *)result;
}

static PyObject *Strs_sample(Strs *self, PyObject *const *args, Py_ssize_t positional_args_count,
                             PyObject *args_names_tuple) {
    PyObject *sample_size_obj = NULL;
    PyObject *seed_obj = NULL;

    // Check for positional arguments
    if (positional_args_count > 1) {
        PyErr_SetString(PyExc_TypeError, "sample() takes 1 positional argument and 1 keyword argument");
        return NULL;
    }
    else if (positional_args_count == 1) { sample_size_obj = args[0]; }

    // Parse keyword arguments
    if (args_names_tuple) {
        Py_ssize_t args_names_count = PyTuple_GET_SIZE(args_names_tuple);
        for (Py_ssize_t i = 0; i < args_names_count; ++i) {
            PyObject *key = PyTuple_GET_ITEM(args_names_tuple, i);
            PyObject *value = args[positional_args_count + i];
            if (PyUnicode_CompareWithASCIIString(key, "seed") == 0 && !seed_obj) { seed_obj = value; }
            else if (PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key)) { return NULL; }
        }
    }

    // Translate the seed and the sample size to C types
    sz_size_t sample_size = 0;
    if (sample_size_obj) {
        if (!PyLong_Check(sample_size_obj)) {
            PyErr_SetString(PyExc_TypeError, "The sample size must be an integer");
            return NULL;
        }
        sample_size = PyLong_AsSize_t(sample_size_obj);
        if (sample_size == (sz_size_t)-1 && PyErr_Occurred()) return NULL; // Reject negative / oversized
    }
    unsigned int seed = (unsigned int)time(NULL); // Default seed
    if (seed_obj) {
        if (!PyLong_Check(seed_obj)) {
            PyErr_SetString(PyExc_TypeError, "The seed must be an integer");
            return NULL;
        }
        seed = PyLong_AsUnsignedLong(seed_obj);
        if (PyErr_Occurred()) return NULL; // Reject negative / oversized seed
    }

    // Introspect the source before allocating, so an empty request or empty source short-circuits
    Py_ssize_t source_count = Strs_len(self);
    get_string_at_offset_t getter = str_at_offset_getter(self);
    if (!getter) {
        PyErr_SetString(PyExc_TypeError, "Unknown Strs kind");
        return NULL;
    }
    if (sample_size == 0 || source_count == 0) return (PyObject *)strs_make_empty_fragmented_();

    // Allocate the sampled spans with an overflow-checked size
    if (sample_size > SIZE_MAX / sizeof(sz_string_view_t)) return PyErr_NoMemory();
    sz_string_view_t *sampled_spans = malloc(sample_size * sizeof(sz_string_view_t));
    if (!sampled_spans) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for the sample");
        return NULL;
    }

    Strs *result = strs_make_empty_fragmented_();
    if (result == NULL) {
        free(sampled_spans);
        return NULL;
    }

    // Randomly sample the strings (with replacement)
    srand(seed);
    PyObject *parent_string;
    for (Py_ssize_t i = 0; i < (Py_ssize_t)sample_size; i++) {
        sz_size_t index = rand() % source_count;
        getter(self, index, source_count, &parent_string, &sampled_spans[i].start, &sampled_spans[i].length);
    }

    // Update the `Strs` object
    result->data.fragmented.count = sample_size;
    result->data.fragmented.spans = sampled_spans;
    result->data.fragmented.parent = parent_string;
    // Hold a reference to the parent backing buffer while this view is alive
    Py_XINCREF(result->data.fragmented.parent);
    return result;
}

static PyObject *Strs_get_layout(Strs *self, void *Py_UNUSED(closure)) {
    char buffer[1024];

    switch (self->layout) {
    case STRS_U32_TAPE_VIEW:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U32_TAPE_VIEW, count=%zu, data=%p, offsets=%p, parent=%p]",
                 self->data.u32_tape_view.count, (void *)self->data.u32_tape_view.data,
                 (void *)self->data.u32_tape_view.offsets, (void *)self->data.u32_tape_view.parent);
        break;

    case STRS_U64_TAPE_VIEW:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U64_TAPE_VIEW, count=%zu, data=%p, offsets=%p, parent=%p]",
                 self->data.u64_tape_view.count, (void *)self->data.u64_tape_view.data,
                 (void *)self->data.u64_tape_view.offsets, (void *)self->data.u64_tape_view.parent);
        break;

    case STRS_U32_TAPE:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U32_TAPE, count=%zu, data=%p, offsets=%p]",
                 self->data.u32_tape.count, (void *)self->data.u32_tape.data, (void *)self->data.u32_tape.offsets);
        break;

    case STRS_U64_TAPE:
        snprintf(buffer, sizeof(buffer), "Strs[layout=U64_TAPE, count=%zu, data=%p, offsets=%p]",
                 self->data.u64_tape.count, (void *)self->data.u64_tape.data, (void *)self->data.u64_tape.offsets);
        break;

    case STRS_FRAGMENTED:
        snprintf(buffer, sizeof(buffer), "Strs[layout=FRAGMENTED, count=%zu, spans=%p, parent=%p]",
                 self->data.fragmented.count, (void *)self->data.fragmented.spans,
                 (void *)self->data.fragmented.parent);
        break;

    default: snprintf(buffer, sizeof(buffer), "Strs[layout=UNKNOWN(%d)]", self->layout); break;
    }

    return PyUnicode_FromString(buffer);
}

/**
 *  @brief Exports a string to a UTF-8 buffer, escaping single quotes.
 *  @param[in] cstr The input string to export.
 *  @param[in] cstr_length The length of the input string.
 *  @param[out] buffer The output buffer to write to.
 *  @param[in] buffer_length The size of the output buffer.
 *  @param[out] did_fit Populated with 1 if the string is fully exported, 0 if it didn't fit, -1 if invalid UTF-8.
 *  @return Pointer to the end of the written data in the buffer, or buffer position where error occurred.
 */
static sz_cptr_t export_escaped_unquoted_to_utf8_buffer(sz_cptr_t cstr, sz_size_t cstr_length,    //
                                                        sz_ptr_t buffer, sz_size_t buffer_length, //
                                                        int *did_fit) {
    sz_cptr_t const cstr_end = cstr + cstr_length;
    sz_ptr_t buffer_ptr = buffer;
    *did_fit = 1;

    // Validate UTF-8 first
    if (sz_utf8_find_malformed(cstr, cstr_length) != SZ_NULL_CHAR) {
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
            sz_copy(buffer_ptr, cstr, rune_length);
            buffer_ptr += rune_length;
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
    // Format: b'\x00\x01...'  -> 3 bytes prefix + 4 bytes per byte + 1 byte suffix
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
    sz_ptr_t repr_buffer_ptr = &repr_buffer[0];
    sz_cptr_t const repr_buffer_end = repr_buffer_ptr + 1024;

    // Start of the array
    sz_copy(repr_buffer_ptr, "sz.Strs([", 9);
    repr_buffer_ptr += 9;

    sz_size_t count = Strs_len(self);
    PyObject *parent_string;

    // In the worst case, we must have enough space for `...', ...])`
    // That's extra 11 bytes of content.
    sz_cptr_t non_fitting_array_tail = "... ])";
    int const non_fitting_array_tail_length = 6;

    // If the whole string doesn't fit, even before the `non_fitting_array_tail` tail,
    // we need to add `, '` separator of 3 bytes.
    for (sz_size_t i = 0; i < count && repr_buffer_ptr + (non_fitting_array_tail_length + 3) < repr_buffer_end; i++) {
        sz_cptr_t cstr_start = NULL;
        sz_size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);

        if (i > 0) { *(repr_buffer_ptr++) = ',', *(repr_buffer_ptr++) = ' '; }

        // Check if the string contains valid UTF-8
        int did_fit;
        repr_buffer_ptr = sz_utf8_find_malformed(cstr_start, cstr_length) == SZ_NULL_CHAR
                              ? export_escaped_unquoted_to_utf8_buffer(
                                    cstr_start, cstr_length, repr_buffer_ptr,
                                    repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length, &did_fit)
                              : export_escaped_unquoted_to_binary_buffer(
                                    cstr_start, cstr_length, repr_buffer_ptr,
                                    repr_buffer_end - repr_buffer_ptr - non_fitting_array_tail_length, &did_fit);

        // If it didn't fit, let's put an ellipsis
        if (!did_fit) {
            sz_copy(repr_buffer_ptr, non_fitting_array_tail, non_fitting_array_tail_length);
            repr_buffer_ptr += non_fitting_array_tail_length;
            return PyUnicode_FromStringAndSize(repr_buffer, repr_buffer_ptr - repr_buffer);
        }
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
    sz_size_t count = Strs_len(self);
    PyObject *parent_string;
    sz_size_t total_bytes = 2; // opening and closing square brackets
    for (sz_size_t i = 0; i < count; i++) {
        sz_cptr_t cstr_start = NULL;
        sz_size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);

        if (i != 0) total_bytes += 2; // For the preceding comma and space

        // Check if string is valid UTF-8 to determine format
        if (sz_utf8_find_malformed(cstr_start, cstr_length) == SZ_NULL_CHAR) {
            // Valid UTF-8: format as '...' with escaped quotes
            total_bytes += 2;           // Opening and closing quotes
            total_bytes += cstr_length; // Base string length

            // Count the number of single quotes that need escaping
            sz_cptr_t scan_ptr = cstr_start;
            sz_size_t scan_length = cstr_length;
            while (scan_length) {
                char quote = '\'';
                sz_cptr_t next_quote = sz_find_byte(scan_ptr, scan_length, &quote);
                if (next_quote == NULL) break;
                total_bytes++; // Extra byte for escaping
                scan_length -= next_quote - scan_ptr + 1;
                scan_ptr = next_quote + 1;
            }
        }
        else {
            // Invalid UTF-8: format as b'\x...'
            total_bytes += 3;               // "b'" prefix
            total_bytes += cstr_length * 4; // Each byte becomes \xNN (4 chars)
            total_bytes += 1;               // Closing quote
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
        sz_cptr_t cstr_start = NULL;
        sz_size_t cstr_length = 0;
        getter(self, i, count, &parent_string, &cstr_start, &cstr_length);
        int did_fit;
        // Check if the string contains valid UTF-8 and export appropriately
        result_ptr = sz_utf8_find_malformed(cstr_start, cstr_length) == SZ_NULL_CHAR
                         ? export_escaped_unquoted_to_utf8_buffer(cstr_start, cstr_length, result_ptr,
                                                                  total_bytes - (result_ptr - result_buffer), &did_fit)
                         : export_escaped_unquoted_to_binary_buffer(cstr_start, cstr_length, result_ptr,
                                                                    total_bytes - (result_ptr - result_buffer),
                                                                    &did_fit);

        // Note: If did_fit is 0, we have a buffer size calculation error, but we continue for robustness
    }

    *result_ptr++ = ']';
    sz_size_t actual_bytes = result_ptr - result_buffer;
    PyObject *result = PyUnicode_FromStringAndSize(result_buffer, actual_bytes);
    free(result_buffer);
    return result;
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

static char const doc_Strs_tape[] =                                                         //
    "In-place transform the internal representation into the Apache Arrow string layout.\n" //
    "\n"                                                                                    //
    "Compacts fragmented strings into a single contiguous buffer with an offsets array,\n"  //
    "enabling zero-copy export to Arrow/StringTape consumers.";

static char const doc_Strs_tape_address[] = //
    "Memory address of the first byte of the contiguous tape buffer, as an integer.";

static char const doc_Strs_tape_nbytes[] = //
    "Total length of the tape (all string bytes) in bytes.";

static char const doc_Strs_offsets_address[] = //
    "Memory address of the first byte of the offsets array, as an integer.";

static char const doc_Strs_offsets_nbytes[] = //
    "Length of the offsets array in bytes.";

static char const doc_Strs_offsets_are_large[] = //
    "True if 64-bit offsets are needed to address the tape (vs. 32-bit) for Arrow export.";

static char const doc_Strs_layout[] = //
    "Debug string describing the internal storage layout (TAPE, TAPE_VIEW, or FRAGMENTED).";

static PyGetSetDef Strs_getsetters[] = {
    // Compatibility with PyArrow
    {"tape", (getter)Strs_get_tape, NULL, doc_Strs_tape, NULL},                                        //
    {"tape_address", (getter)Strs_get_tape_address, NULL, doc_Strs_tape_address, NULL},                //
    {"tape_nbytes", (getter)Strs_get_tape_nbytes, NULL, doc_Strs_tape_nbytes, NULL},                   //
    {"offsets_address", (getter)Strs_get_offsets_address, NULL, doc_Strs_offsets_address, NULL},       //
    {"offsets_nbytes", (getter)Strs_get_offsets_nbytes, NULL, doc_Strs_offsets_nbytes, NULL},          //
    {"offsets_are_large", (getter)Strs_get_offsets_are_large, NULL, doc_Strs_offsets_are_large, NULL}, //
    {"__layout__", (getter)Strs_get_layout, NULL, doc_Strs_layout, NULL},                              //
    {NULL}                                                                                             // Sentinel
};

// The efficient `Strs_init` path initializing from PyArrow array capsules.
static int Strs_init_from_pyarrow(Strs *self, PyObject *sequence_obj, int view) {
    // Handle Arrow array
    PyObject *capsules = PyObject_CallMethod(sequence_obj, "__arrow_c_array__", NULL);
    if (!capsules || !PyTuple_Check(capsules) || PyTuple_Size(capsules) != 2) {
        Py_XDECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "__arrow_c_array__ must return a tuple of 2 capsules");
        return -1;
    }

    PyObject *schema_capsule = PyTuple_GET_ITEM(capsules, 0);
    PyObject *array_capsule = PyTuple_GET_ITEM(capsules, 1);

    if (!PyCapsule_CheckExact(schema_capsule) || !PyCapsule_CheckExact(array_capsule)) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "Expected PyCapsule objects from __arrow_c_array__");
        return -1;
    }

    struct ArrowSchema *schema = (struct ArrowSchema *)PyCapsule_GetPointer(schema_capsule, "arrow_schema");
    struct ArrowArray *array = (struct ArrowArray *)PyCapsule_GetPointer(array_capsule, "arrow_array");

    if (!schema || !array) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "Failed to extract Arrow C structures");
        return -1;
    }

    // Validate string array layout
    if (!schema->format || (strcmp(schema->format, "u") != 0 && strcmp(schema->format, "U") != 0 &&
                            strcmp(schema->format, "z") != 0 && strcmp(schema->format, "Z") != 0)) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "Arrow array must be string layout");
        return -1;
    }

    if (array->n_buffers != 3) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_ValueError, "String Arrow array must have 3 buffers");
        return -1;
    }

    // Determine if 32-bit or 64-bit offsets
    int use_64bit = (strcmp(schema->format, "U") == 0 || strcmp(schema->format, "Z") == 0);
    void const **buffers = (void const **)array->buffers;
    sz_u8_t const *validity = (sz_u8_t const *)buffers[0]; // May be NULL
    sz_cptr_t data_buffer = (sz_cptr_t)buffers[2];
    sz_size_t length = array->length;

    // Zero-copy mode for Arrow arrays
    if (view) {
        if (use_64bit) {
            sz_i64_t const *offsets_64 = (sz_i64_t const *)buffers[1];
            self->layout = STRS_U64_TAPE_VIEW;
            self->data.u64_tape_view.count = length;
            self->data.u64_tape_view.parent = capsules;
            self->data.u64_tape_view.data = data_buffer;
            self->data.u64_tape_view.offsets = (sz_u64_t *)offsets_64;
            Py_INCREF(capsules);
        }
        else {
            sz_i32_t const *offsets_32 = (sz_i32_t const *)buffers[1];
            self->layout = STRS_U32_TAPE_VIEW;
            self->data.u32_tape_view.count = length;
            self->data.u32_tape_view.parent = capsules;
            self->data.u32_tape_view.data = data_buffer;
            self->data.u32_tape_view.offsets = (sz_u32_t *)offsets_32;
            Py_INCREF(capsules);
        }
    }
    // Copy mode for Arrow arrays
    else {
        // Copy mode for Arrow arrays - use allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        if (use_64bit) {
            sz_i64_t const *offsets_64 = (sz_i64_t const *)buffers[1];
            sz_size_t total_bytes = offsets_64[length] - offsets_64[0];

            // Allocate new buffer and offsets using the allocator
            sz_ptr_t new_data = total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle)
                                            : (sz_ptr_t)NULL;
            sz_u64_t *new_offsets = (sz_u64_t *)allocator.allocate((length + 1) * sizeof(sz_u64_t), allocator.handle);
            int const failed_to_allocate_data = total_bytes && !new_data;
            if (failed_to_allocate_data || !new_offsets) {
                if (new_data) allocator.free(new_data, total_bytes, allocator.handle);
                if (new_offsets) allocator.free(new_offsets, (length + 1) * sizeof(sz_u64_t), allocator.handle);
                Py_DECREF(capsules);
                PyErr_NoMemory();
                return -1;
            }

            // Copy data and adjust offsets (Apache Arrow format)
            sz_size_t actual_bytes = offsets_64[length] - offsets_64[0];
            if (actual_bytes > 0) sz_copy(new_data, data_buffer + offsets_64[0], actual_bytes);
            new_offsets[0] = 0; // First offset is always 0
            for (sz_size_t i = 0; i < length; i++) {
                // Handle null values by checking validity bitmap
                if (validity && !(validity[i / 8] & (1 << (i % 8)))) { new_offsets[i + 1] = new_offsets[i]; }
                else { new_offsets[i + 1] = offsets_64[i + 1] - offsets_64[0]; }
            }

            self->layout = STRS_U64_TAPE;
            self->data.u64_tape.count = length;
            self->data.u64_tape.data = new_data;
            self->data.u64_tape.offsets = new_offsets;
            self->data.u64_tape.allocator = allocator;
        }
        else {
            sz_i32_t const *offsets_32 = (sz_i32_t const *)buffers[1];
            sz_size_t total_bytes = offsets_32[length] - offsets_32[0];

            // Allocate new buffer and offsets using the allocator
            sz_ptr_t new_data = total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle)
                                            : (sz_ptr_t)NULL;
            sz_u32_t *new_offsets = (sz_u32_t *)allocator.allocate((length + 1) * sizeof(sz_u32_t), allocator.handle);
            int const failed_to_allocate_data = total_bytes && !new_data;
            if (failed_to_allocate_data || !new_offsets) {
                if (new_data) allocator.free(new_data, total_bytes, allocator.handle);
                if (new_offsets) allocator.free(new_offsets, (length + 1) * sizeof(sz_u32_t), allocator.handle);
                Py_DECREF(capsules);
                PyErr_NoMemory();
                return -1;
            }

            // Copy data and adjust offsets (Apache Arrow format)
            sz_size_t actual_bytes = offsets_32[length] - offsets_32[0];
            if (actual_bytes > 0) sz_copy(new_data, data_buffer + offsets_32[0], actual_bytes);
            new_offsets[0] = 0; // First offset is always 0
            for (sz_size_t i = 0; i < length; i++) {
                // Handle null values by checking validity bitmap
                if (validity && !(validity[i / 8] & (1 << (i % 8)))) { new_offsets[i + 1] = new_offsets[i]; }
                else { new_offsets[i + 1] = offsets_32[i + 1] - offsets_32[0]; }
            }

            self->layout = STRS_U32_TAPE;
            self->data.u32_tape.count = length;
            self->data.u32_tape.data = new_data;
            self->data.u32_tape.offsets = new_offsets;
            self->data.u32_tape.allocator = allocator;
        }
    }

    Py_DECREF(capsules);
    return 0;
}

// The less efficient `Strs_init` path initializing from a Pythonic tuple of strings.
static int Strs_init_from_tuple(Strs *self, PyObject *sequence_obj, int view) {
    Py_ssize_t count = PyTuple_GET_SIZE(sequence_obj);

    // Empty tuple, create empty Strs
    if (count == 0) {
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        self->data.fragmented.parent = NULL;
        sz_memory_allocator_init_default(&self->data.fragmented.allocator);
        return 0;
    }

    // Zero-copy mode for Python sequences - use reordered layout for memory-scattered strings
    if (view) {
        // Initialize allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        sz_string_view_t *parts = (sz_string_view_t *)allocator.allocate(count * sizeof(sz_string_view_t),
                                                                         allocator.handle);
        if (!parts) {
            PyErr_NoMemory();
            return -1;
        }

        // Create views directly to Python string objects
        for (sz_size_t i = 0; i < (sz_size_t)count; i++) {
            PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
            sz_cptr_t item_start;
            sz_size_t item_length;
            if (!sz_py_export_string_like(item, &item_start, &item_length)) {
                allocator.free(parts, count * sizeof(sz_string_view_t), allocator.handle);
                PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", i);
                return -1;
            }
            parts[i].start = item_start;
            parts[i].length = item_length;
        }

        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = count;
        self->data.fragmented.spans = parts;
        self->data.fragmented.allocator = allocator;
        self->data.fragmented.parent = sequence_obj; // Keep sequence alive
        Py_INCREF(sequence_obj);
    }
    // Allocate a new tape to fit all of the items
    else {
        // Estimate the overall size of strings in bytes
        sz_size_t total_bytes = 0;
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
            sz_cptr_t item_start;
            sz_size_t item_length;
            if (!sz_py_export_string_like(item, &item_start, &item_length)) {
                PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", i);
                return -1;
            }
            total_bytes += item_length;
        }

        int use_64bit = (total_bytes >= UINT32_MAX);

        // Initialize allocator for memory management
        sz_memory_allocator_t allocator;
        sz_memory_allocator_init_default(&allocator);

        // Allocate data buffer using allocator
        sz_ptr_t data_buffer = total_bytes ? (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle)
                                           : (sz_ptr_t)NULL;
        int const failed_to_allocate_data = total_bytes && !data_buffer;
        if (failed_to_allocate_data) {
            PyErr_NoMemory();
            return -1;
        }

        if (use_64bit) {
            // Apache Arrow format: N+1 offsets for N strings
            sz_u64_t *offsets = (sz_u64_t *)allocator.allocate((count + 1) * sizeof(sz_u64_t), allocator.handle);
            if (!offsets) {
                if (data_buffer) allocator.free(data_buffer, total_bytes, allocator.handle);
                PyErr_NoMemory();
                return -1;
            }

            sz_size_t offset = 0;
            offsets[0] = 0; // First offset is always 0
            for (Py_ssize_t i = 0; i < count; i++) {
                PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
                sz_cptr_t item_start;
                sz_size_t item_length;
                sz_py_export_string_like(item, &item_start, &item_length);

                sz_copy(data_buffer + offset, item_start, item_length);
                offset += item_length;
                offsets[i + 1] = offset; // Apache Arrow format: offset after this string
            }

            self->layout = STRS_U64_TAPE;
            self->data.u64_tape.count = count;
            self->data.u64_tape.data = data_buffer;
            self->data.u64_tape.offsets = offsets;
            self->data.u64_tape.allocator = allocator;
        }
        else {
            // Apache Arrow format: N+1 offsets for N strings
            sz_u32_t *offsets = (sz_u32_t *)allocator.allocate((count + 1) * sizeof(sz_u32_t), allocator.handle);
            if (!offsets) {
                if (data_buffer) allocator.free(data_buffer, total_bytes, allocator.handle);
                PyErr_NoMemory();
                return -1;
            }

            sz_size_t offset = 0;
            offsets[0] = 0; // First offset is always 0
            for (Py_ssize_t i = 0; i < count; i++) {
                PyObject *item = PyTuple_GET_ITEM(sequence_obj, i);
                sz_cptr_t item_start;
                sz_size_t item_length;
                sz_py_export_string_like(item, &item_start, &item_length);

                sz_copy(data_buffer + offset, item_start, item_length);
                offset += item_length;
                offsets[i + 1] = offset; // Apache Arrow format: offset after this string
            }

            self->layout = STRS_U32_TAPE;
            self->data.u32_tape.count = count;
            self->data.u32_tape.data = data_buffer;
            self->data.u32_tape.offsets = offsets;
            self->data.u32_tape.allocator = allocator;
        }
    }

    return 0;
}

/**
 *  @brief The inefficient `Strs_init` path initializing from a Pythonic list of strings.
 *
 *  A list is walked through a tuple snapshot rather than directly. Two things follow from that, and the
 *  list path needs both: a tuple cannot be resized, so no concurrent `del`/`append` can leave the walk
 *  indexing past the end, and a tuple holds a strong reference to every item, so the spans that `view`
 *  mode exports keep pointing at live strings even after the caller empties the list it passed in.
 *  Holding the list itself pins the container while its contents are free to go.
 *
 *  The snapshot copies one pointer per element, never the string data, so `view` mode stays zero-copy.
 */
static int Strs_init_from_list(Strs *self, PyObject *sequence_obj, int view) {
    PyObject *snapshot = PySequence_Tuple(sequence_obj);
    if (!snapshot) return -1;
    int const result = Strs_init_from_tuple(self, snapshot, view);
    Py_DECREF(snapshot); // `view` mode took its own reference; every other layout owns copied bytes
    return result;
}

// The inefficient `Strs_init` path initializing from a Pythonic iterable of strings.
static int Strs_init_from_iterable(Strs *self, PyObject *sequence_obj, int view) {
    // Get an iterator from the object
    PyObject *iterator = PyObject_GetIter(sequence_obj);
    if (!iterator) {
        PyErr_SetString(PyExc_TypeError, "Object is not iterable");
        return -1;
    }

    if (view) {
        // View mode is not supported for iterators because we can't safely keep references
        // to all the individual string objects without significant overhead
        Py_DECREF(iterator);
        PyErr_SetString(                                             //
            PyExc_ValueError,                                        //
            "View mode (view=True) is not supported for iterators. " //
            "Use view=False to create a copy, or convert to a list/tuple first.");
        return -1;
    }

    // Initialize allocator for memory management
    sz_memory_allocator_t allocator;
    sz_memory_allocator_init_default(&allocator);

    // Incrementally allocate a new tape to fit all of the items
    sz_size_t data_capacity = 4096;
    sz_size_t offsets_capacity = 16;
    sz_size_t count = 0;
    sz_size_t total_bytes = 0;
    int use_64bit = 0; // Start with 32-bit

    sz_ptr_t data_buffer = (sz_ptr_t)allocator.allocate(data_capacity, allocator.handle);
    void *offsets = allocator.allocate(offsets_capacity * sizeof(sz_u32_t), allocator.handle); // Start with 32-bit

    if (!data_buffer || !offsets) {
        if (data_buffer) allocator.free(data_buffer, data_capacity, allocator.handle);
        if (offsets) allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
        Py_DECREF(iterator);
        PyErr_NoMemory();
        return -1;
    }

    // Set initial offset to 0 (Apache Arrow format: N+1 offsets for N strings)
    if (use_64bit) { ((sz_u64_t *)offsets)[0] = 0; }
    else { ((sz_u32_t *)offsets)[0] = 0; }

    // Iterate through all items
    PyObject *item;
    while ((item = PyIter_Next(iterator))) {
        sz_cptr_t item_start;
        sz_size_t item_length;
        if (!sz_py_export_string_like(item, &item_start, &item_length)) {
            Py_DECREF(item);
            allocator.free(data_buffer, data_capacity, allocator.handle);
            allocator.free(offsets, offsets_capacity * (use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t)),
                           allocator.handle);
            Py_DECREF(iterator);
            PyErr_Format(PyExc_TypeError, "Item %zd is not a string-like object", count);
            return -1;
        }

        // Check if adding this string would exceed UINT32_MAX and switch to 64-bit
        if (!use_64bit && total_bytes + item_length > UINT32_MAX) {
            // Convert offsets from 32-bit to 64-bit
            sz_size_t new_offsets_size = offsets_capacity * sizeof(sz_u64_t);
            sz_u64_t *new_offsets = (sz_u64_t *)allocator.allocate(new_offsets_size, allocator.handle);
            if (!new_offsets) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
                Py_DECREF(iterator);
                PyErr_NoMemory();
                return -1;
            }

            // Copy existing 32-bit offsets to 64-bit (including initial 0 and all current offsets)
            sz_u32_t *old_offsets = (sz_u32_t *)offsets;
            for (sz_size_t i = 0; i <= count; i++) { new_offsets[i] = old_offsets[i]; }

            allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
            offsets = new_offsets;
            use_64bit = 1;
        }

        // Grow data buffer if needed (doubling strategy)
        while (total_bytes + item_length > data_capacity) {
            sz_size_t new_capacity = data_capacity * 2;
            sz_ptr_t new_buffer = (sz_ptr_t)allocator.allocate(new_capacity, allocator.handle);
            if (!new_buffer) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * (use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t)),
                               allocator.handle);
                Py_DECREF(iterator);
                PyErr_NoMemory();
                return -1;
            }
            memcpy(new_buffer, data_buffer, total_bytes);
            allocator.free(data_buffer, data_capacity, allocator.handle);
            data_buffer = new_buffer;
            data_capacity = new_capacity;
        }

        // Grow offsets array if needed (doubling strategy)
        // Need space for count+2 offsets total (0, 1, ..., count+1)
        if (count + 1 >= offsets_capacity) {
            sz_size_t new_capacity = offsets_capacity * 2;
            sz_size_t element_size = use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t);
            if (new_capacity > SIZE_MAX / element_size) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
                Py_DECREF(iterator);
                PyErr_SetString(PyExc_MemoryError, "Too many strings");
                return -1;
            }

            void *new_offsets = allocator.allocate(new_capacity * element_size, allocator.handle);
            if (!new_offsets) {
                Py_DECREF(item);
                allocator.free(data_buffer, data_capacity, allocator.handle);
                allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
                Py_DECREF(iterator);
                PyErr_NoMemory();
                return -1;
            }
            memcpy(new_offsets, offsets, (count + 1) * element_size);
            allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
            offsets = new_offsets;
            offsets_capacity = new_capacity;
        }

        // Copy the string data
        memcpy(data_buffer + total_bytes, item_start, item_length);
        total_bytes += item_length;
        count++;

        // Store next offset (end of the string we just added)
        if (use_64bit) { ((sz_u64_t *)offsets)[count] = total_bytes; }
        else { ((sz_u32_t *)offsets)[count] = total_bytes; }

        Py_DECREF(item);
    }

    Py_DECREF(iterator);

    // Check for errors during iteration
    if (PyErr_Occurred()) {
        allocator.free(data_buffer, data_capacity, allocator.handle);
        allocator.free(offsets, offsets_capacity * (use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t)), allocator.handle);
        return -1;
    }

    // Handle empty iterator
    if (count == 0) {
        allocator.free(data_buffer, data_capacity, allocator.handle);
        allocator.free(offsets, offsets_capacity * sizeof(sz_u32_t), allocator.handle);
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        self->data.fragmented.allocator = allocator;
        self->data.fragmented.parent = NULL;
        return 0;
    }

    // Shrink buffers to actual size
    sz_ptr_t final_buffer = (sz_ptr_t)allocator.allocate(total_bytes, allocator.handle);
    if (final_buffer) {
        memcpy(final_buffer, data_buffer, total_bytes);
        allocator.free(data_buffer, data_capacity, allocator.handle);
        data_buffer = final_buffer;
    }

    sz_size_t element_size = use_64bit ? sizeof(sz_u64_t) : sizeof(sz_u32_t);
    sz_size_t final_offsets_size = (count + 1) * element_size;
    void *final_offsets = allocator.allocate(final_offsets_size, allocator.handle);
    if (final_offsets) {
        memcpy(final_offsets, offsets, final_offsets_size);
        allocator.free(offsets, offsets_capacity * element_size, allocator.handle);
        offsets = final_offsets;
    }

    // Setup the consecutive layout (32-bit or 64-bit)
    if (use_64bit) {
        self->layout = STRS_U64_TAPE;
        self->data.u64_tape.count = count;
        self->data.u64_tape.data = data_buffer;
        self->data.u64_tape.offsets = (sz_u64_t *)offsets;
        self->data.u64_tape.allocator = allocator;
    }
    else {
        self->layout = STRS_U32_TAPE;
        self->data.u32_tape.count = count;
        self->data.u32_tape.data = data_buffer;
        self->data.u32_tape.offsets = (sz_u32_t *)offsets;
        self->data.u32_tape.allocator = allocator;
    }

    return 0;
}

static void Strs_release_(Strs *self); // Defined alongside Strs_dealloc below

static int Strs_init(Strs *self, PyObject *args, PyObject *kwargs) {

    // Manual argument parsing for performance
    Py_ssize_t nargs = PyTuple_Size(args);
    if (nargs > 2) {
        PyErr_SetString(PyExc_TypeError,
                        "Strs() takes at most 2 arguments: sequence of strings and a boolean indicator");
        return -1;
    }

    PyObject *sequence_obj = nargs >= 1 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *view_obj = nargs >= 2 ? PyTuple_GET_ITEM(args, 1) : NULL;
    int view = 0; // Default to copy mode

    // Parse keyword arguments if provided
    if (kwargs) {
        Py_ssize_t pos = 0;
        PyObject *key, *value;
        while (PyDict_Next(kwargs, &pos, &key, &value)) {
            if (PyUnicode_CompareWithASCIIString(key, "sequence") == 0 && !sequence_obj) { sequence_obj = value; }
            else if (PyUnicode_CompareWithASCIIString(key, "view") == 0 && !view_obj) { view_obj = value; }
            else {
                PyErr_Format(PyExc_TypeError, "Got an unexpected keyword argument '%U'", key);
                return -1;
            }
        }
    }

    // Parse view flag
    if (view_obj) {
        view = PyObject_IsTrue(view_obj);
        if (view == -1) return -1;
    }

    // Re-initialization: release any state bound by a prior __init__, then reset to the blank state
    // produced by Strs_alloc_ so the rebind below (or a later dealloc) starts from clean ground.
    Strs_release_(self);
    self->layout = STRS_U32_TAPE_VIEW;
    memset(&self->data, 0, sizeof(self->data));

    // If no sequence provided, create empty Strs
    if (!sequence_obj) {
        self->layout = STRS_FRAGMENTED;
        self->data.fragmented.count = 0;
        self->data.fragmented.spans = NULL;
        sz_memory_allocator_init_default(&self->data.fragmented.allocator);
        self->data.fragmented.parent = NULL;
        return 0;
    }

    // Check if it's an Arrow array (has `__arrow_c_array__` method)
    PyObject *arrow_method = PyObject_GetAttrString(sequence_obj, "__arrow_c_array__");
    if (arrow_method) {
        Py_DECREF(arrow_method);
        return Strs_init_from_pyarrow(self, sequence_obj, view);
    }

    // Handle more traditional Python sequences
    PyErr_Clear(); // Clear the attribute error from checking for `__arrow_c_array__`

    if (PyTuple_Check(sequence_obj)) { return Strs_init_from_tuple(self, sequence_obj, view); }
    else if (PyList_Check(sequence_obj)) { return Strs_init_from_list(self, sequence_obj, view); }
    else if (PyObject_HasAttrString(sequence_obj, "__iter__")) {
        return Strs_init_from_iterable(self, sequence_obj, view);
    }
    else {
        PyErr_SetString(PyExc_TypeError, "Strs() argument must be a tuple, list, or iterable");
        return -1;
    }

    return 0;
}

/**
 *  @brief  Frees the owned data/offsets/spans and releases the parent reference held by a `Strs`,
 *          per its layout. Shared by `Strs_dealloc` and the `Strs_init` re-initialization guard.
 */
static void Strs_release_(Strs *self) {
    switch (self->layout) {
    case STRS_U32_TAPE:
        // Free owned data and offsets
        if (self->data.u32_tape.data) {
            sz_size_t data_size = self->data.u32_tape.offsets[self->data.u32_tape.count];
            self->data.u32_tape.allocator.free((sz_ptr_t)self->data.u32_tape.data, data_size,
                                               self->data.u32_tape.allocator.handle);
        }
        if (self->data.u32_tape.offsets) {
            sz_size_t offsets_size = (self->data.u32_tape.count + 1) * sizeof(sz_u32_t);
            self->data.u32_tape.allocator.free(self->data.u32_tape.offsets, offsets_size,
                                               self->data.u32_tape.allocator.handle);
        }
        break;

    case STRS_U64_TAPE:
        // Free owned data and offsets
        if (self->data.u64_tape.data) {
            sz_size_t data_size = self->data.u64_tape.offsets[self->data.u64_tape.count];
            self->data.u64_tape.allocator.free((sz_ptr_t)self->data.u64_tape.data, data_size,
                                               self->data.u64_tape.allocator.handle);
        }
        if (self->data.u64_tape.offsets) {
            sz_size_t offsets_size = (self->data.u64_tape.count + 1) * sizeof(sz_u64_t);
            self->data.u64_tape.allocator.free(self->data.u64_tape.offsets, offsets_size,
                                               self->data.u64_tape.allocator.handle);
        }
        break;

    case STRS_U32_TAPE_VIEW:
        // Views don't own data, just release parent reference
        Py_XDECREF(self->data.u32_tape_view.parent);
        break;

    case STRS_U64_TAPE_VIEW:
        // Views don't own data, just release parent reference
        Py_XDECREF(self->data.u64_tape_view.parent);
        break;

    case STRS_FRAGMENTED:
        // Free owned spans array and release parent reference
        if (self->data.fragmented.spans) {
            sz_size_t spans_size = self->data.fragmented.count * sizeof(sz_string_view_t);
            self->data.fragmented.allocator.free(self->data.fragmented.spans, spans_size,
                                                 self->data.fragmented.allocator.handle);
        }
        Py_XDECREF(self->data.fragmented.parent);
        break;
    }
}

static void Strs_dealloc(Strs *self) {
    Strs_release_(self);

    // Park the dead header on the free-list instead of freeing it, threading the `next` link through
    // the now-unused `data` union; fall through to a real free when the cache is full or absent.
    stringzilla_state_t *state = stringzilla_state_();
    if (state) sz_freelist_lock_(state);
    if (state && state->strs_freelist_count < sz_freelist_capacity_k) {
        *Strs_freelist_next_(self) = state->strs_freelist_head;
        state->strs_freelist_head = self;
        state->strs_freelist_count++;
        sz_freelist_unlock_(state);
        return;
    }
    if (state) sz_freelist_unlock_(state);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static char const doc_Strs_shuffled[] =                                                     //
    "shuffled(seed=None) -> Strs\n"                                                         //
    "\n"                                                                                    //
    "Return a new Strs with the elements randomly permuted; the original is unchanged.\n"   //
    "\n"                                                                                    //
    "Args:\n"                                                                               //
    "  seed (int, optional): Seed for reproducible shuffling. Defaults to a random seed.\n" //
    "Returns:\n"                                                                            //
    "  Strs: A new shuffled collection.\n"                                                  //
    "Example:\n"                                                                            //
    "  >>> sorted(sz.Strs(['a', 'b', 'c']).shuffled(seed=42)) == ['a', 'b', 'c']\n"         //
    "  True";

static char const doc_Strs_sample[] =                                                      //
    "sample(size, seed=None) -> Strs\n"                                                    //
    "\n"                                                                                   //
    "Return a new Strs with `size` elements drawn at random (with replacement).\n"         //
    "\n"                                                                                   //
    "Args:\n"                                                                              //
    "  size (int): Number of elements to draw.\n"                                          //
    "  seed (int, optional): Seed for reproducible sampling. Defaults to a random seed.\n" //
    "Returns:\n"                                                                           //
    "  Strs: A new collection of the sampled elements.\n"                                  //
    "Example:\n"                                                                           //
    "  >>> len(sz.Strs(['a', 'b', 'c', 'd']).sample(2))\n"                                 //
    "  2";

static PyMethodDef Strs_methods[] = {
    {"shuffled", Strs_shuffled, SZ_METHOD_FLAGS, doc_Strs_shuffled},    //
    {"sorted", Strs_sorted, SZ_METHOD_FLAGS, doc_sorted},               //
    {"argsort", Strs_argsort, SZ_METHOD_FLAGS, doc_argsort},            //
    {"sample", Strs_sample, SZ_METHOD_FLAGS, doc_Strs_sample},          //
    {"intersect", Strs_intersect, SZ_METHOD_FLAGS, doc_Strs_intersect}, //
    {NULL, NULL, 0, NULL} // Sentinel
};

static char const doc_Strs[] =                                                                   //
    "Strs(sequence, view=False)\n"                                                               //
    "\n"                                                                                         //
    "Space-efficient container for large collections of strings and their slices.\n"             //
    "Optimized for memory efficiency and bulk operations on string collections.\n"               //
    "Compatible with StringTape format and Apache Arrow string arrays.\n"                        //
    "\n"                                                                                         //
    "Args:\n"                                                                                    //
    "  sequence (list | tuple | generator | pyarrow.Array): Collection of strings to store.\n"   //
    "  view (bool): If True, create a view into the original data instead of copying it.\n"      //
    "\n"                                                                                         //
    "Storage Layouts:\n"                                                                         //
    "  - TAPE: Owns contiguous data buffer with offset array (StringTape compatible)\n"          //
    "  - TAPE_VIEW: Zero-copy view into existing data (Arrow/StringTape slice)\n"                //
    "  - FRAGMENTED: Non-contiguous strings with individual pointers\n"                          //
    "\n"                                                                                         //
    "Features:\n"                                                                                //
    "  - Memory-efficient storage with shared backing buffers\n"                                 //
    "  - Zero-copy slicing and indexing operations\n"                                            //
    "  - StringTape format compatibility for interoperability\n"                                 //
    "  - Bulk operations: sort(), shuffle(), sample()\n"                                         //
    "  - Arrow integration: from_arrow() for zero-copy imports\n"                                //
    "  - GPU kernel compatibility with automatic memory management\n"                            //
    "  - Fast comparison operations with native containers\n"                                    //
    "\n"                                                                                         //
    "Methods:\n"                                                                                 //
    "  - sort(): In-place sorting with custom comparison\n"                                      //
    "  - argsort(): Get indices for sorted order\n"                                              //
    "  - shuffle(): Randomize element order\n"                                                   //
    "  - sample(): Get random subset of elements\n"                                              //
    "\n"                                                                                         //
    "Slicing Behavior:\n"                                                                        //
    "  Slicing creates lightweight views that reference the original data.\n"                    //
    "  Views are automatically converted to owned layouts when needed for\n"                     //
    "  GPU operations, maintaining StringTape format compatibility.\n"                           //
    "\n"                                                                                         //
    "Example:\n"                                                                                 //
    "  >>> names = sz.Strs(['banana', 'apple', 'cherry', 'date'])\n"                             //
    "  >>> [str(x) for x in names.sorted()]      # stable sort over views, no element copies\n"  //
    "  ['apple', 'banana', 'cherry', 'date']\n"                                                  //
    "  >>> [str(x) for x in names.sorted(top=2)]  # partial top-k is cheaper than a full sort\n" //
    "  ['apple', 'banana']";

PyTypeObject StrsType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Strs",
    .tp_doc = doc_Strs,
    .tp_basicsize = sizeof(Strs),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Strs_init,
    .tp_dealloc = (destructor)Strs_dealloc,
    .tp_methods = Strs_methods,
    .tp_as_sequence = &Strs_as_sequence,
    .tp_as_mapping = &Strs_as_mapping,
    .tp_getset = Strs_getsetters,
    .tp_richcompare = Strs_richcompare,
    .tp_repr = (reprfunc)Strs_repr,
    .tp_str = (reprfunc)Strs_str,
};
