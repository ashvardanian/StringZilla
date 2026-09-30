/**
 *  @file javascript/stringzilla.c
 *  @author Ash Vardanian
 *  @date September 18, 2023
 *  @brief JavaScript bindings for StringZilla.
 *
 *  @copyright Copyright (c) 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/n-api.html
 */
#include <stdio.h>  // `printf` for debug builds
#include <stdlib.h> // `malloc` to export strings into UTF-8

#include <node_api.h> // `napi_*` functions

#include <stringzilla/stringzilla.h> // `sz_*` functions

/** The mask kernels run with: detected and compiled, unless @c capabilitiesEnable narrows it. */
static sz_capability_t default_capabilities = sz_cap_serial_k;

/** Throws the @c sz_status_name of a failed @p status; returns whether it succeeded. */
static bool check_status(napi_env env, sz_status_t status) {
    if (status == sz_success_k) return true;
    napi_throw_error(env, NULL, sz_status_name(status));
    return false;
}

static void external_buffer_cleanup(napi_env env, void *data, void *hint) { free(data); }

/** Unwraps the native state from @p js_this, or throws and returns NULL for a foreign receiver. */
static void *unwrap_this(napi_env env, napi_value js_this) {
    void *native = NULL;
    if (napi_unwrap(env, js_this, &native) == napi_ok && native) return native;
    napi_throw_type_error(env, NULL, "Method called on an object that is not an instance of its class");
    return NULL;
}

/** Reads a BigInt or Number hash seed, treating @c undefined as 0, or throws and returns false. */
static bool seed_from_js(napi_env env, napi_value value, sz_u64_t *seed) {
    napi_valuetype type;
    bool lossless = false;
    double number = 0;
    if (napi_typeof(env, value, &type) != napi_ok) return false;
    if (type == napi_undefined) return true;
    if (type == napi_bigint && napi_get_value_bigint_uint64(env, value, seed, &lossless) == napi_ok && lossless)
        return true;
    // Range-check first: converting an out-of-range double to an integer is undefined behavior.
    if (type == napi_number && napi_get_value_double(env, value, &number) == napi_ok && number >= 0 &&
        number < 18446744073709551616.0 && number == (double)(sz_u64_t)number) {
        *seed = (sz_u64_t)number;
        return true;
    }
    napi_throw_range_error(env, NULL, "Seed must be a non-negative integer that fits in 64 bits");
    return false;
}

static napi_value makeFindResultObject(napi_env env, int64_t index, uint64_t length) {
    napi_value js_obj;
    napi_create_object(env, &js_obj);

    napi_value js_index;
    if (index < 0) napi_create_bigint_int64(env, -1, &js_index);
    else napi_create_bigint_uint64(env, (uint64_t)index, &js_index);

    napi_value js_length;
    napi_create_bigint_uint64(env, (uint64_t)length, &js_length);

    napi_set_named_property(env, js_obj, "index", js_index);
    napi_set_named_property(env, js_obj, "length", js_length);

    return js_obj;
}

napi_value indexOfAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    void *haystack_data, *needle_data;
    size_t haystack_length, needle_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }
    status = napi_get_buffer_info(env, args[1], &needle_data, &needle_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    napi_value js_result;
    if (needle_length == 0) { napi_create_bigint_int64(env, 0, &js_result); }
    else {
        sz_cptr_t result;
        if (!check_status(env, sz_find_best((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle_data,
                                            needle_length, &result, default_capabilities, NULL)))
            return NULL;
        if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }
    }

    return js_result;
}

napi_value utf8UncasedFoldAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 1) {
        napi_throw_error(env, NULL, "utf8UncasedFold(buffer, validate?) expects at least 1 argument");
        return NULL;
    }

    void *source_data;
    size_t source_length;
    napi_status status = napi_get_buffer_info(env, args[0], &source_data, &source_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    bool validate = false;
    if (argc > 1) { napi_get_value_bool(env, args[1], &validate); }
    if (validate && sz_utf8_find_malformed((sz_cptr_t)source_data, source_length) != STRINGZILLA_NULL_CHAR) {
        napi_throw_error(env, NULL, "Input is not valid UTF-8");
        return NULL;
    }

    // Worst-case expansion is 3x. See `sz_utf8_uncased_fold_best` docs.
    size_t capacity = source_length * 3;
    void *destination = capacity ? malloc(capacity) : NULL;
    if (capacity && !destination) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }

    sz_size_t out_length = 0;
    if (source_length &&
        !check_status(env, sz_utf8_uncased_fold_best((sz_cptr_t)source_data, source_length, (sz_ptr_t)destination,
                                                     &out_length, default_capabilities, NULL))) {
        free(destination);
        return NULL;
    }

    if (out_length == 0) {
        if (destination) free(destination);
        napi_value js_empty;
        void *unused;
        napi_create_buffer(env, 0, &unused, &js_empty);
        return js_empty;
    }

    // Shrink to the actual size without a second pass or a copy.
    void *shrunk = realloc(destination, (size_t)out_length);
    if (shrunk) destination = shrunk;

    napi_value js_result;
    napi_create_external_buffer(env, (size_t)out_length, destination, external_buffer_cleanup, NULL, &js_result);
    return js_result;
}

napi_value utf8UncasedFindAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_error(env, NULL, "utf8UncasedFind(haystack, needle, validate?) expects at least 2 arguments");
        return NULL;
    }

    void *haystack_data, *needle_data;
    size_t haystack_length, needle_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }
    status = napi_get_buffer_info(env, args[1], &needle_data, &needle_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    bool validate = false;
    if (argc > 2) { napi_get_value_bool(env, args[2], &validate); }
    if (validate && (sz_utf8_find_malformed((sz_cptr_t)haystack_data, haystack_length) != STRINGZILLA_NULL_CHAR ||
                     sz_utf8_find_malformed((sz_cptr_t)needle_data, needle_length) != STRINGZILLA_NULL_CHAR)) {
        napi_throw_error(env, NULL, "Input is not valid UTF-8");
        return NULL;
    }

    sz_utf8_uncased_needle_metadata_t metadata = {0};
    sz_cptr_t match;
    sz_size_t match_length = 0;
    if (!check_status(env, sz_utf8_uncased_search_best((sz_cptr_t)haystack_data, haystack_length,
                                                       (sz_cptr_t)needle_data, needle_length, &metadata, &match,
                                                       &match_length, default_capabilities, NULL)))
        return NULL;

    if (!match) return makeFindResultObject(env, -1, 0);
    return makeFindResultObject(env, (int64_t)(match - (sz_cptr_t)haystack_data), (uint64_t)match_length);
}

typedef struct {
    sz_u8_t *needle_data;
    size_t needle_length;
    sz_utf8_uncased_needle_metadata_t metadata;
} utf8_uncased_needle_t;

static void utf8_uncased_needle_cleanup(napi_env env, void *data, void *hint) {
    utf8_uncased_needle_t *needle = (utf8_uncased_needle_t *)data;
    if (!needle) return;
    if (needle->needle_data) free(needle->needle_data);
    free(needle);
}

napi_value utf8UncasedNeedleConstructor(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    if (argc < 1) {
        napi_throw_error(env, NULL, "Utf8UncasedNeedle(needle, validate?) expects at least 1 argument");
        return NULL;
    }

    void *needle_data;
    size_t needle_length;
    napi_status status = napi_get_buffer_info(env, args[0], &needle_data, &needle_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    bool validate = false;
    if (argc > 1) { napi_get_value_bool(env, args[1], &validate); }
    if (validate && sz_utf8_find_malformed((sz_cptr_t)needle_data, needle_length) != STRINGZILLA_NULL_CHAR) {
        napi_throw_error(env, NULL, "Needle is not valid UTF-8");
        return NULL;
    }

    utf8_uncased_needle_t *needle = (utf8_uncased_needle_t *)malloc(sizeof(*needle));
    if (!needle) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }
    needle->needle_length = needle_length;
    needle->metadata = (sz_utf8_uncased_needle_metadata_t) {0};
    needle->needle_data = NULL;

    if (needle_length) {
        needle->needle_data = (sz_u8_t *)malloc(needle_length);
        if (!needle->needle_data) {
            free(needle);
            napi_throw_error(env, NULL, "Memory allocation failed");
            return NULL;
        }
        if (!check_status(env, sz_copy_best((sz_ptr_t)needle->needle_data, (sz_cptr_t)needle_data, needle_length,
                                            default_capabilities, NULL))) {
            free(needle->needle_data);
            free(needle);
            return NULL;
        }
    }

    napi_wrap(env, js_this, needle, utf8_uncased_needle_cleanup, NULL, NULL);
    return js_this;
}

napi_value utf8UncasedNeedleFindIn(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    if (argc < 1) {
        napi_throw_error(env, NULL, "findIn(haystack, validate?) expects at least 1 argument");
        return NULL;
    }

    utf8_uncased_needle_t *needle = unwrap_this(env, js_this);
    if (!needle) return NULL;

    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    bool validate = false;
    if (argc > 1) { napi_get_value_bool(env, args[1], &validate); }
    if (validate && sz_utf8_find_malformed((sz_cptr_t)haystack_data, haystack_length) != STRINGZILLA_NULL_CHAR) {
        napi_throw_error(env, NULL, "Haystack is not valid UTF-8");
        return NULL;
    }

    sz_cptr_t match;
    sz_size_t match_length = 0;
    if (!check_status(
            env, sz_utf8_uncased_search_best((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle->needle_data,
                                             needle->needle_length, &needle->metadata, &match, &match_length,
                                             default_capabilities, NULL)))
        return NULL;
    if (!match) return makeFindResultObject(env, -1, 0);
    return makeFindResultObject(env, (int64_t)(match - (sz_cptr_t)haystack_data), (uint64_t)match_length);
}

napi_value utf8NormAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_error(env, NULL, "utf8Norm(buffer, form, validate?) expects at least 2 arguments");
        return NULL;
    }

    void *source_data;
    size_t source_length;
    napi_status status = napi_get_buffer_info(env, args[0], &source_data, &source_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    double form_double;
    status = napi_get_value_double(env, args[1], &form_double);
    if (status != napi_ok || form_double < 0 || form_double > 3) {
        napi_throw_error(env, NULL, "Second argument must be a Utf8NormalForm value (NFD, NFC, NFKD, or NFKC)");
        return NULL;
    }
    sz_normal_form_t form = (sz_normal_form_t)form_double;

    bool validate = false;
    if (argc > 2) { napi_get_value_bool(env, args[2], &validate); }
    if (validate && sz_utf8_find_malformed((sz_cptr_t)source_data, source_length) != STRINGZILLA_NULL_CHAR) {
        napi_throw_error(env, NULL, "Input is not valid UTF-8");
        return NULL;
    }

    // A single-codepoint compatibility decomposition expands up to 18×, per `sz_utf8_norm_best`.
    size_t capacity = source_length * 18;
    void *destination = capacity ? malloc(capacity) : NULL;
    if (capacity && !destination) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }

    sz_size_t out_length = 0;
    if (source_length &&
        !check_status(env, sz_utf8_norm_best((sz_cptr_t)source_data, source_length, form, (sz_ptr_t)destination,
                                             &out_length, default_capabilities, NULL))) {
        free(destination);
        return NULL;
    }

    if (out_length == 0) {
        if (destination) free(destination);
        napi_value js_empty;
        void *unused;
        napi_create_buffer(env, 0, &unused, &js_empty);
        return js_empty;
    }

    // Shrink to the actual size without a second pass or a copy.
    void *shrunk = realloc(destination, (size_t)out_length);
    if (shrunk) destination = shrunk;

    napi_value js_result;
    napi_create_external_buffer(env, (size_t)out_length, destination, external_buffer_cleanup, NULL, &js_result);
    return js_result;
}

napi_value utf8FindDenormalizedAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_error(env, NULL, "utf8FindDenormalized(buffer, form) expects 2 arguments");
        return NULL;
    }

    void *source_data;
    size_t source_length;
    napi_status status = napi_get_buffer_info(env, args[0], &source_data, &source_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    double form_double;
    status = napi_get_value_double(env, args[1], &form_double);
    if (status != napi_ok || form_double < 0 || form_double > 3) {
        napi_throw_error(env, NULL, "Second argument must be a Utf8NormalForm value (NFD, NFC, NFKD, or NFKC)");
        return NULL;
    }
    sz_normal_form_t form = (sz_normal_form_t)form_double;

    sz_cptr_t violation;
    if (!check_status(env, sz_utf8_find_denormalized_best((sz_cptr_t)source_data, source_length, form, &violation,
                                                          default_capabilities, NULL)))
        return NULL;

    napi_value js_result;
    if (violation == STRINGZILLA_NULL_CHAR) { napi_create_bigint_int64(env, -1, &js_result); }
    else { napi_create_bigint_uint64(env, violation - (sz_cptr_t)source_data, &js_result); }
    return js_result;
}

/** The dispatch point a segmenter class carries, like @c sz_utf8_wordbreaks_best. */
typedef sz_status_t (*utf8_segmenter_best_t)(sz_cptr_t, sz_size_t, sz_size_t *, sz_size_t *, sz_size_t, sz_size_t *,
                                             sz_size_t *, sz_capability_t, void *);

typedef struct {
    napi_ref text_ref; // Keeps the source Buffer alive while the iterator holds pointers into it
    sz_cptr_t text_data;
    sz_size_t text_length;
    sz_size_t cursor;
    utf8_segmenter_best_t kernel;
    sz_size_t batch_starts[sz_iterators_default_steps_k];
    sz_size_t batch_lengths[sz_iterators_default_steps_k];
    sz_size_t batch_count;
    sz_size_t batch_index;
} utf8_segments_t;

static void utf8_segments_cleanup(napi_env env, void *data, void *hint) {
    utf8_segments_t *segments = (utf8_segments_t *)data;
    if (!segments) return;
    if (segments->text_ref) napi_delete_reference(env, segments->text_ref);
    free(segments);
}

napi_value utf8SegmentsConstructor(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value js_this;
    void *kernel;
    napi_get_cb_info(env, info, &argc, args, &js_this, &kernel);

    if (argc < 1) {
        napi_throw_error(env, NULL, "Segmenter(buffer, validate?) expects at least 1 argument");
        return NULL;
    }

    void *text_data;
    size_t text_length;
    napi_status status = napi_get_buffer_info(env, args[0], &text_data, &text_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    bool validate = false;
    if (argc > 1) { napi_get_value_bool(env, args[1], &validate); }
    if (validate && sz_utf8_find_malformed((sz_cptr_t)text_data, text_length) != STRINGZILLA_NULL_CHAR) {
        napi_throw_error(env, NULL, "Input is not valid UTF-8");
        return NULL;
    }

    utf8_segments_t *segments = (utf8_segments_t *)malloc(sizeof(*segments));
    if (!segments) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }
    segments->text_data = (sz_cptr_t)text_data;
    segments->text_length = (sz_size_t)text_length;
    segments->cursor = 0;
    segments->kernel = (utf8_segmenter_best_t)kernel;
    segments->batch_count = 0;
    segments->batch_index = 0;
    segments->text_ref = NULL;
    napi_create_reference(env, args[0], 1, &segments->text_ref);

    napi_wrap(env, js_this, segments, utf8_segments_cleanup, NULL, NULL);
    return js_this;
}

napi_value utf8SegmentsNext(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    utf8_segments_t *segments = unwrap_this(env, js_this);
    if (!segments) return NULL;

    if (segments->batch_index >= segments->batch_count) {
        // Resume from the end of the last buffered segment - a guaranteed break boundary.
        if (segments->batch_count) {
            sz_size_t const last = segments->batch_count - 1;
            segments->cursor += segments->batch_starts[last] + segments->batch_lengths[last];
        }
        napi_value js_null;
        napi_get_null(env, &js_null);
        if (segments->cursor >= segments->text_length) return js_null;
        sz_size_t count = 0, consumed = 0;
        if (!check_status(
                env, segments->kernel(segments->text_data + segments->cursor, segments->text_length - segments->cursor,
                                      segments->batch_starts, segments->batch_lengths, sz_iterators_default_steps_k,
                                      &count, &consumed, default_capabilities, NULL)))
            return NULL;
        segments->batch_count = count;
        segments->batch_index = 0;
        if (segments->batch_count == 0) {
            // Trailing bytes without a single segment (e.g. closing whitespace) - the iteration is over.
            segments->cursor = segments->text_length;
            return js_null;
        }
    }

    sz_size_t const i = segments->batch_index++;
    napi_value js_obj, js_start, js_length;
    napi_create_object(env, &js_obj);
    napi_create_bigint_uint64(env, (uint64_t)(segments->cursor + segments->batch_starts[i]), &js_start);
    napi_create_bigint_uint64(env, (uint64_t)segments->batch_lengths[i], &js_length);
    napi_set_named_property(env, js_obj, "start", js_start);
    napi_set_named_property(env, js_obj, "length", js_length);
    return js_obj;
}

napi_value countAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    void *haystack_data, *needle_data;
    size_t haystack_length, needle_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }
    status = napi_get_buffer_info(env, args[1], &needle_data, &needle_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    bool overlap = false;
    if (argc > 2) { napi_get_value_bool(env, args[2], &overlap); }

    sz_string_view_t haystack = {(sz_cptr_t)haystack_data, haystack_length};
    sz_string_view_t needle = {(sz_cptr_t)needle_data, needle_length};

    size_t count = 0;
    if (needle.length == 0 || haystack.length == 0 || haystack.length < needle.length) { count = 0; }
    else if (overlap) {
        while (haystack.length) {
            sz_cptr_t ptr;
            if (!check_status(env, sz_find_best(haystack.start, haystack.length, needle.start, needle.length, &ptr,
                                                default_capabilities, NULL)))
                return NULL;
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + found;
            haystack.length -= offset + found;
        }
    }
    else {
        while (haystack.length) {
            sz_cptr_t ptr;
            if (!check_status(env, sz_find_best(haystack.start, haystack.length, needle.start, needle.length, &ptr,
                                                default_capabilities, NULL)))
                return NULL;
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + needle.length;
            haystack.length -= offset + needle.length * found;
        }
    }

    napi_value js_count;
    napi_create_bigint_uint64(env, count, &js_count);
    return js_count;
}

napi_value hashAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for data (zero-copy)
    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get optional seed parameter (default to 0)
    sz_u64_t seed = 0;
    if (argc > 1 && !seed_from_js(env, args[1], &seed)) return NULL;

    // Compute hash using StringZilla
    sz_u64_t hash_result;
    if (!check_status(
            env, sz_hash_best((sz_cptr_t)buffer_data, buffer_length, seed, &hash_result, default_capabilities, NULL)))
        return NULL;

    // Convert result to JavaScript BigInt
    napi_value js_result;
    napi_create_bigint_uint64(env, hash_result, &js_result);

    return js_result;
}

static void hasher_cleanup(napi_env env, void *data, void *hint) { free(data); }
typedef struct {
    sz_hash_state_t state;
    sz_u64_t seed; // Used for `reset`
} hasher_t;

napi_value hasherConstructor(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    sz_u64_t seed = 0;
    if (argc > 0 && !seed_from_js(env, args[0], &seed)) return NULL;

    hasher_t *hasher = malloc(sizeof(hasher_t));
    if (!hasher) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }
    hasher->seed = seed;
    if (!check_status(env, sz_hash_state_init_best(&hasher->state, seed, default_capabilities, NULL))) {
        free(hasher);
        return NULL;
    }
    napi_wrap(env, js_this, hasher, hasher_cleanup, NULL, NULL);

    return js_this;
}

napi_value hasherUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    if (!check_status(env, sz_hash_state_update_best(&hasher->state, (sz_cptr_t)buffer_data, buffer_length,
                                                     default_capabilities, NULL)))
        return NULL;
    return js_this;
}

napi_value hasherDigest(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    sz_u64_t hash;
    if (!check_status(env, sz_hash_state_digest_best(&hasher->state, &hash, default_capabilities, NULL))) return NULL;
    napi_value js_result;
    napi_create_bigint_uint64(env, hash, &js_result);

    return js_result;
}

napi_value hasherReset(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    if (!check_status(env, sz_hash_state_init_best(&hasher->state, hasher->seed, default_capabilities, NULL)))
        return NULL;
    return js_this;
}

napi_value sha256API(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for data (zero-copy)
    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    // Compute SHA-256 using StringZilla
    sz_u8_t digest[32];
    sz_sha256_state_t state;
    if (!check_status(env, sz_sha256_state_init_best(&state, default_capabilities, NULL)) ||
        !check_status(env, sz_sha256_state_update_best(&state, (sz_cptr_t)buffer_data, buffer_length,
                                                       default_capabilities, NULL)) ||
        !check_status(env, sz_sha256_state_digest_best(&state, digest, default_capabilities, NULL)))
        return NULL;

    // Convert result to JavaScript Buffer
    napi_value js_result;
    void *result_data;
    napi_create_buffer_copy(env, 32, digest, &result_data, &js_result);

    return js_result;
}

static void sha256_hasher_cleanup(napi_env env, void *data, void *hint) { free(data); }
typedef struct {
    sz_sha256_state_t state;
} sha256_hasher_t;

napi_value sha256HasherConstructor(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    sha256_hasher_t *hasher = malloc(sizeof(sha256_hasher_t));
    if (!hasher) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }
    if (!check_status(env, sz_sha256_state_init_best(&hasher->state, default_capabilities, NULL))) {
        free(hasher);
        return NULL;
    }
    napi_wrap(env, js_this, hasher, sha256_hasher_cleanup, NULL, NULL);

    return js_this;
}

napi_value sha256HasherUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    sha256_hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    if (!check_status(env, sz_sha256_state_update_best(&hasher->state, (sz_cptr_t)buffer_data, buffer_length,
                                                       default_capabilities, NULL)))
        return NULL;
    return js_this;
}

napi_value sha256HasherDigest(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    sha256_hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    sz_u8_t digest[32];
    if (!check_status(env, sz_sha256_state_digest_best(&hasher->state, digest, default_capabilities, NULL)))
        return NULL;

    // Convert result to JavaScript Buffer
    napi_value js_result;
    void *result_data;
    napi_create_buffer_copy(env, 32, digest, &result_data, &js_result);

    return js_result;
}

napi_value sha256HasherHexdigest(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    sha256_hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    sz_u8_t digest[32];
    if (!check_status(env, sz_sha256_state_digest_best(&hasher->state, digest, default_capabilities, NULL)))
        return NULL;

    // Convert to hex string
    char hex[65];
    for (int i = 0; i < 32; i++) { sprintf(&hex[i * 2], "%02x", digest[i]); }
    hex[64] = '\0';

    napi_value js_result;
    napi_create_string_utf8(env, hex, 64, &js_result);

    return js_result;
}

napi_value sha256HasherReset(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    sha256_hasher_t *hasher = unwrap_this(env, js_this);
    if (!hasher) return NULL;

    if (!check_status(env, sz_sha256_state_init_best(&hasher->state, default_capabilities, NULL))) return NULL;
    return js_this;
}

napi_value findLastAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for haystack (zero-copy)
    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get buffer info for needle (zero-copy)
    void *needle_data;
    size_t needle_length;
    status = napi_get_buffer_info(env, args[1], &needle_data, &needle_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (needle_length == 0) { napi_create_bigint_int64(env, haystack_length, &js_result); }
    else {
        sz_cptr_t result;
        if (!check_status(env, sz_rfind_best((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle_data,
                                             needle_length, &result, default_capabilities, NULL)))
            return NULL;

        // In JavaScript, if `lastIndexOf` is unable to find the specified value, then it should return -1
        if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }
    }

    return js_result;
}

napi_value findByteAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for haystack (zero-copy)
    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get byte value (as number)
    double byte_value_double;
    status = napi_get_value_double(env, args[1], &byte_value_double);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a number");
        return NULL;
    }

    sz_u8_t byte_value = (sz_u8_t)byte_value_double;

    // Find the byte using sz_find_byte_best (needs pointer to byte)
    char byte_char = (char)byte_value;
    sz_cptr_t result;
    if (!check_status(env, sz_find_byte_best((sz_cptr_t)haystack_data, haystack_length, &byte_char, &result,
                                             default_capabilities, NULL)))
        return NULL;

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
    else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }

    return js_result;
}

napi_value findLastByteAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for haystack (zero-copy)
    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get byte value (as number)
    double byte_value_double;
    status = napi_get_value_double(env, args[1], &byte_value_double);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a number");
        return NULL;
    }

    sz_u8_t byte_value = (sz_u8_t)byte_value_double;

    // Find the last byte using sz_rfind_byte_best (needs pointer to byte)
    char byte_char = (char)byte_value;
    sz_cptr_t result;
    if (!check_status(env, sz_rfind_byte_best((sz_cptr_t)haystack_data, haystack_length, &byte_char, &result,
                                              default_capabilities, NULL)))
        return NULL;

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
    else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }

    return js_result;
}

napi_value findByteFromAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for haystack (zero-copy)
    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get buffer info for allowed bytes (zero-copy)
    void *allowed_data;
    size_t allowed_length;
    status = napi_get_buffer_info(env, args[1], &allowed_data, &allowed_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    // Find first byte that is in the allowed set using sz_find_byteset_best
    sz_byteset_t byteset;
    sz_byteset_init(&byteset);
    for (size_t i = 0; i < allowed_length; i++) { sz_byteset_add_u8(&byteset, ((sz_u8_t *)allowed_data)[i]); }
    sz_cptr_t result;
    if (!check_status(env, sz_find_byteset_best((sz_cptr_t)haystack_data, haystack_length, &byteset, &result,
                                                default_capabilities, NULL)))
        return NULL;

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
    else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }

    return js_result;
}

napi_value findLastByteFromAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for haystack (zero-copy)
    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get buffer info for allowed bytes (zero-copy)
    void *allowed_data;
    size_t allowed_length;
    status = napi_get_buffer_info(env, args[1], &allowed_data, &allowed_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    // Find last byte that is in the allowed set using sz_rfind_byteset_best
    sz_byteset_t byteset;
    sz_byteset_init(&byteset);
    for (size_t i = 0; i < allowed_length; i++) { sz_byteset_add_u8(&byteset, ((sz_u8_t *)allowed_data)[i]); }
    sz_cptr_t result;
    if (!check_status(env, sz_rfind_byteset_best((sz_cptr_t)haystack_data, haystack_length, &byteset, &result,
                                                 default_capabilities, NULL)))
        return NULL;

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
    else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }

    return js_result;
}

napi_value equalAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for first buffer (zero-copy)
    void *first_data;
    size_t first_length;
    napi_status status = napi_get_buffer_info(env, args[0], &first_data, &first_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get buffer info for second buffer (zero-copy)
    void *second_data;
    size_t second_length;
    status = napi_get_buffer_info(env, args[1], &second_data, &second_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    // Compare for equality - need to check length first, then content
    sz_bool_t equal = first_length == second_length ? sz_true_k : sz_false_k;
    if (equal && first_length &&
        !check_status(env, sz_equal_best((sz_cptr_t)first_data, (sz_cptr_t)second_data, first_length, &equal,
                                         default_capabilities, NULL)))
        return NULL;

    // Convert to JavaScript boolean and return
    napi_value js_result;
    napi_get_boolean(env, equal, &js_result);

    return js_result;
}

napi_value compareAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for first buffer (zero-copy)
    void *first_data;
    size_t first_length;
    napi_status status = napi_get_buffer_info(env, args[0], &first_data, &first_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // Get buffer info for second buffer (zero-copy)
    void *second_data;
    size_t second_length;
    status = napi_get_buffer_info(env, args[1], &second_data, &second_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Second argument must be a Buffer");
        return NULL;
    }

    // Compare using sz_order_best
    sz_ordering_t order;
    if (!check_status(env, sz_order_best((sz_cptr_t)first_data, first_length, (sz_cptr_t)second_data, second_length,
                                         &order, default_capabilities, NULL)))
        return NULL;

    // Convert to JavaScript number and return
    napi_value js_result;
    napi_create_int32(env, order, &js_result);

    return js_result;
}

napi_value byteSumAPI(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for data (zero-copy)
    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    // Compute byte sum using sz_bytesum_best
    sz_u64_t sum;
    if (!check_status(env, sz_bytesum_best((sz_cptr_t)buffer_data, buffer_length, &sum, default_capabilities, NULL)))
        return NULL;

    // Convert to JavaScript BigInt and return
    napi_value js_result;
    napi_create_bigint_uint64(env, sum, &js_result);

    return js_result;
}

napi_value utf8CountAPI(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for data (zero-copy)
    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    // Count the UTF-8 codepoints using `sz_utf8_count_best`
    sz_size_t codepoints;
    if (!check_status(
            env, sz_utf8_count_best((sz_cptr_t)buffer_data, buffer_length, &codepoints, default_capabilities, NULL)))
        return NULL;

    // Convert to JavaScript BigInt and return
    napi_value js_result;
    napi_create_bigint_uint64(env, (uint64_t)codepoints, &js_result);

    return js_result;
}

napi_value utf8SeekAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Get buffer info for data (zero-copy)
    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    // The codepoint index accepts both Number and BigInt, matching the other offset-taking exports
    int64_t codepoint_index = 0;
    napi_valuetype index_type;
    napi_typeof(env, args[1], &index_type);
    if (index_type == napi_bigint) {
        bool lossless;
        napi_get_value_bigint_int64(env, args[1], &codepoint_index, &lossless);
    }
    else {
        double as_double;
        if (napi_get_value_double(env, args[1], &as_double) != napi_ok) {
            napi_throw_error(env, NULL, "Second argument must be a number or BigInt");
            return NULL;
        }
        codepoint_index = (int64_t)as_double;
    }
    if (codepoint_index < 0) {
        napi_throw_error(env, NULL, "Codepoint index must not be negative");
        return NULL;
    }

    // Resolve the codepoint index to a byte offset, or -1 when the text is too short
    sz_cptr_t start = (sz_cptr_t)buffer_data;
    sz_cptr_t found;
    if (!check_status(env, sz_utf8_seek_best(start, buffer_length, (sz_size_t)codepoint_index, &found,
                                             default_capabilities, NULL)))
        return NULL;

    napi_value js_result;
    if (found == NULL) napi_create_bigint_int64(env, -1, &js_result);
    else napi_create_bigint_uint64(env, (uint64_t)(found - start), &js_result);

    return js_result;
}

/** Device kinds, numbered as the JavaScript @c Device passes them. */
enum { device_cpu_k, device_cuda_k, device_rocm_k, device_metal_k };

/** Reads the @p count leading arguments of a @c Device call: a kind, then an ordinal or a mask. */
static bool read_device_arguments(napi_env env, napi_callback_info info, size_t count, uint32_t *kind,
                                  uint32_t *ordinal, uint64_t *mask) {
    size_t argc = 2;
    napi_value args[2];
    bool lossless;
    if (napi_get_cb_info(env, info, &argc, args, NULL, NULL) != napi_ok || argc < count ||
        napi_get_value_uint32(env, args[0], kind) != napi_ok ||
        (count == 2 && ordinal && napi_get_value_uint32(env, args[1], ordinal) != napi_ok) ||
        (count == 2 && mask && napi_get_value_bigint_uint64(env, args[1], mask, &lossless) != napi_ok)) {
        napi_throw_type_error(env, NULL, "Expected a device kind, then an ordinal or a BigInt capability mask");
        return false;
    }
    return true;
}

/** Returns @p capabilities as a BigInt, or throws when @p status failed. */
static napi_value capabilities_or_throw(napi_env env, sz_status_t status, sz_capability_t capabilities) {
    if (!check_status(env, status)) return NULL;
    napi_value result;
    napi_create_bigint_uint64(env, (uint64_t)capabilities, &result);
    return result;
}

/** Counts the devices of a kind: one CPU, or the GPUs its runtime sees, throwing without one. */
napi_value deviceCountAPI(napi_env env, napi_callback_info info) {
    uint32_t kind;
    if (!read_device_arguments(env, info, 1, &kind, NULL, NULL)) return NULL;
    sz_size_t count = 1;
    sz_status_t status = sz_success_k;
    switch (kind) {
    case device_cpu_k: break;
    case device_cuda_k: status = sz_cuda_count_devices(&count); break;
    case device_rocm_k: status = sz_rocm_count_devices(&count); break;
    case device_metal_k: status = sz_metal_count_devices(&count); break;
    default: status = sz_missing_gpu_k;
    }
    if (!check_status(env, status)) return NULL;
    napi_value result;
    napi_create_uint32(env, (uint32_t)count, &result);
    return result;
}

/** The capabilities a device of a kind and ordinal runs, whether or not they were compiled in. */
napi_value capabilitiesDetectedAPI(napi_env env, napi_callback_info info) {
    uint32_t kind, ordinal;
    if (!read_device_arguments(env, info, 2, &kind, &ordinal, NULL)) return NULL;
    sz_capability_t capabilities = 0;
    sz_status_t status;
    switch (kind) {
    case device_cpu_k: status = sz_cpu_capabilities_detected(&capabilities); break;
    case device_cuda_k: status = sz_cuda_capabilities_detected(ordinal, &capabilities); break;
    case device_rocm_k: status = sz_rocm_capabilities_detected(ordinal, &capabilities); break;
    case device_metal_k: status = sz_metal_capabilities_detected(ordinal, &capabilities); break;
    default: status = sz_missing_gpu_k;
    }
    return capabilities_or_throw(env, status, capabilities);
}

/** The capabilities compiled in for devices of a kind, whether or not a device runs them. */
napi_value capabilitiesCompiledAPI(napi_env env, napi_callback_info info) {
    uint32_t kind;
    if (!read_device_arguments(env, info, 1, &kind, NULL, NULL)) return NULL;
    sz_capability_t capabilities = 0;
    sz_status_t status;
    switch (kind) {
    case device_cpu_k: status = sz_cpu_capabilities_compiled(&capabilities); break;
    case device_cuda_k: status = sz_cuda_capabilities_compiled(&capabilities); break;
    case device_rocm_k: status = sz_rocm_capabilities_compiled(&capabilities); break;
    case device_metal_k: status = sz_metal_capabilities_compiled(&capabilities); break;
    default: status = sz_missing_gpu_k;
    }
    return capabilities_or_throw(env, status, capabilities);
}

/** The mask a device's calls pass; on the CPU, the one every call of this addon dispatches over. */
napi_value capabilitiesEnabledAPI(napi_env env, napi_callback_info info) {
    uint32_t kind, ordinal;
    if (!read_device_arguments(env, info, 2, &kind, &ordinal, NULL)) return NULL;
    sz_capability_t capabilities = default_capabilities;
    sz_status_t status;
    switch (kind) {
    case device_cpu_k: status = sz_success_k; break;
    case device_cuda_k: status = sz_cuda_capabilities_enabled(ordinal, &capabilities); break;
    case device_rocm_k: status = sz_rocm_capabilities_enabled(ordinal, &capabilities); break;
    case device_metal_k: status = sz_metal_capabilities_enabled(ordinal, &capabilities); break;
    default: status = sz_missing_gpu_k;
    }
    return capabilities_or_throw(env, status, capabilities);
}

/** Makes a BigInt mask the CPU's enabled set, clamped to detected and compiled and keeping the
 *  serial fallback, and returns the set that took effect; GPUs keep no such set. */
napi_value capabilitiesEnableAPI(napi_env env, napi_callback_info info) {
    uint32_t kind;
    uint64_t wanted;
    if (!read_device_arguments(env, info, 2, &kind, NULL, &wanted)) return NULL;
    if (kind != device_cpu_k) return capabilities_or_throw(env, sz_missing_kernel_k, 0);
    sz_capability_t available = sz_cap_serial_k;
    if (!check_status(env, sz_cpu_capabilities_enabled(&available))) return NULL;
    default_capabilities = ((sz_capability_t)wanted & available) | sz_cap_serial_k;
    return capabilities_or_throw(env, sz_success_k, default_capabilities);
}

/** Builds the @c Capability object, mapping each capability's name to its BigInt bit, and the
 *  @c cpus, @c devices and @c any groups to theirs. */
static napi_status create_capability_names(napi_env env, napi_value *names) {
    napi_value value;
    napi_status status = napi_create_object(env, names);
    if (status != napi_ok) return status;
    for (unsigned shift = 0; shift != 64; ++shift) {
        sz_capability_t const bit = (sz_capability_t)1 << shift;
        char name[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
        if (!sz_capabilities_name(bit, name, sizeof(name))) continue;
        if ((status = napi_create_bigint_uint64(env, (uint64_t)bit, &value)) != napi_ok ||
            (status = napi_set_named_property(env, *names, name, value)) != napi_ok)
            return status;
    }
    struct {
        char const *name;
        sz_capability_t mask;
    } const groups[] = {{"cpus", sz_cap_cpus_k}, {"devices", sz_cap_devices_k}, {"any", sz_cap_any_k}};
    for (size_t group = 0; group != sizeof(groups) / sizeof(groups[0]); ++group)
        if ((status = napi_create_bigint_uint64(env, (uint64_t)groups[group].mask, &value)) != napi_ok ||
            (status = napi_set_named_property(env, *names, groups[group].name, value)) != napi_ok)
            return status;
    return napi_ok;
}

napi_value Init(napi_env env, napi_value exports) {
    sz_cpu_capabilities_enabled(&default_capabilities);
    sz_cpu_configure_thread(default_capabilities);

    // Create Hasher class constructor
    napi_value hasherClass;
    napi_property_descriptor hasherProps[] = {
        {"update", 0, hasherUpdate, 0, 0, 0, napi_default, 0},
        {"digest", 0, hasherDigest, 0, 0, 0, napi_default, 0},
        {"reset", 0, hasherReset, 0, 0, 0, napi_default, 0},
    };
    napi_define_class(env, "Hasher", NAPI_AUTO_LENGTH, hasherConstructor, NULL,
                      sizeof(hasherProps) / sizeof(hasherProps[0]), hasherProps, &hasherClass);

    // Create Sha256 class constructor
    napi_value sha256HasherClass;
    napi_property_descriptor sha256HasherProps[] = {
        {"update", 0, sha256HasherUpdate, 0, 0, 0, napi_default, 0},
        {"digest", 0, sha256HasherDigest, 0, 0, 0, napi_default, 0},
        {"hexdigest", 0, sha256HasherHexdigest, 0, 0, 0, napi_default, 0},
        {"reset", 0, sha256HasherReset, 0, 0, 0, napi_default, 0},
    };
    napi_define_class(env, "Sha256", NAPI_AUTO_LENGTH, sha256HasherConstructor, NULL,
                      sizeof(sha256HasherProps) / sizeof(sha256HasherProps[0]), sha256HasherProps, &sha256HasherClass);

    // Create Utf8UncasedNeedle class constructor
    napi_value utf8NeedleClass;
    napi_property_descriptor utf8NeedleProps[] = {
        {"findIn", 0, utf8UncasedNeedleFindIn, 0, 0, 0, napi_default, 0},
    };
    napi_define_class(env, "Utf8UncasedNeedle", NAPI_AUTO_LENGTH, utf8UncasedNeedleConstructor, NULL,
                      sizeof(utf8NeedleProps) / sizeof(utf8NeedleProps[0]), utf8NeedleProps, &utf8NeedleClass);

    // Create the four TR29/UAX14 segmenter classes sharing one implementation, parameterized by kernel
    napi_property_descriptor segmenterProps[] = {
        {"next", 0, utf8SegmentsNext, 0, 0, 0, napi_default, 0},
    };
    napi_value utf8WordbreaksClass, utf8GraphemesClass, utf8SentencesClass, utf8LinebreaksClass;
    napi_define_class(env, "Utf8Wordbreaks", NAPI_AUTO_LENGTH, utf8SegmentsConstructor,
                      (void *)&sz_utf8_wordbreaks_best, sizeof(segmenterProps) / sizeof(segmenterProps[0]),
                      segmenterProps, &utf8WordbreaksClass);
    napi_define_class(env, "Utf8Graphemes", NAPI_AUTO_LENGTH, utf8SegmentsConstructor, (void *)&sz_utf8_graphemes_best,
                      sizeof(segmenterProps) / sizeof(segmenterProps[0]), segmenterProps, &utf8GraphemesClass);
    napi_define_class(env, "Utf8Sentences", NAPI_AUTO_LENGTH, utf8SegmentsConstructor, (void *)&sz_utf8_sentences_best,
                      sizeof(segmenterProps) / sizeof(segmenterProps[0]), segmenterProps, &utf8SentencesClass);
    napi_define_class(env, "Utf8Linebreaks", NAPI_AUTO_LENGTH, utf8SegmentsConstructor,
                      (void *)&sz_utf8_linebreaks_best, sizeof(segmenterProps) / sizeof(segmenterProps[0]),
                      segmenterProps, &utf8LinebreaksClass);

    // Define function exports
    napi_property_descriptor findDesc = {"indexOf", 0, indexOfAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor findLastDesc = {"lastIndexOf", 0, findLastAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor findByteDesc = {"findByte", 0, findByteAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor findLastByteDesc = {"findLastByte", 0, findLastByteAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor findByteFromDesc = {"findByteFrom", 0, findByteFromAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor findLastByteFromDesc = {"findLastByteFrom", 0, findLastByteFromAPI, 0, 0, 0,
                                                     napi_default,       0};
    napi_property_descriptor countDesc = {"count", 0, countAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor hashDesc = {"hash", 0, hashAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor sha256Desc = {"sha256", 0, sha256API, 0, 0, 0, napi_default, 0};
    napi_property_descriptor equalDesc = {"equal", 0, equalAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor compareDesc = {"compare", 0, compareAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor byteSumDesc = {"byteSum", 0, byteSumAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8UncasedFoldDesc = {"utf8UncasedFold", 0, utf8UncasedFoldAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8UncasedFindDesc = {"utf8UncasedFind", 0, utf8UncasedFindAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8NormDesc = {"utf8Norm", 0, utf8NormAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8CountDesc = {"utf8Count", 0, utf8CountAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8SeekDesc = {"utf8Seek", 0, utf8SeekAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8FindDenormalizedDesc = {"utf8FindDenormalized", 0, utf8FindDenormalizedAPI, 0, 0, 0,
                                                         napi_default,           0};
    napi_property_descriptor utf8WordsDesc = {"Utf8Wordbreaks", 0, 0, 0, 0, utf8WordbreaksClass, napi_default, 0};
    napi_property_descriptor utf8GraphemesDesc = {"Utf8Graphemes", 0, 0, 0, 0, utf8GraphemesClass, napi_default, 0};
    napi_property_descriptor utf8SentencesDesc = {"Utf8Sentences", 0, 0, 0, 0, utf8SentencesClass, napi_default, 0};
    napi_property_descriptor utf8LinebreaksDesc = {"Utf8Linebreaks", 0, 0, 0, 0, utf8LinebreaksClass, napi_default, 0};
    napi_property_descriptor hasherDesc = {"Hasher", 0, 0, 0, 0, hasherClass, napi_default, 0};
    napi_property_descriptor sha256HasherDesc = {"Sha256", 0, 0, 0, 0, sha256HasherClass, napi_default, 0};
    napi_property_descriptor utf8NeedleDesc = {"Utf8UncasedNeedle", 0, 0, 0, 0, utf8NeedleClass, napi_default, 0};

    napi_property_descriptor deviceCountDesc = {"deviceCount", 0, deviceCountAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor capabilitiesDetectedDesc = {"capabilitiesDetected", 0, capabilitiesDetectedAPI, 0, 0, 0,
                                                         napi_default,           0};
    napi_property_descriptor capabilitiesCompiledDesc = {"capabilitiesCompiled", 0, capabilitiesCompiledAPI, 0, 0, 0,
                                                         napi_default,           0};
    napi_property_descriptor capabilitiesEnabledDesc = {"capabilitiesEnabled", 0, capabilitiesEnabledAPI, 0, 0, 0,
                                                        napi_default,          0};
    napi_property_descriptor capabilitiesEnableDesc = {"capabilitiesEnable", 0, capabilitiesEnableAPI, 0, 0, 0,
                                                       napi_default,         0};
    napi_value capability_names;
    if (create_capability_names(env, &capability_names) != napi_ok) return NULL;
    napi_property_descriptor capabilityDesc = {"Capability", 0, 0, 0, 0, capability_names, napi_default, 0};

    napi_property_descriptor properties[] = {
        findDesc,
        findLastDesc,
        findByteDesc,
        findLastByteDesc,
        findByteFromDesc,
        findLastByteFromDesc,
        countDesc,
        hashDesc,
        sha256Desc,
        equalDesc,
        compareDesc,
        byteSumDesc,
        utf8UncasedFoldDesc,
        utf8UncasedFindDesc,
        utf8NormDesc,
        utf8CountDesc,
        utf8SeekDesc,
        utf8FindDenormalizedDesc,
        utf8WordsDesc,
        utf8GraphemesDesc,
        utf8SentencesDesc,
        utf8LinebreaksDesc,
        hasherDesc,
        sha256HasherDesc,
        utf8NeedleDesc,
        deviceCountDesc,
        capabilitiesDetectedDesc,
        capabilitiesCompiledDesc,
        capabilitiesEnabledDesc,
        capabilitiesEnableDesc,
        capabilityDesc,
    };

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
