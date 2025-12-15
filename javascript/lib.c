/**
 *  @file       lib.c
 *  @brief      JavaScript bindings for StringZilla.
 *  @author     Ash Vardanian
 *  @date       September 18, 2023
 *
 *  @copyright  Copyright (c) 2023
 *  @see        NodeJS docs: https://nodejs.org/api/n-api.html
 */
#include <stdio.h>  // `printf` for debug builds
#include <stdlib.h> // `malloc` to export strings into UTF-8

#include <node_api.h> // `napi_*` functions

#include <stringzilla/stringzilla.h> // `sz_*` functions

static void external_buffer_cleanup(napi_env env, void *data, void *hint) { free(data); }

static napi_value makeFindResultObject(napi_env env, int64_t index, uint64_t length) {
    napi_value js_obj;
    napi_create_object(env, &js_obj);

    napi_value js_index;
    if (index < 0) napi_create_bigint_int64(env, -1, &js_index);
    else
        napi_create_bigint_uint64(env, (uint64_t)index, &js_index);

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
        sz_cptr_t result = sz_find((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle_data, needle_length);
        if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        else { napi_create_bigint_uint64(env, result - (sz_cptr_t)haystack_data, &js_result); }
    }

    return js_result;
}

napi_value utf8CaseFoldAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 1) {
        napi_throw_error(env, NULL, "utf8CaseFold(buffer, validate?) expects at least 1 argument");
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
    if (validate && sz_utf8_valid((sz_cptr_t)source_data, source_length) == sz_false_k) {
        napi_throw_error(env, NULL, "Input is not valid UTF-8");
        return NULL;
    }

    // Worst-case expansion is 3x. See `sz_utf8_case_fold` docs.
    size_t capacity = source_length * 3;
    void *destination = capacity ? malloc(capacity) : NULL;
    if (capacity && !destination) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }

    sz_size_t out_length = 0;
    if (source_length) out_length = sz_utf8_case_fold((sz_cptr_t)source_data, source_length, (sz_ptr_t)destination);

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

napi_value utf8CaseInsensitiveFindAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    if (argc < 2) {
        napi_throw_error(env, NULL,
                         "utf8CaseInsensitiveFind(haystack, needle, validate?) expects at least 2 arguments");
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
    if (validate && (sz_utf8_valid((sz_cptr_t)haystack_data, haystack_length) == sz_false_k ||
                     sz_utf8_valid((sz_cptr_t)needle_data, needle_length) == sz_false_k)) {
        napi_throw_error(env, NULL, "Input is not valid UTF-8");
        return NULL;
    }

    sz_utf8_case_insensitive_needle_metadata_t metadata = {0};
    sz_size_t matched_length = 0;
    sz_cptr_t match = sz_utf8_case_insensitive_find((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle_data,
                                                    needle_length, &metadata, &matched_length);

    if (!match) return makeFindResultObject(env, -1, 0);
    return makeFindResultObject(env, (int64_t)(match - (sz_cptr_t)haystack_data), (uint64_t)matched_length);
}

typedef struct {
    sz_u8_t *needle_data;
    size_t needle_length;
    sz_utf8_case_insensitive_needle_metadata_t metadata;
} utf8_case_insensitive_needle_t;

static void utf8_case_insensitive_needle_cleanup(napi_env env, void *data, void *hint) {
    utf8_case_insensitive_needle_t *needle = (utf8_case_insensitive_needle_t *)data;
    if (!needle) return;
    if (needle->needle_data) free(needle->needle_data);
    free(needle);
}

napi_value utf8CaseInsensitiveNeedleConstructor(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    if (argc < 1) {
        napi_throw_error(env, NULL, "Utf8CaseInsensitiveNeedle(needle, validate?) expects at least 1 argument");
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
    if (validate && sz_utf8_valid((sz_cptr_t)needle_data, needle_length) == sz_false_k) {
        napi_throw_error(env, NULL, "Needle is not valid UTF-8");
        return NULL;
    }

    utf8_case_insensitive_needle_t *needle = (utf8_case_insensitive_needle_t *)malloc(sizeof(*needle));
    if (!needle) {
        napi_throw_error(env, NULL, "Memory allocation failed");
        return NULL;
    }
    needle->needle_length = needle_length;
    needle->metadata = (sz_utf8_case_insensitive_needle_metadata_t) {0};
    needle->needle_data = NULL;

    if (needle_length) {
        needle->needle_data = (sz_u8_t *)malloc(needle_length);
        if (!needle->needle_data) {
            free(needle);
            napi_throw_error(env, NULL, "Memory allocation failed");
            return NULL;
        }
        sz_copy((sz_ptr_t)needle->needle_data, (sz_cptr_t)needle_data, needle_length);
    }

    napi_wrap(env, js_this, needle, utf8_case_insensitive_needle_cleanup, NULL, NULL);
    return js_this;
}

napi_value utf8CaseInsensitiveNeedleFindIn(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    if (argc < 1) {
        napi_throw_error(env, NULL, "findIn(haystack, validate?) expects at least 1 argument");
        return NULL;
    }

    utf8_case_insensitive_needle_t *needle;
    napi_unwrap(env, js_this, (void **)&needle);
    if (!needle) {
        napi_throw_error(env, NULL, "Internal error: missing needle");
        return NULL;
    }

    void *haystack_data;
    size_t haystack_length;
    napi_status status = napi_get_buffer_info(env, args[0], &haystack_data, &haystack_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "First argument must be a Buffer");
        return NULL;
    }

    bool validate = false;
    if (argc > 1) { napi_get_value_bool(env, args[1], &validate); }
    if (validate && sz_utf8_valid((sz_cptr_t)haystack_data, haystack_length) == sz_false_k) {
        napi_throw_error(env, NULL, "Haystack is not valid UTF-8");
        return NULL;
    }

    sz_size_t matched_length = 0;
    sz_cptr_t match =
        sz_utf8_case_insensitive_find((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle->needle_data,
                                      needle->needle_length, &needle->metadata, &matched_length);
    if (!match) return makeFindResultObject(env, -1, 0);
    return makeFindResultObject(env, (int64_t)(match - (sz_cptr_t)haystack_data), (uint64_t)matched_length);
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
            sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? (sz_size_t)(ptr - haystack.start) : haystack.length;
            count += found;
            haystack.start += offset + found;
            haystack.length -= offset + found;
        }
    }
    else {
        while (haystack.length) {
            sz_cptr_t ptr = sz_find(haystack.start, haystack.length, needle.start, needle.length);
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
    if (argc > 1) {
        bool lossless;
        napi_get_value_bigint_uint64(env, args[1], &seed, &lossless);
        if (!lossless) {
            // Try regular number if BigInt fails
            double seed_double;
            if (napi_get_value_double(env, args[1], &seed_double) == napi_ok) { seed = (sz_u64_t)seed_double; }
        }
    }

    // Compute hash using StringZilla
    sz_u64_t hash_result = sz_hash((sz_cptr_t)buffer_data, buffer_length, seed);

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
    if (argc > 0) {
        bool lossless;
        napi_get_value_bigint_uint64(env, args[0], &seed, &lossless);
        if (!lossless) {
            double seed_double;
            if (napi_get_value_double(env, args[0], &seed_double) == napi_ok) { seed = (sz_u64_t)seed_double; }
        }
    }

    hasher_t *hasher = malloc(sizeof(hasher_t));
    hasher->seed = seed;
    sz_hash_state_init(&hasher->state, seed);
    napi_wrap(env, js_this, hasher, hasher_cleanup, NULL, NULL);

    return js_this;
}

napi_value hasherUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    sz_hash_state_update(&hasher->state, (sz_cptr_t)buffer_data, buffer_length);
    return js_this;
}

napi_value hasherDigest(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    sz_u64_t hash = sz_hash_state_digest(&hasher->state);
    napi_value js_result;
    napi_create_bigint_uint64(env, hash, &js_result);

    return js_result;
}

napi_value hasherReset(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    sz_hash_state_init(&hasher->state, hasher->seed);
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
    sz_sha256_state_init(&state);
    sz_sha256_state_update(&state, (sz_cptr_t)buffer_data, buffer_length);
    sz_sha256_state_digest(&state, digest);

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
    sz_sha256_state_init(&hasher->state);
    napi_wrap(env, js_this, hasher, sha256_hasher_cleanup, NULL, NULL);

    return js_this;
}

napi_value sha256HasherUpdate(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value js_this;
    napi_get_cb_info(env, info, &argc, args, &js_this, NULL);

    sha256_hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    void *buffer_data;
    size_t buffer_length;
    napi_status status = napi_get_buffer_info(env, args[0], &buffer_data, &buffer_length);
    if (status != napi_ok) {
        napi_throw_error(env, NULL, "Argument must be a Buffer");
        return NULL;
    }

    sz_sha256_state_update(&hasher->state, (sz_cptr_t)buffer_data, buffer_length);
    return js_this;
}

napi_value sha256HasherDigest(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    sha256_hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    sz_u8_t digest[32];
    sz_sha256_state_digest(&hasher->state, digest);

    // Convert result to JavaScript Buffer
    napi_value js_result;
    void *result_data;
    napi_create_buffer_copy(env, 32, digest, &result_data, &js_result);

    return js_result;
}

napi_value sha256HasherHexdigest(napi_env env, napi_callback_info info) {
    napi_value js_this;
    napi_get_cb_info(env, info, NULL, NULL, &js_this, NULL);

    sha256_hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    sz_u8_t digest[32];
    sz_sha256_state_digest(&hasher->state, digest);

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

    sha256_hasher_t *hasher;
    napi_unwrap(env, js_this, (void **)&hasher);

    sz_sha256_state_init(&hasher->state);
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
        sz_cptr_t result = sz_rfind((sz_cptr_t)haystack_data, haystack_length, (sz_cptr_t)needle_data, needle_length);

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

    // Find the byte using sz_find_byte (needs pointer to byte)
    char byte_char = (char)byte_value;
    sz_cptr_t result = sz_find_byte((sz_cptr_t)haystack_data, haystack_length, &byte_char);

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

    // Find the last byte using sz_rfind_byte (needs pointer to byte)
    char byte_char = (char)byte_value;
    sz_cptr_t result = sz_rfind_byte((sz_cptr_t)haystack_data, haystack_length, &byte_char);

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

    // Find first byte that is in the allowed set using sz_find_byteset
    sz_byteset_t byteset;
    sz_byteset_init(&byteset);
    for (size_t i = 0; i < allowed_length; i++) { sz_byteset_add_u8(&byteset, ((sz_u8_t *)allowed_data)[i]); }
    sz_cptr_t result = sz_find_byteset((sz_cptr_t)haystack_data, haystack_length, &byteset);

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

    // Find last byte that is in the allowed set using sz_rfind_byteset
    sz_byteset_t byteset;
    sz_byteset_init(&byteset);
    for (size_t i = 0; i < allowed_length; i++) { sz_byteset_add_u8(&byteset, ((sz_u8_t *)allowed_data)[i]); }
    sz_cptr_t result = sz_rfind_byteset((sz_cptr_t)haystack_data, haystack_length, &byteset);

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
    sz_bool_t equal = (first_length == second_length) &&
                      (first_length == 0 || sz_equal((sz_cptr_t)first_data, (sz_cptr_t)second_data, first_length));

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

    // Compare using sz_order
    int order = sz_order((sz_cptr_t)first_data, first_length, (sz_cptr_t)second_data, second_length);

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

    // Compute byte sum using sz_bytesum
    sz_u64_t sum = sz_bytesum((sz_cptr_t)buffer_data, buffer_length);

    // Convert to JavaScript BigInt and return
    napi_value js_result;
    napi_create_bigint_uint64(env, sum, &js_result);

    return js_result;
}

napi_value Init(napi_env env, napi_value exports) {

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

    // Create Utf8CaseInsensitiveNeedle class constructor
    napi_value utf8NeedleClass;
    napi_property_descriptor utf8NeedleProps[] = {
        {"findIn", 0, utf8CaseInsensitiveNeedleFindIn, 0, 0, 0, napi_default, 0},
    };
    napi_define_class(env, "Utf8CaseInsensitiveNeedle", NAPI_AUTO_LENGTH, utf8CaseInsensitiveNeedleConstructor, NULL,
                      sizeof(utf8NeedleProps) / sizeof(utf8NeedleProps[0]), utf8NeedleProps, &utf8NeedleClass);

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
    napi_property_descriptor utf8CaseFoldDesc = {"utf8CaseFold", 0, utf8CaseFoldAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor utf8CaseInsensitiveFindDesc = {
        "utf8CaseInsensitiveFind", 0, utf8CaseInsensitiveFindAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor hasherDesc = {"Hasher", 0, 0, 0, 0, hasherClass, napi_default, 0};
    napi_property_descriptor sha256HasherDesc = {"Sha256", 0, 0, 0, 0, sha256HasherClass, napi_default, 0};
    napi_property_descriptor utf8NeedleDesc = {
        "Utf8CaseInsensitiveNeedle", 0, 0, 0, 0, utf8NeedleClass, napi_default, 0};

    // Export the `capabilities` string for debugging
    napi_value caps_str_value;
    const char *caps_cstr = (const char *)sz_capabilities_to_string(sz_capabilities());
    napi_create_string_utf8(env, caps_cstr, NAPI_AUTO_LENGTH, &caps_str_value);
    napi_property_descriptor capabilitiesDesc = {"capabilities", 0, 0, 0, 0, caps_str_value, napi_default, 0};

    napi_property_descriptor properties[] = {
        findDesc,         findLastDesc,
        findByteDesc,     findLastByteDesc,
        findByteFromDesc, findLastByteFromDesc,
        countDesc,        hashDesc,
        sha256Desc,       equalDesc,
        compareDesc,      byteSumDesc,
        utf8CaseFoldDesc, utf8CaseInsensitiveFindDesc,
        hasherDesc,       sha256HasherDesc,
        utf8NeedleDesc,   capabilitiesDesc,
    };

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
