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
    napi_property_descriptor equalDesc = {"equal", 0, equalAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor compareDesc = {"compare", 0, compareAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor byteSumDesc = {"byteSum", 0, byteSumAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor hasherDesc = {"Hasher", 0, 0, 0, 0, hasherClass, napi_default, 0};
    napi_property_descriptor properties[] = {
        findDesc,  findLastDesc, findByteDesc, findLastByteDesc, findByteFromDesc, findLastByteFromDesc,
        countDesc, hashDesc,     equalDesc,    compareDesc,      byteSumDesc,      hasherDesc,
    };

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
