/**
 *  @file lib.c
 *  @author Ash Vardanian
 *  @brief JavaScript bindings for StringZilla.
 *  @date 2023-09-18
 *
 *  @copyright Copyright (c) 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <node_api.h>
#include <stringzilla.h>

napi_value FindAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    struct strzl_haystack_t strzl_haystack = {NULL, 0};
    struct strzl_needle_t strzl_needle = {NULL, 0, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, &strzl_haystack.len);
    char *haystack = malloc(strzl_haystack.len);
    napi_get_value_string_utf8(env, args[0], haystack, strzl_haystack.len, &strzl_haystack.len);
    strzl_haystack.ptr = haystack;

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, &strzl_needle.len);
    char *needle = malloc(strzl_needle.len);
    napi_get_value_string_utf8(env, args[1], needle, strzl_needle.len, &strzl_needle.len);
    strzl_needle.ptr = needle;

// Perform the find operation
#if defined(__AVX2__)
    uint64_t result = strzl_avx2_find_substr(strzl_haystack, strzl_needle);
#elif defined(__ARM_NEON)
    uint64_t result = strzl_neon_find_substr(strzl_haystack, strzl_needle);
#else
    uint64_t result = strzl_naive_find_substr(strzl_haystack, strzl_needle);
#endif

    // Cleanup
    free(haystack);
    free(needle);

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;

    // In JavaScript, if `find` is unable to find the specified value, then it should return -1
    if (result == strzl_haystack.len)
        napi_create_bigint_int64(env, -1, &js_result);
    else
        napi_create_bigint_uint64(env, result, &js_result);

    return js_result;
}

size_t count_char(strzl_haystack_t strzl_haystack, char needle) {
    size_t result = strzl_naive_count_char(strzl_haystack, needle);

    return result;
}

napi_value CountAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    struct strzl_haystack_t strzl_haystack = {NULL, 0};
    struct strzl_needle_t strzl_needle = {NULL, 0, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, &strzl_haystack.len);
    char *haystack = malloc(strzl_haystack.len);
    napi_get_value_string_utf8(env, args[0], haystack, strzl_haystack.len, &strzl_haystack.len);
    strzl_haystack.ptr = haystack;

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, &strzl_needle.len);
    char *needle = malloc(strzl_needle.len);
    napi_get_value_string_utf8(env, args[1], needle, strzl_needle.len, &strzl_needle.len);
    strzl_needle.ptr = needle;

    bool overlap = false;
    if (argc > 2) {
        napi_get_value_bool(env, args[2], &overlap);
    }

    size_t result;

    if (strzl_needle.len == 0 || strzl_haystack.len == 0 || strzl_haystack.len < strzl_needle.len)
        result = 0;
    else if (strzl_needle.len == 1)
        result = count_char(strzl_haystack, strzl_needle.ptr[0]);
    else if (overlap) {
        while (strzl_haystack.len) {
#if defined(__AVX2__)
            size_t offset = strzl_avx2_find_substr(strzl_haystack, strzl_needle);
#elif defined(__ARM_NEON)
            size_t offset = strzl_neon_find_substr(strzl_haystack, strzl_needle);
#else
            size_t offset = strzl_naive_find_substr(strzl_haystack, strzl_needle);
#endif

            bool found = offset != strzl_haystack.len;
            result += found;
            strzl_haystack.ptr += offset + found;
            strzl_haystack.len -= offset + found;
        }
    }

    else {
        while (strzl_haystack.len) {
#if defined(__AVX2__)
            size_t offset = strzl_avx2_find_substr(strzl_haystack, strzl_needle);
#elif defined(__ARM_NEON)
            size_t offset = strzl_neon_find_substr(strzl_haystack, strzl_needle);
#else
            size_t offset = strzl_naive_find_substr(strzl_haystack, strzl_needle);
#endif

            bool found = offset != strzl_haystack.len;
            result += found;
            strzl_haystack.ptr += offset + strzl_needle.len;
            strzl_haystack.len -= offset + strzl_needle.len * found;
        }
    }

    // Cleanup
    free(haystack);
    free(needle);

    // Convert the result to JavaScript `BigInt` and return
    napi_value js_result;
    napi_create_bigint_uint64(env, result, &js_result);

    return js_result;
}

napi_value Init(napi_env env, napi_value exports) {
    // Define the "find" property
    napi_property_descriptor findDesc = {"find", 0, FindAPI, 0, 0, 0, napi_default, 0};

    // Define the "count" property
    napi_property_descriptor countDesc = {"count", 0, CountAPI, 0, 0, 0, napi_default, 0};

    // Define an array of property descriptors
    napi_property_descriptor properties[] = {findDesc, countDesc};

    // Define the number of properties in the array
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);

    // Define the properties on the `exports` object
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
