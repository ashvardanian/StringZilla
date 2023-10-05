/**
 *  @file       lib.c
 *  @brief      JavaScript bindings for StringZilla.
 *  @author     Ash Vardanian
 *  @date       September 18, 2023
 *
 *  @copyright  Copyright (c) 2023
 *  @see        NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <node_api.h>
#include <stringzilla.h>

napi_value FindAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    sz_haystack_t haystack_sz = {NULL, 0};
    sz_needle_t needle_sz = {NULL, 0, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, &haystack_sz.length);
    haystack_sz.start = malloc(haystack_sz.length + 1);
    napi_get_value_string_utf8(env, args[0], haystack_sz.start, haystack_sz.length + 1, &haystack_sz.length);

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, &needle_sz.length);
    needle_sz.start = malloc(needle_sz.length + 1);
    napi_get_value_string_utf8(env, args[1], needle_sz.start, needle_sz.length + 1, &needle_sz.length);

    // Perform the find operation
    sz_size_t result = sz_find_substr(haystack_sz, needle_sz);

    // Cleanup
    free(haystack_sz.start);
    free(needle_sz.start);

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;

    // In JavaScript, if `find` is unable to find the specified value, then it should return -1
    if (result == haystack_sz.length) napi_create_bigint_int64(env, -1, &js_result);
    else
        napi_create_bigint_uint64(env, result, &js_result);

    return js_result;
}

size_t count_char(sz_haystack_t haystack_sz, char needle) {
    size_t result = sz_count_char(haystack_sz, needle);
    return result;
}

napi_value CountAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    sz_haystack_t haystack_sz = {NULL, 0};
    sz_needle_t needle_sz = {NULL, 0, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, &haystack_sz.length);
    haystack_sz.start = malloc(haystack_sz.length + 1);
    napi_get_value_string_utf8(env, args[0], haystack_sz.start, haystack_sz.length + 1, &haystack_sz.length);

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, &needle_sz.length);
    needle_sz.start = malloc(needle_sz.length + 1);
    napi_get_value_string_utf8(env, args[1], needle_sz.start, needle_sz.length + 1, &needle_sz.length);

    bool overlap = false;
    if (argc > 2) { napi_get_value_bool(env, args[2], &overlap); }

    void const *haystack_start = haystack_sz.start, *needle_start = needle_sz.start;

    size_t count = 0;
    if (needle_sz.length == 0 || haystack_sz.length == 0 || haystack_sz.length < needle_sz.length) { count = 0; }
    else if (needle_sz.length == 1) { count = count_char(haystack_sz, needle_sz.start[0]); }
    else if (overlap) {
        while (haystack_sz.length) {
            sz_size_t offset = sz_find_substr(haystack_sz, needle_sz);
            int found = offset != haystack_sz.length;
            count += found;
            haystack_sz.start += offset + found;
            haystack_sz.length -= offset + found;
        }
    }
    else {
        while (haystack_sz.length) {
            sz_size_t offset = sz_find_substr(haystack_sz, needle_sz);
            int found = offset != haystack_sz.length;
            count += found;
            haystack_sz.start += offset + needle_sz.length;
            haystack_sz.length -= offset + needle_sz.length * found;
        }
    }

    // Cleanup
    free(haystack_start);
    free(needle_start);

    // Convert the `count` to JavaScript `BigInt` and return
    napi_value js_count;
    napi_create_bigint_uint64(env, count, &js_count);

    return js_count;
}

napi_value Init(napi_env env, napi_value exports) {

    // Define an array of property descriptors
    napi_property_descriptor findDesc = {"find", 0, FindAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor countDesc = {"count", 0, CountAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor properties[] = {findDesc, countDesc};

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
