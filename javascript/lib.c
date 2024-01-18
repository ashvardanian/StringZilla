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

    // Extract the C string from the JavaScript string for haystack and needle
    sz_string_view_t haystack = {NULL, 0};
    sz_string_view_t needle = {NULL, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, (size_t *)&haystack.length);
    haystack.start = malloc(haystack.length + 1);
    napi_get_value_string_utf8(env, args[0], (char *)haystack.start, haystack.length + 1, (size_t *)&haystack.length);

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, (size_t *)&needle.length);
    needle.start = malloc(needle.length + 1);
    napi_get_value_string_utf8(env, args[1], (char *)needle.start, needle.length + 1, (size_t *)&needle.length);

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (needle.length == 0) { napi_create_bigint_int64(env, 0, &js_result); }
    else {
        sz_cptr_t result = sz_find(haystack.start, haystack.length, needle.start, needle.length);

        // In JavaScript, if `indexOf` is unable to indexOf the specified value, then it should return -1
        if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        else { napi_create_bigint_uint64(env, result - haystack.start, &js_result); }
    }

    // Cleanup
    free((void *)haystack.start);
    free((void *)needle.start);
    return js_result;
}

napi_value countAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    sz_string_view_t haystack = {NULL, 0};
    sz_string_view_t needle = {NULL, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, (size_t *)&haystack.length);
    haystack.start = malloc(haystack.length + 1);
    napi_get_value_string_utf8(env, args[0], (char *)haystack.start, haystack.length + 1, (size_t *)&haystack.length);

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, (size_t *)&needle.length);
    needle.start = malloc(needle.length + 1);
    napi_get_value_string_utf8(env, args[1], (char *)needle.start, needle.length + 1, (size_t *)&needle.length);

    bool overlap = false;
    if (argc > 2) { napi_get_value_bool(env, args[2], &overlap); }

    void const *haystack_start = haystack.start, *needle_start = needle.start;

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

    // Cleanup
    free((void *)haystack_start);
    free((void *)needle_start);

    // Convert the `count` to JavaScript `BigInt` and return
    napi_value js_count;
    napi_create_bigint_uint64(env, count, &js_count);

    return js_count;
}

napi_value Init(napi_env env, napi_value exports) {

    // Define an array of property descriptors
    napi_property_descriptor findDesc = {"indexOf", 0, indexOfAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor countDesc = {"count", 0, countAPI, 0, 0, 0, napi_default, 0};
    napi_property_descriptor properties[] = {findDesc, countDesc};

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
