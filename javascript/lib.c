/**
 *  @file       lib.c
 *  @brief      JavaScript bindings for StringZilla.
 *  @author     Ash Vardanian
 *  @date       September 18, 2023
 *
 *  @copyright  Copyright (c) 2023
 *  @see        NodeJS docs: https://nodejs.org/api/n-api.html
 */

#include <node_api.h>    // `napi_*` functions
#include <stdlib.h>      // `malloc`
#include <stringzilla.h> // `sz_*` functions

napi_value indexOfAPI(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    sz_string_view_t haystack_sz = {NULL, 0};
    sz_string_view_t needle_sz = {NULL, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, (size_t *)&haystack_sz.length);
    haystack_sz.start = malloc(haystack_sz.length + 1);
    napi_get_value_string_utf8(env,
                               args[0],
                               (char *)haystack_sz.start,
                               haystack_sz.length + 1,
                               (size_t *)&haystack_sz.length);

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, (size_t *)&needle_sz.length);
    needle_sz.start = malloc(needle_sz.length + 1);
    napi_get_value_string_utf8(env,
                               args[1],
                               (char *)needle_sz.start,
                               needle_sz.length + 1,
                               (size_t *)&needle_sz.length);

    // Convert the result to JavaScript BigInt and return
    napi_value js_result;
    if (needle_sz.length == 0) { napi_create_bigint_int64(env, 0, &js_result); }
    else {
        sz_string_ptr_t result =
            sz_find_substr(haystack_sz.start, haystack_sz.length, needle_sz.start, needle_sz.length);

        // In JavaScript, if `indexOf` is unable to indexOf the specified value, then it should return -1
        if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        else { napi_create_bigint_uint64(env, result - haystack_sz.start, &js_result); }
    }

    // Cleanup
    free((void *)haystack_sz.start);
    free((void *)needle_sz.start);
    return js_result;
}

napi_value countAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
    napi_get_cb_info(env, info, &argc, args, NULL, NULL);

    // Extract the C string from the JavaScript string for haystack and needle
    sz_string_view_t haystack_sz = {NULL, 0};
    sz_string_view_t needle_sz = {NULL, 0};

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, (size_t *)&haystack_sz.length);
    haystack_sz.start = malloc(haystack_sz.length + 1);
    napi_get_value_string_utf8(env,
                               args[0],
                               (char *)haystack_sz.start,
                               haystack_sz.length + 1,
                               (size_t *)&haystack_sz.length);

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, (size_t *)&needle_sz.length);
    needle_sz.start = malloc(needle_sz.length + 1);
    napi_get_value_string_utf8(env,
                               args[1],
                               (char *)needle_sz.start,
                               needle_sz.length + 1,
                               (size_t *)&needle_sz.length);

    bool overlap = false;
    if (argc > 2) { napi_get_value_bool(env, args[2], &overlap); }

    void const *haystack_start = haystack_sz.start, *needle_start = needle_sz.start;

    size_t count = 0;
    if (needle_sz.length == 0 || haystack_sz.length == 0 || haystack_sz.length < needle_sz.length) { count = 0; }
    else if (needle_sz.length == 1) { count = sz_count_char(haystack_sz.start, haystack_sz.length, needle_sz.start); }
    else if (overlap) {
        while (haystack_sz.length) {
            sz_string_ptr_t ptr =
                sz_find_substr(haystack_sz.start, haystack_sz.length, needle_sz.start, needle_sz.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? ptr - haystack_sz.start : haystack_sz.length;
            count += found;
            haystack_sz.start += offset + found;
            haystack_sz.length -= offset + found;
        }
    }
    else {
        while (haystack_sz.length) {
            sz_string_ptr_t ptr =
                sz_find_substr(haystack_sz.start, haystack_sz.length, needle_sz.start, needle_sz.length);
            sz_bool_t found = ptr != NULL;
            sz_size_t offset = found ? ptr - haystack_sz.start : haystack_sz.length;
            count += found;
            haystack_sz.start += offset + needle_sz.length;
            haystack_sz.length -= offset + needle_sz.length * found;
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
