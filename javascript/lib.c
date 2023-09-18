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
    size_t str_size;
    size_t str_len;

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, &str_size);
    char *haystack = malloc(str_size + 1);
    napi_get_value_string_utf8(env, args[0], haystack, str_size + 1, &str_len);
    struct strzl_haystack_t strzl_haystack = {haystack, str_len};

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, &str_size);
    char *needle = malloc(str_size + 1);
    napi_get_value_string_utf8(env, args[1], needle, str_size + 1, &str_len);
    struct strzl_needle_t strzl_needle = {needle, str_len, 0};

    // Perform the find operation
    uint64_t result = stringzilla_find(&strzl_haystack, &strzl_needle);

    // Cleanup
    free(haystack);
    free(needle);

    // Convert result to JavaScript BigInt and return
    napi_value js_result;
    napi_create_bigint_uint64(env, result, &js_result);

    return js_result;
}

napi_value Init(napi_env env, napi_value exports) {
    napi_property_descriptor desc = {"find", 0, FindAPI, 0, 0, 0, napi_default, 0};
    napi_define_properties(env, exports, 1, &desc);
    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
