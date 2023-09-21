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
    size_t haystack_l;
    size_t needle_l;

    // For haystack
    napi_get_value_string_utf8(env, args[0], NULL, 0, &haystack_l);
    char *haystack = malloc(haystack_l + 1);
    napi_get_value_string_utf8(env, args[0], haystack, haystack_l + 1, &needle_l);
    struct strzl_haystack_t strzl_haystack = {haystack, needle_l};

    // For needle
    napi_get_value_string_utf8(env, args[1], NULL, 0, &haystack_l);
    char *needle = malloc(haystack_l + 1);
    napi_get_value_string_utf8(env, args[1], needle, haystack_l + 1, &needle_l);
    struct strzl_needle_t strzl_needle = {needle, needle_l, 0};

    // Perform the find operation
    #if defined(__AVX2__)
        uint64_t result = strzl_avx2_find_substr(strzl_haystack, strzl_needle);
    #elif defined(__ARM_NEON)
        uint64_t result = strzl_neon_find_substr(strzl_haystack, strzl_needle);
    #else
        uint64_t result = strzl_naive_find_substr(strzl_haystack, strzl_needle);
    #endif

    // Restore length of haystack as it's lost
    haystack_l = strlen(haystack);

    // In JavaScript if find unable to find the specified value then it should return -1
    if (haystack_l == (size_t)result) {
        napi_value js_result;
        napi_create_int32(env, -1, &js_result);

        return js_result;
    }

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
