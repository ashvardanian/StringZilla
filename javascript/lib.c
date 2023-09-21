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

    // Convert result to JavaScript BigInt and return
    napi_value js_result;
    napi_create_bigint_uint64(env, result, &js_result);

    return js_result;
}

napi_value CountSubstrAPI(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value args[3];
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

    bool overlap = false;
    napi_get_value_bool(env, args[2], &overlap);

    size_t haystack_l = strlen(haystack);
    size_t needle_l = strlen(needle);

    size_t result = 0;

    if (haystack_l == 1)
        result = count_char(haystack, *needle);
    else if (haystack_l < needle_l)
        result = 0;
    else if (overlap) {
        while (strlen(haystack)) {
            #if defined(__AVX2__)
                size_t offset = strzl_avx2_find_substr(strzl_haystack, strzl_needle);
            #elif defined(__ARM_NEON)
                size_t offset = strzl_neon_find_substr(strzl_haystack, strzl_needle);
            #else
                size_t offset = strzl_naive_find_substr(strzl_haystack, strzl_needle);
            #endif


            bool found = offset != haystack_l;
            result += found;
            haystack += offset + found;
            haystack_l -= offset + found;
        }
    }

    else {
        while (haystack_l) {
            #if defined(__AVX2__)
                size_t offset = strzl_avx2_find_substr(strzl_haystack, strzl_needle);
            #elif defined(__ARM_NEON)
                size_t offset = strzl_neon_find_substr(strzl_haystack, strzl_needle);
            #else
                size_t offset = strzl_naive_find_substr(strzl_haystack, strzl_needle);
            #endif

            bool found = offset != haystack_l;
            result += found;
            haystack += offset + needle_l;
            haystack_l -= offset + needle_l * found;
        }
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
    // Define the "find" property
    napi_property_descriptor findDesc = {"find", 0, FindAPI, 0, 0, 0, napi_default, 0};
    
    // Define the "countSubstr" property
    napi_property_descriptor countSubstrDesc = {"countSubstr", 0, CountSubstrAPI, 0, 0, 0, napi_default, 0};
    
    // Define an array of property descriptors
    napi_property_descriptor properties[] = {findDesc, countSubstrDesc};
    
    // Define the number of properties in the array
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);

    // Define the properties on the exports object
    napi_define_properties(env, exports, propertyCount, properties);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)

