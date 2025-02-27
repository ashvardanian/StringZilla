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
#include <string.h> // strncmp

#include <node_api.h> // `napi_*` functions

#include <stringzilla/stringzilla.h> // `sz_*` functions

typedef struct {
    napi_env env_;
    napi_ref wrapper_;
    sz_cptr_t start;
    sz_size_t length;
} sz_wrapper_t;

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
napi_value str_startswith(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value jsthis;
    sz_wrapper_t *this;
    sz_wrapper_t *arg;

    napi_get_cb_info(env, info, &argc, args, &jsthis, NULL);
    napi_unwrap(env, jsthis, (void **)&this);
    napi_unwrap(env, args[0], (void **)&arg);

    napi_value ret;
    if (this->length < arg->length) { napi_get_boolean(env, false, &ret); }
    else if (strncmp(this->start, arg->start, arg->length) == 0) { napi_get_boolean(env, true, &ret); }
    else { napi_get_boolean(env, false, &ret); }
    return ret;
}
napi_value str_endswith(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value jsthis;
    sz_wrapper_t *this;
    sz_wrapper_t *arg;

    napi_get_cb_info(env, info, &argc, args, &jsthis, NULL);
    napi_unwrap(env, jsthis, (void **)&this);
    napi_unwrap(env, args[0], (void **)&arg);

    napi_value ret;
    if (this->length < arg->length) { napi_get_boolean(env, false, &ret); }
    else if (strncmp(this->start + (this->length - arg->length), arg->start, arg->length) == 0) { 
        napi_get_boolean(env, true, &ret); 
    }
    else { napi_get_boolean(env, false, &ret); }
    return ret;
}


napi_value str_count(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value jsthis;
    sz_wrapper_t *this;
    sz_wrapper_t *arg;

    napi_get_cb_info(env, info, &argc, args, &jsthis, NULL);
    napi_unwrap(env, jsthis, (void **)&this);
    napi_unwrap(env, args[0], (void **)&arg);

    sz_string_view_t haystack = {this->start, this->length};
    sz_string_view_t needle = {arg->start, arg->length};

    bool overlap = false;
    if (argc > 1) { napi_get_value_bool(env, args[1], &overlap); }

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
napi_value _str_find(sz_find_t finder, napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value args[2];
    napi_value jsthis;
    sz_wrapper_t *this;
    sz_wrapper_t *needle;

    napi_get_cb_info(env, info, &argc, args, &jsthis, NULL);
    napi_unwrap(env, jsthis, (void **)&this);
    napi_unwrap(env, args[0], (void **)&needle);


    napi_value js_result;
    if (needle->length == 0) { napi_create_bigint_int64(env, 0, &js_result); }
    else {
        sz_cptr_t result = finder(this->start, this->length, needle->start, needle->length);

        // In JavaScript, if `indexOf` is unable to indexOf the specified value, then it should return -1
        if (result == NULL) { napi_create_bigint_int64(env, -1, &js_result); }
        else { napi_create_bigint_uint64(env, result - this->start, &js_result); }
    }

    return js_result;
}

napi_value str_find(napi_env env, napi_callback_info info) { 
    return _str_find(sz_find, env, info); 
}
napi_value str_rfind(napi_env env, napi_callback_info info) { 
    return _str_find(sz_rfind, env, info); 
}

static void destroy(napi_env _unused_env, void *obj, void *_unused_hint) {
    sz_wrapper_t *sz = (sz_wrapper_t *)obj;
    napi_delete_reference(sz->env_, sz->wrapper_);
    free((void *)sz->start);
    free(sz);
}

static napi_value create_instance(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value args[1];
    napi_value jsthis;
    sz_wrapper_t *obj = malloc(sizeof(sz_wrapper_t));
    napi_get_cb_info(env, info, &argc, args, &jsthis, NULL);
    // if (type_check_data_buffer(env, args, argc) != ADDON_OK) {
    // return NULL;
    //}
    // napi_get_buffer_info(env, args[0], &data, &buf_len);
    // TODO type check
    napi_get_value_string_utf8(env, args[0], NULL, 0, (size_t *)&obj->length);
    obj->start = malloc(obj->length + 1);
    napi_get_value_string_utf8(env, args[0], (char *)obj->start, obj->length + 1, (size_t *)&obj->length);

    obj->env_ = env;
    napi_wrap(env, jsthis, (void *)obj, destroy, NULL, &obj->wrapper_);
    return jsthis;
}

napi_value Init(napi_env env, napi_value exports) {

    // Define an array of property descriptors
    napi_property_descriptor properties[] = {
        {"indexOf", NULL, indexOfAPI, NULL, NULL, NULL, napi_default, NULL},
        {"find", NULL, indexOfAPI, NULL, NULL, NULL, napi_default, NULL},
        {"count", NULL, countAPI, NULL, NULL, NULL, napi_default, NULL}
    };

    // Define the properties on the `exports` object
    size_t propertyCount = sizeof(properties) / sizeof(properties[0]);
    napi_define_properties(env, exports, propertyCount, properties);

    napi_property_descriptor obj_properties[] = {
        {"indexOf", NULL, str_find, NULL, NULL, NULL, napi_default, NULL},
        {"find", NULL, str_find, NULL, NULL, NULL, napi_default, NULL},
        {"rfind", NULL, str_rfind, NULL, NULL, NULL, napi_default, NULL},
        {"startswith", NULL, str_startswith, NULL, NULL, NULL, napi_default, NULL},
        {"endswith", NULL, str_endswith, NULL, NULL, NULL, napi_default, NULL},
        {"count", NULL, str_count, NULL, NULL, NULL, napi_default, NULL}
    };
    napi_value cons;

    napi_define_class(env, "Str", NAPI_AUTO_LENGTH, create_instance, NULL,
                      sizeof(obj_properties) / sizeof(*obj_properties), obj_properties, &cons);
    napi_set_named_property(env, exports, "Str", cons);

    return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
