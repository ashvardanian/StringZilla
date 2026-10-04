/**
 *  @file python/stringzilla/stringzilla.c
 *  @author Ash Vardanian
 *  @date September 2, 2023
 *  @brief The module-init glue: @c PyModuleDef, type registration, and @c PyInit_stringzilla.
 *
 *  - Doesn't use PyBind11, NanoBind, Boost.Python, or any other high-level libs, only CPython API.
 *  - To minimize latency this implementation avoids @c PyArg_ParseTupleAndKeywords calls.
 *  - Reimplements all of the @c str functionality in C as a @c Str type.
 *  - Provides a highly generic @c Strs class for collections of strings, Arrow-style or not.
 *
 *  Pandas doesn't provide a C API, and even in 2.0 the Apache Arrow representation is opt-in, not
 *  the default. The PyCapsule protocol, together with the @b __arrow_c_array__ dunder methods, can
 *  be used to extract strings.
 *
 *  This module exports C functions via a @c PyCapsule of @c PyAPI, for another extension to import:
 *  - @c sz_py_export_string_like.
 *  - @c sz_py_export_strings_as_sequence.
 *  - @c sz_py_export_strings_as_u32tape.
 *  - @c sz_py_export_strings_as_u64tape.
 *  - @c sz_py_replace_strings_allocator.
 *
 *  Function Naming Convention:
 *  - `Str_like_*`: Functions callable both as module-level functions and as member methods.
 *  - `Str_*`: Functions that are member-only methods or have simpler calling conventions.
 *
 *  This translation unit owns the module-init glue - @c PyModuleDef, the type-registration table,
 *  and @c PyInit_stringzilla - plus the argument plumbing the three engine files share. The
 *  @c File, @c Str, and @c Strs struct layouts and the @c PyTypeObject forward declarations every
 *  domain file needs live in `stringzilla.h`; the domains themselves are split across `shared.c`,
 *  `file.c`, `str.c`, `strs.c`, `memory.c`, `hash.c`, `cipher.c`, `find.c`, `compare.c`, `sort.c`,
 *  `intersect.c`, `levenshtein.c`, `overlap.c`, `substrings.c`, and the `utf8_*.c` files.
 *
 *  @see PyArrow arrays: https://arrow.apache.org/docs/python/generated/pyarrow.array.html
 */
#include "stringzilla.h"

/**
 *  @brief The function table an importing extension reads out of the @c _sz_py_api capsule.
 *
 *  An importer carries its own copy of this layout and casts the capsule pointer to it, so the two
 *  must stay identical: a field added, removed, or reordered on one side alone makes the other
 *  misread memory with no diagnostic, since the capsule carries no version tag.
 */
typedef struct PyAPI {
    sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
    sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
    sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_allocator_t *);
} PyAPI;

#pragma region Capabilities

/** The @c stringzilla.Capability flags class every capability query speaks, built at import. */
static PyObject *capability_type = NULL;

int sz_py_export_capabilities(PyObject *capabilities_object, sz_capability_t *capabilities) {
    if (!capabilities_object || capabilities_object == Py_None) {
        *capabilities = sz_cap_cpus_k;
        return 0;
    }
    unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities_object);
    if (bits == (unsigned long long)-1 && PyErr_Occurred()) return -1;
    // The lookup drops what this CPU lacks, and a GPU capability would get host pointers
    *capabilities = (sz_capability_t)bits & sz_cap_cpus_k;
    return 0;
}

static PyObject *capability_from_mask_(sz_capability_t capabilities) {
    PyObject *bits = PyLong_FromUnsignedLongLong((unsigned long long)capabilities);
    if (!bits) return NULL;
    PyObject *capability = PyObject_CallOneArg(capability_type, bits);
    Py_DECREF(bits);
    return capability;
}

int sz_py_export_engine_placement(PyObject *capabilities_object, PyObject *stream_object, sz_capability_t *capabilities,
                                  sz_stream_t *stream) {
    *capabilities = sz_cap_cpus_k;
    if (capabilities_object && capabilities_object != Py_None) {
        unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities_object);
        if (bits == (unsigned long long)-1 && PyErr_Occurred()) return -1;
        *capabilities = (sz_capability_t)bits;
    }
    if (sz_py_export_stream(stream_object, stream) != 0) return -1;
    // Dispatch picks the group of the highest bit, so only a mask below every GPU's runs on the CPU
    if (*capabilities >= sz_cap_cuda_k) return 0;
    if (*stream) {
        PyErr_SetString(PyExc_ValueError, "a CPU engine takes no stream");
        return -1;
    }
    return 0;
}

/** Wraps what one capability query reported, raising the status of a query that failed. */
static PyObject *capabilities_reported_(sz_status_t status, sz_capability_t capabilities, char const *context) {
    if (status != sz_success_k) {
        sz_py_raise_status(status, context);
        return NULL;
    }
    return capability_from_mask_(capabilities);
}

/** One GPU producer's device count, raising the status of a vendor with none. */
static PyObject *devices_counted_(sz_status_t (*count_devices)(sz_size_t *), char const *context) {
    sz_size_t count = 0;
    sz_status_t const status = count_devices(&count);
    if (status == sz_success_k) return PyLong_FromSize_t(count);
    sz_py_raise_status(status, context);
    return NULL;
}

/** One GPU producer's capabilities of the device numbered @p ordinal_object. */
static PyObject *capabilities_of_ordinal_(sz_status_t (*query)(sz_size_t, sz_capability_t *), PyObject *ordinal_object,
                                          char const *context) {
    Py_ssize_t const ordinal = PyLong_AsSsize_t(ordinal_object);
    if (ordinal == -1 && PyErr_Occurred()) return NULL;
    if (ordinal < 0)
        return PyErr_Format(PyExc_ValueError, "%s: ordinal must be non-negative, got %zd", context, ordinal);
    sz_capability_t capabilities = 0;
    sz_status_t const status = query((sz_size_t)ordinal, &capabilities);
    return capabilities_reported_(status, capabilities, context);
}

/** One producer's capabilities that take no ordinal. */
static PyObject *capabilities_read_(sz_status_t (*query)(sz_capability_t *), char const *context) {
    sz_capability_t capabilities = 0;
    sz_status_t const status = query(&capabilities);
    return capabilities_reported_(status, capabilities, context);
}

/** One GPU producer's new stream on the device numbered @p ordinal_object, as an integer. */
static PyObject *stream_created_(sz_status_t (*init)(sz_size_t, sz_stream_t *), PyObject *ordinal_object,
                                 char const *context) {
    Py_ssize_t const ordinal = PyLong_AsSsize_t(ordinal_object);
    if (ordinal == -1 && PyErr_Occurred()) return NULL;
    if (ordinal < 0)
        return PyErr_Format(PyExc_ValueError, "%s: ordinal must be non-negative, got %zd", context, ordinal);
    sz_stream_t stream = NULL;
    sz_status_t const status = init((sz_size_t)ordinal, &stream);
    if (status == sz_success_k) return PyLong_FromVoidPtr(stream);
    sz_py_raise_status(status, context);
    return NULL;
}

/** Frees a stream one GPU producer made, carried as an integer. */
static PyObject *stream_freed_(sz_status_t (*release)(sz_stream_t), PyObject *stream_object, char const *context) {
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    sz_status_t const status = release(stream);
    if (status == sz_success_k) Py_RETURN_NONE;
    sz_py_raise_status(status, context);
    return NULL;
}

static char const doc_capabilities_detected_cpu[] =                                                  //
    "Get the capabilities this CPU can execute.\n\n"                                                 //
    "Detected from CPUID or HWCAP. Says nothing about whether the kernels were compiled in, which\n" //
    "`cpu_capabilities_compiled` reports.\n\n"                                                       //
    "Signature:\n"                                                                                   //
    "    >>> def cpu_capabilities_detected() -> sz.Capability: ...";

static PyObject *capabilities_detected_cpu(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return capabilities_read_(sz_capabilities_detected_cpu, "cpu_capabilities_detected()");
}

static char const doc_capabilities_compiled_cpu[] =                                                      //
    "Get the CPU capabilities whose kernels were compiled into this binary.\n\n"                         //
    "Decided at build time. Independent of the hardware: a binary built with a broken probe toolchain\n" //
    "still detects this machine's full set while containing no SIMD kernels at all, which is what\n"     //
    "makes a scalar build hard to spot.\n\n"                                                             //
    "Signature:\n"                                                                                       //
    "    >>> def cpu_capabilities_compiled() -> sz.Capability: ...";

static PyObject *capabilities_compiled_cpu(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return capabilities_read_(sz_capabilities_compiled_cpu, "cpu_capabilities_compiled()");
}

static char const doc_capabilities_enabled_cpu[] =                                                  //
    "Get the CPU capabilities kernels run with, unless a call passes `capabilities=`.\n\n"          //
    "It is `cpu_capabilities_detected() & cpu_capabilities_compiled()`, and always has SERIAL.\n\n" //
    "Signature:\n"                                                                                  //
    "    >>> def cpu_capabilities_enabled() -> sz.Capability: ...";

static PyObject *capabilities_enabled_cpu(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return capabilities_read_(sz_capabilities_enabled_cpu, "cpu_capabilities_enabled()");
}

static char const doc_thread_configure_cpu[] =                                     //
    "Prepare the calling thread for kernels of `capabilities`.\n\n"                //
    "Call it on every thread that runs kernels, before the first of them.\n\n"     //
    "Args:\n"                                                                      //
    "    capabilities (Capability): The mask kernels on this thread run with.\n\n" //
    "Signature:\n"                                                                 //
    "    >>> def cpu_configure_thread(capabilities, /) -> None: ...";

static PyObject *thread_configure_cpu(PyObject *unused_module, PyObject *capabilities) {
    sz_unused_(unused_module);
    unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities);
    if (bits == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    sz_status_t const status = sz_thread_configure_cpu((sz_capability_t)bits);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "cpu_configure_thread()");
        return NULL;
    }
    Py_RETURN_NONE;
}

static char const doc_device_count_gpu[] =                                                           //
    "Count the devices of one GPU vendor this process sees: `cuda_count_devices`, and the `rocm_`\n" //
    "and `metal_` ones alike.\n\n"                                                                   //
    "Raises:\n"                                                                                      //
    "    RuntimeError: No device of that vendor answers, or this build lacks its kernels.\n\n"       //
    "Signature:\n"                                                                                   //
    "    >>> def cuda_count_devices() -> int: ...";

static char const doc_capabilities_detected_gpu[] =                                                  //
    "Get the capabilities a GPU can execute, by its runtime's ordinal, whether or not the kernels\n" //
    "were compiled in: `cuda_capabilities_detected`, and the `rocm_` and `metal_` ones alike.\n\n"   //
    "Returns:\n"                                                                                     //
    "    Capability: GPU bits sit above every member, so they print as a number.\n\n"                //
    "Raises:\n"                                                                                      //
    "    RuntimeError: No device with that ordinal answers.\n\n"                                     //
    "Signature:\n"                                                                                   //
    "    >>> def cuda_capabilities_detected(ordinal, /) -> sz.Capability: ...";

static char const doc_capabilities_compiled_gpu[] =                                                      //
    "Get the capabilities of one GPU vendor whose kernels were compiled into this binary, whether or\n"  //
    "not a device runs them: `cuda_capabilities_compiled`, and the `rocm_` and `metal_` ones alike.\n\n" //
    "Signature:\n"                                                                                       //
    "    >>> def cuda_capabilities_compiled() -> sz.Capability: ...";

static char const doc_capabilities_enabled_gpu[] =                                                      //
    "Get the capabilities kernels run with on a GPU, by its runtime's ordinal, both detected and\n"     //
    "compiled: `cuda_capabilities_enabled`, and the `rocm_` and `metal_` ones alike. Engines,\n"        //
    "`Strs.copy` and `synchronize` take this mask, which names the vendor, and a stream, which names\n" //
    "the device.\n\n"                                                                                   //
    "Raises:\n"                                                                                         //
    "    RuntimeError: No device with that ordinal answers.\n\n"                                        //
    "Signature:\n"                                                                                      //
    "    >>> def cuda_capabilities_enabled(ordinal, /) -> sz.Capability: ...";

static PyObject *device_count_cuda(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return devices_counted_(sz_device_count_cuda, "cuda_count_devices()");
}

static PyObject *capabilities_detected_cuda(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return capabilities_of_ordinal_(sz_capabilities_detected_cuda, ordinal, "cuda_capabilities_detected()");
}

static PyObject *capabilities_compiled_cuda(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return capabilities_read_(sz_capabilities_compiled_cuda, "cuda_capabilities_compiled()");
}

static PyObject *capabilities_enabled_cuda(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return capabilities_of_ordinal_(sz_capabilities_enabled_cuda, ordinal, "cuda_capabilities_enabled()");
}

static PyObject *device_count_rocm(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return devices_counted_(sz_device_count_rocm, "rocm_count_devices()");
}

static PyObject *capabilities_detected_rocm(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return capabilities_of_ordinal_(sz_capabilities_detected_rocm, ordinal, "rocm_capabilities_detected()");
}

static PyObject *capabilities_compiled_rocm(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return capabilities_read_(sz_capabilities_compiled_rocm, "rocm_capabilities_compiled()");
}

static PyObject *capabilities_enabled_rocm(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return capabilities_of_ordinal_(sz_capabilities_enabled_rocm, ordinal, "rocm_capabilities_enabled()");
}

static PyObject *device_count_metal(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return devices_counted_(sz_device_count_metal, "metal_count_devices()");
}

static PyObject *capabilities_detected_metal(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return capabilities_of_ordinal_(sz_capabilities_detected_metal, ordinal, "metal_capabilities_detected()");
}

static PyObject *capabilities_compiled_metal(PyObject *unused_module, PyObject *unused_args) {
    sz_unused_(unused_module);
    sz_unused_(unused_args);
    return capabilities_read_(sz_capabilities_compiled_metal, "metal_capabilities_compiled()");
}

static PyObject *capabilities_enabled_metal(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return capabilities_of_ordinal_(sz_capabilities_enabled_metal, ordinal, "metal_capabilities_enabled()");
}

static char const doc_stream_init_gpu[] =                                                               //
    "Create a stream on a GPU, by its runtime's ordinal, which names that device to the engines,\n"     //
    "`Strs.copy` and `synchronize` it is passed to: `cuda_stream_init`, and the `rocm_` and `metal_`\n" //
    "ones alike. Free it with the matching `*_stream_free` once its work is synchronized.\n\n"          //
    "Returns:\n"                                                                                        //
    "    int: The `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>` handle.\n\n"                   //
    "Raises:\n"                                                                                         //
    "    RuntimeError: No device with that ordinal answers, or this build lacks its vendor.\n\n"        //
    "Signature:\n"                                                                                      //
    "    >>> def cuda_stream_init(ordinal, /) -> int: ...";

static char const doc_stream_free_gpu[] =                                                        //
    "Free a stream `cuda_stream_init` made, and the `rocm_` and `metal_` ones alike, once\n"     //
    "`synchronize` joined whatever was queued on it; nothing may use the stream afterwards.\n\n" //
    "Signature:\n"                                                                               //
    "    >>> def cuda_stream_free(stream, /) -> None: ...";

static PyObject *stream_init_cuda(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return stream_created_(sz_stream_init_cuda, ordinal, "cuda_stream_init()");
}

static PyObject *stream_free_cuda(PyObject *unused_module, PyObject *stream) {
    sz_unused_(unused_module);
    return stream_freed_(sz_stream_free_cuda, stream, "cuda_stream_free()");
}

static PyObject *stream_init_rocm(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return stream_created_(sz_stream_init_rocm, ordinal, "rocm_stream_init()");
}

static PyObject *stream_free_rocm(PyObject *unused_module, PyObject *stream) {
    sz_unused_(unused_module);
    return stream_freed_(sz_stream_free_rocm, stream, "rocm_stream_free()");
}

static PyObject *stream_init_metal(PyObject *unused_module, PyObject *ordinal) {
    sz_unused_(unused_module);
    return stream_created_(sz_stream_init_metal, ordinal, "metal_stream_init()");
}

static PyObject *stream_free_metal(PyObject *unused_module, PyObject *stream) {
    sz_unused_(unused_module);
    return stream_freed_(sz_stream_free_metal, stream, "metal_stream_free()");
}

static char const doc_synchronize[] =                                                                    //
    "Wait for everything enqueued on one stream, releasing the GIL meanwhile.\n\n"                       //
    "A GPU engine's verbs enqueue and return, so their outputs are read, and their inputs freed, only\n" //
    "after this returns. Returns at once for CPU capabilities.\n\n"                                      //
    "Args:\n"                                                                                            //
    "    capabilities (Capability): A producer's mask, which names the vendor.\n"                        //
    "    stream (int, optional): A stream of that vendor as an integer, or None for the default.\n\n"    //
    "Raises:\n"                                                                                          //
    "    BufferError: The stream belongs to another device.\n"                                           //
    "    RuntimeError: A launch on the stream failed, or the device does not answer.\n\n"                //
    "Signature:\n"                                                                                       //
    "    >>> def synchronize(capabilities, stream=None) -> None: ...";

static PyObject *module_synchronize(PyObject *unused_module, PyObject *const *args, Py_ssize_t positional_args_count,
                                    PyObject *args_names_tuple) {
    sz_unused_(unused_module);
    if (positional_args_count > 2) {
        PyErr_Format(PyExc_TypeError, "synchronize() takes at most 2 positional arguments, got %zd",
                     positional_args_count);
        return NULL;
    }
    PyObject *capabilities_object = positional_args_count > 0 ? args[0] : NULL;
    PyObject *stream_object = positional_args_count > 1 ? args[1] : NULL;
    Py_ssize_t const args_names_count = args_names_tuple ? PyTuple_GET_SIZE(args_names_tuple) : 0;
    for (Py_ssize_t keyword_index = 0; keyword_index < args_names_count; ++keyword_index) {
        PyObject *const key = PyTuple_GET_ITEM(args_names_tuple, keyword_index);
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "capabilities") == 0 ? &capabilities_object
                          : PyUnicode_CompareWithASCIIString(key, "stream") == 0     ? &stream_object
                                                                                     : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "synchronize() got an unexpected keyword argument '%U'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "synchronize() got multiple values for argument '%U'", key);
        *slot = args[positional_args_count + keyword_index];
    }
    if (!capabilities_object) {
        PyErr_SetString(PyExc_TypeError, "synchronize() missing required argument 'capabilities'");
        return NULL;
    }
    unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities_object);
    if (bits == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    sz_stream_t stream = NULL;
    if (sz_py_export_stream(stream_object, &stream) != 0) return NULL;
    sz_status_t status;
    Py_BEGIN_ALLOW_THREADS;
    status = sz_stream_synchronize_best((sz_capability_t)bits, stream);
    Py_END_ALLOW_THREADS;
    if (status != sz_success_k) {
        sz_py_raise_status(status, "synchronize()");
        return NULL;
    }
    Py_RETURN_NONE;
}

/** Builds the @c Capability IntFlag with one member per CPU capability, each spelled as the C
 *  library spells it, or returns @c NULL. */
static PyObject *capability_type_new_(void) {
    PyObject *members = PyDict_New();
    for (unsigned bit = 0; members && bit != 64; ++bit) {
        sz_capability_t const flag = (sz_capability_t)1 << bit;
        char name[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
        if ((flag & sz_cap_gpus_k) || !sz_capabilities_name(flag, name, sizeof(name))) continue;
        for (char *letter = name; *letter; ++letter) *letter = (char)Py_TOUPPER(*letter);
        PyObject *value = PyLong_FromUnsignedLongLong(flag);
        if (!value || PyDict_SetItemString(members, name, value) < 0) Py_CLEAR(members);
        Py_XDECREF(value);
    }
    PyObject *enum_module = members ? PyImport_ImportModule("enum") : NULL;
    PyObject *int_flag = enum_module ? PyObject_GetAttrString(enum_module, "IntFlag") : NULL;
    PyObject *enum_name = int_flag ? PyUnicode_FromString("Capability") : NULL;
    PyObject *int_flag_args = enum_name ? PyTuple_Pack(2, enum_name, members) : NULL;
    // Without `module`, the enum takes the import machinery's module name and cannot unpickle
    PyObject *module_name = int_flag_args ? PyUnicode_FromString("stringzilla") : NULL;
    PyObject *int_flag_kwargs = module_name ? PyDict_New() : NULL;
    if (int_flag_kwargs && PyDict_SetItemString(int_flag_kwargs, "module", module_name) < 0) Py_CLEAR(int_flag_kwargs);
    PyObject *type = int_flag_kwargs ? PyObject_Call(int_flag, int_flag_args, int_flag_kwargs) : NULL;
    Py_XDECREF(int_flag_kwargs);
    Py_XDECREF(module_name);
    Py_XDECREF(int_flag_args);
    Py_XDECREF(enum_name);
    Py_XDECREF(int_flag);
    Py_XDECREF(enum_module);
    Py_XDECREF(members);
    return type;
}

#pragma endregion Capabilities

static void stringzilla_cleanup(PyObject *m) {
    // Drain both free-lists, releasing the headers parked for reuse during the interpreter's lifetime.
    stringzilla_state_t *state = (stringzilla_state_t *)PyModule_GetState(m);
    if (!state) return;
    for (Str *node = state->str_freelist_head; node;) {
        Str *next = (Str *)node->parent;
        PyObject_Free(node);
        node = next;
    }
    for (Strs *node = state->strs_freelist_head; node;) {
        Strs *next = *Strs_freelist_next_(node);
        PyObject_Free(node);
        node = next;
    }
    state->str_freelist_head = NULL;
    state->str_freelist_count = 0;
    state->strs_freelist_head = NULL;
    state->strs_freelist_count = 0;
}

#pragma region Engine Arguments

void sz_py_raise_status(sz_status_t status, char const *context) {
    switch (status) {
    case sz_bad_alloc_k: PyErr_Format(PyExc_MemoryError, "%s: could not allocate", context); break;
    case sz_invalid_utf8_k: PyErr_Format(PyExc_ValueError, "%s: input is not well-formed UTF-8", context); break;
    case sz_contains_duplicates_k: PyErr_Format(PyExc_ValueError, "%s: input repeats an entry", context); break;
    case sz_overflow_risk_k: PyErr_Format(PyExc_OverflowError, "%s: input outgrows the index width", context); break;
    case sz_unexpected_dimensions_k: PyErr_Format(PyExc_ValueError, "%s: arguments disagree on shape", context); break;
    case sz_missing_gpu_k: PyErr_Format(PyExc_RuntimeError, "%s: no GPU device answers here", context); break;
    case sz_device_code_mismatch_k:
        PyErr_Format(PyExc_RuntimeError, "%s: the engine lives on another device, or the device refused the work",
                     context);
        break;
    case sz_device_memory_mismatch_k:
        PyErr_Format(PyExc_BufferError, "%s: memory the device cannot reach", context);
        break;
    case sz_missing_kernel_k: PyErr_Format(PyExc_LookupError, "%s: no kernel for these capabilities", context); break;
    case sz_missing_library_k: PyErr_Format(PyExc_RuntimeError, "%s: no StringZilla library is linked", context); break;
    default: PyErr_Format(PyExc_RuntimeError, "%s: failed with status %d", context, (int)status); break;
    }
}

int sz_py_export_strings(PyObject *object, char const *name, sz_sequence_t *sequence) {
    if (sz_py_export_strings_as_sequence(object, sequence)) return 0;
    PyErr_Format(PyExc_TypeError, "%s must be a stringzilla.Strs, got %s", name, Py_TYPE(object)->tp_name);
    return -1;
}

int sz_py_export_stream(PyObject *stream_object, sz_stream_t *stream) {
    if (!stream_object || stream_object == Py_None) {
        *stream = NULL;
        return 0;
    }
    if (!PyLong_Check(stream_object)) {
        PyErr_SetString(PyExc_TypeError, "stream must be an int holding a device stream handle");
        return -1;
    }
    *stream = PyLong_AsVoidPtr(stream_object);
    if (PyErr_Occurred()) return -1;
    return 0;
}

int sz_py_export_output_buffer(PyObject *object, char const *name, Py_ssize_t itemsize, int rank,
                               sz_size_t const *extents, Py_buffer *view, sz_size_t *strides) {
    if (PyObject_GetBuffer(object, view, PyBUF_STRIDES | PyBUF_WRITABLE) != 0) return -1;
    if (view->ndim != rank) {
        PyErr_Format(PyExc_ValueError, "%s must be a %d-dimensional buffer, got %d", name, rank, view->ndim);
        PyBuffer_Release(view);
        return -1;
    }
    if (view->itemsize != itemsize) {
        PyErr_Format(PyExc_TypeError, "%s must hold %zd-byte items, got %zd-byte ones", name, itemsize, view->itemsize);
        PyBuffer_Release(view);
        return -1;
    }
    for (int axis = 0; axis != rank; ++axis) {
        if ((sz_size_t)view->shape[axis] < extents[axis]) {
            PyErr_Format(PyExc_ValueError, "%s axis %d holds %zd entries, need %zu", name, axis, view->shape[axis],
                         extents[axis]);
            PyBuffer_Release(view);
            return -1;
        }
        // A stride the verb cannot express: it counts whole items forward, never bytes and never backwards.
        if (view->strides[axis] < itemsize || view->strides[axis] % itemsize != 0) {
            PyErr_Format(PyExc_ValueError, "%s axis %d must step whole items forward", name, axis);
            PyBuffer_Release(view);
            return -1;
        }
        strides[axis] = (sz_size_t)(view->strides[axis] / itemsize);
    }
    return 0;
}

int sz_py_export_input_buffer(PyObject *object, char const *name, Py_ssize_t itemsize, sz_size_t count,
                              Py_buffer *view) {
    if (PyObject_GetBuffer(object, view, PyBUF_CONTIG_RO) != 0) return -1;
    if (view->itemsize != itemsize) {
        PyErr_Format(PyExc_TypeError, "%s must hold %zd-byte items, got %zd-byte ones", name, itemsize, view->itemsize);
        PyBuffer_Release(view);
        return -1;
    }
    if ((sz_size_t)(view->len / view->itemsize) < count) {
        PyErr_Format(PyExc_ValueError, "%s holds %zd entries, need %zu", name, (Py_ssize_t)(view->len / itemsize),
                     count);
        PyBuffer_Release(view);
        return -1;
    }
    return 0;
}

#pragma endregion Engine Arguments

static PyMethodDef stringzilla_methods[] = {
    // Basic `str`, `bytes`, and `bytearray`-like functionality
    {"contains", (PyCFunction)Str_like_contains, STRINGZILLA_METHOD_FLAGS, doc_contains},
    {"count", (PyCFunction)Str_like_count, STRINGZILLA_METHOD_FLAGS, doc_count},
    {"splitlines", (PyCFunction)Str_like_splitlines, STRINGZILLA_METHOD_FLAGS, doc_splitlines},
    {"startswith", (PyCFunction)Str_like_startswith, STRINGZILLA_METHOD_FLAGS, doc_startswith},
    {"endswith", (PyCFunction)Str_like_endswith, STRINGZILLA_METHOD_FLAGS, doc_endswith},
    {"decode", (PyCFunction)Str_like_decode, STRINGZILLA_METHOD_FLAGS, doc_decode},
    {"equal", (PyCFunction)Str_like_equal, STRINGZILLA_METHOD_FLAGS, doc_like_equal},
    {"lstrip", (PyCFunction)Str_like_lstrip, STRINGZILLA_METHOD_FLAGS, doc_lstrip},
    {"rstrip", (PyCFunction)Str_like_rstrip, STRINGZILLA_METHOD_FLAGS, doc_rstrip},
    {"strip", (PyCFunction)Str_like_strip, STRINGZILLA_METHOD_FLAGS, doc_strip},

    // Bidirectional operations
    {"find", (PyCFunction)Str_like_find, STRINGZILLA_METHOD_FLAGS, doc_find},
    {"index", (PyCFunction)Str_like_index, STRINGZILLA_METHOD_FLAGS, doc_index},
    {"partition", (PyCFunction)Str_like_partition, STRINGZILLA_METHOD_FLAGS, doc_partition},
    {"split", (PyCFunction)Str_like_split, STRINGZILLA_METHOD_FLAGS, doc_split},
    {"rfind", (PyCFunction)Str_like_rfind, STRINGZILLA_METHOD_FLAGS, doc_rfind},
    {"rindex", (PyCFunction)Str_like_rindex, STRINGZILLA_METHOD_FLAGS, doc_rindex},
    {"rpartition", (PyCFunction)Str_like_rpartition, STRINGZILLA_METHOD_FLAGS, doc_rpartition},
    {"rsplit", (PyCFunction)Str_like_rsplit, STRINGZILLA_METHOD_FLAGS, doc_rsplit},

    // Character search extensions
    {"find_first_of", (PyCFunction)Str_like_find_first_of, STRINGZILLA_METHOD_FLAGS, doc_find_first_of},
    {"find_last_of", (PyCFunction)Str_like_find_last_of, STRINGZILLA_METHOD_FLAGS, doc_find_last_of},
    {"find_first_not_of", (PyCFunction)Str_like_find_first_not_of, STRINGZILLA_METHOD_FLAGS, doc_find_first_not_of},
    {"find_last_not_of", (PyCFunction)Str_like_find_last_not_of, STRINGZILLA_METHOD_FLAGS, doc_find_last_not_of},
    {"count_byteset", (PyCFunction)Str_like_count_byteset, STRINGZILLA_METHOD_FLAGS, doc_count_byteset},
    {"split_byteset", (PyCFunction)Str_like_split_byteset, STRINGZILLA_METHOD_FLAGS, doc_split_byteset},
    {"rsplit_byteset", (PyCFunction)Str_like_rsplit_byteset, STRINGZILLA_METHOD_FLAGS, doc_rsplit_byteset},

    // Lazily evaluated iterators
    {"split_iter", (PyCFunction)Str_like_split_iter, STRINGZILLA_METHOD_FLAGS, doc_split_iter},
    {"rsplit_iter", (PyCFunction)Str_like_rsplit_iter, STRINGZILLA_METHOD_FLAGS, doc_rsplit_iter},
    {"split_byteset_iter", (PyCFunction)Str_like_split_byteset_iter, STRINGZILLA_METHOD_FLAGS, doc_split_byteset_iter},
    {"rsplit_byteset_iter", (PyCFunction)Str_like_rsplit_byteset_iter, STRINGZILLA_METHOD_FLAGS,
     doc_rsplit_byteset_iter},

    // UTF-8 aware operations
    {"utf8_count", (PyCFunction)Str_like_utf8_count, STRINGZILLA_METHOD_FLAGS, doc_utf8_count},
    {"utf8_split_newlines", (PyCFunction)Str_like_utf8_split_newlines, STRINGZILLA_METHOD_FLAGS,
     doc_utf8_split_newlines},
    {"utf8_newlines", (PyCFunction)Str_like_utf8_newlines, STRINGZILLA_METHOD_FLAGS, doc_utf8_newlines},
    {"utf8_split_whitespaces", (PyCFunction)Str_like_utf8_split_whitespaces, STRINGZILLA_METHOD_FLAGS,
     doc_utf8_split_whitespaces},
    {"utf8_whitespaces", (PyCFunction)Str_like_utf8_whitespaces, STRINGZILLA_METHOD_FLAGS, doc_utf8_whitespaces},
    {"utf8_split_delimiters", (PyCFunction)Str_like_utf8_split_delimiters, STRINGZILLA_METHOD_FLAGS,
     doc_utf8_split_delimiters},
    {"utf8_delimiters", (PyCFunction)Str_like_utf8_delimiters, STRINGZILLA_METHOD_FLAGS, doc_utf8_delimiters},
    {"utf8_wordbreaks", (PyCFunction)Str_like_utf8_wordbreaks, STRINGZILLA_METHOD_FLAGS, doc_utf8_wordbreaks},
    {"utf8_codepoints", (PyCFunction)Str_like_utf8_codepoints, STRINGZILLA_METHOD_FLAGS, doc_utf8_codepoints},
    {"utf8_graphemes", (PyCFunction)Str_like_utf8_graphemes, STRINGZILLA_METHOD_FLAGS, doc_utf8_graphemes},
    {"utf8_sentences", (PyCFunction)Str_like_utf8_sentences, STRINGZILLA_METHOD_FLAGS, doc_utf8_sentences},
    {"utf8_linebreaks", (PyCFunction)Str_like_utf8_linebreaks, STRINGZILLA_METHOD_FLAGS, doc_utf8_linebreaks},
    {"utf8_uncased_fold", (PyCFunction)Str_like_utf8_uncased_fold, STRINGZILLA_METHOD_FLAGS, doc_utf8_uncased_fold},
    {"utf8_norm", (PyCFunction)Str_like_utf8_norm, STRINGZILLA_METHOD_FLAGS, doc_utf8_norm},
    {"utf8_find_denormalized", (PyCFunction)Str_like_utf8_find_denormalized, STRINGZILLA_METHOD_FLAGS,
     doc_utf8_find_denormalized},
    {"utf8_uncased_search", (PyCFunction)Str_like_utf8_uncased_search, STRINGZILLA_METHOD_FLAGS,
     doc_utf8_uncased_search},
    {"utf8_uncased_matches", (PyCFunction)Str_like_utf8_uncased_matches, STRINGZILLA_METHOD_FLAGS,
     doc_utf8_uncased_matches},
    {"utf8_uncased_order", (PyCFunction)Str_like_utf8_uncased_order, STRINGZILLA_METHOD_FLAGS, doc_utf8_uncased_order},

    // Dealing with larger-than-memory datasets
    {"offset_within", (PyCFunction)Str_offset_within, STRINGZILLA_METHOD_FLAGS, doc_offset_within},
    {"write_to", (PyCFunction)Str_write_to, STRINGZILLA_METHOD_FLAGS, doc_write_to},

    // In-place transforms
    {"translate", (PyCFunction)Str_like_translate, STRINGZILLA_METHOD_FLAGS, doc_translate},

    // Global unary extensions
    {"hash", (PyCFunction)Str_like_hash, STRINGZILLA_METHOD_FLAGS, doc_like_hash},
    {"hash_multiseed", (PyCFunction)Str_like_hash_multiseed, STRINGZILLA_METHOD_FLAGS, doc_hash_multiseed},
    {"bytesum", (PyCFunction)Str_like_bytesum, STRINGZILLA_METHOD_FLAGS, doc_like_bytesum},
    {"sha256", (PyCFunction)Str_like_sha256, STRINGZILLA_METHOD_FLAGS, doc_like_sha256},
    {"hmac_sha256", (PyCFunction)hmac_sha256, STRINGZILLA_METHOD_FLAGS, doc_hmac_sha256},
    {"fill_random", (PyCFunction)Str_like_fill_random, STRINGZILLA_METHOD_FLAGS, doc_fill_random},

    // Module-level functionality
    {"random", (PyCFunction)module_random, STRINGZILLA_METHOD_FLAGS, doc_random},

    // Capability and stream producers, the only calls taking an ordinal, and the stream join
    {"cpu_capabilities_detected", capabilities_detected_cpu, METH_NOARGS, doc_capabilities_detected_cpu},
    {"cpu_capabilities_compiled", capabilities_compiled_cpu, METH_NOARGS, doc_capabilities_compiled_cpu},
    {"cpu_capabilities_enabled", capabilities_enabled_cpu, METH_NOARGS, doc_capabilities_enabled_cpu},
    {"cpu_configure_thread", thread_configure_cpu, METH_O, doc_thread_configure_cpu},
    {"cuda_count_devices", device_count_cuda, METH_NOARGS, doc_device_count_gpu},
    {"cuda_capabilities_detected", capabilities_detected_cuda, METH_O, doc_capabilities_detected_gpu},
    {"cuda_capabilities_compiled", capabilities_compiled_cuda, METH_NOARGS, doc_capabilities_compiled_gpu},
    {"cuda_capabilities_enabled", capabilities_enabled_cuda, METH_O, doc_capabilities_enabled_gpu},
    {"rocm_count_devices", device_count_rocm, METH_NOARGS, doc_device_count_gpu},
    {"rocm_capabilities_detected", capabilities_detected_rocm, METH_O, doc_capabilities_detected_gpu},
    {"rocm_capabilities_compiled", capabilities_compiled_rocm, METH_NOARGS, doc_capabilities_compiled_gpu},
    {"rocm_capabilities_enabled", capabilities_enabled_rocm, METH_O, doc_capabilities_enabled_gpu},
    {"metal_count_devices", device_count_metal, METH_NOARGS, doc_device_count_gpu},
    {"metal_capabilities_detected", capabilities_detected_metal, METH_O, doc_capabilities_detected_gpu},
    {"metal_capabilities_compiled", capabilities_compiled_metal, METH_NOARGS, doc_capabilities_compiled_gpu},
    {"metal_capabilities_enabled", capabilities_enabled_metal, METH_O, doc_capabilities_enabled_gpu},
    {"cuda_stream_init", stream_init_cuda, METH_O, doc_stream_init_gpu},
    {"cuda_stream_free", stream_free_cuda, METH_O, doc_stream_free_gpu},
    {"rocm_stream_init", stream_init_rocm, METH_O, doc_stream_init_gpu},
    {"rocm_stream_free", stream_free_rocm, METH_O, doc_stream_free_gpu},
    {"metal_stream_init", stream_init_metal, METH_O, doc_stream_init_gpu},
    {"metal_stream_free", stream_free_metal, METH_O, doc_stream_free_gpu},
    {"synchronize", (PyCFunction)module_synchronize, STRINGZILLA_METHOD_FLAGS, doc_synchronize},

    {NULL, NULL, 0, NULL}};

PyModuleDef stringzilla_module = {
    PyModuleDef_HEAD_INIT,
    "stringzilla",
    "Search, hash, sort, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU",
    sizeof(stringzilla_state_t), // Per-interpreter free-list state; also enables `PyState_FindModule`.
    stringzilla_methods,
    NULL,
    NULL,
    NULL,
    stringzilla_cleanup,
};

/**
 *  @brief Every type the module readies, and the name it is exported under.
 *
 *  A @c NULL name means the type is readied but not exported: the UTF-8 iterators are returned by
 *  methods and never constructed by name, so they need @c tp_dict filled but no module attribute.
 */
static struct {
    char const *name;
    PyTypeObject *type;
} const stringzilla_types[] = {
    {"Str", &StrType},
    {"File", &FileType},
    {"Strs", &StrsType},
    {"FindSplits", &FindSplitsType},
    {"Utf8SplitNewlines", &Utf8SplitNewlinesType},
    {"Utf8Newlines", &Utf8NewlinesType},
    {"Utf8SplitWhitespaces", &Utf8SplitWhitespacesType},
    {"Utf8Whitespaces", &Utf8WhitespacesType},
    {"Utf8SplitDelimiters", &Utf8SplitDelimitersType},
    {"Utf8Delimiters", &Utf8DelimitersType},
    {"Utf8Wordbreaks", &Utf8WordbreaksType},
    {NULL, &Utf8CodepointsType},
    {NULL, &Utf8GraphemesType},
    {NULL, &Utf8SentencesType},
    {NULL, &Utf8LinebreaksType},
    {NULL, &Utf8UncasedMatchesType},
    {"Hasher", &HasherType},
    {"Sha256", &Sha256Type},
    {"Sha256s", &Sha256sType},
    {"Aes256CtrKey", &Aes256CtrKeyType},
    {"Aes256GcmKey", &Aes256GcmKeyType},
    {"Aes256GcmEncryptor", &Aes256GcmEncryptorType},
    {"Aes256GcmDecryptor", &Aes256GcmDecryptorType},
    {"LevenshteinEngine", &LevenshteinEngineType},
    {"OverlapEngine", &OverlapEngineType},
    {"SubstringsEngine", &SubstringsEngineType},
};

PyMODINIT_FUNC PyInit_stringzilla(void) {
    PyObject *m;
    sz_size_t const types_count = sizeof(stringzilla_types) / sizeof(stringzilla_types[0]);

    for (sz_size_t type_index = 0; type_index != types_count; ++type_index)
        if (PyType_Ready(stringzilla_types[type_index].type) < 0) return NULL;

    m = PyModule_Create(&stringzilla_module);
    if (m == NULL) return NULL;

#ifdef Py_GIL_DISABLED
    // Declare that this module is safe for free-threaded Python
    PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

    // Add version metadata
    {
        char version_str[50];
        sprintf(version_str, "%d.%d.%d", sz_version_major(), sz_version_minor(), sz_version_patch());
        PyModule_AddStringConstant(m, "__version__", version_str);
    }

    // Publish the digest width on both hasher types, so callers can size an output matrix without
    // hardcoding it. A class attribute rather than a property, as it describes the algorithm.
    {
        PyObject *digest_length = PyLong_FromSize_t(STRINGZILLA_SHA256_DIGEST_LENGTH);
        if (!digest_length) goto failed;
        int const published = PyDict_SetItemString(Sha256Type.tp_dict, "digest_length", digest_length) |
                              PyDict_SetItemString(Sha256sType.tp_dict, "digest_length", digest_length);
        Py_DECREF(digest_length);
        if (published < 0) goto failed;
    }

    capability_type = capability_type_new_();
    if (!capability_type) goto failed;
    Py_INCREF(capability_type);
    if (PyModule_AddObject(m, "Capability", capability_type) < 0) {
        Py_DECREF(capability_type);
        goto failed;
    }

    for (sz_size_t type_index = 0; type_index != types_count; ++type_index) {
        char const *const name = stringzilla_types[type_index].name;
        if (!name) continue;
        PyTypeObject *const type = stringzilla_types[type_index].type;
        Py_INCREF(type);
        // Only a successful `PyModule_AddObject` steals the reference, so undo just this one and let the
        // dying module release the types it already owns.
        if (PyModule_AddObject(m, name, (PyObject *)type) < 0) {
            Py_DECREF(type);
            goto failed;
        }
    }

    // A refused tag needs a class of its own, created here because a static exception object cannot
    // be built before the interpreter exists.
    AuthenticationErrorType = PyErr_NewExceptionWithDoc("stringzilla.AuthenticationError", doc_AuthenticationError,
                                                        PyExc_ValueError, NULL);
    if (!AuthenticationErrorType) goto failed;
    Py_INCREF(AuthenticationErrorType);
    if (PyModule_AddObject(m, "AuthenticationError", AuthenticationErrorType) < 0) {
        Py_DECREF(AuthenticationErrorType); // The reference the module refused to take
        Py_DECREF(AuthenticationErrorType); // The reference this translation unit holds
        AuthenticationErrorType = NULL;
        goto failed;
    }

    // Export C API functions as a single capsule structure for StringZillas
    static PyAPI sz_py_api = {
        .sz_py_export_string_like = sz_py_export_string_like,
        .sz_py_export_strings_as_sequence = sz_py_export_strings_as_sequence,
        .sz_py_export_strings_as_u32tape = sz_py_export_strings_as_u32tape,
        .sz_py_export_strings_as_u64tape = sz_py_export_strings_as_u64tape,
        .sz_py_replace_strings_allocator = sz_py_replace_strings_allocator,
    };
    if (PyModule_AddObject(m, "_sz_py_api", PyCapsule_New(&sz_py_api, "_sz_py_api", NULL)) < 0) goto failed;

    return m;

failed:
    Py_DECREF(m);
    return NULL;
}
