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
    sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *);
} PyAPI;

#pragma region Devices

/** The @c stringzilla.Capability flags class every capability query speaks, built at import. */
static PyObject *capability_type = NULL;

sz_capability_t sz_py_enabled_capabilities = sz_cap_serial_k;

int sz_py_export_capabilities(PyObject *capabilities_object, sz_capability_t *capabilities) {
    if (!capabilities_object || capabilities_object == Py_None) {
        *capabilities = sz_py_enabled_capabilities;
        return 0;
    }
    unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities_object);
    if (bits == (unsigned long long)-1 && PyErr_Occurred()) return -1;
    sz_capability_t detected = sz_cap_serial_k;
    sz_cpu_capabilities_detected(&detected);
    // A capability this CPU lacks would fault, and a GPU one would get host pointers
    *capabilities = (sz_capability_t)bits & detected;
    return 0;
}

static PyObject *capability_from_mask_(sz_capability_t capabilities) {
    PyObject *bits = PyLong_FromUnsignedLongLong((unsigned long long)capabilities);
    if (!bits) return NULL;
    PyObject *capability = PyObject_CallOneArg(capability_type, bits);
    Py_DECREF(bits);
    return capability;
}

static sz_status_t cpu_count_devices_(sz_size_t *count) {
    *count = 1;
    return sz_success_k;
}

static sz_status_t cpu_capabilities_detected_(sz_size_t device, sz_capability_t *capabilities) {
    sz_unused_(device);
    return sz_cpu_capabilities_detected(capabilities);
}

static sz_status_t cpu_capabilities_enabled_(sz_size_t device, sz_capability_t *capabilities) {
    sz_unused_(device);
    *capabilities = sz_py_enabled_capabilities;
    return sz_success_k;
}

/** One kind of @c stringzilla.Device, and the C queries that answer for it. */
typedef struct DeviceKind {
    char const *name;
    sz_status_t (*count_devices)(sz_size_t *count);
    sz_status_t (*capabilities_detected)(sz_size_t device, sz_capability_t *capabilities);
    sz_status_t (*capabilities_compiled)(sz_capability_t *capabilities);
    sz_status_t (*capabilities_enabled)(sz_size_t device, sz_capability_t *capabilities);
} DeviceKind;

static DeviceKind const device_kinds[] = {
    {"cpu", cpu_count_devices_, cpu_capabilities_detected_, sz_cpu_capabilities_compiled, cpu_capabilities_enabled_},
    {"cuda", sz_cuda_count_devices, sz_cuda_capabilities_detected, sz_cuda_capabilities_compiled,
     sz_cuda_capabilities_enabled},
    {"rocm", sz_rocm_count_devices, sz_rocm_capabilities_detected, sz_rocm_capabilities_compiled,
     sz_rocm_capabilities_enabled},
    {"metal", sz_metal_count_devices, sz_metal_capabilities_detected, sz_metal_capabilities_compiled,
     sz_metal_capabilities_enabled},
};

static sz_size_t const device_kinds_count = sizeof(device_kinds) / sizeof(device_kinds[0]);

static DeviceKind const *device_kind_named_(PyObject *name) {
    if (!PyUnicode_Check(name)) {
        PyErr_Format(PyExc_TypeError, "device kind must be a string, got %s", Py_TYPE(name)->tp_name);
        return NULL;
    }
    for (sz_size_t index = 0; index != device_kinds_count; ++index)
        if (PyUnicode_CompareWithASCIIString(name, device_kinds[index].name) == 0) return &device_kinds[index];
    PyErr_Format(PyExc_ValueError, "device kind must be 'cpu', 'cuda', 'rocm' or 'metal', got %R", name);
    return NULL;
}

/** Counts the devices of @p kind, raising the status C reports when there are none. */
static int device_kind_count_(DeviceKind const *kind, sz_size_t *count) {
    sz_status_t const status = kind->count_devices(count);
    if (status == sz_success_k) return 1;
    sz_py_raise_status(status, kind->name);
    return 0;
}

static PyObject *device_new_(DeviceKind const *kind, sz_size_t ordinal) {
    Device *device = PyObject_New(Device, &DeviceType);
    if (!device) return NULL;
    device->kind = kind;
    device->ordinal = ordinal;
    return (PyObject *)device;
}

int sz_py_export_engine_placement(PyObject *device_object, PyObject *capabilities_object, PyObject *stream_object,
                                  sz_capability_t *capabilities, sz_size_t *ordinal, void **stream) {
    DeviceKind const *kind = &device_kinds[0];
    *ordinal = 0;
    if (device_object && device_object != Py_None) {
        if (!PyObject_TypeCheck(device_object, &DeviceType)) {
            PyErr_Format(PyExc_TypeError, "device must be a stringzilla.Device, got %s",
                         Py_TYPE(device_object)->tp_name);
            return -1;
        }
        kind = ((Device const *)device_object)->kind;
        *ordinal = ((Device const *)device_object)->ordinal;
    }
    sz_capability_t available = 0;
    sz_status_t const status = kind->capabilities_enabled(*ordinal, &available);
    if (status != sz_success_k || !available) {
        PyErr_Format(PyExc_ValueError, "StringZilla has no kernels for Device('%s', %zu): %s", kind->name,
                     (size_t)*ordinal,
                     status == sz_success_k ? "this build lacks its capabilities" : sz_status_name(status));
        return -1;
    }
    *capabilities = available;
    if (capabilities_object && capabilities_object != Py_None) {
        unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities_object);
        if (bits == (unsigned long long)-1 && PyErr_Occurred()) return -1;
        if ((sz_capability_t)bits & ~available) {
            char names[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
            sz_capabilities_name(available, names, sizeof(names));
            PyErr_Format(PyExc_ValueError, "capabilities must narrow those of Device('%s', %zu): %s", kind->name,
                         (size_t)*ordinal, names);
            return -1;
        }
        *capabilities = (sz_capability_t)bits;
    }
    if (sz_py_export_stream(stream_object, stream) != 0) return -1;
    if (kind == &device_kinds[0] && *stream) {
        PyErr_SetString(PyExc_ValueError, "a CPU engine takes no stream");
        return -1;
    }
    return 0;
}

static PyObject *Device_new(PyTypeObject *type, PyObject *args, PyObject *kwargs) {
    sz_unused_(type);
    Py_ssize_t const positional_count = PyTuple_GET_SIZE(args);
    if (positional_count > 2) {
        PyErr_Format(PyExc_TypeError, "Device() takes at most 2 arguments, got %zd", positional_count);
        return NULL;
    }
    PyObject *kind_object = positional_count > 0 ? PyTuple_GET_ITEM(args, 0) : NULL;
    PyObject *ordinal_object = positional_count > 1 ? PyTuple_GET_ITEM(args, 1) : NULL;
    Py_ssize_t position = 0;
    PyObject *key, *value;
    while (kwargs && PyDict_Next(kwargs, &position, &key, &value)) {
        PyObject **slot = PyUnicode_CompareWithASCIIString(key, "kind") == 0      ? &kind_object
                          : PyUnicode_CompareWithASCIIString(key, "ordinal") == 0 ? &ordinal_object
                                                                                  : NULL;
        if (!slot) return PyErr_Format(PyExc_TypeError, "Device() got an unexpected keyword argument '%S'", key);
        if (*slot) return PyErr_Format(PyExc_TypeError, "Device() got multiple values for argument '%S'", key);
        *slot = value;
    }
    if (!kind_object) {
        PyErr_SetString(PyExc_TypeError, "Device() missing required argument 'kind'");
        return NULL;
    }

    DeviceKind const *kind = device_kind_named_(kind_object);
    if (!kind) return NULL;
    Py_ssize_t ordinal = 0;
    if (ordinal_object) {
        ordinal = PyLong_AsSsize_t(ordinal_object);
        if (ordinal == -1 && PyErr_Occurred()) return NULL;
    }
    if (ordinal < 0) return PyErr_Format(PyExc_ValueError, "device ordinal must be non-negative, got %zd", ordinal);
    sz_size_t count = 0;
    if (!device_kind_count_(kind, &count)) return NULL;
    if ((sz_size_t)ordinal >= count)
        return PyErr_Format(PyExc_ValueError, "no %s device %zd, as this process sees %zu", kind->name, ordinal,
                            (size_t)count);
    return device_new_(kind, (sz_size_t)ordinal);
}

static PyObject *Device_repr(PyObject *self) {
    Device const *device = (Device const *)self;
    return PyUnicode_FromFormat("Device('%s', %zu)", device->kind->name, (size_t)device->ordinal);
}

static PyObject *Device_richcompare(PyObject *self, PyObject *other, int operation) {
    if (!PyObject_TypeCheck(other, &DeviceType) || (operation != Py_EQ && operation != Py_NE)) Py_RETURN_NOTIMPLEMENTED;
    Device const *first = (Device const *)self, *second = (Device const *)other;
    int const same = first->kind == second->kind && first->ordinal == second->ordinal;
    return PyBool_FromLong(same == (operation == Py_EQ));
}

static Py_hash_t Device_hash(PyObject *self) {
    Device const *device = (Device const *)self;
    return (Py_hash_t)(device->ordinal * device_kinds_count + (sz_size_t)(device->kind - device_kinds));
}

static PyObject *Device_get_kind(PyObject *self, void *closure) {
    sz_unused_(closure);
    return PyUnicode_FromString(((Device const *)self)->kind->name);
}

static PyObject *Device_get_ordinal(PyObject *self, void *closure) {
    sz_unused_(closure);
    return PyLong_FromSize_t(((Device const *)self)->ordinal);
}

static char const doc_Device_cpu[] =                     //
    "Get the CPU, the one device every process has.\n\n" //
    "Signature:\n"                                       //
    "    >>> def cpu() -> sz.Device: ...";

static PyObject *Device_cpu(PyObject *unused_self, PyObject *unused_args) {
    sz_unused_(unused_self);
    sz_unused_(unused_args);
    return device_new_(&device_kinds[0], 0);
}

static char const doc_Device_count[] =                                                       //
    "Count the devices of one kind this process sees.\n\n"                                   //
    "Args:\n"                                                                                //
    "    kind: One of 'cpu', 'cuda', 'rocm' or 'metal'.\n\n"                                 //
    "Returns:\n"                                                                             //
    "    int: One for the CPU, and how many GPUs of that kind the runtime sees.\n\n"         //
    "Raises:\n"                                                                              //
    "    RuntimeError: No device of that kind answers, or this build lacks its kernels.\n\n" //
    "Signature:\n"                                                                           //
    "    >>> def count(kind, /) -> int: ...";

static PyObject *Device_count(PyObject *unused_self, PyObject *kind_object) {
    sz_unused_(unused_self);
    DeviceKind const *kind = device_kind_named_(kind_object);
    sz_size_t count = 0;
    if (!kind || !device_kind_count_(kind, &count)) return NULL;
    return PyLong_FromSize_t(count);
}

/** Wraps what one capability query reported, raising the status of a query that failed. */
static PyObject *device_capabilities_reported_(sz_status_t status, sz_capability_t capabilities) {
    if (status != sz_success_k) {
        sz_py_raise_status(status, "Device");
        return NULL;
    }
    return capability_from_mask_(capabilities);
}

static char const doc_Device_capabilities_detected[] =                                               //
    "Get the capabilities this device can execute.\n\n"                                              //
    "Detected from CPUID or HWCAP on the CPU, and from the runtime on a GPU. Says nothing about\n"   //
    "whether the kernels were compiled in, which `capabilities_compiled` reports.\n\n"               //
    "Returns:\n"                                                                                     //
    "    Capability: One flag per capability; GPU bits sit above every member, so they print as a\n" //
    "        number.\n\n"                                                                            //
    "Signature:\n"                                                                                   //
    "    >>> def capabilities_detected(self, /) -> sz.Capability: ...";

static PyObject *Device_capabilities_detected(PyObject *self, PyObject *unused_args) {
    sz_unused_(unused_args);
    Device const *device = (Device const *)self;
    sz_capability_t capabilities = 0;
    sz_status_t const status = device->kind->capabilities_detected(device->ordinal, &capabilities);
    return device_capabilities_reported_(status, capabilities);
}

static char const doc_Device_capabilities_compiled[] =                                                   //
    "Get the capabilities of this device's kind whose kernels were compiled into this binary.\n\n"       //
    "Decided at build time. Independent of the hardware: a binary built with a broken probe toolchain\n" //
    "still detects this machine's full set while containing no SIMD kernels at all, which is what\n"     //
    "makes a scalar build hard to spot.\n\n"                                                             //
    "Signature:\n"                                                                                       //
    "    >>> def capabilities_compiled(self, /) -> sz.Capability: ...";

static PyObject *Device_capabilities_compiled(PyObject *self, PyObject *unused_args) {
    sz_unused_(unused_args);
    sz_capability_t capabilities = 0;
    sz_status_t const status = ((Device const *)self)->kind->capabilities_compiled(&capabilities);
    return device_capabilities_reported_(status, capabilities);
}

static char const doc_Device_capabilities_enabled[] =                                                     //
    "Get the capabilities kernels run with on this device, unless a call passes `capabilities=`.\n\n"     //
    "On the CPU it starts as `capabilities_detected() & capabilities_compiled()`, changes only through\n" //
    "`capabilities_enable`, and always has SERIAL. On a GPU it is that intersection, the mask an\n"       //
    "engine built with `device=` this device dispatches with.\n\n"                                        //
    "Signature:\n"                                                                                        //
    "    >>> def capabilities_enabled(self, /) -> sz.Capability: ...";

static PyObject *Device_capabilities_enabled(PyObject *self, PyObject *unused_args) {
    sz_unused_(unused_args);
    Device const *device = (Device const *)self;
    sz_capability_t capabilities = 0;
    sz_status_t const status = device->kind->capabilities_enabled(device->ordinal, &capabilities);
    return device_capabilities_reported_(status, capabilities);
}

static char const doc_Device_capabilities_enable[] =                                                      //
    "Make `wanted` the CPU capabilities kernels run with, and configure the calling thread for them.\n\n" //
    "Capabilities this CPU cannot execute or this binary lacks are dropped and SERIAL is always kept,\n"  //
    "so dispatch never reaches a kernel that cannot run here. Engines and stateful objects keep the\n"    //
    "capabilities they were built with. Mostly useful for testing capabilities one by one. Raises\n"      //
    "ValueError on a GPU.\n\n"                                                                            //
    "Args:\n"                                                                                             //
    "    wanted (Capability): Capabilities to dispatch between, for example `Capability.HASWELL`.\n\n"    //
    "Returns:\n"                                                                                          //
    "    Capability: The capabilities enabled after clamping.\n\n"                                        //
    "Signature:\n"                                                                                        //
    "    >>> def capabilities_enable(self, wanted, /) -> sz.Capability: ...";

static PyObject *Device_capabilities_enable(PyObject *self, PyObject *wanted) {
    if (((Device const *)self)->kind != &device_kinds[0])
        return PyErr_Format(PyExc_ValueError, "capabilities_enable applies to the CPU only, not %R", self);
    unsigned long long const wanted_bits = PyLong_AsUnsignedLongLong(wanted);
    if (wanted_bits == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    sz_capability_t available = sz_cap_serial_k;
    sz_cpu_capabilities_enabled(&available);
    sz_capability_t const enabled = ((sz_capability_t)wanted_bits & available) | sz_cap_serial_k;
    sz_status_t const status = sz_cpu_configure_thread(enabled);
    if (status == sz_success_k) sz_py_enabled_capabilities = enabled;
    return device_capabilities_reported_(status, enabled);
}

static char const doc_Device_configure_thread[] =                                                 //
    "Prepare the calling thread for kernels of `capabilities`.\n\n"                               //
    "`capabilities_enable` already does it for its own thread; call this on every other thread\n" //
    "that runs kernels. Raises ValueError on a GPU.\n\n"                                          //
    "Args:\n"                                                                                     //
    "    capabilities (Capability): The mask kernels on this thread run with.\n\n"                //
    "Signature:\n"                                                                                //
    "    >>> def configure_thread(self, capabilities, /) -> None: ...";

static PyObject *Device_configure_thread(PyObject *self, PyObject *capabilities) {
    if (((Device const *)self)->kind != &device_kinds[0])
        return PyErr_Format(PyExc_ValueError, "configure_thread applies to the CPU only, not %R", self);
    unsigned long long const bits = PyLong_AsUnsignedLongLong(capabilities);
    if (bits == (unsigned long long)-1 && PyErr_Occurred()) return NULL;
    sz_status_t const status = sz_cpu_configure_thread((sz_capability_t)bits);
    if (status != sz_success_k) {
        sz_py_raise_status(status, "configure_thread");
        return NULL;
    }
    Py_RETURN_NONE;
}

static PyGetSetDef Device_getset[] = {
    {"kind", Device_get_kind, NULL, "The kind of device: 'cpu', 'cuda', 'rocm' or 'metal'", NULL},
    {"ordinal", Device_get_ordinal, NULL, "The device's ordinal in its runtime, always 0 for the CPU", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef Device_methods[] = {
    {"cpu", (PyCFunction)Device_cpu, METH_STATIC | METH_NOARGS, doc_Device_cpu},
    {"count", (PyCFunction)Device_count, METH_STATIC | METH_O, doc_Device_count},
    {"capabilities_detected", (PyCFunction)Device_capabilities_detected, METH_NOARGS, doc_Device_capabilities_detected},
    {"capabilities_compiled", (PyCFunction)Device_capabilities_compiled, METH_NOARGS, doc_Device_capabilities_compiled},
    {"capabilities_enabled", (PyCFunction)Device_capabilities_enabled, METH_NOARGS, doc_Device_capabilities_enabled},
    {"capabilities_enable", (PyCFunction)Device_capabilities_enable, METH_O, doc_Device_capabilities_enable},
    {"configure_thread", (PyCFunction)Device_configure_thread, METH_O, doc_Device_configure_thread},
    {NULL, NULL, 0, NULL},
};

PyTypeObject DeviceType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "stringzilla.Device",
    .tp_doc = "One device StringZilla runs kernels on: a kind, 'cpu', 'cuda', 'rocm' or 'metal', and an ordinal.\n\n" //
              "Raises ValueError for a negative or past-the-end ordinal, and RuntimeError when no device\n"           //
              "of that kind answers.\n\n"                                                                             //
              "Signature:\n"                                                                                          //
              "    >>> def Device(kind, ordinal=0): ...",
    .tp_basicsize = sizeof(Device),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = Device_new,
    .tp_repr = Device_repr,
    .tp_richcompare = Device_richcompare,
    .tp_hash = Device_hash,
    .tp_getset = Device_getset,
    .tp_methods = Device_methods,
};

/** Builds the @c Capability IntFlag with one member per CPU capability, each spelled as the C
 *  library spells it, or returns @c NULL. */
static PyObject *capability_type_new_(void) {
    PyObject *members = PyDict_New();
    for (unsigned bit = 0; members && bit != 64; ++bit) {
        sz_capability_t const flag = (sz_capability_t)1 << bit;
        char name[STRINGZILLA_CAPABILITIES_NAME_CAPACITY];
        if ((flag & sz_cap_devices_k) || !sz_capabilities_name(flag, name, sizeof(name))) continue;
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

#pragma endregion Devices

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

int sz_py_export_stream(PyObject *stream_object, void **stream) {
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
    {"Device", &DeviceType},
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
    sz_cpu_capabilities_enabled(&sz_py_enabled_capabilities);
    sz_cpu_configure_thread(sz_py_enabled_capabilities);

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
