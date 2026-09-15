/**
 *  @brief Shared types and forward declarations for the per-domain `stringzillas` files.
 *  @file python/stringzillas/stringzillas.h
 *  @author Ash Vardanian
 *
 *  The `stringzillas` extension is split into one translation unit per domain - `similarities.c` and
 *  `fingerprints.c` - alongside the `device_scope.c` execution-context object and the `stringzillas.c`
 *  module-init file. This header carries everything more than one of those files touches: the
 *  `DeviceScope` struct layout, the `PyTypeObject` forward-declaration set the module-init file needs, the
 *  import-time capsule function pointers, and the shared allocator and capability state. Structs read by a
 *  single translation unit live in that file instead.
 *
 *  The sibling `stringzilla` extension has its own private header, `stringzilla.h`. The two must never be
 *  mixed: both define `SZ_METHOD_FLAGS`, to different values, and this extension shadows the sibling's
 *  capsule-exported function names as pointers resolved at import time.
 *
 *  Not installed; private to this extension's build.
 */
#ifndef STRINGZILLAS_PYTHON_STRINGZILLAS_H_
#define STRINGZILLAS_PYTHON_STRINGZILLAS_H_

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>    // `O_RDNLY`
#include <sys/mman.h> // `mmap`
#include <sys/stat.h> // `stat`
#include <sys/types.h>
#endif

#ifdef _MSC_VER
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#else
#include <limits.h> // `SSIZE_MAX`
#include <unistd.h> // `ssize_t`
#endif

// It seems like some Python versions forget to include a header, so we should:
// https://github.com/ashvardanian/StringZilla/actions/runs/7706636733/job/21002535521
#ifndef SSIZE_MAX
#define SSIZE_MAX (SIZE_MAX / 2)
#endif

#include <errno.h>  // `errno`
#include <stdio.h>  // `fopen`
#include <stdlib.h> // `rand`, `srand`
#include <time.h>   // `time`

#include <Python.h> // CPython API

/**
 *  NumPy resolves its C API through one function table per extension, so the table has to be named and
 *  shared across the four translation units. `PyInit_stringzillas` is the single owner that calls
 *  `import_array()`; every other unit sees the same table as an `extern` pointer.
 */
#define PY_ARRAY_UNIQUE_SYMBOL SZS_NUMPY_API_
#ifndef SZS_NUMPY_API_OWNER_
#define NO_IMPORT_ARRAY
#endif
#include <numpy/arrayobject.h> // NumPy C API

#include <stringzillas/stringzillas.h>

/// @brief  Varargs + keywords, the calling convention every `stringzillas` engine method uses.
#define SZ_METHOD_FLAGS METH_VARARGS | METH_KEYWORDS

/**
 *  Per-object serialization for free-threaded builds.
 *
 *  Engines keep a grow-only `scratch_` buffer that every call writes through, and an explicit
 *  `DeviceScope` owns either a fork-union pool - which admits a single driver - or a CUDA executor.
 *  The GIL is what serializes both today, so a build without it has to say so out loud. The default
 *  scope needs no lock: it is an empty struct handing out a stateless executor by value.
 *
 *  `PyMutex` is zero-initialized, and `tp_alloc` zeroes the object, so these fields need no setup or
 *  teardown. On a GIL build every macro compiles away. Locks are always taken scope-first, then
 *  engine, so two threads sharing either object cannot deadlock.
 */
#ifdef Py_GIL_DISABLED
#define SZS_LOCK_FIELD_ PyMutex lock;
#define SZS_LOCK_(mutex) PyMutex_Lock(mutex)
#define SZS_UNLOCK_(mutex) PyMutex_Unlock(mutex)
#else
#define SZS_LOCK_FIELD_
#define SZS_LOCK_(mutex) ((void)0)
#define SZS_UNLOCK_(mutex) ((void)0)
#endif

/**
 *  @brief  The C API the sibling `stringzilla` module exports through its `_sz_py_api` capsule.
 *
 *  Must stay byte-identical to the definition in `python/stringzilla.c`: the capsule carries no version
 *  tag, so a drifted or reordered field makes the cast below misread memory silently.
 */
typedef struct PyAPI {
    sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
    sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
    sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
    sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *);
} PyAPI;

/**
 *  Function pointers unpacked from that capsule at import time, deliberately shadowing the sibling
 *  extension's exported names. Defined in `stringzillas.c` next to the import logic that fills them.
 */
extern sz_bool_t (*sz_py_export_string_like)(PyObject *, sz_cptr_t *, sz_size_t *);
extern sz_bool_t (*sz_py_export_strings_as_sequence)(PyObject *, sz_sequence_t *);
extern sz_bool_t (*sz_py_export_strings_as_u32tape)(PyObject *, sz_cptr_t *, sz_u32_t const **, sz_size_t *);
extern sz_bool_t (*sz_py_export_strings_as_u64tape)(PyObject *, sz_cptr_t *, sz_u64_t const **, sz_size_t *);
extern sz_bool_t (*sz_py_replace_strings_allocator)(PyObject *, sz_memory_allocator_t *);

/** Whether NumPy resolved at import; every engine refuses array arguments without it. */
extern int numpy_available;
/** Default device scope that can be safely reused across calls; stateless and thread-safe underneath. */
extern szs_device_scope_t default_device_scope;
/** What this binary ships, probed once at import and never rewritten. */
extern sz_capability_t comptime_capabilities_;
/** What this machine offers, probed once at import and never rewritten. */
extern sz_capability_t runtime_capabilities_;
/** The selected subset of `comptime_capabilities_ & runtime_capabilities_` that dispatch currently uses. */
extern sz_capability_t active_capabilities_;
/** Static unified memory allocator for GPU compatibility. */
extern sz_memory_allocator_t unified_allocator;
/** Default CPU-side allocator for buffer-based flows. */
extern sz_memory_allocator_t default_allocator;

extern PyTypeObject DeviceScopeType;
extern PyTypeObject LevenshteinDistancesType;
extern PyTypeObject LevenshteinDistancesUTF8Type;
extern PyTypeObject NeedlemanWunschType;
extern PyTypeObject SmithWatermanType;
extern PyTypeObject FingerprintsType;
extern PyTypeObject SubstringsType;

/** Shared documentation of the `__capabilities__` getter every engine exposes. */
extern char const doc_capabilities[];

/**
 *  @brief  Device scope for controlling execution context (CPU cores or GPU device).
 */
typedef struct {
    PyObject ob_base;
    szs_device_scope_t handle;
    char description[32];
    SZS_LOCK_FIELD_
} DeviceScope;

/**
 * @brief Set appropriate Python exception based on StringZilla status code and error detail.
 * @param[in] status The StringZilla status code
 * @param[in] error_detail Detailed error message from StringZilla (never NULL)
 * @param[in] context Context string describing the operation (e.g., "Levenshtein initialization")
 */
extern void set_stringzilla_error(sz_status_t status, char const *error_detail, char const *context);

/**
 *  @brief Creates a Python tuple from capabilities mask.
 *  @param[in] caps Capabilities mask
 *  @return New reference to Python tuple, or NULL on error
 */
extern PyObject *capabilities_to_tuple(sz_capability_t caps);

/**
 *  @brief Parse capabilities from a sequence of strings and clamp them to what this build can run.
 *  @param[in] caps_obj Sequence of capability strings (e.g., ('serial', 'haswell')), or a `DeviceScope`.
 *  @param[out] result Output capability mask after intersection.
 *  @return 0 on success, -1 on error (with Python exception set).
 */
extern int parse_and_intersect_capabilities(PyObject *caps_obj, sz_capability_t *result);

/**
 *  @brief Helper function to automatically swap a Strs object's allocator to unified memory for GPU kernels.
 *  @param[in] strs_obj The Strs object to swap allocator for
 *  @return sz_true_k on success, sz_false_k on failure
 *  @note Sets Pythonic error on failure.
 */
SZ_HELPER_AUTO sz_bool_t try_swap_to_unified_allocator(PyObject *strs_obj) {
    if (!strs_obj || !sz_py_replace_strings_allocator || !sz_py_export_strings_as_sequence) return sz_false_k;

    // Nothing but a `Strs` owns an allocator, so anything else is a type error rather than a memory one.
    sz_sequence_t probe_sequence;
    if (!sz_py_export_strings_as_sequence(strs_obj, &probe_sequence)) {
        PyErr_Format(PyExc_TypeError, "Expected a Strs collection, received %s", Py_TYPE(strs_obj)->tp_name);
        return sz_false_k;
    }

    // Try to swap to unified allocator - this will be a no-op if already using it
    sz_bool_t success = sz_py_replace_strings_allocator(strs_obj, &unified_allocator);

    if (!success) {
        // Always fatal: GPU kernels require unified/device-accessible memory
        PyErr_SetString( //
            PyExc_BufferError,
            "Device memory mismatch: GPU kernels require unified/device-accessible memory. " //
            "Consider reducing input size, freeing memory, or using CPU capabilities.");
        return sz_false_k;
    }
    return sz_true_k;
}

/**
 *  @brief Whether an engine addresses its buffers from the device, so every one of them must be reachable there.
 *
 *  Unified memory is one way to satisfy this and plain device memory is another, so the question is device
 *  accessibility rather than any single allocator.
 *
 *  @param[in] capabilities The capabilities bitmask of the current engine.
 */
SZ_HELPER_AUTO sz_bool_t requires_device_memory(sz_capability_t capabilities) {
    return (capabilities & sz_cap_cuda_k) != 0;
}

/**
 *  @brief Frees the unified allocation a NumPy array was built over, once that array is collected.
 *  @param[in] capsule Holds the pointer under the name `"szs_unified"`, with the byte count as its context.
 */
SZ_HELPER_AUTO void free_unified_capsule(PyObject *capsule) {
    void *const allocation = PyCapsule_GetPointer(capsule, "szs_unified");
    if (allocation)
        unified_allocator.free(allocation, (sz_size_t)(uintptr_t)PyCapsule_GetContext(capsule),
                               unified_allocator.handle);
}

/**
 *  @brief A NumPy array over freshly allocated unified memory, which both the host and a kernel can address.
 *
 *  Keeps the engines' outputs zero-copy on a GPU scope without changing what callers receive: the result is
 *  an ordinary `ndarray`, and the allocation dies with it.
 *
 *  @param[in] dimensions One or two, matching @p shape.
 *  @param[in] element_bytes Width of one element, which must agree with @p type_number.
 *  @return A new reference, or NULL with a Python exception set.
 */
SZ_HELPER_AUTO PyObject *new_unified_array(int dimensions, npy_intp const *shape, int type_number,
                                           sz_size_t element_bytes) {
    sz_size_t elements = 1;
    for (int axis = 0; axis < dimensions; ++axis) elements *= (sz_size_t)shape[axis];
    // An empty result still allocates one element, so the array never wraps a null pointer.
    sz_size_t const total_bytes = elements ? elements * element_bytes : element_bytes;

    void *const allocation = unified_allocator.allocate(total_bytes, unified_allocator.handle);
    if (!allocation) return PyErr_NoMemory();

    PyObject *const array = PyArray_SimpleNewFromData(dimensions, (npy_intp *)shape, type_number, allocation);
    if (!array) {
        unified_allocator.free(allocation, total_bytes, unified_allocator.handle);
        return PyErr_NoMemory();
    }

    // The array does not own foreign memory, so a capsule base carries the lifetime and the size to free.
    PyObject *const owner = PyCapsule_New(allocation, "szs_unified", free_unified_capsule);
    if (!owner || PyCapsule_SetContext(owner, (void *)(uintptr_t)total_bytes) != 0 ||
        PyArray_SetBaseObject((PyArrayObject *)array, owner) != 0) {
        Py_XDECREF(owner);
        Py_DECREF(array);
        unified_allocator.free(allocation, total_bytes, unified_allocator.handle);
        return NULL;
    }
    return array;
}

/** @brief One caller-supplied device buffer, as `__cuda_array_interface__` or `__dlpack__` described it. */
typedef struct device_buffer_t {
    void *data;
    sz_size_t rows;          //? One for a vector, the query count for a matrix
    sz_size_t columns;       //? Elements per row
    sz_size_t row_stride;    //? Elements between consecutive rows, at least `columns`
    sz_size_t element_bytes; //? Width of one element, matched against what the engine writes
} device_buffer_t;

/** @brief The DLPack device kinds a kernel can reach; the rest are host memory as far as we are concerned. */
enum { dlpack_device_cuda_k = 2, dlpack_device_cuda_managed_k = 13 };

/** @brief DLPack's tensor descriptor, as its stable C ABI lays it out. */
typedef struct dlpack_tensor_t {
    void *data;
    struct {
        int32_t device_type;
        int32_t device_id;
    } device;
    int32_t dimensions;
    struct {
        uint8_t code;
        uint8_t bits;
        uint16_t lanes;
    } dtype;
    int64_t *shape;
    int64_t *strides; //? Null means C-contiguous
    uint64_t byte_offset;
} dlpack_tensor_t;

/** @brief DLPack's owning wrapper, whose deleter this binding calls once it has read the descriptor. */
typedef struct dlpack_managed_tensor_t {
    dlpack_tensor_t tensor;
    void *manager_context;
    void (*deleter)(struct dlpack_managed_tensor_t *self);
} dlpack_managed_tensor_t;

/**
 *  @brief Fills @p result from a shape, an element width and a row stride, refusing shapes the engines cannot write.
 *
 *  One or two dimensions only, rows at least as wide as they are long, and a stride that is a whole number of
 *  elements - the C ABI expresses a row stride in elements and has no way to say "half an element".
 */
SZ_HELPER_AUTO int fill_device_buffer(device_buffer_t *result, void *data, sz_size_t dimensions, int64_t const *shape,
                                      sz_size_t row_stride_bytes, sz_size_t element_bytes) {
    if (dimensions != 1 && dimensions != 2) {
        PyErr_SetString(PyExc_ValueError, "Device buffers must be 1- or 2-dimensional");
        return -1;
    }
    result->data = data;
    result->element_bytes = element_bytes;
    result->rows = dimensions == 2 ? (sz_size_t)shape[0] : 1;
    result->columns = dimensions == 2 ? (sz_size_t)shape[1] : (sz_size_t)shape[0];
    if (row_stride_bytes % element_bytes != 0) {
        PyErr_SetString(PyExc_ValueError, "Device buffer rows must be strided by a whole number of elements");
        return -1;
    }
    result->row_stride = row_stride_bytes / element_bytes;
    if (result->row_stride < result->columns) {
        PyErr_SetString(PyExc_ValueError, "Device buffer rows overlap; the row stride is narrower than a row");
        return -1;
    }
    return 0;
}

/**
 *  @brief Reads a device pointer out of an object exposing `__cuda_array_interface__`.
 *
 *  Version 2 and above, C-contiguous along the last axis, and a non-null `mask` is refused - the engines write
 *  dense rows and honour no mask.
 *
 *  @param[in] buffer_obj The candidate; one exposing no interface reports `sz_false_k` with no exception set,
 *             so the caller can fall through to another protocol.
 */
SZ_HELPER_AUTO sz_bool_t try_read_cuda_array_interface(PyObject *buffer_obj, sz_size_t expected_element_bytes,
                                                       device_buffer_t *result) {
    PyObject *interface = PyObject_GetAttrString(buffer_obj, "__cuda_array_interface__");
    if (!interface) {
        PyErr_Clear();
        return sz_false_k;
    }
    sz_bool_t parsed = sz_false_k;
    PyObject *shape_obj = NULL, *data_obj = NULL, *typestr_obj = NULL, *strides_obj = NULL, *mask_obj = NULL;
    if (!PyDict_Check(interface)) {
        PyErr_SetString(PyExc_TypeError, "__cuda_array_interface__ must be a dict");
        goto done;
    }

    mask_obj = PyDict_GetItemString(interface, "mask"); //? Borrowed
    if (mask_obj && mask_obj != Py_None) {
        PyErr_SetString(PyExc_ValueError, "Masked device buffers are not supported");
        goto done;
    }
    shape_obj = PyDict_GetItemString(interface, "shape");
    data_obj = PyDict_GetItemString(interface, "data");
    typestr_obj = PyDict_GetItemString(interface, "typestr");
    strides_obj = PyDict_GetItemString(interface, "strides");
    if (!shape_obj || !data_obj || !typestr_obj || !PyTuple_Check(shape_obj) || !PyTuple_Check(data_obj)) {
        PyErr_SetString(PyExc_ValueError, "__cuda_array_interface__ is missing shape, data or typestr");
        goto done;
    }

    Py_ssize_t const dimensions = PyTuple_GET_SIZE(shape_obj);
    int64_t shape[2] = {0, 0};
    if (dimensions < 1 || dimensions > 2) {
        PyErr_SetString(PyExc_ValueError, "Device buffers must be 1- or 2-dimensional");
        goto done;
    }
    for (Py_ssize_t axis = 0; axis < dimensions; ++axis)
        shape[axis] = (int64_t)PyLong_AsLongLong(PyTuple_GET_ITEM(shape_obj, axis));
    if (PyErr_Occurred()) goto done;

    void *const data = PyLong_AsVoidPtr(PyTuple_GET_ITEM(data_obj, 0));
    if (PyErr_Occurred()) goto done;

    // A dtype narrower or wider than the engine writes would silently reinterpret every cell.
    char const *const typestr = PyUnicode_AsUTF8(typestr_obj);
    if (!typestr) goto done;
    sz_size_t const element_bytes = (sz_size_t)strtoul(typestr + 2, NULL, 10);
    if (element_bytes != expected_element_bytes) {
        PyErr_Format(PyExc_TypeError, "Device buffer has %zu-byte elements, expected %zu", (size_t)element_bytes,
                     (size_t)expected_element_bytes);
        goto done;
    }

    // A null `strides` is the protocol's way of saying C-contiguous, which makes the row stride the row width.
    sz_size_t row_stride_bytes = (sz_size_t)shape[dimensions - 1] * element_bytes;
    if (strides_obj && strides_obj != Py_None) {
        if (!PyTuple_Check(strides_obj) || PyTuple_GET_SIZE(strides_obj) != dimensions) {
            PyErr_SetString(PyExc_ValueError, "__cuda_array_interface__ strides must match the shape");
            goto done;
        }
        if ((sz_size_t)PyLong_AsSsize_t(PyTuple_GET_ITEM(strides_obj, dimensions - 1)) != element_bytes) {
            PyErr_SetString(PyExc_ValueError, "Device buffer rows must be contiguous along the last axis");
            goto done;
        }
        if (dimensions == 2) row_stride_bytes = (sz_size_t)PyLong_AsSsize_t(PyTuple_GET_ITEM(strides_obj, 0));
        if (PyErr_Occurred()) goto done;
    }

    parsed = fill_device_buffer(result, data, (sz_size_t)dimensions, shape, row_stride_bytes, element_bytes) == 0
                 ? sz_true_k
                 : sz_false_k;

done:
    Py_DECREF(interface);
    return parsed;
}

/**
 *  @brief Reads a device pointer out of an object exposing `__dlpack__`, gated on `__dlpack_device__`.
 *  @param[in] buffer_obj The candidate; see @ref try_read_cuda_array_interface for the return contract.
 */
SZ_HELPER_AUTO sz_bool_t try_read_dlpack(PyObject *buffer_obj, sz_size_t expected_element_bytes,
                                         device_buffer_t *result) {
    PyObject *device_method = PyObject_GetAttrString(buffer_obj, "__dlpack_device__");
    if (!device_method) {
        PyErr_Clear();
        return sz_false_k;
    }
    PyObject *const device_pair = PyObject_CallNoArgs(device_method);
    Py_DECREF(device_method);
    if (!device_pair) return sz_false_k;
    if (!PyTuple_Check(device_pair) || PyTuple_GET_SIZE(device_pair) != 2) {
        Py_DECREF(device_pair);
        PyErr_SetString(PyExc_ValueError, "__dlpack_device__ must return a (device_type, device_id) pair");
        return sz_false_k;
    }
    long const device_type = PyLong_AsLong(PyTuple_GET_ITEM(device_pair, 0));
    Py_DECREF(device_pair);
    if (device_type != dlpack_device_cuda_k && device_type != dlpack_device_cuda_managed_k) {
        PyErr_SetString(PyExc_BufferError, "A GPU scope needs a CUDA device buffer; this one lives on the host");
        return sz_false_k;
    }

    PyObject *const capsule = PyObject_CallMethod(buffer_obj, "__dlpack__", NULL);
    if (!capsule) return sz_false_k;
    dlpack_managed_tensor_t *const managed = (dlpack_managed_tensor_t *)PyCapsule_GetPointer(capsule, "dltensor");
    if (!managed) {
        Py_DECREF(capsule);
        return sz_false_k;
    }

    dlpack_tensor_t const *const tensor = &managed->tensor;
    sz_bool_t parsed = sz_false_k;
    sz_size_t const element_bytes = (sz_size_t)(tensor->dtype.bits / 8) * tensor->dtype.lanes;
    if (element_bytes != expected_element_bytes)
        PyErr_Format(PyExc_TypeError, "Device buffer has %zu-byte elements, expected %zu", (size_t)element_bytes,
                     (size_t)expected_element_bytes);
    else if (tensor->strides && tensor->strides[tensor->dimensions - 1] != 1)
        PyErr_SetString(PyExc_ValueError, "Device buffer rows must be contiguous along the last axis");
    else {
        // DLPack counts strides in elements, and a null `strides` means C-contiguous.
        sz_size_t const row_stride_bytes = (tensor->dimensions == 2 && tensor->strides)
                                               ? (sz_size_t)tensor->strides[0] * element_bytes
                                               : (sz_size_t)tensor->shape[tensor->dimensions - 1] * element_bytes;
        void *const data = (char *)tensor->data + tensor->byte_offset;
        parsed = fill_device_buffer(result, data, (sz_size_t)tensor->dimensions, tensor->shape, row_stride_bytes,
                                    element_bytes) == 0
                     ? sz_true_k
                     : sz_false_k;
    }

    // The capsule is ours once read, so rename it spent and run the deleter rather than leaking the tensor.
    if (managed->deleter) managed->deleter(managed);
    PyCapsule_SetName(capsule, "used_dltensor");
    Py_DECREF(capsule);
    return parsed;
}

/**
 *  @brief Reads a caller's device buffer through whichever protocol it speaks, refusing host memory.
 *
 *  `__cuda_array_interface__` is tried first because it is a plain dictionary, where `__dlpack__` allocates a
 *  capsule that has to be consumed.
 *
 *  @param[in] expected_element_bytes Width the engine will write; another dtype is refused rather than reinterpreted.
 *  @return 0 when @p result was filled, -1 with a Python exception set otherwise.
 */
SZ_HELPER_AUTO int parse_device_buffer(PyObject *buffer_obj, sz_size_t expected_element_bytes,
                                       device_buffer_t *result) {
    if (try_read_cuda_array_interface(buffer_obj, expected_element_bytes, result)) return 0;
    if (PyErr_Occurred()) return -1;
    if (try_read_dlpack(buffer_obj, expected_element_bytes, result)) return 0;
    if (PyErr_Occurred()) return -1;
    PyErr_Format(PyExc_BufferError,
                 "A GPU scope writes its results from the device, so '%s' must expose " //
                 "__cuda_array_interface__ or __dlpack__ over CUDA memory",
                 Py_TYPE(buffer_obj)->tp_name);
    return -1;
}

#endif // STRINGZILLAS_PYTHON_STRINGZILLAS_H_
