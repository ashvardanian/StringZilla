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
    if (!strs_obj || !sz_py_replace_strings_allocator) return sz_false_k;

    // Try to swap to unified allocator - this will be a no-op if already using it
    sz_bool_t success = sz_py_replace_strings_allocator(strs_obj, &unified_allocator);

    if (!success) {
        // Always fatal: GPU kernels require unified/device-accessible memory
        PyErr_SetString( //
            PyExc_RuntimeError,
            "Device memory mismatch: GPU kernels require unified/device-accessible memory. " //
            "Consider reducing input size, freeing memory, or using CPU capabilities.");
        return sz_false_k;
    }
    return sz_true_k;
}

/**
 *  @brief Helper function to determine if unified memory is required based on capabilities and device scope.
 *  @param[in] capabilities The capabilities bitmask of the current engine.
 */
SZ_HELPER_AUTO sz_bool_t requires_unified_memory(sz_capability_t capabilities) {
    return (capabilities & sz_cap_cuda_k) != 0;
}

#endif // STRINGZILLAS_PYTHON_STRINGZILLAS_H_
