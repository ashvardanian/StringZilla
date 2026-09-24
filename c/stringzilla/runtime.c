/**
 *  @file c/stringzilla/runtime.c
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *  @brief StringZilla C library with dynamic dispatch to the most appropriate implementation.
 *
 *  This translation unit owns only the cross-cutting glue: the shared
 *  @c sz_dispatch_cpu_table definition, its one-time initialization, the version &
 *  capabilities exports, and the device and allocator constructors a binding reaches a GPU
 *  engine through. The per-domain backends and @c SZ_API_RUNTIME wrappers live in sibling
 *  translation units: `compare.c`, `memory.c`, `hash.c`, `find.c`, `sort.c`, `intersect.c`,
 *  `utf8_runes.c`, `utf8_tokens.c`, `utf8_wordbreaks.c`, `utf8_graphemes.c`,
 *  `utf8_sentences.c`, `utf8_linebreaks.c`, `utf8_uncased_fold.c`, and `utf8_uncased.c`.
 */
#if defined(_WIN32)
#include <windows.h> // `DllMain`
#endif

#include <stringzilla/stringzilla.h> // `sz_capabilities_*_implementation_`, version macros
#include <stringzilla/types.cuh>     // `sz_cuda_device_t`, the unified, device, and pinned allocators

#include "dispatch.h"

#if SZ_AVOID_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

#if defined(_MSC_VER)
__declspec(align(64)) SZ_DISPATCH_INTERNAL sz_implementations_t sz_dispatch_cpu_table;
#else
__attribute__((aligned(64))) SZ_DISPATCH_INTERNAL sz_implementations_t sz_dispatch_cpu_table;
#endif

/*  The device table needs no alignment pragma: it is read once per round rather than per verb. */
SZ_DISPATCH_INTERNAL sz_implementations_gpu_t sz_dispatch_gpu_table;

static void sz_dispatch_cpu_table_update_implementation_(sz_capability_t caps) {
    sz_dispatch_compare_update_(caps);
    sz_dispatch_memory_update_(caps);
    sz_dispatch_hash_update_(caps);
    sz_dispatch_cipher_update_(caps);
    sz_dispatch_find_update_(caps);
    sz_dispatch_sort_update_(caps);
    sz_dispatch_intersect_update_(caps);
    sz_dispatch_levenshtein_update_(caps);
    sz_dispatch_overlap_update_(caps);
    sz_dispatch_substrings_update_(caps);
    sz_dispatch_utf8_norm_update_(caps);
    sz_dispatch_utf8_runes_update_(caps);
    sz_dispatch_utf8_tokens_update_(caps);
    sz_dispatch_utf8_wordbreaks_update_(caps);
    sz_dispatch_utf8_graphemes_update_(caps);
    sz_dispatch_utf8_sentences_update_(caps);
    sz_dispatch_utf8_linebreaks_update_(caps);
    sz_dispatch_utf8_uncased_fold_update_(caps);
    sz_dispatch_utf8_uncased_update_(caps);
}

/** Initializes a global static "virtual table" of supported backends. Run it just once to avoid
 *  unnecessary @c if checks. */
SZ_API_RUNTIME void sz_dispatch_cpu_table_init(void) {
    sz_capability_t caps = sz_capabilities();
    sz_dispatch_cpu_table_update_implementation_(caps);
}

SZ_API_RUNTIME void sz_dispatch_cpu_table_update(sz_capability_t caps) {
    sz_dispatch_cpu_table_update_implementation_(caps);
}

/**
 *  @brief Fills every device slot, or leaves them null where no runtime is compiled in.
 *
 *  Called by the engines' @c _init_gpu rather than from the startup constructor, so a process that
 *  only ever scores on the host never initializes a driver it does not use.
 */
SZ_DISPATCH_INTERNAL void sz_dispatch_gpu_table_init(void) {
    sz_dispatch_levenshtein_gpu_update_();
    sz_dispatch_overlap_gpu_update_();
    sz_dispatch_substrings_gpu_update_();
}

/*  Makes sure the @c sz_dispatch_cpu_table_init function is called at startup, from either an
 *  executable or when loading a DLL. The section name must be no more than 8 characters long, and
 *  must sort strictly between `.CRT$XCA` and `.CRT$XCZ`. The Microsoft C++ compiler puts C++
 *  initialisation code in `.CRT$XCU`, so avoid that section:
 *  https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-initialization?view=msvc-170 */
#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/INCLUDE:sz_dispatch_cpu_table_init_")
#else
#pragma comment(linker, "/INCLUDE:_sz_dispatch_cpu_table_init_")
#endif
#pragma section(".CRT$XCS", read)
__declspec(allocate(".CRT$XCS")) void (*sz_dispatch_cpu_table_init_)() = sz_dispatch_cpu_table_init;

/*  Called either from CRT code or our own @c _DLLMainCRTStartup, when a DLL is loaded. */
BOOL WINAPI DllMain(HINSTANCE instance_handle, DWORD reason, LPVOID reserved_pointer) {
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        sz_dispatch_cpu_table_init(); // Ensure initialization
        return TRUE;
    case DLL_THREAD_ATTACH: return TRUE;
    case DLL_THREAD_DETACH: return TRUE;
    case DLL_PROCESS_DETACH: return TRUE;
    }
    return TRUE;
}

/*  Called when the DLL is loaded, and there is no CRT code. */
#if SZ_AVOID_LIBC
BOOL WINAPI _DllMainCRTStartup(HINSTANCE instance_handle, DWORD reason, LPVOID reserved_pointer) {
    DllMain(instance_handle, reason, reserved_pointer);
    return TRUE;
}
#endif

#else
__attribute__((constructor)) static void sz_dispatch_cpu_table_init_on_gcc_or_clang(void) {
    sz_dispatch_cpu_table_init();
}
#endif

SZ_API_RUNTIME int sz_dynamic_dispatch(void) { return 1; }
SZ_API_RUNTIME int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_API_RUNTIME int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_API_RUNTIME int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
SZ_API_RUNTIME sz_capability_t sz_capabilities_comptime(void) { return sz_capabilities_comptime_implementation_(); }
SZ_API_RUNTIME sz_capability_t sz_capabilities_runtime(void) { return sz_capabilities_runtime_implementation_(); }
SZ_API_RUNTIME sz_capability_t sz_capabilities(void) {
    return (sz_capability_t)(sz_capabilities_comptime_implementation_() & sz_capabilities_runtime_implementation_());
}
SZ_API_RUNTIME sz_cptr_t sz_capabilities_to_string(sz_capability_t caps) {
    // The one place that must own storage, because the signature returns a string it does not receive.
    static char names[256];
    sz_capabilities_to_string_implementation_(caps, names, sizeof(names));
    return names;
}

/*  The device bodies live in `types.cuh`, so a header-only consumer inlines the same code these
 *  exports call. */
#pragma region Devices

#if SZ_USE_CUDA

SZ_API_RUNTIME sz_status_t sz_cuda_device_init(int ordinal, sz_cuda_device_t *device) {
    return sz_cuda_device_init_implementation_(ordinal, device);
}

SZ_API_RUNTIME void sz_cuda_device_free(sz_cuda_device_t *device) { sz_cuda_device_free_implementation_(device); }

SZ_API_RUNTIME sz_bool_t sz_memory_reaches_device(void const *pointer) {
    return sz_memory_reaches_device_implementation_(pointer);
}

SZ_API_RUNTIME void sz_memory_allocator_init_unified(sz_memory_allocator_t *allocator, sz_cuda_device_t *device) {
    sz_memory_allocator_init_unified_implementation_(allocator, device);
}

SZ_API_RUNTIME void sz_memory_allocator_init_device(sz_memory_allocator_t *allocator, sz_cuda_device_t *device) {
    sz_memory_allocator_init_device_implementation_(allocator, device);
}

SZ_API_RUNTIME void sz_memory_allocator_init_pinned(sz_memory_allocator_t *allocator, sz_cuda_device_t *device) {
    sz_memory_allocator_init_pinned_implementation_(allocator, device);
}

/*  The accessors' addresses are link-time device values, readable only by a device compilation. */
#ifdef __CUDACC__
SZ_API_RUNTIME sz_status_t sz_sequence_from_string_views_cuda(sz_string_view_t const *views, sz_size_t count,
                                                              sz_sequence_t *sequence) {
    return sz_sequence_from_string_views_cuda_implementation_(views, count, sequence);
}
#endif

#endif // SZ_USE_CUDA

#pragma endregion Devices
