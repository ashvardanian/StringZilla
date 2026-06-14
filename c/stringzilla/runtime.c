/**
 *  @file c/stringzilla/runtime.c
 *  @brief StringZilla C library with dynamic backed dispatch for the most appropriate implementation.
 *  @author Ash Vardanian
 *  @date January 16, 2024
 *
 *  This translation unit owns only the cross-cutting glue: the shared `sz_dispatch_table`
 *  definition, its one-time initialization, and the version & capabilities exports. The
 *  per-domain backends and `SZ_DYNAMIC` wrappers live in the sibling `compare.c`, `memory.c`,
 *  `hash.c`, `find.c`, `sort.c`, `intersect.c`, `utf8_iterate.c`, `utf8_case_fold.c`, and
 *  `utf8_case_insensitive.c`.
 */
#include "dispatch.h"                // `sz_dispatch_table`, per-domain updaters, sets `SZ_DYNAMIC_DISPATCH 1`
#include <stringzilla/stringzilla.h> // `sz_capabilities_*_implementation_`, version macros

#if SZ_AVOID_LIBC
#ifdef _MSC_VER
typedef sz_size_t size_t; // Reuse the type definition we've inferred from `stringzilla.h`
#else
typedef __SIZE_TYPE__ size_t; // For GCC/Clang
#endif
#endif

#if defined(SZ_IS_WINDOWS_)
#include <windows.h> // `DllMain`
#endif

#if defined(_MSC_VER)
__declspec(align(64)) SZ_DISPATCH_INTERNAL sz_implementations_t sz_dispatch_table;
#else
__attribute__((aligned(64))) SZ_DISPATCH_INTERNAL sz_implementations_t sz_dispatch_table;
#endif

static void sz_dispatch_table_update_implementation_(sz_capability_t caps) {
    sz_dispatch_compare_update_(caps);
    sz_dispatch_memory_update_(caps);
    sz_dispatch_hash_update_(caps);
    sz_dispatch_find_update_(caps);
    sz_dispatch_sort_update_(caps);
    sz_dispatch_intersect_update_(caps);
    sz_dispatch_utf8_norm_update_(caps);
    sz_dispatch_utf8_iterate_update_(caps);
    sz_dispatch_utf8_case_fold_update_(caps);
    sz_dispatch_utf8_case_insensitive_update_(caps);
}

/**
 *  @brief Initializes a global static "virtual table" of supported backends
 *         Run it just once to avoid unnecessary `if`-s.
 */
SZ_DYNAMIC void sz_dispatch_table_init(void) {
    sz_capability_t caps = sz_capabilities();
    sz_dispatch_table_update_implementation_(caps);
}

SZ_DYNAMIC void sz_dispatch_table_update(sz_capability_t caps) { sz_dispatch_table_update_implementation_(caps); }

#if defined(_MSC_VER)
/*
 *  Makes sure the `sz_dispatch_table_init` function is called at startup, from either an executable or when loading
 *  a DLL. The section name must be no more than 8 characters long, and must be between .CRT$XCA and .CRT$XCZ
 *  alphabetically (exclusive). The Microsoft C++ compiler puts C++ initialisation code in .CRT$XCU, so avoid that
 *  section: https://learn.microsoft.com/en-us/cpp/c-runtime-library/crt-initialization?view=msvc-170
 */
#if defined(_WIN64)
#pragma comment(linker, "/INCLUDE:sz_dispatch_table_init_")
#else
#pragma comment(linker, "/INCLUDE:_sz_dispatch_table_init_")
#endif
#pragma section(".CRT$XCS", read)
__declspec(allocate(".CRT$XCS")) void (*sz_dispatch_table_init_)() = sz_dispatch_table_init;

/*  Called either from CRT code or out own `_DLLMainCRTStartup`, when a DLL is loaded. */
BOOL WINAPI DllMain(HINSTANCE instance_handle, DWORD reason, LPVOID reserved_pointer) {
    switch (reason) {
    case DLL_PROCESS_ATTACH:
        sz_dispatch_table_init(); // Ensure initialization
        return TRUE;
    case DLL_THREAD_ATTACH: return TRUE;
    case DLL_THREAD_DETACH: return TRUE;
    case DLL_PROCESS_DETACH: return TRUE;
    }
    return TRUE;
}

#if SZ_AVOID_LIBC
/*  Called when the DLL is loaded, and there is no CRT code. */
BOOL WINAPI _DllMainCRTStartup(HINSTANCE instance_handle, DWORD reason, LPVOID reserved_pointer) {
    DllMain(instance_handle, reason, reserved_pointer);
    return TRUE;
}
#endif

#else
__attribute__((constructor)) static void sz_dispatch_table_init_on_gcc_or_clang(void) { sz_dispatch_table_init(); }
#endif

SZ_DYNAMIC int sz_dynamic_dispatch(void) { return 1; }
SZ_DYNAMIC int sz_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_DYNAMIC int sz_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_DYNAMIC int sz_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }
SZ_DYNAMIC sz_capability_t sz_capabilities_comptime(void) { return sz_capabilities_comptime_implementation_(); }
SZ_DYNAMIC sz_capability_t sz_capabilities_runtime(void) { return sz_capabilities_runtime_implementation_(); }
SZ_DYNAMIC sz_capability_t sz_capabilities(void) {
    return (sz_capability_t)(sz_capabilities_comptime_implementation_() & sz_capabilities_runtime_implementation_());
}
SZ_DYNAMIC sz_cptr_t sz_capabilities_to_string(sz_capability_t caps) {
    return sz_capabilities_to_string_implementation_(caps);
}
