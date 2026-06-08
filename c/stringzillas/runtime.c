/**
 *  @file c/stringzillas/runtime.c
 *  @brief StringZillas parallel runtime: dispatch-table ownership, device scopes, unified allocator.
 *  @author Ash Vardanian
 *
 *  This translation unit is the StringZillas analogue of `c/stringzilla/runtime.c`: it owns the shared
 *  `szs_dispatch_table` definition, runs its one-time `__attribute__((constructor))` initialization by
 *  calling each per-op updater, and exports the version / capabilities metadata. It also implements the
 *  device-scope lifecycle (`szs_device_scope_*`) over the plain-C `szs_device_scope_impl_t` POD (backed
 *  by a fork_union thread pool on the CPU side) and the unified-memory allocator shim. The per-op
 *  function pointers and the `SZ_DYNAMIC` algorithm wrappers live in the sibling `similarities.c` and
 *  `fingerprints.c`.
 */
#include "dispatch.h"                  // `szs_dispatch_table`, per-op updaters, sets `SZ_DYNAMIC_DISPATCH 1`
#include "stringzillas/stringzillas.h" // Public `szs_*` ABI declarations
#include "fork_union.h"                // `fu_pool_new`, `fu_pool_spawn`, `fu_count_logical_cores`, ...

#include <stringzilla/stringzilla.h> // `sz_capabilities_*_implementation_`, version macros

#if SZ_USE_CUDA
#include <cuda.h>         // Driver API: `cuInit`, `cuDeviceGetCount`, `cuDeviceGetAttribute`
#include <cuda_runtime.h> // `cudaMallocManaged`, `cudaFree` for the unified allocator
#endif

#include <stdlib.h> // `malloc`, `free`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region - Dispatch Table

#if defined(_MSC_VER)
__declspec(align(64)) SZ_DISPATCH_INTERNAL szs_implementations_t szs_dispatch_table;
#else
__attribute__((aligned(64))) SZ_DISPATCH_INTERNAL szs_implementations_t szs_dispatch_table;
#endif

static void szs_dispatch_table_update_(sz_capability_t caps) {
    szs_dispatch_levenshtein_update_(caps);
    szs_dispatch_levenshtein_utf8_update_(caps);
    szs_dispatch_needleman_wunsch_update_(caps);
    szs_dispatch_smith_waterman_update_(caps);
    szs_dispatch_fingerprints_update_(caps);
}

#pragma endregion

#pragma region - Metadata

SZ_DYNAMIC int szs_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_DYNAMIC int szs_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_DYNAMIC int szs_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }

/**
 *  @brief System capabilities: CPU ISA bits ANDed comptime x runtime, plus GPU tier bits if a CUDA
 *         device is present. Cached after the first call.
 */
SZ_DYNAMIC sz_capability_t szs_capabilities(void) {
    static sz_capability_t static_caps = sz_caps_none_k;
    if (static_caps != sz_caps_none_k) return static_caps;

    sz_capability_t cpu_caps = (sz_capability_t)(sz_capabilities_comptime_implementation_() &
                                                 sz_capabilities_runtime_implementation_());

#if SZ_USE_CUDA
    sz_capability_t gpu_caps = sz_caps_none_k;
    if (cuInit(0) == CUDA_SUCCESS) {
        int device_count = 0;
        if (cuDeviceGetCount(&device_count) == CUDA_SUCCESS && device_count > 0) {
            CUdevice device;
            if (cuDeviceGet(&device, 0) == CUDA_SUCCESS) {
                int major = 0, minor = 0;
                cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
                cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
                int const sm_code = major * 10 + minor;
                gpu_caps = (sz_capability_t)(gpu_caps | sz_cap_cuda_k);
                if (sm_code >= 30) gpu_caps = (sz_capability_t)(gpu_caps | sz_cap_kepler_k);
                if (sm_code >= 90) gpu_caps = (sz_capability_t)(gpu_caps | sz_cap_hopper_k);
            }
        }
    }
    static_caps = (sz_capability_t)(cpu_caps | gpu_caps);
#else
    static_caps = cpu_caps;
#endif
    return static_caps;
}

#pragma endregion

#pragma region - One-Time Init

static void szs_dispatch_table_init_(void) { szs_dispatch_table_update_(szs_capabilities()); }

#if defined(_MSC_VER)
#if defined(_WIN64)
#pragma comment(linker, "/INCLUDE:szs_dispatch_table_init_ptr_")
#else
#pragma comment(linker, "/INCLUDE:_szs_dispatch_table_init_ptr_")
#endif
#pragma section(".CRT$XCS", read)
__declspec(allocate(".CRT$XCS")) void (*szs_dispatch_table_init_ptr_)(void) = szs_dispatch_table_init_;
#else
__attribute__((constructor)) static void szs_dispatch_table_init_on_gcc_or_clang_(void) { szs_dispatch_table_init_(); }
#endif

#pragma endregion

#pragma region - Device Scopes

SZ_DYNAMIC sz_status_t szs_device_scope_init_default(szs_device_scope_t *scope_punned, char const **error_message) {
    if (!scope_punned) {
        if (error_message) *error_message = "Scope handle must not be NULL";
        return sz_unexpected_dimensions_k;
    }
    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)malloc(sizeof(szs_device_scope_impl_t));
    if (!scope) {
        if (error_message) *error_message = "Failed to allocate device scope";
        return sz_bad_alloc_k;
    }
    scope->kind = szs_scope_default_k;
    scope->cpu.pool = SZ_NULL;
    scope->cpu.cpu_cores = 1;
#if SZ_USE_CUDA
    scope->gpu.device_id = 0;
    scope->gpu.stream = SZ_NULL;
#endif
    scope->capabilities = szs_capabilities();
    *scope_punned = (szs_device_scope_t)scope;
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t szs_device_scope_init_cpu_cores(sz_size_t cpu_cores, szs_device_scope_t *scope_punned,
                                                       char const **error_message) {
    if (!scope_punned) {
        if (error_message) *error_message = "Scope handle must not be NULL";
        return sz_unexpected_dimensions_k;
    }
    if (cpu_cores == 0) cpu_cores = fu_count_logical_cores();
    // A single core is identical to the default (single-threaded) scope.
    if (cpu_cores <= 1) return szs_device_scope_init_default(scope_punned, error_message);

    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)malloc(sizeof(szs_device_scope_impl_t));
    if (!scope) {
        if (error_message) *error_message = "Failed to allocate CPU device scope";
        return sz_bad_alloc_k;
    }
    fu_pool_t *pool = fu_pool_new("stringzillas");
    if (!pool || !fu_pool_spawn(pool, cpu_cores, fu_caller_inclusive_k)) {
        if (pool) fu_pool_delete(pool);
        free(scope);
        if (error_message) *error_message = "Failed to spawn thread pool";
        return sz_bad_alloc_k;
    }
    scope->kind = szs_scope_cpu_k;
    scope->cpu.pool = (void *)pool;
    scope->cpu.cpu_cores = cpu_cores;
#if SZ_USE_CUDA
    scope->gpu.device_id = 0;
    scope->gpu.stream = SZ_NULL;
#endif
    scope->capabilities = (sz_capability_t)(szs_capabilities() & sz_caps_cpus_k);
    *scope_punned = (szs_device_scope_t)scope;
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t szs_device_scope_init_gpu_device(sz_size_t gpu_device, szs_device_scope_t *scope_punned,
                                                        char const **error_message) {
    if (!scope_punned) {
        if (error_message) *error_message = "Scope handle must not be NULL";
        return sz_unexpected_dimensions_k;
    }
#if SZ_USE_CUDA
    sz_capability_t caps = szs_capabilities();
    if ((caps & sz_cap_cuda_k) != sz_cap_cuda_k) {
        if (error_message) *error_message = "No CUDA-capable device available";
        return sz_missing_gpu_k;
    }
    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)malloc(sizeof(szs_device_scope_impl_t));
    if (!scope) {
        if (error_message) *error_message = "Failed to allocate GPU device scope";
        return sz_bad_alloc_k;
    }
    scope->kind = szs_scope_gpu_k;
    scope->cpu.pool = SZ_NULL;
    scope->cpu.cpu_cores = 1;
    scope->gpu.device_id = (int)gpu_device;
    scope->gpu.stream = SZ_NULL;
    scope->capabilities = (sz_capability_t)(caps & sz_caps_cuda_k);
    *scope_punned = (szs_device_scope_t)scope;
    return sz_success_k;
#else
    sz_unused_(gpu_device);
    if (error_message) *error_message = "CUDA support not compiled in";
    return sz_missing_gpu_k;
#endif
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_cpu_cores(szs_device_scope_t scope_punned, sz_size_t *cpu_cores,
                                                      char const **error_message) {
    if (!scope_punned || !cpu_cores) {
        if (error_message) *error_message = "Invalid null pointer argument";
        return sz_unexpected_dimensions_k;
    }
    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)scope_punned;
    if (scope->kind == szs_scope_gpu_k) {
        if (error_message) *error_message = "Device scope is GPU-only";
        return sz_unexpected_dimensions_k;
    }
    *cpu_cores = scope->cpu.pool ? scope->cpu.cpu_cores : 1;
    return sz_success_k;
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_gpu_device(szs_device_scope_t scope_punned, sz_size_t *gpu_device,
                                                       char const **error_message) {
    if (!scope_punned || !gpu_device) {
        if (error_message) *error_message = "Invalid null pointer argument";
        return sz_unexpected_dimensions_k;
    }
#if SZ_USE_CUDA
    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)scope_punned;
    if (scope->kind == szs_scope_gpu_k) {
        *gpu_device = (sz_size_t)scope->gpu.device_id;
        return sz_success_k;
    }
#else
    sz_unused_(scope_punned);
    sz_unused_(gpu_device);
#endif
    if (error_message) *error_message = "Device scope is CPU-only";
    return sz_unexpected_dimensions_k;
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_capabilities(szs_device_scope_t scope_punned, sz_capability_t *capabilities,
                                                         char const **error_message) {
    if (!scope_punned || !capabilities) {
        if (error_message) *error_message = "Invalid null pointer argument";
        return sz_unexpected_dimensions_k;
    }
    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)scope_punned;
    sz_capability_t system_caps = szs_capabilities();
    if (scope->kind == szs_scope_gpu_k) *capabilities = (sz_capability_t)(system_caps & sz_caps_cuda_k);
    else *capabilities = (sz_capability_t)(system_caps & sz_caps_cpus_k);
    return sz_success_k;
}

SZ_DYNAMIC void szs_device_scope_free(szs_device_scope_t scope_punned) {
    if (!scope_punned) return;
    szs_device_scope_impl_t *scope = (szs_device_scope_impl_t *)scope_punned;
    if (scope->cpu.pool) fu_pool_delete((fu_pool_t *)scope->cpu.pool);
    free(scope);
}

#pragma endregion

#pragma region - Unified Allocator

SZ_DYNAMIC sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc, char const **error_message) {
    if (!alloc) {
        if (error_message) *error_message = "Allocator handle must not be NULL";
        return sz_unexpected_dimensions_k;
    }
#if SZ_USE_CUDA
    extern void *szs_memory_allocate_unified_(sz_size_t, void *);
    extern void szs_memory_free_unified_(void *, sz_size_t, void *);
    alloc->allocate = (sz_memory_allocate_t)&szs_memory_allocate_unified_;
    alloc->free = (sz_memory_free_t)&szs_memory_free_unified_;
    alloc->handle = SZ_NULL;
    return sz_success_k;
#else
    sz_unused_(alloc);
    if (error_message) *error_message = "CUDA support not compiled in";
    return sz_missing_gpu_k;
#endif
}

#if SZ_USE_CUDA
void *szs_memory_allocate_unified_(sz_size_t size, void *handle) {
    sz_unused_(handle);
    void *ptr = SZ_NULL;
    if (cudaMallocManaged(&ptr, size, cudaMemAttachGlobal) != cudaSuccess) return SZ_NULL;
    return ptr;
}
void szs_memory_free_unified_(void *ptr, sz_size_t size, void *handle) {
    sz_unused_(size);
    sz_unused_(handle);
    if (ptr) cudaFree(ptr);
}
#endif

SZ_DYNAMIC void *szs_unified_alloc(sz_size_t size_bytes) {
#if SZ_USE_CUDA
    void *ptr = SZ_NULL;
    if (cudaMallocManaged(&ptr, size_bytes, cudaMemAttachGlobal) != cudaSuccess) return SZ_NULL;
    return ptr;
#else
    return malloc(size_bytes);
#endif
}

SZ_DYNAMIC void szs_unified_free(void *ptr, sz_size_t size_bytes) {
    if (!ptr) return;
#if SZ_USE_CUDA
    sz_unused_(size_bytes);
    cudaFree(ptr);
#else
    sz_unused_(size_bytes);
    free(ptr);
#endif
}

#pragma endregion

#ifdef __cplusplus
} // extern "C"
#endif
