/**
 *  @file c/stringzillas/runtime.cuh
 *  @brief Parallel runtime: version, capabilities, device scopes, unified allocator shim (CPU + CUDA backends).
 *  @author Ash Vardanian
 *  @date March 23, 2025
 */
#ifndef STRINGZILLAS_RUNTIME_CUH_
#define STRINGZILLAS_RUNTIME_CUH_
#include "stringzillas.cuh"

extern "C" {

#pragma region Metadata

SZ_DYNAMIC int szs_version_major(void) { return STRINGZILLA_H_VERSION_MAJOR; }
SZ_DYNAMIC int szs_version_minor(void) { return STRINGZILLA_H_VERSION_MINOR; }
SZ_DYNAMIC int szs_version_patch(void) { return STRINGZILLA_H_VERSION_PATCH; }

SZ_DYNAMIC sz_capability_t szs_capabilities(void) {
    // Preserve the static capabilities
    static sz_capability_t static_caps = sz_caps_none_k;
    if (static_caps == sz_caps_none_k) {
        sz_capability_t cpu_caps = (sz_capability_t)(sz_capabilities_comptime_implementation_() &
                                                     sz_capabilities_runtime_implementation_());
#if SZ_USE_CUDA
        sz_capability_t gpu_caps = sz_caps_none_k;
        sz::gpu_specs_t first_gpu_specs;
        auto specs_status = static_cast<sz::status_t>(szs::gpu_specs_fetch(first_gpu_specs));
        if (specs_status == sz::status_t::missing_gpu_k) { return cpu_caps; }        // No GPUs available
        else if (specs_status != sz::status_t::success_k) { return sz_caps_none_k; } // Some bug
        gpu_caps = static_cast<sz_capability_t>(gpu_caps | sz_cap_cuda_k);
        if (first_gpu_specs.sm_code >= 30) gpu_caps = static_cast<sz_capability_t>(gpu_caps | sz_cap_kepler_k);
        if (first_gpu_specs.sm_code >= 90) gpu_caps = static_cast<sz_capability_t>(gpu_caps | sz_cap_hopper_k);
        static_caps = static_cast<sz_capability_t>(cpu_caps | gpu_caps);
#else
        static_caps = cpu_caps;
#endif // SZ_USE_CUDA
    }

    return static_caps;
}

SZ_DYNAMIC sz_status_t sz_memory_allocator_init_unified(sz_memory_allocator_t *alloc, char const **error_message) {
#if SZ_USE_CUDA
    alloc->allocate = &sz_memory_allocate_from_unified_;
    alloc->free = &sz_memory_free_from_unified_;
    alloc->handle = nullptr;
    return propagate_error(sz::status_t::success_k, error_message);
#else
    sz_unused_(alloc); // Suppress unused parameter warning when CUDA is not used
    return propagate_error(sz::status_t::missing_gpu_k, error_message);
#endif
}

#pragma endregion Metadata

#pragma region Device Scopes

SZ_DYNAMIC sz_status_t szs_device_scope_init_default(szs_device_scope_t *scope_punned, char const **error_message) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");
    auto *scope = new device_scope_t {default_scope_t {}};
    if (!scope) return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate device scope");
    *scope_punned = reinterpret_cast<szs_device_scope_t>(scope);
    return propagate_error(sz::status_t::success_k, error_message);
}

SZ_DYNAMIC sz_status_t szs_device_scope_init_cpu_cores(sz_size_t cpu_cores, szs_device_scope_t *scope_punned,
                                                       char const **error_message) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");

    // If `cpu_cores` is 0, use all available cores
    if (cpu_cores == 0) cpu_cores = std::thread::hardware_concurrency();

    // If `cpu_cores` is 1, redirect to default scope
    if (cpu_cores == 1) return szs_device_scope_init_default(scope_punned, error_message);

    sz::cpu_specs_t specs;
    auto executor = std::make_unique<fu::basic_pool_t>();
    if (!executor->try_spawn(cpu_cores))
        return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to spawn thread pool");

    auto *scope = new (std::nothrow)
        device_scope_t(std::in_place_type_t<cpu_scope_t> {}, std::move(executor), std::move(specs));
    if (!scope) return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate CPU device scope");

    *scope_punned = reinterpret_cast<szs_device_scope_t>(scope);
    return propagate_error(sz::status_t::success_k, error_message);
}

SZ_DYNAMIC sz_status_t szs_device_scope_init_gpu_device(sz_size_t gpu_device, szs_device_scope_t *scope_punned,
                                                        char const **error_message) {
    sz_assert_(scope_punned != nullptr && "Scope must not be null");

#if SZ_USE_CUDA
    sz::gpu_specs_t specs;
    auto specs_status = szs::gpu_specs_fetch(specs, static_cast<int>(gpu_device));
    if (specs_status.status != sz::status_t::success_k) { return propagate_error(specs_status, error_message); }
    szs::cuda_executor_t executor;
    auto executor_status = executor.try_scheduling(static_cast<int>(gpu_device));
    if (executor_status.status != sz::status_t::success_k) return propagate_error(executor_status, error_message);

    auto *scope = new (std::nothrow)
        device_scope_t {gpu_scope_t {.executor = std::move(executor), .specs = std::move(specs)}};
    if (!scope) return propagate_error(sz::status_t::bad_alloc_k, error_message, "Failed to allocate GPU device scope");
    *scope_punned = reinterpret_cast<szs_device_scope_t>(scope);
    return propagate_error(sz::status_t::success_k, error_message);
#else
    sz_unused_(gpu_device);
    sz_unused_(scope_punned);
    return propagate_error(sz::status_t::missing_gpu_k, error_message, "CUDA support not compiled in");
#endif
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_cpu_cores(szs_device_scope_t scope_punned, sz_size_t *cpu_cores,
                                                      char const **error_message) {
    if (scope_punned == nullptr || cpu_cores == nullptr)
        return propagate_error(sz::status_t::unknown_k, error_message, "Invalid null pointer argument");
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);

    if (std::holds_alternative<cpu_scope_t>(scope->variants)) {
        auto &cpu_scope = std::get<cpu_scope_t>(scope->variants);
        if (cpu_scope.executor_ptr) {
            *cpu_cores = cpu_scope.executor_ptr->threads_count();
            return propagate_error(sz::status_t::success_k, error_message);
        }
    }
    // Default scope is single-threaded
    else if (std::holds_alternative<default_scope_t>(scope->variants)) {
        *cpu_cores = 1;
        return propagate_error(sz::status_t::success_k, error_message);
    }

    return propagate_error(sz::status_t::unknown_k, error_message, "Device scope is GPU-only");
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_gpu_device(szs_device_scope_t scope_punned, sz_size_t *gpu_device,
                                                       char const **error_message) {
    if (scope_punned == nullptr || gpu_device == nullptr)
        return propagate_error(sz::status_t::unknown_k, error_message, "Invalid null pointer argument");

#if SZ_USE_CUDA
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    if (std::holds_alternative<gpu_scope_t>(scope->variants)) {
        auto &gpu_scope = std::get<gpu_scope_t>(scope->variants);
        *gpu_device = static_cast<sz_size_t>(gpu_scope.executor.device_id());
        return propagate_error(sz::status_t::success_k, error_message);
    }
#else
    sz_unused_(scope_punned);
    sz_unused_(gpu_device);
#endif

    return propagate_error(sz::status_t::unknown_k, error_message, "Device scope is CPU-only");
}

SZ_DYNAMIC void szs_device_scope_free(szs_device_scope_t scope_punned) {
    if (scope_punned == nullptr) return;
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    delete scope;
}

SZ_DYNAMIC sz_status_t szs_device_scope_get_capabilities(szs_device_scope_t scope_punned, sz_capability_t *capabilities,
                                                         char const **error_message) {

    if (scope_punned == nullptr || capabilities == nullptr)
        return propagate_error(sz::status_t::unknown_k, error_message, "Invalid null pointer argument");
    sz_capability_t system_caps = szs_capabilities();

#if SZ_USE_CUDA
    auto *scope = reinterpret_cast<device_scope_t *>(scope_punned);
    if (std::holds_alternative<gpu_scope_t>(scope->variants)) {
        // For GPU scope, intersect system capabilities with CUDA capabilities
        *capabilities = static_cast<sz_capability_t>(system_caps & sz_caps_cuda_k);
        return propagate_error(sz::status_t::success_k, error_message);
    }
#endif

    // For default and CPU scopes, intersect system capabilities with CPU capabilities
    *capabilities = static_cast<sz_capability_t>(system_caps & sz_caps_cpus_k);
    return propagate_error(sz::status_t::success_k, error_message);
}

#pragma endregion Device Scopes

#pragma region Unified Allocator

SZ_DYNAMIC void *szs_unified_alloc(sz_size_t size_bytes) {
#if SZ_USE_CUDA
    return szs::unified_alloc_t {}.allocate(size_bytes);
#else
    return std::malloc(size_bytes);
#endif
}

SZ_DYNAMIC void szs_unified_free(void *ptr, sz_size_t size_bytes) {
    if (!ptr) return;
#if SZ_USE_CUDA
    szs::unified_alloc_t {}.deallocate(static_cast<char *>(ptr), size_bytes);
#else
    sz_unused_(size_bytes);
    std::free(ptr);
#endif
}

#pragma endregion Unified Allocator
}

#endif // STRINGZILLAS_RUNTIME_CUH_
