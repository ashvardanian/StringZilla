/**
 *  @file cpp/stringzilla/devices.hpp
 *  @author Ash Vardanian
 *  @date March 28, 2025
 *  @brief Devices, the default capability mask and unified allocators, which link the library.
 */
#ifndef STRINGZILLA_DEVICES_HPP_
#define STRINGZILLA_DEVICES_HPP_

#if defined(STRINGZILLA_HEADER_ONLY) && STRINGZILLA_HEADER_ONLY
#error "The StringZilla C++ SDK links the library; link stringzilla::cpp, not stringzilla::header"
#endif

#include "stringzilla/capabilities.h" // `sz_capabilities_enabled_cpu`, `sz_count_devices_cuda`
#include "stringzilla/strings.h"      // `sz_allocator_init_unified_best`

#include "stringzilla/types.hpp"

namespace ashvardanian {
namespace stringzilla {

/** One device StringZilla runs kernels on: the host CPU, or a GPU by its runtime's own ordinal. */
class device_t {
    device_kind_t kind_;
    std::size_t ordinal_;

    constexpr device_t(device_kind_t kind, std::size_t ordinal) noexcept : kind_(kind), ordinal_(ordinal) {}

  public:
    /** The host CPU, which every build has. */
    static constexpr device_t cpu() noexcept { return {device_kind_t::cpu_k, 0}; }

    /** How many devices of @p kind the process sees: one CPU, or @c missing_gpu_k and zero GPUs. */
    static expected<std::size_t> count(device_kind_t kind) noexcept {
        sz_size_t count = 1;
        sz_status_t status = sz_success_k;
        switch (kind) {
        case device_kind_t::cpu_k: break;
        case device_kind_t::cuda_k: status = sz_count_devices_cuda(&count); break;
        case device_kind_t::rocm_k: status = sz_count_devices_rocm(&count); break;
        case device_kind_t::metal_k: status = sz_count_devices_metal(&count); break;
        }
        return {static_cast<std::size_t>(count), static_cast<status_t>(status)};
    }

    /** Device @p ordinal of @p kind, or @c missing_gpu_k past the last one. */
    static expected<device_t> make(device_kind_t kind, std::size_t ordinal) noexcept {
        auto const [devices, devices_status] = count(kind);
        bool const past_last = succeeded(devices_status) && ordinal >= devices;
        return {device_t {kind, ordinal}, past_last ? status_t::missing_gpu_k : devices_status};
    }

    constexpr device_kind_t kind() const noexcept { return kind_; }
    constexpr std::size_t ordinal() const noexcept { return ordinal_; }

    /** What this device runs, whether or not this binary holds kernels for it; zero on failure. */
    expected<sz_capability_t> capabilities_detected() const noexcept {
        sz_capability_t capabilities = 0;
        sz_status_t status = sz_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = sz_capabilities_detected_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = sz_capabilities_detected_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = sz_capabilities_detected_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = sz_capabilities_detected_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** What this binary holds kernels for that this device runs; zero on failure. */
    expected<sz_capability_t> capabilities_compiled() const noexcept {
        sz_capability_t capabilities = 0;
        sz_status_t status = sz_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = sz_capabilities_compiled_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = sz_capabilities_compiled_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = sz_capabilities_compiled_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = sz_capabilities_compiled_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** Both at once: the mask for this device's dispatch points; zero on failure. */
    expected<sz_capability_t> capabilities_enabled() const noexcept {
        sz_capability_t capabilities = 0;
        sz_status_t status = sz_success_k;
        switch (kind_) {
        case device_kind_t::cpu_k: status = sz_capabilities_enabled_cpu(&capabilities); break;
        case device_kind_t::cuda_k: status = sz_capabilities_enabled_cuda(ordinal_, &capabilities); break;
        case device_kind_t::rocm_k: status = sz_capabilities_enabled_rocm(ordinal_, &capabilities); break;
        case device_kind_t::metal_k: status = sz_capabilities_enabled_metal(ordinal_, &capabilities); break;
        }
        return {capabilities, static_cast<status_t>(status)};
    }

    /** Prepares the calling thread for the CPU kernels of @p capabilities; GPUs have no such
     *  kernels, so they report @c missing_kernel_k. */
    status_t configure_thread(sz_capability_t capabilities) const noexcept {
        if (kind_ != device_kind_t::cpu_k) return status_t::missing_kernel_k;
        return static_cast<status_t>(sz_configure_thread_cpu(capabilities));
    }
};

/** The mask every verb dispatches over by default: the CPU's enabled capabilities, as
 *  @c device_t::cpu().capabilities_enabled() reports them. */
inline sz_capability_t default_capabilities() noexcept {
    sz_capability_t capabilities = 0;
    return sz_capabilities_enabled_cpu(&capabilities) == sz_success_k ? capabilities : sz_cap_serial_k;
}

/** The value of a verb run over @ref default_capabilities, which always holds serial once the
 *  library links, so debug builds assert the call succeeded. */
template <typename value_type_>
value_type_ default_mask_value_(expected<value_type_> const &result) noexcept {
    sz_assert_(succeeded(result.status));
    return result.value;
}

/**
 *  @brief A @ref unified_alloc over the unified memory of @p capabilities_, or the heap for serial.
 *  @param[in] stream Null for the vendor's default device, or a stream of the device to serve.
 */
template <typename value_type_, sz_capability_t capabilities_ = sz_cap_serial_k>
expected<unified_alloc<value_type_, capabilities_>> make_unified_alloc(sz_stream_t stream = nullptr) noexcept {
    sz_allocator_t policy {};
    sz_status_t status = sz_success_k;
    if constexpr (capabilities_ == sz_cap_serial_k) status = sz_allocator_init_heap(&policy);
    else status = sz_allocator_init_unified_best(&policy, capabilities_);
    return {unified_alloc<value_type_, capabilities_>(policy, stream), static_cast<status_t>(status)};
}

} // namespace stringzilla
} // namespace ashvardanian

#endif // STRINGZILLA_DEVICES_HPP_
