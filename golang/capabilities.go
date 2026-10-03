// Devices, their capability bits, the mask every call passes, and the statuses calls return.
//
// File: golang/capabilities.go
// Author: Ash Vardanian

package sz

// #include <stringzilla/stringzilla.h>
import "C"
import (
	"errors"
	"runtime"
)

// Capability is one capability of a CPU or a GPU, or a set of them.
type Capability uint64

// Capability bit masks, each architecture's ascending by dispatch preference
const (
	CapSerial      Capability = C.sz_cap_serial_k      // Always: Fallback
	CapWestmere    Capability = C.sz_cap_westmere_k    // 2010: Intel SSE4.2, AES-NI
	CapGoldmont    Capability = C.sz_cap_goldmont_k    // 2016: Intel SHA-NI
	CapHaswell     Capability = C.sz_cap_haswell_k     // 2013: Intel AVX2
	CapSkylake     Capability = C.sz_cap_skylake_k     // 2017: Intel AVX-512
	CapIceLake     Capability = C.sz_cap_icelake_k     // 2019: Intel AVX-512 VBMI2, VAES
	CapNEON        Capability = C.sz_cap_neon_k        // 2013: ARM NEON
	CapNEONAES     Capability = C.sz_cap_neonaes_k     // 2013: ARM NEON AES
	CapNEONSHA     Capability = C.sz_cap_neonsha_k     // 2013: ARM NEON SHA
	CapSVE         Capability = C.sz_cap_sve_k         // 2020: ARM SVE
	CapSVE2        Capability = C.sz_cap_sve2_k        // 2022: ARM SVE2
	CapSVE2AES     Capability = C.sz_cap_sve2aes_k     // 2022: ARM SVE2 AES
	CapRVV         Capability = C.sz_cap_rvv_k         // 2023: RISC-V Vector
	CapRVVCrypto   Capability = C.sz_cap_rvvcrypto_k   // RISC-V Vector Crypto
	CapV128        Capability = C.sz_cap_v128_k        // 2021: WASM SIMD128
	CapV128Relaxed Capability = C.sz_cap_v128relaxed_k // 2022: WASM Relaxed SIMD
	CapLoongsonASX Capability = C.sz_cap_loongsonasx_k // LoongArch LASX 256-bit SIMD
	CapPowerVSX    Capability = C.sz_cap_powervsx_k    // Power VSX 128-bit SIMD

	CapCUDA  Capability = C.sz_cap_cuda_k  // Any CUDA device
	CapROCm  Capability = C.sz_cap_rocm_k  // Any ROCm device
	CapMetal Capability = C.sz_cap_metal_k // Any Metal device

	CapCPUs Capability = C.sz_cap_cpus_k // Every CPU capability
	CapGPUs Capability = C.sz_cap_gpus_k // Every GPU capability
	CapAny  Capability = ^Capability(0)  // Every capability
)

// DeviceKind is the runtime a device belongs to, as the `sz_<kind>_*` C functions name it.
type DeviceKind int

// Device kinds, one per family of `sz_<kind>_*` C functions.
const (
	DeviceCPU DeviceKind = iota
	DeviceCUDA
	DeviceROCm
	DeviceMetal
)

// Device is one device StringZilla knows: the host CPU, or a GPU by its runtime's own ordinal, the
// one `cudaSetDevice` or `hipSetDevice` takes, or the position in Metal's device list. Every call
// of this package runs on the CPU, so a GPU device only reports its capabilities here.
type Device struct {
	Kind    DeviceKind
	Ordinal int
}

// CPU returns the host CPU, which every build has.
func CPU() Device { return Device{Kind: DeviceCPU} }

// CountDevices returns how many devices of kind the process sees: one CPU, or the GPUs its runtime
// counts, failing without one.
func CountDevices(kind DeviceKind) (int, error) {
	count := C.sz_size_t(1)
	var status C.sz_status_t = C.sz_success_k
	switch kind {
	case DeviceCPU:
	case DeviceCUDA:
		status = C.sz_cuda_count_devices(&count)
	case DeviceROCm:
		status = C.sz_rocm_count_devices(&count)
	case DeviceMetal:
		status = C.sz_metal_count_devices(&count)
	default:
		count, status = 0, C.sz_missing_gpu_k
	}
	return int(count), statusError(status)
}

// NewDevice returns device ordinal of kind, failing past the last one.
func NewDevice(kind DeviceKind, ordinal int) (Device, error) {
	count, err := CountDevices(kind)
	if err == nil && (ordinal < 0 || ordinal >= count) {
		err = statusError(C.sz_missing_gpu_k)
	}
	return Device{Kind: kind, Ordinal: ordinal}, err
}

// CapabilitiesDetected returns the capabilities d runs, whether or not they were compiled in.
func (d Device) CapabilitiesDetected() (Capability, error) {
	var capabilities C.sz_capability_t
	ordinal := C.sz_size_t(d.Ordinal)
	var status C.sz_status_t
	switch d.Kind {
	case DeviceCPU:
		status = C.sz_cpu_capabilities_detected(&capabilities)
	case DeviceCUDA:
		status = C.sz_cuda_capabilities_detected(ordinal, &capabilities)
	case DeviceROCm:
		status = C.sz_rocm_capabilities_detected(ordinal, &capabilities)
	case DeviceMetal:
		status = C.sz_metal_capabilities_detected(ordinal, &capabilities)
	default:
		status = C.sz_missing_gpu_k
	}
	return Capability(capabilities), statusError(status)
}

// CapabilitiesCompiled returns the capabilities whose kernels were compiled in for devices of d's
// kind, whether or not d runs them.
func (d Device) CapabilitiesCompiled() Capability {
	var capabilities C.sz_capability_t
	switch d.Kind {
	case DeviceCPU:
		C.sz_cpu_capabilities_compiled(&capabilities)
	case DeviceCUDA:
		C.sz_cuda_capabilities_compiled(&capabilities)
	case DeviceROCm:
		C.sz_rocm_capabilities_compiled(&capabilities)
	case DeviceMetal:
		C.sz_metal_capabilities_compiled(&capabilities)
	}
	return Capability(capabilities)
}

// CapabilitiesEnabled returns the mask for d's calls: [Device.CapabilitiesDetected] and
// [Device.CapabilitiesCompiled] at once. On the CPU it always has [CapSerial].
func (d Device) CapabilitiesEnabled() (Capability, error) {
	var capabilities C.sz_capability_t
	ordinal := C.sz_size_t(d.Ordinal)
	var status C.sz_status_t
	switch d.Kind {
	case DeviceCPU:
		status = C.sz_cpu_capabilities_enabled(&capabilities)
	case DeviceCUDA:
		status = C.sz_cuda_capabilities_enabled(ordinal, &capabilities)
	case DeviceROCm:
		status = C.sz_rocm_capabilities_enabled(ordinal, &capabilities)
	case DeviceMetal:
		status = C.sz_metal_capabilities_enabled(ordinal, &capabilities)
	default:
		status = C.sz_missing_gpu_k
	}
	return Capability(capabilities), statusError(status)
}

// ConfigureThread pins the goroutine to an OS thread, configures it for capabilities, usually the
// CPU's [Device.CapabilitiesEnabled], and returns the unlock function. Call it, typically via
// defer, once the work is done. GPUs have no thread state to configure.
func (d Device) ConfigureThread(capabilities Capability) (func(), error) {
	if d.Kind != DeviceCPU {
		return func() {}, statusError(C.sz_missing_kernel_k)
	}
	runtime.LockOSThread()
	return runtime.UnlockOSThread, statusError(C.sz_cpu_configure_thread(C.sz_capability_t(capabilities)))
}

// statusError names a failed call's status, or returns nil on success.
func statusError(status C.sz_status_t) error {
	if status == C.sz_success_k {
		return nil
	}
	return errors.New(C.GoString(C.sz_status_name(status)))
}

// check panics when a call without an error result reports a failure, as the package does on
// invalid inputs.
func check(status C.sz_status_t) {
	if err := statusError(status); err != nil {
		panic(err)
	}
}

// Has reports whether any bit of capability is in c.
func (c Capability) Has(capability Capability) bool { return c&capability != 0 }

// String names the capabilities in c, comma-separated, like "serial,haswell".
func (c Capability) String() string {
	var names [C.STRINGZILLA_CAPABILITIES_NAME_CAPACITY]C.char
	length := C.sz_capabilities_name(C.sz_capability_t(c), &names[0], C.STRINGZILLA_CAPABILITIES_NAME_CAPACITY)
	return C.GoStringN(&names[0], C.int(length))
}
