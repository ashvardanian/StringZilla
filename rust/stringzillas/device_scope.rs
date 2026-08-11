//! Execution context selecting the CPU or GPU hardware the engines run on.

use super::types::rust_error_from_c_message;
use super::*;
use core::ffi::{c_char, c_void};
use core::ptr;

/// Manages execution context and hardware resource allocation.
///
/// Auto-detects available hardware (CPU SIMD, GPU) and selects optimal implementations.
///
/// ```rust
/// use stringzilla::szs::DeviceScope;
/// let device = DeviceScope::default().unwrap();
/// let cpu_device = DeviceScope::cpu_cores(4).unwrap();
/// ```
pub struct DeviceScope {
    pub(crate) handle: *mut c_void,
}

impl DeviceScope {
    /// Create device scope with auto-detected optimal hardware configuration.
    pub fn default() -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_device_scope_init_default(&mut handle, &mut error_msg) };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Create a device scope for explicit CPU core count.
    ///
    /// Forces CPU-only execution with a specific number of threads. Useful for
    /// benchmarking, testing, or when you need predictable performance characteristics.
    ///
    /// # Parameters
    ///
    /// - `cpu_cores`: Number of CPU cores to use, or zero for all cores
    ///
    /// # Returns
    ///
    /// - `Ok(DeviceScope)`: Successfully created CPU device scope
    /// - `Err(Error)`: Invalid configuration or allocation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// // Create scope for 4 CPU threads
    /// let device = DeviceScope::cpu_cores(4).expect("Failed to create CPU scope");
    ///
    /// assert_eq!(device.get_cpu_cores().unwrap(), 4);
    /// assert!(!device.is_gpu());
    ///
    /// // Use for reproducible benchmarks
    /// let benchmark_device = DeviceScope::cpu_cores(8).unwrap();
    /// // ... run benchmark with consistent thread count
    /// ```
    ///
    /// # Performance
    ///
    /// - Optimal core count is usually equal to physical cores
    /// - Hyperthreading may not provide linear scaling for SIMD workloads
    /// - Consider NUMA topology for systems with >16 cores
    pub fn cpu_cores(cpu_cores: usize) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_device_scope_init_cpu_cores(cpu_cores, &mut handle, &mut error_msg) };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Create a device scope for a specific GPU device.
    ///
    /// Configures execution to use the specified GPU device. Requires CUDA or ROCm
    /// to be available and the device ID to be valid.
    ///
    /// # Parameters
    ///
    /// - `gpu_device`: GPU device index (0-based)
    ///
    /// # Returns
    ///
    /// - `Ok(DeviceScope)`: Successfully configured GPU device
    /// - `Err(Error)`: CUDA/ROCm unavailable, invalid device, or allocation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// // Try to use first GPU
    /// match DeviceScope::gpu_device(0) {
    ///     Ok(device) => {
    ///         println!("Using GPU device: {}", device.get_gpu_device().unwrap());
    ///         assert!(device.is_gpu());
    ///     }
    ///     Err(e) => println!("GPU not available: {:?}", e),
    /// }
    /// ```
    ///
    /// # GPU Selection Strategy
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// // Try multiple GPUs in order of preference
    /// let devices = [0, 1, 2];
    /// let gpu_device = devices
    ///     .iter()
    ///     .find_map(|&id| DeviceScope::gpu_device(id).ok())
    ///     .unwrap_or_else(|| DeviceScope::default().unwrap());
    /// ```
    ///
    /// # Performance
    ///
    /// - GPU is optimal for batch sizes >1000 string pairs
    /// - Memory transfer overhead affects small workloads
    /// - Use unified memory allocation for best GPU performance
    pub fn gpu_device(gpu_device: usize) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_device_scope_init_gpu_device(gpu_device, &mut handle, &mut error_msg) };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Get the hardware capabilities mask for this device scope.
    ///
    /// Returns a bitmask indicating available hardware features like SIMD instructions,
    /// GPU compute capabilities, and memory features. This can be used to verify
    /// that required features are available before creating engines.
    ///
    /// # Returns
    ///
    /// - `Ok(Capability)`: Hardware capabilities bitmask
    /// - `Err(Error)`: Failed to query capabilities
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// let device = DeviceScope::default().unwrap();
    /// let caps = device.get_capabilities().unwrap();
    ///
    /// // Check specific capabilities (values depend on sz_cap_* constants)
    /// println!("Capabilities: 0x{:x}", caps);
    /// if caps & 0x1 != 0 { println!("Basic SIMD available"); }
    /// if caps & 0x2 != 0 { println!("Advanced SIMD available"); }
    /// ```
    pub fn get_capabilities(&self) -> Result<Capability, Error> {
        let mut capabilities: Capability = 0;
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_device_scope_get_capabilities(self.handle, &mut capabilities, &mut error_msg) };
        match status {
            Status::Success => Ok(capabilities),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Get the number of CPU cores configured for this device scope.
    ///
    /// Returns the number of CPU threads that will be used for parallel execution.
    /// For GPU device scopes, this may return 0 or a fallback CPU count.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)`: Number of configured CPU cores
    /// - `Err(Error)`: Failed to query configuration
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// let device = DeviceScope::cpu_cores(8).unwrap();
    /// assert_eq!(device.get_cpu_cores().unwrap(), 8);
    ///
    /// // Default scope may use different count
    /// let default_device = DeviceScope::default().unwrap();
    /// let cores = default_device.get_cpu_cores().unwrap();
    /// println!("Default device using {} CPU cores", cores);
    /// ```
    pub fn get_cpu_cores(&self) -> Result<usize, Error> {
        let mut cpu_cores: usize = 0;
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_device_scope_get_cpu_cores(self.handle, &mut cpu_cores, &mut error_msg) };
        match status {
            Status::Success => Ok(cpu_cores),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Get the GPU device ID configured for this device scope.
    ///
    /// Returns the GPU device index if this scope is configured for GPU execution.
    /// For CPU-only device scopes, this will return an error.
    ///
    /// # Returns
    ///
    /// - `Ok(usize)`: GPU device index (0-based)
    /// - `Err(Status::Unknown)`: Not configured for GPU or GPU unavailable
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// // GPU device scope
    /// if let Ok(gpu_device) = DeviceScope::gpu_device(1) {
    ///     assert_eq!(gpu_device.get_gpu_device().unwrap(), 1);
    ///     assert!(gpu_device.is_gpu());
    /// }
    ///
    /// // CPU device scope
    /// let cpu_device = DeviceScope::cpu_cores(4).unwrap();
    /// assert!(cpu_device.get_gpu_device().is_err());
    /// assert!(!cpu_device.is_gpu());
    /// ```
    pub fn get_gpu_device(&self) -> Result<usize, Error> {
        let mut gpu_device: usize = 0;
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_device_scope_get_gpu_device(self.handle, &mut gpu_device, &mut error_msg) };
        match status {
            Status::Success => Ok(gpu_device),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Check if this device scope is configured for GPU execution.
    ///
    /// This is a convenience method that checks whether `get_gpu_device()` would succeed.
    /// Use this to branch between GPU and CPU code paths.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::DeviceScope;
    /// let device = DeviceScope::default().unwrap();
    ///
    /// if device.is_gpu() {
    ///     println!("GPU acceleration available on device {}",
    ///              device.get_gpu_device().unwrap());
    /// } else {
    ///     println!("Using CPU with {} cores",
    ///              device.get_cpu_cores().unwrap());
    /// }
    /// ```
    pub fn is_gpu(&self) -> bool {
        self.get_gpu_device().is_ok()
    }
}

impl Drop for DeviceScope {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_device_scope_free(self.handle) };
        }
    }
}

unsafe impl Send for DeviceScope {}
unsafe impl Sync for DeviceScope {}

// C API bindings
extern "C" {

    // Device scope functions
    fn szs_device_scope_init_default(scope: *mut *mut c_void, error_message: *mut *const c_char) -> Status;
    fn szs_device_scope_init_cpu_cores(
        cpu_cores: usize,
        scope: *mut *mut c_void,
        error_message: *mut *const c_char,
    ) -> Status;
    fn szs_device_scope_init_gpu_device(
        gpu_device: usize,
        scope: *mut *mut c_void,
        error_message: *mut *const c_char,
    ) -> Status;
    fn szs_device_scope_get_capabilities(
        scope: *mut c_void,
        capabilities: *mut Capability,
        error_message: *mut *const c_char,
    ) -> Status;
    fn szs_device_scope_get_cpu_cores(
        scope: *mut c_void,
        cpu_cores: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;
    fn szs_device_scope_get_gpu_device(
        scope: *mut c_void,
        gpu_device: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;
    fn szs_device_scope_free(scope: *mut c_void);

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_scope_creation() {
        // Test default device scope
        let default_device = DeviceScope::default();
        match default_device {
            Ok(device) => {
                // Test capability query
                let _caps = device.get_capabilities();
                println!("Default device capabilities: {:?}", _caps);
            }
            Err(e) => println!("Default device creation failed: {:?}", e),
        }

        // Test CPU device scope with valid core count
        let cpu_device = DeviceScope::cpu_cores(4);
        match cpu_device {
            Ok(device) => {
                assert!(!device.is_gpu());
                if let Ok(cores) = device.get_cpu_cores() {
                    assert_eq!(cores, 4);
                }
            }
            Err(e) => println!("CPU device creation failed: {:?}", e),
        }

        // Test GPU device scope (may fail if no GPU)
        let gpu_device = DeviceScope::gpu_device(0);
        match gpu_device {
            Ok(device) => {
                assert!(device.is_gpu());
                if let Ok(gpu_id) = device.get_gpu_device() {
                    assert_eq!(gpu_id, 0);
                }
            }
            Err(e) => println!("GPU device creation failed (expected if no GPU): {:?}", e),
        }
    }

    #[test]
    fn device_scope_validation() {
        // Test valid CPU core count - 0 means use all cores
        let all_cores = DeviceScope::cpu_cores(0);
        assert!(all_cores.is_ok(), "CPU cores 0 should mean all cores");

        // Test single core - valid, redirects to default
        let single_core = DeviceScope::cpu_cores(1);
        assert!(single_core.is_ok(), "Single core should be valid");

        // Test multiple cores
        let multi_cores = DeviceScope::cpu_cores(4);
        assert!(multi_cores.is_ok(), "Multiple cores should be valid");
    }

    #[test]
    fn device_scope_invalid_gpu_id_does_not_panic() {
        // An out-of-range GPU id must return an Err, never panic - it may legitimately succeed or
        // fail depending on how many GPUs the machine has, so only the "no panic" contract is checked.
        match DeviceScope::gpu_device(999) {
            Ok(_) => println!("GPU device 999 unexpectedly available"),
            Err(e) => println!("GPU device 999 correctly failed: {:?}", e),
        }
    }
}
