extern crate alloc;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void, CStr};
use core::ptr;

use allocator_api2::{alloc::AllocError, alloc::Allocator, alloc::Layout};
use stringtape::{BytesTape, BytesTapeView, StringTape, StringTapeView};

// Re-export common types from stringzilla
pub use crate::stringzilla::{SortedIdx, Status as SzStatus};

/// Capability flags
pub type Capability = u32;

// Import from stringzilla module
pub use crate::stringzilla::Status;

/// Custom error type that preserves detailed error messages from the C API.
#[derive(Debug)]
pub struct Error {
    pub status: Status,
    pub message: Option<String>,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.message {
            Some(msg) => write!(f, "{}", msg),
            None => write!(f, "{:?}", self.status),
        }
    }
}

impl std::error::Error for Error {}

impl From<Status> for Error {
    fn from(status: Status) -> Self {
        Error { status, message: None }
    }
}

fn rust_error_from_c_message(status: Status, error_msg: *const c_char) -> Error {
    let message = if !error_msg.is_null() && status != Status::Success {
        unsafe { CStr::from_ptr(error_msg).to_str().ok().map(|s| s.to_string()) }
    } else {
        None
    };

    Error { status, message }
}

/// Tape variant that can hold either 32-bit or 64-bit string tapes with unsigned offsets
pub enum AnyCharsTape<'a> {
    Tape32(StringTape<u32, UnifiedAlloc>),
    Tape64(StringTape<u64, UnifiedAlloc>),
    // Zero-copy FFI views (UTF-8)
    View32(StringTapeView<'a, u32>),
    View64(StringTapeView<'a, u64>),
}

/// Tape variant that can hold either 32-bit or 64-bit byte tapes with unsigned offsets
pub enum AnyBytesTape<'a> {
    Tape32(BytesTape<u32, UnifiedAlloc>),
    Tape64(BytesTape<u64, UnifiedAlloc>),
    // Zero-copy FFI views (bytes)
    View32(BytesTapeView<'a, u32>),
    View64(BytesTapeView<'a, u64>),
}

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
    handle: *mut c_void,
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

/// Internal representation of `sz_sequence_t` for passing to C
#[repr(C)]
struct SzSequence {
    handle: *mut c_void,
    count: usize,
    get_start: extern "C" fn(*mut c_void, usize) -> *const u8,
    get_length: extern "C" fn(*mut c_void, usize) -> usize,
    // Additional fields for our implementation
    starts: *const *const u8,
    lengths: *const usize,
}

/// Raw C API tape structure for 32-bit offsets (data < 4GB),
/// matching `sz_sequence_u32tape_t` in the C API
#[repr(C)]
#[derive(Copy, Clone)]
struct SzSequenceU32Tape {
    data: *const u8,
    offsets: *const u32,
    count: usize,
}

/// Raw C API tape structure for 64-bit offsets (data >= 4GB),
/// matching `sz_sequence_u64tape_t` in the C API
#[repr(C)]
#[derive(Copy, Clone)]
struct SzSequenceU64Tape {
    data: *const u8,
    offsets: *const u64,
    count: usize,
}

// Conversions from tape containers to FFI views
impl From<&BytesTape<u32, UnifiedAlloc>> for SzSequenceU32Tape {
    fn from(tape: &BytesTape<u32, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU32Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

impl From<&StringTape<u32, UnifiedAlloc>> for SzSequenceU32Tape {
    fn from(tape: &StringTape<u32, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU32Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

impl From<&BytesTape<u64, UnifiedAlloc>> for SzSequenceU64Tape {
    fn from(tape: &BytesTape<u64, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU64Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

impl From<&StringTape<u64, UnifiedAlloc>> for SzSequenceU64Tape {
    fn from(tape: &StringTape<u64, UnifiedAlloc>) -> Self {
        let parts = tape.as_raw_parts();
        SzSequenceU64Tape {
            data: parts.data_ptr,
            offsets: parts.offsets_ptr,
            count: parts.items_count,
        }
    }
}

// Conversions from stringtape views to FFI views
impl<'a> From<BytesTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: BytesTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<BytesTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: BytesTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&BytesTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: &BytesTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&BytesTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: &BytesTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<StringTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: StringTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<StringTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: StringTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&StringTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: &StringTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&StringTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: &StringTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

/// Generic callback to get start of string at index for byte slices
extern "C" fn sz_sequence_get_start_generic<T: AsRef<[u8]>>(handle: *mut c_void, index: usize) -> *const u8 {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
        strings[index].as_ref().as_ptr()
    }
}

/// Generic callback to get length of string at index for byte slices
extern "C" fn sz_sequence_get_length_generic<T: AsRef<[u8]>>(handle: *mut c_void, index: usize) -> usize {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
        strings[index].as_ref().len()
    }
}

/// Generic callback to get start of string at index for string slices
extern "C" fn sz_sequence_get_start_str<T: AsRef<str>>(handle: *mut c_void, index: usize) -> *const u8 {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
        strings[index].as_ref().as_bytes().as_ptr()
    }
}

/// Generic callback to get length of string at index for string slices
extern "C" fn sz_sequence_get_length_str<T: AsRef<str>>(handle: *mut c_void, index: usize) -> usize {
    unsafe {
        let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
        strings[index].as_ref().as_bytes().len()
    }
}

/// Trait for types that can be converted to SzSequence for byte sequences
trait SzSequenceFromBytes {
    fn to_sz_sequence(&self) -> SzSequence;
}

impl<T: AsRef<[u8]>> SzSequenceFromBytes for [T] {
    fn to_sz_sequence(&self) -> SzSequence {
        SzSequence {
            handle: self.as_ptr() as *mut c_void,
            count: self.len(),
            get_start: sz_sequence_get_start_generic::<T>,
            get_length: sz_sequence_get_length_generic::<T>,
            starts: ptr::null(),
            lengths: ptr::null(),
        }
    }
}

/// Trait for types that can be converted to SzSequence for string sequences
trait SzSequenceFromChars {
    fn to_sz_sequence(&self) -> SzSequence;
}

impl<T: AsRef<str>> SzSequenceFromChars for [T] {
    fn to_sz_sequence(&self) -> SzSequence {
        SzSequence {
            handle: self.as_ptr() as *mut c_void,
            count: self.len(),
            get_start: sz_sequence_get_start_str::<T>,
            get_length: sz_sequence_get_length_str::<T>,
            starts: ptr::null(),
            lengths: ptr::null(),
        }
    }
}

/// Opaque handles for similarity engines
pub type FingerprintsHandle = *mut c_void;
pub type LevenshteinDistancesHandle = *mut c_void;
pub type LevenshteinDistancesUtf8Handle = *mut c_void;
pub type NeedlemanWunschScoresHandle = *mut c_void;
pub type SmithWatermanScoresHandle = *mut c_void;

// C API bindings
extern "C" {

    // Metadata functions
    fn szs_version_major() -> i32;
    fn szs_version_minor() -> i32;
    fn szs_version_patch() -> i32;
    fn szs_capabilities() -> u32;

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

    // Levenshtein distance functions
    fn szs_levenshtein_distances_init(
        match_cost: i8,
        mismatch_cost: i8,
        open_cost: i8,
        extend_cost: i8,
        alloc: *const c_void,
        capabilities: Capability,
        engine: *mut LevenshteinDistancesHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_sequence(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_t
        b: *const c_void, // sz_sequence_t
        results: *mut usize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_u32tape(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut usize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_u64tape(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
        results: *mut usize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_free(engine: LevenshteinDistancesHandle);

    // Levenshtein distance UTF-8 functions
    fn szs_levenshtein_distances_utf8_init(
        match_cost: i8,
        mismatch_cost: i8,
        open_cost: i8,
        extend_cost: i8,
        alloc: *const c_void,
        capabilities: Capability,
        engine: *mut LevenshteinDistancesUtf8Handle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_sequence(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_t
        b: *const c_void, // sz_sequence_t
        results: *mut usize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_u32tape(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut usize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_u64tape(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
        results: *mut usize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_free(engine: LevenshteinDistancesUtf8Handle);

    // Needleman-Wunsch scoring functions
    fn szs_needleman_wunsch_scores_init(
        subs: *const i8, // 256x256 substitution matrix
        open_cost: i8,
        extend_cost: i8,
        alloc: *const c_void,
        capabilities: Capability,
        engine: *mut NeedlemanWunschScoresHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_sequence(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_t
        b: *const c_void, // sz_sequence_t
        results: *mut isize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_u32tape(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut isize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_u64tape(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
        results: *mut isize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_free(engine: NeedlemanWunschScoresHandle);

    // Smith-Waterman scoring functions
    fn szs_smith_waterman_scores_init(
        subs: *const i8, // 256x256 substitution matrix
        open_cost: i8,
        extend_cost: i8,
        alloc: *const c_void,
        capabilities: Capability,
        engine: *mut SmithWatermanScoresHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_sequence(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_t
        b: *const c_void, // sz_sequence_t
        results: *mut isize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_u32tape(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut isize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_u64tape(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
        results: *mut isize,
        results_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_free(engine: SmithWatermanScoresHandle);

    // Fingerprinting functions
    fn szs_fingerprints_init(
        dimensions: usize,
        alphabet_size: usize,
        window_widths: *const usize,
        window_widths_count: usize,
        alloc: *const c_void, // MemoryAllocator - using null for default
        capabilities: Capability,
        engine: *mut FingerprintsHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_fingerprints_sequence(
        engine: FingerprintsHandle,
        device: *mut c_void,  // DeviceScope
        texts: *const c_void, // sz_sequence_t
        min_hashes: *mut u32,
        min_hashes_stride: usize,
        min_counts: *mut u32,
        min_counts_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_fingerprints_u32tape(
        engine: FingerprintsHandle,
        device: *mut c_void,  // DeviceScope
        texts: *const c_void, // sz_sequence_u32tape_t
        min_hashes: *mut u32,
        min_hashes_stride: usize,
        min_counts: *mut u32,
        min_counts_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_fingerprints_u64tape(
        engine: FingerprintsHandle,
        device: *mut c_void,  // DeviceScope
        texts: *const c_void, // sz_sequence_u64tape_t
        min_hashes: *mut u32,
        min_hashes_stride: usize,
        min_counts: *mut u32,
        min_counts_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_fingerprints_free(engine: FingerprintsHandle);

    // Unified allocator functions
    fn szs_unified_alloc(size_bytes: usize) -> *mut c_void;
    fn szs_unified_free(ptr: *mut c_void, size_bytes: usize);

}

/// Unified memory allocator that uses CUDA unified memory when available,
/// falls back to malloc otherwise. Works with allocator-api2.
pub struct UnifiedAlloc;

unsafe impl Allocator for UnifiedAlloc {
    fn allocate(&self, layout: Layout) -> Result<core::ptr::NonNull<[u8]>, AllocError> {
        let size = layout.size();
        if size == 0 {
            // For zero-sized allocations, return a properly aligned non-null dangling pointer
            let ptr = core::ptr::NonNull::new(layout.align() as *mut u8).ok_or(AllocError)?;
            return Ok(core::ptr::NonNull::slice_from_raw_parts(ptr, 0));
        }

        let ptr = unsafe { szs_unified_alloc(size) };
        if ptr.is_null() {
            return Err(AllocError);
        }

        let ptr = core::ptr::NonNull::new(ptr as *mut u8).ok_or(AllocError)?;
        Ok(core::ptr::NonNull::slice_from_raw_parts(ptr, size))
    }

    unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            szs_unified_free(ptr.as_ptr() as *mut c_void, layout.size());
        }
    }
}

/// Type alias for Vec with unified allocator
pub type UnifiedVec<T> = allocator_api2::vec::Vec<T, UnifiedAlloc>;

/// Returns StringZillas similarity engine version information.
pub fn version() -> crate::stringzilla::SemVer {
    crate::stringzilla::SemVer {
        major: unsafe { szs_version_major() },
        minor: unsafe { szs_version_minor() },
        patch: unsafe { szs_version_patch() },
    }
}

/// Copies the capabilities C-string into a fixed buffer and returns it.
/// The returned SmallCString is guaranteed to be null-terminated.
pub fn capabilities() -> crate::stringzilla::SmallCString {
    let caps = unsafe { szs_capabilities() };
    crate::stringzilla::capabilities_from_enum(caps)
}

/// Levenshtein distance engine for batch processing of binary sequences.
///
/// Computes edit distances between byte sequence pairs using configurable gap costs.
/// Optimized for processing large batches in parallel.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
///
/// let strings_a = vec!["kitten", "saturday"];
/// let strings_b = vec!["sitting", "sunday"];
/// let distances = engine.compute(&device, &strings_a, &strings_b).unwrap();
/// assert_eq!(&distances[..], &[3, 3]);
/// ```
pub struct LevenshteinDistances {
    handle: LevenshteinDistancesHandle,
}

impl LevenshteinDistances {
    /// Create a new Levenshtein distances engine with specified costs.
    ///
    /// # Parameters
    /// - `match_cost`: Cost when characters match (typically ≤ 0)
    /// - `mismatch_cost`: Cost when characters differ (typically > 0)  
    /// - `open_cost`: Cost to open a gap (insertion/deletion)
    /// - `extend_cost`: Cost to extend existing gap (usually ≤ open_cost)
    pub fn new(
        device: &DeviceScope,
        match_cost: i8,
        mismatch_cost: i8,
        open_cost: i8,
        extend_cost: i8,
    ) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_levenshtein_distances_init(
                match_cost,
                mismatch_cost,
                open_cost,
                extend_cost,
                ptr::null(),
                capabilities,
                &mut handle,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute Levenshtein distances between sequence pairs.
    ///
    /// Processes pairs of sequences in parallel, computing edit distance for each pair.
    /// The function automatically handles both CPU SIMD and GPU acceleration based
    /// on the device scope configuration.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution
    /// - `sequences_a`: First collection of sequences
    /// - `sequences_b`: Second collection of sequences  
    ///
    /// # Returns
    ///
    /// - `Ok(UnifiedVec<usize>)`: Vector of distances, one per sequence pair
    /// - `Err(Error)`: Computation failed
    ///
    /// # Behavior
    ///
    /// - Pairs sequences by index: (a[0], b[0]), (a[1], b[1]), etc.
    /// - Result length equals `min(sequences_a.len(), sequences_b.len())`
    /// - Uses unified memory allocation for GPU compatibility
    /// - Empty sequences are handled correctly (distance equals other sequence length)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// // Basic usage
    /// let words_a = vec!["cat", "dog", "bird"];
    /// let words_b = vec!["bat", "fog", "word"];
    /// let distances = engine.compute(&device, &words_a, &words_b).unwrap();
    ///
    /// assert_eq!(distances.len(), 3);
    /// println!("Distances: {:?}", distances); // [1, 1, 3]
    /// ```
    ///
    /// # Performance
    ///
    /// - CPU performance scales with SIMD width and core count
    /// - GPU optimal for batches >1000 pairs with medium-length sequences
    /// - Memory layout optimized for cache efficiency
    /// - Consider sequence length distribution for optimal performance
    pub fn compute<T, S>(
        &self,
        device: &DeviceScope,
        sequences_a: T,
        sequences_b: T,
    ) -> Result<UnifiedVec<usize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let seq_a_slice = sequences_a.as_ref();
        let seq_b_slice = sequences_b.as_ref();
        let num_pairs = seq_a_slice.len().min(seq_b_slice.len());

        let mut results = UnifiedVec::with_capacity_in(num_pairs, UnifiedAlloc);
        results.resize(num_pairs, 0);

        let results_stride = core::mem::size_of::<usize>();

        if device.is_gpu() {
            let force_64bit = should_use_64bit_for_bytes(seq_a_slice, seq_b_slice);
            let tape_a = copy_bytes_into_tape(seq_a_slice, force_64bit)?;
            let tape_b = copy_bytes_into_tape(seq_b_slice, force_64bit)?;

            // Forward to the in-place variant to avoid code duplication
            self.compute_into(device, tape_a, tape_b, &mut results)?;
            Ok(results)
        } else {
            let seq_a = SzSequenceFromBytes::to_sz_sequence(seq_a_slice);
            let seq_b = SzSequenceFromBytes::to_sz_sequence(seq_b_slice);
            let mut error_msg: *const c_char = ptr::null();
            let status = unsafe {
                szs_levenshtein_distances_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            match status {
                Status::Success => Ok(results),
                err => Err(rust_error_from_c_message(err, error_msg)),
            }
        }
    }

    /// Compute Levenshtein distances into an existing results buffer.
    ///
    /// - Accepts `AnyBytesTape<'_>` for both sides: either owned `BytesTape` or `BytesTapeView`.
    /// - Supports 32-bit or 64-bit offsets; both inputs must use the same width.
    /// - Writes distances into `results` without reallocating.
    ///
    /// Requirements
    /// - `results.len() >= min(a.len(), b.len())`
    /// - For GPU devices, inputs should be allocated in unified/device memory.
    ///
    /// Errors
    /// - `UnexpectedDimensions` if buffer is too small or widths are mixed.
    /// - Underlying engine errors forwarded from the FFI.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        a: AnyBytesTape<'a>,
        b: AnyBytesTape<'a>,
        results: &mut UnifiedVec<usize>,
    ) -> Result<(), Error> {
        // Convert to FFI views and validate matching offset widths
        let mut error_msg: *const c_char = ptr::null();
        let results_stride = core::mem::size_of::<usize>();

        // Convert both inputs to 64-bit views if possible, else to 32-bit views.
        let a64 = match &a {
            AnyBytesTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyBytesTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        let b64 = match &b {
            AnyBytesTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyBytesTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a64, b64) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_levenshtein_distances_u64tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let a32 = match &a {
            AnyBytesTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyBytesTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        let b32 = match &b {
            AnyBytesTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyBytesTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a32, b32) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_levenshtein_distances_u32tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        // Mixed widths are unsupported to avoid implicit widening and extra copies
        Err(Error::from(SzStatus::UnexpectedDimensions))
    }
}

impl Drop for LevenshteinDistances {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_levenshtein_distances_free(self.handle) };
        }
    }
}

unsafe impl Send for LevenshteinDistances {}
unsafe impl Sync for LevenshteinDistances {}

/// UTF-8 aware Levenshtein distance engine for Unicode text processing.
///
/// Computes edit distances at the character level, properly handling multi-byte
/// UTF-8 sequences. Use for international text, emoji, or when character boundaries matter.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, LevenshteinDistancesUtf8};
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
///
/// let strings_a = vec!["café", "🦀 rust"];
/// let strings_b = vec!["cafe", "🔥 rust"];
/// let distances = engine.compute(&device, &strings_a, &strings_b).unwrap();
/// assert_eq!(&distances[..], &[1, 1]); // Character-level edits
/// ```
pub struct LevenshteinDistancesUtf8 {
    handle: LevenshteinDistancesUtf8Handle,
}

impl LevenshteinDistancesUtf8 {
    /// Create a new UTF-8 aware Levenshtein distances engine.
    ///
    /// Initializes an engine that processes UTF-8 strings at the character level,
    /// properly handling multi-byte Unicode sequences. Essential for international
    /// text processing and semantic correctness.
    ///
    /// # Parameters
    ///
    /// Same as binary engine, but costs apply to Unicode code points:
    /// - `match_cost`: Cost when Unicode characters match
    /// - `mismatch_cost`: Cost when Unicode characters differ
    /// - `open_cost`: Cost to insert/delete a Unicode character
    /// - `extend_cost`: Cost to continue insertion/deletion
    ///
    /// # Returns
    ///
    /// - `Ok(LevenshteinDistancesUtf8)`: Successfully initialized engine
    /// - `Err(Error)`: Invalid cost configuration or allocation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, LevenshteinDistancesUtf8};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Standard Unicode-aware engine
    /// let engine = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// // Test with international text
    /// let greetings_a = vec!["Hello", "Bonjour", "こんにちは"];
    /// let greetings_b = vec!["Hallo", "Bonjoir", "こんばんは"];
    /// let distances = engine.compute(&device, &greetings_a, &greetings_b).unwrap();
    /// ```
    pub fn new(
        device: &DeviceScope,
        match_cost: i8,
        mismatch_cost: i8,
        open_cost: i8,
        extend_cost: i8,
    ) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_levenshtein_distances_utf8_init(
                match_cost,
                mismatch_cost,
                open_cost,
                extend_cost,
                ptr::null(),
                capabilities,
                &mut handle,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute UTF-8 aware Levenshtein distances between string pairs.
    ///
    /// Processes Unicode strings character by character, ensuring proper handling
    /// of multi-byte UTF-8 sequences. Critical for applications requiring semantic
    /// correctness with international text.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, LevenshteinDistancesUtf8};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// // Unicode strings (same container type for both sides)
    /// let strings_a: Vec<String> = vec!["résumé".to_string(), "naïve".to_string()];
    /// let strings_b: Vec<String> = vec!["resume".to_string(), "naive".to_string()];
    /// let distances = engine.compute(&device, &strings_a, &strings_b).unwrap();
    ///
    /// // Each accented character counts as 1 edit
    /// assert_eq!(distances[0], 2); // é->e, é->e
    /// assert_eq!(distances[1], 1); // ï->i
    /// ```
    ///
    /// # Unicode Normalization
    ///
    /// Note: This engine does NOT perform Unicode normalization. Pre-normalize
    /// your strings if you need to handle composed vs decomposed characters:
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, LevenshteinDistancesUtf8};
    /// // These are different at the code point level:
    /// let composed = vec!["café"];     // é as single code point U+00E9
    /// let decomposed = vec!["cafe\u{0301}"]; // e + combining acute accent
    ///
    /// // Distance would be non-zero without normalization
    /// // Use unicode-normalization crate if needed
    /// ```
    pub fn compute<T, S>(
        &self,
        device: &DeviceScope,
        sequences_a: T,
        sequences_b: T,
    ) -> Result<UnifiedVec<usize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<str>,
    {
        let seq_a_slice = sequences_a.as_ref();
        let seq_b_slice = sequences_b.as_ref();
        let num_pairs = seq_a_slice.len().min(seq_b_slice.len());

        let mut results = UnifiedVec::with_capacity_in(num_pairs, UnifiedAlloc);
        results.resize(num_pairs, 0);

        let results_stride = core::mem::size_of::<usize>();

        if device.is_gpu() {
            let force_64bit = should_use_64bit_for_strings(seq_a_slice, seq_b_slice);
            let tape_a = copy_chars_into_tape(seq_a_slice, force_64bit)?;
            let tape_b = copy_chars_into_tape(seq_b_slice, force_64bit)?;
            self.compute_into(device, tape_a, tape_b, &mut results)?;
            Ok(results)
        } else {
            let seq_a = SzSequenceFromChars::to_sz_sequence(seq_a_slice);
            let seq_b = SzSequenceFromChars::to_sz_sequence(seq_b_slice);
            let mut error_msg: *const c_char = ptr::null();
            let status = unsafe {
                szs_levenshtein_distances_utf8_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            match status {
                Status::Success => Ok(results),
                err => Err(rust_error_from_c_message(err, error_msg)),
            }
        }
    }

    /// Compute UTF-8 Levenshtein distances into an existing results buffer.
    ///
    /// - Accepts `AnyCharsTape<'_>` for both sides: `StringTape` or `StringTapeView`.
    /// - Supports 32-bit or 64-bit offsets; both inputs must use the same width.
    /// - No result allocations; writes into `results`.
    ///
    /// Requirements and errors are the same as the bytes variant.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        a: AnyCharsTape<'a>,
        b: AnyCharsTape<'a>,
        results: &mut UnifiedVec<usize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();
        let results_stride = core::mem::size_of::<usize>();

        // Try 64-bit first
        let a64 = match &a {
            AnyCharsTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyCharsTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        let b64 = match &b {
            AnyCharsTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyCharsTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a64, b64) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_levenshtein_distances_utf8_u64tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        // Then 32-bit
        let a32 = match &a {
            AnyCharsTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyCharsTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        let b32 = match &b {
            AnyCharsTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyCharsTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a32, b32) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_levenshtein_distances_utf8_u32tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        Err(Error::from(SzStatus::UnexpectedDimensions))
    }
}

impl Drop for LevenshteinDistancesUtf8 {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_levenshtein_distances_utf8_free(self.handle) };
        }
    }
}

unsafe impl Send for LevenshteinDistancesUtf8 {}
unsafe impl Sync for LevenshteinDistancesUtf8 {}

/// Needleman-Wunsch global sequence alignment scoring engine.
///
/// Finds optimal global alignments using a substitution matrix and gap penalties.
/// Returns alignment scores rather than distances.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, NeedlemanWunschScores};
/// // Create scoring matrix (match=2, mismatch=-1)
/// let mut matrix = [[-1i8; 256]; 256];
/// for i in 0..256 {
///     matrix[i][i] = 2;
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let engine = NeedlemanWunschScores::new(&device, &matrix, -2, -1).unwrap();
///
/// let seq_a = vec!["ACGT"];
/// let seq_b = vec!["AGCT"];
/// let scores = engine.compute(&device, &seq_a, &seq_b).unwrap();
/// ```
pub struct NeedlemanWunschScores {
    handle: NeedlemanWunschScoresHandle,
}

impl NeedlemanWunschScores {
    /// Create a new Needleman-Wunsch global alignment scoring engine.
    ///
    /// # Parameters
    /// - `substitution_matrix`: 256x256 matrix of alignment scores
    /// - `open_cost`: Penalty for opening a gap (typically negative)
    /// - `extend_cost`: Penalty for extending a gap (typically negative, ≤ open_cost)
    pub fn new(
        device: &DeviceScope,
        substitution_matrix: &[[i8; 256]; 256],
        open_cost: i8,
        extend_cost: i8,
    ) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_needleman_wunsch_scores_init(
                substitution_matrix.as_ptr() as *const i8,
                open_cost,
                extend_cost,
                ptr::null(),
                capabilities,
                &mut handle,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute Needleman-Wunsch global alignment scores between sequence pairs.
    ///
    /// Finds the optimal global alignment score for each pair of sequences using
    /// the configured substitution matrix and gap penalties. Returns positive scores
    /// for good alignments, negative for poor alignments.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for parallel execution
    /// - `sequences_a`: First collection of sequences to align
    /// - `sequences_b`: Second collection of sequences to align
    ///
    /// # Returns
    ///
    /// - `Ok(UnifiedVec<isize>)`: Vector of alignment scores (can be negative)
    /// - `Err(Status)`: Computation failed
    ///
    /// # Score Interpretation
    ///
    /// - **Positive scores**: Good alignment, sequences are similar
    /// - **Zero scores**: Neutral alignment
    /// - **Negative scores**: Poor alignment, sequences are dissimilar
    /// - **Magnitude**: Higher absolute values indicate stronger alignment quality
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, NeedlemanWunschScores};
    /// # let mut matrix = [[0i8; 256]; 256];
    /// # for i in 0..256 { matrix[i][i] = 2; for j in 0..256 { if i != j { matrix[i][j] = -1; } } }
    /// let device = DeviceScope::default().unwrap();
    /// let engine = NeedlemanWunschScores::new(&device, &matrix, -2, -1).unwrap();
    ///
    /// // Compare DNA sequences
    /// let dna_a = vec!["ATCGATCG", "GGCCTTAA"];
    /// let dna_b = vec!["ATCGATCC", "GGCCTTAA"]; // One mismatch, one exact
    /// let scores = engine.compute(&device, &dna_a, &dna_b).unwrap();
    ///
    /// println!("DNA alignment scores: {:?}", scores);
    /// // Expect: [positive but lower for mismatch, high positive for exact match]
    /// ```
    ///
    /// # Batch Processing
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, NeedlemanWunschScores};
    /// # let mut matrix = [[0i8; 256]; 256];
    /// # let device = DeviceScope::default().unwrap();
    /// # let engine = NeedlemanWunschScores::new(&device, &matrix, -2, -1).unwrap();
    /// // Process large batches efficiently
    /// let sequences: Vec<&str> = vec![
    ///     "PROTEIN_SEQUENCE_1", "PROTEIN_SEQUENCE_2", /* ... */
    /// ];
    /// let references: Vec<&str> = vec![
    ///     "REFERENCE_SEQ_1", "REFERENCE_SEQ_2", /* ... */
    /// ];
    ///
    /// let scores = engine.compute(&device, &sequences, &references).unwrap();
    ///
    /// // Find best alignments
    /// let best_idx = scores.iter().enumerate()
    ///     .max_by_key(|(_, &score)| score)
    ///     .map(|(idx, _)| idx);
    /// ```
    pub fn compute<T, S>(
        &self,
        device: &DeviceScope,
        sequences_a: T,
        sequences_b: T,
    ) -> Result<UnifiedVec<isize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let seq_a_slice = sequences_a.as_ref();
        let seq_b_slice = sequences_b.as_ref();
        let num_pairs = seq_a_slice.len().min(seq_b_slice.len());

        let mut results = UnifiedVec::with_capacity_in(num_pairs, UnifiedAlloc);
        results.resize(num_pairs, 0);

        let results_stride = core::mem::size_of::<isize>();

        if device.is_gpu() {
            let force_64bit = should_use_64bit_for_bytes(seq_a_slice, seq_b_slice);
            let tape_a = copy_bytes_into_tape(seq_a_slice, force_64bit)?;
            let tape_b = copy_bytes_into_tape(seq_b_slice, force_64bit)?;
            self.compute_into(device, tape_a, tape_b, &mut results)?;
            Ok(results)
        } else {
            let seq_a = SzSequenceFromBytes::to_sz_sequence(seq_a_slice);
            let seq_b = SzSequenceFromBytes::to_sz_sequence(seq_b_slice);
            let mut error_msg: *const c_char = ptr::null();
            let status = unsafe {
                szs_needleman_wunsch_scores_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            match status {
                Status::Success => Ok(results),
                err => Err(rust_error_from_c_message(err, error_msg)),
            }
        }
    }

    /// Compute Needleman–Wunsch scores into an existing results buffer.
    ///
    /// - Accepts `AnyBytesTape<'_>` inputs (owned tapes or views), with matching offset widths.
    /// - Writes scores into `results` without allocating. On GPU, inputs should be device-accessible.
    /// - Errors if `results.len()` is insufficient or widths are mixed.
    /// Compute Smith–Waterman scores into an existing results buffer.
    ///
    /// - Accepts `AnyBytesTape<'_>` inputs (owned tapes or views), with matching offset widths.
    /// - Writes scores into `results` without allocating. On GPU, inputs should be device-accessible.
    /// - Errors if `results.len()` is insufficient or widths are mixed.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        a: AnyBytesTape<'a>,
        b: AnyBytesTape<'a>,
        results: &mut UnifiedVec<isize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();
        let results_stride = core::mem::size_of::<isize>();

        let a64 = match &a {
            AnyBytesTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyBytesTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        let b64 = match &b {
            AnyBytesTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyBytesTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a64, b64) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_needleman_wunsch_scores_u64tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let a32 = match &a {
            AnyBytesTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyBytesTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        let b32 = match &b {
            AnyBytesTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyBytesTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a32, b32) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_needleman_wunsch_scores_u32tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }
        Err(Error::from(SzStatus::UnexpectedDimensions))
    }
}

impl Drop for NeedlemanWunschScores {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_needleman_wunsch_scores_free(self.handle) };
        }
    }
}

unsafe impl Send for NeedlemanWunschScores {}
unsafe impl Sync for NeedlemanWunschScores {}

/// Smith-Waterman local sequence alignment scoring engine.
///
/// Finds optimal local alignments within sequences using a substitution matrix
/// and gap penalties. Returns maximum scores found anywhere in the alignment matrix.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
/// // Create scoring matrix
/// let mut matrix = [[-1i8; 256]; 256];
/// for i in 0..256 {
///     matrix[i][i] = 2;
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let engine = SmithWatermanScores::new(&device, &matrix, -2, -1).unwrap();
///
/// let seq_a = vec!["ACGTAAACGT"];
/// let seq_b = vec!["ACGT"];
/// let scores = engine.compute(&device, &seq_a, &seq_b).unwrap();
/// ```
pub struct SmithWatermanScores {
    handle: SmithWatermanScoresHandle,
}

impl SmithWatermanScores {
    /// Create a new Smith-Waterman local alignment scoring engine.
    ///
    /// Initializes the engine for local sequence alignment with custom scoring parameters.
    /// The engine automatically adapts to available hardware capabilities.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution context
    /// - `substitution_matrix`: 256x256 scoring matrix for character pairs
    /// - `open_cost`: Gap opening penalty (typically negative)
    /// - `extend_cost`: Gap extension penalty (typically negative, ≥ open_cost)
    ///
    /// # Matrix Design for Local Alignment
    ///
    /// For effective local alignment, the matrix should have:
    /// - **Positive match scores**: Reward similar characters
    /// - **Negative mismatch scores**: Penalize dissimilar characters
    /// - **Balanced penalties**: Prevent excessive gap formation
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Protein alignment matrix (simplified)
    /// let mut protein_matrix = [[-1i8; 256]; 256];  // Default mismatch
    ///
    /// // Set positive scores for similar amino acids
    /// let amino_acids = b"ACDEFGHIKLMNPQRSTVWY";
    /// for &aa in amino_acids {
    ///     protein_matrix[aa as usize][aa as usize] = 5; // Identity
    /// }
    ///
    /// // Similar amino acids get positive but lower scores
    /// protein_matrix[b'L' as usize][b'I' as usize] = 2; // Leucine-Isoleucine
    /// protein_matrix[b'I' as usize][b'L' as usize] = 2;
    ///
    /// let engine = SmithWatermanScores::new(&device, &protein_matrix, -3, -1).unwrap();
    /// ```
    ///
    /// # Gap Penalty Strategy
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// # let mut matrix = [[0i8; 256]; 256];
    /// # let device = DeviceScope::default().unwrap();
    /// // Conservative gaps (discourage insertions/deletions)
    /// let conservative = SmithWatermanScores::new(&device, &matrix, -10, -2).unwrap();
    ///
    /// // Permissive gaps (allow more insertions/deletions)
    /// let permissive = SmithWatermanScores::new(&device, &matrix, -2, -1).unwrap();
    /// ```
    pub fn new(
        device: &DeviceScope,
        substitution_matrix: &[[i8; 256]; 256],
        open_cost: i8,
        extend_cost: i8,
    ) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_smith_waterman_scores_init(
                substitution_matrix.as_ptr() as *const i8,
                open_cost,
                extend_cost,
                ptr::null(),
                capabilities,
                &mut handle,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute Smith-Waterman local alignment scores between sequence pairs.
    ///
    /// Finds the optimal local alignment score for each sequence pair. Returns
    /// the maximum alignment score found within the sequences, representing
    /// the best possible local match.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution
    /// - `sequences_a`: First collection of sequences
    /// - `sequences_b`: Second collection of sequences
    ///
    /// # Returns
    ///
    /// - `Ok(UnifiedVec<isize>)`: Vector of local alignment scores (≥ 0)
    /// - `Err(Error)`: Computation failed
    ///
    /// # Score Interpretation
    ///
    /// - **High scores**: Strong local similarity found
    /// - **Low scores**: Weak or no local similarity
    /// - **Zero scores**: No positive-scoring alignment possible
    /// - **Never negative**: Smith-Waterman scores are always ≥ 0
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// # let mut matrix = [[0i8; 256]; 256];
    /// # for i in 0..256 { matrix[i][i] = 3; for j in 0..256 { if i != j { matrix[i][j] = -1; } } }
    /// let device = DeviceScope::default().unwrap();
    /// let engine = SmithWatermanScores::new(&device, &matrix, -2, -1).unwrap();
    ///
    /// // Local similarity search
    /// let sequences = vec![
    ///     "ATCGATCGATCG_LONG_SEQUENCE_WITH_NOISE",
    ///     "DIFFERENT_SEQUENCE_ATCGATCGATCG_MORE_NOISE",
    ///     "COMPLETELY_UNRELATED_SEQUENCE",
    /// ];
    /// let pattern = vec!["ATCGATCGATCG"; 3];  // Search for this pattern
    ///
    /// let scores = engine.compute(&device, &sequences, &pattern).unwrap();
    ///
    /// for (i, &score) in scores.iter().enumerate() {
    ///     if score > 20 {  // Threshold for significant similarity
    ///         println!("Sequence {} contains similar region (score: {})", i, score);
    ///     }
    /// }
    /// ```
    ///
    /// # Homology Search
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// # let mut matrix = [[0i8; 256]; 256];
    /// # let device = DeviceScope::default().unwrap();
    /// # let engine = SmithWatermanScores::new(&device, &matrix, -2, -1).unwrap();
    /// // Find homologous sequences in a database
    /// let query_seq = vec!["PROTEIN_QUERY_SEQUENCE"];
    /// let database_seqs = vec![
    ///     "HOMOLOGOUS_PROTEIN_SEQUENCE_VARIANT_1",
    ///     "HOMOLOGOUS_PROTEIN_SEQUENCE_VARIANT_2",
    ///     "UNRELATED_PROTEIN_SEQUENCE",
    /// ];
    ///
    /// let queries = vec![query_seq[0]; database_seqs.len()];
    /// let scores = engine.compute(&device, &queries, &database_seqs).unwrap();
    ///
    /// // Sort by score to find best matches
    /// let mut scored_results: Vec<_> = scores.iter().enumerate()
    ///     .map(|(i, &score)| (i, score))
    ///     .collect();
    /// scored_results.sort_by_key(|(_, score)| -score);  // Descending
    ///
    /// println!("Best matches:");
    /// for (idx, score) in scored_results.iter().take(3) {
    ///     println!("Database[{}]: score {}", idx, score);
    /// }
    /// ```
    pub fn compute<T, S>(
        &self,
        device: &DeviceScope,
        sequences_a: T,
        sequences_b: T,
    ) -> Result<UnifiedVec<isize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let seq_a_slice = sequences_a.as_ref();
        let seq_b_slice = sequences_b.as_ref();
        let num_pairs = seq_a_slice.len().min(seq_b_slice.len());

        let mut results = UnifiedVec::with_capacity_in(num_pairs, UnifiedAlloc);
        results.resize(num_pairs, 0);

        let results_stride = core::mem::size_of::<isize>();

        if device.is_gpu() {
            let force_64bit = should_use_64bit_for_bytes(seq_a_slice, seq_b_slice);
            let tape_a = copy_bytes_into_tape(seq_a_slice, force_64bit)?;
            let tape_b = copy_bytes_into_tape(seq_b_slice, force_64bit)?;
            self.compute_into(device, tape_a, tape_b, &mut results)?;
            Ok(results)
        } else {
            let seq_a = SzSequenceFromBytes::to_sz_sequence(seq_a_slice);
            let seq_b = SzSequenceFromBytes::to_sz_sequence(seq_b_slice);
            let mut error_msg: *const c_char = ptr::null();
            let status = unsafe {
                szs_smith_waterman_scores_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            match status {
                Status::Success => Ok(results),
                err => Err(rust_error_from_c_message(err, error_msg)),
            }
        }
    }

    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        a: AnyBytesTape<'a>,
        b: AnyBytesTape<'a>,
        results: &mut UnifiedVec<isize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();
        let results_stride = core::mem::size_of::<isize>();

        let a64 = match &a {
            AnyBytesTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyBytesTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        let b64 = match &b {
            AnyBytesTape::Tape64(t) => Some(SzSequenceU64Tape::from(t)),
            AnyBytesTape::View64(v) => Some(SzSequenceU64Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a64, b64) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_smith_waterman_scores_u64tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let a32 = match &a {
            AnyBytesTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyBytesTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        let b32 = match &b {
            AnyBytesTape::Tape32(t) => Some(SzSequenceU32Tape::from(t)),
            AnyBytesTape::View32(v) => Some(SzSequenceU32Tape::from(v)),
            _ => None,
        };
        if let (Some(va), Some(vb)) = (a32, b32) {
            let need = core::cmp::min(va.count, vb.count);
            if results.len() < need {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let status = unsafe {
                szs_smith_waterman_scores_u32tape(
                    self.handle,
                    device.handle,
                    &va as *const _ as *const c_void,
                    &vb as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }
        Err(Error::from(SzStatus::UnexpectedDimensions))
    }
}

impl Drop for SmithWatermanScores {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_smith_waterman_scores_free(self.handle) };
        }
    }
}

unsafe impl Send for SmithWatermanScores {}
unsafe impl Sync for SmithWatermanScores {}

/// Builder for configuring fingerprinting engines with optimal parameters.
///
/// Provides preset configurations for common use cases and allows fine-tuning
/// of parameters for specific applications.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{Fingerprints, DeviceScope};
/// let device = DeviceScope::default().unwrap();
///
/// // DNA sequence analysis
/// let dna_engine = Fingerprints::builder()
///     .dna()
///     .dimensions(256)
///     .build(&device)
///     .unwrap();
///
/// // Text processing
/// let text_engine = Fingerprints::builder()
///     .ascii()
///     .dimensions(512)
///     .build(&device)
///     .unwrap();
/// ```
pub struct FingerprintsBuilder {
    alphabet_size: usize,
    window_widths: Option<Vec<usize>>,
    dimensions: usize,
}

impl FingerprintsBuilder {
    /// Create a new builder with system-optimized defaults.
    ///
    /// Uses intelligent defaults that adapt to available hardware capabilities:
    /// - Alphabet size: 256 (suitable for binary data and most text)
    /// - Window widths: Hardware-optimized selection
    /// - Dimensions: 1024 (balances accuracy and performance)
    ///
    /// # Returns
    ///
    /// - `Self`: New builder with defaults
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::FingerprintsBuilder;
    /// let builder = FingerprintsBuilder::new();
    /// // Further customize with method chaining...
    /// ```
    pub fn new() -> Self {
        Self {
            alphabet_size: 0,
            window_widths: None,
            dimensions: 1024, // Default dimensions
        }
    }

    /// Configure for binary data processing (256-character alphabet).
    ///
    /// Optimizes the engine for processing arbitrary binary data, including:
    /// - File content analysis
    /// - Network packet inspection
    /// - Binary protocol parsing
    /// - Raw data deduplication
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .binary()
    ///     .dimensions(256)
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Process binary data
    /// let binary_data = vec![
    ///     &[0x89, 0x50, 0x4E, 0x47][..], // PNG header
    ///     &[0xFF, 0xD8, 0xFF, 0xE0][..], // JPEG header  
    ///     &[0x50, 0x4B, 0x03, 0x04][..], // ZIP header
    /// ];
    /// let (hashes, counts) = engine.compute(&device, &binary_data, 256).unwrap();
    /// ```
    pub fn binary(mut self) -> Self {
        self.alphabet_size = 256;
        self
    }

    /// Configure for ASCII text processing (128-character alphabet).
    ///
    /// Optimizes for English text and ASCII-only content:
    /// - Plain text documents
    /// - Source code analysis
    /// - Log file processing
    /// - ASCII-based data formats
    ///
    /// Provides better hash distribution than binary mode for ASCII content.
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .ascii()
    ///     .window_widths(&[3, 5, 7])  // Good for word-level analysis
    ///     .dimensions(256)
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Analyze text documents
    /// let documents = vec![
    ///     "The quick brown fox jumps over the lazy dog",
    ///     "A journey of a thousand miles begins with a single step",
    ///     "To be or not to be, that is the question",
    /// ];
    /// let (hashes, counts) = engine.compute(&device, &documents, 256).unwrap();
    /// ```
    pub fn ascii(mut self) -> Self {
        self.alphabet_size = 128;
        self
    }

    /// Configure for DNA sequence analysis (4-character alphabet: A, C, G, T).
    ///
    /// Highly optimized for genomic applications:
    /// - DNA sequencing analysis
    /// - Genome assembly
    /// - Variant detection
    /// - Phylogenetic analysis
    /// - k-mer counting
    ///
    /// The small alphabet size provides excellent hash quality and performance.
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .dna()
    ///     .window_widths(&[21, 31])  // Common k-mer sizes in genomics
    ///     .dimensions(128)  // 64 * 2 window widths
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Analyze DNA sequences
    /// let sequences = vec![
    ///     "ATCGATCGATCGATCGATCGATCG",
    ///     "GCTAGCTAGCTAGCTAGCTAGCTA",
    ///     "TTAAGGCCTTAAGGCCTTAAGGCC",
    /// ];
    /// let (k_mer_hashes, k_mer_counts) = engine.compute(&device, &sequences, 128).unwrap();
    /// ```
    pub fn dna(mut self) -> Self {
        self.alphabet_size = 4;
        self
    }

    /// Configure for protein sequence analysis (22-character amino acid alphabet).
    ///
    /// Optimized for proteomics and structural biology:
    /// - Protein similarity search
    /// - Structural motif discovery
    /// - Functional domain analysis
    /// - Evolutionary studies
    /// - Mass spectrometry data analysis
    ///
    /// Uses the 20 standard amino acids plus Selenocysteine (U) and Pyrrolysine (O).
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .protein()
    ///     .window_widths(&[5, 7, 9])  // Good for motif detection
    ///     .dimensions(192)  // 64 * 3 window widths
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Analyze protein sequences
    /// let proteins = vec![
    ///     "ACDEFGHIKLMNPQRSTVWY",  // Standard amino acids
    ///     "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASED",
    ///     "GSHMVKVALYDYMPMNANDLQLRKGMHFRFKVAEQAARLIQPQEKKLAKAQQTLDLRSQIQQQQEQLGQ",
    /// ];
    /// let (peptide_hashes, peptide_counts) = engine.compute(&device, &proteins, 192).unwrap();
    /// ```
    pub fn protein(mut self) -> Self {
        self.alphabet_size = 22;
        self
    }

    /// Set a custom alphabet size for specialized applications.
    ///
    /// Use this for domain-specific alphabets or when you know the exact
    /// character set size in your data. Common custom sizes:
    /// - 16: Hexadecimal data
    /// - 64: Base64 encoded data  
    /// - 85: Base85 encoded data
    /// - Custom: Domain-specific character sets
    ///
    /// # Parameters
    ///
    /// - `size`: Number of unique characters in your alphabet (> 0)
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Hexadecimal data (0-9, A-F)
    /// let hex_engine = Fingerprints::builder()
    ///     .alphabet_size(16)
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Custom alphabet for specific domain
    /// let custom_engine = Fingerprints::builder()
    ///     .alphabet_size(32)  // Custom 32-character set
    ///     .window_widths(&[4, 6, 8])
    ///     .build(&device)
    ///     .unwrap();
    /// ```
    pub fn alphabet_size(mut self, size: usize) -> Self {
        self.alphabet_size = size;
        self
    }

    /// Configure window widths (n-gram sizes) for rolling hash computation.
    ///
    /// Window widths determine the size of substrings used for hashing. Different
    /// widths capture patterns at different scales. If not specified, the system
    /// selects optimal widths based on hardware capabilities and alphabet size.
    ///
    /// # Guidelines
    ///
    /// - **Small widths (3-5)**: Capture local patterns, good for noisy data
    /// - **Medium widths (7-15)**: Balance between specificity and robustness
    /// - **Large widths (31+)**: Capture longer patterns, sensitive to changes
    /// - **Multiple widths**: Provide multi-scale pattern detection
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    ///
    /// # Domain-Specific Recommendations
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Natural language (word-level patterns)
    /// let text_engine = Fingerprints::builder()
    ///     .ascii()
    ///     .window_widths(&[3, 4, 5, 7])  // Character n-grams
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Genomics (k-mer analysis)
    /// let genomics_engine = Fingerprints::builder()
    ///     .dna()
    ///     .window_widths(&[15, 21, 31])  // Standard k-mer sizes
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // Document similarity (longer patterns)
    /// let doc_engine = Fingerprints::builder()
    ///     .binary()
    ///     .window_widths(&[5, 7, 11, 15, 31])  // Multi-scale analysis
    ///     .build(&device)
    ///     .unwrap();
    /// ```
    ///
    /// # Performance
    ///
    /// - More windows → better accuracy but slower computation
    /// - Use multiples of the number of hash functions for SIMD efficiency
    /// - Consider total dimensions = 64 × number_of_windows for optimal performance
    pub fn window_widths(mut self, widths: &[usize]) -> Self {
        self.window_widths = Some(widths.to_vec());
        self
    }

    /// Set the total number of dimensions (hash functions) per fingerprint.
    ///
    /// Higher dimensions provide better accuracy and collision resistance at the
    /// cost of increased memory usage and computation time. The optimal value
    /// depends on your accuracy requirements and available resources.
    ///
    /// # Performance
    ///
    /// For optimal SIMD performance, use dimensions that are multiples of 64:
    /// - **64**: Minimal configuration, suitable for rapid prototyping
    /// - **128**: Good for small-scale similarity detection
    /// - **256**: Balanced accuracy/performance for most applications
    /// - **512**: High accuracy for critical applications
    /// - **1024**: Maximum accuracy, use when precision is paramount
    ///
    /// # Recommended Formulas
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Basic formula: 64 * number_of_window_widths
    /// let balanced_engine = Fingerprints::builder()
    ///     .dna()
    ///     .window_widths(&[3, 5, 7, 9])  // 4 widths
    ///     .dimensions(256)  // 64 * 4 = 256
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// // High-precision configuration
    /// let precision_engine = Fingerprints::builder()
    ///     .binary()
    ///     .window_widths(&[5, 7, 11, 15])  // 4 widths
    ///     .dimensions(512)  // 128 * 4 = 512 for extra precision
    ///     .build(&device)
    ///     .unwrap();
    /// ```
    ///
    /// # Memory Usage
    ///
    /// Each fingerprint uses `dimensions * sizeof(u32)` bytes for hashes plus
    /// the same for counts. With 1024 dimensions:
    /// - Per fingerprint: 8KB (4KB hashes + 4KB counts)
    /// - 1000 fingerprints: ~8MB total memory
    ///
    /// # Returns
    ///
    /// - `Self`: Updated builder
    pub fn dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = dimensions;
        self
    }

    /// Build the fingerprinting engine with the configured parameters.
    ///
    /// Creates an optimized fingerprinting engine based on the builder configuration
    /// and the target device capabilities. The engine automatically adapts its
    /// implementation strategy based on available hardware features.
    ///
    /// # Parameter Resolution
    ///
    /// - **alphabet_size = 0**: Defaults to 256 (binary mode)
    /// - **window_widths = None**: Uses hardware-optimized defaults
    /// - **dimensions**: Used as specified, should be multiple of 64
    ///
    /// # Returns
    ///
    /// - `Ok(Fingerprints)`: Successfully created engine
    /// - `Err(Error)`: Invalid parameter combination or allocation failure
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Build with validation
    /// let engine = Fingerprints::builder()
    ///     .dna()
    ///     .dimensions(256)
    ///     .build(&device)
    ///     .expect("Failed to create fingerprinting engine");
    ///
    /// // Verify engine is ready for use
    /// let test_data = vec!["ATCGATCG"];
    /// let result = engine.compute(&device, &test_data, 256);
    /// assert!(result.is_ok());
    /// ```
    pub fn build(self, device: &DeviceScope) -> Result<Fingerprints, Error> {
        let mut engine: FingerprintsHandle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);

        let (widths_ptr, widths_len) = match &self.window_widths {
            Some(widths) => (widths.as_ptr(), widths.len()),
            None => (ptr::null(), 0),
        };

        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_fingerprints_init(
                self.dimensions,
                self.alphabet_size,
                widths_ptr,
                widths_len,
                ptr::null(), // No custom allocator
                capabilities,
                &mut engine,
                &mut error_msg,
            )
        };

        match status {
            Status::Success => Ok(Fingerprints { handle: engine }),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }
}

/// High-performance fingerprinting engine for similarity detection and clustering.
///
/// Computes Min-Hash signatures and Count-Min-Sketch data structures for efficient
/// similarity estimation, duplicate detection, and approximate set operations.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{Fingerprints, DeviceScope};
/// let device = DeviceScope::cpu_cores(1).unwrap();
/// let engine = Fingerprints::builder()
///     .ascii()
///     .dimensions(256)
///     .build(&device)
///     .unwrap();
///
/// let documents = vec![
///     "The quick brown fox jumps over the lazy dog",
///     "A quick brown fox leaps over a lazy dog",
/// ];
///
/// let (hashes, counts) = engine.compute(&device, &documents, 256).unwrap();
/// ```
pub struct Fingerprints {
    handle: FingerprintsHandle,
}

impl Fingerprints {
    /// Create a new builder for configuring the fingerprinting engine.
    ///
    /// Returns a builder instance with intelligent defaults that can be customized
    /// for specific use cases. The builder pattern provides a fluent interface
    /// for configuring complex fingerprinting parameters.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::Fingerprints;
    /// // Start with default configuration
    /// let builder = Fingerprints::builder();
    ///
    /// // Customize as needed
    /// // let engine = builder.dna().dimensions(256).build(&device)?;
    /// ```
    pub fn builder() -> FingerprintsBuilder {
        FingerprintsBuilder::new()
    }

    /// Compute Min-Hash fingerprints and Count-Min-Sketch data for a collection of strings.
    ///
    /// Processes the input strings and generates two types of output:
    /// - **Min-Hashes**: Locality-sensitive hash signatures for similarity detection
    /// - **Min-Counts**: Frequency sketches for approximate counting queries
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution (CPU or GPU)
    /// - `strings`: Collection of input strings to fingerprint
    /// - `dimensions`: Number of hash functions per fingerprint (should match engine config)
    ///
    /// # Returns
    ///
    /// - `Ok((UnifiedVec<u32>, UnifiedVec<u32>))`: (min_hashes, min_counts) in unified memory
    /// - `Err(Error)`: Computation failed
    ///
    /// # Output Format
    ///
    /// Both output vectors have layout: `num_strings × dimensions`
    /// - `min_hashes[i * dimensions + j]`: j-th hash of i-th string
    /// - `min_counts[i * dimensions + j]`: j-th count of i-th string
    ///
    /// # Similarity Estimation
    ///
    /// ```rust
    /// # use stringzilla::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let dimensions = 256;
    /// let engine = Fingerprints::builder()
    ///     .dimensions(dimensions)
    ///     .build(&device)
    ///     .unwrap();
    ///
    /// let strings = vec!["hello world", "hello word", "goodbye world"];
    ///
    /// let (hashes, _counts) = engine.compute(&device, &strings, dimensions).unwrap();
    ///
    /// // Estimate Jaccard similarity between strings 0 and 1
    /// let mut matches = 0;
    /// for i in 0..dimensions {
    ///     if hashes[0 * dimensions + i] == hashes[1 * dimensions + i] {
    ///         matches += 1;
    ///     }
    /// }
    /// let similarity = matches as f64 / dimensions as f64;
    /// println!("Estimated Jaccard similarity: {:.3}", similarity);
    /// ```
    ///
    /// # Memory Management
    ///
    /// Uses unified memory allocation for optimal GPU performance:
    /// - CPU: Standard heap allocation
    /// - GPU: CUDA unified memory (accessible from both CPU and GPU)
    /// - Automatic memory cleanup when vectors are dropped
    ///
    /// # Performance
    ///
    /// - GPU optimal for large batches (>1000 strings)
    /// - Memory usage: 8 bytes per string per dimension
    /// - Processing time scales linearly with total input size
    /// - SIMD acceleration provides significant speedup on modern CPUs
    pub fn compute<T, S>(
        &self,
        device: &DeviceScope,
        strings: T,
        dimensions: usize,
    ) -> Result<(UnifiedVec<u32>, UnifiedVec<u32>), Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let strings_slice = strings.as_ref();
        let num_strings = strings_slice.len();
        let hashes_size = num_strings * dimensions;
        let counts_size = num_strings * dimensions;

        let mut min_hashes = UnifiedVec::with_capacity_in(hashes_size, UnifiedAlloc);
        min_hashes.resize(hashes_size, 0);
        let mut min_counts = UnifiedVec::with_capacity_in(counts_size, UnifiedAlloc);
        min_counts.resize(counts_size, 0);

        let hashes_stride = dimensions * core::mem::size_of::<u32>();
        let counts_stride = dimensions * core::mem::size_of::<u32>();

        if device.is_gpu() {
            // For fingerprints we only have one collection, so estimate if it needs 64-bit
            let total_size: usize = strings_slice.iter().map(|s| s.as_ref().len()).sum();
            let force_64bit = total_size > u32::MAX as usize || strings_slice.len() > u32::MAX as usize;
            let tape = copy_bytes_into_tape(strings_slice, force_64bit)?;

            self.compute_into(device, tape, dimensions, &mut min_hashes, &mut min_counts)?;
            Ok((min_hashes, min_counts))
        } else {
            let sequence = SzSequenceFromBytes::to_sz_sequence(strings_slice);
            let mut error_msg: *const c_char = ptr::null();
            let status = unsafe {
                szs_fingerprints_sequence(
                    self.handle,
                    device.handle,
                    &sequence as *const _ as *const c_void,
                    min_hashes.as_mut_ptr(),
                    hashes_stride,
                    min_counts.as_mut_ptr(),
                    counts_stride,
                    &mut error_msg,
                )
            };
            match status {
                Status::Success => Ok((min_hashes, min_counts)),
                err => Err(rust_error_from_c_message(err, error_msg)),
            }
        }
    }

    /// Compute Min-Hash and Count-Min-Sketch into existing buffers.
    ///
    /// - Accepts `AnyBytesTape<'_>` (owned or view) with either 32- or 64-bit offsets.
    /// - Writes `dimensions` hashes and counts per input row into the provided buffers.
    /// - Buffer lengths must be at least `texts.len() * dimensions`.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        texts: AnyBytesTape<'a>,
        dimensions: usize,
        min_hashes: &mut UnifiedVec<u32>,
        min_counts: &mut UnifiedVec<u32>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();
        let count = match &texts {
            AnyBytesTape::Tape64(t) => SzSequenceU64Tape::from(t).count,
            AnyBytesTape::View64(v) => SzSequenceU64Tape::from(v).count,
            AnyBytesTape::Tape32(t) => SzSequenceU32Tape::from(t).count,
            AnyBytesTape::View32(v) => SzSequenceU32Tape::from(v).count,
        };
        let need = count * dimensions;
        if min_hashes.len() < need || min_counts.len() < need {
            return Err(Error::from(SzStatus::UnexpectedDimensions));
        }
        let hashes_stride = dimensions * core::mem::size_of::<u32>();
        let counts_stride = dimensions * core::mem::size_of::<u32>();
        let status = match &texts {
            AnyBytesTape::Tape64(t) => {
                let v = SzSequenceU64Tape::from(t);
                unsafe {
                    szs_fingerprints_u64tape(
                        self.handle,
                        device.handle,
                        &v as *const _ as *const c_void,
                        min_hashes.as_mut_ptr(),
                        hashes_stride,
                        min_counts.as_mut_ptr(),
                        counts_stride,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View64(vv) => {
                let v = SzSequenceU64Tape::from(vv);
                unsafe {
                    szs_fingerprints_u64tape(
                        self.handle,
                        device.handle,
                        &v as *const _ as *const c_void,
                        min_hashes.as_mut_ptr(),
                        hashes_stride,
                        min_counts.as_mut_ptr(),
                        counts_stride,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Tape32(t) => {
                let v = SzSequenceU32Tape::from(t);
                unsafe {
                    szs_fingerprints_u32tape(
                        self.handle,
                        device.handle,
                        &v as *const _ as *const c_void,
                        min_hashes.as_mut_ptr(),
                        hashes_stride,
                        min_counts.as_mut_ptr(),
                        counts_stride,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View32(vv) => {
                let v = SzSequenceU32Tape::from(vv);
                unsafe {
                    szs_fingerprints_u32tape(
                        self.handle,
                        device.handle,
                        &v as *const _ as *const c_void,
                        min_hashes.as_mut_ptr(),
                        hashes_stride,
                        min_counts.as_mut_ptr(),
                        counts_stride,
                        &mut error_msg,
                    )
                }
            }
        };
        match status {
            Status::Success => Ok(()),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }
}

impl Drop for Fingerprints {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_fingerprints_free(self.handle) };
        }
    }
}

unsafe impl Send for Fingerprints {}
unsafe impl Sync for Fingerprints {}

/// Creates a diagonal substitution matrix for sequence alignment.
/// Diagonal entries (matches) get `match_score`, off-diagonal (mismatches) get `mismatch_score`.
/// Equivalent to C++'s `error_costs_256x256_t::diagonal()` method.
pub fn error_costs_256x256_diagonal(match_score: i8, mismatch_score: i8) -> [[i8; 256]; 256] {
    let mut result = [[0i8; 256]; 256];

    for i in 0..256 {
        for j in 0..256 {
            result[i][j] = if i == j { match_score } else { mismatch_score };
        }
    }

    result
}

/// Equivalent to `error_costs_256x256_diagonal(0, -1)`.
pub fn error_costs_256x256_unary() -> [[i8; 256]; 256] {
    error_costs_256x256_diagonal(0, -1)
}

/// Check if either byte collection requires 64-bit tapes
fn should_use_64bit_for_bytes<T: AsRef<[u8]>>(seq_a: &[T], seq_b: &[T]) -> bool {
    let total_size_a: usize = seq_a.iter().map(|s| s.as_ref().len()).sum();
    let total_size_b: usize = seq_b.iter().map(|s| s.as_ref().len()).sum();
    total_size_a > u32::MAX as usize
        || seq_a.len() > u32::MAX as usize
        || total_size_b > u32::MAX as usize
        || seq_b.len() > u32::MAX as usize
}

/// Check if either string collection requires 64-bit tapes
fn should_use_64bit_for_strings<T: AsRef<str>>(seq_a: &[T], seq_b: &[T]) -> bool {
    let total_size_a: usize = seq_a.iter().map(|s| s.as_ref().len()).sum();
    let total_size_b: usize = seq_b.iter().map(|s| s.as_ref().len()).sum();
    total_size_a > u32::MAX as usize
        || seq_a.len() > u32::MAX as usize
        || total_size_b > u32::MAX as usize
        || seq_b.len() > u32::MAX as usize
}

/// Convert byte sequences to BytesTape
fn copy_bytes_into_tape<'a, T>(sequences: &[T], force_64bit: bool) -> Result<AnyBytesTape<'a>, Error>
where
    T: AsRef<[u8]>,
{
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = force_64bit || total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    if use_64bit {
        let mut tape = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyBytesTape::Tape64(tape))
    } else {
        let mut tape = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyBytesTape::Tape32(tape))
    }
}

/// Convert string sequences to StringTape
fn copy_chars_into_tape<'a, T: AsRef<str>>(sequences: &[T], force_64bit: bool) -> Result<AnyCharsTape<'a>, Error> {
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = force_64bit || total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    if use_64bit {
        let mut tape = StringTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyCharsTape::Tape64(tape))
    } else {
        let mut tape = StringTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyCharsTape::Tape32(tape))
    }
}

/// Get information about the compiled backend
///
/// # Examples
///
/// ```
/// # use stringzilla::szs::backend_info;
/// let info = backend_info();
/// println!("Using backend: {}", info);
/// ```
pub fn backend_info() -> &'static str {
    if cfg!(feature = "cuda") {
        "CUDA GPU acceleration enabled"
    } else if cfg!(all(feature = "rocm", not(feature = "cuda"))) {
        "ROCm GPU acceleration enabled"
    } else if cfg!(all(feature = "cpus", not(any(feature = "cuda", feature = "rocm")))) {
        "Multi-threaded CPU backend enabled"
    } else if cfg!(not(any(feature = "cpus", feature = "cuda", feature = "rocm"))) {
        "StringZillas not available - enable cpus, cuda, or rocm feature"
    } else {
        "Unknown backend"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_info() {
        let info = backend_info();
        assert!(!info.is_empty());
        println!("Backend: {}", info);
    }

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
    fn fingerprint_builder_configurations() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping fingerprint tests - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        // Test default configuration
        let default_engine = Fingerprints::builder().build(&device);
        assert!(default_engine.is_ok(), "Default fingerprint engine should initialize");

        // Test binary configuration
        let binary_engine = Fingerprints::builder().binary().dimensions(256).build(&device);
        assert!(binary_engine.is_ok(), "Binary fingerprint engine should initialize");

        // Test ASCII configuration
        let ascii_engine = Fingerprints::builder().ascii().dimensions(256).build(&device);
        assert!(ascii_engine.is_ok(), "ASCII fingerprint engine should initialize");

        // Test DNA configuration
        let dna_engine = Fingerprints::builder()
            .dna()
            .window_widths(&[3, 5, 7])
            .dimensions(192) // 64 * 3 window widths
            .build(&device);
        assert!(dna_engine.is_ok(), "DNA fingerprint engine should initialize");

        // Test protein configuration
        let protein_engine = Fingerprints::builder()
            .protein()
            .window_widths(&[5, 7])
            .dimensions(128) // 64 * 2 window widths
            .build(&device);
        assert!(protein_engine.is_ok(), "Protein fingerprint engine should initialize");

        // Test custom configuration
        let custom_engine = Fingerprints::builder()
            .alphabet_size(16) // Hexadecimal
            .window_widths(&[4, 6, 8])
            .dimensions(192) // 64 * 3 window widths
            .build(&device);
        assert!(custom_engine.is_ok(), "Custom fingerprint engine should initialize");
    }

    #[test]
    fn fingerprint_computation() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping fingerprint computation test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        let engine_result = Fingerprints::builder()
            .binary()
            .dimensions(64) // Small dimensions for testing
            .build(&device);
        if engine_result.is_err() {
            println!("Skipping fingerprint computation test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        // Test basic computation
        let test_strings = vec!["hello", "world", "test"];
        let result = engine.compute(&device, &test_strings, 64);
        match result {
            Ok((hashes, counts)) => {
                assert_eq!(hashes.len(), 3 * 64); // 3 strings * 64 dimensions
                assert_eq!(counts.len(), 3 * 64); // 3 strings * 64 dimensions
                println!("Fingerprint computation successful");
            }
            Err(e) => println!("Fingerprint computation failed: {:?}", e),
        }
    }

    #[test]
    fn levenshtein_distance_engine() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping Levenshtein test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        // Test engine creation
        let engine_result = LevenshteinDistances::new(
            &device, 0, // match cost
            1, // mismatch cost
            1, // open cost
            1, // extend cost
        );
        if engine_result.is_err() {
            println!("Skipping Levenshtein test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        // Test distance computation
        let strings_a = vec!["kitten", "saturday"];
        let strings_b = vec!["sitting", "sunday"];
        let result = engine.compute(&device, &strings_a, &strings_b);
        match result {
            Ok(distances) => {
                assert_eq!(distances.len(), 2);
                println!("Levenshtein distances: {:?}", distances);
                // kitten -> sitting should be 3 (substitute k->s, e->i, insert g)
                // saturday -> sunday should be 3 (delete a,t,r)
            }
            Err(e) => println!("Levenshtein computation failed: {:?}", e),
        }
    }

    #[test]
    fn levenshtein_utf8_engine() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping UTF-8 Levenshtein test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        let engine_result = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1);
        if engine_result.is_err() {
            println!("Skipping UTF-8 Levenshtein test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        // Test with Unicode strings
        let strings_a = vec!["café", "naïve"];
        let strings_b = vec!["cafe", "naive"];
        let result = engine.compute(&device, &strings_a, &strings_b);
        match result {
            Ok(distances) => {
                assert_eq!(distances.len(), 2);
                println!("UTF-8 Levenshtein distances: {:?}", distances);
                // Each accented character should count as 1 substitution
            }
            Err(e) => println!("UTF-8 Levenshtein computation failed: {:?}", e),
        }
    }

    #[test]
    fn needleman_wunsch_engine() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping Needleman-Wunsch test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        // Create simple scoring matrix
        let mut matrix = [[-1i8; 256]; 256];
        for i in 0..256 {
            matrix[i][i] = 2; // Match score
        }

        let engine_result = NeedlemanWunschScores::new(&device, &matrix, -2, -1);
        if engine_result.is_err() {
            println!("Skipping Needleman-Wunsch test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        let sequences_a = vec!["ACGT"];
        let sequences_b = vec!["ACGT"];
        let result = engine.compute(&device, &sequences_a, &sequences_b);
        match result {
            Ok(scores) => {
                assert_eq!(scores.len(), 1);
                println!("Needleman-Wunsch score: {:?}", scores);
                // Perfect match should give positive score
            }
            Err(e) => println!("Needleman-Wunsch computation failed: {:?}", e),
        }
    }

    #[test]
    fn smith_waterman_engine() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping Smith-Waterman test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        // Create simple scoring matrix
        let mut matrix = [[-1i8; 256]; 256];
        for i in 0..256 {
            matrix[i][i] = 3; // Match score
        }

        let engine_result = SmithWatermanScores::new(&device, &matrix, -2, -1);
        if engine_result.is_err() {
            println!("Skipping Smith-Waterman test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        let sequences_a = vec!["ACGTACGT"];
        let sequences_b = vec!["ACGT"];
        let result = engine.compute(&device, &sequences_a, &sequences_b);
        match result {
            Ok(scores) => {
                assert_eq!(scores.len(), 1);
                println!("Smith-Waterman score: {:?}", scores);
                // Should find local alignment with positive score
            }
            Err(e) => println!("Smith-Waterman computation failed: {:?}", e),
        }
    }

    #[test]
    fn unified_allocator() {
        // Test basic allocation
        let layout = std::alloc::Layout::from_size_align(1024, 8).unwrap();
        let alloc = UnifiedAlloc;

        let result = alloc.allocate(layout);
        match result {
            Ok(memory) => {
                println!("Unified allocation successful: {} bytes", memory.len());
                unsafe { alloc.deallocate(memory.cast(), layout) };
            }
            Err(_) => println!("Unified allocation failed"),
        }

        // Test zero-size allocation
        let zero_layout = std::alloc::Layout::from_size_align(0, 1).unwrap();
        let zero_result = alloc.allocate(zero_layout);
        match zero_result {
            Ok(memory) => {
                assert_eq!(memory.len(), 0);
                unsafe { alloc.deallocate(memory.cast(), zero_layout) };
            }
            Err(_) => println!("Zero-size allocation failed"),
        }
    }

    #[test]
    fn error_handling() {
        // Test that valid operations don't panic
        let valid_cpu = DeviceScope::cpu_cores(0); // 0 means all cores - valid
        assert!(valid_cpu.is_ok(), "CPU cores 0 should succeed");

        let invalid_gpu = DeviceScope::gpu_device(999);
        // May succeed or fail depending on system, but shouldn't panic
        match invalid_gpu {
            Ok(_) => println!("GPU device 999 unexpectedly available"),
            Err(e) => println!("GPU device 999 correctly failed: {:?}", e),
        }

        // Test default device scope
        let default_device = DeviceScope::default();
        match default_device {
            Ok(_) => println!("Default device scope created successfully"),
            Err(e) => println!("Default device scope failed: {:?}", e),
        }
    }

    #[test]
    fn thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping thread safety test - device initialization failed");
            return;
        }
        let device = Arc::new(device_result.unwrap());

        let engine_result = Fingerprints::builder().dimensions(64).build(&device);
        if engine_result.is_err() {
            println!("Skipping thread safety test - engine initialization failed");
            return;
        }
        let engine = Arc::new(engine_result.unwrap());

        // Test parallel computation
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let device = Arc::clone(&device);
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    let test_data = vec![format!("thread_{}_data", i)];
                    engine.compute(&device, &test_data, 64)
                })
            })
            .collect();

        let mut success_count = 0;
        for handle in handles {
            match handle.join().unwrap() {
                Ok(_) => success_count += 1,
                Err(e) => println!("Thread computation failed: {:?}", e),
            }
        }

        println!("Thread safety test: {}/4 threads succeeded", success_count);
    }

    #[test]
    fn large_batch_processing() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping large batch test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        let engine_result = Fingerprints::builder().dimensions(64).build(&device);
        if engine_result.is_err() {
            println!("Skipping large batch test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        // Create large batch
        let large_batch: Vec<String> = (0..1000).map(|i| format!("test_string_{}", i)).collect();
        let large_batch_refs: Vec<&str> = large_batch.iter().map(|s| s.as_str()).collect();

        let result = engine.compute(&device, &large_batch_refs, 64);
        match result {
            Ok((hashes, counts)) => {
                assert_eq!(hashes.len(), 1000 * 64);
                assert_eq!(counts.len(), 1000 * 64);
                println!("Large batch processing successful: 1000 strings processed");
            }
            Err(e) => println!("Large batch processing failed: {:?}", e),
        }
    }

    #[test]
    fn similarity_estimation() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping similarity test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        let engine_result = Fingerprints::builder().dimensions(128).build(&device);
        if engine_result.is_err() {
            println!("Skipping similarity test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        let test_strings = vec![
            "the quick brown fox",
            "the quick brown fox",  // Identical
            "the quick brown dog",  // Similar
            "completely different", // Different
        ];

        let result = engine.compute(&device, &test_strings, 128);
        match result {
            Ok((hashes, _counts)) => {
                // Compare fingerprints
                let dimensions = 128;

                // Compare identical strings (should have high similarity)
                let mut matches_identical = 0;
                for i in 0..dimensions {
                    if hashes[0 * dimensions + i] == hashes[1 * dimensions + i] {
                        matches_identical += 1;
                    }
                }
                let similarity_identical = matches_identical as f64 / dimensions as f64;

                // Compare similar strings
                let mut matches_similar = 0;
                for i in 0..dimensions {
                    if hashes[0 * dimensions + i] == hashes[2 * dimensions + i] {
                        matches_similar += 1;
                    }
                }
                let similarity_similar = matches_similar as f64 / dimensions as f64;

                // Compare different strings
                let mut matches_different = 0;
                for i in 0..dimensions {
                    if hashes[0 * dimensions + i] == hashes[3 * dimensions + i] {
                        matches_different += 1;
                    }
                }
                let similarity_different = matches_different as f64 / dimensions as f64;

                println!("Similarity identical: {:.3}", similarity_identical);
                println!("Similarity similar: {:.3}", similarity_similar);
                println!("Similarity different: {:.3}", similarity_different);

                // Basic sanity checks
                assert!(similarity_identical >= similarity_similar);
                assert!(similarity_similar >= similarity_different);
            }
            Err(e) => println!("Similarity estimation failed: {:?}", e),
        }
    }

    #[test]
    fn error_costs_for_needleman_wunsch() {
        let device_result = DeviceScope::default();
        if device_result.is_err() {
            println!("Skipping error_costs test - device initialization failed");
            return;
        }
        let device = device_result.unwrap();

        // Test our diagonal matrix function with NW aligner
        let matrix = error_costs_256x256_diagonal(2, -1);
        let engine_result = NeedlemanWunschScores::new(&device, &matrix, -2, -1);
        if engine_result.is_err() {
            println!("Skipping error_costs test - NW engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        let seq_a = vec!["ABCD"];
        let seq_b = vec!["ABCD"];
        let result = engine.compute(&device, &seq_a, &seq_b);

        match result {
            Ok(scores) => {
                assert!(scores[0] > 0, "Identical sequences should have positive score");
                println!("Error costs matrix integration test passed: score = {}", scores[0]);
            }
            Err(e) => println!("Error costs test failed: {:?}", e),
        }
    }

    #[test]
    fn levenshtein_compute_into_u32_bytes() {
        let device = match DeviceScope::default() {
            Ok(d) => d,
            Err(_) => return, // skip if device unavailable
        };
        let engine = match LevenshteinDistances::new(&device, 0, 1, 1, 1) {
            Ok(e) => e,
            Err(_) => return,
        };

        let a = [b"kitten".as_ref(), b"saturday".as_ref()];
        let b = [b"sitting".as_ref(), b"sunday".as_ref()];

        let mut ta = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        ta.extend(a).unwrap();
        let mut tb = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        tb.extend(b).unwrap();

        let mut results: UnifiedVec<usize> = UnifiedVec::with_capacity_in(2, UnifiedAlloc);
        results.resize(2, 0);

        let res = engine.compute_into(
            &device,
            AnyBytesTape::Tape32(ta),
            AnyBytesTape::Tape32(tb),
            &mut results,
        );
        if let Ok(()) = res {
            assert_eq!(&results[..], &[3, 3]);
        }
    }

    #[test]
    fn levenshtein_compute_into_u64_bytes() {
        let device = match DeviceScope::default() {
            Ok(d) => d,
            Err(_) => return, // skip if device unavailable
        };
        let engine = match LevenshteinDistances::new(&device, 0, 1, 1, 1) {
            Ok(e) => e,
            Err(_) => return,
        };

        let a = [b"abc".as_ref(), b"abcdef".as_ref()];
        let b = [b"yabd".as_ref(), b"abcxef".as_ref()];

        let mut ta = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        ta.extend(a).unwrap();
        let mut tb = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        tb.extend(b).unwrap();

        let mut results: UnifiedVec<usize> = UnifiedVec::with_capacity_in(2, UnifiedAlloc);
        results.resize(2, 0);

        let res = engine.compute_into(
            &device,
            AnyBytesTape::Tape64(ta),
            AnyBytesTape::Tape64(tb),
            &mut results,
        );
        if let Ok(()) = res {
            // abc vs yabd => distance 2, abcdef vs abcxef => distance 1
            assert_eq!(&results[..], &[2, 1]);
        }
    }
}
