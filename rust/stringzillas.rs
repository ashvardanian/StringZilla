extern crate alloc;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void, CStr};
use core::ops::Index;
use core::ptr;

use allocator_api2::{alloc::AllocError, alloc::Allocator, alloc::Layout};
use stringtape::{BytesTape, BytesTapeView, CharsTape, CharsTapeView};

// The `forkunion` crate's build script compiles and links the thread-pool runtime resolving the `fu_*`
// symbols behind the StringZillas engines; referencing the crate keeps that native library in the final
// link even though all calls happen on the C++ side.
use forkunion as _;

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
    Tape32(CharsTape<u32, UnifiedAlloc>),
    Tape64(CharsTape<u64, UnifiedAlloc>),
    // Zero-copy FFI views (UTF-8)
    View32(CharsTapeView<'a, u32>),
    View64(CharsTapeView<'a, u64>),
}

impl<'a> AnyCharsTape<'a> {
    /// Copy string sequences into an owned unified-memory tape, choosing 32- or 64-bit offsets
    /// automatically from the input size. Use this to feed the engines' `compute_into` methods
    /// without depending on the `stringtape` crate directly.
    pub fn from_sequences<T: AsRef<str>>(sequences: &[T]) -> Result<Self, Error> {
        copy_chars_into_tape(sequences, false)
    }
}

/// Tape variant that can hold either 32-bit or 64-bit byte tapes with unsigned offsets
pub enum AnyBytesTape<'a> {
    Tape32(BytesTape<u32, UnifiedAlloc>),
    Tape64(BytesTape<u64, UnifiedAlloc>),
    // Zero-copy FFI views (bytes)
    View32(BytesTapeView<'a, u32>),
    View64(BytesTapeView<'a, u64>),
}

impl<'a> AnyBytesTape<'a> {
    /// Copy byte sequences into an owned unified-memory tape, choosing 32- or 64-bit offsets
    /// automatically from the input size. Use this to feed the engines' `compute_into` methods
    /// without depending on the `stringtape` crate directly.
    pub fn from_sequences<T: AsRef<[u8]>>(sequences: &[T]) -> Result<Self, Error> {
        copy_bytes_into_tape(sequences, false)
    }
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

impl From<&CharsTape<u32, UnifiedAlloc>> for SzSequenceU32Tape {
    fn from(tape: &CharsTape<u32, UnifiedAlloc>) -> Self {
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

impl From<&CharsTape<u64, UnifiedAlloc>> for SzSequenceU64Tape {
    fn from(tape: &CharsTape<u64, UnifiedAlloc>) -> Self {
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

impl<'a> From<CharsTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: CharsTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<CharsTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: CharsTapeView<'a, u64>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU64Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&CharsTapeView<'a, u32>> for SzSequenceU32Tape {
    fn from(view: &CharsTapeView<'a, u32>) -> Self {
        let p = view.as_raw_parts();
        SzSequenceU32Tape {
            data: p.data_ptr,
            offsets: p.offsets_ptr,
            count: p.items_count,
        }
    }
}

impl<'a> From<&CharsTapeView<'a, u64>> for SzSequenceU64Tape {
    fn from(view: &CharsTapeView<'a, u64>) -> Self {
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

    fn szs_levenshtein_distances(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_t
        candidates: *const c_void, // sz_sequence_t; NULL => symmetric self-similarity of queries
        results: *mut usize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_u32tape(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u32tape_t
        candidates: *const c_void, // sz_sequence_u32tape_t; NULL => symmetric self-similarity
        results: *mut usize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_u64tape(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u64tape_t
        candidates: *const c_void, // sz_sequence_u64tape_t; NULL => symmetric self-similarity
        results: *mut usize,
        results_row_stride: usize,
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

    fn szs_levenshtein_distances_utf8(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_t
        candidates: *const c_void, // sz_sequence_t; NULL => symmetric self-similarity of queries
        results: *mut usize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_u32tape(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u32tape_t
        candidates: *const c_void, // sz_sequence_u32tape_t; NULL => symmetric self-similarity
        results: *mut usize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_u64tape(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u64tape_t
        candidates: *const c_void, // sz_sequence_u64tape_t; NULL => symmetric self-similarity
        results: *mut usize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_levenshtein_distances_utf8_free(engine: LevenshteinDistancesUtf8Handle);

    // Needleman-Wunsch scoring functions
    fn szs_needleman_wunsch_scores_init(
        byte_to_class: *const u8,            // 256 byte-to-class map
        class_substitution_costs: *const i8, // 32x32 class substitution matrix
        open_cost: i8,
        extend_cost: i8,
        alloc: *const c_void,
        capabilities: Capability,
        engine: *mut NeedlemanWunschScoresHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_t
        candidates: *const c_void, // sz_sequence_t; NULL => symmetric self-similarity of queries
        results: *mut isize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_u32tape(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u32tape_t
        candidates: *const c_void, // sz_sequence_u32tape_t; NULL => symmetric self-similarity
        results: *mut isize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_u64tape(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u64tape_t
        candidates: *const c_void, // sz_sequence_u64tape_t; NULL => symmetric self-similarity
        results: *mut isize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_needleman_wunsch_scores_free(engine: NeedlemanWunschScoresHandle);

    // Smith-Waterman scoring functions
    fn szs_smith_waterman_scores_init(
        byte_to_class: *const u8,            // 256 byte-to-class map
        class_substitution_costs: *const i8, // 32x32 class substitution matrix
        open_cost: i8,
        extend_cost: i8,
        alloc: *const c_void,
        capabilities: Capability,
        engine: *mut SmithWatermanScoresHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_t
        candidates: *const c_void, // sz_sequence_t; NULL => symmetric self-similarity of queries
        results: *mut isize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_u32tape(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u32tape_t
        candidates: *const c_void, // sz_sequence_u32tape_t; NULL => symmetric self-similarity
        results: *mut isize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_u64tape(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        queries: *const c_void,    // sz_sequence_u64tape_t
        candidates: *const c_void, // sz_sequence_u64tape_t; NULL => symmetric self-similarity
        results: *mut isize,
        results_row_stride: usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_smith_waterman_scores_free(engine: SmithWatermanScoresHandle);

    // Fingerprinting functions
    fn szs_fingerprints_init(
        dimensions: usize,
        alphabet_size: usize,
        window_widths: *const usize,
        window_widths_count: usize,
        seed: u64,
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

/// Row-major cross-product result matrix produced by the similarity engines.
///
/// Every similarity engine computes a dense `queries_count × candidates_count`
/// matrix where `matrix[(query_index, candidate_index)]` holds the distance or
/// score between `queries[query_index]` and `candidates[candidate_index]`.
///
/// The backing storage is a [`UnifiedVec`], so the buffer lives in unified
/// memory and can be consumed by a GPU kernel without an extra copy. Rows are
/// laid out contiguously with a stride of `row_stride` elements; for a freshly
/// allocated matrix `row_stride == candidates_count`.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
///
/// let queries = vec!["kitten", "saturday"];
/// let candidates = vec!["sitting", "sunday"];
/// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
/// assert_eq!(matrix.dimensions(), (2, 2));
/// assert_eq!(matrix[(0, 0)], 3); // kitten vs sitting
/// ```
pub struct Mat<T, A: Allocator = allocator_api2::alloc::Global> {
    data: allocator_api2::vec::Vec<T, A>,
    queries_count: usize,
    candidates_count: usize,
    row_stride: usize,
}

/// A [`Mat`] backed by unified (CPU+GPU) memory - the matrix counterpart of [`UnifiedVec`].
pub type UnifiedMat<T> = Mat<T, UnifiedAlloc>;

impl<T> Mat<T, UnifiedAlloc>
where
    T: Copy + Default,
{
    /// Allocate a zero-initialized `queries_count × candidates_count` matrix in unified memory, guarding
    /// against allocation failure (mirrors the C++ engines' `try_resize` discipline - never panics on OOM).
    ///
    /// The row stride equals `candidates_count`, so rows are stored back-to-back with no padding. The buffer
    /// is suitable for direct GPU consumption.
    pub fn try_allocate(queries_count: usize, candidates_count: usize) -> Result<Self, Error> {
        let element_count = queries_count.saturating_mul(candidates_count);
        let mut data = allocator_api2::vec::Vec::new_in(UnifiedAlloc);
        data.try_reserve_exact(element_count)
            .map_err(|_| Error::from(SzStatus::BadAlloc))?;
        data.resize(element_count, T::default()); // No reallocation: capacity was just reserved.
        Ok(Mat {
            data,
            queries_count,
            candidates_count,
            row_stride: candidates_count,
        })
    }
}

impl<T, A: Allocator> Mat<T, A> {
    /// Returns the `(queries_count, candidates_count)` shape of the matrix.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.queries_count, self.candidates_count)
    }

    /// Returns the number of query rows.
    pub fn queries_count(&self) -> usize {
        self.queries_count
    }

    /// Returns the number of candidate columns.
    pub fn candidates_count(&self) -> usize {
        self.candidates_count
    }

    /// Returns the element stride between consecutive rows in the backing buffer.
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }

    /// Returns the row of distances for `query_index` as a contiguous slice of `candidates_count` elements.
    pub fn row(&self, query_index: usize) -> &[T] {
        let row_start = query_index * self.row_stride;
        &self.data[row_start..row_start + self.candidates_count]
    }

    /// Returns the entire backing buffer (including any inter-row padding) as a flat slice.
    pub fn as_slice(&self) -> &[T] {
        &self.data[..]
    }
}

impl<T, A: Allocator> Index<(usize, usize)> for Mat<T, A> {
    type Output = T;

    fn index(&self, (query_index, candidate_index): (usize, usize)) -> &T {
        &self.data[query_index * self.row_stride + candidate_index]
    }
}

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
/// Computes a cross-product matrix of edit distances between query and candidate
/// byte sequences using configurable gap costs. Optimized for large batches.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
///
/// let queries = vec!["kitten", "saturday"];
/// let candidates = vec!["sitting", "sunday"];
/// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
/// assert_eq!(matrix.dimensions(), (2, 2));
/// assert_eq!(matrix[(0, 0)], 3); // kitten vs sitting
/// assert_eq!(matrix[(1, 1)], 3); // saturday vs sunday
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

    /// Compute the cross-product matrix of Levenshtein distances between queries and candidates.
    ///
    /// Builds a dense row-major `queries × candidates` matrix where
    /// `matrix[(query_index, candidate_index)]` is the edit distance between
    /// `queries[query_index]` and `candidates[candidate_index]`. The function
    /// automatically dispatches to CPU SIMD or GPU based on the device scope.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution
    /// - `queries`: Collection of query sequences (matrix rows)
    /// - `candidates`: Collection of candidate sequences (matrix columns)
    ///
    /// # Returns
    ///
    /// - `Ok(UnifiedMat<usize>)`: `queries.len() × candidates.len()` distance matrix
    /// - `Err(Error)`: Computation failed
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// let queries = vec!["cat", "dog"];
    /// let candidates = vec!["bat", "fog", "word"];
    /// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
    ///
    /// assert_eq!(matrix.dimensions(), (2, 3));
    /// assert_eq!(matrix[(0, 0)], 1); // cat vs bat
    /// ```
    pub fn compute<T, S>(&self, device: &DeviceScope, queries: T, candidates: T) -> Result<UnifiedMat<usize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let queries_slice = queries.as_ref();
        let candidates_slice = candidates.as_ref();
        let mut matrix = UnifiedMat::<usize>::try_allocate(queries_slice.len(), candidates_slice.len())?;
        self.compute_pair(device, queries_slice, Some(candidates_slice), &mut matrix)?;
        Ok(matrix)
    }

    /// Compute the symmetric self-similarity matrix of a single sequence collection.
    ///
    /// Produces a square `sequences × sequences` matrix of pairwise distances by
    /// passing a null candidates pointer to the engine, which then compares the
    /// queries against themselves. The diagonal is the distance of each sequence
    /// to itself and the matrix is symmetric.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, LevenshteinDistances};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// let sequences = vec!["cat", "bat", "rat"];
    /// let matrix = engine.compute_symmetric(&device, &sequences).unwrap();
    /// assert_eq!(matrix.dimensions(), (3, 3));
    /// assert_eq!(matrix[(0, 1)], matrix[(1, 0)]);
    /// ```
    pub fn compute_symmetric<T, S>(&self, device: &DeviceScope, sequences: T) -> Result<UnifiedMat<usize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let sequences_slice = sequences.as_ref();
        let mut matrix = UnifiedMat::<usize>::try_allocate(sequences_slice.len(), sequences_slice.len())?;
        self.compute_pair(device, sequences_slice, None, &mut matrix)?;
        Ok(matrix)
    }

    /// Shared implementation for `compute` and `compute_symmetric`.
    ///
    /// When `candidates` is `None`, a null candidates pointer is forwarded to the
    /// engine to request symmetric self-similarity of `queries`. On GPU devices the
    /// inputs are first copied into unified-memory tapes and dispatched through
    /// `compute_into`.
    fn compute_pair<S>(
        &self,
        device: &DeviceScope,
        queries: &[S],
        candidates: Option<&[S]>,
        matrix: &mut UnifiedMat<usize>,
    ) -> Result<(), Error>
    where
        S: AsRef<[u8]>,
    {
        if device.is_gpu() {
            let force_64bit = match candidates {
                Some(candidates_slice) => should_use_64bit_for_bytes(queries, candidates_slice),
                None => should_use_64bit_for_bytes(queries, queries),
            };
            let queries_tape = copy_bytes_into_tape(queries, force_64bit)?;
            let candidates_tape = match candidates {
                Some(candidates_slice) => Some(copy_bytes_into_tape(candidates_slice, force_64bit)?),
                None => None,
            };
            return self.compute_into(device, queries_tape, candidates_tape, matrix);
        }

        let queries_sequence = SzSequenceFromBytes::to_sz_sequence(queries);
        let candidates_sequence = candidates.map(SzSequenceFromBytes::to_sz_sequence);
        let candidates_ptr = match &candidates_sequence {
            Some(sequence) => sequence as *const _ as *const c_void,
            None => ptr::null(),
        };
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_levenshtein_distances(
                self.handle,
                device.handle,
                &queries_sequence as *const _ as *const c_void,
                candidates_ptr,
                matrix.data.as_mut_ptr(),
                matrix.row_stride,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(()),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute the cross-product distance matrix into a caller-provided matrix.
    ///
    /// - Accepts `AnyBytesTape<'_>` for `queries`: either an owned `BytesTape` or a `BytesTapeView`.
    /// - `candidates` may be `None` to request symmetric self-similarity (a null candidates
    ///   pointer is forwarded to the engine); when `Some`, both inputs must share the same offset width.
    /// - Writes the `queries × candidates` matrix into `matrix` without reallocating.
    ///
    /// Errors
    /// - `UnexpectedDimensions` if the matrix shape does not match the inputs or widths are mixed.
    /// - Underlying engine errors forwarded from the FFI.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        queries: AnyBytesTape<'a>,
        candidates: Option<AnyBytesTape<'a>>,
        matrix: &mut UnifiedMat<usize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();

        // Prefer 64-bit views when the query tape is 64-bit wide.
        let queries64 = match &queries {
            AnyBytesTape::Tape64(tape) => Some(SzSequenceU64Tape::from(tape)),
            AnyBytesTape::View64(view) => Some(SzSequenceU64Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries64 {
            let candidates_view = match &candidates {
                Some(AnyBytesTape::Tape64(tape)) => Some(SzSequenceU64Tape::from(tape)),
                Some(AnyBytesTape::View64(view)) => Some(SzSequenceU64Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_levenshtein_distances_u64tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let queries32 = match &queries {
            AnyBytesTape::Tape32(tape) => Some(SzSequenceU32Tape::from(tape)),
            AnyBytesTape::View32(view) => Some(SzSequenceU32Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries32 {
            let candidates_view = match &candidates {
                Some(AnyBytesTape::Tape32(tape)) => Some(SzSequenceU32Tape::from(tape)),
                Some(AnyBytesTape::View32(view)) => Some(SzSequenceU32Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_levenshtein_distances_u32tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
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
/// let queries = vec!["café", "🦀 rust"];
/// let candidates = vec!["cafe", "🔥 rust"];
/// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
/// assert_eq!(matrix[(0, 0)], 1); // café vs cafe, one character edit
/// assert_eq!(matrix[(1, 1)], 1); // 🦀 rust vs 🔥 rust
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
    /// let queries: Vec<String> = vec!["résumé".to_string(), "naïve".to_string()];
    /// let candidates: Vec<String> = vec!["resume".to_string(), "naive".to_string()];
    /// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
    ///
    /// // Each accented character counts as 1 edit (diagonal of the matrix)
    /// assert_eq!(matrix[(0, 0)], 2); // é->e, é->e
    /// assert_eq!(matrix[(1, 1)], 1); // ï->i
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
    pub fn compute<T, S>(&self, device: &DeviceScope, queries: T, candidates: T) -> Result<UnifiedMat<usize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<str>,
    {
        let queries_slice = queries.as_ref();
        let candidates_slice = candidates.as_ref();
        let mut matrix = UnifiedMat::<usize>::try_allocate(queries_slice.len(), candidates_slice.len())?;
        self.compute_pair(device, queries_slice, Some(candidates_slice), &mut matrix)?;
        Ok(matrix)
    }

    /// Compute the symmetric self-similarity matrix of a single UTF-8 collection.
    ///
    /// Produces a square `sequences × sequences` matrix of character-level edit
    /// distances by forwarding a null candidates pointer to the engine. The matrix
    /// is symmetric and its diagonal holds the distance of each string to itself.
    pub fn compute_symmetric<T, S>(&self, device: &DeviceScope, sequences: T) -> Result<UnifiedMat<usize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<str>,
    {
        let sequences_slice = sequences.as_ref();
        let mut matrix = UnifiedMat::<usize>::try_allocate(sequences_slice.len(), sequences_slice.len())?;
        self.compute_pair(device, sequences_slice, None, &mut matrix)?;
        Ok(matrix)
    }

    /// Shared implementation for `compute` and `compute_symmetric`.
    ///
    /// Forwards a null candidates pointer when `candidates` is `None` to request
    /// symmetric self-similarity. On GPU devices the inputs are first copied into
    /// unified-memory tapes and dispatched through `compute_into`.
    fn compute_pair<S>(
        &self,
        device: &DeviceScope,
        queries: &[S],
        candidates: Option<&[S]>,
        matrix: &mut UnifiedMat<usize>,
    ) -> Result<(), Error>
    where
        S: AsRef<str>,
    {
        if device.is_gpu() {
            let force_64bit = match candidates {
                Some(candidates_slice) => should_use_64bit_for_strings(queries, candidates_slice),
                None => should_use_64bit_for_strings(queries, queries),
            };
            let queries_tape = copy_chars_into_tape(queries, force_64bit)?;
            let candidates_tape = match candidates {
                Some(candidates_slice) => Some(copy_chars_into_tape(candidates_slice, force_64bit)?),
                None => None,
            };
            return self.compute_into(device, queries_tape, candidates_tape, matrix);
        }

        let queries_sequence = SzSequenceFromChars::to_sz_sequence(queries);
        let candidates_sequence = candidates.map(SzSequenceFromChars::to_sz_sequence);
        let candidates_ptr = match &candidates_sequence {
            Some(sequence) => sequence as *const _ as *const c_void,
            None => ptr::null(),
        };
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_levenshtein_distances_utf8(
                self.handle,
                device.handle,
                &queries_sequence as *const _ as *const c_void,
                candidates_ptr,
                matrix.data.as_mut_ptr(),
                matrix.row_stride,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(()),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute the cross-product distance matrix into a caller-provided matrix.
    ///
    /// - Accepts `AnyCharsTape<'_>` for `queries`: `CharsTape` or `CharsTapeView`.
    /// - `candidates` may be `None` for symmetric self-similarity; when `Some`, both
    ///   inputs must share the same offset width.
    /// - Writes the `queries × candidates` matrix into `matrix` without reallocating.
    ///
    /// Requirements and errors are the same as the bytes variant.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        queries: AnyCharsTape<'a>,
        candidates: Option<AnyCharsTape<'a>>,
        matrix: &mut UnifiedMat<usize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();

        // Prefer 64-bit views when the query tape is 64-bit wide.
        let queries64 = match &queries {
            AnyCharsTape::Tape64(tape) => Some(SzSequenceU64Tape::from(tape)),
            AnyCharsTape::View64(view) => Some(SzSequenceU64Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries64 {
            let candidates_view = match &candidates {
                Some(AnyCharsTape::Tape64(tape)) => Some(SzSequenceU64Tape::from(tape)),
                Some(AnyCharsTape::View64(view)) => Some(SzSequenceU64Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_levenshtein_distances_utf8_u64tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let queries32 = match &queries {
            AnyCharsTape::Tape32(tape) => Some(SzSequenceU32Tape::from(tape)),
            AnyCharsTape::View32(view) => Some(SzSequenceU32Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries32 {
            let candidates_view = match &candidates {
                Some(AnyCharsTape::Tape32(tape)) => Some(SzSequenceU32Tape::from(tape)),
                Some(AnyCharsTape::View32(view)) => Some(SzSequenceU32Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_levenshtein_distances_utf8_u32tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
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
/// Finds optimal global alignments using a compact class-based substitution matrix and gap
/// penalties. Returns alignment scores rather than distances.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, NeedlemanWunschScores};
/// // Map each byte to one of 32 classes; here every byte is its own class modulo 32.
/// let mut byte_to_class = [0u8; 256];
/// for i in 0..256 {
///     byte_to_class[i] = (i % 32) as u8;
/// }
/// // Class scoring matrix (match=2, mismatch=-1)
/// let mut class_costs = [[-1i8; 32]; 32];
/// for i in 0..32 {
///     class_costs[i][i] = 2;
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let engine = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
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
    /// - `byte_to_class`: 256-entry map from each input byte to one of 32 character classes
    /// - `class_substitution_costs`: 32x32 matrix of alignment scores between character classes
    /// - `open_cost`: Penalty for opening a gap (typically negative)
    /// - `extend_cost`: Penalty for extending a gap (typically negative, ≤ open_cost)
    pub fn new(
        device: &DeviceScope,
        byte_to_class: &[u8; 256],
        class_substitution_costs: &[[i8; 32]; 32],
        open_cost: i8,
        extend_cost: i8,
    ) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_needleman_wunsch_scores_init(
                byte_to_class.as_ptr() as *const u8,
                class_substitution_costs.as_ptr() as *const i8,
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
    /// # let mut byte_to_class = [0u8; 256];
    /// # for i in 0..256 { byte_to_class[i] = (i % 32) as u8; }
    /// # let mut class_costs = [[-1i8; 32]; 32];
    /// # for i in 0..32 { class_costs[i][i] = 2; }
    /// let device = DeviceScope::default().unwrap();
    /// let engine = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
    ///
    /// // Compare DNA sequences
    /// let queries = vec!["ATCGATCG", "GGCCTTAA"];
    /// let candidates = vec!["ATCGATCC", "GGCCTTAA"]; // One mismatch, one exact
    /// let matrix = engine.compute(&device, &queries, &candidates).unwrap();
    ///
    /// // matrix[(0, 0)] is a lower score (mismatch); matrix[(1, 1)] is the exact match
    /// println!("DNA alignment scores: {:?}", matrix.as_slice());
    /// ```
    ///
    /// # Batch Processing
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, NeedlemanWunschScores};
    /// # let byte_to_class = [0u8; 256];
    /// # let class_costs = [[0i8; 32]; 32];
    /// # let device = DeviceScope::default().unwrap();
    /// # let engine = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
    /// // Process large batches efficiently
    /// let sequences: Vec<&str> = vec![
    ///     "PROTEIN_SEQUENCE_1", "PROTEIN_SEQUENCE_2", /* ... */
    /// ];
    /// let references: Vec<&str> = vec![
    ///     "REFERENCE_SEQ_1", "REFERENCE_SEQ_2", /* ... */
    /// ];
    ///
    /// let matrix = engine.compute(&device, &sequences, &references).unwrap();
    ///
    /// // Find the best-scoring query/candidate cell in the flat matrix buffer
    /// let best_cell = matrix.as_slice().iter().enumerate()
    ///     .max_by_key(|(_, &score)| score)
    ///     .map(|(flat_index, _)| flat_index);
    /// ```
    pub fn compute<T, S>(&self, device: &DeviceScope, queries: T, candidates: T) -> Result<UnifiedMat<isize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let queries_slice = queries.as_ref();
        let candidates_slice = candidates.as_ref();
        let mut matrix = UnifiedMat::<isize>::try_allocate(queries_slice.len(), candidates_slice.len())?;
        self.compute_pair(device, queries_slice, Some(candidates_slice), &mut matrix)?;
        Ok(matrix)
    }

    /// Compute the symmetric self-alignment matrix of a single sequence collection.
    ///
    /// Produces a square `sequences × sequences` matrix of global-alignment scores
    /// by forwarding a null candidates pointer to the engine. The matrix is symmetric
    /// and its diagonal holds the self-alignment score of each sequence.
    pub fn compute_symmetric<T, S>(&self, device: &DeviceScope, sequences: T) -> Result<UnifiedMat<isize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let sequences_slice = sequences.as_ref();
        let mut matrix = UnifiedMat::<isize>::try_allocate(sequences_slice.len(), sequences_slice.len())?;
        self.compute_pair(device, sequences_slice, None, &mut matrix)?;
        Ok(matrix)
    }

    /// Shared implementation for `compute` and `compute_symmetric`.
    ///
    /// Forwards a null candidates pointer when `candidates` is `None` to request
    /// symmetric self-similarity. On GPU devices the inputs are first copied into
    /// unified-memory tapes and dispatched through `compute_into`.
    fn compute_pair<S>(
        &self,
        device: &DeviceScope,
        queries: &[S],
        candidates: Option<&[S]>,
        matrix: &mut UnifiedMat<isize>,
    ) -> Result<(), Error>
    where
        S: AsRef<[u8]>,
    {
        if device.is_gpu() {
            let force_64bit = match candidates {
                Some(candidates_slice) => should_use_64bit_for_bytes(queries, candidates_slice),
                None => should_use_64bit_for_bytes(queries, queries),
            };
            let queries_tape = copy_bytes_into_tape(queries, force_64bit)?;
            let candidates_tape = match candidates {
                Some(candidates_slice) => Some(copy_bytes_into_tape(candidates_slice, force_64bit)?),
                None => None,
            };
            return self.compute_into(device, queries_tape, candidates_tape, matrix);
        }

        let queries_sequence = SzSequenceFromBytes::to_sz_sequence(queries);
        let candidates_sequence = candidates.map(SzSequenceFromBytes::to_sz_sequence);
        let candidates_ptr = match &candidates_sequence {
            Some(sequence) => sequence as *const _ as *const c_void,
            None => ptr::null(),
        };
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_needleman_wunsch_scores(
                self.handle,
                device.handle,
                &queries_sequence as *const _ as *const c_void,
                candidates_ptr,
                matrix.data.as_mut_ptr(),
                matrix.row_stride,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(()),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute the cross-product score matrix into a caller-provided matrix.
    ///
    /// - Accepts `AnyBytesTape<'_>` for `queries` (owned tape or view).
    /// - `candidates` may be `None` for symmetric self-similarity; when `Some`, both
    ///   inputs must share the same offset width.
    /// - Writes the `queries × candidates` matrix into `matrix` without allocating.
    /// - Errors if the matrix shape mismatches the inputs or widths are mixed.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        queries: AnyBytesTape<'a>,
        candidates: Option<AnyBytesTape<'a>>,
        matrix: &mut UnifiedMat<isize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();

        let queries64 = match &queries {
            AnyBytesTape::Tape64(tape) => Some(SzSequenceU64Tape::from(tape)),
            AnyBytesTape::View64(view) => Some(SzSequenceU64Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries64 {
            let candidates_view = match &candidates {
                Some(AnyBytesTape::Tape64(tape)) => Some(SzSequenceU64Tape::from(tape)),
                Some(AnyBytesTape::View64(view)) => Some(SzSequenceU64Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_needleman_wunsch_scores_u64tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let queries32 = match &queries {
            AnyBytesTape::Tape32(tape) => Some(SzSequenceU32Tape::from(tape)),
            AnyBytesTape::View32(view) => Some(SzSequenceU32Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries32 {
            let candidates_view = match &candidates {
                Some(AnyBytesTape::Tape32(tape)) => Some(SzSequenceU32Tape::from(tape)),
                Some(AnyBytesTape::View32(view)) => Some(SzSequenceU32Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_needleman_wunsch_scores_u32tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
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
/// Finds optimal local alignments within sequences using a compact class-based substitution
/// matrix and gap penalties. Returns maximum scores found anywhere in the alignment matrix.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
/// // Map each byte to one of 32 classes; here every byte is its own class modulo 32.
/// let mut byte_to_class = [0u8; 256];
/// for i in 0..256 {
///     byte_to_class[i] = (i % 32) as u8;
/// }
/// // Class scoring matrix (match=2, mismatch=-1)
/// let mut class_costs = [[-1i8; 32]; 32];
/// for i in 0..32 {
///     class_costs[i][i] = 2;
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let engine = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
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
    /// - `byte_to_class`: 256-entry map from each input byte to one of 32 character classes
    /// - `class_substitution_costs`: 32x32 scoring matrix between character classes
    /// - `open_cost`: Gap opening penalty (typically negative)
    /// - `extend_cost`: Gap extension penalty (typically negative, ≥ open_cost)
    ///
    /// # Matrix Design for Local Alignment
    ///
    /// For effective local alignment, the class matrix should have:
    /// - **Positive match scores**: Reward similar classes
    /// - **Negative mismatch scores**: Penalize dissimilar classes
    /// - **Balanced penalties**: Prevent excessive gap formation
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Assign one class per amino-acid, everything else falls into class 0.
    /// let mut byte_to_class = [0u8; 256];
    /// let amino_acids = b"ACDEFGHIKLMNPQRSTVWY";
    /// for (i, &aa) in amino_acids.iter().enumerate() {
    ///     byte_to_class[aa as usize] = (i + 1) as u8;
    /// }
    ///
    /// // Default mismatch, identity on the diagonal, plus a similar-residue bonus.
    /// let mut class_costs = [[-1i8; 32]; 32];
    /// for i in 0..32 {
    ///     class_costs[i][i] = 5; // Identity
    /// }
    /// let leucine = byte_to_class[b'L' as usize] as usize;
    /// let isoleucine = byte_to_class[b'I' as usize] as usize;
    /// class_costs[leucine][isoleucine] = 2; // Leucine-Isoleucine
    /// class_costs[isoleucine][leucine] = 2;
    ///
    /// let engine = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -3, -1).unwrap();
    /// ```
    ///
    /// # Gap Penalty Strategy
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// # let byte_to_class = [0u8; 256];
    /// # let class_costs = [[0i8; 32]; 32];
    /// # let device = DeviceScope::default().unwrap();
    /// // Conservative gaps (discourage insertions/deletions)
    /// let conservative = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -10, -2).unwrap();
    ///
    /// // Permissive gaps (allow more insertions/deletions)
    /// let permissive = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
    /// ```
    pub fn new(
        device: &DeviceScope,
        byte_to_class: &[u8; 256],
        class_substitution_costs: &[[i8; 32]; 32],
        open_cost: i8,
        extend_cost: i8,
    ) -> Result<Self, Error> {
        let mut handle = ptr::null_mut();
        let capabilities = device.get_capabilities().unwrap_or(0);
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_smith_waterman_scores_init(
                byte_to_class.as_ptr() as *const u8,
                class_substitution_costs.as_ptr() as *const i8,
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
    /// # let mut byte_to_class = [0u8; 256];
    /// # for i in 0..256 { byte_to_class[i] = (i % 32) as u8; }
    /// # let mut class_costs = [[-1i8; 32]; 32];
    /// # for i in 0..32 { class_costs[i][i] = 3; }
    /// let device = DeviceScope::default().unwrap();
    /// let engine = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
    ///
    /// // Local similarity search
    /// let sequences = vec![
    ///     "ATCGATCGATCG_LONG_SEQUENCE_WITH_NOISE",
    ///     "DIFFERENT_SEQUENCE_ATCGATCGATCG_MORE_NOISE",
    ///     "COMPLETELY_UNRELATED_SEQUENCE",
    /// ];
    /// let pattern = vec!["ATCGATCGATCG"; 3];  // Search for this pattern
    ///
    /// let matrix = engine.compute(&device, &sequences, &pattern).unwrap();
    ///
    /// for query_index in 0..matrix.queries_count() {
    ///     let score = matrix[(query_index, query_index)];
    ///     if score > 20 {  // Threshold for significant similarity
    ///         println!("Sequence {} contains similar region (score: {})", query_index, score);
    ///     }
    /// }
    /// ```
    ///
    /// # Homology Search
    ///
    /// ```rust
    /// # use stringzilla::szs::{DeviceScope, SmithWatermanScores};
    /// # let byte_to_class = [0u8; 256];
    /// # let class_costs = [[0i8; 32]; 32];
    /// # let device = DeviceScope::default().unwrap();
    /// # let engine = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -2, -1).unwrap();
    /// // Find homologous sequences in a database
    /// let query_seq = vec!["PROTEIN_QUERY_SEQUENCE"];
    /// let database_seqs = vec![
    ///     "HOMOLOGOUS_PROTEIN_SEQUENCE_VARIANT_1",
    ///     "HOMOLOGOUS_PROTEIN_SEQUENCE_VARIANT_2",
    ///     "UNRELATED_PROTEIN_SEQUENCE",
    /// ];
    ///
    /// // One query against the whole database yields a single matrix row.
    /// let matrix = engine.compute(&device, &query_seq, &database_seqs).unwrap();
    ///
    /// // Sort the row by score to find the best matches
    /// let mut scored_results: Vec<_> = matrix.row(0).iter().enumerate()
    ///     .map(|(database_index, &score)| (database_index, score))
    ///     .collect();
    /// scored_results.sort_by_key(|(_, score)| -score);  // Descending
    ///
    /// println!("Best matches:");
    /// for (database_index, score) in scored_results.iter().take(3) {
    ///     println!("Database[{}]: score {}", database_index, score);
    /// }
    /// ```
    pub fn compute<T, S>(&self, device: &DeviceScope, queries: T, candidates: T) -> Result<UnifiedMat<isize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let queries_slice = queries.as_ref();
        let candidates_slice = candidates.as_ref();
        let mut matrix = UnifiedMat::<isize>::try_allocate(queries_slice.len(), candidates_slice.len())?;
        self.compute_pair(device, queries_slice, Some(candidates_slice), &mut matrix)?;
        Ok(matrix)
    }

    /// Compute the symmetric self-alignment matrix of a single sequence collection.
    ///
    /// Produces a square `sequences × sequences` matrix of local-alignment scores
    /// by forwarding a null candidates pointer to the engine. The matrix is symmetric
    /// and its diagonal holds the self-alignment score of each sequence.
    pub fn compute_symmetric<T, S>(&self, device: &DeviceScope, sequences: T) -> Result<UnifiedMat<isize>, Error>
    where
        T: AsRef<[S]>,
        S: AsRef<[u8]>,
    {
        let sequences_slice = sequences.as_ref();
        let mut matrix = UnifiedMat::<isize>::try_allocate(sequences_slice.len(), sequences_slice.len())?;
        self.compute_pair(device, sequences_slice, None, &mut matrix)?;
        Ok(matrix)
    }

    /// Shared implementation for `compute` and `compute_symmetric`.
    ///
    /// Forwards a null candidates pointer when `candidates` is `None` to request
    /// symmetric self-similarity. On GPU devices the inputs are first copied into
    /// unified-memory tapes and dispatched through `compute_into`.
    fn compute_pair<S>(
        &self,
        device: &DeviceScope,
        queries: &[S],
        candidates: Option<&[S]>,
        matrix: &mut UnifiedMat<isize>,
    ) -> Result<(), Error>
    where
        S: AsRef<[u8]>,
    {
        if device.is_gpu() {
            let force_64bit = match candidates {
                Some(candidates_slice) => should_use_64bit_for_bytes(queries, candidates_slice),
                None => should_use_64bit_for_bytes(queries, queries),
            };
            let queries_tape = copy_bytes_into_tape(queries, force_64bit)?;
            let candidates_tape = match candidates {
                Some(candidates_slice) => Some(copy_bytes_into_tape(candidates_slice, force_64bit)?),
                None => None,
            };
            return self.compute_into(device, queries_tape, candidates_tape, matrix);
        }

        let queries_sequence = SzSequenceFromBytes::to_sz_sequence(queries);
        let candidates_sequence = candidates.map(SzSequenceFromBytes::to_sz_sequence);
        let candidates_ptr = match &candidates_sequence {
            Some(sequence) => sequence as *const _ as *const c_void,
            None => ptr::null(),
        };
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_smith_waterman_scores(
                self.handle,
                device.handle,
                &queries_sequence as *const _ as *const c_void,
                candidates_ptr,
                matrix.data.as_mut_ptr(),
                matrix.row_stride,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(()),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Compute the cross-product score matrix into a caller-provided matrix.
    ///
    /// - Accepts `AnyBytesTape<'_>` for `queries` (owned tape or view).
    /// - `candidates` may be `None` for symmetric self-similarity; when `Some`, both
    ///   inputs must share the same offset width.
    /// - Writes the `queries × candidates` matrix into `matrix` without allocating.
    /// - Errors if the matrix shape mismatches the inputs or widths are mixed.
    pub fn compute_into<'a>(
        &self,
        device: &DeviceScope,
        queries: AnyBytesTape<'a>,
        candidates: Option<AnyBytesTape<'a>>,
        matrix: &mut UnifiedMat<isize>,
    ) -> Result<(), Error> {
        let mut error_msg: *const c_char = ptr::null();

        let queries64 = match &queries {
            AnyBytesTape::Tape64(tape) => Some(SzSequenceU64Tape::from(tape)),
            AnyBytesTape::View64(view) => Some(SzSequenceU64Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries64 {
            let candidates_view = match &candidates {
                Some(AnyBytesTape::Tape64(tape)) => Some(SzSequenceU64Tape::from(tape)),
                Some(AnyBytesTape::View64(view)) => Some(SzSequenceU64Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_smith_waterman_scores_u64tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
                    &mut error_msg,
                )
            };
            return match status {
                Status::Success => Ok(()),
                err => Err(rust_error_from_c_message(err, error_msg)),
            };
        }

        let queries32 = match &queries {
            AnyBytesTape::Tape32(tape) => Some(SzSequenceU32Tape::from(tape)),
            AnyBytesTape::View32(view) => Some(SzSequenceU32Tape::from(view)),
            _ => None,
        };
        if let Some(queries_view) = queries32 {
            let candidates_view = match &candidates {
                Some(AnyBytesTape::Tape32(tape)) => Some(SzSequenceU32Tape::from(tape)),
                Some(AnyBytesTape::View32(view)) => Some(SzSequenceU32Tape::from(view)),
                Some(_) => return Err(Error::from(SzStatus::UnexpectedDimensions)),
                None => None,
            };
            let candidates_count = candidates_view.map(|view| view.count).unwrap_or(queries_view.count);
            if matrix.queries_count != queries_view.count || matrix.candidates_count != candidates_count {
                return Err(Error::from(SzStatus::UnexpectedDimensions));
            }
            let candidates_ptr = match &candidates_view {
                Some(view) => view as *const _ as *const c_void,
                None => ptr::null(),
            };
            let status = unsafe {
                szs_smith_waterman_scores_u32tape(
                    self.handle,
                    device.handle,
                    &queries_view as *const _ as *const c_void,
                    candidates_ptr,
                    matrix.data.as_mut_ptr(),
                    matrix.row_stride,
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
    seed: u64,
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
            seed: 0,          // Default reproducibility seed
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

    /// Set the per-dimension diversity seed for the rolling hashers.
    ///
    /// The seed controls how the per-dimension multipliers and moduli are derived. Every value - including the
    /// default `0` - derives both the multiplier and the modulo of every dimension from an independent SplitMix64
    /// stream, which is what makes the resulting MinHashes statistically independent across dimensions.
    ///
    /// # Parameters
    ///
    /// - `seed`: Reproducibility seed; every value yields a distinct, deterministic set of per-dimension parameters.
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
    ///     .dimensions(256)
    ///     .seed(0xC0FFEE)
    ///     .build(&device)
    ///     .unwrap();
    /// ```
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
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
                self.seed,
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

            self.compute_into(device, tape, dimensions, &mut min_hashes[..], &mut min_counts[..])?;
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
        min_hashes: &mut [u32],
        min_counts: &mut [u32],
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

/// Creates a compact class-based diagonal substitution scheme for sequence alignment.
/// Returns a `(byte_to_class, class_substitution_costs)` pair, where each byte maps to one of 32
/// classes (`byte % 32`), matching classes get `match_score`, and mismatching classes get
/// `mismatch_score`.
pub fn error_costs_classes_diagonal(match_score: i8, mismatch_score: i8) -> ([u8; 256], [[i8; 32]; 32]) {
    let mut byte_to_class = [0u8; 256];
    for i in 0..256 {
        byte_to_class[i] = (i % 32) as u8;
    }

    let mut class_costs = [[0i8; 32]; 32];
    for i in 0..32 {
        for j in 0..32 {
            class_costs[i][j] = if i == j { match_score } else { mismatch_score };
        }
    }

    (byte_to_class, class_costs)
}

/// Equivalent to `error_costs_classes_diagonal(0, -1)`.
pub fn error_costs_classes_unary() -> ([u8; 256], [[i8; 32]; 32]) {
    error_costs_classes_diagonal(0, -1)
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

/// Convert string sequences to CharsTape
fn copy_chars_into_tape<'a, T: AsRef<str>>(sequences: &[T], force_64bit: bool) -> Result<AnyCharsTape<'a>, Error> {
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = force_64bit || total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    if use_64bit {
        let mut tape = CharsTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        tape.extend(sequences).map_err(|_| Error::from(SzStatus::BadAlloc))?;
        Ok(AnyCharsTape::Tape64(tape))
    } else {
        let mut tape = CharsTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
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

        // Cross-product distance computation over a non-square query/candidate set.
        let queries = vec!["kitten", "saturday"];
        let candidates = vec!["sitting", "sunday", "kitten"];
        let result = engine.compute(&device, &queries, &candidates);
        match result {
            Ok(matrix) => {
                assert_eq!(matrix.dimensions(), (2, 3));
                // kitten -> sitting is 3 (substitute k->s, e->i, insert g)
                assert_eq!(matrix[(0, 0)], 3);
                // kitten -> kitten is 0 (identical)
                assert_eq!(matrix[(0, 2)], 0);
                // saturday -> sunday is 3 (delete a,t,r)
                assert_eq!(matrix[(1, 1)], 3);
                // The row slice mirrors the indexed cells.
                assert_eq!(matrix.row(0), &[3usize, 6, 0][..]);
                println!("Levenshtein distance matrix: {:?}", matrix.as_slice());
            }
            Err(e) => println!("Levenshtein computation failed: {:?}", e),
        }
    }

    #[test]
    fn levenshtein_distance_symmetric() {
        let device = match DeviceScope::default() {
            Ok(device) => device,
            Err(_) => return,
        };
        let engine = match LevenshteinDistances::new(&device, 0, 1, 1, 1) {
            Ok(engine) => engine,
            Err(_) => return,
        };

        let sequences = vec!["cat", "bat", "cart"];
        let matrix = match engine.compute_symmetric(&device, &sequences) {
            Ok(matrix) => matrix,
            Err(error) => {
                println!("Symmetric Levenshtein computation failed: {:?}", error);
                return;
            }
        };

        assert_eq!(matrix.dimensions(), (3, 3));
        // The diagonal of a self-similarity matrix is all zeros.
        for diagonal_index in 0..3 {
            assert_eq!(matrix[(diagonal_index, diagonal_index)], 0);
        }
        // The matrix is symmetric across its diagonal.
        for first_index in 0..3 {
            for second_index in 0..3 {
                assert_eq!(matrix[(first_index, second_index)], matrix[(second_index, first_index)]);
            }
        }
        // cat vs bat differs by one substitution.
        assert_eq!(matrix[(0, 1)], 1);

        // The diagonal entry equals a one-by-one cross-product of the same string.
        let single = vec!["cat"];
        let single_matrix = engine.compute(&device, &single, &single).unwrap();
        assert_eq!(single_matrix[(0, 0)], matrix[(0, 0)]);
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

        // Cross-product over Unicode strings.
        let queries = vec!["café", "naïve"];
        let candidates = vec!["cafe", "naive"];
        let result = engine.compute(&device, &queries, &candidates);
        match result {
            Ok(matrix) => {
                assert_eq!(matrix.dimensions(), (2, 2));
                // Each accented character counts as one character-level substitution.
                assert_eq!(matrix[(0, 0)], 1); // café vs cafe
                assert_eq!(matrix[(1, 1)], 1); // naïve vs naive
                println!("UTF-8 Levenshtein distance matrix: {:?}", matrix.as_slice());
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

        // Create a simple class-based scoring scheme
        let (byte_to_class, class_costs) = error_costs_classes_diagonal(2, -1);

        let engine_result = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1);
        if engine_result.is_err() {
            println!("Skipping Needleman-Wunsch test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        let queries = vec!["ACGT", "ACGT"];
        let candidates = vec!["ACGT", "TTTT"];
        let result = engine.compute(&device, &queries, &candidates);
        match result {
            Ok(matrix) => {
                assert_eq!(matrix.dimensions(), (2, 2));
                // Identical sequences score higher than the mismatching pair.
                assert!(matrix[(0, 0)] > matrix[(0, 1)]);
                assert!(matrix[(0, 0)] > 0, "Identical sequences should score positively");
                println!("Needleman-Wunsch score matrix: {:?}", matrix.as_slice());
            }
            Err(e) => println!("Needleman-Wunsch computation failed: {:?}", e),
        }

        // Symmetric self-alignment matrix.
        let sequences = vec!["ACGT", "AGGT", "TTTT"];
        if let Ok(matrix) = engine.compute_symmetric(&device, &sequences) {
            assert_eq!(matrix.dimensions(), (3, 3));
            for first_index in 0..3 {
                for second_index in 0..3 {
                    assert_eq!(matrix[(first_index, second_index)], matrix[(second_index, first_index)]);
                }
            }
            // The diagonal self-alignment score is the strongest in its row.
            for diagonal_index in 0..3 {
                let diagonal_score = matrix[(diagonal_index, diagonal_index)];
                for candidate_index in 0..3 {
                    assert!(diagonal_score >= matrix[(diagonal_index, candidate_index)]);
                }
            }
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

        // Create a simple class-based scoring scheme
        let (byte_to_class, class_costs) = error_costs_classes_diagonal(3, -1);

        let engine_result = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -2, -1);
        if engine_result.is_err() {
            println!("Skipping Smith-Waterman test - engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        // One query against two candidates yields a single-row cross-product matrix.
        let queries = vec!["ACGTACGT"];
        let candidates = vec!["ACGT", "TTTT"];
        let result = engine.compute(&device, &queries, &candidates);
        match result {
            Ok(matrix) => {
                assert_eq!(matrix.dimensions(), (1, 2));
                // The embedded "ACGT" run aligns better than the all-mismatch candidate.
                assert!(matrix[(0, 0)] > matrix[(0, 1)]);
                assert!(matrix[(0, 0)] > 0, "Local alignment should be positive");
                println!("Smith-Waterman score matrix: {:?}", matrix.as_slice());
            }
            Err(e) => println!("Smith-Waterman computation failed: {:?}", e),
        }

        // Symmetric self-alignment matrix is symmetric with a dominant diagonal.
        let sequences = vec!["ACGTACGT", "ACGT", "TTTT"];
        if let Ok(matrix) = engine.compute_symmetric(&device, &sequences) {
            assert_eq!(matrix.dimensions(), (3, 3));
            for first_index in 0..3 {
                for second_index in 0..3 {
                    assert_eq!(matrix[(first_index, second_index)], matrix[(second_index, first_index)]);
                }
            }
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

        // Test our diagonal class-based scoring function with NW aligner
        let (byte_to_class, class_costs) = error_costs_classes_diagonal(2, -1);
        let engine_result = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1);
        if engine_result.is_err() {
            println!("Skipping error_costs test - NW engine initialization failed");
            return;
        }
        let engine = engine_result.unwrap();

        let queries = vec!["ABCD"];
        let candidates = vec!["ABCD"];
        let result = engine.compute(&device, &queries, &candidates);

        match result {
            Ok(matrix) => {
                assert_eq!(matrix.dimensions(), (1, 1));
                assert!(matrix[(0, 0)] > 0, "Identical sequences should have positive score");
                println!("Error costs matrix integration test passed: score = {}", matrix[(0, 0)]);
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

        let queries = [b"kitten".as_ref(), b"saturday".as_ref()];
        let candidates = [b"sitting".as_ref(), b"sunday".as_ref()];

        let mut queries_tape = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        queries_tape.extend(queries).unwrap();
        let mut candidates_tape = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        candidates_tape.extend(candidates).unwrap();

        let mut matrix = UnifiedMat::<usize>::try_allocate(2, 2).expect("matrix allocation");

        let outcome = engine.compute_into(
            &device,
            AnyBytesTape::Tape32(queries_tape),
            Some(AnyBytesTape::Tape32(candidates_tape)),
            &mut matrix,
        );
        if let Ok(()) = outcome {
            // Diagonal: kitten vs sitting = 3, saturday vs sunday = 3.
            assert_eq!(matrix[(0, 0)], 3);
            assert_eq!(matrix[(1, 1)], 3);
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

        let queries = [b"abc".as_ref(), b"abcdef".as_ref()];
        let candidates = [b"yabd".as_ref(), b"abcxef".as_ref()];

        let mut queries_tape = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        queries_tape.extend(queries).unwrap();
        let mut candidates_tape = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        candidates_tape.extend(candidates).unwrap();

        let mut matrix = UnifiedMat::<usize>::try_allocate(2, 2).expect("matrix allocation");

        let outcome = engine.compute_into(
            &device,
            AnyBytesTape::Tape64(queries_tape),
            Some(AnyBytesTape::Tape64(candidates_tape)),
            &mut matrix,
        );
        if let Ok(()) = outcome {
            // Diagonal: abc vs yabd => 2, abcdef vs abcxef => 1.
            assert_eq!(matrix[(0, 0)], 2);
            assert_eq!(matrix[(1, 1)], 1);
        }
    }
}
