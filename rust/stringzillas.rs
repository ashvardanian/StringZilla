#[doc = r"
The `szs` module provides multi-string parallel operations for StringZillas,
including fingerprinting, Min-Hashes, and Count-Min-Sketches of binary and UTF-8 strings.
This module is available when `cpus`, `cuda`, or `rocm` features are enabled.
"]
extern crate alloc;
use allocator_api2::{alloc::Allocator, alloc::AllocError, alloc::Layout};
use core::ptr;

// Re-export common types from stringzilla
pub use crate::sz::{SortedIdx, Status as SzStatus};

pub mod szs {
    use super::*;
    use alloc::vec::Vec;
    use core::ffi::c_void;

    /// Capability flags
    pub type Capability = u32;

    // Import from stringzilla module
    pub use crate::sz::Status;

    /// Device scope wrapper for parallel execution
    pub struct DeviceScope {
        handle: *mut c_void,
    }

    impl DeviceScope {
        /// Create a default device scope
        ///
        /// # Examples
        ///
        /// ```
        /// # use stringzilla::stringzillas::szs::DeviceScope;
        /// let device = DeviceScope::default().unwrap();
        /// ```
        pub fn default() -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let status = unsafe { sz_device_scope_init_default(&mut handle) };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Create a device scope for CPU cores
        ///
        /// # Examples
        ///
        /// ```
        /// # use stringzilla::stringzillas::szs::DeviceScope;
        /// let device = DeviceScope::cpu_cores(4).unwrap();
        /// ```
        pub fn cpu_cores(cpu_cores: usize) -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let status = unsafe { sz_device_scope_init_cpu_cores(cpu_cores, &mut handle) };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Create a device scope for GPU device
        ///
        /// # Examples
        ///
        /// ```
        /// # use stringzilla::stringzillas::szs::DeviceScope;
        /// let device = DeviceScope::gpu_device(0).unwrap();
        /// ```
        pub fn gpu_device(gpu_device: usize) -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let status = unsafe { sz_device_scope_init_gpu_device(gpu_device, &mut handle) };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Get the capabilities of this device scope
        pub fn get_capabilities(&self) -> Result<Capability, Status> {
            let mut capabilities: Capability = 0;
            let status = unsafe { sz_device_scope_get_capabilities(self.handle, &mut capabilities) };
            match status {
                Status::Success => Ok(capabilities),
                err => Err(err),
            }
        }

        /// Get the raw handle for this device scope
        pub(crate) fn as_ptr(&self) -> *mut c_void {
            self.handle
        }
    }

    impl Drop for DeviceScope {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    sz_device_scope_free(self.handle);
                }
            }
        }
    }

    /// Builder for fingerprinting engine configuration
    ///
    /// By default, uses alphabet_size=0 and null window_widths to let the C layer
    /// infer optimal configurations based on the provided capabilities.
    ///
    /// # Examples
    ///
    /// ```
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .build(&device)
    ///     .unwrap();
    /// ```
    pub struct FingerprintsBuilder {
        alphabet_size: usize,
        window_widths: Option<Vec<usize>>,
        dimensions: usize,
    }

    impl FingerprintsBuilder {
        /// Create a new builder with defaults (alphabet_size=0, no window widths)
        /// The C layer will infer optimal settings based on device capabilities
        pub fn new() -> Self {
            Self {
                alphabet_size: 0,
                window_widths: None,
                dimensions: 1024, // Default dimensions
            }
        }

        /// Set alphabet size to binary (256 characters)
        pub fn binary(mut self) -> Self {
            self.alphabet_size = 256;
            self
        }

        /// Set alphabet size to ASCII (128 characters)
        pub fn ascii(mut self) -> Self {
            self.alphabet_size = 128;
            self
        }

        /// Set alphabet size to DNA (4 characters: A, C, G, T)
        pub fn dna(mut self) -> Self {
            self.alphabet_size = 4;
            self
        }

        /// Set alphabet size to protein (22 characters)
        pub fn protein(mut self) -> Self {
            self.alphabet_size = 22;
            self
        }

        /// Set custom alphabet size
        pub fn alphabet_size(mut self, size: usize) -> Self {
            self.alphabet_size = size;
            self
        }

        /// Set window widths for rolling hashes
        /// If not set, the C layer will use default window widths
        pub fn window_widths(mut self, widths: &[usize]) -> Self {
            self.window_widths = Some(widths.to_vec());
            self
        }

        /// Set total dimensions per fingerprint
        /// Ideally 1024 or a (64 * window_widths_count) multiple
        pub fn dimensions(mut self, dimensions: usize) -> Self {
            self.dimensions = dimensions;
            self
        }

        /// Build the fingerprinting engine using the device scope's capabilities
        ///
        /// If alphabet_size is 0, it will be set to 256 by default.
        /// If window_widths is None, default window widths will be used.
        pub fn build(self, device: &DeviceScope) -> Result<Fingerprints, Status> {
            let mut engine: FingerprintsHandle = ptr::null_mut();
            let capabilities = device.get_capabilities().unwrap_or(0);

            let (widths_ptr, widths_len) = match &self.window_widths {
                Some(widths) => (widths.as_ptr(), widths.len()),
                None => (ptr::null(), 0),
            };

            let status = unsafe {
                sz_fingerprints_init(
                    self.dimensions,
                    self.alphabet_size,
                    widths_ptr,
                    widths_len,
                    ptr::null(), // No custom allocator
                    capabilities,
                    &mut engine,
                )
            };

            match status {
                Status::Success => Ok(Fingerprints { handle: engine }),
                err => Err(err),
            }
        }
    }

    /// StringZillas fingerprinting engine
    pub struct Fingerprints {
        handle: FingerprintsHandle,
    }

    impl Fingerprints {
        /// Create a new builder for configuring the engine
        pub fn builder() -> FingerprintsBuilder {
            FingerprintsBuilder::new()
        }

        /// Process a collection of strings and compute fingerprints
        pub fn fingerprint<T, S>(
            &self,
            device: &DeviceScope,
            strings: &T,
            min_hashes: &mut [u32],
            min_counts: &mut [u32],
        ) -> Result<(), Status>
        where
            T: AsRef<[S]>,
            S: AsRef<[u8]>,
        {
            let strings_slice = strings.as_ref();
            let sequence = create_sequence_view(strings_slice);

            let status = unsafe {
                sz_fingerprints_sequence(
                    self.handle,
                    device.handle,
                    &sequence as *const _ as *const c_void,
                    min_hashes.as_mut_ptr(),
                    min_hashes.len(),
                    min_counts.as_mut_ptr(),
                    min_counts.len(),
                )
            };

            match status {
                Status::Success => Ok(()),
                err => Err(err),
            }
        }
    }

    impl Drop for Fingerprints {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe {
                    sz_fingerprints_free(self.handle);
                }
            }
        }
    }

    /// Internal representation of sz_sequence_t for passing to C
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

    /// Opaque handles for similarity engines
    pub type FingerprintsHandle = *mut c_void;
    pub type LevenshteinDistancesHandle = *mut c_void;
    pub type LevenshteinDistancesUtf8Handle = *mut c_void;
    pub type NeedlemanWunschScoresHandle = *mut c_void;
    pub type SmithWatermanScoresHandle = *mut c_void;

    // C API bindings
    extern "C" {
        // Device scope functions
        fn sz_device_scope_init_default(scope: *mut *mut c_void) -> Status;
        fn sz_device_scope_init_cpu_cores(cpu_cores: usize, scope: *mut *mut c_void) -> Status;
        fn sz_device_scope_init_gpu_device(gpu_device: usize, scope: *mut *mut c_void) -> Status;
        fn sz_device_scope_get_capabilities(scope: *mut c_void, capabilities: *mut Capability) -> Status;
        fn sz_device_scope_free(scope: *mut c_void);

        // Levenshtein distance functions
        fn sz_levenshtein_distances_init(
            match_cost: i8,
            mismatch_cost: i8,
            open_cost: i8,
            extend_cost: i8,
            alloc: *const c_void,
            capabilities: Capability,
            engine: *mut LevenshteinDistancesHandle,
        ) -> Status;

        fn sz_levenshtein_distances_sequence(
            engine: LevenshteinDistancesHandle,
            device: *mut c_void,
            a: *const c_void, // sz_sequence_t
            b: *const c_void, // sz_sequence_t
            results: *mut usize,
            results_stride: usize,
        ) -> Status;

        fn sz_levenshtein_distances_free(engine: LevenshteinDistancesHandle);

        // Levenshtein distance UTF-8 functions
        fn sz_levenshtein_distances_utf8_init(
            match_cost: i8,
            mismatch_cost: i8,
            open_cost: i8,
            extend_cost: i8,
            alloc: *const c_void,
            capabilities: Capability,
            engine: *mut LevenshteinDistancesUtf8Handle,
        ) -> Status;

        fn sz_levenshtein_distances_utf8_sequence(
            engine: LevenshteinDistancesUtf8Handle,
            device: *mut c_void,
            a: *const c_void, // sz_sequence_t
            b: *const c_void, // sz_sequence_t
            results: *mut usize,
            results_stride: usize,
        ) -> Status;

        fn sz_levenshtein_distances_utf8_free(engine: LevenshteinDistancesUtf8Handle);

        // Needleman-Wunsch scoring functions
        fn sz_needleman_wunsch_scores_init(
            subs: *const i8, // 256x256 substitution matrix
            open_cost: i8,
            extend_cost: i8,
            alloc: *const c_void,
            capabilities: Capability,
            engine: *mut NeedlemanWunschScoresHandle,
        ) -> Status;

        fn sz_needleman_wunsch_scores_sequence(
            engine: NeedlemanWunschScoresHandle,
            device: *mut c_void,
            a: *const c_void, // sz_sequence_t
            b: *const c_void, // sz_sequence_t
            results: *mut isize,
            results_stride: usize,
        ) -> Status;

        fn sz_needleman_wunsch_scores_free(engine: NeedlemanWunschScoresHandle);

        // Smith-Waterman scoring functions
        fn sz_smith_waterman_scores_init(
            subs: *const i8, // 256x256 substitution matrix
            open_cost: i8,
            extend_cost: i8,
            alloc: *const c_void,
            capabilities: Capability,
            engine: *mut SmithWatermanScoresHandle,
        ) -> Status;

        fn sz_smith_waterman_scores_sequence(
            engine: SmithWatermanScoresHandle,
            device: *mut c_void,
            a: *const c_void, // sz_sequence_t
            b: *const c_void, // sz_sequence_t
            results: *mut isize,
            results_stride: usize,
        ) -> Status;

        fn sz_smith_waterman_scores_free(engine: SmithWatermanScoresHandle);

        // Fingerprinting functions
        fn sz_fingerprints_init(
            dimensions: usize,
            alphabet_size: usize,
            window_widths: *const usize,
            window_widths_count: usize,
            alloc: *const c_void, // MemoryAllocator - using null for default
            capabilities: Capability,
            engine: *mut FingerprintsHandle,
        ) -> Status;

        fn sz_fingerprints_sequence(
            engine: FingerprintsHandle,
            device: *mut c_void,  // DeviceScope
            texts: *const c_void, // sz_sequence_t
            min_hashes: *mut u32,
            min_hashes_stride: usize,
            min_counts: *mut u32,
            min_counts_stride: usize,
        ) -> Status;

        fn sz_fingerprints_free(engine: FingerprintsHandle);

        // Unified allocator functions
        fn sz_unified_alloc(size_bytes: usize) -> *mut c_void;
        fn sz_unified_free(ptr: *mut c_void, size_bytes: usize);
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

            let ptr = unsafe { sz_unified_alloc(size) };
            if ptr.is_null() {
                return Err(AllocError);
            }

            let ptr = core::ptr::NonNull::new(ptr as *mut u8).ok_or(AllocError)?;
            Ok(core::ptr::NonNull::slice_from_raw_parts(ptr, size))
        }

        unsafe fn deallocate(&self, ptr: core::ptr::NonNull<u8>, layout: Layout) {
            if layout.size() != 0 {
                sz_unified_free(ptr.as_ptr() as *mut c_void, layout.size());
            }
        }
    }

    /// Type alias for Vec with unified allocator
    pub type UnifiedVec<T> = allocator_api2::vec::Vec<T, UnifiedAlloc>;

    /// Levenshtein distance engine for batch processing
    pub struct LevenshteinDistances {
        handle: LevenshteinDistancesHandle,
    }

    impl LevenshteinDistances {
        /// Create a new Levenshtein distances engine
        /// Uses the device scope to infer capabilities
        pub fn new(
            device: &DeviceScope,
            match_cost: i8,
            mismatch_cost: i8,
            open_cost: i8,
            extend_cost: i8,
        ) -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let capabilities = device.get_capabilities().unwrap_or(0);
            let status = unsafe {
                sz_levenshtein_distances_init(
                    match_cost,
                    mismatch_cost,
                    open_cost,
                    extend_cost,
                    ptr::null(),
                    capabilities,
                    &mut handle,
                )
            };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Call operator to compute distances
        pub fn call<T, S>(
            &self,
            device: &DeviceScope,
            sequences_a: T,
            sequences_b: T,
            results: &mut [usize],
        ) -> Result<(), Status>
        where
            T: AsRef<[S]>,
            S: AsRef<[u8]>,
        {
            let seq_a = create_sequence_view(sequences_a.as_ref());
            let seq_b = create_sequence_view(sequences_b.as_ref());

            let results_stride = core::mem::size_of::<usize>(); // stride in bytes
            let status = unsafe {
                sz_levenshtein_distances_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                )
            };
            match status {
                Status::Success => Ok(()),
                err => Err(err),
            }
        }
    }

    impl Drop for LevenshteinDistances {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { sz_levenshtein_distances_free(self.handle) };
            }
        }
    }

    /// UTF-8 aware Levenshtein distance engine
    pub struct LevenshteinDistancesUtf8 {
        handle: LevenshteinDistancesUtf8Handle,
    }

    impl LevenshteinDistancesUtf8 {
        /// Create a new UTF-8 Levenshtein distances engine
        /// Uses the device scope to infer capabilities
        pub fn new(
            device: &DeviceScope,
            match_cost: i8,
            mismatch_cost: i8,
            open_cost: i8,
            extend_cost: i8,
        ) -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let capabilities = device.get_capabilities().unwrap_or(0);
            let status = unsafe {
                sz_levenshtein_distances_utf8_init(
                    match_cost,
                    mismatch_cost,
                    open_cost,
                    extend_cost,
                    ptr::null(),
                    capabilities,
                    &mut handle,
                )
            };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Call operator to compute UTF-8 distances
        pub fn call<T, S>(
            &self,
            device: &DeviceScope,
            sequences_a: T,
            sequences_b: T,
            results: &mut [usize],
        ) -> Result<(), Status>
        where
            T: AsRef<[S]>,
            S: AsRef<str>,
        {
            let seq_a = create_sequence_view_str(sequences_a.as_ref());
            let seq_b = create_sequence_view_str(sequences_b.as_ref());

            let results_stride = core::mem::size_of::<usize>(); // stride in bytes
            let status = unsafe {
                sz_levenshtein_distances_utf8_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                )
            };
            match status {
                Status::Success => Ok(()),
                err => Err(err),
            }
        }
    }

    impl Drop for LevenshteinDistancesUtf8 {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { sz_levenshtein_distances_utf8_free(self.handle) };
            }
        }
    }

    /// Needleman-Wunsch alignment scoring engine
    pub struct NeedlemanWunschScores {
        handle: NeedlemanWunschScoresHandle,
    }

    impl NeedlemanWunschScores {
        /// Create a new Needleman-Wunsch scoring engine
        /// Uses the device scope to infer capabilities
        pub fn new(
            device: &DeviceScope,
            substitution_matrix: &[[i8; 256]; 256],
            open_cost: i8,
            extend_cost: i8,
        ) -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let capabilities = device.get_capabilities().unwrap_or(0);
            let status = unsafe {
                sz_needleman_wunsch_scores_init(
                    substitution_matrix.as_ptr() as *const i8,
                    open_cost,
                    extend_cost,
                    ptr::null(),
                    capabilities,
                    &mut handle,
                )
            };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Call operator to compute alignment scores
        pub fn call<T, S>(
            &self,
            device: &DeviceScope,
            sequences_a: T,
            sequences_b: T,
            results: &mut [isize],
        ) -> Result<(), Status>
        where
            T: AsRef<[S]>,
            S: AsRef<[u8]>,
        {
            let seq_a = create_sequence_view(sequences_a.as_ref());
            let seq_b = create_sequence_view(sequences_b.as_ref());

            let results_stride = core::mem::size_of::<isize>(); // stride in bytes
            let status = unsafe {
                sz_needleman_wunsch_scores_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                )
            };
            match status {
                Status::Success => Ok(()),
                err => Err(err),
            }
        }
    }

    impl Drop for NeedlemanWunschScores {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { sz_needleman_wunsch_scores_free(self.handle) };
            }
        }
    }

    /// Smith-Waterman local alignment scoring engine
    pub struct SmithWatermanScores {
        handle: SmithWatermanScoresHandle,
    }

    impl SmithWatermanScores {
        /// Create a new Smith-Waterman scoring engine
        /// Uses the device scope to infer capabilities
        pub fn new(
            device: &DeviceScope,
            substitution_matrix: &[[i8; 256]; 256],
            open_cost: i8,
            extend_cost: i8,
        ) -> Result<Self, Status> {
            let mut handle = ptr::null_mut();
            let capabilities = device.get_capabilities().unwrap_or(0);
            let status = unsafe {
                sz_smith_waterman_scores_init(
                    substitution_matrix.as_ptr() as *const i8,
                    open_cost,
                    extend_cost,
                    ptr::null(),
                    capabilities,
                    &mut handle,
                )
            };
            match status {
                Status::Success => Ok(Self { handle }),
                err => Err(err),
            }
        }

        /// Call operator to compute local alignment scores
        pub fn call<T, S>(
            &self,
            device: &DeviceScope,
            sequences_a: T,
            sequences_b: T,
            results: &mut [isize],
        ) -> Result<(), Status>
        where
            T: AsRef<[S]>,
            S: AsRef<[u8]>,
        {
            let seq_a = create_sequence_view(sequences_a.as_ref());
            let seq_b = create_sequence_view(sequences_b.as_ref());

            let results_stride = core::mem::size_of::<isize>(); // stride in bytes
            let status = unsafe {
                sz_smith_waterman_scores_sequence(
                    self.handle,
                    device.handle,
                    &seq_a as *const _ as *const c_void,
                    &seq_b as *const _ as *const c_void,
                    results.as_mut_ptr(),
                    results_stride,
                )
            };
            match status {
                Status::Success => Ok(()),
                err => Err(err),
            }
        }
    }

    impl Drop for SmithWatermanScores {
        fn drop(&mut self) {
            if !self.handle.is_null() {
                unsafe { sz_smith_waterman_scores_free(self.handle) };
            }
        }
    }

    /// Zero-copy helper to create sz_sequence_t view from any container of byte slices
    fn create_sequence_view<T: AsRef<[u8]>>(strings: &[T]) -> SzSequence {
        SzSequence {
            handle: strings.as_ptr() as *mut c_void,
            count: strings.len(),
            get_start: sz_sequence_get_start_generic::<T>,
            get_length: sz_sequence_get_length_generic::<T>,
            starts: ptr::null(),
            lengths: ptr::null(),
        }
    }

    /// Zero-copy helper to create sz_sequence_t view from any container of strings
    fn create_sequence_view_str<T: AsRef<str>>(strings: &[T]) -> SzSequence {
        SzSequence {
            handle: strings.as_ptr() as *mut c_void,
            count: strings.len(),
            get_start: sz_sequence_get_start_str::<T>,
            get_length: sz_sequence_get_length_str::<T>,
            starts: ptr::null(),
            lengths: ptr::null(),
        }
    }

    /// Generic C callback to get start of string at index for byte slices
    extern "C" fn sz_sequence_get_start_generic<T: AsRef<[u8]>>(handle: *mut c_void, index: usize) -> *const u8 {
        unsafe {
            let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
            strings[index].as_ref().as_ptr()
        }
    }

    /// Generic C callback to get length of string at index for byte slices
    extern "C" fn sz_sequence_get_length_generic<T: AsRef<[u8]>>(handle: *mut c_void, index: usize) -> usize {
        unsafe {
            let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
            strings[index].as_ref().len()
        }
    }

    /// Generic C callback to get start of string at index for str slices
    extern "C" fn sz_sequence_get_start_str<T: AsRef<str>>(handle: *mut c_void, index: usize) -> *const u8 {
        unsafe {
            let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
            strings[index].as_ref().as_bytes().as_ptr()
        }
    }

    /// Generic C callback to get length of string at index for str slices
    extern "C" fn sz_sequence_get_length_str<T: AsRef<str>>(handle: *mut c_void, index: usize) -> usize {
        unsafe {
            let strings = core::slice::from_raw_parts(handle as *const T, index + 1);
            strings[index].as_ref().as_bytes().len()
        }
    }

    /// Get information about the compiled backend
    ///
    /// # Examples
    ///
    /// ```
    /// # use stringzilla::stringzillas::szs::backend_info;
    /// let info = backend_info();
    /// println!("Using backend: {}", info);
    /// ```
    pub fn backend_info() -> &'static str {
        #[cfg(feature = "cuda")]
        return "CUDA GPU acceleration enabled";

        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        return "ROCm GPU acceleration enabled";

        #[cfg(all(feature = "cpus", not(any(feature = "cuda", feature = "rocm"))))]
        return "Multi-threaded CPU backend enabled";

        #[cfg(not(any(feature = "cpus", feature = "cuda", feature = "rocm")))]
        return "StringZillas not available - enable cpus, cuda, or rocm feature";
    }
}

#[cfg(test)]
mod tests {
    use super::szs::*;

    #[test]
    fn test_backend_info() {
        let info = backend_info();
        assert!(!info.is_empty());
        println!("Backend: {}", info);
    }

    #[test]
    fn test_fingerprint_engine_builder() {
        // Test builder pattern for different use cases
        let device = DeviceScope::default().unwrap();

        let _binary_engine = Fingerprints::builder().binary().build(&device).unwrap();

        let _dna_engine = Fingerprints::builder()
            .dna()
            .window_widths(&[3, 5, 7])
            .dimensions(192) // 64 * 3 window widths
            .build(&device)
            .unwrap();

        let _custom_engine = Fingerprints::builder()
            .alphabet_size(64)
            .window_widths(&[5, 7, 11, 13])
            .dimensions(256) // 64 * 4 window widths
            .build(&device)
            .unwrap();
    }

    #[test]
    fn test_device_scope() {
        // Note: These will fail until the C functions are implemented
        // but they test the Rust API structure
        let _default_device = DeviceScope::default();
        let _cpu_device = DeviceScope::cpu_cores(4);
        let _gpu_device = DeviceScope::gpu_device(0);
    }
}
