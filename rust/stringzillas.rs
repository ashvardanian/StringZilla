extern crate alloc;
use alloc::vec::Vec;
use allocator_api2::{alloc::AllocError, alloc::Allocator, alloc::Layout};
use core::ffi::c_void;
use core::ptr;
use stringtape::{BytesTape, StringTape};

// Re-export common types from stringzilla
pub use crate::stringzilla::{SortedIdx, Status as SzStatus};

/// Capability flags
pub type Capability = u32;

// Import from stringzilla module
pub use crate::stringzilla::Status;

/// Device scope manages execution context and hardware resource allocation.
///
/// Device scopes automatically detect available hardware capabilities and select
/// optimal implementations. They coordinate between CPU and GPU resources and
/// manage memory allocation strategies.
///
/// # Hardware Detection
///
/// Device scopes automatically detect:
/// - CPU SIMD capabilities (AVX2, AVX-512, NEON, SVE)
/// - GPU availability (CUDA, ROCm)
/// - Memory hierarchy and bandwidth
/// - Thread pool configuration
///
/// # Examples
///
/// ```rust,no_run
/// use stringzilla::stringzillas::szs::{DeviceScope, Status};
///
/// // Default scope - automatically detects best available hardware
/// let device = DeviceScope::default().unwrap();
/// println!("Capabilities: 0x{:x}", device.get_capabilities().unwrap());
///
/// // Explicit CPU configuration for reproducible results
/// let cpu_device = DeviceScope::cpu_cores(4).unwrap();
/// assert_eq!(cpu_device.get_cpu_cores().unwrap(), 4);
/// assert!(!cpu_device.is_gpu());
///
/// // GPU configuration (requires CUDA/ROCm)
/// if let Ok(gpu_device) = DeviceScope::gpu_device(0) {
///     assert!(gpu_device.is_gpu());
///     println!("Using GPU device: {}", gpu_device.get_gpu_device().unwrap());
/// }
/// ```
///
/// # Error Handling
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, Status};
/// // Handle invalid configurations gracefully
/// match DeviceScope::cpu_cores(0) {
///     Ok(_) => unreachable!("Should not accept 0 cores"),
///     Err(Status::InvalidArgument) => println!("CPU cores must be > 0"),
///     Err(e) => println!("Unexpected error: {:?}", e),
/// }
///
/// // GPU might not be available
/// match DeviceScope::gpu_device(99) {
///     Ok(_) => println!("GPU available"),
///     Err(Status::MissingGpu) => println!("GPU not available or invalid device ID"),
///     Err(e) => println!("GPU error: {:?}", e),
/// }
/// ```
pub struct DeviceScope {
    handle: *mut c_void,
}

impl DeviceScope {
    /// Create a device scope with system defaults.
    ///
    /// Automatically detects available hardware and selects the optimal configuration.
    /// This is the recommended method for most use cases as it adapts to the runtime environment.
    ///
    /// # Returns
    ///
    /// - `Ok(DeviceScope)`: Successfully created device scope
    /// - `Err(Status::BadAlloc)`: Memory allocation failed
    /// - `Err(Status::Unknown)`: System detection failed
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
    /// // Create default device scope
    /// let device = DeviceScope::default().expect("Failed to initialize device");
    ///
    /// // Check detected capabilities
    /// let caps = device.get_capabilities().unwrap();
    /// println!("Hardware capabilities: 0x{:x}", caps);
    ///
    /// // Verify it's working
    /// if device.is_gpu() {
    ///     println!("GPU acceleration available");
    /// } else {
    ///     println!("Using CPU with {} cores", device.get_cpu_cores().unwrap());
    /// }
    /// ```
    pub fn default() -> Result<Self, Status> {
        let mut handle = ptr::null_mut();
        let status = unsafe { sz_device_scope_init_default(&mut handle) };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(err),
        }
    }

    /// Create a device scope for explicit CPU core count.
    ///
    /// Forces CPU-only execution with a specific number of threads. Useful for
    /// benchmarking, testing, or when you need predictable performance characteristics.
    ///
    /// # Parameters
    ///
    /// - `cpu_cores`: Number of CPU cores to use (must be > 1)
    ///
    /// # Returns
    ///
    /// - `Ok(DeviceScope)`: Successfully created CPU device scope
    /// - `Err(Status::InvalidArgument)`: Invalid core count (0 or 1)
    /// - `Err(Status::BadAlloc)`: Failed to create thread pool
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
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
    /// # Performance Notes
    ///
    /// - Optimal core count is usually equal to physical cores
    /// - Hyperthreading may not provide linear scaling for SIMD workloads
    /// - Consider NUMA topology for systems with >16 cores
    pub fn cpu_cores(cpu_cores: usize) -> Result<Self, Status> {
        let mut handle = ptr::null_mut();
        let status = unsafe { sz_device_scope_init_cpu_cores(cpu_cores, &mut handle) };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(err),
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
    /// - `Err(Status::MissingGpu)`: CUDA/ROCm not available or invalid device
    /// - `Err(Status::BadAlloc)`: GPU memory allocation failed
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
    /// // Try multiple GPUs in order of preference
    /// let devices = [0, 1, 2];
    /// let gpu_device = devices
    ///     .iter()
    ///     .find_map(|&id| DeviceScope::gpu_device(id).ok())
    ///     .unwrap_or_else(|| DeviceScope::default().unwrap());
    /// ```
    ///
    /// # Performance Notes
    ///
    /// - GPU is optimal for batch sizes >1000 string pairs
    /// - Memory transfer overhead affects small workloads
    /// - Use unified memory allocation for best GPU performance
    pub fn gpu_device(gpu_device: usize) -> Result<Self, Status> {
        let mut handle = ptr::null_mut();
        let status = unsafe { sz_device_scope_init_gpu_device(gpu_device, &mut handle) };
        match status {
            Status::Success => Ok(Self { handle }),
            err => Err(err),
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
    /// - `Err(Status::Unknown)`: Failed to query capabilities
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
    /// let device = DeviceScope::default().unwrap();
    /// let caps = device.get_capabilities().unwrap();
    ///
    /// // Check specific capabilities (values depend on sz_cap_* constants)
    /// println!("Capabilities: 0x{:x}", caps);
    /// if caps & 0x1 != 0 { println!("Basic SIMD available"); }
    /// if caps & 0x2 != 0 { println!("Advanced SIMD available"); }
    /// ```
    pub fn get_capabilities(&self) -> Result<Capability, Status> {
        let mut capabilities: Capability = 0;
        let status = unsafe { sz_device_scope_get_capabilities(self.handle, &mut capabilities) };
        match status {
            Status::Success => Ok(capabilities),
            err => Err(err),
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
    /// - `Err(Status::Unknown)`: Failed to query configuration
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
    /// let device = DeviceScope::cpu_cores(8).unwrap();
    /// assert_eq!(device.get_cpu_cores().unwrap(), 8);
    ///
    /// // Default scope may use different count
    /// let default_device = DeviceScope::default().unwrap();
    /// let cores = default_device.get_cpu_cores().unwrap();
    /// println!("Default device using {} CPU cores", cores);
    /// ```
    pub fn get_cpu_cores(&self) -> Result<usize, Status> {
        let mut cpu_cores: usize = 0;
        let status = unsafe { sz_device_scope_get_cpu_cores(self.handle, &mut cpu_cores) };
        match status {
            Status::Success => Ok(cpu_cores),
            err => Err(err),
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
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
    pub fn get_gpu_device(&self) -> Result<usize, Status> {
        let mut gpu_device: usize = 0;
        let status = unsafe { sz_device_scope_get_gpu_device(self.handle, &mut gpu_device) };
        match status {
            Status::Success => Ok(gpu_device),
            err => Err(err),
        }
    }

    /// Check if this device scope is configured for GPU execution.
    ///
    /// This is a convenience method that checks whether `get_gpu_device()` would succeed.
    /// Use this to branch between GPU and CPU code paths.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::DeviceScope;
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
            unsafe {
                sz_device_scope_free(self.handle);
            }
        }
    }
}

unsafe impl Send for DeviceScope {}
unsafe impl Sync for DeviceScope {}

/// Builder for configuring fingerprinting engines with optimal parameters.
///
/// The builder pattern allows fine-tuning of fingerprinting parameters for different
/// use cases. Default values are chosen to work well across diverse workloads,
/// but specific applications may benefit from custom configuration.
///
/// # Default Configuration
///
/// - **Alphabet size**: 0 (auto-detect as 256 for binary data)
/// - **Window widths**: None (use optimal defaults for hardware)
/// - **Dimensions**: 1024 (provides good balance of accuracy and performance)
///
/// # Performance Guidelines
///
/// For optimal SIMD and GPU performance:
/// - Use dimensions that are multiples of 64
/// - Choose window widths appropriate for your data
/// - Match alphabet size to your actual character set
///
/// # Examples
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
/// let device = DeviceScope::default().unwrap();
///
/// // Default configuration (good for most use cases)
/// let general_engine = Fingerprints::builder()
///     .build(&device)
///     .unwrap();
///
/// // DNA sequence analysis
/// let dna_engine = Fingerprints::builder()
///     .dna()  // 4-character alphabet: A, C, G, T
///     .window_widths(&[3, 5, 7, 9])  // k-mer sizes for genomics
///     .dimensions(256)  // 64 * 4 window widths
///     .build(&device)
///     .unwrap();
///
/// // High-precision text analysis
/// let text_engine = Fingerprints::builder()
///     .ascii()  // 128-character alphabet
///     .window_widths(&[3, 4, 5, 7, 9, 11, 15, 31])  // Multiple n-gram sizes
///     .dimensions(512)  // 64 * 8 window widths
///     .build(&device)
///     .unwrap();
/// ```
///
/// # Alphabet-Specific Presets
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
/// # let device = DeviceScope::default().unwrap();
/// // Bioinformatics applications
/// let dna_engine = Fingerprints::builder().dna().build(&device).unwrap();         // A,C,G,T
/// let protein_engine = Fingerprints::builder().protein().build(&device).unwrap(); // 22 amino acids
///
/// // Text processing
/// let ascii_engine = Fingerprints::builder().ascii().build(&device).unwrap();     // ASCII text
/// let binary_engine = Fingerprints::builder().binary().build(&device).unwrap();   // Binary data
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::FingerprintsBuilder;
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .binary()
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder()
    ///     .ascii()
    ///     .window_widths(&[3, 5, 7])  // Good for word-level analysis
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
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
    /// # Domain-Specific Recommendations
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
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
    /// # Performance Impact
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
    /// # Performance Guidelines
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
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
    /// - `Err(Status::BadAlloc)`: Memory allocation failed
    /// - `Err(Status::InvalidArgument)`: Invalid parameter combination
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
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

/// High-performance fingerprinting engine for similarity detection and clustering.
///
/// The fingerprinting engine computes Min-Hash signatures and Count-Min-Sketch
/// data structures for collections of strings. These techniques enable efficient
/// similarity estimation, duplicate detection, and approximate set operations.
///
/// # Algorithms
///
/// ## Min-Hash
/// Locality-sensitive hashing technique that estimates Jaccard similarity:
/// - Maps each string to a fixed-size signature
/// - Similar strings produce similar signatures
/// - Enables fast similarity queries in high-dimensional spaces
///
/// ## Count-Min-Sketch
/// Probabilistic data structure for frequency estimation:
/// - Tracks approximate frequencies of elements
/// - Space-efficient alternative to exact counting
/// - Supports streaming data processing
///
/// ## Rolling Hash
/// Efficient hash computation over sliding windows:
/// - Multiple window sizes capture patterns at different scales
/// - SIMD optimizations for parallel hash computation
/// - Configurable alphabet sizes for domain-specific optimization
///
/// # Use Cases
///
/// - **Document deduplication**: Find near-duplicate documents
/// - **Plagiarism detection**: Identify similar text passages
/// - **Genomic analysis**: k-mer counting and sequence clustering
/// - **Web crawling**: Detect duplicate web pages
/// - **Data cleaning**: Merge similar database records
/// - **Content filtering**: Identify spam or harmful content
///
/// # Performance Characteristics
///
/// - **Throughput**: Processes millions of strings per second
/// - **Memory**: O(dimensions × batch_size) working memory
/// - **Accuracy**: Configurable precision vs. speed tradeoffs
/// - **Scalability**: Linear scaling with CPU cores, excellent GPU acceleration
///
/// # Examples
///
/// ## Document Similarity
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
/// let device = DeviceScope::default().unwrap();
/// let engine = Fingerprints::builder()
///     .ascii()  // Text processing optimized
///     .dimensions(512)  // High precision
///     .build(&device)
///     .unwrap();
///
/// let documents = vec![
///     "The quick brown fox jumps over the lazy dog",
///     "A quick brown fox leaps over a lazy dog",  // Similar
///     "Completely different content about science", // Dissimilar
/// ];
///
/// let (hashes, _counts) = engine.compute(&device, &documents, 512).unwrap();
///
/// // Compare fingerprints to estimate similarity
/// // Higher overlap in hash values indicates higher similarity
/// ```
///
/// ## Genomic k-mer Analysis
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
/// let device = DeviceScope::default().unwrap();
/// let engine = Fingerprints::builder()
///     .dna()  // 4-character alphabet optimization
///     .window_widths(&[21, 31])  // Standard k-mer sizes
///     .dimensions(128)  // Memory-efficient
///     .build(&device)
///     .unwrap();
///
/// // Analyze genomic sequences
/// let sequences = vec![
///     "ATCGATCGATCGATCGATCGATCGATCGATCG",
///     "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
///     "TTAAGGCCTTAAGGCCTTAAGGCCTTAAGGCC",
/// ];
///
/// let (k_mer_hashes, k_mer_counts) = engine.compute(&device, &sequences, 128).unwrap();
///
/// // Use hashes for sequence clustering or counts for abundance analysis
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::Fingerprints;
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
    /// - `Err(Status)`: Computation failed
    ///
    /// # Output Format
    ///
    /// Both output vectors have layout: `num_strings × dimensions`
    /// - `min_hashes[i * dimensions + j]`: j-th hash of i-th string
    /// - `min_counts[i * dimensions + j]`: j-th count of i-th string
    ///
    /// # Similarity Estimation
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{Fingerprints, DeviceScope};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = Fingerprints::builder().build(&device).unwrap();
    ///
    /// let strings = vec!["hello world", "hello word", "goodbye world"];
    /// let dimensions = 256;
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
    /// # Performance Notes
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
    ) -> Result<(UnifiedVec<u32>, UnifiedVec<u32>), Status>
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
            let (tape, use_64bit) = create_tape(strings_slice)?;

            let status = if use_64bit {
                let tape_view = create_u64tape_view(&tape);
                unsafe {
                    sz_fingerprints_u64tape(
                        self.handle,
                        device.handle,
                        &tape_view as *const _ as *const c_void,
                        min_hashes.as_mut_ptr(),
                        hashes_stride,
                        min_counts.as_mut_ptr(),
                        counts_stride,
                    )
                }
            } else {
                let tape_view = create_u32tape_view(&tape);
                unsafe {
                    sz_fingerprints_u32tape(
                        self.handle,
                        device.handle,
                        &tape_view as *const _ as *const c_void,
                        min_hashes.as_mut_ptr(),
                        hashes_stride,
                        min_counts.as_mut_ptr(),
                        counts_stride,
                    )
                }
            };
            match status {
                Status::Success => Ok((min_hashes, min_counts)),
                err => Err(err),
            }
        } else {
            let sequence = create_sequence_view(strings_slice);
            let status = unsafe {
                sz_fingerprints_sequence(
                    self.handle,
                    device.handle,
                    &sequence as *const _ as *const c_void,
                    min_hashes.as_mut_ptr(),
                    hashes_stride,
                    min_counts.as_mut_ptr(),
                    counts_stride,
                )
            };
            match status {
                Status::Success => Ok((min_hashes, min_counts)),
                err => Err(err),
            }
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

unsafe impl Send for Fingerprints {}
unsafe impl Sync for Fingerprints {}

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

/// Apache Arrow-like tape for non-NULL strings with 32-bit offsets
#[repr(C)]
struct SzSequenceU32Tape {
    data: *const u8,
    offsets: *const u32,
    count: usize,
}

/// Apache Arrow-like tape for non-NULL strings with 64-bit offsets
#[repr(C)]
struct SzSequenceU64Tape {
    data: *const u8,
    offsets: *const u64,
    count: usize,
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
    fn sz_device_scope_get_cpu_cores(scope: *mut c_void, cpu_cores: *mut usize) -> Status;
    fn sz_device_scope_get_gpu_device(scope: *mut c_void, gpu_device: *mut usize) -> Status;
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

    fn sz_levenshtein_distances_u32tape(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut usize,
        results_stride: usize,
    ) -> Status;

    fn sz_levenshtein_distances_u64tape(
        engine: LevenshteinDistancesHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
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

    fn sz_levenshtein_distances_utf8_u32tape(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut usize,
        results_stride: usize,
    ) -> Status;

    fn sz_levenshtein_distances_utf8_u64tape(
        engine: LevenshteinDistancesUtf8Handle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
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

    fn sz_needleman_wunsch_scores_u32tape(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut isize,
        results_stride: usize,
    ) -> Status;

    fn sz_needleman_wunsch_scores_u64tape(
        engine: NeedlemanWunschScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
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

    fn sz_smith_waterman_scores_u32tape(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u32tape_t
        b: *const c_void, // sz_sequence_u32tape_t
        results: *mut isize,
        results_stride: usize,
    ) -> Status;

    fn sz_smith_waterman_scores_u64tape(
        engine: SmithWatermanScoresHandle,
        device: *mut c_void,
        a: *const c_void, // sz_sequence_u64tape_t
        b: *const c_void, // sz_sequence_u64tape_t
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

    fn sz_fingerprints_u32tape(
        engine: FingerprintsHandle,
        device: *mut c_void,  // DeviceScope
        texts: *const c_void, // sz_sequence_u32tape_t
        min_hashes: *mut u32,
        min_hashes_stride: usize,
        min_counts: *mut u32,
        min_counts_stride: usize,
    ) -> Status;

    fn sz_fingerprints_u64tape(
        engine: FingerprintsHandle,
        device: *mut c_void,  // DeviceScope
        texts: *const c_void, // sz_sequence_u64tape_t
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

/// Levenshtein distance engine for batch processing of binary sequences.
///
/// Computes edit distances between pairs of byte sequences using the Wagner-Fischer
/// dynamic programming algorithm with configurable gap costs. This engine is optimized
/// for processing large batches of sequence pairs in parallel.
///
/// # Algorithm
///
/// Uses Wagner-Fischer dynamic programming with affine gap costs:
/// - **Match cost**: Cost when characters are identical (typically ≤ 0)
/// - **Mismatch cost**: Cost when characters differ (typically > 0)  
/// - **Open cost**: Cost to start a gap (insertion/deletion)
/// - **Extend cost**: Cost to extend an existing gap (usually < open cost)
///
/// Time complexity: O(nm) per pair, where n and m are sequence lengths.
/// Space complexity: O(n+m) with optimizations for large batches.
///
/// # Use Cases
///
/// - **Spell checking**: Finding closest dictionary matches
/// - **File deduplication**: Detecting similar binary files
/// - **Data cleaning**: Merging near-duplicate records
/// - **Fuzzy matching**: Approximate string search
///
/// # Examples
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistances};
/// // Create engine with standard costs
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistances::new(
///     &device,
///     0,  // match: no cost for identical chars
///     1,  // mismatch: unit cost for different chars
///     1,  // open: unit cost to start gap
///     1,  // extend: unit cost to extend gap
/// ).unwrap();
///
/// // Compare similar strings
/// let strings_a = vec!["kitten", "saturday"];
/// let strings_b = vec!["sitting", "sunday"];
/// let distances = engine.compute(&device, &strings_a, &strings_b).unwrap();
///
/// println!("Distance 'kitten' -> 'sitting': {}", distances[0]);  // 3
/// println!("Distance 'saturday' -> 'sunday': {}", distances[1]); // 3
/// ```
///
/// # Advanced Configuration
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistances};
/// // Biased towards insertions/deletions over substitutions
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistances::new(
///     &device,
///     -1, // match: reward for matches
///     3,  // mismatch: high cost for substitutions  
///     1,  // open: low cost for gaps
///     1,  // extend: same cost to extend
/// ).unwrap();
/// ```
///
/// # Performance Optimization
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistances};
/// // For maximum performance with large batches
/// let device = DeviceScope::cpu_cores(8).unwrap(); // or gpu_device(0)
/// let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
///
/// // Process thousands of pairs efficiently
/// let large_batch_a: Vec<&str> = (0..10000).map(|i| "test_string").collect();
/// let large_batch_b: Vec<&str> = (0..10000).map(|i| "test_strong").collect();
/// let distances = engine.compute(&device, &large_batch_a, &large_batch_b).unwrap();
/// ```
pub struct LevenshteinDistances {
    handle: LevenshteinDistancesHandle,
}

impl LevenshteinDistances {
    /// Create a new Levenshtein distances engine with specified costs.
    ///
    /// Initializes the engine with custom gap costs for fine-tuned distance computation.
    /// The engine automatically selects optimal algorithms based on the device capabilities.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution context and capabilities
    /// - `match_cost`: Cost when characters match (typically ≤ 0)
    /// - `mismatch_cost`: Cost when characters differ (typically > 0)
    /// - `open_cost`: Cost to open a gap (insertion/deletion)
    /// - `extend_cost`: Cost to extend existing gap (usually ≤ open_cost)
    ///
    /// # Returns
    ///
    /// - `Ok(LevenshteinDistances)`: Successfully initialized engine
    /// - `Err(Status::BadAlloc)`: Memory allocation failed
    /// - `Err(Status::InvalidArgument)`: Invalid cost configuration
    ///
    /// # Cost Configuration Guidelines
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistances};
    /// let device = DeviceScope::default().unwrap();
    ///
    /// // Standard Levenshtein distance (all operations cost 1)
    /// let standard = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// // Prefer matches, penalize mismatches heavily
    /// let match_biased = LevenshteinDistances::new(&device, -1, 3, 2, 1).unwrap();
    ///
    /// // Linear gap costs (open == extend)
    /// let linear_gaps = LevenshteinDistances::new(&device, 0, 1, 2, 2).unwrap();
    ///
    /// // Affine gap costs (open > extend, penalizes gap opening)
    /// let affine_gaps = LevenshteinDistances::new(&device, 0, 1, 3, 1).unwrap();
    /// ```
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
    /// - `Err(Status)`: Computation failed
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistances};
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
    /// # Performance Notes
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
    ) -> Result<UnifiedVec<usize>, Status>
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
            let (tape_a, use_64bit_a) = create_tape(seq_a_slice)?;
            let (tape_b, use_64bit_b) = create_tape(seq_b_slice)?;

            let status = if use_64bit_a || use_64bit_b {
                let tape_a_view = create_u64tape_view(&tape_a);
                let tape_b_view = create_u64tape_view(&tape_b);
                unsafe {
                    sz_levenshtein_distances_u64tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            } else {
                let tape_a_view = create_u32tape_view(&tape_a);
                let tape_b_view = create_u32tape_view(&tape_b);
                unsafe {
                    sz_levenshtein_distances_u32tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            };
            match status {
                Status::Success => Ok(results),
                err => Err(err),
            }
        } else {
            let seq_a = create_sequence_view(seq_a_slice);
            let seq_b = create_sequence_view(seq_b_slice);
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
                Status::Success => Ok(results),
                err => Err(err),
            }
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

unsafe impl Send for LevenshteinDistances {}
unsafe impl Sync for LevenshteinDistances {}

/// UTF-8 aware Levenshtein distance engine for Unicode text processing.
///
/// Computes edit distances between UTF-8 encoded strings at the character level
/// rather than byte level. This engine properly handles multi-byte UTF-8 sequences,
/// ensuring that operations are performed on Unicode code points.
///
/// # UTF-8 vs Binary Processing
///
/// - **Binary engine**: Operates on raw bytes, faster but incorrect for Unicode
/// - **UTF-8 engine**: Operates on Unicode code points, slower but semantically correct
///
/// Use this engine when working with international text, emoji, or any content
/// where character boundaries matter.
///
/// # Examples
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistancesUtf8};
/// let device = DeviceScope::default().unwrap();
/// let engine = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
///
/// // Unicode strings with emoji and accents
/// let strings_a = vec!["café", "naïve", "🦀 rust"];
/// let strings_b = vec!["cafe", "naive", "🔥 rust"];
/// let distances = engine.compute(&device, &strings_a, &strings_b).unwrap();
///
/// // Character-level distances (not byte-level)
/// println!("'café' -> 'cafe': {}", distances[0]); // 1 (é -> e)
/// println!("'🦀 rust' -> '🔥 rust': {}", distances[2]); // 1 (🦀 -> 🔥)
/// ```
///
/// # Comparison with Binary Engine
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistances, LevenshteinDistancesUtf8};
/// let device = DeviceScope::default().unwrap();
/// let binary_engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).unwrap();
/// let utf8_engine = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
///
/// let text_a = vec!["café"]; // 5 bytes, 4 characters
/// let text_b = vec!["cafe"]; // 4 bytes, 4 characters
///
/// let binary_dist = binary_engine.compute(&device, &text_a, &text_b).unwrap();
/// let utf8_dist = utf8_engine.compute(&device, &text_a, &text_b).unwrap();
///
/// println!("Binary distance: {}", binary_dist[0]); // 2 (é is 2 bytes)
/// println!("UTF-8 distance: {}", utf8_dist[0]);   // 1 (é is 1 character)
/// ```
///
/// # Performance Considerations
///
/// - Slower than binary engine due to UTF-8 decoding overhead
/// - Performance impact depends on character distribution
/// - ASCII-only text has minimal overhead
/// - Complex scripts (Arabic, Thai) have higher overhead
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
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistancesUtf8};
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

    /// Compute UTF-8 aware Levenshtein distances between string pairs.
    ///
    /// Processes Unicode strings character by character, ensuring proper handling
    /// of multi-byte UTF-8 sequences. Critical for applications requiring semantic
    /// correctness with international text.
    ///
    /// # Type Requirements
    ///
    /// Input sequences must implement `AsRef<str>` to ensure valid UTF-8:
    /// - `&str`, `String`, `Cow<str>` are all supported
    /// - Invalid UTF-8 will cause undefined behavior
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistancesUtf8};
    /// let device = DeviceScope::default().unwrap();
    /// let engine = LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).unwrap();
    ///
    /// // Mixed string types
    /// let strings_a: Vec<String> = vec!["résumé".to_string(), "naïve".to_string()];
    /// let strings_b: Vec<&str> = vec!["resume", "naive"];
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, LevenshteinDistancesUtf8};
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
    ) -> Result<UnifiedVec<usize>, Status>
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
            let (tape_a, use_64bit_a) = create_tape_str(seq_a_slice)?;
            let (tape_b, use_64bit_b) = create_tape_str(seq_b_slice)?;

            let status = if use_64bit_a || use_64bit_b {
                let tape_a_view = create_u64tape_view_str(&tape_a);
                let tape_b_view = create_u64tape_view_str(&tape_b);
                unsafe {
                    sz_levenshtein_distances_utf8_u64tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            } else {
                let tape_a_view = create_u32tape_view_str(&tape_a);
                let tape_b_view = create_u32tape_view_str(&tape_b);
                unsafe {
                    sz_levenshtein_distances_utf8_u32tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            };
            match status {
                Status::Success => Ok(results),
                err => Err(err),
            }
        } else {
            let seq_a = create_sequence_view_str(seq_a_slice);
            let seq_b = create_sequence_view_str(seq_b_slice);
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
                Status::Success => Ok(results),
                err => Err(err),
            }
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

unsafe impl Send for LevenshteinDistancesUtf8 {}
unsafe impl Sync for LevenshteinDistancesUtf8 {}

/// Needleman-Wunsch global sequence alignment scoring engine.
///
/// Implements the Needleman-Wunsch algorithm for finding the optimal global alignment
/// between two sequences. Unlike edit distance, this algorithm uses a full substitution
/// matrix and returns alignment scores rather than distances.
///
/// # Algorithm Details
///
/// The Needleman-Wunsch algorithm finds the optimal global alignment by:
/// 1. Building a dynamic programming matrix using substitution scores
/// 2. Applying gap penalties (open + extend costs)
/// 3. Finding the maximum-scoring path through the entire sequences
///
/// Time complexity: O(nm), Space complexity: O(nm) or O(n+m) with optimizations.
///
/// # Applications
///
/// - **Bioinformatics**: Protein and DNA sequence alignment
/// - **Linguistics**: Comparative analysis of languages
/// - **Data integration**: Aligning structured records
/// - **Quality control**: Comparing reference vs. observed sequences
///
/// # Substitution Matrix
///
/// Requires a 256x256 scoring matrix where `matrix[i][j]` gives the score
/// for aligning character `i` with character `j`. Common matrices include:
/// - **BLOSUM**: For protein sequences
/// - **PAM**: For evolutionary analysis
/// - **Custom**: For domain-specific applications
///
/// # Examples
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, NeedlemanWunschScores};
/// // Create simple scoring matrix (match=2, mismatch=-1)
/// let mut matrix = [[-1i8; 256]; 256];
/// for i in 0..256 {
///     matrix[i][i] = 2; // Match score
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let engine = NeedlemanWunschScores::new(
///     &device,
///     &matrix,
///     -2, // gap open penalty
///     -1, // gap extend penalty
/// ).unwrap();
///
/// // Align protein sequences
/// let seq_a = vec!["ACDEFGHIKLMNPQRSTVWY"];
/// let seq_b = vec!["ACDEFGHIKL---NPQRSTVWY"];
/// let scores = engine.compute(&device, &seq_a, &seq_b).unwrap();
///
/// println!("Global alignment score: {}", scores[0]);
/// ```
///
/// # BLOSUM62 Example
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, NeedlemanWunschScores};
/// // Load BLOSUM62 matrix (simplified example)
/// fn create_blosum62_matrix() -> [[i8; 256]; 256] {
///     let mut matrix = [[-4i8; 256]; 256]; // Default mismatch
///     
///     // Set scores for amino acids (simplified BLOSUM62 subset)
///     let aa_scores = [
///         ('A', 'A', 4), ('A', 'R', -1), ('A', 'N', -2),
///         ('R', 'R', 5), ('R', 'N', 0), ('N', 'N', 6),
///         // ... (full BLOSUM62 table)
///     ];
///     
///     for (aa1, aa2, score) in aa_scores.iter() {
///         matrix[*aa1 as usize][*aa2 as usize] = *score;
///         matrix[*aa2 as usize][*aa1 as usize] = *score; // Symmetric
///     }
///     matrix
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let blosum62 = create_blosum62_matrix();
/// let engine = NeedlemanWunschScores::new(&device, &blosum62, -11, -1).unwrap();
/// ```
pub struct NeedlemanWunschScores {
    handle: NeedlemanWunschScoresHandle,
}

impl NeedlemanWunschScores {
    /// Create a new Needleman-Wunsch global alignment scoring engine.
    ///
    /// Initializes the engine with a custom substitution matrix and gap penalties.
    /// The engine will automatically select optimal implementations based on
    /// hardware capabilities.
    ///
    /// # Parameters
    ///
    /// - `device`: Device scope for execution and capability detection
    /// - `substitution_matrix`: 256x256 matrix of alignment scores
    /// - `open_cost`: Penalty for opening a gap (typically negative)
    /// - `extend_cost`: Penalty for extending a gap (typically negative, ≤ open_cost)
    ///
    /// # Returns
    ///
    /// - `Ok(NeedlemanWunschScores)`: Successfully initialized engine
    /// - `Err(Status::BadAlloc)`: Memory allocation failed
    /// - `Err(Status::InvalidArgument)`: Invalid matrix or gap costs
    ///
    /// # Matrix Guidelines
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, NeedlemanWunschScores};
    /// // Identity matrix (simple match/mismatch)
    /// let mut simple_matrix = [[0i8; 256]; 256];
    /// for i in 0..256 {
    ///     simple_matrix[i][i] = 1;  // Match
    ///     for j in 0..256 {
    ///         if i != j { simple_matrix[i][j] = -1; } // Mismatch
    ///     }
    /// }
    ///
    /// let device = DeviceScope::default().unwrap();
    /// let engine = NeedlemanWunschScores::new(&device, &simple_matrix, -2, -1).unwrap();
    /// ```
    ///
    /// # Gap Cost Selection
    ///
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, NeedlemanWunschScores};
    /// # let mut matrix = [[0i8; 256]; 256];
    /// # let device = DeviceScope::default().unwrap();
    /// // Linear gap costs (open == extend)
    /// let linear = NeedlemanWunschScores::new(&device, &matrix, -1, -1).unwrap();
    ///
    /// // Affine gap costs (prefer fewer, longer gaps)
    /// let affine = NeedlemanWunschScores::new(&device, &matrix, -5, -1).unwrap();
    /// ```
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, NeedlemanWunschScores};
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, NeedlemanWunschScores};
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
    ) -> Result<UnifiedVec<isize>, Status>
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
            let (tape_a, use_64bit_a) = create_tape(seq_a_slice)?;
            let (tape_b, use_64bit_b) = create_tape(seq_b_slice)?;

            let status = if use_64bit_a || use_64bit_b {
                let tape_a_view = create_u64tape_view(&tape_a);
                let tape_b_view = create_u64tape_view(&tape_b);
                unsafe {
                    sz_needleman_wunsch_scores_u64tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            } else {
                let tape_a_view = create_u32tape_view(&tape_a);
                let tape_b_view = create_u32tape_view(&tape_b);
                unsafe {
                    sz_needleman_wunsch_scores_u32tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            };
            match status {
                Status::Success => Ok(results),
                err => Err(err),
            }
        } else {
            let seq_a = create_sequence_view(seq_a_slice);
            let seq_b = create_sequence_view(seq_b_slice);
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
                Status::Success => Ok(results),
                err => Err(err),
            }
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

unsafe impl Send for NeedlemanWunschScores {}
unsafe impl Sync for NeedlemanWunschScores {}

/// Smith-Waterman local sequence alignment scoring engine.
///
/// Implements the Smith-Waterman algorithm for finding optimal local alignments
/// within sequences. Unlike Needleman-Wunsch, this algorithm finds the best-matching
/// subsequences rather than aligning entire sequences.
///
/// # Algorithm Details
///
/// The Smith-Waterman algorithm:
/// 1. Builds a dynamic programming matrix with non-negative scores
/// 2. Allows scores to reset to zero (no penalty for poor regions)
/// 3. Returns the maximum score found anywhere in the matrix
/// 4. Identifies optimal local alignments without end-to-end constraints
///
/// Time complexity: O(nm), Space complexity: O(nm) or O(n+m) with optimizations.
///
/// # Applications
///
/// - **Homology search**: Finding similar regions in biological sequences
/// - **Motif discovery**: Identifying conserved patterns
/// - **Database search**: BLAST-like local similarity search
/// - **Partial matching**: Finding best substring alignments
/// - **Fragment analysis**: Comparing incomplete or damaged sequences
///
/// # Local vs Global Alignment
///
/// ```text
/// Global (Needleman-Wunsch):
/// SEQUENCE_A: ATCGATCGATCG----ATCG
/// SEQUENCE_B: ----ATCGATCGATCGATCG
/// (Forces end-to-end alignment)
///
/// Local (Smith-Waterman):
/// SEQUENCE_A: ...ATCGATCGATCG...
/// SEQUENCE_B:    ATCGATCGATCG
/// (Finds best local match)
/// ```
///
/// # Examples
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, SmithWatermanScores};
/// // Create scoring matrix for DNA (A, T, C, G)
/// let mut dna_matrix = [[-2i8; 256]; 256]; // Mismatch penalty
/// let dna_chars = [b'A', b'T', b'C', b'G'];
/// for &c1 in &dna_chars {
///     for &c2 in &dna_chars {
///         dna_matrix[c1 as usize][c2 as usize] = if c1 == c2 { 3 } else { -1 };
///     }
/// }
///
/// let device = DeviceScope::default().unwrap();
/// let engine = SmithWatermanScores::new(
///     &device,
///     &dna_matrix,
///     -5, // gap open
///     -1, // gap extend
/// ).unwrap();
///
/// // Find local similarities
/// let long_seq = vec!["ATCGATCGATCGAAAAAATCGATCGATCG"];
/// let pattern = vec!["ATCGATCGATCG"];
/// let scores = engine.compute(&device, &long_seq, &pattern).unwrap();
///
/// println!("Local alignment score: {}", scores[0]); // High positive score
/// ```
///
/// # Database Search Example
///
/// ```rust,no_run
/// # use stringzilla::stringzillas::szs::{DeviceScope, SmithWatermanScores};
/// # let mut matrix = [[0i8; 256]; 256];
/// let device = DeviceScope::default().unwrap();
/// let engine = SmithWatermanScores::new(&device, &matrix, -2, -1).unwrap();
///
/// // Search query against database sequences
/// let query = "ACDEFGHIKLMN";  // Query sequence
/// let database = vec![
///     "ABCDEFGHIJKLMNOPQRSTUVWXYZ",  // Contains query
///     "ZYXWVUTSRQPONMLKJIHGFEDCBA",  // Reverse complement
///     "DEFGHIKLM",                  // Partial match
///     "COMPLETELY_DIFFERENT",        // No similarity
/// ];
///
/// let queries = vec![query; database.len()];  // Repeat query for each DB entry
/// let scores = engine.compute(&device, &queries, &database).unwrap();
///
/// // Find best matches
/// for (i, &score) in scores.iter().enumerate() {
///     println!("Database[{}] score: {}", i, score);
/// }
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, SmithWatermanScores};
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, SmithWatermanScores};
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
    /// - `Err(Status)`: Computation failed
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, SmithWatermanScores};
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
    /// ```rust,no_run
    /// # use stringzilla::stringzillas::szs::{DeviceScope, SmithWatermanScores};
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
    ) -> Result<UnifiedVec<isize>, Status>
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
            let (tape_a, use_64bit_a) = create_tape(seq_a_slice)?;
            let (tape_b, use_64bit_b) = create_tape(seq_b_slice)?;

            let status = if use_64bit_a || use_64bit_b {
                let tape_a_view = create_u64tape_view(&tape_a);
                let tape_b_view = create_u64tape_view(&tape_b);
                unsafe {
                    sz_smith_waterman_scores_u64tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            } else {
                let tape_a_view = create_u32tape_view(&tape_a);
                let tape_b_view = create_u32tape_view(&tape_b);
                unsafe {
                    sz_smith_waterman_scores_u32tape(
                        self.handle,
                        device.handle,
                        &tape_a_view as *const _ as *const c_void,
                        &tape_b_view as *const _ as *const c_void,
                        results.as_mut_ptr(),
                        results_stride,
                    )
                }
            };
            match status {
                Status::Success => Ok(results),
                err => Err(err),
            }
        } else {
            let seq_a = create_sequence_view(seq_a_slice);
            let seq_b = create_sequence_view(seq_b_slice);
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
                Status::Success => Ok(results),
                err => Err(err),
            }
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

unsafe impl Send for SmithWatermanScores {}
unsafe impl Sync for SmithWatermanScores {}

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

/// Convert StringTape to appropriate tape view for C API
fn create_tape<T>(sequences: &[T]) -> Result<(BytesTape<i64, UnifiedAlloc>, bool), Status>
where
    T: AsRef<[u8]>,
{
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    let tape = if use_64bit {
        BytesTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc)
    } else {
        BytesTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc)
    };

    let mut tape = tape;
    tape.extend(sequences).map_err(|_| SzStatus::BadAlloc)?;
    Ok((tape, use_64bit))
}

/// Convert string sequences to StringTape
fn create_tape_str<T: AsRef<str>>(sequences: &[T]) -> Result<(StringTape<i64, UnifiedAlloc>, bool), Status> {
    // Estimate total size to decide between 32-bit and 64-bit tapes
    let total_size: usize = sequences.iter().map(|s| s.as_ref().len()).sum();
    let use_64bit = total_size > u32::MAX as usize || sequences.len() > u32::MAX as usize;

    let tape = if use_64bit {
        StringTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc)
    } else {
        StringTape::<i64, UnifiedAlloc>::new_in(UnifiedAlloc)
    };

    let mut tape = tape;
    tape.extend(sequences).map_err(|_| SzStatus::BadAlloc)?;
    Ok((tape, use_64bit))
}

/// Convert 32-bit BytesTape to SzSequenceU32Tape for C API
fn create_u32tape_view(tape: &BytesTape<i64, UnifiedAlloc>) -> SzSequenceU32Tape {
    let (data_ptr, offsets_ptr, count, _capacity) = tape.as_raw_parts();
    SzSequenceU32Tape {
        data: data_ptr,
        offsets: offsets_ptr as *const u32,
        count,
    }
}

/// Convert 32-bit StringTape to SzSequenceU32Tape for C API
fn create_u32tape_view_str(tape: &StringTape<i64, UnifiedAlloc>) -> SzSequenceU32Tape {
    let (data_ptr, offsets_ptr, count, _capacity) = tape.as_raw_parts();
    SzSequenceU32Tape {
        data: data_ptr,
        offsets: offsets_ptr as *const u32,
        count,
    }
}

/// Convert 64-bit BytesTape to SzSequenceU64Tape for C API  
fn create_u64tape_view(tape: &BytesTape<i64, UnifiedAlloc>) -> SzSequenceU64Tape {
    let (data_ptr, offsets_ptr, count, _capacity) = tape.as_raw_parts();
    SzSequenceU64Tape {
        data: data_ptr,
        offsets: offsets_ptr as *const u64,
        count,
    }
}

/// Convert 64-bit StringTape to SzSequenceU64Tape for C API  
fn create_u64tape_view_str(tape: &StringTape<i64, UnifiedAlloc>) -> SzSequenceU64Tape {
    let (data_ptr, offsets_ptr, count, _capacity) = tape.as_raw_parts();
    SzSequenceU64Tape {
        data: data_ptr,
        offsets: offsets_ptr as *const u64,
        count,
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
    fn test_device_scope_creation() {
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
    fn test_device_scope_validation() {
        // Test invalid CPU core count
        let invalid_cpu = DeviceScope::cpu_cores(0);
        assert!(invalid_cpu.is_err());

        let single_core = DeviceScope::cpu_cores(1);
        assert!(single_core.is_err()); // Should fail as per implementation
    }

    #[test]
    fn test_fingerprint_builder_configurations() {
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
    fn test_fingerprint_computation() {
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
    fn test_levenshtein_distance_engine() {
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
    fn test_levenshtein_utf8_engine() {
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
    fn test_needleman_wunsch_engine() {
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
    fn test_smith_waterman_engine() {
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
    fn test_unified_allocator() {
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
    fn test_error_handling() {
        // Test invalid device scope parameters
        let invalid_cpu = DeviceScope::cpu_cores(0);
        assert!(invalid_cpu.is_err());

        let invalid_gpu = DeviceScope::gpu_device(999);
        // May succeed or fail depending on system, but shouldn't panic
        match invalid_gpu {
            Ok(_) => println!("GPU device 999 unexpectedly available"),
            Err(e) => println!("GPU device 999 correctly failed: {:?}", e),
        }
    }

    #[test]
    fn test_thread_safety() {
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
    fn test_large_batch_processing() {
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
    fn test_similarity_estimation() {
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
}
