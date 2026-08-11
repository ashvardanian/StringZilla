//! Min-Hash and Count-Min-Sketch fingerprinting engines.

use super::types::{
    copy_bytes_into_tape, rust_error_from_c_message, SzSequenceFromBytes, SzSequenceU32Tape, SzSequenceU64Tape,
};
use super::*;
use alloc::vec::Vec;
use core::ffi::{c_char, c_void};
use core::ptr;

/// Opaque handle for the fingerprinting engine
pub type FingerprintsHandle = *mut c_void;

// C API bindings
extern "C" {

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

}

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
    ///     if hashes[i] == hashes[1 * dimensions + i] {
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
    pub fn compute<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        strings: Sequences,
        dimensions: usize,
    ) -> Result<(UnifiedVec<u32>, UnifiedVec<u32>), Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzillas::fixtures::device_or_skip;

    const TEST_FINGERPRINT_DIMS_SMALL: usize = 64;
    const TEST_FINGERPRINT_DIMS_LARGE: usize = 128;
    const TEST_LARGE_BATCH_SIZE: usize = 1000;

    #[test]
    fn fingerprint_builder_configurations() {
        let Some(device) = device_or_skip("fingerprint_builder_configurations") else {
            return;
        };

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
        let Some(device) = device_or_skip("fingerprint_computation") else {
            return;
        };

        let engine = Fingerprints::builder()
            .binary()
            .dimensions(TEST_FINGERPRINT_DIMS_SMALL) // Small dimensions for testing
            .build(&device)
            .expect("binary fingerprint engine should build on CPU");

        // Test basic computation
        let test_strings = vec!["hello", "world", "test"];
        let (hashes, counts) = engine
            .compute(&device, &test_strings, TEST_FINGERPRINT_DIMS_SMALL)
            .expect("fingerprint computation should succeed on CPU");
        assert_eq!(hashes.len(), 3 * TEST_FINGERPRINT_DIMS_SMALL);
        assert_eq!(counts.len(), 3 * TEST_FINGERPRINT_DIMS_SMALL);
    }

    #[test]
    fn thread_safety() {
        use std::sync::Arc;
        use std::thread;

        const THREAD_COUNT: usize = 4;

        let Some(device) = device_or_skip("thread_safety") else {
            return;
        };
        let device = Arc::new(device);

        let engine = Fingerprints::builder()
            .dimensions(TEST_FINGERPRINT_DIMS_SMALL)
            .build(&device)
            .expect("fingerprint engine should build on CPU");
        let engine = Arc::new(engine);

        // Test parallel computation
        let handles: Vec<_> = (0..THREAD_COUNT)
            .map(|i| {
                let device = Arc::clone(&device);
                let engine = Arc::clone(&engine);
                thread::spawn(move || {
                    let test_data = vec![format!("thread_{}_data", i)];
                    engine.compute(&device, &test_data, TEST_FINGERPRINT_DIMS_SMALL)
                })
            })
            .collect();

        let mut success_count = 0;
        for handle in handles {
            match handle.join().expect("worker thread should not panic") {
                Ok(_) => success_count += 1,
                Err(e) => println!("Thread computation failed: {:?}", e),
            }
        }

        assert_eq!(success_count, THREAD_COUNT, "not all threads succeeded");
    }

    #[test]
    fn large_batch_processing() {
        let Some(device) = device_or_skip("large_batch_processing") else {
            return;
        };

        let engine = Fingerprints::builder()
            .dimensions(TEST_FINGERPRINT_DIMS_SMALL)
            .build(&device)
            .expect("fingerprint engine should build on CPU");

        // Create large batch
        let large_batch: Vec<String> = (0..TEST_LARGE_BATCH_SIZE)
            .map(|i| format!("test_string_{}", i))
            .collect();
        let large_batch_refs: Vec<&str> = large_batch.iter().map(|s| s.as_str()).collect();

        let (hashes, counts) = engine
            .compute(&device, &large_batch_refs, TEST_FINGERPRINT_DIMS_SMALL)
            .expect("fingerprint computation should succeed on CPU");
        assert_eq!(hashes.len(), TEST_LARGE_BATCH_SIZE * TEST_FINGERPRINT_DIMS_SMALL);
        assert_eq!(counts.len(), TEST_LARGE_BATCH_SIZE * TEST_FINGERPRINT_DIMS_SMALL);
    }

    #[test]
    fn similarity_estimation() {
        let Some(device) = device_or_skip("similarity_estimation") else {
            return;
        };

        let engine = Fingerprints::builder()
            .dimensions(TEST_FINGERPRINT_DIMS_LARGE)
            .build(&device)
            .expect("fingerprint engine should build on CPU");

        let test_strings = vec![
            "the quick brown fox",
            "the quick brown fox",  // Identical
            "the quick brown dog",  // Similar
            "completely different", // Different
        ];

        let (hashes, _counts) = engine
            .compute(&device, &test_strings, TEST_FINGERPRINT_DIMS_LARGE)
            .expect("fingerprint computation should succeed on CPU");
        {
            let dimensions = TEST_FINGERPRINT_DIMS_LARGE;

            // Compare identical strings (should have high similarity)
            let mut matches_identical = 0;
            for i in 0..dimensions {
                if hashes[i] == hashes[1 * dimensions + i] {
                    matches_identical += 1;
                }
            }
            let similarity_identical = matches_identical as f64 / dimensions as f64;

            // Compare similar strings
            let mut matches_similar = 0;
            for i in 0..dimensions {
                if hashes[i] == hashes[2 * dimensions + i] {
                    matches_similar += 1;
                }
            }
            let similarity_similar = matches_similar as f64 / dimensions as f64;

            // Compare different strings
            let mut matches_different = 0;
            for i in 0..dimensions {
                if hashes[i] == hashes[3 * dimensions + i] {
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
    }
}
