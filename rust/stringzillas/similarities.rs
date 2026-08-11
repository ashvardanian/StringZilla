//! Batch edit-distance and sequence-alignment engines.

use super::types::{
    copy_bytes_into_tape, copy_chars_into_tape, rust_error_from_c_message, should_use_64bit_for_bytes,
    should_use_64bit_for_strings, SzSequenceFromBytes, SzSequenceFromChars, SzSequenceU32Tape, SzSequenceU64Tape,
};
use super::*;
use core::ffi::{c_char, c_void};
use core::ptr;

/// Opaque handles for similarity engines
pub type LevenshteinDistancesHandle = *mut c_void;
pub type LevenshteinDistancesUtf8Handle = *mut c_void;
pub type NeedlemanWunschScoresHandle = *mut c_void;
pub type SmithWatermanScoresHandle = *mut c_void;

// C API bindings
extern "C" {

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
    pub fn compute<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        queries: Sequences,
        candidates: Sequences,
    ) -> Result<UnifiedMat<usize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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
    pub fn compute_symmetric<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        sequences: Sequences,
    ) -> Result<UnifiedMat<usize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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
    fn compute_pair<Sequence>(
        &self,
        device: &DeviceScope,
        queries: &[Sequence],
        candidates: Option<&[Sequence]>,
        matrix: &mut UnifiedMat<usize>,
    ) -> Result<(), Error>
    where
        Sequence: AsRef<[u8]>,
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
    pub fn compute<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        queries: Sequences,
        candidates: Sequences,
    ) -> Result<UnifiedMat<usize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<str>,
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
    pub fn compute_symmetric<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        sequences: Sequences,
    ) -> Result<UnifiedMat<usize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<str>,
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
    fn compute_pair<Sequence>(
        &self,
        device: &DeviceScope,
        queries: &[Sequence],
        candidates: Option<&[Sequence]>,
        matrix: &mut UnifiedMat<usize>,
    ) -> Result<(), Error>
    where
        Sequence: AsRef<str>,
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
    pub fn compute<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        queries: Sequences,
        candidates: Sequences,
    ) -> Result<UnifiedMat<isize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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
    pub fn compute_symmetric<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        sequences: Sequences,
    ) -> Result<UnifiedMat<isize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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
    fn compute_pair<Sequence>(
        &self,
        device: &DeviceScope,
        queries: &[Sequence],
        candidates: Option<&[Sequence]>,
        matrix: &mut UnifiedMat<isize>,
    ) -> Result<(), Error>
    where
        Sequence: AsRef<[u8]>,
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
    pub fn compute<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        queries: Sequences,
        candidates: Sequences,
    ) -> Result<UnifiedMat<isize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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
    pub fn compute_symmetric<Sequences, Sequence>(
        &self,
        device: &DeviceScope,
        sequences: Sequences,
    ) -> Result<UnifiedMat<isize>, Error>
    where
        Sequences: AsRef<[Sequence]>,
        Sequence: AsRef<[u8]>,
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
    fn compute_pair<Sequence>(
        &self,
        device: &DeviceScope,
        queries: &[Sequence],
        candidates: Option<&[Sequence]>,
        matrix: &mut UnifiedMat<isize>,
    ) -> Result<(), Error>
    where
        Sequence: AsRef<[u8]>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzillas::fixtures::device_or_skip;
    use stringtape::BytesTape;

    #[test]
    fn levenshtein_distance_engine() {
        let Some(device) = device_or_skip("levenshtein_distance_engine") else {
            return;
        };

        // Test engine creation
        let engine = LevenshteinDistances::new(
            &device, 0, // match cost
            1, // mismatch cost
            1, // open cost
            1, // extend cost
        )
        .expect("Levenshtein engine should build on CPU");

        // Cross-product distance computation over a non-square query/candidate set.
        let queries = vec!["kitten", "saturday"];
        let candidates = vec!["sitting", "sunday", "kitten"];
        let matrix = engine
            .compute(&device, &queries, &candidates)
            .expect("Levenshtein computation should succeed on CPU");
        assert_eq!(matrix.dimensions(), (2, 3));
        // kitten -> sitting is 3 (substitute k->s, e->i, insert g)
        assert_eq!(matrix[(0, 0)], 3);
        // kitten -> kitten is 0 (identical)
        assert_eq!(matrix[(0, 2)], 0);
        // saturday -> sunday is 3 (delete a,t,r)
        assert_eq!(matrix[(1, 1)], 3);
        // The row slice mirrors the indexed cells.
        assert_eq!(matrix.row(0), &[3usize, 6, 0][..]);
    }

    #[test]
    fn levenshtein_distance_symmetric() {
        let Some(device) = device_or_skip("levenshtein_distance_symmetric") else {
            return;
        };
        let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).expect("Levenshtein engine should build on CPU");

        let sequences = vec!["cat", "bat", "cart"];
        let matrix = engine
            .compute_symmetric(&device, &sequences)
            .expect("symmetric Levenshtein computation should succeed on CPU");

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
        let single_matrix = engine
            .compute(&device, &single, &single)
            .expect("Levenshtein computation should succeed on CPU");
        assert_eq!(single_matrix[(0, 0)], matrix[(0, 0)]);
    }

    #[test]
    fn levenshtein_utf8_engine() {
        let Some(device) = device_or_skip("levenshtein_utf8_engine") else {
            return;
        };

        let engine =
            LevenshteinDistancesUtf8::new(&device, 0, 1, 1, 1).expect("UTF-8 Levenshtein engine should build on CPU");

        // Cross-product over Unicode strings.
        let queries = vec!["café", "naïve"];
        let candidates = vec!["cafe", "naive"];
        let matrix = engine
            .compute(&device, &queries, &candidates)
            .expect("UTF-8 Levenshtein computation should succeed on CPU");
        assert_eq!(matrix.dimensions(), (2, 2));
        // Each accented character counts as one character-level substitution.
        assert_eq!(matrix[(0, 0)], 1); // café vs cafe
        assert_eq!(matrix[(1, 1)], 1); // naïve vs naive
    }

    #[test]
    fn needleman_wunsch_engine() {
        let Some(device) = device_or_skip("needleman_wunsch_engine") else {
            return;
        };

        // Create a simple class-based scoring scheme
        let (byte_to_class, class_costs) = error_costs_classes_diagonal(2, -1);

        let engine = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1)
            .expect("Needleman-Wunsch engine should build on CPU");

        let queries = vec!["ACGT", "ACGT"];
        let candidates = vec!["ACGT", "TTTT"];
        let matrix = engine
            .compute(&device, &queries, &candidates)
            .expect("Needleman-Wunsch computation should succeed on CPU");
        assert_eq!(matrix.dimensions(), (2, 2));
        // Identical sequences score higher than the mismatching pair.
        assert!(matrix[(0, 0)] > matrix[(0, 1)]);
        assert!(matrix[(0, 0)] > 0, "Identical sequences should score positively");

        // Symmetric self-alignment matrix.
        let sequences = vec!["ACGT", "AGGT", "TTTT"];
        let symmetric = engine
            .compute_symmetric(&device, &sequences)
            .expect("Needleman-Wunsch symmetric computation should succeed on CPU");
        assert_eq!(symmetric.dimensions(), (3, 3));
        for first_index in 0..3 {
            for second_index in 0..3 {
                assert_eq!(
                    symmetric[(first_index, second_index)],
                    symmetric[(second_index, first_index)]
                );
            }
        }
        // The diagonal self-alignment score is the strongest in its row.
        for diagonal_index in 0..3 {
            let diagonal_score = symmetric[(diagonal_index, diagonal_index)];
            for candidate_index in 0..3 {
                assert!(diagonal_score >= symmetric[(diagonal_index, candidate_index)]);
            }
        }
    }

    #[test]
    fn smith_waterman_engine() {
        let Some(device) = device_or_skip("smith_waterman_engine") else {
            return;
        };

        // Create a simple class-based scoring scheme
        let (byte_to_class, class_costs) = error_costs_classes_diagonal(3, -1);

        let engine = SmithWatermanScores::new(&device, &byte_to_class, &class_costs, -2, -1)
            .expect("Smith-Waterman engine should build on CPU");

        // One query against two candidates yields a single-row cross-product matrix.
        let queries = vec!["ACGTACGT"];
        let candidates = vec!["ACGT", "TTTT"];
        let matrix = engine
            .compute(&device, &queries, &candidates)
            .expect("Smith-Waterman computation should succeed on CPU");
        assert_eq!(matrix.dimensions(), (1, 2));
        // The embedded "ACGT" run aligns better than the all-mismatch candidate.
        assert!(matrix[(0, 0)] > matrix[(0, 1)]);
        assert!(matrix[(0, 0)] > 0, "Local alignment should be positive");

        // Symmetric self-alignment matrix is symmetric with a dominant diagonal.
        let sequences = vec!["ACGTACGT", "ACGT", "TTTT"];
        let symmetric = engine
            .compute_symmetric(&device, &sequences)
            .expect("Smith-Waterman symmetric computation should succeed on CPU");
        assert_eq!(symmetric.dimensions(), (3, 3));
        for first_index in 0..3 {
            for second_index in 0..3 {
                assert_eq!(
                    symmetric[(first_index, second_index)],
                    symmetric[(second_index, first_index)]
                );
            }
        }
    }

    #[test]
    fn error_costs_for_needleman_wunsch() {
        let Some(device) = device_or_skip("error_costs_for_needleman_wunsch") else {
            return;
        };

        // Test our diagonal class-based scoring function with NW aligner
        let (byte_to_class, class_costs) = error_costs_classes_diagonal(2, -1);
        let engine = NeedlemanWunschScores::new(&device, &byte_to_class, &class_costs, -2, -1)
            .expect("Needleman-Wunsch engine should build on CPU");

        let queries = vec!["ABCD"];
        let candidates = vec!["ABCD"];
        let matrix = engine
            .compute(&device, &queries, &candidates)
            .expect("Needleman-Wunsch computation should succeed on CPU");
        assert_eq!(matrix.dimensions(), (1, 1));
        assert!(matrix[(0, 0)] > 0, "Identical sequences should have positive score");
    }

    #[test]
    fn levenshtein_compute_into_u32_bytes() {
        let Some(device) = device_or_skip("levenshtein_compute_into_u32_bytes") else {
            return;
        };
        let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).expect("Levenshtein engine should build on CPU");

        let queries = [b"kitten".as_ref(), b"saturday".as_ref()];
        let candidates = [b"sitting".as_ref(), b"sunday".as_ref()];

        let mut queries_tape = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        queries_tape.extend(queries).unwrap();
        let mut candidates_tape = BytesTape::<u32, UnifiedAlloc>::new_in(UnifiedAlloc);
        candidates_tape.extend(candidates).unwrap();

        let mut matrix = UnifiedMat::<usize>::try_allocate(2, 2).expect("matrix allocation");

        engine
            .compute_into(
                &device,
                AnyBytesTape::Tape32(queries_tape),
                Some(AnyBytesTape::Tape32(candidates_tape)),
                &mut matrix,
            )
            .expect("Levenshtein compute_into should succeed on CPU");
        // Diagonal: kitten vs sitting = 3, saturday vs sunday = 3.
        assert_eq!(matrix[(0, 0)], 3);
        assert_eq!(matrix[(1, 1)], 3);
    }

    #[test]
    fn levenshtein_compute_into_u64_bytes() {
        let Some(device) = device_or_skip("levenshtein_compute_into_u64_bytes") else {
            return;
        };
        let engine = LevenshteinDistances::new(&device, 0, 1, 1, 1).expect("Levenshtein engine should build on CPU");

        let queries = [b"abc".as_ref(), b"abcdef".as_ref()];
        let candidates = [b"yabd".as_ref(), b"abcxef".as_ref()];

        let mut queries_tape = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        queries_tape.extend(queries).unwrap();
        let mut candidates_tape = BytesTape::<u64, UnifiedAlloc>::new_in(UnifiedAlloc);
        candidates_tape.extend(candidates).unwrap();

        let mut matrix = UnifiedMat::<usize>::try_allocate(2, 2).expect("matrix allocation");

        engine
            .compute_into(
                &device,
                AnyBytesTape::Tape64(queries_tape),
                Some(AnyBytesTape::Tape64(candidates_tape)),
                &mut matrix,
            )
            .expect("Levenshtein compute_into should succeed on CPU");
        // Diagonal: abc vs yabd => 2, abcdef vs abcxef => 1.
        assert_eq!(matrix[(0, 0)], 2);
        assert_eq!(matrix[(1, 1)], 1);
    }
}
