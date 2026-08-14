//! Multi-pattern substring search engines - one compiled dictionary of needles walked over many haystacks.

use super::types::{rust_error_from_c_message, SzSequenceFromBytes};
use super::*;
use core::ffi::{c_char, c_void};
use core::ptr;

/// Opaque handle for the multi-pattern search engine
pub type SubstringsHandle = *mut c_void;

/// Whether a dictionary matches needles byte-for-byte or folds both sides to a shared case first.
///
/// Corresponds to `szs_substrings_case_sensitivity_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseSensitivity {
    /// Byte-exact matching; needles may be arbitrary bytes.
    Cased = 0,
    /// Full Unicode case folding; needles must be valid UTF-8.
    Uncased = 1,
}

/// How overlapping matches resolve: reported in full, or reduced to a non-overlapping cover.
///
/// The three states map one-to-one onto the `MatchKind` trio of the reference Rust engines. The
/// policy travels per call rather than per engine, since it never shapes the compiled automaton -
/// one dictionary serves all three.
///
/// Corresponds to `szs_substrings_overlap_policy_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapPolicy {
    /// Every match of every needle, including overlapping and nested ones.
    Overlapping = 0,
    /// Non-overlapping cover: earliest start, then longest span, then lower needle index.
    LeftmostLongest = 1,
    /// Non-overlapping cover: earliest start, then lower needle index, even when a longer needle matches.
    LeftmostFirst = 2,
}

/// One reported match, locating it by haystack, by needle, and by byte span.
///
/// Under case folding a needle's own byte length is not the length of every match - needle "k" matches
/// both the 1-byte "k" and the 3-byte Kelvin sign - so the span is carried per match.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SubstringsMatch {
    /// Which haystack the match was found in.
    pub haystack_index: usize,
    /// Which needle matched.
    pub needle_index: usize,
    /// Offset of the match within its haystack, in bytes.
    pub byte_offset: usize,
    /// Length of the matched span, in bytes.
    pub byte_length: usize,
}

/// Classic BM25's continuous parameters.
///
/// There is no correct default for the corpus mean, so there is no `Default`: reach for
/// [`Bm25Params::normalized`] or [`Bm25Params::unnormalized`], which name the two configurations that
/// exist. A positive `length_normalization` with a non-positive `average_document_length` is refused,
/// since it would divide by a mean that is not there.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// The literature's `k1`: how slowly repeated occurrences stop adding score; 1.2 is customary.
    pub term_frequency_saturation: f32,
    /// The literature's `b`, in [0, 1]: 0 ignores document length and every length input with it,
    /// 1 normalizes fully; 0.75 is customary.
    pub length_normalization: f32,
    /// Corpus-wide mean of the document lengths, in the same unit; never derived from the batch, and
    /// read only when `length_normalization` is positive.
    pub average_document_length: f32,
}

impl Bm25Params {
    /// Classic BM25 with the literature's customary `k1 = 1.2` and `b = 0.75`, normalized against a
    /// corpus whose mean document length is `average_document_length`, in the unit the per-document
    /// lengths use.
    pub fn normalized(average_document_length: f32) -> Self {
        Self {
            term_frequency_saturation: 1.2,
            length_normalization: 0.75,
            average_document_length,
        }
    }

    /// BM25 with length normalization switched off, for a corpus whose mean length is unknown or whose
    /// documents are uniform enough not to need it; the per-document lengths go unread.
    pub fn unnormalized() -> Self {
        Self {
            term_frequency_saturation: 1.2,
            length_normalization: 0.0,
            average_document_length: 0.0,
        }
    }
}

// C API bindings
extern "C" {

    fn szs_substrings_init(
        alloc: *const c_void, // MemoryAllocator - using null for default
        capabilities: Capability,
        engine: *mut SubstringsHandle,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_index(
        engine: SubstringsHandle,
        needles: *const c_void, // SzSequence
        case_sensitivity: CaseSensitivity,
        device: *mut c_void,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_count(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequence
        overlap_policy: OverlapPolicy,
        counts: *mut usize,
        matches_total: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_count_u32tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU32Tape
        overlap_policy: OverlapPolicy,
        counts: *mut usize,
        matches_total: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_count_u64tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU64Tape
        overlap_policy: OverlapPolicy,
        counts: *mut usize,
        matches_total: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_find(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequence
        overlap_policy: OverlapPolicy,
        matches: *mut SubstringsMatch,
        matches_capacity: usize,
        matches_found: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_find_u32tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU32Tape
        overlap_policy: OverlapPolicy,
        matches: *mut SubstringsMatch,
        matches_capacity: usize,
        matches_found: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_find_u64tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU64Tape
        overlap_policy: OverlapPolicy,
        matches: *mut SubstringsMatch,
        matches_capacity: usize,
        matches_found: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_score_bm25(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequence
        document_lengths: *const f32,
        parameters: Bm25Params,
        needle_weights: *const f32,
        scores: *mut f32,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_score_bm25_u32tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU32Tape
        document_lengths: *const f32,
        parameters: Bm25Params,
        needle_weights: *const f32,
        scores: *mut f32,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_score_bm25_u64tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU64Tape
        document_lengths: *const f32,
        parameters: Bm25Params,
        needle_weights: *const f32,
        scores: *mut f32,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_replace_bound(
        engine: SubstringsHandle,
        replacements: *const c_void, // SzSequence
        input_bytes: usize,
        output_bytes_bound: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_replace_u32tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU32Tape
        overlap_policy: OverlapPolicy,
        replacements: *const c_void, // SzSequence
        output_data: *mut u8,
        output_data_capacity: usize,
        output_offsets: *mut u64,
        output_bytes_written: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_replace_u64tape(
        engine: SubstringsHandle,
        device: *mut c_void,
        haystacks: *const c_void, // SzSequenceU64Tape
        overlap_policy: OverlapPolicy,
        replacements: *const c_void, // SzSequence
        output_data: *mut u8,
        output_data_capacity: usize,
        output_offsets: *mut u64,
        output_bytes_written: *mut usize,
        error_message: *mut *const c_char,
    ) -> Status;

    fn szs_substrings_free(engine: SubstringsHandle);
}

/// Multi-pattern exact and case-folded substring search engine.
///
/// Compiles a needle set into an Aho-Corasick automaton once, then reuses it across every later
/// call, so the dictionary is paid for once rather than per haystack. Every needle is tested
/// against every haystack in a single walk, and matches are resolved under a caller-chosen
/// [`OverlapPolicy`] - every overlapping match, or a non-overlapping leftmost cover. The same
/// automaton also scores haystacks with BM25 and rewrites them by substituting their matches.
///
/// Every verb writes into caller-owned buffers, so a pipeline allocates once and reuses those buffers
/// across every batch. Haystacks arrive as an [`AnyBytesTape`], which spells all three shapes the C API
/// accepts - a 32- or 64-bit tape, or borrowed slices addressed by callback and copied nowhere. Only
/// [`Substrings::replace_into`] narrows that to the tapes, since a rewrite's product is itself a tape.
///
/// # Examples
///
/// ```rust
/// # use stringzilla::szs::{AnyBytesTape, Substrings, CaseSensitivity, DeviceScope, OverlapPolicy};
/// let device = DeviceScope::cpu_cores(1).unwrap();
/// let engine = Substrings::new(&device, &["needle", "haystack"], CaseSensitivity::Cased).unwrap();
///
/// let documents = vec!["a needle in a haystack", "no matches here"];
/// let haystacks = AnyBytesTape::from_slices(&documents);
///
/// let mut counts = vec![0usize; documents.len()];
/// let total = engine.count_into(&device, &haystacks, OverlapPolicy::Overlapping, &mut counts).unwrap();
/// assert_eq!(counts, vec![2, 0]);
/// assert_eq!(total, 2);
///
/// let mut matches = vec![Default::default(); total];
/// let found = engine.find_into(&device, &haystacks, OverlapPolicy::Overlapping, &mut matches).unwrap();
/// assert_eq!(found, 2);
/// ```
pub struct Substrings {
    handle: SubstringsHandle,
    /// Fixed once the automaton is built, and what every per-needle array is validated against.
    needles_count: usize,
}

impl Substrings {
    /// Compile a dictionary of `needles` into a search engine on `device`'s capabilities.
    ///
    /// Needles must be non-empty, and under [`CaseSensitivity::Uncased`] must be valid UTF-8.
    pub fn new<Sequence>(
        device: &DeviceScope,
        needles: &[Sequence],
        case_sensitivity: CaseSensitivity,
    ) -> Result<Self, Error>
    where
        Sequence: AsRef<[u8]>,
    {
        let capabilities = device.get_capabilities()?;
        let sequence = SzSequenceFromBytes::to_sz_sequence(needles);
        let mut handle: SubstringsHandle = ptr::null_mut();
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe { szs_substrings_init(ptr::null(), capabilities, &mut handle, &mut error_msg) };
        if status != Status::Success {
            return Err(rust_error_from_c_message(status, error_msg));
        }

        // The automaton is tiered against the cache `device` walks through, so indexing names it explicitly.
        let status = unsafe {
            szs_substrings_index(
                handle,
                &sequence as *const _ as *const c_void,
                case_sensitivity,
                device.handle,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(Substrings {
                handle,
                needles_count: needles.len(),
            }),
            // `Substrings` is never constructed on this path, so its `Drop` never runs and the handle
            // would leak; the engine owns allocations from `init` alone, so it has to be freed here.
            err => {
                unsafe { szs_substrings_free(handle) };
                Err(rust_error_from_c_message(err, error_msg))
            }
        }
    }

    /// Needles compiled into the automaton; every reported `needle_index` is below this.
    pub fn needles_count(&self) -> usize {
        self.needles_count
    }

    /// Count matches of every needle in every haystack into `counts`, one total per haystack.
    ///
    /// `counts` must hold one entry per haystack. The tape is borrowed, so one materialized corpus
    /// serves a whole count-then-find-then-replace pipeline without being rebuilt.
    pub fn count_into(
        &self,
        device: &DeviceScope,
        haystacks: &AnyBytesTape<'_>,
        overlap_policy: OverlapPolicy,
        counts: &mut [usize],
    ) -> Result<usize, Error> {
        if counts.len() < Self::haystacks_count(haystacks) {
            return Err(Error::from(Status::UnexpectedDimensions));
        }

        let mut matches_total: usize = 0;
        let mut error_msg: *const c_char = ptr::null();
        let status = match &haystacks {
            AnyBytesTape::Tape64(tape) => {
                let view = SzSequenceU64Tape::from(tape);
                unsafe {
                    szs_substrings_count_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        counts.as_mut_ptr(),
                        &mut matches_total,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View64(borrowed) => {
                let view = SzSequenceU64Tape::from(borrowed);
                unsafe {
                    szs_substrings_count_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        counts.as_mut_ptr(),
                        &mut matches_total,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Tape32(tape) => {
                let view = SzSequenceU32Tape::from(tape);
                unsafe {
                    szs_substrings_count_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        counts.as_mut_ptr(),
                        &mut matches_total,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View32(borrowed) => {
                let view = SzSequenceU32Tape::from(borrowed);
                unsafe {
                    szs_substrings_count_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        counts.as_mut_ptr(),
                        &mut matches_total,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Slices(sequence) => unsafe {
                szs_substrings_count(
                    self.handle,
                    device.handle,
                    sequence as *const _ as *const c_void,
                    overlap_policy,
                    counts.as_mut_ptr(),
                    &mut matches_total,
                    &mut error_msg,
                )
            },
        };
        match status {
            Status::Success => Ok(matches_total),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Locate matches of every needle in every haystack into `matches`, under `overlap_policy`.
    ///
    /// `matches` is sized by the total [`Substrings::count_into`] returns under the same policy. Returns
    /// the number of matches written; a short buffer fails with [`Status::UnexpectedDimensions`] and
    /// writes nothing.
    pub fn find_into(
        &self,
        device: &DeviceScope,
        haystacks: &AnyBytesTape<'_>,
        overlap_policy: OverlapPolicy,
        matches: &mut [SubstringsMatch],
    ) -> Result<usize, Error> {
        let mut matches_found = 0usize;
        let mut error_msg: *const c_char = ptr::null();
        let status = match &haystacks {
            AnyBytesTape::Tape64(tape) => {
                let view = SzSequenceU64Tape::from(tape);
                unsafe {
                    szs_substrings_find_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        matches.as_mut_ptr(),
                        matches.len(),
                        &mut matches_found,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View64(borrowed) => {
                let view = SzSequenceU64Tape::from(borrowed);
                unsafe {
                    szs_substrings_find_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        matches.as_mut_ptr(),
                        matches.len(),
                        &mut matches_found,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Tape32(tape) => {
                let view = SzSequenceU32Tape::from(tape);
                unsafe {
                    szs_substrings_find_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        matches.as_mut_ptr(),
                        matches.len(),
                        &mut matches_found,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View32(borrowed) => {
                let view = SzSequenceU32Tape::from(borrowed);
                unsafe {
                    szs_substrings_find_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        matches.as_mut_ptr(),
                        matches.len(),
                        &mut matches_found,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Slices(sequence) => unsafe {
                szs_substrings_find(
                    self.handle,
                    device.handle,
                    sequence as *const _ as *const c_void,
                    overlap_policy,
                    matches.as_mut_ptr(),
                    matches.len(),
                    &mut matches_found,
                    &mut error_msg,
                )
            },
        };
        match status {
            Status::Success => Ok(matches_found),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Score every haystack against the compiled dictionary in one automaton walk.
    ///
    /// `needle_weights` holds one IDF or boost per needle, and `document_lengths` defaults to byte
    /// lengths when `None`. Term frequencies are raw overlapping counts, which is classic BM25.
    pub fn score_bm25_into(
        &self,
        device: &DeviceScope,
        haystacks: &AnyBytesTape<'_>,
        needle_weights: &[f32],
        document_lengths: Option<&[f32]>,
        parameters: Bm25Params,
        scores: &mut [f32],
    ) -> Result<(), Error> {
        let haystacks_count = Self::haystacks_count(haystacks);
        if needle_weights.len() < self.needles_count
            || scores.len() < haystacks_count
            || document_lengths.is_some_and(|lengths| lengths.len() < haystacks_count)
        {
            return Err(Error::from(Status::UnexpectedDimensions));
        }

        let document_lengths_ptr = document_lengths.map_or(ptr::null(), |lengths| lengths.as_ptr());
        let mut error_msg: *const c_char = ptr::null();
        let status = match haystacks {
            AnyBytesTape::Tape64(tape) => {
                let view = SzSequenceU64Tape::from(tape);
                unsafe {
                    szs_substrings_score_bm25_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        document_lengths_ptr,
                        parameters,
                        needle_weights.as_ptr(),
                        scores.as_mut_ptr(),
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View64(borrowed) => {
                let view = SzSequenceU64Tape::from(borrowed);
                unsafe {
                    szs_substrings_score_bm25_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        document_lengths_ptr,
                        parameters,
                        needle_weights.as_ptr(),
                        scores.as_mut_ptr(),
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Tape32(tape) => {
                let view = SzSequenceU32Tape::from(tape);
                unsafe {
                    szs_substrings_score_bm25_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        document_lengths_ptr,
                        parameters,
                        needle_weights.as_ptr(),
                        scores.as_mut_ptr(),
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View32(borrowed) => {
                let view = SzSequenceU32Tape::from(borrowed);
                unsafe {
                    szs_substrings_score_bm25_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        document_lengths_ptr,
                        parameters,
                        needle_weights.as_ptr(),
                        scores.as_mut_ptr(),
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Slices(sequence) => unsafe {
                szs_substrings_score_bm25(
                    self.handle,
                    device.handle,
                    sequence as *const _ as *const c_void,
                    document_lengths_ptr,
                    parameters,
                    needle_weights.as_ptr(),
                    scores.as_mut_ptr(),
                    &mut error_msg,
                )
            },
        };
        match status {
            Status::Success => Ok(()),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Bound the bytes a rewrite of `input_bytes` can produce, from the dictionary and `replacements`.
    ///
    /// Needs no haystacks, no walk, and no device. Sizing an output buffer to this bound makes
    /// [`Substrings::replace_into`] a single call that cannot be refused, at the cost of
    /// over-allocating whenever the corpus is not entirely the widest-expanding needle.
    pub fn replace_bound<Replacement>(&self, replacements: &[Replacement], input_bytes: usize) -> Result<usize, Error>
    where
        Replacement: AsRef<[u8]>,
    {
        let replacements_sequence = SzSequenceFromBytes::to_sz_sequence(replacements);
        let mut output_bytes_bound = 0usize;
        let mut error_msg: *const c_char = ptr::null();
        let status = unsafe {
            szs_substrings_replace_bound(
                self.handle,
                &replacements_sequence as *const _ as *const c_void,
                input_bytes,
                &mut output_bytes_bound,
                &mut error_msg,
            )
        };
        match status {
            Status::Success => Ok(output_bytes_bound),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Rewrite every haystack of a tape into caller-owned buffers, substituting matches.
    ///
    /// Tape in, tape out: `overlap_policy` must name a non-overlapping cover, replacements are indexed
    /// by needle and inserted verbatim, and an empty replacement deletes. `output_offsets` takes
    /// `haystacks.len() + 1` entries and is 64-bit so the pair reads directly as a `BytesTapeView<u64>`.
    /// Returns the bytes written; size `output_data` with [`Substrings::replace_bound`] to be sure the
    /// call cannot be refused.
    pub fn replace_into<Replacement>(
        &self,
        device: &DeviceScope,
        haystacks: &AnyBytesTape<'_>,
        overlap_policy: OverlapPolicy,
        replacements: &[Replacement],
        output_data: &mut [u8],
        output_offsets: &mut [u64],
    ) -> Result<usize, Error>
    where
        Replacement: AsRef<[u8]>,
    {
        // The engine writes one boundary per haystack plus a trailing total, and one replacement is read per
        // needle - neither is negotiable, and a short slice here would be an out-of-bounds write. The
        // callback-addressed arm is refused outright: a rewrite's product is a tape, and the C API offers
        // this verb no `sz_sequence_t` overload to put one in.
        if output_offsets.len() < Self::haystacks_count(haystacks) + 1
            || replacements.len() != self.needles_count
            || matches!(haystacks, AnyBytesTape::Slices(_))
        {
            return Err(Error::from(Status::UnexpectedDimensions));
        }

        let replacements_sequence = SzSequenceFromBytes::to_sz_sequence(replacements);
        let mut bytes_written = 0usize;
        let mut error_msg: *const c_char = ptr::null();
        let status = match haystacks {
            AnyBytesTape::Tape64(tape) => {
                let view = SzSequenceU64Tape::from(tape);
                unsafe {
                    szs_substrings_replace_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        &replacements_sequence as *const _ as *const c_void,
                        output_data.as_mut_ptr(),
                        output_data.len(),
                        output_offsets.as_mut_ptr(),
                        &mut bytes_written,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View64(borrowed) => {
                let view = SzSequenceU64Tape::from(borrowed);
                unsafe {
                    szs_substrings_replace_u64tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        &replacements_sequence as *const _ as *const c_void,
                        output_data.as_mut_ptr(),
                        output_data.len(),
                        output_offsets.as_mut_ptr(),
                        &mut bytes_written,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Tape32(tape) => {
                let view = SzSequenceU32Tape::from(tape);
                unsafe {
                    szs_substrings_replace_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        &replacements_sequence as *const _ as *const c_void,
                        output_data.as_mut_ptr(),
                        output_data.len(),
                        output_offsets.as_mut_ptr(),
                        &mut bytes_written,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::View32(borrowed) => {
                let view = SzSequenceU32Tape::from(borrowed);
                unsafe {
                    szs_substrings_replace_u32tape(
                        self.handle,
                        device.handle,
                        &view as *const _ as *const c_void,
                        overlap_policy,
                        &replacements_sequence as *const _ as *const c_void,
                        output_data.as_mut_ptr(),
                        output_data.len(),
                        output_offsets.as_mut_ptr(),
                        &mut bytes_written,
                        &mut error_msg,
                    )
                }
            }
            AnyBytesTape::Slices(_) => unreachable!("refused above"),
        };
        match status {
            Status::Success => Ok(bytes_written),
            err => Err(rust_error_from_c_message(err, error_msg)),
        }
    }

    /// Haystacks the input carries, which every output buffer above is sized against.
    fn haystacks_count(haystacks: &AnyBytesTape<'_>) -> usize {
        match haystacks {
            AnyBytesTape::Tape64(tape) => SzSequenceU64Tape::from(tape).count,
            AnyBytesTape::View64(view) => SzSequenceU64Tape::from(view).count,
            AnyBytesTape::Tape32(tape) => SzSequenceU32Tape::from(tape).count,
            AnyBytesTape::View32(view) => SzSequenceU32Tape::from(view).count,
            AnyBytesTape::Slices(sequence) => sequence.count(),
        }
    }
}

impl Drop for Substrings {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { szs_substrings_free(self.handle) };
        }
    }
}

unsafe impl Send for Substrings {}
unsafe impl Sync for Substrings {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stringzillas::fixtures::device_or_skip;
    use crate::stringzillas::types::copy_bytes_into_tape;
    use alloc::format;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn substrings_count_and_find_known_answers() {
        let Some(device) = device_or_skip("substrings_count_and_find_known_answers") else {
            return;
        };
        let engine = Substrings::new(&device, &["aa", "b"], CaseSensitivity::Cased).unwrap();
        let haystacks = ["aaab", "xyz"];

        // Overlapping matches are all reported: "aa" at 0 and 1, "b" at 3.
        let (counts, matches) = count_and_find(&engine, &device, &haystacks, OverlapPolicy::Overlapping);
        assert_eq!(counts[0], 3);
        assert_eq!(counts[1], 0);

        assert_eq!(matches.len(), 3);
        for one_match in matches.iter() {
            assert_eq!(one_match.haystack_index, 0);
            let expected_needle = ["aa", "b"][one_match.needle_index].as_bytes();
            let span = &haystacks[0].as_bytes()[one_match.byte_offset..one_match.byte_offset + one_match.byte_length];
            assert_eq!(span, expected_needle);
        }
    }

    #[test]
    fn substrings_uncased_kelvin_sign() {
        let Some(device) = device_or_skip("substrings_uncased_kelvin_sign") else {
            return;
        };
        let engine = Substrings::new(&device, &["k"], CaseSensitivity::Uncased).unwrap();
        // The 3-byte Kelvin sign U+212A folds to "k", so the 1-byte needle matches a 3-byte span.
        let kelvin_sign = "\u{212A}";
        let haystacks = ["K", "k", kelvin_sign];

        let (counts, matches) = count_and_find(&engine, &device, &haystacks, OverlapPolicy::Overlapping);
        assert_eq!(&counts[..], &[1, 1, 1]);
        assert_eq!(matches.len(), 3);
        let kelvin_match = matches.iter().find(|one_match| one_match.haystack_index == 2).unwrap();
        assert_eq!(kelvin_match.byte_length, 3);
    }

    #[test]
    fn substrings_needles_count() {
        let Some(device) = device_or_skip("substrings_needles_count") else {
            return;
        };
        let engine = Substrings::new(&device, &["aa", "b"], CaseSensitivity::Cased).unwrap();
        assert_eq!(engine.needles_count(), 2);
    }

    #[test]
    fn substrings_empty_needle_rejected() {
        let Some(device) = device_or_skip("substrings_empty_needle_rejected") else {
            return;
        };
        let Err(error) = Substrings::new(&device, &["good", ""], CaseSensitivity::Cased) else {
            panic!("an empty needle must be rejected");
        };
        assert_eq!(error.status, Status::UnexpectedDimensions);
    }

    /// Counts, then locates, sizing the match buffer from the counts exactly as a caller would.
    fn count_and_find<Sequence: AsRef<[u8]>>(
        engine: &Substrings,
        device: &DeviceScope,
        haystacks: &[Sequence],
        overlap_policy: OverlapPolicy,
    ) -> (Vec<usize>, Vec<SubstringsMatch>) {
        let tape = AnyBytesTape::from_sequences(haystacks).unwrap();
        let mut counts = vec![0usize; haystacks.len()];
        engine.count_into(device, &tape, overlap_policy, &mut counts).unwrap();

        let mut matches = vec![SubstringsMatch::default(); counts.iter().sum()];
        let found = engine.find_into(device, &tape, overlap_policy, &mut matches).unwrap();
        assert_eq!(found, matches.len(), "counting and locating must agree on the total");
        (counts, matches)
    }

    /// Matches sorted by start offset, so a test asserts a cover's shape without depending on the
    /// walk's own emission order, which follows match ends.
    fn matches_by_offset(matches: &[SubstringsMatch]) -> Vec<(usize, usize, usize)> {
        let mut ordered: Vec<(usize, usize, usize)> = matches
            .iter()
            .map(|one_match| (one_match.byte_offset, one_match.byte_length, one_match.needle_index))
            .collect();
        ordered.sort_unstable();
        ordered
    }

    #[test]
    fn substrings_leftmost_longest_shadows_shorter_needles() {
        let Some(device) = device_or_skip("substrings_leftmost_longest_shadows_shorter_needles") else {
            return;
        };
        let engine = Substrings::new(&device, &["cat", "catalog"], CaseSensitivity::Cased).unwrap();
        let haystacks = ["catalog"];

        // Both needles start at 0, so the longer span wins and "cat" never surfaces.
        let (counts, matches) = count_and_find(&engine, &device, &haystacks, OverlapPolicy::LeftmostLongest);
        assert_eq!(counts[0], 1);
        assert_eq!(matches_by_offset(&matches), vec![(0, 7, 1)]);

        // The same dictionary under the overlapping policy reports both.
        let (_, overlapping) = count_and_find(&engine, &device, &haystacks, OverlapPolicy::Overlapping);
        assert_eq!(matches_by_offset(&overlapping), vec![(0, 3, 0), (0, 7, 1)]);
    }

    #[test]
    fn substrings_leftmost_first_prefers_the_lower_needle_index() {
        let Some(device) = device_or_skip("substrings_leftmost_first_prefers_the_lower_needle_index") else {
            return;
        };
        let engine = Substrings::new(&device, &["cat", "catalog"], CaseSensitivity::Cased).unwrap();

        // Same start, so the lower needle index wins even though "catalog" spans further; the walk then
        // resumes past "cat", where "alog" matches nothing.
        let (_, matches) = count_and_find(&engine, &device, &["catalog"], OverlapPolicy::LeftmostFirst);
        assert_eq!(matches_by_offset(&matches), vec![(0, 3, 0)]);
    }

    #[test]
    fn substrings_cover_skips_matches_starting_inside_a_commit() {
        let Some(device) = device_or_skip("substrings_cover_skips_matches_starting_inside_a_commit") else {
            return;
        };
        let engine = Substrings::new(&device, &["ab", "ba"], CaseSensitivity::Cased).unwrap();

        // "ab" commits at 0, so "ba" at 1 starts inside the committed span and is dropped.
        let (_, matches) = count_and_find(&engine, &device, &["aba"], OverlapPolicy::LeftmostLongest);
        assert_eq!(matches_by_offset(&matches), vec![(0, 2, 0)]);

        // A self-overlapping needle tiles its haystack instead: commits at 0 and 2, not at 1.
        let tiling = Substrings::new(&device, &["aa"], CaseSensitivity::Cased).unwrap();
        let (_, tiled) = count_and_find(&tiling, &device, &["aaaa"], OverlapPolicy::LeftmostLongest);
        assert_eq!(matches_by_offset(&tiled), vec![(0, 2, 0), (2, 2, 0)]);
    }

    #[test]
    fn substrings_score_bm25_saturates_term_frequency() {
        let Some(device) = device_or_skip("substrings_score_bm25_saturates_term_frequency") else {
            return;
        };
        let engine = Substrings::new(&device, &["cat"], CaseSensitivity::Cased).unwrap();
        let haystacks = ["cat cat", "dog"];
        let parameters = Bm25Params {
            term_frequency_saturation: 1.2,
            length_normalization: 0.75,
            average_document_length: 5.0,
        };

        // Byte lengths stand in for document lengths, so haystack 0 is 7 bytes against a 5-byte mean:
        //   1.0 * 2 * (1.2 + 1) / (2 + 1.2 * (1 - 0.75 + 0.75 * 7 / 5)) = 4.4 / 3.56
        let tape = AnyBytesTape::from_sequences(&haystacks).unwrap();
        let mut scores = vec![0.0f32; haystacks.len()];
        engine
            .score_bm25_into(&device, &tape, &[1.0], None, parameters, &mut scores)
            .unwrap();
        assert!((scores[0] - 4.4f32 / 3.56f32).abs() < 1e-4);
        assert_eq!(scores[1], 0.0); // ? A needle absent from a haystack contributes nothing.
    }

    /// Rewrites @p haystacks into buffers sized by `replace_bound`, returning each output string.
    ///
    /// The whole point of the bound is that this is one call, so a test that needed two would be
    /// testing something the API does not promise.
    fn replace_all<Sequence, Replacement>(
        engine: &Substrings,
        device: &DeviceScope,
        haystacks: &[Sequence],
        overlap_policy: OverlapPolicy,
        replacements: &[Replacement],
    ) -> Vec<Vec<u8>>
    where
        Sequence: AsRef<[u8]>,
        Replacement: AsRef<[u8]>,
    {
        let input_bytes: usize = haystacks.iter().map(|one| one.as_ref().len()).sum();
        let bound = engine.replace_bound(replacements, input_bytes).unwrap();

        let mut output_data = vec![0u8; bound];
        let mut output_offsets = vec![0u64; haystacks.len() + 1];
        let tape = AnyBytesTape::from_sequences(haystacks).unwrap();
        engine
            .replace_into(
                device,
                &tape,
                overlap_policy,
                replacements,
                &mut output_data,
                &mut output_offsets,
            )
            .unwrap();

        (0..haystacks.len())
            .map(|index| {
                let start = output_offsets[index] as usize;
                let end = output_offsets[index + 1] as usize;
                output_data[start..end].to_vec()
            })
            .collect()
    }

    #[test]
    fn substrings_replace_rewrites_every_match() {
        let Some(device) = device_or_skip("substrings_replace_rewrites_every_match") else {
            return;
        };
        let engine = Substrings::new(&device, &["cat", "dog"], CaseSensitivity::Cased).unwrap();
        let haystacks = ["cat and dog", "nothing here"];

        let rewritten = replace_all(
            &engine,
            &device,
            &haystacks,
            OverlapPolicy::LeftmostLongest,
            &["feline", "canine"],
        );
        assert_eq!(rewritten[0], b"feline and canine");
        assert_eq!(rewritten[1], b"nothing here"); // ? An unmatched haystack passes through byte for byte.
    }

    #[test]
    fn substrings_replace_deletes_on_an_empty_replacement() {
        let Some(device) = device_or_skip("substrings_replace_deletes_on_an_empty_replacement") else {
            return;
        };
        let engine = Substrings::new(&device, &["bad "], CaseSensitivity::Cased).unwrap();

        // An empty replacement is legal and means deletion.
        let rewritten = replace_all(&engine, &device, &["bad word"], OverlapPolicy::LeftmostLongest, &[""]);
        assert_eq!(rewritten[0], b"word");
    }

    #[test]
    fn substrings_replace_inserts_verbatim_under_folding() {
        let Some(device) = device_or_skip("substrings_replace_inserts_verbatim_under_folding") else {
            return;
        };
        let engine = Substrings::new(&device, &["k"], CaseSensitivity::Uncased).unwrap();

        // No case adaptation: every fold preimage - including the 3-byte Kelvin sign - takes the
        // replacement's exact bytes, so the rewritten haystack shrinks.
        let kelvin_sign = "\u{212A}";
        let haystacks = [format!("K{kelvin_sign}k")];
        let rewritten = replace_all(&engine, &device, &haystacks, OverlapPolicy::LeftmostLongest, &["x"]);
        assert_eq!(rewritten[0], b"xxx");
    }

    #[test]
    fn substrings_replace_shadows_under_leftmost_longest() {
        let Some(device) = device_or_skip("substrings_replace_shadows_under_leftmost_longest") else {
            return;
        };
        let engine = Substrings::new(&device, &["cat", "catalog"], CaseSensitivity::Cased).unwrap();

        // The longer needle shadows the shorter at the same start, so "cat" never fires.
        let rewritten = replace_all(
            &engine,
            &device,
            &["catalog"],
            OverlapPolicy::LeftmostLongest,
            &["feline", "directory"],
        );
        assert_eq!(rewritten[0], b"directory");
    }

    #[test]
    fn substrings_replace_bound_covers_the_widest_expansion() {
        let Some(device) = device_or_skip("substrings_replace_bound_covers_the_widest_expansion") else {
            return;
        };
        let engine = Substrings::new(&device, &["a", "bb"], CaseSensitivity::Cased).unwrap();

        // "a" expands 1 byte into 4 and "bb" expands 2 into 4, so the widest ratio is 4 and a fully
        // tiled haystack of 8 bytes cannot exceed 32.
        let bound = engine.replace_bound(&["wxyz", "wxyz"], 8).unwrap();
        assert_eq!(bound, 32);

        // A dictionary that only ever shrinks still bounds at the input length, never below it.
        let shrinking = Substrings::new(&device, &["aaaa"], CaseSensitivity::Cased).unwrap();
        assert_eq!(shrinking.replace_bound(&["z"], 8).unwrap(), 8);
    }

    #[test]
    fn substrings_input_variants_agree() {
        let Some(device) = device_or_skip("substrings_input_variants_agree") else {
            return;
        };
        let engine = Substrings::new(&device, &["cat", "dog"], CaseSensitivity::Cased).unwrap();
        let corpus = ["cat and dog", "nothing here", "dog dog"];

        // The three shapes the C API accepts reach three different entry points - `_u32tape`,
        // `_u64tape` and the callback-addressed `sz_sequence_t` - and must reach one answer.
        let variants = [
            AnyBytesTape::from_sequences(&corpus).unwrap(),
            copy_bytes_into_tape(&corpus, true).unwrap(),
            AnyBytesTape::from_slices(&corpus),
        ];
        let mut answers = Vec::new();
        for haystacks in &variants {
            let mut counts = vec![0usize; corpus.len()];
            let total = engine
                .count_into(&device, haystacks, OverlapPolicy::Overlapping, &mut counts)
                .unwrap();
            let mut matches = vec![SubstringsMatch::default(); total];
            engine
                .find_into(&device, haystacks, OverlapPolicy::Overlapping, &mut matches)
                .unwrap();

            let parameters = Bm25Params::normalized(10.0);
            let mut scores = vec![0.0f32; corpus.len()];
            engine
                .score_bm25_into(&device, haystacks, &[1.0, 1.0], None, parameters, &mut scores)
                .unwrap();
            answers.push((counts, matches_by_offset(&matches), scores));
        }
        assert_eq!(answers[1], answers[0], "the 64-bit tape must agree with the 32-bit one");
        assert_eq!(answers[2], answers[0], "borrowed slices must agree with a tape");
        assert_eq!(answers[0].2[1], 0.0); // ? "nothing here" holds neither needle.

        // Tape in, tape out: a rewrite has nowhere to put its product when the input is addressed by
        // callback, and the C API gives `replace` no sequence overload to try.
        let mut output_data = vec![0u8; 256];
        let mut output_offsets = vec![0u64; corpus.len() + 1];
        let refused = engine.replace_into(
            &device,
            &AnyBytesTape::from_slices(&corpus),
            OverlapPolicy::LeftmostLongest,
            &["feline", "canine"],
            &mut output_data,
            &mut output_offsets,
        );
        assert_eq!(refused.unwrap_err().status, Status::UnexpectedDimensions);
    }

    #[test]
    fn substrings_parallel_scope_agrees_with_the_default() {
        // `device_or_skip` only ever yields the default scope, so nothing else here reaches the
        // multi-core arm of `szs_substrings_init` or the parallel walkers behind it.
        let Ok(parallel) = DeviceScope::cpu_cores(2) else {
            return;
        };
        let Some(device) = device_or_skip("substrings_parallel_scope_agrees_with_the_default") else {
            return;
        };
        let haystacks = ["cat and dog", "nothing here", "dog dog", "a catalog of cats"];
        let needles = ["cat", "dog", "catalog"];

        let serial_engine = Substrings::new(&device, &needles, CaseSensitivity::Cased).unwrap();
        let parallel_engine = Substrings::new(&parallel, &needles, CaseSensitivity::Cased).unwrap();
        for policy in [
            OverlapPolicy::Overlapping,
            OverlapPolicy::LeftmostLongest,
            OverlapPolicy::LeftmostFirst,
        ] {
            let (serial_counts, serial_matches) = count_and_find(&serial_engine, &device, &haystacks, policy);
            let (parallel_counts, parallel_matches) = count_and_find(&parallel_engine, &parallel, &haystacks, policy);
            assert_eq!(parallel_counts, serial_counts);
            assert_eq!(matches_by_offset(&parallel_matches), matches_by_offset(&serial_matches));
        }
    }

    /// A tiny xorshift, so a seeded corpus needs no dependency and reproduces across platforms.
    fn next_random(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// A needle set and corpus drawn from a five-letter alphabet, small enough that short needles hit
    /// often rather than measuring an empty walk.
    fn random_corpus(seed: u64) -> (Vec<String>, Vec<String>) {
        let mut state = seed | 1;
        let alphabet = b"abcde";
        let mut draw = |max_length: usize, state: &mut u64| -> String {
            let length = 1 + (next_random(state) as usize) % max_length;
            (0..length)
                .map(|_| alphabet[(next_random(state) as usize) % alphabet.len()] as char)
                .collect()
        };
        let mut needles: Vec<String> = (0..12).map(|_| draw(4, &mut state)).collect();
        needles.sort_unstable();
        needles.dedup();
        let haystacks: Vec<String> = (0..20).map(|_| draw(180, &mut state)).collect();
        (needles, haystacks)
    }

    #[test]
    fn substrings_matches_aho_corasick() {
        use aho_corasick::{AhoCorasick, MatchKind};

        let Some(device) = device_or_skip("substrings_matches_aho_corasick") else {
            return;
        };
        for seed in [42u64, 1, 314159, 2718281828] {
            let (needles, haystacks) = random_corpus(seed);
            let engine = Substrings::new(&device, &needles, CaseSensitivity::Cased).unwrap();

            // The three `MatchKind`s map one-to-one onto `OverlapPolicy`, so an independent automaton
            // witnesses every cover rule rather than only the overlapping walk.
            for (policy, kind) in [
                (OverlapPolicy::Overlapping, MatchKind::Standard),
                (OverlapPolicy::LeftmostLongest, MatchKind::LeftmostLongest),
                (OverlapPolicy::LeftmostFirst, MatchKind::LeftmostFirst),
            ] {
                let oracle_automaton = AhoCorasick::builder().match_kind(kind).build(&needles).unwrap();
                let mut oracle: Vec<(usize, usize, usize, usize)> = Vec::new();
                for (haystack_index, haystack) in haystacks.iter().enumerate() {
                    let found: Vec<_> = match kind {
                        MatchKind::Standard => oracle_automaton.find_overlapping_iter(haystack).collect(),
                        _ => oracle_automaton.find_iter(haystack).collect(),
                    };
                    for one_match in found {
                        oracle.push((
                            haystack_index,
                            one_match.pattern().as_usize(),
                            one_match.start(),
                            one_match.len(),
                        ));
                    }
                }
                oracle.sort_unstable();

                let (_, matches) = count_and_find(&engine, &device, &haystacks, policy);
                let mut found: Vec<(usize, usize, usize, usize)> = matches
                    .iter()
                    .map(|one| (one.haystack_index, one.needle_index, one.byte_offset, one.byte_length))
                    .collect();
                found.sort_unstable();
                assert_eq!(
                    found, oracle,
                    "seed {seed} disagrees with aho-corasick under {policy:?}"
                );
            }
        }
    }

    #[test]
    fn substrings_replace_matches_aho_corasick() {
        use aho_corasick::{AhoCorasick, MatchKind};

        let Some(device) = device_or_skip("substrings_replace_matches_aho_corasick") else {
            return;
        };
        for seed in [7u64, 99, 123456789] {
            let (needles, haystacks) = random_corpus(seed);
            let replacements: Vec<String> = needles.iter().map(|needle| format!("<{needle}>")).collect();
            let engine = Substrings::new(&device, &needles, CaseSensitivity::Cased).unwrap();

            for (policy, kind) in [
                (OverlapPolicy::LeftmostLongest, MatchKind::LeftmostLongest),
                (OverlapPolicy::LeftmostFirst, MatchKind::LeftmostFirst),
            ] {
                let oracle_automaton = AhoCorasick::builder().match_kind(kind).build(&needles).unwrap();
                let rewritten = replace_all(&engine, &device, &haystacks, policy, &replacements);
                for (haystack, rewritten_one) in haystacks.iter().zip(rewritten.iter()) {
                    let expected = oracle_automaton.replace_all_bytes(haystack.as_bytes(), &replacements);
                    assert_eq!(
                        rewritten_one, &expected,
                        "seed {seed} rewrote differently under {policy:?}"
                    );
                }
            }
        }
    }
}
