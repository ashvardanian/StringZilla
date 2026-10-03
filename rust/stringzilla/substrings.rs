//! Multi-pattern substring search: one compiled Aho-Corasick automaton walked over many haystacks.
//!
//! File: rust/stringzilla/substrings.rs
//! Author: Ash Vardanian

use core::ffi::c_void;
use core::mem::MaybeUninit;

use super::*;

/// The `hot_states` argument's "size the hot tier yourself" value, so zero stays a
/// real all-cold request.
pub const SUBSTRINGS_HOT_STATES_AUTO: usize = usize::MAX;

/// The `matches_budget` argument's "let the tier choose" value; a host tier reads no budget at all.
pub const SUBSTRINGS_MATCHES_BUDGET_AUTO: usize = 0;

/// Whether a vocabulary matches needles byte-for-byte, or folds both sides to a shared case first.
///
/// Corresponds to `sz_substrings_case_sensitivity_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseSensitivity {
    /// Byte-exact matching; needles may be arbitrary bytes, including malformed UTF-8.
    Cased = 0,
    /// Full Unicode case folding as `CaseFolding.txt` defines it; needles must be valid UTF-8.
    Uncased = 1,
}

/// How matches that share bytes resolve: reported in full, or thinned to a leftmost run.
///
/// The policy sizes the engine's arena, so it is fixed at construction rather than travelling per
/// call, and one engine runs exactly one of the three.
///
/// Corresponds to `sz_substrings_overlap_policy_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstringsOverlapPolicy {
    /// Every match of every needle, including ones that share bytes and ones nested in others.
    Overlapping = 0,
    /// Matches sharing no bytes: earliest start, then longest span, then lower needle index.
    LeftmostLongest = 1,
    /// Matches sharing no bytes: earliest start, then lower needle index, however long the rival.
    LeftmostFirst = 2,
}

/// One reported match, locating it by haystack, by needle, and by byte span.
///
/// Under case folding a needle's own byte length is not the length of every match - needle `k`
/// matches both the 1-byte `k` and the 3-byte Kelvin sign `U+212A` - so the span travels per match.
///
/// Corresponds to `sz_substrings_match_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubstringsMatch {
    /// Which haystack of the sequence this match was found in.
    pub haystack_index: usize,
    /// Which needle of the vocabulary matched.
    pub needle_index: usize,
    /// Where the match starts inside that haystack, in its own bytes.
    pub byte_offset: usize,
    /// Haystack bytes the match spans, which folding can make differ from the needle's own length.
    pub byte_length: usize,
}

/// What a round found, which is how a capacity shortfall is reported rather than as an error.
///
/// The sizing walk always runs, so `matches_emitted` is the truth whatever the caller's
/// output could hold, and a nonzero `shortfall` is the one signal that an output is
/// incomplete rather than wrong.
///
/// Corresponds to `sz_substrings_report_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SubstringsReport {
    /// Matches the sizing walk found, which is the truth whatever the output held.
    pub matches_emitted: usize,
    /// Matches written out, which is `matches_emitted` clipped at the capacity.
    pub matches_stored: usize,
    /// Bytes a rewrite needs, which is the truth whatever the target held.
    pub target_length: usize,
    /// Matches or bytes the round could not hold, zero when everything fit.
    pub shortfall: usize,
}

/// BM25's continuous parameters.
///
/// There is no correct default for the corpus mean, so there is no `Default`: reach for
/// [`Bm25Params::normalized`] or [`Bm25Params::unnormalized`], which name the two configurations
/// that exist. A positive `length_normalization` beside a non-positive `average_document_length` is
/// refused, since it would divide by a mean that is not there.
///
/// Corresponds to `sz_substrings_bm25_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Params {
    /// The literature's `k1`: how slowly repeated occurrences stop adding score; 1.2 is customary.
    pub term_frequency_saturation: f32,
    /// The literature's `b`, in `[0, 1]`: 0 ignores document length, 1 normalizes it fully.
    pub length_normalization: f32,
    /// The corpus-wide mean document length, in the unit of the lengths scored; read only when
    /// `length_normalization` is positive.
    pub average_document_length: f32,
}

impl Bm25Params {
    /// Classic BM25 with the literature's customary `k1 = 1.2` and `b = 0.75`, normalized against a
    /// corpus whose mean document length is `average_document_length`, in the unit the per-document
    /// lengths use.
    pub const fn normalized(average_document_length: f32) -> Self {
        Self {
            term_frequency_saturation: 1.2,
            length_normalization: 0.75,
            average_document_length,
        }
    }

    /// BM25 with length normalization switched off, for a corpus whose mean length is unknown or
    /// whose documents are uniform enough not to need it; the per-document lengths then go unread.
    pub const fn unnormalized() -> Self {
        Self {
            term_frequency_saturation: 1.2,
            length_normalization: 0.0,
            average_document_length: 0.0,
        }
    }
}

/// A vocabulary compiled into one automaton, walked over as many batches of haystacks as
/// a caller has.
///
/// Building the automaton is the expensive half and every verb below reuses it, so one long-lived
/// engine amortizes that across every later batch: a dictionary of thousands of terms costs one
/// pass over a haystack rather than thousands.
///
/// The fields mirror `sz_substrings_engine_t` one for one and only `needles_count`,
/// `overlap_policy` and `report` are read from Rust, so the layout is load-bearing and the engine
/// travels to C by pointer. Owning raw pointers makes it neither `Send` nor `Sync`, which is what
/// the C contract wants: a compute verb writes the engine's round arena, so it mutates, and every
/// verb below takes `&mut self` for that reason.
///
/// # Examples
///
/// ```rust
/// use stringzilla::sz::{
///     Capabilities, CaseSensitivity, Stream, SubstringsEngine, SubstringsOverlapPolicy,
///     SUBSTRINGS_HOT_STATES_AUTO, SUBSTRINGS_MATCHES_BUDGET_AUTO,
/// };
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let mut engine = SubstringsEngine::new(
///     &["cat", "catalog"],
///     CaseSensitivity::Cased,
///     SubstringsOverlapPolicy::Overlapping,
///     SUBSTRINGS_HOT_STATES_AUTO,
///     SUBSTRINGS_MATCHES_BUDGET_AUTO,
///     0,
///     &cpu,
/// )
/// .unwrap();
///
/// let documents = ["a catalog of cats", "nothing here"];
/// let mut counts = [0usize; 2];
/// cpu.scope(|scope| engine.counts(scope, &documents, &mut counts, 1)).unwrap();
/// assert_eq!(counts, [3, 0]); // "catalog", the "cat" inside it, and the "cat" of "cats"
/// ```
#[repr(C)]
#[allow(dead_code)] // Every member is the C side's to write; the layout is what Rust keeps.
pub struct SubstringsEngine {
    hot_rows: *const u32,
    byte_to_class: *const u8,
    base: *const u32,
    check: *const u32,
    fail: *const u32,
    accepts_words: *const u32,
    outputs: *const c_void,
    outputs_counts: *const u32,
    outputs_offsets: *const usize,
    outputs_total: usize,
    slots_count: usize,
    hot_count: u32,
    classes_count: u32,
    state_count: u32,
    root: u32,
    needles_count: u32,
    max_source_match_bytes: u32,
    min_source_match_bytes: u32,
    max_outputs_per_state: u32,
    case_sensitivity: CaseSensitivity,
    root_live: [u64; 4],
    overlap_policy: SubstringsOverlapPolicy,
    matches_budget: usize,
    chunk_budget: usize,
    haystacks_budget: usize,
    report: *mut SubstringsReport,
    capability: Capabilities,
    copy: *const c_void,
    allocator: _SzMemoryAllocator,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl SubstringsEngine {
    /// Compiles `needles` into an automaton the matching verbs read, with the capabilities of
    /// `stream`, like [`Capabilities::cpu_enabled`] or a GPU's [`Capabilities::cuda_enabled`], on
    /// the device `stream` names, which it may join.
    ///
    /// An empty needle is refused rather than skipped, since it would match at every position
    /// and dropping it would shift every later needle's reported index. `hot_states` is the
    /// count kept in the dense hot rows, or [`SUBSTRINGS_HOT_STATES_AUTO`] to fill a fixed byte
    /// budget instead.
    ///
    /// The budgets are read by a device kernel alone, so a host engine takes
    /// [`SUBSTRINGS_MATCHES_BUDGET_AUTO`] and zero and walks straight into the caller's output.
    /// `matches_budget` is what a device round may emit before every later kernel retires, and
    /// `haystacks_budget` how many haystacks it may carry; the round's arena is sized for both
    /// here, zero asking for a default, so no verb allocates, a round carrying more haystacks is
    /// refused, and matches past the budget surface as the report's `shortfall`. A GPU engine's
    /// verbs take a [`Sequence`] for haystacks and replacements, and every buffer, weights and
    /// lengths included, in memory the device reaches, such as a `Vec` in a [`UnifiedAllocator`].
    pub fn new<Needle>(
        needles: &[Needle],
        case_sensitivity: CaseSensitivity,
        overlap_policy: SubstringsOverlapPolicy,
        hot_states: usize,
        matches_budget: usize,
        haystacks_budget: usize,
        stream: &Stream,
    ) -> Result<Self, Status>
    where
        Needle: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        with_sequence(needles, |sequence| unsafe {
            sz_substrings_engine_init(
                engine.as_mut_ptr(),
                sequence,
                case_sensitivity,
                overlap_policy,
                hot_states,
                matches_budget,
                haystacks_budget,
                stream.capabilities().bits(),
                core::ptr::null(),
                stream.as_raw(),
            )
        })
        .check()?;
        Ok(unsafe { engine.assume_init() })
    }

    /// Needles the vocabulary holds, which bounds every reported `needle_index`.
    pub fn needles_count(&self) -> usize {
        self.needles_count as usize
    }

    /// The cover every round runs, fixed when the arena was sized.
    pub fn overlap_policy(&self) -> SubstringsOverlapPolicy {
        self.overlap_policy
    }

    /// What the last round found, which is how a capacity shortfall surfaces instead of
    /// as an error.
    ///
    /// A verb borrows the engine until its scope joins, so the record a device wrote is complete.
    pub fn report(&self) -> SubstringsReport {
        unsafe { *self.report }
    }

    /// Queues on the stream of `scope` the count of every needle's matches in every haystack, one
    /// count per haystack.
    ///
    /// `counts` receives haystack `h` at `counts[h * counts_stride]`, the stride counting entries
    /// rather than bytes and being at least one, so a strided call writes one column of a
    /// `[haystacks, vocabularies]` feature matrix. A stream of another group than the engine's is
    /// refused with [`Status::DeviceMemoryMismatch`], here and by every verb below.
    pub fn counts<'scope, Haystacks>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: &'scope Haystacks,
        counts: &'scope mut [usize],
        counts_stride: usize,
    ) -> Result<(), Status>
    where
        Haystacks: Strings + ?Sized,
    {
        let stream = scope.stream_for(self.capability)?;
        let engine = self as *mut Self;
        haystacks.with_sequence(stream, |sequence| {
            strided_column_check(sequence.count, counts.len(), counts_stride)?;
            let output = counts.as_mut_ptr();
            unsafe { sz_substrings_counts(engine, sequence, output, counts_stride, stream.as_raw()) }.check()
        })?
    }

    /// Queues on the stream of `scope` a report of every match of every needle in every haystack,
    /// ascending by haystack.
    ///
    /// `matches_offsets` holds one boundary per haystack plus a final total, and is filled whether
    /// or not the matches fit, which is what sizes the next call. A capacity too small is not an
    /// error: [`SubstringsEngine::report`] names the true total and what did not fit, so a sizing
    /// call with an empty `matches` followed by one filling call needs no walk in between.
    pub fn find<'scope, Haystacks>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: &'scope Haystacks,
        matches: &'scope mut [SubstringsMatch],
        matches_offsets: &'scope mut [usize],
    ) -> Result<(), Status>
    where
        Haystacks: Strings + ?Sized,
    {
        let stream = scope.stream_for(self.capability)?;
        let engine = self as *mut Self;
        let capacity = matches.len();
        let output = optional_mut_ptr(matches);
        haystacks.with_sequence(stream, |sequence| {
            if matches_offsets.len() < sequence.count + 1 {
                return Err(Status::UnexpectedDimensions);
            }
            let offsets = matches_offsets.as_mut_ptr();
            unsafe { sz_substrings_find(engine, sequence, output, capacity, offsets, stream.as_raw()) }.check()
        })?
    }

    /// Queues on the stream of `scope` a rewrite of every haystack into one `target` buffer,
    /// substituting one replacement per needle.
    ///
    /// `replacements` is indexed by needle and an empty one deletes the match, so it holds exactly
    /// [`SubstringsEngine::needles_count`] entries. `offsets` holds one boundary per haystack plus
    /// a final total and is filled whether or not `target` held the result, which is what sizes the
    /// next call; a `target` too small leaves its contents unspecified rather than failing, and
    /// [`SubstringsEngine::report`] names the bytes the rewrite needed.
    ///
    /// The engine's policy must be a cover, since a substitution over matches that share bytes is
    /// not a function.
    pub fn replace<'scope, Haystacks, Replacements>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: &'scope Haystacks,
        replacements: &'scope Replacements,
        target: &'scope mut [u8],
        offsets: &'scope mut [usize],
    ) -> Result<(), Status>
    where
        Haystacks: Strings + ?Sized,
        Replacements: Strings + ?Sized,
    {
        let stream = scope.stream_for(self.capability)?;
        let needles_count = self.needles_count();
        let engine = self as *mut Self;
        let capacity = target.len();
        let output = optional_mut_ptr(target);
        haystacks.with_sequence(stream, |haystacks_sequence| {
            replacements.with_sequence(stream, |replacements_sequence| {
                if replacements_sequence.count != needles_count || offsets.len() < haystacks_sequence.count + 1 {
                    return Err(Status::UnexpectedDimensions);
                }
                unsafe {
                    sz_substrings_replace(
                        engine,
                        haystacks_sequence,
                        replacements_sequence,
                        output,
                        capacity,
                        offsets.as_mut_ptr(),
                        stream.as_raw(),
                    )
                }
                .check()
            })?
        })?
    }

    /// Queues on the stream of `scope` the score of every haystack against the whole vocabulary
    /// as one BM25 query, one score per haystack.
    ///
    /// The vocabulary is the query and `needle_weights` holds each needle's IDF or boost, so a
    /// many-term query over a large corpus never materializes per-term frequency rows. Term
    /// frequencies are raw overlapping counts, since a cover would suppress genuine occurrences of
    /// a needle nested in another, so the engine's own policy does not apply here.
    /// `document_lengths` is `None` to normalize by each haystack's byte length instead.
    ///
    /// `scores` receives haystack `h` at `scores[h * scores_stride]`, on the same convention as
    /// [`SubstringsEngine::counts`].
    #[allow(clippy::too_many_arguments)] // The C verb's own arguments, plus the scope.
    pub fn bm25_scores<'scope, Haystacks>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: &'scope Haystacks,
        document_lengths: Option<&'scope [f32]>,
        parameters: &'scope Bm25Params,
        needle_weights: &'scope [f32],
        scores: &'scope mut [f32],
        scores_stride: usize,
    ) -> Result<(), Status>
    where
        Haystacks: Strings + ?Sized,
    {
        if needle_weights.len() < self.needles_count() {
            return Err(Status::UnexpectedDimensions);
        }
        let stream = scope.stream_for(self.capability)?;
        let engine = self as *mut Self;
        haystacks.with_sequence(stream, |sequence| {
            strided_column_check(sequence.count, scores.len(), scores_stride)?;
            let lengths = match document_lengths {
                Some(lengths) if lengths.len() < sequence.count => return Err(Status::UnexpectedDimensions),
                Some(lengths) => lengths.as_ptr(),
                None => core::ptr::null(),
            };
            unsafe {
                sz_substrings_bm25_scores(
                    engine,
                    sequence,
                    lengths,
                    parameters as *const Bm25Params,
                    needle_weights.as_ptr(),
                    scores.as_mut_ptr(),
                    scores_stride,
                    stream.as_raw(),
                )
            }
            .check()
        })?
    }
}

impl Drop for SubstringsEngine {
    fn drop(&mut self) {
        unsafe { sz_substrings_engine_free(self as *mut Self, core::ptr::null_mut()) };
    }
}

/// Refuses an output that cannot hold one entry per haystack at `stride` entries apart.
fn strided_column_check(haystacks: usize, entries: usize, stride: usize) -> Result<(), Status> {
    if stride == 0 {
        return Err(Status::UnexpectedDimensions);
    }
    let span = match haystacks {
        0 => 0,
        count => (count - 1)
            .checked_mul(stride)
            .and_then(|offset| offset.checked_add(1))
            .ok_or(Status::OverflowRisk)?,
    };
    if entries < span {
        return Err(Status::UnexpectedDimensions);
    }
    Ok(())
}

/// The C verbs read a null output as a pure size query, which an empty slice's dangling
/// pointer is not.
fn optional_mut_ptr<Element>(buffer: &mut [Element]) -> *mut Element {
    if buffer.is_empty() {
        core::ptr::null_mut()
    } else {
        buffer.as_mut_ptr()
    }
}
