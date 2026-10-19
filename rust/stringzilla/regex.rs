//! Multi-pattern regular expressions: one compiled pair of automata walked over many haystacks.
//!
//! File: rust/stringzilla/regex.rs
//! Author: Ash Vardanian

use core::ffi::c_void;
use core::mem::MaybeUninit;

use super::*;

extern "C" {
    fn sz_regex_engine_init(
        engine: *mut RegexEngine,
        patterns: *const sz_strings_view_t,
        case_sensitivity: CaseSensitivity,
        malformed_index: *mut sz_size_t,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: sz_stream_t,
    ) -> sz_status_t;
    fn sz_regex_engine_free(engine: *mut RegexEngine, stream: sz_stream_t);
    fn sz_regex_counts(
        engine: *mut RegexEngine,
        haystacks: *const sz_strings_view_t,
        counts: *mut sz_size_t,
        counts_stride: sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    fn sz_regex_find(
        engine: *mut RegexEngine,
        haystacks: *const sz_strings_view_t,
        matches: *mut Match,
        matches_capacity: sz_size_t,
        matches_offsets: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    fn sz_regex_replace(
        engine: *mut RegexEngine,
        haystacks: *const sz_strings_view_t,
        replacements: *const sz_strings_view_t,
        target: *mut u8,
        target_capacity: sz_size_t,
        offsets: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    fn sz_regex_extract(
        engine: *mut RegexEngine,
        haystacks: *const sz_strings_view_t,
        pattern_index: sz_size_t,
        groups: *mut Match,
        groups_capacity: sz_size_t,
        groups_offsets: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
}

/// A vocabulary of RE2-syntax patterns compiled into two automata, walked over as many batches of
/// haystacks as a caller has.
///
/// Matches are leftmost-first across the union of patterns, ties going to the lower pattern index,
/// which is what Rust's `regex`, RE2 and Python's `re` report; a [`Match`] names its pattern as
/// `needle_index`. Every haystack streams through the automata once whatever the pattern count.
///
/// The fields mirror `sz_regex_engine_t` one for one and only `patterns_count` is read from Rust,
/// so the layout is load-bearing and the engine travels to C by pointer. Every verb grows the
/// engine's round scratch, so it takes `&mut self`.
///
/// # Examples
///
/// ```rust
/// use stringzilla::sz::{Capabilities, CaseSensitivity, Match, RegexEngine, Stream, StringsView};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let mut engine = RegexEngine::new(&["[0-9]+", "[a-z]+"], CaseSensitivity::Cased, &cpu).unwrap();
///
/// let documents = StringsView::from_slices(&["abc 123 de", "!"]);
/// let (mut matches, mut offsets) = ([Match::default(); 4], [0usize; 3]);
/// cpu.scope(|scope| engine.find(scope, documents, &mut matches, &mut offsets)).unwrap();
/// assert_eq!(offsets, [0, 3, 3]);
/// assert_eq!((matches[1].needle_index, matches[1].byte_offset, matches[1].byte_length), (0, 4, 3));
/// ```
#[repr(C)]
#[allow(dead_code)] // Every member is the C side's to write; the layout is what Rust keeps.
pub struct RegexEngine {
    byte_to_class: *const u8,
    reverse_rows: *const u32,
    forward_rows: *const u32,
    forward_patterns: *const u32,
    forward_escapes: *const Byteset,
    prefixes: *const c_void,
    forward_starts: [u32; 4],
    reverse_starts: [u32; 4],
    forward_accepts_begin: u32,
    reverse_accepts_begin: u32,
    classes_count: u32,
    forward_states_count: u32,
    reverse_states_count: u32,
    patterns_count: u32,
    reverse_live: Byteset,
    positions_words: u32,
    positions_shifts_count: u32,
    positions_groups_count: u32,
    positions_ranges_count: u32,
    positions_reach: *const u64,
    positions_shifts: *const u64,
    positions_groups: *const u64,
    positions_all: *const u64,
    positions_initial: *const u64,
    positions_ascii: *const u32,
    positions_ranges: *const u32,
    nfa_states: *const c_void,
    nfa_links: *const c_void,
    nfa_states_count: u32,
    nfa_links_count: u32,
    nfa_start: u32,
    patterns_starts: *const u32,
    patterns_groups: *const u32,
    lookarounds: *const c_void,
    bodies: *const c_void,
    looks_bytes: *const u32,
    lookarounds_count: u32,
    groups_names: sz_strings_view_t,
    case_sensitivity: CaseSensitivity,
    capability: Capabilities,
    allocator: sz_allocator_t,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl RegexEngine {
    /// Compiles `patterns` with the capabilities of `stream`, like [`Capabilities::cpu_enabled`],
    /// on the device `stream` names.
    ///
    /// A pattern outside the supported syntax, like a backreference, a capturing group inside a
    /// lookaround or an inline `(?i)`, is refused as [`StringsError::Malformed`] with its index;
    /// case folding is the engine-wide `case_sensitivity` instead.
    pub fn new<Pattern>(
        patterns: &[Pattern],
        case_sensitivity: CaseSensitivity,
        stream: &Stream,
    ) -> Result<Self, StringsError>
    where
        Pattern: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        let mut malformed_index = 0;
        let status = unsafe {
            sz_regex_engine_init(
                engine.as_mut_ptr(),
                &StringsView::from_slices(patterns).view,
                case_sensitivity,
                &mut malformed_index,
                stream.capabilities().bits(),
                core::ptr::null(),
                stream.as_raw(),
            )
        };
        match status.check() {
            Err(Status::InvalidSyntax) => Err(StringsError::Malformed(malformed_index)),
            checked => checked.map_err(StringsError::Failed),
        }?;
        Ok(unsafe { engine.assume_init() })
    }

    /// Patterns the vocabulary holds, which bounds every reported `needle_index`.
    pub fn patterns_count(&self) -> usize {
        self.patterns_count as usize
    }

    /// Queues on the stream of `scope` the count of every haystack's leftmost-first matches, one
    /// count per haystack.
    ///
    /// `counts` receives haystack `h` at `counts[h * counts_stride]`, the stride counting entries
    /// rather than bytes and being at least one. A stream of another group than the engine's is
    /// refused with [`Status::DeviceMemoryMismatch`], here and by every verb below.
    pub fn counts<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: StringsView<'scope>,
        counts: &'scope mut [usize],
        counts_stride: usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        strided_column_check(haystacks.len(), counts.len(), counts_stride)?;
        let output = counts.as_mut_ptr();
        unsafe { sz_regex_counts(self, &haystacks.view, output, counts_stride, stream.as_raw()) }.check()
    }

    /// Queues on the stream of `scope` a report of every leftmost-first match, ascending by
    /// haystack, then by start.
    ///
    /// `matches_offsets` holds one boundary per haystack plus a final total, filled whether or not
    /// the matches fit, which sizes the next call; an empty `matches` makes the call a size query.
    pub fn find<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: StringsView<'scope>,
        matches: &'scope mut [Match],
        matches_offsets: &'scope mut [usize],
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if matches_offsets.len() < haystacks.len() + 1 {
            return Err(Status::UnexpectedDimensions);
        }
        let capacity = matches.len();
        let output = optional_mut_ptr(matches);
        let offsets = matches_offsets.as_mut_ptr();
        unsafe { sz_regex_find(self, &haystacks.view, output, capacity, offsets, stream.as_raw()) }.check()
    }

    /// Rewrites every haystack into one `target` buffer, substituting each match with its
    /// pattern's replacement, on the CPU only.
    ///
    /// `replacements` is indexed by pattern, so it holds exactly [`RegexEngine::patterns_count`]
    /// entries; a replacement names groups as `$1`, `${1}` or `${name}`, `$0` being the whole match
    /// and `$$` one dollar. `offsets` holds one boundary per haystack plus a final total and is
    /// filled whether or not `target` held the result, which sizes the next call.
    pub fn replace<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: StringsView<'scope>,
        replacements: StringsView<'scope>,
        target: &'scope mut [u8],
        offsets: &'scope mut [usize],
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if replacements.len() != self.patterns_count() || offsets.len() < haystacks.len() + 1 {
            return Err(Status::UnexpectedDimensions);
        }
        let capacity = target.len();
        let output = optional_mut_ptr(target);
        unsafe {
            sz_regex_replace(
                self,
                &haystacks.view,
                &replacements.view,
                output,
                capacity,
                offsets.as_mut_ptr(),
                stream.as_raw(),
            )
        }
        .check()
    }

    /// Queues on the stream of `scope` the groups of each haystack's first match of
    /// `pattern_index`, as Pandas `str.extract` reads them.
    ///
    /// A haystack with such a match gets a [`Match`] row for the whole match, `needle_index` zero,
    /// then one per group that took part, `needle_index` naming the group, so a group left out
    /// differs from an empty one; a haystack without one gets no rows. `groups_offsets` holds one
    /// boundary per haystack plus a final total, filled whether or not the rows fit, which sizes
    /// the next call; an empty `groups` makes the call a size query.
    ///
    /// ```rust
    /// use stringzilla::sz::{Capabilities, CaseSensitivity, Match, RegexEngine, Stream, StringsView};
    ///
    /// let cpu = Stream::default(Capabilities::cpu_enabled());
    /// let mut engine = RegexEngine::new(&["(\\d+)-(\\d+)"], CaseSensitivity::Cased, &cpu).unwrap();
    ///
    /// let documents = StringsView::from_slices(&["a 12-34", "none"]);
    /// let (mut groups, mut offsets) = ([Match::default(); 3], [0usize; 3]);
    /// cpu.scope(|scope| engine.extract(scope, documents, 0, &mut groups, &mut offsets)).unwrap();
    /// assert_eq!(offsets, [0, 3, 3]);
    /// assert_eq!((groups[2].needle_index, groups[2].byte_offset, groups[2].byte_length), (2, 5, 2));
    /// ```
    pub fn extract<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        haystacks: StringsView<'scope>,
        pattern_index: usize,
        groups: &'scope mut [Match],
        groups_offsets: &'scope mut [usize],
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if pattern_index >= self.patterns_count() || groups_offsets.len() < haystacks.len() + 1 {
            return Err(Status::UnexpectedDimensions);
        }
        let capacity = groups.len();
        let output = optional_mut_ptr(groups);
        let offsets = groups_offsets.as_mut_ptr();
        unsafe {
            sz_regex_extract(
                self,
                &haystacks.view,
                pattern_index,
                output,
                capacity,
                offsets,
                stream.as_raw(),
            )
        }
        .check()
    }
}

impl Drop for RegexEngine {
    fn drop(&mut self) {
        unsafe { sz_regex_engine_free(self as *mut Self, core::ptr::null_mut()) };
    }
}
