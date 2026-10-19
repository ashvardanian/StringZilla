//! Path queries: RFC 9535 JSONPath over JSON and XPath over XML.
//!
//! One compiled engine walks over many documents.
//!
//! File: rust/stringzilla/subdocuments.rs
//! Author: Ash Vardanian

use core::ffi::{c_char, c_void};
use core::mem::{ManuallyDrop, MaybeUninit};

use super::*;

extern "C" {
    fn sz_subdocuments_engine_init(
        engine: *mut SubdocumentsEngine,
        paths: *const sz_strings_view_t,
        format: SubdocumentsFormat,
        malformed_index: *mut sz_size_t,
        capabilities: sz_capability_t,
        allocator: *const c_void,
        stream: sz_stream_t,
    ) -> sz_status_t;
    fn sz_subdocuments_engine_free(engine: *mut SubdocumentsEngine, stream: sz_stream_t);
    fn sz_subdocuments_counts(
        engine: *mut SubdocumentsEngine,
        documents: *const sz_strings_view_t,
        counts: *mut sz_size_t,
        counts_stride: sz_size_t,
        malformed_index: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
    fn sz_subdocuments_find(
        engine: *mut SubdocumentsEngine,
        documents: *const sz_strings_view_t,
        matches: *mut Match,
        matches_capacity: sz_size_t,
        matches_offsets: *mut sz_size_t,
        malformed_index: *mut sz_size_t,
        stream: sz_stream_t,
    ) -> sz_status_t;
}

/// `sz_invalid_syntax_k`, the status of a path or a document that breaks its grammar.
const INVALID_SYNTAX: sz_status_t = -22;

/// The grammar of every document an engine reads, and the path language that addresses it.
///
/// Corresponds to `sz_subdocuments_format_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubdocumentsFormat {
    /// RFC 8259 documents, addressed by RFC 9535 JSONPath.
    Json = 0,
    /// XML 1.0 documents, addressed by XPath 1.0 location paths along forward axes.
    Xml = 1,
}

/// A vocabulary of paths compiled into one automaton over keys, walked over as many batches of
/// documents as a caller has.
///
/// Every document is walked once whatever the path count, and no tree is built: a container no path
/// can enter is skipped by counting its brackets. A [`Match`] spans the selected node's exact bytes
/// inside its document, in JSON a string with its quotes, a number or literal, or a container
/// through its closing bracket, and in XML an element, an attribute's value or a text node, which a
/// caller reads in place or hands to another parser.
///
/// The fields mirror `sz_subdocuments_engine_t` one for one and only `paths_count` and `format` are
/// read from Rust, so the layout is load-bearing and the engine travels to C by pointer. The regex
/// engine of `match()` and `search()` filters is embedded by value and freed by the C side, hence
/// `ManuallyDrop`. Every verb grows the engine's depth stack, so it takes `&mut self`.
///
/// # Examples
///
/// ```rust
/// use stringzilla::sz::{Capabilities, Match, Stream, StringsView, SubdocumentsEngine, SubdocumentsFormat};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let paths = ["$.store.book[?@.price < 10].title"];
/// let mut engine = SubdocumentsEngine::new(&paths, SubdocumentsFormat::Json, &cpu).unwrap();
///
/// let document = r#"{"store": {"book": [{"title": "A", "price": 8}, {"title": "B", "price": 12}]}}"#;
/// let texts = [document];
/// let documents = StringsView::from_slices(&texts);
/// let (mut matches, mut offsets, mut malformed_index) = ([Match::default(); 4], [0usize; 2], 0);
/// cpu.scope(|scope| engine.find(scope, documents, &mut matches, &mut offsets, &mut malformed_index)).unwrap();
/// assert_eq!(offsets[1], 1);
/// let title = &matches[0];
/// assert_eq!(&document[title.byte_offset..title.byte_offset + title.byte_length], "\"A\"");
/// ```
#[repr(C)]
#[allow(dead_code)] // Every member is the C side's to write; the layout is what Rust keeps.
pub struct SubdocumentsEngine {
    names_data: *const c_char,
    names_offsets: *const u32,
    names_slots: *const u32,
    names_steps: *const u64,
    wildcard_steps: *const u64,
    descendant_steps: *const u64,
    relative_steps: *const u64,
    attribute_steps: *const u64,
    text_steps: *const u64,
    filter_steps: *const u64,
    filters_roots: *const u64,
    filters: *const c_void,
    instructions: *const c_void,
    strings_data: *const c_char,
    final_steps: *const u64,
    root_steps: *const u64,
    slices: *const c_void,
    slices_offsets: *const u32,
    steps_paths: *const u32,
    steps_count: u32,
    words_count: u32,
    names_slots_count: u32,
    paths_count: u32,
    filters_count: u32,
    regex: ManuallyDrop<RegexEngine>,
    patterns_count: u32,
    format: SubdocumentsFormat,
    capability: Capabilities,
    allocator: sz_allocator_t,
    memory: *mut c_void,
    memory_bytes: usize,
    scratch: *mut c_void,
    scratch_bytes: usize,
}

impl SubdocumentsEngine {
    /// Compiles `paths` for `format` with the capabilities of `stream`, like
    /// [`Capabilities::cpu_enabled`], on the device `stream` names.
    ///
    /// JSON paths follow RFC 9535, filters and their functions included, and XML paths follow
    /// XPath 1.0 along forward axes. A path that breaks its grammar, or that no single walk can
    /// evaluate, like an absolute query inside a filter, is refused as
    /// [`StringsError::Malformed`] with its index.
    pub fn new<Path>(paths: &[Path], format: SubdocumentsFormat, stream: &Stream) -> Result<Self, StringsError>
    where
        Path: AsRef<[u8]>,
    {
        let mut engine = MaybeUninit::<Self>::uninit();
        let mut malformed_index = 0;
        let status = unsafe {
            sz_subdocuments_engine_init(
                engine.as_mut_ptr(),
                &StringsView::from_slices(paths).view,
                format,
                &mut malformed_index,
                stream.capabilities().bits(),
                core::ptr::null(),
                stream.as_raw(),
            )
        };
        match status {
            INVALID_SYNTAX => Err(StringsError::Malformed(malformed_index)),
            status => status.check().map_err(StringsError::Failed),
        }?;
        Ok(unsafe { engine.assume_init() })
    }

    /// Paths the vocabulary holds, which bounds every reported `needle_index`.
    pub fn paths_count(&self) -> usize {
        self.paths_count as usize
    }

    /// The grammar every document of every verb follows.
    pub fn format(&self) -> SubdocumentsFormat {
        self.format
    }

    /// Queues on the stream of `scope` the count of every document's nodes that some path selects,
    /// once per path, one count per document.
    ///
    /// `counts` receives document `d` at `counts[d * counts_stride]`, the stride counting entries
    /// rather than bytes and being at least one. A malformed document fails the call, and
    /// `malformed_index` then names the first one once the scope joins.
    pub fn counts<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        documents: StringsView<'scope>,
        counts: &'scope mut [usize],
        counts_stride: usize,
        malformed_index: &'scope mut usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        strided_column_check(documents.len(), counts.len(), counts_stride)?;
        let output = counts.as_mut_ptr();
        unsafe {
            sz_subdocuments_counts(
                self,
                &documents.view,
                output,
                counts_stride,
                malformed_index,
                stream.as_raw(),
            )
        }
        .check()
    }

    /// Queues on the stream of `scope` a report of every node of every document that some path
    /// selects, by document, then by start offset, then by path, so a container precedes what it
    /// holds.
    ///
    /// `matches_offsets` holds one boundary per document plus a final total, and is filled whether
    /// or not the matches fit, which is what sizes the next call; an empty `matches` makes the call
    /// a pure size query. A malformed document fails the call, and `malformed_index` then names the
    /// first one once the scope joins.
    pub fn find<'scope>(
        &'scope mut self,
        scope: &'scope Scope<'scope, '_>,
        documents: StringsView<'scope>,
        matches: &'scope mut [Match],
        matches_offsets: &'scope mut [usize],
        malformed_index: &'scope mut usize,
    ) -> Result<(), Status> {
        let stream = scope.stream_for(self.capability)?;
        if matches_offsets.len() < documents.len() + 1 {
            return Err(Status::UnexpectedDimensions);
        }
        let capacity = matches.len();
        let output = optional_mut_ptr(matches);
        let offsets = matches_offsets.as_mut_ptr();
        unsafe {
            sz_subdocuments_find(
                self,
                &documents.view,
                output,
                capacity,
                offsets,
                malformed_index,
                stream.as_raw(),
            )
        }
        .check()
    }
}

impl Drop for SubdocumentsEngine {
    fn drop(&mut self) {
        unsafe { sz_subdocuments_engine_free(self as *mut Self, core::ptr::null_mut()) };
    }
}
