//! String collections and byte transforms: owned [`Strings`] and borrowed [`StringsView`] columns
//! with their splitting, joining, escaping, lookup and normalization conversions, lookup tables and
//! random fills over one buffer, and in-place replacement.
//!
//! File: rust/stringzilla/strings.rs
//! Author: Ash Vardanian

use alloc::alloc::Global;
use alloc::vec::Vec;
use core::alloc::{Allocator, Layout};
use core::ffi::c_void;
use core::fmt;
use core::marker::PhantomData;
use core::ptr::NonNull;

use super::*;

/// Performs a lookup-table transformation, mapping contents of a buffer into the same or other
/// memory region, taking a byte substitution value from the provided table.
///
/// # Arguments
///
/// - `target`: A mutable buffer to populate.
/// - `source`: An immutable buffer to map from.
/// - `table`: Lookup table of 256 substitution values.
///
/// Returns [`Status::UnexpectedDimensions`] if `target` is shorter than `source`.
///
/// # Examples
///
/// To convert uppercase ASCII characters to lowercase:
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let mut to_lower: [u8; 256] = core::array::from_fn(|i| i as u8);
/// for (upper, lower) in ('A'..='Z').zip('a'..='z') {
///     to_lower[upper as usize] = lower as u8;
/// }
/// let source = "HELLO WORLD!";
/// let mut target = vec![0u8; source.len()];
/// sz::lookup(&mut target, &source, to_lower).unwrap();
/// let result = String::from_utf8(target).expect("Invalid UTF-8 sequence");
/// assert_eq!(result, "hello world!");
/// ```
///
pub fn lookup<Target, Source>(target: &mut Target, source: &Source, table: [u8; 256]) -> Result<(), Status>
where
    Target: AsMut<[u8]> + ?Sized,
    Source: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    if target_slice.len() < source_slice.len() {
        return Err(Status::UnexpectedDimensions);
    }
    // Two empty slices may share one dangling address, which would read as a single tape.
    if source_slice.is_empty() {
        return Ok(());
    }
    unsafe {
        lookup_tape(
            source_slice.as_ptr(),
            target_slice.as_mut_ptr(),
            source_slice.len(),
            &table,
        )
    }
    .check()
}

/// Performs a lookup-table transformation, mapping contents of a buffer into the same or other
/// memory region, taking a byte substitution value from the provided table.
///
/// # Arguments
///
/// - `buffer`: A mutable buffer to update inplace.
/// - `table`: Lookup table of 256 substitution values.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let mut to_lower: [u8; 256] = core::array::from_fn(|i| i as u8);
/// for (upper, lower) in ('A'..='Z').zip('a'..='z') {
///     to_lower[upper as usize] = lower as u8;
/// }
/// let mut text = *b"HELLO WORLD!";
/// sz::lookup_inplace(&mut text, to_lower);
/// assert_eq!(text, *b"hello world!");
/// ```
///
pub fn lookup_inplace<Buffer>(buffer: &mut Buffer, table: [u8; 256])
where
    Buffer: AsMut<[u8]> + ?Sized,
{
    let buffer_slice = buffer.as_mut();
    let target = buffer_slice.as_mut_ptr();
    // In place both blocks keep their sizes, so the kernel allocates nothing and cannot fail.
    unsafe { lookup_tape(target.cast_const(), target, buffer_slice.len(), &table) }.infallible();
}

/// Maps `length` bytes of `source` into `target` through `table`, as tapes of one string over the
/// two buffers, whose matching sizes leave nothing to allocate; in place, they share the offsets.
unsafe fn lookup_tape(source: *const u8, target: *mut u8, length: usize, table: &[u8; 256]) -> sz_status_t {
    let mut narrow = [[0, length as u32]; 2];
    let mut wide = [[0, length as u64]; 2];
    let (layout, source_offsets, target_offsets): (_, *mut c_void, *mut c_void) = match u32::try_from(length) {
        Ok(_) => (
            sz_strings_layout_t::U32Tape,
            narrow[0].as_mut_ptr().cast(),
            narrow[1].as_mut_ptr().cast(),
        ),
        Err(_) => (
            sz_strings_layout_t::U64Tape,
            wide[0].as_mut_ptr().cast(),
            wide[1].as_mut_ptr().cast(),
        ),
    };
    let in_place = core::ptr::eq(source, target);
    let strings = sz_strings_view_t {
        layout,
        count: 1,
        data: sz_strings_data_t {
            tape: sz_strings_tape_t {
                data: source.cast_mut(),
                offsets: source_offsets,
            },
        },
    };
    let mut tape = sz_strings_span_t {
        layout,
        count: 1,
        tape: sz_strings_tape_t {
            data: target,
            offsets: if in_place { source_offsets } else { target_offsets },
        },
    };
    sz_strings_lookup_best(
        &strings,
        core::ptr::null(),
        1,
        table.as_ptr(),
        core::ptr::null_mut(),
        &mut tape,
        Capabilities::CPUS.bits(),
        core::ptr::null_mut(),
    )
}

/// Randomizes the contents of a given byte slice `text` using characters from a specified
/// `alphabet`. This function mutates `text` in place, replacing each byte with a random one
/// from `alphabet`. It is designed for situations where you need to generate random strings or
/// data sequences based on a specific set of characters, such as generating random DNA
/// sequences or testing inputs.
///
/// # Arguments
///
/// - `buffer`: A mutable reference to the data to randomize. This data will be mutated in place.
/// - `nonce`: A 64-bit "number used once", or nonce, value to seed the random number generator.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
/// let mut buffer = vec![0; 10];
/// sz::fill_random(&mut buffer, 42);
/// ```
///
/// After than,  `buffer` is filled with random byte values from 0 to 255.
pub fn fill_random<Buffer>(buffer: &mut Buffer, nonce: u64)
where
    Buffer: AsMut<[u8]> + ?Sized, // Allows for mutable references to dynamically sized types.
{
    let buffer_slice = buffer.as_mut();
    unsafe {
        sz_fill_random_best(
            buffer_slice.as_mut_ptr() as *mut c_void,
            buffer_slice.len(),
            nonce,
            Capabilities::CPUS.bits(),
            core::ptr::null_mut(),
        )
    }
    .infallible();
}

#[cfg(feature = "std")]
fn replace_all_with_finder<FindNext, FindPrev>(
    buffer: &mut Vec<u8>,
    needle_length: usize,
    replacement: &[u8],
    mut find_next: FindNext,
    mut find_prev: FindPrev,
) -> Result<usize, Status>
where
    FindNext: FnMut(&[u8], usize) -> Option<usize>,
    FindPrev: FnMut(&[u8], usize) -> Option<usize>,
{
    if needle_length == 0 || buffer.is_empty() {
        return Ok(0);
    }

    // Case 1: needle and replacement are the same length – overwrite each match in place.
    if needle_length == replacement.len() {
        let mut replaced = 0;
        let mut search_from = 0;
        while let Some(pos) = find_next(buffer.as_slice(), search_from) {
            buffer[pos..pos + needle_length].copy_from_slice(replacement);
            search_from = pos + needle_length;
            replaced += 1;
        }
        return Ok(replaced);
    }

    // Case 2: replacement is shorter – compact forward to minimize memmoves and avoid allocations.
    if needle_length > replacement.len() {
        let mut replaced = 0;
        let mut read = 0;
        let mut write = 0;
        let len = buffer.len();

        while let Some(pos) = find_next(buffer.as_slice(), read) {
            if pos > read {
                buffer.copy_within(read..pos, write);
                write += pos - read;
            }
            buffer[write..write + replacement.len()].copy_from_slice(replacement);
            write += replacement.len();
            read = pos + needle_length;
            replaced += 1;
        }

        if read < len {
            buffer.copy_within(read..len, write);
            write += len - read;
        }
        buffer.truncate(write);
        return Ok(replaced);
    }

    // Case 3: replacement is longer – collect match positions once, resize once, then rewrite
    // from the back.
    let mut match_count = 0usize;
    let mut search_from = 0;
    while let Some(pos) = find_next(buffer.as_slice(), search_from) {
        match_count += 1;
        search_from = pos + needle_length;
    }

    if match_count == 0 {
        return Ok(0);
    }

    let original_len = buffer.len();
    let delta = replacement.len() - needle_length;
    let added = match match_count.checked_mul(delta) {
        Some(v) => v,
        None => return Err(Status::OverflowRisk),
    };
    let new_len = match original_len.checked_add(added) {
        Some(v) => v,
        None => return Err(Status::OverflowRisk),
    };
    if buffer.try_reserve_exact(added).is_err() {
        return Err(Status::BadAlloc);
    }
    buffer.resize(new_len, 0);

    let mut read_end = original_len;
    let mut write_end = new_len;

    while let Some(pos) = find_prev(buffer.as_slice(), read_end) {
        let match_end = pos + needle_length;
        let tail_len = read_end - match_end;
        buffer.copy_within(match_end..read_end, write_end - tail_len);
        write_end -= tail_len;
        write_end -= replacement.len();
        buffer[write_end..write_end + replacement.len()].copy_from_slice(replacement);
        read_end = pos;
    }

    debug_assert_eq!(write_end, read_end, "replace_all backfill mismatch");
    Ok(match_count)
}

/// Replaces all non-overlapping occurrences of `needle` inside `buffer` in place.
///
/// - equal-length replacements simply overwrite matches,
/// - shorter replacements compact forward without allocating,
/// - longer replacements count matches once, resize once, and rewrite from the back.
///
/// Returns the number of replacements performed.
#[cfg(feature = "std")]
pub fn replace_all(buffer: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) -> Result<usize, Status> {
    replace_all_with_finder(
        buffer,
        needle.len(),
        replacement,
        |haystack, start| {
            if start >= haystack.len() {
                None
            } else {
                find(&haystack[start..], needle).map(|offset| start + offset)
            }
        },
        |haystack, end| {
            if end == 0 {
                None
            } else {
                rfind(&haystack[..end], needle)
            }
        },
    )
}

/// Replaces all non-overlapping bytes in `buffer` that belong to `byteset` with `replacement`.
///
/// Uses the same three-way strategy as [`replace_all`]. If the byteset is empty, the buffer is
/// left untouched. Returns the number of replacements performed.
#[cfg(feature = "std")]
pub fn replace_all_byteset(buffer: &mut Vec<u8>, byteset: Byteset, replacement: &[u8]) -> Result<usize, Status> {
    if byteset.bits.iter().all(|&b| b == 0) {
        return Ok(0);
    }

    replace_all_with_finder(
        buffer,
        1,
        replacement,
        |haystack, start| {
            if start >= haystack.len() {
                None
            } else {
                find_byteset(&haystack[start..], byteset).map(|offset| start + offset)
            }
        },
        |haystack, end| {
            if end == 0 {
                None
            } else {
                rfind_byteset(&haystack[..end], byteset)
            }
        },
    )
}

/// The text formats [`StringsView::escape_in`] and [`StringsView::unescape_in`] convert strings to
/// and from.
///
/// Corresponds to `sz_escape_kind_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EscapeKind {
    /// RFC 8259 string bodies: `\"`, `\\`, short forms like `\n`, and `\u00XX` for the other bytes
    /// below 0x20, every other byte kept as it is.
    Json = 0,
    /// Two lowercase hexadecimal digits per byte, decoding either case.
    Hex = 1,
    /// HTML and XML text and attribute values: `&amp;`, `&lt;`, `&gt;`, `&quot;` and `&#39;`.
    Html = 2,
    /// RFC 3986 percent-encoding of every byte outside `A-Z a-z 0-9 - . _ ~`.
    Url = 3,
    /// Terminal-safe logs: short forms like `\n`, `\xHH` for the other control bytes and ill-formed
    /// UTF-8, and `\uXXXX` for C1 controls and the bidirectional overrides that reorder what a
    /// reader sees.
    Console = 4,
    /// RFC 4180 fields, quoted only when holding a comma, a quote, CR or LF, with inner quotes
    /// doubled.
    Csv = 5,
    /// RFC 8259 string bodies in ASCII alone: every other codepoint becomes `\uXXXX`, a surrogate
    /// pair beyond the Basic Multilingual Plane, and every ill-formed byte the escaped U+FFFD.
    JsonAscii = 6,
}

/// How [`Strings::split_in`] ends its parts.
///
/// Corresponds to `sz_split_kind_t` in the C API.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitKind {
    /// Every delimiter ends a part, so neighbouring delimiters leave empty parts, as
    /// [`slice::split`] does at one byte.
    Each = 0,
    /// A run of delimiters ends one part and empty parts are skipped, as
    /// [`str::split_ascii_whitespace`] does at ASCII whitespace.
    Runs = 1,
}

/// Why [`StringsView::unescape_in`] or [`StringsView::utf8_count`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StringsError {
    /// The position of the first string holding a malformed sequence, like a lone surrogate in
    /// JSON, an unbalanced quote in a CSV field, or ill-formed UTF-8.
    Malformed(usize),
    /// Any other failure, like a block that could not grow.
    Failed(Status),
}

impl fmt::Display for StringsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StringsError::Malformed(index) => write!(f, "string {index} holds a malformed sequence"),
            StringsError::Failed(status) => fmt::Display::fmt(status, f),
        }
    }
}

impl core::error::Error for StringsError {}

/// The alignment of every block lent to C, which asks for no more than `malloc` gives.
const BLOCK_ALIGNMENT: usize = 16;

/// An owned column: one payload and its u32 or u64 offsets, both from `A`.
///
/// It is C's `sz_strings_span_t`, the layout of Apache Arrow's `String` and `LargeString` arrays,
/// with 32-bit offsets while the payload fits in 4 GiB and 64-bit beyond. Both blocks go back to
/// `A` on drop, and every conversion joins its stream first, so a column is always readable.
///
/// # Examples
///
/// ```
/// use std::alloc::Global;
/// use stringzilla::sz::{Capabilities, EscapeKind, Stream, Strings, StringsView};
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let names = Strings::copy_in(StringsView::from_slices(&["Ada", "Grace \"Amazing\" Hopper"]), Global, &cpu)?;
/// let fields = names.view().escape_in(EscapeKind::Csv, Global, &cpu)?;
/// assert_eq!(fields.view().get(1), Some(&br#""Grace ""Amazing"" Hopper""#[..]));
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
pub struct Strings<A: Allocator = Global> {
    span: sz_strings_span_t,
    allocator: A,
}

impl<A: Allocator> Strings<A> {
    /// Copies `strings` into a column from `allocator`, gathered on the device of `stream`, which
    /// on a GPU reads only tapes in memory it reaches.
    pub fn copy_in(strings: StringsView<'_>, allocator: A, stream: &Stream) -> Result<Self, Status> {
        Self::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_to_tape_best(
                &strings.view,
                core::ptr::null(),
                strings.view.count,
                allocator,
                span,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
    }

    /// Splits `text` at the bytes of `delimiters` into a column of the parts between them, every
    /// delimiter or every run of them ending a part as `kind` says.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::alloc::Global;
    /// use stringzilla::sz::{Byteset, Capabilities, SplitKind, Stream, Strings};
    ///
    /// let cpu = Stream::default(Capabilities::cpu_enabled());
    /// let fields = Strings::split_in(b"Ada,,Grace", Byteset::from(","), SplitKind::Each, Global, &cpu)?;
    /// assert!(fields.view().iter().eq([&b"Ada"[..], b"", b"Grace"]));
    /// let words = Strings::split_in(b" Ada  Grace ", Byteset::from(" "), SplitKind::Runs, Global, &cpu)?;
    /// assert!(words.view().iter().eq([&b"Ada"[..], b"Grace"]));
    /// # Ok::<(), stringzilla::sz::Status>(())
    /// ```
    pub fn split_in(
        text: &[u8],
        delimiters: Byteset,
        kind: SplitKind,
        allocator: A,
        stream: &Stream,
    ) -> Result<Self, Status> {
        Self::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_split_best(
                text.as_ptr(),
                text.len(),
                &delimiters,
                kind,
                allocator,
                span,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
    }

    /// Splits UTF-8 `text` into a column of its lines without their breaks, like Python's
    /// `str.splitlines()`, which takes CR LF as one break and leaves no empty line after the last.
    pub fn split_lines_in(text: &[u8], allocator: A, stream: &Stream) -> Result<Self, Status> {
        Self::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_split_lines_best(
                text.as_ptr(),
                text.len(),
                allocator,
                span,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
    }

    /// Borrows the column, to read it or convert it into another one.
    pub fn view(&self) -> StringsView<'_> {
        StringsView {
            view: sz_strings_view_t {
                layout: self.span.layout,
                count: self.span.count,
                data: sz_strings_data_t { tape: self.span.tape },
            },
            bytes: PhantomData,
        }
    }

    /// Fills an empty column from `allocator` through `write`, joining `stream` before anything
    /// reads it, and reports the failure of `write` ahead of the join's.
    fn written(
        allocator: A,
        stream: &Stream,
        write: impl FnOnce(&mut sz_allocator_t, &mut sz_strings_span_t) -> sz_status_t,
    ) -> Result<Self, Status> {
        let mut strings = Strings {
            span: sz_strings_span_t {
                layout: sz_strings_layout_t::U32Tape,
                count: 0,
                tape: sz_strings_tape_t {
                    data: core::ptr::null_mut(),
                    offsets: core::ptr::null_mut(),
                },
            },
            allocator,
        };
        let mut lent = lend(&strings.allocator);
        let status = write(&mut lent, &mut strings.span);
        let joined = stream.synchronize();
        status.check().and(joined).map(|()| strings)
    }
}

impl<A: Allocator> Drop for Strings<A> {
    fn drop(&mut self) {
        // C's `sz_strings_span_free`, which its header keeps inline, so the library exports none.
        let span = self.span;
        if span.tape.offsets.is_null() {
            return;
        }
        let width = match span.layout {
            sz_strings_layout_t::U64Tape => size_of::<u64>(),
            _ => size_of::<u32>(),
        };
        let payload = self.view().offset(span.count);
        let lent = lend(&self.allocator);
        unsafe {
            free_in::<A>(span.tape.data.cast(), payload, lent.handle, core::ptr::null_mut());
            free_in::<A>(
                span.tape.offsets,
                (span.count + 1) * width,
                lent.handle,
                core::ptr::null_mut(),
            );
        }
    }
}

/// Lends `allocator` to C for one call, as the handle its trampolines read it back from.
fn lend<A: Allocator>(allocator: &A) -> sz_allocator_t {
    sz_allocator_t {
        allocate: Some(allocate_in::<A>),
        reallocate: Some(reallocate_in::<A>),
        free: Some(free_in::<A>),
        handle: (allocator as *const A).cast_mut().cast(),
    }
}

/// The layout of a block of `bytes`, none for zero bytes, which C keeps as a null block.
fn block_layout(bytes: usize) -> Option<Layout> {
    if bytes == 0 {
        return None;
    }
    Layout::from_size_align(bytes, BLOCK_ALIGNMENT).ok()
}

unsafe extern "C" fn allocate_in<A: Allocator>(
    bytes: sz_size_t,
    handle: *mut c_void,
    _stream: sz_stream_t,
) -> *mut c_void {
    let allocator = &*handle.cast_const().cast::<A>();
    block_layout(bytes)
        .and_then(|layout| allocator.allocate(layout).ok())
        .map_or(core::ptr::null_mut(), |block| block.as_ptr().cast())
}

unsafe extern "C" fn reallocate_in<A: Allocator>(
    block: *mut c_void,
    old_bytes: sz_size_t,
    new_bytes: sz_size_t,
    handle: *mut c_void,
    stream: sz_stream_t,
) -> *mut c_void {
    let allocator = &*handle.cast_const().cast::<A>();
    let Some((block, old)) = NonNull::new(block.cast::<u8>()).zip(block_layout(old_bytes)) else {
        return allocate_in::<A>(new_bytes, handle, stream);
    };
    if new_bytes == 0 {
        allocator.deallocate(block, old);
        return core::ptr::null_mut();
    }
    // A size no layout holds fails like a refused one, leaving the block to C.
    let Some(new) = block_layout(new_bytes) else {
        return core::ptr::null_mut();
    };
    let moved = if new_bytes > old_bytes {
        allocator.grow(block, old, new)
    } else {
        allocator.shrink(block, old, new)
    };
    moved.map_or(core::ptr::null_mut(), |moved| moved.as_ptr().cast())
}

unsafe extern "C" fn free_in<A: Allocator>(
    block: *mut c_void,
    bytes: sz_size_t,
    handle: *mut c_void,
    _stream: sz_stream_t,
) {
    let allocator = &*handle.cast_const().cast::<A>();
    if let Some((block, layout)) = NonNull::new(block.cast::<u8>()).zip(block_layout(bytes)) {
        allocator.deallocate(block, layout);
    }
}

/// A borrowed column in any layout of `sz_strings_view_t`: an Arrow tape, or slices read through
/// an accessor.
///
/// Engines and conversions read it where it lies. A GPU reads only tapes in memory it reaches, like
/// a [`Strings`] from a [`UnifiedAllocator`], refusing any other view, accessors included, with
/// [`Status::DeviceMemoryMismatch`].
///
/// # Examples
///
/// ```
/// use stringzilla::sz::{Capabilities, LevenshteinEngine, LevenshteinSymbol, Status, Stream, StringsView};
///
/// // An Arrow `String` array sliced from its second element, so its offsets start past zero.
/// let (data, offsets) = (b"catsittingkitten", [3u32, 10, 16]);
/// let candidates = StringsView::from_u32_tape(data, &offsets)?;
/// assert!(candidates.iter().eq([&b"sitting"[..], b"kitten"]));
/// assert_eq!(StringsView::from_u32_tape(data, &[3, 17]).err(), Some(Status::UnexpectedDimensions));
///
/// let cpu = Stream::default(Capabilities::cpu_enabled());
/// let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &cpu)?;
/// let mut distances = [0usize; 2];
/// cpu.scope(|scope| engine.distances(scope, candidates, None, &mut distances, 2))?;
/// assert_eq!(distances, [3, 0]);
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
// A slice of views reaches C as an array of its `sz_strings_view_t`.
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct StringsView<'a> {
    pub(crate) view: sz_strings_view_t,
    bytes: PhantomData<&'a [u8]>,
}

impl<'a> StringsView<'a> {
    /// Views an Arrow `String` tape: string `i` spans `data[offsets[i]..offsets[i + 1]]`.
    ///
    /// Offsets that are empty, fall, or run past `data` are refused with
    /// [`Status::UnexpectedDimensions`].
    pub fn from_u32_tape(data: &'a [u8], offsets: &'a [u32]) -> Result<Self, Status> {
        Self::from_tape(data, offsets, sz_strings_layout_t::U32Tape)
    }

    /// [`StringsView::from_u32_tape`], for an Arrow `LargeString` tape.
    pub fn from_u64_tape(data: &'a [u8], offsets: &'a [u64]) -> Result<Self, Status> {
        Self::from_tape(data, offsets, sz_strings_layout_t::U64Tape)
    }

    /// Views `texts` where they lie, read through an accessor rather than copied.
    pub fn from_slices<Text: AsRef<[u8]>>(texts: &'a [Text]) -> Self {
        StringsView {
            view: sz_strings_view_t {
                layout: sz_strings_layout_t::Accessor,
                count: texts.len(),
                data: sz_strings_data_t {
                    accessor: sz_strings_accessor_t {
                        get: slices_get::<Text>,
                        handle: texts.as_ptr().cast(),
                    },
                },
            },
            bytes: PhantomData,
        }
    }

    /// Strings the column holds.
    pub fn len(&self) -> usize {
        self.view.count
    }

    /// Whether the column holds no strings.
    pub fn is_empty(&self) -> bool {
        self.view.count == 0
    }

    /// The bytes of string `index`, or `None` past the last one, like [`slice::get`].
    pub fn get(&self, index: usize) -> Option<&'a [u8]> {
        (index < self.view.count).then(|| self.string(index))
    }

    /// The strings in order, like [`slice::iter`].
    pub fn iter(&self) -> StringsIter<'a> {
        StringsIter {
            strings: *self,
            indices: 0..self.view.count,
        }
    }

    /// Escapes every string for `kind` into a new column from `allocator`, on the device of
    /// `stream`.
    pub fn escape_in<A: Allocator>(
        &self,
        kind: EscapeKind,
        allocator: A,
        stream: &Stream,
    ) -> Result<Strings<A>, Status> {
        Strings::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_escape_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                kind,
                allocator,
                span,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
    }

    /// Reverses [`StringsView::escape_in`] for `kind`, stopping at the first string holding a
    /// malformed sequence, whose position [`StringsError::Malformed`] reports.
    pub fn unescape_in<A: Allocator>(
        &self,
        kind: EscapeKind,
        allocator: A,
        stream: &Stream,
    ) -> Result<Strings<A>, StringsError> {
        let mut malformed = 0;
        Strings::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_unescape_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                kind,
                allocator,
                span,
                &mut malformed,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
        .map_err(|status| match status {
            Status::InvalidUtf8 => StringsError::Malformed(malformed),
            status => StringsError::Failed(status),
        })
    }

    /// Maps every byte of every string through `table` into a new column, as [`lookup`] does for
    /// one buffer.
    pub fn lookup_in<A: Allocator>(
        &self,
        table: &[u8; 256],
        allocator: A,
        stream: &Stream,
    ) -> Result<Strings<A>, Status> {
        Strings::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_lookup_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                table.as_ptr(),
                allocator,
                span,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
    }

    /// Brings every string into the Unicode `form` into a new column, as [`utf8_norm`] does for one
    /// string, copying the strings already in it as they are.
    pub fn normalize_in<A: Allocator>(
        &self,
        form: NormalForm,
        allocator: A,
        stream: &Stream,
    ) -> Result<Strings<A>, Status> {
        Strings::written(allocator, stream, |allocator, span| unsafe {
            sz_strings_normalize_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                form,
                allocator,
                span,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        })
    }

    /// Joins the strings into one buffer from `allocator`, with `between` between neighbours, like
    /// [`slice::join`].
    pub fn join_in<A: Allocator>(&self, between: &[u8], allocator: A, stream: &Stream) -> Result<Vec<u8, A>, Status> {
        Self::join_columns_in(&[*self], &[b"", b""], between, allocator, stream)
    }

    /// Writes row `r` of `columns` as `frames[0]`, `columns[0][r]`, `frames[1]` and so on through
    /// `frames[columns.len()]`, with `between` between rows, into one buffer from `allocator`, to
    /// serialize CSV lines or JSON records without a string per cell.
    ///
    /// Columns of different lengths, or `frames` not one longer than `columns`, are refused with
    /// [`Status::UnexpectedDimensions`].
    ///
    /// # Examples
    ///
    /// ```
    /// use std::alloc::Global;
    /// use stringzilla::sz::{Capabilities, Stream, StringsView};
    ///
    /// let cpu = Stream::default(Capabilities::cpu_enabled());
    /// let names = StringsView::from_slices(&["Ada", "Grace"]);
    /// let years = StringsView::from_u32_tape(b"18151906", &[0, 4, 8])?;
    /// let csv = StringsView::join_columns_in(&[names, years], &[b"", b",", b""], b"\n", Global, &cpu)?;
    /// assert_eq!(csv, b"Ada,1815\nGrace,1906");
    /// # Ok::<(), stringzilla::sz::Status>(())
    /// ```
    pub fn join_columns_in<A: Allocator>(
        columns: &[StringsView<'_>],
        frames: &[&[u8]],
        between: &[u8],
        allocator: A,
        stream: &Stream,
    ) -> Result<Vec<u8, A>, Status> {
        let rows = columns.first().map_or(0, StringsView::len);
        if frames.len() != columns.len() + 1 || columns.iter().any(|column| column.len() != rows) {
            return Err(Status::UnexpectedDimensions);
        }
        let mut views = Vec::new_in(&allocator);
        views.try_reserve_exact(frames.len()).map_err(|_| Status::BadAlloc)?;
        views.extend(frames.iter().map(|frame| sz_string_view_t {
            start: frame.as_ptr(),
            length: frame.len(),
        }));
        let join = |output: *mut u8, length: &mut usize| unsafe {
            sz_strings_join_best(
                columns.as_ptr().cast(),
                columns.len(),
                rows,
                views.as_ptr(),
                sz_string_view_t {
                    start: between.as_ptr(),
                    length: between.len(),
                },
                output,
                length,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        };
        let mut length = 0;
        join(core::ptr::null_mut(), &mut length).check()?;
        let mut output = Vec::new_in(&allocator);
        output.try_reserve_exact(length).map_err(|_| Status::BadAlloc)?;
        let status = join(output.as_mut_ptr(), &mut length);
        let joined = stream.synchronize();
        status.check().and(joined)?;
        // A block from `&A` is a block from `A`, which takes it over once nothing borrows `A`.
        let (pointer, capacity) = (output.as_mut_ptr(), output.capacity());
        core::mem::forget(output);
        drop(views);
        Ok(unsafe { Vec::from_raw_parts_in(pointer, length, capacity, allocator) })
    }

    /// Writes every string into its own NUL-padded row of `width` bytes, back to back in `rows`,
    /// truncating longer ones, and returns the longest length, to retry with wider rows.
    ///
    /// `rows` holds at least `len() * width` bytes, or [`Status::UnexpectedDimensions`] is
    /// returned, and on a GPU it lies in memory the device reaches, like a `Vec` in a
    /// [`UnifiedAllocator`].
    pub fn to_rows(&self, rows: &mut [u8], width: usize, stream: &Stream) -> Result<usize, Status> {
        let rows_bytes = self.view.count.checked_mul(width).ok_or(Status::OverflowRisk)?;
        if rows.len() < rows_bytes {
            return Err(Status::UnexpectedDimensions);
        }
        let target = sz_strings_rows_t {
            data: rows.as_mut_ptr(),
            stride: width,
            width,
        };
        // A device writes the longest length too, so it lands in memory that device reaches.
        let unified = UnifiedAllocator::new(stream);
        let layout = Layout::new::<usize>();
        let longest = unified.allocate(layout).map_err(|_| Status::BadAlloc)?.cast::<usize>();
        let status = unsafe {
            longest.write(0);
            sz_strings_to_rows_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                &target,
                longest.as_ptr(),
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        };
        let joined = stream.synchronize();
        let result = status.check().and(joined).map(|()| unsafe { longest.read() });
        unsafe { unified.deallocate(longest.cast(), layout) };
        result
    }

    /// Finds `needle` in every string, as [`find`] does in one, writing the offset of each first
    /// match into `positions`, or `usize::MAX` where a string holds none.
    ///
    /// `positions` holds exactly `len()` entries, or [`Status::UnexpectedDimensions`] is returned.
    /// A match crossing from one string of a tape into the next counts for neither.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::{Capabilities, Stream, StringsView};
    ///
    /// let cpu = Stream::default(Capabilities::cpu_enabled());
    /// let requests = StringsView::from_slices(&["GET /index", "POST /login", "GET /login"]);
    /// let mut positions = [0usize; 3];
    /// requests.find(b"login", &mut positions, &cpu)?;
    /// assert_eq!(positions, [usize::MAX, 6, 5]);
    /// # Ok::<(), stringzilla::sz::Status>(())
    /// ```
    pub fn find(&self, needle: &[u8], positions: &mut [usize], stream: &Stream) -> Result<(), Status> {
        if positions.len() != self.view.count {
            return Err(Status::UnexpectedDimensions);
        }
        let status = unsafe {
            sz_find_strings_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                needle.as_ptr(),
                needle.len(),
                positions.as_mut_ptr(),
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        };
        let joined = stream.synchronize();
        status.check().and(joined)
    }

    /// Hashes every string under `seed` into `hashes`, each exactly as [`hash_with_seed`] would.
    ///
    /// `hashes` holds exactly `len()` entries, or [`Status::UnexpectedDimensions`] is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::{self, Capabilities, Stream, StringsView};
    ///
    /// let cpu = Stream::default(Capabilities::cpu_enabled());
    /// let keys = StringsView::from_u32_tape(b"adagrace", &[0, 3, 8])?;
    /// let mut hashes = [0u64; 2];
    /// keys.hashes(42, &mut hashes, &cpu)?;
    /// assert_eq!(hashes, [sz::hash_with_seed("ada", 42), sz::hash_with_seed("grace", 42)]);
    /// # Ok::<(), stringzilla::sz::Status>(())
    /// ```
    pub fn hashes(&self, seed: u64, hashes: &mut [u64], stream: &Stream) -> Result<(), Status> {
        if hashes.len() != self.view.count {
            return Err(Status::UnexpectedDimensions);
        }
        let status = unsafe {
            sz_hash_strings_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                seed,
                hashes.as_mut_ptr(),
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        };
        let joined = stream.synchronize();
        status.check().and(joined)
    }

    /// Checks that every string is well-formed UTF-8, reporting the first one that is not as
    /// [`StringsError::Malformed`], and writes each one's rune count into `runes`, unless it is
    /// `None`, leaving the counts from the malformed string on unspecified.
    ///
    /// `runes` holds exactly `len()` entries, or [`Status::UnexpectedDimensions`] is returned. A
    /// sequence crossing from one string of a tape into the next is ill-formed.
    pub fn utf8_count(&self, runes: Option<&mut [usize]>, stream: &Stream) -> Result<(), StringsError> {
        if runes.as_ref().is_some_and(|runes| runes.len() != self.view.count) {
            return Err(StringsError::Failed(Status::UnexpectedDimensions));
        }
        let mut malformed = 0;
        let status = unsafe {
            sz_utf8_count_strings_best(
                &self.view,
                core::ptr::null(),
                self.view.count,
                runes.map_or(core::ptr::null_mut(), |runes| runes.as_mut_ptr()),
                &mut malformed,
                stream.capabilities().bits(),
                stream.as_raw(),
            )
        };
        let joined = stream.synchronize();
        status.check().and(joined).map_err(|status| match status {
            Status::InvalidUtf8 => StringsError::Malformed(malformed),
            status => StringsError::Failed(status),
        })
    }

    /// Builds a tape view over `offsets` of the width `layout` names, if they ascend within `data`.
    fn from_tape<Offset: Copy + Into<u64>>(
        data: &'a [u8],
        offsets: &'a [Offset],
        layout: sz_strings_layout_t,
    ) -> Result<Self, Status> {
        let count = offsets.len().checked_sub(1).ok_or(Status::UnexpectedDimensions)?;
        let ascending = offsets.windows(2).all(|pair| pair[0].into() <= pair[1].into());
        if !ascending || offsets[count].into() > data.len() as u64 {
            return Err(Status::UnexpectedDimensions);
        }
        Ok(StringsView {
            view: sz_strings_view_t {
                layout,
                count,
                data: sz_strings_data_t {
                    tape: sz_strings_tape_t {
                        data: data.as_ptr().cast_mut(),
                        offsets: offsets.as_ptr().cast_mut().cast(),
                    },
                },
            },
            bytes: PhantomData,
        })
    }

    /// Offset `index` of a tape, 32-bit or 64-bit as its layout says.
    fn offset(&self, index: usize) -> usize {
        let offsets = unsafe { self.view.data.tape.offsets };
        match self.view.layout {
            sz_strings_layout_t::U64Tape => unsafe { *offsets.cast::<u64>().add(index) as usize },
            _ => unsafe { *offsets.cast::<u32>().add(index) as usize },
        }
    }

    /// The bytes of string `index`, which must be below the count.
    fn string(&self, index: usize) -> &'a [u8] {
        let (start, length) = match self.view.layout {
            sz_strings_layout_t::Accessor => {
                let accessor = unsafe { self.view.data.accessor };
                let mut string = sz_string_view_t {
                    start: core::ptr::null(),
                    length: 0,
                };
                unsafe { (accessor.get)(accessor.handle, &index, 1, &mut string) };
                (string.start, string.length)
            }
            _ => {
                let (first, end) = (self.offset(index), self.offset(index + 1));
                (
                    unsafe { self.view.data.tape.data.cast_const().wrapping_add(first) },
                    end - first,
                )
            }
        };
        match length {
            0 => &[],
            length => unsafe { core::slice::from_raw_parts(start, length) },
        }
    }
}

unsafe extern "C" fn slices_get<Text: AsRef<[u8]>>(
    handle: *const c_void,
    indices: *const sz_size_t,
    count: sz_size_t,
    views: *mut sz_string_view_t,
) {
    let texts = handle.cast::<Text>();
    for position in 0..count {
        let bytes = (*texts.add(*indices.add(position))).as_ref();
        views.add(position).write(sz_string_view_t {
            start: bytes.as_ptr(),
            length: bytes.len(),
        });
    }
}

/// The strings of a [`StringsView`] in order, borrowing what the view borrows.
#[derive(Clone)]
pub struct StringsIter<'a> {
    strings: StringsView<'a>,
    indices: core::ops::Range<usize>,
}

impl<'a> Iterator for StringsIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<&'a [u8]> {
        self.indices.next().map(|index| self.strings.string(index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.indices.size_hint()
    }
}

impl ExactSizeIterator for StringsIter<'_> {}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::alloc::Global;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::alloc::{AllocError, Allocator, Layout};
    use core::cell::Cell;
    use core::ptr::NonNull;

    use crate::sz;
    use crate::sz::{Byteset, EscapeKind, SplitKind, Status, Strings, StringsError, StringsView};

    const KINDS: [EscapeKind; 7] = [
        EscapeKind::Json,
        EscapeKind::Hex,
        EscapeKind::Html,
        EscapeKind::Url,
        EscapeKind::Console,
        EscapeKind::Csv,
        EscapeKind::JsonAscii,
    ];

    fn cpu() -> sz::Stream {
        sz::Stream::default(sz::Capabilities::cpu_enabled())
    }

    // region: Buffers

    #[test]
    fn fill_random() {
        let mut first_buffer: Vec<u8> = vec![0; 10]; // Ten zeros
        let mut second_buffer: Vec<u8> = vec![1; 10]; // Ten ones
        sz::fill_random(&mut first_buffer, 42);
        sz::fill_random(&mut second_buffer, 42);

        // Same nonce will produce the same outputs
        assert_eq!(first_buffer, second_buffer);
    }

    #[test]
    fn size_checks() {
        let long: Vec<u8> = vec![7; 20];
        let mut less_long: Vec<u8> = vec![0; 10];
        let lut: [u8; 256] = core::array::from_fn(|i| i as u8);

        assert_eq!(
            sz::lookup(&mut less_long, &long, lut),
            Err(sz::Status::UnexpectedDimensions)
        );
        assert_eq!(less_long, vec![0; 10]);

        assert_eq!(sz::lookup(&mut less_long, &long[..10], lut), Ok(()));
        assert_eq!(less_long, vec![7; 10]);
    }

    // endregion

    // region: Views

    #[test]
    fn tapes_refuse_offsets_that_are_empty_fall_or_overrun_their_data() {
        let data = b"kittensitting";
        assert_eq!(
            StringsView::from_u32_tape(data, &[]).err(),
            Some(Status::UnexpectedDimensions)
        );
        assert_eq!(
            StringsView::from_u32_tape(data, &[0, 6, 4]).err(),
            Some(Status::UnexpectedDimensions)
        );
        assert_eq!(
            StringsView::from_u32_tape(data, &[0, 6, 14]).err(),
            Some(Status::UnexpectedDimensions)
        );
        assert_eq!(
            StringsView::from_u64_tape(data, &[7, 6]).err(),
            Some(Status::UnexpectedDimensions)
        );
        assert_eq!(
            StringsView::from_u64_tape(data, &[0, u64::MAX]).err(),
            Some(Status::UnexpectedDimensions)
        );

        let single = StringsView::from_u32_tape(data, &[13]).unwrap();
        assert!(single.is_empty() && single.get(0).is_none());
        let wide = StringsView::from_u64_tape(data, &[0, 6, 6, 13]).unwrap();
        assert!(wide.iter().eq([&b"kitten"[..], b"", b"sitting"]));
    }

    /// A tape copied from a sliced Arrow array starts its payload at the slice's first offset.
    #[test]
    fn copies_of_sliced_tapes_rebase_their_offsets() {
        let cpu = cpu();
        let sliced = StringsView::from_u32_tape(b"skipsittingkitten", &[4, 11, 17]).unwrap();
        let copy = Strings::copy_in(sliced, Global, &cpu).unwrap();
        assert!(copy.view().iter().eq(sliced.iter()));
        assert_eq!(copy.view().offset(0), 0);
    }

    // endregion

    // region: Conversions

    /// A column reads back what it copied, escapes to the RFC forms, and unescapes every kind back.
    #[test]
    fn columns_escape_and_unescape() {
        let cpu = cpu();
        let names = [
            "Ada",
            "Grace \"Amazing\" Hopper",
            "Linus\nTorvalds",
            "",
            "caf\u{e9} \u{1f600}",
        ];
        let column = Strings::copy_in(StringsView::from_slices(&names), Global, &cpu).unwrap();
        let view = column.view();
        assert!(view.iter().eq(names.iter().map(|name| name.as_bytes())));
        assert_eq!(view.get(names.len()), None);
        let nothing = Strings::copy_in(StringsView::from_slices(&[] as &[&str]), Global, &cpu).unwrap();
        let nothing_escaped = nothing.view().escape_in(EscapeKind::Json, Global, &cpu).unwrap();
        assert!(nothing.view().is_empty() && nothing_escaped.view().is_empty());

        let json = view.escape_in(EscapeKind::Json, Global, &cpu).unwrap();
        assert_eq!(json.view().get(1), Some(&br#"Grace \"Amazing\" Hopper"#[..]));
        assert_eq!(json.view().get(2), Some(&br"Linus\nTorvalds"[..]));
        let ascii = view.escape_in(EscapeKind::JsonAscii, Global, &cpu).unwrap();
        assert_eq!(ascii.view().get(4), Some(&b"caf\\u00e9 \\ud83d\\ude00"[..]));
        let csv = view.escape_in(EscapeKind::Csv, Global, &cpu).unwrap();
        assert_eq!(csv.view().get(0), Some(&b"Ada"[..]));
        assert_eq!(csv.view().get(1), Some(&br#""Grace ""Amazing"" Hopper""#[..]));

        for kind in KINDS {
            let escaped = view.escape_in(kind, Global, &cpu).unwrap();
            let restored = escaped.view().unescape_in(kind, Global, &cpu).unwrap();
            assert!(restored.view().iter().eq(view.iter()), "{kind:?}");
        }
    }

    #[test]
    fn unescape_reports_the_first_malformed_string() {
        let fields = StringsView::from_slices(&["plain", "\"closed\"", "\"open", "\"also open"]);
        assert_eq!(
            fields.unescape_in(EscapeKind::Csv, Global, &cpu()).err(),
            Some(StringsError::Malformed(2))
        );
    }

    /// Lookup, normalization and rows read a column like the strings it was copied from.
    #[test]
    fn columns_map_normalize_and_pad() {
        let cpu = cpu();
        let words = StringsView::from_slices(&["Stra\u{df}e", "\u{fb01}le", "Ab"]);

        let upper: [u8; 256] = core::array::from_fn(|byte| (byte as u8).to_ascii_uppercase());
        let shouted = words.lookup_in(&upper, Global, &cpu).unwrap();
        assert!(shouted
            .view()
            .iter()
            .eq(["STRA\u{df}E".as_bytes(), "\u{fb01}LE".as_bytes(), b"AB"]));

        let folded = words.normalize_in(sz::NormalForm::NfkcCasefold, Global, &cpu).unwrap();
        assert!(folded.view().iter().eq([&b"strasse"[..], b"file", b"ab"]));

        let mut rows = [0xFF; 12];
        assert_eq!(words.to_rows(&mut rows, 4, &cpu), Ok(7));
        assert_eq!(&rows, b"Stra\xEF\xAC\x81lAb\0\0");
        assert_eq!(
            words.to_rows(&mut rows[..11], 4, &cpu),
            Err(Status::UnexpectedDimensions)
        );
    }

    /// Texts past one SIMD block, with delimiters leading, trailing, neighbouring, or absent.
    const SPLIT_TEXTS: [&str; 6] = [
        "",
        ",",
        "Ada",
        ",Ada,,Grace,",
        "a comma-free text running well past sixteen bytes and then past sixty-four of them, too",
        "Ada,Grace,Linus,Barbara,Edsger,Donald,Frances,Margaret,Radia,Ken,Dennis,Bjarne,Guido,,",
    ];

    #[test]
    fn each_delimiter_splits_like_slice_split() {
        let cpu = cpu();
        for text in SPLIT_TEXTS.map(str::as_bytes) {
            let parts = Strings::split_in(text, Byteset::from(","), SplitKind::Each, Global, &cpu).unwrap();
            assert!(parts.view().iter().eq(text.split(|&byte| byte == b',')), "{text:?}");
        }
    }

    #[test]
    fn delimiter_runs_split_like_whitespace_split() {
        let cpu = cpu();
        let whitespace = Byteset::from(" \t\n\x0c\r");
        for text in SPLIT_TEXTS.map(|text| text.replace(',', " \t")) {
            let words = Strings::split_in(text.as_bytes(), whitespace, SplitKind::Runs, Global, &cpu).unwrap();
            let expected = text.split_ascii_whitespace().map(str::as_bytes);
            assert!(words.view().iter().eq(expected), "{text:?}");
        }
    }

    #[test]
    fn line_splits_follow_python_splitlines() {
        let cpu = cpu();
        let every_break = "1\r\n2\r3\n4\u{85}5\u{2028}6\u{2029}7\x0b8\x0c9\x1c10\x1d11\x1e12\r\n";
        let cases: [(&str, &[&str]); 6] = [
            ("", &[]),
            ("\n", &[""]),
            ("a\n\nb\n", &["a", "", "b"]),
            ("a\r\r\nb", &["a", "", "b"]),
            ("no break", &["no break"]),
            (
                every_break,
                &["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
            ),
        ];
        for (text, expected) in cases {
            let lines = Strings::split_lines_in(text.as_bytes(), Global, &cpu).unwrap();
            assert!(
                lines.view().iter().eq(expected.iter().map(|line| line.as_bytes())),
                "{text:?}"
            );
        }
    }

    /// Kernels place and copy rows 64 at a time, so the longest column spans three such blocks.
    #[test]
    fn joins_put_the_separator_only_between_strings() {
        let cpu = cpu();
        let nothing = StringsView::from_slices(&[] as &[&str]);
        assert_eq!(nothing.join_in(b", ", Global, &cpu).unwrap(), b"");
        let one = StringsView::from_slices(&["Ada"]);
        assert_eq!(one.join_in(b", ", Global, &cpu).unwrap(), b"Ada");
        every_layout(|haystacks| {
            assert_eq!(
                haystacks.join_in(b"|", Global, &cpu).unwrap(),
                HAYSTACKS.join("|").as_bytes()
            );
        });
        let letters: Vec<u8> = (0..130).map(|index| b'a' + index % 26).collect();
        let offsets: Vec<u32> = (0..=130).collect();
        let column = StringsView::from_u32_tape(&letters, &offsets).unwrap();
        let joined = column.join_in(b",", Global, &cpu).unwrap();
        assert!(joined.split(|&byte| byte == b',').eq(column.iter()));
    }

    #[test]
    fn column_joins_frame_every_row() {
        let cpu = cpu();
        let names = StringsView::from_slices(&["Ada", "Grace"]);
        let years = StringsView::from_u32_tape(b"18151906", &[0, 4, 8]).unwrap();
        let csv = StringsView::join_columns_in(&[names, years], &[b"", b",", b""], b"\r\n", Global, &cpu).unwrap();
        assert_eq!(csv, b"Ada,1815\r\nGrace,1906");
        let frames: [&[u8]; 3] = [br#"{"name":""#, br#"","born":"#, b"}"];
        let json = StringsView::join_columns_in(&[names, years], &frames, b",", Global, &cpu).unwrap();
        assert_eq!(json, br#"{"name":"Ada","born":1815},{"name":"Grace","born":1906}"#);

        let refused = Some(Status::UnexpectedDimensions);
        let unframed = StringsView::join_columns_in(&[names, years], &[b"", b""], b"\n", Global, &cpu);
        assert_eq!(unframed.err(), refused);
        let short = StringsView::from_slices(&["Ada"]);
        let ragged = StringsView::join_columns_in(&[names, short], &[b"", b",", b""], b"\n", Global, &cpu);
        assert_eq!(ragged.err(), refused);
    }

    // endregion

    // region: Searching and Hashing

    /// Empty, short and past-16-byte strings, ending in "nee" and "dle", which a tape holds back to
    /// back as "needle".
    const HAYSTACKS: [&str; 8] = [
        "",
        "a",
        "needle",
        "haystack with a needle inside",
        "needlessly long haystack without the word, past sixteen bytes",
        "caf\u{e9} needle \u{1f600}",
        "nee",
        "dle",
    ];

    /// Runs `check` over the haystacks as slices, as a sliced Arrow tape of either offset width,
    /// and as an owned column.
    fn every_layout(check: impl Fn(StringsView<'_>)) {
        let mut data = b"skip".to_vec();
        let mut narrow = vec![data.len() as u32];
        for haystack in HAYSTACKS {
            data.extend_from_slice(haystack.as_bytes());
            narrow.push(data.len() as u32);
        }
        let wide: Vec<u64> = narrow.iter().map(|&offset| offset.into()).collect();
        let slices = StringsView::from_slices(&HAYSTACKS);
        let owned = Strings::copy_in(slices, Global, &cpu()).unwrap();
        check(slices);
        check(StringsView::from_u32_tape(&data, &narrow).unwrap());
        check(StringsView::from_u64_tape(&data, &wide).unwrap());
        check(owned.view());
    }

    /// A match crossing from one string of a tape into the next, like "a" + "nee" or "nee" +
    /// "dle", counts for neither.
    #[test]
    fn column_finds_agree_with_finds_in_each_string() {
        let cpu = cpu();
        let needles: [&[u8]; 6] = [
            b"needle",
            b"",
            b"e",
            "\u{1f600}".as_bytes(),
            b"anee",
            b"no such text in any haystack",
        ];
        every_layout(|haystacks| {
            for needle in needles {
                let mut positions = vec![0; haystacks.len()];
                haystacks.find(needle, &mut positions, &cpu).unwrap();
                let expected = haystacks
                    .iter()
                    .map(|haystack| sz::find(haystack, needle).unwrap_or(usize::MAX));
                assert!(positions.iter().copied().eq(expected), "{needle:?}");
            }
        });
    }

    #[test]
    fn column_hashes_agree_with_hashes_of_each_string() {
        let cpu = cpu();
        every_layout(|keys| {
            for seed in [0, 42, u64::MAX] {
                let mut hashes = vec![0; keys.len()];
                keys.hashes(seed, &mut hashes, &cpu).unwrap();
                let expected = keys.iter().map(|key| sz::hash_with_seed(key, seed));
                assert!(hashes.iter().copied().eq(expected), "{seed}");
            }
        });
    }

    #[test]
    fn column_rune_counts_agree_with_char_counts() {
        let cpu = cpu();
        every_layout(|texts| {
            let mut runes = vec![0; texts.len()];
            texts.utf8_count(Some(&mut runes), &cpu).unwrap();
            let expected = texts
                .iter()
                .map(|text| core::str::from_utf8(text).unwrap().chars().count());
            assert!(runes.iter().copied().eq(expected));
            assert_eq!(texts.utf8_count(None, &cpu), Ok(()));
        });
    }

    #[test]
    fn rune_counts_report_the_first_malformed_string() {
        let cpu = cpu();
        let rows: [&[u8]; 4] = [b"fine", "caf\u{e9}".as_bytes(), b"\xFF", b"\xC0\x80"];
        let rows = StringsView::from_slices(&rows);
        assert_eq!(rows.utf8_count(None, &cpu), Err(StringsError::Malformed(2)));
        assert_eq!(
            rows.utf8_count(Some(&mut [0; 4]), &cpu),
            Err(StringsError::Malformed(2))
        );
        let crossing = StringsView::from_u32_tape("caf\u{e9}".as_bytes(), &[0, 4, 5]).unwrap();
        assert_eq!(crossing.utf8_count(None, &cpu), Err(StringsError::Malformed(0)));
    }

    #[test]
    fn column_finds_hashes_and_counts_refuse_outputs_of_another_length() {
        let cpu = cpu();
        let column = StringsView::from_slices(&HAYSTACKS);
        for length in [HAYSTACKS.len() - 1, HAYSTACKS.len() + 1] {
            let refused = Err(Status::UnexpectedDimensions);
            assert_eq!(column.find(b"needle", &mut vec![0; length], &cpu), refused);
            assert_eq!(column.hashes(0, &mut vec![0; length], &cpu), refused);
            assert_eq!(
                column.utf8_count(Some(&mut vec![0; length]), &cpu),
                refused.map_err(StringsError::Failed)
            );
        }
    }

    // endregion

    // region: Allocators

    /// A distinct allocator forwarding to the global heap, to prove any allocator holds a column.
    #[derive(Clone, Copy)]
    struct Passthrough;

    unsafe impl Allocator for Passthrough {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            Global.allocate(layout)
        }
        unsafe fn deallocate(&self, block: NonNull<u8>, layout: Layout) {
            Global.deallocate(block, layout)
        }
    }

    /// Refuses its `refused`-th request, counting the blocks it has handed out and not taken back.
    struct Budget {
        refused: usize,
        requests: Cell<usize>,
        live: Cell<isize>,
    }

    impl Budget {
        fn new(refused: usize) -> Self {
            Budget {
                refused,
                requests: Cell::new(0),
                live: Cell::new(0),
            }
        }
    }

    // Growing and shrinking fall back to `allocate`, so a refused resize is swept too.
    unsafe impl Allocator for &Budget {
        fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
            let request = self.requests.get();
            self.requests.set(request + 1);
            if request == self.refused {
                return Err(AllocError);
            }
            let block = Global.allocate(layout)?;
            self.live.set(self.live.get() + 1);
            Ok(block)
        }
        unsafe fn deallocate(&self, block: NonNull<u8>, layout: Layout) {
            self.live.set(self.live.get() - 1);
            Global.deallocate(block, layout)
        }
    }

    /// Runs `convert` refusing each of its allocations in turn, every attempt either succeeding or
    /// failing with [`Status::BadAlloc`], and leaving no block behind once its column drops.
    fn sweep_allocation_failures(convert: impl Fn(&Budget) -> Result<(), StringsError>) {
        let unlimited = Budget::new(usize::MAX);
        convert(&unlimited).unwrap();
        let needed = unlimited.requests.get();
        assert!(needed > 0 && unlimited.live.get() == 0);
        for refused in 0..needed {
            let budget = Budget::new(refused);
            match convert(&budget) {
                Ok(()) | Err(StringsError::Failed(Status::BadAlloc)) => {}
                Err(other) => panic!("refusing request {refused} failed with {other:?}"),
            }
            assert_eq!(budget.live.get(), 0, "refusing request {refused} leaked");
        }
    }

    const NAMES: [&str; 5] = [
        "Ada",
        "Grace \"Amazing\" Hopper",
        "Linus\nTorvalds",
        "",
        "caf\u{e9} \u{1f600}",
    ];

    #[test]
    fn copies_fail_cleanly_at_every_refused_allocation() {
        let cpu = cpu();
        let slices = StringsView::from_slices(&NAMES);
        let tape = Strings::copy_in(slices, Global, &cpu).unwrap();
        for source in [slices, tape.view()] {
            sweep_allocation_failures(|budget| {
                let copy = Strings::copy_in(source, budget, &cpu).map_err(StringsError::Failed)?;
                assert!(copy.view().iter().eq(slices.iter()));
                Ok(())
            });
        }
    }

    #[test]
    fn escapes_and_unescapes_fail_cleanly_at_every_refused_allocation() {
        let cpu = cpu();
        let names = StringsView::from_slices(&NAMES);
        for kind in KINDS {
            let escaped = names.escape_in(kind, Global, &cpu).unwrap();
            sweep_allocation_failures(|budget| {
                let copy = names.escape_in(kind, budget, &cpu).map_err(StringsError::Failed)?;
                assert!(copy.view().iter().eq(escaped.view().iter()));
                Ok(())
            });
            sweep_allocation_failures(|budget| {
                let restored = escaped.view().unescape_in(kind, budget, &cpu)?;
                assert!(restored.view().iter().eq(names.iter()));
                Ok(())
            });
        }
    }

    #[test]
    fn splits_and_joins_fail_cleanly_at_every_refused_allocation() {
        let cpu = cpu();
        let text = b"Ada,Grace,,Linus,";
        sweep_allocation_failures(|budget| {
            let parts = Strings::split_in(text, Byteset::from(","), SplitKind::Each, budget, &cpu)
                .map_err(StringsError::Failed)?;
            assert!(parts.view().iter().eq(text.split(|&byte| byte == b',')));
            let joined = parts.view().join_in(b",", budget, &cpu).map_err(StringsError::Failed)?;
            assert_eq!(joined, text);
            Ok(())
        });
    }

    /// Empty strings take no payload block, which C keeps null and frees as zero bytes.
    #[test]
    fn columns_of_empty_strings_hold_only_their_offsets() {
        let budget = Budget::new(usize::MAX);
        let blanks = Strings::copy_in(StringsView::from_slices(&["", "", ""]), &budget, &cpu()).unwrap();
        assert_eq!(budget.live.get(), 1);
        assert!(blanks.view().iter().eq([&b""[..]; 3]));
        drop(blanks);
        assert_eq!(budget.live.get(), 0);
    }

    /// C's resizes map onto the allocator's own, a zero-byte block being a null one either way.
    #[test]
    fn lent_reallocations_grow_shrink_and_free_through_the_allocator() {
        let budget = Budget::new(usize::MAX);
        let allocator = &budget;
        let lent = super::lend(&allocator);
        let (allocate, reallocate, free) = (lent.allocate.unwrap(), lent.reallocate.unwrap(), lent.free.unwrap());
        let stream = core::ptr::null_mut();
        unsafe {
            assert!(allocate(0, lent.handle, stream).is_null());
            let block = reallocate(core::ptr::null_mut(), 0, 3, lent.handle, stream).cast::<u8>();
            assert_eq!(block.addr() % super::BLOCK_ALIGNMENT, 0);
            block.copy_from_nonoverlapping(b"abc".as_ptr(), 3);
            let grown = reallocate(block.cast(), 3, 300, lent.handle, stream).cast::<u8>();
            let shrunk = reallocate(grown.cast(), 300, 2, lent.handle, stream).cast::<u8>();
            assert_eq!(core::slice::from_raw_parts(shrunk, 2), b"ab");
            assert_eq!(budget.live.get(), 1);
            assert!(reallocate(shrunk.cast(), 2, 0, lent.handle, stream).is_null());
            free(core::ptr::null_mut(), 0, lent.handle, stream);
        }
        assert_eq!(budget.live.get(), 0);
    }

    #[test]
    fn any_allocator_holds_a_column() {
        let cpu = cpu();
        let column = Strings::copy_in(StringsView::from_slices(&NAMES), Passthrough, &cpu).unwrap();
        let escaped = column.view().escape_in(EscapeKind::Url, Passthrough, &cpu).unwrap();
        let restored = escaped.view().unescape_in(EscapeKind::Url, Passthrough, &cpu).unwrap();
        assert!(restored.view().iter().eq(NAMES.iter().map(|name| name.as_bytes())));
    }

    // endregion

    // region: Metal

    /// A column in Metal memory converts on the GPU into what the host writes, and a Metal engine
    /// reads it in place, while blocks or views the GPU cannot reach are refused.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_columns_convert_like_host_ones() {
        let Ok(capabilities) = sz::Capabilities::metal_enabled(0) else {
            return;
        };
        if !capabilities.contains(sz::Capability::Metal) {
            return;
        }
        let gpu = sz::Stream::new(capabilities, 0).unwrap();
        let cpu = cpu();
        let unified = sz::UnifiedAllocator::new(&gpu);
        let words = ["Grace \"Amazing\" Hopper", "", "Stra\u{df}e", "a,b\r\n", "kitten"];
        let slices = StringsView::from_slices(&words);
        // Only the host reads slices, gathering them into blocks the GPU reaches.
        let device = Strings::copy_in(slices, unified, &cpu).unwrap();
        let host = Strings::copy_in(slices, Global, &cpu).unwrap();
        let (device_view, host_view) = (device.view(), host.view());
        assert!(Strings::copy_in(device_view, unified, &gpu)
            .unwrap()
            .view()
            .iter()
            .eq(slices.iter()));
        assert_eq!(
            slices.escape_in(EscapeKind::Json, unified, &gpu).err(),
            Some(Status::DeviceMemoryMismatch)
        );
        assert_eq!(
            device_view.escape_in(EscapeKind::Json, Global, &gpu).err(),
            Some(Status::DeviceMemoryMismatch)
        );

        for kind in [EscapeKind::Json, EscapeKind::Csv, EscapeKind::Url] {
            let device_escaped = device_view.escape_in(kind, unified, &gpu).unwrap();
            let host_escaped = host_view.escape_in(kind, Global, &cpu).unwrap();
            assert!(device_escaped.view().iter().eq(host_escaped.view().iter()), "{kind:?}");
            let restored = device_escaped.view().unescape_in(kind, unified, &gpu).unwrap();
            assert!(restored.view().iter().eq(slices.iter()));
        }
        let upper: [u8; 256] = core::array::from_fn(|byte| (byte as u8).to_ascii_uppercase());
        let device_upper = device_view.lookup_in(&upper, unified, &gpu).unwrap();
        let host_upper = host_view.lookup_in(&upper, Global, &cpu).unwrap();
        assert!(device_upper.view().iter().eq(host_upper.view().iter()));
        let form = sz::NormalForm::NfkcCasefold;
        let device_folded = device_view.normalize_in(form, unified, &gpu).unwrap();
        let host_folded = host_view.normalize_in(form, Global, &cpu).unwrap();
        assert!(device_folded.view().iter().eq(host_folded.view().iter()));

        let mut device_rows = Vec::new_in(unified);
        device_rows.resize(words.len() * 8, 0xFFu8);
        let mut host_rows = vec![0xFFu8; words.len() * 8];
        assert_eq!(
            device_view.to_rows(&mut device_rows, 8, &gpu),
            host_view.to_rows(&mut host_rows, 8, &cpu)
        );
        assert_eq!(device_rows[..], host_rows[..]);

        let mut engine = sz::LevenshteinEngine::new(&["kitten"], sz::LevenshteinSymbol::Bytes, &gpu).unwrap();
        let mut distances = Vec::new_in(unified);
        distances.resize(words.len(), usize::MAX);
        gpu.scope(|scope| engine.distances(scope, device_view, None, &mut distances, words.len()))
            .unwrap();
        assert_eq!(distances[4], 0);
    }

    // endregion
}
