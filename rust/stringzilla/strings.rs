//! String collections and byte transforms: [`Tape`] with its escaping, lookup and normalization
//! conversions, lookup tables and random fills over one buffer, and in-place replacement.
//!
//! File: rust/stringzilla/strings.rs
//! Author: Ash Vardanian

use core::alloc::{Allocator, Layout};
use core::ffi::c_void;
use core::fmt;

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

/// The text formats [`Tape::escape`] and [`Tape::unescape`] convert strings to and from.
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

/// Why [`Tape::unescape`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnescapeError {
    /// The position of the first string holding a malformed sequence, like a lone surrogate in
    /// JSON or an unbalanced quote in a CSV field.
    Malformed(usize),
    /// Any other failure, like a block that could not grow.
    Failed(Status),
}

impl fmt::Display for UnescapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnescapeError::Malformed(index) => write!(f, "string {index} holds a malformed escape sequence"),
            UnescapeError::Failed(status) => fmt::Display::fmt(status, f),
        }
    }
}

impl core::error::Error for UnescapeError {}

/// A batch of strings in one tape: a payload block with every string back to back, and an offsets
/// block one entry longer than the batch, both from a [`UnifiedAllocator`], which engines read in
/// place.
///
/// It is C's `sz_strings_span_t`, with 32-bit offsets while the payload fits in 4 GiB and 64-bit
/// beyond, and both blocks are freed when it drops. Every conversion runs on the device of the
/// tape's stream and joins it before returning, so a tape is always readable from the host. A host
/// engine reads a tape made for any capabilities, and a GPU engine one its device reaches, refusing
/// any other with [`Status::DeviceMemoryMismatch`].
///
/// # Examples
///
/// ```
/// use stringzilla::sz::{Capabilities, EscapeKind, LevenshteinEngine, LevenshteinSymbol, Stream, Tape, UnifiedAllocator};
///
/// let stream = Stream::default(Capabilities::cpu_enabled());
/// let unified = UnifiedAllocator::new(&stream);
/// let candidates = Tape::copy(&["sitting", "kitten"], &unified, &stream)?;
/// let mut engine = LevenshteinEngine::new(&["kitten"], LevenshteinSymbol::Bytes, &stream)?;
/// let mut distances = [0usize; 2];
/// stream.scope(|scope| engine.distances(scope, &candidates, &mut distances, 2))?;
/// assert_eq!(distances, [3, 0]);
///
/// let names = Tape::copy(&["Ada", "Grace \"Amazing\" Hopper"], &unified, &stream)?;
/// let fields = names.escape(EscapeKind::Csv)?;
/// assert_eq!(fields.get(1), Some(&br#""Grace ""Amazing"" Hopper""#[..]));
/// # Ok::<(), stringzilla::sz::Status>(())
/// ```
pub struct Tape<'stream> {
    stream: &'stream Stream,
    span: sz_strings_span_t,
    allocator: sz_allocator_t,
}

impl<'stream> Tape<'stream> {
    /// Copies `texts` into a tape from `allocator`, gathering them on the host into blocks the
    /// device of `stream` reaches.
    ///
    /// An `allocator` bound to a different stream handle or capability group is refused with
    /// [`Status::DeviceMemoryMismatch`].
    pub fn copy<Text: AsRef<[u8]>>(
        texts: &[Text],
        allocator: &UnifiedAllocator,
        stream: &'stream Stream,
    ) -> Result<Self, Status> {
        if allocator.capabilities().vendor() != stream.capabilities().vendor()
            || allocator.stream.as_raw() != stream.as_raw()
        {
            return Err(Status::DeviceMemoryMismatch);
        }
        // Only the host calls an accessor, so it gathers the strings whichever device they are for.
        Self::written(stream, allocator.c_allocator()?, |allocator, tape| {
            with_strings(texts, |strings| unsafe {
                sz_strings_to_tape_best(
                    strings,
                    core::ptr::null(),
                    strings.count,
                    allocator,
                    tape,
                    Capabilities::CPUS.bits(),
                    core::ptr::null_mut(),
                )
            })
        })
    }

    /// Strings the tape holds.
    pub fn len(&self) -> usize {
        self.span.count
    }

    /// Whether the tape holds no strings.
    pub fn is_empty(&self) -> bool {
        self.span.count == 0
    }

    /// The bytes of string `index`, or `None` past the last one, like [`slice::get`].
    pub fn get(&self, index: usize) -> Option<&[u8]> {
        (index < self.span.count).then(|| self.string(index))
    }

    /// The strings in order, like [`slice::iter`].
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + '_ {
        (0..self.span.count).map(|index| self.string(index))
    }

    /// Escapes every string for `kind` into a new tape.
    pub fn escape(&self, kind: EscapeKind) -> Result<Self, Status> {
        let (strings, capabilities, stream) = self.dispatch();
        Self::written(self.stream, self.allocator, |allocator, tape| unsafe {
            sz_strings_escape_best(
                &strings,
                core::ptr::null(),
                strings.count,
                kind,
                allocator,
                tape,
                capabilities,
                stream,
            )
        })
    }

    /// Reverses [`Tape::escape`] for `kind` into a new tape, stopping at the first string holding
    /// a malformed sequence, whose position [`UnescapeError::Malformed`] reports.
    pub fn unescape(&self, kind: EscapeKind) -> Result<Self, UnescapeError> {
        let (strings, capabilities, stream) = self.dispatch();
        let mut malformed = 0;
        Self::written(self.stream, self.allocator, |allocator, tape| unsafe {
            sz_strings_unescape_best(
                &strings,
                core::ptr::null(),
                strings.count,
                kind,
                allocator,
                tape,
                &mut malformed,
                capabilities,
                stream,
            )
        })
        .map_err(|status| match status {
            Status::InvalidUtf8 => UnescapeError::Malformed(malformed),
            status => UnescapeError::Failed(status),
        })
    }

    /// Maps every byte of every string through `table` into a new tape, as [`lookup`] does for one
    /// buffer.
    pub fn lookup(&self, table: [u8; 256]) -> Result<Self, Status> {
        let (strings, capabilities, stream) = self.dispatch();
        Self::written(self.stream, self.allocator, |allocator, tape| unsafe {
            sz_strings_lookup_best(
                &strings,
                core::ptr::null(),
                strings.count,
                table.as_ptr(),
                allocator,
                tape,
                capabilities,
                stream,
            )
        })
    }

    /// Brings every string into the Unicode `form` into a new tape, as [`utf8_norm`] does for one
    /// string, copying the strings already in it as they are.
    pub fn normalize(&self, form: Utf8NormalForm) -> Result<Self, Status> {
        let (strings, capabilities, stream) = self.dispatch();
        Self::written(self.stream, self.allocator, |allocator, tape| unsafe {
            sz_strings_normalize_best(
                &strings,
                core::ptr::null(),
                strings.count,
                form,
                allocator,
                tape,
                capabilities,
                stream,
            )
        })
    }

    /// Writes every string into its own NUL-padded row of `width` bytes, back to back in `rows`,
    /// truncating longer ones, and returns the longest length, to retry with wider rows.
    ///
    /// `rows` holds at least `len() * width` bytes, or [`Status::UnexpectedDimensions`] is
    /// returned, and on a GPU it lies in memory the device reaches, like a `Vec` in a
    /// [`UnifiedAllocator`].
    pub fn to_rows(&self, rows: &mut [u8], width: usize) -> Result<usize, Status> {
        let rows_bytes = self.span.count.checked_mul(width).ok_or(Status::OverflowRisk)?;
        if rows.len() < rows_bytes {
            return Err(Status::UnexpectedDimensions);
        }
        let target = sz_strings_rows_t {
            data: rows.as_mut_ptr(),
            stride: width,
            width,
        };
        let (strings, capabilities, stream) = self.dispatch();
        // A device writes the longest length too, so it lands in memory that device reaches.
        let unified = UnifiedAllocator::new(self.stream);
        let layout = Layout::new::<usize>();
        let longest = unified.allocate(layout).map_err(|_| Status::BadAlloc)?.cast::<usize>();
        let status = unsafe {
            longest.write(0);
            sz_strings_to_rows_best(
                &strings,
                core::ptr::null(),
                strings.count,
                &target,
                longest.as_ptr(),
                capabilities,
                stream,
            )
        };
        let joined = self.stream.synchronize();
        let result = status.check().and(joined).map(|()| unsafe { longest.read() });
        unsafe { unified.deallocate(longest.cast(), layout) };
        result
    }

    /// Fills a zeroed span from `allocator` through `write`, joining `stream` before anything reads
    /// the tape, and reports the failure of `write` ahead of the join's.
    fn written(
        stream: &'stream Stream,
        allocator: sz_allocator_t,
        write: impl FnOnce(&mut sz_allocator_t, &mut sz_strings_span_t) -> sz_status_t,
    ) -> Result<Self, Status> {
        let mut tape = Tape {
            stream,
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
        let status = write(&mut tape.allocator, &mut tape.span);
        let joined = stream.synchronize();
        status.check().and(joined).map(|()| tape)
    }

    /// The view a conversion reads this tape through, with the capabilities and the stream it
    /// dispatches on.
    fn dispatch(&self) -> (sz_strings_view_t, sz_capability_t, sz_stream_t) {
        let strings = sz_strings_view_t {
            layout: self.span.layout,
            count: self.span.count,
            data: sz_strings_data_t { tape: self.span.tape },
        };
        (strings, self.stream.capabilities().bits(), self.stream.as_raw())
    }

    /// Offset `index` of the tape, 32-bit or 64-bit as its layout says.
    fn offset(&self, index: usize) -> usize {
        let offsets = self.span.tape.offsets;
        match self.span.layout {
            sz_strings_layout_t::U64Tape => unsafe { *offsets.cast::<u64>().add(index) as usize },
            _ => unsafe { *offsets.cast::<u32>().add(index) as usize },
        }
    }

    /// The bytes of string `index`, which must be below the count.
    fn string(&self, index: usize) -> &[u8] {
        let (start, end) = (self.offset(index), self.offset(index + 1));
        match end - start {
            0 => &[],
            length => unsafe { core::slice::from_raw_parts(self.span.tape.data.add(start), length) },
        }
    }
}

unsafe impl Strings for Tape<'_> {
    fn with_strings<Return>(
        &self,
        _stream: &Stream,
        call: impl FnOnce(&sz_strings_view_t) -> Return,
    ) -> Result<Return, Status> {
        Ok(call(&self.dispatch().0))
    }
}

impl Drop for Tape<'_> {
    fn drop(&mut self) {
        // C's `sz_strings_span_free`, which its header keeps inline: the offsets size both blocks.
        let span = self.span;
        let Some(free) = self.allocator.free.filter(|_| !span.tape.offsets.is_null()) else {
            return;
        };
        let width = match span.layout {
            sz_strings_layout_t::U64Tape => size_of::<u64>(),
            _ => size_of::<u32>(),
        };
        let (handle, stream) = (self.allocator.handle, self.stream.as_raw());
        unsafe {
            free(span.tape.data.cast(), self.offset(span.count), handle, stream);
            free(span.tape.offsets, (span.count + 1) * width, handle, stream);
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use crate::sz;

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

    /// A tape reads back what it copied, escapes to the RFC forms, and unescapes every kind back.
    #[test]
    fn tapes_escape_and_unescape() {
        let cpu = sz::Stream::default(sz::Capabilities::cpu_enabled());
        let unified = sz::UnifiedAllocator::new(&cpu);
        let names = [
            "Ada",
            "Grace \"Amazing\" Hopper",
            "Linus\nTorvalds",
            "",
            "caf\u{e9} \u{1f600}",
        ];
        let tape = sz::Tape::copy(&names, &unified, &cpu).unwrap();
        assert!(tape.iter().eq(names.iter().map(|name| name.as_bytes())));
        assert_eq!(tape.get(names.len()), None);
        let nothing = sz::Tape::copy(&[] as &[&str], &unified, &cpu).unwrap();
        assert!(nothing.is_empty() && nothing.escape(sz::EscapeKind::Json).unwrap().is_empty());

        let json = tape.escape(sz::EscapeKind::Json).unwrap();
        assert_eq!(json.get(1), Some(&br#"Grace \"Amazing\" Hopper"#[..]));
        assert_eq!(json.get(2), Some(&br"Linus\nTorvalds"[..]));
        let ascii = tape.escape(sz::EscapeKind::JsonAscii).unwrap();
        assert_eq!(ascii.get(4), Some(&b"caf\\u00e9 \\ud83d\\ude00"[..]));
        let csv = tape.escape(sz::EscapeKind::Csv).unwrap();
        assert_eq!(csv.get(0), Some(&b"Ada"[..]));
        assert_eq!(csv.get(1), Some(&br#""Grace ""Amazing"" Hopper""#[..]));

        let kinds = [
            sz::EscapeKind::Json,
            sz::EscapeKind::Hex,
            sz::EscapeKind::Html,
            sz::EscapeKind::Url,
            sz::EscapeKind::Console,
            sz::EscapeKind::Csv,
            sz::EscapeKind::JsonAscii,
        ];
        for kind in kinds {
            let restored = tape.escape(kind).unwrap().unescape(kind).unwrap();
            assert!(restored.iter().eq(tape.iter()), "{kind:?}");
        }
    }

    #[test]
    fn unescape_reports_the_first_malformed_string() {
        let cpu = sz::Stream::default(sz::Capabilities::cpu_enabled());
        let unified = sz::UnifiedAllocator::new(&cpu);
        let fields = sz::Tape::copy(&["plain", "\"closed\"", "\"open", "\"also open"], &unified, &cpu).unwrap();
        assert_eq!(
            fields.unescape(sz::EscapeKind::Csv).err(),
            Some(sz::UnescapeError::Malformed(2))
        );
    }

    /// Lookup, normalization and rows read a tape like the strings it was copied from.
    #[test]
    fn tapes_map_normalize_and_pad() {
        let cpu = sz::Stream::default(sz::Capabilities::cpu_enabled());
        let unified = sz::UnifiedAllocator::new(&cpu);
        let words = ["Stra\u{df}e", "\u{fb01}le", "Ab"];
        let tape = sz::Tape::copy(&words, &unified, &cpu).unwrap();

        let upper: [u8; 256] = core::array::from_fn(|byte| (byte as u8).to_ascii_uppercase());
        let shouted = tape.lookup(upper).unwrap();
        assert!(shouted
            .iter()
            .eq(["STRA\u{df}E".as_bytes(), "\u{fb01}LE".as_bytes(), b"AB"]));

        let folded = tape.normalize(sz::Utf8NormalForm::NfkcCasefold).unwrap();
        assert!(folded.iter().eq([&b"strasse"[..], b"file", b"ab"]));

        let mut rows = [0xFF; 12];
        assert_eq!(tape.to_rows(&mut rows, 4), Ok(7));
        assert_eq!(&rows, b"Stra\xEF\xAC\x81lAb\0\0");
        assert_eq!(tape.to_rows(&mut rows[..11], 4), Err(sz::Status::UnexpectedDimensions));
    }

    /// A tape on a Metal stream converts on the GPU into what the host writes, and a Metal engine
    /// reads it in place.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_tapes_convert_like_host_ones() {
        let Ok(capabilities) = sz::Capabilities::metal_enabled(0) else {
            return;
        };
        let gpu = sz::Stream::new(capabilities, 0).unwrap();
        let cpu = sz::Stream::default(sz::Capabilities::cpu_enabled());
        let (device_memory, host_memory) = (sz::UnifiedAllocator::new(&gpu), sz::UnifiedAllocator::new(&cpu));
        let words = ["Grace \"Amazing\" Hopper", "", "Stra\u{df}e", "a,b\r\n", "kitten"];
        let device = sz::Tape::copy(&words, &device_memory, &gpu).unwrap();
        let host = sz::Tape::copy(&words, &host_memory, &cpu).unwrap();

        for kind in [sz::EscapeKind::Json, sz::EscapeKind::Csv, sz::EscapeKind::Url] {
            let (device_escaped, host_escaped) = (device.escape(kind).unwrap(), host.escape(kind).unwrap());
            assert!(device_escaped.iter().eq(host_escaped.iter()), "{kind:?}");
            assert!(device_escaped
                .unescape(kind)
                .unwrap()
                .iter()
                .eq(words.iter().map(|word| word.as_bytes())));
        }
        let upper: [u8; 256] = core::array::from_fn(|byte| (byte as u8).to_ascii_uppercase());
        assert!(device
            .lookup(upper)
            .unwrap()
            .iter()
            .eq(host.lookup(upper).unwrap().iter()));
        let form = sz::Utf8NormalForm::NfkcCasefold;
        assert!(device
            .normalize(form)
            .unwrap()
            .iter()
            .eq(host.normalize(form).unwrap().iter()));

        let mut device_rows = Vec::new_in(device_memory);
        device_rows.resize(words.len() * 8, 0xFFu8);
        let mut host_rows = vec![0xFFu8; words.len() * 8];
        assert_eq!(device.to_rows(&mut device_rows, 8), host.to_rows(&mut host_rows, 8));
        assert_eq!(device_rows[..], host_rows[..]);

        let mut engine = sz::LevenshteinEngine::new(&["kitten"], sz::LevenshteinSymbol::Bytes, &gpu).unwrap();
        let mut distances = Vec::new_in(device_memory);
        distances.resize(words.len(), usize::MAX);
        gpu.scope(|scope| engine.distances(scope, &device, &mut distances, words.len()))
            .unwrap();
        assert_eq!(distances[4], 0);
    }
}
