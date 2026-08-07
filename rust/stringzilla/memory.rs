//! Byte-level memory kernels — copies, moves, fills, and lookup transforms.
//!
//! Also home to the in-place substring replacement kernels built on those primitives.

use super::*;
use core::ffi::c_void;

/// Moves the contents of `source` into `target`, overwriting the existing contents of `target`.
/// This function is useful for scenarios where you need to replace the contents of a byte slice
/// with the contents of another byte slice.
#[inline(always)]
pub fn move_<Target, Source>(target: &mut Target, source: &Source)
where
    Target: AsMut<[u8]> + ?Sized,
    Source: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    assert!(
        target_slice.len() >= source_slice.len(),
        "target must be at least as long as source"
    );
    unsafe {
        sz_move(
            target_slice.as_mut_ptr() as *const c_void,
            source_slice.as_ptr() as *const c_void,
            source_slice.len(),
        );
    }
}

/// Fills the contents of `target` with the specified `value`. This function is useful for
/// scenarios where you need to set all bytes in a byte slice to a specific value, such as
/// zeroing out a buffer or initializing a buffer with a specific byte pattern.
#[inline(always)]
pub fn fill<Target>(target: &mut Target, value: u8)
where
    Target: AsMut<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    unsafe {
        sz_fill(target_slice.as_ptr() as *const c_void, target_slice.len(), value);
    }
}

/// Copies the contents of `source` into `target`, overwriting the existing contents of `target`.
/// This function is useful for scenarios where you need to replace the contents of a byte slice
/// with the contents of another byte slice.
#[inline(always)]
pub fn copy<Target, Source>(target: &mut Target, source: &Source)
where
    Target: AsMut<[u8]> + ?Sized,
    Source: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    assert!(
        target_slice.len() >= source_slice.len(),
        "target must be at least as long as source"
    );
    unsafe {
        sz_copy(
            target_slice.as_mut_ptr() as *mut c_void,
            source_slice.as_ptr() as *const c_void,
            source_slice.len(),
        );
    }
}

/// Performs a lookup transformation (LUT), mapping contents of a buffer into the same or other
/// memory region, taking a byte substitution value from the provided table.
///
/// # Arguments
///
/// * `target`: A mutable buffer to populate.
/// * `source`: An immutable buffer to map from.
/// * `table`: Lookup table of 256 substitution values.
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
/// sz::lookup(&mut target, &source, to_lower);
/// let result = String::from_utf8(target).expect("Invalid UTF-8 sequence");
/// assert_eq!(result, "hello world!");
/// ```
///
pub fn lookup<Target, Source>(target: &mut Target, source: &Source, table: [u8; 256])
where
    Target: AsMut<[u8]> + ?Sized,
    Source: AsRef<[u8]> + ?Sized,
{
    let target_slice = target.as_mut();
    let source_slice = source.as_ref();
    assert!(
        target_slice.len() >= source_slice.len(),
        "target must be at least as long as source"
    );
    unsafe {
        sz_lookup(
            target_slice.as_mut_ptr() as *mut c_void,
            source_slice.len(),
            source_slice.as_ptr() as *const c_void,
            table.as_ptr() as _,
        );
    }
}

/// Performs a lookup transformation (LUT), mapping contents of a buffer into the same or other
/// memory region, taking a byte substitution value from the provided table.
///
/// # Arguments
///
/// * `buffer`: A mutable buffer to update inplace.
/// * `table`: Lookup table of 256 substitution values.
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
    unsafe {
        sz_lookup(
            buffer_slice.as_mut_ptr() as *mut c_void,
            buffer_slice.len(),
            buffer_slice.as_ptr() as *const c_void,
            table.as_ptr() as _,
        );
    }
}

/// Randomizes the contents of a given byte slice `text` using characters from
/// a specified `alphabet`. This function mutates `text` in place, replacing each
/// byte with a random one from `alphabet`. It is designed for situations where
/// you need to generate random strings or data sequences based on a specific set
/// of characters, such as generating random DNA sequences or testing inputs.
///
/// # Arguments
///
/// * `buffer`: A mutable reference to the data to randomize. This data will be mutated in place.
/// * `nonce`: A 64-bit "number used once" (nonce) value to seed the random number generator.
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
        sz_fill_random(buffer_slice.as_ptr() as _, buffer_slice.len(), nonce);
    }
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
            copy(&mut buffer[pos..pos + needle_length], &replacement);
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
                let chunk = pos - read;
                unsafe {
                    sz_move(
                        buffer.as_mut_ptr().add(write) as *const c_void,
                        buffer.as_ptr().add(read) as *const c_void,
                        chunk,
                    );
                }
                write += chunk;
            }
            copy(&mut buffer[write..write + replacement.len()], replacement);
            write += replacement.len();
            read = pos + needle_length;
            replaced += 1;
        }

        if read < len {
            let chunk = len - read;
            unsafe {
                sz_move(
                    buffer.as_mut_ptr().add(write) as *const c_void,
                    buffer.as_ptr().add(read) as *const c_void,
                    chunk,
                );
            }
            write += len - read;
        }
        buffer.truncate(write);
        return Ok(replaced);
    }

    // Case 3: replacement is longer – collect match positions once, resize once, then rewrite from the back.
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
    if let Err(_) = buffer.try_reserve_exact(added) {
        return Err(Status::BadAlloc);
    }
    buffer.resize(new_len, 0);

    let mut read_end = original_len;
    let mut write_end = new_len;

    while let Some(pos) = find_prev(buffer.as_slice(), read_end) {
        let match_end = pos + needle_length;
        let tail_len = read_end - match_end;
        if tail_len > 0 {
            unsafe {
                sz_move(
                    buffer.as_mut_ptr().add(write_end - tail_len) as *const c_void,
                    buffer.as_ptr().add(match_end) as *const c_void,
                    tail_len,
                );
            }
        }
        write_end -= tail_len;
        write_end -= replacement.len();
        copy(&mut buffer[write_end..write_end + replacement.len()], replacement);
        read_end = pos;
    }

    debug_assert_eq!(write_end, read_end, "replace_all backfill mismatch");
    Ok(match_count)
}

/// Tries to replace all non-overlapping occurrences of `needle` inside `buffer` in place.
///
/// - equal-length replacements simply overwrite matches,
/// - shorter replacements compact forward without allocating,
/// - longer replacements count matches once, resize once, and rewrite from the back.
///
/// Returns the number of replacements performed.
#[cfg(feature = "std")]
pub fn try_replace_all(buffer: &mut Vec<u8>, needle: &[u8], replacement: &[u8]) -> Result<usize, Status> {
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

/// Tries to replace all non-overlapping bytes in `buffer` that belong to `byteset` with `replacement`.
///
/// Uses the same three-way strategy as [`try_replace_all`]. If the byteset is empty, the buffer is
/// left untouched. Returns the number of replacements performed.
#[cfg(feature = "std")]
pub fn try_replace_all_byteset(buffer: &mut Vec<u8>, byteset: Byteset, replacement: &[u8]) -> Result<usize, Status> {
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

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
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
    #[should_panic(expected = "target must be at least as long as source")]
    fn copy_size_checks() {
        let long: Vec<u8> = vec![0; 20];
        let mut less_long: Vec<u8> = vec![0; 10];

        sz::copy(&mut less_long, &long);
    }

    #[test]
    #[should_panic(expected = "target must be at least as long as source")]
    fn move_size_checks() {
        let long: Vec<u8> = vec![0; 20];
        let mut less_long: Vec<u8> = vec![0; 10];

        sz::move_(&mut less_long, &long);
    }

    #[test]
    #[should_panic(expected = "target must be at least as long as source")]
    fn lookup_size_checks() {
        let long: Vec<u8> = vec![0; 20];
        let mut less_long: Vec<u8> = vec![0; 10];

        let lut: [u8; 256] = (0..=255u8).collect::<Vec<_>>().try_into().unwrap();
        sz::lookup(&mut less_long, &long, lut);
    }
}
