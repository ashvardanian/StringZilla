//! Set intersections over string sequences.

use super::*;
use core::ffi::c_void;

/// Intersects two sequences (inner join) using their default byte-slice views.
///
/// Both sequences must have an output buffer provided (for first and second positions)
/// whose length is at least the minimum of the two input lengths.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let set1 = ["banana", "apple", "cherry"];
/// let set2 = ["cherry", "orange", "pineapple", "banana"];
/// let mut positions1 = [0; 3]; // at least min(3, 4) == 3 elements.
/// let mut positions2 = [0; 3];
/// let n = sz::intersection(&set1, &set2, 0, &mut positions1, &mut positions2).expect("intersection failed");
/// assert!(n == 2); // "banana" and "cherry" are common.
/// ```
pub fn intersection<Element: AsRef<[u8]>>(
    data1: &[Element],
    data2: &[Element],
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
) -> Result<usize, Status> {
    let min_count = data1.len().min(data2.len());
    if positions1.len() < min_count || positions2.len() < min_count {
        return Err(Status::BadAlloc);
    }

    // Call the lower-level implementation with accurate counts for both sequences.
    let adapter1 = move |i: usize| -> &'static [u8] {
        // SAFETY: used only during the FFI call
        unsafe { core::mem::transmute::<&[u8], &'static [u8]>(data1[i].as_ref()) }
    };
    let adapter2 = move |j: usize| -> &'static [u8] {
        // SAFETY: used only during the FFI call
        unsafe { core::mem::transmute::<&[u8], &'static [u8]>(data2[j].as_ref()) }
    };
    _intersection_by_impl(
        adapter1,
        adapter2,
        seed,
        positions1,
        positions2,
        data1.len(),
        data2.len(),
    )
}

/// Intersects two sequences (inner join) using their elements corresponding byte-slice views.
/// The caller must provide a closure that maps an index to the byte slice representation of
/// the corresponding element in the first and second sequences.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// #[derive(Debug)]
/// struct Person { name: &'static str, age: u32 }
///
/// let people1 = [
///     Person { name: "Charlie", age: 20 },
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
/// ];
/// let people2 = [
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
///     Person { name: "Charlie", age: 20 },
/// ];
/// let mut positions1 = [0; 3]; // min(people1.len(), people2.len())
/// let mut positions2 = [0; 3]; // min(people1.len(), people2.len())
/// let n = sz::intersection_by(
///     |i| people1[i].name.as_bytes(),
///     |j| people2[j].name.as_bytes(),
///     0,
///     &mut positions1,
///     &mut positions2,
/// ).expect("intersection_by failed");
/// assert!(n == 3); // "Alice", "Bob", and "Charlie" are common.
/// ```
pub fn intersection_by<Mapper1, Mapper2, Key1, Key2>(
    mapper1: Mapper1,
    mapper2: Mapper2,
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
) -> Result<usize, Status>
where
    Mapper1: Fn(usize) -> Key1,
    Key1: AsRef<[u8]>,
    Mapper2: Fn(usize) -> Key2,
    Key2: AsRef<[u8]>,
{
    if positions1.len() != positions2.len() {
        return Err(Status::BadAlloc);
    }

    // Adapter closure: given an index, call the provided mapper and then transmute the
    // resulting slice to have a `'static` lifetime. This transmute is safe as long as
    // the FFI call is synchronous and the returned slices are only used during the call.
    let adapter1 = move |i: usize| -> &'static [u8] {
        let binding = mapper1(i);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };
    let adapter2 = move |i: usize| -> &'static [u8] {
        let binding = mapper2(i);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };

    _intersection_by_impl(
        adapter1,
        adapter2,
        seed,
        positions1,
        positions2,
        positions1.len(),
        positions2.len(),
    )
}

fn _intersection_by_impl<Adapter1, Adapter2>(
    adapter1: Adapter1,
    adapter2: Adapter2,
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
    count1: usize,
    count2: usize,
) -> Result<usize, Status>
where
    Adapter1: Fn(usize) -> &'static [u8],
    Adapter2: Fn(usize) -> &'static [u8],
{
    let wrapper1 = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<Adapter1>() },
        data: &adapter1 as *const Adapter1 as *const c_void,
    };
    let wrapper2 = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<Adapter2>() },
        data: &adapter2 as *const Adapter2 as *const c_void,
    };
    let seq1 = _SzSequence {
        handle: &wrapper1 as *const _ as *const c_void,
        count: count1,
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    let seq2 = _SzSequence {
        handle: &wrapper2 as *const _ as *const c_void,
        count: count2,
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    let mut inter_size: usize = 0;
    let status = unsafe {
        sz_sequence_intersect(
            &seq1,
            &seq2,
            core::ptr::null(),
            seed,
            &mut inter_size as *mut usize,
            positions1.as_mut_ptr(),
            positions2.as_mut_ptr(),
        )
    };
    if status == Status::Success {
        Ok(inter_size)
    } else {
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    #[cfg(feature = "std")]
    use std::collections::HashSet;

    use super::*;
    use crate::sz;

    #[test]
    #[cfg(feature = "std")]
    fn intersection_default() {
        // Two slices of string literals.
        let set1 = ["banana", "apple", "cherry"];
        let set2 = ["cherry", "orange", "pineapple", "banana"];
        // Output buffers: size must be at least min(set1.len(), set2.len()).
        let mut out1 = [0; 3];
        let mut out2 = [0; 3];

        let n = sz::intersection(&set1, &set2, 0, &mut out1, &mut out2).expect("intersection failed");
        assert!(n <= set1.len().min(set2.len()));

        // For simplicity, we will compare the intersection from the first set.
        // Our API returns indices (for set1 in out1).
        let common_from_api: HashSet<_> = out1[..n].iter().map(|&i| set1[i]).collect();

        // Compute the expected intersection using a `HashSet`.
        let expected: HashSet<_> = set1
            .iter()
            .cloned()
            .collect::<HashSet<_>>()
            .intersection(&set2.iter().cloned().collect())
            .cloned()
            .collect();

        assert_eq!(common_from_api, expected);
    }

    #[test]
    #[cfg(feature = "std")]
    fn intersection_by_custom() {
        // Define a custom type.
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Person {
            name: &'static str,
            age: u32, //? We won't use this field for intersection
        }

        let group1 = [
            Person { name: "Alice", age: 25 },
            Person { name: "Bob", age: 30 },
            Person {
                name: "Charlie",
                age: 35,
            },
        ];
        let group2 = [
            Person { name: "David", age: 40 },
            Person {
                name: "Charlie",
                age: 50,
            },
            Person { name: "Alice", age: 60 },
        ];
        let mut out1 = [0; 3];
        let mut out2 = [0; 3];

        let n = sz::intersection_by(
            |i: sz::SortedIdx| group1[i].name.as_bytes(),
            |j: sz::SortedIdx| group2[j].name.as_bytes(),
            0,
            &mut out1,
            &mut out2,
        )
        .expect("intersection_by failed");
        assert!(n <= group1.len().min(group2.len()));

        // Use the indices for `group1` to get common names.
        let common_from_api: HashSet<_> = out1[..n].iter().map(|&i| group1[i].name).collect();

        // Compute expected common names using a `HashSet`.
        let expected: HashSet<_> = group1
            .iter()
            .map(|p| p.name)
            .collect::<HashSet<_>>()
            .intersection(&group2.iter().map(|p| p.name).collect())
            .cloned()
            .collect();

        assert_eq!(common_from_api, expected);
    }

    #[test]
    #[should_panic(expected = "BadAlloc")]
    fn intersection_size_checks() {
        let mut indices = [0usize; 10];
        let mut indices2 = [0usize; 5];
        let data = vec![0x41u8; 12];

        sz::intersection_by(|_: usize| &data, |_: usize| &data, 1, &mut indices, &mut indices2).unwrap();
    }

    #[test]
    fn intersection_sequences_sharing_an_empty_string() {
        // Regression check for sequences that are each individually duplicate-free but happen
        // to share an empty string - a corner case that must keep working correctly.
        let set1 = ["", "p", "q"];
        let set2 = ["", "z"];
        let mut positions1 = [0usize; 2];
        let mut positions2 = [0usize; 2];
        let matched = sz::intersection(&set1, &set2, 0, &mut positions1, &mut positions2).expect("intersection failed");
        assert_eq!(matched, 1);
        assert_eq!(set1[positions1[0]], set2[positions2[0]]);
    }
}
