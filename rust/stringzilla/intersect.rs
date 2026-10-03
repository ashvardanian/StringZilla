//! Set intersections over string sequences.
//!
//! File: rust/stringzilla/intersect.rs
//! Author: Ash Vardanian

use super::*;

/// Intersects two sequences, as an inner join, using their default byte-slice views.
///
/// Both sequences must have an output buffer provided, for the first and second positions,
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
    intersection_by(
        data1,
        |item| item.as_ref(),
        data2,
        |item| item.as_ref(),
        seed,
        positions1,
        positions2,
    )
}

/// Intersects two sequences, as an inner join, on a byte-slice key of each element.
///
/// The sequences may differ in length and element type, and each output buffer must hold at least
/// as many positions as the shorter sequence has elements, or [`Status::UnexpectedDimensions`] is
/// returned. The keys run inside the C call, where a panic cannot unwind and aborts the process.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// struct Person { name: &'static str, age: u32 }
///
/// let people = [
///     Person { name: "Charlie", age: 20 },
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
/// ];
/// let invited = ["Alice", "Bob", "Dave", "Eve"];
/// let mut positions1 = [0; 3]; // min(people.len(), invited.len())
/// let mut positions2 = [0; 3];
/// let n = sz::intersection_by(
///     &people,
///     |person| person.name.as_bytes(),
///     &invited,
///     |name| name.as_bytes(),
///     0,
///     &mut positions1,
///     &mut positions2,
/// ).expect("intersection_by failed");
/// assert!(n == 2); // "Alice" and "Bob" are common.
/// ```
pub fn intersection_by<Element1, Element2, Key1, Key2>(
    data1: &[Element1],
    key1: Key1,
    data2: &[Element2],
    key2: Key2,
    seed: u64,
    positions1: &mut [SortedIdx],
    positions2: &mut [SortedIdx],
) -> Result<usize, Status>
where
    Key1: Fn(&Element1) -> &[u8],
    Key2: Fn(&Element2) -> &[u8],
{
    let min_count = data1.len().min(data2.len());
    if positions1.len() < min_count || positions2.len() < min_count {
        return Err(Status::UnexpectedDimensions);
    }

    let mut intersection_count: usize = 0;
    with_sequence_by(data1, key1, |sequence1| {
        with_sequence_by(data2, key2, |sequence2| unsafe {
            sz_sequence_intersect_best(
                sequence1,
                sequence2,
                core::ptr::null(),
                seed,
                &mut intersection_count,
                positions1.as_mut_ptr(),
                positions2.as_mut_ptr(),
                Capabilities::CPUS.bits(),
                core::ptr::null_mut(),
            )
        })
    })
    .check()?;
    Ok(intersection_count)
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    #[cfg(feature = "std")]
    use std::collections::HashSet;

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
        // Our API returns indices, for set1 in out1.
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
        // A different length and element type on the second side, which is all the C call needs.
        let group2 = ["David", "Charlie", "Alice", "Eve", "Mallory"];
        let mut out1 = [0; 3];
        let mut out2 = [0; 3];

        let n = sz::intersection_by(
            &group1,
            |person| person.name.as_bytes(),
            &group2,
            |name| name.as_bytes(),
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
            .intersection(&group2.iter().cloned().collect())
            .cloned()
            .collect();

        assert_eq!(common_from_api, expected);
        for position in 0..n {
            assert_eq!(group1[out1[position]].name, group2[out2[position]]);
        }
    }

    #[test]
    fn intersection_size_checks() {
        let data = [vec![0x41u8; 12], vec![0x42u8; 12], vec![0x43u8; 12]];
        let mut indices = [0usize; 10];
        let mut too_few = [0usize; 2];

        assert_eq!(
            sz::intersection(&data, &data, 1, &mut indices, &mut too_few),
            Err(sz::Status::UnexpectedDimensions)
        );
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
