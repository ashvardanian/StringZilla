//! Argument-sorting of string sequences.

use super::*;
use core::ffi::c_void;

/// Knobs for [`argsort`] and [`argsort_by`].
///
/// The default is a full, ascending, byte-lexicographic, **stable** sort (equal elements keep their
/// input order). Tweak the public fields directly or chain the builder methods:
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let descending = sz::ArgsortOptions::default().reversed();
/// let top_10_folded = sz::ArgsortOptions { uncased: true, top: Some(10), ..Default::default() };
/// # let _ = (descending, top_10_folded);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArgsortOptions {
    /// Sort in descending order; equal elements still keep their input order (stable).
    pub reverse: bool,
    /// Order under Unicode case-folding instead of raw bytes.
    pub uncased: bool,
    /// Only fully order the leading `Some(k)` elements (top-K / partial sort); `None` sorts everything.
    /// The remaining entries of `order` stay a valid - but arbitrary - permutation of the leftover indices.
    pub top: Option<usize>,
}

impl ArgsortOptions {
    /// Sort in descending order.
    pub const fn reversed(mut self) -> Self {
        self.reverse = true;
        self
    }
    /// Order under Unicode case-folding instead of raw bytes.
    pub const fn uncased(mut self) -> Self {
        self.uncased = true;
        self
    }
    /// Only fully order the leading `count` elements (top-K / partial sort).
    pub const fn top(mut self, count: usize) -> Self {
        self.top = Some(count);
        self
    }
}

/// Computes the permutation that sorts `data` by its byte-slice representations.
///
/// The caller supplies an output buffer `order` of length at least `data.len()`; on success the sorted
/// permutation indices are written into its first `data.len()` slots. See [`ArgsortOptions`] for
/// descending, uncased, and top-K variants.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// let fruits = ["banana", "apple", "cherry"];
/// let mut order = [0; 3];
/// sz::argsort(&fruits, &mut order, Default::default()).expect("sort failed");
/// assert_eq!(&order, &[1, 0, 2]); // "apple", "banana", "cherry"
///
/// // Descending, uncased:
/// let labels = ["beta", "Alpha", "BETA"];
/// let mut order = [0; 3];
/// sz::argsort(&labels, &mut order, sz::ArgsortOptions::default().reversed().uncased()).unwrap();
/// assert_eq!(labels[order[0]], "beta"); // "beta"/"BETA" (fold-equal) before "Alpha", stable on ties
/// ```
pub fn argsort<Element: AsRef<[u8]>>(
    data: &[Element],
    order: &mut [SortedIdx],
    options: ArgsortOptions,
) -> Result<(), Status> {
    if data.len() > order.len() {
        return Err(Status::BadAlloc);
    }
    argsort_by(|i| data[i].as_ref(), &mut order[..data.len()], options)
}

/// Computes the permutation that sorts items by a caller-provided byte-slice key.
/// The number of items is inferred from the length of the `order` slice.
///
/// # Example
///
/// ```rust
/// use stringzilla::stringzilla as sz;
///
/// struct Person { name: &'static str, age: u32 }
/// let people = [
///     Person { name: "Charlie", age: 20 },
///     Person { name: "Alice", age: 25 },
///     Person { name: "Bob", age: 30 },
/// ];
/// let mut order = [0; 3];
/// sz::argsort_by(|i| people[i].name.as_bytes(), &mut order, Default::default()).expect("sort failed");
/// assert_eq!(&order, &[1, 2, 0]); // "Alice", "Bob", "Charlie"
/// ```
pub fn argsort_by<Mapper, Key>(mapper: Mapper, order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>
where
    Mapper: Fn(usize) -> Key,
    Key: AsRef<[u8]>,
{
    // Adapter closure: given an index, call the provided mapper and then transmute the
    // resulting slice to have a `'static` lifetime. This transmute is safe as long as
    // the FFI call is synchronous and the returned slices are only used during the call.
    let adapter = move |i: usize| -> &'static [u8] {
        let binding = mapper(i);
        let slice = binding.as_ref();
        unsafe { core::mem::transmute(slice) }
    };

    _argsort_impl(adapter, order, options)
}

/// Helper that takes an adapter (with a concrete type) and performs the FFI call.
fn _argsort_impl<Adapter>(adapter: Adapter, order: &mut [SortedIdx], options: ArgsortOptions) -> Result<(), Status>
where
    Adapter: Fn(usize) -> &'static [u8],
{
    let wrapper = _PunnedSliceLookupView {
        get_slice: unsafe { _get_slice_fn::<Adapter>() },
        data: &adapter as *const Adapter as *const c_void,
    };
    let seq = _SzSequence {
        handle: &wrapper as *const _ as *const c_void,
        count: order.len(),
        get_start: Some(_slice_get_start_punned),
        get_length: Some(_slice_get_length_punned),
    };
    let top_count = options.top.unwrap_or(0);
    let reverse = options.reverse as i32;
    let status = unsafe {
        if options.uncased {
            sz_sequence_argsort_uncased(&seq, core::ptr::null(), order.as_mut_ptr(), top_count, reverse)
        } else {
            sz_sequence_argsort(&seq, core::ptr::null(), order.as_mut_ptr(), top_count, reverse)
        }
    };
    if status == Status::Success {
        Ok(())
    } else {
        Err(status)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz;

    #[test]
    fn argsort_default() {
        // Test with a slice of string literals.
        let fruits = ["banana", "apple", "cherry"];
        let mut order = [0; 3]; // output buffer must be at least fruits.len()
        sz::argsort(&fruits, &mut order, Default::default()).expect("argsort failed");

        // Reconstruct sorted order using the returned indices.
        let sorted_from_api: Vec<_> = order.iter().map(|&i| fruits[i]).collect();

        // Compute expected order using the standard sort.
        let mut expected = fruits.to_vec();
        expected.sort();

        assert_eq!(sorted_from_api, expected);
    }

    #[test]
    fn argsort_by_custom() {
        // Define a custom type.
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Person {
            name: &'static str,
            age: u32, //? We won't use this field for intersection
        }

        let people = [
            Person {
                name: "Charlie",
                age: 30,
            },
            Person { name: "Alice", age: 25 },
            Person { name: "Bob", age: 40 },
        ];
        let mut order = [0; 3];
        sz::argsort_by(|i: usize| people[i].name.as_bytes(), &mut order, Default::default())
            .expect("argsort_by failed");

        let sorted_from_api: Vec<_> = order.iter().map(|&i| people[i].name).collect();

        // Compute expected order using standard sorting on the names.
        let mut expected: Vec<_> = people.iter().map(|p| p.name).collect();
        expected.sort();

        assert_eq!(sorted_from_api, expected);
    }

    #[test]
    fn argsort_reverse_is_stable() {
        // Two equal "beta"s must keep their input order even when sorting descending.
        let labels = ["beta", "alpha", "beta", "gamma"];
        let mut order = [0; 4];
        sz::argsort(&labels, &mut order, sz::ArgsortOptions::default().reversed()).expect("argsort failed");
        let sorted: Vec<_> = order.iter().map(|&i| labels[i]).collect();
        assert_eq!(sorted, vec!["gamma", "beta", "beta", "alpha"]);
        // Stability: the first "beta" (index 0) precedes the second (index 2).
        let beta_positions: Vec<_> = order.iter().filter(|&&i| labels[i] == "beta").copied().collect();
        assert_eq!(beta_positions, vec![0, 2]);
    }

    #[test]
    fn argsort_top_k_prefix() {
        let words = ["delta", "alpha", "echo", "bravo", "charlie"];
        let mut order = [0; 5];
        sz::argsort(&words, &mut order, sz::ArgsortOptions::default().top(2)).expect("argsort failed");
        // Only the first two entries are guaranteed sorted (the two smallest).
        assert_eq!(words[order[0]], "alpha");
        assert_eq!(words[order[1]], "bravo");
        // `order` is still a full permutation.
        let mut seen = order.to_vec();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn argsort_uncased() {
        let labels = ["Banana", "apple", "BANANA", "Apple"];
        let mut order = [0; 4];
        sz::argsort(&labels, &mut order, sz::ArgsortOptions::default().uncased()).expect("argsort failed");
        let sorted: Vec<_> = order.iter().map(|&i| labels[i]).collect();
        // Fold-equal strings group together and stay in input order: "apple","Apple" then "Banana","BANANA".
        assert_eq!(sorted, vec!["apple", "Apple", "Banana", "BANANA"]);
    }
}
