//! Byte-level equality and ordering.

use super::*;
use core::cmp::Ordering;
use core::ffi::c_void;

/// Lexicographic (byte-order) comparison of two strings, SIMD-accelerated.
///
/// Mirrors `Ord` on `&[u8]` but uses StringZilla's vectorized `sz_order`.
///
/// # Examples
///
/// ```
/// use std::cmp::Ordering;
/// use stringzilla::stringzilla as sz;
///
/// assert_eq!(sz::order("apple", "banana"), Ordering::Less);
/// assert_eq!(sz::order("abc", "abc"), Ordering::Equal);
/// ```
pub fn order<First, Second>(first: First, second: Second) -> Ordering
where
    First: AsRef<[u8]>,
    Second: AsRef<[u8]>,
{
    let first_ref = first.as_ref();
    let second_ref = second.as_ref();
    let result = unsafe {
        sz_order(
            first_ref.as_ptr() as *const c_void,
            first_ref.len(),
            second_ref.as_ptr() as *const c_void,
            second_ref.len(),
        )
    };
    match result {
        x if x < 0 => Ordering::Less,
        0 => Ordering::Equal,
        _ => Ordering::Greater,
    }
}

/// Byte-level equality of two strings, SIMD-accelerated via `sz_equal`.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla as sz;
///
/// assert!(sz::equal("abc", "abc"));
/// assert!(!sz::equal("abc", "abd"));
/// ```
pub fn equal<First, Second>(first: First, second: Second) -> bool
where
    First: AsRef<[u8]>,
    Second: AsRef<[u8]>,
{
    let first_ref = first.as_ref();
    let second_ref = second.as_ref();
    // `sz_equal` assumes equal lengths; differing lengths can never be byte-equal.
    first_ref.len() == second_ref.len()
        && unsafe {
            sz_equal(
                first_ref.as_ptr() as *const c_void,
                second_ref.as_ptr() as *const c_void,
                first_ref.len(),
            ) != 0
        }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::string::String;
    use core::cmp::Ordering;

    use super::*;
    use crate::sz;

    #[test]
    fn equal_matches_bytes() {
        assert!(sz::equal("Hello, world!", "Hello, world!"));
        assert!(!sz::equal("Hello, world!", "Hello, World!"));
        assert!(!sz::equal("abc", "abcd"));
        assert!(sz::equal("", ""));
        assert!(!sz::equal("", "a"));

        // Long enough to reach the vectorized path, differing only in the final byte.
        let long: String = core::iter::repeat('z').take(1000).collect();
        let mut altered = long.clone();
        altered.pop();
        altered.push('y');
        assert!(sz::equal(&long, &long));
        assert!(sz::equal(long.as_bytes(), long.as_bytes()));
        assert!(!sz::equal(&long, &altered));
    }

    #[test]
    fn order_is_lexicographic() {
        assert_eq!(sz::order("apple", "banana"), Ordering::Less);
        assert_eq!(sz::order("banana", "apple"), Ordering::Greater);
        assert_eq!(sz::order("abc", "abc"), Ordering::Equal);
        assert_eq!(sz::order("", ""), Ordering::Equal);

        // A prefix orders before the longer string extending it.
        assert_eq!(sz::order("abc", "abcd"), Ordering::Less);
        assert_eq!(sz::order("abcd", "abc"), Ordering::Greater);
        assert_eq!(sz::order("", "a"), Ordering::Less);

        // Byte order wins over length: "b" outranks the longer "abcd".
        assert_eq!(sz::order("b", "abcd"), Ordering::Greater);
        assert_eq!(sz::order("abcd", "b"), Ordering::Less);
        assert_eq!(sz::order("Zebra", "apple"), Ordering::Less);
    }
}
