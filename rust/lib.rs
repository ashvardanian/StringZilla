#![cfg_attr(not(test), no_std)]

use core::ffi::c_void;

type SzRandomGeneratorT = extern "C" fn(*mut c_void) -> u64;

// Import the functions from the StringZilla C library.
extern "C" {
    fn sz_find(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_rfind(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_find_char_from(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_rfind_char_from(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_find_char_not_from(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_rfind_char_not_from(
        haystack: *const c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *const c_void;

    fn sz_edit_distance(
        haystack1: *const c_void,
        haystack1_length: usize,
        haystack2: *const c_void,
        haystack2_length: usize,
        bound: usize,
        allocator: *const c_void,
    ) -> usize;

    fn sz_alignment_score(
        haystack1: *const c_void,
        haystack1_length: usize,
        haystack2: *const c_void,
        haystack2_length: usize,
        matrix: *const c_void,
        gap: i8,
        allocator: *const c_void,
    ) -> isize;

    fn sz_generate(
        alphabet: *const c_void,
        alphabet_size: usize,
        text: *mut c_void,
        length: usize,
        generate: SzRandomGeneratorT,
        generator: *mut c_void,
    );
}

/// The [StringZilla] trait provides a collection of string searching and manipulation functionalities.
pub trait StringZilla<N>
where
    N: AsRef<[u8]>,
{
    /// Locates first matching substring. Equivalent to `memmem(haystack, h_length, needle, n_length)` in LibC. Similar
    /// to `strstr(haystack, needle)` in LibC, but requires known length.
    fn sz_find(&self, needle: N) -> Option<usize>;

    /// Locates the last matching substring.
    fn sz_rfind(&self, needle: N) -> Option<usize>;

    /// Finds the first character in the haystack, that is present in the needle.
    fn sz_find_char_from(&self, needles: N) -> Option<usize>;

    /// Finds the last character in the haystack, that is present in the needle.
    fn sz_rfind_char_from(&self, needles: N) -> Option<usize>;

    /// Finds the first character in the haystack, that is __not__ present in the needle.
    fn sz_find_char_not_from(&self, needles: N) -> Option<usize>;

    /// Finds the last character in the haystack, that is __not__ present in the needle.
    fn sz_rfind_char_not_from(&self, needles: N) -> Option<usize>;

    /// Computes the Levenshtein edit-distance between two strings using the Wagner-Fisher algorithm.
    /// Similar to the Needleman-Wunsch alignment algorithm. Often used in fuzzy string matching.
    fn sz_edit_distance(&self, needle: N) -> usize;

    /// Computes Needleman–Wunsch alignment score for two strings. Often used in bioinformatics and cheminformatics.
    /// Similar to the Levenshtein edit-distance, parameterized for gap and substitution penalties.
    fn sz_alignment_score(&self, needle: N, matrix: [[i8; 256]; 256], gap: i8) -> isize;
}

trait MutableStringZilla<N>
where
    N: AsRef<[u8]>,
{
    /// Generates a random string for a given alphabet.
    /// Replaces the buffer with a random string of the same length.
    // Cannot be String, as it is not AsMut<[u8]>
    fn randomize(&mut self, alphabet: N, generate: SzRandomGeneratorT);
}

impl<T, N> MutableStringZilla<N> for T
where
    T: AsMut<[u8]>,
    N: AsRef<[u8]>,
{
    fn randomize(&mut self, alphabet: N, generate: SzRandomGeneratorT) {
        let text = self.as_mut();
        let text_len = text.len();
        let alphabet_slice = alphabet.as_ref(); // Convert N to &[u8];

        unsafe {
            sz_generate(
                alphabet_slice.as_ptr() as *const c_void,
                alphabet_slice.len(),
                text.as_mut_ptr() as *mut c_void,
                text_len,
                generate,              // Directly use the function pointer
                core::ptr::null_mut(), // No need for a generator context
            );
        }
    }
}

impl<T, N> StringZilla<N> for T
where
    T: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    fn sz_find(&self, needle: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needle_ref = needle.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let haystack_length = haystack_ref.len();
        let needle_pointer = needle_ref.as_ptr() as _;
        let needle_length = needle_ref.len();
        let result = unsafe {
            sz_find(
                haystack_pointer,
                haystack_length,
                needle_pointer,
                needle_length,
            )
        };

        if result.is_null() {
            None
        } else {
            Some(unsafe { result.offset_from(haystack_pointer) } as usize)
        }
    }

    fn sz_rfind(&self, needle: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needle_ref = needle.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let haystack_length = haystack_ref.len();
        let needle_pointer = needle_ref.as_ptr() as _;
        let needle_length = needle_ref.len();
        let result = unsafe {
            sz_rfind(
                haystack_pointer,
                haystack_length,
                needle_pointer,
                needle_length,
            )
        };

        if result.is_null() {
            None
        } else {
            Some(unsafe { result.offset_from(haystack_pointer) } as usize)
        }
    }

    fn sz_find_char_from(&self, needles: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needles_ref = needles.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as _;
        let needles_length = needles_ref.len();
        let result = unsafe {
            sz_find_char_from(
                haystack_pointer,
                haystack_length,
                needles_pointer,
                needles_length,
            )
        };
        if result.is_null() {
            None
        } else {
            Some(unsafe { result.offset_from(haystack_pointer) } as usize)
        }
    }

    fn sz_rfind_char_from(&self, needles: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needles_ref = needles.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as _;
        let needles_length = needles_ref.len();
        let result = unsafe {
            sz_rfind_char_from(
                haystack_pointer,
                haystack_length,
                needles_pointer,
                needles_length,
            )
        };
        if result.is_null() {
            None
        } else {
            Some(unsafe { result.offset_from(haystack_pointer) } as usize)
        }
    }

    fn sz_find_char_not_from(&self, needles: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needles_ref = needles.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as _;
        let needles_length = needles_ref.len();
        let result = unsafe {
            sz_find_char_not_from(
                haystack_pointer,
                haystack_length,
                needles_pointer,
                needles_length,
            )
        };
        if result.is_null() {
            None
        } else {
            Some(unsafe { result.offset_from(haystack_pointer) } as usize)
        }
    }

    fn sz_rfind_char_not_from(&self, needles: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needles_ref = needles.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as _;
        let needles_length = needles_ref.len();
        let result = unsafe {
            sz_rfind_char_not_from(
                haystack_pointer,
                haystack_length,
                needles_pointer,
                needles_length,
            )
        };
        if result.is_null() {
            None
        } else {
            Some(unsafe { result.offset_from(haystack_pointer) } as usize)
        }
    }

    fn sz_edit_distance(&self, needle: N) -> usize {
        let haystack_ref = self.as_ref();
        let needle_ref = needle.as_ref();
        let haystack_length = haystack_ref.len();
        let needle_length = needle_ref.len();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let needle_pointer = needle_ref.as_ptr() as _;
        unsafe {
            sz_edit_distance(
                haystack_pointer,
                haystack_length,
                needle_pointer,
                needle_length,
                // Upper bound on the distance, that allows us to exit early. If zero is
                // passed, the maximum possible distance will be equal to the length of
                // the longer input.
                0,
                // Uses the default allocator
                core::ptr::null(),
            )
        }
    }

    fn sz_alignment_score(&self, needle: N, matrix: [[i8; 256]; 256], gap: i8) -> isize {
        let haystack_ref = self.as_ref();
        let needle_ref = needle.as_ref();
        let haystack_length = haystack_ref.len();
        let needle_length = needle_ref.len();
        let haystack_pointer = haystack_ref.as_ptr() as _;
        let needle_pointer = needle_ref.as_ptr() as _;
        unsafe {
            sz_alignment_score(
                haystack_pointer,
                haystack_length,
                needle_pointer,
                needle_length,
                matrix.as_ptr() as _,
                gap,
                core::ptr::null(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;
    use std::borrow::Cow;

    use crate::{MutableStringZilla, StringZilla, SzRandomGeneratorT};

    fn unary_substitution_costs() -> [[i8; 256]; 256] {
        let mut result = [[0; 256]; 256];

        for i in 0..256 {
            for j in 0..256 {
                result[i][j] = if i == j { 0 } else { -1 };
            }
        }

        result
    }

    // Define a simple deterministic generator for testing purposes.
    extern "C" fn test_generator(_: *mut c_void) -> u64 {
        4 // Always returns 4 for predictability in tests
    }

    #[test]
    fn levenshtein() {
        assert_eq!("hello".sz_edit_distance("hell"), 1);
        assert_eq!("hello".sz_edit_distance("hell"), 1);
        assert_eq!("abc".sz_edit_distance(""), 3);
        assert_eq!("abc".sz_edit_distance("ac"), 1);
        assert_eq!("abc".sz_edit_distance("a_bc"), 1);
        assert_eq!("abc".sz_edit_distance("adc"), 1);
        assert_eq!("ggbuzgjux{}l".sz_edit_distance("gbuzgjux{}l"), 1);
        assert_eq!("abcdefgABCDEFG".sz_edit_distance("ABCDEFGabcdefg"), 14);
        assert_eq!("fitting".sz_edit_distance("kitty"), 4);
        assert_eq!("smitten".sz_edit_distance("mitten"), 1);
    }

    #[test]
    fn needleman() {
        let costs_vector = unary_substitution_costs();
        assert_eq!("listen".sz_alignment_score("silent", costs_vector, -1), -4);
        assert_eq!(
            "abcdefgABCDEFG".sz_alignment_score("ABCDEFGabcdefg", costs_vector, -1),
            -14
        );
        assert_eq!("hello".sz_alignment_score("hello", costs_vector, -1), 0);
        assert_eq!("hello".sz_alignment_score("hell", costs_vector, -1), -1);
    }

    #[test]
    fn basics() {
        let my_string: String = String::from("Hello, world!");
        let my_str: &str = my_string.as_str();
        let my_cow_str: Cow<'_, str> = Cow::from(&my_string);

        // Use the generic function with a String
        assert_eq!(my_string.sz_find("world"), Some(7));
        assert_eq!(my_string.sz_rfind("world"), Some(7));
        assert_eq!(my_string.sz_find_char_from("world"), Some(2));
        assert_eq!(my_string.sz_rfind_char_from("world"), Some(11));
        assert_eq!(my_string.sz_find_char_not_from("world"), Some(0));
        assert_eq!(my_string.sz_rfind_char_not_from("world"), Some(12));

        // Use the generic function with a &str
        assert_eq!(my_str.sz_find("world"), Some(7));
        assert_eq!(my_str.sz_find("world"), Some(7));
        assert_eq!(my_str.sz_find_char_from("world"), Some(2));
        assert_eq!(my_str.sz_rfind_char_from("world"), Some(11));
        assert_eq!(my_str.sz_find_char_not_from("world"), Some(0));
        assert_eq!(my_str.sz_rfind_char_not_from("world"), Some(12));

        // Use the generic function with a Cow<'_, str>
        assert_eq!(my_cow_str.as_ref().sz_find("world"), Some(7));
        assert_eq!(my_cow_str.as_ref().sz_find("world"), Some(7));
        assert_eq!(my_cow_str.as_ref().sz_find_char_from("world"), Some(2));
        assert_eq!(my_cow_str.as_ref().sz_rfind_char_from("world"), Some(11));
        assert_eq!(my_cow_str.as_ref().sz_find_char_not_from("world"), Some(0));
        assert_eq!(
            my_cow_str.as_ref().sz_rfind_char_not_from("world"),
            Some(12)
        );
    }

    #[test]
    fn test_randomize_with_byte_slice() {
        let mut my_bytes: Vec<u8> = vec![0; 10]; // A buffer of ten zeros
        let alphabet: &[u8] = b"abcd"; // A byte slice alphabet
        my_bytes.randomize(alphabet, test_generator);

        // Assert that all bytes in `my_bytes` are now 'd' (ASCII 100), based on the test_generator
        assert!(my_bytes.iter().all(|&b| b == b'd'));
    }

    #[test]
    fn test_randomize_with_vec() {
        let mut my_bytes: Vec<u8> = vec![0; 10];
        let alphabet = vec![b'a', b'b', b'c', b'd']; // A Vec<u8> alphabet
        my_bytes.randomize(&alphabet, test_generator);

        // Assert similar to the previous test
        assert!(my_bytes.iter().all(|&b| b == b'd'));
    }

    #[test]
    fn test_randomize_with_string() {
        let mut my_bytes: Vec<u8> = vec![0; 10];
        let alphabet = "abcd".to_string(); // A String alphabet
        my_bytes.randomize(&alphabet, test_generator);

        // Assert similar to the previous test
        assert!(my_bytes.iter().all(|&b| b == b'd'));
    }
}
