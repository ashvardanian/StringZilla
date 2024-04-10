#![cfg_attr(not(test), no_std)]

/// The `sz` module provides a collection of string searching and manipulation functionality,
/// designed for high efficiency and compatibility with no_std environments. This module offers
/// various utilities for byte string manipulation, including search, reverse search, and
/// edit-distance calculations, suitable for a wide range of applications from basic string
/// processing to complex text analysis tasks.

pub mod sz {

    use core::ffi::c_void;

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

        fn sz_edit_distance_utf8(
            haystack1: *const c_void,
            haystack1_length: usize,
            haystack2: *const c_void,
            haystack2_length: usize,
            bound: usize,
            allocator: *const c_void,
        ) -> usize;

        fn sz_hamming_distance(
            haystack1: *const c_void,
            haystack1_length: usize,
            haystack2: *const c_void,
            haystack2_length: usize,
            bound: usize,
        ) -> usize;

        fn sz_hamming_distance_utf8(
            haystack1: *const c_void,
            haystack1_length: usize,
            haystack2: *const c_void,
            haystack2_length: usize,
            bound: usize,
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

        // type RandomGeneratorT = fn(*mut c_void) -> u64;

        fn sz_generate(
            alphabet: *const c_void,
            alphabet_size: usize,
            text: *mut c_void,
            length: usize,
            generate: *const c_void,
            generator: *mut c_void,
        );
    }

    /// Locates the first matching substring within `haystack` that equals `needle`.
    /// This function is similar to the `memmem()` function in LibC, but, unlike `strstr()`,
    /// it requires the length of both haystack and needle to be known beforehand.
    ///
    /// # Arguments
    ///
    /// * `haystack`: The byte slice to search.
    /// * `needle`: The byte slice to find within the haystack.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` representing the starting index of the first occurrence of `needle`
    /// within `haystack` if found, otherwise `None`.
    pub fn find<H, N>(haystack: H, needle: N) -> Option<usize>
    where
        H: AsRef<[u8]>,
        N: AsRef<[u8]>,
    {
        let haystack_ref = haystack.as_ref();
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

    /// Locates the last matching substring within `haystack` that equals `needle`.
    /// This function is useful for finding the most recent or last occurrence of a pattern
    /// within a byte slice.
    ///
    /// # Arguments
    ///
    /// * `haystack`: The byte slice to search.
    /// * `needle`: The byte slice to find within the haystack.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` representing the starting index of the last occurrence of `needle`
    /// within `haystack` if found, otherwise `None`.
    pub fn rfind<H, N>(haystack: H, needle: N) -> Option<usize>
    where
        H: AsRef<[u8]>,
        N: AsRef<[u8]>,
    {
        let haystack_ref = haystack.as_ref();
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

    /// Finds the index of the first character in `haystack` that is also present in `needles`.
    /// This function is particularly useful for parsing and tokenization tasks where a set of
    /// delimiter characters is used.
    ///
    /// # Arguments
    ///
    /// * `haystack`: The byte slice to search.
    /// * `needles`: The set of bytes to search for within the haystack.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` representing the index of the first occurrence of any byte from
    /// `needles` within `haystack`, if found, otherwise `None`.
    pub fn find_char_from<H, N>(haystack: H, needles: N) -> Option<usize>
    where
        H: AsRef<[u8]>,
        N: AsRef<[u8]>,
    {
        let haystack_ref = haystack.as_ref();
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

    /// Finds the index of the last character in `haystack` that is also present in `needles`.
    /// This can be used to find the last occurrence of any character from a specified set,
    /// useful in parsing scenarios such as finding the last delimiter in a string.
    ///
    /// # Arguments
    ///
    /// * `haystack`: The byte slice to search.
    /// * `needles`: The set of bytes to search for within the haystack.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` representing the index of the last occurrence of any byte from
    /// `needles` within `haystack`, if found, otherwise `None`.
    pub fn rfind_char_from<H, N>(haystack: H, needles: N) -> Option<usize>
    where
        H: AsRef<[u8]>,
        N: AsRef<[u8]>,
    {
        let haystack_ref = haystack.as_ref();
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

    /// Finds the index of the first character in `haystack` that is not present in `needles`.
    /// This function is useful for skipping over a known set of characters and finding the
    /// first character that does not belong to that set.
    ///
    /// # Arguments
    ///
    /// * `haystack`: The byte slice to search.
    /// * `needles`: The set of bytes that should not be matched within the haystack.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` representing the index of the first occurrence of any byte not in
    /// `needles` within `haystack`, if found, otherwise `None`.
    pub fn find_char_not_from<H, N>(haystack: H, needles: N) -> Option<usize>
    where
        H: AsRef<[u8]>,
        N: AsRef<[u8]>,
    {
        let haystack_ref = haystack.as_ref();
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

    /// Finds the index of the last character in `haystack` that is not present in `needles`.
    /// Useful for text processing tasks such as trimming trailing characters that belong to
    /// a specified set.
    ///
    /// # Arguments
    ///
    /// * `haystack`: The byte slice to search.
    /// * `needles`: The set of bytes that should not be matched within the haystack.
    ///
    /// # Returns
    ///
    /// An `Option<usize>` representing the index of the last occurrence of any byte not in
    /// `needles` within `haystack`, if found, otherwise `None`.
    pub fn rfind_char_not_from<H, N>(haystack: H, needles: N) -> Option<usize>
    where
        H: AsRef<[u8]>,
        N: AsRef<[u8]>,
    {
        let haystack_ref = haystack.as_ref();
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

    /// Computes the Levenshtein edit distance between two strings, using the Wagner-Fisher
    /// algorithm. This measure is widely used in applications like spell-checking, DNA sequence
    /// analysis.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    /// * `bound`: The maximum distance to compute, allowing for early exit.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (insertions,
    /// deletions, or substitutions) required to change `first` into `second`.
    pub fn edit_distance_bounded<F, S>(first: F, second: S, bound: usize) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        let first_ref = first.as_ref();
        let second_ref = second.as_ref();
        let first_length = first_ref.len();
        let second_length = second_ref.len();
        let first_pointer = first_ref.as_ptr() as _;
        let second_pointer = second_ref.as_ptr() as _;
        unsafe {
            sz_edit_distance(
                first_pointer,
                first_length,
                second_pointer,
                second_length,
                // Upper bound on the distance, that allows us to exit early. If zero is
                // passed, the maximum possible distance will be equal to the length of
                // the longer input.
                bound,
                // Uses the default allocator
                core::ptr::null(),
            )
        }
    }

    /// Computes the Levenshtein edit distance between two UTF8 strings, using the Wagner-Fisher
    /// algorithm. This measure is widely used in applications like spell-checking.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    /// * `bound`: The maximum distance to compute, allowing for early exit.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (insertions,
    /// deletions, or substitutions) required to change `first` into `second`.
    pub fn edit_distance_utf8_bounded<F, S>(first: F, second: S, bound: usize) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        let first_ref = first.as_ref();
        let second_ref = second.as_ref();
        let first_length = first_ref.len();
        let second_length = second_ref.len();
        let first_pointer = first_ref.as_ptr() as _;
        let second_pointer = second_ref.as_ptr() as _;
        unsafe {
            sz_edit_distance_utf8(
                first_pointer,
                first_length,
                second_pointer,
                second_length,
                // Upper bound on the distance, that allows us to exit early. If zero is
                // passed, the maximum possible distance will be equal to the length of
                // the longer input.
                bound,
                // Uses the default allocator
                core::ptr::null(),
            )
        }
    }

    /// Computes the Levenshtein edit distance between two strings, using the Wagner-Fisher
    /// algorithm. This measure is widely used in applications like spell-checking, DNA sequence
    /// analysis.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (insertions,
    /// deletions, or substitutions) required to change `first` into `second`.
    pub fn edit_distance<F, S>(first: F, second: S) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        edit_distance_bounded(first, second, 0)
    }

    /// Computes the Levenshtein edit distance between two UTF8 strings, using the Wagner-Fisher
    /// algorithm. This measure is widely used in applications like spell-checking.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (insertions,
    /// deletions, or substitutions) required to change `first` into `second`.
    pub fn edit_distance_utf8<F, S>(first: F, second: S) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        edit_distance_utf8_bounded(first, second, 0)
    }

    /// Computes the Hamming edit distance between two strings, counting the number of substituted characters.
    /// Difference in length is added to the result as well.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    /// * `bound`: The maximum distance to compute, allowing for early exit.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (substitutions) required to
    /// change `first` into `second`.
    pub fn hamming_distance_bounded<F, S>(first: F, second: S, bound: usize) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        let first_ref = first.as_ref();
        let second_ref = second.as_ref();
        let first_length = first_ref.len();
        let second_length = second_ref.len();
        let first_pointer = first_ref.as_ptr() as _;
        let second_pointer = second_ref.as_ptr() as _;
        unsafe {
            sz_hamming_distance(
                first_pointer,
                first_length,
                second_pointer,
                second_length,
                // Upper bound on the distance, that allows us to exit early. If zero is
                // passed, the maximum possible distance will be equal to the length of
                // the longer input.
                bound,
            )
        }
    }

    /// Computes the Hamming edit distance between two UTF8 strings, counting the number of substituted
    /// variable-length characters. Difference in length is added to the result as well.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    /// * `bound`: The maximum distance to compute, allowing for early exit.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (substitutions) required to
    /// change `first` into `second`.
    pub fn hamming_distance_utf8_bounded<F, S>(first: F, second: S, bound: usize) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        let first_ref = first.as_ref();
        let second_ref = second.as_ref();
        let first_length = first_ref.len();
        let second_length = second_ref.len();
        let first_pointer = first_ref.as_ptr() as _;
        let second_pointer = second_ref.as_ptr() as _;
        unsafe {
            sz_hamming_distance_utf8(
                first_pointer,
                first_length,
                second_pointer,
                second_length,
                // Upper bound on the distance, that allows us to exit early. If zero is
                // passed, the maximum possible distance will be equal to the length of
                // the longer input.
                bound,
            )
        }
    }

    /// Computes the Hamming edit distance between two strings, counting the number of substituted characters.
    /// Difference in length is added to the result as well.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (substitutions) required to
    /// change `first` into `second`.
    pub fn hamming_distance<F, S>(first: F, second: S) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        hamming_distance_bounded(first, second, 0)
    }

    /// Computes the Hamming edit distance between two UTF8 strings, counting the number of substituted
    /// variable-length characters. Difference in length is added to the result as well.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice.
    /// * `second`: The second byte slice.
    ///
    /// # Returns
    ///
    /// A `usize` representing the minimum number of single-character edits (substitutions) required to
    /// change `first` into `second`.
    pub fn hamming_distance_utf8<F, S>(first: F, second: S) -> usize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        hamming_distance_utf8_bounded(first, second, 0)
    }

    /// Computes the Needleman-Wunsch alignment score for two strings. This function is
    /// particularly used in bioinformatics for sequence alignment but is also applicable in
    /// other domains requiring detailed comparison between two strings, including gap and
    /// substitution penalties.
    ///
    /// # Arguments
    ///
    /// * `first`: The first byte slice to align.
    /// * `second`: The second byte slice to align.
    /// * `matrix`: The substitution matrix used for scoring.
    /// * `gap`: The penalty for each gap introduced during alignment.
    ///
    /// # Returns
    ///
    /// An `isize` representing the total alignment score, where higher scores indicate better
    /// alignment between the two strings, considering the specified gap penalties and
    /// substitution matrix.
    pub fn alignment_score<F, S>(first: F, second: S, matrix: [[i8; 256]; 256], gap: i8) -> isize
    where
        F: AsRef<[u8]>,
        S: AsRef<[u8]>,
    {
        let first_ref = first.as_ref();
        let second_ref = second.as_ref();
        let first_length = first_ref.len();
        let second_length = second_ref.len();
        let first_pointer = first_ref.as_ptr() as _;
        let second_pointer = second_ref.as_ptr() as _;
        unsafe {
            sz_alignment_score(
                first_pointer,
                first_length,
                second_pointer,
                second_length,
                matrix.as_ptr() as _,
                gap,
                core::ptr::null(),
            )
        }
    }

    /// Generates a default substitution matrix for use with the Needleman-Wunsch
    /// alignment algorithm. This matrix is initialized such that diagonal entries
    /// (representing matching characters) are zero, and off-diagonal entries
    /// (representing mismatches) are -1. This setup effectively produces distances
    /// equal to the negative Levenshtein edit distance, suitable for basic sequence
    /// alignment tasks where all mismatches are penalized equally and there are no
    /// rewards for matches.
    ///
    /// # Returns
    ///
    /// A 256x256 array of `i8`, where each element represents the substitution cost
    /// between two characters (byte values). Matching characters are assigned a cost
    /// of 0, and non-matching characters are assigned a cost of -1.
    pub fn unary_substitution_costs() -> [[i8; 256]; 256] {
        let mut result = [[0; 256]; 256];

        for i in 0..256 {
            for j in 0..256 {
                result[i][j] = if i == j { 0 } else { -1 };
            }
        }

        result
    }

    /// Randomizes the contents of a given byte slice `text` using characters from
    /// a specified `alphabet`. This function mutates `text` in place, replacing each
    /// byte with a random one from `alphabet`. It is designed for situations where
    /// you need to generate random strings or data sequences based on a specific set
    /// of characters, such as generating random DNA sequences or testing inputs.
    ///
    /// # Type Parameters
    ///
    /// * `T`: The type of the text to be randomized. Must be mutable and convertible to a byte slice.
    /// * `A`: The type of the alphabet. Must be convertible to a byte slice.
    ///
    /// # Arguments
    ///
    /// * `text`: A mutable reference to the data to randomize. This data will be mutated in place.
    /// * `alphabet`: A reference to the byte slice representing the alphabet to use for randomization.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz;
    /// let mut my_text = vec![0; 10]; // A buffer to randomize
    /// let alphabet = b"ACTG"; // Using a DNA alphabet
    /// sz::randomize(&mut my_text, &alphabet);
    /// ```
    ///
    /// After than,  `my_text` is filled with random 'A', 'C', 'T', or 'G' values.
    pub fn randomize<T, A>(text: &mut T, alphabet: &A)
    where
        T: AsMut<[u8]> + ?Sized, // Allows for mutable references to dynamically sized types.
        A: AsRef<[u8]> + ?Sized, // Allows for references to dynamically sized types.
    {
        let text_slice = text.as_mut();
        let alphabet_slice = alphabet.as_ref();
        unsafe {
            sz_generate(
                alphabet_slice.as_ptr() as *const c_void,
                alphabet_slice.len(),
                text_slice.as_mut_ptr() as *mut c_void,
                text_slice.len(),
                core::ptr::null(),
                core::ptr::null_mut(),
            );
        }
    }
}

/// Provides extensions for string searching and manipulation functionalities
/// on types that can reference byte slices ([u8]). This trait extends the capability
/// of any type implementing `AsRef<[u8]>`, allowing easy integration of SIMD-accelerated
/// string processing functions.
///
/// # Examples
///
/// Basic usage on a `Vec<u8>`:
///
/// ```
/// use stringzilla::StringZilla;
///
/// let haystack: &[u8] = &[b'a', b'b', b'c', b'd', b'e'];
/// let needle: &[u8] = &[b'c', b'd'];
///
/// assert_eq!(haystack.sz_find(needle.as_ref()), Some(2));
/// ```
///
/// Searching in a string slice:
///
/// ```
/// use stringzilla::StringZilla;
///
/// let haystack = "abcdef";
/// let needle = "cd";
///
/// assert_eq!(haystack.sz_find(needle.as_bytes()), Some(2));
/// ```
pub trait StringZilla<N>
where
    N: AsRef<[u8]>,
{
    /// Searches for the first occurrence of `needle` in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find("world".as_bytes()), Some(7));
    /// ```
    fn sz_find(&self, needle: N) -> Option<usize>;

    /// Searches for the last occurrence of `needle` in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let haystack = "Hello, world, world!";
    /// assert_eq!(haystack.sz_rfind("world".as_bytes()), Some(14));
    /// ```
    fn sz_rfind(&self, needle: N) -> Option<usize>;

    /// Finds the index of the first character in `self` that is also present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find_char_from("aeiou".as_bytes()), Some(1));
    /// ```
    fn sz_find_char_from(&self, needles: N) -> Option<usize>;

    /// Finds the index of the last character in `self` that is also present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_rfind_char_from("aeiou".as_bytes()), Some(8));
    /// ```
    fn sz_rfind_char_from(&self, needles: N) -> Option<usize>;

    /// Finds the index of the first character in `self` that is not present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find_char_not_from("aeiou".as_bytes()), Some(0));
    /// ```
    fn sz_find_char_not_from(&self, needles: N) -> Option<usize>;

    /// Finds the index of the last character in `self` that is not present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_rfind_char_not_from("aeiou".as_bytes()), Some(12));
    /// ```
    fn sz_rfind_char_not_from(&self, needles: N) -> Option<usize>;

    /// Computes the Levenshtein edit distance between `self` and `other`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::StringZilla;
    ///
    /// let first = "kitten";
    /// let second = "sitting";
    /// assert_eq!(first.sz_edit_distance(second.as_bytes()), 3);
    /// ```
    fn sz_edit_distance(&self, other: N) -> usize;

    /// Computes the alignment score between `self` and `other` using the specified
    /// substitution matrix and gap penalty.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::{sz, StringZilla};
    ///
    /// let first = "kitten";
    /// let second = "sitting";
    /// let matrix = sz::unary_substitution_costs();
    /// let gap_penalty = -1;
    /// assert_eq!(first.sz_alignment_score(second.as_bytes(), matrix, gap_penalty), -3);
    /// ```
    fn sz_alignment_score(&self, other: N, matrix: [[i8; 256]; 256], gap: i8) -> isize;
}

impl<T, N> StringZilla<N> for T
where
    T: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    fn sz_find(&self, needle: N) -> Option<usize> {
        sz::find(self, needle)
    }

    fn sz_rfind(&self, needle: N) -> Option<usize> {
        sz::rfind(self, needle)
    }

    fn sz_find_char_from(&self, needles: N) -> Option<usize> {
        sz::find_char_from(self, needles)
    }

    fn sz_rfind_char_from(&self, needles: N) -> Option<usize> {
        sz::rfind_char_from(self, needles)
    }

    fn sz_find_char_not_from(&self, needles: N) -> Option<usize> {
        sz::find_char_not_from(self, needles)
    }

    fn sz_rfind_char_not_from(&self, needles: N) -> Option<usize> {
        sz::rfind_char_not_from(self, needles)
    }

    fn sz_edit_distance(&self, other: N) -> usize {
        sz::edit_distance(self, other)
    }

    fn sz_alignment_score(&self, other: N, matrix: [[i8; 256]; 256], gap: i8) -> isize {
        sz::alignment_score(self, other, matrix, gap)
    }
}

/// Provides a tool for mutating a byte slice by filling it with random data from a specified alphabet.
/// This trait is especially useful for types that need to be mutable and can reference or be converted to byte slices.
///
/// # Examples
///
/// Filling a mutable byte buffer with random ASCII letters:
///
/// ```
/// use stringzilla::MutableStringZilla;
///
/// let mut buffer = vec![0u8; 10]; // A buffer to randomize
/// let alphabet = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"; // Alphabet to use
/// buffer.sz_randomize(alphabet);
///
/// println!("Random buffer: {:?}", buffer);
/// // The buffer will now contain random ASCII letters.
/// ```
pub trait MutableStringZilla<A>
where
    A: AsRef<[u8]>,
{
    /// Fills the implementing byte slice with random bytes from the specified `alphabet`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::MutableStringZilla;
    ///
    /// let mut text = vec![0; 1000]; // A buffer to randomize
    /// let alphabet = b"AGTC"; // Using a DNA alphabet
    /// text.sz_randomize(alphabet);
    ///
    /// // `text` is now filled with random 'A', 'G', 'T', or 'C' values.
    /// ```
    fn sz_randomize(&mut self, alphabet: A);
}

impl<T, A> MutableStringZilla<A> for T
where
    T: AsMut<[u8]>,
    A: AsRef<[u8]>,
{
    fn sz_randomize(&mut self, alphabet: A) {
        let self_mut = self.as_mut();
        let alphabet_ref = alphabet.as_ref();
        sz::randomize(self_mut, alphabet_ref);
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::sz;
    use crate::MutableStringZilla;
    use crate::StringZilla;

    #[test]
    fn hamming() {
        assert_eq!(sz::hamming_distance("hello", "hello"), 0);
        assert_eq!(sz::hamming_distance("hello", "hell"), 1);
        assert_eq!(sz::hamming_distance("abc", "adc"), 1);

        assert_eq!(sz::hamming_distance_bounded("abcdefgh", "ABCDEFGH", 2), 2);
        assert_eq!(sz::hamming_distance_utf8("αβγδ", "αγγδ"), 1);
    }

    #[test]
    fn levenshtein() {
        assert_eq!(sz::edit_distance("hello", "hell"), 1);
        assert_eq!(sz::edit_distance("hello", "hell"), 1);
        assert_eq!(sz::edit_distance("abc", ""), 3);
        assert_eq!(sz::edit_distance("abc", "ac"), 1);
        assert_eq!(sz::edit_distance("abc", "a_bc"), 1);
        assert_eq!(sz::edit_distance("abc", "adc"), 1);
        assert_eq!(sz::edit_distance("fitting", "kitty"), 4);
        assert_eq!(sz::edit_distance("smitten", "mitten"), 1);
        assert_eq!(sz::edit_distance("ggbuzgjux{}l", "gbuzgjux{}l"), 1);
        assert_eq!(sz::edit_distance("abcdefgABCDEFG", "ABCDEFGabcdefg"), 14);

        assert_eq!(sz::edit_distance_bounded("fitting", "kitty", 2), 2);
        assert_eq!(sz::edit_distance_utf8("façade", "facade"), 1);
    }

    #[test]
    fn needleman() {
        let costs_vector = sz::unary_substitution_costs();
        assert_eq!(
            sz::alignment_score("listen", "silent", costs_vector, -1),
            -4
        );
        assert_eq!(
            sz::alignment_score("abcdefgABCDEFG", "ABCDEFGabcdefg", costs_vector, -1),
            -14
        );
        assert_eq!(sz::alignment_score("hello", "hello", costs_vector, -1), 0);
        assert_eq!(sz::alignment_score("hello", "hell", costs_vector, -1), -1);
    }

    #[test]
    fn search() {
        let my_string: String = String::from("Hello, world!");
        let my_str: &str = my_string.as_str();
        let my_cow_str: Cow<'_, str> = Cow::from(&my_string);

        // Identical to `memchr::memmem::find` and `memchr::memmem::rfind` functions
        assert_eq!(sz::find("Hello, world!", "world"), Some(7));
        assert_eq!(sz::rfind("Hello, world!", "world"), Some(7));

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
    fn randomize() {
        let mut text: Vec<u8> = vec![0; 10]; // A buffer of ten zeros
        let alphabet: &[u8] = b"abcd"; // A byte slice alphabet
        text.sz_randomize(alphabet);

        // Iterate throught text and check that it only contains letters from the alphabet
        assert!(text
            .iter()
            .all(|&b| b == b'd' || b == b'c' || b == b'b' || b == b'a'));
    }
}
