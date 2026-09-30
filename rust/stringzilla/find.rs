//! Substring and byteset search, partitioning, counting, and splitting.
//!
//! Also home to the binary-operation trait shared by the match and split iterators.
//!
//! File: rust/stringzilla/find.rs
//! Author: Ash Vardanian

use super::*;
use core::ffi::c_void;
use core::marker::PhantomData;

/// Locates the first matching substring within `haystack` that equals `needle`.
/// This function is similar to the `memmem()` function in LibC, but, unlike `strstr()`,
/// it requires the length of both haystack and needle to be known beforehand.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needle`: The byte slice to find within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the starting index of the first occurrence of `needle`
/// within `haystack` if found, otherwise `None`.
///
/// # Empty needle
///
/// The C core returns the start of `haystack` for an empty needle, like `strstr`, so
/// `find(haystack, b"")` is always `Some(0)`, matching `"abc".find("") == Some(0)`. This holds even
/// for an empty `haystack`.
pub fn find<Haystack, Needle>(haystack: Haystack, needle: Needle) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let needle_ref = needle.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();
    let needle_pointer = needle_ref.as_ptr() as _;
    let needle_length = needle_ref.len();
    let mut result = core::ptr::null();
    unsafe {
        sz_find_best(
            haystack_pointer,
            haystack_length,
            needle_pointer,
            needle_length,
            &mut result,
            enabled_cpu_capabilities_mask(),
            core::ptr::null_mut(),
        )
    }
    .infallible();

    if result.is_null() {
        None
    } else {
        Some(match_offset(result, haystack_pointer))
    }
}

/// Locates the last matching substring within `haystack` that equals `needle`.
/// This function is useful for finding the most recent or last occurrence of a pattern
/// within a byte slice.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needle`: The byte slice to find within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the starting index of the last occurrence of `needle`
/// within `haystack` if found, otherwise `None`.
///
/// # Empty needle
///
/// The C core returns the end of `haystack` for an empty needle, the reverse mirror of `strstr`, so
/// `rfind(haystack, b"")` is always `Some(haystack.len())`, matching `"abc".rfind("") == Some(3)`.
/// This holds even for an empty `haystack`.
#[inline(always)]
pub fn rfind<Haystack, Needle>(haystack: Haystack, needle: Needle) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let needle_ref = needle.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();
    let needle_pointer = needle_ref.as_ptr() as _;
    let needle_length = needle_ref.len();
    let mut result = core::ptr::null();
    unsafe {
        sz_rfind_best(
            haystack_pointer,
            haystack_length,
            needle_pointer,
            needle_length,
            &mut result,
            enabled_cpu_capabilities_mask(),
            core::ptr::null_mut(),
        )
    }
    .infallible();

    if result.is_null() {
        None
    } else {
        Some(match_offset(result, haystack_pointer))
    }
}

/// Checks whether `needle` occurs anywhere within `haystack`.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needle`: The byte slice to look for within the haystack.
///
/// # Returns
///
/// `true` if `needle` occurs within `haystack`, `false` otherwise.
///
/// # Empty needle
///
/// Mirrors `str::contains`: an empty needle is always present, so `contains(haystack, b"")` is
/// always `true`, matching `"abc".contains("") == true`, even for an empty `haystack`.
#[inline(always)]
pub fn contains<Haystack, Needle>(haystack: Haystack, needle: Needle) -> bool
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    find(haystack, needle).is_some()
}

/// Finds the index of the first character in `haystack` that is also present in `needles`.
/// This function is particularly useful for parsing and tokenization tasks where a set of
/// delimiter characters is used.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the first occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
#[inline(always)]
pub fn find_byteset<Haystack>(haystack: Haystack, needles: Byteset) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();

    let mut result = core::ptr::null();
    unsafe {
        sz_find_byteset_best(
            haystack_pointer,
            haystack_length,
            &needles as *const _ as *const c_void,
            &mut result,
            enabled_cpu_capabilities_mask(),
            core::ptr::null_mut(),
        )
    }
    .infallible();
    if result.is_null() {
        None
    } else {
        Some(match_offset(result, haystack_pointer))
    }
}

/// Finds the index of the last character in `haystack` that is also present in `needles`.
/// This can be used to find the last occurrence of any character from a specified set,
/// useful in parsing scenarios such as finding the last delimiter in a string.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the last occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn rfind_byteset<Haystack>(haystack: Haystack, needles: Byteset) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();

    let mut result = core::ptr::null();
    unsafe {
        sz_rfind_byteset_best(
            haystack_pointer,
            haystack_length,
            &needles as *const _ as *const c_void,
            &mut result,
            enabled_cpu_capabilities_mask(),
            core::ptr::null_mut(),
        )
    }
    .infallible();
    if result.is_null() {
        None
    } else {
        Some(match_offset(result, haystack_pointer))
    }
}

/// Finds the index of the first character in `haystack` that is also present in `needles`.
/// This function is particularly useful for parsing and tokenization tasks where a set of
/// delimiter characters is used.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the first occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
#[inline(always)]
pub fn find_byte_from<Haystack, Needle>(haystack: Haystack, needles: Needle) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    find_byteset(haystack, Byteset::from(needles))
}

/// Finds the index of the last character in `haystack` that is also present in `needles`.
/// This can be used to find the last occurrence of any character from a specified set,
/// useful in parsing scenarios such as finding the last delimiter in a string.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needles`: The set of bytes to search for within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the last occurrence of any byte from
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn rfind_byte_from<Haystack, Needle>(haystack: Haystack, needles: Needle) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    rfind_byteset(haystack, Byteset::from(needles))
}

/// Finds the index of the first character in `haystack` that is not present in `needles`.
/// This function is useful for skipping over a known set of characters and finding the
/// first character that does not belong to that set.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needles`: The set of bytes that should not be matched within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the first occurrence of any byte not in
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn find_byte_not_from<Haystack, Needle>(haystack: Haystack, needles: Needle) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    find_byteset(haystack, Byteset::from(needles).inverted())
}

/// Finds the index of the last character in `haystack` that is not present in `needles`.
/// Useful for text processing tasks such as trimming trailing characters that belong to
/// a specified set.
///
/// # Arguments
///
/// - `haystack`: The byte slice to search.
/// - `needles`: The set of bytes that should not be matched within the haystack.
///
/// # Returns
///
/// An `Option<usize>` representing the index of the last occurrence of any byte not in
/// `needles` within `haystack`, if found, otherwise `None`.
pub fn rfind_byte_not_from<Haystack, Needle>(haystack: Haystack, needles: Needle) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
    Needle: AsRef<[u8]>,
{
    rfind_byteset(haystack, Byteset::from(needles).inverted())
}

/// What a match or split iterator looks for; the iterator type picks the direction.
#[derive(Debug, Clone, Copy)]
pub enum Matcher<'a> {
    /// An exact byte sequence.
    Substring(&'a [u8]),
    /// Any one byte of the set; invert the set to match the bytes outside it.
    Bytes(Byteset),
}

impl Matcher<'_> {
    /// Offset of the first match in `haystack`.
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        match *self {
            Matcher::Substring(needle) => find(haystack, needle),
            Matcher::Bytes(set) => find_byteset(haystack, set),
        }
    }

    /// Offset of the last match in `haystack`.
    pub fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        match *self {
            Matcher::Substring(needle) => rfind(haystack, needle),
            Matcher::Bytes(set) => rfind_byteset(haystack, set),
        }
    }

    /// Bytes every match spans: the needle's length, or one for a byteset.
    pub fn width(&self) -> usize {
        match self {
            Matcher::Substring(needle) => needle.len(),
            Matcher::Bytes(_) => 1,
        }
    }
}

/// An iterator over the matches of a pattern in a byte slice, front to back.
///
/// # Empty needle
///
/// An empty needle matches at every offset from 0 to `haystack.len()`, so an `n`-byte haystack
/// yields `n + 1` empty matches, mirroring `"abc".matches("").count() == 4`.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{FindMatches, Matcher};
///
/// let matches: Vec<&[u8]> = FindMatches::new(b"abababa", Matcher::Substring(b"aba")).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct FindMatches<'a, Overlap: Overlaps = NonOverlapping> {
    matcher: Matcher<'a>,
    rest: Option<&'a [u8]>, // The text after the last match's stride, `None` once exhausted
    _overlaps: PhantomData<Overlap>,
}

impl<'a> FindMatches<'a, NonOverlapping> {
    pub fn new(haystack: &'a [u8], matcher: Matcher<'a>) -> Self {
        Self {
            matcher,
            rest: Some(haystack),
            _overlaps: PhantomData,
        }
    }

    /// Report overlapping matches too, a compile-time policy returning the `Overlapping` variant.
    pub fn overlapping(self) -> FindMatches<'a, Overlapping> {
        FindMatches {
            matcher: self.matcher,
            rest: self.rest,
            _overlaps: PhantomData,
        }
    }
}

impl<'a, Overlap: Overlaps> Iterator for FindMatches<'a, Overlap> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.rest?;
        let Some(start) = self.matcher.find(rest) else {
            self.rest = None;
            return None;
        };
        let width = self.matcher.width();
        let stride = if Overlap::OVERLAP { 1 } else { width.max(1) };
        self.rest = rest.get(start + stride..);
        Some(&rest[start..start + width])
    }
}

/// An iterator over the matches of a pattern in a byte slice, back to front.
///
/// # Empty needle
///
/// An empty needle matches at every offset from `haystack.len()` down to 0, so an `n`-byte haystack
/// yields `n + 1` empty matches.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Matcher, RFindMatches};
///
/// let matches: Vec<&[u8]> = RFindMatches::new(b"abababa", Matcher::Substring(b"aba")).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct RFindMatches<'a, Overlap: Overlaps = NonOverlapping> {
    matcher: Matcher<'a>,
    rest: Option<&'a [u8]>, // The text before the last match's stride, `None` once exhausted
    _overlaps: PhantomData<Overlap>,
}

impl<'a> RFindMatches<'a, NonOverlapping> {
    pub fn new(haystack: &'a [u8], matcher: Matcher<'a>) -> Self {
        Self {
            matcher,
            rest: Some(haystack),
            _overlaps: PhantomData,
        }
    }

    /// Report overlapping matches too, a compile-time policy returning the `Overlapping` variant.
    pub fn overlapping(self) -> RFindMatches<'a, Overlapping> {
        RFindMatches {
            matcher: self.matcher,
            rest: self.rest,
            _overlaps: PhantomData,
        }
    }
}

impl<'a, Overlap: Overlaps> Iterator for RFindMatches<'a, Overlap> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let rest = self.rest?;
        let Some(start) = self.matcher.rfind(rest) else {
            self.rest = None;
            return None;
        };
        let width = self.matcher.width();
        let stride = if Overlap::OVERLAP { 1 } else { width.max(1) };
        self.rest = (start + width).checked_sub(stride).map(|kept| &rest[..kept]);
        Some(&rest[start..start + width])
    }
}

/// An iterator over the segments of a byte slice between the matches of a pattern, front to back.
///
/// By default empty segments are __kept__, mirroring `str::split`, so adjacent separators and a
/// leading or trailing one yield empty slices. Call [`Self::skip_empty`] to drop them.
///
/// # Empty needle and empty haystack
///
/// An empty needle never splits, yielding the whole haystack as one segment. An empty haystack
/// yields one empty segment, mirroring `"".split(",") == [""]`.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{FindSplits, Matcher};
///
/// let splits: Vec<&[u8]> = FindSplits::new(b"a,b,c,d", Matcher::Substring(b",")).collect();
/// assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
/// ```
pub struct FindSplits<'a, Empty: EmptySegments = KeepEmpty> {
    matcher: Matcher<'a>,
    rest: Option<&'a [u8]>, // The text after the last separator, `None` after the final segment
    _empties: PhantomData<Empty>,
}

impl<'a> FindSplits<'a, KeepEmpty> {
    pub fn new(haystack: &'a [u8], matcher: Matcher<'a>) -> Self {
        Self {
            matcher,
            rest: Some(haystack),
            _empties: PhantomData,
        }
    }

    /// Drop zero-length segments, a compile-time policy returning the `SkipEmpty` variant.
    pub fn skip_empty(self) -> FindSplits<'a, SkipEmpty> {
        FindSplits {
            matcher: self.matcher,
            rest: self.rest,
            _empties: PhantomData,
        }
    }
}

impl<'a, Empty: EmptySegments> Iterator for FindSplits<'a, Empty> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let width = self.matcher.width();
        loop {
            let rest = self.rest?;
            let found = if width == 0 { None } else { self.matcher.find(rest) };
            let segment = match found {
                Some(start) => {
                    self.rest = Some(&rest[start + width..]);
                    &rest[..start]
                }
                None => {
                    self.rest = None;
                    rest
                }
            };
            if !(Empty::SKIP && segment.is_empty()) {
                return Some(segment);
            }
        }
    }
}

/// An iterator over the segments of a byte slice between the matches of a pattern, back to front.
///
/// By default empty segments are __kept__, mirroring `str::rsplit`. Call [`Self::skip_empty`] to
/// drop them. An empty needle never splits, yielding the whole haystack as one segment.
///
/// # Examples
///
/// ```
/// use stringzilla::stringzilla::{Matcher, RFindSplits};
///
/// let splits: Vec<&[u8]> = RFindSplits::new(b"a,b,c,d", Matcher::Substring(b",")).collect();
/// assert_eq!(splits, vec![b"d", b"c", b"b", b"a"]);
/// ```
pub struct RFindSplits<'a, Empty: EmptySegments = KeepEmpty> {
    matcher: Matcher<'a>,
    rest: Option<&'a [u8]>, // The text before the last separator, `None` after the final segment
    _empties: PhantomData<Empty>,
}

impl<'a> RFindSplits<'a, KeepEmpty> {
    pub fn new(haystack: &'a [u8], matcher: Matcher<'a>) -> Self {
        Self {
            matcher,
            rest: Some(haystack),
            _empties: PhantomData,
        }
    }

    /// Drop zero-length segments, a compile-time policy returning the `SkipEmpty` variant.
    pub fn skip_empty(self) -> RFindSplits<'a, SkipEmpty> {
        RFindSplits {
            matcher: self.matcher,
            rest: self.rest,
            _empties: PhantomData,
        }
    }
}

impl<'a, Empty: EmptySegments> Iterator for RFindSplits<'a, Empty> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let width = self.matcher.width();
        loop {
            let rest = self.rest?;
            let found = if width == 0 { None } else { self.matcher.rfind(rest) };
            let segment = match found {
                Some(start) => {
                    self.rest = Some(&rest[..start]);
                    &rest[start + width..]
                }
                None => {
                    self.rest = None;
                    rest
                }
            };
            if !(Empty::SKIP && segment.is_empty()) {
                return Some(segment);
            }
        }
    }
}

/// Trait for binary string operations that take a needle parameter.
/// These operations include searching, splitting, and pattern matching.
///
/// # Examples
///
/// Basic usage on a string slice:
///
/// ```
/// use stringzilla::sz::StringZillableBinary;
///
/// let haystack = "Hello, world!";
/// assert_eq!(haystack.sz_find("world".as_bytes()), Some(7));
/// ```
pub trait StringZillableBinary<'a, Needle>
where
    Needle: AsRef<[u8]> + 'a,
{
    /// Searches for the first occurrence of `needle` in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find("world".as_bytes()), Some(7));
    /// ```
    fn sz_find(&self, needle: Needle) -> Option<usize>;

    /// Searches for the last occurrence of `needle` in `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world, world!";
    /// assert_eq!(haystack.sz_rfind("world".as_bytes()), Some(14));
    /// ```
    fn sz_rfind(&self, needle: Needle) -> Option<usize>;

    /// Finds the index of the first character in `self` that is also present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find_byte_from("aeiou".as_bytes()), Some(1));
    /// ```
    fn sz_find_byte_from(&self, needles: Needle) -> Option<usize>;

    /// Finds the index of the last character in `self` that is also present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_rfind_byte_from("aeiou".as_bytes()), Some(8));
    /// ```
    fn sz_rfind_byte_from(&self, needles: Needle) -> Option<usize>;

    /// Finds the index of the first character in `self` that is not present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_find_byte_not_from("aeiou".as_bytes()), Some(0));
    /// ```
    fn sz_find_byte_not_from(&self, needles: Needle) -> Option<usize>;

    /// Finds the index of the last character in `self` that is not present in `needles`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = "Hello, world!";
    /// assert_eq!(haystack.sz_rfind_byte_not_from("aeiou".as_bytes()), Some(12));
    /// ```
    fn sz_rfind_byte_not_from(&self, needles: Needle) -> Option<usize>;

    /// Returns an iterator over all non-overlapping matches of the given `needle` in `self`.
    ///
    /// # Arguments
    ///
    /// - `needle`: The byte slice to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"abababa";
    /// let needle = b"aba";
    /// let matches: Vec<&[u8]> = haystack.sz_matches(needle).collect();
    /// assert_eq!(matches, vec![b"aba", b"aba"]); // non-overlapping by default (like str::matches)
    /// let overlapping: Vec<&[u8]> = haystack.sz_matches(needle).overlapping().collect();
    /// assert_eq!(overlapping, vec![b"aba", b"aba", b"aba"]); // opt in with .overlapping()
    /// ```
    fn sz_matches(&'a self, needle: &'a Needle) -> FindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of the given `needle` in `self`,
    /// searching from the end.
    ///
    /// # Arguments
    ///
    /// - `needle`: The byte slice to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"abababa";
    /// let needle = b"aba";
    /// let matches: Vec<&[u8]> = haystack.sz_rmatches(needle).collect();
    /// assert_eq!(matches, vec![b"aba", b"aba"]); // non-overlapping by default
    /// let overlapping: Vec<&[u8]> = haystack.sz_rmatches(needle).overlapping().collect();
    /// assert_eq!(overlapping, vec![b"aba", b"aba", b"aba"]); // opt in with .overlapping()
    /// ```
    fn sz_rmatches(&'a self, needle: &'a Needle) -> RFindMatches<'a>;

    /// Returns an iterator over the substrings of `self` that are separated by the given `needle`.
    ///
    /// # Arguments
    ///
    /// - `needle`: The byte slice to split `self` by.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"a,b,c,d";
    /// let needle = b",";
    /// let splits: Vec<&[u8]> = haystack.sz_splits(needle).collect();
    /// assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
    /// ```
    fn sz_splits(&'a self, needle: &'a Needle) -> FindSplits<'a>;

    /// Returns an iterator over the substrings of `self` that are separated by the given `needle`,
    /// searching from the end.
    ///
    /// # Arguments
    ///
    /// - `needle`: The byte slice to split `self` by.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"a,b,c,d";
    /// let needle = b",";
    /// let splits: Vec<&[u8]> = haystack.sz_rsplits(needle).collect();
    /// assert_eq!(splits, vec![b"d", b"c", b"b", b"a"]);
    /// ```
    fn sz_rsplits(&'a self, needle: &'a Needle) -> RFindSplits<'a>;

    /// Returns an iterator over all non-overlapping matches of any of the bytes in
    /// `needles` within `self`.
    ///
    /// # Arguments
    ///
    /// - `needles`: The set of bytes to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_first_of(needles).collect();
    /// assert_eq!(matches, vec![b"e", b"o", b"o"]);
    /// ```
    fn sz_find_first_of(&'a self, needles: &'a Needle) -> FindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of any of the bytes in `needles` within
    /// `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// - `needles`: The set of bytes to search for within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_last_of(needles).collect();
    /// assert_eq!(matches, vec![b"o", b"o", b"e"]);
    /// ```
    fn sz_find_last_of(&'a self, needles: &'a Needle) -> RFindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of any byte not in
    /// `needles` within `self`.
    ///
    /// # Arguments
    ///
    /// - `needles`: The set of bytes that should not be matched within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_first_not_of(needles).collect();
    /// assert_eq!(matches, vec![b"H", b"l", b"l", b",", b" ", b"w", b"r", b"l", b"d", b"!"]);
    /// ```
    fn sz_find_first_not_of(&'a self, needles: &'a Needle) -> FindMatches<'a>;

    /// Returns an iterator over all non-overlapping matches of any byte not in `needles` within
    /// `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// - `needles`: The set of bytes that should not be matched within `self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stringzilla::sz::StringZillableBinary;
    ///
    /// let haystack = b"Hello, world!";
    /// let needles = b"aeiou";
    /// let matches: Vec<&[u8]> = haystack.sz_find_last_not_of(needles).collect();
    /// assert_eq!(matches, vec![b"!", b"d", b"l", b"r", b"w", b" ", b",", b"l", b"l", b"H"]);
    /// ```
    fn sz_find_last_not_of(&'a self, needles: &'a Needle) -> RFindMatches<'a>;
}

impl<'a, Source, Needle> StringZillableBinary<'a, Needle> for Source
where
    Source: AsRef<[u8]> + ?Sized,
    Needle: AsRef<[u8]> + 'a,
{
    fn sz_find(&self, needle: Needle) -> Option<usize> {
        find(self, needle)
    }

    fn sz_rfind(&self, needle: Needle) -> Option<usize> {
        rfind(self, needle)
    }

    fn sz_find_byte_from(&self, needles: Needle) -> Option<usize> {
        find_byte_from(self, needles)
    }

    fn sz_rfind_byte_from(&self, needles: Needle) -> Option<usize> {
        rfind_byte_from(self, needles)
    }

    fn sz_find_byte_not_from(&self, needles: Needle) -> Option<usize> {
        find_byte_not_from(self, needles)
    }

    fn sz_rfind_byte_not_from(&self, needles: Needle) -> Option<usize> {
        rfind_byte_not_from(self, needles)
    }

    fn sz_matches(&'a self, needle: &'a Needle) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), Matcher::Substring(needle.as_ref()))
    }

    fn sz_rmatches(&'a self, needle: &'a Needle) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), Matcher::Substring(needle.as_ref()))
    }

    fn sz_splits(&'a self, needle: &'a Needle) -> FindSplits<'a> {
        FindSplits::new(self.as_ref(), Matcher::Substring(needle.as_ref()))
    }

    fn sz_rsplits(&'a self, needle: &'a Needle) -> RFindSplits<'a> {
        RFindSplits::new(self.as_ref(), Matcher::Substring(needle.as_ref()))
    }

    fn sz_find_first_of(&'a self, needles: &'a Needle) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), Matcher::Bytes(Byteset::from(needles)))
    }

    fn sz_find_last_of(&'a self, needles: &'a Needle) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), Matcher::Bytes(Byteset::from(needles)))
    }

    fn sz_find_first_not_of(&'a self, needles: &'a Needle) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), Matcher::Bytes(Byteset::from(needles).inverted()))
    }

    fn sz_find_last_not_of(&'a self, needles: &'a Needle) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), Matcher::Bytes(Byteset::from(needles).inverted()))
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::borrow::Cow;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    use super::*;
    use crate::sz;

    #[test]
    fn search() {
        let my_string: String = String::from("Hello, world!");
        let my_str: &str = my_string.as_str();
        let my_cow_str: Cow<'_, str> = Cow::from(&my_string);

        // Identical to `memchr::memmem::find` and `memchr::memmem::rfind` functions
        assert_eq!(sz::find("Hello, world!", "world"), Some(7));
        assert_eq!(sz::rfind("Hello, world!", "world"), Some(7));

        // Use the generic function with a String
        let world_string = String::from("world");
        assert_eq!(my_string.sz_find(&world_string), Some(7));
        assert_eq!(my_string.sz_rfind(&world_string), Some(7));
        assert_eq!(my_string.sz_find_byte_from(&world_string), Some(2));
        assert_eq!(my_string.sz_rfind_byte_from(&world_string), Some(11));
        assert_eq!(my_string.sz_find_byte_not_from(&world_string), Some(0));
        assert_eq!(my_string.sz_rfind_byte_not_from(&world_string), Some(12));

        // Use the generic function with a &str
        assert_eq!(my_str.sz_find("world"), Some(7));
        assert_eq!(my_str.sz_rfind("world"), Some(7));
        assert_eq!(my_str.sz_find_byte_from("world"), Some(2));
        assert_eq!(my_str.sz_rfind_byte_from("world"), Some(11));
        assert_eq!(my_str.sz_find_byte_not_from("world"), Some(0));
        assert_eq!(my_str.sz_rfind_byte_not_from("world"), Some(12));

        // Use the generic function with a Cow<'_, str>
        assert_eq!(my_cow_str.as_ref().sz_find("world"), Some(7));
        assert_eq!(my_cow_str.as_ref().sz_rfind("world"), Some(7));
        assert_eq!(my_cow_str.as_ref().sz_find_byte_from("world"), Some(2));
        assert_eq!(my_cow_str.as_ref().sz_rfind_byte_from("world"), Some(11));
        assert_eq!(my_cow_str.as_ref().sz_find_byte_not_from("world"), Some(0));
        assert_eq!(my_cow_str.as_ref().sz_rfind_byte_not_from("world"), Some(12));
    }

    #[test]
    fn empty_needle_matches_std() {
        // The C core reports an empty needle as "not found" by design, but `find`/`rfind`/
        // `contains` synthesize the `str` answer instead.
        assert_eq!(sz::find("abc", ""), Some(0));
        assert_eq!("abc".find(""), Some(0));
        assert_eq!(sz::rfind("abc", ""), Some(3));
        assert_eq!("abc".rfind(""), Some(3));
        assert!(sz::contains("abc", ""));
        assert!("abc".contains(""));

        // An empty haystack is a degenerate but well-defined case too.
        assert_eq!(sz::find("", ""), Some(0));
        assert_eq!("".find(""), Some(0));
        assert_eq!(sz::rfind("", ""), Some(0));
        assert_eq!("".rfind(""), Some(0));
        assert!(sz::contains("", ""));
        assert!("".contains(""));

        // Non-empty needles are unaffected.
        assert_eq!(sz::find("abc", "b"), Some(1));
        assert_eq!(sz::rfind("abc", "b"), Some(1));
        assert!(sz::contains("abc", "b"));
        assert!(!sz::contains("abc", "z"));
    }

    #[test]
    fn iter_matches_forward() {
        let haystack = b"hello world hello universe";
        let needle = b"hello";
        let matches: Vec<_> = haystack.sz_matches(needle).collect();
        assert_eq!(matches, vec![b"hello", b"hello"]);
    }

    #[test]
    fn iter_matches_reverse() {
        let haystack = b"hello world hello universe";
        let needle = b"hello";
        let matches: Vec<_> = haystack.sz_rmatches(needle).collect();
        assert_eq!(matches, vec![b"hello", b"hello"]);
    }

    #[test]
    fn iter_splits_forward() {
        let haystack = b"alpha,beta;gamma";
        let needle = b",";
        let splits: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(splits, vec![&b"alpha"[..], &b"beta;gamma"[..]]);
    }

    #[test]
    fn iter_splits_reverse() {
        let haystack = b"alpha,beta;gamma";
        let needle = b";";
        let splits: Vec<_> = haystack.sz_rsplits(needle).collect();
        assert_eq!(splits, vec![&b"gamma"[..], &b"alpha,beta"[..]]);
    }

    #[test]
    fn iter_splits_with_empty_parts() {
        let haystack = b"a,,b,";
        let needle = b",";
        let splits: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(splits, vec![b"a", &b""[..], b"b", &b""[..]]);
    }

    #[test]
    fn iter_splits_empty_haystack_yields_one_empty_segment() {
        // Mirrors `"".split(",") == [""]`, not zero segments.
        let matcher = Matcher::Substring(b",");
        let splits: Vec<_> = FindSplits::new(b"", matcher).collect();
        assert_eq!(splits, vec![&b""[..]]);
    }

    #[test]
    fn iter_matches_forward_empty_needle_matches_std() {
        let empty = Matcher::Substring(b"");
        let matches: Vec<_> = FindMatches::new(b"abc", empty).collect();
        assert_eq!(matches, vec![&b""[..]; 4]);
        assert_eq!("abc".matches("").count(), 4);
        assert_eq!(FindMatches::new(b"abc", empty).overlapping().count(), 4);
        assert_eq!(FindMatches::new(b"", empty).count(), 1);
    }

    #[test]
    fn iter_matches_reverse_empty_needle() {
        let empty = Matcher::Substring(b"");
        let matches: Vec<_> = RFindMatches::new(b"abc", empty).collect();
        assert_eq!(matches, vec![&b""[..]; 4]);
        assert_eq!(RFindMatches::new(b"abc", empty).overlapping().count(), 4);
        assert_eq!(RFindMatches::new(b"", empty).count(), 1);
    }

    #[test]
    fn iter_splits_forward_empty_needle() {
        let splits: Vec<_> = FindSplits::new(b"abc", Matcher::Substring(b"")).collect();
        assert_eq!(splits, vec![&b"abc"[..]]);
    }

    #[test]
    fn iter_splits_reverse_empty_needle() {
        let splits: Vec<_> = RFindSplits::new(b"abc", Matcher::Substring(b"")).collect();
        assert_eq!(splits, vec![&b"abc"[..]]);
    }

    #[test]
    fn iter_splits_forward_skip_empty() {
        // Default KEEP yields empties; skip_empty drops every zero-length segment.
        let haystack = b"a,,b,";
        let needle = b",";
        let kept: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(kept, vec![b"a", &b""[..], b"b", &b""[..]]);
        let nonempty: Vec<_> = haystack.sz_splits(needle).skip_empty().collect();
        assert_eq!(nonempty, vec![b"a", b"b"]);
    }

    #[test]
    fn iter_splits_reverse_skip_empty() {
        // KEEP rsplit of "a,,b," is the reverse of the forward split, empties included.
        let haystack = b"a,,b,";
        let needle = b",";
        let kept: Vec<_> = haystack.sz_rsplits(needle).collect();
        assert_eq!(kept, vec![&b""[..], b"b", &b""[..], b"a"]);
        let nonempty: Vec<_> = haystack.sz_rsplits(needle).skip_empty().collect();
        assert_eq!(nonempty, vec![b"b", b"a"]);
    }

    #[test]
    fn iter_splits_byteset_skip_empty() {
        // Byteset matcher (split on any of ",;"): adjacent delimiters yield empties under
        // the KEEP default.
        let haystack = b",a;;b,";
        let separators = Matcher::Bytes(Byteset::from(b",;"));
        let kept: Vec<_> = FindSplits::new(haystack, separators).collect();
        assert_eq!(kept, vec![&b""[..], b"a", &b""[..], b"b", &b""[..]]);
        let nonempty: Vec<_> = FindSplits::new(haystack, separators).skip_empty().collect();
        assert_eq!(nonempty, vec![b"a", b"b"]);
        let reversed: Vec<_> = RFindSplits::new(haystack, separators).collect();
        assert_eq!(reversed, vec![&b""[..], b"b", &b""[..], b"a", &b""[..]]);
    }

    #[test]
    fn iter_matches_with_overlaps() {
        let haystack = b"aaaa";
        let needle = b"aa";
        // Default is non-overlapping; `.overlapping()` opts into the
        // compile-time Overlapping policy.
        let non_overlapping: Vec<_> = haystack.sz_matches(needle).collect();
        assert_eq!(non_overlapping, vec![b"aa", b"aa"]);
        let matches: Vec<_> = haystack.sz_matches(needle).overlapping().collect();
        assert_eq!(matches, vec![b"aa", b"aa", b"aa"]);
    }

    #[test]
    fn iter_splits_with_utf8_haystack() {
        let haystack = "こんにちは,世界".as_bytes();
        let needle = b",";
        let splits: Vec<_> = haystack.sz_splits(needle).collect();
        assert_eq!(splits, vec!["こんにちは".as_bytes(), "世界".as_bytes()]);
    }

    #[test]
    fn iter_find_first_of() {
        let haystack = b"hello world";
        let needles = b"or";
        let matches: Vec<_> = haystack.sz_find_first_of(needles).collect();
        assert_eq!(matches, vec![b"o", b"o", b"r"]);
    }

    #[test]
    fn iter_find_last_of() {
        let haystack = b"hello world";
        let needles = b"or";
        let matches: Vec<_> = haystack.sz_find_last_of(needles).collect();
        assert_eq!(matches, vec![b"r", b"o", b"o"]);
    }

    #[test]
    fn iter_find_first_not_of() {
        let haystack = b"aabbbcccd";
        let needles = b"ab";
        let matches: Vec<_> = haystack.sz_find_first_not_of(needles).collect();
        assert_eq!(matches, vec![b"c", b"c", b"c", b"d"]);
    }

    #[test]
    fn iter_find_last_not_of() {
        let haystack = b"aabbbcccd";
        let needles = b"cd";
        let matches: Vec<_> = haystack.sz_find_last_not_of(needles).collect();
        assert_eq!(matches, vec![b"b", b"b", b"b", b"a", b"a"]);
    }

    #[test]
    fn iter_find_first_of_empty_needles() {
        let haystack = b"hello world";
        let needles = b"";
        let matches: Vec<_> = haystack.sz_find_first_of(needles).collect();
        assert_eq!(matches, Vec::<&[u8]>::new());
    }

    #[test]
    fn iter_find_last_of_empty_haystack() {
        let haystack = b"";
        let needles = b"abc";
        let matches: Vec<_> = haystack.sz_find_last_of(needles).collect();
        assert_eq!(matches, Vec::<&[u8]>::new());
    }

    #[test]
    fn iter_find_first_not_of_all_matching() {
        let haystack = b"aaabbbccc";
        let needles = b"abc";
        let matches: Vec<_> = haystack.sz_find_first_not_of(needles).collect();
        assert_eq!(matches, Vec::<&[u8]>::new());
    }

    #[test]
    fn iter_find_last_not_of_all_not_matching() {
        let haystack = b"hello world";
        let needles = b"xyz";
        let matches: Vec<_> = haystack.sz_find_last_not_of(needles).collect();
        assert_eq!(
            matches,
            vec![b"d", b"l", b"r", b"o", b"w", b" ", b"o", b"l", b"l", b"e", b"h"]
        );
    }

    #[test]
    fn iter_find_matches_overlapping() {
        let haystack = b"aaaa";
        let matcher = Matcher::Substring(b"aa");
        let matches: Vec<_> = FindMatches::new(haystack, matcher).overlapping().collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_find_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = Matcher::Substring(b"aa");
        let matches: Vec<_> = FindMatches::new(haystack, matcher).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_rfind_matches_overlapping() {
        let haystack = b"aaaa";
        let matcher = Matcher::Substring(b"aa");
        let matches: Vec<_> = RFindMatches::new(haystack, matcher).overlapping().collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_rfind_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = Matcher::Substring(b"aa");
        let matches: Vec<_> = RFindMatches::new(haystack, matcher).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_same_length() {
        let mut buffer = b"abcabc".to_vec();
        let replaced = sz::replace_all(&mut buffer, b"ab", b"XY").expect("replace_all failed");
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"XYcXYc");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_shrinks() {
        let mut buffer = b"aaaa".to_vec();
        let replaced = sz::replace_all(&mut buffer, b"aa", b"b").expect("replace_all failed");
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"bb");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_grows() {
        let mut buffer = b"aba".to_vec();
        let replaced = sz::replace_all(&mut buffer, b"a", b"XYZ").expect("replace_all failed");
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"XYZbXYZ");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_byteset_basic() {
        let mut buffer = b"hello world".to_vec();
        let vowels = sz::Byteset::from("aeiou");
        let replaced = sz::replace_all_byteset(&mut buffer, vowels, b"_").expect("replace_all_byteset failed");
        assert_eq!(replaced, 3);
        assert_eq!(buffer, b"h_ll_ w_rld");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_byteset_grows() {
        let mut buffer = b"yzz".to_vec();
        let vowels = sz::Byteset::from("y");
        let replaced = sz::replace_all_byteset(&mut buffer, vowels, b"(y)").expect("replace_all_byteset failed");
        assert_eq!(replaced, 1);
        assert_eq!(buffer, b"(y)zz");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_noop_on_empty_pattern() {
        let mut buffer = b"unchanged".to_vec();
        let replaced = sz::replace_all(&mut buffer, b"", b"anything").expect("replace_all failed");
        assert_eq!(replaced, 0);
        assert_eq!(buffer, b"unchanged");
    }
}
