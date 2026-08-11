//! Substring and byteset search, partitioning, counting, and splitting.
//!
//! Also home to the binary-operation trait shared by the match and split iterators.

use super::*;
use core::ffi::c_void;
use core::marker::PhantomData;

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
    let result = unsafe { sz_find(haystack_pointer, haystack_length, needle_pointer, needle_length) };

    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
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
    let result = unsafe { sz_rfind(haystack_pointer, haystack_length, needle_pointer, needle_length) };

    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
    }
}

/// Checks whether `needle` occurs anywhere within `haystack`.
///
/// # Arguments
///
/// * `haystack`: The byte slice to search.
/// * `needle`: The byte slice to look for within the haystack.
///
/// # Returns
///
/// `true` if `needle` occurs within `haystack`, `false` otherwise.
///
/// # Empty needle
///
/// Mirrors `str::contains`: an empty needle is always present, so `contains(haystack, b"")` is
/// always `true`, matching `"abc".contains("") == true` (even for an empty `haystack`).
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
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes to search for within the haystack.
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

    let result = unsafe { sz_find_byteset(haystack_pointer, haystack_length, &needles as *const _ as *const c_void) };
    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
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
pub fn rfind_byteset<Haystack>(haystack: Haystack, needles: Byteset) -> Option<usize>
where
    Haystack: AsRef<[u8]>,
{
    let haystack_ref = haystack.as_ref();
    let haystack_pointer = haystack_ref.as_ptr() as _;
    let haystack_length = haystack_ref.len();

    let result = unsafe { sz_rfind_byteset(haystack_pointer, haystack_length, &needles as *const _ as *const c_void) };
    if result.is_null() {
        None
    } else {
        Some(unsafe { result.offset_from(haystack_pointer) }.try_into().unwrap())
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
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes to search for within the haystack.
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
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes that should not be matched within the haystack.
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
/// * `haystack`: The byte slice to search.
/// * `needles`: The set of bytes that should not be matched within the haystack.
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

pub enum MatcherType<'a> {
    Find(&'a [u8]),
    RFind(&'a [u8]),
    FindFirstOf(&'a [u8]),
    FindLastOf(&'a [u8]),
    FindFirstNotOf(&'a [u8]),
    FindLastNotOf(&'a [u8]),
}

impl<'a> MatcherType<'a> {
    /// Runs this matcher's search mode over `haystack`, yielding the offset of the first hit.
    pub fn find(&self, haystack: &'a [u8]) -> Option<usize> {
        match self {
            MatcherType::Find(needle) => find(haystack, needle),
            MatcherType::RFind(needle) => rfind(haystack, needle),
            MatcherType::FindFirstOf(needles) => find_byte_from(haystack, needles),
            MatcherType::FindLastOf(needles) => rfind_byte_from(haystack, needles),
            MatcherType::FindFirstNotOf(needles) => find_byte_not_from(haystack, needles),
            MatcherType::FindLastNotOf(needles) => rfind_byte_not_from(haystack, needles),
        }
    }

    /// Bytes to skip past a hit: the needle's own length for substring modes, one byte for byteset modes.
    pub fn needle_length(&self) -> usize {
        match self {
            MatcherType::Find(needle) | MatcherType::RFind(needle) => needle.len(),
            _ => 1,
        }
    }
}

/// An iterator over non-overlapping matches of a pattern in a string slice.
/// This iterator yields the matched substrings in the order they are found.
///
/// # Empty needle
///
/// An empty needle matches at every position, including past the last byte: iterating over an
/// `n`-byte haystack yields `n + 1` empty matches, mirroring `"abc".matches("").count() == 4`.
/// Each zero-length match still advances the search position by at least one byte, so the
/// iterator always terminates instead of looping forever on the same spot.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, FindMatches}};
///
/// let haystack = b"abababa";
/// let matcher = MatcherType::Find(b"aba");
/// let matches: Vec<&[u8]> = FindMatches::new(haystack, matcher).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct FindMatches<'a, Overlap: Overlaps = NonOverlapping> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    _overlaps: PhantomData<Overlap>,
}

impl<'a> FindMatches<'a, NonOverlapping> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: 0,
            _overlaps: PhantomData,
        }
    }

    /// Report overlapping matches too (compile-time policy; returns the `Overlapping` variant).
    pub fn overlapping(self) -> FindMatches<'a, Overlapping> {
        FindMatches {
            haystack: self.haystack,
            matcher: self.matcher,
            position: self.position,
            _overlaps: PhantomData,
        }
    }
}

impl<'a, Overlap: Overlaps> Iterator for FindMatches<'a, Overlap> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        // An empty needle matches even in the empty slice at `haystack.len()`, so the bound is
        // exclusive on the *next* sentinel position, not on `haystack.len()` itself; once
        // exhausted, `position` is parked one past `haystack.len()` so this guard is stable.
        if self.position > self.haystack.len() {
            return None;
        }

        if let Some(index) = self.matcher.find(&self.haystack[self.position..]) {
            debug_assert!(
                self.position + index + self.matcher.needle_length() <= self.haystack.len(),
                "matcher returned a match span past the haystack end"
            );
            let start = self.position + index;
            let end = start + self.matcher.needle_length();
            // A zero-length match (empty needle) must still advance by at least one byte, or
            // this would loop forever re-matching the same position.
            let step = if Overlap::OVERLAP {
                1
            } else {
                self.matcher.needle_length().max(1)
            };
            self.position = start + step;
            Some(&self.haystack[start..end])
        } else {
            self.position = self.haystack.len() + 1;
            None
        }
    }
}

/// An iterator over non-overlapping splits of a string slice by a pattern.
/// This iterator yields the substrings between the matches of the pattern.
///
/// By default empty segments are **kept** (adjacent delimiters and leading/trailing matches yield empty
/// slices, mirroring `str::split`). Call [`Self::skip_empty`] to drop zero-length segments. The `STEPS`
/// const-generic mirrors the UTF-8 split iterators for API uniformity; substring/byteset splits search
/// match-by-match, so it does not affect the yielded segments.
///
/// # Empty needle and empty haystack
///
/// An empty haystack always yields exactly one (empty) segment, mirroring `"".split(",") == [""]`.
/// An empty needle matches at every position - including past the last byte - so it still yields one
/// empty segment per position instead of hanging: each zero-length match advances the search position
/// by at least one byte.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, FindSplits}};
///
/// let haystack = b"a,b,c,d";
/// let matcher = MatcherType::Find(b",");
/// let splits: Vec<&[u8]> = FindSplits::new(haystack, matcher).collect();
/// assert_eq!(splits, vec![b"a", b"b", b"c", b"d"]);
/// ```
pub struct FindSplits<'a, Empty: EmptySegments = KeepEmpty, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: usize,
    _empties: PhantomData<Empty>,
}

impl<'a> FindSplits<'a, KeepEmpty, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self::with_steps(haystack, matcher)
    }
}

impl<'a, const STEPS: usize> FindSplits<'a, KeepEmpty, STEPS> {
    /// Constructs an iterator with an explicit batch size (kept for API uniformity with the UTF-8 splits).
    pub fn with_steps(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: 0,
            _empties: PhantomData,
        }
    }

    /// Drop zero-length segments (compile-time policy; returns the `SkipEmpty` variant).
    pub fn skip_empty(self) -> FindSplits<'a, SkipEmpty, STEPS> {
        FindSplits {
            haystack: self.haystack,
            matcher: self.matcher,
            position: self.position,
            _empties: PhantomData,
        }
    }
}

impl<'a, Empty: EmptySegments, const STEPS: usize> FindSplits<'a, Empty, STEPS> {
    /// Yields the next raw segment without the empty-segment filter.
    #[inline(always)]
    fn next_raw(&mut self) -> Option<&'a [u8]> {
        // Empty delimiter: no split.
        if self.matcher.needle_length() == 0 {
            if self.position > self.haystack.len() {
                return None;
            }
            self.position = self.haystack.len() + 1;
            return Some(self.haystack);
        }
        // `position` only ever exceeds `haystack.len()` once the trailing segment below has
        // already been emitted; that sentinel, rather than tracking "did we ever match", is
        // what makes this correctly yield one empty segment for a completely empty haystack.
        if self.position > self.haystack.len() {
            return None;
        }

        if let Some(index) = self.matcher.find(&self.haystack[self.position..]) {
            debug_assert!(
                self.position + index + self.matcher.needle_length() <= self.haystack.len(),
                "matcher returned a match span past the haystack end"
            );
            let start = self.position;
            let end = self.position + index;
            // A zero-length match (empty needle) must still advance by at least one byte, or
            // this would loop forever re-matching the same position.
            self.position = end + self.matcher.needle_length().max(1);
            Some(&self.haystack[start..end])
        } else {
            let start = self.position;
            self.position = self.haystack.len() + 1;
            Some(&self.haystack[start..])
        }
    }
}

impl<'a, Empty: EmptySegments, const STEPS: usize> Iterator for FindSplits<'a, Empty, STEPS> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = self.next_raw()?;
            if Empty::SKIP && segment.is_empty() {
                continue;
            }
            return Some(segment);
        }
    }
}

/// An iterator over non-overlapping matches of a pattern in a string slice, searching from the end.
/// This iterator yields the matched substrings in reverse order.
///
/// # Empty needle
///
/// An empty needle matches at every position, including past the last byte: iterating over an
/// `n`-byte haystack yields `n + 1` empty matches, in reverse order. Each zero-length match still
/// shrinks the remaining search window by at least one byte, so the iterator always terminates
/// instead of looping forever on the same spot.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RFindMatches}};
///
/// let haystack = b"abababa";
/// let matcher = MatcherType::RFind(b"aba");
/// let matches: Vec<&[u8]> = RFindMatches::new(haystack, matcher).collect();
/// assert_eq!(matches, vec![b"aba", b"aba"]);
/// ```
pub struct RFindMatches<'a, Overlap: Overlaps = NonOverlapping> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    // Right-exclusive bound of the unsearched prefix; `usize::MAX` means exhausted.
    position: usize,
    _overlaps: PhantomData<Overlap>,
}

impl<'a> RFindMatches<'a, NonOverlapping> {
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: haystack.len(),
            _overlaps: PhantomData,
        }
    }

    /// Report overlapping matches too (compile-time policy; returns the `Overlapping` variant).
    pub fn overlapping(self) -> RFindMatches<'a, Overlapping> {
        RFindMatches {
            haystack: self.haystack,
            matcher: self.matcher,
            position: self.position,
            _overlaps: PhantomData,
        }
    }
}

impl<'a, Overlap: Overlaps> Iterator for RFindMatches<'a, Overlap> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position == usize::MAX {
            return None;
        }

        let previous_position = self.position;
        let search_area = &self.haystack[..self.position];
        if let Some(index) = self.matcher.find(search_area) {
            let start = index;
            let end = start + self.matcher.needle_length();
            let result = Some(&self.haystack[start..end]);

            let skip = if Overlap::OVERLAP {
                self.matcher.needle_length().saturating_sub(1)
            } else {
                0
            };
            let next_position = start + skip;
            // A zero-length match (empty needle) can land exactly at the current window's
            // right edge, leaving `next_position == previous_position`; shrink by one more so
            // the window keeps making progress. Once there is nothing left to shrink, mark the
            // iterator exhausted via the `usize::MAX` sentinel instead of wrapping around.
            self.position = if next_position < previous_position {
                next_position
            } else if next_position == 0 {
                usize::MAX
            } else {
                next_position - 1
            };

            result
        } else {
            None
        }
    }
}

/// An iterator over non-overlapping splits of a string slice by a pattern, searching from the end.
/// This iterator yields the substrings between the matches of the pattern in reverse order.
///
/// By default empty segments are **kept** (mirroring `str::rsplit`). Call [`Self::skip_empty`] to drop
/// zero-length segments. The `STEPS` const-generic mirrors the UTF-8 split iterators for API uniformity;
/// substring/byteset splits search match-by-match, so it does not affect the yielded segments.
///
/// # Empty needle
///
/// An empty needle matches at every position, including past the last byte, so it still yields one
/// empty segment per position instead of hanging: each zero-length match shrinks the remaining
/// search window by at least one byte.
///
/// # Examples
///
/// ```
/// use stringzilla::{stringzilla as sz, stringzilla::{MatcherType, RFindSplits}};
///
/// let haystack = b"a,b,c,d";
/// let matcher = MatcherType::RFind(b",");
/// let splits: Vec<&[u8]> = RFindSplits::new(haystack, matcher).collect();
/// assert_eq!(splits, vec![b"d", b"c", b"b", b"a"]);
/// ```
pub struct RFindSplits<'a, Empty: EmptySegments = KeepEmpty, const STEPS: usize = ITERATORS_DEFAULT_STEPS> {
    haystack: &'a [u8],
    matcher: MatcherType<'a>,
    position: Option<usize>, // End of the not-yet-segmented prefix; `None` once the final segment is yielded
    _empties: PhantomData<Empty>,
}

impl<'a> RFindSplits<'a, KeepEmpty, ITERATORS_DEFAULT_STEPS> {
    /// Constructs an iterator with the default batch size ([`ITERATORS_DEFAULT_STEPS`]).
    pub fn new(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self::with_steps(haystack, matcher)
    }
}

impl<'a, const STEPS: usize> RFindSplits<'a, KeepEmpty, STEPS> {
    /// Constructs an iterator with an explicit batch size (kept for API uniformity with the UTF-8 splits).
    pub fn with_steps(haystack: &'a [u8], matcher: MatcherType<'a>) -> Self {
        Self {
            haystack,
            matcher,
            position: Some(haystack.len()),
            _empties: PhantomData,
        }
    }

    /// Drop zero-length segments (compile-time policy; returns the `SkipEmpty` variant).
    pub fn skip_empty(self) -> RFindSplits<'a, SkipEmpty, STEPS> {
        RFindSplits {
            haystack: self.haystack,
            matcher: self.matcher,
            position: self.position,
            _empties: PhantomData,
        }
    }
}

impl<'a, Empty: EmptySegments, const STEPS: usize> RFindSplits<'a, Empty, STEPS> {
    /// Yields the next raw segment (reverse order) without the empty-segment filter.
    #[inline(always)]
    fn next_raw(&mut self) -> Option<&'a [u8]> {
        let position = self.position?;
        // Empty delimiter: no split.
        if self.matcher.needle_length() == 0 {
            self.position = None;
            return Some(&self.haystack[..position]);
        }
        let search_area = &self.haystack[..position];
        if let Some(index) = self.matcher.find(search_area) {
            let start = index + self.matcher.needle_length();
            // A non-empty needle always matches strictly inside `search_area`, so `index <
            // position` and the window keeps shrinking. An empty needle instead matches right
            // at the window's own edge (`index == position`); shrink by one more byte there so
            // the next call doesn't re-match the same spot, and stop once nothing is left.
            self.position = if index < position {
                Some(index)
            } else {
                index.checked_sub(1)
            };
            Some(&self.haystack[start..position])
        } else {
            self.position = None;
            Some(&self.haystack[..position])
        }
    }
}

impl<'a, Empty: EmptySegments, const STEPS: usize> Iterator for RFindSplits<'a, Empty, STEPS> {
    type Item = &'a [u8];

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let segment = self.next_raw()?;
            if Empty::SKIP && segment.is_empty() {
                continue;
            }
            return Some(segment);
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
    /// * `needle`: The byte slice to search for within `self`.
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

    /// Returns an iterator over all non-overlapping matches of the given `needle` in `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needle`: The byte slice to search for within `self`.
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
    /// * `needle`: The byte slice to split `self` by.
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

    /// Returns an iterator over the substrings of `self` that are separated by the given `needle`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needle`: The byte slice to split `self` by.
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

    /// Returns an iterator over all non-overlapping matches of any of the bytes in `needles` within `self`.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes to search for within `self`.
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

    /// Returns an iterator over all non-overlapping matches of any of the bytes in `needles` within `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes to search for within `self`.
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

    /// Returns an iterator over all non-overlapping matches of any byte not in `needles` within `self`.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes that should not be matched within `self`.
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

    /// Returns an iterator over all non-overlapping matches of any byte not in `needles` within `self`, searching from the end.
    ///
    /// # Arguments
    ///
    /// * `needles`: The set of bytes that should not be matched within `self`.
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
        FindMatches::new(self.as_ref(), MatcherType::Find(needle.as_ref()))
    }

    fn sz_rmatches(&'a self, needle: &'a Needle) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), MatcherType::RFind(needle.as_ref()))
    }

    fn sz_splits(&'a self, needle: &'a Needle) -> FindSplits<'a> {
        FindSplits::new(self.as_ref(), MatcherType::Find(needle.as_ref()))
    }

    fn sz_rsplits(&'a self, needle: &'a Needle) -> RFindSplits<'a> {
        RFindSplits::new(self.as_ref(), MatcherType::RFind(needle.as_ref()))
    }

    fn sz_find_first_of(&'a self, needles: &'a Needle) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), MatcherType::FindFirstOf(needles.as_ref()))
    }

    fn sz_find_last_of(&'a self, needles: &'a Needle) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), MatcherType::FindLastOf(needles.as_ref()))
    }

    fn sz_find_first_not_of(&'a self, needles: &'a Needle) -> FindMatches<'a> {
        FindMatches::new(self.as_ref(), MatcherType::FindFirstNotOf(needles.as_ref()))
    }

    fn sz_find_last_not_of(&'a self, needles: &'a Needle) -> RFindMatches<'a> {
        RFindMatches::new(self.as_ref(), MatcherType::FindLastNotOf(needles.as_ref()))
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
        let matcher = MatcherType::Find(b",");
        let splits: Vec<_> = FindSplits::new(b"", matcher).collect();
        assert_eq!(splits, vec![&b""[..]]);
    }

    #[test]
    fn iter_matches_forward_empty_needle_matches_std() {
        let matches: Vec<_> = FindMatches::new(b"abc", MatcherType::Find(b"")).collect();
        assert_eq!(matches, vec![&b""[..]; 4]);
        assert_eq!("abc".matches("").count(), 4);
    }

    #[test]
    fn iter_matches_reverse_empty_needle() {
        let matches: Vec<_> = RFindMatches::new(b"abc", MatcherType::RFind(b"")).collect();
        assert_eq!(matches, vec![&b""[..]; 4]);
    }

    #[test]
    fn iter_splits_forward_empty_needle() {
        let splits: Vec<_> = FindSplits::new(b"abc", MatcherType::Find(b"")).collect();
        assert_eq!(splits, vec![&b"abc"[..]]);
    }

    #[test]
    fn iter_splits_reverse_empty_needle() {
        let splits: Vec<_> = RFindSplits::new(b"abc", MatcherType::RFind(b"")).collect();
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
        // Byteset matcher (split on any of ",;"): adjacent delimiters yield empties under the KEEP default.
        let haystack = b",a;;b,";
        let kept: Vec<_> = FindSplits::new(haystack, MatcherType::FindFirstOf(b",;")).collect();
        assert_eq!(kept, vec![&b""[..], b"a", &b""[..], b"b", &b""[..]]);
        let nonempty: Vec<_> = FindSplits::new(haystack, MatcherType::FindFirstOf(b",;"))
            .skip_empty()
            .collect();
        assert_eq!(nonempty, vec![b"a", b"b"]);
    }

    #[test]
    fn iter_matches_with_overlaps() {
        let haystack = b"aaaa";
        let needle = b"aa";
        // Default is non-overlapping; `.overlapping()` opts into the compile-time Overlapping policy.
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
        let matcher = MatcherType::Find(b"aa");
        let matches: Vec<_> = FindMatches::new(haystack, matcher).overlapping().collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_find_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::Find(b"aa");
        let matches: Vec<_> = FindMatches::new(haystack, matcher).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_rfind_matches_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::RFind(b"aa");
        let matches: Vec<_> = RFindMatches::new(haystack, matcher).overlapping().collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    fn iter_rfind_matches_non_overlapping() {
        let haystack = b"aaaa";
        let matcher = MatcherType::RFind(b"aa");
        let matches: Vec<_> = RFindMatches::new(haystack, matcher).collect();
        assert_eq!(matches, vec![&b"aa"[..], &b"aa"[..]]);
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_same_length() {
        let mut buffer = b"abcabc".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"ab", b"XY").expect("try_replace_all failed");
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"XYcXYc");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_shrinks() {
        let mut buffer = b"aaaa".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"aa", b"b").expect("try_replace_all failed");
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"bb");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_grows() {
        let mut buffer = b"aba".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"a", b"XYZ").expect("try_replace_all failed");
        assert_eq!(replaced, 2);
        assert_eq!(buffer, b"XYZbXYZ");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_byteset_basic() {
        let mut buffer = b"hello world".to_vec();
        let vowels = sz::Byteset::from("aeiou");
        let replaced = sz::try_replace_all_byteset(&mut buffer, vowels, b"_").expect("try_replace_all_byteset failed");
        assert_eq!(replaced, 3);
        assert_eq!(buffer, b"h_ll_ w_rld");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_byteset_grows() {
        let mut buffer = b"yzz".to_vec();
        let vowels = sz::Byteset::from("y");
        let replaced =
            sz::try_replace_all_byteset(&mut buffer, vowels, b"(y)").expect("try_replace_all_byteset failed");
        assert_eq!(replaced, 1);
        assert_eq!(buffer, b"(y)zz");
    }

    #[test]
    #[cfg(feature = "std")]
    fn replace_all_noop_on_empty_pattern() {
        let mut buffer = b"unchanged".to_vec();
        let replaced = sz::try_replace_all(&mut buffer, b"", b"anything").expect("try_replace_all failed");
        assert_eq!(replaced, 0);
        assert_eq!(buffer, b"unchanged");
    }
}
