#![forbid(
    unused_unsafe,
    unused_allocation,
    arithmetic_overflow,
    temporary_cstring_as_ptr
)]
#![doc = include_str!("README.md")]

mod bindings {
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// String implementation with StringZilla C bindings under the hood.
pub struct String {
    inner: bindings::sz_string_view_t,
}

impl String {
    pub fn new(input: impl AsRef<str>) -> Self {
        let value = input.as_ref();
        String {
            inner: bindings::sz_string_view_t {
                start: value.as_ptr() as *const i8,
                length: value.as_bytes().len() as bindings::sz_size_t,
            },
        }
    }

    pub fn find(&self, start: usize, end: usize) -> Option<usize> {
        unimplemented!()
    }

    pub fn contains(&self, input: impl AsRef<str>, start: usize, end: usize) -> bool {
        unimplemented!()
    }

    pub fn split_lines(&self, keep_linebreaks: bool, separator: char) -> Vec<String> {
        unimplemented!()
    }

    pub fn split(&self, separator: char, max_split: usize, keep_separator: bool) -> Vec<String> {
        unimplemented!()
    }

    pub fn count_char(&self, input: impl AsRef<str>) -> usize {
        let value = input.as_ref();
        unsafe {
            bindings::sz_count_char(
                self.inner.start,
                self.inner.length,
                value.as_ptr() as *const i8,
            ) as usize
        }
    }
}

impl std::ops::Index<std::ops::Range<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: std::ops::Range<usize>) -> &Self::Output {
        todo!()
    }
}

impl std::ops::Index<std::ops::RangeInclusive<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: std::ops::RangeInclusive<usize>) -> &Self::Output {
        todo!()
    }
}

impl std::ops::Index<std::ops::RangeFrom<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: std::ops::RangeFrom<usize>) -> &Self::Output {
        todo!()
    }
}

impl std::ops::Index<std::ops::RangeTo<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: std::ops::RangeTo<usize>) -> &Self::Output {
        todo!()
    }
}

impl From<std::string::String> for String {
    #[inline]
    fn from(value: std::string::String) -> Self {
        String::new(value)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn count_char() {
        let haystack = super::String::new("abba");
        assert_eq!(haystack.count_char("b"), 2);
    }

    // #[test]
    // fn ops_string_indexing() {
    //     let haystack = super::String::new("abcdef");
    //     assert_eq!(&haystack[3..], "def");
    //     assert_eq!(&haystack[..2], "ab");
    //     // assert_eq!(&haystack[2], "abc");
    //     assert_eq!(&haystack[0..2], "ab");
    //     assert_eq!(&haystack[0..=2], "abc");
    // }
}
