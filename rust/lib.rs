#![forbid(
    unused_unsafe,
    unused_allocation,
    arithmetic_overflow,
    temporary_cstring_as_ptr
)]

use std::ffi::CString;

mod bindings {
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub struct String {
    #[allow(dead_code)]
    c_str: CString,
    inner: bindings::sz_string_view_t,
}

impl String {
    pub fn new(input: impl AsRef<str>) -> Self {
        let c_str = CString::new(input.as_ref()).expect("failed to convert to CString");
        String {
            inner: bindings::sz_string_view_t {
                start: c_str.as_ptr() as *const i8,
                length: c_str.as_bytes().len() as bindings::sz_size_t,
            },
            c_str,
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
                CString::new(value).unwrap_unchecked().as_ptr(),
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

impl From<std::string::String> for String {
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
}
