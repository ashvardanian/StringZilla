use std::ffi::CString;

mod bindings {
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub type String = bindings::sz_string_view_t;

impl String {
    pub fn new<'a>(input: impl Into<&'a str>) -> Self {
        let value = input.into();
        bindings::sz_string_view_t {
            start: unsafe { CString::new(value).unwrap_unchecked() }.as_ptr() as *const i8,
            length: value.len() as bindings::sz_size_t,
        }
    }

    pub fn count_char<'a>(&self, input: impl Into<&'a str>) -> usize {
        let value = input.into();
        unsafe {
            bindings::si__count_char(
                self.start,
                self.length,
                std::ffi::CString::new(value).unwrap_unchecked().as_ptr(),
            ) as usize
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn count_char() {
        let haystack = super::String::new("abba");
        assert_eq!(haystack.count_char("b"), 2);
        // assert_eq!(haystack.count_char("a"), 2);
        // assert_eq!(haystack.count_char("ba"), 1);
        // assert_eq!(haystack.count_char("ab"), 1);
    }
}
