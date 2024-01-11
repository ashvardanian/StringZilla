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
            bindings::sz_count_char(
                self.start,
                self.length,
                CString::new(value).unwrap_unchecked().as_ptr(),
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
    }
}
