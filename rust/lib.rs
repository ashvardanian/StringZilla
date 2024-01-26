#![cfg_attr(not(test), no_std)]

use core::ffi::c_void;

// Import the functions from the StringZilla C library.
extern "C" {
    fn sz_find(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;

    fn sz_rfind(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;

    fn sz_find_char_from(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;

    fn sz_rfind_char_from(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;

    fn sz_find_char_not_from(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;

    fn sz_rfind_char_not_from(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;
}

pub trait StringZilla<N>
where
    N: AsRef<[u8]>,
{
    /// Generic function to find the first occurrence of a substring or a subarray.
    fn sz_find(&self, needle: N) -> Option<usize>;
    /// Generic function to find the last occurrence of a substring or a subarray.
    fn sz_rfind(&self, needle: N) -> Option<usize>;
    /// Generic function to find the first occurrence of a character/element from the second argument.
    fn sz_find_char_from(&self, needles: N) -> Option<usize>;
    /// Generic function to find the last occurrence of a character/element from the second argument.
    fn sz_rfind_char_from(&self, needles: N) -> Option<usize>;
    /// Generic function to find the first occurrence of a character/element from the second argument.
    fn sz_find_char_not_from(&self, needles: N) -> Option<usize>;
    /// Generic function to find the last occurrence of a character/element from the second argument.
    fn sz_rfind_char_not_from(&self, needles: N) -> Option<usize>;
}

impl<T, N> StringZilla<N> for T
where
    T: AsRef<[u8]>,
    N: AsRef<[u8]>,
{
    fn sz_find(&self, needle: N) -> Option<usize> {
        let haystack_ref = self.as_ref();
        let needle_ref = needle.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needle_pointer = needle_ref.as_ptr() as *const c_void;
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
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needle_pointer = needle_ref.as_ptr() as *const c_void;
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
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as *const c_void;
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
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as *const c_void;
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
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as *const c_void;
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
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needles_pointer = needles_ref.as_ptr() as *const c_void;
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
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::StringZilla;

    #[test]
    fn basics() {
        let my_string = String::from("Hello, world!");
        let my_str = my_string.as_str();
        let my_cow_str = Cow::from(&my_string);

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
}
