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

/// Generic function to find the first occurrence of a substring or a subarray
pub fn find<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needle: N) -> Option<usize> {
    let haystack_ref = haystack.as_ref();
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

/// Generic function to find the last occurrence of a substring or a subarray
pub fn rfind<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needle: N) -> Option<usize> {
    let haystack_ref = haystack.as_ref();
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

/// Generic function to find the first occurrence of a character/element from the second argument
pub fn find_char_from<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needles: N) -> Option<usize> {
    let haystack_ref = haystack.as_ref();
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

/// Generic function to find the last occurrence of a character/element from the second argument
pub fn rfind_char_from<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needles: N) -> Option<usize> {
    let haystack_ref = haystack.as_ref();
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

/// Generic function to find the first occurrence of a character/element from the second argument
pub fn find_char_not_from<H: AsRef<[u8]>, N: AsRef<[u8]>>(
    haystack: H,
    needles: N,
) -> Option<usize> {
    let haystack_ref = haystack.as_ref();
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

/// Generic function to find the last occurrence of a character/element from the second argument
pub fn rfind_char_not_from<H: AsRef<[u8]>, N: AsRef<[u8]>>(
    haystack: H,
    needles: N,
) -> Option<usize> {
    let haystack_ref = haystack.as_ref();
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

#[cfg(test)]
mod tests {
    use crate::find;
    use crate::find_char_from;
    use crate::find_char_not_from;
    use crate::rfind;
    use crate::rfind_char_from;
    use crate::rfind_char_not_from;

    #[test]
    fn basics() {
        let my_string = String::from("Hello, world!");
        let my_str = "Hello, world!";

        // Use the generic function with a String
        assert_eq!(find(&my_string, "world"), Some(7));
        assert_eq!(rfind(&my_string, "world"), Some(7));
        assert_eq!(find_char_from(&my_string, "world"), Some(2));
        assert_eq!(rfind_char_from(&my_string, "world"), Some(11));
        assert_eq!(find_char_not_from(&my_string, "world"), Some(0));
        assert_eq!(rfind_char_not_from(&my_string, "world"), Some(12));

        // Use the generic function with a &str
        assert_eq!(find(my_str, "world"), Some(7));
        assert_eq!(rfind(my_str, "world"), Some(7));
        assert_eq!(find_char_from(my_str, "world"), Some(2));
        assert_eq!(rfind_char_from(my_str, "world"), Some(11));
        assert_eq!(find_char_not_from(my_str, "world"), Some(0));
        assert_eq!(rfind_char_not_from(my_str, "world"), Some(12));
    }
}
