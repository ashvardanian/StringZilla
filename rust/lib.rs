use std::os::raw::c_void;

// Import the functions from the StringZilla C library.
extern "C" {
    fn sz_find(
        haystack: *mut c_void,
        haystack_length: usize,
        needle: *const c_void,
        needle_length: usize,
    ) -> *mut c_void;
}

// Generic function to find a substring or a subarray
pub fn find<H: AsRef<[u8]>, N: AsRef<[u8]>>(haystack: H, needle: N) -> Option<usize> {
    unsafe {
        let haystack_ref = haystack.as_ref();
        let needle_ref = needle.as_ref();
        let haystack_pointer = haystack_ref.as_ptr() as *mut c_void;
        let haystack_length = haystack_ref.len();
        let needle_pointer = needle_ref.as_ptr() as *const c_void;
        let needle_length = needle_ref.len();
        let result = sz_find(
            haystack_pointer,
            haystack_length,
            needle_pointer,
            needle_length,
        );
        if result.is_null() {
            None
        } else {
            Some(result.offset_from(haystack_pointer) as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::find;

    #[test]
    fn basics() {
        let my_string = String::from("Hello, world!");
        let my_str = "Hello, world!";

        // Use the generic function with a String
        let result_string = find(&my_string, "world");
        assert_eq!(result_string, Some(7));

        // Use the generic function with a &str
        let result_str = find(my_str, "world");
        assert_eq!(result_str, Some(7));
    }
}
