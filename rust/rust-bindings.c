// Functions prefixed with si__ are manually implemented, since
// bindgen does not support static inline functions.

#include "rust-bindings.h"

sz_size_t si__count_char(sz_string_start_t const haystack,
                         sz_size_t const haystack_length,
                         sz_string_start_t const needle) {
    return sz_count_char(haystack, haystack_length, needle);
}
