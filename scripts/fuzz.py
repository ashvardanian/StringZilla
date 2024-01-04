# PyTest + Cppyy test of the `sz_edit_distance` utility function.
#
# This file is useful for quick iteration on the underlying C implementation,
# validating the core algorithm on examples produced by the Python test below.
import pytest
import cppyy
import random

from scripts.similarity_baseline import levenshtein

cppyy.include("include/stringzilla/stringzilla.h")
cppyy.cppdef(
    """
static char native_buffer[4096];    
sz_string_view_t native_view{&native_buffer[0], 4096};

sz_ptr_t _sz_malloc(sz_size_t length, void *handle) { return (sz_ptr_t)malloc(length); }
void _sz_free(sz_ptr_t start, sz_size_t length, void *handle) { free(start); }

sz_size_t native_implementation(std::string a, std::string b) {
    sz_memory_allocator_t alloc;
    alloc.allocate = _sz_malloc;
    alloc.free = _sz_free;
    alloc.handle = NULL;
    return sz_edit_distance_serial(a.data(), a.size(), b.data(), b.size(), 200, &alloc);
}
"""
)


@pytest.mark.repeat(5000)
@pytest.mark.parametrize("alphabet", ["abc"])
@pytest.mark.parametrize("length", [10, 50, 200, 300])
def test(alphabet: str, length: int):
    a = "".join(random.choice(alphabet) for _ in range(length))
    b = "".join(random.choice(alphabet) for _ in range(length))
    sz_edit_distance = cppyy.gbl.native_implementation

    pythonic = levenshtein(a, b)
    native = sz_edit_distance(a, b)
    assert pythonic == native
