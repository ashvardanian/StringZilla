"""PyTest + Cppyy test of the `sz_u8_divide` utility function."""

import pytest
import cppyy

cppyy.include("include/stringzilla/stringzilla.h")
cppyy.cppdef(
    """
sz_u32_t sz_u8_divide_as_u32(sz_u8_t number, sz_u8_t divisor) {
    return sz_u8_divide(number, divisor);
}
"""
)


@pytest.mark.parametrize("number", range(0, 256))
@pytest.mark.parametrize("divisor", range(2, 256))
def test_efficient_division(number: int, divisor: int):
    sz_u8_divide = cppyy.gbl.sz_u8_divide_as_u32
    assert (number // divisor) == sz_u8_divide(number, divisor)
