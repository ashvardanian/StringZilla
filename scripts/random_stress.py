# PyTest + Cppyy test of the random string generators and related utility functions
#
import pytest
import cppyy

cppyy.include("include/stringzilla/stringzilla.h")
cppyy.cppdef(
    """
sz_u32_t native_division(sz_u8_t number, sz_u8_t divisor) {
    return sz_u8_divide(number, divisor);
}
"""
)


@pytest.mark.parametrize("number", range(0, 256))
@pytest.mark.parametrize("divisor", range(2, 256))
def test_fast_division(number: int, divisor: int):
    sz_u8_divide = cppyy.gbl.native_division
    assert (number // divisor) == sz_u8_divide(number, divisor)
