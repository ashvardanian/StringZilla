"""Writable outputs and error boundaries for byte lookup translation."""

import array
import sys

import pytest
import stringzilla as sz


@pytest.mark.parametrize("method", [False, True])
@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255, 256, 257, 4097])
def test_translate_out_bounds(method, length):
    source = (b"ab" * (length // 2)) + (b"a" if length % 2 else b"")
    expected = (b"xy" * (length // 2)) + (b"x" if length % 2 else b"")
    backing = bytearray(b"!" * (length + 6))
    output = memoryview(backing)[3:-3]
    table = {"a": "x", "b": "y"}

    result = sz.Str(source).translate(table, out=output) if method else sz.translate(source, table, out=output)

    assert result is output
    assert bytes(output) == expected
    assert backing[:3] == b"!!!"
    assert backing[-3:] == b"!!!"
    assert source == (b"ab" * (length // 2)) + (b"a" if length % 2 else b"")
    output.release()
    backing.extend(b"!")


def test_translate_out_bytearray_and_slice():
    source = memoryview(b"_abc_")
    output = bytearray(3)
    result = sz.translate(source, {"a": "A", "b": "B", "c": "C"}, start=1, end=-1, out=output)
    assert result is output
    assert output == b"ABC"
    output.extend(b"!")


def test_translate_out_exact_alias():
    image = bytearray(b"abcabc")
    result = sz.translate(image, {"a": "x", "b": "y", "c": "z"}, out=image)
    assert result is image
    assert image == b"xyzxyz"
    image.extend(b"!")


@pytest.mark.parametrize("offset", [-1, 1])
def test_translate_out_rejects_partial_overlap(offset):
    data = bytearray(b"abcde")
    source = memoryview(data)[1:4]
    output_start = 1 + offset
    output_end = 4 + offset
    output = memoryview(data)[output_start:output_end]
    with pytest.raises(ValueError, match="overlap"):
        sz.translate(source, {"a": "x", "b": "y", "c": "z"}, out=output)
    assert data == b"abcde"
    source.release()
    output.release()
    data.extend(b"!")


@pytest.mark.parametrize("output", [b"000", memoryview(b"000"), sz.Str(b"000")])
def test_translate_out_rejects_readonly(output):
    with pytest.raises(BufferError):
        sz.translate(b"abc", {"a": "A"}, out=output)


@pytest.mark.parametrize("size", [0, 2, 4])
def test_translate_out_requires_matching_length(size):
    output = bytearray(size)
    with pytest.raises(ValueError, match="length"):
        sz.translate(b"abc", {"a": "A"}, out=output)
    assert output == bytes(size)
    output.extend(b"!")


def test_translate_out_rejects_nonbyte_elements():
    output = array.array("I", [0, 0, 0])
    with pytest.raises(TypeError, match="bytes"):
        sz.translate(b"abc", {"a": "A"}, out=output)
    output.append(0)


def test_translate_out_rejects_strided_buffer():
    backing = bytearray(b"......")
    output = memoryview(backing)[::2]
    with pytest.raises(BufferError):
        sz.translate(b"abc", {"a": "A"}, out=output)
    assert backing == b"......"


def test_translate_out_rejects_inplace_combination():
    source = bytearray(b"abc")
    output = bytearray(b"...")
    with pytest.raises(ValueError, match="inplace"):
        sz.translate(source, {"a": "A"}, inplace=True, out=output)
    assert source == b"abc"
    assert output == b"..."


@pytest.mark.parametrize("method", [False, True])
def test_translate_rejects_unknown_keyword_before_mutation(method):
    source = bytearray(b"abc")
    table = {"a": "A"}
    with pytest.raises(TypeError, match="unexpected"):
        if method:
            sz.Str(source).translate(table, inplace=True, unexpected=True)
        else:
            sz.translate(source, table, inplace=True, unexpected=True)
    assert source == b"abc"


@pytest.mark.parametrize("method", [False, True])
def test_translate_rejects_duplicate_inplace(method):
    source = bytearray(b"abc")
    with pytest.raises(TypeError, match="inplace"):
        if method:
            sz.Str(source).translate({"a": "A"}, True, inplace=True)
        else:
            sz.translate(source, {"a": "A"}, True, inplace=True)
    assert source == b"abc"


def test_translate_out_all_bytes():
    source = bytes(range(256))
    table = bytes(range(255, -1, -1))
    output = bytearray(256)
    assert sz.translate(source, table, out=output) is output
    assert output == bytes(range(255, -1, -1))


def test_translate_out_numpy_buffer():
    np = pytest.importorskip("numpy")
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    output = np.empty_like(image)
    table = bytes(range(255, -1, -1))
    result = sz.translate(memoryview(image), table, out=output)
    assert result is output
    np.testing.assert_array_equal(output, 255 - image)


def test_translate_none_output_preserves_allocating_api():
    assert sz.translate("abc", {"a": "A"}, out=None) == "Abc"
    assert sz.translate(b"abc", {"a": "A"}, out=None) == b"Abc"


@pytest.mark.parametrize("provided_output", [False, True])
def test_translate_refreshes_source_after_control_callback(provided_output):
    source = bytearray(b"ab")
    output = bytearray(34)

    class ResizeOnTruth:
        def __bool__(self):
            source.extend(b"c" * 32)
            return False

    result = sz.translate(
        source, {"a": "x", "b": "y", "c": "z"}, inplace=ResizeOnTruth(), out=output if provided_output else None
    )
    if provided_output:
        assert result is output
    assert result == b"xy" + b"z" * 32
    assert source == b"ab" + b"c" * 32


@pytest.mark.skipif(sys.version_info < (3, 12), reason="Python buffer protocol requires Python 3.12")
def test_translate_out_rejects_source_released_by_output_callback():
    backing = bytearray(b"abc")
    source = memoryview(backing)
    output = bytearray(b"...")

    class ReleaseSourceOnExport:
        def __buffer__(self, flags):
            source.release()
            backing.extend(b"d" * 4096)
            return memoryview(output)

    with pytest.raises(ValueError, match="released memoryview"):
        sz.translate(source, {"a": "A"}, out=ReleaseSourceOnExport())
    assert backing == b"abc" + b"d" * 4096
    assert output == b"..."
    output.extend(b"!")
