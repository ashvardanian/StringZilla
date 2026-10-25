from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import stringzilla as sz
    from typing_extensions import assert_type

    class Index:
        def __index__(self) -> int:
            return 1

    def valid_calls(flag: bool) -> None:
        table = bytes(range(256))
        assert_type(sz.translate(b"abc", table), bytes)
        assert_type(sz.translate(bytearray(b"abc"), table), bytes)
        assert_type(sz.translate(memoryview(b"abc"), memoryview(table), inplace=False), bytes)
        assert_type(sz.translate("abc", {"a": "A"}), str)
        assert_type(sz.translate(bytearray(b"abc"), table, True), None)
        assert_type(sz.translate(memoryview(bytearray(b"abc")), table, inplace=True), None)
        assert_type(sz.translate(b"abc", table, flag, start=Index(), end=None), bytes | None)
        assert_type(sz.translate("abc", table, flag), str | None)
        assert_type(sz.translate(b"abc", table, False, None, -1), bytes)
        assert_type(sz.__version__, str)
        assert_type(sz.__capabilities__, tuple[str, ...])
        assert_type(sz.__capabilities_str__, str)
        assert_type(sz.reset_capabilities(["serial"]), None)
