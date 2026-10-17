import stringzilla as sz


def invalid_calls() -> None:
    table = bytes(range(256))
    sz.translate(b"abc", table, inpalce=True)  # error
    sz.translate(b"abc", table, start="1")  # error
    sz.translate(137, table)  # error
    sz.translate(b"abc", {137: 137})  # error
    sz.translate(b"abc", table=table)  # error
    _text: str = sz.translate(b"abc", table)  # error
    sz.reset_capabilities([137])  # error
