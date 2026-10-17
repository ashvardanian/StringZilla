"""Initial typing for translation and capability metadata."""

import sys
from collections.abc import Iterable
from typing import Any, Literal, SupportsIndex, TypeAlias, overload

if sys.version_info >= (3, 14):
    _MemoryView: TypeAlias = memoryview[Any]
else:
    _MemoryView: TypeAlias = memoryview

_BinaryInput: TypeAlias = bytes | bytearray | _MemoryView
_StringInput: TypeAlias = str | _BinaryInput
_Table: TypeAlias = _StringInput | dict[str, str]
_Index: TypeAlias = SupportsIndex | None

__version__: str
__capabilities__: tuple[str, ...]
__capabilities_str__: str

def reset_capabilities(names: Iterable[str], /) -> None: ...
@overload
def translate(
    text: str, table: _Table, /, inplace: Literal[False] = False, start: _Index = ..., end: _Index = ...
) -> str: ...
@overload
def translate(
    text: _BinaryInput, table: _Table, /, inplace: Literal[False] = False, start: _Index = ..., end: _Index = ...
) -> bytes: ...
@overload
def translate(
    text: _StringInput, table: _Table, /, inplace: Literal[True], start: _Index = ..., end: _Index = ...
) -> None: ...
@overload
def translate(text: str, table: _Table, /, inplace: bool, start: _Index = ..., end: _Index = ...) -> str | None: ...
@overload
def translate(
    text: _BinaryInput, table: _Table, /, inplace: bool, start: _Index = ..., end: _Index = ...
) -> bytes | None: ...

# Native APIs outside this initial scope remain untyped.
def __getattr__(name: str) -> Any: ...
