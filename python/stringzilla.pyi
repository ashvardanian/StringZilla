"""Search, hash, sort, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU

File: python/stringzilla.pyi
Author: Ash Vardanian
Date: October 27, 2026
"""

import enum
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, overload

__version__: str

type _Buffer = Any
type _StringLike = String | str | bytes | bytearray | memoryview

class Capability(enum.IntFlag):
    SERIAL = ...
    WESTMERE = ...
    GOLDMONT = ...
    HASWELL = ...
    SKYLAKE = ...
    ICELAKE = ...
    NEON = ...
    NEONAES = ...
    NEONSHA = ...
    SVE = ...
    SVE2 = ...
    SVE2AES = ...
    RVV = ...
    RVVCRYPTO = ...
    V128 = ...
    V128RELAXED = ...
    LOONGSONASX = ...
    POWERVSX = ...

class AuthenticationError(ValueError):
    """Raised when a Galois/counter mode tag does not authenticate the bytes it accompanies.

    Derives from `ValueError`, so code guarding only against malformed input still stops a
    forgery, and the plaintext is never returned alongside it.

    Example:
      >>> issubclass(sz.AuthenticationError, ValueError)
      True
    """

class Aes256CtrKey:
    """Aes256CtrKey(secret, *, capabilities=None)

    AES-256 in counter mode, abbreviated CTR: unauthenticated, seekable, and its own inverse.
    The 32-byte secret is expanded once into a round-key schedule the object holds and wipes
    when it is collected.

    Args:
      secret (str | bytes): Exactly 32 secret bytes.
      capabilities (Capability, optional): Capabilities this key and its calls run,
        defaulting to the CPU's enabled ones.
    Raises:
      ValueError: If the secret is not exactly 32 bytes long.
    Note:
      A nonce must never repeat under one secret; reuse exposes the exclusive-or of two texts.

    Example:
      >>> key = sz.Aes256CtrKey(bytes(32))
      >>> key.xor(key.xor(b'hello', bytes(12)), bytes(12))
      b'hello'
    """

    def __init__(self, secret: str | bytes, *, capabilities: Capability | None = None) -> None: ...
    def xor(self, text: str | bytes, nonce: str | bytes, offset: int = 0) -> bytes:
        """xor(text, nonce, offset=0) -> bytes

        Exclusive-or `text` against the keystream, encrypting or decrypting with the same call.

        Args:
          text (str | bytes): The bytes to transform.
          nonce (str | bytes): Exactly 12 nonce bytes, never repeated under one secret.
          offset (int, optional): Absolute byte position of `text` in the stream. Defaults to 0.
        Returns:
          bytes: As many bytes as `text` holds.
        Raises:
          ValueError: If the nonce is not exactly 12 bytes long.

        Example:
          >>> key, nonce = sz.Aes256CtrKey(bytes(32)), bytes(12)
          >>> whole = key.xor(bytes(64), nonce)
          >>> key.xor(bytes(16), nonce, offset=32) == whole[32:48]
          True
        """

class Aes256GcmKey:
    """Aes256GcmKey(secret, *, capabilities=None)

    AES-256 in Galois/counter mode, abbreviated GCM: authenticated, and therefore not seekable.
    The 32-byte secret is expanded once into a schedule and the Galois hash powers, also called
    GHASH, and is wiped on collection.

    Args:
      secret (str | bytes): Exactly 32 secret bytes.
      capabilities (Capability, optional): Capabilities this key and its calls run,
        defaulting to the CPU's enabled ones.
    Raises:
      ValueError: If the secret is not exactly 32 bytes long.
    Note:
      Reusing a nonce here exposes the hash subkey, after which every message can be forged.

    Example:
      >>> key = sz.Aes256GcmKey(bytes(32))
      >>> ciphertext, tag = key.encrypt(b'hello', bytes(12))
      >>> key.decrypt(ciphertext, bytes(12), tag)
      b'hello'
    """

    def __init__(self, secret: str | bytes, *, capabilities: Capability | None = None) -> None: ...
    def encrypt(
        self, text: str | bytes, nonce: str | bytes, associated: str | bytes | None = None
    ) -> tuple[bytes, bytes]:
        """encrypt(text, nonce, associated=None) -> (bytes, bytes)

        Encrypt `text` and authenticate both it and `associated`, returning ciphertext and tag.

        Args:
          text (str | bytes): The plaintext.
          nonce (str | bytes): Exactly 12 nonce bytes, never repeated under one secret.
          associated (str | bytes, optional): Bytes authenticated but not encrypted.
        Returns:
          tuple: The ciphertext, as long as `text`, and the 16-byte tag.
        Raises:
          ValueError: If the nonce is not exactly 12 bytes long.

        Example:
          >>> ciphertext, tag = sz.Aes256GcmKey(bytes(32)).encrypt(b'hi', bytes(12), b'header')
          >>> len(ciphertext), len(tag)
          (2, 16)
        """
    def decrypt(
        self, text: str | bytes, nonce: str | bytes, tag: str | bytes, associated: str | bytes | None = None
    ) -> bytes:
        """decrypt(text, nonce, tag, associated=None) -> bytes

        Check `tag` against the ciphertext and associated data, then return the plaintext.

        Args:
          text (str | bytes): The ciphertext.
          nonce (str | bytes): The 12 nonce bytes used to encrypt.
          tag (str | bytes): The 16-byte tag to check.
          associated (str | bytes, optional): The same bytes passed to `encrypt`.
        Returns:
          bytes: The plaintext, as long as `text`.
        Raises:
          AuthenticationError: If the tag does not match, in which case nothing is returned.

        Example:
          >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
          >>> ciphertext, tag = key.encrypt(b'hello', nonce)
          >>> key.decrypt(ciphertext, nonce, tag)
          b'hello'
        """

class Aes256GcmEncryptor:
    """Aes256GcmEncryptor(key, nonce, *, capabilities=None)

    The same Galois/counter mode construction as `Aes256GcmKey`, sealing a message in chunks.
    Chunk boundaries are invisible to the result. Opening a message uses `Aes256GcmDecryptor`,
    a separate type because the tag absorbs ciphertext in both directions, which is the output
    when sealing and the input when opening. Copies the key, so it cannot outlive it.

    Args:
      key (Aes256GcmKey): The expanded key, copied into the encryptor.
      nonce (str | bytes): Exactly 12 nonce bytes, never repeated under one secret.
      capabilities (Capability, optional): Capabilities this encryptor and its calls run,
        defaulting to the CPU's enabled ones.

    Example:
      >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
      >>> sealed = sz.Aes256GcmEncryptor(key, nonce)
      >>> ciphertext, tag = key.encrypt(b'hello', nonce)
      >>> sealed.encrypt(b'hel') + sealed.encrypt(b'lo') == ciphertext
      True
    """

    def __init__(self, key: Aes256GcmKey, nonce: str | bytes, *, capabilities: Capability | None = None) -> None: ...
    def associate(self, data: str | bytes) -> Aes256GcmEncryptor:
        """associate(data) -> Aes256GcmEncryptor

        Absorb bytes that are authenticated but not encrypted, and return self for chaining.
        All associated data must be absorbed before the first message chunk.

        Args:
          data (str | bytes): Associated bytes, such as a routing header.
        Returns:
          Aes256GcmEncryptor: The same object, enabling `sealed.associate(a).associate(b)`.
        Example:
          >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
          >>> sealed = sz.Aes256GcmEncryptor(key, nonce).associate(b'head')
          >>> ciphertext, tag = key.encrypt(b'body', nonce, b'head')
          >>> sealed.encrypt(b'body') == ciphertext
          True
        """
    def encrypt(self, data: str | bytes) -> bytes:
        """encrypt(data) -> bytes

        Encrypt one chunk and absorb its ciphertext into the running tag.

        Args:
          data (str | bytes): The plaintext chunk.
        Returns:
          bytes: The ciphertext chunk, as long as `data`.
        Example:
          >>> sealed = sz.Aes256GcmEncryptor(sz.Aes256GcmKey(bytes(32)), bytes(12))
          >>> len(sealed.encrypt(b'chunk'))
          5
        """
    def digest(self) -> bytes:
        """digest() -> bytes

        Return the 16-byte tag over everything absorbed so far, leaving the seal appendable.

        Returns:
          bytes: The 16-byte authentication tag.
        Example:
          >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
          >>> sealed = sz.Aes256GcmEncryptor(key, nonce)
          >>> _ = sealed.encrypt(b'hello')
          >>> ciphertext, tag = key.encrypt(b'hello', nonce)
          >>> sealed.digest() == tag
          True
        """

class Aes256GcmDecryptor:
    """Aes256GcmDecryptor(key, nonce, *, capabilities=None)

    The same Galois/counter mode construction as `Aes256GcmKey`, opening a message in chunks
    and checking its tag at the end. Chunk boundaries are invisible to the result. Sealing a
    message uses `Aes256GcmEncryptor`. Copies the key, so it cannot outlive it.

    Args:
      key (Aes256GcmKey): The expanded key, copied into the decryptor.
      nonce (str | bytes): The 12 nonce bytes the message was sealed under.
      capabilities (Capability, optional): Capabilities this decryptor and its calls run,
        defaulting to the CPU's enabled ones.

    Example:
      >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
      >>> ciphertext, tag = key.encrypt(b'hello', nonce)
      >>> sz.Aes256GcmDecryptor(key, nonce).decrypt_unverified(ciphertext)
      b'hello'
    """

    def __init__(self, key: Aes256GcmKey, nonce: str | bytes, *, capabilities: Capability | None = None) -> None: ...
    def associate(self, data: str | bytes) -> Aes256GcmDecryptor:
        """associate(data) -> Aes256GcmDecryptor

        Absorb bytes that are authenticated but not encrypted, and return self for chaining.
        All associated data must be absorbed before the first message chunk.

        Args:
          data (str | bytes): The same associated bytes the sender authenticated.
        Returns:
          Aes256GcmDecryptor: The same object, enabling `opened.associate(a).associate(b)`.
        Example:
          >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
          >>> ciphertext, tag = key.encrypt(b'body', nonce, b'head')
          >>> opened = sz.Aes256GcmDecryptor(key, nonce).associate(b'head')
          >>> opened.decrypt_unverified(ciphertext)
          b'body'
        """
    def decrypt_unverified(self, data: str | bytes) -> bytes:
        """decrypt_unverified(data) -> bytes

        Decrypt one chunk and absorb its ciphertext into the running tag. The bytes returned are
        not yet authenticated: nothing has checked that this ciphertext is genuine, so buffer them
        until `verify()` succeeds rather than acting on them.

        Args:
          data (str | bytes): The ciphertext chunk.
        Returns:
          bytes: The unverified plaintext chunk, as long as `data`.
        Example:
          >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
          >>> ciphertext, tag = key.encrypt(b'hello', nonce)
          >>> opened = sz.Aes256GcmDecryptor(key, nonce)
          >>> opened.decrypt_unverified(ciphertext)
          b'hello'
        """
    def verify(self, tag: str | bytes) -> None:
        """verify(tag) -> None

        Check `tag` against everything absorbed so far, in time independent of where it differs.

        Args:
          tag (str | bytes): The 16-byte tag to check.
        Raises:
          AuthenticationError: If the tag does not match.
          ValueError: If the tag is not exactly 16 bytes long.
        Example:
          >>> key, nonce = sz.Aes256GcmKey(bytes(32)), bytes(12)
          >>> ciphertext, tag = key.encrypt(b'hello', nonce)
          >>> opened = sz.Aes256GcmDecryptor(key, nonce)
          >>> _ = opened.decrypt_unverified(ciphertext)
          >>> opened.verify(tag) is None
          True
        """

class File:
    r"""File(path, mode='r')

    Memory-mapped file class that exposes the memory range for low-level access.
    Provides efficient read-only access to file contents without loading into memory.

    Args:
      path (str): Path to the file to memory-map.
      mode (str): File access mode, defaulting to 'r' for read-only.

    Example:
      >>> import os, tempfile, pathlib
      >>> path = os.path.join(tempfile.mkdtemp(), 'log.txt')
      >>> _ = pathlib.Path(path).write_text('error: disk full\nok\nerror: timeout\n')
      >>> # Memory-map and scan in place - no copy into RAM, SIMD-accelerated:
      >>> sz.String(sz.File(path)).count('error')
      2
    """

    def __init__(self, path: str, mode: str = "r") -> None: ...

def equal(first: _StringLike, second: _StringLike, /, *, capabilities: Capability | None = None) -> bool:
    """Check if two strings are equal.

    This function can be called as a method on a String object or as a standalone function.
    Args:
      first (String or str or bytes): The first string object.
      second (String or str or bytes): The second string object.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bool: True if the strings are equal, False otherwise.
    Note:
      Comparison is byte-level: a `str` and a `bytes` with identical bytes compare equal.
    Raises:
      TypeError: If the argument is not string-like or incorrect number of arguments is provided.

    Example:
      >>> sz.equal('abc', 'abc')
      True
    """

def contains(
    text: _StringLike,
    substring: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> bool:
    """Check if a string contains a substring.

    Args:
      text (String or str or bytes): The string object.
      substring (str): The substring to search for.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bool: True if the substring is found, False otherwise.

    Example:
      >>> sz.String('hello').contains('ell')
      True
    """

def find(
    text: _StringLike,
    substring: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the first occurrence of a substring.

    Args:
      text (String or str or bytes): The string object.
      substring (str): The substring to find.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The index of the first occurrence, or -1 if not found.

    Example:
      >>> sz.String('hello').find('l')
      2
    """

def index(
    text: _StringLike,
    substring: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the first occurrence of a substring or raise an error if not found.

    Args:
      text (String or str or bytes): The string object.
      substring (str): The substring to find.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The index of the first occurrence.
    Raises:
      ValueError: If the substring is not found.

    Example:
      >>> sz.String('hello').index('l')
      2
    """

def rfind(
    text: _StringLike,
    substring: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the last occurrence of a substring.

    Args:
      text (String or str or bytes): The string object.
      substring (str): The substring to find.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The index of the last occurrence, or -1 if not found.

    Example:
      >>> sz.String('hello').rfind('l')
      3
    """

def rindex(
    text: _StringLike,
    substring: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the last occurrence of a substring or raise an error if not found.

    Args:
      text (String or str or bytes): The string object.
      substring (str): The substring to find.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The index of the last occurrence.
    Raises:
      ValueError: If the substring is not found.

    Example:
      >>> sz.String('hello').rindex('l')
      3
    """

def partition(
    text: _StringLike, separator: _StringLike, *, capabilities: Capability | None = None
) -> tuple[String, String, String]:
    """Split the string into a 3-tuple around the first occurrence of a separator.

    Args:
      text (String or str or bytes): The string object.
      separator (str): The separator to partition by.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      tuple: A 3-tuple (head, separator, tail), or `(self, '', '')` if the separator is not found.

    Example:
      >>> tuple(map(str, sz.String('a=b=c').partition('=')))
      ('a', '=', 'b=c')
    """

def rpartition(
    text: _StringLike, separator: _StringLike, *, capabilities: Capability | None = None
) -> tuple[String, String, String]:
    """Split the string into a 3-tuple around the last occurrence of a separator.

    Args:
      text (String or str or bytes): The string object.
      separator (str): The separator to partition by.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      tuple: A 3-tuple (head, separator, tail). If the separator is not found, returns ('', '', self).

    Example:
      >>> tuple(map(str, sz.String('a=b=c').rpartition('=')))
      ('a=b', '=', 'c')
    """

def count(
    text: _StringLike,
    substring: _StringLike,
    start: int = 0,
    end: int | None = None,
    allowoverlap: bool = False,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Count the occurrences of a substring.

    Args:
      text (String or str or bytes): The string object.
      substring (str): The substring to count.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      allowoverlap (bool, optional): Count overlapping occurrences, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The number of occurrences of the substring.

    Example:
      >>> sz.String('banana').count('a')
      3
    """

def startswith(
    text: _StringLike,
    prefix: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> bool:
    """Check if a string starts with a given prefix.

    Args:
      text (String or str or bytes): The string object.
      prefix (str): The prefix to check.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bool: True if the string starts with the prefix, False otherwise.

    Example:
      >>> sz.String('hello').startswith('he')
      True
    """

def endswith(
    text: _StringLike,
    suffix: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> bool:
    """Check if a string ends with a given suffix.

    Args:
      text (String or str or bytes): The string object.
      suffix (str): The suffix to check.
      start (int, optional): The starting index, defaulting to 0.
      end (int, optional): The ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bool: True if the string ends with the suffix, False otherwise.

    Example:
      >>> sz.String('hello').endswith('lo')
      True
    """

def find_first_of(
    text: _StringLike,
    chars: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the index of the first occurrence of any character from another string.

    Args:
      text (String or str or bytes): The string object.
      chars (str): A string containing characters to search for.
      start (int, optional): Starting index, defaulting to 0.
      end (int, optional): Ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: Index of the first matching character, or -1 if none found.

    Example:
      >>> sz.String('hello').find_first_of('aeiou')
      1
    """

def find_first_not_of(
    text: _StringLike,
    chars: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the index of the first character not in another string.

    Args:
      text (String or str or bytes): The string object.
      chars (str): A string containing characters to exclude.
      start (int, optional): Starting index, defaulting to 0.
      end (int, optional): Ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: Index of the first non-matching character, or -1 if all match.

    Example:
      >>> sz.String('hello').find_first_not_of('he')
      2
    """

def find_last_of(
    text: _StringLike,
    chars: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the index of the last occurrence of any character from another string.

    Args:
      text (String or str or bytes): The string object.
      chars (str): A string containing characters to search for.
      start (int, optional): Starting index, defaulting to 0.
      end (int, optional): Ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: Index of the last matching character, or -1 if none found.

    Example:
      >>> sz.String('hello').find_last_of('aeiou')
      4
    """

def find_last_not_of(
    text: _StringLike,
    chars: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Find the index of the last character not in another string.

    Args:
      text (String or str or bytes): The string object.
      chars (str): A string containing characters to exclude.
      start (int, optional): Starting index, defaulting to 0.
      end (int, optional): Ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: Index of the last non-matching character, or -1 if all match.

    Example:
      >>> sz.String('hello').find_last_not_of('lo')
      1
    """

def count_byteset(
    text: _StringLike,
    chars: _StringLike,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> int:
    """Count the occurrences of any character from a set of characters.

    Args:
      text (String or str or bytes): The string object.
      chars (str): A string containing characters to count.
      start (int, optional): Starting index, defaulting to 0.
      end (int, optional): Ending index, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The number of occurrences of any character from the set.

    Example:
      >>> sz.String('hello world').count_byteset('lo')
      5
    """

def split(
    text: _StringLike,
    separator: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> Strings:
    """Split a string by a separator, like `str.split`.

    Args:
      text (String or str or bytes): The string object.
      separator (str): The separator to split by; an empty one leaves the text whole.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include the separator in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      Strings: The parts, copied onto a tape of their own.

    Example:
      >>> list(map(str, sz.String('a,b,c').split(',')))
      ['a', 'b', 'c']
    """

def rsplit(
    text: _StringLike,
    separator: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> Strings:
    """Split a string by a separator starting from the end, like `str.rsplit`.

    Args:
      text (String or str or bytes): The string object.
      separator (str): The separator to split by; an empty one leaves the text whole.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include the separator in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      Strings: The parts, copied onto a tape of their own.

    Example:
      >>> list(map(str, sz.String('a,b,c').rsplit(',')))
      ['a', 'b', 'c']
    """

def split_byteset(
    text: _StringLike,
    separators: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> Strings:
    """Split a string by a set of character separators.

    Args:
      text (String or str or bytes): The string object.
      separators (str): A string containing separator characters.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include separators in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      Strings: The parts, copied onto a tape of their own.

    Example:
      >>> list(map(str, sz.String('a,b;c').split_byteset(',;')))
      ['a', 'b', 'c']
    """

def rsplit_byteset(
    text: _StringLike,
    separators: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> Strings:
    """Split a string by a set of character separators in reverse order.

    Args:
      text (String or str or bytes): The string object.
      separators (str): A string containing separator characters.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include separators in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      Strings: The parts, copied onto a tape of their own.

    Example:
      >>> list(map(str, sz.String('a,b;c').rsplit_byteset(',;')))
      ['a', 'b', 'c']
    """

def split_iter(
    text: _StringLike,
    separator: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> FindSplits:
    """Create an iterator for splitting a string by a separator.

    Args:
      text (String or str or bytes): The string object.
      separator (str): The separator to split by; an empty one leaves the text whole.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include separator in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding split substrings.

    Example:
      >>> # Stream parts lazily instead of materializing a list (that is what split() is for):
      >>> sum(1 for _ in sz.String('a,b,c').split_iter(','))
      3
    """

def rsplit_iter(
    text: _StringLike,
    separator: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> FindSplits:
    """Create an iterator for splitting a string by a separator in reverse order.

    Args:
      text (String or str or bytes): The string object.
      separator (str): The separator to split by; an empty one leaves the text whole.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include separator in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding split substrings in reverse.

    Example:
      >>> # Iterates from the end; the first yielded part is the last field:
      >>> str(next(iter(sz.String('a/b/c').rsplit_iter('/'))))
      'c'
    """

def split_byteset_iter(
    text: _StringLike,
    separators: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> FindSplits:
    """Create an iterator for splitting a string by a set of character separators.

    Args:
      text (String or str or bytes): The string object.
      separators (str): A string containing separator characters.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include separators in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding split substrings.

    Example:
      >>> # Splits on ANY byte in the set, streamed lazily:
      >>> str(next(iter(sz.String('a,b;c').split_byteset_iter(',;'))))
      'a'
    """

def rsplit_byteset_iter(
    text: _StringLike,
    separators: _StringLike,
    maxsplit: int = -1,
    keepseparator: bool = False,
    skip_empty: bool = False,
    *,
    capabilities: Capability | None = None,
) -> FindSplits:
    """Create an iterator for splitting a string by a set of character separators in reverse order.

    Args:
      text (String or str or bytes): The string object.
      separators (str): A string containing separator characters.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      keepseparator (bool, optional): Include separators in results, defaulting to False.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding split substrings in reverse.

    Example:
      >>> # Reverse byteset split; first yielded part is the last field:
      >>> str(next(iter(sz.String('a,b;c').rsplit_byteset_iter(',;'))))
      'c'
    """

def splitlines(
    text: _StringLike, keeplinebreaks: bool = False, maxsplit: int = -1, *, capabilities: Capability | None = None
) -> Strings:
    r"""Split a string by line breaks, like `str.splitlines`.

    Args:
      text (String or str or bytes): The string object.
      keeplinebreaks (bool, optional): Include line breaks in the results, defaulting to False.
      maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      Strings: The lines, copied onto a tape of their own.

    Example:
      >>> list(map(str, sz.String('a\nb\nc').splitlines()))
      ['a', 'b', 'c']
    """

def lstrip(text: _StringLike, chars: _StringLike | None = None, *, capabilities: Capability | None = None) -> String:
    """Remove leading characters from a string.

    Args:
      text (String or str or bytes): The string object.
      chars (str, optional): Characters to remove, defaulting to whitespace.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      String: A new string with leading characters removed.

    Example:
      >>> sz.String('  hi').lstrip() == 'hi'
      True
    """

def rstrip(text: _StringLike, chars: _StringLike | None = None, *, capabilities: Capability | None = None) -> String:
    """Remove trailing characters from a string.

    Args:
      text (String or str or bytes): The string object.
      chars (str, optional): Characters to remove, defaulting to whitespace.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      String: A new string with trailing characters removed.

    Example:
      >>> sz.String('hi  ').rstrip() == 'hi'
      True
    """

def strip(text: _StringLike, chars: _StringLike | None = None, *, capabilities: Capability | None = None) -> String:
    """Remove leading and trailing characters from a string.

    Args:
      text (String or str or bytes): The string object.
      chars (str, optional): Characters to remove, defaulting to whitespace.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      String: A new string with leading and trailing characters removed.

    Example:
      >>> sz.String('  hi  ').strip() == 'hi'
      True
    """

def hash(text: _StringLike, seed: int = 0, /, *, capabilities: Capability | None = None) -> int:
    """Compute the hash value of the string.

    This function can be called as a method on a String object or as a standalone function.
    Args:
      text (String or str or bytes): The string to hash, positional-only when standalone.
      seed (int, optional): The seed value for hashing. Defaults to 0. Can be positional or keyword.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The hash value as an unsigned 64-bit integer. This differs from Python's
           built-in `hash()` which returns a `Py_hash_t` and may be platform-dependent.
    Raises:
      TypeError: If the argument is not string-like or incorrect number of arguments is provided.
    Signature:
      >>> def hash(text, seed=0, /, *, capabilities=None) -> int: ...
    """

def hash_multiseed(
    text: _StringLike, seeds: _Buffer, /, out: _Buffer | None = None, *, capabilities: Capability | None = None
) -> tuple[int, ...] | None:
    """Hash one string under many seeds at once.

    Equivalent to `tuple(hash(text, s) for s in seeds)`, but normalizes the input into AES
    blocks once and replays the cheap per-seed rounds - markedly faster for short strings under
    many seeds (feature hashing, Count-Min sketches, Bloom/cuckoo filters, MinHash/LSH).

    Args:
      text (String or str or bytes): The string to hash, positional-only when standalone.
      seeds (buffer of uint64): Contiguous 64-bit seeds, e.g. numpy.uint64 array or array('Q', ...).
           Plain int lists are not accepted - wrap them in array('Q', seeds) or numpy.
      out (writable uint64 buffer, optional): A contiguous buffer of at least len(seeds) elements.
           If given, it is filled in place and None returned; otherwise a tuple of ints is returned.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      tuple[int, ...] | None: Tuple of 64-bit hashes, or None when writing into `out`.

    Example:
      >>> def hash_multiseed(text, seeds, /, out=None, *, capabilities=None) -> tuple | None: ...
    """

def fill_random(
    buffer: _Buffer,
    nonce: int = 0,
    alphabet: str | bytes | None = None,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> None:
    """Fill a string-like buffer in place with pseudo-random bytes.

    Args:
      buffer (String or bytes-like): Writable, contiguous byte buffer (e.g., memoryview/bytearray).
      nonce (int, optional): Seed ensuring reproducible output for the same inputs, defaulting to 0.
      alphabet (str or bytes, optional): If provided, remaps random bytes to its characters.
      start (int, optional): Starting index, defaulting to 0.
      end (int, optional): Ending index, defaulting to len(buffer).
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      None: Mutates the buffer slice in place.

    Example:
      >>> buf = bytearray(16)
      >>> sz.fill_random(buf)
      >>> len(buf)
      16
    """

def bytesum(text: _StringLike, *, capabilities: Capability | None = None) -> int:
    """Compute the checksum of individual byte values in a string.

    This function can be called as a method on a String object or as a standalone function.
    Args:
      text (String or str or bytes): The string to hash.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: The checksum of individual byte values in a string.
    Raises:
      TypeError: If the argument is not string-like or incorrect number of arguments is provided.

    Example:
      >>> sz.bytesum('abc')
      294
    """

def sha256(text: _StringLike, *, capabilities: Capability | None = None) -> bytes:
    """Compute SHA256 cryptographic hash of the input data.

    This function can be called as a method on a String object or as a standalone function.
    Args:
      text (String or str or bytes): The input data to hash.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bytes: The 32-byte (256-bit) SHA256 digest.
    Raises:
      TypeError: If the argument is not string-like or incorrect number of arguments is provided.

    Example:
      >>> digest = sz.String('abc').sha256().hex()
      >>> digest[:8]
      'ba7816bf'
    """

def hmac_sha256(
    key: str | bytes,
    message: str | bytes | Strings | Sequence[str | bytes],
    out: _Buffer | None = None,
    *,
    capabilities: Capability | None = None,
) -> bytes | list[bytes] | _Buffer:
    """hmac_sha256(key, message, out=None, *, capabilities=None) -> bytes | list[bytes] | buffer

    Compute the HMAC-SHA256 authentication code of one message, or of many under one key.

    Authenticating one message is a serial dependency chain no instruction set can speed up, but
    independent messages compress in parallel lanes: sixteen at a time on AVX-512, eight on
    AVX2. Pass a Strings of messages - a batch of tokens or requests sharing a secret - and both
    the message pass and the outer wrap run lane-parallel.

    Args:
      key (str | bytes): The secret key.
      message (str | bytes | Strings | sequence): One message, or many to authenticate together.
      out (buffer, optional): Writable, C-contiguous buffer of bytes holding at least 32 per
           message - a `(len(message), 32)` numpy.uint8 matrix receives one tag per row with
           zero allocation. Defaults to None.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bytes: The 32-byte tag, for a single string-like message.
      list[bytes]: One tag per message, in order, for a collection.
      buffer: `out` itself, when an output buffer is given.
    Example:
      >>> len(sz.hmac_sha256(b'key', b'message'))
      32
      >>> sz.hmac_sha256(b'key', sz.Strings(['a', 'b'])) == [
      ...     sz.hmac_sha256(b'key', b'a'), sz.hmac_sha256(b'key', b'b')]
      True
    """

def random(
    length: int, *, nonce: int = 0, alphabet: str | bytes | None = None, capabilities: Capability | None = None
) -> bytes:
    """random(length, *, nonce=0, alphabet=None, capabilities=None) -> bytes

    Generate a new random byte string, optionally remapped to a given alphabet.
    If alphabet is provided, each byte is mapped to alphabet[b % len(alphabet)].
    Capabilities default to the CPU's enabled ones.

    Example:
      >>> len(sz.random(16))
      16
    """

def decode(text: _StringLike, encoding: str = "utf-8", errors: str = "strict") -> str:
    """Decode the bytes into a Unicode string with a given encoding.

    Args:
      text (String or str or bytes): The string object.
      encoding (str, optional): The encoding to use, defaulting to 'utf-8'.
      errors (str, optional): Error handling scheme, defaulting to 'strict'.
    Returns:
      str: The decoded Unicode string.
    Raises:
      UnicodeDecodeError: If decoding fails.

    Example:
      >>> sz.String('abc').decode() == 'abc'
      True
    """

def write_to(text: _StringLike, filename: str) -> None:
    """Write the string to a file.

    Args:
      text (String or str or bytes): The string object.
      filename (str): The file path to write to.
    Returns:
      None.

    Example:
      >>> import os, tempfile
      >>> path = os.path.join(tempfile.gettempdir(), 'sz_demo.bin')
      >>> sz.String('payload').write_to(path)
    """

def offset_within(text: _StringLike, larger: String) -> int:
    """Return the raw byte offset of this StringZilla string within a larger StringZilla string.

    Args:
      text (String or str or bytes): The substring.
      larger (String): The larger string to search within.
    Returns:
      int: The byte offset where 'self' is found within 'larger', or -1 if not found.

    Example:
      >>> big = sz.String('hello world')
      >>> big[6:].offset_within(big)
      6
    """

def translate(
    text: _StringLike,
    table: str | bytes | dict[str, str],
    inplace: bool = False,
    start: int = 0,
    end: int | None = None,
    *,
    capabilities: Capability | None = None,
) -> None | str | bytes:
    """Perform transformation of a string using a look-up table.

    Args:
      text (String or str or bytes): The string object.
      table (str or dict): A 256-byte string, or a dict mapping characters below U+0100 to such.
      inplace (bool, optional): If True, the string is modified in place, defaulting to False.

      start (int, optional): The starting index for translation, defaulting to 0.
      end (int, optional): The ending index for translation, defaulting to the string length.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      Union[None, str, bytes]: If inplace is False, a translated copy of the [start, end) slice is
        returned, otherwise None.
    Raises:
      ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.
      UnicodeDecodeError: If a str input translates into malformed UTF-8.
      TypeError: If the table is not a string or dictionary.

    Example:
      >>> sz.String('abc').translate({'a': 'A'}) == b'Abc'
      True
    """

def to_json(
    columns: Mapping[str, Strings | Sequence[str]], ensure_ascii: bool = True, *, capabilities: Capability | None = None
) -> bytes:
    r"""to_json(columns, ensure_ascii=True, *, capabilities=None) -> bytes

    Dump columns of strings as a compact JSON array of records, one object per row, the bytes of
    `json.dumps([dict(zip(columns, row)) for row in zip(*columns.values())], separators=(',', ':'))`.
    Each column is escaped in one pass, and the rows are written straight into the result.

    Args:
      columns (Mapping[str, Strings | Sequence[str]]): Columns of equal length, keyed by name.
      ensure_ascii (bool, optional): Write every non-ASCII codepoint as `\uXXXX`, defaulting to True.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bytes: The UTF-8 JSON text.
    Raises:
      ValueError: If the columns differ in length.
    Example:
      >>> sz.to_json({'name': sz.Strings(['Ada', 'Linus']), 'city': ['London', 'Helsinki']})
      b'[{"name":"Ada","city":"London"},{"name":"Linus","city":"Helsinki"}]'
    """

def to_csv(
    columns: Mapping[str, Strings | Sequence[str]], lineterminator: str = "\n", *, capabilities: Capability | None = None
) -> bytes:
    r"""to_csv(columns, lineterminator='\n', *, capabilities=None) -> bytes

    Dump columns of strings as CSV with a header row of their names, the bytes `csv.writer` writes
    with its default dialect, quoting a field only around a comma, a quote, CR or LF. Each column
    is escaped in one pass, and the rows are written straight into the result.

    Args:
      columns (Mapping[str, Strings | Sequence[str]]): Columns of equal length, keyed by name.
      lineterminator (str, optional): '\n' or '\r\n' after every row, defaulting to '\n'.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      bytes: The UTF-8 CSV text.
    Raises:
      ValueError: If the columns differ in length, or the terminator is another string.
    Example:
      >>> sz.to_csv({'name': ['Ada', 'Grace "Amazing" Hopper'], 'city': ['London', 'New York']})
      b'name,city\nAda,London\n"Grace ""Amazing"" Hopper",New York\n'
    """

def cpu_capabilities_detected() -> Capability:
    """Get the capabilities this CPU can execute.

    Detected from CPUID or HWCAP. Says nothing about whether the kernels were compiled in, which
    `cpu_capabilities_compiled` reports.

    Signature:
        >>> def cpu_capabilities_detected() -> sz.Capability: ...
    """

def cpu_capabilities_compiled() -> Capability:
    """Get the CPU capabilities whose kernels were compiled into this binary.

    Decided at build time. Independent of the hardware: a binary built with a broken probe toolchain
    still detects this machine's full set while containing no SIMD kernels at all, which is what
    makes a scalar build hard to spot.

    Signature:
        >>> def cpu_capabilities_compiled() -> sz.Capability: ...
    """

def cpu_capabilities_enabled() -> Capability:
    """Get the CPU capabilities kernels run with, unless a call passes `capabilities=`.

    It is `cpu_capabilities_detected() & cpu_capabilities_compiled()`, and always has SERIAL.

    Signature:
        >>> def cpu_capabilities_enabled() -> sz.Capability: ...
    """

def cpu_configure_thread(capabilities: Capability, /) -> None:
    """Prepare the calling thread for kernels of `capabilities`.

    Call it on every thread that runs kernels, before the first of them.

    Args:
        capabilities (Capability): The mask kernels on this thread run with.

    Signature:
        >>> def cpu_configure_thread(capabilities, /) -> None: ...
    """

def cuda_count_devices() -> int:
    """Count the devices of one GPU vendor this process sees: `cuda_count_devices`, and the `rocm_`
    and `metal_` ones alike.

    Raises:
        RuntimeError: No device of that vendor answers, or this build lacks its kernels.

    Signature:
        >>> def cuda_count_devices() -> int: ...
    """

def rocm_count_devices() -> int:
    """Count the devices of one GPU vendor this process sees: `cuda_count_devices`, and the `rocm_`
    and `metal_` ones alike.

    Raises:
        RuntimeError: No device of that vendor answers, or this build lacks its kernels.

    Signature:
        >>> def cuda_count_devices() -> int: ...
    """

def metal_count_devices() -> int:
    """Count the devices of one GPU vendor this process sees: `cuda_count_devices`, and the `rocm_`
    and `metal_` ones alike.

    Raises:
        RuntimeError: No device of that vendor answers, or this build lacks its kernels.

    Signature:
        >>> def cuda_count_devices() -> int: ...
    """

def cuda_capabilities_detected(ordinal: int, /) -> Capability:
    """Get the capabilities a GPU can execute, by its runtime's ordinal, whether or not the kernels
    were compiled in: `cuda_capabilities_detected`, and the `rocm_` and `metal_` ones alike.

    Returns:
        Capability: GPU bits sit above every member, so they print as a number.

    Raises:
        RuntimeError: No device with that ordinal answers.

    Signature:
        >>> def cuda_capabilities_detected(ordinal, /) -> sz.Capability: ...
    """

def rocm_capabilities_detected(ordinal: int, /) -> Capability:
    """Get the capabilities a GPU can execute, by its runtime's ordinal, whether or not the kernels
    were compiled in: `cuda_capabilities_detected`, and the `rocm_` and `metal_` ones alike.

    Returns:
        Capability: GPU bits sit above every member, so they print as a number.

    Raises:
        RuntimeError: No device with that ordinal answers.

    Signature:
        >>> def cuda_capabilities_detected(ordinal, /) -> sz.Capability: ...
    """

def metal_capabilities_detected(ordinal: int, /) -> Capability:
    """Get the capabilities a GPU can execute, by its runtime's ordinal, whether or not the kernels
    were compiled in: `cuda_capabilities_detected`, and the `rocm_` and `metal_` ones alike.

    Returns:
        Capability: GPU bits sit above every member, so they print as a number.

    Raises:
        RuntimeError: No device with that ordinal answers.

    Signature:
        >>> def cuda_capabilities_detected(ordinal, /) -> sz.Capability: ...
    """

def cuda_capabilities_compiled() -> Capability:
    """Get the capabilities of one GPU vendor whose kernels were compiled into this binary, whether or
    not a device runs them: `cuda_capabilities_compiled`, and the `rocm_` and `metal_` ones alike.

    Signature:
        >>> def cuda_capabilities_compiled() -> sz.Capability: ...
    """

def rocm_capabilities_compiled() -> Capability:
    """Get the capabilities of one GPU vendor whose kernels were compiled into this binary, whether or
    not a device runs them: `cuda_capabilities_compiled`, and the `rocm_` and `metal_` ones alike.

    Signature:
        >>> def cuda_capabilities_compiled() -> sz.Capability: ...
    """

def metal_capabilities_compiled() -> Capability:
    """Get the capabilities of one GPU vendor whose kernels were compiled into this binary, whether or
    not a device runs them: `cuda_capabilities_compiled`, and the `rocm_` and `metal_` ones alike.

    Signature:
        >>> def cuda_capabilities_compiled() -> sz.Capability: ...
    """

def cuda_capabilities_enabled(ordinal: int, /) -> Capability:
    """Get the capabilities kernels run with on a GPU, by its runtime's ordinal, both detected and
    compiled: `cuda_capabilities_enabled`, and the `rocm_` and `metal_` ones alike. Engines,
    `Strings.copy` and `synchronize` take this mask, which names the vendor, and a stream, which names
    the device.

    Raises:
        RuntimeError: No device with that ordinal answers.

    Signature:
        >>> def cuda_capabilities_enabled(ordinal, /) -> sz.Capability: ...
    """

def rocm_capabilities_enabled(ordinal: int, /) -> Capability:
    """Get the capabilities kernels run with on a GPU, by its runtime's ordinal, both detected and
    compiled: `cuda_capabilities_enabled`, and the `rocm_` and `metal_` ones alike. Engines,
    `Strings.copy` and `synchronize` take this mask, which names the vendor, and a stream, which names
    the device.

    Raises:
        RuntimeError: No device with that ordinal answers.

    Signature:
        >>> def cuda_capabilities_enabled(ordinal, /) -> sz.Capability: ...
    """

def metal_capabilities_enabled(ordinal: int, /) -> Capability:
    """Get the capabilities kernels run with on a GPU, by its runtime's ordinal, both detected and
    compiled: `cuda_capabilities_enabled`, and the `rocm_` and `metal_` ones alike. Engines,
    `Strings.copy` and `synchronize` take this mask, which names the vendor, and a stream, which names
    the device.

    Raises:
        RuntimeError: No device with that ordinal answers.

    Signature:
        >>> def cuda_capabilities_enabled(ordinal, /) -> sz.Capability: ...
    """

def cuda_stream_init(ordinal: int, /) -> int:
    """Create a stream on a GPU, by its runtime's ordinal, which names that device to the engines,
    `Strings.copy` and `synchronize` it is passed to: `cuda_stream_init`, and the `rocm_` and `metal_`
    ones alike. Free it with the matching `*_stream_free` once its work is synchronized.

    Returns:
        int: The `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>` handle.

    Raises:
        RuntimeError: No device with that ordinal answers, or this build lacks its vendor.

    Signature:
        >>> def cuda_stream_init(ordinal, /) -> int: ...
    """

def rocm_stream_init(ordinal: int, /) -> int:
    """Create a stream on a GPU, by its runtime's ordinal, which names that device to the engines,
    `Strings.copy` and `synchronize` it is passed to: `cuda_stream_init`, and the `rocm_` and `metal_`
    ones alike. Free it with the matching `*_stream_free` once its work is synchronized.

    Returns:
        int: The `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>` handle.

    Raises:
        RuntimeError: No device with that ordinal answers, or this build lacks its vendor.

    Signature:
        >>> def cuda_stream_init(ordinal, /) -> int: ...
    """

def metal_stream_init(ordinal: int, /) -> int:
    """Create a stream on a GPU, by its runtime's ordinal, which names that device to the engines,
    `Strings.copy` and `synchronize` it is passed to: `cuda_stream_init`, and the `rocm_` and `metal_`
    ones alike. Free it with the matching `*_stream_free` once its work is synchronized.

    Returns:
        int: The `cudaStream_t`, `hipStream_t` or `id<MTLCommandQueue>` handle.

    Raises:
        RuntimeError: No device with that ordinal answers, or this build lacks its vendor.

    Signature:
        >>> def cuda_stream_init(ordinal, /) -> int: ...
    """

def cuda_stream_free(stream: int, /) -> None:
    """Free a stream `cuda_stream_init` made, and the `rocm_` and `metal_` ones alike, once
    `synchronize` joined whatever was queued on it; nothing may use the stream afterwards.

    Signature:
        >>> def cuda_stream_free(stream, /) -> None: ...
    """

def rocm_stream_free(stream: int, /) -> None:
    """Free a stream `cuda_stream_init` made, and the `rocm_` and `metal_` ones alike, once
    `synchronize` joined whatever was queued on it; nothing may use the stream afterwards.

    Signature:
        >>> def cuda_stream_free(stream, /) -> None: ...
    """

def metal_stream_free(stream: int, /) -> None:
    """Free a stream `cuda_stream_init` made, and the `rocm_` and `metal_` ones alike, once
    `synchronize` joined whatever was queued on it; nothing may use the stream afterwards.

    Signature:
        >>> def cuda_stream_free(stream, /) -> None: ...
    """

def synchronize(capabilities: Capability, stream: int | None = None) -> None:
    """Wait for everything enqueued on one stream, releasing the GIL meanwhile.

    A GPU engine's verbs enqueue and return, so their outputs are read, and their inputs freed, only
    after this returns. Returns at once for CPU capabilities.

    Args:
        capabilities (Capability): A producer's mask, which names the vendor.
        stream (int, optional): A stream of that vendor as an integer, or None for the default.

    Raises:
        BufferError: The stream belongs to another device.
        RuntimeError: A launch on the stream failed, or the device does not answer.

    Signature:
        >>> def synchronize(capabilities, stream=None) -> None: ...
    """

def utf8_count(text: _StringLike, *, capabilities: Capability | None = None) -> int:
    r"""Count the number of UTF-8 characters in a string.

    Unlike len() which returns bytes, this counts actual Unicode characters,
    handling multi-byte UTF-8 sequences correctly.

    Args:
      text (String or str or bytes): The string object.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      int: Number of UTF-8 characters in the string.

    Example:
      >>> sz.utf8_count('hello')  # 5 ASCII chars = 5
      5
      >>> sz.utf8_count('\xc3\xa9')  # 1 e-acute char = 1
      1
    """

def utf8_split_newlines(
    text: _StringLike, skip_empty: bool = False, with_separators: bool = False, *, capabilities: Capability | None = None
) -> Utf8SplitNewlines:
    r"""Create an iterator over the lines, the content between Unicode newlines.

    Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.
    Unlike splitlines(), this returns an iterator for memory-efficient processing.

    Args:
      text (String or str or bytes): The string object.
      skip_empty (bool, optional): Skip empty lines, defaulting to False.
      with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding the lines as String objects.

    Recognized newlines:
      LF (\n), VT (\v), FF (\f), CR (\r), NEL (U+0085),
      LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\r\n)

    Example:
      >>> sum(1 for _ in sz.String('a\nb').utf8_split_newlines())
      2
    """

def utf8_newlines(
    text: _StringLike, skip_empty: bool = False, *, capabilities: Capability | None = None
) -> Utf8Newlines:
    r"""Create an iterator over the Unicode newline separators, one per codepoint, CR+LF kept together.

    Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.
    Yields each newline separator; utf8_split_newlines() yields the lines between them.

    Args:
      text (String or str or bytes): The string object.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding the newline separators as String objects.

    Recognized newlines:
      LF (\n), VT (\v), FF (\f), CR (\r), NEL (U+0085),
      LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\r\n)

    Example:
      >>> sum(1 for _ in sz.String('a\nb').utf8_newlines())
      1
    """

def utf8_split_whitespaces(
    text: _StringLike, skip_empty: bool = False, with_separators: bool = False, *, capabilities: Capability | None = None
) -> Utf8SplitWhitespaces:
    """Create an iterator over the tokens, the content between Unicode whitespace.

    Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.
    Splits on each whitespace codepoint; pass skip_empty=True for str.split()-style tokens.

    Args:
      text (String or str or bytes): The string object.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding the non-whitespace tokens as String objects.

    Recognized whitespace:
      ASCII: TAB, LF, VT, FF, CR, SPACE
      Latin-1: NEXT LINE, NO-BREAK SPACE
      General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.
      CJK: IDEOGRAPHIC SPACE (U+3000)

    Example:
      >>> sum(1 for _ in sz.String('foo bar baz').utf8_split_whitespaces())
      3
    """

def utf8_whitespaces(
    text: _StringLike, skip_empty: bool = False, *, capabilities: Capability | None = None
) -> Utf8Whitespaces:
    """Create an iterator over Unicode whitespace separators, one per codepoint, CR+LF kept together.

    Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.
    Yields each whitespace codepoint; utf8_split_whitespaces() yields the tokens between them.

    Args:
      text (String or str or bytes): The string object.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding the whitespace separators as String objects.

    Recognized whitespace:
      ASCII: TAB, LF, VT, FF, CR, SPACE
      Latin-1: NEXT LINE, NO-BREAK SPACE
      General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.
      CJK: IDEOGRAPHIC SPACE (U+3000)

    Example:
      >>> sum(1 for _ in sz.String('foo bar').utf8_whitespaces())
      1
    """

def utf8_split_delimiters(
    text: _StringLike, skip_empty: bool = False, with_separators: bool = False, *, capabilities: Capability | None = None
) -> Utf8SplitDelimiters:
    """Create an iterator over the fields, the content between Unicode delimiters.

    Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and
    separator/whitespace (Z*) codepoint - the superset of utf8_split_whitespaces().
    Splits on delimiters, one separator per codepoint, and yields the segments between them.

    Args:
      text (String or str or bytes): The string object.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding the non-delimiter segments as String objects.

    Example:
      >>> list(str(t) for t in sz.String('Hi, world').utf8_split_delimiters(skip_empty=True))
      ['Hi', 'world']
    """

def utf8_delimiters(
    text: _StringLike, skip_empty: bool = False, *, capabilities: Capability | None = None
) -> Utf8Delimiters:
    """Create an iterator over Unicode delimiter separators, one per codepoint, CR+LF kept together.

    Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and
    separator/whitespace (Z*) codepoint. Yields each delimiter codepoint;
    utf8_split_delimiters() yields the fields between them.

    Args:
      text (String or str or bytes): The string object.
      skip_empty (bool, optional): Skip empty segments, defaulting to False.
      capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
    Returns:
      iterator: An iterator yielding the delimiter separators as String objects.

    Example:
      >>> list(str(d) for d in sz.String('a.b').utf8_delimiters())
      ['.']
    """

def utf8_wordbreaks(string: _StringLike, /, *, capabilities: Capability | None = None) -> Utf8Wordbreaks:
    """utf8_wordbreaks(string, /, *, capabilities=None)

    Return an iterator yielding words per Unicode UAX-29 word boundary rules.
    Unlike str.split(), this is UAX-29 compliant and supports all Unicode scripts.

    Args:
        string: The input UTF-8 string to split into words.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        Iterator yielding String objects for each word.


    Example:
      >>> # Stream UAX-29 word tokens lazily:
      >>> 'world' in (str(w) for w in sz.utf8_wordbreaks('Hi, world'))
      True
    """

def utf8_codepoints(string: _StringLike, /, *, capabilities: Capability | None = None) -> Utf8Codepoints:
    """utf8_codepoints(string, /, *, capabilities=None)

    Return an iterator yielding Unicode code points as int, decoded from UTF-8.
    Ill-formed bytes decode to U+FFFD, the replacement character, so iteration is
    total and never raises on malformed input.

    Args:
        string: The input UTF-8 string to decode into code points.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        Iterator yielding int code points, one per Unicode scalar value.


    Example:
      >>> list(sz.utf8_codepoints('AB'))
      [65, 66]
    """

def utf8_graphemes(string: _StringLike, /, *, capabilities: Capability | None = None) -> Utf8Graphemes:
    """utf8_graphemes(string, /, *, capabilities=None)

    Return an iterator yielding grapheme clusters per Unicode UAX-29 rules.
    A grapheme cluster is a user-perceived character (e.g. a base plus combining
    marks, or an emoji ZWJ sequence) and may span several code points.

    Args:
        string: The input UTF-8 string to split into grapheme clusters.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        Iterator yielding String objects for each grapheme cluster.


    Example:
      >>> # Stream UAX-29 grapheme clusters lazily:
      >>> [str(g) for g in sz.utf8_graphemes('abc')]
      ['a', 'b', 'c']
    """

def utf8_sentences(string: _StringLike, /, *, capabilities: Capability | None = None) -> Utf8Sentences:
    """utf8_sentences(string, /, *, capabilities=None)

    Return an iterator yielding sentences per Unicode UAX-29 sentence boundary rules.
    UAX-29 compliant and Unicode-script aware, unlike naive period splitting.

    Args:
        string: The input UTF-8 string to split into sentences.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        Iterator yielding String objects for each sentence.


    Example:
      >>> # Stream UAX-29 sentences lazily:
      >>> len(list(sz.utf8_sentences('Hi. Bye.'))) >= 2
      True
    """

def utf8_linebreaks(string: _StringLike, /, *, capabilities: Capability | None = None) -> Utf8Linebreaks:
    r"""utf8_linebreaks(string, /, *, capabilities=None)

    Return an iterator yielding segments at line-break opportunities per UAX-14.
    Each segment ends at a line-break opportunity, a soft wrap point.
    For hard-line splitting (str.splitlines()), use utf8_split_newlines().

    Args:
        string: The input UTF-8 string to split at line-break opportunities.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        Iterator yielding String objects for each line-break-opportunity segment.


    Example:
      >>> # Stream UAX-14 line-break opportunities lazily:
      >>> len(list(sz.utf8_linebreaks('a\nb'))) >= 2
      True
    """

def utf8_uncased_fold(
    text: _StringLike, validate: bool = False, *, capabilities: Capability | None = None
) -> bytes:
    r"""Apply Unicode case folding to a UTF-8 string.

    Case folding normalizes text for uncased comparisons,
    handling one-to-many expansions, like German sharp S to 'ss'.

    Args:
        text (String or str or bytes): The input UTF-8 string.
        validate (bool): If True, validate UTF-8 before processing. Default: False.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        bytes: The case-folded UTF-8 string.

    Example:
        >>> sz.utf8_uncased_fold('HELLO')
        b'hello'
        >>> sz.utf8_uncased_fold('Stra\u00dfe')  # German sharp S
        b'strasse'
    """

def utf8_norm(
    text: _StringLike, form: str, validate: bool = False, *, capabilities: Capability | None = None
) -> bytes:
    r"""Normalize a UTF-8 string to a Unicode normalization form.

    Transforms the input into NFD, NFC, NFKD, or NFKC, mirroring `unicodedata.normalize` but
    returning raw UTF-8 bytes, or into NFKC_Casefold, the Unicode toNFKC_Casefold: NFKC with full
    case folding and no default-ignorable codepoints.

    Args:
        text (String or str or bytes): The input UTF-8 string.
        form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD', 'NFKC_Casefold'.
        validate (bool): If True, validate UTF-8 before processing. Default: False.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        bytes: The normalized UTF-8 string.

    Example:
        >>> sz.utf8_norm('caf\u00e9', 'NFD')  # decompose
        b'cafe\xcc\x81'
        >>> sz.utf8_norm('\ufb01', 'NFKD')  # fi ligature
        b'fi'
        >>> sz.utf8_norm('Stra\u00dfe\u00ad', 'NFKC_Casefold')  # folded, soft hyphen dropped
        b'strasse'
    """

def utf8_find_denormalized(
    text: _StringLike, form: str, *, capabilities: Capability | None = None
) -> int | None:
    r"""Return the byte offset of the first normalization violation, or None.

    Scans the UTF-8 string and returns the byte offset of the first codepoint
    that breaks the given normalization form (first non-Yes Quick-Check result
    or canonical-ordering violation). Returns None when the string is already
    fully normalized.

    Args:
        text (String or str or bytes): The input UTF-8 string.
        form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD', 'NFKC_Casefold'.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        int | None: Byte offset of first violation, or None if already normalized.

    Example:
        >>> sz.utf8_find_denormalized('cafe\u0301', 'NFC')  # 'e' + combining acute
        3
        >>> sz.utf8_find_denormalized('caf\u00e9', 'NFC') is None  # precomposed, already NFC
        True
    """

def utf8_uncased_search(
    haystack: _StringLike,
    needle: _StringLike,
    start: int = 0,
    end: int | None = None,
    validate: bool = False,
    *,
    capabilities: Capability | None = None,
) -> int:
    r"""Find substring using Unicode uncased matching.

    Performs a uncased search using Unicode case folding rules,
    correctly handling one-to-many expansions, like 'ß' matching 'SS'.

    IMPORTANT - Type-dependent behavior:
      - str input:   start/end are CODEPOINT offsets, returns CODEPOINT offset
      - bytes input: start/end are BYTE offsets, returns BYTE offset

    Args:
        haystack (String or str or bytes): The string to search in.
        needle (String or str or bytes): The substring to find.
        start (int, optional): Starting index, defaulting to 0.
        end (int, optional): Ending index, defaulting to length.
        validate (bool): If True, validate UTF-8 before processing. Default: False.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        int: Index of the first match, or -1 if not found.

    Example:
        >>> sz.utf8_uncased_search('Hello World', 'WORLD')  # str: codepoint offset
        6
        >>> sz.utf8_uncased_search('Straße', 'STRASSE')  # 'ß' = 1 codepoint
        0
        >>> sz.utf8_uncased_search(b'Stra\xc3\x9fe', b'STRASSE')  # 'ß' = 2 bytes
        0
    """

def utf8_uncased_order(
    a: _StringLike, b: _StringLike, validate: bool = False, *, capabilities: Capability | None = None
) -> int:
    """Compare two UTF-8 strings uncasedly.

    Performs lexicographical comparison using Unicode case folding, correctly handling
    one-to-many expansions, like 'Straße' equaling 'STRASSE'.

    Args:
        a (String or str or bytes): First string to compare.
        b (String or str or bytes): Second string to compare.
        validate (bool): If True, validate UTF-8 before processing. Default: False.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Returns:
        int: Negative if a < b, zero if equal, positive if a > b.

    Example:
        >>> sz.utf8_uncased_order('hello', 'HELLO')
        0
        >>> sz.utf8_uncased_order('apple', 'BANANA')
        -1
    """

def utf8_uncased_matches(
    haystack: _StringLike,
    needle: _StringLike,
    /,
    include_overlapping: bool = False,
    *,
    capabilities: Capability | None = None,
) -> Utf8UncasedMatches:
    """utf8_uncased_matches(haystack, needle, /, include_overlapping=False, *, capabilities=None)

    Iterate over all uncased matches of needle in haystack.

    This function uses Unicode case folding for proper handling of
    international text. The matched region length may differ from the
    needle length due to case folding expansions, like 'ß' matching 'SS'.
    An empty needle matches at every codepoint boundary, the end included.

    Args:
        haystack (String or str or bytes): The string to search in.
        needle (String or str or bytes): The pattern to find.
        include_overlapping (bool, optional): Allow overlapping matches, defaulting to False.
        capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

    Yields:
        String: Each matched region as a view into the original haystack.

    Examples:
        >>> list(sz.utf8_uncased_matches('Hello HELLO hello', 'hello'))
        [sz.String('Hello'), sz.String('HELLO'), sz.String('hello')]
        >>> list(sz.utf8_uncased_matches('Straße STRASSE', 'strasse'))
        [sz.String('Straße'), sz.String('STRASSE')]
    """

class FindSplits:
    """FindSplits(string, separator, ...)

    Text-splitting iterator for efficient string processing.
    Provides lazy evaluation of string splits without materializing all results.

    Created by:
      - String.split_iter()
      - String.rsplit_iter()
      - String.split_byteset_iter()
      - String.rsplit_byteset_iter()

    Features:
      - Memory-efficient: yields results one at a time
      - Forward and reverse iteration support
      - Character set and string separator support

    Example:
      >>> # split_iter streams parts lazily - prefer it over split() on large inputs:
      >>> for field in sz.String('2024-01-15').split_iter('-'):
      ...     print(field)
      2024
      01
      15
    """

    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Hasher:
    """Hasher(seed=0, *, capabilities=None)

    Incremental, AES-accelerated hasher producing the same 64-bit value as `sz.hash`.
    Feed data in chunks with `update()`, read the digest, and reuse the object via `reset()`.

    Args:
      seed (int, optional): 64-bit seed mixed into the initial state. Defaults to 0.
      capabilities (Capability, optional): Capabilities this hasher and its calls run,
        defaulting to the CPU's enabled ones.
    Example:
      >>> h = sz.Hasher()
      >>> _ = h.update(b'hello').update(b' world')
      >>> h.digest() == sz.hash(b'hello world')
      True
    """

    def __init__(self, seed: int = 0, *, capabilities: Capability | None = None) -> None: ...
    def update(self, data: str | bytes) -> Hasher:
        """update(data) -> Hasher

        Absorb more bytes into the running hash and return self for chaining.

        Args:
          data (str | bytes): Chunk to mix into the state.
        Returns:
          Hasher: The same object, enabling `h.update(a).update(b)`.
        Example:
          >>> sz.Hasher().update(b'ab').update(b'c').digest() == sz.hash(b'abc')
          True
        """
    def digest(self) -> int:
        """digest() -> int

        Return the current hash as an unsigned 64-bit integer without consuming state.

        Returns:
          int: Hash of all data absorbed so far.
        Example:
          >>> isinstance(sz.Hasher().update(b'x').digest(), int)
          True
        """
    def hexdigest(self) -> str:
        """hexdigest() -> str

        Return the current hash as a 16-character lowercase hex string.

        Returns:
          str: Zero-padded 16-digit hex of the 64-bit hash.
        Example:
          >>> len(sz.Hasher().hexdigest())
          16
        """
    def reset(self) -> Hasher:
        """reset() -> Hasher

        Reset the state to the initial seed and return self.

        Returns:
          Hasher: The same object, emptied and reusable.
        Example:
          >>> h = sz.Hasher().update(b'data').reset()
          >>> h.digest() == sz.Hasher().digest()
          True
        """

class Sha256:
    """Sha256(*, capabilities=None)

    Incremental SHA-256 hasher with a hashlib-compatible interface, hardware-accelerated
    where available. Feed data with `update()`, then read `digest()` or `hexdigest()`.
    The capabilities it and its calls run default to the CPU's enabled ones.

    Example:
      >>> import hashlib
      >>> sz.Sha256().update(b'abc').hexdigest() == hashlib.sha256(b'abc').hexdigest()
      True
    """

    digest_length: int
    def __init__(self, *, capabilities: Capability | None = None) -> None: ...
    def update(self, data: str | bytes) -> Sha256:
        """update(data) -> Sha256

        Absorb more bytes into the running SHA-256 state and return self for chaining.

        Args:
          data (str | bytes): Chunk to mix into the state.
        Returns:
          Sha256: The same object, enabling `h.update(a).update(b)`.
        Example:
          >>> hexdigest = sz.Sha256().update(b'a').update(b'bc').hexdigest()
          >>> hexdigest[:8]
          'ba7816bf'
        """
    def digest(self) -> bytes:
        """digest() -> bytes

        Return the current 32-byte SHA-256 digest without consuming state.

        Returns:
          bytes: The 32-byte digest.
        Example:
          >>> len(sz.Sha256().update(b'abc').digest())
          32
        """
    def hexdigest(self) -> str:
        """hexdigest() -> str

        Return the current SHA-256 digest as a 64-character lowercase hex string.

        Returns:
          str: 64-digit hex of the 32-byte digest.
        Example:
          >>> import hashlib
          >>> sz.Sha256().update(b'abc').hexdigest() == hashlib.sha256(b'abc').hexdigest()
          True
        """
    def reset(self) -> Sha256:
        """reset() -> Sha256

        Reset the state to the initial SHA-256 constants and return self.

        Returns:
          Sha256: The same object, emptied and reusable.
        Example:
          >>> h = sz.Sha256().update(b'x').reset()
          >>> h.hexdigest() == sz.Sha256().hexdigest()
          True
        """
    def copy(self) -> Sha256:
        """copy() -> Sha256

        Return an independent copy of the hasher with identical internal state.

        Returns:
          Sha256: A new object that can be advanced separately from the original.
        Example:
          >>> a = sz.Sha256().update(b'ab')
          >>> a.copy().update(b'c').hexdigest() == sz.Sha256().update(b'abc').hexdigest()
          True
        """

class Sha256s:
    """Sha256s(lanes, *, capabilities=None)

    Incremental SHA-256 over many independent messages at once, one message per lane.

    Hashing a single message is a serial dependency chain no instruction set can speed
    up, but independent messages compress in parallel lanes: sixteen at a time on
    AVX-512, eight on AVX2. Feed at least a few kilobytes per lane per call, as shorter
    chunks never reach the lane-parallel path.

    Lanes are workers rather than messages: assigning `lanes[i] = sz.Sha256()` retires a
    finished lane and starts the next message in it, leaving every other lane streaming.

    Args:
      lanes (int): How many independent messages to advance together.
      capabilities (Capability, optional): Capabilities these lanes and their calls run,
        defaulting to the CPU's enabled ones.
    Example:
      >>> lanes = sz.Sha256s(2)
      >>> _ = lanes.update([b'Hello, ', b'Goodbye, '])
      >>> _ = lanes.update([b'world!', b'world!'])
      >>> hello, goodbye = lanes.digest()
      >>> hello == sz.Sha256().update(b'Hello, world!').digest()
      True
    """

    digest_length: int
    def __init__(self, lanes: int, *, capabilities: Capability | None = None) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Sha256: ...
    def __setitem__(self, index: int, lane: Sha256) -> None: ...
    def update(self, chunks: Strings | Sequence[str | bytes]) -> Sha256s:
        """update(chunks) -> Sha256s

        Append the next chunk of every lane's message.

        Args:
          chunks (Strings | sequence): Exactly one string-like chunk per lane; may be empty. A Strings
               is read in place, whatever its layout, so a tape never leaves its buffer.
        Returns:
          Sha256s: The same object, to allow chaining.
        """
    def digest(self, out: _Buffer | None = None) -> list[bytes] | _Buffer:
        """digest(out=None) -> list[bytes] | buffer

        Return each lane's 32-byte digest, leaving every lane able to accept more data.

        Args:
          out (buffer, optional): Writable, C-contiguous buffer of bytes holding at least
               `len(lanes) * 32` of them - a `(len(lanes), 32)` numpy.uint8 matrix receives one
               digest per row with zero allocation. Defaults to None.
        Returns:
          list[bytes]: One digest per lane in lane order, or `out` itself when a buffer is given.
        Example:
          >>> lanes = sz.Sha256s(2)
          >>> digests = bytearray(len(lanes) * sz.Sha256.digest_length)
          >>> _ = lanes.update([b'abc', b'abc']).digest(out=digests)
          >>> bytes(digests[:sz.Sha256.digest_length]) == sz.Sha256().update(b'abc').digest()
          True
        """
    def hexdigest(self) -> list[str]:
        """hexdigest() -> list[str]

        Return each lane's digest as a 64-character lowercase hex string, in lane order.

        Returns:
          list[str]: One 64-digit hex string per lane.
        """
    def reset(self) -> Sha256s:
        """reset() -> Sha256s

        Return every lane to the initial state, reusing the allocation.

        Returns:
          Sha256s: The same object, to allow chaining.
        """
    def copy(self) -> Sha256s:
        """copy() -> Sha256s

        Return an independent batch whose lanes hold identical internal state.

        Returns:
          Sha256s: A new object that can be advanced separately from the original.
        """

class LevenshteinEngine:
    """LevenshteinEngine(queries, symbol='bytes', *, capabilities=None, stream=None)

    Prepare a batch of queries once and score it against many collections of candidates.

    Unit-cost Myers bit-parallel edit distance: every query is packed into match masks at
    construction, and each round streams the candidates past them.

    Args:
      queries (Strings): Patterns every candidate is scored against.
      symbol (str, optional): 'bytes' counts byte edits, 'runes' counts UTF-8 rune edits.
      capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),
        defaulting to cpu_capabilities_enabled().
      stream (int, optional): A stream of that vendor as an integer, naming the device, or None
        for the default.
    Example:
      >>> engine = sz.LevenshteinEngine(sz.Strings(['hello']), symbol='bytes')
      >>> out = memoryview(bytearray(8)).cast('Q', (1, 1))
      >>> engine.distances(sz.Strings(['hallo']), out)
      >>> out[0, 0]
      1
    """

    def __init__(
        self, queries: Strings, symbol: str = "bytes", *, capabilities: Capability | None = None, stream: int | None = None
    ) -> None: ...
    def distances(self, candidates: Strings, out: _Buffer, *, bound: int | None = None, stream: int | None = None) -> None:
        """distances(candidates, out, *, bound=None, stream=None) -> None

        Score every prepared query against every candidate, into `out`.

        Args:
          candidates (Strings): Texts forming the matrix columns, from `Strings.copy` on a GPU.
          out (buffer): Writable 2-D buffer of pointer-width unsigned integers like numpy.uintp,
            at least (len(queries), len(candidates)), contiguous along its candidate axis.
          bound (int, optional): The largest distance reported exactly. A candidate past it reports
            exactly bound + 1, and None keeps every distance exact.
          stream (int, optional): A stream of the engine's device as an integer, or None for the
            default. A device engine enqueues there and returns, so `out` must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Example:
          >>> engine = sz.LevenshteinEngine(sz.Strings(['hello', 'world']))
          >>> out = memoryview(bytearray(32)).cast('Q', (2, 2))
          >>> engine.distances(sz.Strings(['hallo', 'word']), out)
          >>> out[0, 0]
          1
        """

class OverlapEngine:
    """OverlapEngine(queries, widths, *, candidates_budget=0, capabilities=None, stream=None)

    Hash a batch of queries into one window forest and probe it with many candidates.

    A window is a fixed-width byte n-gram, and a score is the share of a candidate's windows
    that the query also spells, in [0, 1]. The score is asymmetric in the two sides.

    Args:
      queries (Strings): Texts whose windows are sorted into the forest.
      widths (sequence of int): Window widths in bytes, the last axis of every output.
      candidates_budget (int, optional): The most candidates one round may carry on a device,
        ignored on the CPU.
      capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),
        defaulting to cpu_capabilities_enabled().
      stream (int, optional): A stream of that vendor as an integer, naming the device, or None
        for the default.
    Example:
      >>> engine = sz.OverlapEngine(sz.Strings(['abcdef']), [3, 4])
      >>> out = memoryview(bytearray(8)).cast('f', (1, 1, 2))
      >>> engine.scores(sz.Strings(['abcdefg']), out)
      >>> 0.0 <= out[0, 0, 0] <= 1.0
      True
    """

    def __init__(
        self,
        queries: Strings,
        widths: Sequence[int],
        *,
        candidates_budget: int = 0,
        capabilities: Capability | None = None,
        stream: int | None = None,
    ) -> None: ...
    def scores(self, candidates: Strings, out: _Buffer, *, radius: float | None = None, stream: int | None = None) -> None:
        """scores(candidates, out, *, radius=None, stream=None) -> None

        Score every prepared query against every candidate, at every width, into `out`.

        Args:
          candidates (Strings): Texts whose windows probe the forest, from `Strings.copy` on a GPU.
          out (buffer): Writable 3-D buffer of 32-bit floats, at least
            (len(queries), len(candidates), len(widths)), contiguous along its width axis.
          radius (float, optional): The furthest distance a pair may lie at and still score exactly,
            a distance being the windows it misses over its longer side's, summed over every width.
            A pair past it scores 0.0 at every width, and None or 1.0 keep every score exact.
          stream (int, optional): A stream of the engine's device as an integer, or None for the
            default. A device engine joins it before returning, so `out` must be device-reachable;
            a host engine ignores it.
        Example:
          >>> engine = sz.OverlapEngine(sz.Strings(['abcdef']), [3])
          >>> out = memoryview(bytearray(4)).cast('f', (1, 1, 1))
          >>> engine.scores(sz.Strings(['abcdef']), out)
          >>> out[0, 0, 0]
          1.0
        """

class Utf8Graphemes:
    """Utf8Graphemes(string, ...)

    UTF-8 aware grapheme cluster boundary iterator per Unicode UAX-29 algorithm.
    Yields grapheme clusters, the user-perceived characters.

    Created by:
      - String.utf8_graphemes()
      - sz.utf8_graphemes()

    UAX-29 Grapheme_Cluster_Break rules implemented:
      - GB3: CR x LF, no break
      - GB4-GB5: Control/CR/LF breaks
      - GB6-GB8: Hangul syllable sequences
      - GB9-GB9c: Extend/ZWJ/SpacingMark, Indic conjuncts
      - GB11-GB12: Emoji ZWJ and Regional Indicator pairs


    Example:
      >>> len(list(sz.utf8_graphemes('abc'))) == 3
      True
    """

    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8Linebreaks:
    r"""Utf8Linebreaks(string, ...)

    UTF-8 aware line-break-opportunity iterator per Unicode UAX-14 algorithm.
    Yields String views for each line-break-opportunity segment, a soft wrap.
    For hard-line splitting (str.splitlines()), use utf8_split_newlines().

    Created by:
      - String.utf8_linebreaks()
      - sz.utf8_linebreaks()

    UAX-14 Line_Break rules implemented:
      - LB4-LB6: Mandatory breaks at BK, CR, LF, NL
      - LB7-LB8: Spaces and ZWSP break opportunities
      - LB9-LB14: Combining marks, opening/closing punctuation
      - LB15-LB31: Quotation, numbers, words, CJK rules


    Example:
      >>> len(list(sz.utf8_linebreaks('a\nb'))) >= 2
      True
    """

    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8Codepoints:
    """Utf8Codepoints(string)

    UTF-8 aware code point iterator yielding Unicode scalar values as int.
    Ill-formed bytes decode to U+FFFD, so iteration is total.

    Created by:
      - String.utf8_codepoints()
      - sz.utf8_codepoints()

    Example:
      >>> list(sz.utf8_codepoints('AB'))
      [65, 66]
    """

    def __iter__(self) -> Iterator[int]: ...
    def __next__(self) -> int: ...

class Utf8Sentences:
    """Utf8Sentences(string, ...)

    UTF-8 aware sentence boundary iterator per Unicode UAX-29 algorithm.
    Yields sentences, the text segments between consecutive sentence boundaries.

    Created by:
      - String.utf8_sentences()
      - sz.utf8_sentences()

    UAX-29 Sentence_Break rules implemented:
      - SB3: CR x LF, no break
      - SB4: ParaSep breaks
      - SB6-SB8: ATerm/STerm sentence-final sequences
      - SB9-SB11: Close/Sp/ParaSep continuations
      - SB998: Otherwise no break


    Example:
      >>> len(list(sz.utf8_sentences('Hi. Bye.'))) >= 2
      True
    """

    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8Wordbreaks:
    """Utf8Wordbreaks(string, ...)

    UTF-8 aware word boundary iterator per Unicode UAX-29 algorithm.
    Yields words, the text segments between consecutive word boundaries.

    Created by:
      - String.utf8_wordbreaks()
      - sz.utf8_wordbreaks()

    UAX-29 Word_Break rules implemented:
      - WB3: CR x LF, no break
      - WB4: Ignore Extend/Format/ZWJ
      - WB5-WB13: Letter, number, punctuation rules
      - WB15-WB16: Regional Indicator pairs


    Example:
      >>> len(list(sz.utf8_wordbreaks('Hi there'))) >= 2
      True
    """

    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8UncasedMatches:
    """Utf8UncasedMatches(haystack, needle, ...)

    Iterator yielding all uncased matches of needle in haystack.
    Uses Unicode case folding for proper handling of international text.

    Created by:
      - String.utf8_uncased_matches()
      - sz.utf8_uncased_matches()

    Each iteration yields a String view of the matched region in the haystack.
    The matched length may differ from needle length due to case folding
    expansions, like German 'ß' matching 'SS'.

    Example:
      >>> len(list(sz.utf8_uncased_matches('aAaA', 'a')))
      4
    """

    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8SplitNewlines:
    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8Newlines:
    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8SplitWhitespaces:
    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8Whitespaces:
    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8SplitDelimiters:
    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class Utf8Delimiters:
    def __iter__(self) -> Iterator[String]: ...
    def __next__(self) -> String: ...

class RegexEngine:
    """RegexEngine(patterns, case_sensitivity='cased', *, capabilities=None, stream=None)

    Compile a vocabulary of RE2-syntax patterns once and stream many collections of haystacks
    through it, reporting leftmost-first matches across all of them, the lower index winning ties.

    Args:
      patterns (Strings): The vocabulary to compile; a malformed pattern raises a ValueError
        naming its index.
      case_sensitivity (str, optional): 'cased' matches bytes, 'uncased' folds both sides.
      capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),
        defaulting to cpu_capabilities_enabled().
      stream (int, optional): A stream of that vendor as an integer, naming the device, or None
        for the default.
    Example:
      >>> engine = sz.RegexEngine(sz.Strings(['[0-9]+', '[a-z]+']))
      >>> out = memoryview(bytearray(8)).cast('Q')
      >>> engine.counts(sz.Strings(['abc 123 de']), out)
      >>> out[0]
      3
    """

    def __init__(
        self,
        patterns: Strings,
        case_sensitivity: str = "cased",
        *,
        capabilities: Capability | None = None,
        stream: int | None = None,
    ) -> None: ...
    def counts(self, haystacks: Strings, out: _Buffer, *, stream: int | None = None) -> None:
        """counts(haystacks, out, *, stream=None) -> None

        Count the leftmost-first matches of the vocabulary in every haystack, into `out`.

        Args:
          haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.
          out (buffer): Writable 1-D buffer of pointer-width unsigned integers, one per haystack.
          stream (int, optional): A stream of the engine's device as an integer, or None.
        Example:
          >>> engine = sz.RegexEngine(sz.Strings(['[0-9]+', '[a-z]+']))
          >>> out = memoryview(bytearray(8)).cast('Q')
          >>> engine.counts(sz.Strings(['abc 123 de']), out)
          >>> out[0]
          3
        """
    def find(self, haystacks: Strings, matches: _Buffer | None, offsets: _Buffer, *, stream: int | None = None) -> None:
        """find(haystacks, matches, offsets, *, stream=None) -> None

        Locate every leftmost-first match, writing one row per match and one boundary per haystack.

        A capacity too small is not an error: `offsets` is filled either way, which sizes the next call.

        Args:
          haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.
          matches (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned
            integers, shaped `(capacity, 4)` - haystack index, pattern index, byte offset, byte length.
            None makes the call a pure size query.
          offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding
            len(haystacks) + 1 boundaries into `matches`, the last being the total.
          stream (int, optional): A stream of the engine's device as an integer, or None.
        Example:
          >>> engine = sz.RegexEngine(sz.Strings(['[0-9]+']))
          >>> matches = memoryview(bytearray(2 * 4 * 8)).cast('Q', (2, 4))
          >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')
          >>> engine.find(sz.Strings(['a 12 b 345']), matches, offsets)
          >>> matches[1, 2], matches[1, 3]
          (7, 3)
        """
    def replace(self, haystacks: Strings, replacements: Strings, target: _Buffer | None, offsets: _Buffer) -> None:
        r"""replace(haystacks, replacements, target, offsets) -> None

        Rewrite every haystack into one target buffer, one replacement per pattern.

        A replacement names groups as `$1`, `${1}` or `${name}`, `$0` being the whole match and `$$`
        one dollar. A target too small is not an error: `offsets` is filled either way. CPU only.

        Args:
          haystacks (Strings): Texts to rewrite.
          replacements (Strings): One replacement per pattern; an empty one deletes the match.
          target (buffer or None): Writable, contiguous byte buffer; None makes the call a size query.
          offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding
            len(haystacks) + 1 rewritten boundaries, the last being the total.
        Example:
          >>> engine = sz.RegexEngine(sz.Strings(['(\\w+)@(\\w+)']))
          >>> target = memoryview(bytearray(16))
          >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')
          >>> engine.replace(sz.Strings(['bob@host']), sz.Strings(['$2/$1']), target, offsets)
          >>> bytes(target[offsets[0]:offsets[1]])
          b'host/bob'
        """
    def extract(
        self, haystacks: Strings, pattern_index: int, groups: _Buffer | None, offsets: _Buffer, *, stream: int | None = None
    ) -> None:
        r"""extract(haystacks, pattern_index, groups, offsets, *, stream=None) -> None

        Locate the groups of each haystack's first match of a pattern, as Pandas `str.extract` reads them.

        A haystack with such a match gets a row for the whole match, group index zero, then one row per
        group that took part, so a group left out differs from an empty one; a haystack without one gets
        no rows. A capacity too small is not an error: `offsets` is filled either way, which sizes the
        next call. `haystacks.matched(groups)` reads the rows as strings.

        Args:
          haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.
          pattern_index (int): The pattern whose groups to locate.
          groups (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned
            integers, shaped `(capacity, 4)` - haystack index, group index, byte offset, byte length.
            None makes the call a pure size query.
          offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding
            len(haystacks) + 1 boundaries into `groups`, the last being the total.
          stream (int, optional): A stream of the engine's device as an integer, or None.
        Example:
          >>> engine = sz.RegexEngine(sz.Strings(['(\\d+)-(\\d+)']))
          >>> groups = memoryview(bytearray(3 * 4 * 8)).cast('Q', (3, 4))
          >>> offsets = memoryview(bytearray(3 * 8)).cast('Q')
          >>> haystacks = sz.Strings(['a 12-34', 'none'])
          >>> engine.extract(haystacks, 0, groups, offsets)
          >>> list(map(bytes, haystacks.matched(groups))), offsets[2]
          ([b'12-34', b'12', b'34'], 3)
        """

class SubdocumentsEngine:
    """SubdocumentsEngine(paths, format='json', *, capabilities=None, stream=None)

    Compile JSONPath or XPath queries once and pull what they select out of many documents.

    Every document is walked once, whatever the path count, and no tree is built. A match spans
    the node's exact bytes: in JSON a string with its quotes, a number or literal, or a container
    through its closing bracket; in XML an element, an attribute's value, or a text node.

    Args:
      paths (Strings): RFC 9535 JSONPath queries, or for XML, XPath 1.0 location paths along
        forward axes.
      format (str, optional): 'json' or 'xml', the grammar of every document.
      capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),
        defaulting to cpu_capabilities_enabled().
      stream (int, optional): A stream of that vendor as an integer, naming the device, or None
        for the default.
    Raises:
      ValueError: Naming the first path that breaks its grammar or leaves the supported subset.
    Example:
      >>> engine = sz.SubdocumentsEngine(sz.Strings(['$.a[?@ > 1]']))
      >>> out = memoryview(bytearray(8)).cast('Q')
      >>> engine.counts(sz.Strings(['{"a": [1, 2, 3]}']), out)
      >>> out[0]
      2
    """

    def __init__(
        self, paths: Strings, format: str = "json", *, capabilities: Capability | None = None, stream: int | None = None
    ) -> None: ...
    def counts(self, documents: Strings, out: _Buffer, *, stream: int | None = None) -> None:
        """counts(documents, out, *, stream=None) -> None

        Count the nodes of every document that some path selects, once per path, into `out`.

        Args:
          documents (Strings): Documents to walk, from `Strings.copy` on a GPU.
          out (buffer): Writable 1-D buffer of pointer-width unsigned integers, one per document.
          stream (int, optional): A stream of the engine's device as an integer, or None.
            A device engine enqueues there and returns, so `out` must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Raises:
          ValueError: Naming the first malformed document.
        Example:
          >>> engine = sz.SubdocumentsEngine(sz.Strings(['$..name']))
          >>> out = memoryview(bytearray(8)).cast('Q')
          >>> engine.counts(sz.Strings(['{"name": "a", "b": {"name": "c"}}']), out)
          >>> out[0]
          2
        """
    def find(self, documents: Strings, matches: _Buffer | None, offsets: _Buffer, *, stream: int | None = None) -> None:
        """find(documents, matches, offsets, *, stream=None) -> None

        Locate every selected node, writing one row per match and one boundary per document.

        Matches come by document, then start offset, then path, so a container precedes what it holds.
        A capacity too small is not an error: `offsets` is filled either way, which is what sizes the next
        call, and `Strings.matched` gathers the spans the rows locate.

        Args:
          documents (Strings): Documents to walk, from `Strings.copy` on a GPU.
          matches (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned
            integers, shaped `(capacity, 4)` - document index, path index, byte offset, byte length.
            None makes the call a pure size query.
          offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding
            len(documents) + 1 boundaries into `matches`, the last being the total.
          stream (int, optional): A stream of the engine's device as an integer, or None.
            A device engine enqueues there and returns, so both buffers must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Raises:
          ValueError: Naming the first malformed document.
        Example:
          >>> engine = sz.SubdocumentsEngine(sz.Strings(['/a/b/text()']), format='xml')
          >>> matches = memoryview(bytearray(2 * 4 * 8)).cast('Q', (2, 4))
          >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')
          >>> engine.find(sz.Strings(['<a><b>x</b><b>y</b></a>']), matches, offsets)
          >>> offsets[1]
          2
        """

class SubstringsEngine:
    """SubstringsEngine(needles, case_sensitivity='cased', overlap_policy='overlapping',
                     hot_states=None, matches_budget=0, *, haystacks_budget=0,
                     capabilities=None, stream=None)

    Compile a vocabulary once and stream many collections of haystacks through it.

    One Aho-Corasick pass answers every needle at once, whatever the vocabulary size.

    Args:
      needles (Strings): The vocabulary to compile; an empty needle is refused.
      case_sensitivity (str, optional): 'cased' matches bytes, 'uncased' folds both sides.
      overlap_policy (str, optional): 'overlapping', 'leftmost-longest', or 'leftmost-first'.
      hot_states (int, optional): States kept in the dense hot rows; None sizes them automatically.
      matches_budget (int, optional): Matches one round may emit, read by a device tier only.
      haystacks_budget (int, optional): Haystacks one round may carry, read by a device tier only.
      capabilities (Capability, optional): A producer's mask, like cuda_capabilities_enabled(0),
        defaulting to cpu_capabilities_enabled().
      stream (int, optional): A stream of that vendor as an integer, naming the device, or None
        for the default.
    Example:
      >>> engine = sz.SubstringsEngine(sz.Strings(['he', 'she']), case_sensitivity='cased')
      >>> out = memoryview(bytearray(8)).cast('Q')
      >>> engine.counts(sz.Strings(['hershey']), out)
      >>> out[0]
      3
    """

    def __init__(
        self,
        needles: Strings,
        case_sensitivity: str = "cased",
        overlap_policy: str = "overlapping",
        hot_states: int | None = None,
        matches_budget: int = 0,
        *,
        haystacks_budget: int = 0,
        capabilities: Capability | None = None,
        stream: int | None = None,
    ) -> None: ...
    @property
    def report(self) -> dict[str, int]:
        """What the last round found, as a dict of four counts.

        `matches_emitted` is the truth whatever the outputs could hold, `matches_stored` what was
        written, `target_length` what a rewrite needs, and `shortfall` what did not fit. On a device
        engine these are written by the device, so join the stream before reading them.

        Example:
          >>> engine = sz.SubstringsEngine(sz.Strings(['he', 'she']))
          >>> out = memoryview(bytearray(8)).cast('Q')
          >>> engine.counts(sz.Strings(['hershey']), out)
          >>> engine.report['matches_emitted']
          3
        """
    def counts(self, haystacks: Strings, out: _Buffer, *, stream: int | None = None) -> None:
        """counts(haystacks, out, *, stream=None) -> None

        Count the matches of every needle in every haystack, into `out`.

        Args:
          haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.
          out (buffer): Writable 1-D buffer of pointer-width unsigned integers, one per haystack.
          stream (int, optional): A stream of the engine's device as an integer, or None.
            A device engine enqueues there and returns, so `out` must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Example:
          >>> engine = sz.SubstringsEngine(sz.Strings(['he', 'she']))
          >>> out = memoryview(bytearray(8)).cast('Q')
          >>> engine.counts(sz.Strings(['hershey']), out)
          >>> out[0]
          3
        """
    def find(self, haystacks: Strings, matches: _Buffer | None, offsets: _Buffer, *, stream: int | None = None) -> None:
        """find(haystacks, matches, offsets, *, stream=None) -> None

        Locate every match, writing one row per match and one boundary per haystack.

        A capacity too small is not an error: `offsets` and `report` are filled either way, which is
        what sizes the next call.

        Args:
          haystacks (Strings): Texts to search, from `Strings.copy` on a GPU.
          matches (buffer or None): Writable, fully contiguous 2-D buffer of pointer-width unsigned
            integers, shaped `(capacity, 4)` - haystack index, needle index, byte offset, byte length.
            None makes the call a pure size query.
          offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding
            len(haystacks) + 1 boundaries into `matches`, the last being the total.
          stream (int, optional): A stream of the engine's device as an integer, or None.
            A device engine enqueues there and returns, so both buffers must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Example:
          >>> engine = sz.SubstringsEngine(sz.Strings(['he', 'she']))
          >>> matches = memoryview(bytearray(3 * 4 * 8)).cast('Q', (3, 4))
          >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')
          >>> engine.find(sz.Strings(['hershey']), matches, offsets)
          >>> offsets[1]
          3
        """
    def replace(
        self,
        haystacks: Strings,
        replacements: Strings,
        target: _Buffer | None,
        offsets: _Buffer,
        *,
        stream: int | None = None,
    ) -> None:
        """replace(haystacks, replacements, target, offsets, *, stream=None) -> None

        Rewrite every haystack into one target buffer, one replacement per needle.

        Needs a leftmost policy, since a substitution over matches that share bytes is not a function.
        A target too small is not an error: `offsets` and `report` are filled either way.

        Args:
          haystacks (Strings): Texts to rewrite, from `Strings.copy` on a GPU.
          replacements (Strings): One replacement per needle; an empty one deletes the match. On a GPU
            engine, from `Strings.copy` too.
          target (buffer or None): Writable, contiguous byte buffer; None makes the call a size query.
          offsets (buffer): Writable, contiguous 1-D buffer of pointer-width unsigned integers holding
            len(haystacks) + 1 rewritten boundaries, the last being the total.
          stream (int, optional): A stream of the engine's device as an integer, or None.
            A device engine enqueues there and returns, so both buffers must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Example:
          >>> engine = sz.SubstringsEngine(sz.Strings(['he', 'she']), overlap_policy='leftmost-longest')
          >>> target = memoryview(bytearray(16))
          >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')
          >>> engine.replace(sz.Strings(['hershey']), sz.Strings(['HE', 'SHE']), target, offsets)
          >>> bytes(target[offsets[0]:offsets[1]])
          b'HErSHEy'
        """
    def bm25_scores(
        self,
        haystacks: Strings,
        needle_weights: _Buffer,
        out: _Buffer,
        *,
        document_lengths: _Buffer | None = None,
        term_frequency_saturation: float = 1.2,
        length_normalization: float = 0.75,
        average_document_length: float = 0.0,
        stream: int | None = None,
    ) -> None:
        """bm25_scores(haystacks, needle_weights, out, *, document_lengths=None,
                    term_frequency_saturation=1.2, length_normalization=0.75,
                    average_document_length=0.0, stream=None) -> None

        Score every haystack against the vocabulary, which is the query, into `out`.

        Term frequencies are raw overlapping counts, so the engine's own overlap policy does not apply.

        Args:
          haystacks (Strings): Documents to score, from `Strings.copy` on a GPU.
          needle_weights (buffer): One 32-bit float IDF or boost per needle.
          out (buffer): Writable 1-D buffer of 32-bit floats, one score per haystack.
          document_lengths (buffer, optional): One 32-bit float length per haystack; byte lengths
            when None.
          term_frequency_saturation (float, optional): The literature's k1.
          length_normalization (float, optional): The literature's b, in [0, 1].
          average_document_length (float, optional): Corpus-wide mean of the lengths above; required
            whenever length_normalization is positive.
          stream (int, optional): A stream of the engine's device as an integer, or None.
            A device engine enqueues there and returns, so `out` must be device-reachable
            and read only after the caller joins the stream; a host engine ignores it.
        Example:
          >>> engine = sz.SubstringsEngine(sz.Strings(['cat', 'dog']))
          >>> weights = memoryview(bytearray(2 * 4)).cast('f')
          >>> weights[0], weights[1] = 1.0, 1.0
          >>> out = memoryview(bytearray(2 * 4)).cast('f')
          >>> engine.bm25_scores(sz.Strings(['cat and dog', 'nothing here']), weights, out,
          ...                    average_document_length=10.0)
          >>> out[1]
          0.0
        """

class Strings:
    """Strings(sequence=())

    An immutable sequence of byte strings on one tape - their bytes back to back and an array of
    offsets, the Apache Arrow `string` or `large_string` layout - like a `tuple` of `bytes` that
    costs two allocations rather than one per element.

    Args:
      sequence (iterable | pyarrow.Array, optional): The strings to hold. An Arrow string or
        binary array is borrowed in place; any other iterable of `str` or bytes-like objects
        is copied once.

    Slicing with a step of one borrows the tape, while any other step, `sorted`, `shuffled` and
    `sample` gather into a new one, and `argsort` returns the order without copying, as `find`,
    `hashes` and `utf8_count` return one number per string in a single pass over the tape.

    Example:
      >>> names = sz.Strings(['banana', 'apple', 'cherry', 'date'])
      >>> [str(x) for x in names.sorted()]
      ['apple', 'banana', 'cherry', 'date']
      >>> [str(x) for x in names.sorted(top=2)]  # partial top-k is cheaper than a full sort
      ['apple', 'banana']
    """

    def __init__(self, sequence: Any = ()) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: _StringLike) -> bool: ...
    def __iter__(self) -> Iterator[String]: ...
    @overload
    def __getitem__(self, index: int) -> String: ...
    @overload
    def __getitem__(self, index: slice) -> Strings: ...
    @property
    def tape_address(self) -> int:
        """Memory address of the first byte of the tape, which the offsets count from, as an integer."""
    @property
    def tape_nbytes(self) -> int:
        """Bytes from the start of the tape to the end of its last string."""
    @property
    def offsets_address(self) -> int:
        """Memory address of the first offset, as an integer."""
    @property
    def offsets_nbytes(self) -> int:
        """Length of the offsets array in bytes, one offset per string and one past the last."""
    @property
    def offsets_are_large(self) -> bool:
        """True for 64-bit offsets, the Arrow `large_string` layout, and False for the 32-bit `string` one."""
    def shuffled(self, seed: int | None = None) -> Strings:
        """shuffled(seed=None) -> Strings

        Return a new Strings with the elements randomly permuted, like `random.sample` drawing all
        of them; the original is unchanged.

        Args:
          seed (int, optional): Seed for reproducible shuffling. Defaults to a random seed.
        Returns:
          Strings: A new shuffled collection on a tape of its own.
        Example:
          >>> sorted(map(str, sz.Strings(['a', 'b', 'c']).shuffled(seed=42))) == ['a', 'b', 'c']
          True
        """
    def sorted(
        self,
        *,
        reverse: bool = False,
        uncased: bool = False,
        top: int | None = None,
        capabilities: Capability | None = None,
    ) -> Strings:
        """sorted(*, reverse=False, uncased=False, top=None, capabilities=None) -> Strings

        Return a new, stably sorted Strings, like the built-in `sorted`; the original is unchanged.

        Args:
          reverse (bool, optional): Sort in descending order. Defaults to False.
          uncased (bool, optional): Order by Unicode case-folding. Defaults to False.
          top (int, optional): Keep only the `top` smallest elements, or largest if reversed. Defaults to None, keeping all.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: A new, sorted collection on a tape of its own.
        Example:
          >>> list(map(str, sz.Strings(['banana', 'apple', 'cherry']).sorted()))
          ['apple', 'banana', 'cherry']
        """
    def argsort(
        self,
        *,
        reverse: bool = False,
        uncased: bool = False,
        top: int | None = None,
        out: _Buffer | None = None,
        capabilities: Capability | None = None,
    ) -> tuple[int, ...] | _Buffer:
        """argsort(*, reverse=False, uncased=False, top=None, out=None, capabilities=None) -> tuple | buffer

        Return the stable permutation of indices that sorts the Strings, like
        `sorted(range(len(strings)), key=strings.__getitem__)`, copying no strings.

        Args:
          reverse (bool, optional): Sort in descending order. Defaults to False.
          uncased (bool, optional): Order by Unicode case-folding. Defaults to False.
          top (int, optional): Keep only the `top` leading indices. Defaults to None, keeping all.
          out (buffer, optional): Writable, C-contiguous buffer of pointer-width unsigned integers like numpy.uintp to receive the indices with zero allocation. Defaults to None.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple[int, ...]: The sorting permutation, or `out` itself when an `out` buffer is given.
        Example:
          >>> sz.Strings(['banana', 'apple', 'cherry']).argsort()
          (1, 0, 2)
        """
    def find(
        self, needle: _StringLike, *, out: _Buffer | None = None, capabilities: Capability | None = None
    ) -> tuple[int, ...] | _Buffer:
        """find(needle, *, out=None, capabilities=None) -> tuple | buffer

        Find the first occurrence of `needle` in every string, like `str.find` over each, counting bytes
        as `bytes.find` does, in one pass over the whole tape.

        Args:
          needle (String or str or bytes): The substring to find; an empty one matches at 0.
          out (buffer, optional): Writable, C-contiguous buffer of pointer-width signed integers like
            numpy.intp to receive the offsets with zero allocation. Defaults to None.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple[int, ...]: The byte offset of the first match in each string, or -1 where it holds none,
            or `out` itself when an `out` buffer is given.
        Example:
          >>> sz.Strings(['hello', 'world', 'help']).find('el')
          (1, -1, 1)
        """
    def hashes(
        self, seed: int = 0, *, out: _Buffer | None = None, capabilities: Capability | None = None
    ) -> tuple[int, ...] | _Buffer:
        """hashes(seed=0, *, out=None, capabilities=None) -> tuple | buffer

        Hash every string under one seed, each equal to `sz.hash(text, seed)`, like the built-in `hash`
        over each but stable across runs and platforms, for group-by keys, joins and sharding.

        Args:
          seed (int, optional): The seed every string is hashed under, defaulting to 0.
          out (buffer, optional): Writable, C-contiguous buffer of 64-bit unsigned integers like
            numpy.uint64 to receive the hashes with zero allocation. Defaults to None.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple[int, ...]: One unsigned 64-bit hash per string, or `out` itself when an `out` buffer is
            given.
        Example:
          >>> sz.Strings(['a', 'b']).hashes(7) == (sz.hash('a', 7), sz.hash('b', 7))
          True
        """
    def utf8_count(
        self, *, out: _Buffer | None = None, capabilities: Capability | None = None
    ) -> tuple[int, ...] | _Buffer:
        r"""utf8_count(*, out=None, capabilities=None) -> tuple | buffer

        Count the codepoints of every string, like `len(text.decode())` over each, validating every one
        as UTF-8 in the same pass over the whole tape.

        Args:
          out (buffer, optional): Writable, C-contiguous buffer of pointer-width unsigned integers like
            numpy.uintp to receive the counts with zero allocation. Defaults to None.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple[int, ...]: The codepoints of each string, or `out` itself when an `out` buffer is given.
        Raises:
          ValueError: Naming the index of the first string holding ill-formed UTF-8.
        Example:
          >>> sz.Strings(['hello', 'caf\u00e9', '']).utf8_count()
          (5, 4, 0)
        """
    def sample(self, size: int, seed: int | None = None) -> Strings:
        """sample(size, seed=None) -> Strings

        Return a new Strings with `size` elements drawn at random with replacement, like
        `random.choices(strings, k=size)`.

        Args:
          size (int): Number of elements to draw.
          seed (int, optional): Seed for reproducible sampling. Defaults to a random seed.
        Returns:
          Strings: A new collection of the sampled elements on a tape of its own.
        Example:
          >>> list(map(bytes, sz.Strings(['a', 'a', 'a']).sample(2)))
          [b'a', b'a']
        """
    def matched(self, matches: _Buffer) -> Strings:
        """matched(matches) -> Strings

        Return a new Strings holding the bytes each match row spans, in row order, gathering what
        a `find` of `SubstringsEngine` or `SubdocumentsEngine` located in this collection.

        Args:
          matches (buffer): Contiguous 2-D buffer of pointer-width unsigned integers shaped
            `(count, 4)` - haystack index, needle index, byte offset, byte length.
        Returns:
          Strings: A new collection of the matched spans on a tape of its own.
        Raises:
          IndexError: When a row reaches outside this collection.
        Example:
          >>> documents = sz.Strings(['{"a": [1, "x"]}'])
          >>> engine = sz.SubdocumentsEngine(sz.Strings(['$.a[*]']))
          >>> matches = memoryview(bytearray(2 * 4 * 8)).cast('Q', (2, 4))
          >>> offsets = memoryview(bytearray(2 * 8)).cast('Q')
          >>> engine.find(documents, matches, offsets)
          >>> list(map(bytes, documents.matched(matches)))
          [b'1', b'"x"']
        """
    def intersect(
        self, other: Strings, *, seed: int = 0, capabilities: Capability | None = None
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """intersect(other, *, seed=0, capabilities=None) -> tuple[tuple[int, ...], tuple[int, ...]]

        Return the positions of strings present in both collections.
        Each distinct shared value is matched exactly once, even if either side has duplicates.

        Args:
          other (Strings): The collection to intersect with.
          seed (int, optional): Seed for the hash table to avoid attacks. Defaults to 0.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple[tuple[int, ...], tuple[int, ...]]: Parallel position tuples - `result[0][i]` in
          this collection and `result[1][i]` in `other` point to equal strings.
        Example:
          >>> ours, theirs = sz.Strings(['banana', 'apple', 'cherry']).intersect(
          ...     sz.Strings(['cherry', 'banana']))
          >>> sorted(ours)
          [0, 2]
        """
    def copy(self, capabilities: int, stream: int | None = None) -> Strings:
        """copy(capabilities, stream=None) -> Strings

        Copy the strings into a tape in the unified memory of `capabilities`, which engines of those
        capabilities read in place and the host reads as a normal read-only Strings.

        Args:
          capabilities (int): A producer's mask, like sz.cuda_capabilities_enabled(0).
          stream (int, optional): A stream of that vendor as an integer, naming the device the tape is
            made on, or None for the default.
        Returns:
          Strings: A tape owning its unified memory, freed with it.
        Example:
          >>> tape = sz.Strings(['kitten', 'sitting']).copy(sz.cpu_capabilities_enabled())
          >>> [str(text) for text in tape]
          ['kitten', 'sitting']
        """
    def translate(
        self, table: str | bytes | dict[str, str], *, capabilities: Capability | None = None
    ) -> Strings:
        """translate(table, *, capabilities=None) -> Strings

        Map every byte of every string through a look-up table, like `bytes.translate` over each.

        Args:
          table (str or bytes or dict): A 256-byte table, or a dict mapping characters below U+0100.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: A new collection, each string as long as its source.
        Raises:
          ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.
        Example:
          >>> [str(text) for text in sz.Strings(['cab', 'bee']).translate({'a': 'A', 'b': 'B'})]
          ['cAB', 'Bee']
        """
    def escape(self, kind: str = "json", *, capabilities: Capability | None = None) -> Strings:
        r"""escape(kind='json', *, capabilities=None) -> Strings

        Escape every string for a text format, in one pass over the whole collection.

        Kinds:
          json: A JSON string body, as `json.dumps(text, ensure_ascii=False)` writes it unquoted.
          json_ascii: The same with every non-ASCII codepoint as `\uXXXX`, like `json.dumps(text)`.
          hex: Two lowercase digits per byte, as `bytes.hex` writes them.
          html: `&`, `<`, `>`, `"` and `'` as entities, like `html.escape`, with `&#39;` for `'`.
          url: Percent-encoding, as `urllib.parse.quote(text, safe='')` writes it.
          console: Controls, ill-formed UTF-8 and bidirectional overrides as `\n`, `\xHH` or `\uXXXX`.
          csv: A field quoted only around a comma, a quote, CR or LF, as `csv.writer` writes it.

        Args:
          kind (str, optional): One of the kinds above, defaulting to 'json'.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: A new collection of the escaped strings.
        Example:
          >>> [str(text) for text in sz.Strings(['<b>', 'Tom & Jerry']).escape('html')]
          ['&lt;b&gt;', 'Tom &amp; Jerry']
        """
    def unescape(self, kind: str = "json", *, capabilities: Capability | None = None) -> Strings:
        """unescape(kind='json', *, capabilities=None) -> Strings

        Reverse `escape` for every string, in one pass over the whole collection. The json kinds
        decode what `json.loads` decodes in a string body, html what `html.unescape` decodes of the
        five XML entities and numeric references, url what `urllib.parse.unquote_to_bytes` decodes,
        and csv one field as `csv.reader` reads it.

        Args:
          kind (str, optional): One of the kinds `escape` takes, defaulting to 'json'.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: A new collection of the unescaped strings.
        Raises:
          ValueError: Naming the index of the first string holding a malformed escape.
        Example:
          >>> [str(text) for text in sz.Strings(['a%20b', '100%25']).unescape('url')]
          ['a b', '100%']
        """
    def normalize(self, form: str = "NFC", *, capabilities: Capability | None = None) -> Strings:
        r"""normalize(form='NFC', *, capabilities=None) -> Strings

        Bring every string into a Unicode normalization form, in one pass over the whole collection,
        like `unicodedata.normalize` over each. 'NFKC_Casefold' is the Unicode toNFKC_Casefold, the
        keys search engines match on: NFKC with full case folding and no default-ignorable codepoints.

        Args:
          form (str, optional): 'NFC', 'NFD', 'NFKC', 'NFKD' or 'NFKC_Casefold', defaulting to 'NFC'.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: A new collection of the normalized strings.
        Example:
          >>> [str(text) for text in sz.Strings(['\ufb01', 'Stra\u00dfe']).normalize('NFKC_Casefold')]
          ['fi', 'strasse']
        """
    def to_json(self, ensure_ascii: bool = True, *, capabilities: Capability | None = None) -> bytes:
        r"""to_json(ensure_ascii=True, *, capabilities=None) -> bytes

        Dump the strings as one compact JSON array, the bytes of
        `json.dumps(strings, ensure_ascii=ensure_ascii, separators=(',', ':')).encode()`.

        Args:
          ensure_ascii (bool, optional): Write every non-ASCII codepoint as `\uXXXX`, defaulting to True.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          bytes: The UTF-8 JSON text.
        Example:
          >>> sz.Strings(['Ada', 'Grace']).to_json()
          b'["Ada","Grace"]'
        """

class String:
    """String(source)

    Immutable byte-string/slice class with SIMD and SWAR-accelerated operations.
    Provides high-performance byte-string operations using modern CPU instructions.

    Args:
      source (str, bytes, bytearray, or buffer): Source data to wrap.

    Methods:
      - find(), rfind(): Fast substring search with SIMD acceleration
      - count(): Count occurrences with optional overlap support
      - split(), rsplit(): String splitting with various separators
      - contains(): Fast membership testing
      - translate(): Byte-level translations with lookup tables

    Example:
      >>> text = sz.String('the quick brown fox')
      >>> text.find('brown')        # SIMD-accelerated substring search
      10
      >>> str(text[10:15])          # slicing returns a zero-copy view, not a new buffer
      'brown'
    """

    def __init__(self, source: Any) -> None: ...
    def __len__(self) -> int: ...
    def __contains__(self, item: _StringLike) -> bool: ...
    def __add__(self, other: _StringLike) -> String: ...
    def __hash__(self) -> int: ...
    @overload
    def __getitem__(self, index: int) -> str: ...
    @overload
    def __getitem__(self, index: slice) -> String: ...
    @property
    def address(self) -> int:
        """Memory address of the first byte of the string, as an integer.

        Enables zero-copy interop, e.g. with PyArrow or ctypes.

        Example:
          >>> isinstance(sz.String('abc').address, int)
          True
        """
    @property
    def nbytes(self) -> int:
        """Length of the string in bytes.

        Example:
          >>> sz.String('abc').nbytes
          3
        """
    def contains(
        self, substring: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> bool:
        """Check if a string contains a substring.

        Args:
          text (String or str or bytes): The string object.
          substring (str): The substring to search for.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          bool: True if the substring is found, False otherwise.

        Example:
          >>> sz.String('hello').contains('ell')
          True
        """
    def count(
        self,
        substring: _StringLike,
        start: int = 0,
        end: int | None = None,
        allowoverlap: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> int:
        """Count the occurrences of a substring.

        Args:
          text (String or str or bytes): The string object.
          substring (str): The substring to count.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          allowoverlap (bool, optional): Count overlapping occurrences, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The number of occurrences of the substring.

        Example:
          >>> sz.String('banana').count('a')
          3
        """
    def splitlines(
        self, keeplinebreaks: bool = False, maxsplit: int = -1, *, capabilities: Capability | None = None
    ) -> Strings:
        r"""Split a string by line breaks, like `str.splitlines`.

        Args:
          text (String or str or bytes): The string object.
          keeplinebreaks (bool, optional): Include line breaks in the results, defaulting to False.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: The lines, copied onto a tape of their own.

        Example:
          >>> list(map(str, sz.String('a\nb\nc').splitlines()))
          ['a', 'b', 'c']
        """
    def startswith(
        self, prefix: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> bool:
        """Check if a string starts with a given prefix.

        Args:
          text (String or str or bytes): The string object.
          prefix (str): The prefix to check.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          bool: True if the string starts with the prefix, False otherwise.

        Example:
          >>> sz.String('hello').startswith('he')
          True
        """
    def endswith(
        self, suffix: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> bool:
        """Check if a string ends with a given suffix.

        Args:
          text (String or str or bytes): The string object.
          suffix (str): The suffix to check.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          bool: True if the string ends with the suffix, False otherwise.

        Example:
          >>> sz.String('hello').endswith('lo')
          True
        """
    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Decode the bytes into a Unicode string with a given encoding.

        Args:
          text (String or str or bytes): The string object.
          encoding (str, optional): The encoding to use, defaulting to 'utf-8'.
          errors (str, optional): Error handling scheme, defaulting to 'strict'.
        Returns:
          str: The decoded Unicode string.
        Raises:
          UnicodeDecodeError: If decoding fails.

        Example:
          >>> sz.String('abc').decode() == 'abc'
          True
        """
    def hash(self, seed: int = 0, /, *, capabilities: Capability | None = None) -> int:
        """Compute the hash value of the string.

        This function can be called as a method on a String object or as a standalone function.
        Args:
          text (String or str or bytes): The string to hash, positional-only when standalone.
          seed (int, optional): The seed value for hashing. Defaults to 0. Can be positional or keyword.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The hash value as an unsigned 64-bit integer. This differs from Python's
               built-in `hash()` which returns a `Py_hash_t` and may be platform-dependent.
        Raises:
          TypeError: If the argument is not string-like or incorrect number of arguments is provided.
        Signature:
          >>> def hash(text, seed=0, /, *, capabilities=None) -> int: ...
        """
    def hash_multiseed(
        self, seeds: _Buffer, /, out: _Buffer | None = None, *, capabilities: Capability | None = None
    ) -> tuple[int, ...] | None:
        """Hash one string under many seeds at once.

        Equivalent to `tuple(hash(text, s) for s in seeds)`, but normalizes the input into AES
        blocks once and replays the cheap per-seed rounds - markedly faster for short strings under
        many seeds (feature hashing, Count-Min sketches, Bloom/cuckoo filters, MinHash/LSH).

        Args:
          text (String or str or bytes): The string to hash, positional-only when standalone.
          seeds (buffer of uint64): Contiguous 64-bit seeds, e.g. numpy.uint64 array or array('Q', ...).
               Plain int lists are not accepted - wrap them in array('Q', seeds) or numpy.
          out (writable uint64 buffer, optional): A contiguous buffer of at least len(seeds) elements.
               If given, it is filled in place and None returned; otherwise a tuple of ints is returned.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple[int, ...] | None: Tuple of 64-bit hashes, or None when writing into `out`.

        Example:
          >>> def hash_multiseed(text, seeds, /, out=None, *, capabilities=None) -> tuple | None: ...
        """
    def sha256(self, *, capabilities: Capability | None = None) -> bytes:
        """Compute SHA256 cryptographic hash of the input data.

        This function can be called as a method on a String object or as a standalone function.
        Args:
          text (String or str or bytes): The input data to hash.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          bytes: The 32-byte (256-bit) SHA256 digest.
        Raises:
          TypeError: If the argument is not string-like or incorrect number of arguments is provided.

        Example:
          >>> digest = sz.String('abc').sha256().hex()
          >>> digest[:8]
          'ba7816bf'
        """
    def lstrip(self, chars: _StringLike | None = None, *, capabilities: Capability | None = None) -> String:
        """Remove leading characters from a string.

        Args:
          text (String or str or bytes): The string object.
          chars (str, optional): Characters to remove, defaulting to whitespace.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          String: A new string with leading characters removed.

        Example:
          >>> sz.String('  hi').lstrip() == 'hi'
          True
        """
    def rstrip(self, chars: _StringLike | None = None, *, capabilities: Capability | None = None) -> String:
        """Remove trailing characters from a string.

        Args:
          text (String or str or bytes): The string object.
          chars (str, optional): Characters to remove, defaulting to whitespace.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          String: A new string with trailing characters removed.

        Example:
          >>> sz.String('hi  ').rstrip() == 'hi'
          True
        """
    def strip(self, chars: _StringLike | None = None, *, capabilities: Capability | None = None) -> String:
        """Remove leading and trailing characters from a string.

        Args:
          text (String or str or bytes): The string object.
          chars (str, optional): Characters to remove, defaulting to whitespace.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          String: A new string with leading and trailing characters removed.

        Example:
          >>> sz.String('  hi  ').strip() == 'hi'
          True
        """
    def join(self, strings: Strings | Iterable[str | bytes], /, *, capabilities: Capability | None = None) -> String:
        """join(strings, /, *, capabilities=None) -> String

        Concatenate the strings with this one between every two, like `str.join`, measuring the result
        first and writing it once.

        Args:
          strings (Strings or Iterable[str | bytes]): The strings to join; a `Strings` is read in place.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          String: The joined bytes.
        Example:
          >>> str(sz.String(', ').join(['Ada', 'Grace', 'Linus']))
          'Ada, Grace, Linus'
        """
    def find(
        self, substring: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the first occurrence of a substring.

        Args:
          text (String or str or bytes): The string object.
          substring (str): The substring to find.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The index of the first occurrence, or -1 if not found.

        Example:
          >>> sz.String('hello').find('l')
          2
        """
    def index(
        self, substring: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the first occurrence of a substring or raise an error if not found.

        Args:
          text (String or str or bytes): The string object.
          substring (str): The substring to find.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The index of the first occurrence.
        Raises:
          ValueError: If the substring is not found.

        Example:
          >>> sz.String('hello').index('l')
          2
        """
    def partition(
        self, separator: _StringLike, *, capabilities: Capability | None = None
    ) -> tuple[String, String, String]:
        """Split the string into a 3-tuple around the first occurrence of a separator.

        Args:
          text (String or str or bytes): The string object.
          separator (str): The separator to partition by.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple: A 3-tuple (head, separator, tail), or `(self, '', '')` if the separator is not found.

        Example:
          >>> tuple(map(str, sz.String('a=b=c').partition('=')))
          ('a', '=', 'b=c')
        """
    def split(
        self,
        separator: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> Strings:
        """Split a string by a separator, like `str.split`.

        Args:
          text (String or str or bytes): The string object.
          separator (str): The separator to split by; an empty one leaves the text whole.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include the separator in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: The parts, copied onto a tape of their own.

        Example:
          >>> list(map(str, sz.String('a,b,c').split(',')))
          ['a', 'b', 'c']
        """
    def rfind(
        self, substring: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the last occurrence of a substring.

        Args:
          text (String or str or bytes): The string object.
          substring (str): The substring to find.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The index of the last occurrence, or -1 if not found.

        Example:
          >>> sz.String('hello').rfind('l')
          3
        """
    def rindex(
        self, substring: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the last occurrence of a substring or raise an error if not found.

        Args:
          text (String or str or bytes): The string object.
          substring (str): The substring to find.
          start (int, optional): The starting index, defaulting to 0.
          end (int, optional): The ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The index of the last occurrence.
        Raises:
          ValueError: If the substring is not found.

        Example:
          >>> sz.String('hello').rindex('l')
          3
        """
    def rpartition(
        self, separator: _StringLike, *, capabilities: Capability | None = None
    ) -> tuple[String, String, String]:
        """Split the string into a 3-tuple around the last occurrence of a separator.

        Args:
          text (String or str or bytes): The string object.
          separator (str): The separator to partition by.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          tuple: A 3-tuple (head, separator, tail). If the separator is not found, returns ('', '', self).

        Example:
          >>> tuple(map(str, sz.String('a=b=c').rpartition('=')))
          ('a=b', '=', 'c')
        """
    def rsplit(
        self,
        separator: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> Strings:
        """Split a string by a separator starting from the end, like `str.rsplit`.

        Args:
          text (String or str or bytes): The string object.
          separator (str): The separator to split by; an empty one leaves the text whole.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include the separator in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: The parts, copied onto a tape of their own.

        Example:
          >>> list(map(str, sz.String('a,b,c').rsplit(',')))
          ['a', 'b', 'c']
        """
    def find_first_of(
        self, chars: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the index of the first occurrence of any character from another string.

        Args:
          text (String or str or bytes): The string object.
          chars (str): A string containing characters to search for.
          start (int, optional): Starting index, defaulting to 0.
          end (int, optional): Ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: Index of the first matching character, or -1 if none found.

        Example:
          >>> sz.String('hello').find_first_of('aeiou')
          1
        """
    def find_last_of(
        self, chars: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the index of the last occurrence of any character from another string.

        Args:
          text (String or str or bytes): The string object.
          chars (str): A string containing characters to search for.
          start (int, optional): Starting index, defaulting to 0.
          end (int, optional): Ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: Index of the last matching character, or -1 if none found.

        Example:
          >>> sz.String('hello').find_last_of('aeiou')
          4
        """
    def find_first_not_of(
        self, chars: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the index of the first character not in another string.

        Args:
          text (String or str or bytes): The string object.
          chars (str): A string containing characters to exclude.
          start (int, optional): Starting index, defaulting to 0.
          end (int, optional): Ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: Index of the first non-matching character, or -1 if all match.

        Example:
          >>> sz.String('hello').find_first_not_of('he')
          2
        """
    def find_last_not_of(
        self, chars: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Find the index of the last character not in another string.

        Args:
          text (String or str or bytes): The string object.
          chars (str): A string containing characters to exclude.
          start (int, optional): Starting index, defaulting to 0.
          end (int, optional): Ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: Index of the last non-matching character, or -1 if all match.

        Example:
          >>> sz.String('hello').find_last_not_of('lo')
          1
        """
    def count_byteset(
        self, chars: _StringLike, start: int = 0, end: int | None = None, *, capabilities: Capability | None = None
    ) -> int:
        """Count the occurrences of any character from a set of characters.

        Args:
          text (String or str or bytes): The string object.
          chars (str): A string containing characters to count.
          start (int, optional): Starting index, defaulting to 0.
          end (int, optional): Ending index, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: The number of occurrences of any character from the set.

        Example:
          >>> sz.String('hello world').count_byteset('lo')
          5
        """
    def split_byteset(
        self,
        separators: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> Strings:
        """Split a string by a set of character separators.

        Args:
          text (String or str or bytes): The string object.
          separators (str): A string containing separator characters.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include separators in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: The parts, copied onto a tape of their own.

        Example:
          >>> list(map(str, sz.String('a,b;c').split_byteset(',;')))
          ['a', 'b', 'c']
        """
    def rsplit_byteset(
        self,
        separators: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> Strings:
        """Split a string by a set of character separators in reverse order.

        Args:
          text (String or str or bytes): The string object.
          separators (str): A string containing separator characters.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include separators in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Strings: The parts, copied onto a tape of their own.

        Example:
          >>> list(map(str, sz.String('a,b;c').rsplit_byteset(',;')))
          ['a', 'b', 'c']
        """
    def split_iter(
        self,
        separator: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> FindSplits:
        """Create an iterator for splitting a string by a separator.

        Args:
          text (String or str or bytes): The string object.
          separator (str): The separator to split by; an empty one leaves the text whole.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include separator in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding split substrings.

        Example:
          >>> # Stream parts lazily instead of materializing a list (that is what split() is for):
          >>> sum(1 for _ in sz.String('a,b,c').split_iter(','))
          3
        """
    def rsplit_iter(
        self,
        separator: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> FindSplits:
        """Create an iterator for splitting a string by a separator in reverse order.

        Args:
          text (String or str or bytes): The string object.
          separator (str): The separator to split by; an empty one leaves the text whole.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include separator in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding split substrings in reverse.

        Example:
          >>> # Iterates from the end; the first yielded part is the last field:
          >>> str(next(iter(sz.String('a/b/c').rsplit_iter('/'))))
          'c'
        """
    def split_byteset_iter(
        self,
        separators: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> FindSplits:
        """Create an iterator for splitting a string by a set of character separators.

        Args:
          text (String or str or bytes): The string object.
          separators (str): A string containing separator characters.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include separators in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding split substrings.

        Example:
          >>> # Splits on ANY byte in the set, streamed lazily:
          >>> str(next(iter(sz.String('a,b;c').split_byteset_iter(',;'))))
          'a'
        """
    def rsplit_byteset_iter(
        self,
        separators: _StringLike,
        maxsplit: int = -1,
        keepseparator: bool = False,
        skip_empty: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> FindSplits:
        """Create an iterator for splitting a string by a set of character separators in reverse order.

        Args:
          text (String or str or bytes): The string object.
          separators (str): A string containing separator characters.
          maxsplit (int, optional): Maximum number of splits, defaulting to no limit.
          keepseparator (bool, optional): Include separators in results, defaulting to False.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding split substrings in reverse.

        Example:
          >>> # Reverse byteset split; first yielded part is the last field:
          >>> str(next(iter(sz.String('a,b;c').rsplit_byteset_iter(',;'))))
          'c'
        """
    def utf8_count(self, *, capabilities: Capability | None = None) -> int:
        r"""Count the number of UTF-8 characters in a string.

        Unlike len() which returns bytes, this counts actual Unicode characters,
        handling multi-byte UTF-8 sequences correctly.

        Args:
          text (String or str or bytes): The string object.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          int: Number of UTF-8 characters in the string.

        Example:
          >>> sz.utf8_count('hello')  # 5 ASCII chars = 5
          5
          >>> sz.utf8_count('\xc3\xa9')  # 1 e-acute char = 1
          1
        """
    def utf8_split_newlines(
        self, skip_empty: bool = False, with_separators: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8SplitNewlines:
        r"""Create an iterator over the lines, the content between Unicode newlines.

        Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.
        Unlike splitlines(), this returns an iterator for memory-efficient processing.

        Args:
          text (String or str or bytes): The string object.
          skip_empty (bool, optional): Skip empty lines, defaulting to False.
          with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding the lines as String objects.

        Recognized newlines:
          LF (\n), VT (\v), FF (\f), CR (\r), NEL (U+0085),
          LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\r\n)

        Example:
          >>> sum(1 for _ in sz.String('a\nb').utf8_split_newlines())
          2
        """
    def utf8_newlines(
        self, skip_empty: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8Newlines:
        r"""Create an iterator over the Unicode newline separators, one per codepoint, CR+LF kept together.

        Uses SIMD-accelerated detection of all 7 Unicode newline characters plus CRLF.
        Yields each newline separator; utf8_split_newlines() yields the lines between them.

        Args:
          text (String or str or bytes): The string object.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding the newline separators as String objects.

        Recognized newlines:
          LF (\n), VT (\v), FF (\f), CR (\r), NEL (U+0085),
          LINE SEPARATOR (U+2028), PARAGRAPH SEPARATOR (U+2029), CRLF (\r\n)

        Example:
          >>> sum(1 for _ in sz.String('a\nb').utf8_newlines())
          1
        """
    def utf8_split_whitespaces(
        self, skip_empty: bool = False, with_separators: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8SplitWhitespaces:
        """Create an iterator over the tokens, the content between Unicode whitespace.

        Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.
        Splits on each whitespace codepoint; pass skip_empty=True for str.split()-style tokens.

        Args:
          text (String or str or bytes): The string object.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding the non-whitespace tokens as String objects.

        Recognized whitespace:
          ASCII: TAB, LF, VT, FF, CR, SPACE
          Latin-1: NEXT LINE, NO-BREAK SPACE
          General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.
          CJK: IDEOGRAPHIC SPACE (U+3000)

        Example:
          >>> sum(1 for _ in sz.String('foo bar baz').utf8_split_whitespaces())
          3
        """
    def utf8_whitespaces(
        self, skip_empty: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8Whitespaces:
        """Create an iterator over Unicode whitespace separators, one per codepoint, CR+LF kept together.

        Uses SIMD-accelerated detection of all 25 Unicode White_Space characters.
        Yields each whitespace codepoint; utf8_split_whitespaces() yields the tokens between them.

        Args:
          text (String or str or bytes): The string object.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding the whitespace separators as String objects.

        Recognized whitespace:
          ASCII: TAB, LF, VT, FF, CR, SPACE
          Latin-1: NEXT LINE, NO-BREAK SPACE
          General Punctuation: EN/EM QUAD/SPACE, THIN SPACE, etc.
          CJK: IDEOGRAPHIC SPACE (U+3000)

        Example:
          >>> sum(1 for _ in sz.String('foo bar').utf8_whitespaces())
          1
        """
    def utf8_split_delimiters(
        self, skip_empty: bool = False, with_separators: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8SplitDelimiters:
        """Create an iterator over the fields, the content between Unicode delimiters.

        Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and
        separator/whitespace (Z*) codepoint - the superset of utf8_split_whitespaces().
        Splits on delimiters, one separator per codepoint, and yields the segments between them.

        Args:
          text (String or str or bytes): The string object.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          with_separators (bool, optional): Interleave the separators losslessly, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding the non-delimiter segments as String objects.

        Example:
          >>> list(str(t) for t in sz.String('Hi, world').utf8_split_delimiters(skip_empty=True))
          ['Hi', 'world']
        """
    def utf8_delimiters(
        self, skip_empty: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8Delimiters:
        """Create an iterator over Unicode delimiter separators, one per codepoint, CR+LF kept together.

        Uses SIMD-accelerated detection of every punctuation (P*), symbol (S*), and
        separator/whitespace (Z*) codepoint. Yields each delimiter codepoint;
        utf8_split_delimiters() yields the fields between them.

        Args:
          text (String or str or bytes): The string object.
          skip_empty (bool, optional): Skip empty segments, defaulting to False.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          iterator: An iterator yielding the delimiter separators as String objects.

        Example:
          >>> list(str(d) for d in sz.String('a.b').utf8_delimiters())
          ['.']
        """
    def utf8_wordbreaks(self, /, *, capabilities: Capability | None = None) -> Utf8Wordbreaks:
        """utf8_wordbreaks(string, /, *, capabilities=None)

        Return an iterator yielding words per Unicode UAX-29 word boundary rules.
        Unlike str.split(), this is UAX-29 compliant and supports all Unicode scripts.

        Args:
            string: The input UTF-8 string to split into words.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            Iterator yielding String objects for each word.


        Example:
          >>> # Stream UAX-29 word tokens lazily:
          >>> 'world' in (str(w) for w in sz.utf8_wordbreaks('Hi, world'))
          True
        """
    def utf8_codepoints(self, /, *, capabilities: Capability | None = None) -> Utf8Codepoints:
        """utf8_codepoints(string, /, *, capabilities=None)

        Return an iterator yielding Unicode code points as int, decoded from UTF-8.
        Ill-formed bytes decode to U+FFFD, the replacement character, so iteration is
        total and never raises on malformed input.

        Args:
            string: The input UTF-8 string to decode into code points.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            Iterator yielding int code points, one per Unicode scalar value.


        Example:
          >>> list(sz.utf8_codepoints('AB'))
          [65, 66]
        """
    def utf8_graphemes(self, /, *, capabilities: Capability | None = None) -> Utf8Graphemes:
        """utf8_graphemes(string, /, *, capabilities=None)

        Return an iterator yielding grapheme clusters per Unicode UAX-29 rules.
        A grapheme cluster is a user-perceived character (e.g. a base plus combining
        marks, or an emoji ZWJ sequence) and may span several code points.

        Args:
            string: The input UTF-8 string to split into grapheme clusters.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            Iterator yielding String objects for each grapheme cluster.


        Example:
          >>> # Stream UAX-29 grapheme clusters lazily:
          >>> [str(g) for g in sz.utf8_graphemes('abc')]
          ['a', 'b', 'c']
        """
    def utf8_sentences(self, /, *, capabilities: Capability | None = None) -> Utf8Sentences:
        """utf8_sentences(string, /, *, capabilities=None)

        Return an iterator yielding sentences per Unicode UAX-29 sentence boundary rules.
        UAX-29 compliant and Unicode-script aware, unlike naive period splitting.

        Args:
            string: The input UTF-8 string to split into sentences.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            Iterator yielding String objects for each sentence.


        Example:
          >>> # Stream UAX-29 sentences lazily:
          >>> len(list(sz.utf8_sentences('Hi. Bye.'))) >= 2
          True
        """
    def utf8_linebreaks(self, /, *, capabilities: Capability | None = None) -> Utf8Linebreaks:
        r"""utf8_linebreaks(string, /, *, capabilities=None)

        Return an iterator yielding segments at line-break opportunities per UAX-14.
        Each segment ends at a line-break opportunity, a soft wrap point.
        For hard-line splitting (str.splitlines()), use utf8_split_newlines().

        Args:
            string: The input UTF-8 string to split at line-break opportunities.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            Iterator yielding String objects for each line-break-opportunity segment.


        Example:
          >>> # Stream UAX-14 line-break opportunities lazily:
          >>> len(list(sz.utf8_linebreaks('a\nb'))) >= 2
          True
        """
    def utf8_uncased_fold(self, validate: bool = False, *, capabilities: Capability | None = None) -> bytes:
        r"""Apply Unicode case folding to a UTF-8 string.

        Case folding normalizes text for uncased comparisons,
        handling one-to-many expansions, like German sharp S to 'ss'.

        Args:
            text (String or str or bytes): The input UTF-8 string.
            validate (bool): If True, validate UTF-8 before processing. Default: False.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            bytes: The case-folded UTF-8 string.

        Example:
            >>> sz.utf8_uncased_fold('HELLO')
            b'hello'
            >>> sz.utf8_uncased_fold('Stra\u00dfe')  # German sharp S
            b'strasse'
        """
    def utf8_norm(self, form: str, validate: bool = False, *, capabilities: Capability | None = None) -> bytes:
        r"""Normalize a UTF-8 string to a Unicode normalization form.

        Transforms the input into NFD, NFC, NFKD, or NFKC, mirroring `unicodedata.normalize` but
        returning raw UTF-8 bytes, or into NFKC_Casefold, the Unicode toNFKC_Casefold: NFKC with full
        case folding and no default-ignorable codepoints.

        Args:
            text (String or str or bytes): The input UTF-8 string.
            form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD', 'NFKC_Casefold'.
            validate (bool): If True, validate UTF-8 before processing. Default: False.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            bytes: The normalized UTF-8 string.

        Example:
            >>> sz.utf8_norm('caf\u00e9', 'NFD')  # decompose
            b'cafe\xcc\x81'
            >>> sz.utf8_norm('\ufb01', 'NFKD')  # fi ligature
            b'fi'
            >>> sz.utf8_norm('Stra\u00dfe\u00ad', 'NFKC_Casefold')  # folded, soft hyphen dropped
            b'strasse'
        """
    def utf8_find_denormalized(self, form: str, *, capabilities: Capability | None = None) -> int | None:
        r"""Return the byte offset of the first normalization violation, or None.

        Scans the UTF-8 string and returns the byte offset of the first codepoint
        that breaks the given normalization form (first non-Yes Quick-Check result
        or canonical-ordering violation). Returns None when the string is already
        fully normalized.

        Args:
            text (String or str or bytes): The input UTF-8 string.
            form (str): One of 'NFC', 'NFD', 'NFKC', 'NFKD', 'NFKC_Casefold'.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            int | None: Byte offset of first violation, or None if already normalized.

        Example:
            >>> sz.utf8_find_denormalized('cafe\u0301', 'NFC')  # 'e' + combining acute
            3
            >>> sz.utf8_find_denormalized('caf\u00e9', 'NFC') is None  # precomposed, already NFC
            True
        """
    def utf8_uncased_search(
        self,
        needle: _StringLike,
        start: int = 0,
        end: int | None = None,
        validate: bool = False,
        *,
        capabilities: Capability | None = None,
    ) -> int:
        r"""Find substring using Unicode uncased matching.

        Performs a uncased search using Unicode case folding rules,
        correctly handling one-to-many expansions, like 'ß' matching 'SS'.

        IMPORTANT - Type-dependent behavior:
          - str input:   start/end are CODEPOINT offsets, returns CODEPOINT offset
          - bytes input: start/end are BYTE offsets, returns BYTE offset

        Args:
            haystack (String or str or bytes): The string to search in.
            needle (String or str or bytes): The substring to find.
            start (int, optional): Starting index, defaulting to 0.
            end (int, optional): Ending index, defaulting to length.
            validate (bool): If True, validate UTF-8 before processing. Default: False.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            int: Index of the first match, or -1 if not found.

        Example:
            >>> sz.utf8_uncased_search('Hello World', 'WORLD')  # str: codepoint offset
            6
            >>> sz.utf8_uncased_search('Straße', 'STRASSE')  # 'ß' = 1 codepoint
            0
            >>> sz.utf8_uncased_search(b'Stra\xc3\x9fe', b'STRASSE')  # 'ß' = 2 bytes
            0
        """
    def utf8_uncased_matches(
        self, needle: _StringLike, /, include_overlapping: bool = False, *, capabilities: Capability | None = None
    ) -> Utf8UncasedMatches:
        """utf8_uncased_matches(haystack, needle, /, include_overlapping=False, *, capabilities=None)

        Iterate over all uncased matches of needle in haystack.

        This function uses Unicode case folding for proper handling of
        international text. The matched region length may differ from the
        needle length due to case folding expansions, like 'ß' matching 'SS'.
        An empty needle matches at every codepoint boundary, the end included.

        Args:
            haystack (String or str or bytes): The string to search in.
            needle (String or str or bytes): The pattern to find.
            include_overlapping (bool, optional): Allow overlapping matches, defaulting to False.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Yields:
            String: Each matched region as a view into the original haystack.

        Examples:
            >>> list(sz.utf8_uncased_matches('Hello HELLO hello', 'hello'))
            [sz.String('Hello'), sz.String('HELLO'), sz.String('hello')]
            >>> list(sz.utf8_uncased_matches('Straße STRASSE', 'strasse'))
            [sz.String('Straße'), sz.String('STRASSE')]
        """
    def utf8_uncased_order(
        self, other: _StringLike, validate: bool = False, *, capabilities: Capability | None = None
    ) -> int:
        """Compare two UTF-8 strings uncasedly.

        Performs lexicographical comparison using Unicode case folding, correctly handling
        one-to-many expansions, like 'Straße' equaling 'STRASSE'.

        Args:
            a (String or str or bytes): First string to compare.
            b (String or str or bytes): Second string to compare.
            validate (bool): If True, validate UTF-8 before processing. Default: False.
            capabilities (Capability, optional): Capabilities to run, by default the CPU's enabled ones.

        Returns:
            int: Negative if a < b, zero if equal, positive if a > b.

        Example:
            >>> sz.utf8_uncased_order('hello', 'HELLO')
            0
            >>> sz.utf8_uncased_order('apple', 'BANANA')
            -1
        """
    def offset_within(self, larger: String) -> int:
        """Return the raw byte offset of this StringZilla string within a larger StringZilla string.

        Args:
          text (String or str or bytes): The substring.
          larger (String): The larger string to search within.
        Returns:
          int: The byte offset where 'self' is found within 'larger', or -1 if not found.

        Example:
          >>> big = sz.String('hello world')
          >>> big[6:].offset_within(big)
          6
        """
    def write_to(self, filename: str) -> None:
        """Write the string to a file.

        Args:
          text (String or str or bytes): The string object.
          filename (str): The file path to write to.
        Returns:
          None.

        Example:
          >>> import os, tempfile
          >>> path = os.path.join(tempfile.gettempdir(), 'sz_demo.bin')
          >>> sz.String('payload').write_to(path)
        """
    def translate(
        self,
        table: str | bytes | dict[str, str],
        inplace: bool = False,
        start: int = 0,
        end: int | None = None,
        *,
        capabilities: Capability | None = None,
    ) -> None | str | bytes:
        """Perform transformation of a string using a look-up table.

        Args:
          text (String or str or bytes): The string object.
          table (str or dict): A 256-byte string, or a dict mapping characters below U+0100 to such.
          inplace (bool, optional): If True, the string is modified in place, defaulting to False.

          start (int, optional): The starting index for translation, defaulting to 0.
          end (int, optional): The ending index for translation, defaulting to the string length.
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          Union[None, str, bytes]: If inplace is False, a translated copy of the [start, end) slice is
            returned, otherwise None.
        Raises:
          ValueError: If the table is not 256 bytes long, or a dict names a character above U+00FF.
          UnicodeDecodeError: If a str input translates into malformed UTF-8.
          TypeError: If the table is not a string or dictionary.

        Example:
          >>> sz.String('abc').translate({'a': 'A'}) == b'Abc'
          True
        """
    def fill_random(
        self,
        nonce: int = 0,
        alphabet: str | bytes | None = None,
        start: int = 0,
        end: int | None = None,
        *,
        capabilities: Capability | None = None,
    ) -> None:
        """Fill a string-like buffer in place with pseudo-random bytes.

        Args:
          buffer (String or bytes-like): Writable, contiguous byte buffer (e.g., memoryview/bytearray).
          nonce (int, optional): Seed ensuring reproducible output for the same inputs, defaulting to 0.
          alphabet (str or bytes, optional): If provided, remaps random bytes to its characters.
          start (int, optional): Starting index, defaulting to 0.
          end (int, optional): Ending index, defaulting to len(buffer).
          capabilities (Capability, optional): Capabilities to run, defaulting to the CPU's enabled ones.
        Returns:
          None: Mutates the buffer slice in place.

        Example:
          >>> buf = bytearray(16)
          >>> sz.fill_random(buf)
          >>> len(buf)
          16
        """
