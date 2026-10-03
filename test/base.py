"""
Shared Unicode data loading utilities for StringZilla tests and notebooks.

This module provides functions to download and cache Unicode data files
(UCD XML, CaseFolding.txt, DerivedNormalizationProps.txt) for use in
tests and exploration notebooks.

File: test/base.py
Author: Ash Vardanian
Date: June 18, 2023
"""

import io
import math
import os
import re
import secrets
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from string import ascii_lowercase
from typing import NewType

# NumPy is available on most platforms and is required for many tests. PyPy on some platforms raises
# a weird error that is not an `ImportError`, so the naked `except` is a necessary evil.
try:
    import numpy as np  # noqa: F401

    numpy_available = True
except:  # noqa: E722
    numpy_available = False

# PyArrow is not available on most platforms; same defensive import contract as NumPy above.
try:
    import pyarrow as pa  # noqa: F401

    pyarrow_available = True
except:  # noqa: E722
    pyarrow_available = False


def _write_cache_atomically(cache_path: str, payload: bytes) -> None:
    """Write payload to cache_path through a temporary file in the same directory, then rename.

    Concurrent tests race to download the same URL, and a rename is the only way a reader is guaranteed
    to see either the previous file or a complete new one, never a half-written one.
    """
    directory = os.path.dirname(cache_path) or "."
    handle, staging_path = tempfile.mkstemp(dir=directory, suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as staging_file:
            staging_file.write(payload)
        os.replace(staging_path, cache_path)
    except BaseException:
        try:
            os.unlink(staging_path)
        except OSError:
            pass
        raise


class UnicodeDataDownloadError(Exception):
    """Raised when Unicode data files cannot be downloaded."""

    pass


class UnicodeDataParseError(Exception):
    """Raised when a downloaded Unicode data file yields no usable entries.

    Distinct from `UnicodeDataDownloadError` on purpose: an unreachable network is a legitimate reason to
    skip a test, while a file that parses to an empty table means the cache is truncated or the upstream
    format moved, and a test running against an empty table silently validates nothing.
    """

    pass


def _parsed_or_raise(entries, cache_path: str, description: str):
    """Return the parsed entries, or raise if the file yielded none."""
    if not entries:
        raise UnicodeDataParseError(f"No {description} parsed from {cache_path}")
    return entries


UNICODE_VERSION = "17.0.0"
"""Unicode version used for all Unicode data files."""


def get_unicode_xml_data(version: str = UNICODE_VERSION) -> ET.Element:
    """Download and parse Unicode UCD XML, caching in temp directory.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Root Element of the parsed XML tree
    """
    cache_path = os.path.join(tempfile.gettempdir(), f"ucd-{version}.all.flat.xml")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucdxml/ucd.all.flat.zip"
        print(f"Downloading Unicode {version} UCD XML from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                zip_data = response.read()
            zip_bytes = io.BytesIO(zip_data)
            with zipfile.ZipFile(zip_bytes) as zip_file:
                xml_filename = zip_file.namelist()[0]
                with zip_file.open(xml_filename) as xml_file:
                    xml_content = xml_file.read()
                    _write_cache_atomically(cache_path, xml_content)
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download UCD XML from {url}: {error}")
    else:
        print(f"Using cached Unicode {version} UCD XML: {cache_path}")

    tree = ET.parse(cache_path)
    return tree.getroot()


def get_all_codepoints(version: str = UNICODE_VERSION) -> list[int]:
    """Return all assigned/defined codepoints in Unicode."""
    root = get_unicode_xml_data(version)
    namespace = {"ucd": "http://www.unicode.org/ns/2003/ucd/1.0"}
    codepoints = []
    for char in root.findall(".//ucd:char", namespace):
        codepoint_string = char.get("cp")
        if codepoint_string:
            codepoints.append(int(codepoint_string, 16))
    return sorted(codepoints)


def parse_uncased_folding_file(filepath: str) -> dict[int, bytes]:
    """Parse Unicode CaseFolding.txt into a dict: codepoint -> folded UTF-8 bytes.

    Uses status C (common) and F (full) mappings for full case folding.
    """
    folds = {}
    with open(filepath, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            status = parts[1].strip()
            # C = common, F = full (for expansions like ß → ss)
            # Skip S (simple) and T (Turkic) for full case folding
            if status not in ("C", "F"):
                continue
            try:
                codepoint = int(parts[0].strip(), 16)
                # Mapping can be multiple codepoints separated by spaces (e.g., "0073 0073" for ß → ss)
                target_codepoints = [int(hex_part, 16) for hex_part in parts[2].split("#")[0].strip().split()]
                # Convert target codepoints to UTF-8 bytes
                folded_string = "".join(chr(codepoint) for codepoint in target_codepoints)
                folds[codepoint] = folded_string.encode("utf-8")
            except (ValueError, IndexError):
                continue
    return folds


def _download_uncased_folding_file(version: str) -> str:
    """Download CaseFolding.txt and return the cache path.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Path to cached file

    Raises:
        UnicodeDataDownloadError: If download fails
    """
    cache_path = os.path.join(tempfile.gettempdir(), f"CaseFolding-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/CaseFolding.txt"
        try:
            # Use urlopen with 30-second timeout instead of urlretrieve
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download CaseFolding.txt from {url}: {error}")

    return cache_path


def get_uncased_folding_rules(version: str = UNICODE_VERSION) -> dict[int, bytes]:
    """Download and parse Unicode CaseFolding.txt, caching in temp directory.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping codepoints to their folded UTF-8 bytes
    """
    cache_path = _download_uncased_folding_file(version)
    return parse_uncased_folding_file(cache_path)


def get_uncased_folding_rules_as_codepoints(
    version: str = UNICODE_VERSION,
) -> dict[int, list[int]]:
    """Download and parse Unicode CaseFolding.txt, returning target codepoints.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping source codepoints to list of target codepoints
    """
    cache_path = _download_uncased_folding_file(version)

    folds = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            status = parts[1].strip()
            if status not in ("C", "F"):
                continue
            try:
                source_codepoint = int(parts[0].strip(), 16)
                target_codepoints = [int(hex_part, 16) for hex_part in parts[2].split("#")[0].strip().split()]
                folds[source_codepoint] = target_codepoints
            except (ValueError, IndexError):
                continue
    return folds


def get_normalization_props(version: str = UNICODE_VERSION) -> dict[int, dict[str, str]]:
    """Download and parse Unicode DerivedNormalizationProps.txt, caching in temp directory.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping codepoints to their normalization properties.
        Properties include: NFC_QC, NFD_QC, NFKC_QC, NFKD_QC (Quick_Check values)
    """
    cache_path = os.path.join(tempfile.gettempdir(), f"DerivedNormalizationProps-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/DerivedNormalizationProps.txt"
        print(f"Downloading Unicode {version} DerivedNormalizationProps.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download DerivedNormalizationProps.txt from {url}: {error}")
    else:
        print(f"Using cached Unicode {version} DerivedNormalizationProps.txt: {cache_path}")

    properties: dict[int, dict[str, str]] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_field = parts[1].strip()

                # Parse property name and value (e.g., "NFC_QC" or "NFC_QC; N")
                if len(parts) >= 3:
                    property_name = property_field
                    property_default = parts[2].strip()
                else:
                    # Properties like "Full_Composition_Exclusion" are boolean
                    property_name = property_field
                    property_default = "Y"

                # Parse codepoint range
                if ".." in codepoint_range:
                    start, end = codepoint_range.split("..")
                    start_codepoint = int(start, 16)
                    end_codepoint = int(end, 16)
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)

                for codepoint in range(start_codepoint, end_codepoint + 1):
                    if codepoint not in properties:
                        properties[codepoint] = {}
                    properties[codepoint][property_name] = property_default
            except (ValueError, IndexError):
                continue

    return properties


def _download_word_break_property_file(version: str) -> str:
    """Download WordBreakProperty.txt and return the cache path."""
    cache_path = os.path.join(tempfile.gettempdir(), f"WordBreakProperty-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/WordBreakProperty.txt"
        print(f"Downloading Unicode {version} WordBreakProperty.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download WordBreakProperty.txt from {url}: {error}")

    return cache_path


def get_word_break_properties(version: str = UNICODE_VERSION) -> dict[int, str]:
    """Download and parse WordBreakProperty.txt.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping codepoints to their Word_Break property name.
        Property names: ALetter, CR, Double_Quote, Extend, ExtendNumLet, Format,
        Hebrew_Letter, Katakana, LF, MidLetter, MidNum, MidNumLet, Newline,
        Numeric, Regional_Indicator, Single_Quote, WSegSpace, ZWJ
    """
    cache_path = _download_word_break_property_file(version)

    properties: dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_name = parts[1].strip()

                # Parse codepoint range
                if ".." in codepoint_range:
                    start, end = codepoint_range.split("..")
                    start_codepoint = int(start, 16)
                    end_codepoint = int(end, 16)
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)

                for codepoint in range(start_codepoint, end_codepoint + 1):
                    properties[codepoint] = property_name
            except (ValueError, IndexError):
                continue

    return properties


def _download_word_break_test_file(version: str) -> str:
    """Download WordBreakTest.txt and return the cache path."""
    cache_path = os.path.join(tempfile.gettempdir(), f"WordBreakTest-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/WordBreakTest.txt"
        print(f"Downloading Unicode {version} WordBreakTest.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download WordBreakTest.txt from {url}: {error}")

    return cache_path


def get_word_break_test_cases(version: str = UNICODE_VERSION) -> list[tuple]:
    """Download and parse WordBreakTest.txt.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        List of tuples: (text: str, boundary_positions: list[int])
        boundary_positions includes 0 and len(text) as TR29 specifies.
    """
    cache_path = _download_word_break_test_file(version)

    test_cases = []
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue

            # Parse format: ÷ 0061 × 0027 × 0061 ÷
            # ÷ = break, × = no break
            # Numbers are hex codepoints
            parts = line.replace("÷", " ÷ ").replace("×", " × ").split()

            codepoints = []
            boundaries = []
            byte_position = 0

            for part in parts:
                part = part.strip()
                if part == "÷":
                    boundaries.append(byte_position)
                elif part == "×":
                    pass  # No boundary
                elif part:
                    try:
                        codepoint = int(part, 16)
                        codepoints.append(codepoint)
                        byte_position += len(chr(codepoint).encode("utf-8"))
                    except ValueError:
                        continue

            if codepoints:
                text = "".join(chr(codepoint) for codepoint in codepoints)
                test_cases.append((text, boundaries))

    return test_cases


def baseline_word_boundaries(text: str, word_break_properties: dict[int, str] | None = None) -> list[int]:
    """Pure Python implementation of TR29 word boundary algorithm.

    This serves as a reference baseline for testing the C implementation.

    Args:
        text: UTF-8 string to find word boundaries in.
        wb_props: Optional pre-loaded word break properties dict.

    Returns:
        List of byte positions that are word boundaries.
    """
    if word_break_properties is None:
        word_break_properties = get_word_break_properties()

    def word_break_property_of(codepoint: int) -> str:
        return word_break_properties.get(codepoint, "Other")

    # Convert text to list of (codepoint, byte_offset, byte_length)
    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)

    if not codepoints:
        return [0]

    total_bytes = byte_offset
    boundaries = [0]  # WB1: sot ÷

    # Helper to check if a property is "ignorable" per WB4
    def is_ignorable(property_value: str) -> bool:
        return property_value in ("Extend", "Format", "ZWJ")

    # Helper to get effective property (skipping WB4 ignorables)
    def effective_property_at(relative_index: int, direction: int = 1) -> str:
        """Get property at idx, or skip ignorables in direction."""
        while 0 <= relative_index < len(codepoints):
            property_value = word_break_property_of(codepoints[relative_index][0])
            if not is_ignorable(property_value):
                return property_value
            relative_index += direction
        return "Other"

    def is_aletter_or_hebrew_letter(property_value: str) -> bool:
        return property_value in ("ALetter", "Hebrew_Letter")

    def is_midnumlet_or_single_quote(property_value: str) -> bool:
        return property_value in ("MidNumLet", "Single_Quote")

    # Check each position between codepoints
    for index in range(1, len(codepoints)):
        boundary_byte_position = codepoints[index][1]

        # Get properties before and after this position
        previous_codepoint = codepoints[index - 1][0]
        current_codepoint = codepoints[index][0]
        previous_property = word_break_property_of(previous_codepoint)
        current_property = word_break_property_of(current_codepoint)

        # WB3: Do not break between CR and LF
        if previous_property == "CR" and current_property == "LF":
            continue

        # WB3a: Break after CR, LF, Newline
        if previous_property in ("CR", "LF", "Newline"):
            boundaries.append(boundary_byte_position)
            continue

        # WB3b: Break before CR, LF, Newline
        if current_property in ("CR", "LF", "Newline"):
            boundaries.append(boundary_byte_position)
            continue

        # WB3c: Do not break within emoji ZWJ sequences
        if previous_property == "ZWJ":
            # Extended_Pictographic check would go here
            # For simplicity, just don't break after ZWJ
            continue

        # WB3d: Keep horizontal whitespace together
        if previous_property == "WSegSpace" and current_property == "WSegSpace":
            continue

        # WB4: Ignore Format and Extend
        # Get effective properties skipping ignorables
        effective_previous = previous_property
        if is_ignorable(previous_property):
            # Look back for non-ignorable
            for lookahead_index in range(index - 2, -1, -1):
                scanned_property = word_break_property_of(codepoints[lookahead_index][0])
                if not is_ignorable(scanned_property):
                    effective_previous = scanned_property
                    break
            else:
                effective_previous = "Other"

        effective_current = current_property
        # Note: we don't skip ignorables for current in simple cases

        # WB5: Do not break between letters
        if is_aletter_or_hebrew_letter(effective_previous) and is_aletter_or_hebrew_letter(effective_current):
            continue

        # WB6: Do not break letters across punctuation (looking ahead)
        if is_aletter_or_hebrew_letter(effective_previous) and current_property in (
            "MidLetter",
            "MidNumLet",
            "Single_Quote",
        ):
            # Look ahead for AHLetter
            for lookahead_index in range(index + 1, len(codepoints)):
                next_property = word_break_property_of(codepoints[lookahead_index][0])
                if is_ignorable(next_property):
                    continue
                if is_aletter_or_hebrew_letter(next_property):
                    # Don't break: AHLetter × (MidLetter|MidNumLetQ) × AHLetter
                    break
                else:
                    boundaries.append(boundary_byte_position)
                    break
            else:
                boundaries.append(boundary_byte_position)
            continue

        # WB7: Reverse of WB6
        if current_property in ("MidLetter", "MidNumLet", "Single_Quote"):
            # Already handled by WB6
            pass

        if is_midnumlet_or_single_quote(effective_previous) or effective_previous == "MidLetter":
            if is_aletter_or_hebrew_letter(effective_current):
                # Check if preceded by AHLetter
                for lookahead_index in range(index - 2, -1, -1):
                    scanned_property = word_break_property_of(codepoints[lookahead_index][0])
                    if is_ignorable(scanned_property):
                        continue
                    if is_aletter_or_hebrew_letter(scanned_property):
                        # Don't break: AHLetter × (MidLetter|MidNumLetQ) × AHLetter
                        break
                    else:
                        boundaries.append(boundary_byte_position)
                        break
                else:
                    boundaries.append(boundary_byte_position)
                continue

        # WB8: Do not break between digits
        if effective_previous == "Numeric" and effective_current == "Numeric":
            continue

        # WB9: Do not break letter-digit
        if is_aletter_or_hebrew_letter(effective_previous) and effective_current == "Numeric":
            continue

        # WB10: Do not break digit-letter
        if effective_previous == "Numeric" and is_aletter_or_hebrew_letter(effective_current):
            continue

        # WB11/12: Numeric separators
        if effective_previous == "Numeric" and current_property in ("MidNum", "MidNumLet", "Single_Quote"):
            # Look ahead for Numeric
            for lookahead_index in range(index + 1, len(codepoints)):
                next_property = word_break_property_of(codepoints[lookahead_index][0])
                if is_ignorable(next_property):
                    continue
                if next_property == "Numeric":
                    break
                else:
                    boundaries.append(boundary_byte_position)
                    break
            else:
                boundaries.append(boundary_byte_position)
            continue

        if current_property in ("MidNum", "MidNumLet", "Single_Quote") and effective_current == "Numeric":
            # WB12 - check preceding Numeric
            pass  # Handled above

        # WB13: Katakana
        if effective_previous == "Katakana" and effective_current == "Katakana":
            continue

        # WB13a: Connect with ExtendNumLet
        if effective_previous in ("ALetter", "Hebrew_Letter", "Numeric", "Katakana", "ExtendNumLet"):
            if effective_current == "ExtendNumLet":
                continue

        # WB13b: ExtendNumLet continues words
        if effective_previous == "ExtendNumLet":
            if effective_current in ("ALetter", "Hebrew_Letter", "Numeric", "Katakana"):
                continue

        # WB15/16: Regional Indicators - keep pairs together
        if effective_previous == "Regional_Indicator" and effective_current == "Regional_Indicator":
            # Count preceding RIs - if odd, don't break
            regional_indicator_count = 0
            for lookahead_index in range(index - 1, -1, -1):
                scanned_property = word_break_property_of(codepoints[lookahead_index][0])
                if scanned_property == "Regional_Indicator":
                    regional_indicator_count += 1
                elif is_ignorable(scanned_property):
                    continue
                else:
                    break
            if regional_indicator_count % 2 == 1:
                continue

        # WB999: Otherwise, break
        boundaries.append(boundary_byte_position)

    # WB2: ÷ eot
    boundaries.append(total_bytes)

    return sorted(set(boundaries))


def _download_break_property_file(filename: str, version: str) -> str:
    """Download an auxiliary break-property file (e.g. GraphemeBreakProperty.txt) and return the cache path.

    ``LineBreak.txt`` lives directly under ``ucd/`` while the others live under ``ucd/auxiliary/``.
    """
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")

    subdir = "ucd" if filename == "LineBreak.txt" else "ucd/auxiliary"
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/{subdir}/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {error}")

    return cache_path


def _parse_break_property_file(cache_path: str) -> dict[int, str]:
    """Parse a ``codepoint(..range) ; Property`` style UCD break-property file into a dict.

    An empty result means a truncated cache, so it raises rather than returning a table that would
    silently make every consumer agree with a broken oracle.
    """
    properties: dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_name = parts[1].strip()
                if ".." in codepoint_range:
                    start, end = codepoint_range.split("..")
                    start_codepoint = int(start, 16)
                    end_codepoint = int(end, 16)
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)
                for codepoint in range(start_codepoint, end_codepoint + 1):
                    properties[codepoint] = property_name
            except (ValueError, IndexError):
                continue
    return _parsed_or_raise(properties, cache_path, "break properties")


def _download_break_test_file(filename: str, version: str) -> str:
    """Download an auxiliary break-test file (e.g. GraphemeBreakTest.txt) and return the cache path."""
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {error}")

    return cache_path


def _parse_break_test_file(cache_path: str) -> list[tuple]:
    """Parse a UAX break-test file (``÷`` = break, ``×`` = no break) into (text, boundary_positions)."""
    test_cases = []
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace("÷", " ÷ ").replace("×", " × ").split()
            codepoints = []
            boundaries = []
            byte_position = 0
            for part in parts:
                part = part.strip()
                if part == "÷":
                    boundaries.append(byte_position)
                elif part == "×":
                    pass
                elif part:
                    try:
                        codepoint = int(part, 16)
                        codepoints.append(codepoint)
                        byte_position += len(chr(codepoint).encode("utf-8"))
                    except ValueError:
                        continue
            if codepoints:
                text = "".join(chr(codepoint) for codepoint in codepoints)
                test_cases.append((text, boundaries))
    return _parsed_or_raise(test_cases, cache_path, "break-test cases")


# region Grapheme


def get_grapheme_break_properties(version: str = UNICODE_VERSION) -> dict[int, str]:
    """Download and parse GraphemeBreakProperty.txt.

    Returns a dict mapping codepoints to their Grapheme_Cluster_Break property name.
    Property names: CR, LF, Control, Extend, ZWJ, Regional_Indicator, Prepend,
    SpacingMark, L, V, T, LV, LVT, and (implicitly) Other.
    """
    cache_path = _download_break_property_file("GraphemeBreakProperty.txt", version)
    return _parse_break_property_file(cache_path)


def get_indic_conjunct_break_properties(version: str = UNICODE_VERSION) -> dict[int, str]:
    """Download and parse the Indic_Conjunct_Break (InCB) property from DerivedCoreProperties.txt.

    Returns a dict mapping codepoints to their InCB value ("Linker", "Consonant", or "Extend");
    codepoints absent from the map are InCB=None. Used by `baseline_grapheme_boundaries` for GB9c.
    """
    cache_path = os.path.join(tempfile.gettempdir(), f"DerivedCoreProperties-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/DerivedCoreProperties.txt"
        print(f"Downloading Unicode {version} DerivedCoreProperties.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download DerivedCoreProperties.txt from {url}: {error}")
    indic_conjunct_breaks: dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            parts = [part.strip() for part in line.split(";")]
            if len(parts) < 3 or parts[1] != "InCB":
                continue
            codepoint_range = parts[0]
            if ".." in codepoint_range:
                start_codepoint, end_codepoint = (int(bound, 16) for bound in codepoint_range.split(".."))
            else:
                start_codepoint = end_codepoint = int(codepoint_range, 16)
            for codepoint in range(start_codepoint, end_codepoint + 1):
                indic_conjunct_breaks[codepoint] = parts[2]
    return _parsed_or_raise(indic_conjunct_breaks, cache_path, "Indic_Conjunct_Break entries")


def get_grapheme_break_test_cases(version: str = UNICODE_VERSION) -> list[tuple]:
    """Download and parse the official GraphemeBreakTest.txt.

    Returns a list of (text: str, boundary_positions: list[int]) tuples; boundary
    positions are byte offsets and include 0 and len(text).
    """
    cache_path = _download_break_test_file("GraphemeBreakTest.txt", version)
    return _parse_break_test_file(cache_path)


def baseline_grapheme_boundaries(
    text: str,
    properties: dict[int, str] | None = None,
    indic_conjunct_breaks: dict[int, str] | None = None,
    extended_pictographic: set = None,
) -> list[int]:
    """Pure Python implementation of the UAX-29 extended grapheme cluster algorithm.

    Reference baseline for the C implementation. Returns byte positions of cluster boundaries.
    Covers GB1-GB999 including the Hangul, GB9c Indic-conjunct, emoji-ZWJ (GB11) and
    Regional_Indicator (GB12/13) rules.
    """
    if properties is None:
        properties = get_grapheme_break_properties()
    if indic_conjunct_breaks is None:
        indic_conjunct_breaks = get_indic_conjunct_break_properties()
    if extended_pictographic is None:
        extended_pictographic = get_extended_pictographic()

    def property_of(codepoint: int) -> str:
        return properties.get(codepoint, "Other")

    def indic_conjunct_of(codepoint: int) -> str:
        return indic_conjunct_breaks.get(codepoint, "None")

    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)

    if not codepoints:
        return [0]

    total_bytes = byte_offset
    boundaries = [0]  # GB1: sot ÷

    def is_extended_pictographic(codepoint: int) -> bool:
        # Real Extended_Pictographic from emoji-data.txt. Regional_Indicators have their own GCB class and are
        # not Extended_Pictographic, so GB11's ZWJ bridge correctly does not fire on `RI ZWJ RI`.
        return codepoint in extended_pictographic

    for index in range(1, len(codepoints)):
        boundary_byte_position = codepoints[index][1]
        previous_codepoint = codepoints[index - 1][0]
        current_codepoint = codepoints[index][0]
        previous_property = property_of(previous_codepoint)
        current_property = property_of(current_codepoint)

        # GB3: CR x LF
        if previous_property == "CR" and current_property == "LF":
            continue
        # GB4: (Control | CR | LF) ÷
        if previous_property in ("Control", "CR", "LF"):
            boundaries.append(boundary_byte_position)
            continue
        # GB5: ÷ (Control | CR | LF)
        if current_property in ("Control", "CR", "LF"):
            boundaries.append(boundary_byte_position)
            continue
        # GB6: L x (L | V | LV | LVT)
        if previous_property == "L" and current_property in ("L", "V", "LV", "LVT"):
            continue
        # GB7: (LV | V) x (V | T)
        if previous_property in ("LV", "V") and current_property in ("V", "T"):
            continue
        # GB8: (LVT | T) x T
        if previous_property in ("LVT", "T") and current_property == "T":
            continue
        # GB9: x (Extend | ZWJ)
        if current_property in ("Extend", "ZWJ"):
            continue
        # GB9a: x SpacingMark
        if current_property == "SpacingMark":
            continue
        # GB9b: Prepend x
        if previous_property == "Prepend":
            continue
        # GB9c: Consonant [Extend|Linker]* Linker [Extend|Linker]* x Consonant (Indic conjunct)
        if indic_conjunct_of(current_codepoint) == "Consonant":
            seen_linker = False
            scan_index = index - 1
            while scan_index >= 0 and indic_conjunct_of(codepoints[scan_index][0]) in ("Extend", "Linker"):
                if indic_conjunct_of(codepoints[scan_index][0]) == "Linker":
                    seen_linker = True
                scan_index -= 1
            if scan_index >= 0 and seen_linker and indic_conjunct_of(codepoints[scan_index][0]) == "Consonant":
                continue
        # GB11: ExtPict Extend* ZWJ x ExtPict
        if previous_property == "ZWJ" and is_extended_pictographic(current_codepoint):
            lookahead_index = index - 1
            # skip the ZWJ, then any Extend, looking for an Extended_Pictographic base
            scan_index = lookahead_index - 1
            while scan_index >= 0 and property_of(codepoints[scan_index][0]) == "Extend":
                scan_index -= 1
            if scan_index >= 0 and is_extended_pictographic(codepoints[scan_index][0]):
                continue
        # GB12/GB13: RI x RI when preceding RI count is even
        if previous_property == "Regional_Indicator" and current_property == "Regional_Indicator":
            regional_indicator_count = 0
            for scan_index in range(index - 1, -1, -1):
                if property_of(codepoints[scan_index][0]) == "Regional_Indicator":
                    regional_indicator_count += 1
                else:
                    break
            if regional_indicator_count % 2 == 1:
                continue
        # GB999: otherwise break
        boundaries.append(boundary_byte_position)

    boundaries.append(total_bytes)  # GB2: eot ÷
    return sorted(set(boundaries))


# endregion Grapheme

# region Sentence


def get_sentence_break_properties(version: str = UNICODE_VERSION) -> dict[int, str]:
    """Download and parse SentenceBreakProperty.txt.

    Returns a dict mapping codepoints to their Sentence_Break property name.
    Property names: CR, LF, Extend, Sep, Format, Sp, Lower, Upper, OLetter,
    Numeric, ATerm, SContinue, STerm, Close, and (implicitly) Other.
    """
    cache_path = _download_break_property_file("SentenceBreakProperty.txt", version)
    return _parse_break_property_file(cache_path)


def get_sentence_break_test_cases(version: str = UNICODE_VERSION) -> list[tuple]:
    """Download and parse the official SentenceBreakTest.txt.

    Returns a list of (text: str, boundary_positions: list[int]) tuples; boundary
    positions are byte offsets and include 0 and len(text).
    """
    cache_path = _download_break_test_file("SentenceBreakTest.txt", version)
    return _parse_break_test_file(cache_path)


def baseline_sentence_boundaries(text: str, properties: dict[int, str] | None = None) -> list[int]:
    """Pure Python implementation of the UAX-29 sentence boundary algorithm.

    Reference baseline for the C implementation. Returns byte positions of sentence boundaries.
    Implements SB1-SB998 including the SB8/SB8a/SB9/SB10/SB11 ATerm/STerm machinery.
    """
    if properties is None:
        properties = get_sentence_break_properties()

    def property_of(codepoint: int) -> str:
        return properties.get(codepoint, "Other")

    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)

    if not codepoints:
        return [0]

    total_bytes = byte_offset
    property_list = [property_of(codepoint) for codepoint, _, _ in codepoints]

    def is_ignorable(property_value: str) -> bool:
        # SB5: ignore Extend and Format for the purpose of the "before" context.
        return property_value in ("Extend", "Format")

    def effective_before(relative_index: int) -> str:
        """Property of the last non-ignorable codepoint at or before idx (inclusive)."""
        scan_index = relative_index
        while scan_index >= 0 and is_ignorable(property_list[scan_index]):
            scan_index -= 1
        return property_list[scan_index] if scan_index >= 0 else "Other"

    def is_paragraph_separator(property_value: str) -> bool:
        return property_value in ("Sep", "CR", "LF")

    boundaries = [0]  # SB1: sot ÷

    codepoint_count = len(codepoints)
    for index in range(1, codepoint_count):
        boundary_byte_position = codepoints[index][1]
        previous_property = property_list[index - 1]
        current_property = property_list[index]

        # SB3: CR x LF
        if previous_property == "CR" and current_property == "LF":
            continue
        # SB4: Sep | CR | LF ÷
        if is_paragraph_separator(previous_property):
            boundaries.append(boundary_byte_position)
            continue
        # SB5: x (Extend | Format) -> ignore (no break)
        if current_property in ("Extend", "Format"):
            continue

        effective_previous = effective_before(index - 1)

        # SB6: ATerm x Numeric
        if effective_previous == "ATerm" and current_property == "Numeric":
            continue
        # SB7: (Upper | Lower) ATerm x Upper
        if effective_previous == "ATerm" and current_property == "Upper":
            # find the codepoint before the ATerm (skipping ignorables)
            scan_index = index - 2
            while scan_index >= 0 and is_ignorable(property_list[scan_index]):
                scan_index -= 1
            if scan_index >= 0 and property_list[scan_index] in ("Upper", "Lower"):
                continue
        # Scan back over Close* and Sp* after an ATerm/STerm to find the sentence-terminator context.
        # SB8: ATerm Close* Sp* x (not (OLetter|Upper|Lower|Sep|CR|LF|STerm|ATerm))* Lower
        # SB8a: (STerm|ATerm) Close* Sp* x (SContinue | STerm | ATerm)
        scan_index = index - 1
        while scan_index >= 0 and is_ignorable(property_list[scan_index]):
            scan_index -= 1
        # walk back through Sp
        while scan_index >= 0 and (property_list[scan_index] == "Sp" or is_ignorable(property_list[scan_index])):
            scan_index -= 1
        # walk back through Close
        while scan_index >= 0 and (property_list[scan_index] == "Close" or is_ignorable(property_list[scan_index])):
            scan_index -= 1
        terminator_property = property_list[scan_index] if scan_index >= 0 else "Other"

        if terminator_property in ("ATerm", "STerm"):
            # SB8a
            if current_property in ("SContinue", "STerm", "ATerm"):
                continue
            if terminator_property == "ATerm":
                # SB8: look ahead skipping the "not separator/terminator/letter" run to a Lower
                lookahead_index = index
                lower_ahead = False
                while lookahead_index < codepoint_count:
                    scanned_property = property_list[lookahead_index]
                    if scanned_property == "Lower":
                        lower_ahead = True
                        break
                    if scanned_property in ("OLetter", "Upper", "Sep", "CR", "LF", "STerm", "ATerm"):
                        break
                    lookahead_index += 1
                if lower_ahead:
                    continue
            # SB9: (STerm|ATerm) Close* x (Close | Sp | Sep | CR | LF)
            if current_property in ("Close", "Sp", "Sep", "CR", "LF"):
                continue
            # SB11: after STerm/ATerm (with optional Close*/Sp*/ParaSep) -> break
            boundaries.append(boundary_byte_position)
            continue

        # SB998: otherwise, do not break
        continue

    boundaries.append(total_bytes)  # SB2: eot ÷
    return sorted(set(boundaries))


# endregion Sentence

# region Line


def get_line_break_properties(version: str = UNICODE_VERSION) -> dict[int, str]:
    """Download and parse LineBreak.txt.

    Returns a dict mapping codepoints to their Line_Break property name
    (e.g. BK, CR, LF, NL, SP, OP, CL, GL, BA, BB, B2, HY, AL, ID, NU, ...).
    """
    cache_path = _download_break_property_file("LineBreak.txt", version)
    return _parse_break_property_file(cache_path)


def get_line_break_test_cases(version: str = UNICODE_VERSION) -> list[tuple]:
    """Download and parse the official LineBreakTest.txt.

    Returns a list of (text: str, boundary_positions: list[int]) tuples; boundary
    positions are byte offsets and include 0 and len(text). The test file marks only
    break opportunities, not whether they are mandatory.
    """
    cache_path = _download_break_test_file("LineBreakTest.txt", version)
    return _parse_break_test_file(cache_path)


def baseline_line_boundaries(text: str, properties: dict[int, str] | None = None) -> list[tuple]:
    """Pure Python reference for UAX-14 mandatory line breaks.

    Rather than reimplement the full pair-table (LB1-LB31), this baseline returns the set of
    *mandatory* break boundaries (LB4/LB5: BK, CR, LF, NL), which is the property the
    StringZilla line iterator's ``is_mandatory`` flag exposes. Soft break opportunities are
    library-specific and validated separately against ``uniseg``.

    Returns:
        List of (byte_offset, mandatory) tuples. ``mandatory`` is always True here (every
        returned offset is a hard break); the final eot boundary is reported as non-mandatory.
    """
    if properties is None:
        properties = get_line_break_properties()

    def property_of(codepoint: int) -> str:
        return properties.get(codepoint, "AL")

    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)

    if not codepoints:
        return [(0, False)]

    total_bytes = byte_offset
    boundaries = []  # list of (offset, mandatory)
    seen = set()

    def add(offset: int, mandatory: bool):
        if offset not in seen:
            seen.add(offset)
            boundaries.append((offset, mandatory))

    add(0, False)  # LB2: sot, never a mandatory break

    codepoint_count = len(codepoints)
    for index in range(1, codepoint_count):
        boundary_byte_position = codepoints[index][1]
        previous_property = property_of(codepoints[index - 1][0])
        current_property = property_of(codepoints[index][0])

        # LB5: CR x LF -> no break between
        if previous_property == "CR" and current_property == "LF":
            continue
        # LB4: BK !  (mandatory break after BK)
        # LB5: CR ! , LF ! , NL !  (mandatory break after each)
        if previous_property in ("BK", "CR", "LF", "NL"):
            add(boundary_byte_position, True)
            continue

    # eot boundary (LB3) - reported but not mandatory
    add(total_bytes, False)
    return sorted(boundaries)


# endregion Line

# region Hardening extractors

# Extractors that derive hard synthetic corner cases and authoritative oracles directly from the
# UCD, rather than from a hand-picked palette: the official NormalizationTest.txt, emoji-data
# Extended_Pictographic, the canonical combining classes and decomposition mappings, and a
# per-break-class representative inverter.


def _download_ucd_text(filename: str, subdir: str, version: str = UNICODE_VERSION) -> str:
    """Download a UCD text file from an arbitrary subdir (e.g. ``ucd`` or ``ucd/emoji``) and cache it."""
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/{subdir}/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {error}")
    return cache_path


def get_normalization_test_cases(version: str = UNICODE_VERSION) -> list[tuple]:
    """Download and parse the official NormalizationTest.txt.

    Returns a list of (source, nfc, nfd, nfkc, nfkd) tuples, each a decoded `str`. This is the authoritative,
    version-pinned conformance data for NFC/NFD/NFKC/NFKD, independent of ICU's or the host Python's Unicode
    version. ``@PartN`` section markers and comments are skipped.
    """
    cache_path = _download_ucd_text("NormalizationTest.txt", "ucd", version)

    def field_to_string(field: str) -> str:
        return "".join(chr(int(codepoint, 16)) for codepoint in field.split())

    cases = []
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line or line.startswith("@"):
                continue
            fields = [field.strip() for field in line.split(";")]
            if len(fields) < 5:
                continue
            try:
                cases.append(tuple(field_to_string(fields[index]) for index in range(5)))
            except ValueError:
                continue
    return cases


def get_emoji_properties(version: str = UNICODE_VERSION) -> dict[int, set]:
    """Download and parse emoji-data.txt into a dict mapping codepoints to their set of emoji property names.

    Property names: Emoji, Emoji_Presentation, Emoji_Modifier, Emoji_Modifier_Base, Emoji_Component,
    Extended_Pictographic.
    """
    cache_path = _download_ucd_text("emoji-data.txt", "ucd/emoji", version)

    properties: dict[int, set] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_name = parts[1].strip()
                if ".." in codepoint_range:
                    start_codepoint, end_codepoint = (int(bound, 16) for bound in codepoint_range.split(".."))
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)
                for codepoint in range(start_codepoint, end_codepoint + 1):
                    properties.setdefault(codepoint, set()).add(property_name)
            except (ValueError, IndexError):
                continue
    return properties


def get_extended_pictographic(version: str = UNICODE_VERSION) -> set:
    """Return the set of Extended_Pictographic codepoints (the real GB11 / WB3c property from emoji-data.txt)."""
    return {codepoint for codepoint, names in get_emoji_properties(version).items() if "Extended_Pictographic" in names}


def get_combining_classes(version: str = UNICODE_VERSION) -> dict[int, int]:
    """Return a dict mapping codepoints to their Canonical_Combining_Class (from the UCD XML ``ccc`` attribute).

    Only non-default entries matter for normalization reordering; codepoints absent from the map are ccc=0.
    """
    root = get_unicode_xml_data(version)
    namespace = "{http://www.unicode.org/ns/2003/ucd/1.0}"
    combining_classes: dict[int, int] = {}
    for char in root.iter(namespace + "char"):
        codepoint = char.get("cp")
        raw_combining_class = char.get("ccc")
        if codepoint is None or raw_combining_class is None:
            continue
        value = int(raw_combining_class)
        if value:
            combining_classes[int(codepoint, 16)] = value
    return combining_classes


def get_decomposition_mappings(version: str = UNICODE_VERSION) -> dict[int, tuple]:
    """Return a dict mapping codepoints to ``(kind, [target_codepoints])`` decomposition mappings.

    ``kind`` is ``"canonical"`` (UCD ``dt == "can"``) or ``"compatibility"`` (any other decomposition type).
    Codepoints with no decomposition (``dm == "#"``) are excluded. Sourced from the UCD XML ``dm`` / ``dt``
    attributes; used to build canonical-equivalent precomposed/decomposed pairs.
    """
    root = get_unicode_xml_data(version)
    namespace = "{http://www.unicode.org/ns/2003/ucd/1.0}"
    mappings: dict[int, tuple] = {}
    for char in root.iter(namespace + "char"):
        codepoint = char.get("cp")
        decomposition_mapping = char.get("dm")
        if codepoint is None or not decomposition_mapping or decomposition_mapping == "#":
            continue
        decomposition_type = char.get("dt")
        kind = "canonical" if decomposition_type in (None, "can") else "compatibility"
        targets = [int(part, 16) for part in decomposition_mapping.split()]
        mappings[int(codepoint, 16)] = (kind, targets)
    return mappings


def representatives_by_class(
    properties: dict[int, str], count: int = 3, bmp_only: bool = False
) -> dict[str, list[int]]:
    """Invert a ``{codepoint: class_name}`` break-property dict into ``{class_name: [first-`count` codepoints]}``.

    The enabler for class-adjacency corpora: with one representative codepoint per break class, a test can
    construct strings exercising every class-adjacency pair (or triple) the segmentation rules can encounter.
    """
    representatives: dict[str, list[int]] = {}
    for codepoint in sorted(properties):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue  # surrogates are never valid in UTF-8 text
        if bmp_only and codepoint > 0xFFFF:
            continue
        bucket = representatives.setdefault(properties[codepoint], [])
        if len(bucket) < count:
            bucket.append(codepoint)
    return representatives


# endregion Hardening extractors

# region General test scaffolding

# General fixtures and seeded-RNG helpers shared by every per-family test module, the Python analog
# of the C++ `test/harness.hpp` harness. Kept here so a split test file imports one place for both
# the Unicode data loaders above and the seeding / random-string utilities below. The NumPy /
# PyArrow availability flags and their defensive imports live in the top import block.


def env_text(name: str) -> str | None:
    """Reads `name`, or `None` when it is unset or empty."""
    return os.environ.get(name) or None


def env_parsed(name: str, fallback, parse: Callable[[str], object], expected: str):
    """Reads `name` through `parse`, or `fallback` when unset or empty; exits with status 1 if it does not parse."""
    text = env_text(name)
    if text is None:
        return fallback
    try:
        parsed = parse(text)
    except (ValueError, TypeError):
        parsed = None
    if parsed is None:
        raise SystemExit(f'{name}="{text}" does not parse, expected {expected}')
    return parsed


def env_flag(name: str, fallback: bool) -> bool:
    """Reads `0`, `1`, `true` or `false`, or `fallback` when unset or empty; exits if it does not parse."""
    return env_parsed(name, fallback, {"0": False, "false": False, "1": True, "true": True}.get, "0, 1, true or false")


Seed = NewType("Seed", int)  # 32-bit; derive streams from it with `stream_key`, never add to it
StreamKey = NewType("StreamKey", int)  # 64-bit, from `stream_key`; seeds one test's generators


def env_seed(name: str, fallback: Seed) -> Seed:
    """Reads an unsigned 32-bit seed or `random`, or `fallback` when unset or empty."""
    return env_parsed(name, fallback, parse_seed, "an unsigned integer or random")


def parse_seed(text: str) -> Seed | None:
    """Parses an unsigned 32-bit integer like `42`, or `random` to draw one from system entropy."""
    if text == "random":
        return Seed(secrets.randbits(32))
    digits = re.fullmatch(r"[0-9]+", text) is not None
    return Seed(int(text)) if digits and int(text) < 2**32 else None


_UINT64_MASK = 2**64 - 1


def mix(value: int) -> int:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def stream_key(seed: Seed, name: str) -> StreamKey:
    """The key of stream `name`, `mix(mix(seed ^ fnv1a64(name)))`, bit-identical to C++ `stream_key`."""
    hashed = 0xCBF29CE484222325
    for byte in name.encode():
        hashed = ((hashed ^ byte) * 0x100000001B3) & _UINT64_MASK
    return StreamKey(mix(mix(seed ^ hashed)))


@dataclass(frozen=True)
class Settings:
    """Every `STRINGZILLA_*` variable the Python suite reads, parsed once at import."""

    seed: Seed
    filter: str
    filter_pattern: re.Pattern | None
    scale: float
    in_qemu: bool

    def selects(self, name: str) -> bool:
        """Whether `STRINGZILLA_FILTER` selects the test `name`, as a regex or else as a substring."""
        return bool(self.filter_pattern.search(name)) if self.filter_pattern else self.filter in name


def read_settings() -> Settings:
    """Reads every `STRINGZILLA_*` variable, exiting with status 1 on the first that does not parse."""
    filter = env_text("STRINGZILLA_FILTER") or ""
    try:
        filter_pattern = re.compile(filter) if filter else None
    except re.error:
        filter_pattern = None
    return Settings(
        seed=env_seed("STRINGZILLA_SEED", Seed(42)),
        filter=filter,
        filter_pattern=filter_pattern,
        # Stress depth shared with the C++ suite: scales every fuzz baseline, so CI can shrink it.
        scale=env_parsed(
            "STRINGZILLA_SCALE",
            1.0,
            lambda text: scale if 0 < (scale := float(text)) < math.inf else None,
            "a positive number like 0.1 or 10",
        ),
        in_qemu=env_flag("STRINGZILLA_IN_QEMU", False),
    )


SETTINGS = read_settings()


def scale_iterations(baseline: int) -> int:
    return max(1, int(baseline * SETTINGS.scale))


def get_random_string(
    rng: Random, length: int | None = None, variability: int | None = None, alphabet: str = ascii_lowercase
) -> str:
    """Build a random string with optional fixed `length`, `alphabet`, and alphabet `variability`."""
    length = rng.randint(3, 300) if length is None else length
    variability = len(alphabet) if variability is None else variability
    return "".join(rng.choice(alphabet[:variability]) for _ in range(length))


def is_equal_strings(native_strings, big_strings):
    """Assert each StringZilla slice equals its native counterpart, pairwise."""
    for native_slice, big_slice in zip(native_strings, big_strings):
        assert native_slice == big_slice, f"Mismatch between `{native_slice}` and `{str(big_slice)}`"


DEGENERATE_HAYSTACKS = ["", "a", "hello world"]
DEGENERATE_BOUNDS = [-16, -6, -5, -1, 0, 1, 2, 5, 6, 10, 11, 12, 16, 99]
UNCASED_DEGENERATE_BOUNDS = [0, 1, 3, 4, 5, 6, 7, 8, 11, 99]
"""`utf8_uncased_search` counts negative offsets from 0, not from the end, so its bounds are
non-negative."""


# endregion General test scaffolding


# region Backend differential sweep
#
# A call passing `capabilities=` dispatches within that mask, so running the same input under
# `SERIAL` and under `SERIAL | NEON` exercises different kernel code through one binding, and any
# divergence is a kernel bug, not a binding bug. The helpers below drive that comparison over inputs
# engineered to stress the SIMD tail/boundary logic.


def capability_sweep():
    """All capability configurations every differential test should sweep over, the same list for every
    API, derived from the live hardware so no test hardcodes which backend its kernel uses.

    Returns: the serial baseline, then each enabled SIMD capability on its own (atop serial), then all of
    them together. Built from ``cpu_capabilities_enabled()``, so it adapts to the host (NEON / NEON-AES /
    NEON-SHA on Arm; Westmere/Haswell/Skylake/Ice Lake on x86) and collapses to just ``SERIAL`` on a
    machine with no SIMD capability. Each config is one ``sz.Capability`` mask, passed to a call as its
    ``capabilities=config`` argument. Sweeping every config across
    every API means an API whose kernel ignores a given capability simply re-runs the serial path under
    that config, harmless redundancy, while the config that does enable its SIMD path exercises it. Any
    divergence across the sweep is a kernel bug.
    """
    import stringzilla as sz

    enabled = sz.cpu_capabilities_enabled()
    simd = [capability for capability in sz.Capability if capability in enabled and capability != sz.Capability.SERIAL]
    sweep = [sz.Capability.SERIAL] + [sz.Capability.SERIAL | capability for capability in simd]
    if len(simd) > 1:
        sweep.append(enabled)
    return sweep


def assert_backends_agree(results, *, oracle=None, format_inputs=None):
    """Assert every backend agrees with the baseline, and the baseline matches ``oracle`` if given.
    ``format_inputs`` is a zero-arg callable rendered only on failure."""
    context = f" [{format_inputs()}]" if format_inputs is not None else ""
    baseline_config = next(iter(results))
    baseline = results[baseline_config]
    diverged = {config: value for config, value in results.items() if value != baseline}
    assert not diverged, f"backend divergence{context}: baseline {baseline_config}={baseline!r}; diverged={diverged!r}"
    if oracle is not None:
        assert baseline == oracle, f"kernels agree but disagree with the oracle{context}: {baseline!r} != {oracle!r}"


VECTOR_WIDTH_LENGTHS = (
    [0, 1, 2, 3]  # Degenerate and sub-word inputs
    + [7, 8, 9]  # 64-bit SWAR
    + [15, 16, 17]  # 128-bit NEON and SSE
    + [31, 32, 33, 47, 48]  # 256-bit AVX2, plus one and a half windows
    + [63, 64, 65, 95, 96]  # 512-bit AVX-512, plus one and a half windows
    + [127, 128, 129, 255, 256, 257]  # Several whole windows
    + [1024, 4096]  # Bulk, where the tail is a small fraction
)
"""Lengths bracketing the 16/32/64-byte SIMD register widths, plus a few larger tiers. Tail handling
and vector-boundary logic in the kernels is most likely to diverge from serial exactly at these sizes."""


def boundary_strings(alphabet: str = "ab") -> list[str]:
    """One deterministic string per `VECTOR_WIDTH_LENGTHS`, tiled from a small `alphabet` so substring
    and byteset kernels see frequent partial and full matches straddling the vector boundaries."""
    repeated = alphabet * (max(VECTOR_WIDTH_LENGTHS) // len(alphabet) + 1)
    return [repeated[:length] for length in VECTOR_WIDTH_LENGTHS]


def unaligned_views(text, offsets=(0, 1, 3, 7, 15)):
    """Yield ``(offset, sz.Str view)`` pairs that start at small byte offsets into a shared parent buffer.

    Slicing a ``Str`` shares the parent's allocation at the sliced offset, so the kernel receives a
    misaligned base pointer, the cheap way to exercise unaligned loads from Python."""
    import stringzilla as sz

    parent = sz.Str(text)
    for offset in offsets:
        if offset <= len(parent):
            yield offset, parent[offset:]


def malformed_utf8_corpus() -> list[bytes]:
    """Byte strings that are not valid UTF-8, to feed the ``utf8_*`` kernels. Every backend must agree
    on the result and stay in-bounds without crashing, the same malformed shapes the C++ `utf8.hpp`
    safety sweep uses, ported to Python `bytes`."""
    return [
        b"\x80",  # lone continuation byte
        b"\xc3",  # truncated 2-byte lead
        b"\xe0\xa0",  # truncated 3-byte sequence
        b"\xf0\x9f\x98",  # truncated 4-byte emoji (missing final byte)
        b"\xc0\x80",  # overlong encoding of NUL
        b"\xed\xa0\x80",  # UTF-16 surrogate U+D800 encoded as UTF-8
        b"\xf5\x80\x80\x80",  # lead byte for a codepoint past U+10FFFF
        b"a\x00b",  # embedded NUL
        b"\xff\xfe",  # bytes that never appear in valid UTF-8
        b"valid\xfftext",  # valid text with one invalid byte spliced in
    ]


def vector_width_bracketing_strings() -> list[str]:
    """ASCII-padded strings with a 2, 3, and 4-byte codepoint planted around the 16/32/64-byte SIMD lanes."""
    multibyte_codepoints = ["é", "中", "\U0001f600"]
    lane_widths = [16, 32, 64]
    strings = []
    for lane_width in lane_widths:
        for byte_offset in (-2, -1, 0, 1, 2):
            for codepoint in multibyte_codepoints:
                pad_length = max(0, lane_width + byte_offset - len(codepoint.encode("utf-8")))
                strings.append("a" * pad_length + codepoint + "tail")
    return strings


_BOUNDARY_TILED_STRINGS = boundary_strings()
_BOUNDARY_SAME_CHAR_STRINGS = boundary_strings(alphabet="a")
_BOUNDARY_STRINGS_BY_LENGTH = {
    length: (tiled, same_char)
    for length, tiled, same_char in zip(VECTOR_WIDTH_LENGTHS, _BOUNDARY_TILED_STRINGS, _BOUNDARY_SAME_CHAR_STRINGS)
}


def differential_bodies(length: int, rng: Random) -> tuple[str, str, str]:
    """Random, tiled, and all-same-character bodies of `length`; the random one drawn from `rng`."""
    tiled_body, same_char_body = _BOUNDARY_STRINGS_BY_LENGTH[length]
    return get_random_string(rng, length=length), tiled_body, same_char_body


# endregion Backend differential sweep
