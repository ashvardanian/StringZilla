"""
Shared Unicode data loading utilities for StringZilla tests and notebooks.

This module provides functions to download and cache Unicode data files
(UCD XML, CaseFolding.txt, DerivedNormalizationProps.txt) for use in
tests and exploration notebooks.
"""

import os
import io
import tempfile
import zipfile
import urllib.request
import xml.etree.ElementTree as ET
from random import choice, randint, seed
from string import ascii_lowercase
from typing import Dict, List, Optional

# NumPy is available on most platforms and is required for many tests. PyPy on some platforms raises a weird
# error that is not an `ImportError`, so the naked `except` is a necessary evil mirrored from the old monolith.
try:
    import numpy as np

    numpy_available = True
except:  # noqa: E722
    numpy_available = False

# PyArrow is not available on most platforms; same defensive import contract as NumPy above.
try:
    import pyarrow as pa  # noqa: F401

    pyarrow_available = True
except:  # noqa: E722
    pyarrow_available = False


class UnicodeDataDownloadError(Exception):
    """Raised when Unicode data files cannot be downloaded."""

    pass


# Unicode version used for all Unicode data files
UNICODE_VERSION = "17.0.0"


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
            with zipfile.ZipFile(zip_bytes) as zf:
                xml_filename = zf.namelist()[0]
                with zf.open(xml_filename) as xml_file:
                    xml_content = xml_file.read()
                    with open(cache_path, "wb") as f:
                        f.write(xml_content)
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download UCD XML from {url}: {e}")
    else:
        print(f"Using cached Unicode {version} UCD XML: {cache_path}")

    tree = ET.parse(cache_path)
    return tree.getroot()


def get_all_codepoints(version: str = UNICODE_VERSION) -> List[int]:
    """Return all assigned/defined codepoints in Unicode."""
    root = get_unicode_xml_data(version)
    ns = {"ucd": "http://www.unicode.org/ns/2003/ucd/1.0"}
    codepoints = []
    for char in root.findall(".//ucd:char", ns):
        cp_str = char.get("cp")
        if cp_str:
            codepoints.append(int(cp_str, 16))
    return sorted(codepoints)


def parse_uncased_folding_file(filepath: str) -> Dict[int, bytes]:
    """Parse Unicode CaseFolding.txt into a dict: codepoint -> folded UTF-8 bytes.

    Uses status C (common) and F (full) mappings for full case folding.
    """
    folds = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
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
                target_cps = [int(x, 16) for x in parts[2].split("#")[0].strip().split()]
                # Convert target codepoints to UTF-8 bytes
                folded_str = "".join(chr(cp) for cp in target_cps)
                folds[codepoint] = folded_str.encode("utf-8")
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
                with open(cache_path, "wb") as f:
                    f.write(response.read())
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download CaseFolding.txt from {url}: {e}")

    return cache_path


def get_uncased_folding_rules(version: str = UNICODE_VERSION) -> Dict[int, bytes]:
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
) -> Dict[int, List[int]]:
    """Download and parse Unicode CaseFolding.txt, returning target codepoints.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping source codepoints to list of target codepoints
    """
    cache_path = _download_uncased_folding_file(version)

    folds = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
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
                source_cp = int(parts[0].strip(), 16)
                target_cps = [int(x, 16) for x in parts[2].split("#")[0].strip().split()]
                folds[source_cp] = target_cps
            except (ValueError, IndexError):
                continue
    return folds


def get_normalization_props(version: str = UNICODE_VERSION) -> Dict[int, Dict[str, str]]:
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
            urllib.request.urlretrieve(url, cache_path)
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download DerivedNormalizationProps.txt from {url}: {e}")
    else:
        print(f"Using cached Unicode {version} DerivedNormalizationProps.txt: {cache_path}")

    props: Dict[int, Dict[str, str]] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                cp_range = parts[0].strip()
                prop_value = parts[1].strip()

                # Parse property name and value (e.g., "NFC_QC" or "NFC_QC; N")
                if len(parts) >= 3:
                    prop_name = prop_value
                    prop_val = parts[2].strip()
                else:
                    # Properties like "Full_Composition_Exclusion" are boolean
                    prop_name = prop_value
                    prop_val = "Y"

                # Parse codepoint range
                if ".." in cp_range:
                    start, end = cp_range.split("..")
                    start_cp = int(start, 16)
                    end_cp = int(end, 16)
                else:
                    start_cp = end_cp = int(cp_range, 16)

                for cp in range(start_cp, end_cp + 1):
                    if cp not in props:
                        props[cp] = {}
                    props[cp][prop_name] = prop_val
            except (ValueError, IndexError):
                continue

    return props


def _download_word_break_property_file(version: str) -> str:
    """Download WordBreakProperty.txt and return the cache path."""
    cache_path = os.path.join(tempfile.gettempdir(), f"WordBreakProperty-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/WordBreakProperty.txt"
        print(f"Downloading Unicode {version} WordBreakProperty.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with open(cache_path, "wb") as f:
                    f.write(response.read())
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download WordBreakProperty.txt from {url}: {e}")

    return cache_path


def get_word_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
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

    props: Dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                cp_range = parts[0].strip()
                prop_name = parts[1].strip()

                # Parse codepoint range
                if ".." in cp_range:
                    start, end = cp_range.split("..")
                    start_cp = int(start, 16)
                    end_cp = int(end, 16)
                else:
                    start_cp = end_cp = int(cp_range, 16)

                for cp in range(start_cp, end_cp + 1):
                    props[cp] = prop_name
            except (ValueError, IndexError):
                continue

    return props


def _download_word_break_test_file(version: str) -> str:
    """Download WordBreakTest.txt and return the cache path."""
    cache_path = os.path.join(tempfile.gettempdir(), f"WordBreakTest-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/WordBreakTest.txt"
        print(f"Downloading Unicode {version} WordBreakTest.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with open(cache_path, "wb") as f:
                    f.write(response.read())
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download WordBreakTest.txt from {url}: {e}")

    return cache_path


def get_word_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    """Download and parse WordBreakTest.txt.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        List of tuples: (text: str, boundary_positions: List[int])
        boundary_positions includes 0 and len(text) as TR29 specifies.
    """
    cache_path = _download_word_break_test_file(version)

    test_cases = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue

            # Parse format: ÷ 0061 × 0027 × 0061 ÷
            # ÷ = break, × = no break
            # Numbers are hex codepoints
            parts = line.replace("÷", " ÷ ").replace("×", " × ").split()

            codepoints = []
            boundaries = []
            byte_pos = 0

            for part in parts:
                part = part.strip()
                if part == "÷":
                    boundaries.append(byte_pos)
                elif part == "×":
                    pass  # No boundary
                elif part:
                    try:
                        cp = int(part, 16)
                        codepoints.append(cp)
                        byte_pos += len(chr(cp).encode("utf-8"))
                    except ValueError:
                        continue

            if codepoints:
                text = "".join(chr(cp) for cp in codepoints)
                test_cases.append((text, boundaries))

    return test_cases


def baseline_word_boundaries(text: str, wb_props: Dict[int, str] = None) -> List[int]:
    """Pure Python implementation of TR29 word boundary algorithm.

    This serves as a reference baseline for testing the C implementation.

    Args:
        text: UTF-8 string to find word boundaries in.
        wb_props: Optional pre-loaded word break properties dict.

    Returns:
        List of byte positions that are word boundaries.
    """
    if wb_props is None:
        wb_props = get_word_break_properties()

    def get_wb_prop(cp: int) -> str:
        return wb_props.get(cp, "Other")

    # Convert text to list of (codepoint, byte_offset, byte_length)
    codepoints = []
    byte_offset = 0
    for char in text:
        cp = ord(char)
        cp_bytes = char.encode("utf-8")
        codepoints.append((cp, byte_offset, len(cp_bytes)))
        byte_offset += len(cp_bytes)

    if not codepoints:
        return [0]

    total_bytes = byte_offset
    boundaries = [0]  # WB1: sot ÷

    # Helper to check if a property is "ignorable" per WB4
    def is_ignorable(prop: str) -> bool:
        return prop in ("Extend", "Format", "ZWJ")

    # Helper to get effective property (skipping WB4 ignorables)
    def get_effective_prop_at(idx: int, direction: int = 1) -> str:
        """Get property at idx, or skip ignorables in direction."""
        while 0 <= idx < len(codepoints):
            prop = get_wb_prop(codepoints[idx][0])
            if not is_ignorable(prop):
                return prop
            idx += direction
        return "Other"

    def is_ahletter(prop: str) -> bool:
        return prop in ("ALetter", "Hebrew_Letter")

    def is_midnumletq(prop: str) -> bool:
        return prop in ("MidNumLet", "Single_Quote")

    # Check each position between codepoints
    for i in range(1, len(codepoints)):
        pos_bytes = codepoints[i][1]

        # Get properties before and after this position
        prev_cp = codepoints[i - 1][0]
        curr_cp = codepoints[i][0]
        prev_prop = get_wb_prop(prev_cp)
        curr_prop = get_wb_prop(curr_cp)

        # WB3: Do not break between CR and LF
        if prev_prop == "CR" and curr_prop == "LF":
            continue

        # WB3a: Break after CR, LF, Newline
        if prev_prop in ("CR", "LF", "Newline"):
            boundaries.append(pos_bytes)
            continue

        # WB3b: Break before CR, LF, Newline
        if curr_prop in ("CR", "LF", "Newline"):
            boundaries.append(pos_bytes)
            continue

        # WB3c: Do not break within emoji ZWJ sequences
        if prev_prop == "ZWJ":
            # Extended_Pictographic check would go here
            # For simplicity, just don't break after ZWJ
            continue

        # WB3d: Keep horizontal whitespace together
        if prev_prop == "WSegSpace" and curr_prop == "WSegSpace":
            continue

        # WB4: Ignore Format and Extend
        # Get effective properties skipping ignorables
        eff_prev = prev_prop
        if is_ignorable(prev_prop):
            # Look back for non-ignorable
            for j in range(i - 2, -1, -1):
                p = get_wb_prop(codepoints[j][0])
                if not is_ignorable(p):
                    eff_prev = p
                    break
            else:
                eff_prev = "Other"

        eff_curr = curr_prop
        # Note: we don't skip ignorables for current in simple cases

        # WB5: Do not break between letters
        if is_ahletter(eff_prev) and is_ahletter(eff_curr):
            continue

        # WB6: Do not break letters across punctuation (looking ahead)
        if is_ahletter(eff_prev) and curr_prop in ("MidLetter", "MidNumLet", "Single_Quote"):
            # Look ahead for AHLetter
            for j in range(i + 1, len(codepoints)):
                next_prop = get_wb_prop(codepoints[j][0])
                if is_ignorable(next_prop):
                    continue
                if is_ahletter(next_prop):
                    # Don't break: AHLetter × (MidLetter|MidNumLetQ) × AHLetter
                    break
                else:
                    boundaries.append(pos_bytes)
                    break
            else:
                boundaries.append(pos_bytes)
            continue

        # WB7: Reverse of WB6
        if curr_prop in ("MidLetter", "MidNumLet", "Single_Quote"):
            # Already handled by WB6
            pass

        if is_midnumletq(eff_prev) or eff_prev == "MidLetter":
            if is_ahletter(eff_curr):
                # Check if preceded by AHLetter
                for j in range(i - 2, -1, -1):
                    p = get_wb_prop(codepoints[j][0])
                    if is_ignorable(p):
                        continue
                    if is_ahletter(p):
                        # Don't break: AHLetter × (MidLetter|MidNumLetQ) × AHLetter
                        break
                    else:
                        boundaries.append(pos_bytes)
                        break
                else:
                    boundaries.append(pos_bytes)
                continue

        # WB8: Do not break between digits
        if eff_prev == "Numeric" and eff_curr == "Numeric":
            continue

        # WB9: Do not break letter-digit
        if is_ahletter(eff_prev) and eff_curr == "Numeric":
            continue

        # WB10: Do not break digit-letter
        if eff_prev == "Numeric" and is_ahletter(eff_curr):
            continue

        # WB11/12: Numeric separators
        if eff_prev == "Numeric" and curr_prop in ("MidNum", "MidNumLet", "Single_Quote"):
            # Look ahead for Numeric
            for j in range(i + 1, len(codepoints)):
                next_prop = get_wb_prop(codepoints[j][0])
                if is_ignorable(next_prop):
                    continue
                if next_prop == "Numeric":
                    break
                else:
                    boundaries.append(pos_bytes)
                    break
            else:
                boundaries.append(pos_bytes)
            continue

        if curr_prop in ("MidNum", "MidNumLet", "Single_Quote") and eff_curr == "Numeric":
            # WB12 - check preceding Numeric
            pass  # Handled above

        # WB13: Katakana
        if eff_prev == "Katakana" and eff_curr == "Katakana":
            continue

        # WB13a: Connect with ExtendNumLet
        if eff_prev in ("ALetter", "Hebrew_Letter", "Numeric", "Katakana", "ExtendNumLet"):
            if eff_curr == "ExtendNumLet":
                continue

        # WB13b: ExtendNumLet continues words
        if eff_prev == "ExtendNumLet":
            if eff_curr in ("ALetter", "Hebrew_Letter", "Numeric", "Katakana"):
                continue

        # WB15/16: Regional Indicators - keep pairs together
        if eff_prev == "Regional_Indicator" and eff_curr == "Regional_Indicator":
            # Count preceding RIs - if odd, don't break
            ri_count = 0
            for j in range(i - 1, -1, -1):
                p = get_wb_prop(codepoints[j][0])
                if p == "Regional_Indicator":
                    ri_count += 1
                elif is_ignorable(p):
                    continue
                else:
                    break
            if ri_count % 2 == 1:
                continue

        # WB999: Otherwise, break
        boundaries.append(pos_bytes)

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
                with open(cache_path, "wb") as f:
                    f.write(response.read())
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {e}")

    return cache_path


def _parse_break_property_file(cache_path: str) -> Dict[int, str]:
    """Parse a ``codepoint(..range) ; Property`` style UCD break-property file into a dict."""
    props: Dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                cp_range = parts[0].strip()
                prop_name = parts[1].strip()
                if ".." in cp_range:
                    start, end = cp_range.split("..")
                    start_cp = int(start, 16)
                    end_cp = int(end, 16)
                else:
                    start_cp = end_cp = int(cp_range, 16)
                for cp in range(start_cp, end_cp + 1):
                    props[cp] = prop_name
            except (ValueError, IndexError):
                continue
    return props


def _download_break_test_file(filename: str, version: str) -> str:
    """Download an auxiliary break-test file (e.g. GraphemeBreakTest.txt) and return the cache path."""
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with open(cache_path, "wb") as f:
                    f.write(response.read())
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {e}")

    return cache_path


def _parse_break_test_file(cache_path: str) -> List[tuple]:
    """Parse a UAX break-test file (``÷`` = break, ``×`` = no break) into (text, boundary_positions)."""
    test_cases = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace("÷", " ÷ ").replace("×", " × ").split()
            codepoints = []
            boundaries = []
            byte_pos = 0
            for part in parts:
                part = part.strip()
                if part == "÷":
                    boundaries.append(byte_pos)
                elif part == "×":
                    pass
                elif part:
                    try:
                        cp = int(part, 16)
                        codepoints.append(cp)
                        byte_pos += len(chr(cp).encode("utf-8"))
                    except ValueError:
                        continue
            if codepoints:
                text = "".join(chr(cp) for cp in codepoints)
                test_cases.append((text, boundaries))
    return test_cases


#  region Grapheme


def get_grapheme_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    """Download and parse GraphemeBreakProperty.txt.

    Returns a dict mapping codepoints to their Grapheme_Cluster_Break property name.
    Property names: CR, LF, Control, Extend, ZWJ, Regional_Indicator, Prepend,
    SpacingMark, L, V, T, LV, LVT, and (implicitly) Other.
    """
    cache_path = _download_break_property_file("GraphemeBreakProperty.txt", version)
    return _parse_break_property_file(cache_path)


def get_grapheme_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    """Download and parse the official GraphemeBreakTest.txt.

    Returns a list of (text: str, boundary_positions: List[int]) tuples; boundary
    positions are byte offsets and include 0 and len(text).
    """
    cache_path = _download_break_test_file("GraphemeBreakTest.txt", version)
    return _parse_break_test_file(cache_path)


def baseline_grapheme_boundaries(text: str, props: Dict[int, str] = None) -> List[int]:
    """Pure Python implementation of the UAX-29 extended grapheme cluster algorithm.

    Reference baseline for the C implementation. Returns byte positions of cluster boundaries.
    Covers GB1-GB999 including the Hangul, emoji-ZWJ (GB11) and Regional_Indicator (GB12/13) rules.
    """
    if props is None:
        props = get_grapheme_break_properties()

    def prop_of(cp: int) -> str:
        return props.get(cp, "Other")

    codepoints = []
    byte_offset = 0
    for char in text:
        cp = ord(char)
        cp_bytes = char.encode("utf-8")
        codepoints.append((cp, byte_offset, len(cp_bytes)))
        byte_offset += len(cp_bytes)

    if not codepoints:
        return [0]

    total_bytes = byte_offset
    boundaries = [0]  # GB1: sot ÷

    def is_extended_pictographic(cp: int) -> bool:
        # Extended_Pictographic is supplied via Emoji-Data; approximate from common ranges.
        return (
            0x1F000 <= cp <= 0x1FAFF
            or 0x2600 <= cp <= 0x27BF
            or cp in (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122, 0x2139, 0x2328, 0x2388)
            or 0x2194 <= cp <= 0x21AA
            or 0x231A <= cp <= 0x231B
            or 0x24C2 == cp
            or 0x25AA <= cp <= 0x25FE
            or 0x2934 <= cp <= 0x2935
            or 0x2B00 <= cp <= 0x2BFF
            or 0x1F1E6 <= cp <= 0x1F1FF
        )

    for i in range(1, len(codepoints)):
        pos_bytes = codepoints[i][1]
        prev_cp = codepoints[i - 1][0]
        curr_cp = codepoints[i][0]
        prev_prop = prop_of(prev_cp)
        curr_prop = prop_of(curr_cp)

        # GB3: CR x LF
        if prev_prop == "CR" and curr_prop == "LF":
            continue
        # GB4: (Control | CR | LF) ÷
        if prev_prop in ("Control", "CR", "LF"):
            boundaries.append(pos_bytes)
            continue
        # GB5: ÷ (Control | CR | LF)
        if curr_prop in ("Control", "CR", "LF"):
            boundaries.append(pos_bytes)
            continue
        # GB6: L x (L | V | LV | LVT)
        if prev_prop == "L" and curr_prop in ("L", "V", "LV", "LVT"):
            continue
        # GB7: (LV | V) x (V | T)
        if prev_prop in ("LV", "V") and curr_prop in ("V", "T"):
            continue
        # GB8: (LVT | T) x T
        if prev_prop in ("LVT", "T") and curr_prop == "T":
            continue
        # GB9: x (Extend | ZWJ)
        if curr_prop in ("Extend", "ZWJ"):
            continue
        # GB9a: x SpacingMark
        if curr_prop == "SpacingMark":
            continue
        # GB9b: Prepend x
        if prev_prop == "Prepend":
            continue
        # GB11: ExtPict Extend* ZWJ x ExtPict
        if prev_prop == "ZWJ" and is_extended_pictographic(curr_cp):
            j = i - 1
            # skip the ZWJ, then any Extend, looking for an Extended_Pictographic base
            k = j - 1
            while k >= 0 and prop_of(codepoints[k][0]) == "Extend":
                k -= 1
            if k >= 0 and is_extended_pictographic(codepoints[k][0]):
                continue
        # GB12/GB13: RI x RI when preceding RI count is even
        if prev_prop == "Regional_Indicator" and curr_prop == "Regional_Indicator":
            ri_count = 0
            for k in range(i - 1, -1, -1):
                if prop_of(codepoints[k][0]) == "Regional_Indicator":
                    ri_count += 1
                else:
                    break
            if ri_count % 2 == 1:
                continue
        # GB999: otherwise break
        boundaries.append(pos_bytes)

    boundaries.append(total_bytes)  # GB2: eot ÷
    return sorted(set(boundaries))


#  endregion Grapheme

#  region Sentence


def get_sentence_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    """Download and parse SentenceBreakProperty.txt.

    Returns a dict mapping codepoints to their Sentence_Break property name.
    Property names: CR, LF, Extend, Sep, Format, Sp, Lower, Upper, OLetter,
    Numeric, ATerm, SContinue, STerm, Close, and (implicitly) Other.
    """
    cache_path = _download_break_property_file("SentenceBreakProperty.txt", version)
    return _parse_break_property_file(cache_path)


def get_sentence_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    """Download and parse the official SentenceBreakTest.txt.

    Returns a list of (text: str, boundary_positions: List[int]) tuples; boundary
    positions are byte offsets and include 0 and len(text).
    """
    cache_path = _download_break_test_file("SentenceBreakTest.txt", version)
    return _parse_break_test_file(cache_path)


def baseline_sentence_boundaries(text: str, props: Dict[int, str] = None) -> List[int]:
    """Pure Python implementation of the UAX-29 sentence boundary algorithm.

    Reference baseline for the C implementation. Returns byte positions of sentence boundaries.
    Implements SB1-SB998 including the SB8/SB8a/SB9/SB10/SB11 ATerm/STerm machinery.
    """
    if props is None:
        props = get_sentence_break_properties()

    def prop_of(cp: int) -> str:
        return props.get(cp, "Other")

    codepoints = []
    byte_offset = 0
    for char in text:
        cp = ord(char)
        cp_bytes = char.encode("utf-8")
        codepoints.append((cp, byte_offset, len(cp_bytes)))
        byte_offset += len(cp_bytes)

    if not codepoints:
        return [0]

    total_bytes = byte_offset
    props_list = [prop_of(cp) for cp, _, _ in codepoints]

    def is_ignorable(prop: str) -> bool:
        # SB5: ignore Extend and Format for the purpose of the "before" context.
        return prop in ("Extend", "Format")

    def effective_before(idx: int) -> str:
        """Property of the last non-ignorable codepoint at or before idx (inclusive)."""
        k = idx
        while k >= 0 and is_ignorable(props_list[k]):
            k -= 1
        return props_list[k] if k >= 0 else "Other"

    def parasep(prop: str) -> bool:
        return prop in ("Sep", "CR", "LF")

    boundaries = [0]  # SB1: sot ÷

    n = len(codepoints)
    for i in range(1, n):
        pos_bytes = codepoints[i][1]
        prev_prop = props_list[i - 1]
        curr_prop = props_list[i]

        # SB3: CR x LF
        if prev_prop == "CR" and curr_prop == "LF":
            continue
        # SB4: Sep | CR | LF ÷
        if parasep(prev_prop):
            boundaries.append(pos_bytes)
            continue
        # SB5: x (Extend | Format) -> ignore (no break)
        if curr_prop in ("Extend", "Format"):
            continue

        eff_prev = effective_before(i - 1)

        # SB6: ATerm x Numeric
        if eff_prev == "ATerm" and curr_prop == "Numeric":
            continue
        # SB7: (Upper | Lower) ATerm x Upper
        if eff_prev == "ATerm" and curr_prop == "Upper":
            # find the codepoint before the ATerm (skipping ignorables)
            k = i - 2
            while k >= 0 and is_ignorable(props_list[k]):
                k -= 1
            if k >= 0 and props_list[k] in ("Upper", "Lower"):
                continue
        # Scan back over Close* and Sp* after an ATerm/STerm to find the sentence-terminator context.
        # SB8: ATerm Close* Sp* x (not (OLetter|Upper|Lower|Sep|CR|LF|STerm|ATerm))* Lower
        # SB8a: (STerm|ATerm) Close* Sp* x (SContinue | STerm | ATerm)
        k = i - 1
        while k >= 0 and is_ignorable(props_list[k]):
            k -= 1
        # walk back through Sp
        while k >= 0 and (props_list[k] == "Sp" or is_ignorable(props_list[k])):
            k -= 1
        # walk back through Close
        while k >= 0 and (props_list[k] == "Close" or is_ignorable(props_list[k])):
            k -= 1
        term_prop = props_list[k] if k >= 0 else "Other"

        if term_prop in ("ATerm", "STerm"):
            # SB8a
            if curr_prop in ("SContinue", "STerm", "ATerm"):
                continue
            if term_prop == "ATerm":
                # SB8: look ahead skipping the "not separator/terminator/letter" run to a Lower
                j = i
                lower_ahead = False
                while j < n:
                    p = props_list[j]
                    if p == "Lower":
                        lower_ahead = True
                        break
                    if p in ("OLetter", "Upper", "Sep", "CR", "LF", "STerm", "ATerm"):
                        break
                    j += 1
                if lower_ahead:
                    continue
            # SB9: (STerm|ATerm) Close* x (Close | Sp | Sep | CR | LF)
            if curr_prop in ("Close", "Sp", "Sep", "CR", "LF"):
                continue
            # SB11: after STerm/ATerm (with optional Close*/Sp*/ParaSep) -> break
            boundaries.append(pos_bytes)
            continue

        # SB998: otherwise, do not break
        continue

    boundaries.append(total_bytes)  # SB2: eot ÷
    return sorted(set(boundaries))


#  endregion Sentence

#  region Line


def get_line_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    """Download and parse LineBreak.txt.

    Returns a dict mapping codepoints to their Line_Break property name
    (e.g. BK, CR, LF, NL, SP, OP, CL, GL, BA, BB, B2, HY, AL, ID, NU, ...).
    """
    cache_path = _download_break_property_file("LineBreak.txt", version)
    return _parse_break_property_file(cache_path)


def get_line_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    """Download and parse the official LineBreakTest.txt.

    Returns a list of (text: str, boundary_positions: List[int]) tuples; boundary
    positions are byte offsets and include 0 and len(text). The test file marks only
    break opportunities, not whether they are mandatory.
    """
    cache_path = _download_break_test_file("LineBreakTest.txt", version)
    return _parse_break_test_file(cache_path)


def baseline_line_boundaries(text: str, props: Dict[int, str] = None) -> List[tuple]:
    """Pure Python reference for UAX-14 mandatory line breaks.

    Rather than reimplement the full pair-table (LB1-LB31), this baseline returns the set of
    *mandatory* break boundaries (LB4/LB5: BK, CR, LF, NL), which is the property the
    StringZilla line iterator's ``is_mandatory`` flag exposes. Soft break opportunities are
    library-specific and validated separately against ``uniseg``.

    Returns:
        List of (byte_offset, mandatory) tuples. ``mandatory`` is always True here (every
        returned offset is a hard break); the final eot boundary is reported as non-mandatory.
    """
    if props is None:
        props = get_line_break_properties()

    def prop_of(cp: int) -> str:
        return props.get(cp, "AL")

    codepoints = []
    byte_offset = 0
    for char in text:
        cp = ord(char)
        cp_bytes = char.encode("utf-8")
        codepoints.append((cp, byte_offset, len(cp_bytes)))
        byte_offset += len(cp_bytes)

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

    n = len(codepoints)
    for i in range(1, n):
        pos_bytes = codepoints[i][1]
        prev_prop = prop_of(codepoints[i - 1][0])
        curr_prop = prop_of(codepoints[i][0])

        # LB5: CR x LF -> no break between
        if prev_prop == "CR" and curr_prop == "LF":
            continue
        # LB4: BK !  (mandatory break after BK)
        # LB5: CR ! , LF ! , NL !  (mandatory break after each)
        if prev_prop in ("BK", "CR", "LF", "NL"):
            add(pos_bytes, True)
            continue

    # eot boundary (LB3) - reported but not mandatory
    add(total_bytes, False)
    return sorted(boundaries)


#  endregion Line

#  region General test scaffolding

# General fixtures and seeded-RNG helpers shared by every per-family test module (the Python analog of the
# C++ `test_stringzilla.hpp` harness). Kept here so a split test file imports one place for both the Unicode
# data loaders above and the seeding / random-string utilities below. The NumPy / PyArrow availability flags
# and their defensive imports live in the top import block.

# A random seed generated once at import time for this test run, logged by the `log_test_environment` fixture.
# `SystemRandom` gives true randomness independent of the seeded RNG state.
_random_seed_for_run = int.from_bytes(os.urandom(4), "little")

# Reproducible test seeds for consistent CI runs (kept in sync with test_stringzillas.py).
SEED_VALUES = [
    42,  # Classic test seed
    0,  # Edge case: zero seed
    1,  # Minimal positive seed
    314159,  # Pi digits
    _random_seed_for_run,  # Random seed for this run (logged at startup)
]

# Override SEED_VALUES with an environment variable if set (for reproducible CI fuzzing).
_env_seed = os.environ.get("SZ_TESTS_SEED")
if _env_seed:
    try:
        _parsed_seed = int(_env_seed)
        SEED_VALUES = [_parsed_seed]
        print(f"SZ_TESTS_SEED={_parsed_seed} (from environment, overriding default seeds)")
    except ValueError:
        pass  # Keep default SEED_VALUES if parsing fails


def seed_random_generators(seed_value: Optional[int] = None):
    """Seed Python and NumPy RNGs for reproducibility."""
    if seed_value is None:
        return
    seed(seed_value)
    # Handle both NumPy 1.x and 2.x, and any import issues.
    if numpy_available:
        try:
            np.random.seed(seed_value)
        except (ImportError, AttributeError, Exception):
            pass


def get_random_string(length: Optional[int] = None, variability: Optional[int] = None) -> str:
    """Build a random lowercase-ASCII string, with optional fixed `length` and alphabet `variability`."""
    if length is None:
        length = randint(3, 300)
    if variability is None:
        variability = len(ascii_lowercase)
    return "".join(choice(ascii_lowercase[:variability]) for _ in range(length))


def is_equal_strings(native_strings, big_strings):
    """Assert each StringZilla slice equals its native counterpart, pairwise."""
    for native_slice, big_slice in zip(native_strings, big_strings):
        assert native_slice == big_slice, f"Mismatch between `{native_slice}` and `{str(big_slice)}`"


#  endregion General test scaffolding
