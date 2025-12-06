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
from typing import Dict, List


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
            raise RuntimeError(f"Could not download UCD XML from {url}: {e}")
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

def parse_case_folding_file(filepath: str) -> Dict[int, bytes]:
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


def _download_case_folding_file(version: str) -> str:
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


def get_case_folding_rules(version: str = UNICODE_VERSION) -> Dict[int, bytes]:
    """Download and parse Unicode CaseFolding.txt, caching in temp directory.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping codepoints to their folded UTF-8 bytes
    """
    cache_path = _download_case_folding_file(version)
    return parse_case_folding_file(cache_path)


def get_case_folding_rules_as_codepoints(
    version: str = UNICODE_VERSION,
) -> Dict[int, List[int]]:
    """Download and parse Unicode CaseFolding.txt, returning target codepoints.

    Args:
        version: Unicode version string (e.g., "17.0.0")

    Returns:
        Dict mapping source codepoints to list of target codepoints
    """
    cache_path = _download_case_folding_file(version)

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
    cache_path = os.path.join(
        tempfile.gettempdir(), f"DerivedNormalizationProps-{version}.txt"
    )

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/DerivedNormalizationProps.txt"
        print(f"Downloading Unicode {version} DerivedNormalizationProps.txt from {url}...")
        try:
            urllib.request.urlretrieve(url, cache_path)
            print(f"Cached to {cache_path}")
        except Exception as e:
            raise RuntimeError(
                f"Could not download DerivedNormalizationProps.txt from {url}: {e}"
            )
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
