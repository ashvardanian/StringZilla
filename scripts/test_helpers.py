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
        RuntimeError: If download fails
    """
    cache_path = os.path.join(tempfile.gettempdir(), f"CaseFolding-{version}.txt")

    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/CaseFolding.txt"
        try:
            urllib.request.urlretrieve(url, cache_path)
        except Exception as e:
            raise RuntimeError(f"Could not download CaseFolding.txt from {url}: {e}")

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
