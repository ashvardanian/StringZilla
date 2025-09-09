"""
Custom build backend that adds NumPy as a conditional build dependency.

This backend wraps setuptools.build_meta and adds NumPy as a build
requirement only for stringzillas-cpus and stringzillas-cuda variants.

It also forwards PEP 660 editable builds to setuptools' build_meta,
so tools like `uv pip install -e .` work.
"""

import os
from setuptools import build_meta as _orig_build_meta


def get_requires_for_build_wheel(config_settings=None):
    """Get build requirements, conditionally including numpy."""
    requirements = _orig_build_meta.get_requires_for_build_wheel(config_settings)

    # Add NumPy for variants that need it
    sz_target = os.environ.get("SZ_TARGET", "stringzilla")
    if sz_target in ("stringzillas-cpus", "stringzillas-cuda"):
        requirements.append("numpy")

    return requirements


def get_requires_for_build_editable(config_settings=None):
    """Get build requirements for editable, conditionally including numpy (PEP 660)."""
    # Prefer setuptools' own hook if available; otherwise mimic wheel behavior
    if hasattr(_orig_build_meta, "get_requires_for_build_editable"):
        requirements = _orig_build_meta.get_requires_for_build_editable(config_settings)
    else:
        requirements = _orig_build_meta.get_requires_for_build_wheel(config_settings)

    sz_target = os.environ.get("SZ_TARGET", "stringzilla")
    if sz_target in ("stringzillas-cpus", "stringzillas-cuda"):
        requirements.append("numpy")
    return requirements


def get_requires_for_build_sdist(config_settings=None):
    """Get build requirements for sdist."""
    return _orig_build_meta.get_requires_for_build_sdist(config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build wheel."""
    return _orig_build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    """Build source distribution."""
    return _orig_build_meta.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    """Prepare metadata for wheel build."""
    return _orig_build_meta.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    """Prepare metadata for editable build (PEP 660)."""
    if hasattr(_orig_build_meta, "prepare_metadata_for_build_editable"):
        return _orig_build_meta.prepare_metadata_for_build_editable(metadata_directory, config_settings)
    raise RuntimeError(
        "Editable installs require setuptools with PEP 660 support. "
        "Please upgrade setuptools (setuptools>=61)."
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build editable wheel (PEP 660)."""
    if hasattr(_orig_build_meta, "build_editable"):
        return _orig_build_meta.build_editable(wheel_directory, config_settings, metadata_directory)
    raise RuntimeError(
        "Editable installs require setuptools with PEP 660 support. "
        "Please upgrade setuptools (setuptools>=61)."
    )
