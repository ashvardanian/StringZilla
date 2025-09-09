"""
Custom build backend that adds NumPy as a conditional build dependency.

This backend wraps setuptools.build_meta and adds NumPy as a build
requirement only for stringzillas-cpus and stringzillas-cuda variants.
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
