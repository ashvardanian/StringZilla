"""Release coverage checks using wheel filenames without building native extensions."""

import contextlib
import tempfile
import unittest
from pathlib import Path

from build_backend import cli_check_wheels


class WheelCoverageTests(unittest.TestCase):
    platforms = (
        "manylinux_2_17_x86_64",
        "manylinux_2_17_aarch64",
        "macosx_11_0_x86_64",
        "macosx_11_0_arm64",
        "win_amd64",
    )
    required = ("*manylinux*_x86_64*", "*manylinux*_aarch64*", "*macosx_*_x86_64*", "*macosx_*_arm64*", "*win_amd64*")
    pythons = ("310", "311", "312", "313", "314", "314t")

    @contextlib.contextmanager
    def wheels(self, pythons=None, version="5.3.0"):
        # The parallel runner shares the TestCase, so each invocation owns its files.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for python in self.pythons if pythons is None else pythons:
                for platform in self.platforms:
                    (directory / f"stringzilla-{version}-cp{python.rstrip('t')}-cp{python}-{platform}.whl").touch()
            yield directory

    def check(self, directory):
        cli_check_wheels(str(directory), self.required, self.pythons, "5.3.0")

    def test_complete_core_matrix_passes_without_long_tail_wheels(self):
        with self.wheels() as directory:
            self.check(directory)

    def test_a_single_python_does_not_cover_all_supported_versions(self):
        with self.wheels(["310"]) as directory:
            with self.assertRaises(SystemExit):
                self.check(directory)

    def test_a_missing_python_platform_pair_is_rejected(self):
        with self.wheels() as directory:
            (directory / "stringzilla-5.3.0-cp311-cp311-win_amd64.whl").unlink()
            with self.assertRaises(SystemExit):
                self.check(directory)

    def test_free_threaded_abi_is_required_independently(self):
        with self.wheels(self.pythons[:-1]) as directory:
            with self.assertRaises(SystemExit):
                self.check(directory)

    def test_an_older_release_cannot_fill_a_gap(self):
        with self.wheels(version="5.2.0") as directory:
            with self.assertRaises(SystemExit):
                self.check(directory)

    def test_empty_python_coverage_is_rejected(self):
        with self.wheels(()) as directory:
            with self.assertRaises(ValueError):
                cli_check_wheels(str(directory), self.required, [], "5.3.0")

    def test_existing_platform_only_callers_keep_their_policy(self):
        with self.wheels(["310"]) as directory:
            cli_check_wheels(str(directory), self.required)


if __name__ == "__main__":
    unittest.main()
