"""Check typing from a non-editable installed wheel, outside the source tree."""

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
from importlib.metadata import distributions
from pathlib import Path

import pytest
import stringzilla as sz

pytestmark = pytest.mark.skipif(
    os.environ.get("SZ_TARGET", "stringzilla") != "stringzilla", reason="base module typing"
)


@pytest.fixture
def installed_stubs():
    package = next(
        package
        for package in distributions(path=[str(Path(sz.__file__).parent)])
        if package.metadata["Name"] == "stringzilla"
    )
    if package.read_text("RECORD") is None:
        pytest.skip("typing discovery requires a non-editable wheel installation")
    source = package.read_text("direct_url.json")
    if source and json.loads(source).get("dir_info", {}).get("editable"):
        pytest.skip("typing discovery requires a non-editable wheel installation")
    stub = Path(package.locate_file("stringzilla-stubs/__init__.pyi"))
    assert stub.is_file()
    assert stub.with_name("py.typed").read_text() == "partial\n"
    return stub


def test_stub_package(installed_stubs):
    ast.parse(installed_stubs.read_text(), feature_version=(3, 10))


@pytest.mark.parametrize("checker", ["mypy", "pyrefly"])
@pytest.mark.parametrize("case", ["valid", "invalid"])
def test_installed_types(installed_stubs, tmp_path, checker, case):
    if importlib.util.find_spec(checker) is None:
        pytest.skip(f"{checker} is not installed")
    source = (Path(__file__).parent / "typing_cases" / f"{case}.py").read_text()
    consumer = tmp_path / f"{case}.py"
    consumer.write_text(source)
    environment = os.environ.copy()
    for key in ("PYTHONPATH", "MYPYPATH"):
        environment.pop(key, None)
    if checker == "mypy":
        command = [sys.executable, "-m", "mypy", "--strict", "--no-incremental", str(consumer)]
        location = rf"{case}\.py:(\d+): error:"
    else:
        command = [
            sys.executable,
            "-m",
            "pyrefly",
            "check",
            "--preset",
            "strict",
            "--python-interpreter-path",
            sys.executable,
            "--output-format=github",
            str(consumer),
        ]
        location = r"\bline=(\d+)"
    result = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, check=False, timeout=60
    )
    output = result.stdout + result.stderr
    expected = {index for index, line in enumerate(source.splitlines(), 1) if "# error" in line}
    if expected:
        assert result.returncode == 1, output
        assert {int(line) for line in re.findall(location, output)} == expected, output
    else:
        assert result.returncode == 0, output
