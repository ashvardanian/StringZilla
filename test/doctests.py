"""Executes the ``>>>`` examples embedded in the StringZilla docstrings.

Every public method, property, and class documents itself with a runnable example.
This keeps those examples honest: if an API changes, the example fails here.

Run with::

    python -m pytest scripts/test_doctests.py -v
"""

import doctest
import hashlib

import pytest

import stringzilla as sz

# Names referenced by the docstring examples (e.g. ``>>> sz.Str(...)``). Doctests run in a
# namespace seeded with these, mirroring how a user would ``import stringzilla as sz``.
_OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL


def _run(module, extraglobs):
    result = doctest.testmod(module, extraglobs=extraglobs, optionflags=_OPTIONFLAGS, verbose=False)
    assert result.failed == 0, f"{result.failed} of {result.attempted} doctests failed in {module.__name__}"
    return result.attempted


def test_stringzilla_doctests():
    attempted = _run(sz, extraglobs={"sz": sz, "hashlib": hashlib})
    assert attempted > 0, "no doctests were discovered in stringzilla"


def test_stringzillas_doctests():
    # `stringzillas` is a separate, optional extension (parallel/GPU engines). Skip cleanly when it
    # is not importable here, including a partial or link-failed build, so the suite still runs the
    # core `stringzilla` doctests. CI builds it fully, where these examples execute for real.
    try:
        import stringzillas as szs
    except ImportError as error:
        pytest.skip(f"stringzillas not importable: {error}")
    _run(szs, extraglobs={"sz": sz, "szs": szs})
