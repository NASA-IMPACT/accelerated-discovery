"""Regression tests for lazy-loading of heavy optional dependencies.

akd-core must stay lightweight: importing the base package (or common
submodules) must not eagerly pull in slow/heavy dependencies like litellm
or gdown. Those are deferred to the call sites that actually use them.

Each guarantee runs in a fresh subprocess so this module's own imports (or
other tests in the same process) cannot mask a regression via an
already-populated sys.modules.
"""

import subprocess
import sys

import pytest

# Modules that must NOT be loaded as a side effect of the given import.
LAZY_GUARANTEES = [
    # Bare import must stay free of every heavy/optional dependency.
    ("import akd", ["litellm", "gdown", "instructor", "aiohttp", "requests", "dateparser"]),
    ("import akd._base.session", ["litellm"]),
    ("import akd.mapping.mappers", ["litellm"]),
    ("import akd.utils", ["gdown", "requests", "dateparser"]),
    # source_validator is on the bare path (pulled via akd.tools); aiohttp must stay lazy.
    ("import akd.tools.source_validator", ["aiohttp"]),
    # Scrapers must not drag in the docling/torch tree (lazy DoclingScraper gate).
    ("import akd.tools.scrapers", ["torch", "docling", "transformers", "fitz"]),
]


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    ("import_stmt", "forbidden"),
    LAZY_GUARANTEES,
    ids=[stmt for stmt, _ in LAZY_GUARANTEES],
)
def test_import_does_not_load_heavy_deps(import_stmt: str, forbidden: list[str]):
    code = (
        f"import sys; {import_stmt}; "
        f"loaded = [m for m in {forbidden!r} if m in sys.modules]; "
        f"assert not loaded, f'unexpectedly loaded: {{loaded}}'"
    )
    result = _run_in_subprocess(code)
    assert result.returncode == 0, f"{import_stmt!r} loaded forbidden modules:\n{result.stderr}"
