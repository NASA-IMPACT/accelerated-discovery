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
#
# These guarantees are what let a downstream service install a slim extra
# (e.g. `akd[guardrails]`) and import the relevant surface without the heavy
# ML / browser / LLM-SDK trees. Heavy deps are declared in pyproject extras and
# lazy-imported at their call sites; each row below pins one import surface.
LAZY_GUARANTEES = [
    # Bare import must stay free of every heavy/optional dependency.
    ("import akd", ["litellm", "gdown", "instructor", "aiohttp", "requests", "dateparser"]),
    ("import akd._base.session", ["litellm"]),
    ("import akd.mapping.mappers", ["litellm"]),
    ("import akd.utils", ["gdown", "requests", "dateparser"]),
    # source_validator is on the bare path (pulled via akd.tools); aiohttp must stay lazy.
    ("import akd.tools.source_validator", ["aiohttp"]),
    # Scrapers must not drag in the docling/torch tree nor crawl4ai/playwright
    # (lazy DoclingScraper gate + deferred crawl4ai in web_scrapers).
    ("import akd.tools.scrapers", ["torch", "docling", "transformers", "fitz", "crawl4ai", "playwright"]),
    # Agents package stays light: the LiteLLM/instructor stack loads only when an
    # agent is instantiated/run, not at import.
    ("import akd.agents", ["litellm", "instructor", "openai", "tiktoken"]),
    # Guardrail contracts (schemas/protocol/categories) are ML/LLM-free — this is
    # the akd[guardrails] surface a downstream service codes against.
    ("import akd.guardrails", ["deepeval", "litellm", "torch", "crawl4ai"]),
    # Granite Guardian tool is httpx-only; must not pull the RiskAgent deps.
    ("from akd.guardrails.providers import GraniteGuardianTool", ["deepeval", "litellm"]),
    # The RiskAgent module itself is importable without deepeval loaded (deferred
    # to the methods that build/measure the DAG).
    ("import akd.guardrails.providers.risk_agent", ["deepeval"]),
    # Search tools stay free of numpy/reranker-ML and the scraper browser tree.
    ("import akd.tools.search", ["numpy", "crawl4ai", "playwright"]),
    # Reranker module imports without numpy/sentence-transformers/torch.
    ("import akd.tools.reranker", ["numpy", "sentence_transformers", "torch"]),
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


def test_pypaperbot_default_construction_stays_light():
    """Default PyPaperBotScraper() must not pull docling/torch at construction.

    The default DoclingScraper converter is built lazily on first `.scraper`
    access, so merely constructing a PyPaperBotScraper stays light.
    """
    code = (
        "import sys\n"
        "from akd.tools.scrapers.pypaperbot import PyPaperBotScraper\n"
        "PyPaperBotScraper()  # default construction — must not build DoclingScraper\n"
        "loaded = [m for m in ['torch', 'docling', 'transformers'] if m in sys.modules]\n"
        "assert not loaded, f'default construction eagerly loaded: {loaded}'"
    )
    result = _run_in_subprocess(code)
    assert result.returncode == 0, result.stderr
