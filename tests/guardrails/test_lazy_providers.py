"""Regression tests for lazy-loaded guardrail providers.

Importing ``akd.guardrails`` (and ``akd.guardrails.providers``) must NOT pull
in heavy provider dependencies (deepeval via RiskAgent, litellm via the agent
stack). Providers are resolved lazily through a module ``__getattr__`` so that
downstream consumers can depend on the guardrails protocol layer cheaply.

Each guarantee runs in a fresh subprocess so this module's own imports (or
other tests in the same process) cannot mask a regression via an
already-populated sys.modules.
"""

import subprocess
import sys

import pytest

# Importing these must NOT load the listed heavy modules.
LAZY_GUARANTEES = [
    ("import akd.guardrails", ["litellm", "deepeval"]),
    ("import akd.guardrails.providers", ["litellm", "deepeval"]),
    ("from akd.guardrails.providers import GraniteGuardianTool", ["deepeval"]),
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
def test_guardrails_import_does_not_load_heavy_deps(import_stmt: str, forbidden: list[str]):
    code = (
        f"import sys; {import_stmt}; "
        f"loaded = [m for m in {forbidden!r} if m in sys.modules]; "
        f"assert not loaded, f'unexpectedly loaded: {{loaded}}'"
    )
    result = _run_in_subprocess(code)
    assert result.returncode == 0, f"{import_stmt!r} loaded forbidden modules:\n{result.stderr}"


def test_providers_lazy_names_resolve():
    """Every name in providers.__all__ resolves via module __getattr__."""
    code = "import akd.guardrails.providers as p; [getattr(p, name) for name in p.__all__]"
    result = _run_in_subprocess(code)
    assert result.returncode == 0, result.stderr


def test_providers_dir_lists_lazy_names():
    """dir() shows lazy names (REPL/tab-completion support)."""
    code = "import akd.guardrails.providers as p; assert 'RiskAgent' in dir(p) and 'GraniteGuardianTool' in dir(p)"
    result = _run_in_subprocess(code)
    assert result.returncode == 0, result.stderr


def test_risk_agent_importable_without_deepeval():
    """The RiskAgent class imports without deepeval (deferred to method calls)."""
    code = (
        "import sys; sys.modules['deepeval'] = None\n"
        "from akd.guardrails.providers import RiskAgent  # must not raise\n"
        "assert RiskAgent is not None"
    )
    result = _run_in_subprocess(code)
    assert result.returncode == 0, result.stderr


def test_risk_agent_missing_deepeval_error_is_friendly():
    """When deepeval is absent, instantiating RiskAgent explains the [risk_agent] extra."""
    code = (
        "import sys; sys.modules['deepeval'] = None\n"
        "from akd.guardrails.providers import RiskAgent\n"
        "try:\n"
        "    RiskAgent()\n"
        "except ModuleNotFoundError as e:\n"
        "    assert 'akd[risk_agent]' in str(e), str(e)\n"
        "else:\n"
        "    raise AssertionError('expected ModuleNotFoundError')"
    )
    result = _run_in_subprocess(code)
    assert result.returncode == 0, result.stderr


def test_guardrails_protocol_layer_importable_without_heavy_deps():
    """The contract downstream guardrail consumers depend on: protocol +
    categories + decorators + composite must import with heavy deps blocked."""
    blocked = ["deepeval", "litellm", "instructor", "crawl4ai", "gdown"]
    code = (
        f"import sys\n"
        f"for m in {blocked!r}: sys.modules[m] = None\n"
        "from akd.guardrails import (\n"
        "    CompositeGuardrail,\n"
        "    GuardrailInput,\n"
        "    GuardrailOutput,\n"
        "    GuardrailProtocol,\n"
        "    apply_guardrails,\n"
        "    guardrail,\n"
        ")\n"
        "from akd.guardrails.categories import RiskCategory, RiskMetadata\n"
    )
    result = _run_in_subprocess(code)
    assert result.returncode == 0, result.stderr
