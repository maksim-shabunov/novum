"""THE core architectural rule: the serving layer imports no training stack.

If this test fails, the API image is about to grow by ~2 GB and the separation
between offline training and online serving has been broken. The same check
runs inside docker/Dockerfile.api at build time.

The runtime check is the real one -- it imports the API in a clean subprocess
and inspects sys.modules, which catches a lazy import that a source grep would
miss.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BANNED = ("torch", "sklearn", "scipy", "pandas", "pyarrow", "matplotlib")

_PROBE = """
import sys

import {module}  # noqa: F401

banned = {banned!r}
leaked = sorted(m for m in banned if m in sys.modules)
if leaked:
    print("LEAKED:" + ",".join(leaked))
    raise SystemExit(1)
print("CLEAN")
"""


def _probe(module: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, banned=BANNED)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


@pytest.mark.parametrize(
    "module",
    ["core", "core.dataset", "core.scoring", "core.budgets", "core.models", "core.models.pca"],
)
def test_core_imports_no_training_dependency(module: str) -> None:
    result = _probe(module)
    assert result.returncode == 0, (
        f"importing {module} pulled in a training dependency: "
        f"{result.stdout.strip()} {result.stderr.strip()}"
    )


def test_api_imports_no_training_dependency() -> None:
    pytest.importorskip("fastapi", reason="serve extras not installed")
    result = _probe("api.main")
    assert result.returncode == 0, (
        "THE core architectural rule is broken: importing api.main pulled in a training "
        f"dependency. {result.stdout.strip()} {result.stderr.strip()}"
    )
    assert "CLEAN" in result.stdout


def test_the_registry_lists_stub_tiers_without_importing_torch() -> None:
    """Listing models must not import the modules that implement them."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from core.models.registry import available_models; "
            "names = available_models(); "
            "assert 'conv_ae_myriad' in names, names; "
            "assert 'torch' not in sys.modules; print('CLEAN')",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "CLEAN" in result.stdout


def test_api_sources_contain_no_training_imports() -> None:
    """Static backstop over the AST, so prose about torch does not trip it."""
    import ast

    offenders: list[str] = []
    for path in (PROJECT_ROOT / "api").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            offenders += [f"{path.name}:{node.lineno} {r}" for r in roots if r in BANNED]
    assert not offenders, f"training imports found in api/: {offenders}"
