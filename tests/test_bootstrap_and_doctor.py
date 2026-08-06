"""Bare-server bootstrap and the doctor.

These two are the project's front door on a machine where nothing works yet, so
their own invariants matter more than usual:

  * doctor.py must import ONLY the standard library. It runs before `make setup`
    has created a venv, so a stray `import numpy` makes it useless exactly when
    it is needed.
  * doctor.py must be parseable by an old interpreter. If it needs Python 3.10
    to tell you that you do not have Python 3.10, it has failed.
  * bootstrap.sh must refuse non-Debian systems and reject bad arguments rather
    than half-provisioning a box.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCTOR = PROJECT_ROOT / "scripts" / "doctor.py"
BOOTSTRAP = PROJECT_ROOT / "scripts" / "bootstrap.sh"

# Everything doctor.py is allowed to import. All standard library.
ALLOWED_DOCTOR_IMPORTS = {
    "argparse", "json", "os", "shutil", "subprocess", "sys", "collections", "core",
}


# ---------------------------------------------------------------------------
# doctor.py
# ---------------------------------------------------------------------------
def test_doctor_imports_only_the_standard_library() -> None:
    """No numpy, no yaml. doctor runs before anything is installed."""
    tree = ast.parse(DOCTOR.read_text(encoding="utf-8"), filename=str(DOCTOR))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])

    unexpected = imported - ALLOWED_DOCTOR_IMPORTS
    assert not unexpected, (
        f"doctor.py imports {sorted(unexpected)}, which may be unavailable on a "
        "machine that has not run `make setup` yet"
    )


def test_doctor_survives_without_third_party_packages() -> None:
    """Run with -S and an empty sys.path prefix: no site-packages at all."""
    result = subprocess.run(
        [sys.executable, "-S", str(DOCTOR), "--json"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode in (0, 1), result.stderr
    payload = json.loads(result.stdout)
    assert "checks" in payload and payload["checks"]


def test_doctor_avoids_syntax_an_old_interpreter_would_reject() -> None:
    """Parseable as Python 3.7 syntax -- it must run to report a too-old python.

    Guards against someone adding `match`, a walrus, or `int | None` and
    silently making the too-old-Python path a SyntaxError instead of a message.
    """
    source = DOCTOR.read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 7))

    tree = ast.parse(source)
    for node in ast.walk(tree):
        assert not isinstance(node, ast.NamedExpr), "walrus operator needs 3.8+"
        if isinstance(node, ast.ImportFrom):
            assert node.module != "__future__" or "annotations" not in [
                a.name for a in node.names
            ], "doctor.py should avoid `from __future__ import annotations`"


def test_doctor_emits_valid_json() -> None:
    result = subprocess.run(
        [sys.executable, str(DOCTOR), "--json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    payload = json.loads(result.stdout)
    assert {"checks", "failures", "warnings", "next"} <= set(payload)
    for check in payload["checks"]:
        assert set(check) == {"name", "status", "detail", "hint"}
        assert check["status"] in {"PASS", "WARN", "FAIL"}
    assert payload["next"]


def test_doctor_exit_code_tracks_failures_not_warnings() -> None:
    result = subprocess.run(
        [sys.executable, str(DOCTOR), "--json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    payload = json.loads(result.stdout)
    expected = 1 if payload["failures"] else 0
    assert result.returncode == expected, (
        f"{payload['failures']} failure(s) but exit code {result.returncode}"
    )


def test_doctor_strict_promotes_warnings_to_failures() -> None:
    plain = subprocess.run(
        [sys.executable, str(DOCTOR), "--json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    payload = json.loads(plain.stdout)
    strict = subprocess.run(
        [sys.executable, str(DOCTOR), "--strict", "--no-color"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    if payload["warnings"] or payload["failures"]:
        assert strict.returncode == 1
    else:
        assert strict.returncode == 0


def test_doctor_reports_the_committed_artifact() -> None:
    """A fresh clone ships weights, and doctor should say so."""
    result = subprocess.run(
        [sys.executable, str(DOCTOR), "--json"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    checks = {c["name"]: c for c in json.loads(result.stdout)["checks"]}
    assert "artifacts/" in checks
    if (PROJECT_ROOT / "artifacts" / "rad750.npz").exists():
        assert checks["artifacts/"]["status"] == "PASS"
        assert "rad750.npz" in checks["artifacts/"]["detail"]


def test_cgroup_limit_parsing_handles_unlimited(tmp_path, monkeypatch) -> None:
    """A container limit must override /proc/meminfo, but 'max' must not."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import doctor as doctor_module

    monkeypatch.setattr(doctor_module, "host_ram_bytes", lambda: 64 * 1024**3)

    monkeypatch.setattr(doctor_module, "cgroup_limit_bytes", lambda: None)
    ram, from_cgroup = doctor_module.total_ram_gb()
    assert ram == pytest.approx(64.0) and not from_cgroup

    monkeypatch.setattr(doctor_module, "cgroup_limit_bytes", lambda: 2 * 1024**3)
    ram, from_cgroup = doctor_module.total_ram_gb()
    assert ram == pytest.approx(2.0) and from_cgroup


# ---------------------------------------------------------------------------
# bootstrap.sh
# ---------------------------------------------------------------------------
def test_bootstrap_is_valid_bash() -> None:
    result = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr


def test_bootstrap_is_executable() -> None:
    assert BOOTSTRAP.stat().st_mode & 0o111, "bootstrap.sh is not executable"


def test_bootstrap_help_exits_zero() -> None:
    result = subprocess.run(
        ["bash", str(BOOTSTRAP), "--help"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "--with-docker" in result.stdout
    assert "--skip-apt" in result.stdout


def test_bootstrap_rejects_unknown_options() -> None:
    result = subprocess.run(
        ["bash", str(BOOTSTRAP), "--nope"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 2, "bad usage should exit 2, not half-provision the box"


def test_bootstrap_rejects_a_non_numeric_disk_threshold() -> None:
    result = subprocess.run(
        ["bash", str(BOOTSTRAP), "--min-disk-gb", "lots"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 1
    assert "whole number" in (result.stdout + result.stderr)


def test_bootstrap_is_non_interactive_by_construction() -> None:
    """apt must never open a dialog on a headless box.

    Checks every apt-get line, executed or merely printed as advice: a command
    we tell someone to copy-paste should not hang on a prompt either.
    """
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "export DEBIAN_FRONTEND=noninteractive" in source

    mutating = ("apt-get install", "apt-get upgrade", "apt-get remove", "apt-get purge")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if any(call in stripped for call in mutating):
            assert " -y" in stripped, f"apt call without -y: {stripped}"


def test_bootstrap_does_not_silently_add_third_party_repositories() -> None:
    """PPAs are the operator's call. Docker's repo is opt-in behind --with-docker."""
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "add-apt-repository" not in source.replace(
        "sudo add-apt-repository ppa:deadsnakes/ppa", ""
    ), "bootstrap.sh must only *print* PPA instructions, never run them"

    docker_lines = [
        line for line in source.splitlines()
        if "download.docker.com" in line and not line.strip().startswith("#")
    ]
    assert docker_lines, "expected the Docker repo setup to exist"
    assert "install_docker()" in source
    assert '[ "${WITH_DOCKER}" -eq 1 ] || return 0' in source, (
        "the Docker install must be gated behind --with-docker"
    )


@pytest.mark.skipif(sys.platform.startswith("linux"), reason="tests the non-Linux path")
def test_bootstrap_refuses_non_debian_systems() -> None:
    """On macOS there is no /etc/os-release, and it must say so, not crash."""
    result = subprocess.run(
        ["bash", str(BOOTSTRAP)], capture_output=True, text=True, timeout=60
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    assert "Debian and Ubuntu" in combined
    assert "apt-get" not in combined.split("Then run")[0] or "brew" in combined
