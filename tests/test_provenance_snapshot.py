"""`git_dirty` must report the tree the run started from, not the one it made.

The field was structurally incapable of ever being false. `scripts/train.py`
writes `artifacts/<tier>.npz` and *then* records provenance, and the artifact is
tracked -- so by the time `git_dirty()` ran, the run had dirtied the tree itself.
Every sidecar in the repository said `git_dirty: true`, including ones produced
from a spotless checkout, which meant no result could be tied to a commit.

These tests pin the fix: state is frozen at process entry, before any write.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core import provenance


@pytest.fixture(autouse=True)
def _fresh_snapshot():
    provenance.reset_git_snapshot()
    yield
    provenance.reset_git_snapshot()


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real git repo with one committed file, so the tree starts clean."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    (root / "artifact.npz").write_text("v1", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")

    # core.provenance runs git against the project root; point it at ours.
    def _run(*args: str) -> tuple[bool, str]:
        out = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False
        )
        return out.returncode == 0, out.stdout.strip()

    monkeypatch.setattr(provenance, "_git_output", _run)
    monkeypatch.setattr(
        provenance, "_git",
        lambda *a: (lambda ok, t: (t or None) if ok else None)(*_run(*a)),
    )
    return root


def test_a_clean_tree_reports_clean(repo: Path) -> None:
    assert provenance.git_dirty() is False


def test_writing_an_artifact_dirties_the_live_tree(repo: Path) -> None:
    """The behaviour that made the field useless, demonstrated."""
    assert provenance.git_dirty() is False
    (repo / "artifact.npz").write_text("v2", encoding="utf-8")
    assert provenance.git_dirty() is True


def test_the_snapshot_survives_the_write(repo: Path) -> None:
    """The whole point: a run from a clean tree records itself as clean."""
    provenance.snapshot_git_state()          # process entry
    (repo / "artifact.npz").write_text("v2", encoding="utf-8")   # the run's output

    assert provenance.git_dirty() is True, "the live tree really is dirty now"
    assert provenance.git_state()["git_dirty"] is False, (
        "the sidecar would still claim the tree was dirty at entry"
    )


def test_a_genuinely_dirty_tree_is_still_reported(repo: Path) -> None:
    """The field must keep its meaning -- this is not a way to always say clean."""
    (repo / "artifact.npz").write_text("uncommitted", encoding="utf-8")
    provenance.snapshot_git_state()
    assert provenance.git_state()["git_dirty"] is True


def test_the_snapshot_is_idempotent(repo: Path) -> None:
    """Any entry point may call it; the first call is the one that counts."""
    first = provenance.snapshot_git_state()
    (repo / "artifact.npz").write_text("v2", encoding="utf-8")
    assert provenance.snapshot_git_state() == first


def test_without_a_snapshot_it_falls_back_to_the_live_tree(repo: Path) -> None:
    """A caller that never snapshots gets the old behaviour, not a stale lie."""
    (repo / "artifact.npz").write_text("v2", encoding="utf-8")
    assert provenance.git_state()["git_dirty"] is True


def test_every_writing_entry_point_snapshots_first() -> None:
    """A new script that writes a sidecar must not reintroduce the bug."""
    root = Path(__file__).resolve().parents[1]
    for name in ("train.py", "evaluate.py", "simulate.py"):
        source = (root / "scripts" / name).read_text(encoding="utf-8")
        assert "snapshot_git_state()" in source, (
            f"scripts/{name} writes provenance but never snapshots git state, "
            "so its sidecar will report the tree its own output dirtied"
        )
