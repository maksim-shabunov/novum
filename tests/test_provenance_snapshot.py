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


# ---------------------------------------------------------------------------
# Source-only dirtiness
# ---------------------------------------------------------------------------


def test_regenerating_an_artifact_is_not_a_dirty_source(repo: Path) -> None:
    """Training three tiers in sequence must not make the last two untraceable.

    The second run sees a tree the first run's artifact dirtied. That says
    nothing about whether the code was committed, which is the only thing the
    field is asked to answer.
    """
    (repo / "artifacts").mkdir()
    (repo / "artifacts" / "rad750.npz").write_text("weights", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "artifact")

    (repo / "artifacts" / "rad750.npz").write_text("retrained", encoding="utf-8")
    assert provenance.git_dirty(sources_only=False) is True, "the tree really did change"
    assert provenance.git_dirty() is False, "a regenerated artifact is not dirty source"


def test_uncommitted_code_is_still_dirty(repo: Path) -> None:
    """And the field must keep its teeth."""
    (repo / "train.py").write_text("print('edited')", encoding="utf-8")
    assert provenance.git_dirty() is True


# ---------------------------------------------------------------------------
# Building where the repository is not present
# ---------------------------------------------------------------------------


def test_a_container_can_declare_the_commit_it_was_built_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Modal or Docker build has no .git, so it must be told.

    Without this a container-built artifact records git_commit: null and is
    untraceable -- the exact failure the field exists to prevent.
    """
    monkeypatch.setenv(provenance.ENV_GIT_COMMIT, "deadbeefcafe")
    monkeypatch.setenv(provenance.ENV_GIT_BRANCH, "submission")
    monkeypatch.setenv(provenance.ENV_GIT_DIRTY, "false")
    provenance.reset_git_snapshot()

    assert provenance.git_commit() == "deadbeefcafe"
    assert provenance.git_branch() == "submission"
    assert provenance.git_dirty() is False


def test_a_declared_dirty_tree_is_believed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch must not be a way to always claim clean."""
    monkeypatch.setenv(provenance.ENV_GIT_DIRTY, "true")
    assert provenance.git_dirty() is True


def test_the_local_repository_still_wins_when_nothing_is_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (provenance.ENV_GIT_COMMIT, provenance.ENV_GIT_BRANCH,
                provenance.ENV_GIT_DIRTY):
        monkeypatch.delenv(var, raising=False)
    assert provenance.git_commit() is not None, "should read the real repository"
