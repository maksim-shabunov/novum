"""NOVUM FastAPI application.

    make serve          # uvicorn on NOVUM_API_HOST:NOVUM_API_PORT
    curl localhost:8000/health

STUB, but a live one. The artifact endpoints are real -- they read the
committed sidecars and metrics, so the API is useful the moment the repo is
cloned, with no training step. Scoring and simulation endpoints return 501 and
say what they will do.

ARCHITECTURAL RULE: no training dependency may be importable from here. This
module imports numpy (via `core`) and nothing heavier. `core.models.registry`
resolves model classes lazily, so listing artifacts never drags in a trainer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from core import __version__ as novum_version
from core import paths
from core.env import load_env
from core.logging_utils import get_logger, setup_logging

from .settings import Settings, get_settings

# Under uvicorn, importing this module IS process entry. Load credentials once,
# here, before any settings are read.
load_env()

log = get_logger("novum.api")

app = FastAPI(
    title="NOVUM",
    version=novum_version,
    summary="Onboard science data triage for a planetary rover.",
    description=(
        "Novelty scoring under a downlink budget and a compute budget. "
        "Training happens offline; this service only consumes trained weight artifacts."
    ),
)

_settings = get_settings()
setup_logging(_settings.log_level)

_cors_origins = _settings.allowed_origins
_cors_regex = _settings.cors_origin_regex
if _cors_origins or _cors_regex:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_origin_regex=_cors_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# ---------------------------------------------------------------------------
# Auth: optional, off unless NOVUM_API_KEY is set.
# ---------------------------------------------------------------------------
def require_api_key(
    authorization: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.auth_enabled:
        return
    expected = f"Bearer {settings.api_key}"
    if authorization != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Artifact discovery (real)
# ---------------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _artifact_summary(npz_path: Path) -> dict[str, Any]:
    sidecar = _read_json(npz_path.with_suffix(".json")) or {}
    metrics = _read_json(paths.artifacts_dir() / "metrics" / f"{npz_path.stem}.json") or {}
    return {
        "name": npz_path.stem,
        "file": npz_path.name,
        "bytes": npz_path.stat().st_size,
        "tier": sidecar.get("tier"),
        "model_type": (sidecar.get("config") or {}).get("model", {}).get("type"),
        "config_hash": sidecar.get("config_hash"),
        "git_commit": sidecar.get("git_commit"),
        "created_utc": sidecar.get("created_utc"),
        "param_count": sidecar.get("param_count"),
        "flops_per_inference": sidecar.get("flops_per_inference"),
        "roc_auc": (metrics.get("metrics") or {}).get("roc_auc"),
        "has_sidecar": bool(sidecar),
        "has_metrics": bool(metrics),
    }


def _list_artifacts() -> list[dict[str, Any]]:
    root = paths.artifacts_dir()
    if not root.is_dir():
        return []
    return [_artifact_summary(p) for p in sorted(root.glob("*.npz"))]


def _resolve(name: str) -> Path:
    """Resolve an artifact name, refusing anything that escapes artifacts/."""
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise HTTPException(status_code=400, detail=f"invalid artifact name {name!r}")
    root = paths.artifacts_dir().resolve()
    candidate = (root / (name if name.endswith(".npz") else f"{name}.npz")).resolve()
    if candidate.parent != root:
        raise HTTPException(status_code=400, detail=f"invalid artifact name {name!r}")
    if not candidate.exists():
        raise HTTPException(status_code=404, detail=f"no artifact named {name!r}")
    return candidate


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    """Liveness plus enough context to tell which build is running."""
    artifacts = _list_artifacts()
    return {
        "status": "ok",
        "version": novum_version,
        "artifacts": len(artifacts),
        "default_artifact": _settings.default_artifact,
        "auth_required": _settings.auth_enabled,
    }


@app.get("/api/artifacts", tags=["artifacts"])
def list_artifacts(_: None = Depends(require_api_key)) -> dict[str, Any]:
    """Every trained artifact committed to this repo, with its provenance."""
    artifacts = _list_artifacts()
    return {"count": len(artifacts), "artifacts": artifacts}


@app.get("/api/artifacts/{name}", tags=["artifacts"])
def get_artifact(name: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Full sidecar for one artifact: config, git commit, cost accounting."""
    path = _resolve(name)
    sidecar = _read_json(path.with_suffix(".json"))
    if sidecar is None:
        raise HTTPException(status_code=404, detail=f"no sidecar for artifact {name!r}")
    return {"summary": _artifact_summary(path), "provenance": sidecar}


@app.get("/api/artifacts/{name}/metrics", tags=["artifacts"])
def get_metrics(name: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Published evaluation metrics for one artifact."""
    path = _resolve(name)
    metrics = _read_json(paths.artifacts_dir() / "metrics" / f"{path.stem}.json")
    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"no published metrics for {name!r}; run `make eval`",
        )
    return metrics


# ---------------------------------------------------------------------------
# Mission-control console
#
# The console reads its precomputed grid as a static file straight from the web
# server; nothing here is on that path. What it does need from a Python process
# is the briefing, which is prose generated from the same decision log -- and
# which must work with no API key, so the deterministic template is the default
# rather than the fallback.
# ---------------------------------------------------------------------------
@app.get("/api/brief", tags=["console"])
def mission_brief(
    hardware: str = "rad750",
    tier: str = "rad750",
    budget: float = 0.25,
    adaptation: str = "frozen",
    policy: str = "score_first",
    llm: bool = False,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """Operator briefing for one console cell.

    `llm=true` asks the configured model to arrange the prose. Every figure is
    substituted from the decision log after generation either way, so the two
    modes differ in wording and never in numbers.
    """
    from core.ground.console_brief import brief_for_cell

    try:
        return brief_for_cell(
            hardware, tier, budget, adaptation, policy, offline=not llm
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"no precomputed run for cell {exc}"
        ) from exc


@app.get("/api/console/axes", tags=["console"])
def console_axes(_: None = Depends(require_api_key)) -> dict[str, Any]:
    """What the console can offer: every axis value with a precomputed run."""
    from core.ground.console_brief import load_grid

    try:
        grid = load_grid()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "axes": grid["axes"],
        "processors": grid.get("processors", {}),
        "default": grid["default"],
        "n_cells": len(grid["cells"]),
    }


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------
@app.post("/api/score", tags=["stub"], status_code=status.HTTP_501_NOT_IMPLEMENTED)
def score_frames(_: None = Depends(require_api_key)) -> dict[str, str]:
    """NOT IMPLEMENTED. Will score uploaded frames with a loaded artifact.

    Scoring needs only numpy: `core.models.registry.load_model(path).score(frames)`.
    The training stack stays out of this image.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "scoring endpoint not implemented yet. It will accept (N, 64, 64, 6) frames "
            "and return novelty scores from core.models.registry.load_model(...).score()."
        ),
    )


@app.get("/api/simulate", tags=["stub"], status_code=status.HTTP_501_NOT_IMPLEMENTED)
def simulate(_: None = Depends(require_api_key)) -> dict[str, str]:
    """NOT IMPLEMENTED. Will replay downlink windows via `sim.replay`."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="downlink simulation not implemented yet; see sim/window.py",
    )


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
