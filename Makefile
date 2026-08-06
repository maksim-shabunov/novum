# =============================================================================
# NOVUM
#
#   make setup && make data && make train && make eval
#
# Every target is non-interactive and safe to run under tmux on a remote box.
# Override any variable on the command line:  make train TIER=myriad SEED=3
# =============================================================================

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

PYTHON      ?= python3
VENV        ?= .venv
BIN         := $(VENV)/bin
PY          := $(BIN)/python
PIP         := $(BIN)/pip

# Training + serving by default, per the project contract. Use
# `make setup EXTRAS=serve,dev` for a serving-only box (no torch, ~40x smaller).
EXTRAS      ?= train,serve,dev
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu

TIER        ?= rad750
SEED        ?= 0
TIERS       ?= rad750,myriad,snapdragon
SEEDS       ?= 0,1,2
CONFIG      ?= configs/tier_$(TIER).yaml
ARTIFACT    ?= artifacts/$(TIER).npz

HOST        ?= 127.0.0.1
PORT        ?= 8000

# Pinned versions, so a run today and a run in three weeks agree. See the file
# header for why these are not simply the newest releases.
CONSTRAINTS ?= constraints.txt
PIP_C       := $(if $(wildcard $(CONSTRAINTS)),-c $(CONSTRAINTS),)

# make doctor STRICT=1  -> warnings are failures too
STRICT      ?=
DOCTOR_ARGS := $(if $(STRICT),--strict,)

COMPOSE     ?= docker compose -f docker/docker-compose.yml

.PHONY: help bootstrap doctor setup data fetch preprocess train sweep eval serve \
        test lint fmt lock docker-train docker-serve docker-build docker-down \
        clean clean-data clean-venv guard-venv check-python check-deps

help: ## Show this help
	@echo "NOVUM -- onboard science data triage"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables: TIER=$(TIER) SEED=$(SEED) TIERS=$(TIERS) SEEDS=$(SEEDS) EXTRAS=$(EXTRAS)"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
bootstrap: ## Install system packages on a bare Debian/Ubuntu server
	bash scripts/bootstrap.sh

doctor: ## Diagnose the environment (run this first when something breaks)
	@$(PYTHON) scripts/doctor.py $(DOCTOR_ARGS)

# Everything `make setup` assumes about the host, checked before we touch
# anything -- each failure prints the exact command that fixes it, because a
# raw traceback from `python3 -m venv` helps nobody at 2am on a fresh box.
check-python:
	@command -v $(PYTHON) >/dev/null 2>&1 || { \
		echo "ERROR: '$(PYTHON)' is not on PATH."; \
		echo "  Debian/Ubuntu:  sudo apt-get install -y python3 python3-venv python3-pip"; \
		echo "  Or run:         bash scripts/bootstrap.sh"; \
		exit 1; }
	@$(PYTHON) -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || { \
		echo "ERROR: $(PYTHON) is $$($(PYTHON) -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])'), but NOVUM needs >= 3.10."; \
		echo "  bash scripts/bootstrap.sh     # prints your upgrade options"; \
		echo "  make setup PYTHON=python3.12  # if you already have a newer one"; \
		exit 1; }
	@$(PYTHON) -c 'import ensurepip' >/dev/null 2>&1 || { \
		echo "ERROR: $(PYTHON) cannot create a virtualenv -- ensurepip is missing."; \
		echo "  On Debian/Ubuntu the venv module ships in a separate package. Run:"; \
		echo "    sudo apt-get install -y python3-venv python$$($(PYTHON) -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv"; \
		echo "  Or run:  bash scripts/bootstrap.sh"; \
		exit 1; }

setup: check-python ## Create the venv and install training + serving deps
	@test -x $(PY) || $(PYTHON) -m venv $(VENV) || { \
		echo "ERROR: could not create a virtualenv at $(VENV)."; \
		echo "  sudo apt-get install -y python3-venv"; \
		echo "  Then:  rm -rf $(VENV) && make setup"; \
		exit 1; }
	$(PIP) install --upgrade pip setuptools wheel
	@if echo "$(EXTRAS)" | grep -q train; then \
		echo ">> installing CPU-only torch from $(TORCH_INDEX)"; \
		$(PIP) install --index-url $(TORCH_INDEX) $(PIP_C) torch; \
	fi
	$(PIP) install $(PIP_C) -e ".[$(EXTRAS)]"
	@$(PY) -c "import core; print('>> novum', core.__version__, 'importable')"
	@echo ""
	@$(PY) scripts/doctor.py || true
	@echo ">> ready. Next:  make data"

lock: guard-venv ## Freeze the full transitive dependency set for THIS machine
	$(PIP) freeze --exclude-editable > requirements-lock.txt
	@echo ">> wrote requirements-lock.txt ($$(wc -l < requirements-lock.txt) packages)"
	@echo "   This file is platform and interpreter specific. Regenerate it on the"
	@echo "   target machine; do not copy one between architectures."

guard-venv:
	@test -x $(PY) || { echo "No venv at $(VENV). Run: make setup"; exit 1; }

# -----------------------------------------------------------------------------
# Data pipeline
# -----------------------------------------------------------------------------
data: fetch preprocess ## Download and preprocess the dataset (~332 MB download)

fetch: guard-venv ## Download + extract the Zenodo archives (resumable)
	$(PY) -m scripts.fetch_data

preprocess: guard-venv ## Convert to memmapped float32 arrays + manifest
	$(PY) -m scripts.preprocess

# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
train: guard-venv ## Train one tier (default TIER=rad750)
	$(PY) -m scripts.train --config $(CONFIG) --out $(ARTIFACT) --seed $(SEED)

sweep: guard-venv ## Run the full (tier x seed) matrix, unattended
	$(PY) -m scripts.sweep --tiers $(TIERS) --seeds $(SEEDS)

eval: guard-venv ## Evaluate the current artifact and print ROC AUC
	$(PY) -m scripts.evaluate --artifact $(ARTIFACT)

# -----------------------------------------------------------------------------
# Serving
# -----------------------------------------------------------------------------
serve: guard-venv ## Run the API locally with reload
	$(BIN)/uvicorn api.main:app --host $(HOST) --port $(PORT) --reload

# -----------------------------------------------------------------------------
# Quality
# -----------------------------------------------------------------------------
test: guard-venv ## Run the test suite
	$(PY) -m pytest

check-deps: guard-venv ## Assert the API layer imports no training dependency
	$(PY) -m pytest tests/test_no_training_deps.py -v

lint: guard-venv ## Lint
	$(BIN)/ruff check .

fmt: guard-venv ## Auto-fix lint and format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

# -----------------------------------------------------------------------------
# Docker
# -----------------------------------------------------------------------------
docker-build: ## Build both images
	$(COMPOSE) --profile train build

docker-train: ## Run the full data + train pipeline in the training image
	$(COMPOSE) --profile train run --rm train

docker-serve: ## Build and run the API image (artifacts mounted read-only)
	$(COMPOSE) up --build api

docker-down: ## Stop and remove containers
	$(COMPOSE) --profile train down

# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------
clean: ## Remove caches and run logs (keeps data/ and artifacts/)
	rm -rf runs/sweep runs/metrics .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-data: ## Remove processed arrays only (keeps the downloaded zips)
	rm -rf data/processed

clean-venv: ## Remove the virtualenv
	rm -rf $(VENV)
