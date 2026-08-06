"""NOVUM doctor -- diagnose the environment. Run this first when something breaks.

    make doctor
    make doctor STRICT=1      # warnings become failures too
    python3 scripts/doctor.py --json

DELIBERATELY WRITTEN FOR OLD PYTHON AND ZERO DEPENDENCIES.

This is the one file in the project that must run on an interpreter that might
be too old, inside a checkout that might have no virtualenv, before anything is
installed. So: standard library only, no numpy, no yaml, no dataclasses, no
`from __future__ import annotations`, no PEP 604 unions. If this file needs
Python 3.10 to tell you that you do not have Python 3.10, it is useless.

Check categories:

  FAIL  a prerequisite for the *next* command is missing. Exit code 1.
  WARN  something a later step produces is absent, or is merely recommended.
        Not fatal, because `make doctor` is meant to be run immediately after
        bootstrap, before `make setup` has created any of it.
  PASS  fine.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import namedtuple

MIN_PYTHON = (3, 10)
MIN_FREE_DISK_GB = 12.0
RECOMMENDED_RAM_GB = 2.0

#: Without these, the very next step fails.
REQUIRED_BINARIES = ("git", "make", "curl", "unzip")
#: Strongly recommended, but nothing breaks immediately.
OPTIONAL_BINARIES = ("tmux",)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

Check = namedtuple("Check", ["name", "status", "detail", "hint"])

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Paths. Prefer core.paths so NOVUM_DATA_DIR is honoured identically, but never
# depend on it -- core/ may be unimportable on the interpreter running this.
# ---------------------------------------------------------------------------
def _resolve_dirs():
    try:
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from core import paths as core_paths

        return {
            "data": str(core_paths.data_dir()),
            "processed": str(core_paths.processed_dir()),
            "artifacts": str(core_paths.artifacts_dir()),
            "runs": str(core_paths.runs_dir()),
        }
    except Exception:
        def env_dir(var, default):
            raw = os.environ.get(var, "").strip()
            if not raw:
                return default
            return raw if os.path.isabs(raw) else os.path.join(PROJECT_ROOT, raw)

        data = env_dir("NOVUM_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
        return {
            "data": data,
            "processed": os.path.join(data, "processed"),
            "artifacts": env_dir("NOVUM_ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "artifacts")),
            "runs": env_dir("NOVUM_RUNS_DIR", os.path.join(PROJECT_ROOT, "runs")),
        }


DIRS = _resolve_dirs()


def venv_python():
    """Path to the venv interpreter, honouring a VENV override."""
    venv = os.environ.get("VENV") or ".venv"
    if not os.path.isabs(venv):
        venv = os.path.join(PROJECT_ROOT, venv)
    exe = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return os.path.join(venv, exe)


def free_gb(path):
    try:
        return shutil.disk_usage(path).free / (1024.0 ** 3)
    except OSError:
        return None


def cgroup_limit_bytes():
    """Memory ceiling imposed by a container, or None outside one.

    /proc/meminfo reports the HOST's memory inside a container, so a 2 GB
    container on a 64 GB host looks like 64 GB and the memory check passes
    right up until the OOM killer disagrees. NOVUM ships Docker images, so
    this matters.
    """
    candidates = (
        "/sys/fs/cgroup/memory.max",                    # cgroup v2
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",  # cgroup v1
    )
    for path in candidates:
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 uses a huge sentinel for "unlimited".
        if value <= 0 or value >= (1 << 62):
            return None
        return value
    return None


def host_ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    try:  # macOS and BSD have no SC_PHYS_PAGES
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL)
        return int(out.strip())
    except Exception:
        return None


def total_ram_gb():
    """Usable memory in GB, and whether a cgroup limit is what is binding."""
    host = host_ram_bytes()
    limit = cgroup_limit_bytes()
    if limit is not None and (host is None or limit < host):
        return limit / (1024.0 ** 3), True
    if host is None:
        return None, False
    return host / (1024.0 ** 3), False


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------
def check_python(venv_version):
    """Version of the interpreter that would build the venv.

    `venv_version` is the existing virtualenv's version tuple, or None. It
    matters: if a good venv already exists, everything runs on that, and a stale
    system python3 is a latent problem rather than a present one. Reporting it
    as FAIL then would send people chasing an interpreter they never use.
    """
    version = "%d.%d.%d" % sys.version_info[:3]
    want = "%d.%d" % MIN_PYTHON

    if sys.version_info >= MIN_PYTHON:
        return Check("python3 >= " + want, PASS, version + "  " + sys.executable, "")

    if venv_version is not None and venv_version >= MIN_PYTHON:
        return Check(
            "python3 >= " + want,
            WARN,
            "system python3 is %s, but the venv is %s and runs everything"
            % (version, "%d.%d.%d" % venv_version),
            "Fine as-is. Recreating the venv would fail; then use: make setup PYTHON=python3.12",
        )

    return Check(
        "python3 >= " + want,
        FAIL,
        version + " is too old",
        "Upgrade the distro (Ubuntu 22.04+ / Debian 12+), or: make setup PYTHON=python3.12",
    )


def check_binaries():
    checks = []
    for name in REQUIRED_BINARIES:
        found = shutil.which(name)
        checks.append(
            Check(
                "binary: " + name,
                PASS if found else FAIL,
                found or "not on PATH",
                "" if found else "bash scripts/bootstrap.sh",
            )
        )
    for name in OPTIONAL_BINARIES:
        found = shutil.which(name)
        checks.append(
            Check(
                "binary: " + name,
                PASS if found else WARN,
                found or "not on PATH (recommended for long runs)",
                "" if found else "bash scripts/bootstrap.sh",
            )
        )
    return checks


def check_disk():
    available = free_gb(PROJECT_ROOT)
    if available is None:
        return Check("free disk", WARN, "could not determine", "")
    detail = "%.1f GB free, need %.0f GB" % (available, MIN_FREE_DISK_GB)
    if available >= MIN_FREE_DISK_GB:
        return Check("free disk", PASS, "%.1f GB free" % available, "")
    return Check(
        "free disk",
        FAIL,
        detail,
        "Free space, or: export NOVUM_DATA_DIR=/mnt/big/novum-data",
    )


def check_ram():
    ram, from_cgroup = total_ram_gb()
    if ram is None:
        return Check("memory", WARN, "could not determine", "")
    source = " (container limit)" if from_cgroup else ""
    if ram >= RECOMMENDED_RAM_GB:
        return Check("memory", PASS, "%.1f GB%s" % (ram, source), "")
    return Check(
        "memory",
        WARN,
        "%.1f GB%s (%.0f GB recommended)" % (ram, source, RECOMMENDED_RAM_GB),
        "Preprocessing streams and fits in ~120 MB; installing torch is the tight "
        "step. Use: make setup EXTRAS=data,serve,dev",
    )


def venv_version():
    """Version tuple of the virtualenv interpreter, or None if unusable."""
    python = venv_python()
    if not os.path.exists(python):
        return None
    try:
        out = subprocess.check_output(
            [python, "-c", 'import sys; print("%d %d %d" % sys.version_info[:3])'],
            stderr=subprocess.DEVNULL,
        )
        return tuple(int(part) for part in out.decode("utf-8", "replace").split())
    except Exception:
        return None


def check_venv(version):
    python = venv_python()
    if not os.path.exists(python):
        return Check("virtualenv", WARN, "not created yet", "make setup")
    if version is None:
        return Check("virtualenv", FAIL, "present but not runnable",
                     "rm -rf .venv && make setup")
    if version < MIN_PYTHON:
        return Check(
            "virtualenv", FAIL,
            "python %d.%d.%d is below the %d.%d minimum" % (version + MIN_PYTHON),
            "rm -rf .venv && make setup PYTHON=python3.12",
        )
    return Check("virtualenv", PASS, "python %d.%d.%d" % version, "")


def check_venv_packages():
    python = venv_python()
    if not os.path.exists(python):
        return [Check("deps: core (numpy, yaml)", WARN, "no virtualenv yet", "make setup")]

    groups = (
        ("core (numpy, pyyaml)", "import numpy, yaml", "make setup", WARN),
        ("data (pandas, pyarrow)", "import pandas, pyarrow", "make setup EXTRAS=data,serve,dev", WARN),
        ("serve (fastapi)", "import fastapi", "make setup", WARN),
        ("train (torch)", "import torch", "make setup EXTRAS=train,serve,dev  # not needed for rad750", WARN),
    )
    checks = []
    for label, statement, hint, missing_status in groups:
        code = subprocess.call(
            [python, "-c", statement], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        checks.append(
            Check(
                "deps: " + label,
                PASS if code == 0 else missing_status,
                "importable" if code == 0 else "not installed",
                "" if code == 0 else hint,
            )
        )
    return checks


def check_processed_data():
    processed = DIRS["processed"]
    meta_path = os.path.join(processed, "meta.json")
    manifest = os.path.join(processed, "manifest.csv")

    if not os.path.isdir(processed):
        return [Check("data/processed", WARN, "not built yet", "make data")]
    if not os.path.exists(meta_path):
        return [Check("data/processed", WARN, "no meta.json", "make data")]

    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (ValueError, OSError) as exc:
        return [Check("data/processed", FAIL, "meta.json unreadable: %s" % exc,
                      "python3 -m scripts.preprocess --force")]

    splits = meta.get("splits", {})
    total = sum(int(s.get("count", 0)) for s in splits.values())
    missing = [
        name for name, info in splits.items()
        if not os.path.exists(os.path.join(processed, info.get("path", "")))
    ]

    checks = [
        Check(
            "data/processed",
            FAIL if missing else PASS,
            ("missing arrays: %s" % ", ".join(sorted(missing))) if missing
            else "%d splits, %d frames" % (len(splits), total),
            "python3 -m scripts.preprocess --force" if missing else "",
        ),
        Check(
            "data manifest",
            PASS if os.path.exists(manifest) else WARN,
            "manifest.csv" if os.path.exists(manifest) else "missing",
            "" if os.path.exists(manifest) else "make data",
        ),
    ]
    return checks


def check_artifacts():
    artifacts = DIRS["artifacts"]
    if not os.path.isdir(artifacts):
        return Check("artifacts/", WARN, "directory missing", "make train")
    weights = sorted(
        name for name in os.listdir(artifacts) if name.endswith(".npz")
    )
    if not weights:
        return Check("artifacts/", WARN, "no .npz weights", "make train")
    total_kb = sum(
        os.path.getsize(os.path.join(artifacts, name)) for name in weights
    ) / 1024.0
    return Check(
        "artifacts/",
        PASS,
        "%d artifact(s): %s (%.0f KB)" % (len(weights), ", ".join(weights), total_kb),
        "",
    )


def collect():
    version = venv_version()
    checks = [check_python(version)]
    checks.extend(check_binaries())
    checks.append(check_disk())
    checks.append(check_ram())
    checks.append(check_venv(version))
    checks.extend(check_venv_packages())
    checks.extend(check_processed_data())
    checks.append(check_artifacts())
    return checks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def colourise(status, enabled):
    if not enabled:
        return status
    colours = {PASS: "\033[32m", WARN: "\033[33m", FAIL: "\033[31m"}
    return colours.get(status, "") + status + "\033[0m"


def report(checks, use_colour):
    name_width = max(len(c.name) for c in checks)
    print("")
    print("=" * 72)
    print("  NOVUM doctor")
    print("=" * 72)
    for check in checks:
        print("  %-*s  %s  %s" % (name_width, check.name, colourise(check.status, use_colour), check.detail))
        if check.hint and check.status != PASS:
            print("  %-*s        -> %s" % (name_width, "", check.hint))
    print("=" * 72)

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    print("  %d passed, %d warning(s), %d failure(s)"
          % (len(checks) - len(failures) - len(warnings), len(warnings), len(failures)))
    return failures, warnings


def next_steps(checks):
    """Tell the operator the single next command, not a menu."""
    by_name = dict((c.name, c) for c in checks)

    def failing(name):
        c = by_name.get(name)
        return c is not None and c.status != PASS

    # A hard failure outranks any progress made further down the pipeline.
    hard = [c for c in checks if c.status == FAIL]
    if hard:
        return hard[0].hint or "bash scripts/bootstrap.sh"

    if failing("virtualenv") or failing("deps: core (numpy, pyyaml)"):
        return "make setup"
    if failing("data/processed") or failing("data manifest"):
        return "make data"
    if failing("artifacts/"):
        return "make train"
    return "make eval        # everything is in place"


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python3 scripts/doctor.py",
        description="Diagnose the NOVUM environment. Run this first when something breaks.",
    )
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures (useful in CI)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--no-color", action="store_true", help="disable colour")
    args = parser.parse_args(argv)

    checks = collect()

    if args.json:
        print(json.dumps(
            {
                "checks": [c._asdict() for c in checks],
                "failures": sum(1 for c in checks if c.status == FAIL),
                "warnings": sum(1 for c in checks if c.status == WARN),
                "next": next_steps(checks),
            },
            indent=2,
        ))
    else:
        use_colour = (
            not args.no_color
            and not os.environ.get("NO_COLOR")
            and hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
        )
        failures, warnings = report(checks, use_colour)
        print("  next: %s" % next_steps(checks))
        print("")
        if failures:
            print("  Required checks failed. Start with:  bash scripts/bootstrap.sh")
            print("")

    failures = [c for c in checks if c.status == FAIL]
    warnings = [c for c in checks if c.status == WARN]
    if failures:
        return 1
    if args.strict and warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
