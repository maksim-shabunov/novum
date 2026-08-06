#!/usr/bin/env bash
# =============================================================================
# NOVUM bootstrap -- bring a bare Debian/Ubuntu server to the point where
# `make setup` can run.
#
#     bash scripts/bootstrap.sh                # system packages only
#     bash scripts/bootstrap.sh --with-docker  # ...plus Docker from Docker's repo
#
# This script is the entry point precisely because `make` might not exist yet,
# so it assumes nothing beyond bash, coreutils and apt. It is idempotent: a
# second run installs nothing and simply reports state.
#
# It installs system packages and nothing else. It does not create the venv,
# download data, or train -- `make setup` and `make data` own those, so this
# stays re-runnable and quick.
# =============================================================================
set -euo pipefail

# apt must never open a dialog: this may run over a dropped-then-resumed ssh
# session, from cloud-init, or from CI.
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MIN_DISK_GB="${MIN_DISK_GB:-12}"
MIN_PY_MAJOR=3
MIN_PY_MINOR=10
RECOMMENDED_RAM_KB=$((2 * 1024 * 1024))   # 2 GiB

WITH_DOCKER=0
SKIP_APT=0

APT_PACKAGES=(
  make
  git
  curl
  unzip
  ca-certificates
  tmux
  python3
  python3-venv
  python3-pip
)

# --- output ------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
  C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'
else
  C_RESET=''; C_BOLD=''; C_RED=''; C_GREEN=''; C_YELLOW=''; C_BLUE=''
fi

info()  { printf '%s==>%s %s\n' "${C_BLUE}"   "${C_RESET}" "$*"; }
ok()    { printf '%s  ok%s %s\n' "${C_GREEN}"  "${C_RESET}" "$*"; }
warn()  { printf '%swarn%s %s\n' "${C_YELLOW}" "${C_RESET}" "$*" >&2; }
err()   { printf '%sFAIL%s %s\n' "${C_RED}"    "${C_RESET}" "$*" >&2; }
die()   { err "$*"; exit 1; }
have()  { command -v "$1" >/dev/null 2>&1; }

usage() {
  cat <<'EOF'
NOVUM bootstrap -- prepare a bare Debian/Ubuntu server.

Usage: bash scripts/bootstrap.sh [options]

Options:
  --with-docker     Also install Docker Engine + the compose plugin from
                    Docker's official apt repository (not the distro package,
                    which ships a Compose too old for docker-compose.yml).
                    Adds the invoking user to the docker group.
  --skip-apt        Run the preflight and version checks, install nothing.
                    Useful for re-verifying an already-provisioned box.
  --min-disk-gb N   Override the free-space requirement (default: 12).
  -h, --help        Show this message.

Environment:
  MIN_DISK_GB       Same as --min-disk-gb.
  NO_COLOR          Disable coloured output.

Installs: make git curl unzip ca-certificates tmux python3 python3-venv python3-pip
Exit codes: 0 ok | 1 precondition failed or install error | 2 bad usage
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --with-docker)   WITH_DOCKER=1 ;;
    --skip-apt)      SKIP_APT=1 ;;
    --min-disk-gb)   shift; [ $# -gt 0 ] || { usage >&2; exit 2; }; MIN_DISK_GB="$1" ;;
    --min-disk-gb=*) MIN_DISK_GB="${1#*=}" ;;
    -h|--help)       usage; exit 0 ;;
    *)               err "unknown option: $1"; echo; usage >&2; exit 2 ;;
  esac
  shift
done

case "${MIN_DISK_GB}" in
  ''|*[!0-9]*) die "--min-disk-gb must be a whole number, got '${MIN_DISK_GB}'" ;;
esac

printf '%s\n' "${C_BOLD}NOVUM bootstrap${C_RESET}"
printf 'project: %s\n\n' "${PROJECT_DIR}"

# --- 1. refuse non-Debian ----------------------------------------------------
require_debian() {
  info "checking operating system"

  if [ ! -r /etc/os-release ]; then
    die "no /etc/os-release: this does not look like a Linux distribution.
     bootstrap.sh provisions Debian and Ubuntu servers only.
     On macOS use Homebrew:  brew install python@3.12 git make tmux
     Then run:  make setup"
  fi

  # shellcheck disable=SC1091
  . /etc/os-release

  OS_ID="${ID:-unknown}"
  OS_NAME="${PRETTY_NAME:-${OS_ID}}"
  OS_CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"

  local is_debian=0 word
  for word in "${OS_ID}" ${ID_LIKE:-}; do
    case "${word}" in debian|ubuntu) is_debian=1 ;; esac
  done

  if [ "${is_debian}" -ne 1 ]; then
    die "unsupported distribution: ${OS_NAME}
     bootstrap.sh uses apt-get and targets Debian and Ubuntu.
     Install these yourself, then run 'make setup':
       make git curl unzip ca-certificates tmux python3 (>= 3.10) python3-venv python3-pip"
  fi

  if ! have apt-get; then
    die "${OS_NAME} reports as Debian-like but has no apt-get. Cannot continue."
  fi

  ok "${OS_NAME} (${OS_ID}${OS_CODENAME:+ ${OS_CODENAME}})"
}

# --- 2. preflight ------------------------------------------------------------
preflight_disk() {
  info "checking free disk space in ${PROJECT_DIR}"

  local free_kb free_gb
  # -P forces POSIX single-line output; without it long device names wrap and
  # the awk column index silently shifts.
  if ! free_kb="$(df -Pk "${PROJECT_DIR}" 2>/dev/null | awk 'NR==2 {print $4}')"; then
    warn "could not determine free space; continuing"
    return 0
  fi
  case "${free_kb}" in ''|*[!0-9]*) warn "unexpected df output; continuing"; return 0 ;; esac

  free_gb=$(awk -v kb="${free_kb}" 'BEGIN { printf "%.1f", kb / 1048576 }')
  FREE_DISK_GB="${free_gb}"

  if [ "${free_kb}" -lt $((MIN_DISK_GB * 1024 * 1024)) ]; then
    die "only ${free_gb} GB free in ${PROJECT_DIR}, need at least ${MIN_DISK_GB} GB.

     NOVUM needs roughly:
       0.4 GB  downloaded archives
       2.3 GB  extracted float64 frames
       1.2 GB  processed float32 arrays
       3.0 GB  virtualenv with CPU torch
       ~5 GB   headroom for apt, pip caches and sweep output

     Free some space, mount a larger volume, or point data elsewhere:
       export NOVUM_DATA_DIR=/mnt/big/novum-data
     Then re-run. To install anyway:  --min-disk-gb ${free_gb%.*}"
  fi

  ok "${free_gb} GB free (need ${MIN_DISK_GB} GB)"
}

# Inside a container /proc/meminfo shows the HOST's memory, so a 2 GB container
# on a 64 GB host looks fine right up until the OOM killer disagrees. Prefer the
# cgroup ceiling when one is set.
cgroup_limit_kb() {
  local path raw
  for path in /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    [ -r "${path}" ] || continue
    raw="$(cat "${path}" 2>/dev/null)" || continue
    if [ "${raw}" = "max" ]; then
      return 1
    fi
    case "${raw}" in ''|*[!0-9]*) continue ;; esac
    # cgroup v1 uses a huge sentinel value to mean "unlimited".
    if [ "${raw}" -ge 4611686018427387904 ]; then
      return 1
    fi
    echo $((raw / 1024))
    return 0
  done
  return 1
}

preflight_ram() {
  info "checking memory"

  local mem_kb mem_gb limit_kb mem_source=""
  if [ ! -r /proc/meminfo ]; then
    warn "cannot read /proc/meminfo; skipping the memory check"
    return 0
  fi
  mem_kb="$(awk '/^MemTotal:/ {print $2; exit}' /proc/meminfo)"
  case "${mem_kb}" in ''|*[!0-9]*) warn "unexpected /proc/meminfo; skipping"; return 0 ;; esac

  if limit_kb="$(cgroup_limit_kb)" && [ "${limit_kb}" -lt "${mem_kb}" ]; then
    mem_kb="${limit_kb}"
    mem_source=" (container limit)"
  fi

  mem_gb=$(awk -v kb="${mem_kb}" 'BEGIN { printf "%.1f", kb / 1048576 }')
  TOTAL_RAM_GB="${mem_gb}${mem_source}"

  if [ "${mem_kb}" -lt "${RECOMMENDED_RAM_KB}" ]; then
    warn "only ${mem_gb} GB RAM${mem_source} detected (2 GB recommended)."
    warn "  Preprocessing streams frame by frame and peaks near 120 MB, so it will"
    warn "  fit. The tight step is installing CPU torch. If pip is OOM-killed:"
    warn "    make setup EXTRAS=data,serve,dev   # rad750 needs no torch at all"
    warn "  or add swap:  fallocate -l 2G /swapfile && chmod 600 /swapfile \\"
    warn "                && mkswap /swapfile && swapon /swapfile"
  else
    ok "${mem_gb} GB RAM${mem_source}"
  fi
}

# --- privilege ---------------------------------------------------------------
resolve_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
    TARGET_USER="${SUDO_USER:-root}"
    return 0
  fi
  have sudo || die "not running as root and sudo is not installed.
     Re-run as root:  su -c 'bash scripts/bootstrap.sh'"
  SUDO="sudo"
  TARGET_USER="$(id -un)"
  if ! sudo -n true 2>/dev/null; then
    info "sudo will prompt for your password (apt needs root)"
  fi
}

# --- 3. packages -------------------------------------------------------------
apt_install() {
  if [ "${SKIP_APT}" -eq 1 ]; then
    info "--skip-apt: not installing system packages"
    return 0
  fi

  info "apt-get update"
  ${SUDO} apt-get update -qq

  info "installing: ${APT_PACKAGES[*]}"
  ${SUDO} apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"

  ok "system packages installed"
}

# --- 4. python version -------------------------------------------------------
check_python() {
  info "checking python3 version (need >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR})"

  have python3 || die "python3 is still not on PATH after apt install. Something is wrong
     with this system's package state. Try:  ${SUDO:-sudo} apt-get install -y --reinstall python3"

  PY_VERSION="$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"

  if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (${MIN_PY_MAJOR}, ${MIN_PY_MINOR}) else 1)"; then
    err "python3 is ${PY_VERSION}, but NOVUM needs >= ${MIN_PY_MAJOR}.${MIN_PY_MINOR}."
    cat >&2 <<EOF

     ${OS_NAME} does not ship a new enough Python.

     This script will NOT add a third-party PPA on your behalf -- that is your
     decision to make on your own server. Pick one:

     1. Upgrade the distribution (recommended).
          Ubuntu 22.04 ships 3.10, Ubuntu 24.04 ships 3.12,
          Debian 12 ships 3.11. All satisfy this requirement.
            sudo do-release-upgrade

     2. Add the deadsnakes PPA yourself (Ubuntu only), then re-run:
            sudo add-apt-repository ppa:deadsnakes/ppa
            sudo apt-get update
            sudo apt-get install -y python3.12 python3.12-venv
            make setup PYTHON=python3.12

     3. Install a private interpreter with pyenv (no root, no PPA):
            curl -fsSL https://pyenv.run | bash
            pyenv install 3.12 && pyenv local 3.12
            make setup PYTHON="\$(pyenv which python)"

     4. Skip the host Python entirely and use Docker:
            bash scripts/bootstrap.sh --with-docker
            make docker-train

EOF
    exit 1
  fi

  ok "python3 ${PY_VERSION}"

  # ensurepip lives in a separate package on Debian/Ubuntu and its absence is
  # the single most common way `make setup` fails on a fresh box. Catch it here,
  # where we can still fix it, rather than in the middle of setup.
  if ! python3 -c 'import ensurepip' >/dev/null 2>&1; then
    if [ "${SKIP_APT}" -eq 1 ]; then
      warn "python3 cannot create virtualenvs (ensurepip missing); --skip-apt so not fixing"
    else
      warn "python3-venv is incomplete (no ensurepip); installing the versioned package"
      ${SUDO} apt-get install -y --no-install-recommends \
        "python$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')-venv" \
        || die "could not install the venv package. Install it manually:
     ${SUDO:-sudo} apt-get install -y python3-venv"
    fi
  fi
  ok "python3 can create virtualenvs"
}

# --- 5. docker (optional) ----------------------------------------------------
install_docker() {
  [ "${WITH_DOCKER}" -eq 1 ] || return 0

  info "installing Docker Engine from Docker's official repository"

  if have docker && docker compose version >/dev/null 2>&1; then
    ok "docker and the compose plugin are already installed"
    DOCKER_INSTALLED=1
    return 0
  fi

  if dpkg -l docker.io 2>/dev/null | grep -q '^ii'; then
    warn "the distro 'docker.io' package is installed. Its Compose is too old for"
    warn "  docker-compose.yml (needs v2.24+). Consider: sudo apt-get remove -y docker.io"
  fi

  case "${OS_ID}" in
    ubuntu|debian) DOCKER_REPO_OS="${OS_ID}" ;;
    *)
      # Derivatives (Mint, Pop!_OS, Raspbian) have no Docker repo of their own.
      for word in ${ID_LIKE:-}; do
        case "${word}" in ubuntu) DOCKER_REPO_OS="ubuntu"; break ;; debian) DOCKER_REPO_OS="debian"; break ;; esac
      done
      ;;
  esac

  if [ -z "${DOCKER_REPO_OS:-}" ] || [ -z "${OS_CODENAME}" ]; then
    warn "cannot map ${OS_NAME} to a Docker repository; skipping the Docker install."
    warn "  Install it yourself:  https://docs.docker.com/engine/install/"
    return 0
  fi

  ${SUDO} install -m 0755 -d /etc/apt/keyrings
  if [ ! -s /etc/apt/keyrings/docker.asc ]; then
    ${SUDO} curl -fsSL "https://download.docker.com/linux/${DOCKER_REPO_OS}/gpg" \
      -o /etc/apt/keyrings/docker.asc
    ${SUDO} chmod a+r /etc/apt/keyrings/docker.asc
  fi

  printf 'deb [arch=%s signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/%s %s stable\n' \
    "$(dpkg --print-architecture)" "${DOCKER_REPO_OS}" "${OS_CODENAME}" \
    | ${SUDO} tee /etc/apt/sources.list.d/docker.list >/dev/null

  ${SUDO} apt-get update -qq
  ${SUDO} apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  DOCKER_INSTALLED=1

  if [ "${TARGET_USER}" != "root" ]; then
    ${SUDO} usermod -aG docker "${TARGET_USER}"
    DOCKER_GROUP_ADDED="${TARGET_USER}"
  fi

  ok "docker installed"
}

# --- 6. summary --------------------------------------------------------------
version_of() {
  if have "$1"; then
    "$@" 2>&1 | head -n1
  else
    echo "MISSING"
  fi
}

summary() {
  local free_now
  free_now="$(df -Pk "${PROJECT_DIR}" 2>/dev/null | awk 'NR==2 {printf "%.1f GB", $4 / 1048576}')"

  printf '\n%s\n' "${C_BOLD}================================================================${C_RESET}"
  printf '%s\n' "${C_BOLD}  bootstrap complete${C_RESET}"
  printf '%s\n\n' "${C_BOLD}================================================================${C_RESET}"

  printf '  %-14s %s\n' "os"      "${OS_NAME}"
  printf '  %-14s %s\n' "python3" "$(version_of python3 --version)"
  printf '  %-14s %s\n' "pip"     "$(version_of python3 -m pip --version)"
  printf '  %-14s %s\n' "make"    "$(version_of make --version)"
  printf '  %-14s %s\n' "git"     "$(version_of git --version)"
  printf '  %-14s %s\n' "curl"    "$(version_of curl --version)"
  printf '  %-14s %s\n' "unzip"   "$(unzip -v 2>/dev/null | head -n1 || echo MISSING)"
  printf '  %-14s %s\n' "tmux"    "$(version_of tmux -V)"
  if [ "${WITH_DOCKER}" -eq 1 ]; then
    printf '  %-14s %s\n' "docker"  "$(version_of docker --version)"
    printf '  %-14s %s\n' "compose" "$(docker compose version 2>/dev/null | head -n1 || echo MISSING)"
  fi
  printf '  %-14s %s\n' "free disk" "${free_now:-unknown}"
  printf '  %-14s %s\n' "ram"       "${TOTAL_RAM_GB:-unknown} GB"

  if [ -n "${DOCKER_GROUP_ADDED:-}" ]; then
    printf '\n%s\n' "${C_YELLOW}  NOTE: ${DOCKER_GROUP_ADDED} was added to the 'docker' group.${C_RESET}"
    printf '%s\n' "  Group membership is only picked up by NEW logins. Either:"
    printf '%s\n' "      exit and ssh back in"
    printf '%s\n' "  or, for this shell only:"
    printf '%s\n' "      newgrp docker"
    printf '%s\n' "  Until then, docker commands need sudo."
  fi

  printf '\n%s\n' "${C_BOLD}  next:${C_RESET}"
  printf '%s\n' "      cd ${PROJECT_DIR}"
  printf '%s\n' "      make doctor"
  printf '%s\n' "      make setup && make data && make train && make eval"
  printf '\n%s\n' "  Long runs belong in tmux so a dropped ssh session does not kill them:"
  printf '%s\n' "      tmux new -s novum"
  printf '%s\n' "      make setup && make data && make train && make eval"
  printf '%s\n' "      # detach with Ctrl-b then d ; reattach with: tmux attach -t novum"
  printf '\n'
}

main() {
  require_debian
  preflight_disk
  preflight_ram
  resolve_sudo
  apt_install
  check_python
  install_docker
  summary
}

main "$@"
