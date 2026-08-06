"""Logging and progress reporting that behave under tmux, nohup and CI.

The progress reporter deliberately has two modes. On a terminal it redraws a
single line. Everywhere else -- a detached tmux pane piped to a file, a Docker
build log, CI -- it emits periodic newline-terminated updates instead, because
carriage-return bars turn multi-hour logs into unreadable single lines.
"""

from __future__ import annotations

import logging
import os
import sys
import time

_CONFIGURED = False


def _env_level(default: str = "INFO") -> int:
    name = os.environ.get("NOVUM_LOG_LEVEL", default).strip().upper()
    return getattr(logging, name, logging.INFO)


def setup_logging(level: int | str | None = None, *, force: bool = False) -> None:
    """Configure root logging once. Safe to call from every entry point."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    if level is None:
        level = _env_level()
    elif isinstance(level, str):
        level = getattr(logging, level.strip().upper(), logging.INFO)

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def plain_output() -> bool:
    """True when we should not use carriage-return redraws."""
    override = os.environ.get("NOVUM_PLAIN_PROGRESS", "").strip()
    if override:
        return override not in ("0", "false", "no", "")
    try:
        return not sys.stderr.isatty()
    except Exception:
        return True


def human_bytes(n: float) -> str:
    step = 1024.0
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < step or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} TiB"


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


class Progress:
    """Minimal dependency-free progress reporter (no tqdm in the dep tree).

    Usage:
        with Progress(total, "download train_typical.zip", unit="bytes") as p:
            p.advance(len(chunk))
    """

    def __init__(
        self,
        total: int | None,
        label: str,
        *,
        unit: str = "items",
        logger: logging.Logger | None = None,
        interval: float = 5.0,
        width: int = 28,
    ) -> None:
        self.total = int(total) if total else None
        self.label = label
        self.unit = unit
        self.log = logger or get_logger("novum.progress")
        self.interval = interval
        self.width = width
        self.current = 0
        self.plain = plain_output()
        self._start = time.monotonic()
        self._last_emit = 0.0
        self._closed = False

    def __enter__(self) -> Progress:
        return self

    def __exit__(self, *exc) -> None:
        self.close(failed=exc[0] is not None)

    def advance(self, n: int = 1) -> None:
        self.set(self.current + n)

    def set(self, value: int) -> None:
        self.current = int(value)
        now = time.monotonic()
        if now - self._last_emit >= (self.interval if self.plain else 0.1):
            self._last_emit = now
            self._emit()

    def _fmt_amount(self, n: float) -> str:
        return human_bytes(n) if self.unit == "bytes" else f"{int(n):,} {self.unit}"

    def _rate(self, elapsed: float) -> str:
        if elapsed <= 0:
            return ""
        r = self.current / elapsed
        return f" | {human_bytes(r)}/s" if self.unit == "bytes" else f" | {r:,.0f} {self.unit}/s"

    def _emit(self, *, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start
        amount = self._fmt_amount(self.current)
        if self.total:
            frac = min(1.0, self.current / self.total)
            pct = f"{frac * 100:5.1f}%"
            eta = ""
            if 0 < frac < 1 and elapsed > 1:
                eta = f" | eta {human_duration(elapsed / frac - elapsed)}"
            body = f"{pct} {amount}/{self._fmt_amount(self.total)}{self._rate(elapsed)}{eta}"
        else:
            frac = 0.0
            body = f"{amount}{self._rate(elapsed)}"

        if self.plain:
            if final:
                self.log.info(
                    "%s: done %s in %s", self.label, amount, human_duration(elapsed)
                )
            else:
                self.log.info("%s: %s", self.label, body)
        else:
            filled = int(self.width * frac) if self.total else 0
            bar = "#" * filled + "-" * (self.width - filled)
            end = "\n" if final else ""
            sys.stderr.write(f"\r  {self.label} [{bar}] {body}   {end}")
            sys.stderr.flush()

    def close(self, *, failed: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if failed and not self.plain:
            sys.stderr.write("\n")
            sys.stderr.flush()
            return
        self._emit(final=True)
