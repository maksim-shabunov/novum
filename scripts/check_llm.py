"""Is the ground-side LLM actually reachable, right now, with this key?

    make check-llm
    python -m scripts.check_llm --model anthropic/claude-3.5-sonnet

Sends ONE minimal request and reports what came back: model, latency, tokens,
estimated cost. When it cannot, it says which of the four things went wrong --
key missing, key rejected, rate limited, request failed -- and what to do about
it. The mission brief silently falling back to templates is the failure mode
this target exists to make loud, because "LLM unavailable" in a generated
report is not something an operator can act on.

Exit status is 0 only when a completion actually came back, so CI can gate on
it. Nothing else in the project depends on this passing.
"""

from __future__ import annotations

import argparse
import sys
import time

from core import paths
from core.env import load_env
from core.ground.llm_provider import (
    REASON_HELP,
    ProviderError,
    complete,
    get_model,
)
from core.logging_utils import setup_logging

#: Deliberately tiny: this is a reachability check, not a capability test.
PROBE_SYSTEM = "You are a connectivity probe. Answer with one word."
PROBE_USER = "Reply with the single word: OK"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.check_llm",
        description=(
            "Send one minimal request to the configured LLM provider and report "
            "the model, latency, tokens and cost -- or the exact reason it failed."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=None, help="Override the model id for this check.")
    p.add_argument(
        "--max-tokens", type=int, default=8, help="Cap on the probe's completion length."
    )
    p.add_argument("--log-level", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level, force=args.log_level is not None)

    loaded = load_env()
    print(f"env file:  {paths.rel(loaded) if loaded else 'none (os.environ only)'}")

    model = args.model or get_model()
    print(f"model:     {model}")

    started = time.monotonic()
    try:
        resp = complete(
            PROBE_SYSTEM,
            PROBE_USER,
            model=model,
            max_tokens=args.max_tokens,
            temperature=0.0,
        )
    except ProviderError as exc:
        elapsed_ms = (time.monotonic() - started) * 1000
        print(f"latency:   {elapsed_ms:.0f} ms")
        print(f"result:    UNAVAILABLE ({exc.reason})")
        print(f"why:       {REASON_HELP.get(exc.reason, 'no further detail')}")
        print(f"detail:    {str(exc)[:500]}")
        if exc.reason == "key_missing":
            print(
                "\nSet it in .env at the project root (the file is gitignored):\n"
                "    OPENROUTER_API_KEY=sk-or-...\n"
                "An exported OPENROUTER_API_KEY always wins over .env."
            )
        return 1

    elapsed_ms = (time.monotonic() - started) * 1000
    usage = resp.usage
    print(f"latency:   {elapsed_ms:.0f} ms")
    print(f"result:    OK — {resp.text.strip()[:60]!r}")
    print(
        f"tokens:    {usage.prompt_tokens} in + {usage.completion_tokens} out "
        f"= {usage.total_tokens} total"
    )
    print(f"cost:      ${usage.cost_usd:.6f} (estimated from the local pricing table)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
