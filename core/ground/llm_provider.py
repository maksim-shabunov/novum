"""Thin, swappable OpenRouter provider for the ground-side report.

The model id is a config value so any OpenRouter model can be swapped in:
  NOVUM_REPORT_MODEL=anthropic/claude-3.5-sonnet

If the key is absent or the request fails the caller receives ProviderError
and must fall back to the template path -- nothing in the demo may depend on
an external API being alive.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

# Pricing table for cost estimation: ($/1M input tokens, $/1M output tokens).
# Used only for the logged estimate; never billed here.
_PRICING: dict[str, tuple[float, float]] = {
    "ibm-granite/granite-4.1-8b": (0.05, 0.10),
    # fallback: assume a mid-range model if not in table
    "__default__": (0.50, 1.50),
}

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "ibm-granite/granite-4.1-8b"
REQUEST_TIMEOUT = 90  # seconds


# Why a request did not produce text. "LLM unavailable" tells an operator
# nothing they can act on; these do. Each maps to exactly one remedy.
REASON_NO_KEY = "key_missing"
REASON_AUTH_FAILED = "auth_failed"
REASON_RATE_LIMITED = "rate_limited"
REASON_REQUEST_FAILED = "request_failed"
REASON_OFFLINE_REQUESTED = "offline_requested"
REASON_UNSANCTIONED_FIGURES = "unsanctioned_figures"

#: One line per reason, written for whoever has to fix it.
REASON_HELP: dict[str, str] = {
    REASON_NO_KEY: "OPENROUTER_API_KEY is not set -- add it to .env or export it",
    REASON_AUTH_FAILED: "OpenRouter rejected the API key -- check it is current and funded",
    REASON_RATE_LIMITED: "OpenRouter rate-limited the request -- retry later or slow the run",
    REASON_REQUEST_FAILED: "the OpenRouter request failed -- see the detail below",
    REASON_OFFLINE_REQUESTED: "--offline was requested; the provider was never called",
    REASON_UNSANCTIONED_FIGURES: (
        "the model wrote figures instead of the placeholders it was given, so the "
        "generation was discarded (the offending values are in the run log)"
    ),
}

#: Reasons whose detail must never be echoed into the published report.
#: `unsanctioned_figures` carries the very numbers we refused to publish --
#: quoting them in the footer would put the hallucination back in the document
#: through the door marked "error message". They stay in the log and in
#: report_meta.json, where a maintainer looks and a reader does not.
DETAIL_NOT_FOR_PUBLICATION = frozenset(
    {REASON_OFFLINE_REQUESTED, REASON_UNSANCTIONED_FIGURES}
)


class ProviderError(RuntimeError):
    """Raised when the LLM provider is unavailable or returns an error.

    Carries a machine-readable `reason` so the caller can say *why* it fell
    back to templates rather than reporting an unactionable "unavailable".
    """

    def __init__(self, message: str, *, reason: str = REASON_REQUEST_FAILED) -> None:
        super().__init__(message)
        self.reason = reason

    @property
    def detail(self) -> str:
        return str(self)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class LLMResponse:
    text: str
    usage: Usage = field(default_factory=Usage)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = _PRICING.get(model, _PRICING["__default__"])
    return (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000.0


def get_model() -> str:
    """Model id from env, falling back to the Granite default."""
    return os.environ.get("NOVUM_REPORT_MODEL", "").strip() or DEFAULT_MODEL


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ProviderError("OPENROUTER_API_KEY is not set", reason=REASON_NO_KEY)
    return key


def complete(
    system: str,
    user: str,
    *,
    model: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> LLMResponse:
    """Send a single chat completion request to OpenRouter.

    Raises ProviderError on any failure -- absent key, network error, non-200
    response, or malformed JSON.  The caller is responsible for the fallback.
    """
    resolved_model = model or get_model()
    api_key = _get_api_key()  # raises ProviderError if missing

    payload = json.dumps(
        {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        OPENROUTER_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/novum-mars",
            "X-Title": "NOVUM Mission Brief",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        # 401/403 and 429 have different remedies from a generic failure, and
        # from each other: a bad key is not a busy server.
        if exc.code in (401, 403):
            reason = REASON_AUTH_FAILED
        elif exc.code == 429:
            reason = REASON_RATE_LIMITED
        else:
            reason = REASON_REQUEST_FAILED
        raise ProviderError(
            f"HTTP {exc.code} from OpenRouter: {detail[:400]}", reason=reason
        ) from exc
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise ProviderError(
            f"OpenRouter request failed: {exc}", reason=REASON_REQUEST_FAILED
        ) from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(
            f"OpenRouter returned non-JSON: {body[:200]}", reason=REASON_REQUEST_FAILED
        ) from exc

    if "error" in data:
        # A 200 body can still carry an error object, rate limits included.
        code = str((data["error"] or {}).get("code", "")) if isinstance(data["error"], dict) else ""
        reason = REASON_RATE_LIMITED if code == "429" else REASON_REQUEST_FAILED
        raise ProviderError(f"OpenRouter error: {data['error']}", reason=reason)

    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise ProviderError(
            f"unexpected OpenRouter response shape: {body[:300]}",
            reason=REASON_REQUEST_FAILED,
        ) from exc

    raw_usage = data.get("usage") or {}
    prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
    completion_tokens = int(raw_usage.get("completion_tokens", 0))
    cost = _estimate_cost(resolved_model, prompt_tokens, completion_tokens)

    return LLMResponse(
        text=text,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=resolved_model,
            cost_usd=cost,
        ),
    )
