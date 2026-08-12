"""Model client used by the workflow harness and the evaluation judge.

Not in the specification's file tree. It exists because the pinned `anthropic`
SDK was replaced with Gemini on the repo owner's instruction, and the free tier
that decision implies needs request throttling and retry handling that does not
belong inside a workflow definition. That substitution is a documented deviation
from the pinned stack; see the README limitations.

Two things here are load-bearing for the research, not conveniences:

*Notional cost.* Generation runs on the free tier, so actual spend is zero. If
`cost_usd` were recorded as zero the cost view (FR-28) would be vacuous and
invariant 5 would reconcile trivially. Costs are therefore computed from the
model's published list price and are *notional*. The README states this
plainly; no number in this repository should be read as money that was spent.

*Stub mode.* `StubClient` produces deterministic canned responses so the state
machines, fault injection and corpus plumbing can be exercised without a key or
quota. Traces it produces are for smoke-testing only and are never committed as
corpus: `generate_corpus.py` writes them to a separate directory and stamps
`workflow_version` with a `+stub` suffix so they cannot be mistaken for real
runs.
"""

from __future__ import annotations

import hashlib
import os
import random
import re
import threading
import time
from dataclasses import dataclass

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMUnavailable",
    "GeminiClient",
    "StubClient",
    "build_client",
    "notional_cost_usd",
]


# Published list prices in USD per million tokens (input, output), used only to
# compute the notional cost described in the module docstring. Models absent
# from this table fall back to `_DEFAULT_PRICE`, which keeps cost accounting
# working rather than crashing when a new model id appears.
_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.6-flash": (0.30, 2.50),
    "gemini-3-flash-preview": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.10, 0.40),
}

_DEFAULT_PRICE = (0.10, 0.40)


class LLMUnavailable(RuntimeError):
    """Raised when a real model call cannot be made, rather than silently
    degrading to stub output and contaminating the corpus."""


def notional_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Cost at the model's published list price. See the module docstring."""
    key = model.split("/")[-1]
    price_in, price_out = _PRICES_PER_MTOK.get(key, _DEFAULT_PRICE)
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model: str

    @property
    def cost_usd(self) -> float:
        return notional_cost_usd(self.model, self.prompt_tokens, self.completion_tokens)


class _RateLimiter:
    """Spaces calls to stay inside a requests-per-minute quota.

    The free tier's limit is low enough that corpus generation would otherwise
    spend most of its time being rejected and retried.
    """

    def __init__(self, rpm: int) -> None:
        self._min_interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            sleep_for = self._next_allowed - now
            if sleep_for > 0:
                time.sleep(sleep_for)
                now = time.monotonic()
            self._next_allowed = now + self._min_interval


class LLMClient:
    """Interface the workflows and the judge depend on.

    ``hint`` is an out-of-band channel for the stub client only. It carries the
    expected answer and the calling role so stub output can imitate correct and
    incorrect runs. It is deliberately *not* part of the prompt: putting it
    there would leak ground truth into real model calls and invalidate every
    run in the corpus. `GeminiClient` ignores it entirely.
    """

    model: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        hint: dict[str, str] | None = None,
    ) -> LLMResponse:
        raise NotImplementedError


class GeminiClient(LLMClient):
    """Gemini via the `google-genai` SDK.

    Temperature is fixed at 0. Generation is still not bit-reproducible, so the
    corpus is committed rather than regenerated; `make reproduce` recomputes
    every published number from the committed corpus.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        *,
        rpm_limit: int = 10,
        max_retries: int = 5,
        temperature: float = 0.0,
    ) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise LLMUnavailable(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and fill "
                "it in, or pass --stub-llm to smoke-test without a key (stub "
                "output is never committed as corpus)."
            )
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - install-time failure
            raise LLMUnavailable(
                "google-genai is not installed. Run: "
                'uv pip install -e "backend[harness]"'
            ) from exc

        self._genai = genai
        self._client = genai.Client(api_key=key)
        self.model = model
        self._limiter = _RateLimiter(rpm_limit)
        self._max_retries = max_retries
        self._temperature = temperature

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        hint: dict[str, str] | None = None,
    ) -> LLMResponse:
        del hint  # stub-only channel; never reaches a real model call
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self._temperature,
            system_instruction=system,
        )

        last_error: Exception | None = None
        for attempt in range(self._max_retries):
            self._limiter.wait()
            started = time.monotonic()
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=config
                )
            except Exception as exc:  # SDK raises varied types per failure mode
                last_error = exc
                if not _is_retryable(exc) or attempt == self._max_retries - 1:
                    raise LLMUnavailable(
                        f"Gemini call failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                # Free-tier quota rejections are the expected failure here, not
                # transport errors, and the API states how long to wait. Honour
                # that when present: a shorter exponential backoff just burns
                # retries against a per-minute window that has not reset yet.
                time.sleep(_retry_delay_seconds(exc, attempt))
                continue

            latency_ms = int((time.monotonic() - started) * 1000)
            usage = getattr(response, "usage_metadata", None)
            return LLMResponse(
                text=(response.text or "").strip(),
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                latency_ms=latency_ms,
                model=self.model,
            )

        raise LLMUnavailable(f"Gemini call failed: {last_error}")


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """How long to wait before retrying.

    Prefers the ``retryDelay`` the API reports, since a per-minute quota window
    does not reset any sooner. Falls back to exponential backoff with a floor
    high enough to outlast a one-minute window.
    """
    match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", str(exc))
    if match:
        # A small margin: retrying exactly on the boundary tends to fail again.
        return float(match.group(1)) + 2.0
    return float(min(15 * (attempt + 1), 70))


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "500",
            "503",
            "unavailable",
            "deadline",
            "timeout",
        )
    )


class StubClient(LLMClient):
    """Deterministic canned responses for smoke-testing without a key.

    Answers are derived from the prompt, so the state machines, fault injection
    and success grading all exercise real code paths. The text is not model
    output and traces built from it are never committed as corpus.
    """

    def __init__(self, model: str = "stub", *, correct_rate: float = 0.7) -> None:
        self.model = model
        self._correct_rate = correct_rate

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        hint: dict[str, str] | None = None,
    ) -> LLMResponse:
        seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        hint = hint or {}
        expected = hint.get("expected")
        role = hint.get("role", "executor")

        if role == "reviewer":
            text = (
                "The reasoning is sound and the arithmetic checks out. ACCEPT"
                if rng.random() < 0.6
                else "Step two divides when it should multiply. REJECT"
            )
        elif expected is not None:
            if rng.random() < self._correct_rate:
                text = f"Working through it step by step, the answer is {expected}."
            else:
                wrong = _perturb(expected, rng)
                text = f"Working through it step by step, the answer is {wrong}."
        else:
            text = f"Stub response for a {role} step."

        return LLMResponse(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            latency_ms=rng.randint(200, 900),
            model=self.model,
        )


def _perturb(answer: str, rng: random.Random) -> str:
    """Produce a plausibly wrong answer, so graded failures look like real ones."""
    try:
        value = float(answer)
    except ValueError:
        return f"not {answer}"
    delta = rng.choice([-2, -1, 1, 2, 10])
    result = value + delta
    return str(int(result)) if float(result).is_integer() else f"{result:g}"


def build_client(
    model: str,
    *,
    stub: bool = False,
    rpm_limit: int | None = None,
) -> LLMClient:
    """Construct the client the harness and judge should use."""
    if stub:
        return StubClient()
    if rpm_limit is None:
        rpm_limit = int(os.environ.get("LLM_RPM_LIMIT", "10"))
    return GeminiClient(model, rpm_limit=rpm_limit)
