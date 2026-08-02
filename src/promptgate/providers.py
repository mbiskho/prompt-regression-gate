"""Providers.

`MockProvider` is the interesting one, and the design constraint is unusual:
**it has to react to the prompt.** A regression-testing tool whose fake model
ignores the prompt cannot demonstrate that it catches prompt regressions - the
demo would be theatre.

So the mock reads the prompt for instruction markers and only honours the ones
it finds. Ask for JSON and it returns JSON; delete that line and it returns
prose with a preamble. Ask it to stay under a word count and it does; delete
that and it rambles. That is a caricature of how models behave, but it is a
faithful caricature of the *failure* being tested: an instruction was dropped
and the output stopped conforming.

Per-case variation is a deterministic hash, so a run is reproducible and the
same case fails the same way every time. Without that, a regression diff is
noise.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class Completion:
    text: str
    latency_ms: float
    cost_usd: float = 0.0
    model: str = ""


class Provider(Protocol):
    name: str

    def complete(
        self, prompt: str, *, model: str, max_tokens: int, temperature: float
    ) -> Completion: ...


def _unit(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


# --------------------------------------------------------------------------
# Instruction detection
# --------------------------------------------------------------------------

#: Marker -> the regex that means "the prompt asked for this".
#: Deliberately loose. The point is to detect intent, not to grade phrasing.
INSTRUCTIONS = {
    "json": r"\b(?:json only|return json|as json|valid json|respond with json)\b",
    "schema": r"\bschema\b|\brequired fields?\b|\bmust include\b",
    "no_pii": r"\b(?:redact|do not include|never include|omit)\b.{0,40}\b(?:pii|personal|email|phone|ssn)\b",
    "brevity": r"\b(?:at most|no more than|under|maximum of)\s+\d+\s+(?:words|sentences)\b",
    "label_only": r"\b(?:reply with (?:the )?label only|one of the following|respond with only)\b",
    "no_preamble": r"\b(?:no preamble|do not explain|without commentary|omit any preamble)\b",
}
_COMPILED = {k: re.compile(v, re.IGNORECASE) for k, v in INSTRUCTIONS.items()}


def detect_instructions(prompt: str) -> set[str]:
    """Which behaviours the prompt actually asks for."""
    return {name for name, pattern in _COMPILED.items() if pattern.search(prompt)}


class MockProvider:
    """Deterministic, prompt-sensitive fake model. The default.

    Exists so the suite, the CI gate and the seeded-regression demo all run
    with no API key and no spend, and produce identical results on every
    machine.

    Parameters
    ----------
    competence:
        Probability a case is handled correctly *given* the prompt asked for
        the right thing. Below 1.0, some cases fail even with a good prompt -
        which is realistic, and is what makes a baseline necessary rather than
        just asserting "everything passes".
    """

    name = "mock"

    def __init__(self, *, competence: float = 0.92, seed: str = "v1", price_per_mtok: float = 0.0):
        self.competence = competence
        self.seed = seed
        self.price_per_mtok = price_per_mtok

    def complete(
        self, prompt: str, *, model: str, max_tokens: int = 512, temperature: float = 0.0
    ) -> Completion:
        started = time.perf_counter()
        asked = detect_instructions(prompt)
        payload = _extract_payload(prompt)
        case_key = payload.get("__case_id__", prompt[-120:])

        competent = _unit(self.seed, "skill", str(case_key), model) < self.competence
        text = self._render(asked, payload, competent)

        tokens = max(1, len(prompt) // 4 + len(text) // 4)
        return Completion(
            text=text,
            latency_ms=(time.perf_counter() - started) * 1000
            + 40
            + 60 * _unit(self.seed, str(case_key)),
            cost_usd=tokens * self.price_per_mtok / 1_000_000,
            model=model,
        )

    def _render(self, asked: set[str], payload: dict[str, Any], competent: bool) -> str:
        answer = payload.get("__answer__")
        labels = payload.get("__labels__")

        # Classification-shaped case.
        if labels:
            chosen = answer if competent else _wrong_label(labels, answer)
            if "label_only" in asked:
                return str(chosen)
            return f"Based on the ticket, I would classify this as: {chosen}. Let me know if you need more detail."

        # Structured-extraction case.
        if isinstance(answer, dict):
            if "json" not in asked:
                # The instruction was dropped, so the model explains itself in
                # prose - the classic silent regression.
                fields = ", ".join(f"{k} is {v}" for k, v in list(answer.items())[:3])
                return f"Sure! Here is what I found: {fields}."
            if not competent:
                partial = dict(answer)
                partial.pop(next(iter(partial), ""), None)
                return json.dumps(partial)
            body = json.dumps(answer)
            return body if "no_preamble" in asked else f"Here is the JSON:\n```json\n{body}\n```"

        # Free-text case.
        text = str(answer or "The request was processed.")
        if "brevity" not in asked:
            text += (
                " In addition, it is worth noting that this outcome depends on a number of "
                "contextual factors which may vary between reporting periods, and readers "
                "should interpret the figures accordingly rather than in isolation."
            )
        if "no_pii" not in asked and payload.get("__leak__"):
            text += f" You can reach the account owner at {payload['__leak__']}."
        return text


def _wrong_label(labels: list[str], correct: Any) -> str:
    options = [str(x) for x in labels if str(x) != str(correct)]
    return options[0] if options else str(correct)


_PAYLOAD = re.compile(r"<!--promptgate:(.*?)-->", re.DOTALL)


def _extract_payload(prompt: str) -> dict[str, Any]:
    """Read the case fixture the runner embeds in the rendered prompt.

    The mock needs to know what the right answer *is* in order to return it.
    Real providers ignore the comment; it is stripped before any real call.
    """
    match = _PAYLOAD.search(prompt)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def strip_payload(prompt: str) -> str:
    return _PAYLOAD.sub("", prompt).strip()


# --------------------------------------------------------------------------
# Real providers
# --------------------------------------------------------------------------

PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5": (3.00, 15.00),
}


class OpenAIProvider:
    """Real calls, priced so the cost guard means something."""

    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url

    def complete(
        self, prompt: str, *, model: str, max_tokens: int = 512, temperature: float = 0.0
    ) -> Completion:
        import urllib.request

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        body = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": strip_payload(prompt)}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode()
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read())
        latency = (time.perf_counter() - started) * 1000

        usage = data.get("usage", {})
        rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
        cost = (
            usage.get("prompt_tokens", 0) * rate_in + usage.get("completion_tokens", 0) * rate_out
        ) / 1_000_000
        return Completion(
            text=data["choices"][0]["message"]["content"] or "",
            latency_ms=latency,
            cost_usd=cost,
            model=model,
        )


def get_provider(name: str = "mock", **kwargs: Any) -> Provider:
    if name in ("mock", "offline"):
        return MockProvider(**kwargs)
    if name in ("openai", "gpt"):
        return OpenAIProvider(**kwargs)
    raise KeyError(f"unknown provider {name!r}")
