"""Deterministic assertions.

**Use these wherever they can be written.** They are free, exact, instant and
reproducible, and every case they cover is a case the judge does not have to -
which matters, because the judge is the expensive, noisy, slow part.

The rule of thumb that survived contact with the golden sets here: if you can
describe what "wrong" looks like without using the word "good", it is an
assertion, not a rubric.

Each check takes the model output plus its spec and returns a `CheckResult`.
Checks never raise - a broken check is reported as a failed check, because a
test framework that crashes on malformed model output is a test framework that
gets switched off.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from .models import CheckResult

# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Tolerates the three things models do instead of returning bare JSON: wrap
    it in a code fence, prefix it with "Sure, here you go", or append a
    trailing explanation.

    Being tolerant here is a deliberate choice. Strictness would make the
    assertion fire on formatting noise rather than on content, and a gate that
    goes red for cosmetic reasons is a gate people learn to ignore.
    """
    if not text or not text.strip():
        raise ValueError("empty output")
    candidate = _FENCE.sub("", text.strip())
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = candidate.find(opener), candidate.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("no parseable JSON found")


def check_json_valid(output: str, spec: Any) -> CheckResult:
    try:
        extract_json(output)
    except ValueError as exc:
        return CheckResult("json_valid", False, str(exc))
    return CheckResult("json_valid", True)


def check_required_fields(output: str, spec: Any) -> CheckResult:
    """Named fields must be present and non-null.

    Scores partial credit, because "five of six fields" and "returned an
    apology" are different failures and the diff should be able to tell them
    apart.
    """
    fields = list(spec)
    try:
        data = extract_json(output)
    except ValueError as exc:
        return CheckResult("required_fields", False, f"not JSON: {exc}", 0.0)
    if not isinstance(data, dict):
        return CheckResult(
            "required_fields", False, f"expected object, got {type(data).__name__}", 0.0
        )

    missing = [f for f in fields if data.get(f) is None]
    score = (len(fields) - len(missing)) / len(fields) if fields else 1.0
    if missing:
        return CheckResult("required_fields", False, f"missing {missing}", round(score, 4))
    return CheckResult("required_fields", True, score=1.0)


def check_json_schema(output: str, spec: Any) -> CheckResult:
    """Validate against a JSON Schema subset.

    Uses `jsonschema` when installed, otherwise a built-in checker covering
    type, required, properties, enum, and numeric bounds - which is most of
    what LLM output schemas use. Keeps the core install dependency-light.
    """
    try:
        data = extract_json(output)
    except ValueError as exc:
        return CheckResult("json_schema", False, f"not JSON: {exc}", 0.0)

    try:
        import jsonschema  # type: ignore

        try:
            jsonschema.validate(data, spec)
            return CheckResult("json_schema", True, score=1.0)
        except jsonschema.ValidationError as exc:
            return CheckResult("json_schema", False, exc.message[:160], 0.0)
    except ImportError:
        errors = _validate(data, spec, "$")
        if errors:
            return CheckResult("json_schema", False, "; ".join(errors[:3]), 0.0)
        return CheckResult("json_schema", True, score=1.0)


_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "null": type(None),
}


def _validate(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    expected = schema.get("type")
    if expected:
        types = expected if isinstance(expected, list) else [expected]
        if not any(_is_type(value, t) for t in types):
            return [f"{path}: expected {'/'.join(types)}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in {schema['enum']}")

    if isinstance(value, dict):
        for name in schema.get("required", []):
            if name not in value:
                errors.append(f"{path}: missing {name!r}")
        for name, sub in (schema.get("properties") or {}).items():
            if name in value and isinstance(sub, dict):
                errors.extend(_validate(value[name], sub, f"{path}.{name}"))

    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(value):
            errors.extend(_validate(item, schema["items"], f"{path}[{i}]"))

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")

    return errors


def _is_type(value: Any, name: str) -> bool:
    expected = _TYPES.get(name)
    if expected is None:
        return True
    # JSON booleans are not numbers, whatever Python thinks.
    if name in ("number", "integer") and isinstance(value, bool):
        return False
    return isinstance(value, expected)


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------


def check_contains(output: str, spec: Any) -> CheckResult:
    missing = [s for s in spec if s.lower() not in output.lower()]
    if missing:
        return CheckResult("contains", False, f"missing {missing[:3]}")
    return CheckResult("contains", True)


def check_not_contains(output: str, spec: Any) -> CheckResult:
    """Banned strings. The workhorse for refusal leakage and stock phrases."""
    found = [s for s in spec if s.lower() in output.lower()]
    if found:
        return CheckResult("not_contains", False, f"contains banned {found[:3]}")
    return CheckResult("not_contains", True)


def check_regex(output: str, spec: Any) -> CheckResult:
    if not re.search(spec, output, re.IGNORECASE | re.MULTILINE):
        return CheckResult("regex", False, f"no match for {spec!r}")
    return CheckResult("regex", True)


def check_max_words(output: str, spec: Any) -> CheckResult:
    words = len(output.split())
    if words > int(spec):
        return CheckResult("max_words", False, f"{words} words > {spec}")
    return CheckResult("max_words", True)


def check_min_words(output: str, spec: Any) -> CheckResult:
    words = len(output.split())
    if words < int(spec):
        return CheckResult("min_words", False, f"{words} words < {spec}")
    return CheckResult("min_words", True)


def check_one_of(output: str, spec: Any) -> CheckResult:
    """Output must be one of a fixed set. Classification's whole quality bar.

    Worth being explicit about the limit: this checks the answer is a *valid*
    label, not the *right* one. A model that confidently returns the wrong
    label from the list passes. Pair it with `expected_label` when the case
    knows the answer.
    """
    cleaned = output.strip().strip(".\"'`").lower()
    options = [str(s).lower() for s in spec]
    if cleaned in options:
        return CheckResult("one_of", True)
    hits = [o for o in options if o and o in cleaned]
    if len(hits) == 1:
        return CheckResult("one_of", True, "label embedded in prose", 0.8)
    return CheckResult("one_of", False, f"{output.strip()[:40]!r} not in {spec}")


def check_expected_label(output: str, spec: Any) -> CheckResult:
    """Exact-match against the correct label. Only valid for closed sets."""
    cleaned = output.strip().strip(".\"'`").lower()
    if cleaned == str(spec).lower():
        return CheckResult("expected_label", True)
    return CheckResult("expected_label", False, f"got {output.strip()[:40]!r}, want {spec!r}")


# --------------------------------------------------------------------------
# Safety and budget
# --------------------------------------------------------------------------

#: Patterns that should never appear in generated output. Narrow on purpose:
#: a broad PII regex produces false positives on ordinary numbers, and a gate
#: that cries wolf gets disabled.
PII_PATTERNS = {
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "phone_us": r"\b(?:\+1[ -]?)?\(?\d{3}\)?[ -]\d{3}[ -]\d{4}\b",
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


def check_no_pii(output: str, spec: Any) -> CheckResult:
    """Scan for PII shapes. `spec` is True, or a list of pattern names."""
    names = list(spec) if isinstance(spec, list) else list(PII_PATTERNS)
    hits = []
    for name in names:
        pattern = PII_PATTERNS.get(name)
        if pattern and re.search(pattern, output):
            hits.append(name)
    if hits:
        # The matched value is deliberately not echoed - a CI log is a bad
        # place to reproduce the PII you just detected.
        return CheckResult("no_pii", False, f"matched {hits}")
    return CheckResult("no_pii", True)


def check_max_latency_ms(output: str, spec: Any) -> CheckResult:  # pragma: no cover - see runner
    """Placeholder: latency is known by the runner, not by the output."""
    return CheckResult("max_latency_ms", True)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CHECKS: dict[str, Callable[[str, Any], CheckResult]] = {
    "json_valid": check_json_valid,
    "json_schema": check_json_schema,
    "required_fields": check_required_fields,
    "contains": check_contains,
    "not_contains": check_not_contains,
    "regex": check_regex,
    "max_words": check_max_words,
    "min_words": check_min_words,
    "one_of": check_one_of,
    "expected_label": check_expected_label,
    "no_pii": check_no_pii,
    "max_latency_ms": check_max_latency_ms,
}


def run_checks(
    output: str, checks: dict[str, Any], *, latency_ms: float = 0.0
) -> list[CheckResult]:
    """Run every configured check. Never raises."""
    results: list[CheckResult] = []
    for name, spec in checks.items():
        if name == "max_latency_ms":
            budget = float(spec)
            ok = latency_ms <= budget
            results.append(
                CheckResult(
                    "max_latency_ms", ok, "" if ok else f"{latency_ms:.0f}ms > {budget:.0f}ms"
                )
            )
            continue

        fn = CHECKS.get(name)
        if fn is None:
            results.append(CheckResult(name, False, f"unknown check {name!r}"))
            continue
        try:
            results.append(fn(output, spec))
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
            results.append(CheckResult(name, False, f"check raised {type(exc).__name__}: {exc}"))
    return results


def register(name: str, fn: Callable[[str, Any], CheckResult]) -> None:
    """Add a domain-specific check. These are the highest-value ones."""
    CHECKS[name] = fn
