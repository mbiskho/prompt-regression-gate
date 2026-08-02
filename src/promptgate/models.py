"""Core types.

The one idea worth stating up front: **a result is only comparable to a
baseline if the case that produced it has not changed.** Every case carries a
content hash, and the diff refuses to compare results whose hashes differ.

Without that, editing a case's expected output silently "fixes" a regression -
the suite goes green and nobody can tell whether the prompt improved or the
test was moved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Status(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"  # the provider failed; not the prompt's fault

    @property
    def ok(self) -> bool:
        return self is Status.PASS


@dataclass
class Case:
    """One test case: inputs, and what a good answer looks like.

    `checks` are deterministic assertion specs. `rubric` is used only when a
    judge is configured - and a case with neither is a case that cannot fail,
    which `GoldenSet.validate()` rejects.
    """

    id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    #: Deterministic assertions, e.g. {"json_schema": {...}, "max_latency_ms": 2000}
    checks: dict[str, Any] = field(default_factory=dict)
    #: Free-text criteria for the judge, used only where no assertion works.
    rubric: str = ""
    #: Reference answer. Optional, and never compared verbatim - exact-match
    #: scoring on generated text is noise.
    expected: str = ""
    tags: list[str] = field(default_factory=list)
    #: Cases can be marked known-failing so the suite stays green while a fix
    #: is in flight, without deleting the evidence.
    xfail: bool = False

    @property
    def content_hash(self) -> str:
        """Fingerprint of everything that affects the verdict.

        Deliberately excludes `tags` and `xfail`: retagging a case or marking
        it known-failing does not change what it tests, so it should not
        invalidate its baseline.
        """
        payload = json.dumps(
            {
                "id": self.id,
                "inputs": self.inputs,
                "checks": self.checks,
                "rubric": self.rubric,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in ({}, [], "", False)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Case:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class PromptSpec:
    """The thing under test: a template, a model, and a version label."""

    id: str
    version: str
    template: str
    model: str = "mock/small"
    temperature: float = 0.0
    max_output_tokens: int = 512

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def render(self, inputs: dict[str, Any]) -> str:
        """Fill the template. Missing keys are an error, not a blank.

        A silently-empty substitution produces a prompt that still runs and
        quietly tests nothing.
        """
        try:
            return self.template.format(**inputs)
        except KeyError as exc:
            raise KeyError(
                f"prompt {self.id!r} needs input {exc.args[0]!r}, which the case does not provide"
            ) from exc

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "prompt_hash": self.prompt_hash}


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaseResult:
    case_id: str
    case_hash: str
    status: Status
    output: str = ""
    checks: list[CheckResult] = field(default_factory=list)
    judge_score: float | None = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    error: str = ""
    xfail: bool = False

    @property
    def failures(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.passed]

    @property
    def counts_as_failure(self) -> bool:
        """An xfail case that fails is expected; one that passes is not.

        The second half matters - a case marked known-failing that starts
        passing means the fix landed and the marker is now hiding a real
        signal, so it should be surfaced rather than silently tolerated.
        """
        if self.xfail:
            return False
        return self.status is not Status.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "case_hash": self.case_hash,
            "status": self.status.value,
            "output": self.output,
            "checks": [c.to_dict() for c in self.checks],
            "judge_score": self.judge_score,
            "latency_ms": round(self.latency_ms, 2),
            "cost_usd": round(self.cost_usd, 8),
            "error": self.error,
            "xfail": self.xfail,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseResult:
        return cls(
            case_id=data["case_id"],
            case_hash=data["case_hash"],
            status=Status(data["status"]),
            output=data.get("output", ""),
            checks=[CheckResult(**c) for c in data.get("checks", [])],
            judge_score=data.get("judge_score"),
            latency_ms=data.get("latency_ms", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            error=data.get("error", ""),
            xfail=data.get("xfail", False),
        )


@dataclass
class RunReport:
    """One execution of a golden set against one prompt version."""

    run_id: str
    prompt_id: str
    prompt_version: str
    prompt_hash: str
    goldenset_version: str
    results: list[CaseResult] = field(default_factory=list)
    started_at: float = 0.0
    duration_s: float = 0.0
    cost_usd: float = 0.0
    budget_usd: float | None = None
    stopped_early: str = ""

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status is Status.PASS)

    @property
    def failed(self) -> list[CaseResult]:
        return [r for r in self.results if r.counts_as_failure]

    @property
    def unexpected_passes(self) -> list[CaseResult]:
        return [r for r in self.results if r.xfail and r.status is Status.PASS]

    @property
    def pass_rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def ok(self) -> bool:
        return not self.failed and not self.stopped_early

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "prompt_id": self.prompt_id,
            "prompt_version": self.prompt_version,
            "prompt_hash": self.prompt_hash,
            "goldenset_version": self.goldenset_version,
            "started_at": self.started_at,
            "duration_s": round(self.duration_s, 3),
            "cost_usd": round(self.cost_usd, 8),
            "budget_usd": self.budget_usd,
            "stopped_early": self.stopped_early,
            "summary": {
                "cases": len(self.results),
                "passed": self.passed,
                "failed": len(self.failed),
                "pass_rate": round(self.pass_rate, 4),
            },
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunReport:
        return cls(
            run_id=data["run_id"],
            prompt_id=data["prompt_id"],
            prompt_version=data["prompt_version"],
            prompt_hash=data.get("prompt_hash", ""),
            goldenset_version=data.get("goldenset_version", ""),
            results=[CaseResult.from_dict(r) for r in data.get("results", [])],
            started_at=data.get("started_at", 0.0),
            duration_s=data.get("duration_s", 0.0),
            cost_usd=data.get("cost_usd", 0.0),
            budget_usd=data.get("budget_usd"),
            stopped_early=data.get("stopped_early", ""),
        )


class BudgetExceeded(RuntimeError):
    """Raised when a run would breach its spend ceiling."""
