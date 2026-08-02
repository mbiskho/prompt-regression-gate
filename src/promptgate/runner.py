"""Suite execution.

Order of operations per case, and the order is the whole cost argument:

    render prompt -> call model -> deterministic checks -> judge (only if needed)

The judge runs only when the case has a rubric *and* every deterministic check
already passed. A response that failed schema validation is already a failure;
paying a model to confirm it is money spent to learn nothing.

The cost guard is checked **before** each call, not after. Checking afterwards
means the ceiling is discovered by exceeding it, which is not a ceiling.
"""

from __future__ import annotations

import json
import time
import uuid

from .assertions import run_checks
from .goldenset import GoldenSet
from .judge import Judge
from .models import BudgetExceeded, Case, CaseResult, RunReport, Status
from .providers import MockProvider, Provider, get_provider


def _fixture(case: Case) -> str:
    """Fixture the mock provider reads to know what a right answer looks like.

    Embedded as an HTML comment and stripped before any real provider call.
    Real models never see it.
    """
    payload: dict = {"__case_id__": case.id}
    if case.checks.get("one_of"):
        payload["__labels__"] = case.checks["one_of"]
    if "expected_label" in case.checks:
        payload["__answer__"] = case.checks["expected_label"]
    elif case.inputs.get("expected_json"):
        payload["__answer__"] = case.inputs["expected_json"]
    elif case.expected:
        payload["__answer__"] = case.expected
    if case.inputs.get("leak"):
        payload["__leak__"] = case.inputs["leak"]
    return f"\n<!--promptgate:{json.dumps(payload)}-->"


class Runner:
    """Runs a golden set against a prompt version.

    Parameters
    ----------
    budget_usd:
        Spend ceiling for the whole run. When the next call would breach it,
        the run stops and is reported as incomplete rather than partially
        green - a suite that silently skipped half its cases and passed is
        worse than one that failed.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        judge: Judge | None = None,
        budget_usd: float | None = 1.0,
        fail_fast: bool = False,
    ) -> None:
        self.provider = provider or get_provider("mock")
        self.judge = judge
        self.budget_usd = budget_usd
        self.fail_fast = fail_fast

    def run(self, golden: GoldenSet, *, verbose: bool = False) -> RunReport:
        golden.validate()
        started = time.time()
        clock = time.perf_counter()

        report = RunReport(
            run_id=uuid.uuid4().hex[:12],
            prompt_id=golden.prompt.id,
            prompt_version=golden.prompt.version,
            prompt_hash=golden.prompt.prompt_hash,
            goldenset_version=golden.version,
            started_at=started,
            budget_usd=self.budget_usd,
        )

        for case in golden.cases:
            if self.budget_usd is not None and report.cost_usd >= self.budget_usd:
                report.stopped_early = (
                    f"spend ceiling ${self.budget_usd:.4f} reached after "
                    f"{len(report.results)} of {len(golden.cases)} cases"
                )
                break

            result = self._run_case(case, golden)
            report.results.append(result)
            report.cost_usd += result.cost_usd

            if verbose:
                mark = "PASS" if result.status is Status.PASS else result.status.value.upper()
                print(
                    f"  {mark:<5} {case.id}"
                    + (f"  ({'; '.join(result.failures)[:90]})" if result.failures else "")
                )

            if self.fail_fast and result.counts_as_failure:
                report.stopped_early = f"fail-fast on {case.id}"
                break

        report.duration_s = time.perf_counter() - clock
        if self.judge is not None:
            report.cost_usd += self.judge.cost_usd
            self.judge.cost_usd = 0.0
        return report

    def _run_case(self, case: Case, golden: GoldenSet) -> CaseResult:
        prompt = golden.prompt.render(case.inputs)
        if isinstance(self.provider, MockProvider):
            prompt += _fixture(case)

        result = CaseResult(
            case_id=case.id, case_hash=case.content_hash, status=Status.PASS, xfail=case.xfail
        )

        try:
            completion = self.provider.complete(
                prompt,
                model=golden.prompt.model,
                max_tokens=golden.prompt.max_output_tokens,
                temperature=golden.prompt.temperature,
            )
        except Exception as exc:  # noqa: BLE001 - a provider failure is not a prompt failure
            result.status = Status.ERROR
            result.error = f"{type(exc).__name__}: {exc}"
            return result

        result.output = completion.text
        result.latency_ms = completion.latency_ms
        result.cost_usd = completion.cost_usd
        result.checks = run_checks(completion.text, case.checks, latency_ms=completion.latency_ms)

        deterministic_ok = all(c.passed for c in result.checks)

        # The judge only sees responses that already cleared every free check.
        if case.rubric and self.judge is not None and deterministic_ok:
            check, score = self.judge.score(completion.text, case.rubric, case_id=case.id)
            result.checks.append(check)
            result.judge_score = score

        result.status = Status.PASS if all(c.passed for c in result.checks) else Status.FAIL
        return result


def check_budget(estimated_usd: float, ceiling_usd: float | None) -> None:
    if ceiling_usd is not None and estimated_usd > ceiling_usd:
        raise BudgetExceeded(
            f"estimated ${estimated_usd:.4f} exceeds the run ceiling ${ceiling_usd:.4f}"
        )
