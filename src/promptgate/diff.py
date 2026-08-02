"""Regression diff.

Pass/fail on its own is not actionable. The question a reviewer has is "what
did my edit break", and the answer is a list of case ids with the check that
started failing.

One rule does most of the work: **a case whose content hash changed is not
compared.** If the case itself was edited, "it used to pass and now fails" is
meaningless - and worse, editing a case is the easiest way to make a genuine
regression disappear. Those cases are reported separately as `changed`, so the
reviewer sees that the test moved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import CaseResult, RunReport, Status


@dataclass
class CaseDelta:
    case_id: str
    before: str
    after: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "before": self.before,
            "after": self.after,
            "detail": self.detail,
        }


@dataclass
class Diff:
    baseline_run: str
    candidate_run: str
    regressed: list[CaseDelta] = field(default_factory=list)
    fixed: list[CaseDelta] = field(default_factory=list)
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unexpected_passes: list[str] = field(default_factory=list)
    pass_rate_before: float = 0.0
    pass_rate_after: float = 0.0

    @property
    def has_regression(self) -> bool:
        return bool(self.regressed)

    @property
    def pass_rate_delta(self) -> float:
        return self.pass_rate_after - self.pass_rate_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run": self.baseline_run,
            "candidate_run": self.candidate_run,
            "has_regression": self.has_regression,
            "pass_rate_before": round(self.pass_rate_before, 4),
            "pass_rate_after": round(self.pass_rate_after, 4),
            "pass_rate_delta": round(self.pass_rate_delta, 4),
            "regressed": [d.to_dict() for d in self.regressed],
            "fixed": [d.to_dict() for d in self.fixed],
            "added": self.added,
            "removed": self.removed,
            "changed": self.changed,
            "unexpected_passes": self.unexpected_passes,
        }


def _detail(result: CaseResult) -> str:
    if result.error:
        return f"provider error: {result.error}"
    failures = result.failures
    return "; ".join(failures[:2]) if failures else result.status.value


def diff_runs(baseline: RunReport, candidate: RunReport) -> Diff:
    """Compare two runs of the same golden set."""
    before = {r.case_id: r for r in baseline.results}
    after = {r.case_id: r for r in candidate.results}

    out = Diff(
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        pass_rate_before=baseline.pass_rate,
        pass_rate_after=candidate.pass_rate,
        added=sorted(set(after) - set(before)),
        removed=sorted(set(before) - set(after)),
        unexpected_passes=sorted(r.case_id for r in candidate.unexpected_passes),
    )

    for case_id in sorted(set(before) & set(after)):
        old, new = before[case_id], after[case_id]

        if old.case_hash != new.case_hash:
            # The case was edited. Comparing verdicts across a changed
            # definition would let a rewritten test bury a real regression.
            out.changed.append(case_id)
            continue

        was_ok = old.status is Status.PASS
        is_ok = new.status is Status.PASS
        if was_ok and not is_ok:
            out.regressed.append(CaseDelta(case_id, "pass", new.status.value, _detail(new)))
        elif not was_ok and is_ok:
            out.fixed.append(CaseDelta(case_id, old.status.value, "pass", _detail(old)))

    return out


def render_diff(diff: Diff, *, use_colour: bool = True) -> str:
    """Human-readable diff for a CI log or a PR comment."""

    def paint(text: str, colour: str) -> str:
        if not use_colour:
            return text
        codes = {"red": "31", "green": "32", "yellow": "33", "dim": "2"}
        return f"\033[{codes[colour]}m{text}\033[0m"

    lines: list[str] = []
    delta = diff.pass_rate_delta
    arrow = "→"
    lines.append(
        f"pass rate {diff.pass_rate_before:.1%} {arrow} {diff.pass_rate_after:.1%} ({delta:+.1%})"
    )
    lines.append("")

    if diff.regressed:
        lines.append(paint(f"REGRESSED ({len(diff.regressed)})", "red"))
        for d in diff.regressed:
            lines.append(f"  {d.case_id}  {d.detail}")
        lines.append("")

    if diff.fixed:
        lines.append(paint(f"fixed ({len(diff.fixed)})", "green"))
        for d in diff.fixed:
            lines.append(f"  {d.case_id}")
        lines.append("")

    if diff.unexpected_passes:
        lines.append(paint(f"xfail cases now passing ({len(diff.unexpected_passes)})", "yellow"))
        lines.append("  remove the xfail marker so these are gated again:")
        for case_id in diff.unexpected_passes:
            lines.append(f"  {case_id}")
        lines.append("")

    if diff.changed:
        lines.append(paint(f"cases edited, not compared ({len(diff.changed)})", "yellow"))
        for case_id in diff.changed:
            lines.append(f"  {case_id}")
        lines.append("")

    if diff.added:
        lines.append(paint(f"new cases ({len(diff.added)}): ", "dim") + ", ".join(diff.added))
    if diff.removed:
        lines.append(
            paint(f"removed cases ({len(diff.removed)}): ", "dim") + ", ".join(diff.removed)
        )

    if not diff.regressed:
        lines.append(paint("no regressions", "green"))

    return "\n".join(lines)


def render_markdown(diff: Diff, report: RunReport) -> str:
    """PR-comment format."""
    verdict = "🔴 **Regression detected**" if diff.has_regression else "🟢 **No regressions**"
    lines = [
        f"### Prompt regression gate — {verdict}",
        "",
        f"`{report.prompt_id}` version `{report.prompt_version}` "
        f"(`{report.prompt_hash}`) against golden set `{report.goldenset_version}`",
        "",
        "| | before | after |",
        "|---|---:|---:|",
        f"| pass rate | {diff.pass_rate_before:.1%} | {diff.pass_rate_after:.1%} |",
        f"| cases | {len(diff.regressed) + len(diff.fixed)} changed | "
        f"{len(report.results)} total |",
        "",
    ]
    if diff.regressed:
        lines += ["**Regressed**", "", "| case | why |", "|---|---|"]
        lines += [f"| `{d.case_id}` | {d.detail} |" for d in diff.regressed]
        lines.append("")
    if diff.fixed:
        lines.append(f"**Fixed:** {', '.join(f'`{d.case_id}`' for d in diff.fixed)}")
        lines.append("")
    if diff.changed:
        lines.append(
            f"⚠️ **Edited, so not compared:** {', '.join(f'`{c}`' for c in diff.changed)} — "
            "a changed case cannot be judged against its old result."
        )
        lines.append("")
    lines.append(
        f"<sub>{len(report.results)} cases · {report.duration_s:.1f}s · "
        f"${report.cost_usd:.4f}</sub>"
    )
    return "\n".join(lines)
