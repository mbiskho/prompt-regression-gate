"""Command line interface.

promptgate run     golden.yaml              # run the suite
promptgate gate    golden.yaml              # run, diff vs baseline, exit 1 on regression
promptgate baseline golden.yaml             # record the current run as the baseline
promptgate calibrate labels.jsonl           # judge agreement, with kappa
promptgate history <prompt-id>              # recent runs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .calibration import calibrate, load_labels
from .diff import diff_runs, render_diff, render_markdown
from .goldenset import GoldenSetError, load_goldenset
from .judge import Judge
from .providers import get_provider
from .runner import Runner
from .store import ResultStore

DEFAULT_DB = ".promptgate/results.db"


def _build(args) -> tuple:
    golden = load_goldenset(args.goldenset, prompt_file=getattr(args, "prompt_file", None))
    provider = get_provider(args.provider)
    judge = None if args.no_judge else Judge(provider, model=args.judge_model)
    runner = Runner(provider, judge=judge, budget_usd=args.budget, fail_fast=args.fail_fast)
    return golden, runner


def cmd_run(args) -> int:
    golden, runner = _build(args)
    print(
        f"{golden.prompt.id} v{golden.prompt.version} ({golden.prompt.prompt_hash}) "
        f"· {len(golden.cases)} cases"
    )
    report = runner.run(golden, verbose=not args.quiet)

    print()
    print(
        f"{report.passed}/{len(report.results)} passed "
        f"({report.pass_rate:.1%}) · {report.duration_s:.1f}s · ${report.cost_usd:.4f}"
    )
    if report.stopped_early:
        print(f"STOPPED EARLY: {report.stopped_early}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"wrote {args.json}")

    ResultStore(args.db).save(report, label=args.label)
    return 0 if report.ok else 1


def cmd_baseline(args) -> int:
    args.label = args.label or "baseline"
    golden, runner = _build(args)
    report = runner.run(golden, verbose=not args.quiet)
    ResultStore(args.db).save(report, label=args.label)
    print(
        f"baseline recorded: {report.run_id} · {report.passed}/{len(report.results)} passed "
        f"({report.pass_rate:.1%})"
    )
    return 0


def cmd_gate(args) -> int:
    """Run, compare against the baseline, and fail the build on a regression."""
    golden, runner = _build(args)
    store = ResultStore(args.db)

    baseline = store.latest(golden.prompt.id, label=args.baseline_label)
    if baseline is None:
        report = runner.run(golden, verbose=not args.quiet)
        store.save(report, label=args.baseline_label)
        print(
            f"no baseline for {golden.prompt.id!r}; recorded this run as one "
            f"({report.passed}/{len(report.results)} passed).",
            file=sys.stderr,
        )
        # First run cannot regress against nothing, but it can still be broken
        # on its own terms.
        return 0 if report.ok else 1

    report = runner.run(golden, verbose=not args.quiet)
    store.save(report, label=args.label)
    delta = diff_runs(baseline, report)

    print()
    print(render_diff(delta, use_colour=sys.stdout.isatty()))

    if args.markdown:
        Path(args.markdown).write_text(render_markdown(delta, report))
        print(f"\nwrote {args.markdown}")
    if args.json:
        Path(args.json).write_text(json.dumps(delta.to_dict(), indent=2))

    if delta.has_regression:
        print(f"\n{len(delta.regressed)} case(s) regressed — failing the build", file=sys.stderr)
        return 1
    if report.stopped_early:
        print(f"\nrun incomplete: {report.stopped_early}", file=sys.stderr)
        return 1
    if golden.min_pass_rate is not None and report.pass_rate < golden.min_pass_rate:
        print(
            f"\npass rate {report.pass_rate:.1%} below the floor "
            f"{golden.min_pass_rate:.1%} — failing the build",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_calibrate(args) -> int:
    examples = load_labels(args.labels)
    judge = Judge(get_provider(args.provider), model=args.judge_model, threshold=args.threshold)
    report = calibrate(judge, examples)

    print(report.summary())
    if args.json:
        Path(args.json).write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\nwrote {args.json}")

    if report.label_source != "human":
        print(
            "\nNote: these labels are not human-generated, so this kappa measures the "
            "calibration machinery rather than agreement with a reviewer.",
            file=sys.stderr,
        )
    if not report.meets_bar:
        print(
            f"\nkappa {report.kappa:.3f} is below 0.60 — do not gate on this judge yet",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_history(args) -> int:
    rows = ResultStore(args.db).history(args.prompt_id, limit=args.limit)
    if not rows:
        print(f"no runs recorded for {args.prompt_id!r}")
        return 0
    print(f"{'run':<14}{'version':<12}{'passed':>8}{'failed':>8}{'cost':>10}  label")
    for row in rows:
        print(
            f"{row['run_id']:<14}{row['prompt_version']:<12}"
            f"{row['passed']:>8}{row['failed']:>8}{row['cost_usd']:>10.4f}  {row['label']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptgate",
        description="Fails the build when a prompt edit makes the output worse.",
    )
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--provider", default="mock", help="mock | openai")
    parser.add_argument("--judge-model", default="mock/small")
    parser.add_argument("--no-judge", action="store_true", help="deterministic checks only")
    parser.add_argument("--budget", type=float, default=1.0, help="spend ceiling per run, USD")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--json", help="write machine-readable output here")

    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, helptext in (
        ("run", cmd_run, "run the suite"),
        ("baseline", cmd_baseline, "record the current run as the baseline"),
        ("gate", cmd_gate, "run, diff against the baseline, fail on regression"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("goldenset")
        p.add_argument("--prompt-file", help="override the prompt template file")
        p.add_argument("--label", default="", help="label this run, e.g. a branch name")
        if name == "gate":
            p.add_argument("--baseline-label", default="baseline")
            p.add_argument("--markdown", help="write a PR comment here")
        p.set_defaults(func=fn)

    p = sub.add_parser("calibrate", help="measure judge agreement with reference labels")
    p.add_argument("labels")
    p.add_argument("--threshold", type=float, default=0.6)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("history", help="recent runs for a prompt")
    p.add_argument("prompt_id")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_history)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except GoldenSetError as exc:
        print(f"golden set is invalid: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
