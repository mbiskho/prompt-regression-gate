"""promptgate - fails the build when a prompt edit makes the output worse.

from promptgate import load_goldenset, Runner, diff_runs

golden = load_goldenset("examples/golden/triage.yaml")
report = Runner().run(golden)
print(report.pass_rate, report.ok)
"""

from __future__ import annotations

from .assertions import CHECKS, register, run_checks
from .calibration import CalibrationReport, LabelledExample, calibrate, cohens_kappa, interpret
from .diff import Diff, diff_runs, render_diff, render_markdown
from .goldenset import GoldenSet, GoldenSetError, load_goldenset, save_goldenset
from .judge import Judge
from .models import (
    BudgetExceeded,
    Case,
    CaseResult,
    CheckResult,
    PromptSpec,
    RunReport,
    Status,
)
from .providers import MockProvider, OpenAIProvider, get_provider
from .runner import Runner
from .store import ResultStore

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Case",
    "CaseResult",
    "CheckResult",
    "PromptSpec",
    "RunReport",
    "Status",
    "BudgetExceeded",
    "GoldenSet",
    "GoldenSetError",
    "load_goldenset",
    "save_goldenset",
    "Runner",
    "Judge",
    "ResultStore",
    "Diff",
    "diff_runs",
    "render_diff",
    "render_markdown",
    "CHECKS",
    "run_checks",
    "register",
    "calibrate",
    "cohens_kappa",
    "interpret",
    "CalibrationReport",
    "LabelledExample",
    "MockProvider",
    "OpenAIProvider",
    "get_provider",
]
