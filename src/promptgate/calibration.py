"""Judge calibration.

**An uncalibrated judge is decoration.** Until you know how often it agrees
with a reference label, a red build is an opinion and nobody on the team will
believe it twice.

Cohen's kappa rather than raw agreement, because raw agreement is inflated by
class imbalance: if 90% of your cases are good, a judge that says "good" every
time scores 90% agreement and is useless. Kappa corrects for the agreement you
would get by chance.

Rough reading, following Landis & Koch: below 0.4 is poor, 0.4-0.6 moderate,
0.6-0.8 substantial, above 0.8 almost perfect. A gate should want at least 0.6
before anyone is asked to act on a red build.

**On the labels shipped in this repo.** They are generated, not human. I am
not going to hand-label two hundred examples and call it a human baseline when
it is not one - the whole point of this tool is to stop people asserting
quality they have not measured. `examples/labels.jsonl` is a *synthetic*
reference produced by a known rule, so the kappa it yields measures whether
the calibration machinery works, not whether the judge agrees with a domain
expert. Point `promptgate calibrate` at your own labelled file to get a number
that means something.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LabelledExample:
    """One human (or reference) verdict on one response."""

    case_id: str
    response: str
    criteria: str
    #: True if a reviewer considered this acceptable.
    label: bool
    source: str = "synthetic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "response": self.response,
            "criteria": self.criteria,
            "label": self.label,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelledExample:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two binary raters, corrected for chance."""
    if len(a) != len(b):
        raise ValueError("rater lists must be the same length")
    n = len(a)
    if n == 0:
        return 0.0

    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    a_true, b_true = sum(a) / n, sum(b) / n
    expected = a_true * b_true + (1 - a_true) * (1 - b_true)

    if expected >= 1.0:
        # Both raters were unanimous and identical. Kappa is undefined here;
        # reporting 1.0 would overstate a dataset that cannot discriminate.
        return 0.0 if observed < 1.0 else 1.0
    return (observed - expected) / (1 - expected)


def interpret(kappa: float) -> str:
    if kappa < 0.0:
        return "worse than chance"
    if kappa < 0.20:
        return "slight"
    if kappa < 0.40:
        return "fair"
    if kappa < 0.60:
        return "moderate"
    if kappa < 0.80:
        return "substantial"
    return "almost perfect"


@dataclass
class CalibrationReport:
    n: int
    kappa: float
    agreement: float
    judge_positive_rate: float
    reference_positive_rate: float
    false_positives: int = 0
    false_negatives: int = 0
    threshold: float = 0.6
    label_source: str = "synthetic"
    disagreements: list[dict[str, Any]] = field(default_factory=list)

    @property
    def meets_bar(self) -> bool:
        return self.kappa >= 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "cohens_kappa": round(self.kappa, 4),
            "interpretation": interpret(self.kappa),
            "raw_agreement": round(self.agreement, 4),
            "judge_positive_rate": round(self.judge_positive_rate, 4),
            "reference_positive_rate": round(self.reference_positive_rate, 4),
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "judge_threshold": self.threshold,
            "label_source": self.label_source,
            "meets_bar": self.meets_bar,
            "bar": "kappa >= 0.60 (substantial)",
            "disagreements": self.disagreements[:20],
        }

    def summary(self) -> str:
        verdict = "OK" if self.meets_bar else "BELOW BAR"
        return (
            f"kappa {self.kappa:.3f} ({interpret(self.kappa)}) over {self.n} examples "
            f"[{verdict}]\n"
            f"  raw agreement      {self.agreement:.1%}\n"
            f"  judge says good    {self.judge_positive_rate:.1%}\n"
            f"  reference says good {self.reference_positive_rate:.1%}\n"
            f"  false positives    {self.false_positives}  "
            f"(judge accepted what the reference rejected)\n"
            f"  false negatives    {self.false_negatives}\n"
            f"  label source       {self.label_source}"
        )


def calibrate(judge, examples: list[LabelledExample]) -> CalibrationReport:
    """Score the judge against reference labels.

    False positives are called out separately because they are the expensive
    direction: a judge that accepts bad output makes the gate green while the
    product degrades, which is precisely the failure the gate exists to stop.
    """
    if not examples:
        raise ValueError("no labelled examples supplied")

    judge_verdicts: list[bool] = []
    reference: list[bool] = []
    disagreements: list[dict[str, Any]] = []

    for example in examples:
        check, score = judge.score(example.response, example.criteria, case_id=example.case_id)
        judge_verdicts.append(check.passed)
        reference.append(example.label)
        if check.passed != example.label:
            disagreements.append(
                {
                    "case_id": example.case_id,
                    "judge": "pass" if check.passed else "fail",
                    "reference": "pass" if example.label else "fail",
                    "judge_score": score,
                    "response": example.response[:120],
                }
            )

    n = len(examples)
    agreement = sum(1 for x, y in zip(judge_verdicts, reference, strict=True) if x == y) / n
    return CalibrationReport(
        n=n,
        kappa=cohens_kappa(judge_verdicts, reference),
        agreement=agreement,
        judge_positive_rate=sum(judge_verdicts) / n,
        reference_positive_rate=sum(reference) / n,
        false_positives=sum(
            1 for j, r in zip(judge_verdicts, reference, strict=True) if j and not r
        ),
        false_negatives=sum(
            1 for j, r in zip(judge_verdicts, reference, strict=True) if not j and r
        ),
        threshold=judge.threshold,
        label_source=examples[0].source,
        disagreements=disagreements,
    )


def load_labels(path: Path | str) -> list[LabelledExample]:
    return [
        LabelledExample.from_dict(json.loads(line))
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def save_labels(examples: list[LabelledExample], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(e.to_dict()) for e in examples) + "\n")
    return path
