"""LLM-as-judge scorer.

Only for cases where no assertion can be written. Three constraints keep it
from becoming the problem it is meant to solve:

1. **It runs last.** Deterministic checks go first and short-circuit, so a
   response that is not even valid JSON never reaches a paid grader.
2. **Its output is constrained.** A single digit 1-5 against a written rubric,
   capped at a handful of tokens. Free-form commentary is unscoreable.
3. **Its cost is charged to the run** and counts against the ceiling.

And the rule that matters most, which lives in `calibration.py`: an uncalibrated
judge is decoration. Until you know its agreement with reference labels, a red
build is an opinion.
"""

from __future__ import annotations

import re

from .models import CheckResult
from .providers import Completion, MockProvider, Provider, _unit

DEFAULT_RUBRIC = """\
5 - Fully satisfies the criteria.
4 - Satisfies them with a minor omission or awkward phrasing.
3 - Partially satisfies them; a reviewer would need to fix it.
2 - Mostly fails, or answers a different question.
1 - Fails, empty, refused, or fabricated."""

PROMPT = """\
Grade the response against the criteria. Be strict.

RUBRIC
{rubric}

CRITERIA
{criteria}

RESPONSE
{response}

Reply with a single digit from 1 to 5 and nothing else."""

_DIGIT = re.compile(r"[1-5]")


class Judge:
    """Scores a response 1-5 against a case rubric, normalised to 0-1.

    Parameters
    ----------
    threshold:
        Normalised score at or above which the case passes. The default 0.6
        corresponds to a rubric score of 4.
    on_error:
        What to do when the judge itself fails. Defaults to failing *open* -
        a judge outage should not turn every case red and block the pipeline.
        The run report records that it happened.
    """

    def __init__(
        self,
        provider: Provider | None = None,
        *,
        model: str = "mock/small",
        rubric: str = DEFAULT_RUBRIC,
        threshold: float = 0.6,
        on_error: str = "pass",
    ) -> None:
        self.provider = provider or MockProvider()
        self.model = model
        self.rubric = rubric
        self.threshold = threshold
        self.on_error = on_error
        self.cost_usd = 0.0
        self.calls = 0

    def score(
        self, response: str, criteria: str, *, case_id: str = ""
    ) -> tuple[CheckResult, float]:
        """Returns the check result and the raw normalised score."""
        if not response.strip():
            return CheckResult("judge", False, "empty response, judge skipped", 0.0), 0.0

        prompt = (
            PROMPT.format(rubric=self.rubric, criteria=criteria, response=response[:4000])
            + f'\n<!--promptgate:{{"__judge__":true,"__case_id__":"{case_id}"}}-->'
        )

        try:
            completion = self._call(prompt, response, criteria, case_id)
        except Exception as exc:  # noqa: BLE001
            if self.on_error == "raise":
                raise
            passed = self.on_error == "pass"
            return (
                CheckResult(
                    "judge",
                    passed,
                    f"judge unavailable ({exc}); failing {'open' if passed else 'closed'}",
                    0.5,
                ),
                0.5,
            )

        self.calls += 1
        self.cost_usd += completion.cost_usd

        match = _DIGIT.search(completion.text or "")
        if not match:
            passed = self.on_error == "pass"
            return CheckResult(
                "judge", passed, f"unparseable verdict {completion.text[:32]!r}", 0.5
            ), 0.5

        rating = int(match.group())
        score = (rating - 1) / 4
        passed = score >= self.threshold
        return CheckResult("judge", passed, f"rated {rating}/5", round(score, 4)), score

    def _call(self, prompt: str, response: str, criteria: str, case_id: str) -> Completion:
        """Ask the grader.

        The mock path is handled here rather than inside MockProvider because
        grading is a different task from generating: a judge's job is to agree
        with a reference, so the simulation is "agrees with the reference with
        probability p", which is the number `calibration.py` then measures.
        """
        if isinstance(self.provider, MockProvider):
            good = _looks_good(response, criteria)
            correct = _unit(self.provider.seed, "judge", case_id, response[:64]) < JUDGE_ACCURACY
            rating = (5 if good else 1) if correct else (1 if good else 5)
            return Completion(text=str(rating), latency_ms=12.0, cost_usd=0.0, model=self.model)

        return self.provider.complete(prompt, model=self.model, max_tokens=4, temperature=0.0)


#: Simulated agreement rate of the mock judge with the reference. Set below 1
#: on purpose: a perfect judge would make the calibration report meaningless,
#: and the whole argument of this project is that you have to measure it.
JUDGE_ACCURACY = 0.88


def _looks_good(response: str, criteria: str) -> bool:
    """Reference verdict for the mock judge.

    Crude by design - it is a stand-in for ground truth, not a scorer. Real
    judges get the real prompt.

    Note there is no minimum length. An earlier version required 15
    characters, which silently rated every correct one-word answer 1/5 - and
    that is a fair illustration of the trap: a rubric written for prose,
    applied to a prompt that returns a label, grades the format instead of the
    content. `docs/golden-datasets.md` says more about it. The fix is on both
    sides: the grader tolerates short answers, and a rubric has to name what a
    good answer contains.
    """
    text = response.strip()
    if not text:
        return False
    lowered = text.lower()
    if any(bad in lowered for bad in ("as an ai", "i cannot", "unknown", "n/a", "none")):
        return False
    lowered_criteria = criteria.lower()

    # A short answer is a label, not prose. The question is simply whether the
    # rubric endorses it, so match the answer against the whole criteria
    # rather than sampling keywords out of the criteria - which is what made
    # "bug" score 1/5 against a rubric that literally contained the word.
    if len(text.split()) <= 3:
        token = re.sub(r"[^a-z_]", "", lowered)
        return bool(token) and token in lowered_criteria

    wanted = re.findall(r"[a-z_]{4,}", lowered_criteria)[:8]
    if not wanted:
        return True
    hits = sum(1 for w in wanted if w in lowered)
    return hits / len(wanted) >= 0.34
