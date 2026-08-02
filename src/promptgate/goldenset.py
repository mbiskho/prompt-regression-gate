"""Golden set loading, versioning and validation.

A golden set is a YAML file: a version, a prompt, and a list of cases. It is
meant to be reviewed in a pull request like any other test, which is why the
format is flat and readable rather than clever.

Two rules the loader enforces, both learned the hard way:

* **Every case must be able to fail.** A case with no checks and no rubric
  always passes and inflates the pass rate while testing nothing.
* **Case ids must be unique.** Results are keyed by id, so a duplicate
  silently overwrites its twin in the baseline and the diff goes quiet.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Case, PromptSpec


class GoldenSetError(ValueError):
    """The golden set is malformed. Always fatal - never run a broken suite."""


@dataclass
class GoldenSet:
    version: str
    prompt: PromptSpec
    cases: list[Case] = field(default_factory=list)
    description: str = ""
    #: Fail the run if the pass rate drops below this. Optional; the diff
    #: against a baseline is the primary gate.
    min_pass_rate: float | None = None

    @property
    def content_hash(self) -> str:
        joined = "|".join(sorted(c.content_hash for c in self.cases))
        return hashlib.sha256(joined.encode()).hexdigest()[:12]

    def case(self, case_id: str) -> Case | None:
        return next((c for c in self.cases if c.id == case_id), None)

    def filter(self, tags: list[str] | None = None) -> GoldenSet:
        if not tags:
            return self
        wanted = set(tags)
        return GoldenSet(
            version=self.version,
            prompt=self.prompt,
            cases=[c for c in self.cases if wanted & set(c.tags)],
            description=self.description,
            min_pass_rate=self.min_pass_rate,
        )

    def validate(self) -> None:
        if not self.cases:
            raise GoldenSetError("golden set contains no cases")

        seen: set[str] = set()
        problems: list[str] = []
        for case in self.cases:
            if case.id in seen:
                problems.append(f"duplicate case id {case.id!r}")
            seen.add(case.id)
            if not case.checks and not case.rubric:
                problems.append(
                    f"case {case.id!r} has no checks and no rubric, so it can never fail"
                )
            for name in case.checks:
                from .assertions import CHECKS

                if name not in CHECKS:
                    problems.append(f"case {case.id!r} uses unknown check {name!r}")
        if problems:
            raise GoldenSetError("; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "version": self.version,
            "prompt": {
                "id": self.prompt.id,
                "version": self.prompt.version,
                "model": self.prompt.model,
                "template": self.prompt.template,
            },
            "cases": [c.to_dict() for c in self.cases],
        }
        if self.description:
            payload["description"] = self.description
        if self.min_pass_rate is not None:
            payload["min_pass_rate"] = self.min_pass_rate
        return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pyyaml is required; pip install promptgate") from exc
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise GoldenSetError(f"{path}: expected a mapping at the top level")
    return data


def load_goldenset(path: Path | str, *, prompt_file: Path | str | None = None) -> GoldenSet:
    """Load and validate a golden set.

    The prompt template can live inline or in its own file. Separate files are
    the better default: a prompt diff is then readable in review, rather than
    buried in YAML escaping.
    """
    path = Path(path)
    data = _load_yaml(path)

    prompt_data = data.get("prompt")
    if not isinstance(prompt_data, dict):
        raise GoldenSetError(f"{path}: missing 'prompt' section")

    template = prompt_data.get("template", "")
    if prompt_file:
        # An explicit override comes from the command line, so it resolves
        # against the working directory. A `template_file` inside the YAML
        # resolves against the YAML, so a golden set stays portable.
        resolved = Path(prompt_file)
        if not resolved.exists():
            raise GoldenSetError(f"prompt template file not found: {resolved}")
        template = resolved.read_text()
    elif prompt_data.get("template_file"):
        resolved = Path(prompt_data["template_file"])
        if not resolved.is_absolute():
            resolved = path.parent / resolved
        if not resolved.exists():
            raise GoldenSetError(f"{path}: prompt template file not found: {resolved}")
        template = resolved.read_text()
    if not template.strip():
        raise GoldenSetError(f"{path}: prompt has no template")

    prompt = PromptSpec(
        id=prompt_data.get("id", path.stem),
        version=str(prompt_data.get("version", "unversioned")),
        template=template,
        model=prompt_data.get("model", "mock/small"),
        temperature=float(prompt_data.get("temperature", 0.0)),
        max_output_tokens=int(prompt_data.get("max_output_tokens", 512)),
    )

    raw_cases = data.get("cases") or []
    if not isinstance(raw_cases, list):
        raise GoldenSetError(f"{path}: 'cases' must be a list")

    golden = GoldenSet(
        version=str(data.get("version", "1")),
        prompt=prompt,
        cases=[Case.from_dict(c) for c in raw_cases],
        description=data.get("description", ""),
        min_pass_rate=data.get("min_pass_rate"),
    )
    golden.validate()
    return golden


def save_goldenset(golden: GoldenSet, path: Path | str) -> Path:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(golden.to_dict(), sort_keys=False, width=100))
    return path
