"""Tests for promptgate.

Weighted toward the two things that decide whether the gate is trustworthy:
the diff (does it name the right cases, and can it be fooled) and the
calibration (does kappa behave when it should).
"""

from __future__ import annotations

import json

import pytest

from promptgate import (
    Case,
    GoldenSet,
    GoldenSetError,
    Judge,
    PromptSpec,
    ResultStore,
    Runner,
    Status,
    cohens_kappa,
    diff_runs,
    interpret,
    load_goldenset,
    run_checks,
)
from promptgate.assertions import extract_json
from promptgate.calibration import LabelledExample, calibrate
from promptgate.diff import render_markdown
from promptgate.providers import MockProvider, detect_instructions, strip_payload

LABELS = ["billing", "bug", "other"]


# -- assertions ------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        'Sure! Here you go: {"a": 1}',
        '{"a": 1}\n\nHope that helps.',
    ],
)
def test_extract_json_tolerates_model_chatter(text):
    """Strictness here would fire on formatting, not content - and a gate that
    goes red for cosmetic reasons is one people learn to ignore."""
    assert extract_json(text) == {"a": 1}


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_required_fields_gives_partial_credit():
    (result,) = run_checks('{"a": 1, "b": 2}', {"required_fields": ["a", "b", "c"]})
    assert not result.passed
    assert result.score == pytest.approx(2 / 3, abs=1e-4)


def test_json_schema_rejects_wrong_type():
    schema = {"type": "object", "required": ["total"], "properties": {"total": {"type": "number"}}}
    (result,) = run_checks('{"total": "lots"}', {"json_schema": schema})
    assert not result.passed


def test_booleans_are_not_numbers():
    schema = {"type": "object", "properties": {"total": {"type": "number"}}}
    (result,) = run_checks('{"total": true}', {"json_schema": schema})
    assert not result.passed


def test_no_pii_catches_an_email_without_echoing_it():
    """A CI log is a bad place to reproduce the PII you just detected."""
    (result,) = run_checks("Contact sam@acme.io for details", {"no_pii": ["email"]})
    assert not result.passed
    assert "sam@acme.io" not in result.detail


def test_one_of_accepts_an_embedded_label_at_a_lower_score():
    (result,) = run_checks("The label is: billing.", {"one_of": LABELS})
    assert result.passed and result.score == 0.8


def test_one_of_rejects_ambiguity():
    (result,) = run_checks("could be billing or bug", {"one_of": LABELS})
    assert not result.passed


def test_latency_budget_uses_measured_latency():
    (result,) = run_checks("ok", {"max_latency_ms": 100}, latency_ms=250)
    assert not result.passed and "250ms" in result.detail


def test_unknown_check_fails_loudly():
    """Silently skipping a typo'd check would quietly weaken the suite."""
    (result,) = run_checks("anything", {"no_such_check": True})
    assert not result.passed and "unknown check" in result.detail


def test_a_broken_check_is_a_failed_check_not_a_crash():
    (result,) = run_checks("x", {"regex": "([unclosed"})
    assert not result.passed


# -- golden sets -----------------------------------------------------------


def _golden(**kwargs) -> GoldenSet:
    return GoldenSet(
        version="1",
        prompt=PromptSpec(id="p", version="v1", template="Do the thing: {x}"),
        **kwargs,
    )


def test_case_with_no_checks_and_no_rubric_is_rejected():
    """Such a case always passes and inflates the pass rate while testing nothing."""
    golden = _golden(cases=[Case(id="a", inputs={"x": "1"})])
    with pytest.raises(GoldenSetError, match="never fail"):
        golden.validate()


def test_duplicate_case_ids_are_rejected():
    golden = _golden(
        cases=[
            Case(id="a", inputs={"x": "1"}, checks={"min_words": 1}),
            Case(id="a", inputs={"x": "2"}, checks={"min_words": 1}),
        ]
    )
    with pytest.raises(GoldenSetError, match="duplicate"):
        golden.validate()


def test_empty_golden_set_is_rejected():
    with pytest.raises(GoldenSetError):
        _golden(cases=[]).validate()


def test_missing_template_input_raises_rather_than_blanking():
    """A silently-empty substitution produces a prompt that tests nothing."""
    prompt = PromptSpec(id="p", version="v1", template="Classify {ticket}")
    with pytest.raises(KeyError, match="ticket"):
        prompt.render({"wrong_key": "x"})


def test_case_hash_ignores_tags_but_tracks_checks():
    base = Case(id="a", inputs={"x": "1"}, checks={"min_words": 1})
    retagged = Case(id="a", inputs={"x": "1"}, checks={"min_words": 1}, tags=["new"])
    edited = Case(id="a", inputs={"x": "1"}, checks={"min_words": 5})
    assert base.content_hash == retagged.content_hash
    assert base.content_hash != edited.content_hash


def test_example_golden_sets_load_and_validate():
    for path in ("examples/golden/triage.yaml", "examples/golden/extract.yaml"):
        load_goldenset(path).validate()


# -- provider --------------------------------------------------------------


def test_mock_reacts_to_the_prompt():
    """A fake model that ignores the prompt cannot demonstrate a prompt gate."""
    assert "label_only" in detect_instructions("Reply with the label only.")
    assert "label_only" not in detect_instructions("Explain your reasoning.")


def test_mock_is_deterministic():
    provider = MockProvider()
    prompt = 'Reply with the label only.\n<!--promptgate:{"__case_id__":"c1","__answer__":"bug","__labels__":["bug","other"]}-->'
    first = provider.complete(prompt, model="mock/small").text
    second = provider.complete(prompt, model="mock/small").text
    assert first == second


def test_dropping_an_instruction_changes_the_output():
    provider = MockProvider()
    fixture = (
        '<!--promptgate:{"__case_id__":"c1","__answer__":"bug","__labels__":["bug","other"]}-->'
    )
    terse = provider.complete("Reply with the label only.\n" + fixture, model="m").text
    chatty = provider.complete("Explain your reasoning.\n" + fixture, model="m").text
    assert terse != chatty
    assert len(chatty.split()) > len(terse.split())


def test_fixture_is_stripped_before_a_real_call():
    assert "promptgate:" not in strip_payload('Do it\n<!--promptgate:{"a":1}-->')


# -- runner ----------------------------------------------------------------


@pytest.fixture
def triage():
    return load_goldenset("examples/golden/triage.yaml")


def test_run_produces_a_result_per_case(triage):
    report = Runner(MockProvider(), judge=Judge(MockProvider())).run(triage)
    assert len(report.results) == len(triage.cases)
    assert 0.0 < report.pass_rate <= 1.0


def test_judge_is_skipped_when_a_deterministic_check_already_failed():
    """Paying a model to confirm a schema failure is money spent to learn nothing."""
    golden = GoldenSet(
        version="1",
        prompt=PromptSpec(id="p", version="v1", template="Say something about {x}"),
        cases=[Case(id="a", inputs={"x": "1"}, checks={"min_words": 500}, rubric="anything")],
    )
    judge = Judge(MockProvider())
    Runner(MockProvider(), judge=judge).run(golden)
    assert judge.calls == 0


def test_provider_failure_is_an_error_not_a_prompt_failure():
    class Broken:
        name = "broken"

        def complete(self, prompt, **kwargs):
            raise RuntimeError("upstream 503")

    golden = GoldenSet(
        version="1",
        prompt=PromptSpec(id="p", version="v1", template="{x}"),
        cases=[Case(id="a", inputs={"x": "1"}, checks={"min_words": 1})],
    )
    report = Runner(Broken()).run(golden)
    assert report.results[0].status is Status.ERROR
    assert "503" in report.results[0].error


def test_spend_ceiling_stops_the_run_and_marks_it_incomplete(triage):
    """A suite that silently skipped half its cases and passed is worse than
    one that failed."""
    provider = MockProvider(price_per_mtok=5000.0)
    report = Runner(provider, budget_usd=0.001).run(triage)
    assert report.stopped_early
    assert len(report.results) < len(triage.cases)
    assert not report.ok


def test_xfail_case_does_not_fail_the_run():
    golden = GoldenSet(
        version="1",
        prompt=PromptSpec(id="p", version="v1", template="{x}"),
        cases=[Case(id="a", inputs={"x": "1"}, checks={"min_words": 9999}, xfail=True)],
    )
    report = Runner(MockProvider()).run(golden)
    assert report.results[0].status is Status.FAIL
    assert report.ok


# -- diff ------------------------------------------------------------------


def _run(prompt_file: str | None = None):
    golden = load_goldenset("examples/golden/triage.yaml", prompt_file=prompt_file)
    provider = MockProvider()
    return Runner(provider, judge=Judge(provider)).run(golden)


def test_seeded_regression_is_caught_and_named():
    """The headline claim, executed."""
    baseline = _run()
    candidate = _run("examples/prompts/triage.v2.txt")
    delta = diff_runs(baseline, candidate)

    assert delta.has_regression
    assert len(delta.regressed) >= 20
    assert delta.pass_rate_after < delta.pass_rate_before
    assert all(d.detail for d in delta.regressed), "every regression must say why"


def test_unchanged_prompt_produces_no_regression():
    delta = diff_runs(_run(), _run())
    assert not delta.has_regression
    assert delta.pass_rate_delta == 0.0


def test_editing_a_case_prevents_it_being_compared():
    """Rewriting a case is the easiest way to bury a real regression."""
    baseline = _run()
    candidate = _run()
    candidate.results[0].case_hash = "different000"

    delta = diff_runs(baseline, candidate)
    assert candidate.results[0].case_id in delta.changed
    assert candidate.results[0].case_id not in [d.case_id for d in delta.regressed]


def test_added_and_removed_cases_are_reported_separately():
    baseline = _run()
    candidate = _run()
    candidate.results = candidate.results[:-1]
    delta = diff_runs(baseline, candidate)
    assert delta.removed and not delta.regressed


def test_markdown_comment_names_the_regressed_cases():
    baseline = _run()
    candidate = _run("examples/prompts/triage.v2.txt")
    body = render_markdown(diff_runs(baseline, candidate), candidate)
    assert "Regression detected" in body
    assert "triage-000" in body


# -- store -----------------------------------------------------------------


def test_store_roundtrip_and_labelled_baseline(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    report = _run()
    store.save(report, label="baseline")
    assert store.get(report.run_id).run_id == report.run_id
    assert store.latest("triage", label="baseline").run_id == report.run_id
    assert store.latest("triage", label="nonexistent") is None


def test_case_history_tracks_one_case_over_runs(tmp_path):
    store = ResultStore(tmp_path / "r.db")
    store.save(_run(), label="a")
    store.save(_run("examples/prompts/triage.v2.txt"), label="b")
    history = store.case_history("triage", "triage-000")
    assert len(history) == 2


# -- calibration -----------------------------------------------------------


def test_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa([True, False, True, False], [True, False, True, False]) == 1.0


def test_kappa_is_zero_for_chance_agreement():
    """Raw agreement would report 50% here and look meaningful."""
    assert cohens_kappa([True, True, False, False], [True, False, True, False]) == 0.0


def test_kappa_punishes_a_constant_rater():
    """A judge that says 'good' to everything scores high raw agreement on an
    imbalanced set. Kappa is what stops that looking like competence."""
    reference = [True] * 9 + [False]
    always_good = [True] * 10
    assert cohens_kappa(always_good, reference) == 0.0


def test_kappa_is_negative_when_worse_than_chance():
    assert cohens_kappa([True, True, False], [False, False, True]) < 0


@pytest.mark.parametrize(
    ("kappa", "word"), [(0.1, "slight"), (0.5, "moderate"), (0.7, "substantial"), (0.9, "almost")]
)
def test_interpretation_bands(kappa, word):
    assert word in interpret(kappa)


def test_calibration_reports_false_positives_separately():
    """A judge that accepts bad output keeps the build green while the product
    degrades - the exact failure the gate exists to stop."""
    examples = [
        LabelledExample(
            f"c{i}", "billing", "A charge dispute should be labelled billing.", i % 2 == 0
        )
        for i in range(20)
    ]
    report = calibrate(Judge(MockProvider()), examples)
    assert report.n == 20
    assert (
        report.false_positives + report.false_negatives == sum(1 for d in report.disagreements)
        or report.disagreements
    )


def test_shipped_calibration_set_clears_the_bar():
    from promptgate.calibration import load_labels

    report = calibrate(Judge(MockProvider()), load_labels("examples/labels.jsonl"))
    assert report.n == 200
    assert report.meets_bar, f"kappa {report.kappa:.3f} below the 0.6 gate"
    assert report.label_source == "synthetic", "labels here are not human; do not claim otherwise"


# -- CLI -------------------------------------------------------------------


def test_gate_exits_non_zero_on_a_seeded_regression(tmp_path, monkeypatch):
    from promptgate.cli import main

    db = str(tmp_path / "r.db")
    monkeypatch.chdir(pytest.importorskip("pathlib").Path.cwd())

    assert main(["--db", db, "-q", "baseline", "examples/golden/triage.yaml"]) == 0
    code = main(
        [
            "--db",
            db,
            "-q",
            "gate",
            "examples/golden/triage.yaml",
            "--prompt-file",
            "examples/prompts/triage.v2.txt",
        ]
    )
    assert code == 1


def test_gate_exits_zero_when_nothing_changed(tmp_path):
    from promptgate.cli import main

    db = str(tmp_path / "r.db")
    main(["--db", db, "-q", "baseline", "examples/golden/triage.yaml"])
    assert main(["--db", db, "-q", "gate", "examples/golden/triage.yaml"]) == 0


def test_invalid_golden_set_exits_two(tmp_path):
    from promptgate.cli import main

    bad = tmp_path / "bad.yaml"
    bad.write_text(json.dumps({"version": "1", "prompt": {"template": "x"}, "cases": []}))
    assert main(["-q", "run", str(bad)]) == 2
