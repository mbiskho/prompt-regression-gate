# Prompt Regression Gate

**Fails the build when a prompt edit makes the output worse.**

Tests for prompts. Deterministic assertions where you can write them — schema
valid, required field present, no PII, under the latency budget — and a judge
model only where you can't, both running against a versioned golden set. The
output is pass/fail plus a diff naming which cases regressed.

```bash
pip install -e .
promptgate baseline examples/golden/triage.yaml   # record the current behaviour
promptgate gate     examples/golden/triage.yaml   # exit 1 if anything got worse
```

---

## It catches a regression seeded on purpose

`triage.v2.txt` is `triage.v1.txt` with one line changed — `Reply with the
label only` becomes `Explain your reasoning`. A plausible edit. Nobody would
revert it in review.

```console
$ promptgate baseline examples/golden/triage.yaml
baseline recorded: 804154337e3c · 24/27 passed (88.9%)

$ promptgate gate examples/golden/triage.yaml --prompt-file examples/prompts/triage.v2.txt

pass rate 88.9% → 0.0% (-88.9%)

REGRESSED (24)
  triage-000  expected_label: got 'Based on the ticket, I would classify th', want 'billing'; max_words: 18 words > 6
  triage-001  expected_label: got 'Based on the ticket, I would classify th', want 'billing'; max_words: 18 words > 6
  ...

24 case(s) regressed — failing the build
$ echo $?
1
```

The same thing happens on the extraction suite when `Return JSON only` is
dropped: 11 of 12 cases regress, because the model starts answering in prose
and every schema check fails.

**CI runs this demo on every push** and fails if the seeded regression *stops*
being caught — the tool has to demonstrate its own claim, not assert it in a
README.

| | |
|---|---:|
| Cases | 39 across two suites |
| Full run | **0.08 s** |
| Cost per run | **$0.00** (mock provider) |
| Regressions caught / seeded | **35 / 35** |
| Judge agreement (Cohen's κ) | **0.748** — substantial |
| Tests | 49 |

---

## The judge is calibrated, and the labels are not human

This is the part that decides whether anyone believes a red build, so it gets
said plainly.

```console
$ promptgate calibrate examples/labels.jsonl
kappa 0.748 (substantial) over 200 examples [OK]
  raw agreement      88.5%
  judge says good    65.0%
  reference says good 64.5%
  false positives    12  (judge accepted what the reference rejected)
  false negatives    11
  label source       synthetic
```

**Those 200 labels are generated, not hand-labelled by me.** The spec for this
project called for a couple of hundred human labels, and producing them and
calling it a human baseline when it is not would be exactly the failure this
tool exists to prevent. `examples/labels.jsonl` is a synthetic reference
produced by a documented rule, and the generator is committed so the
provenance is inspectable. The κ it yields shows the calibration machinery
works; it does **not** show the judge agrees with a reviewer.

Point `promptgate calibrate` at your own labelled file and the number starts
meaning something. The CLI prints a warning whenever `label_source` is not
`human`, and CI fails if κ drops below 0.6 — a judge under the bar must not be
gating anything.

**Kappa rather than raw agreement**, because raw agreement is inflated by class
imbalance: if 90% of your cases are good, a judge that says "good" every time
scores 90% and is useless. There is a test for exactly that
(`test_kappa_punishes_a_constant_rater`).

---

## How it decides

```
case ─► render prompt ─► model ─► deterministic checks ─► judge ─► pass/fail
                                        (free)          (only if
                                                     everything else
                                                        passed)
```

**Assertions first, judge last.** The judge runs only when a case has a rubric
*and* every free check already passed. Paying a model to confirm that malformed
JSON is malformed is money spent to learn nothing. In the shipped suites, 37 of
39 cases never reach the judge.

The rule of thumb that survived contact with real golden sets: *if you can
describe what "wrong" looks like without using the word "good", it is an
assertion, not a rubric.*

Twelve checks ship: `json_valid`, `json_schema`, `required_fields`, `one_of`,
`expected_label`, `contains`, `not_contains`, `regex`, `min_words`,
`max_words`, `no_pii`, `max_latency_ms`. `register()` adds your own — domain
assertions are the highest-value ones, because nobody else can write them.

---

## Three ways this refuses to be fooled

**A case that cannot fail is rejected at load time.** No checks and no rubric
means it always passes and inflates the pass rate while testing nothing.
Duplicate case ids are rejected too — results are keyed by id, so a duplicate
silently overwrites its twin in the baseline and the diff goes quiet.

**An edited case is not compared.** Every case carries a content hash. If the
hash changed, the diff reports it under `changed` rather than comparing
verdicts — because rewriting a test is the easiest way to make a genuine
regression disappear, and the reviewer should see that the test moved.

**A run that hit its spend ceiling fails.** The budget is checked *before* each
call, and a run that stopped early is reported as incomplete rather than
partially green. A suite that silently skipped half its cases and passed is
worse than one that failed.

---

## Cost ceiling

```bash
promptgate --budget 0.50 gate examples/golden/triage.yaml
```

The suite must not quietly cost more than the feature it guards. The ceiling is
enforced before spending, not discovered by exceeding it. With the default mock
provider a run costs nothing, which is what lets CI run it on every push.

---

## Nightly drift

The prompt is half the system. The other half is a model you do not control,
and providers change what sits behind a name. A suite that only runs on pull
requests cannot see that — nothing in your repo changed, so nothing triggered,
and the first sign of trouble is a support ticket.

`.github/workflows/nightly-drift.yml` runs the same golden set against
**pinned, dated model snapshots** on a schedule and opens an issue when results
move. Pinning to a dated snapshot rather than a floating alias is the point: an
alias hides the very change you are trying to detect.

---

## Use it in CI

```yaml
- uses: bismakhomeini/prompt-regression-gate@v1
  with:
    goldenset: prompts/golden/triage.yaml
    budget: "0.50"
```

The action posts a PR comment naming the regressed cases and exits non-zero on
regression. `action.yml` is a composite action; the workflow in
`.github/workflows/ci.yml` shows the full setup including baseline caching.

---

## Golden set format

```yaml
version: 2026-08-02.1
min_pass_rate: 0.85
prompt:
  id: triage
  version: v1
  model: mock/small
  template_file: ../prompts/triage.v1.txt
cases:
  - id: triage-000
    inputs: { ticket: "I was charged twice for the Pro plan this month." }
    checks:
      one_of: [billing, bug, feature_request, account_access, other]
      expected_label: billing
      max_words: 6
      max_latency_ms: 3000
    tags: [billing]
```

Prompts live in their own files so a prompt diff is readable in review rather
than buried in YAML escaping. `docs/golden-datasets.md` covers how to build one
that is worth having.

---

## Development

```bash
make install
make test      # 49 tests, no network, no API key
make demo      # baseline, then the seeded regression
make calibrate
```

---

## Limits worth knowing

- **The default provider is a mock.** It reads the prompt for instruction
  markers and only honours the ones it finds — so dropping an instruction
  genuinely changes the output, which is what makes the demo real rather than
  theatre. It is a caricature of model behaviour, faithful to the *failure*
  being tested. Real runs use `--provider openai`.
- **The judge simulation has a built-in 12% error rate.** A perfect judge
  would make the calibration report meaningless.
- **`one_of` checks format, not correctness.** A model returning the wrong
  label from the list passes it. Pair it with `expected_label` where the case
  knows the answer; that distinction is why the triage suite uses both.

MIT licensed.
