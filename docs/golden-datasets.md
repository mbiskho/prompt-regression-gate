# Designing a golden set worth having

The tool is easy. The golden set is the work, and a bad one is worse than none
because it produces confident green builds.

## Start from failures, not from features

The instinct is to write cases covering what the prompt should do. That
produces a suite full of easy cases that pass forever and catch nothing.

Better sources, in order:

1. **Bugs you have already shipped.** Every support ticket about the AI feature
   is a case. These are free and they are the ones that recur.
2. **Inputs that embarrass the model.** Empty strings, very long inputs, mixed
   languages, adversarial phrasing, inputs where the honest answer is "I don't
   know".
3. **The boring middle.** A handful of ordinary inputs, so you notice when
   something breaks for everyone rather than only at the edges.

Twenty cases from the first two categories beat two hundred from the third.

## Write assertions, not rubrics

*If you can describe what "wrong" looks like without using the word "good", it
is an assertion.*

| you want to check | use |
|---|---|
| output parses as JSON | `json_valid` |
| the schema holds | `json_schema` |
| a field is present | `required_fields` |
| the answer is a valid label | `one_of` |
| the answer is the *right* label | `expected_label` |
| no contact details leaked | `no_pii` |
| it stayed brief | `max_words` |
| it did not refuse | `not_contains: ["as an ai", "i cannot"]` |
| it answered fast enough | `max_latency_ms` |

Only reach for a rubric when none of these fit — open-ended generation,
summarisation, tone. In the two suites shipped here, 37 of 39 cases are pure
assertions.

### The trap: format checks are not correctness checks

`one_of` verifies the model returned *a* label from your list. It does not
verify the *right* one. A model that confidently answers `billing` for a login
problem sails through, the suite stays green, and quality erodes invisibly.

Whenever the case knows the answer, add `expected_label`. Where it genuinely
cannot know, you need a rubric and a calibrated judge — which is a much more
expensive commitment, so it is worth checking twice whether the case really
cannot be made decidable.

### A real mistake from building this

The triage suite originally had two judged cases with rubrics like *"identifies
an escalation risk and does not dismiss the customer"* — applied to a prompt
whose entire job is to emit one word. The judge rated every correct answer 1/5.

The rubric was written for prose and the prompt returns a label, so the judge
was grading the format. Two fixes were needed, on both sides: the grader had to
tolerate short answers, and the rubric had to name what a good answer actually
contains. A rubric that does not describe the shape of a correct answer is not
a rubric, it is a mood.

## Version the set and hash the cases

The golden set has a `version`, and every case has a content hash covering its
inputs, checks and rubric.

That hash is what stops the most common way of "fixing" a red build: editing
the case until it passes. When a case's hash changes, the diff reports it under
`changed` and refuses to compare it against its old result. The reviewer sees
that the test moved, which is the information they needed.

Tags and `xfail` are deliberately excluded from the hash — retagging a case
does not change what it tests.

## Use xfail instead of deleting

A case that fails because of a known bug should be marked `xfail: true`, not
removed. The suite stays green, the evidence stays in the repo, and when the
fix lands the diff reports it under "xfail cases now passing" so the marker
gets removed rather than quietly hiding a real signal forever.

## Keep the suite cheaper than the feature

Set `--budget`. A regression suite that costs more per run than the feature it
guards will be moved to a nightly job, then to a manual job, then deleted.

The cheapest way to stay under budget is to write assertions instead of
rubrics: assertions are free and instant, and the judge only runs on cases that
already passed everything else.

## Calibrate the judge before you trust it

If any case uses a rubric, you need to know how often the judge agrees with a
human on your data. Two hundred hand-labelled examples is the usual figure and
the work is genuinely dull.

Skip it and the whole thing is decoration: a red build nobody believes gets
overridden the second time it blocks a release, and after that the gate is
theatre.

```bash
promptgate calibrate labels.jsonl
```

Report Cohen's κ, not raw agreement — raw agreement is inflated by class
imbalance, and a judge that says "good" to everything scores 90% on a set that
is 90% good.

Below κ 0.6, do not gate on the judge. Use it as an advisory signal, or go back
and turn more cases into assertions.

**Watch false positives specifically.** A judge that accepts bad output keeps
the build green while the product degrades, which is the failure the gate
exists to stop. A judge that rejects good output is annoying; a judge that
accepts bad output is useless.

## Nightly, against pinned versions

Prompt changes are not the only thing that can break you. Pin your model to a
dated snapshot and run the suite on a schedule, so a provider changing the
model behind a name shows up as an issue rather than a support ticket.

If you pin to a floating alias, this job cannot work — the alias is the thing
hiding the change.
