# Tasks

A task is a JSON file: a `description`, the allowed `labels`, and examples in
three splits. `train` drives proposals, `validation` picks the winner, and
`test` is scored once at the end and never reaches the optimizer.

## Describing the task

Pick the case that matches your data.

### I only have inputs

List the allowed labels and your raw inputs in `inputs.json`:

```json
{
  "description": "Route each support ticket to billing, technical, or sales.",
  "labels": ["billing", "technical", "sales"],
  "inputs": [
    "I was charged twice this month.",
    "The export button returns a 500 error.",
    "Can I get a quote for twenty seats?",
    "Please refund the charge after my cancellation.",
    "The app crashes when I open a project.",
    "Can we schedule a product demo before buying?"
  ]
}
```

Then have the proposer model label them. `--validation` and `--test` set how
many inputs are held out for selection and final scoring:

```sh
uv run rulesmith draft-labels inputs.json --output task.json \
  --labeler-model "$REFLECTION_MODEL" --labeler-base-url "$REFLECTION_BASE_URL" \
  --validation 1 --test 1
```

Output:

```json
{
  "labels": {
    "technical": 2,
    "billing": 2,
    "sales": 2
  },
  "output": "task.json"
}
```

This leaves four training inputs, one validation input, and one test input: a
workflow example, too small to measure model quality reliably. For real work,
use more examples and larger held-out splits.

> [!IMPORTANT]
> Review and correct `task.json` before optimizing. Search learns to agree with
> these labels, including their mistakes.

This is DSPy's teacher–student pattern: a large model labels, and the small
judge is fit to its judgment.

### My inputs are more than text

An input may be a string, a JSON object, or an array. Structure is worth giving
it: a rule can only `read` what has a path, so on a bare string every read is
missing and the graph has nothing to go on but judgments.

```json
{
  "description": "Route each refund request.",
  "labels": ["auto_refund", "review", "decline"],
  "inputs": [
    {"charge": {"amount": 40, "days_ago": 2}, "account": {"plan": "pro", "chargebacks": 0},
     "message": "Charged twice for the same seat."},
    {"charge": {"amount": 900, "days_ago": 60}, "account": {"plan": "free", "chargebacks": 2},
     "message": "I want all of it back."},
    {"charge": {"amount": 12, "days_ago": 1}, "account": {"plan": "enterprise", "chargebacks": 0},
     "message": "Wrong seat count on this invoice."}
  ]
}
```

Rules then mix what is known with what must be judged:

```
read charge.amount in [0, 100000]
read account.chargebacks in [0, 100]
ask genuine = "The message describes a mistake the customer did not cause."
when account.chargebacks >= 1: review
when charge.amount < 50 and genuine >= 0.8: auto_refund
always: review
```

### I have labeled examples

Start with the complete labeled sample:

```sh
cp examples/support-routing/task.json task.json
```

Or write `task.json` directly. This minimal example shows the required shape:

```json
{
  "description": "Route each support ticket to billing, technical, or sales.",
  "labels": ["billing", "technical", "sales"],
  "train": [{"id": "t1", "state": "I was charged twice.", "label": "billing"}],
  "validation": [{"id": "v1", "state": "The app crashes.", "label": "technical"}],
  "test": [{"id": "e1", "state": "Can I get a price quote?", "label": "sales"}]
}
```

`train` drives proposals, `validation` picks the winner, and `test` is scored once at the end.

Two optional additions keep the held-out figures honest:

- **`calibration`**, a fourth list of examples search never sees. `rulesmith
  calibrate` fits its lines and counts its rules on these when they are there.
  Without them it fits on the training and validation examples, which search
  chose the graph by, so the counts flatter it; the calibrate report says which
  with `fitted_on`.
- **`group`** on an example, for examples that are not independent: a ticket and
  its follow-up, two reviews by one customer. A task whose group appears in two
  splits is refused, since whichever split is held out would have seen the other.

```json
"calibration": [{"id": "k1", "state": "Refund my second charge.", "label": "billing", "group": "acct-17"}]
```

## How answers are scored

By default an answer scores 1 when it matches the example's label and 0
otherwise. A few additions in `task.json` refine that, and none calls a model.

**Some mistakes cost more.** `costs[expected][answered]` prices a wrong answer;
anything unlisted costs 1. The dearest mistake scores 0 and a cheaper one scores
in between, so search avoids the expensive ones first:

```json
"costs": {"fraud": {"approve": 10, "review": 1}}
```

Approving fraud is ten times worse than sending it to review. The feedback search
sees names what each mistake cost.

**Examples sampled at another rate than real traffic.** A rare label is often
sampled up, half scams and half genuine postings, so there are enough of each to
learn from; scored as sampled, a rule that flags too much looks better than it
is. `prevalence` gives each label's share of real traffic, and each example's
mistakes then count as much as the real traffic it stands for, in search, in
`calibrate`'s cutoff fit and in `evaluate`:

```json
"prevalence": {"fraudulent": 0.05, "genuine": 0.95}
```

Sampled half and half, a false flag now costs nineteen times what a missed scam
does, as it would on real traffic. `rulesmith evaluate` adds `at_prevalence`:
accuracy, and each label's precision and recall, as they would be at those
rates. Shares must cover every label and sum to 1; `costs` still apply on top.

**Some examples have more than one right answer.** `accepted` lists labels that
count as fully right besides `label`:

```json
{"id": "t7", "state": "Refund my duplicate charge and fix the export bug.", "label": "billing", "accepted": ["technical"]}
```

**Passing a case to a person.** Add a label for it and say what it costs, as a
share of the worst mistake, and search can learn to hand over the cases it is
unsure of rather than guess:

```json
"labels": ["approve", "decline", "human"],
"abstain": {"label": "human", "cost": 0.3}
```

No example may expect `human`; it is an answer, not a truth.

**No labels at all.** A stronger model can grade the answers instead. Examples
may then leave out `label`; where one is present, the grader is shown it as a
reference:

```sh
uv run rulesmith optimize task.json \
  --backend ruling \
  --reflection-model "$REFLECTION_MODEL" --reflection-base-url "$REFLECTION_BASE_URL" \
  --grader-model "$GRADER_MODEL" --grader-base-url "$GRADER_BASE_URL" \
  --output runs/my-task
```

Output (trimmed; with `GRADER_MODEL=openrouter/qwen/qwen3.8-27b`):

```text
...
{
  ...
  "seed_validation_score": 0.3333333333333333,
  "validation_score": 1.0,
  "improvement_p": 0.25,
  "kept_seed": false,
  "candidates": 2,
  "reflection_calls": 4,
  "metric_calls": 102,
  "backend": "ruling",
  "reflection_model": "openrouter/qwen/qwen3.8-27b",
  "scored_by": "grader openrouter/qwen/qwen3.8-27b"
}
```

The grader scores each answer from 0 to 1 and says why, and search reads the
reason. Each answer to each input is graded once, so a search pays for at most
one grade per distinct answer per example. Use a different, stronger model from
the one your graph asks. Otherwise search finds the graph that asks the grader's
own question and scores perfectly for nothing. rulesmith stops when the grader
and the judge report the same model name, but it cannot tell that two services
serve the same weights under different names, so that part is on you. Reports record `scored_by`, because a graded score is a
model's opinion, not accuracy against labels.
