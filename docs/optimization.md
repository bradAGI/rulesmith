# Optimization

`rulesmith optimize` searches for the graph with the best validation
score: agreement with the labels for classification (exact match, or weighted
by the task's `costs` and `accepted` labels), a grader model's score with
`--grader-model`, and normalized reward for [Doom](doom.md). DSPy writes candidate graphs, and
[GEPA](https://gepa-ai.github.io/gepa/guides/adapters/) decides which to
evaluate and keep.

## The search

1. **Seed.** Search starts from the graph you pass, or from a call-free graph
   that always answers the most common training label (the first listed button
   for Doom).
2. **Reflect.** When the current graph misses a training example, `GraphProposer`
   (a `dspy.Predict` with plain-text `ChatAdapter` fields, which any chat
   endpoint accepts) gets the graph, the inputs, its outputs, and the expected
   labels or episode rewards. Where the inputs have fields, it is told what each
   holds; with `--evidence`, it is also shown the few rules the training examples
   support best, each with how often it held, ranked by how many more examples
   it gets right than guessing its label would. That is the same for every search
   on the same examples, so finding the rule the data holds depends less on a
   lucky proposal, but the counts come from a sampled training set: on the
   jobs postings it had search drop the judge and lose on the seeds that had
   been best, so it is off unless asked for. Search revises one part at a time:
   **readings** holds reads and judgments; **rules** holds names, votes and the
   rules that decide. The proposer returns the replacement part, preserving
   references to nodes owned by the unchanged one, and names the move it took
   from a short list, each shown with an example in the task's own labels:
   `statements` (yes/no statements summed and cut), `rubric` (a score with a
   cut), `decompose` (a layer of statements for every label, offered when the
   node cap has room for it), `second_opinion` (a vote), `gate` (a judgment
   asked only where a reading says so, offered when the inputs have a number
   to read), `escalate` (a named judge for the own judge's unsure answers,
   offered when `--judge` names one) and `free` (anything else). Before the
   moves, every searched graph was read nodes, a choice and a coalesce.
3. **Check.** The replacement is assembled with the unchanged part and
   validated as a complete graph; no generated code runs. Validation errors
   go back to the proposer for up to `--repair-attempts` repairs. Candidates
   that remain invalid score zero, and their errors inform later proposals.
4. **Accept.** A proposal that beats its parent on the training batch is scored
   on validation.
5. **Select.** The graph with the best validation score is saved. The seed
   stays eligible, so a graph that overfits training will not replace it.
   Validation itself can be fitted when a search starts from an earlier winner:
   on TREC one went from 0.74 to 0.84 on validation and 0.83 on test, against
   its seed's 0.94. Before trusting a gain measured against another judge, score
   the seed's rules on that judge too.

The test split never reaches the optimizer. Score it once, after selection:

```sh
uv run rulesmith evaluate task.json runs/my-task/plan.json --split test \
  --backend ruling --output runs/my-task/test.json
```

## Settings

| Flag | Default | Meaning |
| --- | --- | --- |
| `--reflection-model` | required | DSPy model ID for the proposer |
| `--reflection-base-url` | — | Proposer endpoint, e.g. a local server's `/v1` |
| `--max-metric-calls` | 100 | Budget in example evaluations |
| `--reflection-batch` | 3 | Training examples per reflection |
| `--repair-attempts` | 2 | Times an invalid proposal goes back to the proposer with its validation error before search sees it |
| `--reflection-max-tokens` | 4096 | Token limit on each proposal |
| `--reflection-options` | — | Extra request JSON for the proposer, such as `{"reasoning": {"max_tokens": 2048}}` to cap a hosted thinking model that would otherwise reason past its reply limit |
| `--reflection-timeout` | none | Seconds before a proposal request fails, so a hung proposer stops the search instead of stalling it |
| `--retries` | 0 | Retries for judge requests that fail or time out |
| `--parallel-examples` | 1 | Examples scored at once; Doom allows more than 1 only for deathmatch tasks, whose duels each run their own games |
| `--mine` | off | Before any proposal, mine rules that one field, or two together, settle (a star rating, a flag, an empty text by its length) from the training data, put them ahead of the starting graph so its judgments are asked only about what they leave, and start search there if that scores better on validation |
| `--evidence` | off | Show the proposer the single-field rules the training examples support best, with how often each held. Counted on the training examples, so with a label sampled up it can over-fit the sample's rate; measure before leaving it on |
| `--significance` | off | Keep the starting graph unless the winner's gain on validation passes a one-sided sign test at this level, so a candidate that was only lucky on a small validation set does not replace it |
| `--call-cost` | 0 | Score taken off an example for each judge question it asks, so search prefers the graph that asks less when two answer alike |
| `--judge-cost NAME=COST` | `--call-cost` | Score taken off for each question to a judge named with `--judge`, so search escalates to a dearer judge only where it pays |
| `--extract-cost` | 0 | Score taken off for each extraction a graph runs |
| `--unit` | — | Search one unit of a map: the plan is the map, the task is the unit's own, search starts from the unit's graph, and the winner is placed back where the unit was and refused if the map's hand-offs no longer hold. The report's `unit` names it |
| `--resume` | off | Continue the search checkpointed under `<output>/search`, to `--max-metric-calls`; a larger budget than before runs the same search on. The task and starting graph must be the ones it began with |
| `--map-task` | — | With `--unit`: a labeled task for the whole map, scored on its validation examples with the old unit and the new; a winner that makes the map worse is refused, and the report's `unit.map_validation` records both scores |
| `--keep-judge` | off | Keep only a graph whose rules, wherever they answer without asking the judge, are right at least this often on the labeled training and validation examples (Laplace-smoothed, so a rule needs about 100 right answers in a row to reach 0.99). Search falls back to the next best candidate, then to the starting graph, and fails if none qualifies. Use it where a rule over a field anyone can set would be a way around the judge, such as fraud |
| `--merges` | 5 | Times search may combine two candidates, taking each part from whichever graph scored better |
| `--max-nodes` / `--max-depth` | 16 / 8 | Limits on generated graphs |
| `--prefer` | small | `small` tells the proposer to keep graphs small; `large` tells it to build layers of judgments, one or more per label, and raises the default `--max-nodes` to 64. With a 4B judge, `large` lost nine points on jobs and held level on TREC against `small`: every layer of statements scored below one well-written question. Use it with a judge that scores statements well, and score the result on test either way |
| `--seed` | 0 | Search seed |
| `--grader-model` / `--grader-base-url` | — | DSPy model that grades answers instead of the labels; examples may then omit `label`. Must differ from the judge the graph asks. Also on `evaluate`. |

All commands that ask a judge also take `--backend`, `--model`, `--base-url`,
`--timeout`, `--retries` and `--max-workers`. Retries default to 0, so a failed
judge request stops the run unless you raise them. With `--backend typesafe`,
the model defaults to `jev-latest`; pin a version for reproducible runs.

> [!TIP]
> Reflection only runs when the graph misses something in a training batch. If
> one label dominates your data, raise `--reflection-batch` so each batch
> includes the rarer labels.

## Reports

Each run writes a new directory with `plan.json`, `plan.rules` (the same graph
as rules, for a graph that has a rule form), `report.json`, and `search/`, where
GEPA checkpoints its state after every round. Output paths that already exist
are refused before any request is sent, unless `--resume` continues the search
checkpointed there. `touch <output>/search/gepa.stop` stops a running search
after its current round with everything kept; `rulesmith optimize ... --resume`
picks it up again, and with a larger `--max-metric-calls` runs it further. The
benchmark drivers stop this way at their deadline and resume on the next run.

| Field | Meaning |
| --- | --- |
| `validation_score` | Score of the saved graph, 0–1 |
| `seed_validation_score` | Score of the starting graph |
| `candidates` | Graphs that reached validation, including the seed. `1` means nothing new did. |
| `reflection_calls` | Proposer calls |
| `repairs` | Proposals that failed validation and were sent back to the proposer with the error, up to `--repair-attempts` each |
| `moves` | Each proposal's part, the move the proposer named for it (null where it named none), and whether the program it made became a candidate |
| `invalid_candidate_evaluations` | Evaluations rejected by schema or graph checks |
| `metric_calls` | Example evaluations used |
| `mined` | With `--mine`: how many rules were mined, their validation score, and whether search started from them |
| `improvement_p`, `kept_seed` | The sign test of the winner against the seed, and whether `--significance` kept the seed |
| `validation_scores` | Validation score of each candidate, in candidate order |
| `discovered_at` | Metric calls spent when each candidate was found, in candidate order, the seed's being 0 |
| `keep_judge_passed_over` | With `--keep-judge`: better-scoring candidates refused because a rule answered alone too often wrongly |
| `scored_by` | `labels`, `grader <model>`, or `games`: a graded score is not accuracy |
| `resumed`, `stopped_early` | Whether this run continued a checkpoint, and whether it ended on a `gepa.stop` file rather than its budget |
| `rules` | `plan.rules`, or null for a graph beyond rules, which is saved as JSON only |

Reports also record resolved model IDs, dependency versions, settings, and
task/seed hashes.

Search drops any rule counts the starting graph carries (see
[measuring rules](rules.md#how-sure-a-rule-is)), since they describe that
graph and not its rewrites. Run `rulesmith calibrate` on the graph search returns:
it also moves each line a rule draws on a judgment to where the training examples
put it, a numeric fit search would otherwise pay proposer calls and examples for.

## From Python

```python
import dspy
from typesafe_sdk import TypeSafeClient

from rulesmith.graph import Task
from rulesmith.optimize import GraphProposer, SearchConfig, optimize_plan

task = Task.model_validate_json(open("task.json").read())
proposer = GraphProposer(dspy.LM("openai/<served-model-id>", api_base="http://127.0.0.1:8080/v1"))
with TypeSafeClient(base_url="http://127.0.0.1:8010", api_key="local-no-auth") as client:
    plan, report = optimize_plan(
        task, task.baseline_plan(), client, "default", proposer, SearchConfig()
    )
```

## Testing

Default tests need no credentials. They use a local HTTP fixture and fixed
proposals, so they verify wiring, not model quality.

```sh
uv run pytest -q
```

Real inference against a running Ruling server:

```sh
uv run pytest tests/test_ruling.py --ruling-url http://127.0.0.1:8010 -q
```

Add real DSPy graph generation. This fails unless a generated graph reaches
validation and at least matches the seed:

```sh
uv run pytest tests/test_ruling.py --ruling-url http://127.0.0.1:8010 \
  --ruling-reflection-model "$REFLECTION_MODEL" \
  --ruling-reflection-url "$REFLECTION_BASE_URL" -q
```

Hosted providers (paid):

```sh
export RULESMITH_REFLECTION_MODEL="$REFLECTION_MODEL"
uv run pytest --live tests/test_live.py -q
```

A past local smoke run, whose output is not kept in the repo, with a `Qwen3.5-9B-4bit` proposer and Ruling's `Qwen3.5-4B-4bit`
judge went from an always-`sales` seed to a generated one-Choice graph: 1/3 → 3/3
on validation and on test, in 48 metric calls. That's 12 authored examples, which
shows the pipeline works end to end, not a benchmark result.

## Limits

- Search optimizes a `label` output: a class for classification, a button for Doom.
- Graphs are finite and static: no loops, tools, or dynamic nodes.
- The budget counts example evaluations, not requests or dollars, and excludes
  proposer calls. It can overshoot by one iteration.
- Provider errors stop the search rather than counting as wrong answers.
  In-flight requests are not canceled.
- Node inputs go to the judge; training examples and feedback go to the proposer.
- Extra structure is kept only when it raises the score. Small validation sets
  make selection noisy.
- DSPy's LiteLLM dependency may load `.env` files, so shell variables may not be
  the only configuration source.

## What search writes

A rule list, not graph JSON: the language is in [Rules](rules.md), with its
[reference](rules.md#reference).
