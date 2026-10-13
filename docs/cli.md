# Command line

`rulesmith` is one command with subcommands; `rulesmith <command> --help` lists
every flag.

| Command | Does |
| --- | --- |
| `validate` | Check a graph against a task's labels and size limits, without running it |
| `run` | Execute a graph once on one input |
| `serve` | Answer decisions over HTTP, one endpoint per decision folder, with an optional trace |
| `drift` | Check a decision's production traces against what its rules were calibrated on; exits 1 when a rule has drifted |
| `evaluate` | Score a graph on a task split |
| `optimize` | Search for a better graph, writing `plan.json`, editable `plan.rules` and `report.json` |
| `arena` | Rank Doom graphs by Elo from deathmatch duels |
| `record` | Play a Doom episode to `.mp4` or `.gif` |
| `calibrate` | Fit how sure each judgment should read and where each rule draws its line on one, then count how often each rule is right, on the task's `calibration` examples or else its training and validation ones; with `--call-cost`, `--judge-cost` and `--extract-cost`, an escalation line lands where escalating pays. A rules file written from a rules file keeps its comments |
| `draw` | Write a graph as a Mermaid flowchart; `--closed` draws a map's units as single boxes |
| `unit-task` | Write the task of one unit of a map: the map runs over its task's examples, and the unit gets the cases it was handed, as handed, whose answer it can give. Calibrate and search a unit on this; `calibrate` refuses a whole map |
| `ablate` | Measure what each reading, judgment and name in a graph is worth |
| `draft-labels` | Label raw inputs with a strong model, for you to review as a task |
| `tune` | Fine-tune the judge on a task's examples with Ruling, test split held out |
| `distill` | Fine-tune the judge on a larger model's answers to a task's inputs, labeled or not, as soft targets |

Every command that asks a judge (`run`, `evaluate`, `optimize`, `calibrate`,
`arena`, `record`, `ablate`) takes `--backend`, `--model`, `--base-url`, `--timeout`,
`--retries` and `--max-workers`; see [backends](#backends). A graph whose
judgments name another judge with `using <name>` also needs
`--judge <name>=BACKEND[:MODEL][@URL]` for each, such as
`--judge large=ruling:mlx-community/Qwen3.6-35B-A3B-4bit@http://127.0.0.1:8012`;
the model and endpoint default to the backend's own. A graph that extracts fields needs
`--extractor-model`, any model DSPy reaches, and `--extractor-base-url` for an
OpenAI-compatible endpoint. `draft-labels` calls
a generative model instead, set with `--labeler-model` and `--labeler-base-url`.

`optimize` has flags of its own for the search budget, the proposer, checkpoints
and `--resume`, listed under [settings](optimization.md#settings); its `--unit`
and `--map-task` flags, for improving one unit of a map, are on the
[Maps](maps.md#working-with-a-map) page.

A `.rules` file is a graph wherever the CLI takes one, so the examples run as
they stand:

```sh
uv run rulesmith validate examples/support-routing/task.json examples/support-routing/graph.rules
uv run rulesmith run examples/support-routing/graph.rules examples/support-routing/input.json \
  --backend ruling
```

Output (trimmed):

```text
{
  "nodes": 5,
  "labels": ["billing", "technical", "sales"]
}
{
  "outputs": {"label": "billing"},
  ...
  "answers": {
    "team": {
      "type": "choice",
      "value": "billing",
      "probabilities": {"billing": 0.997, "technical": 0.003, "sales": 0.0003},
      "confidence": 0.995
    }
  },
  "skipped": ["rule_2"],
  ...
  "label": "billing",
  "decided_by": "rule_1",
  "confidence": 0.997
}
```

`run --trace decisions.jsonl` also appends the decision to that file, as
described under [Use from Python](python.md).

## Serving decisions

`serve` makes each decision folder an endpoint, so a workflow in any language
asks it at its branch points. It listens on 127.0.0.1 unless `--host` says
otherwise, takes the same judge flags as `run`, and prints one line with its
address once it is ready:

```sh
uv run rulesmith serve examples/order-review examples/support-routing \
  --backend ruling --trace decisions.jsonl
```

```text
{"url": "http://127.0.0.1:8020", "decisions": ["order-review", "support-routing"]}
```

```sh
curl -s -X POST http://127.0.0.1:8020/decisions/order-review -d '{"order": {"total": 900}}'
```

The answer is what `run` prints, label, rule and confidence included.
`GET /decisions` lists what is served. An input no rule applies to is still
answered, with 200: `label` and `decided_by` are null and `undecided` says why
(`"no_rule_applied"`, the one reason a valid graph can end without a decision),
so a workflow branches on that field rather than on a status code. A decision
that was made has `undecided: null`. A body that is not JSON gets 400; an
unknown name gets 404. With `--trace`, every decision is appended to the
file the console's production panel reads.

## Watching a decision in production

A rule that decides a far larger or smaller share of real traffic than of the
examples it was calibrated on is a sign the inputs have moved, before anyone
knows whether its accuracy did. `drift` reads a decision's traces, keeps the
decisions its current graph made, and tests each measured rule's share against
its calibrated one:

```sh
uv run rulesmith drift examples/ticket-triage traces/
```

```json
{
  "graph_sha256": "…",
  "decisions": 1200,
  "undecided": 3,
  "calls": {"judge large": 41, "the judge": 388},
  "drift": [{"rule": "rule_1", "measured_share": 0.12, "share": 0.31, "p": 1.2e-41}]
}
```

`calls` counts every model call by the model it went to, so an escalation rate
that creeps up shows here too. `undecided` counts decisions no rule made; they
count toward the traffic every rule's share is tested against, so a rise in them
moves the shares too. The command exits 1 when any rule drifted and 0
otherwise, so a scheduled job can alert on it. Decisions by earlier versions of
the graph are left out: a rule's name is its place in the list. Each trace record
names the model each call went to under `called`.

## Backends

| Backend | Flag | Model | Endpoint | Credential |
| --- | --- | --- | --- | --- |
| Jev (hosted) | `--backend typesafe` (default) | `jev-latest`, or `--model` | `TYPESAFE_BASE_URL`, default `https://api.typesafe.ai` | `TYPESAFE_API_KEY` |
| Ruling (local) | `--backend ruling` | `default`, or any name Ruling serves (`GET /v1/models` lists them) | `RULING_BASE_URL`, default `http://127.0.0.1:8010` | `RULING_API_KEY` if auth is on |
| Clef-flash (local) | `--backend clef` | `clef-flash` | `CLEF_BASE_URL`, default `http://127.0.0.1:8000` | none |
| Any OpenAI-compatible chat API | `--backend openai` | `gpt-4.1`, or `--model` | `OPENAI_BASE_URL`, default `https://api.openai.com/v1` | `OPENAI_API_KEY`, none for a local server |

The `openai` backend asks a chat model each question on its own, with the
options lettered (a yes-or-no question is A true, B false; a score's levels run
from lowest to highest), for a single token at temperature 0 with the ten most
likely first tokens' log-probabilities. An option's probability is the softmax
of its letter's log-probability over the option letters; a letter not among the
ten is taken to sit one nat below the least likely one listed. A choice is the
most probable option, a score the expected level, and confidence is computed
with TypeSafe's published formulas, so graphs and calibration read these
answers as they read Jev's. It needs a model that returns log-probabilities:
OpenAI's non-reasoning models do, as do local servers such as `mlx_lm.server`
and vLLM; reasoning models and some providers do not, and rulesmith stops with
an error naming the model. A model must also answer with the letter at once: one
that opens with a thinking block, such as Qwen3 with thinking on, is refused too.
A choice may have at most 26 options.

`--max-workers` (default 4) is how many requests rulesmith keeps open at once,
so pick it by what the judge does with them. Jev answers them in parallel, so
more workers finish sooner. Ruling on a Mac answers one at a time per model and
queues the rest, so more workers are safe but only add waiting; Ruling serving
through vLLM batches them on the GPU, so more workers pay there. Ruling turns
away requests beyond `RULING_MAX_PENDING` (default 64) with a 429 and a
`Retry-After`, which rulesmith waits out when `--retries` allows, so keep the
workers of every run sharing one server under that. Clef-flash's servers
answer one at a time, so use `--max-workers 1` with them. Clef-flash is
Cloudflare's trained 9B decision model, and there are two ways to serve it.

On Linux with an NVIDIA GPU, `rulesmith clef-serve` runs
[Cloudflare's own release](https://huggingface.co/Cloudflare/clef-flash) with
PyTorch. It downloads the weights on first start (about 19 GB), imports the
model code that ships with them, as the model card does, and prints its URL
once the model is loaded:

```sh
pip install 'rulesmith[clef]'
rulesmith clef-serve --port 8000
rulesmith run graph.rules input.json --backend clef --max-workers 1
```

The weights are bf16, so the GPU needs about 20 GB free: an L4 or A10G (24
GB), an A100, or anything larger. A free Colab T4 has 16 GB and is not enough.
`--device` picks `cuda`, `mps`, or `cpu` (default: the first one this machine
has), and `--weights Cloudflare/clef` serves the 27B Clef instead, which needs
about 55 GB, an 80 GB A100 or H100. A long state needs more memory than this;
a request that runs out is answered with a 500 and the server keeps serving.

On a Mac, the [MLX port](https://huggingface.co/mlx-community/clef-flash-4bit)
needs far less memory, being 4-bit (about 6 GB), and runs faster: on an M5 Max
it answered 300 one-question requests in 38 seconds, against 59 for
`clef-serve --device mps`, and chose the same answer for every one. Its own
server answers the same requests rulesmith sends:

```sh
cd "$(huggingface-cli download mlx-community/clef-flash-4bit)"
uv run --no-project --with mlx==0.32.3 --with mlx-lm==0.32.0 --with mlx-vlm==0.7.4 \
  python clef_mlx.py serve --port 8000
uv run rulesmith run graph.rules input.json --backend clef --max-workers 1
```

A Ruling judge fine-tuned on
your labels is served (`RULING_ADAPTER`) and asked the same way, and
`rulesmith tune task.json --output tuning` makes one: it writes the records
Ruling's trainer reads, every training and validation example asked the direct
question with its label in `train.jsonl` and the test examples in
`holdout.jsonl`, which the trainer refuses to overlap, then runs `ruling train`
from the checkout at `--ruling-dir` (default `$RULING_DIR`, else `../ruling`)
with the recipe the benchmark validated, 200 steps with the best of ten
validation checkpoints kept, on `--base`, a pure-attention model Ruling trains
at full speed. Anything after `--` goes to `ruling train` and overrides the
recipe. It prints where the adapter is and the line that serves it; rulesmith
then asks that judge like any other. The
[benchmarks](benchmarks.md#a-judge-tuned-on-the-same-examples) measure what a
tune on a hundred examples changes. Jev calls are billed: the
search budget counts examples, not requests or dollars, and `--call-cost` makes
search prefer graphs that ask less. The same search against Jev:

```sh
export TYPESAFE_API_KEY=<your key>
uv run rulesmith optimize task.json \
  --reflection-model "$REFLECTION_MODEL" --reflection-base-url "$REFLECTION_BASE_URL" \
  --output runs/my-task-jev
```
