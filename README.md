# rulesmith

![rulesmith: rules that ask a local model for typed decisions](image.png)

**rulesmith lets DSPy build decision graphs for you.** Give it a classification
task and labeled examples. It searches for a small program of rules: cheap tests
on the input's fields settle the cases they can, and a local or hosted judge is
asked typed questions about the rest. [GEPA](https://github.com/gepa-ai/gepa)
keeps the revisions that score better, and the result is a rules file you can
read, edit, run, and serve.

```
read has_company_logo in [0, 1]
read length of company_profile in [0, 10000]
when has_company_logo == 0 and length of company_profile == 0: fraudulent
always: ask "Is this job posting a scam?" [fraudulent: "fraudulent"] [genuine: "genuine"]
```

![Same 4B judge, same 100 examples: rulesmith beat DSPy's tuned classifier by 6.7 points on job-fraud detection with 29% of the model calls, and tied it on clothing reviews with 6%](docs/assets/claim.svg)

The [benchmarks](docs/benchmarks.md) have every dataset, including where rulesmith
does not help: on one-sentence inputs it ties DSPy, and on job postings a TF-IDF
classifier still beats it.

An independent research prototype, not affiliated with TypeSafe.

## Installation

```sh
pip install rulesmith
```

Python 3.12 to 3.14. Extras for the games and benchmarks are listed in the
[quickstart](docs/quickstart.md#installation).

## Quickstart

With a judge running ([Ruling](https://github.com/bradAGI/ruling) on a Mac or through vLLM,
Cloudflare's Clef-flash on an NVIDIA GPU or a Mac, TypeSafe's hosted Jev, or any
OpenAI-compatible chat API that returns log-probabilities), any OpenAI-compatible model to write the rules, and a
labeled task such as [this one](examples/support-routing/task.json):

```sh
export OPENAI_API_KEY=local-no-auth   # a local model server takes any key
rulesmith optimize task.json --backend ruling \
  --reflection-model "openai/<model>" --reflection-base-url http://127.0.0.1:8080/v1 \
  --output runs/my-task
rulesmith run runs/my-task/plan.rules input.json --backend ruling
```

The [quickstart](docs/quickstart.md) walks through it with every command's output.

## Beyond classification

The same search plays games. Pitted against a league of other graphs, this one learned to
hunt in Doom deathmatch and beat the champion 12-0, with no model calls at all
([how](docs/doom.md)):

![A searched rules graph hunting down the Doom champion it dethroned](docs/assets/doom/duel-hunter.gif)

## Learn more

- [Quickstart](docs/quickstart.md): installation, judges, and a task to a running graph.
- [Notebook](notebooks/quickstart.ipynb): rules for spotting fake job postings, end to end in Colab with an OpenAI key.
- [Local notebook](notebooks/local.ipynb): the same with Clef-flash judging on your Mac or NVIDIA GPU, free per call.
- [Tasks](docs/tasks.md): the task file, drafting labels, structured inputs, costs, and graders.
- [Rules](docs/rules.md): the language search writes, confidence, calibration, escalation, and extraction.
- [Graph format](docs/graphs.md): the JSON behind each node, execution, and drawing a graph.
- [Maps](docs/maps.md): many decisions as one map, searched a unit at a time.
- [Optimization](docs/optimization.md): how search runs, its settings, and its limits.
- [Command line](docs/cli.md): every command, including `serve` and `drift`, and the backends.
- [Python](docs/python.md): running and building graphs from code.
- [Benchmarks](docs/benchmarks.md): rulesmith against DSPy, TF-IDF and direct calls on five public datasets.
- [Related work](docs/related-work.md): when to use rulesmith and when DSPy, and what came before.
- [Doom](docs/doom.md) and [Chess](docs/chess.md): graphs that play.
- [Claude Code skill](skills/rulesmith/SKILL.md): copy `skills/rulesmith` into `~/.claude/skills/`.

## Development

```sh
uv sync --locked
uv run pytest -q
```

## License

[MIT](LICENSE).
