"""Records for tuning the judge on a task's examples, in the shape Ruling's trainer reads.
rulesmith never trains a model; it writes what a trainer needs: every training and validation
example asked the direct question over the task's labels, with the label it carries, and the
test examples in a holdout file the trainer refuses to overlap, so a tuned judge is scored on
rows it has never seen."""

import json
import shutil
import subprocess
from pathlib import Path

from rulesmith.graph import Choice, Task
from rulesmith.runtime import read_answer

FILES = ("train", "holdout")
# What the benchmark validated for a hundred examples: a light tune, the best of ten validation
# checkpoints kept, on a pure-attention base that Ruling trains at full speed.
RECIPE = ("--steps", "200", "--eval-every", "20", "--valid-fraction", "0.2")
BASE = "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def direct_question(task: Task) -> dict:
    """The simplest question a judge can be asked about a task: a choice over every label, each
    as the task describes it."""
    return {"type": "choice", "instructions": task.description, "criteria": task.criteria()}


def records(task: Task) -> dict[str, list[dict]]:
    question = direct_question(task)

    def record(example) -> dict:
        return {
            "state": example.state,
            "questions": {"label": question},
            "labels": {"label": example.label},
        }

    return {
        "train": [record(e) for e in [*task.train, *task.validation] if e.label is not None],
        "holdout": [record(e) for e in task.test if e.label is not None],
    }


def taught(task: Task, teacher, model: str) -> dict[str, list[dict]]:
    """The task's training and validation inputs, labeled or not, each asked the direct question
    of a teacher, whose probabilities become the record's soft target: the student learns the
    teacher's judgment, its doubt included. A gold label stays the record's label, so the
    trainer's validation measures the student against the truth; an unlabeled input takes the
    teacher's choice. The holdout is the test split, as for tuning."""
    question = direct_question(task)
    asked = Choice.model_validate(question)
    found = []
    for example in [*task.train, *task.validation]:
        response = teacher.system_one(
            state=example.state, questions={"label": question}, model=model
        )
        answer = read_answer(asked, response.answers["label"])
        found.append(
            {
                "state": example.state,
                "questions": {"label": question},
                "labels": {"label": example.label or answer["value"]},
                "references": {"label": {"target": answer["probabilities"]}},
            }
        )
    return {"train": found, "holdout": records(task)["holdout"]}


def export(task: Task, directory: Path, found: dict[str, list[dict]] | None = None) -> dict:
    """Write the task's records, or the `found` ones, under `directory` and say what was
    written. Nothing is overwritten: a tuning set is an input to a run worth keeping."""
    paths = {name: directory / f"{name}.jsonl" for name in FILES}
    taken = [str(path) for path in paths.values() if path.exists()]
    if taken:
        raise FileExistsError(f"already exists: {', '.join(taken)}")
    found = found or records(task)
    directory.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        path.write_text("".join(json.dumps(row) + "\n" for row in found[name]))
    return {name: {"path": str(paths[name]), "examples": len(found[name])} for name in FILES}


def trainer(ruling_dir: Path) -> list[str] | None:
    """How to invoke Ruling: from its checkout, through its own environment when it has one and
    through uv otherwise; or, with no checkout there, the `ruling` command an install from PyPI
    puts on the path. None when there is neither."""
    executable = ruling_dir / ".venv" / "bin" / "ruling"
    if executable.exists():
        return [str(executable)]
    if (ruling_dir / "pyproject.toml").exists():
        return ["uv", "run", "--directory", str(ruling_dir), "ruling"]
    installed = shutil.which("ruling")
    return [installed] if installed else None


def tune(
    task: Task,
    directory: Path,
    ruling_dir: Path,
    base: str,
    extra: list[str],
    found: dict[str, list[dict]] | None = None,
) -> dict:
    """Export the task's records, or the `found` ones a teacher wrote, and train an adapter on
    them with Ruling, under `directory`. The recipe comes first and `extra` after it, so
    anything passed through overrides it. The report says where the adapter is and how to
    serve it."""
    report = export(task, directory, found)
    command = trainer(ruling_dir)
    if command is None:
        raise FileNotFoundError(
            f"no Ruling checkout at {ruling_dir} and no `ruling` command on the path "
            f"(`pip install ruling`); the records are written under {directory}, "
            f"and `ruling train --data {directory / 'train.jsonl'} --holdout "
            f"{directory / 'holdout.jsonl'} --output {directory / 'adapter'}` trains on them"
        )
    adapter = directory / "adapter"
    subprocess.run(
        [
            *command,
            "train",
            "--data",
            report["train"]["path"],
            "--holdout",
            report["holdout"]["path"],
            "--model",
            base,
            *RECIPE,
            *extra,
            "--output",
            str(adapter),
        ],
        check=True,
    )
    return report | {
        "adapter": str(adapter),
        "serve_command": f"RULING_MODEL={base} RULING_ADAPTER={adapter} {' '.join(command)} serve",
    }
