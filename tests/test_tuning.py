"""Exporting a task's examples for tuning the judge, in the shape Ruling's trainer reads."""

import json
import pathlib

import pytest
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.graph import Task
from rulesmith.tuning import export, records


def labeled_task() -> Task:
    data = task_data()
    data["train"] = [
        {"id": f"t{i}", "state": f"request {i}", "label": "accept" if i % 2 else "review"}
        for i in range(6)
    ]
    data["validation"] = [{"id": "v0", "state": "request v", "label": "review"}]
    data["test"] = [{"id": f"h{i}", "state": f"held {i}", "label": "accept"} for i in range(3)]
    return Task.model_validate(data)


def test_records_ask_the_direct_question_with_the_label_and_keep_the_test_split_apart():
    # The adapter is trained on exactly the examples search sees, each asked the direct
    # question with the right label, and the holdout file holds every test example so the
    # trainer can refuse any overlap: a judge tuned on a test row would void the whole setting.
    task = labeled_task()
    found = records(task)
    assert len(found["train"]) == 7 and len(found["holdout"]) == 3
    question = found["train"][0]["questions"]["label"]
    assert question == {
        "type": "choice",
        "instructions": task.description,
        "criteria": task.criteria(),
    }
    assert [r["labels"]["label"] for r in found["train"]] == [
        e.label for e in [*task.train, *task.validation]
    ]
    states = {json.dumps(r["state"], sort_keys=True) for r in found["train"]}
    assert not states & {json.dumps(r["state"], sort_keys=True) for r in found["holdout"]}


def test_export_writes_both_files_once(tmp_path):
    report = export(labeled_task(), tmp_path / "tuning")
    lines = (tmp_path / "tuning" / "train.jsonl").read_text().splitlines()
    assert len([json.loads(line) for line in lines]) == report["train"]["examples"] == 7
    assert report["holdout"]["examples"] == 3
    with pytest.raises(FileExistsError, match="already exists"):
        export(labeled_task(), tmp_path / "tuning")


def fake_ruling(tmp_path) -> pathlib.Path:
    """A Ruling checkout whose trainer records what it was asked and writes an adapter, so the
    whole hand-off is exercised without a model."""
    checkout = tmp_path / "ruling"
    (checkout / ".venv" / "bin").mkdir(parents=True)
    script = checkout / ".venv" / "bin" / "ruling"
    script.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$(dirname "$0")/../../argv.txt"\n'
        'for i in "$@"; do [ "$prev" = "--output" ] && out="$i"; prev="$i"; done\n'
        'mkdir -p "$out" && echo fake > "$out/adapters.safetensors"\n'
    )
    script.chmod(0o755)
    return checkout


def test_tune_runs_ruling_on_the_records_and_says_how_to_serve_the_adapter(tmp_path):
    checkout = fake_ruling(tmp_path)
    task = write(tmp_path / "task.json", labeled_task().model_dump(mode="json"))
    out = tmp_path / "tuning"
    result = invoke("tune", task, "--output", out, "--ruling-dir", checkout, "--", "--steps", "50")
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    argv = (checkout / "argv.txt").read_text().splitlines()
    assert argv[0] == "train"
    assert argv[argv.index("--data") + 1] == str(out / "train.jsonl")
    assert argv[argv.index("--holdout") + 1] == str(out / "holdout.jsonl")
    # The recipe is there, and the passed-through argument comes after it, so it wins.
    assert "--eval-every" in argv and argv.index("--steps", argv.index("--steps") + 1) > 0
    assert argv[-2:] == ["--output", str(out / "adapter")]
    assert (out / "adapter" / "adapters.safetensors").exists()
    assert printed["adapter"] == str(out / "adapter")
    assert "RULING_ADAPTER=" in printed["serve_command"] and " serve" in printed["serve_command"]


def test_tune_refuses_a_game_an_existing_output_and_a_missing_checkout(tmp_path, monkeypatch):
    checkout = fake_ruling(tmp_path)
    # With `ruling` on the path a missing checkout is no longer an error, so keep it off here.
    monkeypatch.setenv("PATH", str(tmp_path / "empty-path"))
    task = write(tmp_path / "task.json", labeled_task().model_dump(mode="json"))
    doom = write(tmp_path / "doom.json", {"scenario": "basic", "labels": ["ATTACK"]})
    assert (
        invoke("tune", doom, "--output", tmp_path / "a", "--ruling-dir", checkout).returncode != 0
    )
    missing = invoke("tune", task, "--output", tmp_path / "b", "--ruling-dir", tmp_path / "nowhere")
    assert missing.returncode != 0 and "no Ruling checkout" in missing.stderr
    # The records were still written, and running again on the same output is refused.
    assert (tmp_path / "b" / "train.jsonl").exists()
    again = invoke("tune", task, "--output", tmp_path / "b", "--ruling-dir", checkout)
    assert again.returncode != 0 and "already exists" in again.stderr


def teach(provider, sure: float = 0.8):
    """A teacher that leans `accept` as sure as told, whatever it is asked about."""
    provider.answers = {
        "label": {
            "type": "choice",
            "choice": "accept",
            "probabilities": {"accept": sure, "review": 1 - sure},
            "confidence": sure,
        }
    }


def test_distilled_records_carry_the_teachers_probabilities_and_teach_from_unlabeled_inputs(
    provider, client
):
    from rulesmith.tuning import taught

    teach(provider)
    task = labeled_task().model_copy(deep=True)
    task.train[0].label = None
    found = taught(task, client, "teacher")
    assert len(found["train"]) == 7 and len(found["holdout"]) == 3
    targets = [r["references"]["label"]["target"] for r in found["train"]]
    assert all(t == {"accept": 0.8, "review": pytest.approx(0.2)} for t in targets)
    # A gold label stays the label, so validation measures the student against the truth;
    # an unlabeled input takes the teacher's answer.
    assert found["train"][0]["labels"]["label"] == "accept"
    assert found["train"][1]["labels"]["label"] == task.train[1].label


RULING = pathlib.Path(__file__).resolve().parents[2] / "ruling"


@pytest.mark.skipif(not (RULING / "pyproject.toml").exists(), reason="no Ruling checkout beside")
def test_ruling_reads_a_distilled_record_as_the_teachers_probabilities(tmp_path, provider, client):
    import subprocess

    from rulesmith.tuning import taught

    teach(provider, 0.7)
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(taught(labeled_task(), client, "teacher")["train"][0]) + "\n")
    script = (
        "import sys, json\n"
        "from ruling.evals import load_dataset\n"
        "from ruling.train import target_vector\n"
        "(record,) = load_dataset(__import__('pathlib').Path(sys.argv[1]))\n"
        "print(json.dumps(target_vector(record, 'label')))\n"
    )
    read = subprocess.run(
        ["uv", "run", "--directory", str(RULING), "python", "-c", script, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(read.stdout.strip().splitlines()[-1]) == {
        "accept": 0.7,
        "review": pytest.approx(0.3),
    }


def test_distill_asks_the_teacher_and_hands_ruling_the_soft_targets(tmp_path, provider):
    teach(provider)
    checkout = fake_ruling(tmp_path)
    task = write(tmp_path / "task.json", labeled_task().model_dump(mode="json"))
    out = tmp_path / "distilled"
    result = invoke(
        "distill",
        task,
        "--output",
        out,
        "--ruling-dir",
        checkout,
        "--teacher",
        f"ruling:big-model@{provider.url}",
    )
    assert result.returncode == 0, result.stderr
    asked = [body for path, body in provider.requests if path == "/v1/systemone"]
    assert len(asked) == 7 and {body["model"] for body in asked} == {"big-model"}
    argv = (checkout / "argv.txt").read_text().splitlines()
    assert argv[0] == "train" and argv[argv.index("--data") + 1] == str(out / "train.jsonl")
    first = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert first["references"]["label"]["target"]["accept"] == 0.8
    assert json.loads(result.stdout)["teacher"] == "big-model"


def test_arguments_after_a_double_dash_are_refused_where_nothing_takes_them(tmp_path):
    # Only tune and distill hand them on; anywhere else they would be dropped without a word.
    task = write(tmp_path / "task.json", labeled_task().model_dump(mode="json"))
    result = invoke("validate", task, tmp_path / "graph.rules", "--", "--steps", "50")
    assert result.returncode == 2 and "after --" in result.stderr


def test_an_installed_ruling_trains_when_there_is_no_checkout(tmp_path, monkeypatch):
    """`pip install ruling` puts a `ruling` command on the path; a checkout is then optional."""
    import shutil

    from rulesmith.tuning import trainer

    nowhere = tmp_path / "nowhere"
    monkeypatch.setattr(shutil, "which", lambda name: None)
    assert trainer(nowhere) is None
    monkeypatch.setattr(
        shutil, "which", lambda name: "/opt/bin/ruling" if name == "ruling" else None
    )
    assert trainer(nowhere) == ["/opt/bin/ruling"]
    # A checkout with its own environment still comes first.
    checkout = fake_ruling(tmp_path)
    assert trainer(checkout) == [str(checkout / ".venv" / "bin" / "ruling")]
