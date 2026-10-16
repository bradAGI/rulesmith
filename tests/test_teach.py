"""Teaching the judge: train labels become outcomes on Ruling's ledger; nothing else does."""

import json
from pathlib import Path

import pytest
from typesafe_sdk import ChoiceAnswer, NoulAnswer, RetryPolicy, ScoreAnswer

from rulesmith.graph import Plan, Task
from rulesmith.optimize import SearchConfig, optimize_plan
from rulesmith.runtime import DecisionProgram
from rulesmith.teach import RulingClient, RulingError, Teaching

ROOT = Path(__file__).resolve().parents[1]
ANSWERS = {
    "department": {
        "type": "choice",
        "choice": "billing",
        "probabilities": {"billing": 0.8, "technical": 0.1, "sales": 0.1},
        "confidence": 0.7,
    },
    "urgency": {
        "type": "score",
        "score": 1.2,
        "legend": {"0": "Can wait", "1": "This week", "2": "Today"},
        "probabilities": {"0": 0.2, "1": 0.4, "2": 0.4},
        "confidence": 0.3,
    },
    "refund": {"type": "noul", "noul": 0.9},
}
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": None, "technical": None, "sales": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent?",
        "criteria": ["Can wait", "This week", "Today"],
    },
    "refund": {"type": "noul", "instructions": "The customer wants a refund."},
}


@pytest.fixture
def task():
    return Task.model_validate_json((ROOT / "examples/support-routing/task.json").read_text())


def example_in(asked, task: Task) -> str:
    """Which example a judge request is about, as `split:index`, from the state it holds."""
    text = json.dumps(asked)
    splits = {"train": task.train, "validation": task.validation, "test": task.test}
    for split, examples in splits.items():
        for index, example in enumerate(examples):
            if json.dumps(example.state) in text:
                return f"{split}:{index}"
    return "unknown"


@pytest.fixture
def ledger(provider, task):
    """A Ruling server with a ledger, whose decision ids say which example each was about."""
    provider.answers = ANSWERS

    def identify(request):
        provider.body = {**provider.body, "id": example_in(request["state"], task)}

    provider.on_request = identify
    return provider


def outcomes(provider):
    return [(path, request) for path, request in provider.requests if "/outcome" in path]


def client(provider, key="s3cret"):
    return RulingClient(
        base_url=provider.url, api_key=key, timeout=5, retry=RetryPolicy(max_retries=0)
    )


def test_the_client_keeps_the_decision_id_and_reads_every_answer_type(ledger, task):
    with client(ledger) as judge:
        decision = judge.system_one(state=task.train[0].state, questions=QUESTIONS, model="m")
        judge.outcome(decision.decision, {"department": "billing"})
    assert decision.decision == "train:0"
    assert isinstance(decision.answers["department"], ChoiceAnswer)
    assert isinstance(decision.answers["urgency"], ScoreAnswer)
    assert decision.answers["urgency"].probabilities == {0: 0.2, 1: 0.4, 2: 0.4}
    assert isinstance(decision.answers["refund"], NoulAnswer)
    assert decision.usage.input_tokens == 12
    assert outcomes(ledger) == [
        ("/v1/decisions/train:0/outcome", {"labels": {"department": "billing"}})
    ]
    assert all(headers["Authorization"] == "Bearer s3cret" for headers in ledger.headers)


def test_a_server_without_a_ledger_gives_no_id_and_is_never_taught(provider, task):
    provider.answers = ANSWERS
    with Teaching(client(provider), task) as judge:
        response = judge.system_one(state=task.train[0].state, questions=QUESTIONS, model="m")
    assert response.answers["department"].choice == "billing"
    assert outcomes(provider) == [] and judge.taught == 0


def test_a_refused_request_raises_with_the_servers_reason(provider):
    provider.status = 422
    provider.body = {"error": {"code": "invalid_options", "message": "a choice needs 2 options"}}
    with client(provider) as judge, pytest.raises(RulingError, match="422.*invalid_options"):
        judge.system_one(state="x", questions=QUESTIONS, model="m")


def test_only_label_questions_about_train_examples_are_taught(ledger, task):
    with Teaching(client(ledger), task) as judge:
        for example in (task.train[0], task.validation[0], task.test[0]):
            # A graph hands the judge its call's inputs, the state among them, sometimes nested.
            state = example.state
            for asked in (state, {"ticket": state}, {"context": {"ticket": state, "n": 1}}):
                judge.system_one(state=asked, questions=QUESTIONS, model="m")
    label = task.train[0].label
    lesson = ("/v1/decisions/train:0/outcome", {"labels": {"department": label}})
    assert outcomes(ledger) == [lesson] * 3
    assert judge.taught == 3


def test_an_example_with_several_right_labels_teaches_nothing(ledger, task):
    ambiguous = task.model_copy(
        update={"train": [task.train[0].model_copy(update={"accepted": ["sales"]})]}
    )
    with Teaching(client(ledger), ambiguous) as judge:
        judge.system_one(state=ambiguous.train[0].state, questions=QUESTIONS, model="m")
    assert outcomes(ledger) == []


def test_a_choice_whose_options_are_not_the_tasks_labels_is_not_taught(ledger, task):
    questions = {
        "mood": {"type": "choice", "instructions": "Tone?", "criteria": {"calm": 1, "angry": 2}},
        "department": QUESTIONS["department"],
    }
    mood = {"type": "choice", "choice": "calm", "probabilities": {"calm": 0.6, "angry": 0.4}}
    ledger.answers = {**ANSWERS, "mood": {**mood, "confidence": 0.2}}
    with Teaching(client(ledger), task) as judge:
        judge.system_one(state=task.train[0].state, questions=questions, model="m")
    [(_, request)] = outcomes(ledger)
    assert request == {"labels": {"department": task.train[0].label}}


def answering_anything(provider, task):
    """Answer in the shape of whatever the graph asks, and name the example the state holds."""

    def reply(request):
        answers = {}
        for name, question in request["questions"].items():
            if question["type"] == "choice":
                keys = list(question["criteria"])
                answers[name] = {
                    "type": "choice",
                    "choice": keys[0],
                    "probabilities": {k: 1 / len(keys) for k in keys},
                    "confidence": 0.0,
                }
            elif question["type"] == "score":
                levels = question["criteria"]
                answers[name] = {
                    "type": "score",
                    "score": 0.0,
                    "legend": {str(i): level for i, level in enumerate(levels)},
                    "probabilities": {str(i): 1 / len(levels) for i in range(len(levels))},
                    "confidence": 0.0,
                }
            else:
                answers[name] = {"type": "noul", "noul": 0.5}
        provider.answers = answers
        provider.body = {**provider.body, "id": example_in(request["state"], task)}

    provider.on_request = reply


def test_a_search_teaches_train_labels_and_never_a_validation_or_test_one(ledger, task):
    """The whole search, scoring train batches and the validation set alike, posts outcomes for
    train examples only."""
    proposed = Plan.model_validate_json((ROOT / "examples/support-routing/dag.json").read_text())
    initial = Plan.model_validate(
        {"nodes": {}, "outputs": {"label": {"type": "literal", "value": "sales"}}}
    )
    answering_anything(ledger, task)
    with Teaching(client(ledger), task) as judge:
        optimize_plan(
            task,
            initial,
            judge,
            "m",
            lambda prompt: proposed.model_dump_json(),
            SearchConfig(max_metric_calls=12, reflection_batch=2),
        )
        assert judge.taught > 0
        taught = {path.removeprefix("/v1/decisions/").split(":")[0] for path, _ in outcomes(ledger)}
        assert taught == {"train"}
        # The program saw the judge as any other client: answers, model and usage unchanged.
        result = DecisionProgram(proposed, judge, "m")(state=task.train[0].state)
    assert result.models == ["fixture-model"] and result.input_tokens > 0
