"""Opt-in integration with a running Ruling server and its actual local model."""

import json
import os
import subprocess
import sys
from pathlib import Path

import dspy
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from rulesmith.graph import Plan, Task
from rulesmith.optimize import GraphProposer, SearchConfig, optimize_plan
from rulesmith.runtime import DecisionProgram

pytestmark = pytest.mark.ruling
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ruling_client(request):
    with TypeSafeClient(
        base_url=request.config.getoption("--ruling-url"),
        api_key=os.environ.get("RULING_API_KEY") or "local-no-auth",
        retry=RetryPolicy(max_retries=0),
    ) as client:
        yield client


@pytest.fixture
def ruling_model(ruling_client):
    """The server's primary model, first in a list that may also name adapters and other bases."""
    models = ruling_client.models.list().models
    assert models
    return models[0].name


@pytest.fixture
def support_task():
    return Task.model_validate_json((ROOT / "examples/support-routing/task.json").read_text())


def assert_routed_assessment(result, plan, model):
    department = result["outputs"]["label"]
    assert department in plan.nodes["department"].question.criteria
    selected = f"{department}_review"
    assert set(result["answers"]) == {"department", "urgency", selected}
    assert set(result["calls"]) == set(result["answers"])
    assert set(result["skipped"]) == {
        f"{other}_review"
        for other in plan.nodes["department"].question.criteria
        if other != department
    }
    assessment = result["outputs"]["assessment"]
    assert assessment["department"] == department
    assert assessment["urgency"] == result["answers"]["urgency"]["value"]
    assert assessment["specialist_confidence"] == result["answers"][selected]["value"]
    assert 0 <= assessment["specialist_confidence"] <= 1
    assert 0 <= assessment["urgency"] <= len(plan.nodes["urgency"].question.criteria) - 1
    assert result["models"] == [model]
    assert result["input_tokens"] > 0
    assert result["output_tokens"] == 0


def test_ruling_executes_dependent_calls_conditional_branches_and_joins(
    ruling_client, ruling_model, support_task
):
    plan = Plan.model_validate_json((ROOT / "examples/support-routing/dag.json").read_text())
    state = support_task.train[0].state
    result = DecisionProgram(plan, ruling_client, ruling_model)(state=state)
    assert_routed_assessment(dict(result), plan, ruling_model)
    assert result.node_values["context"] == {
        "ticket": state,
        "department": result.label,
        "urgency": result.answers["urgency"],
    }


def test_ruling_cli_runs_without_a_hosted_provider_credential(
    request, ruling_model, support_task, tmp_path
):
    state = tmp_path / "state.json"
    state.write_text(json.dumps(support_task.train[0].state))
    environment = dict(os.environ)
    environment.pop("TYPESAFE_API_KEY", None)
    # Prevent DSPy's transitive dotenv support from restoring a hosted credential.
    environment["PYTHON_DOTENV_DISABLED"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rulesmith.cli",
            "run",
            str(ROOT / "examples/support-routing/dag.json"),
            str(state),
            "--backend",
            "ruling",
            "--base-url",
            request.config.getoption("--ruling-url"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    plan = Plan.model_validate_json((ROOT / "examples/support-routing/dag.json").read_text())
    assert_routed_assessment(json.loads(result.stdout), plan, ruling_model)


def test_real_gepa_evaluates_a_fixed_graph_proposal_with_ruling(
    ruling_client, ruling_model, support_task
):
    """Exercise optimizer/model wiring; a fixed proposer makes no quality claim."""
    task = support_task.model_copy(
        update={"train": support_task.train[:1], "validation": support_task.validation[:1]}
    )
    initial = Plan.model_validate(
        {"nodes": {}, "outputs": {"label": {"type": "literal", "value": "sales"}}}
    )
    proposed = Plan.model_validate_json((ROOT / "examples/support-routing/dag.json").read_text())
    reflections = []

    def propose(prompt):
        reflections.append(prompt)
        return proposed.model_dump_json()

    best, report = optimize_plan(
        task,
        initial,
        ruling_client,
        ruling_model,
        propose,
        SearchConfig(max_metric_calls=4, reflection_batch=1),
    )
    best.check(task.labels, max_nodes=16, max_depth=8)
    assert reflections
    assert report["resolved_models"] == [ruling_model]
    assert report["seed_validation_score"] == 0
    assert 0 <= report["validation_score"] <= 1
    assert all(example.state not in json.dumps(reflections) for example in task.test)


def test_dspy_generates_a_graph_that_gepa_evaluates_on_ruling(
    request,
    ruling_client,
    ruling_model,
    support_task,
):
    model = request.config.getoption("--ruling-reflection-model")
    endpoint = request.config.getoption("--ruling-reflection-url")
    if model is None and endpoint is None:
        pytest.skip("real graph generation requires explicit reflection model and URL")
    if not model or not endpoint:
        pytest.fail("provide both --ruling-reflection-model and --ruling-reflection-url")
    lm = dspy.LM(
        model,
        api_base=endpoint,
        api_key="local-no-auth",
        temperature=1,
        max_tokens=4096,
        cache=False,
        num_retries=0,
    )
    generations = []
    proposer = GraphProposer(lm)

    def reflect(prompt):
        generated = proposer(prompt)
        generations.append(generated)
        return generated

    initial = Plan.model_validate(
        {"nodes": {}, "outputs": {"label": {"type": "literal", "value": "sales"}}}
    )
    best, report = optimize_plan(
        support_task,
        initial,
        ruling_client,
        ruling_model,
        reflect,
        SearchConfig(max_metric_calls=24),
    )
    best.check(support_task.labels, max_nodes=16, max_depth=8)
    assert generations, "DSPy did not generate any graph proposals"
    assert report["resolved_models"] == [ruling_model], "no generated graph called Ruling"
    assert report["candidates"] >= 2, "no generated graph reached validation evaluation"
    assert report["validation_score"] >= report["seed_validation_score"]
