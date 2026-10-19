import json
import re
import threading
from typing import get_args

import dspy
import pytest
from conftest import chat_reply
from test_graph import binary_plan, call, constant_plan, ref, task_data

from rulesmith.extract import Extractor
from rulesmith.graph import Call, Condition, Node, Plan, Question, Source, Task
from rulesmith.optimize import (
    GraphAdapter,
    GraphProposer,
    Labels,
    SearchConfig,
    assemble,
    components,
    evaluate_plan,
    improvement_p,
    optimize_plan,
)
from rulesmith.rules import compile_rules, parts
from rulesmith.runtime import DecisionProgram, Models, ResponseError


def scripted(plan):
    """A proposer that answers with whichever part of `plan` the prompt asked it to revise."""
    written = components(plan)

    def reflect(prompt):
        shown = prompt if isinstance(prompt, str) else json.dumps(prompt)
        part = shown.rsplit("Return the part named ", 1)[-1].split(".", 1)[0]
        if part == "readings":
            return written["readings"]
        # A rules part may carry the readings it names, and a rewrite needing new ones must.
        return "\n".join(text for text in (written["readings"], written["rules"]) if text)

    return reflect


def test_official_sdk_http_dspy_and_dag_execute_together(provider, client):
    plan = Plan.model_validate(binary_plan())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    result = DecisionProgram(plan, client, "fixture-model")(state={"request": "Example"})
    assert result.label == "accept"
    assert result.outputs == {"label": "accept"}
    assert result.models == ["fixture-model"] and result.input_tokens == 12
    assert provider.requests == [
        (
            "/v1/systemone",
            {
                "state": {"state": {"request": "Example"}},
                "model": "fixture-model",
                "questions": {"eligible": plan.nodes["eligible"].question.model_dump()},
            },
        )
    ]


def test_provider_contract_failure_does_not_become_a_bad_candidate(provider, client):
    task = Task.model_validate(task_data())
    adapter = GraphAdapter(task, client, "fixture-model", SearchConfig())
    with pytest.raises(ResponseError, match="IDs"):
        adapter.evaluate(
            task.train, {"graph": Plan.model_validate(binary_plan()).model_dump_json()}
        )


def test_invalid_candidate_produces_actionable_feedback_without_http(provider, client):
    task = Task.model_validate(task_data())
    adapter = GraphAdapter(task, client, "fixture-model", SearchConfig())
    batch = adapter.evaluate(task.train, {"graph": "{}"}, capture_traces=True)
    assert batch.scores == [0.0]
    assert batch.outputs == [None]
    feedback = adapter.make_reflective_dataset({"graph": "{}"}, batch, ["graph"])
    assert "validation" in feedback["graph"][0]["Feedback"].lower()
    assert provider.requests == []


def test_real_gepa_evolves_structure_through_the_official_sdk(provider, client):
    data = task_data()
    data["validation"][0]["label"] = "accept"
    task = Task.model_validate(data)
    initial = Plan.model_validate(constant_plan())
    proposal = binary_plan()
    proposal["nodes"].update(left=call(), right=call())
    proposal["nodes"]["eligible"]["inputs"].update(left=ref("left"), right=ref("right"))
    proposed = Plan.model_validate(proposal)
    provider.answers = {key: {"type": "noul", "noul": 0.9} for key in ("left", "right", "eligible")}
    reflections = []

    def scripted_reflection(prompt):
        # A fixed proposal isolates optimizer wiring from paid model quality.
        reflections.append(prompt)
        return proposed.model_dump_json()

    plan, report = optimize_plan(
        task,
        initial,
        client,
        "fixture-model",
        scripted_reflection,
        SearchConfig(max_metric_calls=4, reflection_batch=1),
    )
    # A whole-graph answer is the program entire, so one proposal can settle a round.
    assert reflections
    assert parts(plan) == parts(proposed)
    assert report["validation_score"] == 1.0
    assert report["seed_validation_score"] == 0.0
    assert report["candidates"] >= 2
    # Per example, left and right share inputs and go out as one request before eligible.
    assert len(provider.requests) == 4
    assert report["resolved_models"] == ["fixture-model"]
    assert len(report["task_sha256"]) == 64
    assert len(report["seed_sha256"]) == 64
    assert set(report["versions"]) == {"dspy", "gepa", "typesafe-sdk"}
    assert {body["state"]["state"] for _, body in provider.requests} == {"request A", "request B"}
    assert "request C" not in json.dumps(reflections)


@pytest.mark.parametrize("failure", ["schema", "execution"])
def test_real_gepa_carries_rejected_graph_feedback_into_next_reflection(provider, client, failure):
    data = task_data()
    data["validation"][0]["label"] = "accept"
    task = Task.model_validate(data)
    initial = Plan.model_validate(constant_plan())
    proposed = Plan.model_validate(binary_plan())
    answer = scripted(proposed)
    graph = json.loads(proposed.model_dump_json(exclude_none=True))
    # A part carries the nodes its rules need, so the broken answer is a whole policy part.
    broken = {"policy": None}
    policy = {"nodes": graph["nodes"], "outputs": graph["outputs"]}
    if failure == "schema":
        policy["outputs"] = {"label": {"type": "node", "node": "missing"}}
        expected_error = "unknown nodes"
    else:
        policy["nodes"]["decision"]["when"] = {
            "type": "compare",
            "node": "eligible",
            "op": "lt",
            "value": 0.5,
        }
        expected_error = "required graph outputs were skipped"
    broken["policy"] = json.dumps(policy)
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    answer, reflections, offered = scripted(proposed), [], []

    def reflect(prompt):
        shown = prompt if isinstance(prompt, str) else json.dumps(prompt)
        reflections.append(shown)
        part = shown.rsplit("Return the part named ", 1)[-1].split(".", 1)[0]
        if not offered and part == "rules":
            offered.append(json.dumps(policy))
            return offered[0]
        return answer(prompt)

    optimize_plan(
        task,
        initial,
        client,
        "fixture-model",
        reflect,
        SearchConfig(max_metric_calls=12, reflection_batch=1, repair_attempts=0),
    )
    # Search saw the broken graph fail, and said why on a later round.
    assert offered, "the broken part was never proposed"
    assert any(expected_error in shown for shown in reflections[1:]), expected_error
    assert "request C" not in json.dumps(reflections)


def test_validation_execution_failures_do_not_enter_training_reflection_feedback(client):
    data = constant_plan()
    data["nodes"] = {
        "gate": {"type": "value", "value": 0},
        "label": {
            "type": "value",
            "value": "accept",
            "when": {"type": "compare", "node": "gate", "op": "ge", "value": 1},
        },
    }
    data["outputs"]["label"] = ref("label")
    adapter = GraphAdapter(Task.model_validate(task_data()), client, "fixture", SearchConfig())
    batch = adapter.evaluate(
        adapter.task.validation, {"graph": json.dumps(data)}, capture_traces=False
    )
    assert batch.scores == [0.0]
    assert adapter.proposal_feedback is None


def test_real_gepa_keeps_seed_when_train_improvement_regresses_validation(client):
    task = Task.model_validate(task_data())
    initial = Plan.model_validate(constant_plan())
    proposed = Plan.model_validate(constant_plan("accept"))
    best, report = optimize_plan(
        task,
        initial,
        client,
        "fixture-model",
        lambda prompt: proposed.model_dump_json(),
        SearchConfig(max_metric_calls=4, reflection_batch=1),
    )
    assert best == initial
    assert report["validation_scores"] == [1.0, 0.0]
    assert report["validation_score"] == report["seed_validation_score"] == 1


def test_constant_program_never_calls_provider(provider, client):
    plan = Plan.model_validate(constant_plan())
    assert DecisionProgram(plan, client, "fixture-model")(state="Example").label == "review"
    assert provider.requests == []


def test_assembled_literal_data_reaches_the_provider_unchanged(provider, client):
    payload = {"node": "spare", "source": "spare", "sources": 7}
    graph = binary_plan()
    graph["nodes"]["spare"] = {"type": "value", "value": "unused"}
    graph["nodes"]["eligible"]["inputs"] = {"data": {"type": "literal", "value": payload}}
    graph["outputs"]["data"] = {"type": "literal", "value": payload}
    provider.answers = {"eligible": {"type": "noul", "noul": 0.9}}
    plan = assemble({"graph": json.dumps(graph)})
    result = DecisionProgram(plan, client, "fixture-model")(state="Example")
    assert "spare" not in plan.nodes
    assert result.outputs == {"label": "accept", "data": payload}
    assert provider.requests[0][1]["state"] == {"data": payload}


def test_reflection_outage_is_propagated_and_stops_search(client):
    task = Task.model_validate(task_data())
    initial = Plan.model_validate(constant_plan())
    calls = []

    def unavailable(prompt):
        calls.append(prompt)
        raise RuntimeError("reflection provider unavailable")

    with pytest.raises(RuntimeError, match="reflection provider unavailable"):
        optimize_plan(
            task,
            initial,
            client,
            "fixture-model",
            unavailable,
            SearchConfig(max_metric_calls=4, reflection_batch=1),
        )
    assert len(calls) == 1


def test_calls_sharing_inputs_are_asked_in_one_sdk_request(provider, client, mixed_plan):
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "urgency": {
            "type": "score",
            "score": 0.9,
            "confidence": 0.8,
            "probabilities": {"0": 0.1, "1": 0.9},
            "legend": {"0": "Routine request", "1": "Immediate assistance needed"},
        },
        "refund": {"type": "noul", "noul": 0.9},
        "second": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.7,
            "probabilities": {"billing": 0.8, "technical": 0.2},
        },
    }
    prediction = DecisionProgram(mixed_plan, client, "fixture-model")(
        state="Refund this charge now."
    )
    assert prediction.label == "accept"
    assert set(prediction.answers) == {"team", "urgency", "refund"}
    ((_, request),) = provider.requests
    assert request["questions"] == {
        key: mixed_plan.nodes[key].question.model_dump() for key in ("team", "urgency", "refund")
    }
    # One request is billed once, however many questions it carries.
    assert prediction.input_tokens == 12


def test_calls_with_different_inputs_stay_separate_requests(provider, client):
    plan = binary_plan()
    plan["nodes"]["other"] = call()
    plan["nodes"]["other"]["inputs"] = {"request": {"type": "literal", "value": "Other"}}
    plan["outputs"]["other"] = ref("other")
    provider.answers = {key: {"type": "noul", "noul": 0.9} for key in ("eligible", "other")}
    DecisionProgram(Plan.model_validate(plan), client, "fixture-model")(state="Example")
    assert sorted(list(request["questions"]) for _, request in provider.requests) == [
        ["eligible"],
        ["other"],
    ]


def test_diamond_routes_values_and_answer_records_and_runs_shared_nodes_once(provider, client):
    from threading import Barrier

    from rulesmith.runtime import ExecutionConfig

    plan = Plan.model_validate(
        {
            "nodes": {
                "root": call(),
                # Distinct inputs keep the siblings separate requests, so they must run in parallel.
                "left": call(parent=ref("root"), side={"type": "literal", "value": "left"}),
                "right": call(parent=ref("root"), side={"type": "literal", "value": "right"}),
                "join": call(
                    left=ref("left"),
                    right=ref("right"),
                    evidence=ref("root", view="answer"),
                    original={"type": "state"},
                ),
                "independent": call(other={"type": "literal", "value": "unrelated"}),
            },
            "outputs": {"joined": ref("join"), "other": ref("independent")},
        }
    )
    expected = {"root": 0.4, "left": 0.6, "right": 0.8, "join": 0.9, "independent": 0.1}
    provider.answers = {key: {"type": "noul", "noul": value} for key, value in expected.items()}
    siblings = Barrier(2)

    def synchronize(request):
        (key,) = request["questions"]
        if key in {"left", "right"}:
            # The timeout only bounds a deadlock if parallel execution regresses.
            siblings.wait(timeout=5)

    provider.on_request = synchronize
    result = DecisionProgram(plan, client, "fixture-model", ExecutionConfig(max_workers=2))(
        state={"ticket": "input"}
    )
    requests = {next(iter(body["questions"])): body for _, body in provider.requests}
    assert len(provider.requests) == len(plan.nodes)
    assert requests["left"]["state"] == {"parent": 0.4, "side": "left"}
    assert requests["right"]["state"] == {"parent": 0.4, "side": "right"}
    assert requests["join"]["state"] == {
        "left": 0.6,
        "right": 0.8,
        "evidence": {"type": "noul", "value": 0.4},
        "original": {"ticket": "input"},
    }
    assert result.outputs == {"joined": 0.9, "other": 0.1}
    assert result.node_values == expected
    assert result.input_tokens == 12 * len(plan.nodes)


@pytest.mark.parametrize("probability,active", [(0.2, "low"), (0.8, "high")])
def test_conditional_calls_skip_unused_branches_and_join_available_outputs(
    probability,
    active,
    provider,
    client,
):
    condition = {"type": "compare", "node": "gate", "op": "ge", "value": 0.5}
    plan = Plan.model_validate(
        {
            "nodes": {
                "gate": call(),
                "high": {
                    **call(),
                    "when": {
                        "type": "all",
                        "conditions": [condition, {"type": "present", "node": "gate"}],
                    },
                },
                "low": {
                    **call(),
                    "when": {
                        "type": "any",
                        "conditions": [
                            {"type": "not", "condition": condition},
                            {"type": "compare", "node": "gate", "op": "lt", "value": 0.1},
                        ],
                    },
                },
                "joined": {"type": "coalesce", "sources": ["high", "low"]},
                "final": call(
                    result=ref("joined"),
                    high=ref("high", optional=True),
                    low=ref("low", optional=True),
                ),
            },
            "outputs": {"result": ref("final")},
        }
    )
    provider.answers = {key: {"type": "noul", "noul": probability} for key in plan.nodes}
    result = DecisionProgram(plan, client, "fixture-model")(state="input")
    requests = {next(iter(body["questions"])): body for _, body in provider.requests}
    assert set(requests) == {"gate", active, "final"}
    assert requests["final"]["state"] == {"result": probability, active: probability}
    assert result.skipped == ["low" if active == "high" else "high"]


def test_scheduler_releases_dependents_without_waiting_for_unrelated_roots(provider, client):
    from threading import Event

    from rulesmith.runtime import ExecutionConfig

    plan = Plan.model_validate(
        {
            "nodes": {
                "slow": call(),
                "fast": call(root={"type": "literal", "value": "fast"}),
                "child": call(parent=ref("fast")),
            },
            "outputs": {"slow": ref("slow"), "child": ref("child")},
        }
    )
    provider.answers = {key: {"type": "noul", "noul": 0.6} for key in plan.nodes}
    child_started = Event()

    def coordinate(request):
        (key,) = request["questions"]
        if key == "child":
            child_started.set()
        elif key == "slow":
            assert child_started.wait(timeout=5), "independent root blocked a ready descendant"

    provider.on_request = coordinate
    result = DecisionProgram(plan, client, "fixture-model", ExecutionConfig(max_workers=2))(
        state="input"
    )
    assert result.outputs == {"slow": 0.6, "child": 0.6}
    assert len(provider.requests) == 3


def test_coalesce_uses_declared_priority_and_program_does_not_cache_previous_input(
    provider, client
):
    plan = Plan.model_validate(
        {
            "nodes": {
                "a": call(),
                "z": call(),
                "join": {"type": "coalesce", "sources": ["z", "a"]},
            },
            "outputs": {"value": ref("join")},
        }
    )
    program = DecisionProgram(plan, client, "fixture-model")
    for probability in (0.3, 0.7):
        provider.answers = {
            "a": {"type": "noul", "noul": 0.1},
            "z": {"type": "noul", "noul": probability},
        }
        assert program(state=str(probability)).outputs == {"value": probability}
    # a and z share inputs, so each run asks both in one request.
    assert len(provider.requests) == 2


def test_candidate_with_unavailable_label_gets_feedback_without_poisoning_search(provider, client):
    data = binary_plan()
    data["nodes"]["decision"]["when"] = {
        "type": "compare",
        "node": "eligible",
        "op": "ge",
        "value": 0.5,
    }
    plan = Plan.model_validate(data)
    task = Task.model_validate(task_data())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.1}}
    adapter = GraphAdapter(task, client, "fixture-model", SearchConfig())
    batch = adapter.evaluate(task.train, {"graph": plan.model_dump_json()}, capture_traces=True)
    assert batch.scores == [0.0]
    assert "label" in batch.trajectories[0]["Feedback"]
    assert adapter.failure is None
    trace = batch.trajectories[0]["Generated Outputs"]
    assert trace["node_values"]["eligible"] == 0.1
    assert trace["input_tokens"] == 12
    valid = adapter.evaluate(
        task.train, {"graph": Plan.model_validate(constant_plan("accept")).model_dump_json()}
    )
    assert valid.scores == [1.0]


def test_provider_error_prevents_dependent_calls(provider, client):
    plan = Plan.model_validate(
        {
            "nodes": {
                "root": call(),
                "dependent": call(parent=ref("root")),
            },
            "outputs": {"result": ref("dependent")},
        }
    )
    with pytest.raises(ResponseError, match="IDs"):
        DecisionProgram(plan, client, "fixture-model")(state="input")
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    "team,severity,label",
    [
        ("support", 2, "review"),
        ("billing", 2, "urgent"),
        ("billing", 1, "review"),
    ],
)
def test_switch_routes_to_a_conditional_score_and_rejoins(team, severity, label, provider, client):
    plan = Plan.model_validate(
        {
            "nodes": {
                "team": {
                    "type": "call",
                    "inputs": {"state": {"type": "state"}},
                    "question": {
                        "type": "choice",
                        "instructions": "Choose a team.",
                        "criteria": {"billing": "Payments", "support": "Other requests"},
                    },
                },
                "route": {
                    "type": "switch",
                    "source": "team",
                    "cases": {"billing": "assess", "support": "review"},
                },
                "severity": {
                    "type": "call",
                    "inputs": {"state": {"type": "state"}},
                    "question": {
                        "type": "score",
                        "instructions": "Assess severity.",
                        "criteria": ["Routine", "Disrupted", "Blocked"],
                    },
                    "when": {"type": "compare", "node": "route", "op": "eq", "value": "assess"},
                },
                "decision": {
                    "type": "threshold",
                    "source": "severity",
                    "cutoff": 1.5,
                    "below": "review",
                    "above": "urgent",
                },
                "fallback": {
                    "type": "value",
                    "value": "review",
                    "when": {"type": "compare", "node": "route", "op": "eq", "value": "review"},
                },
                "joined": {"type": "coalesce", "sources": ["decision", "fallback"]},
            },
            "outputs": {"label": ref("joined")},
        }
    )
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": team,
            "confidence": 1.0,
            "probabilities": {key: float(key == team) for key in ("billing", "support")},
        },
        "severity": {
            "type": "score",
            "score": severity,
            "confidence": 1.0,
            "probabilities": {str(i): float(i == severity) for i in range(3)},
            "legend": {
                str(i): level for i, level in enumerate(["Routine", "Disrupted", "Blocked"])
            },
        },
    }
    result = DecisionProgram(plan, client, "fixture-model")(state="input")
    assert result.label == label
    assert len(provider.requests) == (1 if team == "support" else 2)
    if team == "support":
        assert set(result.skipped) == {"severity", "decision"}


def schema_types(annotated):
    union = get_args(annotated)[0]
    return {member.model_fields["type"].default for member in get_args(union)}


def every_construct_plan():
    """Uses every node, question, source, and condition type the schema defines."""
    return {
        "nodes": {
            "team": {
                "type": "call",
                "question": {
                    "type": "choice",
                    "instructions": "Which team owns this request?",
                    "criteria": {"billing": "Charges", "technical": "Bugs"},
                },
                "inputs": {"ticket": {"type": "state"}},
            },
            "urgency": {
                "type": "call",
                "question": {
                    "type": "score",
                    "instructions": "How urgent is it?",
                    "criteria": ["Routine request", "Immediate assistance needed"],
                },
                "inputs": {"ticket": {"type": "state"}},
            },
            "refund": {
                "type": "call",
                "question": {"type": "noul", "instructions": "Is a refund requested?"},
                "inputs": {"ticket": {"type": "state"}, "team": ref("team")},
            },
            "route": {
                "type": "switch",
                "source": "team",
                "cases": {"billing": "accept", "technical": "review"},
            },
            "sure": {"type": "confidence", "source": "team"},
            "second": {
                "type": "call",
                "question": {
                    "type": "choice",
                    "instructions": "Reading it again, which team owns this request?",
                    "criteria": {"billing": "Charges", "technical": "Bugs"},
                },
                "inputs": {"ticket": {"type": "state"}},
            },
            "agreed": {"type": "vote", "rule": "agree", "sources": ["team", "second"]},
            "spread": {
                "type": "sum",
                "terms": [{"node": "urgency"}, {"node": "refund", "weight": -1}],
                "constant": 0.5,
            },
            "gate": {
                "type": "threshold",
                "source": "refund",
                "cutoff": 0.5,
                "below": "review",
                "above": "accept",
            },
            "hold": {
                "type": "value",
                "value": "review",
                "when": {
                    "type": "all",
                    "conditions": [
                        {"type": "present", "node": "team"},
                        {
                            "type": "not",
                            "condition": {
                                "type": "compare",
                                "node": "urgency",
                                "op": "ge",
                                "value": 0.5,
                            },
                        },
                    ],
                },
            },
            "pick": {
                "type": "coalesce",
                "sources": ["hold", "gate", "route"],
                "when": {
                    "type": "any",
                    "conditions": [
                        {"type": "compare", "node": "team", "op": "eq", "value": "billing"},
                        {"type": "compare", "node": "refund", "op": "lt", "value": 0.2},
                        {"type": "relation", "node": "urgency", "op": "ge", "other": "refund"},
                    ],
                },
            },
            "priority": {"type": "read", "path": ["priority"], "categories": ["low", "high"]},
            # A model reading a field out of the text, and a read of that field.
            "details": {
                "type": "extract",
                "instructions": "What amount does the ticket name?",
                "inputs": {"ticket": {"type": "state"}},
            },
            "amount": {"type": "read", "source": "details", "path": ["amount"], "bounds": [0, 1e4]},
            # A list read as its items' fields, and a count of the items that meet a test.
            "lines": {"type": "read", "path": ["lines"], "each": {"cost": {"bounds": [0, 100]}}},
            "costly": {
                "type": "quantify",
                "source": "lines",
                "rule": "count",
                "where": {"type": "compare", "node": "cost", "op": "ge", "value": 50},
            },
            # A unit: another graph, handed the input, answering with its label.
            "policy": {
                "type": "graph",
                "input": {"type": "state"},
                "output": "label",
                "plan": {
                    "nodes": {"standard": {"type": "value", "value": "standard"}},
                    "outputs": {"label": ref("standard")},
                },
            },
            "summary": {
                "type": "merge",
                "inputs": {
                    "priority": {"type": "node", "node": "priority", "optional": True},
                    "amount": {"type": "node", "node": "amount", "optional": True},
                    "costly": {"type": "node", "node": "costly", "optional": True},
                    "policy_applied": ref("policy"),
                    "route": ref("route"),
                    "team": {"type": "node", "node": "team", "view": "answer"},
                    "held": {"type": "node", "node": "hold", "optional": True},
                    "policy": {"type": "literal", "value": {"version": 1}},
                    "sure": ref("sure"),
                    "agreed": ref("agreed"),
                    "spread": ref("spread"),
                },
            },
        },
        "outputs": {"label": ref("pick"), "summary": ref("summary")},
    }


def test_dspy_proposals_can_use_the_whole_graph_language_in_real_search(provider, client):
    graph = every_construct_plan()
    used = set(re.findall(r'"type": "(\w+)"', json.dumps(graph)))
    language = set().union(*map(schema_types, (Node, Question, Source, Condition)))
    assert language <= used
    data = task_data()
    data["validation"][0]["label"] = "accept"
    provider.completion = chat_reply(graph=graph)
    provider.answers = {
        "team": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.8,
            "probabilities": {"billing": 0.9, "technical": 0.1},
        },
        "urgency": {
            "type": "score",
            "score": 0.9,
            "confidence": 0.8,
            "probabilities": {"0": 0.1, "1": 0.9},
            "legend": {"0": "Routine request", "1": "Immediate assistance needed"},
        },
        "refund": {"type": "noul", "noul": 0.9},
        "second": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.7,
            "probabilities": {"billing": 0.8, "technical": 0.2},
        },
    }
    lm = dspy.LM(
        "openai/fixture-model",
        api_base=provider.url + "/v1",
        api_key="test-placeholder",
        cache=False,
        num_retries=0,
    )
    best, report = optimize_plan(
        Task.model_validate(data),
        Plan.model_validate(constant_plan()),
        client,
        "fixture-model",
        GraphProposer(lm),
        # Every construct takes more nodes than search allows a graph by default.
        SearchConfig(max_metric_calls=4, reflection_batch=1, max_nodes=24),
        # The extractor shares the proposer's endpoint, so it reads a proposal, fills no field,
        # and the optional amount stays out of the summary.
        models=Models(extractor=Extractor(lm)),
    )
    assert best == Plan.model_validate(graph)
    assert report["validation_score"] == 1.0
    judged = [body["questions"] for path, body in provider.requests if path == "/v1/systemone"]
    assert sorted(key for questions in judged for key in questions) == sorted(
        ["team", "second", "urgency", "refund"] * 2
    )


def test_reflection_prompt_shows_choice_options_mapped_to_descriptions(client):
    prompts = []

    def reflect(prompt):
        prompts.append(prompt)
        return Plan.model_validate(constant_plan("accept")).model_dump_json()

    optimize_plan(
        Task.model_validate(task_data()),
        Plan.model_validate(constant_plan()),
        client,
        "fixture-model",
        reflect,
        SearchConfig(max_metric_calls=4, reflection_batch=1),
    )
    example = re.search(r"An example of one: ?\n(\{.*\})\n", prompts[0]).group(1)
    (node,) = Plan.model_validate_json(example).nodes.values()
    criteria = node.question.criteria
    assert set(criteria) == {"accept", "review"}
    assert all(description != option for option, description in criteria.items())


def test_a_proposal_that_ties_its_parent_reaches_the_next_reflection(client):
    tie = {
        "nodes": {"hold": {"type": "value", "value": "review"}},
        "outputs": {"label": ref("hold")},
    }
    answer = scripted(Plan.model_validate(constant_plan("accept")))
    prompts, offered = [], []

    def reflect(prompt):
        prompts.append(prompt)
        shown = prompt if isinstance(prompt, str) else json.dumps(prompt)
        part = shown.rsplit("Return the part named ", 1)[-1].split(".", 1)[0]
        if not offered and part == "rules":
            offered.append(json.dumps(tie))
            return offered[0]
        return answer(prompt)

    optimize_plan(
        Task.model_validate(task_data()),
        Plan.model_validate(constant_plan()),
        client,
        "fixture-model",
        reflect,
        SearchConfig(max_metric_calls=6, reflection_batch=1),
    )
    # The tie scored no better than its parent, so it was discarded; its results still
    # reach the next reflection.
    carried = [
        p for p in prompts if json.dumps(tie) in (p if isinstance(p, str) else json.dumps(p))
    ]
    assert carried, "the tied proposal never came back"
    latest = carried[-1] if isinstance(carried[-1], str) else json.dumps(carried[-1])
    assert "Expected label: accept" in latest.split(json.dumps(tie), 1)[1]


def test_proposals_containing_code_fences_reach_search_intact(provider, client):
    graph = binary_plan()
    graph["nodes"]["eligible"]["question"]["instructions"] = "Quote it as ```text``` if unsure."
    proposal = Plan.model_validate(graph)
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    data = task_data()
    data["validation"][0]["label"] = "accept"
    best, report = optimize_plan(
        Task.model_validate(data),
        Plan.model_validate(constant_plan()),
        client,
        "fixture-model",
        scripted(proposal),
        SearchConfig(max_metric_calls=4, reflection_batch=1),
    )
    assert parts(best) == parts(proposal)
    assert "```text```" in best.nodes["eligible"].question.instructions


def test_invalid_proposals_are_repaired_before_search_scores_them(client):
    broken = {"nodes": ["invalid node container"]}
    fixed = Plan.model_validate(constant_plan("accept"))
    prompts = []

    def reflect(prompt):
        prompts.append(prompt)
        return json.dumps(broken) if len(prompts) == 1 else fixed.model_dump_json()

    data = task_data()
    data["validation"][0]["label"] = "accept"
    best, report = optimize_plan(
        Task.model_validate(data),
        Plan.model_validate(constant_plan()),
        client,
        "fixture-model",
        reflect,
        SearchConfig(max_metric_calls=4, reflection_batch=1, repair_attempts=1),
    )
    assert best == fixed
    # The repair happened before scoring; other parts may still propose their own mistakes.
    assert report["repairs"] >= 1
    # The repair prompt quotes what would not compile, and why.
    repair = next(p for p in prompts if json.dumps(broken) in p)
    assert "nodes must be an object" in repair


def rule_plan(cutoff=0.5):
    """binary_plan with its judgment asked about the whole input, which rules can write."""
    plan = binary_plan(cutoff)
    plan["nodes"]["eligible"] = call(observation={"type": "state"})
    return plan


@pytest.mark.parametrize("malformed", [False, True])
def test_real_gepa_repairs_fragments_in_parent_context_over_http(provider, client, malformed):
    data = task_data()
    data["validation"][0]["label"] = "accept"
    proposed = Plan.model_validate(rule_plan())
    written = parts(proposed)
    offered = []

    def respond(request):
        shown = request["messages"][-1]["content"]
        part = shown.rsplit("Return the part named ", 1)[-1].split(".", 1)[0]
        part = part if part in written else "rules"
        answer = written[part]
        if part == "rules":
            answer = "\n".join(text for text in (written["readings"], answer) if text)
        if malformed and not offered:
            answer = "when nonsense >= 1: accept\n" + answer
        offered.append(part)
        provider.completion = chat_reply(graph=answer)

    provider.on_chat = respond
    provider.answers = {"eligible": {"type": "noul", "noul": 0.9}}
    proposer = GraphProposer(
        dspy.LM(
            "openai/fixture-model",
            api_base=provider.url + "/v1",
            api_key="test-placeholder",
            cache=False,
            num_retries=0,
        )
    )
    best, report = optimize_plan(
        Task.model_validate(data),
        Plan.model_validate(rule_plan(cutoff=0.95)),
        client,
        "fixture-model",
        proposer,
        SearchConfig(max_metric_calls=6, reflection_batch=1, repair_attempts=1),
    )
    assert parts(best) == written
    # A rule the compiler cannot read is repaired before anything is scored.
    assert report["repairs"] == int(malformed)
    assert report["invalid_candidate_evaluations"] == 0


def test_repair_prompts_carry_only_the_latest_failed_attempt(client):
    attempts = [json.dumps({"nodes": [f"invalid_container_{n}"]}) for n in range(3)]
    prompts = []

    def reflect(prompt):
        prompts.append(prompt)
        return attempts[len(prompts) - 1] if len(prompts) <= 3 else attempts[-1]

    optimize_plan(
        Task.model_validate(task_data()),
        Plan.model_validate(constant_plan()),
        client,
        "fixture-model",
        reflect,
        SearchConfig(max_metric_calls=4, reflection_batch=1, repair_attempts=2),
    )
    assert attempts[1] in prompts[2] and attempts[0] not in prompts[2]


@pytest.mark.parametrize("entry", ["search", "evaluate"])
def test_parallel_examples_are_scored_together_and_keep_their_order(client, entry):
    data = task_data()
    data["train"] = [{"id": name, "state": f"request {name}", "label": "accept"} for name in "xyz"]
    task = Task.model_validate(data)
    # Each score waits until all three are underway, so serial scoring can never finish.
    meeting = threading.Barrier(3, timeout=10)

    def score(program, example):
        meeting.wait()
        return Labels(task)(program, example)

    plan = constant_plan()
    if entry == "search":
        config = SearchConfig(parallel_examples=3)
        adapter = GraphAdapter(task, client, "fixture", config, score=score)
        records = adapter.evaluate(task.train, {"graph": json.dumps(plan)}).outputs
    else:
        report = evaluate_plan(
            Plan.model_validate(plan), task.train, client, "f", score=score, parallel_examples=3
        )
        records = report["examples"]
    assert [record["id"] for record in records] == ["x", "y", "z"]


def test_a_part_that_names_a_missing_node_is_an_invalid_candidate(client):
    task = Task.model_validate(task_data())
    parts = components(Plan.model_validate(binary_plan()))
    adapter = GraphAdapter(task, client, "fixture", SearchConfig())
    batch = adapter.evaluate(
        task.train, parts | {"perception": '{"nodes": {}}'}, capture_traces=True
    )
    assert batch.scores == [0.0]
    assert "Candidate validation failed" in batch.trajectories[0]["Feedback"]


def test_scoring_the_same_graph_twice_costs_one_evaluation(client):
    task = Task.model_validate(task_data())
    calls = []

    def counted(program, example):
        calls.append(example.id)
        return Labels(task)(program, example)

    adapter = GraphAdapter(task, client, "fixture", SearchConfig(), score=counted)
    parts = components(Plan.model_validate(constant_plan()))
    first = adapter.evaluate(task.train, parts)
    again = adapter.evaluate(task.train, parts)
    assert calls == [example.id for example in task.train]
    assert again.scores == first.scores and again.outputs == first.outputs


def test_candidates_that_ask_the_same_question_pay_for_it_once(provider, client):
    # Most candidates change one part and keep the rest, so search re-asks the judge the same
    # questions about the same examples; only a question it has not heard before costs a call.
    task = Task.model_validate(task_data())
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    adapter = GraphAdapter(task, client, "fixture", SearchConfig())
    for cutoff in (0.5, 0.7):
        adapter.evaluate(task.train, components(Plan.model_validate(rule_plan(cutoff))))
    asked = [path for path, _ in provider.requests if path == "/v1/systemone"]
    assert len(asked) == len(task.train)


def test_with_a_price_on_calls_the_cheaper_of_two_equal_graphs_scores_higher(provider, client):
    # Both answer every example right; one asks the judge first. Only the price tells them apart.
    data = task_data()
    data["train"] = [{"id": "a", "state": "request A", "label": "accept"}]
    task = Task.model_validate(data)
    provider.body["answers"] = {"eligible": {"type": "noul", "noul": 0.9}}
    asking = components(Plan.model_validate(rule_plan(0.5)))
    silent = components(Plan.model_validate(constant_plan("accept")))
    free = GraphAdapter(task, client, "fixture", SearchConfig())
    assert free.evaluate(task.train, asking).scores == free.evaluate(task.train, silent).scores
    priced = GraphAdapter(task, client, "fixture", SearchConfig(call_cost=0.1))
    batch = priced.evaluate(task.train, asking, capture_traces=True)
    assert batch.scores == [pytest.approx(0.9)]
    assert priced.evaluate(task.train, silent).scores == [1.0]
    assert "0.1" in batch.trajectories[0]["Feedback"]


def test_a_win_on_too_few_validation_examples_does_not_replace_the_seed(provider, client):
    # One validation example cannot tell a better graph from a lucky one.
    data = task_data()
    data["validation"][0]["label"] = "accept"
    task = Task.model_validate(data)
    initial = Plan.model_validate(constant_plan())
    provider.answers = {"eligible": {"type": "noul", "noul": 0.9}}
    better = Plan.model_validate(binary_plan()).model_dump_json()
    config = SearchConfig(max_metric_calls=4, reflection_batch=1, significance=0.05)
    plan, report = optimize_plan(task, initial, client, "fixture-model", lambda _: better, config)
    assert plan == initial
    assert report["kept_seed"] and report["improvement_p"] == 0.5
    assert report["validation_score"] == report["seed_validation_score"] == 0.0


def test_one_sided_sign_test():
    assert improvement_p(0, 0) == 1.0
    assert improvement_p(1, 0) == 0.5
    assert improvement_p(10, 0) == pytest.approx(1 / 2**10)
    assert improvement_p(3, 3) > 0.5


def test_a_search_seeded_with_statements_keeps_them_and_records_its_moves(provider, client):
    data = task_data()
    data["train"].append({"id": "d", "state": "request D", "label": "review"})
    task = Task.model_validate(data)
    seed = compile_rules(
        'ask refund = "The customer requests a refund."\nwhen refund >= 0.5: review\nalways: accept'
    )
    # The seed's one line gets every example wrong; two statements summed get them all right,
    # and a single choice gets A right and the rest wrong.
    probabilities = {
        "request A": {"refund": 0.6, "dispute": 0.2},
        "request B": {"refund": 0.4, "dispute": 0.9},
        "request D": {"refund": 0.4, "dispute": 0.9},
    }
    choice = 'always: ask "Classify requests." [accept: "Accept it."] [review: "Review it."]'
    statements = (
        'ask refund = "The customer requests a refund."\n'
        'ask dispute = "The customer disputes a charge."\n'
        "let risk = refund + dispute\n"
        "when risk >= 1.2: review\n"
        "always: accept"
    )

    def answer(request):
        asked = probabilities[request["state"]["observation"]]
        provider.answers = {name: {"type": "noul", "noul": p} for name, p in asked.items()} | {
            "rule_1": {
                "type": "choice",
                "choice": "accept",
                "confidence": 0.9,
                "probabilities": {"accept": 0.9, "review": 0.1},
            }
        }

    rules_asked = []

    def propose(request):
        shown = json.dumps(request["messages"])
        part = shown.rsplit("Return the part named ", 1)[-1].split(".", 1)[0]
        if part != "rules":
            provider.completion = chat_reply(move="free", graph=components(seed)["readings"])
            return
        rules_asked.append(part)
        if len(rules_asked) == 1:
            provider.completion = chat_reply(move="free", graph=choice)
        else:
            provider.completion = chat_reply(move="statements", graph=statements)

    provider.on_request, provider.on_chat = answer, propose
    lm = dspy.LM(
        "openai/fixture-model",
        api_base=provider.url + "/v1",
        api_key="test-placeholder",
        cache=False,
        num_retries=0,
    )
    best, report = optimize_plan(
        task,
        seed,
        client,
        "fixture-model",
        GraphProposer(lm),
        SearchConfig(max_metric_calls=20, reflection_batch=2),
    )
    assert len(rules_asked) >= 2
    assert report["seed_validation_score"] == 0.0 and report["validation_score"] == 1.0
    nouls = [
        node
        for node in best.nodes.values()
        if isinstance(node, Call) and node.question.type == "noul"
    ]
    assert len(nouls) == 2
    tried = {(move["part"], move["move"], move["kept"]) for move in report["moves"]}
    assert {("rules", "free", True), ("rules", "statements", True)} <= tried
    # A readings answer that restates the seed changes nothing, so it is never kept.
    assert all(not move["kept"] for move in report["moves"] if move["part"] == "readings")
