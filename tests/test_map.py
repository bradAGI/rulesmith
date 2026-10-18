"""A map: a graph whose units live in files of their own, loaded by path."""

import json
from pathlib import Path

import pytest
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.diagram import draw
from rulesmith.graph import Plan
from rulesmith.maps import MapError, load
from rulesmith.rules import compile_rules

BILLING = (
    "read charge.amount in [0, 100000]\n"
    'ask genuine = "The message describes a mistake the customer did not cause."\n'
    "when charge.amount < 50 and genuine >= 0.8: auto_refund\n"
    "always: review\n"
)


def router(unit: str) -> dict:
    return {
        "nodes": {
            "team": {
                "type": "call",
                "question": {
                    "type": "choice",
                    "instructions": "Which team should answer this ticket?",
                    "criteria": {"billing": "Charges and refunds.", "technical": "Bugs."},
                },
                "inputs": {"observation": {"type": "state"}},
            },
            "billing": {
                "type": "graph",
                "path": unit,
                "when": {"type": "compare", "node": "team", "op": "eq", "value": "billing"},
            },
            "escalate": {"type": "value", "value": "engineer"},
            "decision": {"type": "coalesce", "sources": ["billing", "escalate"]},
        },
        "outputs": {"label": {"type": "node", "node": "decision"}},
    }


def test_a_unit_is_loaded_from_its_own_file_relative_to_the_map(tmp_path):
    (tmp_path / "units").mkdir()
    (tmp_path / "units" / "billing.rules").write_text(BILLING)
    (tmp_path / "maps").mkdir()
    map_file = write(tmp_path / "maps" / "support.json", router("../units/billing.rules"))
    plan = load(map_file)
    # The loaded map is an ordinary graph with the unit embedded, so everything else is unchanged.
    embedded = router("")
    embedded["nodes"]["billing"] = {
        "type": "graph",
        "plan": compile_rules(BILLING).model_dump(mode="json", exclude_none=True),
        "when": embedded["nodes"]["billing"]["when"],
    }
    assert plan == Plan.model_validate(embedded)


def test_a_unit_file_may_be_json_and_hold_units_of_its_own(tmp_path):
    inner = compile_rules("read x in [0, 9]\nwhen x >= 5: high\nalways: low")
    write(tmp_path / "inner.json", inner.model_dump(mode="json", exclude_none=True))
    write(
        tmp_path / "middle.json",
        {
            "nodes": {"depth": {"type": "graph", "path": "inner.json"}},
            "outputs": {"label": {"type": "node", "node": "depth"}},
        },
    )
    map_file = write(tmp_path / "outer.json", router("middle.json"))
    plan = load(map_file)
    assert plan.nodes["billing"].plan.nodes["depth"].plan == inner


def test_a_map_that_contains_itself_is_refused(tmp_path):
    write(tmp_path / "loop.json", router("loop.json"))
    with pytest.raises(MapError, match="loop.json"):
        load(tmp_path / "loop.json")


def test_a_unit_path_that_reaches_nothing_says_which_map_named_it(tmp_path):
    map_file = write(tmp_path / "support.json", router("missing.rules"))
    with pytest.raises(MapError, match="missing.rules") as failure:
        load(map_file)
    assert "support.json" in str(failure.value)


def test_a_graph_read_without_loading_its_units_is_refused():
    # Validation alone cannot see into a file, so it says what to do instead of guessing.
    with pytest.raises(ValueError, match="load"):
        Plan.model_validate(router("billing.rules"))


def test_the_cli_takes_a_map_wherever_it_takes_a_graph(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    data = task_data() | {"labels": ["auto_refund", "review", "engineer"]}
    for split in ("train", "validation", "test"):
        for example in data[split]:
            example["label"] = "review"
    task = write(tmp_path / "task.json", data)
    result = invoke("validate", task, map_file)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["nodes"] == 4
    drawn = invoke("draw", map_file)
    assert drawn.returncode == 0 and "subgraph" in drawn.stdout


def test_the_example_map_loads_from_its_files():
    root = Path(__file__).resolve().parents[1]
    plan = load(root / "examples" / "ticket-handling" / "map.json")
    assert plan.nodes["billing"].plan.nodes


def test_validate_reports_the_units_and_refuses_a_hand_off_the_unit_cannot_meet(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    graph = router("billing.rules")
    # A rule after the unit tests its answer against a label the unit never gives.
    graph["nodes"]["refunded"] = {
        "type": "value",
        "value": "engineer",
        "when": {"type": "compare", "node": "billing", "op": "eq", "value": "approved"},
    }
    graph["nodes"]["decision"]["sources"] = ["refunded", "billing", "escalate"]
    map_file = write(tmp_path / "support.json", graph)
    data = task_data() | {"labels": ["auto_refund", "review", "engineer"]}
    for split in ("train", "validation", "test"):
        for example in data[split]:
            example["label"] = "review"
    task = write(tmp_path / "task.json", data)
    result = invoke("validate", task, map_file)
    assert result.returncode != 0 and "approved" in result.stderr
    graph["nodes"]["refunded"]["when"]["value"] = "auto_refund"
    write(map_file, graph)
    result = invoke("validate", task, map_file)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["units"] == {"billing": "billing.rules"}


def test_a_map_is_drawn_as_data_with_json(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    drawn = invoke("draw", map_file, "--json")
    assert drawn.returncode == 0, drawn.stderr
    drawing = json.loads(drawn.stdout)
    assert drawing == draw(load(map_file)).to_dict()
    assert {box["name"] for box in drawing["boxes"] if box["kind"] == "unit"} == {"billing"}
    closed = json.loads(invoke("draw", map_file, "--json", "--closed").stdout)
    assert closed["closed"] and all(box["unit"] is None for box in closed["boxes"])


def test_a_map_can_be_drawn_with_its_units_closed(tmp_path):
    (tmp_path / "billing.rules").write_text(BILLING)
    map_file = write(tmp_path / "support.json", router("billing.rules"))
    opened = invoke("draw", map_file)
    closed = invoke("draw", map_file, "--closed")
    assert opened.returncode == 0 and closed.returncode == 0, closed.stderr
    assert "subgraph" in opened.stdout and "genuine" in opened.stdout
    # Closed, a unit is one double-edged box, the sign for a program of its own, and its
    # insides are not drawn.
    assert "subgraph" not in closed.stdout and "genuine" not in closed.stdout
    assert '[["billing<br/>a graph of its own"]]' in closed.stdout


def test_a_map_whose_only_judge_calls_are_inside_a_unit_still_gets_a_judge(tmp_path, provider):
    # The map's top level asks nothing; the unit does. Evaluate and run must still bring a judge.
    (tmp_path / "billing.rules").write_text(BILLING)
    graph = {
        "nodes": {
            "billing": {"type": "graph", "path": "billing.rules"},
        },
        "outputs": {"label": {"type": "node", "node": "billing"}},
    }
    map_file = write(tmp_path / "support.json", graph)
    ticket = write(tmp_path / "ticket.json", {"charge": {"amount": 40}, "message": "twice"})
    provider.answers = {"genuine": {"type": "noul", "noul": 0.9}}
    result = invoke(
        "run",
        map_file,
        ticket,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--max-workers",
        1,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["label"] == "auto_refund"


# Routed on a field, so which cases reach the unit is known without a judge.
REFUNDS = (
    "read charge.amount in [0, 100000]\nwhen charge.amount < 50: auto_refund\nalways: review\n"
)
ROUTED = {
    "nodes": {
        "kind": {
            "type": "read",
            "path": ["ticket", "kind"],
            "categories": ["billing", "technical"],
        },
        "billing": {
            "type": "graph",
            "path": "units/billing.rules",
            "input": {"type": "state"},
            "when": {"type": "compare", "node": "kind", "op": "eq", "value": "billing"},
        },
        "escalate": {"type": "value", "value": "engineer"},
        "decision": {"type": "coalesce", "sources": ["billing", "escalate"]},
    },
    "outputs": {"label": {"type": "node", "node": "decision"}},
}


def case(id: str, kind: str, amount: int, label: str, group: str | None = None) -> dict:
    found = {
        "id": id,
        "state": {"ticket": {"kind": kind}, "charge": {"amount": amount}},
        "label": label,
    }
    return found | ({"group": group} if group else {})


def routed_task() -> dict:
    return {
        "description": "Settle a ticket.",
        "labels": ["auto_refund", "review", "engineer"],
        "train": [
            case("t1", "billing", 10, "auto_refund", "g1"),
            case("t2", "billing", 900, "review"),
            case("t3", "technical", 10, "engineer"),
            # Routed to billing, but the right answer is one billing cannot give: the router's
            # mistake, which no change to the unit can fix.
            case("t4", "billing", 15, "engineer"),
        ],
        "validation": [
            case("v1", "billing", 30, "auto_refund"),
            case("v2", "technical", 5, "engineer"),
        ],
        "calibration": [case("c1", "billing", 70, "review")],
        "test": [case("e1", "billing", 20, "auto_refund"), case("e2", "billing", 400, "review")],
    }


def routed_map(tmp_path: Path) -> Path:
    (tmp_path / "units").mkdir()
    (tmp_path / "units" / "billing.rules").write_text(REFUNDS)
    return write(tmp_path / "map.json", ROUTED)


def test_a_decision_names_what_each_unit_it_ran_was_handed(tmp_path):
    from rulesmith.runtime import DecisionProgram

    program = DecisionProgram(load(routed_map(tmp_path)), None, "none")
    state = {"ticket": {"kind": "billing"}, "charge": {"amount": 10}}
    assert program(state=state).handed == {"billing": state}
    # A unit its guard kept from running was handed nothing.
    assert program(state={"ticket": {"kind": "technical"}}).handed == {}


def test_a_units_task_is_the_cases_the_map_hands_it_that_it_can_answer(tmp_path):
    from rulesmith.graph import Task
    from rulesmith.maps import unit_task
    from rulesmith.runtime import DecisionProgram

    program = DecisionProgram(load(routed_map(tmp_path)), None, "none")
    found = unit_task(program, Task.model_validate(routed_task()), "billing")
    assert found.labels == ["auto_refund", "review"]
    assert [e.id for e in found.train] == ["t1", "t2"]
    assert [e.id for e in found.validation] == ["v1"]
    assert [e.id for e in found.calibration] == ["c1"]
    assert [e.id for e in found.test] == ["e1", "e2"]
    # The unit's state is what it was handed, and an example keeps its group.
    assert found.train[0].state == {"ticket": {"kind": "billing"}, "charge": {"amount": 10}}
    assert found.train[0].group == "g1"


def test_a_unit_the_map_never_reaches_in_a_split_has_no_task(tmp_path):
    from rulesmith.graph import Task
    from rulesmith.maps import MapError, unit_task
    from rulesmith.runtime import DecisionProgram

    data = routed_task()
    data["validation"] = [case("v2", "technical", 5, "engineer")]
    program = DecisionProgram(load(routed_map(tmp_path)), None, "none")
    with pytest.raises(MapError, match="validation"):
        unit_task(program, Task.model_validate(data), "billing")


def test_the_cli_writes_a_units_task_and_search_takes_it(tmp_path):
    map_file = routed_map(tmp_path)
    task = write(tmp_path / "task.json", routed_task())
    out = tmp_path / "billing-task.json"
    result = invoke("unit-task", task, map_file, "--unit", "billing", "--output", out)
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["labels"] == ["auto_refund", "review"]
    # The unit's task is what calibrating the unit's own file reads, and its counts are the
    # unit's cases only.
    calibrated = tmp_path / "billing.rules"
    result = invoke("calibrate", out, tmp_path / "units" / "billing.rules", "--output", calibrated)
    assert result.returncode == 0, result.stderr
    assert "of 1" in calibrated.read_text()


def test_calibrating_a_whole_map_says_to_calibrate_its_units(tmp_path):
    task = write(tmp_path / "task.json", routed_task())
    result = invoke("calibrate", task, routed_map(tmp_path), "--output", tmp_path / "out.rules")
    assert result.returncode != 0
    assert "unit-task" in result.stderr
