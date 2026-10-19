"""A plan drawn as a Mermaid flowchart."""

import re
from pathlib import Path

import pytest
from test_graph import binary_plan

from rulesmith.diagram import draw, mermaid
from rulesmith.graph import Coalesce, Graph, NodeInput, Plan
from rulesmith.maps import load
from rulesmith.rules import compile_rules

GRAPHS = sorted((Path(__file__).resolve().parents[1] / "examples/doom/graphs").glob("*.json"))


def test_every_node_and_edge_appears():
    plan = compile_rules(
        "read variables.HEALTH in [0, 200]\n"
        "when variables.HEALTH < 20: MOVE_BACKWARD\n"
        "always: ATTACK"
    )
    drawn = mermaid(plan)
    assert drawn.startswith("flowchart TD")
    # Every node, every output, and one box for the rule's condition.
    assert len(boxes(drawn)) == len(plan.nodes) + len(plan.outputs) + 1
    assert len(conditions(drawn)) == 1


def boxes(drawn: str) -> dict[str, str]:
    """Each box's id and the text inside it."""
    return dict(re.findall(r'^\s*(\w+)[\[({/]+"([^"]*)"', drawn, re.MULTILINE))


def conditions(drawn: str) -> list[tuple[str, str]]:
    """The dashed arrows from each condition box to the rule it gates, as (from, to) ids."""
    return re.findall(r"^\s*(c\d+) -\.-> (\w+)$", drawn, re.MULTILINE)


@pytest.mark.parametrize("path", GRAPHS, ids=lambda path: path.stem)
def test_every_repository_graph_draws(path):
    plan = Plan.model_validate_json(path.read_text())
    drawn = mermaid(plan)
    # Mermaid takes a label between shape markers, so a bracket inside one breaks the diagram.
    for line in drawn.splitlines()[1:]:
        if '"' in line:
            assert line.count('"') == 2, line


def test_a_graph_rules_cannot_express_still_draws():
    # The case the diagram is for: merges and several outputs have no rule form.
    plan = Plan.model_validate(binary_plan())
    drawn = mermaid(plan)
    assert drawn.count("-->") >= 1


def test_a_guard_keeps_its_grouping():
    # Dropping the parentheses turns 'not (a or b)' into '(not a) or b', a different rule.
    plan = compile_rules(
        "read a in [0, 9]\nread b in [0, 9]\nwhen not (a >= 1 or b >= 1): QUIET\nalways: LOUD"
    )
    assert "not (a >= 1 or b >= 1)" in mermaid(plan)


def test_a_node_named_like_an_output_is_drawn_apart_from_it():
    # Mermaid joins boxes that share an id, which drew a cycle the graph does not have.
    import re
    from graphlib import TopologicalSorter

    plan = compile_rules("read x in [0, 9]\nlet label = x >= 3\nwhen label: A\nalways: B")
    drawn = mermaid(plan)
    edges = re.findall(r"^\s*(\w+) -[.-]*->(?:\|[^|]*\|)? (\w+)$", drawn, re.MULTILINE)
    graph: dict[str, set[str]] = {}
    for source, target in edges:
        graph.setdefault(target, set()).add(source)
    tuple(TopologicalSorter(graph).static_order())


def test_every_judge_is_marked_and_named_as_the_rules_test_it():
    # The arrows read 'team == "billing"'; a reader must find the box that is 'team' and see
    # that it is the one place the model is asked.
    plan = compile_rules(
        (Path(__file__).resolve().parents[1] / "examples/support-routing/graph.rules").read_text()
    )
    drawn = mermaid(plan)
    ids = {name: f"n{index}" for index, name in enumerate(plan.nodes)}
    marked = {
        id_
        for listed in re.findall(r"^\s*class (\S+) judge$", drawn, re.MULTILINE)
        for id_ in listed.split(",")
    }
    calls = {name for name, node in plan.nodes.items() if node.type == "call"}
    assert calls and marked == {ids[name] for name in calls}
    assert all(name in boxes(drawn)[ids[name]] for name in calls)


def test_the_arrows_into_a_list_of_rules_are_numbered_in_the_order_they_are_tried():
    # Drawn in any other order, a judge last in line would look as if it were always asked.
    plan = compile_rules("read x in [0, 9]\nwhen x >= 5: zulu\nwhen x >= 2: alpha\nalways: mike")
    drawn = mermaid(plan)
    ids = {name: f"n{index}" for index, name in enumerate(plan.nodes)}
    ranked = dict(re.findall(r'^\s*(\w+) -->\|"(\d+)"\| ', drawn, re.MULTILINE))
    assert [ranked[ids[source]] for source in plan.nodes["decision"].sources] == ["1", "2", "3"]


def test_every_condition_is_a_box_fed_by_what_it_reads():
    # One form for every condition: on an arrow, one reading two values looked as if it
    # belonged to one of them, and one reading a single value looked unlike it.
    plan = compile_rules(
        "read logo in [0, 1]\nread length of profile in [0, 9999]\n"
        "when logo == 0 and length of profile == 0: scam\nalways: real"
    )
    drawn = mermaid(plan)
    ids = {name: f"n{index}" for index, name in enumerate(plan.nodes)}
    (condition,) = [id_ for id_, text in boxes(drawn).items() if "logo == 0 and" in text]
    fed = re.findall(rf"^\s*(\w+) --> {condition}$", drawn, re.MULTILINE)
    assert sorted(fed) == sorted(ids[name] for name in ("read_logo", "read_length_profile"))
    assert conditions(drawn) == [(condition, ids["rule_1"])]
    single = mermaid(compile_rules("read logo in [0, 1]\nwhen logo == 0: scam\nalways: real"))
    (box,) = [id_ for id_, text in boxes(single).items() if text == "logo == 0"]
    assert re.search(rf"^\s*n0 --> {box}$", single, re.MULTILINE) and conditions(single) == [
        (box, "n1")
    ]


def test_a_drawing_names_every_node_and_frames_a_unit_inside_its_unit():
    # The drawing is what a canvas lays out: every node of the map and of each unit is a box,
    # a unit's boxes name the unit they sit in, and every arrow joins two boxes it holds.
    plan = load(Path(__file__).resolve().parents[1] / "examples/trust-safety/map.json")
    drawn = draw(plan)
    ids = [box.id for box in drawn.boxes]
    assert len(ids) == len(set(ids))
    for edge in drawn.edges:
        assert edge.source in ids and edge.target in ids
    units = [box for box in drawn.boxes if box.kind == "unit"]
    assert {u.name for u in units} == {n for n, node in plan.nodes.items() if node.type == "graph"}
    for unit in units:
        inner = {box.name for box in drawn.boxes if box.unit == unit.id and box.type}
        assert inner == set(plan.nodes[unit.name].plan.nodes)
    # A gate points at the unit as a whole, from a condition box outside it.
    by_id = {box.id: box for box in drawn.boxes}
    gates = [e for e in drawn.edges if e.kind == "gate" and by_id[e.target].kind == "unit"]
    assert gates and all(by_id[e.source].kind == "condition" for e in gates)
    assert all(by_id[e.source].unit is None for e in gates)


def test_a_unit_that_is_one_answer_says_so_on_its_box():
    # `always: hold` compiles to no nodes and a literal output, so there is nothing to draw
    # inside the unit; its box says the answer, since the unit is that answer.
    unit = Graph(plan=compile_rules("always: hold\n", None))
    plan = Plan(nodes={"large": unit, "decision": Coalesce(sources=["large"])}, outputs={"label": NodeInput(node="decision")})
    drawn = draw(plan)
    box = next(b for b in drawn.boxes if b.kind == "unit")
    assert box.lines == ("large", "always hold")
    assert not any(b.unit == box.id for b in drawn.boxes)


def test_a_closed_drawing_keeps_the_units_and_nothing_inside_them():
    plan = load(Path(__file__).resolve().parents[1] / "examples/trust-safety/map.json")
    drawn = draw(plan, closed=True)
    assert drawn.closed and all(box.unit is None for box in drawn.boxes)
    assert [box.kind for box in drawn.boxes].count("unit") == 3
    ids = {box.id for box in drawn.boxes}
    assert all(e.source in ids and e.target in ids for e in drawn.edges)


def test_a_drawing_is_plain_data():
    import json

    plan = compile_rules("read x in [0, 9]\nwhen x >= 5: yes\nalways: no")
    data = json.loads(json.dumps(draw(plan).to_dict()))
    assert set(data) == {"closed", "boxes", "edges"}
    assert {box["kind"] for box in data["boxes"]} == {"read", "condition", "rule", "join", "output"}
    assert all(isinstance(box["lines"], list) for box in data["boxes"])
    ranked = sorted(e["rank"] for e in data["edges"] if e["kind"] == "rank")
    assert ranked == [1, 2]


def test_the_flowchart_draws_exactly_the_drawing():
    # The Mermaid text is one rendering of the drawing: every box and arrow, nothing else.
    plan = load(Path(__file__).resolve().parents[1] / "examples/trust-safety/map.json")
    drawn, text = draw(plan), mermaid(plan)
    for box in drawn.boxes:
        assert re.search(rf"^\s*{box.id}[\[({{/]", text, re.MULTILINE), box.id
    arrows = re.findall(r"^\s*(\w+) [-.=]+>(?:\|[^|]*\|)? (\w+)$", text, re.MULTILINE)
    assert len(arrows) == len(drawn.edges)
    by_id = {box.id: box for box in drawn.boxes}
    for edge in drawn.edges:
        framed = by_id[edge.target].kind == "unit" and by_id[edge.source].unit != edge.target
        assert (edge.source, f"{edge.target}_unit" if framed else edge.target) in arrows, edge


def test_a_gated_ordered_decision_shows_its_gate():
    # A coalesce can carry a condition like any node; drawn without it, the decision looked
    # as if it always ran.
    plan = Plan.model_validate(
        {
            "nodes": {
                "x": {"type": "read", "path": ["x"], "bounds": [0, 9]},
                "a": {"type": "value", "value": "A"},
                "pick": {
                    "type": "coalesce",
                    "sources": ["a"],
                    "when": {"type": "present", "node": "x"},
                },
            },
            "outputs": {"label": {"type": "node", "node": "pick"}},
        }
    )
    drawn = draw(plan)
    (gate,) = [e for e in drawn.edges if e.kind == "gate"]
    by_id = {box.id: box for box in drawn.boxes}
    assert by_id[gate.target].name == "pick" and by_id[gate.source].lines == ("x is present",)
    assert conditions(mermaid(plan)) == [(gate.source, gate.target)]
