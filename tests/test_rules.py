"""The rule surface a proposer writes, and the graph it compiles to."""

import json
import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from test_graph import binary_plan, call, constant_plan, ref, task_data

from rulesmith.graph import Plan, Task
from rulesmith.rules import RuleError, compile_rules, keep_comments, parts, render_rules
from rulesmith.runtime import DecisionProgram

GRAPHS = sorted((Path(__file__).resolve().parents[1] / "examples/doom/graphs").glob("*.json"))


@pytest.mark.parametrize("path", GRAPHS, ids=lambda path: path.stem)
def test_every_repository_graph_survives_the_round_trip(path):
    plan = Plan.model_validate_json(path.read_text())
    rules = render_rules(plan)
    # Compiling picks its own node names, so the text is what must come back unchanged.
    assert render_rules(compile_rules(rules)) == rules


def test_rules_read_as_the_decisions_they_make():
    plan = Plan.model_validate_json((GRAPHS[0].parent / "duel-hunter.json").read_text())
    text = render_rules(plan)
    # One line per rule, in the order the graph tries them.
    assert "TURN_LEFT" in text and "compass.2.bearing >= 45" in text
    assert text.count("\n") + 1 < len(plan.model_dump_json()) // 200


def test_a_rule_may_ask_the_judge_and_fall_through_when_it_declines():
    text = "\n".join(
        [
            "read variables.HEALTH in [0, 200]",
            'when variables.HEALTH >= 25: ask "Fight or run?"'
            ' [ATTACK: "Opponent is lined up."] [MOVE_BACKWARD: "Opponent has the angle."]',
            "always: MOVE_FORWARD",
        ]
    )
    plan = compile_rules(text)
    kinds = [node.type for node in plan.nodes.values()]
    assert "call" in kinds and "value" in kinds
    assert render_rules(compile_rules(render_rules(plan))) == render_rules(plan)


def test_a_proposal_that_fences_its_answer_and_restates_the_sections_still_compiles():
    # What the proposer actually writes when asked for one part: the whole program, in a fence,
    # with the section headings repeated. None of that changes the rules it is proposing.
    reading = "read variables.HEALTH in [0, 200]"
    rules = "when variables.HEALTH < 20: MOVE_BACKWARD\nalways: ATTACK"
    plain = f"{reading}\n{rules}"
    dressed = f"```\nreadings:\n{reading}\n\nrules:\n{rules}\n```"
    assert render_rules(compile_rules(dressed)) == render_rules(compile_rules(plain))


@given(st.floats(min_value=0, max_value=1))
def test_the_ordering_operators_partition_every_reading(value):
    # '<=' and '>' are exact complements, so one arm fires for any reading, the boundary included.
    plan = compile_rules("read score in [0, 1]\nwhen score > 0.5: high\nwhen score <= 0.5: low")
    assert DecisionProgram(plan, None, "none")({"score": value}).label == (
        "high" if value > 0.5 else "low"
    )


def test_a_rule_that_accepts_either_of_two_readings_round_trips():
    # Rendering a graph produced ' or ' that nothing could read back.
    text = (
        "read variables.HEALTH in [0, 200]\n"
        "read target.distance in [0, 6000]\n"
        "when variables.HEALTH < 20 or target.distance < 100: MOVE_BACKWARD\n"
        "always: ATTACK"
    )
    plan = compile_rules(text)
    rules = render_rules(plan)
    assert " or " in rules and render_rules(compile_rules(rules)) == rules
    program = DecisionProgram(plan, None, "none")
    assert (
        program({"variables": {"HEALTH": 10}, "target": {"distance": 900}}).label == "MOVE_BACKWARD"
    )
    assert (
        program({"variables": {"HEALTH": 90}, "target": {"distance": 10}}).label == "MOVE_BACKWARD"
    )
    assert program({"variables": {"HEALTH": 90}, "target": {"distance": 900}}).label == "ATTACK"


def test_a_group_of_alternatives_is_read_as_one_test():
    # The bearing rules a proposer writes put the wrap-around case in parentheses.
    text = (
        "read compass.2.bearing in [-180, 180]\n"
        "when compass.2.bearing and (compass.2.bearing >= 135 or compass.2.bearing < -135): "
        "MOVE_BACKWARD\n"
        "always: MOVE_FORWARD"
    )
    program = DecisionProgram(compile_rules(text), None, "none")

    # compass.2 indexes the third thing on the compass, as the game reports it.
    def seen(bearing):
        return {"compass": [{}, {}, {"bearing": bearing}]}

    assert program(seen(170)).label == "MOVE_BACKWARD"
    assert program(seen(-170)).label == "MOVE_BACKWARD"
    assert program(seen(10)).label == "MOVE_FORWARD"


def test_a_group_inside_a_longer_condition_keeps_its_grouping_through_the_round_trip():
    # Rendered without parentheses, 'a and (b or c)' reads back as '(a and b) or c', which is a
    # different program that still compiles -- the drift has to fail here, not in a duel.
    text = (
        "read variables.HEALTH in [0, 200]\n"
        "read target.distance in [0, 6000]\n"
        "when variables.HEALTH >= 25 and (target.distance < 100 or target.distance >= 900): "
        "ATTACK\n"
        "always: MOVE_FORWARD"
    )
    plan = compile_rules(text)
    assert compile_rules(render_rules(plan)) == plan
    program = DecisionProgram(plan, None, "none")
    assert (
        program({"variables": {"HEALTH": 10}, "target": {"distance": 50}}).label == "MOVE_FORWARD"
    )
    assert program({"variables": {"HEALTH": 90}, "target": {"distance": 50}}).label == "ATTACK"
    assert (
        program({"variables": {"HEALTH": 90}, "target": {"distance": 500}}).label == "MOVE_FORWARD"
    )


def test_a_constant_policy_is_one_line():
    plan = Plan.model_validate(constant_plan("review"))
    assert render_rules(plan).strip() == "always: review"
    assert compile_rules("always: review") == plan


def test_a_rule_naming_an_unknown_reading_says_which_line():
    with pytest.raises(RuleError, match="line 2"):
        compile_rules("always: review\nwhen nonsense.path >= 1: accept")


def test_graphs_beyond_rules_stay_in_json():
    # Merges join several judgments into one value; rules have no way to say that.
    merged = {
        "nodes": {
            "left": call(observation={"type": "state"}),
            "right": call(observation={"type": "state"}),
            "both": {"type": "merge", "inputs": {"a": ref("left"), "b": ref("right")}},
        },
        "outputs": {"label": ref("both")},
    }
    with pytest.raises(RuleError, match="merge"):
        render_rules(Plan.model_validate(merged))
    # A judgment shown the input under its own key cannot be written as rules either.
    pair = json.loads(
        (Path(__file__).resolve().parents[1] / "examples/support-routing/dag.json").read_text()
    )
    with pytest.raises(RuleError, match="ticket"):
        render_rules(Plan.model_validate(pair))


def test_the_compiler_accepts_a_whole_graph_as_json():
    # A proposal that needs the full language can still answer with it.
    graph = binary_plan()
    assert compile_rules(json.dumps(graph)) == Plan.model_validate(graph)


def test_rules_only_name_labels_the_task_allows():
    task = Task.model_validate(task_data())
    plan = compile_rules("always: review")
    plan.check(task.labels, 16, 8)


EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*/graph.rules"))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda path: path.parent.name)
def test_every_example_program_compiles_and_survives_the_round_trip(path):
    plan = compile_rules(path.read_text())
    assert render_rules(compile_rules(render_rules(plan))) == render_rules(plan)
    kinds = {node.question.type for node in plan.nodes.values() if node.type == "call"}
    # Each example shows one kind of judgment doing the work that kind is for, and the last
    # keeps its judgment behind readings, asked only when they cannot settle the case.
    shown = {
        "support-routing": "choice",
        "ticket-triage": "score",
        "reply-check": "noul",
        "order-review": "noul",
    }
    assert kinds == {shown[path.parent.name]}
    if path.parent.name == "order-review":
        assert all(node.when is not None for node in plan.nodes.values() if node.type == "call")


def test_the_proposer_is_told_every_kind_of_judgment_it_may_ask():
    # Search can only use what its instructions describe, and the language has three kinds.
    from rulesmith.graph import Task
    from rulesmith.optimize import SearchConfig, proposer_prompt

    task = Task.model_validate(task_data())
    prompt = proposer_prompt(task, SearchConfig())
    # A form given named values needs readings declared, and has a test of its own.
    asked = [form for form in re.findall(r"'(ask [^']+)'", prompt) if " given [" not in form]
    kinds = set()
    for form in asked:
        name = form.split()[1]
        # A judgment nothing asks about is pruned, so give each one a rule that uses it.
        plan = compile_rules(f"{form}\nwhen {name}: review\nalways: accept", task.criteria())
        kinds |= {node.question.type for node in plan.nodes.values() if node.type == "call"}
    assert kinds == {"choice", "score", "noul"}


def test_the_proposer_is_told_how_to_test_confidence_and_count_votes():
    # A construct the instructions never describe is one search will never write.
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    votes = re.findall(r"'(vote [^']+)'", prompt)
    sure = re.findall(r"'(\w+\.confidence [^']+)'", prompt)
    assert votes and sure
    asked = "\n".join(
        f'ask {name} = "Is it?" [review: "Yes."] [accept: "No."]'
        for name in ("threat", "first", "second", "third")
    )
    for line in votes:
        plan = compile_rules(f"{asked}\n{line}\nwhen {line.split()[1]}: review\nalways: accept")
        assert "vote" in {node.type for node in plan.nodes.values()}
    for test in sure:
        plan = compile_rules(f"{asked}\nwhen {test}: review\nalways: accept")
        assert "confidence" in {node.type for node in plan.nodes.values()}


def test_the_proposer_is_told_how_to_name_a_condition_and_compare_two_readings():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    lets = re.findall(r"'(let [^']+)'", prompt)
    relations = re.findall(r"'([A-Za-z_][\w.]* (?:>=|>|<=|<|==) [A-Za-z_][\w.]*)'", prompt)
    assert lets and relations

    def declared(condition):
        mentioned = set(re.findall(r"[A-Za-z_][\w.]*", condition)) - {"and", "or", "not"}
        return "\n".join(f"read {path} in [-99, 99]" for path in sorted(mentioned))

    kinds = set()
    for line in lets:
        name, condition = line.removeprefix("let ").split(" = ", 1)
        plan = compile_rules(f"{declared(condition)}\n{line}\nwhen {name}: review\nalways: accept")
        kinds.add(plan.nodes[name].type)
    # A named condition and a computed value both have to be described to be written.
    assert kinds == {"value", "sum"}
    for relation in relations:
        plan = compile_rules(f"{declared(relation)}\nwhen {relation}: review\nalways: accept")
        assert '"relation"' in plan.model_dump_json()


def test_the_proposer_is_told_how_to_name_a_reading():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    named = re.findall(r"'(read (\w+) = [^']+)'", prompt)
    assert named
    for line, name in named:
        plan = compile_rules(f"{line}\nwhen {name}: review\nalways: accept")
        assert plan.nodes[name].type == "read"


def test_the_proposer_is_told_how_to_give_a_judgment_only_some_values():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    forms = re.findall(r"'(ask (\w+) = [^']+ given \[([^\]]+)\])'", prompt)
    assert forms
    for line, name, listed in forms:
        given = [part.strip() for part in listed.split(",")]
        reads = "\n".join(f"read {path} in [0, 99]" for path in given if path != "observation")
        plan = compile_rules(f"{reads}\n{line}\nwhen {name}: review\nalways: accept")
        assert list(plan.nodes[name].inputs) == given


def test_the_proposer_is_told_how_other_outputs_are_written():
    # A seed with several outputs reaches the proposer as sections it has to keep intact.
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    headings = re.findall(r"'(output \w+:)'", prompt)
    assert headings
    for heading in headings:
        plan = compile_rules(f"always: review\n{heading}\nalways: high")
        assert set(plan.outputs) == {"label", heading.split()[1].rstrip(":")}


def test_the_proposer_is_asked_for_the_parts_search_sends():
    # The prompt once opened by asking for a JSON component named 'policy' that no longer exists.
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    for part in parts(compile_rules("read x in [0, 1]\nwhen x >= 1: review\nalways: accept")):
        assert f"'{part}'" in prompt or f"{part} part" in prompt
    assert "for policy" not in prompt and "perception" not in prompt


def test_the_proposer_is_told_how_to_read_a_length():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(Task.model_validate(task_data()), SearchConfig())
    forms = re.findall(r"'(read length of [^']+)'", prompt)
    assert forms
    for line in forms:
        path = line.removeprefix("read length of ").split()[0]
        plan = compile_rules(f"{line}\nwhen length of {path} < 20: review\nalways: accept")
        assert any(node.type == "read" and node.measure == "length" for node in plan.nodes.values())


COMMENTED = """# Triage: cheap fields first, the judge only for the rest.

read amount in [0, 1000]
# Asked only when the amount alone cannot settle it.
ask risky = "Does this read like fraud?"
  briefed "Card testing, resale."

# Large orders always go to a person.
when amount >= 500: review
when risky >= 0.7: review
# Everything else ships.
always: accept
# Reviewed 2026-10.
"""


def test_rewritten_rules_keep_the_comments_above_the_lines_they_explain():
    rendered = render_rules(compile_rules(COMMENTED)).replace(">= 500", ">= 450")
    kept = keep_comments(COMMENTED, rendered)
    # Nothing but the comments and blank lines is added back: the rules are the rendered ones.
    assert [line for line in kept.splitlines() if line and not line.startswith("#")] == (
        rendered.splitlines()
    )
    lines = kept.splitlines()
    assert lines[:3] == [
        "# Triage: cheap fields first, the judge only for the rest.",
        "",
        "read amount in [0, 1000]",
    ]
    assert lines[
        lines.index("# Asked only when the amount alone cannot settle it.") + 1
    ].startswith("ask risky =")
    assert lines[lines.index("# Large orders always go to a person.") + 1] == (
        "when amount >= 450: review"
    )
    assert lines[lines.index("# Everything else ships.") + 1] == "always: accept"
    assert lines[-1] == "# Reviewed 2026-10."


def test_a_comment_above_a_line_that_is_gone_stays_with_what_followed_it():
    rendered = "read amount in [0, 1000]\nwhen amount >= 500: review\nalways: accept"
    source = (
        "read amount in [0, 1000]\n# Retired.\nread other in [0, 1]\n"
        "when amount >= 500: review\nalways: accept\n"
    )
    lines = keep_comments(source, rendered).splitlines()
    assert lines == [
        "read amount in [0, 1000]",
        "# Retired.",
        "when amount >= 500: review",
        "always: accept",
    ]


def test_a_rule_after_one_that_always_answers_is_refused():
    # It could never decide anything: the first unconditional rule settles every case. Search
    # gets the refusal as feedback, rather than keeping rules a reader would think are live.
    with pytest.raises(RuleError, match="line 3: .*never decide"):
        compile_rules("read x in [0, 10]\nalways: a\nwhen x >= 5: b\n")
    # A question about the whole input is always answered, so nothing after it runs either.
    with pytest.raises(RuleError, match="line 2: .*never decide"):
        compile_rules('always: ask "Is it a scam?" [scam] [genuine]\nalways: genuine\n')
    # A rule after one that only sometimes answers is live: a condition, a computed answer
    # whose reading may be missing, or a question about a field that may be missing.
    compile_rules("read x in [0, 10]\nwhen x >= 5: b\nalways: a\noutput team:\nalways: billing\n")
    compile_rules("read x in [0, 10]\nalways: x >= 5 ? b : a\nalways: c\n")
    compile_rules('read x in [0, 10]\nalways: ask "Q?" [a] [b] given [x]\nalways: c\n')


SECOND_OPINION = """ask first = "Is this order fraudulent?" [yes: "It is."] [no: "It is not."]
ask second = "Read again: is it fraudulent?" [yes: "It is."] [no: "It is not."]
vote checked = agree of [first, second]
when checked == "yes": first
always: ask "The readings split. Is it fraudulent?" [yes: "It is."] [no: "It is not."]"""


def test_a_rule_may_decide_by_a_named_judgment():
    # A second opinion then needs one rule, whatever the label count, not one per label.
    plan = compile_rules(SECOND_OPINION)
    rule = plan.nodes["rule_1"]
    assert rule.type == "coalesce" and rule.sources == ["first"] and rule.when is not None
    assert render_rules(plan) == SECOND_OPINION
    plan.check(["yes", "no"], 16, 8)


def test_a_name_that_is_also_a_label_cannot_decide():
    rules = 'ask yes = "Is it so?" [task labels]\nalways: yes\n'
    with pytest.raises(RuleError, match="yes"):
        compile_rules(rules, {"yes": "It is.", "no": "It is not."})
