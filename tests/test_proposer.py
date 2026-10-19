import json

import dspy
from conftest import chat_reply
from test_graph import binary_plan, task_data

from rulesmith.graph import Call, Plan, StateInput, Sum, Task, Vote
from rulesmith.label import Labeler
from rulesmith.optimize import (
    GraphProposal,
    GraphProposer,
    Proposal,
    SearchConfig,
    move_examples,
    proposer_prompt,
    text_adapter,
)
from rulesmith.rules import WHOLE, compile_rules, render_rules


def proposer(provider):
    return GraphProposer(
        dspy.LM(
            "openai/fixture-model",
            api_base=provider.url + "/v1",
            api_key="test-placeholder",
            cache=False,
            num_retries=0,
        )
    )


def fielded_task() -> Task:
    """A labeled task whose inputs carry a number a rule could gate on."""
    data = task_data()
    for split in ("train", "validation", "test"):
        for n, example in enumerate(data[split]):
            example["state"] = {"amount": 100 * (n + 1), "note": example["state"]}
    return Task.model_validate(data)


def asked(plan, kind: str) -> list[Call]:
    return [n for n in plan.nodes.values() if isinstance(n, Call) and n.question.type == kind]


def test_a_move_is_named_beside_the_part_and_may_be_left_out():
    rules = "read target.x in [-1, 1]\nwhen target.x >= 0.1: TURN_RIGHT\nalways: MOVE_FORWARD"
    adapter = text_adapter()
    assert adapter.parse(GraphProposal, chat_reply(move="rubric", graph=rules)) == {
        "move": "rubric",
        "graph": rules,
    }
    # A model with thinking off answers with the part alone; that must still be a proposal.
    assert adapter.parse(GraphProposal, chat_reply(graph=rules)) == {"move": None, "graph": rules}


def test_the_whole_graph_example_names_the_input_as_rules_do():
    # A proposer copies the example's input key, and a key rules cannot write leaves a
    # winning graph with no plan.rules.
    task, config = fielded_task(), SearchConfig()
    prompt = proposer_prompt(task, config)
    example = next(line for line in prompt.splitlines() if line.startswith('{"nodes"'))
    plan = Plan.model_validate(json.loads(example))
    assert plan.nodes["decision"].inputs == {WHOLE: StateInput()}
    render_rules(plan, task.criteria())


def test_every_move_example_is_a_valid_program_that_uses_its_construct():
    task, config = fielded_task(), SearchConfig()
    plans = {
        name: compile_rules(example, task.criteria())
        for name, example in move_examples(task, ["large"], config.max_nodes).items()
    }
    for plan in plans.values():
        plan.check(task.labels, config.max_nodes, config.max_depth)
    assert asked(plans["rubric"], "score")
    assert len(asked(plans["statements"], "noul")) >= 2
    assert any(isinstance(node, Sum) for node in plans["statements"].nodes.values())
    assert any(isinstance(node, Vote) for node in plans["second_opinion"].nodes.values())
    assert any(node.when is not None for node in asked(plans["gate"], "choice"))
    assert any(node.judge == "large" for node in asked(plans["escalate"], "choice"))


def test_moves_are_offered_only_where_the_task_can_take_them():
    # An escalation needs a judge to escalate to, and a gate a field to read.
    assert "escalate" not in move_examples(fielded_task(), (), SearchConfig().max_nodes)
    assert "gate" not in move_examples(
        Task.model_validate(task_data()), ["large"], SearchConfig().max_nodes
    )


def test_the_instructions_show_each_move_in_the_tasks_own_terms():
    task, config = fielded_task(), SearchConfig()
    offered = move_examples(task, ["large"], config.max_nodes)
    assert {"rubric", "statements", "second_opinion", "gate", "escalate"} <= offered.keys()
    shown = proposer_prompt(task, config, judges=["large"])
    for name, example in offered.items():
        assert example in shown, name
    assert offered["escalate"] not in proposer_prompt(task, config)


def test_the_proposer_reports_the_move_with_the_part(provider):
    provider.completion = chat_reply(move="statements", graph="always: review")
    assert proposer(provider)("Propose rules.") == Proposal("always: review", "statements")


def test_a_move_the_instructions_do_not_name_is_reported_as_none(provider):
    # Models answer the field with the part's name or the schema note instead of a move.
    provider.completion = chat_reply(move="rules", graph="always: review")
    assert proposer(provider)("Propose rules.") == Proposal("always: review", None)


def test_whatever_the_model_answers_becomes_the_candidate(provider):
    # Rules the compiler cannot read are reported by it, with the line, not hidden here.
    provider.completion = "when nonsense: MOVE_FORWARD"
    assert proposer(provider)("Propose rules.") == Proposal("when nonsense: MOVE_FORWARD")
    assert len(provider.requests) == 1


def test_a_fenced_answer_loses_its_fence(provider):
    provider.completion = "```\nalways: review\n```"
    assert proposer(provider)("Propose rules.").graph == "always: review"


def test_a_graph_wrapped_in_prose_is_still_read(provider):
    # Small models explain themselves before answering; the graph is still in there.
    graph = binary_plan()
    provider.completion = f"Based on the error I fixed the output.\n\n{json.dumps(graph)}\n\nDone."
    assert json.loads(proposer(provider)("Propose a decision graph.").graph) == graph


def test_proposer_and_labeler_work_on_servers_without_json_mode(provider):
    provider.json_mode = False
    lm = dspy.LM(
        "openai/fixture-model",
        api_base=provider.url + "/v1",
        api_key="test-placeholder",
        cache=False,
        num_retries=0,
    )
    graph = binary_plan()
    provider.completion = chat_reply(graph=graph)
    assert json.loads(GraphProposer(lm)("Propose a decision graph.").graph) == graph
    provider.completion = chat_reply(label="review")
    assert Labeler(lm, "Classify requests.", ["accept", "review"])("request A") == "review"


def test_preferring_large_graphs_raises_the_node_cap_unless_one_is_set():
    assert SearchConfig().prefer == "small"
    assert SearchConfig(prefer="large").max_nodes > SearchConfig().max_nodes
    assert SearchConfig(prefer="large", max_nodes=20).max_nodes == 20


def test_a_layer_of_statements_is_offered_only_where_the_budget_holds_it():
    data = task_data()
    data["labels"] = ["accept", "review", "refund", "escalate", "ignore"]
    task = Task.model_validate(data)
    small, large = SearchConfig(), SearchConfig(prefer="large")
    assert "decompose" not in move_examples(task, (), small.max_nodes)
    layer = move_examples(task, (), large.max_nodes)["decompose"]
    plan = compile_rules(layer, task.criteria())
    plan.check(task.labels, large.max_nodes, large.max_depth)
    # Every label but the fallback gets statements of its own, summed and cut.
    assert len(asked(plan, "noul")) >= len(task.labels) - 1
    assert sum(isinstance(node, Sum) for node in plan.nodes.values()) == len(task.labels) - 1
    assert layer in proposer_prompt(task, large)
    assert layer not in proposer_prompt(task, small)
    assert f"{large.max_nodes} nodes" in proposer_prompt(task, large)
