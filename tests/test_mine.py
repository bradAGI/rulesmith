"""Rules mined from the data, offered to search before any proposal is written."""

import re

from test_graph import call

from rulesmith.graph import Plan, Task
from rulesmith.mine import evidence, fields, mine
from rulesmith.rules import compile_rules, render_rules
from rulesmith.runtime import DecisionProgram


def reviews(count: int = 60) -> list[dict]:
    """A rating settles most reviews; a three-star one could go either way."""
    rows = []
    for index in range(count):
        rating = index % 5 + 1
        # Three-star labels vary with nothing in the fields, department included.
        mixed = "yes" if index // 10 % 2 else "no"
        label = "yes" if rating >= 4 else "no" if rating <= 2 else mixed
        department = ["Tops", "Dresses"][index % 2]
        state = {"rating": rating, "department": department, "review": f"review {index}"}
        rows.append({"id": str(index), "state": state, "label": label})
    return rows


def task(rows: list[dict] | None = None) -> Task:
    rows = rows or reviews()
    return Task(
        description="Would the reviewer recommend it?",
        labels=["no", "yes"],
        train=rows[:40],
        validation=rows[40:50],
        test=rows[50:],
    )


def test_the_field_that_settles_most_cases_becomes_rules_ahead_of_the_seed():
    seed = Plan.constant("no")
    mined = mine(task(), seed)
    program = DecisionProgram(mined, None, "none")
    # Four and five stars are settled by a rule; three stars fall through to the seed.
    assert program({"rating": 5, "department": "Tops"}).label == "yes"
    assert program({"rating": 1, "department": "Tops"}).label == "no"
    assert program({"rating": 3, "department": "Tops"}).label == "no"
    assert "rating" in render_rules(mined)


def test_a_mined_graph_asks_the_judge_only_for_what_the_rules_leave(provider, client):
    provider.answers = {"eligible": {"type": "noul", "noul": 0.9}}
    seed = compile_rules('ask eligible = "Recommended?"\nwhen eligible >= 0.5: yes\nalways: no')
    program = DecisionProgram(mine(task(), seed), client, "fixture")
    program({"rating": 5, "department": "Tops"})
    assert provider.requests == []
    program({"rating": 3, "department": "Tops"})
    assert len(provider.requests) == 1


def test_a_mined_graph_reads_back_as_the_same_rules():
    mined = mine(task(), Plan.constant("no"))
    once = render_rules(mined)
    assert render_rules(compile_rules(once)) == once
    mined.check(["no", "yes"], 16, 8)


def test_nothing_is_mined_where_no_field_decides():
    # Labels alternate with nothing in the fields, so no test is anywhere near pure.
    labels = ["no", "yes"]
    noisy = [
        {"id": str(i), "state": {"rating": i % 5 + 1, "review": f"r{i}"}, "label": labels[i % 2]}
        for i in range(60)
    ]
    assert mine(task(noisy), Plan.constant("no")) is None
    text = [
        {"id": str(i), "state": f"review {i}", "label": ["no", "yes"][i % 2]} for i in range(60)
    ]
    assert mine(task(text), Plan.constant("no")) is None


def test_a_seed_with_no_rule_form_is_left_alone():
    merged = Plan.model_validate(
        {
            "nodes": {
                "a": call(observation={"type": "state"}),
                "b": {"type": "merge", "inputs": {"a": {"type": "node", "node": "a"}}},
            },
            "outputs": {
                "label": {"type": "literal", "value": "no"},
                "extra": {"type": "node", "node": "b"},
            },
        }
    )
    assert mine(task(), merged) is None


def test_fields_are_summarized_by_what_values_they_take():
    summary = fields(task().train)
    assert summary["rating"] == "numbers from 1 to 5"
    assert summary["department"] == "one of Dresses, Tops"
    # Text that differs on every example is no category, so it is known by how long it is.
    assert summary["review"] == "text of 8 to 9 characters"


def test_search_starts_from_the_mined_graph_when_it_scores_better(client):
    from rulesmith.optimize import SearchConfig, optimize_plan

    seed = Plan.constant("no")
    # A proposer that only ever offers the seed back: whatever search keeps, mining found.
    config = SearchConfig(max_metric_calls=30, reflection_batch=3, mine=True)
    plan, report = optimize_plan(task(), seed, client, "none", lambda _: "always: no", config)
    assert report["mined"]["used"] and report["mined"]["rules"] >= 2
    assert DecisionProgram(plan, None, "none")({"rating": 5, "department": "Tops"}).label == "yes"


def test_the_proposer_is_told_what_each_field_holds():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    prompt = proposer_prompt(task(), SearchConfig())
    assert "numbers from 1 to 5" in prompt and "one of Dresses, Tops" in prompt


def postings(count: int = 80) -> list[dict]:
    """A scam lacks a logo while remote, or has one while not: every single field is a coin
    flip, at any step of a rule list, and only the two together decide."""
    rows = []
    for index in range(count):
        logo, remote = index % 2, index // 2 % 2
        label = "scam" if logo != remote else "real"
        state = {"logo": logo, "remote": remote, "title": f"job {index}"}
        rows.append({"id": str(index), "state": state, "label": label})
    return rows


def test_two_fields_together_can_settle_what_neither_settles_alone():
    rows = postings()
    plan = mine(task_of(rows, ["real", "scam"]), Plan.constant("real"))
    program = DecisionProgram(plan, None, "none")
    assert program({"logo": 0, "remote": 1, "title": "x"}).label == "scam"
    assert program({"logo": 1, "remote": 1, "title": "x"}).label == "real"
    assert program({"logo": 1, "remote": 0, "title": "x"}).label == "scam"
    once = render_rules(plan)
    assert render_rules(compile_rules(once)) == once


def test_an_empty_text_field_can_settle_a_case():
    rows = [
        {
            "id": str(i),
            "state": {
                "profile": "" if i % 3 == 0 else f"We are a firm founded in {1900 + i}.",
                "title": f"job {i}",
            },
            "label": "scam" if i % 3 == 0 else "real",
        }
        for i in range(60)
    ]
    plan = mine(task_of(rows, ["real", "scam"]), Plan.constant("real"))
    assert "length of profile" in render_rules(plan)
    assert DecisionProgram(plan, None, "none")({"profile": "", "title": "x"}).label == "scam"
    # A length is never negative, so the room around what was seen stops at zero.
    (read,) = [node for node in plan.nodes.values() if node.type == "read"]
    assert read.bounds[0] == 0


def task_of(rows: list[dict], labels: list[str]) -> Task:
    return Task(
        description="Which?",
        labels=labels,
        train=rows[: len(rows) * 2 // 3],
        validation=rows[len(rows) * 2 // 3 : len(rows) * 5 // 6],
        test=rows[len(rows) * 5 // 6 :],
    )


def test_an_empty_string_is_never_offered_as_a_category():
    # Blank fields are common in real data; a category must be a non-empty label.
    rows = [
        {
            "id": str(i),
            "state": {"experience": ["", "Entry", "Senior"][i % 3], "n": i},
            "label": "yes" if i % 3 == 0 else "no",
        }
        for i in range(60)
    ]
    plan = mine(task(rows), Plan.constant("no"))
    reads = [node for node in (plan.nodes.values() if plan else []) if node.type == "read"]
    assert all("" not in (node.categories or []) for node in reads)


def noisy_reviews(count: int = 60) -> list[dict]:
    """Ratings settle reviews nine times in ten, and an age that means nothing covers almost all
    of them: a rule over age is right as often as guessing, one over rating far more often."""
    rows = []
    for index in range(count):
        rating = index % 5 + 1
        label = "yes" if rating >= 4 else "no"
        if index % 10 == 9:
            label = "no" if label == "yes" else "yes"
        state = {"rating": rating, "age": 20 + index % 50, "review": f"review {index}"}
        rows.append({"id": str(index), "state": state, "label": label})
    return rows


def test_evidence_ranks_rules_by_how_far_they_beat_chance_and_reads_as_rules():
    from rulesmith.mine import evidence

    found = evidence(task(noisy_reviews()).train)
    assert found and "rating" in found[0] and "age" not in found[0]
    # Each line is a rule the language reads, with how often it held on the examples it covers.
    rule, counted = found[0].split("  # ")
    compile_rules(f"read rating in [1, 5]\nread age in [0, 100]\n{rule}\nalways: no")
    assert re.fullmatch(r"right on \d+ of the \d+ training examples it covers", counted)
    # Counting the same examples gives the same evidence, so two searches see the same thing.
    assert evidence(task(noisy_reviews()).train) == found


def test_the_proposer_is_shown_the_evidence_only_when_asked():
    # Counted on a sampled training set, the evidence over-fits its rate: on the jobs run it
    # had search drop the judge and lose on the seeds that had been best. So it is opt-in.
    from rulesmith.optimize import SearchConfig, proposer_prompt

    found = evidence(task(noisy_reviews()).train)
    quiet = proposer_prompt(task(noisy_reviews()), SearchConfig())
    assert "Fields of the inputs" in quiet
    assert "counted on the training examples" not in quiet
    assert not any(line in quiet for line in found)
    told = proposer_prompt(task(noisy_reviews()), SearchConfig(evidence=True))
    assert "counted on the training examples" in told
    assert all(line in told for line in found)
