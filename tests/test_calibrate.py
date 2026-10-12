"""Calibrating a judge's probabilities, and passing an example to a person."""

import json
import re

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from test_cli import invoke, write
from test_graph import task_data

from rulesmith.calibrate import cuts, fit_cutoffs, fit_temperature, measure
from rulesmith.chat_judge import choice_confidence, score_confidence
from rulesmith.graph import Plan, Task
from rulesmith.optimize import Labels
from rulesmith.rules import compile_rules, render_rules
from rulesmith.runtime import DecisionProgram, calibrated

CHOICE = {
    "type": "choice",
    "value": "a",
    "probabilities": {"a": 0.8, "b": 0.15, "c": 0.05},
    "confidence": 0.6,
}


@given(st.floats(0.2, 5))
def test_a_temperature_reshapes_probabilities_without_changing_the_answer(temperature):
    answer = calibrated(CHOICE, temperature)
    assert answer["value"] == "a"
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    top = answer["probabilities"]["a"]
    # Above one flattens an overconfident judge; below one sharpens a timid one.
    assert (top < 0.8) if temperature > 1.001 else (top > 0.8) if temperature < 0.999 else True
    noul = calibrated({"type": "noul", "value": 0.9}, temperature)
    assert (noul["value"] < 0.9) == (temperature > 1.001) or abs(temperature - 1) < 0.001


def test_a_score_moves_with_its_calibrated_levels():
    answer = calibrated(
        {
            "type": "score",
            "value": 1.737,
            "probabilities": {"0": 0.066, "1": 0.131, "2": 0.803},
            "confidence": 0.6,
        },
        3.0,
    )
    levels = answer["probabilities"]
    assert answer["value"] == pytest.approx(sum(int(k) * p for k, p in levels.items()))
    assert answer["value"] < 1.737


def test_an_overconfident_judge_is_fitted_a_temperature_above_one():
    # Sure of every answer, right on only three in five: its certainty needs flattening.
    sure = {"a": 0.97, "b": 0.03}
    pairs = [(sure, "a")] * 3 + [(sure, "b")] * 2
    assert fit_temperature(pairs) > 1
    timid = {"a": 0.6, "b": 0.4}
    assert fit_temperature([(timid, "a")] * 20) < 1


def test_a_calibrated_question_reads_back_as_written():
    text = 'ask q = "Which?" [a: "A"] [b: "B"] calibrated 1.7\nwhen q == "a": A\nalways: B'
    plan = compile_rules(text)
    assert plan.nodes["q"].temperature == 1.7
    once = render_rules(plan)
    assert "calibrated 1.7" in once
    assert render_rules(compile_rules(once)) == once


def test_a_temperature_must_be_positive():
    with pytest.raises(ValueError, match="temperature"):
        compile_rules(
            'ask q = "Which?" [a: "A"] [b: "B"] calibrated 0\nwhen q == "a": A\nalways: B'
        )


def test_the_cli_fits_a_temperature_on_training_examples(tmp_path, provider):
    data = task_data()
    data["train"] = [
        {"id": str(i), "state": f"request {i}", "label": "accept" if i % 5 < 3 else "review"}
        for i in range(10)
    ]
    task = write(tmp_path / "task.json", data)
    plan = tmp_path / "plan.rules"
    plan.write_text(
        'ask route = "Accept or review?" [accept: "Fine."] [review: "Check it."]\n'
        'when route == "accept": accept\nalways: review'
    )
    provider.answers = {
        "route": {
            "type": "choice",
            "choice": "accept",
            "confidence": 0.95,
            "probabilities": {"accept": 0.97, "review": 0.03},
        }
    }
    output = tmp_path / "calibrated.rules"
    result = invoke(
        "calibrate",
        task,
        plan,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    fitted = compile_rules(output.read_text()).nodes["route"].temperature
    assert fitted > 1
    assert json.loads(result.stdout)["temperatures"]["route"] == fitted


def test_passing_to_a_person_costs_what_the_task_says():
    data = task_data() | {"labels": ["accept", "review", "human"]}
    data["abstain"] = {"label": "human", "cost": 0.3}
    task = Task.model_validate(data)
    program = DecisionProgram(Plan.constant("human"), None, "none")
    outcome = Labels(task)(program, task.train[0])
    assert outcome.score == pytest.approx(0.7)
    assert "person" in outcome.trace["Feedback"]


@pytest.mark.parametrize(
    "abstain,labels",
    [
        ({"label": "human", "cost": 0.3}, ["accept", "review"]),
        ({"label": "human", "cost": 1.5}, ["accept", "review", "human"]),
    ],
)
def test_an_abstain_label_must_be_a_task_label_that_costs_less_than_a_mistake(abstain, labels):
    with pytest.raises(ValidationError):
        Task.model_validate(task_data() | {"labels": labels, "abstain": abstain})


def test_no_example_may_expect_a_person():
    data = task_data() | {"labels": ["accept", "review", "human"]}
    data["abstain"] = {"label": "human", "cost": 0.3}
    data["train"][0]["label"] = "human"
    with pytest.raises(ValidationError, match="human"):
        Task.model_validate(data)


def test_the_proposer_is_told_how_to_pass_an_example_on():
    from rulesmith.optimize import SearchConfig, proposer_prompt

    data = task_data() | {"labels": ["accept", "review", "human"]}
    data["abstain"] = {"label": "human", "cost": 0.3}
    prompt = proposer_prompt(Task.model_validate(data), SearchConfig())
    assert "human" in prompt and "0.3" in prompt


def stated(request) -> float:
    """The number an example's state is, however the judge was handed it."""
    return float(re.search(r"[0-9.]+", json.dumps(request["state"])).group())


def noul_by_state(provider):
    """The judge's probability is whatever the example's state says it is, so a task can lay out
    exactly where the judge stands on each example."""

    def answer(request):
        provider.body["answers"] = {"scam": {"type": "noul", "noul": stated(request)}}

    provider.on_request = answer


def scam_task(
    positives: list[float],
    negatives: list[float],
    validation=((0.23, "genuine"),),
    calibration=(),
) -> Task:
    examples = [
        {"id": f"p{i}", "state": str(p), "label": "fraudulent"} for i, p in enumerate(positives)
    ]
    examples += [
        {"id": f"n{i}", "state": str(p), "label": "genuine"} for i, p in enumerate(negatives)
    ]
    data = task_data() | {"labels": ["fraudulent", "genuine"], "train": examples}
    data["validation"] = [
        {"id": f"v{i}", "state": str(p), "label": label} for i, (p, label) in enumerate(validation)
    ]
    data["test"] = [{"id": "t", "state": "0.99", "label": "fraudulent"}]
    data["calibration"] = [
        {"id": f"k{i}", "state": str(p), "label": label} for i, (p, label) in enumerate(calibration)
    ]
    return Task.model_validate(data)


# Training examples put the line between 0.25 and 0.3; these put it between 0.6 and 0.7.
TIMID = ([0.3, 0.35, 0.4, 0.45, 0.42], [0.1, 0.15, 0.2, 0.25, 0.22])
CALIBRATION = [(0.7, "fraudulent"), (0.75, "fraudulent"), (0.8, "fraudulent"), (0.85, "fraudulent")]
CALIBRATION += [(0.6, "genuine"), (0.55, "genuine"), (0.5, "genuine"), (0.48, "genuine")]


SCAM_RULES = 'ask scam = "Is this a scam?"\nwhen scam >= 0.5: fraudulent\nalways: genuine'


def test_a_cutoff_moves_to_where_the_judge_separates_the_labels(provider, client):
    noul_by_state(provider)
    # The judge is timid: scams sit at 0.3 to 0.45, real postings below 0.25. The rule's line at
    # 0.5 misses every scam; between 0.25 and 0.3 it gets them all.
    task = scam_task([0.3, 0.35, 0.4, 0.45, 0.42], [0.1, 0.15, 0.2, 0.25, 0.22])
    fitted, report = fit_cutoffs(compile_rules(SCAM_RULES), task, client, "fixture")
    cutoff = fitted.nodes[report[0]["rule"]].when.value
    assert 0.25 < cutoff < 0.3
    assert report[0]["kept"] and report[0]["train_before"] == 0.5 and report[0]["train_after"] == 1
    assert "scam >= " in render_rules(fitted) and "0.5" not in render_rules(fitted)


def test_a_calibration_split_is_what_the_lines_and_counts_are_fitted_on(provider, client):
    noul_by_state(provider)
    task = scam_task(*TIMID, calibration=CALIBRATION)
    fitted, report = fit_cutoffs(compile_rules(SCAM_RULES), task, client, "fixture")
    assert 0.6 < fitted.nodes[report[0]["rule"]].when.value < 0.7
    counted = measure(fitted, task, client, "fixture")
    # Eight calibration examples, none of the ten training ones.
    assert {n.measured.examples for n in counted.nodes.values() if n.type == "value"} == {8}
    assert sum(n.measured.right for n in counted.nodes.values() if n.type == "value") == 8


def test_one_example_is_not_evidence_enough_to_move_a_cutoff(provider, client):
    noul_by_state(provider)
    # Every example is on the right side of 0.5 but one scam at 0.4. A line at 0.325 would get it,
    # and nothing else supports that line: held out, that example is the only one that changes.
    task = scam_task([0.95, 0.9, 0.85, 0.8, 0.4], [0.1, 0.15, 0.2, 0.25, 0.05])
    fitted, report = fit_cutoffs(compile_rules(SCAM_RULES), task, client, "fixture")
    assert fitted.nodes[report[0]["rule"]].when.value == 0.5
    assert not report[0]["kept"] and report[0]["tried"] > 1


def test_confidence_and_threshold_cutoffs_are_fitted_too(provider, client):
    def answer(request):
        sure = stated(request)
        provider.body["answers"] = {
            "route": {
                "type": "choice",
                "choice": "accept",
                "confidence": sure,
                "probabilities": {"accept": sure, "review": 1 - sure},
            }
        }

    provider.on_request = answer
    plan = Plan.model_validate(
        {
            "nodes": {
                "route": {
                    "type": "call",
                    "inputs": {"text": {"type": "state"}},
                    "question": {
                        "type": "choice",
                        "instructions": "Accept or review?",
                        "criteria": {"accept": "Fine.", "review": "Check."},
                    },
                },
                "sure": {"type": "confidence", "source": "route"},
                "settled": {
                    "type": "threshold",
                    "source": "sure",
                    "cutoff": 0.9,
                    "above": "accept",
                    "below": "review",
                },
            },
            "outputs": {"label": {"type": "node", "node": "settled"}},
        }
    )
    # Accepted requests draw 0.6 to 0.8 from the judge, the ones to review 0.51 to 0.58; a
    # threshold at 0.9 sends everything to review.
    data = task_data() | {
        "train": [
            *[
                {"id": f"a{i}", "state": str(p), "label": "accept"}
                for i, p in enumerate([0.6, 0.7, 0.8, 0.75, 0.65])
            ],
            *[
                {"id": f"r{i}", "state": str(p), "label": "review"}
                for i, p in enumerate([0.51, 0.55, 0.58, 0.53, 0.52])
            ],
        ]
    }
    task = Task.model_validate(data)
    assert [cut.node for cut in cuts(plan)] == ["settled"]
    fitted, report = fit_cutoffs(plan, task, client, "fixture")
    assert 0.58 < fitted.nodes["settled"].cutoff < 0.6 and report[0]["tests"] == "sure"


@given(st.floats(0.2, 5))
def test_a_temperature_reshapes_how_sure_a_choice_or_a_score_reads(temperature):
    # A rule tests `.confidence`, so a fitted temperature has to reach it, by the formula the
    # judge reported it with, or calibrating a choice could never change what a rule decides.
    choice = calibrated(CHOICE, temperature)
    score = calibrated(
        {
            "type": "score",
            "value": 1.737,
            "probabilities": {"0": 0.066, "1": 0.131, "2": 0.803},
            "confidence": 0.6,
        },
        temperature,
    )
    if abs(temperature - 1) < 0.001:
        return
    assert choice["confidence"] == pytest.approx(
        choice_confidence(list(choice["probabilities"].values()))
    )
    levels = sorted(score["probabilities"], key=int)
    assert score["confidence"] == pytest.approx(
        score_confidence([score["probabilities"][level] for level in levels])
    )
    # Above one an overconfident judge reads less sure; below one a timid one reads surer.
    untempered = choice_confidence(list(CHOICE["probabilities"].values()))
    assert (choice["confidence"] < untempered) == (temperature > 1)
    # An answer its own probabilities argue against is not made sure by sharpening them.
    assert calibrated(CHOICE | {"value": "c"}, temperature)["confidence"] == 0


def test_a_fitted_temperature_reaches_the_confidence_a_rule_tests(provider, client):
    provider.answers = {
        "threat": {
            "type": "choice",
            "choice": "ATTACK",
            "confidence": choice_confidence([0.9, 0.1]),
            "probabilities": {"ATTACK": 0.9, "MOVE_BACKWARD": 0.1},
        }
    }
    rules = (
        'ask threat = "Fight or fall back?" [ATTACK: "Ammo and health allow it."]'
        ' [MOVE_BACKWARD: "They do not."]{}\n'
        'when threat.confidence >= 0.7 and threat == "ATTACK": ATTACK\n'
        "always: MOVE_FORWARD"
    )
    sure = DecisionProgram(compile_rules(rules.format("")), client, "fixture")(state="contact")
    hedged = DecisionProgram(compile_rules(rules.format(" calibrated 5")), client, "fixture")(
        state="contact"
    )
    assert sure.label == "ATTACK" and sure.answers["threat"]["confidence"] == pytest.approx(0.8)
    # Flattened by five, 0.9 against 0.1 is barely a lean, so the rule no longer acts on it.
    assert hedged.answers["threat"]["confidence"] < 0.7
    assert hedged.label == "MOVE_FORWARD"


def confidence_threshold(cutoff: float) -> Plan:
    """A plan accepting whatever its judge is at least `cutoff` sure of, from the confidence it
    reports, and sending the rest to review."""
    return Plan.model_validate(
        {
            "nodes": {
                "route": {
                    "type": "call",
                    "inputs": {"text": {"type": "state"}},
                    "question": {
                        "type": "choice",
                        "instructions": "Accept or review?",
                        "criteria": {"accept": "Fine.", "review": "Check."},
                    },
                },
                "sure": {"type": "confidence", "source": "route"},
                "settled": {
                    "type": "threshold",
                    "source": "sure",
                    "cutoff": cutoff,
                    "above": "accept",
                    "below": "review",
                },
            },
            "outputs": {"label": {"type": "node", "node": "settled"}},
        }
    )


def test_a_confidence_cutoff_is_fitted_on_the_confidence_the_rule_compares(provider, client):
    # The judge reports confidence the way TypeSafe and the chat judge do, how far its leading
    # option stands above even odds, not the leading option's probability, so on two options
    # p = 0.68 reads 0.36 and p = 0.71 reads 0.42.
    def answer(request):
        p = stated(request)
        provider.body["answers"] = {
            "route": {
                "type": "choice",
                "choice": "accept",
                "confidence": choice_confidence([p, 1 - p]),
                "probabilities": {"accept": p, "review": 1 - p},
            }
        }

    provider.on_request = answer
    review = [0.55, 0.56, 0.57, 0.575, 0.58, 0.585, 0.59, 0.595, 0.6, 0.605]
    review += [0.61, 0.615, 0.62, 0.625, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68]
    accept = [0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.765, 0.77, 0.78, 0.8, 0.82]
    accept += [0.84, 0.85, 0.86, 0.88, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95]
    data = task_data() | {
        "train": [
            *[{"id": f"a{i}", "state": str(p), "label": "accept"} for i, p in enumerate(accept)],
            *[{"id": f"r{i}", "state": str(p), "label": "review"} for i, p in enumerate(review)],
        ]
    }
    task = Task.model_validate(data)
    fitted, report = fit_cutoffs(confidence_threshold(0.95), task, client, "fixture")
    # Every reviewed request reads at most 0.36 and every accepted one at least 0.42, so the line
    # belongs between them and gets all 41 right. Lines drawn between the leading probabilities
    # instead start at 0.555, which sends the eight accepted requests reading 0.42 to 0.54 to
    # review: 33 of 41.
    assert 0.36 < fitted.nodes["settled"].cutoff < 0.42
    assert report[0]["train_before"] == pytest.approx(20 / 41)
    assert report[0]["train_after"] == 1


def test_the_cli_writes_fitted_cutoffs_and_reports_the_validation_change(tmp_path, provider):
    noul_by_state(provider)
    task = scam_task(
        [0.3, 0.35, 0.4, 0.45, 0.42],
        [0.1, 0.15, 0.2, 0.25, 0.22],
        validation=[(0.38, "fraudulent"), (0.12, "genuine"), (0.6, "fraudulent")],
    )
    task_file = write(tmp_path / "task.json", task.model_dump(mode="json"))
    plan = tmp_path / "plan.rules"
    plan.write_text(SCAM_RULES)
    output = tmp_path / "calibrated.rules"
    result = invoke(
        "calibrate",
        task_file,
        plan,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    fitted = compile_rules(output.read_text())
    cutoff = fitted.nodes[
        [n for n, node in fitted.nodes.items() if node.when is not None][0]
    ].when.value
    assert 0.25 < cutoff < 0.3
    printed = json.loads(result.stdout)
    assert printed["cutoffs"][0]["to"] == cutoff
    # Validation is reported, never fitted on: two of three were right before, all three after.
    assert printed["validation"] == {"before": pytest.approx(2 / 3), "after": 1}
    assert printed["fitted_on"] == "train+validation"


def test_the_cli_fits_on_a_calibration_split_and_still_reports_validation(tmp_path, provider):
    noul_by_state(provider)
    task = scam_task(
        *TIMID, validation=[(0.65, "fraudulent"), (0.12, "genuine")], calibration=CALIBRATION
    )
    task_file = write(tmp_path / "task.json", task.model_dump(mode="json"))
    plan = tmp_path / "plan.rules"
    plan.write_text(SCAM_RULES)
    output = tmp_path / "calibrated.rules"
    result = invoke(
        "calibrate",
        task_file,
        plan,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--output",
        output,
    )
    assert result.returncode == 0, result.stderr
    printed = json.loads(result.stdout)
    assert printed["fitted_on"] == "calibration"
    assert 0.6 < printed["cutoffs"][0]["to"] < 0.7
    assert printed["validation"] == {"before": 1, "after": 1}
    assert "calibrated 4/4 of 8" in output.read_text()
    scored = invoke(
        "evaluate",
        task_file,
        output,
        "--backend",
        "ruling",
        "--base-url",
        provider.url,
        "--split",
        "calibration",
        "--output",
        tmp_path / "scored.json",
    )
    assert scored.returncode == 0, scored.stderr
    assert json.loads((tmp_path / "scored.json").read_text())["score"] == 1


def test_at_the_real_rate_a_line_is_not_lowered_to_catch_scams_at_the_cost_of_false_flags(
    provider, client
):
    noul_by_state(provider)
    # Ten scams sit at 0.30 to 0.45 and six of the ten genuine postings among them. Sampled half
    # and half, a line near 0.25 catches every scam for six false flags and wins; at one scam in
    # twenty, each genuine posting it flags costs nineteen times what a missed scam does, and the
    # line stays where it was.
    positives = [0.30, 0.31, 0.33, 0.34, 0.35, 0.37, 0.40, 0.42, 0.43, 0.45]
    negatives = [0.05, 0.1, 0.15, 0.2, 0.36, 0.38, 0.39, 0.41, 0.44, 0.46]
    balanced = scam_task(positives, negatives)
    fitted, _ = fit_cutoffs(compile_rules(SCAM_RULES), balanced, client, "fixture")
    lowered = next(n for n in fitted.nodes.values() if n.when is not None).when.value
    assert lowered < 0.5
    real = Task.model_validate(
        balanced.model_dump() | {"prevalence": {"fraudulent": 0.05, "genuine": 0.95}}
    )
    fitted, report = fit_cutoffs(compile_rules(SCAM_RULES), real, client, "fixture")
    assert next(n for n in fitted.nodes.values() if n.when is not None).when.value == 0.5
    assert not report[0]["kept"]
