"""Fitting how sure a judge should read, and measuring how often each rule is right. A model
that says 0.97 and is right three times in five is overconfident, and a rule testing
`confidence >= 0.9` on it means nothing. One temperature per question, fitted on labeled examples,
makes its probabilities say what they claim. A rule claims nothing at all until it is measured:
counted on labeled examples, its decisions are as sure as it was right, and inputs that reach it
far more or less often than those did are no longer the inputs it was measured on. Where a rule
draws its line on a judgment, `scam >= 0.5`, is a number the proposer guessed; fitted here, it is
the line between the judge's answers that gets the most labeled examples right and holds up on
examples held out of the fit."""

import math
from collections import Counter
from dataclasses import dataclass

from rulesmith.graph import (
    All,
    Any,
    Call,
    Choice,
    Coalesce,
    Compare,
    Condition,
    Confidence,
    Measured,
    NodeInput,
    Not,
    Noul,
    Plan,
    Score,
    Sum,
    Task,
    Term,
    Threshold,
    Value,
)
from rulesmith.runtime import (
    DecisionProgram,
    Models,
    Prices,
    calibrated,
)

# Temperatures tried, evenly spaced in log from a fifth to five: calibration corrects a judge's
# confidence, it does not invert it.
TEMPERATURES = [math.exp(math.log(0.2) + index * (math.log(25) / 80)) for index in range(81)]


def fit_temperature(pairs: list[tuple[dict[str, float], str]]) -> float:
    """The temperature under which these answers give the right labels the most probability,
    judged by log loss. `pairs` holds each answer's probabilities and the label that was right."""

    def loss(temperature: float) -> float:
        total = 0.0
        for probabilities, right in pairs:
            tempered = calibrated(
                {"type": "choice", "value": right, "probabilities": probabilities}, temperature
            )
            total -= math.log(max(tempered["probabilities"][right], 1e-12))
        return total

    return round(min(TEMPERATURES, key=loss), 3)


def calibrate(
    plan: Plan, task: Task, client, model: str, models: Models | None = None
) -> tuple[Plan, dict[str, float]]:
    """The plan with a temperature fitted for every choice question over the task's labels, from
    how its answers on the examples it is fitted on compare with their labels. Other questions
    have no labels to be right about and keep theirs."""
    fitted_nodes = {
        name: node
        for name, node in plan.nodes.items()
        if isinstance(node, Call)
        and isinstance(node.question, Choice)
        and set(node.question.criteria) <= set(task.labels)
    }
    raw = plan.model_copy(
        update={
            "nodes": plan.nodes
            | {
                name: node.model_copy(update={"temperature": 1})
                for name, node in fitted_nodes.items()
            }
        }
    )
    program = DecisionProgram(raw, client, model, models=models)
    pairs: dict[str, list] = {name: [] for name in fitted_nodes}
    for example in task.fitting():
        answers = program(state=example.state).answers
        for name, node in fitted_nodes.items():
            if name in answers and example.label in node.question.criteria:
                pairs[name].append((answers[name]["probabilities"], example.label))
    temperatures = {name: fit_temperature(found) for name, found in pairs.items() if found}
    nodes = plan.nodes | {
        name: plan.nodes[name].model_copy(update={"temperature": temperature})
        for name, temperature in temperatures.items()
    }
    return Plan.model_validate(plan.model_copy(update={"nodes": nodes}).model_dump()), temperatures


@dataclass(frozen=True)
class Cut:
    """One numeric line a rule draws on a judgment: a `>=` or `<` in a rule's condition, reached
    by `path` through its and/or/not groups, or a threshold node's cutoff (`path` None). `tests`
    is the judgment or confidence the line is drawn on."""

    node: str
    path: tuple[int, ...] | None
    tests: str

    def value(self, plan: Plan) -> float:
        if self.path is None:
            return plan.nodes[self.node].cutoff
        return descend(plan.nodes[self.node].when, self.path).value

    def moved(self, plan: Plan, value: float) -> Plan:
        """The plan with this line drawn at `value` and nothing else changed."""
        data = plan.model_dump()
        node = data["nodes"][self.node]
        if self.path is None:
            node["cutoff"] = value
        else:
            condition = node["when"]
            for index in self.path:
                condition = (
                    condition["conditions"][index]
                    if "conditions" in condition
                    else condition["condition"]
                )
            condition["value"] = value
        return Plan.model_validate(data)


def descend(condition: Condition, path: tuple[int, ...]) -> Compare:
    for index in path:
        condition = (
            condition.conditions[index]
            if isinstance(condition, (All, Any))
            else condition.condition
        )
    return condition


def judgments(plan: Plan) -> set[str]:
    """The nodes whose numbers come from the judge: a probability, a score's expected level, how
    sure a judgment was, and a sum with any of those among its terms. A line drawn on one of
    them is about the judge and may be fitted; a line on a reading is what the rule says about
    the input and stays where it was written."""
    judged = {
        name
        for name, node in plan.nodes.items()
        if (isinstance(node, Call) and isinstance(node.question, (Noul, Score)))
        or isinstance(node, Confidence)
    }
    return judged | {
        name
        for name, node in plan.nodes.items()
        if isinstance(node, Sum) and any(term.node in judged for term in node.terms)
    }


def cuts(plan: Plan) -> list[Cut]:
    """Every line a rule draws on a number the judge produced: a probability, a score's expected
    level, or how sure a judgment was. Lines drawn on readings are what the rule says about the
    input, not about the judge, and stay where they were written."""
    judged = judgments(plan)

    def compares(condition: Condition, path: tuple[int, ...]):
        if isinstance(condition, Compare):
            if condition.op != "eq" and condition.node in judged:
                yield path, condition.node
        elif isinstance(condition, Not):
            yield from compares(condition.condition, (*path, 0))
        elif isinstance(condition, (All, Any)):
            for index, child in enumerate(condition.conditions):
                yield from compares(child, (*path, index))

    found = []
    for name, node in plan.nodes.items():
        if isinstance(node, Threshold) and node.source in judged:
            found.append(Cut(name, None, node.source))
        if getattr(node, "when", None) is not None:
            found += [Cut(name, path, tests) for path, tests in compares(node.when, ())]
    return found


FOLDS = 5


def fit_cutoffs(
    plan: Plan,
    task: Task,
    client,
    model: str,
    models: Models | None = None,
    prices: Prices | None = None,
) -> tuple[Plan, list[dict]]:
    """The plan with every cut moved to where it gets the most fitting examples right, one cut
    at a time, holding the questions and the rules' words still. A move is kept only when it
    also wins held out: the fitting examples are dealt into folds, the best line is chosen
    without each fold in turn and scored on it, and those scores together must beat the line as
    it stands. One example on the far side of a line is not evidence to move it; a few are.
    The report has a row per line: where it was and where it ended, how many places were tried,
    the training accuracy just before and after its move, and whether it moved. A task with a
    prevalence counts each example as the real traffic it stands for, so the line is placed for
    the rate the graph will meet, not the rate the examples were sampled at. With `prices`, each
    example also pays for the calls it made, so a line on how sure a small judge was is placed
    where escalating to a dearer one pays, not wherever escalating is never worse."""
    examples = task.fitting()
    lines = cuts(plan)
    if not examples or not lines:
        return plan, []
    weights = task.weights()
    weight = [weights[e.label] for e in examples]
    total = sum(weight)

    prices = prices or Prices()

    def right(candidate: Plan) -> list[float]:
        decided = outcomes(candidate, examples, client, model, models)
        return [
            w * (record["correct"] - prices.charge(record["calls"]))
            for w, record in zip(weight, decided, strict=True)
        ]

    seen = observed(plan, examples, client, model, models)
    started = {cut: cut.value(plan) for cut in lines}
    tried = dict.fromkeys(lines, 0)
    after = right(plan)
    change: dict[Cut, tuple[list[float], list[float]]] = {}
    for _ in range(3):
        moved = False
        for cut in lines:
            current = cut.value(plan)
            candidates = midpoints(seen.get(cut.tests, []))
            if not candidates:
                continue
            tried[cut] += len(candidates)
            scores = {value: right(cut.moved(plan, value)) for value in candidates}
            scores[current] = after
            best = choose(scores, current)
            if best != current and held_out_wins(scores, current, len(examples)):
                change[cut] = (after, scores[best])
                plan, after, moved = cut.moved(plan, best), scores[best], True
        if not moved:
            break
    report = [
        {
            "rule": cut.node,
            "tests": cut.tests,
            "from": started[cut],
            "to": cut.value(plan),
            "tried": tried[cut],
            "train_before": sum(change.get(cut, (after, after))[0]) / total,
            "train_after": sum(change.get(cut, (after, after))[1]) / total,
            "kept": cut.value(plan) != started[cut],
        }
        for cut in lines
        if tried[cut]
    ]
    return plan, report


@dataclass(frozen=True)
class Target:
    """The rule whose label a sum is fitted to tell apart: the first line a rule draws on the
    sum, and the label that rule gives above it (or below it, for a `<`)."""

    rule: str
    path: tuple[int, ...] | None
    label: str
    above: bool


def targets(plan: Plan) -> dict[str, Target]:
    judged = judgments(plan)
    found: dict[str, Target] = {}
    for name, node in plan.nodes.items():
        if isinstance(node, Threshold) and isinstance(plan.nodes[node.source], Sum):
            found.setdefault(node.source, Target(name, None, node.above, True))
        if not isinstance(node, Value) or node.when is None:
            continue
        for cut in cuts(plan):
            compared = cut.tests
            if (
                cut.node != name
                or not isinstance(plan.nodes[compared], Sum)
                or compared not in judged
            ):
                continue
            op = descend(node.when, cut.path).op
            found.setdefault(compared, Target(name, cut.path, node.value, op in ("ge", "gt")))
    return found


def logistic(
    rows: list[list[float]], hits: list[bool], weight: list[float]
) -> tuple[list[float], float]:
    """Weights and a constant that make the rows' sum largest where `hits` holds, by gradient
    descent on the weighted log loss with a little shrinkage so a term that never varies does
    not run away."""
    count = len(rows[0])
    coefficients, constant = [0.0] * count, 0.0
    total = sum(weight)
    for _ in range(500):
        gradient, shift = [0.0] * count, 0.0
        for row, hit, w in zip(rows, hits, weight, strict=True):
            logit = constant + sum(c * x for c, x in zip(coefficients, row, strict=True))
            error = w * (1 / (1 + math.exp(-max(-30.0, min(30.0, logit)))) - hit)
            shift += error
            for index, x in enumerate(row):
                gradient[index] += error * x
        coefficients = [
            c - (g / total + 0.001 * c) for c, g in zip(coefficients, gradient, strict=True)
        ]
        constant -= shift / total
    return coefficients, constant


@dataclass(frozen=True)
class Fit:
    """One sum's terms on the fitting examples, ready to be fitted on any subset of them."""

    plan: Plan
    name: str
    target: Target
    rows: dict[int, list[float]]
    hits: dict[int, bool]
    weight: list[float]

    def on(self, indices: list[int]) -> Plan:
        """The plan with the sum reweighted by a fit on these examples and the rule's line moved
        to the fit's own line at zero, above which the label is more likely than not."""
        coefficients, constant = logistic(
            [self.rows[i] for i in indices],
            [self.hits[i] for i in indices],
            [self.weight[i] for i in indices],
        )
        node = self.plan.nodes[self.name]
        terms = [
            Term(node=term.node, weight=coefficient)
            for term, coefficient in zip(node.terms, coefficients, strict=True)
        ]
        reweighted = node.model_copy(update={"terms": terms, "constant": constant})
        plan = self.plan.model_copy(update={"nodes": self.plan.nodes | {self.name: reweighted}})
        line = Cut(self.target.rule, self.target.path, self.name)
        return line.moved(Plan.model_validate(plan.model_dump()), 0.0)


def fit_weights(
    plan: Plan,
    task: Task,
    client,
    model: str,
    models: Models | None = None,
    prices: Prices | None = None,
) -> tuple[Plan, list[dict]]:
    """The plan with every sum of judgments a rule draws a line on reweighted to tell that rule's
    label apart: a logistic fit of the sum's terms to whether the example carries the label, with
    the rule's line moved to the fit's own line at zero. Like a line, a fit is kept only when it
    wins held out, fitted without each fold in turn and scored on it. The report has a row per
    sum: its weights before and after, the training accuracy around the change, and whether it
    was kept. Run before `fit_cutoffs`, which then places every other line on the new sum."""
    examples = task.fitting()
    found = targets(plan)
    if not examples or not found:
        return plan, []
    weights = task.weights()
    weight = [weights[e.label] for e in examples]
    total = sum(weight)
    prices = prices or Prices()

    def right(candidate: Plan) -> list[float]:
        decided = outcomes(candidate, examples, client, model, models)
        return [
            w * (record["correct"] - prices.charge(record["calls"]))
            for w, record in zip(weight, decided, strict=True)
        ]

    program = DecisionProgram(plan, client, model, models=models)
    values = [program(state=example.state).node_values for example in examples]
    after = right(plan)
    report = []
    for name, target in found.items():
        terms = [term.node for term in plan.nodes[name].terms]
        usable = [i for i, seen in enumerate(values) if all(term in seen for term in terms)]
        if len(usable) < 2:
            continue
        rows = {i: [values[i][term] for term in terms] for i in usable}
        hits = {i: (examples[i].label == target.label) == target.above for i in usable}
        fit = Fit(plan, name, target, rows, hits, weight)
        folds = min(FOLDS, len(usable))
        held_out = 0.0
        for fold in range(folds):
            scored = right(fit.on([i for i in usable if i % FOLDS != fold]))
            held_out += sum(scored[i] for i in usable if i % FOLDS == fold)
        kept = held_out > sum(after[i] for i in usable)
        before, written, candidate = after, terms_of(plan.nodes[name]), fit.on(usable)
        if kept:
            plan, after = candidate, right(candidate)
        report.append(
            {
                "sum": name,
                "from": written,
                "to": terms_of(candidate.nodes[name]),
                "train_before": sum(before) / total,
                "train_after": sum(after) / total,
                "kept": kept,
            }
        )
    return plan, report


def terms_of(node: Sum) -> dict[str, float]:
    return {term.node: term.weight for term in node.terms} | {"constant": node.constant}


def observed(
    plan: Plan, examples: list, client, model: str, models: Models | None = None
) -> dict[str, list[float]]:
    """The numbers each judgment and confidence took on the examples, the values a line can be
    drawn between."""
    program = DecisionProgram(plan, client, model, models=models)
    seen: dict[str, list[float]] = {}
    judged = judgments(plan)
    for example in examples:
        prediction = program(state=example.state)
        answers, values = prediction.answers, prediction.node_values
        for name in judged:
            node = plan.nodes[name]
            # The confidence the judge reported, which is what a rule compares, not the
            # probability of its answer: on two options, 0.68 reads 0.36.
            if isinstance(node, Confidence):
                if "confidence" in answers.get(node.source, {}):
                    seen.setdefault(name, []).append(answers[node.source]["confidence"])
            elif name in values:
                seen.setdefault(name, []).append(values[name])
    return seen


def midpoints(values: list[float]) -> list[float]:
    """A line between each pair of neighboring values, rounded where rounding still separates
    them: any line between the same two answers decides the same examples, and a short number
    reads as a choice rather than an accident."""
    distinct = sorted(set(values))
    found = []
    for low, high in zip(distinct, distinct[1:], strict=False):
        exact = (low + high) / 2
        short = round(exact, 4)
        found.append(short if low < short < high else exact)
    return found


def correct(
    plan: Plan, examples: list, client, model: str, models: Models | None = None
) -> list[bool]:
    return [record["correct"] for record in outcomes(plan, examples, client, model, models)]


def choose(scores: dict[float, list[float]], current: float, fold: int | None = None) -> float:
    """The line with the most right answers, outside `fold` if one is named; among equals, the
    one nearest to where the line stands, so a tie never moves it."""

    def total(value: float) -> int:
        return sum(right for index, right in enumerate(scores[value]) if index % FOLDS != fold)

    return max(scores, key=lambda value: (total(value), -abs(value - current)))


def held_out_wins(scores: dict[float, list[float]], current: float, count: int) -> bool:
    """Whether choosing the line without each fold and scoring it on that fold beats, over all the
    folds, the line as it stands."""
    folds = min(FOLDS, count)
    held_out = 0
    for fold in range(folds):
        chosen = choose(scores, current, fold)
        held_out += sum(
            right for index, right in enumerate(scores[chosen]) if index % FOLDS == fold
        )
    return held_out > sum(scores[current])


def outcomes(
    plan: Plan, examples: list, client, model: str, models: Models | None = None
) -> list[dict]:
    """Each labeled example as the plan decided it: the node that decided, the judgments asked,
    and whether the answer was one the example accepts."""
    program = DecisionProgram(plan, client, model, models=models)
    found = []
    for example in examples:
        prediction = dict(program(state=example.state))
        found.append(
            {
                "label": prediction.get("label"),
                "decided_by": prediction.get("decided_by"),
                "calls": prediction["calls"],
                "correct": prediction.get("label") in (example.label, *example.accepted),
            }
        )
    return found


def tally(records: list[dict]) -> dict[str | None, tuple[int, int]]:
    """For each node that decided an answer, how many of its answers were right and how many it
    gave. A constant graph decides with no node, under None."""
    counts: dict[str | None, tuple[int, int]] = {}
    for record in records:
        if record.get("label") is None:
            continue
        right, decided = counts.get(record["decided_by"], (0, 0))
        counts[record["decided_by"]] = (right + bool(record["correct"]), decided + 1)
    return counts


def rules(plan: Plan) -> list[str]:
    """The rules the label is decided by: every value or threshold in its list of fallbacks."""
    source = plan.outputs["label"]
    pending, found = [source.node] if isinstance(source, NodeInput) else [], []
    while pending:
        name = pending.pop(0)
        node = plan.nodes[name]
        if isinstance(node, Coalesce):
            pending = list(node.sources) + pending
        elif isinstance(node, (Value, Threshold)):
            found.append(name)
    return found


def measure(plan: Plan, task: Task, client, model: str, models: Models | None = None) -> Plan:
    """The plan with every rule that decides its label counted on the task's calibration
    examples, or without those on its labeled training and validation examples. A rule decides
    only some of them, so it is counted on all it can be; the training and validation examples
    are ones search has seen, so those counts flatter a searched graph."""
    examples = task.fitting()
    if not task.calibration:
        examples += [e for e in task.validation if e.label is not None]
    if not examples or not rules(plan):
        return plan
    counts = tally(outcomes(plan, examples, client, model, models))
    nodes = plan.nodes | {
        name: plan.nodes[name].model_copy(
            update={
                "measured": Measured(
                    right=counts.get(name, (0, 0))[0],
                    decided=counts.get(name, (0, 0))[1],
                    examples=len(examples),
                )
            }
        )
        for name in rules(plan)
    }
    return Plan.model_validate(plan.model_copy(update={"nodes": nodes}).model_dump())


def unmeasured(plan: Plan) -> Plan:
    """The plan without its rules' counts, which describe this graph and no rewrite of it."""
    nodes = {
        name: node.model_copy(update={"measured": None})
        if isinstance(node, (Value, Threshold))
        else node
        for name, node in plan.nodes.items()
    }
    return plan.model_copy(update={"nodes": nodes})


def binomial_p(count: int, trials: int, share: float) -> float:
    """Exact two-sided binomial test: the chance, over `trials` examples each reaching a rule
    with probability `share`, of a count no likelier than `count`."""
    if share in (0, 1):
        return float(count == trials * share)

    def log_chance(k: int) -> float:
        ways = math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
        return ways + k * math.log(share) + (trials - k) * math.log1p(-share)

    # A relative tolerance, as exact tests use, so a tie with the observed count is not lost
    # to rounding.
    observed = log_chance(count) + 1e-7
    return min(1.0, sum(math.exp(c) for k in range(trials + 1) if (c := log_chance(k)) <= observed))


def drift(plan: Plan, records: list[dict], alpha: float) -> list[dict]:
    """Every measured rule these examples reach at a rate its measurement makes unlikely, below
    `alpha`: the inputs have moved from the ones it was counted on, and so may how often it is
    right."""
    reached = Counter(record.get("decided_by") for record in records)
    flagged = []
    for name, node in plan.nodes.items():
        measured = getattr(node, "measured", None)
        if measured is None:
            continue
        p = binomial_p(reached[name], len(records), measured.share)
        if p < alpha:
            flagged.append(
                {
                    "rule": name,
                    "measured_share": measured.share,
                    "share": reached[name] / len(records),
                    "p": p,
                }
            )
    return flagged


def at_prevalence(task: Task, records: list[dict]) -> dict | None:
    """Accuracy, and each label's precision and recall, as they would be on real traffic: each
    scored example counted as the share of it that its label stands for. None for a task with no
    prevalence, whose examples are taken to be sampled as they come."""
    if task.prevalence is None:
        return None
    weights = task.weights()
    scored = [(r["expected"], r.get("label"), weights[r["expected"]]) for r in records]
    total = sum(w for _, _, w in scored)
    labels = {}
    for label in task.labels:
        hit = sum(w for e, a, w in scored if e == label and a == label)
        answered = sum(w for _, a, w in scored if a == label)
        present = sum(w for e, _, w in scored if e == label)
        labels[label] = {
            "precision": hit / answered if answered else None,
            "recall": hit / present if present else None,
        }
    return {
        "prevalence": task.prevalence,
        "accuracy": sum(w for e, a, w in scored if e == a) / total,
        "labels": labels,
    }
