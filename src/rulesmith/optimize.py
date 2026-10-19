"""GEPA searches the graph language; the evaluator owns the task contract."""

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import Annotated, Literal, Protocol

import dspy
import gepa
from dspy.utils.exceptions import AdapterParseError
from gepa.core.adapter import EvaluationBatch, GEPAAdapter
from pydantic import Field, ValidationError, model_validator
from typesafe_sdk import TypeSafeClient

from rulesmith.calibrate import tally, unmeasured
from rulesmith.graph import Example, Measured, Name, Plan, StrictModel, Task
from rulesmith.mine import columns, evidence, fields, mine
from rulesmith.rules import RuleError, compile_rules, parts, quoted, whole
from rulesmith.runtime import (
    DecisionProgram,
    ExecutionConfig,
    Models,
    Prices,
    Remembered,
    payer,
    undecided,
)


def text_adapter() -> dspy.ChatAdapter:
    """Plain-text fields work on any chat endpoint; JSON mode is not universal (LM Studio)."""
    return dspy.ChatAdapter(use_json_adapter_fallback=False)


FREE = "free"
# The ways a proposal may revise a part, each named so a run can say which were tried and kept.
# Every searched graph once came back as read nodes and one choice: the language's other
# judgments were described in a sentence each and never written.
MOVES = {
    "statements": "replace or back a choice with yes/no statements, summed and cut at a line",
    "rubric": "ask for a position on an ordered rubric and cut it at a line",
    "decompose": "write a layer of statements for every label, each summed and cut, ending on "
    "a fallback",
    "second_opinion": "ask the question twice and act only where the answers agree",
    "gate": "ask the judge only where a reading says it is needed",
    "escalate": "settle the own judge's sure answers by rule and ask a named judge the rest",
    FREE: "any other revision",
}


class GraphProposal(dspy.Signature):
    """Revise the requested part of a decision program, using the feedback supplied."""

    request: str = dspy.InputField()
    # Optional: a model with thinking off answers with the part alone, and that is a proposal.
    move: str | None = dspy.OutputField(desc=f"The move taken, one of {', '.join(MOVES)}")
    # Rules are text, and so is the whole-graph JSON a program beyond rules falls back to.
    graph: str = dspy.OutputField(desc="The requested part, complete and alone")


@dataclass(frozen=True)
class Proposal:
    """What a proposer answered: the part, and the move it says it took, where it named one."""

    graph: str
    move: str | None = None


class GraphProposer(dspy.Module):
    def __init__(self, lm: dspy.LM):
        super().__init__()
        self.lm = lm
        self.predict = dspy.Predict(GraphProposal)

    def forward(self, prompt: str | list[dict]) -> Proposal:
        request = prompt if isinstance(prompt, str) else json.dumps(prompt)
        move = None
        # Syntax failures remain candidates so GEPA can score and reflect on them.
        with dspy.context(lm=self.lm, adapter=text_adapter()):
            try:
                answered = self.predict(request=request)
                answer, move = answered.graph, answered.move
            except AdapterParseError as error:
                answer = headless_part(error)
            except ValidationError as error:
                answer = json.dumps({"generation_error": str(error)})
        return Proposal(unfenced(answer if isinstance(answer, str) else json.dumps(answer)), move)


def headless_part(error: AdapterParseError) -> str:
    """The answer from a reply that carries no field header. Models with thinking off answer
    with the part alone, sometimes with a sentence either side of it."""
    return error.lm_response


def unfenced(answer: str) -> str:
    """The answer without the code fence models wrap it in, and without the sentence some
    write before a whole-graph JSON reply."""
    text = answer.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text
        text = text.rsplit("```", 1)[0].strip()
    if not text.startswith("{") and '"nodes"' in text:
        start = text.find("{")
        while start != -1:
            try:
                graph, end = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
                continue
            if isinstance(graph, dict) and "nodes" in graph:
                return text[start : start + end]
            start = text.find("{", start + 1)
    return text


@dataclass(frozen=True)
class Outcome:
    """One scored example: its report row and the record reflection sees."""

    score: float
    record: dict
    trace: dict
    models: list[str]
    error: str | None = None


def passed(task: Task, answered: str | None, expected: str, score: float) -> str | None:
    """Feedback for an example handed to a person, which is neither right nor a mistake."""
    if task.abstain is None or answered != task.abstain.label:
        return None
    return (
        f"Passed to a person where the answer was {expected}, which costs "
        f"{task.abstain.cost:g} of the worst mistake. Score: {score:g}. Pass on only what the "
        "readings and judgments leave unsure."
    )


class Labels:
    """Scores an answer against an example's labels. A right or accepted answer scores 1, the
    task's dearest mistake 0, and a cheaper mistake in between; with no costs set, every mistake
    costs the same and this is exact match."""

    def __init__(self, task: Task):
        self.task = task
        # A mistake on an example costs as much as the real traffic it stands for, so search
        # and evaluation rank programs as they would rank on the real rate. Scores stay in [0, 1].
        self.weights = task.weights()
        self.worst = task.worst * max(self.weights.values())

    def __call__(self, program: DecisionProgram, example: Example) -> Outcome:
        if example.label is None:
            raise ValueError(f"example {example.id} has no label to score against; grade it")
        prediction = program(state=example.state)
        error = undecided(program.plan, prediction)
        # Undecided costs what the dearest mistake does; only the abstain label is a pass.
        answered = prediction.label
        right = [example.label, *example.accepted]
        cost = (
            self.task.worst
            if answered is None
            else min(self.task.cost(label, answered) for label in right)
        )
        score = 1 - cost * self.weights[example.label] / self.worst
        also = f" (also right: {', '.join(example.accepted)})" if example.accepted else ""
        rate = (
            f" It stands for {self.weights[example.label]:g} examples of real traffic."
            if self.task.prevalence is not None
            else ""
        )
        return Outcome(
            score=score,
            record={
                "id": example.id,
                "expected": example.label,
                "correct": cost == 0,
                "cost": cost,
                **dict(prediction),
            },
            trace={
                "Inputs": example.state,
                "Generated Outputs": dict(prediction),
                "Feedback": error
                or passed(self.task, answered, example.label, score)
                or (
                    f"Expected label: {example.label}{also}. Answered: {answered}, which costs "
                    f"{cost:g} where the worst mistake costs {self.task.worst:g}.{rate} "
                    f"Score: {score:g}."
                ),
            },
            models=prediction.models,
            error=error,
        )


Scorer = Callable[[DecisionProgram, object], Outcome]


class SearchTask(Protocol):
    """What search needs from a task; scoring its examples is the scorer's job."""

    description: str
    labels: list[str]
    train: Sequence
    validation: Sequence

    def model_dump_json(self) -> str: ...


Preference = Literal["small", "large"]
# Room for two statements and their sum per label, with the rules that cut them, for a task of
# a dozen or so labels; a task of more sets its own cap.
LARGE_GRAPH_NODES = 64


class SearchConfig(StrictModel):
    max_metric_calls: int = Field(default=100, ge=1)
    prefer: Preference = Field(
        default="small",
        description="Whether the proposer is told to keep graphs small or to build layers of "
        "judgments; large also raises the node cap, unless max_nodes is set",
    )
    max_nodes: int = Field(default=16, ge=1)
    max_depth: int = Field(default=8, ge=1)
    reflection_batch: int = Field(default=3, ge=1)
    repair_attempts: int = Field(
        default=2, ge=0, description="Times an invalid proposal is sent back with its error"
    )
    seed: int = 0
    parallel_examples: int = Field(
        default=1, ge=1, description="Examples scored at once; scorers sharing one game need 1"
    )
    merges: int = Field(
        default=5, ge=0, description="Times search may combine two candidates' parts"
    )
    significance: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Replace the starting graph only if the winner's gain over it on "
        "validation passes a one-sided sign test at this level; small validation sets otherwise "
        "reward a lucky candidate",
    )
    mine: bool = Field(
        default=False,
        description="Before any proposal, mine rules a field alone settles and start from them "
        "if they score better on validation",
    )
    call_cost: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Score taken off an example per judge question asked, so that of two "
        "graphs that answer alike, search keeps the one that asks less",
    )
    call_costs: dict[Name, Annotated[float, Field(ge=0, le=1)]] = Field(
        default_factory=dict,
        description="Score taken off per question asked of a named judge, by its name; a named "
        "judge not listed costs what a question to the graph's own judge does",
    )
    extract_cost: float = Field(
        default=0,
        ge=0,
        le=1,
        description="Score taken off an example per extraction it ran",
    )
    evidence: bool = Field(
        default=False,
        description="Tell the proposer the single-field rules the training examples support "
        "best; counted on the training examples, so it can over-fit a sampled rate",
    )
    keep_judge: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Keep only a graph whose rules, where they answer without asking the judge, "
        "are right at least this often on the labeled training and validation examples, smoothed "
        "by Laplace's rule; a rule over a field anyone can set is otherwise a way around it",
    )

    @model_validator(mode="before")
    @classmethod
    def room_for_large_graphs(cls, data):
        """A layer of judgments per label does not fit the cap a small graph is held to."""
        if isinstance(data, dict) and data.get("prefer") == "large" and "max_nodes" not in data:
            return data | {"max_nodes": LARGE_GRAPH_NODES}
        return data

    @property
    def prices(self) -> Prices:
        return Prices(call=self.call_cost, judges=self.call_costs, extract=self.extract_cost)


def score_all(score: Scorer, program: DecisionProgram, examples: Sequence, parallel: int) -> list:
    """Every example's outcome, in order, scoring up to `parallel` examples at once."""
    with ThreadPoolExecutor(parallel) as pool:
        return list(pool.map(lambda example: score(program, example), examples))


def described_labels(task: SearchTask) -> dict[str, str] | None:
    """What a question written '[task labels]' asks: a labeled task's labels, each with its
    description. A game's actions have none, so its rules write every choice out."""
    return task.criteria() if isinstance(task, Task) else None


def components(
    plan: Plan, labels: dict[str, str] | None = None, readable: bool = True
) -> dict[str, str]:
    """The candidate a proposer revises: rules, not the graph they compile to. Over an input with
    no fields to read, an empty readings part is left out, so every proposal goes to the rules
    and the question rather than half of them to a part with nothing to say."""
    found = parts(plan, labels)
    if not readable and not found["readings"]:
        del found["readings"]
    return found


def assemble(candidate: dict[str, str], labels: dict[str, str] | None = None) -> Plan:
    """The plan a candidate describes, however its parts were rewritten."""
    try:
        return whole(candidate, labels)
    except RuleError as error:
        raise ValueError(str(error)) from error


class GraphAdapter(GEPAAdapter[object, dict, dict]):
    def __init__(
        self,
        task: SearchTask,
        client: TypeSafeClient,
        model: str,
        config: SearchConfig,
        execution: ExecutionConfig | None = None,
        score: Scorer | None = None,
        *,
        models: Models | None = None,
    ):
        self.task = task
        self.labels = described_labels(task)
        self.readable = bool(field_summary(task))
        self.score = score or Labels(task)
        self.client = Remembered(client) if client is not None else None
        self.model = model
        self.models = (models or Models()).remembered()
        self.config = config
        self.execution = execution or ExecutionConfig()
        self.failure: Exception | None = None
        self.resolved_models: set[str] = set()
        # GEPA discards proposals that do not beat their parent, along with their
        # results; keep the latest one so the next reflection can learn from it.
        self.proposed: str | None = None
        self.proposal_feedback: dict | None = None
        self.invalid_candidate_evaluations = 0
        # A graph scores the same however often GEPA asks, and the champion is re-scored
        # every round; without this the seed pays for its own validation over and over.
        self.scored: dict[tuple[str, str], Outcome] = {}

    def parse(self, candidate: dict[str, str]) -> Plan:
        plan = assemble(candidate, self.labels)
        plan.check(self.task.labels, self.config.max_nodes, self.config.max_depth)
        return plan

    def remember(self, plan: Plan, program: DecisionProgram, batch: Sequence) -> list:
        """Every example's outcome, scoring only the ones this graph has not seen."""
        graph = plan.model_dump_json()
        wanted = [e for e in batch if (graph, str(e)) not in self.scored]
        fresh = score_all(self.score, program, wanted, self.config.parallel_examples)
        self.scored.update({(graph, str(e)): o for e, o in zip(wanted, fresh, strict=True)})
        return [self.scored[graph, str(example)] for example in batch]

    def evaluate(self, batch, candidate, capture_traces=False):
        if self.failure is not None:
            raise self.failure
        outputs, scores, records = [], [], []
        try:
            plan = self.parse(candidate)
        except ValueError as error:
            self.invalid_candidate_evaluations += 1
            feedback = {"Feedback": f"Candidate validation failed: {error}"}
            return self.batch(
                candidate,
                [None] * len(batch),
                [0.0] * len(batch),
                [feedback] * len(batch) if capture_traces else None,
            )
        program = DecisionProgram(plan, self.client, self.model, self.execution, models=self.models)
        try:
            outcomes = self.remember(plan, program, batch)
        except Exception as error:
            # GEPA catches some adapter failures internally, even with raise_on_exception.
            self.failure = error
            raise
        for outcome in outcomes:
            self.resolved_models.update(outcome.models)
            outputs.append(outcome.record)
            score, trace = self.charged(outcome)
            scores.append(score)
            records.append(trace)
        return self.batch(candidate, outputs, scores, records if capture_traces else None)

    def charged(self, outcome: Outcome) -> tuple[float, dict]:
        """An example's score less the price of the judge questions it asked, each at its own
        judge's price, and feedback that says so, since a proposer cannot lower a cost it is not
        told about."""
        prices = self.config.prices
        asked = Counter(payer(call) for call in outcome.record.get("calls", {}).values())
        price = sum(count * prices.of(paid) for paid, count in asked.items())
        if not price:
            return outcome.score, outcome.trace
        each = ", ".join(
            f"{count} call(s) to {paid} at {prices.of(paid):g} each"
            for paid, count in sorted(asked.items())
        )
        told = (
            f" It made {each}, so it scores {max(0.0, outcome.score - price):g}; answer from "
            "readings where they suffice, and pay for a dearer model only where it helps."
        )
        trace = outcome.trace | {"Feedback": outcome.trace.get("Feedback", "") + told}
        return max(0.0, outcome.score - price), trace

    def batch(self, candidate, outputs, scores, trajectories) -> EvaluationBatch:
        if trajectories is not None and self.proposed in candidate.values():
            self.proposal_feedback = {
                "graph": self.proposed,
                "scores": scores,
                "results": trajectories,
            }
            self.proposed = None
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {component: eval_batch.trajectories for component in components_to_update}


def abstaining(task: SearchTask) -> str:
    """How to hand an example to a person, when the task allows it."""
    abstain = getattr(task, "abstain", None)
    if abstain is None:
        return ""
    return (
        f"Answer {abstain.label!r} to pass an example to a person rather than guess; it costs "
        f"{abstain.cost:g} of a mistake, so use it where the case stays unsure, for example "
        f"'when threat.confidence < 0.6: {abstain.label}'.\n"
    )


def keeping(config: SearchConfig) -> str:
    """What the proposer must know when rules may answer alone only where they are that sure."""
    if config.keep_judge is None:
        return ""
    return (
        f"A rule that answers without asking the judge must be right at least "
        f"{config.keep_judge:.0%} of the time, or the program is refused; where no reading is "
        "that sure, keep the judge in the path.\n"
    )


def field_summary(task: SearchTask, with_evidence: bool = False) -> str:
    """What the fields of the inputs hold, so a proposer can see which a rule could read, and,
    when asked, the rules the training examples support best: the same for every search on the
    same examples, so whether a search finds the rule the data holds depends less on its luck.
    Counted on a sampled training set, that evidence can over-fit the sample's rate, so it is
    shown only when the search asks for it."""
    examples = [example for example in task.train if hasattr(example, "state")]
    summary = fields(examples)
    if not summary:
        return ""
    described = "; ".join(f"{name}: {held}" for name, held in summary.items())
    found = evidence(examples) if with_evidence and isinstance(task, Task) else []
    supported = (
        "Rules the training examples support best, counted on the training examples; small "
        "counts can be chance, so use, combine, or ignore them as the feedback warrants:\n"
        + "\n".join(found)
        + "\n"
        if found
        else ""
    )
    return f"Fields of the inputs, as seen in training: {described}.\n{supported}"


def record_lists(state, path: tuple[str, ...] = ()) -> dict[str, set[str]]:
    """Every list of objects in an input, by path, with the fields its items carry."""
    found: dict[str, set[str]] = {}
    if isinstance(state, dict):
        for key, value in state.items():
            for listed, keys in record_lists(value, (*path, str(key))).items():
                found.setdefault(listed, set()).update(keys)
    elif isinstance(state, list) and state and all(isinstance(item, dict) for item in state):
        found[".".join(path)] = {key for item in state for key in item}
    return found


def listing(task: SearchTask) -> str:
    """How to test a list's items, told only where the inputs hold lists of records."""
    found: dict[str, set[str]] = {}
    for example in task.train:
        for listed, keys in record_lists(getattr(example, "state", None)).items():
            found.setdefault(listed, set()).update(keys)
    if not found:
        return ""
    named = "; ".join(f"{path} ({', '.join(sorted(keys))})" for path, keys in sorted(found.items()))
    path, keys = min(found.items())
    field = min(keys)
    return (
        f"These fields are lists of records, with their items' fields: {named}. Read one as "
        f"'read {path} each {{{field} in [low, high]}}', declaring each field it tests as a "
        f"reading is declared, then test 'any {path} has {field} > 5' or 'all {path} have "
        f"{field} > 5', a compound test in parentheses, or count with 'let n = count of {path} "
        f"where {field} > 5'. An item missing a field fails a test on it.\n"
    )


def text_only(task: SearchTask) -> str:
    """What to tell a proposer whose inputs have no fields: the question is the whole program,
    so improve it, and start with a briefing, which is where this judge takes guidance."""
    if not isinstance(task, Task):
        return ""
    return (
        "The input is one piece of text with no fields to read, so the program is its question. "
        "Improve the question: write the labels out with descriptions that draw the lines the "
        "feedback shows the judge misses, and give the judgment a briefing. Try a briefing "
        "first, in your first proposal, and sharpen it from the feedback after.\n"
    )


def extraction(config: SearchConfig, extracting: bool) -> str:
    """How to have a model read fields out of free text, and what each extraction costs."""
    if not extracting:
        return ""
    return (
        "A model can read fields out of free text for the rules to test, which a judge cannot: "
        "'extract NAME = \"what to read\" {amount in [0, 10000], kind one of [refund, bug]}', "
        "then 'when NAME.amount >= 500' or 'NAME.kind == \"refund\"'. A field it cannot fill "
        "within its domain is absent, like a missing reading. It may end 'given [...]' and "
        f"'when ...' as a judgment may; each extraction costs {config.extract_cost:g}. "
    )


def escalating(config: SearchConfig, judges: Sequence[str]) -> str:
    """The judges besides the graph's own, and what a question to each costs."""
    if not judges:
        return ""
    priced = ", ".join(
        f"{name} ({config.call_costs.get(name, config.call_cost):g} a question)" for name in judges
    )
    return (
        "Besides the graph's own judge, a judgment may be asked of another by ending it "
        f"'using NAME', after any 'given': {priced}; a question to the graph's own judge costs "
        f"{config.call_cost:g}. A larger judge is for the cases the graph's own is unsure of: "
        "ask the own judge first, let rules settle its sure answers, such as "
        "'when team.confidence >= 0.8 and team == \"billing\": billing', and end with the same "
        "question 'using' the larger judge. "
    )


def applies(label: str) -> str:
    return f"When the input calls for {label}."


def statements(labels: Sequence[str]) -> str:
    """Two yes/no statements per label but the last, each pair summed and cut, and the last
    label as the fallback: the shape a choice decomposes into."""
    *decided, fallback = labels
    asked, rules = [], []
    for label in decided:
        name = re.sub(r"\W+", "_", label)
        asked += [
            f"ask said_{name} = {quoted(f'The input calls for {label}.')}",
            f"ask meant_{name} = {quoted(f'Read as a whole, the input is about {label}.')}",
            f"let support_{name} = said_{name} + meant_{name}",
        ]
        rules.append(f"when support_{name} >= 1.2: {label}")
    return "\n".join([*asked, *rules, f"always: {fallback}"])


def gated_reading(task: SearchTask) -> tuple[str, str] | None:
    """A reading of the inputs' first numeric field, and a test at its midpoint, for a gate."""
    examples = [example for example in task.train if hasattr(example, "state")]
    for (path, measure), seen in columns(examples).items():
        if measure == "value" and not isinstance(seen[0], str):
            name, low, high = ".".join(path), min(seen), max(seen)
            return f"read {name} in [{low:g}, {high:g}]", f"{name} >= {(low + high) / 2:g}"
    return None


def move_examples(task: SearchTask, judges: Sequence[str], max_nodes: int) -> dict[str, str]:
    """Each move the task can take, with an example in its own terms, less `free`: an escalation
    needs a judge to escalate to, a gate a field to read, and a layer of statements for every
    label room in the node budget."""
    first, second = task.labels[:2]
    options = " ".join(f"[{label}: {quoted(applies(label))}]" for label in (first, second))
    question = f"{quoted(task.description)} {options}"
    examples = {
        "statements": statements(task.labels[:2]),
        "rubric": (
            f"ask degree = {quoted(f'How strongly does the input call for {first}?')} "
            f"[weakly] [somewhat] [clearly]\nwhen degree >= 1.5: {first}\nalways: {second}"
        ),
        "decompose": statements(task.labels),
        "second_opinion": (
            f"ask first = {question}\n"
            f"ask second = {quoted(f'A second reading. {task.description}')} {options}\n"
            "vote checked = agree of [first, second]\n"
            f'when checked == "yes" and first == "{first}": {first}\nalways: {second}'
        ),
    }
    if (reading := gated_reading(task)) is not None:
        declared, test = reading
        examples["gate"] = (
            f"{declared}\nask kind = {question} when {test}\n"
            f'when kind == "{first}": {first}\nalways: {second}'
        )
    if judges:
        examples["escalate"] = (
            f"ask first = {question}\n"
            f'when first.confidence >= 0.8 and first == "{first}": {first}\n'
            f'when first.confidence >= 0.8 and first == "{second}": {second}\n'
            f"always: ask {question} using {judges[0]}"
        )
    if len(compile_rules(examples["decompose"]).nodes) > max_nodes:
        del examples["decompose"]
    return examples


def revising(task: SearchTask, config: SearchConfig, judges: Sequence[str]) -> str:
    """The moves a proposal may take, each with its example, and the one to name when none fits."""
    offered = "".join(
        f"{name}: {MOVES[name]}, for example:\n{example}\n"
        for name, example in move_examples(task, judges, config.max_nodes).items()
    )
    return (
        "Take one of these moves and name it in the move field; each example is a whole "
        f"program, of which you return the part asked for:\n{offered}{FREE}: {MOVES[FREE]}.\n"
    )


def sizing(config: SearchConfig) -> str:
    """How big a graph the proposer should aim for."""
    if config.prefer == "small":
        return "Prefer small generalizable graphs."
    return (
        "Prefer graphs that ask many judgments: a layer of statements or rubrics, one or more "
        "per label, summed with 'let' and cut with 'when', generalizes better than one "
        "question asked to do everything, so add judgments rather than fold them into one."
    )


def proposer_prompt(
    task: SearchTask, config: SearchConfig, judges: Sequence[str] = (), extracting: bool = False
) -> str:
    """Everything the proposer is told: the language, the task, and the part to return. Named
    `judges` are the ones besides the graph's own that a judgment may be asked of; `extracting`
    says a model is there to read fields out of free text."""
    syntax_example = Plan.model_validate(
        {
            "nodes": {
                "decision": {
                    "type": "call",
                    "question": {
                        "type": "choice",
                        "instructions": task.description,
                        "criteria": {label: applies(label) for label in task.labels[:2]},
                    },
                    "inputs": {"request": {"type": "state"}},
                }
            },
            "outputs": {"label": {"type": "node", "node": "decision"}},
        }
    )
    # Only a labeled task describes its labels; a game's rules have no list to point to.
    every_label = (
        "A choice over every allowed label, each as the task describes it, is written "
        "'ask intent = \"question\" [task labels]' instead of listing them. Where the judge "
        "confuses labels, write them out instead, '[label: \"when it applies\"]' for each, "
        "with descriptions that draw the lines the feedback shows it misses, and make the "
        "question's instructions as full as the judge needs: what to look for, how the labels "
        "differ, and the cases that are easy to confuse. The judge reads the whole question, "
        "so a careful paragraph is a legitimate move, not padding. "
        if isinstance(task, Task)
        else ""
    )
    return (
        "Improve this decision program using observed execution failures. It is written in "
        "two parts, 'readings' (what it may look at and ask) and 'rules' (what it decides), "
        "and you return the one part you are asked for as rule text. Add, remove, or rewrite "
        "readings, questions, names, and conditions. Start from the current program: keep "
        "the readings, names, and rules that work, and change only what the feedback shows is "
        "failing, since rewriting a working graph from scratch usually loses what worked. "
        "Each call receives its mapped inputs, "
        "which can reference original state, literal data, or earlier node values/answers. "
        "Ready calls with identical inputs share one request; others run concurrently. "
        "A call's question MUST be an object nested inside the call, containing lowercase "
        "type, instructions, and criteria when applicable. Choice criteria map each option, "
        "which is the value the node outputs, to a description of when it applies. "
        "Never put instructions or criteria "
        'beside question. Original input uses {"type":"state"}; it is not a node named input. '
        'Every input and output source needs its type, e.g. {"type":"node","node":"decision"}; '
        "a source without type is rejected. A choice call's value is the option it picks; a "
        "score call's value is the expected position on its ordered levels; a noul call's value "
        "is the probability that its yes/no statement is true. Calls judge; they cannot extract "
        "or compute numbers from their inputs. "
        "A read node takes one value from the original input by path, e.g. "
        '{"type":"read","path":["rangefinder",3],"bounds":[0,2048]}, declaring bounds or '
        "categories. Reads give exact values for thresholds, switches, and conditions; calls "
        "let the judge weigh the whole input, which suits decisions no single value settles, "
        "such as whether to fight or fall back. Use whichever scores better. "
        "A call runs on every input that does not skip it, so in a game it runs every step; "
        "gate calls with when-conditions on reads, such as only while a target is present, "
        "so the judge is asked only where no read settles the decision. "
        "A read's path must end at a number or string, never an "
        "object or list. A missing path or a value outside the declared domain skips the "
        "read, so test whether a field exists with "
        '{"type":"present","node":"<read of a scalar inside it>"}. The label must always get '
        "a value: end the final coalesce with an unconditional value node as the default. "
        "Only declared node IDs may be referenced; there is no input_wiring field. "
        "Use view=value for a label output, not view=answer (which returns an object). "
        "Node and control references must form a DAG; every node must feed an output. "
        "Conditions gate execution after dependencies settle. Missing required inputs skip "
        "a node; optional missing inputs are omitted. merge builds an object. coalesce waits "
        "for all sources and picks the first available in listed order. All skipped sources "
        "skip the coalesce. A required missing output invalidates that execution. "
        "Switch cases must cover the source's categorical values; threshold uses >= for above. "
        "The label output must have a statically known domain within the allowed labels. "
        f"Do not encode example-specific answers. {sizing(config)}\n"
        f"Task: {task.description}\nAllowed label values: {json.dumps(task.labels)}\n"
        f"{field_summary(task, config.evidence) or text_only(task)}"
        f"{listing(task)}"
        f"{abstaining(task)}"
        f"{keeping(config)}"
        f"At most {config.max_nodes} nodes and dependency depth {config.max_depth}.\n"
        # The schema is twelve thousand characters and rules need none of it; a proposal
        # that falls back to whole-graph JSON is rare enough to ask for the schema then.
        "A whole-graph JSON answer, for the rare program rules cannot express, is "
        '{"nodes": {...}, "outputs": {"label": {"type": "node", "node": "..."}}} whose nodes '
        "are call, switch, threshold, merge, coalesce, value, read, confidence, vote or sum, "
        "each optionally guarded "
        'by "when". An example of one: \n'
        f"{syntax_example.model_dump_json(exclude_none=True)}\n"
        "Use the task's full label set when appropriate.\n"
        "Write the program as rules, tried in order, the first whose condition holds "
        "deciding. Declare what it may look at: 'read target.x in [-1, 1]' or "
        "'read route.next one of [path, door, exit]'. A reading may be given a name, which "
        "the rules then use in place of its path: 'read health = variables.HEALTH in [0, 200]'. "
        "'read length of company_profile in [0, 10000]' reads how long a text or list is, which "
        "a rule tests as 'length of company_profile < 20': an empty or one-line field is often "
        "telling and costs nothing to check. "
        "A judgment is asked in one of three "
        "ways, and its kind decides how a rule tests it: "
        '\'ask threat = "question" [fight: "when it applies"] [clear: "when it applies"]\' '
        "answers one of those labels, tested 'threat == \"fight\"'; "
        "'ask severity = \"question\" [minor] [major] [critical]' answers a position on that "
        "rubric, ordered, tested 'severity >= 2'; and "
        "'ask winning = \"a statement that is either true or false\"' answers the probability "
        "that it is true, tested 'winning >= 0.8'. "
        f"{every_label}"
        f"{escalating(config, judges)}"
        f"{extraction(config, extracting)}"
        "Ask for a probability when a threshold "
        "suits the decision, and for labels when the answers have no order. Any judgment may "
        "end 'when <condition>' to be asked only then, which is how a costly question is kept "
        "behind a cheap reading. A judgment may end 'briefed \"...\"' with a briefing the "
        "judge reads before the input: what to look for, how the labels differ, and the cases "
        "that are easy to confuse. The judge takes guidance from the briefing, not from the "
        "question, so when it confuses labels write a full briefing rather than a longer "
        "question. A judgment reads the whole input unless it ends 'given' a "
        "list of declared names, which is all it is then shown and costs less to read: "
        "'ask close = \"The enemy is close enough to fight.\" given [enemy.distance, health]'. "
        "'observation' in that list is the whole input. "
        "A judgment may end with calibrated and a number, a temperature fitted to how often "
        "it is right; keep it as written and never invent one. "
        "A choice or a score also reports how sure it was: test "
        "'threat.confidence >= 0.7' to act only on a sure answer and let a hedged one fall "
        "through. 'vote verdict = agree of [first, second]' answers yes when every judgment "
        "listed gave the same label and no otherwise, which is how a second opinion verifies a "
        "first; 'vote verdict = majority of [first, second, third]' answers the label most of "
        "them gave. A vote counts labels, so its judgments must be choices, and each one is "
        "another model call. Then the rules: "
        "'when <condition>: <action>' or 'always: <action>', where an action is a label, an "
        "ask written the same way, or 'route.bearing >= 0 ? TURN_LEFT : TURN_RIGHT'. "
        "Conditions join with 'and' and 'or', group with parentheses, invert with 'not', name "
        "a reading to test that it has a value, or compare one with >=, >, <=, < or ==, "
        "against a number, a quoted label, or another reading: "
        "'capture.best_gain > hanging.worst'. Both sides of such a comparison must be numbers, "
        "or both labels compared with ==. Name a condition the rules test more than once, "
        "before the rules that use it: 'let armed = ammo >= 1 and health >= 25', then "
        "'when armed: ATTACK' or 'when not armed and health < 10: MOVE_BACKWARD'. "
        "'let' also computes a number from readings, with +, - and a number times a reading: "
        "'let margin = 2 * gain - worst + 1', then 'when margin >= 3: CAPTURE'. "
        "Rules decide the label. A line 'output priority:' starts another rule list, which "
        "decides an output of that name; keep any such sections a program already has. "
        "Write no other syntax: no "
        "code fences, no section headings, no comments.\n"
        f"{revising(task, config, judges)}"
        "The readings part holds declarations and the rules part holds "
        "rules; return the part you are asked for, complete and alone, except that a rules "
        "part may declare any reading its rules need. Answer in the same form the part arrives "
        "in: a readings part is always lines beginning 'read' or 'ask', never JSON. Only a rules "
        "part may be answered as whole-graph JSON, for a program rules cannot express, matching "
        "the schema above; invent no field that schema does not list.\n"
        "Part to revise:\n<curr_param>\nObserved examples and feedback:\n<side_info>"
    )


def optimize_plan(
    task: SearchTask,
    initial: Plan,
    client: TypeSafeClient,
    model: str,
    reflection_lm: Callable[[str], str | Proposal],
    config: SearchConfig,
    execution: ExecutionConfig | None = None,
    score: Scorer | None = None,
    run_dir: Path | None = None,
    models: Models | None = None,
) -> tuple[Plan, dict]:
    """Search from `initial`. `reflection_lm` answers a prompt with the part it proposes, or
    with a `Proposal` naming the move it took. With `run_dir`, GEPA checkpoints its state there
    after every round and resumes from it when it exists, so an interrupted search continues
    where it stopped and a finished one runs on under a larger budget; a `gepa.stop` file in it
    stops the search after the current round with everything kept."""
    initial.check(task.labels, config.max_nodes, config.max_depth)
    seed_sha256 = hashlib.sha256(initial.model_dump_json().encode()).hexdigest()
    task_sha256 = hashlib.sha256(task.model_dump_json().encode()).hexdigest()
    # A rule's counts describe the graph they were taken on; a proposer would carry them onto
    # rewrites they no longer describe.
    initial = unmeasured(initial)
    batch_size = min(config.reflection_batch, len(task.train))
    minimum_budget = 2 * (len(task.validation) + batch_size)
    if config.max_metric_calls < minimum_budget:
        raise ValueError(f"max_metric_calls must be at least {minimum_budget} for one search round")
    resumed = run_dir is not None and (run_dir / "gepa_state.bin").exists()
    if run_dir is not None:
        # A checkpoint continues one search; another task or seed would corrupt it quietly.
        identity = {"task_sha256": task_sha256, "seed_sha256": seed_sha256}
        record = run_dir / "search.json"
        if resumed and json.loads(record.read_text()) != identity:
            raise ValueError(
                f"{run_dir} holds another search, of another task or from another starting "
                "graph; resume it with the same, or write to a new output"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(identity, indent=2) + "\n")
    adapter = GraphAdapter(task, client, model, config, execution, score, models=models)
    if config.keep_judge is not None and not isinstance(adapter.score, Labels):
        raise ValueError("keep_judge counts how often rules are right, which needs labels")
    start, mined_report = initial, None
    mined = mine(task, initial) if config.mine and isinstance(task, Task) else None
    if mined is not None:
        try:
            adapter.parse(components(mined, adapter.labels, adapter.readable))
        except ValueError:
            mined = None
    if mined is not None:
        # Two extra scorings of the validation set, outside the budget, to pick where to start.
        scored = {
            name: mean(
                adapter.evaluate(
                    task.validation, components(plan, adapter.labels, adapter.readable)
                ).scores
            )
            for name, plan in (("seed", initial), ("mined", mined))
        }
        used = scored["mined"] >= scored["seed"]
        start = mined if used else initial
        rules = sum(1 for name in mined.nodes if name.startswith("rule_mined"))
        mined_report = {"rules": rules, "validation": scored["mined"], "used": used}
    reflection_calls = 0
    repairs = 0
    # Each proposal's part and move, with its text and the candidate it revised, to say later
    # which were kept.
    proposals: list[tuple[dict, str, dict[str, str]]] = []

    def reflect(prompt, candidate, component):
        nonlocal reflection_calls
        if adapter.failure is not None:
            raise adapter.failure
        # Name the part, and show the rest, so a rule can use a reading that already exists.
        context = "\n".join(f"{part}:\n{text}" for part, text in candidate.items() if text)
        prompt = (
            f"{prompt}\n\nReturn the part named {component}.\n"
            f"The whole program as it stands:\n{context}"
        )
        previous = adapter.proposal_feedback
        if previous is not None:
            feedback = (
                "Your previous proposal was scored on training examples (1 is best per "
                "example) and discarded unless it beat the current graph. Do not repeat its "
                f"mistakes.\nPrevious proposal:\n{previous['graph']}\n"
                f"Scores: {previous['scores']}\nResults: {json.dumps(previous['results'])}"
            )
            prompt = f"{prompt}\n\n{feedback}"
            adapter.proposal_feedback = None
        nonlocal repairs
        request = prompt
        for attempt in range(config.repair_attempts + 1):
            try:
                reflection_calls += 1
                answered = reflection_lm(request)
            except Exception as error:
                # Preserve the provider failure if GEPA treats it as an unsuccessful mutation.
                adapter.failure = error
                raise
            answered = answered if isinstance(answered, Proposal) else Proposal(answered)
            proposal = answered.graph.strip()
            try:
                adapter.parse(candidate | {component: proposal})
                break
            except ValueError as error:
                if attempt == config.repair_attempts:
                    break
                repairs += 1
                # Validation is instant, so fix schema slips before spending episodes on them.
                repair = (
                    f"This graph failed validation and was not run:\n{proposal}\n"
                    f"Error: {error}\nReturn only the corrected {component} component."
                )
                # Only the latest attempt matters; earlier failures would bloat the prompt.
                request = f"{prompt}\n\n{repair}"
        adapter.proposed = proposal
        proposals.append(({"part": component, "move": answered.move}, proposal, dict(candidate)))
        return proposal

    given = models or Models()
    prompt = proposer_prompt(task, config, sorted(given.judges), given.extractor is not None)

    def propose(candidate, reflective_dataset, components_to_update):
        return {
            component: reflect(
                prompt.replace("<curr_param>", candidate[component]).replace(
                    "<side_info>", json.dumps(reflective_dataset[component])
                ),
                candidate,
                component,
            )
            for component in components_to_update
        }

    adapter.propose_new_texts = propose
    result = gepa.optimize(
        seed_candidate=components(start, adapter.labels, adapter.readable),
        trainset=task.train,
        valset=task.validation,
        adapter=adapter,
        reflection_minibatch_size=batch_size,
        max_metric_calls=config.max_metric_calls,
        seed=config.seed,
        use_merge=config.merges > 0,
        max_merge_invocations=config.merges,
        display_progress_bar=False,
        raise_on_exception=True,
        stop_callbacks=lambda state: adapter.failure is not None,
        run_dir=str(run_dir) if run_dir is not None else None,
    )
    if adapter.failure is not None:
        raise adapter.failure
    # A proposal is kept when the program it makes of its parent became a candidate; one that
    # restates its parent never does.
    moves = [
        record
        | {
            "kept": text != parent[record["part"]]
            and parent | {record["part"]: text} in result.candidates
        }
        for record, text, parent in proposals
    ]

    def gain_p(index: int) -> float:
        seed_scores, scores = result.val_subscores[0], result.val_subscores[index]
        shared = seed_scores.keys() & scores.keys()
        return improvement_p(
            sum(scores[i] > seed_scores[i] for i in shared),
            sum(scores[i] < seed_scores[i] for i in shared),
        )

    def admissible(index: int) -> bool:
        return index == 0 or config.significance is None or gain_p(index) <= config.significance

    chosen = result.best_idx
    p = gain_p(chosen)
    kept_seed = not admissible(chosen)
    if kept_seed:
        chosen = 0
    best = adapter.parse(result.candidates[chosen])
    validation_score = result.val_aggregate_scores[chosen]
    passed_over = 0
    if config.keep_judge is not None:
        ranked = sorted(
            range(len(result.candidates)), key=lambda i: -result.val_aggregate_scores[i]
        )
        keep = [chosen] + [i for i in ranked if i != chosen and admissible(i)]
        found = next(
            (i for i in keep if judged(adapter, adapter.parse(result.candidates[i]), task)), None
        )
        passed_over = keep.index(found) if found is not None else len(keep)
        if found is not None:
            chosen, best = found, adapter.parse(result.candidates[found])
            validation_score = result.val_aggregate_scores[found]
        elif judged(adapter, initial, task):
            best = initial
            validation_score = mean(
                adapter.evaluate(
                    task.validation, components(initial, adapter.labels, adapter.readable)
                ).scores
            )
        else:
            raise ValueError(
                f"no graph search found, nor the one it started from, keeps rules that answer "
                f"without the judge right at least {config.keep_judge:g} of the time"
            )
    report = {
        "model": model,
        "resolved_models": sorted(adapter.resolved_models),
        "task_sha256": task_sha256,
        "seed_sha256": seed_sha256,
        "checkpoint": str(run_dir) if run_dir is not None else None,
        "resumed": resumed,
        "stopped_early": run_dir is not None and (run_dir / "gepa.stop").exists(),
        "mined": mined_report,
        "versions": {name: version(name) for name in ("dspy", "gepa", "typesafe-sdk")},
        "search": config.model_dump(),
        "execution": adapter.execution.model_dump(),
        "seed_validation_score": result.val_aggregate_scores[0],
        "validation_score": validation_score,
        "improvement_p": p,
        "kept_seed": kept_seed,
        "keep_judge_passed_over": passed_over,
        "validation_scores": result.val_aggregate_scores,
        # The budget spent when each candidate was found, so a run can be charted by cost.
        "discovered_at": list(result.discovery_eval_counts),
        "candidates": len(result.candidates),
        "reflection_calls": reflection_calls,
        "repairs": repairs,
        "moves": moves,
        "invalid_candidate_evaluations": adapter.invalid_candidate_evaluations,
        "metric_calls": result.total_metric_calls,
    }
    return best, report


def judged(adapter: GraphAdapter, plan: Plan, task: SearchTask) -> bool:
    """Whether every rule that answers some labeled example without asking the judge is right
    as often as search was told to keep: counted on training and validation examples alike,
    since a rule settles few of either."""
    examples = [*task.train, *task.validation]
    program = DecisionProgram(
        plan, adapter.client, adapter.model, adapter.execution, models=adapter.models
    )
    records = [outcome.record for outcome in adapter.remember(plan, program, examples)]
    alone = [record for record in records if not record.get("calls")]
    return all(
        Measured(right=right, decided=decided, examples=len(records)).precision
        >= adapter.config.keep_judge
        for right, decided in tally(alone).values()
    )


def improvement_p(wins: int, losses: int) -> float:
    """One-sided sign test: the chance of at least this many wins among the examples that
    changed, if the new graph were no better. Ties say nothing either way."""
    changed = wins + losses
    return sum(math.comb(changed, k) for k in range(wins, changed + 1)) / 2**changed


def evaluate_plan(
    plan: Plan,
    examples: Sequence,
    client: TypeSafeClient | None,
    model: str,
    score: Scorer,
    execution: ExecutionConfig | None = None,
    parallel_examples: int = 1,
    models: Models | None = None,
) -> dict:
    program = DecisionProgram(plan, client, model, execution, models=models)
    outcomes = score_all(score, program, examples, parallel_examples)
    return {
        "score": sum(outcome.score for outcome in outcomes) / len(outcomes),
        "resolved_models": sorted({name for outcome in outcomes for name in outcome.models}),
        "examples": [outcome.record for outcome in outcomes],
    }
