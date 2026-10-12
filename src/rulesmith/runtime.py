"""Execute typed DAGs as DSPy modules through the official TypeSafe client."""

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import UTC, datetime
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Annotated, NamedTuple

import dspy
from pydantic import ConfigDict, Field, TypeAdapter
from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeClient,
    Usage,
)

from rulesmith.chat_judge import score_confidence
from rulesmith.graph import (
    All,
    Call,
    Choice,
    Coalesce,
    Compare,
    Condition,
    Confidence,
    Extract,
    Graph,
    ItemField,
    LiteralInput,
    Merge,
    Node,
    NodeInput,
    Not,
    Noul,
    Plan,
    Present,
    Quantify,
    Question,
    Read,
    Relation,
    Score,
    Source,
    State,
    StateInput,
    StrictModel,
    Sum,
    Switch,
    Value,
    Vote,
    tested,
)


class ResponseError(ValueError):
    """The provider returned an answer that violates the requested contract."""


def in_range(value: float, upper: float) -> bool:
    return math.isfinite(value) and 0 <= value <= upper


def read_answer(question: Question, answer: NoulAnswer | ChoiceAnswer | ScoreAnswer) -> dict:
    expected = {Noul: NoulAnswer, Choice: ChoiceAnswer, Score: ScoreAnswer}
    if not isinstance(answer, expected[type(question)]):
        raise ResponseError("answer type does not match question type")
    if isinstance(answer, NoulAnswer):
        if not in_range(answer.noul, 1):
            raise ResponseError("Noul probability must be finite and between 0 and 1")
        return {"type": "noul", "value": answer.noul}

    keys = (
        set(question.criteria)
        if isinstance(question, Choice)
        else set(range(len(question.criteria)))
    )
    probabilities = answer.probabilities
    if set(probabilities) != keys or not all(in_range(p, 1) for p in probabilities.values()):
        raise ResponseError(
            "probabilities must cover exactly the declared options and lie in [0, 1]"
        )
    # The hosted service may round individual probabilities in its response.
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=0.02):
        raise ResponseError("probabilities do not sum to 1 within rounding tolerance")
    if not in_range(answer.confidence, 1):
        raise ResponseError("confidence must be finite and between 0 and 1")

    if isinstance(answer, ChoiceAnswer):
        if answer.choice not in keys:
            raise ResponseError("selected option is outside the declared options")
        value = answer.choice
    else:
        if not in_range(answer.score, len(question.criteria) - 1):
            raise ResponseError("score is outside the declared rubric")
        if answer.legend != dict(enumerate(question.criteria)):
            raise ResponseError("returned rubric differs from the requested rubric")
        value = answer.score
    return {
        "type": question.type,
        "value": value,
        "probabilities": probabilities,
        "confidence": answer.confidence,
    }


def calibrated(answer: dict, temperature: float) -> dict:
    """An answer with its log-odds divided by `temperature`. The choice itself never changes,
    only how sure it reads; a score is the expected level of its tempered probabilities, and the
    confidence a rule tests is worked out again from them by the formula the judge reports it
    with."""
    if temperature == 1:
        return answer
    if answer["type"] == "noul":
        odds = (
            math.log(answer["value"]) - math.log1p(-answer["value"])
            if 0 < answer["value"] < 1
            else None
        )
        value = answer["value"] if odds is None else 1 / (1 + math.exp(-odds / temperature))
        return answer | {"value": value}
    weights = {key: p ** (1 / temperature) for key, p in answer["probabilities"].items()}
    total = sum(weights.values())
    probabilities = {key: weight / total for key, weight in weights.items()}
    tempered = answer | {"probabilities": probabilities}
    if answer["type"] == "score":
        tempered["value"] = sum(int(level) * p for level, p in probabilities.items())
        levels = sorted(probabilities, key=int)
        tempered["confidence"] = score_confidence([probabilities[level] for level in levels])
    else:
        # How far the chosen option stands above even odds, as `choice_confidence` reports it
        # for the likeliest option: the same number whenever a judge chose its likeliest option,
        # and less, down to none, for an answer its own probabilities argue against.
        uniform = 1 / len(probabilities)
        chosen = probabilities[answer["value"]]
        tempered["confidence"] = max(0.0, (chosen - uniform) / (1 - uniform))
    return tempered


# Why a decision has no answer. A required output no node produced is the only way a valid graph
# can end without one: a judge that fails raises, since that is a failure, not a decision.
NO_RULE_APPLIED = "no_rule_applied"


def undecided(plan: Plan, prediction: dspy.Prediction) -> str | None:
    """What scoring tells search about a decision that was not made, naming the outputs it lacks
    so the proposer sees what to cover, or None for a decision that was made."""
    if prediction.undecided is None:
        return None
    missing = [
        name
        for name, source in plan.outputs.items()
        if not source.optional and name not in prediction.outputs
    ]
    return (
        f"The graph decided nothing ({prediction.undecided}): "
        f"required graph outputs were skipped: {missing}"
    )


class ExecutionConfig(StrictModel):
    max_workers: int = Field(default=4, ge=1)
    # A tuple, not a set: this config is written into a run's report, and JSON has no set.
    suppress: tuple[str, ...] = Field(
        default=(),
        description="Nodes forced absent, for measuring what a graph decides without them",
    )


# JSON null is a valid value, so absence needs a distinct sentinel.
MISSING = object()


def matches(condition: Condition | None, values: dict) -> bool:
    if condition is None:
        return True
    if isinstance(condition, Present):
        return condition.node in values
    if isinstance(condition, (Compare, Relation)):
        if condition.node not in values:
            return False
        if isinstance(condition, Relation):
            if condition.other not in values:
                return False
            against = values[condition.other]
        else:
            against = condition.value
        value = values[condition.node]
        if condition.op == "eq":
            return value == against
        return value < against if condition.op == "lt" else value >= against
    if isinstance(condition, Not):
        return not matches(condition.condition, values)
    if isinstance(condition, All):
        return all(matches(child, values) for child in condition.conditions)
    return any(matches(child, values) for child in condition.conditions)


def resolve(source: Source, state: State, values: dict, answers: dict):
    if isinstance(source, StateInput):
        return state
    if isinstance(source, LiteralInput):
        return deepcopy(source.value)
    return (answers if source.view == "answer" else values).get(source.node, MISSING)


def resolve_inputs(inputs: dict[str, Source], state: State, values: dict, answers: dict):
    result = {}
    for name, source in inputs.items():
        value = resolve(source, state, values, answers)
        if value is MISSING:
            if isinstance(source, NodeInput) and source.optional:
                continue
            return MISSING
        result[name] = value
    return result


def read_path(node: Read, state: State):
    if state is MISSING:
        return MISSING
    value = state
    for step in node.path:
        if isinstance(step, int) and isinstance(value, list) and -len(value) <= step < len(value):
            value = value[step]
        elif isinstance(value, dict) and str(step) in value:
            # A rule's 'years.2024' is an index into a list and a key into an object alike,
            # since JSON object keys are always text.
            value = value[str(step)]
        else:
            return MISSING
    if node.each is not None:
        if not isinstance(value, list):
            return MISSING
        return [items(node.each, item) for item in value]
    if node.measure == "length":
        if not isinstance(value, str | list):
            return MISSING
        value = len(value)
    if node.categories is not None:
        allowed = isinstance(value, str) and value in node.categories
    else:
        low, high = node.bounds
        allowed = isinstance(value, int | float) and not isinstance(value, bool)
        allowed = allowed and low <= value <= high
    # A value outside the declared domain is skipped like a missing path, so every value
    # downstream nodes see stays within the domain validation relied on.
    return value if allowed else MISSING


def items(fields: dict[str, ItemField], item) -> dict:
    """An item's declared fields that are there and within their domains; a field outside its
    domain is absent, so a test on it fails as a test on a missing reading does."""
    found = {}
    for name, declared in fields.items():
        value = item.get(name, MISSING) if isinstance(item, dict) else MISSING
        if declared.categories is not None:
            allowed = isinstance(value, str) and value in declared.categories
        else:
            allowed = isinstance(value, int | float) and not isinstance(value, bool)
            allowed = allowed and declared.bounds[0] <= value <= declared.bounds[1]
        if allowed:
            found[name] = value
    return found


def resolve_node(node: Node, state: State, values: dict, answers: dict, suppressed: bool = False):
    """Resolve deterministic values, or a call's input object; skipped nodes stay absent."""
    if suppressed or not matches(node.when, values):
        return MISSING
    if isinstance(node, Value):
        return deepcopy(node.value)
    if isinstance(node, Merge):
        return resolve_inputs(node.inputs, state, values, answers)
    if isinstance(node, Extract):
        return resolve_inputs(node.inputs, state, values, answers)
    if isinstance(node, Call):
        inputs = resolve_inputs(node.inputs, state, values, answers)
        if inputs is MISSING or node.briefing is None:
            return inputs
        # First, so the judge reads the guidance before the input it applies to.
        return {"briefing": node.briefing, **inputs}
    if isinstance(node, Graph):
        return resolve(node.input, state, values, answers)
    if isinstance(node, Coalesce):
        return next((values[source] for source in node.sources if source in values), MISSING)
    if isinstance(node, Vote):
        cast = [values[source] for source in node.sources if source in values]
        if not cast:
            return MISSING
        if node.rule == "agree":
            # An opinion that was never given has not agreed, so agreement waits for them all.
            if len(cast) < len(node.sources):
                return MISSING
            return node.same if len(set(cast)) == 1 else node.different
        # Ties go to whichever value a source named first, so a vote never depends on ordering
        # inside the counter.
        return max(set(cast), key=lambda value: (cast.count(value), -cast.index(value)))
    if isinstance(node, Sum):
        if any(term.node not in values for term in node.terms):
            return MISSING
        return node.constant + sum(term.weight * values[term.node] for term in node.terms)
    if isinstance(node, Read):
        # A field is read from what its extraction filled, absent when the extraction never ran.
        return read_path(node, state if node.source is None else values.get(node.source, MISSING))
    if isinstance(node, Quantify):
        if node.source not in values:
            return MISSING
        meeting = [matches(node.where, item) for item in values[node.source]]
        if node.rule == "count":
            return sum(meeting)
        # Like a named condition: a value only when it holds, so a rule tests it by naming it.
        held = any(meeting) if node.rule == "any" else all(meeting)
        return True if held else MISSING
    if isinstance(node, Confidence):
        # A probability answer is its own confidence, so there is nothing separate to report.
        answer = answers.get(node.source)
        return answer["confidence"] if answer and "confidence" in answer else MISSING
    if node.source not in values:
        return MISSING
    value = values[node.source]
    if isinstance(node, Switch):
        return node.cases[value]
    return node.above if value >= node.cutoff else node.below


class Remembered:
    """A judge that answers each question about each input once. Search scores candidates that
    mostly keep one another's questions, and ablation replays a graph with one input switched
    off; either way the same question about the same input would otherwise be paid for again,
    and could come back different. A question is known by what it asks, not its name, so a
    renamed but identical question is not asked twice."""

    def __init__(self, client: TypeSafeClient):
        self.client = client
        self.answers: dict[str, tuple[object, str]] = {}

    def system_one(self, *, state, questions, model):
        def key(name):
            return json.dumps([state, questions[name], model], sort_keys=True)

        unasked = {name: q for name, q in questions.items() if key(name) not in self.answers}
        if unasked:
            response = self.client.system_one(state=state, questions=unasked, model=model)
            # Checked here as well as by the caller, since a bad answer must not be remembered.
            if set(response.answers) != set(unasked):
                raise ResponseError("answer IDs do not match the requested question IDs")
            for name in unasked:
                self.answers[key(name)] = response.answers[name], response.model
        remembered = {name: self.answers[key(name)] for name in questions}
        answers = {name: answer for name, (answer, _) in remembered.items()}
        # Answers already paid for cost nothing now; a fresh request reports its own usage.
        usage = response.usage if unasked else Usage(input_tokens=0, output_tokens=0)
        _, served = next(iter(remembered.values()))
        return SystemOneResponse(model=served, usage=usage, answers=answers)


def fallbacks(plan: Plan) -> dict[str, list[list[str]]]:
    """For each judgment used only as a fallback, the sources ahead of it in every rule order
    that uses it. Rules are tried in order and the first to answer decides, so such a judgment
    is worth asking only once everything ahead of it has settled without an answer. A judgment
    anything else reads (a condition, a merge, a confidence, an output) is always asked."""
    dependencies = plan.dependencies()
    uses: dict[str, list[list[str]] | None] = {}
    for name, node in plan.nodes.items():
        tested_here = tested(node.when)
        if isinstance(node, Coalesce):
            for index, source in enumerate(node.sources):
                if source not in tested_here and uses.get(source, []) is not None:
                    uses.setdefault(source, []).append(node.sources[:index])
        for used in dependencies[name] - (
            set(node.sources) - tested_here if isinstance(node, Coalesce) else set()
        ):
            uses[used] = None
    for source in plan.outputs.values():
        if isinstance(source, NodeInput):
            uses[source.node] = None

    def ancestors(name: str, seen: set[str]) -> set[str]:
        for parent in dependencies[name] - seen:
            seen.add(parent)
            ancestors(parent, seen)
        return seen

    found = {}
    for name, ahead in uses.items():
        if ahead is None or not isinstance(plan.nodes[name], (Call, Graph)):
            continue
        # Waiting on a source that itself waits on this judgment would never end.
        if any(name in ancestors(source, set()) for sources in ahead for source in sources):
            continue
        found[name] = ahead
    return found


def demand(ahead: list[list[str]] | None, values: dict, settled: set[str]) -> str:
    """Whether a judgment is needed now: 'ask', 'wait' until what is ahead of it settles, or
    'skip' because in every rule order that uses it something ahead has already answered."""
    if ahead is None:
        return "ask"
    waiting = False
    for sources in ahead:
        if any(source in values for source in sources):
            continue
        if all(source in settled for source in sources):
            return "ask"
        waiting = True
    return "wait" if waiting else "skip"


def decider(plan: Plan, name: str, values: dict) -> str:
    """The node whose value an output took, followed through each list of fallbacks to the rule
    that answered."""
    node = plan.nodes[name]
    while isinstance(node, Coalesce):
        name = next(source for source in node.sources if source in values)
        node = plan.nodes[name]
    return name


def sureness(answer: dict) -> float:
    """How sure one judgment was of its answer: the probability of the option it chose, its
    own confidence for a score, and for a probability, the likelier of true and false."""
    if answer["type"] == "choice":
        return answer["probabilities"][answer["value"]]
    if answer["type"] == "score":
        return answer["confidence"]
    return max(answer["value"], 1 - answer["value"])


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class Extracted(NamedTuple):
    """What an extraction filled, and what the call to fill it cost."""

    fields: dict
    usage: dict


class Judge(NamedTuple):
    """A judge a graph names: the client that reaches it and the model it asks for."""

    client: TypeSafeClient
    model: str


@dataclass(frozen=True)
class Models:
    """The models a graph may ask beside its own judge, each under the name the graph uses."""

    judges: Mapping[str, Judge] = field(default_factory=dict)
    # Called with an extraction's instructions, its fields and its input; answers the fields it
    # could fill and the call's usage. rulesmith.extract.Extractor is the one the CLI builds.
    extractor: Callable[[str, dict[str, Read], State], tuple[dict, dict]] | None = None

    def remembered(self) -> "Models":
        """The same models, each answering a question about an input once."""
        judges = {name: Judge(Remembered(j.client), j.model) for name, j in self.judges.items()}
        return Models(judges, self.extractor)


OWN_JUDGE = "the judge"
EXTRACTOR = "the extractor"


def payer(call: dict) -> str:
    """Who a recorded call was made to: the graph's own judge, a named one, or the extractor."""
    if "extracted" in call:
        return EXTRACTOR
    return f"judge {call['judge']}" if call.get("judge") else OWN_JUDGE


class Prices(StrictModel):
    """What a model call costs, as a share of an example's score, so that of two graphs that
    answer alike the cheaper one wins: a question to the graph's own judge, a question to each
    named judge, and an extraction. A named judge without a price costs what the own one does."""

    call: float = Field(default=0, ge=0, le=1)
    judges: dict[str, Annotated[float, Field(ge=0, le=1)]] = Field(default_factory=dict)
    extract: float = Field(default=0, ge=0, le=1)

    def of(self, paid: str) -> float:
        if paid == EXTRACTOR:
            return self.extract
        if paid == OWN_JUDGE:
            return self.call
        return self.judges.get(paid.removeprefix("judge "), self.call)

    def charge(self, calls: dict) -> float:
        """The price of every call one decision made."""
        return sum(self.of(payer(call)) for call in calls.values())


class DecisionProgram(dspy.Module):
    def __init__(
        self,
        plan: Plan,
        client: TypeSafeClient | None,
        model: str,
        execution: ExecutionConfig | None = None,
        trace: Path | None = None,
        *,
        models: Models | None = None,
    ):
        super().__init__()
        self.plan = plan
        self.client = client
        self.model = model
        self.execution = execution or ExecutionConfig()
        self.trace = trace
        self.models = models or Models()
        if client is None and any(
            isinstance(node, Call) and node.judge is None for node in plan.nodes.values()
        ):
            raise ValueError("graphs containing calls require a TypeSafe client")
        if missing := sorted(plan.judges() - self.models.judges.keys()):
            raise ValueError(f"the graph asks judges no model was given for: {missing}")
        if plan.extracts() and self.models.extractor is None:
            raise ValueError("the graph extracts fields, so it needs an extractor model")
        self.fallbacks = fallbacks(plan)

    def _ask(self, calls: list[tuple[str, Call]], state: dict) -> dict[str, tuple[dict, dict]]:
        """Ask every call sharing one input object and one judge in a single request."""
        questions = {key: node.question.model_dump() for key, node in calls}
        named = calls[0][1].judge
        client, model = self.models.judges[named] if named else (self.client, self.model)
        response = client.system_one(state=state, questions=questions, model=model)
        if set(response.answers) != set(questions):
            raise ResponseError("answer IDs do not match the requested question IDs")
        results = {}
        for index, (key, node) in enumerate(calls):
            # A request is billed once, so its usage is attributed to its first call only.
            first = index == 0
            usage = {
                "model": response.model,
                "input_tokens": response.usage.input_tokens if first else 0,
                "output_tokens": response.usage.output_tokens if first else 0,
                "request": sorted(questions),
            } | ({"judge": named} if named else {})
            answer = calibrated(read_answer(node.question, response.answers[key]), node.temperature)
            results[key] = answer, usage
        return results

    def _delegate(self, key: str, node: Graph, handed) -> dict[str, dspy.Prediction]:
        """Run a unit's graph on what it was handed, through the same engine, on the same
        judge. A unit whose graph could not decide is a unit that answered nothing."""
        prediction = DecisionProgram(
            node.plan, self.client, self.model, self.execution, models=self.models
        )(state=handed)
        return {key: prediction}

    def _extract(self, key: str, node: Extract, handed) -> dict[str, "Extracted"]:
        fields = self.plan.fields(key)
        return {key: Extracted(*self.models.extractor(node.instructions, fields, handed))}

    def _settle_unit(self, key: str, prediction, values: dict, answers: dict, calls, units):
        """A unit's answer is its graph's chosen output, and its judgments and calls are
        reported under the unit's name, so a map's decision can be audited unit by unit."""
        unit = dict(prediction)
        units[key] = unit
        for name, answer in unit["answers"].items():
            answers[f"{key}.{name}"] = answer
        for name, usage in unit["calls"].items():
            calls[f"{key}.{name}"] = usage
        output = self.plan.nodes[key].output
        if output in unit["outputs"]:
            values[key] = unit["outputs"][output]

    def forward(self, state: State) -> dspy.Prediction:
        state = TypeAdapter(State, config=ConfigDict(allow_inf_nan=False)).validate_python(state)
        prediction = self._decide(state)
        if self.trace is not None:
            self._record(state, prediction)
        return prediction

    def _record(self, state: State, prediction: dspy.Prediction) -> None:
        """Append what was decided, by which rule, on which graph and input, to the trace.
        Inputs are kept as digests, so a trace can be shared without the data it judged."""
        record = {
            "time": datetime.now(UTC).isoformat(),
            "graph_sha256": digest(self.plan.model_dump_json()),
            "input_sha256": digest(json.dumps(state, sort_keys=True)),
            "model": self.model,
            "label": prediction.get("label"),
            "outputs": prediction.outputs,
            "decided_by": prediction.get("decided_by"),
            "confidence": prediction.get("confidence"),
            "asked": list(prediction.answers),
            # Which model each call went to, so escalation is counted in production too.
            "called": {key: payer(call) for key, call in prediction.calls.items()},
            "undecided": prediction.undecided,
        }
        # One unbuffered append per record, so concurrent decisions never split each other's line.
        with self.trace.open("ab", buffering=0) as trace:
            trace.write((json.dumps(record) + "\n").encode())

    def _decide(self, state: State) -> dspy.Prediction:
        values, answers, calls, units, delegated = {}, {}, {}, {}, {}
        sorter = TopologicalSorter(self.plan.dependencies())
        sorter.prepare()
        ready, running, settled, stuck = [], {}, set(), False

        def done(key: str) -> None:
            sorter.done(key)
            settled.add(key)

        with ThreadPoolExecutor(max_workers=self.execution.max_workers) as pool:
            while sorter.is_active():
                # Drain locally executable nodes before waiting on network work.
                progressed = True
                while progressed:
                    ready.extend(sorter.get_ready())
                    deferred, progressed, requests, delegations = [], False, {}, []
                    for key in sorted(ready):
                        node = self.plan.nodes[key]
                        value = resolve_node(
                            node, state, values, answers, key in self.execution.suppress
                        )
                        needed = (
                            "ask" if stuck else demand(self.fallbacks.get(key), values, settled)
                        )
                        asks = isinstance(node, (Call, Graph, Extract))
                        if value is MISSING or (asks and needed == "skip"):
                            done(key)
                            progressed = True
                        elif asks and needed == "wait":
                            deferred.append(key)
                        elif isinstance(node, Call):
                            # One request per input and judge: a judge answers only its own.
                            group = json.dumps([node.judge, value], sort_keys=True)
                            requests.setdefault(group, (value, []))[1].append((key, node))
                        elif isinstance(node, (Graph, Extract)):
                            if isinstance(node, Graph):
                                delegated[key] = value
                            delegations.append((key, node, value))
                        else:
                            values[key] = value
                            done(key)
                            progressed = True
                    for value, group in requests.values():
                        if len(running) < self.execution.max_workers:
                            running[pool.submit(self._ask, group, value)] = group[0][0]
                        else:
                            deferred.extend(key for key, _ in group)
                    for key, node, handed in delegations:
                        if len(running) < self.execution.max_workers:
                            work = self._extract if isinstance(node, Extract) else self._delegate
                            running[pool.submit(work, key, node, handed)] = key
                        else:
                            deferred.append(key)
                    ready = deferred
                # Judgments waiting only on one another would wait for ever; ask them all.
                stuck = not running and bool(ready)
                if running:
                    completed, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in sorted(completed, key=running.get):
                        running.pop(future)
                        for key, result in sorted(future.result().items()):
                            if isinstance(result, dspy.Prediction):
                                self._settle_unit(key, result, values, answers, calls, units)
                            elif isinstance(result, Extracted):
                                values[key], calls[key] = result.fields, result.usage
                            else:
                                answer, usage = result
                                answers[key], calls[key] = answer, usage
                                values[key] = answer["value"]
                            done(key)

        outputs, missing = {}, []
        for name, source in self.plan.outputs.items():
            value = resolve(source, state, values, answers)
            if value is not MISSING:
                outputs[name] = value
            elif not source.optional:
                missing.append(name)
        order = self.plan.order()
        result = dict(
            outputs=outputs,
            node_values={key: values[key] for key in order if key in values},
            # A unit's judgments and calls follow its own name, after the graph's own.
            answers={key: answers[key] for key in order if key in answers}
            | {key: answer for key, answer in answers.items() if key not in order},
            skipped=[key for key in order if key not in values],
            # What each unit that ran was handed: the state its own graph decided on, which is
            # what a unit's own task is made of.
            handed={key: delegated[key] for key in order if key in delegated},
            calls={key: calls[key] for key in order if key in calls}
            | {key: usage for key, usage in calls.items() if key not in order},
            models=sorted({call["model"] for call in calls.values()}),
            input_tokens=sum(call["input_tokens"] for call in calls.values()),
            output_tokens=sum(call["output_tokens"] for call in calls.values()),
        )
        if "label" in outputs:
            result["label"] = outputs["label"]
            source = self.plan.outputs["label"]
            decided = (
                decider(self.plan, source.node, values) if isinstance(source, NodeInput) else None
            )
            measured = getattr(self.plan.nodes.get(decided), "measured", None)
            # A unit's decision names the rule inside it, and is no surer than that unit was.
            if decided in units and units[decided].get("decided_by"):
                decided = f"{decided}.{units[decided]['decided_by']}"
            result["decided_by"] = decided
            sure = [sureness(answer) for answer in result["answers"].values()]
            sure += [unit["confidence"] for unit in units.values() if "confidence" in unit]
            result["confidence"] = min(
                sure + ([measured.precision] if measured else []), default=1.0
            )
        # Without every required output there is no decision, whatever the outputs that were made
        # say; the caller reads why rather than catching an exception.
        result["undecided"] = NO_RULE_APPLIED if missing else None
        if missing:
            result |= {"label": None, "decided_by": None}
        return dspy.Prediction(**result)
