"""A typed, finite DAG language for model calls and deterministic data flow."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from graphlib import CycleError, TopologicalSorter
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# A name a graph and the command line both write, so it is a plain word in both.
Name = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_]\w*$")]
State = str | dict[str, JsonValue] | list[JsonValue]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Choice(StrictModel):
    type: Literal["choice"] = "choice"
    instructions: Text
    criteria: dict[Text, Text] = Field(min_length=2, max_length=255)


class Score(StrictModel):
    type: Literal["score"] = "score"
    instructions: Text
    criteria: list[Text] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def distinct_levels(self) -> Self:
        if len(set(self.criteria)) != len(self.criteria):
            raise ValueError("score criteria must be distinct")
        return self


class Noul(StrictModel):
    type: Literal["noul"] = "noul"
    instructions: Text


Question = Annotated[Choice | Score | Noul, Field(discriminator="type")]


class StateInput(StrictModel):
    type: Literal["state"] = "state"


class LiteralInput(StrictModel):
    type: Literal["literal"] = "literal"
    value: JsonValue


class NodeInput(StrictModel):
    type: Literal["node"] = "node"
    node: Text
    view: Literal["value", "answer"] = "value"
    optional: bool = False


Source = Annotated[StateInput | LiteralInput | NodeInput, Field(discriminator="type")]


class Compare(StrictModel):
    type: Literal["compare"] = "compare"
    node: Text
    op: Literal["eq", "lt", "ge"]
    value: str | float


class Relation(StrictModel):
    """One value compared with another, where Compare holds a node against a constant."""

    type: Literal["relation"] = "relation"
    node: Text
    op: Literal["eq", "lt", "ge"]
    other: Text


class Present(StrictModel):
    type: Literal["present"] = "present"
    node: Text


class All(StrictModel):
    type: Literal["all"] = "all"
    conditions: list[Condition] = Field(min_length=1)


class Any(StrictModel):
    type: Literal["any"] = "any"
    conditions: list[Condition] = Field(min_length=1)


class Not(StrictModel):
    type: Literal["not"] = "not"
    condition: Condition


Condition = Annotated[Compare | Relation | Present | All | Any | Not, Field(discriminator="type")]
All.model_rebuild()
Any.model_rebuild()
Not.model_rebuild()


def reachable(dependencies: dict[str, set[str]], outputs) -> set[str]:
    """The nodes an output depends on, following references as far as they go."""
    used: set[str] = set()
    pending = [source.node for source in outputs if isinstance(source, NodeInput)]
    while pending:
        key = pending.pop()
        if key not in used and key in dependencies:
            used.add(key)
            pending.extend(dependencies[key])
    return used


def references(value: JsonValue) -> set[str]:
    """References in graph syntax; literal payloads and question text are opaque."""
    if isinstance(value, list):
        return set().union(*(references(v) for v in value))
    if not isinstance(value, dict):
        return set()
    named: set[str] = set()
    kind = value.get("type")
    if kind in ("node", "compare", "relation", "present") and isinstance(value.get("node"), str):
        named.add(value["node"])
    if kind == "relation" and isinstance(value.get("other"), str):
        named.add(value["other"])
    if kind in ("switch", "threshold", "confidence", "quantify") and isinstance(
        value.get("source"), str
    ):
        named.add(value["source"])
    if kind in ("coalesce", "vote") and isinstance(value.get("sources"), list):
        named.update(source for source in value["sources"] if isinstance(source, str))
    if kind == "sum" and isinstance(value.get("terms"), list):
        named.update(
            term["node"]
            for term in value["terms"]
            if isinstance(term, dict) and isinstance(term.get("node"), str)
        )
    for key in ("when", "condition", "conditions"):
        named.update(references(value.get(key)))
    if kind == "read" and isinstance(value.get("source"), str):
        named.add(value["source"])
    if kind in ("call", "merge", "extract") and isinstance(value.get("inputs"), dict):
        for source in value["inputs"].values():
            named.update(references(source))
    if kind == "graph":
        named.update(references(value.get("input")))
    return named


def check_items(condition: Condition, fields: dict[str, ItemField]) -> None:
    """A test on a list's items names only fields the items declare, compared as their kind."""
    domains = {
        name: frozenset(field.categories) if field.categories else field.bounds
        for name, field in fields.items()
    }
    for leaf in predicates(condition):
        named = [leaf.node] + ([leaf.other] if isinstance(leaf, Relation) else [])
        for name in named:
            if name not in domains:
                raise ValueError(f"the items have no field {name!r}; they have {sorted(domains)}")
        if isinstance(leaf, Present):
            continue
        domain = domains[leaf.node]
        if isinstance(leaf, Relation):
            if type(domain) is not type(domains[leaf.other]):
                raise ValueError("a relation compares two values of the same kind")
            if leaf.op != "eq" and not isinstance(domain, tuple):
                raise ValueError("an ordering relation requires numeric fields")
        elif isinstance(leaf.value, str):
            if leaf.op != "eq" or not isinstance(domain, frozenset) or leaf.value not in domain:
                raise ValueError(f"the field {leaf.node} is never {leaf.value!r}")
        elif not isinstance(domain, tuple):
            raise ValueError(f"numeric comparison requires a numeric field, and {leaf.node} is not")


def truth(condition: Condition, known: dict[str, bool]) -> bool | None:
    """A condition's value given what is known of its tests, or None while it still depends
    on one that is not known, each test keyed by its JSON."""
    if isinstance(condition, Not):
        inner = truth(condition.condition, known)
        return None if inner is None else not inner
    if isinstance(condition, (All, Any)):
        truths = [truth(child, known) for child in condition.conditions]
        # True settles an any, and false settles an all.
        deciding = isinstance(condition, Any)
        if deciding in truths:
            return deciding
        return None if None in truths else not deciding
    return known.get(condition.model_dump_json())


def predicates(condition: Condition | None) -> list[Compare | Relation | Present]:
    if condition is None:
        return []
    if isinstance(condition, (Compare, Relation, Present)):
        return [condition]
    if isinstance(condition, Not):
        return predicates(condition.condition)
    return [leaf for child in condition.conditions for leaf in predicates(child)]


def tested(condition: Condition | None) -> set[str]:
    """The nodes a condition reads."""
    named = set()
    for leaf in predicates(condition):
        named.add(leaf.node)
        if isinstance(leaf, Relation):
            named.add(leaf.other)
    return named


class Guarded(StrictModel):
    when: Condition | None = None


class Measured(StrictModel):
    """How a rule did on labeled examples: how many it decided, and how many of those rightly.
    Measured by rulesmith calibrate, it says how sure the rule's decisions are and whether new
    inputs still reach it about as often as the ones it was measured on."""

    right: int = Field(ge=0)
    decided: int = Field(ge=0)
    examples: int = Field(ge=1)

    @model_validator(mode="after")
    def nested(self) -> Self:
        if not self.right <= self.decided <= self.examples:
            raise ValueError(
                "a rule is right at most as often as it decides, and decides at most every example"
            )
        return self

    @property
    def precision(self) -> float:
        """How often the rule is right when it decides, smoothed toward even odds by Laplace's
        rule of succession, so a rule seen deciding twice does not claim certainty."""
        return (self.right + 1) / (self.decided + 2)

    @property
    def share(self) -> float:
        return self.decided / self.examples


class Decides(Guarded):
    """A node a rule list can end on, the way a rule answers without asking."""

    measured: Measured | None = None


class Call(Guarded):
    type: Literal["call"] = "call"
    question: Question
    inputs: dict[Text, Source] = Field(min_length=1)
    # What the judge reads before the input: what to look for, how the labels differ, the cases
    # that are easy to confuse. It goes in the state rather than the question because that is
    # where a small judge takes guidance from; the same text in the question changes nothing.
    briefing: Text | None = None
    temperature: float = Field(
        default=1,
        gt=0,
        description="Divides the judge's log-odds before they are read: above 1 tempers an "
        "overconfident judge, below 1 sharpens a timid one. Fitted by rulesmith calibrate",
    )
    judge: Name | None = Field(
        default=None,
        description="A named judge to ask instead of the graph's own, such as a larger model for "
        "the cases the small one is unsure of; where the graph runs, each name is given a model",
    )


class Switch(Guarded):
    type: Literal["switch"] = "switch"
    source: Text
    cases: dict[Text, Text] = Field(min_length=1)


class Threshold(Decides):
    type: Literal["threshold"] = "threshold"
    source: Text
    cutoff: float
    below: Text
    above: Text


class Merge(Guarded):
    type: Literal["merge"] = "merge"
    inputs: dict[Text, Source] = Field(min_length=1)


# What the rules compiler names each rule, so a rule is never read as a judgment of its own.
RULE = re.compile(r"rule_\d+")


class Coalesce(Guarded):
    type: Literal["coalesce"] = "coalesce"
    sources: list[Text] = Field(min_length=1)


class Value(Decides):
    type: Literal["value"] = "value"
    value: JsonValue


class Vote(Guarded):
    """Several judgments settled into one value. `majority` takes the value most of them gave,
    breaking a tie toward the earlier source; `agree` reports whether they all said the same,
    which is how a second opinion verifies a first."""

    type: Literal["vote"] = "vote"
    sources: list[Text] = Field(min_length=2)
    rule: Literal["majority", "agree"] = "majority"
    same: Text = "yes"
    different: Text = "no"


class Confidence(Guarded):
    """How sure a judgment was, as a number a rule can test. A choice or a score answer carries
    its own confidence; a probability answers with the probability itself, so it has none to
    report and this is absent for one."""

    type: Literal["confidence"] = "confidence"
    source: Text


class Term(StrictModel):
    node: Text
    weight: float = 1


class Sum(Guarded):
    """A weighted sum of numeric values plus a constant: a margin, a total, a scaled
    difference. It is skipped when any term is, since a missing term is not a zero."""

    type: Literal["sum"] = "sum"
    terms: list[Term] = Field(min_length=1)
    constant: float = 0


class Extract(Guarded):
    """A generative model reading its inputs into fields: each field is a read of this node, with
    the domain the model must answer in. A field it cannot fill, or fills outside that domain, is
    absent, like an input's missing path."""

    type: Literal["extract"] = "extract"
    instructions: Text
    inputs: dict[Text, Source] = Field(min_length=1)


class ItemField(StrictModel):
    """A field of each item in a list, with its domain, as a reading declares one."""

    bounds: tuple[float, float] | None = None
    categories: list[Text] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def one_domain(self) -> Self:
        if (self.bounds is None) == (self.categories is None):
            raise ValueError("an item's field declares exactly one of bounds or categories")
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError("an item's field bounds must be increasing")
        return self


class Read(Guarded):
    """A value taken from the original input by path, with its declared domain; with `source`,
    a field of that extraction instead."""

    type: Literal["read"] = "read"
    source: Name | None = None
    path: list[str | int] = Field(min_length=1)
    bounds: tuple[float, float] | None = None
    categories: list[Text] | None = Field(default=None, min_length=1)
    each: dict[Name, ItemField] | None = Field(
        default=None,
        min_length=1,
        description="A list, read as its items' fields: an item that is not an object, or a "
        "field missing or outside its domain, leaves that field absent for that item",
    )
    measure: Literal["value", "length"] = Field(
        default="value",
        description="The value itself, or how long it is when it is a text or a list: an empty "
        "field or a one-line answer is often as telling as what it says, and costs nothing",
    )

    @model_validator(mode="after")
    def one_domain(self) -> Self:
        if sum(domain is not None for domain in (self.bounds, self.categories, self.each)) != 1:
            raise ValueError("a read declares exactly one of bounds, categories or each")
        if self.measure == "length" and self.bounds is None:
            raise ValueError("a length is a number, so its read declares bounds")
        if self.bounds is not None and self.bounds[0] > self.bounds[1]:
            raise ValueError("read bounds must be increasing")
        if self.source is not None and (len(self.path) != 1 or self.measure != "value"):
            raise ValueError("a field of an extraction is read by its name alone")
        return self


class Quantify(Guarded):
    """A test over a list's items: `any` holds when some item meets `where`, `all` when every
    item does, and either is absent otherwise, as a named condition is; `count` is how many
    do. `where` tests the items' fields by name, and a field an item lacks fails its test."""

    type: Literal["quantify"] = "quantify"
    source: Text
    rule: Literal["any", "all", "count"]
    where: Condition


class Graph(Guarded):
    """Another plan, run on the one value this node is handed and answering with one of its
    outputs: a unit of a larger map. The value may be the whole input, a node's value, or an
    object a merge built, so the unit's own reads stay relative to what it receives. A parent's
    rules test the unit's answer only against the values its graph can give, so the boundary is
    typed like any other edge."""

    type: Literal["graph"] = "graph"
    plan: Plan | None = None
    path: str | None = Field(
        default=None,
        description="A file holding the unit's graph, .json or .rules, relative to the file "
        "naming it; rulesmith.maps.load embeds it, and a plan is what validation needs",
    )
    input: Source = Field(default_factory=StateInput)
    output: Text = "label"

    @model_validator(mode="after")
    def answers_with_an_output_it_has(self) -> Self:
        if self.plan is None:
            raise ValueError(
                f"a unit named by path ({self.path!r}) must be loaded first: "
                "rulesmith.maps.load reads the map and embeds its units"
            )
        if self.path is not None:
            raise ValueError("a unit is embedded as a plan or named by path, not both")
        if self.output not in self.plan.outputs:
            raise ValueError(f"the graph has no output {self.output!r} to answer with")
        return self


Node = Annotated[
    Call
    | Switch
    | Threshold
    | Merge
    | Coalesce
    | Value
    | Read
    | Extract
    | Quantify
    | Confidence
    | Vote
    | Sum
    | Graph,
    Field(discriminator="type"),
]
Domain = frozenset[str] | tuple[float, float] | None


def literal_domain(value: JsonValue) -> Domain:
    if isinstance(value, str):
        return frozenset([value])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (value, value)
    return None


class Plan(StrictModel):
    nodes: dict[Text, Node]
    outputs: dict[Text, Source] = Field(min_length=1)

    def dependencies(self) -> dict[str, set[str]]:
        result = {}
        for key, node in self.nodes.items():
            refs = tested(node.when)
            if isinstance(node, (Call, Merge, Extract)):
                refs.update(s.node for s in node.inputs.values() if isinstance(s, NodeInput))
            elif isinstance(node, Read) and node.source is not None:
                refs.add(node.source)
            elif isinstance(node, (Switch, Threshold, Confidence, Quantify)):
                refs.add(node.source)
            elif isinstance(node, (Coalesce, Vote)):
                refs.update(node.sources)
            elif isinstance(node, Sum):
                refs.update(term.node for term in node.terms)
            elif isinstance(node, Graph) and isinstance(node.input, NodeInput):
                refs.add(node.input.node)
            result[key] = refs
        return result

    def contributing(self) -> set[str]:
        """Every node an output depends on, directly or through another node."""
        return reachable(self.dependencies(), self.outputs.values())

    def judges(self) -> set[str]:
        """The named judges this graph asks, in it or in a unit of it."""
        return {
            node.judge for node in self.nodes.values() if isinstance(node, Call) and node.judge
        }.union(*(node.plan.judges() for node in self.nodes.values() if isinstance(node, Graph)))

    def extracts(self) -> bool:
        """Whether running this graph asks an extractor, in it or in a unit of it."""
        return any(
            isinstance(node, Extract) or (isinstance(node, Graph) and node.plan.extracts())
            for node in self.nodes.values()
        )

    def fields(self, extraction: str) -> dict[str, Read]:
        """The fields read of an extraction, by name, in the order the graph declares them."""
        return {
            str(node.path[0]): node
            for node in self.nodes.values()
            if isinstance(node, Read) and node.source == extraction
        }

    def asks_own_judge(self) -> bool:
        """Whether running this graph asks the judge it runs with, in it or in a unit of it; a
        judgment asked of a named judge goes to that one instead."""
        return any(
            (isinstance(node, Call) and node.judge is None)
            or (isinstance(node, Graph) and node.plan.asks_own_judge())
            for node in self.nodes.values()
        )

    def answers(self, output: str) -> bool:
        """Whether every input gets a value for this output, an optional input counting as
        given. The check is static and errs toward no: a read is taken as possibly absent, and
        the tests in conditions as independent of one another, except that a judgment's choice
        is one of its labels. So a rule list ending on an unconditional rule, a judgment, or a
        rule for whatever the rules above it left out always answers."""
        sure: dict[str, bool] = {}
        domains = self.domains()

        def given(source: Source) -> bool:
            if isinstance(source, NodeInput):
                return source.optional or answers(source.node)
            return True

        def answers(key: str) -> bool:
            if key not in sure:
                node = self.nodes[key]
                sure[key] = made(node) and (node.when is None or covered([node.when]))
            return sure[key]

        def covered(conditions: list[Condition]) -> bool:
            """Whether one of these conditions holds whatever the tests in them come to."""
            leaves = {leaf.model_dump_json(): leaf for c in conditions for leaf in predicates(c)}
            chosen: dict[str, list[str]] = {}
            unknowns = []
            for key, leaf in leaves.items():
                if isinstance(domains.get(leaf.node), frozenset) and (
                    isinstance(leaf, Present) or (isinstance(leaf, Compare) and leaf.op == "eq")
                ):
                    chosen.setdefault(leaf.node, []).append(key)
                elif isinstance(leaf, Present) and answers(leaf.node):
                    unknowns.append([{key: True}])
                else:
                    unknowns.append([{key: True}, {key: False}])
            for node, keys in chosen.items():
                # One outcome per label the choice can give, and one for no answer at all.
                outcomes = [
                    {
                        key: isinstance(leaves[key], Present) or leaves[key].value == label
                        for key in keys
                    }
                    for label in domains[node]
                ]
                if not answers(node):
                    outcomes.append(dict.fromkeys(keys, False))
                unknowns.append(outcomes)

            def holds(known: dict[str, bool], rest: list) -> bool:
                truths = [truth(condition, known) for condition in conditions]
                if True in truths:
                    return True
                if None not in truths:
                    return False
                return all(holds(known | outcome, rest[1:]) for outcome in rest[0])

            return holds({}, unknowns)

        def made(node: Node) -> bool:
            """Whether the node has a value whenever its own condition lets it run."""
            if isinstance(node, Read):
                return False
            if isinstance(node, Value):
                return True
            if isinstance(node, (Call, Merge, Extract)):
                return all(given(source) for source in node.inputs.values())
            if isinstance(node, Coalesce) or (isinstance(node, Vote) and node.rule == "majority"):
                runs = [self.nodes[source] for source in node.sources]
                guards = [source.when for source in runs if made(source)]
                return None in guards or (bool(guards) and covered(guards))
            if isinstance(node, Vote):
                return all(answers(source) for source in node.sources)
            if isinstance(node, Sum):
                return all(answers(term.node) for term in node.terms)
            if isinstance(node, Quantify):
                return node.rule == "count" and answers(node.source)
            if isinstance(node, Graph):
                return given(node.input) and node.plan.answers(node.output)
            # A threshold, a switch, or a confidence, which validation keeps to choices and
            # scores, the kinds of judgment that report one.
            return answers(node.source)

        return given(self.outputs[output])

    def unanswered(self) -> list[str]:
        """The required outputs some input could leave without a value."""
        return [
            name
            for name, source in self.outputs.items()
            if not (isinstance(source, NodeInput) and source.optional) and not self.answers(name)
        ]

    def order(self) -> tuple[str, ...]:
        dependencies = self.dependencies()
        return tuple(
            TopologicalSorter(
                {k: sorted(dependencies[k]) for k in sorted(self.nodes)}
            ).static_order()
        )

    def domains(self) -> dict[str, Domain]:
        """Infer finite label sets and numeric bounds without running model calls."""
        result: dict[str, Domain] = {}
        for key in self.order():
            node = self.nodes[key]
            domain = None
            if isinstance(node, Call):
                q = node.question
                domain = (
                    frozenset(q.criteria)
                    if isinstance(q, Choice)
                    else (0, len(q.criteria) - 1 if isinstance(q, Score) else 1)
                )
            elif isinstance(node, Switch):
                domain = frozenset(node.cases.values())
            elif isinstance(node, Threshold):
                domain = frozenset([node.below, node.above])
            elif isinstance(node, Value):
                domain = literal_domain(node.value)
            elif isinstance(node, Read):
                domain = frozenset(node.categories) if node.categories else node.bounds
            elif isinstance(node, Quantify) and node.rule == "count":
                domain = (0, math.inf)
            elif isinstance(node, Confidence):
                domain = (0.0, 1.0)
            elif isinstance(node, Graph):
                # The unit answers with what its own graph can put on that output.
                answer = node.plan.outputs[node.output]
                if isinstance(answer, LiteralInput):
                    domain = literal_domain(answer.value)
                elif answer.view == "value" and not answer.optional:
                    domain = node.plan.domains()[answer.node]
            elif isinstance(node, Sum):
                ranges = [result[term.node] for term in node.terms]
                if all(isinstance(r, tuple) for r in ranges):
                    # Each term spans weight * low to weight * high, whichever way the sign runs.
                    ends = [
                        sorted((term.weight * low, term.weight * high))
                        for term, (low, high) in zip(node.terms, ranges, strict=True)
                    ]
                    domain = (
                        node.constant + sum(low for low, _ in ends),
                        node.constant + sum(high for _, high in ends),
                    )
            elif isinstance(node, Vote):
                if node.rule == "agree":
                    domain = frozenset([node.same, node.different])
                else:
                    voted = [result[source] for source in node.sources]
                    if all(isinstance(d, frozenset) for d in voted):
                        domain = frozenset().union(*voted)
            elif isinstance(node, Coalesce):
                inputs = [result[source] for source in node.sources]
                if all(isinstance(d, frozenset) for d in inputs):
                    domain = frozenset().union(*inputs)
                elif all(isinstance(d, tuple) for d in inputs):
                    domain = (min(d[0] for d in inputs), max(d[1] for d in inputs))
            result[key] = domain
        return result

    @model_validator(mode="after")
    def typed_references(self) -> Self:
        dependencies = self.dependencies()
        sources = list(self.outputs.values())
        for node in self.nodes.values():
            if isinstance(node, (Call, Merge, Extract)):
                sources.extend(node.inputs.values())
            elif isinstance(node, Graph):
                sources.append(node.input)
        references = {r for parents in dependencies.values() for r in parents}
        references.update(s.node for s in sources if isinstance(s, NodeInput))
        unknown = references - self.nodes.keys()
        if unknown:
            raise ValueError(f"unknown nodes: {sorted(unknown)}")
        for source in sources:
            if (
                isinstance(source, NodeInput)
                and source.view == "answer"
                and not isinstance(self.nodes[source.node], Call)
            ):
                raise ValueError("answer view requires a call node")
        for key, node in self.nodes.items():
            if isinstance(node, Quantify):
                listed = self.nodes[node.source]
                if not isinstance(listed, Read) or listed.each is None:
                    raise ValueError(
                        f"{key!r} counts items of {node.source!r}, which is not a list"
                    )
                check_items(node.where, listed.each)
            if isinstance(node, Read) and node.source is not None:
                if not isinstance(self.nodes.get(node.source), Extract):
                    raise ValueError(
                        f"read {key!r} takes a field of {node.source!r}, not an extraction"
                    )
        try:
            self.order()
        except CycleError as error:
            raise ValueError(f"graph contains a cycle: {error.args[1]}") from error

        if self.contributing() != self.nodes.keys():
            raise ValueError("every node must contribute to a graph output")
        # A vote counts labels. A probability or a rubric position is a float, and two floats
        # agreeing exactly is a coincidence; a merged object cannot be counted at all.
        domains = self.domains()
        for key, node in self.nodes.items():
            if isinstance(node, Vote) and not all(
                isinstance(domains[source], frozenset) for source in node.sources
            ):
                raise ValueError(
                    f"vote {key!r} counts labels, so every source needs a categorical value"
                )
            if isinstance(node, Sum) and not all(
                isinstance(domains[term.node], tuple) for term in node.terms
            ):
                raise ValueError(f"sum {key!r} adds numbers, so every term needs a numeric value")
            # A reading has no confidence and a probability answers with its own, so a
            # confidence of either is absent on every input and its rules never fire.
            if isinstance(node, Confidence):
                source = self.nodes[node.source]
                if not isinstance(source, Call) or isinstance(source.question, Noul):
                    raise ValueError(
                        f"confidence {key!r} needs a choice or score judgment as its source"
                    )
        for node in self.nodes.values():
            if isinstance(node, Switch):
                domain = domains[node.source]
                if not isinstance(domain, frozenset) or set(node.cases) != domain:
                    raise ValueError("switch cases must exactly cover the source categories")
            elif isinstance(node, Threshold):
                domain = domains[node.source]
                if not isinstance(domain, tuple):
                    raise ValueError("threshold requires a numeric source")
                if not domain[0] <= node.cutoff <= domain[1]:
                    raise ValueError(f"cutoff must be within source bounds {domain}")
            for condition in predicates(node.when):
                if isinstance(condition, Present):
                    continue
                if isinstance(condition, Relation):
                    pair = domains[condition.node], domains[condition.other]
                    kinds = {type(domain) for domain in pair}
                    if len(kinds) != 1 or None in pair:
                        raise ValueError("a relation compares two values of the same kind")
                    if condition.op != "eq" and kinds != {tuple}:
                        raise ValueError("an ordering relation requires numeric sources")
                    continue
                domain = domains[condition.node]
                if isinstance(condition.value, str) and condition.op == "eq":
                    if not isinstance(domain, frozenset) or condition.value not in domain:
                        can = (
                            f"one of {sorted(domain)}"
                            if isinstance(domain, frozenset)
                            else "a number"
                        )
                        raise ValueError(
                            f"the condition {condition.node} == {condition.value!r} tests a value "
                            f"{condition.node} never has; it is {can}"
                        )
                elif not isinstance(domain, tuple) or isinstance(condition.value, str):
                    raise ValueError("numeric comparison requires a numeric source and value")
        return self

    @classmethod
    def constant(cls, label: str) -> Plan:
        """A call-free graph that always outputs one label."""
        return cls(nodes={}, outputs={"label": LiteralInput(value=label)})

    def decides_by(self, name: str) -> str | None:
        """The judgment or vote a rule hands on: the rule is a coalesce of that one node, which
        has a name of its own. A list of fallbacks, however short, is not one."""
        node = self.nodes[name]
        if not isinstance(node, Coalesce) or len(node.sources) != 1:
            return None
        source = node.sources[0]
        if RULE.fullmatch(source) or not isinstance(self.nodes[source], (Call, Vote)):
            return None
        return source

    def check(self, labels: list[str], max_nodes: int, max_depth: int) -> None:
        """Keep classification targets and search bounds outside mutable candidates."""
        if len(self.nodes) > max_nodes:
            raise ValueError(f"node count exceeds {max_nodes}")
        dependencies, depths = self.dependencies(), {}
        for key in self.order():
            depths[key] = 1 + max((depths[p] for p in dependencies[key]), default=0)
        if max(depths.values(), default=0) > max_depth:
            raise ValueError(f"graph depth exceeds {max_depth}")
        label = self.outputs.get("label")
        domain = None
        if isinstance(label, LiteralInput):
            domain = literal_domain(label.value)
        elif isinstance(label, NodeInput) and label.view == "value" and not label.optional:
            domain = self.domains()[label.node]
        if not isinstance(domain, frozenset) or not domain <= set(labels):
            raise ValueError(
                "label output must have a required categorical value within task labels"
            )


# A unit holds a plan and a plan may hold units, so each finishes the other's schema.
Graph.model_rebuild()
Plan.model_rebuild()


class Example(StrictModel):
    id: Text
    state: State
    # Absent only when a grader, not a label, says whether an answer is right.
    label: Text | None = None
    accepted: list[Text] = Field(
        default_factory=list, description="Other labels that are also right for this example"
    )
    group: Text | None = Field(
        default=None,
        description="Examples that are not independent, a ticket and its follow-up, share a "
        "group, which keeps them in one split",
    )

    @model_validator(mode="after")
    def accepted_beside_a_label(self) -> Self:
        if self.accepted and self.label is None:
            raise ValueError(f"example {self.id} accepts other labels but has none of its own")
        if self.label in self.accepted:
            raise ValueError(f"example {self.id} lists its own label as another accepted one")
        return self


class Abstain(StrictModel):
    """A label that passes an example to a person, at a cost between nothing and a mistake."""

    label: Text
    cost: float = Field(gt=0, lt=1, description="As a share of the dearest mistake")


class Task(StrictModel):
    description: Text
    labels: list[Text] = Field(min_length=2)
    train: list[Example] = Field(min_length=1)
    validation: list[Example] = Field(min_length=1)
    test: list[Example] = Field(min_length=1)
    calibration: list[Example] = Field(
        default_factory=list,
        description="Examples search never sees, for calibrate to fit on; without them it fits "
        "on the training and validation examples, which search chose the graph by",
    )
    costs: dict[Text, dict[Text, Annotated[float, Field(ge=0)]]] = Field(
        default_factory=dict,
        description="costs[expected][answered]: what a wrong answer costs; 1 where unlisted",
    )
    descriptions: dict[Text, Text] = Field(
        default_factory=dict,
        description="What each label means, for a question over them all; its words where unlisted",
    )
    abstain: Abstain | None = None
    prevalence: dict[Text, Annotated[float, Field(gt=0)]] | None = Field(
        default=None,
        description="Each label's share of real traffic, where the examples were sampled at "
        "other rates: a rare label sampled to half the examples makes its mistakes look common",
    )

    def examples(self) -> list[Example]:
        return self.train + self.validation + self.calibration + self.test

    def fitting(self) -> list[Example]:
        """The labeled examples calibrate fits lines and counts on: the calibration split when
        the task has one, else the training examples search saw, which flatter a searched graph."""
        return [e for e in (self.calibration or self.train) if e.label is not None]

    def fitted_on(self) -> str:
        return "calibration" if self.calibration else "train+validation"

    def weights(self) -> dict[str, float]:
        """How much an example of each label stands for in real traffic: its prevalence over
        its share of the labeled examples. Every label weighs 1 without a prevalence."""
        if self.prevalence is None:
            return dict.fromkeys(self.labels, 1.0)
        labeled = [e.label for e in self.examples() if e.label]
        return {
            label: self.prevalence[label] / (labeled.count(label) / len(labeled))
            if label in labeled
            else 1.0
            for label in self.labels
        }

    @model_validator(mode="after")
    def disjoint_splits(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("task labels must be distinct")
        ids, states, groups = set(), set(), {}
        splits = {"train": self.train, "validation": self.validation}
        splits |= {"calibration": self.calibration, "test": self.test}
        for split, examples in splits.items():
            for example in examples:
                for label in [example.label, *example.accepted]:
                    if label is not None and label not in self.labels:
                        raise ValueError(f"unknown label in example {example.id}: {label}")
                state = json.dumps(example.state, sort_keys=True, ensure_ascii=False)
                if example.id in ids:
                    raise ValueError(f"duplicate example id: {example.id}")
                if state in states:
                    raise ValueError(f"duplicate state in example {example.id}")
                if example.group is not None and groups.setdefault(example.group, split) != split:
                    raise ValueError(
                        f"group {example.group!r} is in both {groups[example.group]} and {split}; "
                        "examples that belong together are held out together"
                    )
                ids.add(example.id)
                states.add(state)
        for expected, row in self.costs.items():
            for answered in [expected, *row]:
                if answered not in self.labels:
                    raise ValueError(f"costs name a label the task does not have: {answered}")
            if expected in row:
                raise ValueError(f"costs give {expected!r} a cost for answering itself")
        unknown = self.descriptions.keys() - set(self.labels)
        if unknown:
            raise ValueError(f"descriptions name labels the task does not have: {sorted(unknown)}")
        if self.worst == 0:
            raise ValueError("costs make every mistake free, so no answer is worse than another")
        if self.prevalence is not None:
            if set(self.prevalence) != set(self.labels):
                raise ValueError("prevalence must give a share for every label and no other")
            if abs(sum(self.prevalence.values()) - 1) > 1e-6:
                raise ValueError("prevalence shares must sum to 1")
        if self.abstain is not None:
            passed = self.abstain.label
            if passed not in self.labels:
                raise ValueError(f"the abstain label {passed!r} must be one of the task's labels")
            if passed in self.costs or any(passed in row for row in self.costs.values()):
                raise ValueError("the abstain label's cost is set by abstain, not by costs")
            for example in self.examples():
                if passed in (example.label, *example.accepted):
                    raise ValueError(f"example {example.id} expects {passed!r}, which no person is")
        return self

    def cost(self, expected: str, answered: str) -> float:
        if expected == answered:
            return 0.0
        if self.abstain is not None and answered == self.abstain.label:
            return self.abstain.cost * self.worst
        return self.costs.get(expected, {}).get(answered, 1.0)

    def criteria(self) -> dict[str, str]:
        """Every label with what it means: a choice over the whole task, in the task's order."""
        return {
            label: self.descriptions.get(label, label.replace("_", " ")) for label in self.labels
        }

    @property
    def worst(self) -> float:
        """The dearest mistake the task can make, which scores zero. Passing an example to a
        person is not a mistake, and nothing expects it, so it takes no part."""
        passed = self.abstain.label if self.abstain is not None else None
        answers = [label for label in self.labels if label != passed]
        return max(self.costs.get(e, {}).get(a, 1.0) for e in answers for a in answers if e != a)

    def baseline_plan(self) -> Plan:
        """A call-free majority-label seed, so search can start from the task alone. Graded
        tasks may label nothing, and then the seed is the first label allowed."""
        counts = Counter(e.label for e in self.train if e.label is not None)
        return Plan.constant(counts.most_common(1)[0][0] if counts else self.labels[0])
