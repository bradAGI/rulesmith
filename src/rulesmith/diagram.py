"""A plan as a drawing: boxes, the arrows between them, and each unit as a frame around its
own. Rendered as a Mermaid flowchart for the command line, or handed over as data for a canvas
to lay out, so a graph rules cannot express can still be looked at."""

from dataclasses import asdict, dataclass, field
from textwrap import shorten

from rulesmith.graph import (
    Call,
    Coalesce,
    Confidence,
    Extract,
    Graph,
    LiteralInput,
    Merge,
    NodeInput,
    Plan,
    Present,
    Quantify,
    Read,
    Sum,
    Switch,
    Threshold,
    Value,
    Vote,
    tested,
)
from rulesmith.rules import (
    GENERATED,
    name_of,
    number,
    path_of,
    render_condition,
    render_quantifier,
)

SHAPES = {
    "read": ("[/", "/]"),
    "call": ("{{", "}}"),
    "extract": ("{{", "}}"),
    "threshold": ("[", "]"),
    "switch": ("[", "]"),
    "merge": ("[", "]"),
    "coalesce": ("([", "])"),
    "vote": ("[/", "\\]"),
    "confidence": ("[", "]"),
    "sum": ("[", "]"),
    "quantify": ("[", "]"),
    "value": ("((", "))"),
}
# Colors by what a node costs to run, with dark text so each reads on a light or dark page: the
# judge is the one place a model is asked, whichever backend runs the graph.
KINDS = {
    "judge": (Call, Extract),
    "read": (Read,),
    "rule": (Value, Threshold),
    "unit": (Graph,),
}
STYLES = {
    "judge": "fill:#fcd34d,stroke:#b45309,stroke-width:2px,color:#1c1917",
    "read": "fill:#bfdbfe,stroke:#1d4ed8,color:#1c1917",
    "rule": "fill:#bbf7d0,stroke:#15803d,color:#1c1917",
    "unit": "fill:#ede9fe,stroke:#7c3aed,color:#1c1917",
    "condition": "fill:#e5e7eb,stroke:#6b7280,color:#1c1917",
}
ARROWS = {"data": "-->", "gate": "-.->", "answer": "==>"}


@dataclass(frozen=True)
class Box:
    """One drawn thing: a plan node, a condition gating one, or an output."""

    id: str
    # read, judge, rule, unit, condition, join (the ordered decision), step (anything else
    # that runs without a model) or output.
    kind: str
    lines: tuple[str, ...]
    # The plan node or output drawn; a condition draws none.
    name: str | None = None
    # The node's type as the plan writes it; a condition or output has none.
    type: str | None = None
    # The id of the unit it is drawn inside, or None at the top.
    unit: str | None = None
    # A unit's gate, as its frame's title says it.
    when: str | None = None


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    # data: a value flowing; rank: into an ordered decision, numbered as tried; gate: from a
    # condition to what it gates; answer: into an output, or a unit's answer leaving it.
    kind: str
    rank: int | None = None


@dataclass
class Drawing:
    """A plan's boxes in the order its nodes are written, a unit's boxes after the unit's own,
    and the arrows among them. Closed, a unit is one box with nothing drawn inside."""

    closed: bool = False
    boxes: list[Box] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "closed": self.closed,
            "boxes": [asdict(box) | {"lines": list(box.lines)} for box in self.boxes],
            "edges": [asdict(edge) for edge in self.edges],
        }


def label(name: str, node, reads: dict[str, Read], plan: Plan) -> tuple[str, ...]:
    """What a box says, a line at a time: enough to recognize the node, not its whole
    definition."""
    if isinstance(node, Read):
        return (f"read {path_of(node)}",)
    if isinstance(node, Call):
        # Named as the rules test it, and marked as the model call it is.
        named = "judge" if GENERATED.fullmatch(name) else f"judge: {name}"
        named += f", asked of {node.judge}" if node.judge else ""
        question = shorten(node.question.instructions, 48, placeholder="...")
        return (named, f"{node.question.type}: {question}")
    if isinstance(node, Extract):
        # A model call too, so it wears the judge's color; what it does is read, not judge.
        return (f"extract: {name}", shorten(node.instructions, 48, placeholder="..."))
    if isinstance(node, Threshold):
        return (f"{name}: >= {number(node.cutoff)} ? {node.above} : {node.below}",)
    if isinstance(node, Value):
        # A named condition holds a bare true; its name is what the rules test.
        return (name if node.value is True else str(node.value),)
    if isinstance(node, Switch):
        return (f"{name}: switch on {node.source}",)
    if isinstance(node, Vote):
        return (f"{name}: {node.rule} of {len(node.sources)}",)
    if isinstance(node, Coalesce):
        return (name, "first that answers")
    if isinstance(node, Confidence):
        return (f"how sure {node.source} was",)
    if isinstance(node, Sum):
        return (f"{name}: sum",)
    if isinstance(node, Quantify):
        return (render_quantifier(node, reads, plan),)
    if isinstance(node, Merge):
        return (name, "merge")
    return (name,)


def kind_of(node) -> str:
    for kind, types in KINDS.items():
        if isinstance(node, types):
            return kind
    return "join" if isinstance(node, Coalesce) else "step"


def safe(text: str) -> str:
    """Mermaid takes the label between its shape markers, so anything that closes one has to go.
    Labels are quoted, and quoted text may hold parentheses, which a guard needs to keep its
    meaning."""
    for character in '"[]{}|;\n':
        text = text.replace(character, " ")
    return " ".join(text.split())


def guard(condition, reads: dict[str, Read], plan: Plan) -> str:
    """A condition as its box says it. In rules a bare name tests that the node answered, which
    reads fine after `when`; alone in a box it needs the verb."""
    # A test over a list's items already reads as a condition.
    if isinstance(condition, Present) and not isinstance(plan.nodes.get(condition.node), Quantify):
        return f"{name_of(condition.node, reads, plan)} is present"
    return render_condition(condition, reads, plan)


def draw(plan: Plan, closed: bool = False) -> Drawing:
    """The plan as boxes and arrows, each unit's graph drawn inside its box, or, closed, each
    unit as one box, so a map of many units reads as a map."""
    drawing = Drawing(closed=closed)
    build(plan, "", None, drawing)
    return drawing


def build(
    plan: Plan,
    prefix: str,
    unit: str | None,
    drawing: Drawing,
    exit: tuple[str, str] | None = None,
) -> None:
    """One plan's boxes, arrows and outputs, with ids under `prefix` so a unit's graph and the
    graph around it never share an id. `exit` names the output that is a unit's answer and the
    id of the unit's box, which stands in for that output's circle."""
    reads = {name: node for name, node in plan.nodes.items() if isinstance(node, Read)}
    # A node may be named like an output or anything else, so ids are positions and the names
    # go in the labels.
    ids = {name: f"{prefix}n{index}" for index, name in enumerate(plan.nodes)}
    for name, node in plan.nodes.items():
        if isinstance(node, Graph):
            when = guard(node.when, reads, plan) if node.when is not None else None
            # A unit of one line, `always: hold`, has no nodes and is its answer; its box says so,
            # since there is nothing to draw inside it.
            answer = node.plan.outputs.get(node.output)
            said = f"always {answer.value}" if isinstance(answer, LiteralInput) else f"answers with its {node.output}"
            lines = (name, said)
            drawing.boxes.append(Box(ids[name], "unit", lines, name, node.type, unit, when))
            if not drawing.closed:
                build(node.plan, f"{ids[name]}_", ids[name], drawing, (node.output, ids[name]))
            if isinstance(node.input, NodeInput):
                drawing.edges.append(Edge(ids[node.input.node], ids[name], "data"))
            continue
        lines = label(name, node, reads, plan)
        drawing.boxes.append(Box(ids[name], kind_of(node), lines, name, node.type, unit))
    conditions = 0
    for name, node in plan.nodes.items():
        if isinstance(node, Coalesce):
            # Tried in order, so the order is the point: a judge last in line is asked only when
            # every rule ahead of it declined.
            for rank, source in enumerate(node.sources, start=1):
                drawing.edges.append(Edge(ids[source], ids[name], "rank", rank))
        else:
            for source in sorted(feeds(node)):
                drawing.edges.append(Edge(ids[source], ids[name], "data"))
        # A rule is often a bare value with a condition on it, so the guard has no data edge to
        # ride on. It is a box of its own, fed by each value it reads, with a dashed arrow to
        # what it gates: written on one arrow, a condition over two values looked as if it
        # belonged to one of them.
        if node.when is not None:
            box = f"{prefix}c{conditions}"
            conditions += 1
            said = (guard(node.when, reads, plan),)
            drawing.boxes.append(Box(box, "condition", said, None, None, unit))
            for source in sorted(tested(node.when)):
                drawing.edges.append(Edge(ids[source], box, "data"))
            drawing.edges.append(Edge(box, ids[name], "gate"))
    for index, (output, source) in enumerate(plan.outputs.items()):
        # A unit's answer leaves through the unit's box; every other output is a circle.
        if exit is not None and output == exit[0]:
            target = exit[1]
        else:
            target = f"{prefix}o{index}"
            drawing.boxes.append(Box(target, "output", (output,), output, None, unit))
        if isinstance(source, NodeInput):
            drawing.edges.append(Edge(ids[source.node], target, "answer"))


def feeds(node) -> set[str]:
    """The nodes whose values this one consumes, ignoring the ones its guard mentions."""
    if isinstance(node, (Call, Merge, Extract)):
        return {source.node for source in node.inputs.values() if isinstance(source, NodeInput)}
    if isinstance(node, Read) and node.source is not None:
        return {node.source}
    if isinstance(node, (Coalesce, Vote)):
        return set(node.sources)
    if isinstance(node, (Switch, Threshold, Confidence, Quantify)):
        return {node.source}
    if isinstance(node, Sum):
        return {term.node for term in node.terms}
    return set()


def mermaid(plan: Plan, closed: bool = False) -> str:
    """The drawing as a Mermaid flowchart: boxes for nodes, arrows for what feeds what, a box of
    its own for each condition, and each open unit's graph inside a frame of its own."""
    drawing = draw(plan, closed)
    lines = ["flowchart TD"]
    render(drawing, None, lines)
    for kind, style in STYLES.items():
        members = [box.id for box in drawing.boxes if box.kind == kind]
        if members:
            lines.append(f"    classDef {kind} {style}")
            lines.append(f"    class {','.join(members)} {kind}")
    return "\n".join(lines)


def render(drawing: Drawing, unit: str | None, lines: list[str]) -> None:
    """The boxes drawn inside `unit` (the top, for None), each open unit as a frame holding its
    own, then the conditions, the arrows leaving those boxes, and the outputs."""
    boxes = {box.id: box for box in drawing.boxes}
    inside = [box for box in drawing.boxes if box.unit == unit]
    for box in inside:
        if box.kind == "unit":
            if drawing.closed:
                # The double-edged box is the usual sign for a step that is a program of its
                # own, which a closed unit is.
                lines.append(f'    {box.id}[["{safe(box.name)}<br/>a graph of its own"]]')
                continue
            # The gate's arrow lands on the frame's edge, which is easy to miss, so the frame's
            # title says when the unit runs as well; on one line, since Mermaid gives a frame's
            # title no room for a second.
            title = f"{safe(box.name)}: a graph of its own"
            if box.when is not None:
                title += f", when {safe(box.when)}"
            lines.append(f'    subgraph {box.id}_unit["{title}"]')
            lines.append(f'    {box.id}(["{text(box)}"])')
            render(drawing, box.id, lines)
            lines.append("    end")
        elif box.kind == "condition":
            lines.append(f'    {box.id}["{text(box)}"]')
        elif box.kind == "output":
            lines.append(f'    {box.id}(("{text(box)}"))')
        else:
            open_shape, close_shape = SHAPES[box.type]
            lines.append(f'    {box.id}{open_shape}"{text(box)}"{close_shape}')
    for edge in drawing.edges:
        if boxes[edge.source].unit != unit:
            continue
        target = edge.target
        # What a unit's gate and input attach to is the unit's whole frame, since the condition
        # decides whether any of it runs; its own answer, from inside, lands on its box.
        framed = boxes[target].kind == "unit" and not drawing.closed
        if framed and boxes[edge.source].unit != target:
            target = f"{target}_unit"
        arrow = f'-->|"{edge.rank}"|' if edge.kind == "rank" else ARROWS[edge.kind]
        lines.append(f"    {edge.source} {arrow} {target}")


def text(box: Box) -> str:
    return "<br/>".join(safe(line) for line in box.lines)
