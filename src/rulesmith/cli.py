"""Validate, execute, evaluate, and evolve a saved decision graph."""

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
from collections import Counter
from contextlib import ExitStack, nullcontext, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, get_args
from urllib.parse import urlsplit

import dspy
from typesafe_sdk import RetryPolicy, TypeSafeClient, TypeSafeError

from rulesmith import clef
from rulesmith.ablate import effects
from rulesmith.calibrate import at_prevalence, calibrate, correct, drift, fit_cutoffs, measure
from rulesmith.chat_judge import ChatClient, ChatJudgeError
from rulesmith.diagram import mermaid
from rulesmith.extract import Extractor
from rulesmith.grade import Graded, Grader, model_name
from rulesmith.graph import Graph, Plan, Task
from rulesmith.label import Labeler, Unlabeled, draft_task
from rulesmith.maps import MapError, load, unit_task, units
from rulesmith.optimize import (
    GraphProposer,
    Labels,
    Preference,
    SearchConfig,
    described_labels,
    evaluate_plan,
    optimize_plan,
)
from rulesmith.rules import NAME, RuleError, keep_comments, render_rules
from rulesmith.runtime import (
    DecisionProgram,
    ExecutionConfig,
    Judge,
    Models,
    Prices,
    Remembered,
    digest,
)
from rulesmith.serve import make_server
from rulesmith.teach import RulingClient, Teaching
from rulesmith.tuning import BASE, FILES, taught, tune

# The games load only for a task that plays one (see game), so a core install without their
# packages still validates and runs every other graph.
if TYPE_CHECKING:
    from rulesmith.chess import ChessTask
    from rulesmith.doom import DoomTask


@dataclass(frozen=True)
class Backend:
    model: str
    base_url_env: str
    api_key_env: str
    base_url: str | None = None
    api_key: str | None = None
    client: type[TypeSafeClient] | type[ChatClient] = TypeSafeClient


BACKENDS = {
    "typesafe": Backend("jev-latest", "TYPESAFE_BASE_URL", "TYPESAFE_API_KEY"),
    # The SDK requires a nonempty key even when the local server disables auth.
    "ruling": Backend(
        "default", "RULING_BASE_URL", "RULING_API_KEY", "http://127.0.0.1:8010", "local-no-auth"
    ),
    # Cloudflare's Clef-flash, served on this machine by `rulesmith clef-serve` or by the MLX
    # port's own clef_mlx.py, which answer the same /v1/systemone requests, one at a time.
    "clef": Backend(
        "clef-flash", "CLEF_BASE_URL", "CLEF_API_KEY", "http://127.0.0.1:8000", "local-no-auth"
    ),
    # Any OpenAI-compatible chat API whose models return log-probabilities, read as a judge.
    # OpenAI's newer GPT-5 and GPT-6 models return none, so the default is gpt-4.1.
    "openai": Backend(
        "gpt-4.1",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "https://api.openai.com/v1",
        client=ChatClient,
    ),
}


def resolve_backend(name: str, model: str | None = None, base_url: str | None = None):
    backend = BACKENDS[name]
    endpoint = (base_url or os.environ.get(backend.base_url_env, "")).strip() or backend.base_url
    key = os.environ.get(backend.api_key_env, "").strip() or backend.api_key
    return model or backend.model, {"base_url": endpoint, "api_key": key}


def connect(name: str, connection: dict, timeout: float, retry: RetryPolicy):
    """A client for the backend `name`, in the protocol that backend speaks."""
    return BACKENDS[name].client(**connection, timeout=timeout, retry=retry)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    helps = {
        "validate": "Check a graph against a task without calling any model",
        "evaluate": "Score a graph on one split of a task",
        "optimize": "Search for the graph with the best validation score",
        "record": "Play a graph in Doom and save the video",
    }
    for name, summary in helps.items():
        command = commands.add_parser(name, help=summary)
        command.add_argument("task", type=Path)
        if name == "optimize":
            command.add_argument(
                "plan", type=Path, nargs="?", help="Seed graph; defaults to the majority label"
            )
            command.add_argument(
                "--teach-judge",
                action="store_true",
                help="Report each train example's label to the Ruling judge as the outcome of the "
                "label questions it answers, so `ruling retrain` can tune it on the task; needs "
                "--backend ruling and a server started with RULING_LEDGER. Validation and test "
                "labels are never sent",
            )
        else:
            command.add_argument("plan", type=Path)
        command.add_argument(
            "--max-nodes",
            type=int,
            help=f"Most nodes a graph may have; {SearchConfig().max_nodes} unless --prefer large",
        )
        command.add_argument("--max-depth", type=int, default=SearchConfig().max_depth)
        if name == "validate":
            continue
        provider_arguments(command)
        command.add_argument("--output", type=Path, required=True)
        if name in ("evaluate", "optimize"):
            command.add_argument(
                "--parallel-examples",
                type=positive,
                default=SearchConfig().parallel_examples,
                help="Examples or episodes scored at once; deathmatch episodes can run together",
            )
        if name in ("evaluate", "optimize", "record"):
            command.add_argument(
                "--opponent",
                type=Path,
                help="Graph a deathmatch task's episodes are played against",
            )
        if name in ("evaluate", "optimize"):
            command.add_argument(
                "--engine",
                help="A UCI engine, such as stockfish, to play a chess task against instead of "
                "a graph",
            )
            command.add_argument(
                "--engine-skill", type=int, help="The engine's Skill Level (default 0, weakest)"
            )
            command.add_argument(
                "--engine-seconds", type=float, help="The engine's time per move (default 0.05)"
            )
            command.add_argument(
                "--grader-model",
                help="DSPy model identifier that grades answers instead of the task's labels",
            )
            command.add_argument("--grader-base-url")
        if name == "record":
            command.add_argument(
                "--seed", type=int, nargs="+", required=True, help="Episode seeds, played in order"
            )
        elif name == "evaluate":
            command.add_argument(
                "--split", choices=("train", "validation", "calibration", "test"), default="test"
            )
            command.add_argument(
                "--drift",
                type=float,
                default=0.01,
                help="Flag a calibrated rule that decides at a rate this unlikely for it",
            )
            command.add_argument(
                "--visible", action="store_true", help="Show the game window for a Doom task"
            )
        else:
            command.add_argument(
                "--reflection-model",
                required=True,
                help="DSPy model identifier for the generative proposer",
            )
            command.add_argument("--reflection-base-url")
            command.add_argument("--reflection-max-tokens", type=int, default=4096)
            command.add_argument(
                "--reflection-options",
                type=json.loads,
                help="Extra request JSON for the proposer, such as a cap on a thinking model's "
                'reasoning: \'{"reasoning": {"max_tokens": 2048}}\'',
            )
            command.add_argument(
                "--reflection-timeout", type=float, help="Seconds before a proposal request fails"
            )
            command.add_argument(
                "--max-metric-calls", type=int, default=SearchConfig().max_metric_calls
            )
            command.add_argument(
                "--prefer",
                choices=get_args(Preference),
                default=SearchConfig().prefer,
                help="small keeps graphs small; large asks for layers of judgments and raises "
                "the default --max-nodes",
            )
            command.add_argument(
                "--reflection-batch", type=int, default=SearchConfig().reflection_batch
            )
            command.add_argument(
                "--repair-attempts", type=int, default=SearchConfig().repair_attempts
            )
            command.add_argument("--merges", type=int, default=SearchConfig().merges)
            command.add_argument(
                "--mine",
                action="store_true",
                help="Mine rules a field alone settles, and start there if they score better",
            )
            command.add_argument(
                "--evidence",
                action="store_true",
                help="Show the proposer the single-field rules the training examples support "
                "best; counted on a sampled training set, they can over-fit its rate",
            )
            command.add_argument(
                "--significance",
                type=float,
                help="Keep the seed unless the winner's gain passes a sign test at this level",
            )
            command.add_argument(
                "--resume",
                action="store_true",
                help="Continue the search checkpointed under --output/search, to "
                "--max-metric-calls; the same task and starting graph are required",
            )
            command.add_argument(
                "--unit",
                help="Search this unit of the map given as the plan, against the task given, "
                "with the rest of the map frozen; the winner must still fit the map",
            )
            command.add_argument(
                "--map-task",
                type=Path,
                help="With --unit: a task for the whole map, scored on its validation examples "
                "before and after the unit changes; a winner that makes the map worse is refused",
            )
            cost_arguments(command)
            command.add_argument(
                "--keep-judge",
                type=float,
                help="Let rules answer without the judge only where they are right at least this "
                "often on the labeled examples, such as 0.99",
            )
            command.add_argument("--seed", type=int, default=SearchConfig().seed)
    draft = commands.add_parser("draft-labels", help="Label raw inputs for review as a task")
    draft.add_argument("inputs", type=Path, help="JSON with description, labels, and inputs")
    draft.add_argument("--output", type=Path, required=True)
    draft.add_argument(
        "--labeler-model", required=True, help="DSPy model identifier for the labeler"
    )
    draft.add_argument("--labeler-base-url")
    draft.add_argument("--validation", type=int, required=True, help="Validation example count")
    draft.add_argument("--test", type=int, required=True, help="Test example count")
    draft.add_argument("--seed", type=int, default=0, help="Split shuffle seed")
    arena = commands.add_parser("arena", help="Rank Doom graphs by Elo from deathmatch duels")
    arena.add_argument("task", type=Path, help="Doom task whose episodes are the matches")
    arena.add_argument("plans", type=Path, nargs="+", help="Entrants, named by file stem")
    arena.add_argument("--split", choices=("train", "validation", "test"), default="test")
    arena.add_argument("--max-nodes", type=int, default=SearchConfig().max_nodes)
    arena.add_argument("--max-depth", type=int, default=SearchConfig().max_depth)
    arena.add_argument("--output", type=Path, required=True)
    provider_arguments(arena)
    fit = commands.add_parser(
        "calibrate",
        help="Fit how sure each judgment reads and where rules draw lines on it, from the labels",
    )
    fit.add_argument("task", type=Path)
    fit.add_argument("plan", type=Path)
    fit.add_argument("--output", type=Path, required=True, help="A .rules or .json plan")
    provider_arguments(fit)
    cost_arguments(fit)
    unit = commands.add_parser(
        "unit-task",
        help="Write the task of one unit of a map: the cases the map hands it, as handed",
    )
    unit.add_argument("task", type=Path, help="A labeled task for the whole map")
    unit.add_argument("plan", type=Path, help="The map")
    unit.add_argument("--unit", required=True, help="The unit's node in the map")
    unit.add_argument("--output", type=Path, required=True)
    provider_arguments(unit)
    draw = commands.add_parser("draw", help="Write a graph as a Mermaid flowchart")
    draw.add_argument("plan", type=Path)
    draw.add_argument(
        "--closed",
        action="store_true",
        help="Draw each unit as one box rather than opening it, so a map reads as a map",
    )
    for name, summary in (
        ("tune", "Fine-tune the judge on a task's examples with Ruling, test split held out"),
        (
            "distill",
            "Fine-tune the judge on a larger model's answers to a task's inputs, labeled or not",
        ),
    ):
        tuning = commands.add_parser(name, help=summary)
        tuning.add_argument("task", type=Path)
        tuning.add_argument(
            "--output", type=Path, required=True, help="Directory for the records and the adapter"
        )
        tuning.add_argument(
            "--ruling-dir",
            type=Path,
            default=Path(os.environ.get("RULING_DIR", "../ruling")),
            help="A Ruling checkout (default: $RULING_DIR, else ../ruling)",
        )
        tuning.add_argument("--base", default=BASE, help="The model the adapter is trained on")
        if name == "distill":
            tuning.add_argument(
                "--teacher",
                type=judge_spec,
                required=True,
                metavar="BACKEND[:MODEL][@URL]",
                help="The judge whose answers the student learns, such as "
                "clef or ruling:qwen-35b@http://127.0.0.1:8012",
            )
            tuning.add_argument("--timeout", type=float, default=60)
        tuning.epilog = "Anything after -- goes to `ruling train` and overrides the recipe."
    ablation = commands.add_parser(
        "ablate", help="Measure what each reading, judgment and name in a graph is worth"
    )
    ablation.add_argument("task", type=Path)
    ablation.add_argument("plan", type=Path)
    ablation.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    ablation.add_argument("--output", type=Path)
    provider_arguments(ablation)
    watching = commands.add_parser(
        "drift",
        help="Check a decision's traces against what its rules were calibrated on; exits 1 when a "
        "rule has drifted",
    )
    watching.add_argument("decision", type=Path, help="A decision folder or graph file")
    watching.add_argument(
        "traces", type=Path, nargs="+", help="Trace files, or directories of them"
    )
    watching.add_argument(
        "--significance",
        type=float,
        default=0.01,
        help="Flag a rule whose share of decisions is this unlikely under its calibrated share",
    )
    serving = commands.add_parser(
        "serve", help="Answer decisions over HTTP at POST /decisions/<name>, as run prints them"
    )
    serving.add_argument(
        "decisions",
        type=Path,
        nargs="+",
        help="Decision folders, each with a graph.rules or map.json, or graph files",
    )
    serving.add_argument("--host", default="127.0.0.1", help="The address to listen on")
    serving.add_argument("--port", type=int, default=8020, help="0 picks a free port")
    serving.add_argument("--trace", type=Path, help="Append every decision to this JSON-lines file")
    provider_arguments(serving)
    clef_serving = commands.add_parser(
        "clef-serve",
        help="Serve Cloudflare's Clef at POST /v1/systemone for --backend clef, with PyTorch on "
        "an NVIDIA GPU, Apple's, or a CPU (needs the clef extra)",
    )
    clef_serving.add_argument(
        "--weights",
        default="Cloudflare/clef-flash",
        help="A Clef release on Hugging Face, such as Cloudflare/clef for the 27B, or a "
        "downloaded copy of one (default: Cloudflare/clef-flash)",
    )
    clef_serving.add_argument("--device", help="cuda, mps or cpu (default: the best there is)")
    clef_serving.add_argument("--host", default="127.0.0.1", help="The address to listen on")
    clef_serving.add_argument(
        "--port",
        type=int,
        default=urlsplit(BACKENDS["clef"].base_url).port,
        help="0 picks a free port (default: the one --backend clef asks)",
    )
    run = commands.add_parser("run", help="Run a graph on one input and print what it decided")
    run.add_argument("plan", type=Path)
    run.add_argument("state", type=Path, help="JSON file containing the input state")
    run.add_argument("--trace", type=Path, help="Append the decision to this JSON-lines file")
    provider_arguments(run)
    return root


def provider_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--backend", choices=tuple(BACKENDS), default="typesafe")
    command.add_argument("--model", help="Model ID; defaults to the selected backend alias")
    command.add_argument("--base-url", help="Override the selected backend endpoint")
    command.add_argument("--timeout", type=float, default=60)
    command.add_argument(
        "--retries", type=int, default=0, help="Retries for failed or timed-out judge requests"
    )
    command.add_argument("--max-workers", type=positive, default=ExecutionConfig().max_workers)
    command.add_argument(
        "--judge",
        type=named_judge,
        action="append",
        default=[],
        metavar="NAME=BACKEND[:MODEL][@URL]",
        help="A judge the graph asks with 'using NAME', such as "
        "large=ruling:qwen-35b@http://127.0.0.1:8012; repeat for each",
    )
    command.add_argument(
        "--extractor-model",
        help="The generative model that fills a graph's 'extract' fields, as DSPy names it, "
        "such as openai/qwen3.8-27b",
    )
    command.add_argument(
        "--extractor-base-url", help="An OpenAI-compatible endpoint serving --extractor-model"
    )


def judge_spec(text: str) -> tuple[str, str | None, str | None]:
    """'ruling:qwen-35b@http://127.0.0.1:8012' as its backend, model and endpoint; the model
    and endpoint default to the backend's own, as they do for the graph's judge."""
    rest, at, url = text.partition("@")
    backend, _, model = rest.partition(":")
    if backend not in BACKENDS or (at and not url):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not BACKEND[:MODEL][@URL], with BACKEND one of {sorted(BACKENDS)}"
        )
    return backend, model or None, url or None


def named_judge(text: str) -> tuple[str, str, str | None, str | None]:
    """'large=ruling:qwen-35b@http://127.0.0.1:8012' as its name, and the judge it names."""
    name, equals, rest = text.partition("=")
    if not equals or not NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=BACKEND[:MODEL][@URL]")
    return name, *judge_spec(rest)


def cost_arguments(command: argparse.ArgumentParser) -> None:
    """What each model call costs, for search to prefer the cheaper of two graphs and for
    calibrate to place an escalation line where escalating pays."""
    command.add_argument(
        "--call-cost",
        type=float,
        default=Prices().call,
        help="Score off an example per judge question, to prefer graphs that ask less",
    )
    command.add_argument(
        "--judge-cost",
        type=named_cost,
        action="append",
        default=[],
        metavar="NAME=COST",
        help="Score off an example per question to a named judge; a named judge without one "
        "costs --call-cost",
    )
    command.add_argument(
        "--extract-cost",
        type=float,
        default=Prices().extract,
        help="Score off an example per extraction it runs, to keep only those that pay",
    )


def prices(cli: argparse.ArgumentParser, args: argparse.Namespace) -> Prices:
    named = {name for name, *_ in args.judge}
    for name, _ in args.judge_cost:
        if name not in named:
            cli.error(f"--judge-cost {name} prices a judge no --judge names")
    return Prices(call=args.call_cost, judges=dict(args.judge_cost), extract=args.extract_cost)


def named_cost(text: str) -> tuple[str, float]:
    name, equals, cost = text.partition("=")
    if not equals or not NAME.fullmatch(name):
        raise argparse.ArgumentTypeError(f"{text!r} is not NAME=COST, such as large=0.1")
    try:
        return name, float(cost)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"the cost in {text!r} is not a number") from error


def graph_models(cli: argparse.ArgumentParser, args, stack: ExitStack, *plans) -> Models:
    """The models the graphs ask beside their own judge: a client for every judge named on the
    command line, once each name the graphs use is there, and the extractor, if one is given."""
    given = {}
    for name, backend, model, url in args.judge:
        if name in given:
            cli.error(f"--judge {name} is given twice")
        given[name] = backend, model, url
    for name in sorted(set().union(*(plan.judges() for plan in plans if plan)) - given.keys()):
        cli.error(
            f"the graph asks judge {name}; "
            f"say which model with --judge {name}=BACKEND[:MODEL][@URL]"
        )
    judges = {}
    for name, (backend, model, url) in given.items():
        resolved, connection = resolve_backend(backend, model, url)
        retry = RetryPolicy(max_retries=args.retries)
        client = stack.enter_context(connect(backend, connection, args.timeout, retry))
        judges[name] = Judge(client, resolved)
    if args.extractor_base_url and not args.extractor_model:
        cli.error("--extractor-base-url needs an --extractor-model to send extractions to")
    if not args.extractor_model:
        if any(plan.extracts() for plan in plans if plan):
            cli.error(
                "the graph extracts fields; name the model that reads them with --extractor-model"
            )
        return Models(judges)
    options = {"api_base": args.extractor_base_url} if args.extractor_base_url else {}
    lm = dspy.LM(args.extractor_model, temperature=0, cache=False, num_retries=0, **options)
    return Models(judges, Extractor(lm))


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, not {value}")
    return value


def judge(args: argparse.Namespace, needed: bool, task: Task | None = None):
    """The judge model, and a client for it when some graph calls it: a teaching one when the
    search is to report the task's train labels to a Ruling server."""
    model, connection = resolve_backend(args.backend, args.model, args.base_url)
    if not needed:
        return model, nullcontext()
    retry = RetryPolicy(max_retries=args.retries)
    if getattr(args, "teach_judge", False):
        if args.backend != "ruling":
            raise SystemExit(
                "--teach-judge reports outcomes to a Ruling server; use --backend ruling"
            )
        if task is None:
            raise SystemExit(
                "--teach-judge reports a task's labels, and this task has none to report"
            )
        return model, Teaching(RulingClient(**connection, timeout=args.timeout, retry=retry), task)
    return model, connect(args.backend, connection, args.timeout, retry)


def ablation(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """What each input is worth, over the states the task itself supplies."""
    task = load_task(cli, args.task)
    if args.output and args.output.exists():
        cli.error(f"output already exists: {args.output}")
    # A plan is structurally valid once it loads; size limits bound what search may write, and
    # have nothing to say about measuring a graph that already exists.
    plan = read_plan(cli, args.plan, task)
    examples = getattr(task, args.split)
    if not all(hasattr(example, "state") for example in examples):
        cli.error(
            "ablate reads states from a task's examples; a Doom or chess task keeps seeds, "
            "so measure those graphs with rulesmith.ablate.effects over recorded states"
        )
    states = [example.state for example in examples]
    calls = plan.asks_own_judge()
    model, judge_client = judge(args, calls)
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        models = graph_models(cli, args, stack, plan)
        execution = ExecutionConfig(max_workers=args.max_workers)
        found = effects(plan, states, client if calls else None, model, execution, models)
    report = {
        "states": len(states),
        "effects": [
            {
                "node": effect.node,
                "kind": effect.kind,
                "changed": effect.changed,
                "share": round(effect.share, 4),
                "reason": effect.reason,
            }
            for effect in found
        ],
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(dump(report))
    print(dump(report), end="")


def arena(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    task = load_task(cli, args.task)
    if not plays(task, "doom") or not task.deathmatch:
        cli.error("arena requires a Doom task with deathmatch set")
    from rulesmith.arena import Entrant, tournament

    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    plans = {}
    for path in args.plans:
        if path.stem in plans:
            cli.error(f"entrants share the name {path.stem}; rename a plan file")
        plans[path.stem] = read_plan(cli, path, task)
        plans[path.stem].check(task.labels, args.max_nodes, args.max_depth)
    if len(plans) < 2:
        cli.error("the arena needs at least two entrants")
    model, judge_client = judge(args, any(plan.asks_own_judge() for plan in plans.values()))
    execution = ExecutionConfig(max_workers=args.max_workers)
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        models = graph_models(cli, args, stack, *plans.values())
        entrants = [
            Entrant(name, DecisionProgram(plan, client, model, execution, models=models))
            for name, plan in plans.items()
        ]
        report = tournament(task, entrants, getattr(task, args.split))
    report["split"] = args.split
    report["backend"] = args.backend
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dump(report))
    print(dump({"ratings": report["ratings"], "output": str(args.output)}), end="")


def decision(cli: argparse.ArgumentParser, path: Path) -> tuple[str, Plan]:
    """A decision folder's name and graph, read against its task's labels when it has a task, or
    a graph file's stem and graph: the graph as `run` and `serve` would run it."""
    if not path.is_dir():
        return path.stem, read_plan(cli, path)
    file = next((path / f for f in ("graph.rules", "map.json") if (path / f).exists()), None)
    if file is None:
        cli.error(f"{path} holds no graph.rules or map.json")
    task = load_task(cli, path / "task.json") if (path / "task.json").exists() else None
    return path.name, read_plan(cli, file, task)


def watch_drift(cli: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """How the decisions a graph made in production compare with what its rules were counted
    on. Only decisions by this exact graph count: a rule's name is its place in the list, so an
    earlier version's rule_2 is another rule."""
    _, plan = decision(cli, args.decision)
    made = digest(plan.model_dump_json())
    files = [
        f
        for path in args.traces
        for f in (sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    ]
    records = [
        record
        for file in files
        for line in file.read_text().splitlines()
        if line.strip() and (record := json.loads(line))["graph_sha256"] == made
    ]
    if not records:
        cli.error(f"no traced decision was made by this graph ({made[:12]})")
    called = Counter(paid for record in records for paid in record.get("called", {}).values())
    report = {
        "graph_sha256": made,
        "decisions": len(records),
        "undecided": sum(record["undecided"] is not None for record in records),
        "calls": dict(sorted(called.items())),
        "drift": drift(plan, records, args.significance),
    }
    print(dump(report), end="")
    return 1 if report["drift"] else 0


def serve_decisions(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Serve each decision under its folder's name, or its file's, until stopped. A folder with a
    task reads its rules against the task's labels, as calibrate and evaluate do."""
    graphs = {}
    for path in args.decisions:
        name, plan = decision(cli, path)
        if name in graphs:
            cli.error(f"two decisions are named {name}; serve them from folders of their own")
        graphs[name] = plan
    model, judge_client = judge(args, any(plan.asks_own_judge() for plan in graphs.values()))
    execution = ExecutionConfig(max_workers=args.max_workers)
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        models = graph_models(cli, args, stack, *graphs.values())
        programs = {
            name: DecisionProgram(plan, client, model, execution, args.trace, models=models)
            for name, plan in graphs.items()
        }
        server = stack.enter_context(make_server(programs, args.host, args.port))
        # One line, flushed, so whatever started the server knows when it may call it.
        url = f"http://{args.host}:{server.server_port}"
        print(json.dumps({"url": url, "decisions": sorted(programs)}), flush=True)
        server.serve_forever()


def serve_clef(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Load Clef, then answer its requests until stopped."""
    try:
        # Loading prints whatever the libraries print; stdout's first line must be the URL.
        with redirect_stdout(sys.stderr):
            answer, device = clef.load(args.weights, args.device)
    except ImportError as error:
        cli.error(f"clef-serve needs the clef extra: pip install 'rulesmith[clef]' ({error})")
    with clef.make_server(answer, args.host, args.port) as server:
        # One line, flushed, once the model is loaded, so a caller knows when it may ask.
        url = f"http://{args.host}:{server.server_port}"
        print(json.dumps({"url": url, "weights": args.weights, "device": device}), flush=True)
        server.serve_forever()


def dump(value: dict) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def draft_labels(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    unlabeled = Unlabeled.model_validate_json(args.inputs.read_text())
    lm_options = {"api_base": args.labeler_base_url} if args.labeler_base_url else {}
    labeler = Labeler(
        dspy.LM(args.labeler_model, cache=False, num_retries=0, **lm_options),
        unlabeled.description,
        unlabeled.labels,
    )
    task = draft_task(
        unlabeled.description,
        unlabeled.labels,
        unlabeled.inputs,
        labeler,
        args.validation,
        args.test,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(task.model_dump_json(indent=2) + "\n")
    counts = Counter(e.label for e in task.examples())
    print(dump({"labels": dict(counts), "output": str(args.output)}), end="")


def save_video(frames: list, fps: int, path: Path) -> None:
    if path.suffix == ".mp4":
        import imageio_ffmpeg

        height, width, _ = frames[0].shape
        writer = imageio_ffmpeg.write_frames(str(path), (width, height), fps=fps)
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()
        return
    from PIL import Image

    first, *rest = (Image.fromarray(frame) for frame in frames)
    # GIF delays are whole centiseconds, so round to 10 ms rather than truncate.
    delay = round(100 / fps) * 10
    first.save(path, save_all=True, append_images=rest, duration=delay, loop=0)


def scorer(args: argparse.Namespace, task):
    """What scores a labeled task's answers: its labels, or a grader when one is named. A game
    task's scorer is its games, chosen where the games are set up."""
    if not isinstance(task, Task):
        return None
    if not args.grader_model:
        return Labels(task)
    lm_options = {"api_base": args.grader_base_url} if args.grader_base_url else {}
    lm = dspy.LM(args.grader_model, cache=False, num_retries=0, **lm_options)
    return Graded(Grader(lm, task.description))


def scored_by(args: argparse.Namespace, task) -> str:
    """How a report's scores were made, since a grader's score is not accuracy."""
    if getattr(args, "grader_model", None):
        return f"grader {args.grader_model}"
    return "labels" if isinstance(task, Task) else "games"


def opponent_of(args: argparse.Namespace) -> str | None:
    """Who a played task's scores were earned against, since a score means nothing without it."""
    if getattr(args, "engine", None):
        return f"engine {args.engine} at skill {args.engine_skill or 0}"
    return str(args.opponent) if getattr(args, "opponent", None) else None


def write_unit_task(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """The task a unit is calibrated and searched on, made by running the whole map, on its
    judge, over the map's own labeled examples."""
    task = load_task(cli, args.task)
    if not isinstance(task, Task):
        cli.error("unit-task reads a labeled task for the whole map")
    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    plan = read_plan(cli, args.plan, task)
    if not isinstance(plan.nodes.get(args.unit), Graph):
        cli.error(f"{args.plan} has no unit named {args.unit!r}; a unit is a graph node")
    model, judge_client = judge(args, plan.asks_own_judge())
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        models = graph_models(cli, args, stack, plan)
        program = DecisionProgram(
            plan, client, model, ExecutionConfig(max_workers=args.max_workers), models=models
        )
        try:
            found = unit_task(program, task, args.unit)
        except MapError as error:
            cli.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(found.model_dump_json(indent=2, exclude_none=True) + "\n")
    print(
        dump(
            {
                split: len(getattr(found, split))
                for split in ("train", "validation", "calibration", "test")
            }
        ),
        end="",
    )


def calibration(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """A plan whose choice questions read as sure as they are right, whose rules draw their lines
    on judgments where the labeled examples put them, and whose rules are counted."""
    task = load_task(cli, args.task)
    if not isinstance(task, Task):
        cli.error("calibrate fits judgments against a labeled task's examples")
    if units(args.plan):
        # Each unit's rules are counted on the cases that reach it, which a map's task does not
        # say; a unit's own task does.
        cli.error(
            "calibrate one unit of a map at a time: write its task with rulesmith unit-task, "
            "then calibrate the unit's own file on it"
        )
    if args.output.exists():
        cli.error(f"output already exists: {args.output}")
    plan = read_plan(cli, args.plan, task)
    priced = prices(cli, args)
    model, judge_client = judge(args, plan.asks_own_judge())
    held_out = [e for e in task.validation if e.label is not None]
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        # Every pass asks the same questions; the later ones reread the first's answers.
        remembered = Remembered(client)
        models = graph_models(cli, args, stack, plan).remembered()
        fitted, temperatures = calibrate(plan, task, remembered, model, models)
        before = correct(fitted, held_out, remembered, model, models)
        fitted, cutoffs = fit_cutoffs(fitted, task, remembered, model, models, priced)
        after = correct(fitted, held_out, remembered, model, models)
        fitted = measure(fitted, task, remembered, model, models)
    written = (
        render_rules(fitted, task.criteria())
        if args.output.suffix == ".rules"
        else fitted.model_dump_json(indent=2)
    )
    if args.output.suffix == ".rules" and args.plan.suffix == ".rules":
        written = keep_comments(args.plan.read_text(), written)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(written + "\n")
    counted = {
        name: node.measured.model_dump()
        for name, node in fitted.nodes.items()
        if getattr(node, "measured", None) is not None
    }
    # Fitted on examples search never saw, or on the ones it chose the graph by.
    report = {
        "fitted_on": task.fitted_on(),
        "temperatures": temperatures,
        "cutoffs": cutoffs,
        "rules": counted,
    }
    if priced != Prices():
        # A line placed against prices holds for those prices; say which they were.
        report["prices"] = priced.model_dump()
    if held_out:
        # Validation is scored, never fitted on, so this is what the fit is worth on fresh examples.
        report["validation"] = {
            "before": sum(before) / len(held_out),
            "after": sum(after) / len(held_out),
        }
    print(dump(report | {"output": str(args.output)}), end="")


def read_plan(
    cli: argparse.ArgumentParser, path: Path, task: "Task | DoomTask | ChessTask | None" = None
) -> Plan:
    """A graph from its JSON, or from the rules it is written as, which may ask over the labels
    of the task the command names. A file that does not hold one is a mistake in the command,
    reported as such rather than as a crash."""
    try:
        return load(path, described_labels(task) if task else None)
    except (ValueError, OSError) as error:
        cli.error(f"{path}: {error}")


def game(cli: argparse.ArgumentParser, name: str) -> ModuleType:
    """A game's module. Its packages come with the extra of the same name, which a core install
    leaves out, so a missing one is a matter of installing rather than a crash."""
    try:
        return importlib.import_module(f"rulesmith.{name}")
    except ModuleNotFoundError as error:
        cli.error(f"a {name} task needs {error.name}: pip install 'rulesmith[{name}]'")


def plays(task, name: str) -> bool:
    """Whether a task is one of a game's, asked without importing a game that may be missing."""
    return type(task).__module__ == f"rulesmith.{name}"


def load_task(cli: argparse.ArgumentParser, path: Path) -> "Task | DoomTask | ChessTask":
    try:
        data = json.loads(path.read_text())
        if data.get("game") == "chess":
            kind = game(cli, "chess").ChessTask
        elif "scenario" in data:
            kind = game(cli, "doom").DoomTask
        else:
            kind = Task
        return kind.model_validate(data)
    except ValueError as error:
        cli.error(f"{path}: {error}")


def read_state(cli: argparse.ArgumentParser, path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        cli.error(f"{path}: {error}")


def main() -> None:
    # The default SIGTERM action skips cleanup, which leaves ViZDoom engines running (they
    # ignore SIGTERM themselves); exiting normally closes every environment on the way out.
    signal.signal(signal.SIGTERM, lambda number, _: sys.exit(128 + number))
    cli = parser()
    argv = sys.argv[1:]
    # Arguments after -- belong to `ruling train`. Older Python 3.12 releases match an empty
    # positional before the options and then refuse them, so they are split off here instead.
    split = argv.index("--") if "--" in argv else len(argv)
    args = cli.parse_args(argv[:split])
    args.train_args = argv[split + 1 :]
    if args.train_args and args.command not in ("tune", "distill"):
        cli.error("only tune and distill pass arguments after -- on to ruling train")
    try:
        command(cli, args)
    except (TypeSafeError, ChatJudgeError) as error:
        # A missing key or an unreachable judge is something to set or start, said in a line.
        cli.error(str(error))


def command(cli: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.command == "draft-labels":
        draft_labels(cli, args)
        return
    if args.command == "draw":
        print(mermaid(read_plan(cli, args.plan), closed=args.closed))
        return
    if args.command in ("tune", "distill"):
        task = load_task(cli, args.task)
        if not isinstance(task, Task):
            cli.error(f"{args.command} needs a task's inputs; a game has none to learn from")
        extra = list(args.train_args)
        try:
            if args.command == "distill":
                # Refused before the teacher is paid, and the teacher asked before anything is
                # written, so a teacher that cannot answer leaves no half-made tuning set.
                taken = [args.output / f"{name}.jsonl" for name in FILES]
                if any(path.exists() for path in taken):
                    cli.error(f"already exists: {args.output}; distill into a new directory")
                backend, model, url = args.teacher
                teacher, connection = resolve_backend(backend, model, url)
                with connect(backend, connection, args.timeout, RetryPolicy()) as client:
                    found = taught(task, client, teacher)
                report = tune(task, args.output, args.ruling_dir, args.base, extra, found)
                report["teacher"] = teacher
            else:
                report = tune(task, args.output, args.ruling_dir, args.base, extra)
        except (FileExistsError, FileNotFoundError) as error:
            cli.error(str(error))
        except subprocess.CalledProcessError as error:
            cli.error(f"ruling train failed with exit code {error.returncode}")
        print(dump(report), end="")
        return
    if args.command == "arena":
        arena(cli, args)
        return
    if args.command == "ablate":
        ablation(cli, args)
        return
    if args.command == "unit-task":
        return write_unit_task(cli, args)
    if args.command == "calibrate":
        calibration(cli, args)
        return
    if args.command == "serve":
        serve_decisions(cli, args)
        return
    if args.command == "clef-serve":
        serve_clef(cli, args)
        return
    if args.command == "drift":
        sys.exit(watch_drift(cli, args))
    task = None if args.command == "run" else load_task(cli, args.task)
    plan = read_plan(cli, args.plan, task) if args.plan else task.baseline_plan()
    whole, unit = None, getattr(args, "unit", None)
    map_task = getattr(args, "map_task", None)
    if map_task is not None and unit is None:
        cli.error("--map-task scores the map around a --unit; name the unit")
    if map_task is not None:
        map_task = load_task(cli, map_task)
        if not isinstance(map_task, Task):
            cli.error("--map-task must be a labeled task for the whole map")
    if unit is not None:
        # The map is what the winner must fit; the unit's own graph is what search starts from.
        whole = plan
        if not isinstance(whole.nodes.get(unit), Graph):
            cli.error(f"{args.plan} has no unit named {unit!r}; a unit is a graph node")
        plan = whole.nodes[unit].plan
    if args.command != "run":
        if args.max_nodes is None:
            # The cap is the search's to settle, since --prefer large raises it.
            preferred = getattr(args, "prefer", SearchConfig().prefer)
            args.max_nodes = SearchConfig(prefer=preferred).max_nodes
        plan.check(task.labels, args.max_nodes, args.max_depth)
    if args.command == "validate":
        checked = {"nodes": len(plan.nodes), "labels": task.labels}
        # Valid, since the inputs a task sees may always carry what the rules read, but worth
        # knowing: an input that does not is an error at run time, not an answer.
        if unanswered := plan.unanswered():
            checked["unanswered"] = unanswered
        # A map's units, so what was validated across files is on record.
        if named := units(args.plan):
            checked["units"] = named
        print(dump(checked), end="")
        return
    if getattr(args, "visible", False) and not plays(task, "doom"):
        cli.error("--visible requires a Doom task")
    deathmatch = args.command != "run" and plays(task, "doom") and task.deathmatch
    played = deathmatch or (args.command != "run" and plays(task, "chess"))
    engine = getattr(args, "engine", None)
    tuned = [flag for flag in ("engine_skill", "engine_seconds") if getattr(args, flag, None)]
    if tuned and not engine:
        cli.error("--engine-skill and --engine-seconds set the strength of an --engine")
    if engine and not plays(task, "chess"):
        cli.error("--engine plays a chess task; other games are played against a graph")
    if engine and args.opponent:
        cli.error("a chess task is played against --opponent or --engine, not both")
    against = bool(args.opponent or engine) if hasattr(args, "opponent") else False
    if args.command in ("evaluate", "optimize", "record") and played != against:
        cli.error("--opponent is required for, and only for, tasks played against another graph")
    opponent = None
    if played and not engine:
        opponent = read_plan(cli, args.opponent, task)
        opponent.check(task.labels, args.max_nodes, args.max_depth)
    if getattr(args, "parallel_examples", 1) > 1 and plays(task, "doom") and not deathmatch:
        cli.error("--parallel-examples needs a deathmatch task; other Doom episodes share one game")
    if args.command == "record" and not plays(task, "doom"):
        cli.error("record requires a Doom task")
    if args.command == "record" and args.output.suffix not in (".gif", ".mp4"):
        cli.error("record writes .gif or .mp4")
    grading = getattr(args, "grader_model", None)
    if getattr(args, "grader_base_url", None) and not grading:
        cli.error("--grader-base-url needs a --grader-model to send grades to")
    if grading and not isinstance(task, Task):
        cli.error("--grader-model grades a labeled task's examples; games score themselves")
    if grading and model_name(grading) == model_name(resolve_backend(args.backend, args.model)[0]):
        # Search would learn to ask the grader's own question and score perfectly for nothing.
        cli.error("--grader-model is the judge the graph asks; grade with a different model")
    if getattr(args, "visible", False) and deathmatch:
        cli.error("--visible shows a single-player Doom game; a deathmatch has no one window")
    keeping = getattr(args, "keep_judge", None) is not None
    if keeping and (grading or not isinstance(task, Task)):
        cli.error("--keep-judge measures rules against a labeled task's labels")
    if args.command in ("evaluate", "optimize") and isinstance(task, Task) and not grading:
        splits = ("train", "validation") if args.command == "optimize" else (args.split,)
        unlabeled = [e.id for split in splits for e in getattr(task, split) if e.label is None]
        if unlabeled:
            cli.error(f"examples {unlabeled} have no label; label them or pass a --grader-model")
    if args.command in ("evaluate", "optimize", "record"):
        resuming = getattr(args, "resume", False)
        checkpoint = args.output / "search" / "gepa_state.bin"
        if resuming and not checkpoint.exists():
            cli.error(f"no checkpoint to resume in {args.output}: {checkpoint} is missing")
        if args.output.exists() and not resuming:
            hint = "; pass --resume to continue its search" if checkpoint.exists() else ""
            cli.error(f"output already exists: {args.output}{hint}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "optimize":
        priced = prices(cli, args)
        config = SearchConfig(
            prefer=args.prefer,
            max_nodes=args.max_nodes,
            max_depth=args.max_depth,
            max_metric_calls=args.max_metric_calls,
            reflection_batch=args.reflection_batch,
            repair_attempts=args.repair_attempts,
            seed=args.seed,
            parallel_examples=args.parallel_examples,
            merges=args.merges,
            call_cost=priced.call,
            call_costs=priced.judges,
            extract_cost=priced.extract,
            significance=args.significance,
            mine=args.mine,
            evidence=args.evidence,
            keep_judge=args.keep_judge,
        )
        lm_options = {"api_base": args.reflection_base_url} if args.reflection_base_url else {}
        if args.reflection_timeout is not None:
            lm_options["timeout"] = args.reflection_timeout
        if args.reflection_options:
            lm_options["extra_body"] = args.reflection_options
        reflection = dspy.LM(
            args.reflection_model,
            temperature=1,
            max_tokens=args.reflection_max_tokens,
            cache=False,
            num_retries=0,
            **lm_options,
        )

        reflect = GraphProposer(reflection)

    execution = ExecutionConfig(max_workers=args.max_workers)
    model, judge_client = judge(
        args,
        args.command == "optimize"
        or any(graph.asks_own_judge() for graph in (plan, opponent) if graph),
        task if isinstance(task, Task) else None,
    )
    with ExitStack() as stack:
        client = stack.enter_context(judge_client)
        models = graph_models(cli, args, stack, plan, opponent)
        score = scorer(args, task) if args.command in ("evaluate", "optimize") else None
        if deathmatch:
            from rulesmith.arena import Duels

            score = Duels(task, DecisionProgram(opponent, client, model, execution, models=models))
        elif args.command != "run" and plays(task, "chess"):
            from rulesmith.chess import Engine, Games

            if engine:
                rival = Engine(engine, args.engine_skill or 0, args.engine_seconds or 0.05)
                score = Games(task, stack.enter_context(rival))
            else:
                score = Games(
                    task, DecisionProgram(opponent, client, model, execution, models=models)
                )
        elif args.command != "run" and plays(task, "doom"):
            from rulesmith.doom import DoomEnvironment, Episodes

            env = DoomEnvironment(
                task, visible=getattr(args, "visible", False), record=args.command == "record"
            )
            score = Episodes(task, stack.enter_context(env))
        if args.command == "run":
            prediction = DecisionProgram(plan, client, model, execution, args.trace, models=models)(
                state=read_state(cli, args.state)
            )
            print(dump(dict(prediction)), end="")
        elif args.command == "record":
            from vizdoom import DEFAULT_TICRATE

            from rulesmith.arena import duel

            program = DecisionProgram(plan, client, model, execution, models=models)
            frames, episodes = [], []
            for seed in args.seed:
                if deathmatch:
                    played, _ = duel(task, program, score.opponent, seed, True, frames)
                    episodes.append(played.record)
                else:
                    episodes.append(score(program, seed).record)
                    frames += env.frames
            save_video(frames, DEFAULT_TICRATE, args.output)
            print(dump({"episodes": episodes, "output": str(args.output)}), end="")
        elif args.command == "evaluate":
            report = evaluate_plan(
                plan,
                getattr(task, args.split),
                client,
                model,
                score,
                execution,
                args.parallel_examples,
                models,
            )
            report["split"] = args.split
            report["backend"] = args.backend
            report["model"] = model
            report["scored_by"] = scored_by(args, task)
            report["opponent"] = opponent_of(args)
            summary = {"score": report["score"], "output": str(args.output)}
            if isinstance(task, Task):
                real = at_prevalence(task, report["examples"])
                if real is not None:
                    report["at_prevalence"] = summary["at_prevalence"] = real
                report["drift"] = summary["drift"] = drift(plan, report["examples"], args.drift)
                for moved in report["drift"]:
                    print(
                        f"warning: {moved['rule']} decided {moved['share']:.0%} of these "
                        f"examples against {moved['measured_share']:.0%} when calibrated "
                        f"(p = {moved['p']:.2g}); the inputs have moved from the ones it was "
                        "measured on",
                        file=sys.stderr,
                    )
            args.output.write_text(dump(report))
            print(dump(summary), end="")
        else:
            with redirect_stdout(sys.stderr):
                best, report = optimize_plan(
                    task,
                    plan,
                    client,
                    model,
                    reflect,
                    config,
                    execution,
                    score,
                    run_dir=args.output / "search",
                    models=models,
                )
            report["backend"] = args.backend
            report["reflection_model"] = args.reflection_model
            report["scored_by"] = scored_by(args, task)
            report["opponent"] = opponent_of(args)
            if whole is not None:
                # Placed where the unit was, the winner must meet every hand-off the map makes.
                placed = whole.nodes[unit].model_copy(update={"plan": best})
                try:
                    placed_map = Plan.model_validate(
                        whole.model_copy(
                            update={"nodes": whole.nodes | {unit: placed}}
                        ).model_dump()
                    )
                except ValueError as error:
                    cli.error(f"the graph search found does not fit the map at {unit!r}: {error}")
                report["unit"] = {"name": unit, "path": units(args.plan).get(unit), "placed": True}
                if map_task is not None:
                    # A unit is judged by what the whole map does with it, on the map's own
                    # examples: better alone and worse for the map is worse.
                    scores = {}
                    for when, graph in (("before", whole), ("after", placed_map)):
                        scored = evaluate_plan(
                            graph,
                            map_task.validation,
                            client,
                            model,
                            Labels(map_task),
                            execution,
                            args.parallel_examples,
                            models,
                        )
                        scores[when] = scored["score"]
                    if scores["after"] < scores["before"]:
                        cli.error(
                            f"placed in the map, the graph search found scores {scores['after']:g} "
                            f"on the map task's validation examples against {scores['before']:g} "
                            "before: better alone, worse for the map"
                        )
                    report["unit"]["map_validation"] = scores
            args.output.mkdir(exist_ok=True)
            (args.output / "plan.json").write_text(best.model_dump_json(indent=2) + "\n")
            # The same graph as the rules search wrote, which is the form a person edits and
            # any command accepts; a graph beyond rules has no such form, and the report says.
            try:
                written = render_rules(best, described_labels(task))
                (args.output / "plan.rules").write_text(written + "\n")
                report["rules"] = "plan.rules"
            except RuleError:
                report["rules"] = None
            (args.output / "report.json").write_text(dump(report))
            print(dump(report), end="")


if __name__ == "__main__":
    main()
