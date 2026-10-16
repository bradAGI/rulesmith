"""Teach the judge what the task knows: a Ruling server keeps the decisions it answers and takes
each one's outcome later, and retrains on them. During a search, every train example carries
its label, so the label question's right answer is known the moment the judge answers it.

Only train examples teach. Validation and test examples are what the search and the final
score are measured on; a judge that learned their labels would make both meaningless, so
their states never reach the outcome endpoint. A graph hands the judge an object built from a
call's inputs, the example's state as one of its values, often nested; a task keeps every
state unique across its splits, so the example an object holds is known as soon as it is found.
"""

from __future__ import annotations

import json
import time
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import NamedTuple, Self

import httpx
from typesafe_sdk import (
    ChoiceAnswer,
    NoulAnswer,
    RetryPolicy,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

from rulesmith.graph import Task

DECISIONS = "/v1/systemone"
OUTCOMES = "/v1/decisions/{decision}/outcome"


class RulingError(RuntimeError):
    """The Ruling server could not be reached, or refused a request."""


class Decision(NamedTuple):
    """A System One response plus the id Ruling gave the decision, when it keeps a ledger."""

    model: str
    usage: Usage
    answers: dict[str, NoulAnswer | ChoiceAnswer | ScoreAnswer]
    decision: str | None


class RulingClient:
    """Asks a Ruling server System One questions and keeps the decision id its ledger returns,
    which the hosted SDK's client drops as a field it does not know."""

    def __init__(self, *, base_url: str, api_key: str | None, timeout: float, retry: RetryPolicy):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self.retry = retry

    def system_one(self, *, state, questions: dict[str, dict], model: str) -> Decision:
        body = self._post(DECISIONS, {"model": model, "state": state, "questions": questions})
        usage = body.get("usage") or {}
        return Decision(
            model=body["model"],
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            answers={name: parse_answer(answer) for name, answer in body["answers"].items()},
            decision=body.get("id"),
        )

    def outcome(self, decision: str, labels: dict[str, object]) -> None:
        self._post(OUTCOMES.format(decision=decision), {"labels": labels})

    def _post(self, path: str, body: dict) -> dict:
        attempt = 0
        while True:
            last = attempt == self.retry.max_retries
            try:
                response = self.http.post(path, json=body)
            except httpx.TransportError as error:
                if last:
                    raise RulingError(f"{self.http.base_url} is unreachable: {error}") from error
            else:
                if response.is_success:
                    return response.json()
                wait = retry_after(response.headers.get("Retry-After"))
                if last or response.status_code not in self.retry.http_statuses:
                    raise RulingError(
                        f"{self.http.base_url}{path} answered {response.status_code}: "
                        f"{response.text}"
                    )
            # A server that says how long it needs is believed; one that does not gets backoff.
            if wait is None:
                wait = min(self.retry.backoff_initial * 2**attempt, self.retry.backoff_max)
            time.sleep(wait)
            attempt += 1

    def close(self) -> None:
        self.http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def retry_after(header: str | None) -> float | None:
    """Seconds a `Retry-After` header asks for, given as a delay or as a date; None without one."""
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(header).timestamp() - time.time())
        except (ValueError, TypeError):
            return None


def parse_answer(answer: dict) -> NoulAnswer | ChoiceAnswer | ScoreAnswer:
    """The hosted SDK's answer types, built from Ruling's JSON, which is the hosted wire format."""
    kind = answer.get("type")
    if kind == "noul":
        return NoulAnswer(noul=answer["noul"])
    if kind == "choice":
        return ChoiceAnswer(
            choice=answer["choice"],
            probabilities=answer["probabilities"],
            confidence=answer["confidence"],
        )
    if kind == "score":
        return ScoreAnswer(
            score=answer["score"],
            confidence=answer["confidence"],
            legend={int(level): text for level, text in answer["legend"].items()},
            probabilities={int(level): p for level, p in answer["probabilities"].items()},
        )
    raise RulingError(f"an answer of unknown type {kind!r}")


def state_key(state) -> str:
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def _children(value):
    return value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()


class Teaching:
    """The judge's client for a search, which reports the train label as the outcome of every
    label question answered about a train example. A label question is a choice whose options
    are task labels; the label is then exactly its right answer. An example with other accepted
    labels has no single right answer and teaches nothing. A decision about a held-out example
    teaches nothing either, even when that example's state happens to contain a train state."""

    def __init__(self, client: RulingClient, task: Task):
        self.client = client
        self.labels = set(task.labels)
        self.lessons = {
            state_key(example.state): example.label
            for example in task.train
            if example.label is not None and not example.accepted
        }
        held_out = [*task.validation, *task.test, *task.calibration]
        self.held_out = {state_key(example.state) for example in held_out}
        self.taught = 0

    def system_one(self, *, state, questions: dict[str, dict], model: str) -> SystemOneResponse:
        decision = self.client.system_one(state=state, questions=questions, model=model)
        label = self._lesson(state)
        if label is not None and decision.decision is not None:
            lesson = {
                name: label
                for name, question in questions.items()
                if question.get("type") == "choice"
                and label in question["criteria"]
                and set(question["criteria"]) <= self.labels
            }
            if lesson:
                self.client.outcome(decision.decision, lesson)
                self.taught += len(lesson)
        return SystemOneResponse(
            model=decision.model, usage=decision.usage, answers=decision.answers
        )

    def _lesson(self, asked) -> str | None:
        """The train label of the example whose state `asked` is, or holds at any depth, unless
        it is or holds a held-out example's state."""
        return None if self._holds(asked, self.held_out) else self._found(asked)

    def _found(self, asked) -> str | None:
        found = self.lessons.get(state_key(asked))
        if found is not None:
            return found
        return next((label for child in _children(asked) if (label := self._found(child))), None)

    def _holds(self, asked, keys: set[str]) -> bool:
        return state_key(asked) in keys or any(
            self._holds(child, keys) for child in _children(asked)
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
