from __future__ import annotations

from enum import Enum


class RunState(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    SUBMITTED = "submitted"
    RUNNING = "running"
    PARTIAL = "partial"
    FAILED = "failed"
    FINISHED = "finished"


class RunEventKind(str, Enum):
    RESUME = "resume"
    RECOVERY = "recovery"


class InvalidStateTransition(ValueError):
    pass


_NORMAL_TRANSITIONS = {
    RunState.CREATED: {RunState.VALIDATED, RunState.FAILED},
    RunState.VALIDATED: {RunState.SUBMITTED, RunState.RUNNING, RunState.FAILED},
    RunState.SUBMITTED: {RunState.RUNNING, RunState.FAILED},
    RunState.RUNNING: {RunState.PARTIAL, RunState.FAILED, RunState.FINISHED},
    RunState.PARTIAL: {RunState.FAILED},
    RunState.FAILED: set(),
    RunState.FINISHED: set(),
}


def transition(
    current: RunState,
    target: RunState,
    event: RunEventKind | None = None,
) -> RunState:
    if target in _NORMAL_TRANSITIONS[current]:
        return target
    if current in {RunState.FAILED, RunState.PARTIAL} and target is RunState.RUNNING:
        if event in {RunEventKind.RESUME, RunEventKind.RECOVERY}:
            return target
    raise InvalidStateTransition(f"illegal run-state transition: {current.value} -> {target.value}")
