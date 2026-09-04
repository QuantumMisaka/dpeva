import pytest

from dpeva.run.status import (
    InvalidStateTransition,
    RunEventKind,
    RunState,
    transition,
)


def test_canonical_state_values_are_closed() -> None:
    assert {state.value for state in RunState} == {
        "created", "validated", "submitted", "running", "partial", "failed", "finished"
    }


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.CREATED, RunState.VALIDATED),
        (RunState.VALIDATED, RunState.RUNNING),
        (RunState.VALIDATED, RunState.SUBMITTED),
        (RunState.SUBMITTED, RunState.RUNNING),
        (RunState.RUNNING, RunState.PARTIAL),
        (RunState.RUNNING, RunState.FAILED),
        (RunState.RUNNING, RunState.FINISHED),
        (RunState.PARTIAL, RunState.FAILED),
    ],
)
def test_normal_transitions(current: RunState, target: RunState) -> None:
    assert transition(current, target) is target


def test_recovery_requires_an_explicit_event() -> None:
    with pytest.raises(InvalidStateTransition):
        transition(RunState.FAILED, RunState.RUNNING)
    assert transition(RunState.FAILED, RunState.RUNNING, RunEventKind.RECOVERY) is RunState.RUNNING


def test_finished_is_terminal() -> None:
    with pytest.raises(InvalidStateTransition):
        transition(RunState.FINISHED, RunState.RUNNING, RunEventKind.RESUME)
