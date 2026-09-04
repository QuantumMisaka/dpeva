import pytest

from dpeva.run import (
    InvalidStateTransition,
    RunEventKind,
    RunState,
    transition,
)
from dpeva.run.status import (
    InvalidStateTransition as StatusInvalidStateTransition,
)


def test_public_run_package_exports_canonical_symbols() -> None:
    assert InvalidStateTransition is StatusInvalidStateTransition
    assert {RunState.CREATED.value, RunState.FINISHED.value} == {"created", "finished"}
    assert RunEventKind.RESUME.value == "resume"
    assert transition(RunState.CREATED, RunState.VALIDATED) is RunState.VALIDATED


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


def test_local_lifecycle_reaches_finished() -> None:
    state = RunState.CREATED
    for target in (RunState.VALIDATED, RunState.RUNNING, RunState.FINISHED):
        state = transition(state, target)
    assert state is RunState.FINISHED


def test_simulated_slurm_lifecycle_marks_submission_before_running() -> None:
    state = transition(RunState.CREATED, RunState.VALIDATED)
    state = transition(state, RunState.SUBMITTED)
    assert transition(state, RunState.RUNNING) is RunState.RUNNING


@pytest.mark.parametrize("current", [RunState.CREATED, RunState.VALIDATED, RunState.SUBMITTED])
def test_any_pre_execution_state_can_reach_failed(current: RunState) -> None:
    assert transition(current, RunState.FAILED) is RunState.FAILED


def test_simulated_slurm_success_and_failure_terminal_paths() -> None:
    submitted = transition(RunState.VALIDATED, RunState.SUBMITTED)
    assert transition(transition(submitted, RunState.RUNNING), RunState.FINISHED) is RunState.FINISHED
    assert transition(submitted, RunState.FAILED) is RunState.FAILED


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunState.CREATED, RunState.RUNNING),
        (RunState.SUBMITTED, RunState.FINISHED),
        (RunState.PARTIAL, RunState.FINISHED),
        (RunState.FAILED, RunState.FINISHED),
        (RunState.FINISHED, RunState.FAILED),
    ],
)
def test_illegal_transitions_are_rejected(current: RunState, target: RunState) -> None:
    with pytest.raises(InvalidStateTransition):
        transition(current, target)


def test_partial_and_failed_recovery_are_explicit() -> None:
    assert transition(RunState.PARTIAL, RunState.RUNNING, RunEventKind.RESUME) is RunState.RUNNING
    assert transition(RunState.FAILED, RunState.RUNNING, RunEventKind.RECOVERY) is RunState.RUNNING
    with pytest.raises(InvalidStateTransition):
        transition(RunState.PARTIAL, RunState.RUNNING)


@pytest.mark.parametrize("event", [RunEventKind.RESUME, RunEventKind.RECOVERY])
def test_recovered_partial_run_can_finish(event: RunEventKind) -> None:
    running = transition(RunState.PARTIAL, RunState.RUNNING, event)
    assert transition(running, RunState.FINISHED) is RunState.FINISHED


@pytest.mark.parametrize("event", [RunEventKind.RESUME, RunEventKind.RECOVERY])
def test_recovered_failed_run_can_finish_or_fail(event: RunEventKind) -> None:
    running = transition(RunState.FAILED, RunState.RUNNING, event)
    assert transition(running, RunState.FINISHED) is RunState.FINISHED
    assert transition(running, RunState.FAILED) is RunState.FAILED
