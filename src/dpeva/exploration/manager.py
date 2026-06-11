"""Exploration backend registry."""

from __future__ import annotations

from collections.abc import Callable

from .base import ExplorationBackend, ExplorationRequest, ExplorationResult


BackendFactory = Callable[[], ExplorationBackend]


_BACKENDS: dict[str, BackendFactory] = {}


def _register_defaults() -> None:
    if "atst-tools" not in _BACKENDS:
        from .atst_backend import ATSTToolsBackend

        _BACKENDS["atst-tools"] = ATSTToolsBackend


def register_backend(name: str, factory: BackendFactory) -> None:
    _BACKENDS[name] = factory


def get_backend(name: str) -> ExplorationBackend:
    _register_defaults()
    try:
        return _BACKENDS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown exploration backend: {name}") from exc


def run_exploration(request: ExplorationRequest) -> ExplorationResult:
    return get_backend(request.backend).run(request)
