"""Shared scope contract for the DeepMD 3.2 qualification harness."""

from __future__ import annotations

from typing import Any, Iterable


VALID_SCOPES = ("dpa4", "all")
ENVIRONMENT_CASES = ("pip-freeze", "deepmd-version", "torch-cuda", "gpu")
DPA4_CASES = (
    "pt-test",
    "pt-test-ema",
    "pt-eval-desc",
    "pt-eval-desc-ema",
    "pt-embed",
    "pt-embed-ema",
)
DPA4C_CASES = ("dpa4c-periodic-eval-desc",)


def normalize_scope(value: Any) -> str:
    """Normalize a scope, preserving scope-less historical input as ``all``."""

    if value is None:
        return "all"
    if not isinstance(value, str) or value not in VALID_SCOPES:
        raise ValueError(f"qualification scope must be one of {VALID_SCOPES}")
    return value


def bind_scope(payload: dict[str, Any], requested: str | None = None) -> str:
    """Resolve one payload scope and reject caller attempts to reinterpret it."""

    payload_scope = normalize_scope(payload.get("scope"))
    if requested is not None and normalize_scope(requested) != payload_scope:
        raise ValueError(
            f"requested qualification scope {requested!r} does not match input scope {payload_scope!r}"
        )
    return payload_scope


def command_cases(scope: str) -> tuple[str, ...]:
    selected = normalize_scope(scope)
    return ENVIRONMENT_CASES + DPA4_CASES + (DPA4C_CASES if selected == "all" else ())


def record_cases(scope: str) -> tuple[str, ...]:
    return ("preflight", *command_cases(scope))


def attestation_specs(records: Iterable[Any], scope: str) -> list[dict[str, Any]]:
    """Build manifest-bound attestation specs for cases selected by ``scope``."""

    selected_cases = set(DPA4_CASES + (DPA4C_CASES if normalize_scope(scope) == "all" else ()))
    return [
        {
            "case": case,
            "capability_key": record.key.model_dump(),
            "verification_command": record.verification_command,
            "source": "sai-v100-qualification",
        }
        for record in records
        if record.sai_verification_cases
        for case in record.sai_verification_cases
        if case in selected_cases
    ]
