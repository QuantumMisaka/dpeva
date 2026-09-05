#!/usr/bin/env python3
"""Execute the repository's named quality gates from one TOML manifest.

The runner deliberately accepts argv arrays only.  Gate definitions are data,
not shell snippets, so values from the manifest are never interpolated into a
shell command.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


_SCHEMA_VERSION = "1.0"
_GATE_FIELDS = frozenset({"argv", "layer", "owner", "basis"})
_TOP_LEVEL_FIELDS = frozenset({"schema_version", "gates", "profiles"})


@dataclass(frozen=True)
class Gate:
    name: str
    argv: tuple[str, ...]
    layer: str
    owner: str
    basis: str


@dataclass(frozen=True)
class Manifest:
    gates: dict[str, Gate]
    profiles: dict[str, tuple[str, ...]]
    repo_root: Path


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a table")
    return value


def _manifest_repo_root(path: Path) -> Path:
    resolved = path.resolve()
    return resolved.parent.parent if resolved.parent.name == "scripts" else resolved.parent


def load_manifest(path: Path) -> Manifest:
    """Load and strictly validate a gate manifest."""

    manifest_path = Path(path)
    try:
        with manifest_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {manifest_path}: {exc}") from exc

    unknown = set(raw) - _TOP_LEVEL_FIELDS
    if unknown:
        raise ValueError(f"unknown manifest field(s): {', '.join(sorted(unknown))}")
    if raw.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")

    raw_gates = _require_mapping(raw.get("gates"), "gates")
    raw_profiles = _require_mapping(raw.get("profiles"), "profiles")
    gates: dict[str, Gate] = {}
    for name, raw_gate in raw_gates.items():
        _require_string(name, "gate name")
        table = _require_mapping(raw_gate, f"gates.{name}")
        unknown_gate = set(table) - _GATE_FIELDS
        missing_gate = _GATE_FIELDS - set(table)
        if unknown_gate:
            raise ValueError(
                f"gates.{name} has unknown field(s): {', '.join(sorted(unknown_gate))}"
            )
        if missing_gate:
            raise ValueError(
                f"gates.{name} is missing field(s): {', '.join(sorted(missing_gate))}"
            )
        raw_argv = table["argv"]
        if not isinstance(raw_argv, list) or not raw_argv:
            raise ValueError(f"gates.{name}.argv must be a non-empty array")
        argv: list[str] = []
        for index, item in enumerate(raw_argv):
            if not isinstance(item, str) or not item:
                raise ValueError(f"gates.{name}.argv[{index}] must be a non-empty string")
            argv.append(item)
        gates[name] = Gate(
            name=name,
            argv=tuple(argv),
            layer=_require_string(table["layer"], f"gates.{name}.layer"),
            owner=_require_string(table["owner"], f"gates.{name}.owner"),
            basis=_require_string(table["basis"], f"gates.{name}.basis"),
        )

    profiles: dict[str, tuple[str, ...]] = {}
    for name, raw_refs in raw_profiles.items():
        _require_string(name, "profile name")
        if not isinstance(raw_refs, list):
            raise ValueError(f"profiles.{name} must be an array")
        refs: list[str] = []
        for index, ref in enumerate(raw_refs):
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"profiles.{name}[{index}] must be a non-empty string")
            if ref in refs:
                raise ValueError(f"profiles.{name} has duplicate gate reference: {ref}")
            if ref not in gates:
                raise ValueError(f"profiles.{name} references unknown gate: {ref}")
            refs.append(ref)
        profiles[name] = tuple(refs)

    collisions = set(gates).intersection(profiles)
    if collisions:
        raise ValueError(
            "gate/profile name collision(s): " + ", ".join(sorted(collisions))
        )

    return Manifest(gates=gates, profiles=profiles, repo_root=_manifest_repo_root(manifest_path))


def resolve_profile(manifest: Manifest, name: str) -> list[str]:
    """Resolve either one gate name or one named profile to ordered gate names."""

    if name in manifest.profiles:
        return list(manifest.profiles[name])
    if name in manifest.gates:
        return [name]
    raise KeyError(f"unknown gate or profile: {name}")


Runner = Callable[..., subprocess.CompletedProcess[Any]]


def run_names(
    names: Sequence[str],
    manifest: Manifest,
    runner: Runner = subprocess.run,
) -> int:
    """Run ordered gate names and return the first command's non-zero code."""

    for name in names:
        try:
            gate = manifest.gates[name]
        except KeyError as exc:
            raise KeyError(f"unknown gate: {name}") from exc
        try:
            result = runner(
                list(gate.argv),
                check=False,
                shell=False,
                cwd=manifest.repo_root,
            )
        except OSError as exc:
            print(f"gate {name!r} could not start: {exc}", file=sys.stderr)
            return 127
        if result.returncode:
            return int(result.returncode)
    return 0


def run_gate(name: str, manifest: Manifest, runner: Runner = subprocess.run) -> int:
    """Resolve and execute one gate or profile."""

    return run_names(resolve_profile(manifest, name), manifest=manifest, runner=runner)


def _print_catalog(manifest: Manifest) -> None:
    print("Gates:")
    for name, gate in manifest.gates.items():
        print(f"  {name}\t{gate.layer}\t{gate.owner}\t{gate.basis}")
    print("Profiles:")
    for name, refs in manifest.profiles.items():
        print(f"  {name}\t{', '.join(refs)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", help="gate or profile to execute")
    parser.add_argument("--list", action="store_true", help="list gates and profiles without executing")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    manifest = load_manifest(repo_root / "scripts" / "gates.toml")
    if args.list:
        _print_catalog(manifest)
        return 0
    if args.name is None:
        parser.error("a gate or profile name is required (or use --list)")
    try:
        return run_gate(args.name, manifest)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
