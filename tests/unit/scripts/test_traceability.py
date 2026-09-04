from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_traceability import validate_traceability


def write_registry(root: Path, **overrides: object) -> Path:
    entry = {
        "capability_id": "example.capability",
        "code_paths": ["src/example.py"],
        "test_paths": ["tests/test_example.py"],
        "documentation_paths": ["docs/example.md"],
        "owner": "Example Owner",
        "evidence_path": "docs/reports/evidence.md",
    }
    entry.update(overrides)
    path = root / "registry.json"
    path.write_text(json.dumps([entry]), encoding="utf-8")
    return path


def make_repo(tmp_path: Path) -> Path:
    for relative in (
        "src/example.py",
        "tests/test_example.py",
        "docs/example.md",
        "docs/reports/evidence.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture", encoding="utf-8")
    return tmp_path


def test_every_entry_has_existing_paths() -> None:
    failures = validate_traceability(
        Path("docs/governance/traceability/capability-evidence.json"),
        repo_root=Path.cwd(),
    )
    assert failures == []


def test_validation_does_not_require_capability_text_in_source(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    registry = write_registry(repo)
    assert validate_traceability(registry, repo_root=repo) == []


@pytest.mark.parametrize("value", ["../outside.py", "/tmp/outside.py", "src/../outside.py"])
def test_rejects_non_repo_paths(tmp_path: Path, value: str) -> None:
    repo = make_repo(tmp_path)
    registry = write_registry(repo, code_paths=[value])
    failures = validate_traceability(registry, repo_root=repo)
    assert any("code_paths" in failure for failure in failures)


def test_rejects_symlink_escape(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    outside = tmp_path.parent / "outside.py"
    outside.write_text("outside", encoding="utf-8")
    link = repo / "src" / "linked.py"
    link.symlink_to(outside)
    registry = write_registry(repo, code_paths=["src/linked.py"])
    failures = validate_traceability(registry, repo_root=repo)
    assert any("code_paths" in failure for failure in failures)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [{"capability_id": "only-id"}],
        [{
            "capability_id": "example.capability",
            "code_paths": [],
            "test_paths": ["tests/test_example.py"],
            "documentation_paths": ["docs/example.md"],
            "owner": "Example Owner",
            "evidence_path": "docs/reports/evidence.md",
        }],
        [
            {
                "capability_id": "example.capability",
                "code_paths": ["src/example.py"],
                "test_paths": ["tests/test_example.py"],
                "documentation_paths": ["docs/example.md"],
                "owner": "Example Owner",
                "evidence_path": "docs/reports/evidence.md",
                "unexpected": True,
            }
        ],
    ],
)
def test_rejects_invalid_root_or_exact_entry_schema(tmp_path: Path, payload: object) -> None:
    repo = make_repo(tmp_path)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    assert validate_traceability(registry, repo_root=repo)


def test_rejects_duplicate_ids_and_non_file_paths(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    registry = write_registry(repo)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload.append(payload[0])
    payload[0]["evidence_path"] = "docs"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    failures = validate_traceability(registry, repo_root=repo)
    assert any("duplicate" in failure for failure in failures)
    assert any("evidence_path" in failure for failure in failures)
