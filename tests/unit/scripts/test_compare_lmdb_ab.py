"""Tests for the LMDB A/B evidence producer."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.validation.compare_lmdb_ab import (
    compare_metrics,
    parse_aggregate_metrics,
    sha256_tree,
)


def test_parse_aggregate_metrics_uses_last_weighted_block() -> None:
    text = """
    [2026-01-01 00:00:00,000] DEEPMD INFO    # number of test data : 1
    [2026-01-01 00:00:00,000] DEEPMD INFO    Energy MAE/Natoms  : 9.0
    # number of test data : 1
    Energy MAE/Natoms  : 9.0
    Force  MAE         : 8.0
    Virial MAE/Natoms  : 7.0
    Stress MAE         : 6.0
    [2026-01-01 00:00:00,000] DEEPMD INFO    # number of systems : 1
    [2026-01-01 00:00:00,000] DEEPMD INFO    Energy MAE/Natoms  : 1.0
    [2026-01-01 00:00:00,000] DEEPMD INFO    Force  MAE         : 2.0
    [2026-01-01 00:00:00,000] DEEPMD INFO    Virial MAE/Natoms  : 3.0
    [2026-01-01 00:00:00,000] DEEPMD INFO    Stress MAE         : 4.0
    """

    assert parse_aggregate_metrics(text) == {
        "energy_mae_per_atom": 1.0,
        "force_mae": 2.0,
        "virial_mae_per_atom": 3.0,
        "stress_mae": 4.0,
    }


def test_compare_metrics_marks_relative_difference() -> None:
    npy = {
        "energy_mae_per_atom": 1.0,
        "force_mae": 2.0,
        "virial_mae_per_atom": 3.0,
        "stress_mae": 4.0,
    }
    lmdb = {
        "energy_mae_per_atom": 1.0 + 1e-6,
        "force_mae": 2.0 - 2e-6,
        "virial_mae_per_atom": 3.0,
        "stress_mae": 4.0,
    }

    comparison = compare_metrics(npy, lmdb, threshold=2e-5)

    assert all(item["passed"] for item in comparison.values())
    assert comparison["energy_mae_per_atom"]["relative_diff"] == pytest.approx(
        1e-6, abs=2e-12
    )


def test_sha256_tree_changes_with_content(tmp_path: Path) -> None:
    root = tmp_path / "data"
    (root / "a").mkdir(parents=True)
    (root / "a" / "x").write_text("one", encoding="utf-8")
    first = sha256_tree(root)
    (root / "a" / "x").write_text("two", encoding="utf-8")

    assert sha256_tree(root) != first
