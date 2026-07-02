# Dataset Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `dpeva audit` and `analysis mode="audit"` for post-collect and dataset diagnostics, including pool balance, descriptor novelty, composition coverage, overlap, and optional log-det entropy metrics.

**Architecture:** Implement audit as an analysis extension. Pure metric functions live in `src/dpeva/analysis/audit.py`; filesystem orchestration and report writing live in `src/dpeva/analysis/audit_manager.py`; `AnalysisWorkflow` routes `mode="audit"` to the manager; `dpeva audit` is a CLI alias for the same workflow.

**Tech Stack:** Python, Pydantic v2, NumPy, pandas, scikit-learn `NearestNeighbors`, existing `dpeva.io.dataset.load_systems`, existing `dpeva.io.collection.CollectionIOManager`, pytest.

---

## Reference Design

Read before implementing:

- `docs/superpowers/specs/2026-07-03-dataset-audit-design.md`
- `src/dpeva/config.py`
- `src/dpeva/cli.py`
- `src/dpeva/workflows/analysis.py`
- `src/dpeva/analysis/dataset.py`
- `src/dpeva/io/collection.py`
- `tests/unit/analysis/test_dataset_manager.py`
- `tests/unit/workflows/test_analysis_workflow.py`

Do not use iter10 practice outputs as test fixtures. Use small synthetic fixtures in unit tests.

## Files

Create:

- `src/dpeva/analysis/audit.py`
- `src/dpeva/analysis/audit_manager.py`
- `tests/unit/analysis/test_audit_metrics.py`
- `tests/unit/analysis/test_audit_manager.py`
- `examples/recipes/analysis/config_audit.json`

Modify:

- `src/dpeva/config.py`
- `src/dpeva/cli.py`
- `src/dpeva/workflows/analysis.py`
- `src/dpeva/analysis/__init__.py`
- `src/dpeva/constants.py`
- `tests/unit/test_cli.py`
- `tests/unit/workflows/test_analysis_workflow.py`
- `docs/guides/cli.md`
- `docs/guides/configuration.md`
- `docs/source/api/config/analysis.rst`
- `examples/recipes/README.md`
- `examples/recipes/analysis/README.md`

---

### Task 1: Config Models and CLI Alias

**Files:**
- Modify: `src/dpeva/config.py`
- Modify: `src/dpeva/cli.py`
- Modify: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/unit/test_cli.py` or a new `tests/unit/analysis/test_audit_config.py` if that better matches local style:

```python
from pathlib import Path

import pytest

from dpeva.config import AnalysisConfig


def test_analysis_config_accepts_audit_mode_with_target(tmp_path):
    target = tmp_path / "collect"
    target.mkdir()
    config = AnalysisConfig(
        mode="audit",
        output_dir=tmp_path / "audit",
        audit_targets=[{"name": "sampled", "collect_dir": target}],
    )
    assert config.mode == "audit"
    assert config.audit_targets[0].name == "sampled"
    assert config.audit_targets[0].collect_dir == target
    assert config.audit_metrics.basic is True
    assert config.audit_metrics.entropy is False


def test_analysis_config_rejects_audit_without_targets(tmp_path):
    with pytest.raises(ValueError, match="audit_targets is required"):
        AnalysisConfig(mode="audit", output_dir=tmp_path / "audit")
```

- [ ] **Step 2: Run config tests and confirm failure**

Run:

```bash
pytest tests/unit/test_cli.py -q
```

Expected: failure because `AnalysisConfig.mode` does not allow `"audit"` and audit config models do not exist.

- [ ] **Step 3: Add audit config models**

In `src/dpeva/config.py`, add these models near `AnalysisConfig`:

```python
class AuditDatasetConfig(BaseModel):
    """Dataset, descriptor, or collect artifact to audit."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True, protected_namespaces=())

    name: str = Field(..., description="Human-readable dataset or collect-run name.")
    dataset_dir: Optional[Path] = Field(None, description="Path to dpdata-compatible dataset.")
    collect_dir: Optional[Path] = Field(None, description="Path to a DP-EVA collect output directory.")
    final_df_path: Optional[Path] = Field(None, description="Path to collect dataframe/final_df.csv.")
    desc_dir: Optional[Path] = Field(None, description="Path to descriptor directory or HDF5 embedding directory.")
    desc_feature_kind: Literal["descriptor", "fitting_last_layer"] = Field(
        "descriptor",
        description="Descriptor kind used for audit descriptor metrics.",
    )
    hdf5_dataset: Optional[str] = Field(None, description="Explicit HDF5 dataset name for descriptor loading.")

    @model_validator(mode="after")
    def validate_has_input(self):
        if not any([self.dataset_dir, self.collect_dir, self.final_df_path, self.desc_dir]):
            raise ValueError(
                "Each audit target must provide dataset_dir, collect_dir, final_df_path, or desc_dir"
            )
        return self


class AuditMetricsConfig(BaseModel):
    """Metric switches for audit analysis."""

    basic: bool = Field(True, description="Compute frame/system, pool, and log summary metrics.")
    descriptor_nn: bool = Field(True, description="Compute descriptor nearest-neighbor novelty metrics.")
    composition: bool = Field(True, description="Compute composition coverage metrics.")
    overlap: bool = Field(True, description="Compute exact dataname overlap across targets.")
    entropy: bool = Field(False, description="Compute lightweight log-det feature entropy metrics.")
    advanced_entropy: bool = Field(False, description="Enable QUESTS-style advanced entropy metrics.")
    entropy_ridge: float = Field(1e-6, gt=0.0, description="Diagonal ridge for covariance log-det entropy.")
    entropy_pca_dim: Optional[int] = Field(None, gt=0, description="Optional PCA dimension before log-det entropy.")
    entropy_bandwidth: Optional[float] = Field(None, gt=0.0, description="Bandwidth for advanced KDE entropy.")
```

Then update `AnalysisConfig`:

```python
mode: Literal["model_test", "dataset", "audit"] = Field("model_test", description="Analysis mode.")
audit_reference: Optional[AuditDatasetConfig] = Field(None, description="Reference dataset for audit novelty metrics.")
audit_targets: List[AuditDatasetConfig] = Field(default_factory=list, description="Targets to audit.")
audit_metrics: AuditMetricsConfig = Field(default_factory=AuditMetricsConfig, description="Audit metric switches.")
```

Extend `validate_mode_paths()`:

```python
if self.mode == "audit" and not self.audit_targets:
    raise ValueError("audit_targets is required when mode='audit'")
```

- [ ] **Step 4: Add CLI alias test**

Add this test to `tests/unit/test_cli.py`:

```python
import json
from unittest.mock import patch

from dpeva.cli import main


def test_audit_cli_dispatches_analysis_workflow(tmp_path, monkeypatch):
    cfg = tmp_path / "audit.json"
    collect_dir = tmp_path / "collect"
    collect_dir.mkdir()
    cfg.write_text(json.dumps({
        "output_dir": str(tmp_path / "audit"),
        "audit_targets": [{"name": "sampled", "collect_dir": str(collect_dir)}],
    }))
    monkeypatch.setattr("sys.argv", ["dpeva", "--no-banner", "audit", str(cfg)])

    with patch("dpeva.workflows.analysis.AnalysisWorkflow") as workflow_cls:
        main()

    args, kwargs = workflow_cls.call_args
    assert args[0]["mode"] == "audit"
    workflow_cls.return_value.run.assert_called_once()
```

- [ ] **Step 5: Run CLI alias test and confirm failure**

Run:

```bash
pytest tests/unit/test_cli.py::test_audit_cli_dispatches_analysis_workflow -q
```

Expected: failure because the `audit` subparser and handler do not exist.

- [ ] **Step 6: Implement CLI alias**

In `src/dpeva/cli.py`, add:

```python
def handle_audit(args):
    """Handles the 'audit' command as an alias for analysis mode='audit'."""
    from dpeva.workflows.analysis import AnalysisWorkflow
    config = load_and_resolve_config(args.config)
    config["mode"] = "audit"
    workflow = AnalysisWorkflow(config, config_path=os.path.abspath(args.config))
    workflow.run()
```

In `main()`, add after the `analysis` parser:

```python
p_audit = subparsers.add_parser("audit", help="Run Dataset and Collect Audit Workflow")
p_audit.add_argument("config", type=validate_config_path, help="Path to configuration JSON")
p_audit.set_defaults(func=handle_audit)
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/unit/test_cli.py tests/unit/analysis/test_audit_config.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/dpeva/config.py src/dpeva/cli.py tests/unit/test_cli.py tests/unit/analysis/test_audit_config.py
git commit -m "feat: add audit analysis config and cli alias"
```

---

### Task 2: Pure Audit Metric Functions

**Files:**
- Create: `src/dpeva/analysis/audit.py`
- Create: `tests/unit/analysis/test_audit_metrics.py`
- Modify: `src/dpeva/analysis/__init__.py`

- [ ] **Step 1: Write failing metric tests**

Create `tests/unit/analysis/test_audit_metrics.py`:

```python
import math

import numpy as np
import pandas as pd

from dpeva.analysis.audit import (
    descriptor_columns,
    internal_nearest_neighbor_summary,
    logdet_entropy,
    nearest_neighbor_summary,
    normalized_entropy,
    pairwise_overlap,
    pool_name_from_dataname,
    pool_summary,
)


def test_pool_name_from_dataname_handles_nested_systems():
    assert pool_name_from_dataname("DeepCNT/sys_a-3") == "DeepCNT"
    assert pool_name_from_dataname("sys_a-3") == "sys_a"


def test_normalized_entropy_balanced_and_collapsed():
    assert normalized_entropy([5, 5]) == 1.0
    assert normalized_entropy([10, 0]) == 0.0
    assert normalized_entropy([]) == 0.0


def test_pool_summary_counts_and_entropy():
    df = pd.DataFrame({"dataname": ["A/sys-0", "A/sys-1", "B/sys-0"]})
    summary = pool_summary(df)
    assert summary.counts == {"A": 2, "B": 1}
    expected = -(2 / 3) * math.log(2 / 3) - (1 / 3) * math.log(1 / 3)
    assert summary.entropy == expected / math.log(2)


def test_descriptor_columns_are_numeric_sorted():
    df = pd.DataFrame(columns=["desc_stru_10", "dataname", "desc_stru_2", "desc_stru_1"])
    assert descriptor_columns(df) == ["desc_stru_1", "desc_stru_2", "desc_stru_10"]


def test_nearest_neighbor_summary_reports_distribution():
    query = np.array([[0.0, 0.0], [2.0, 0.0]])
    reference = np.array([[0.0, 0.0], [10.0, 0.0]])
    stats = nearest_neighbor_summary(query, reference, "nn_to_reference")
    assert stats["nn_to_reference_min"] == 0.0
    assert stats["nn_to_reference_median"] == 1.0
    assert stats["nn_to_reference_mean"] == 1.0


def test_internal_nearest_neighbor_uses_second_neighbor():
    desc = np.array([[0.0], [2.0], [5.0]])
    stats = internal_nearest_neighbor_summary(desc, "selected_internal_nn")
    assert stats["selected_internal_nn_median"] == 2.0


def test_pairwise_overlap_by_dataname():
    frames = {
        "a": pd.DataFrame({"dataname": ["x-0", "x-1", "x-2"]}),
        "b": pd.DataFrame({"dataname": ["x-1", "x-2", "x-3"]}),
    }
    overlap = pairwise_overlap(frames)
    row = overlap.iloc[0].to_dict()
    assert row["left"] == "a"
    assert row["right"] == "b"
    assert row["overlap"] == 2
    assert row["left_unique"] == 1
    assert row["right_unique"] == 1


def test_logdet_entropy_is_finite_with_collinear_features():
    features = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    value = logdet_entropy(features, ridge=1e-6)
    assert np.isfinite(value)
```

- [ ] **Step 2: Run metric tests and confirm failure**

Run:

```bash
pytest tests/unit/analysis/test_audit_metrics.py -q
```

Expected: import failure because `dpeva.analysis.audit` does not exist.

- [ ] **Step 3: Implement pure metrics**

Create `src/dpeva/analysis/audit.py` with:

```python
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from dpeva.constants import COL_DESC_PREFIX


@dataclass(frozen=True)
class PoolSummary:
    counts: dict[str, int]
    entropy: float
    n_frames: int
    n_pools: int


def system_name_from_dataname(dataname: str) -> str:
    return str(dataname).rsplit("-", 1)[0]


def pool_name_from_dataname(dataname: str) -> str:
    sys_name = system_name_from_dataname(dataname)
    return sys_name.split("/", 1)[0]


def normalized_entropy(counts: list[int] | np.ndarray) -> float:
    values = np.asarray(counts, dtype=float)
    values = values[values > 0]
    if len(values) <= 1:
        return 0.0
    probs = values / values.sum()
    entropy = -float(np.sum(probs * np.log(probs)))
    return entropy / math.log(len(values))


def pool_summary(df: pd.DataFrame) -> PoolSummary:
    if "dataname" not in df:
        raise ValueError("pool_summary requires a dataname column")
    counts_series = df["dataname"].map(pool_name_from_dataname).value_counts().sort_index()
    counts = {str(k): int(v) for k, v in counts_series.items()}
    return PoolSummary(
        counts=counts,
        entropy=normalized_entropy(list(counts.values())),
        n_frames=int(sum(counts.values())),
        n_pools=len(counts),
    )


def descriptor_columns(df: pd.DataFrame) -> list[str]:
    cols = [col for col in df.columns if col.startswith(COL_DESC_PREFIX)]
    return sorted(cols, key=lambda col: int(col.rsplit("_", 1)[1]))


def _summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    if len(values) == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_p05": float("nan"),
            f"{prefix}_p25": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p75": float("nan"),
            f"{prefix}_p95": float("nan"),
        }
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p25": float(np.percentile(values, 25)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_p75": float(np.percentile(values, 75)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def nearest_neighbor_summary(query: np.ndarray, reference: np.ndarray, prefix: str) -> dict[str, float]:
    if len(query) == 0 or len(reference) == 0:
        return _summary(np.asarray([], dtype=float), prefix)
    nn = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="euclidean")
    nn.fit(reference)
    distances = nn.kneighbors(query, return_distance=True)[0][:, 0]
    return _summary(distances, prefix)


def internal_nearest_neighbor_summary(features: np.ndarray, prefix: str) -> dict[str, float]:
    if len(features) < 2:
        return _summary(np.asarray([], dtype=float), prefix)
    nn = NearestNeighbors(n_neighbors=2, algorithm="auto", metric="euclidean")
    nn.fit(features)
    distances = nn.kneighbors(features, return_distance=True)[0][:, 1]
    return _summary(distances, prefix)


def pairwise_overlap(final_dfs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    names = list(final_dfs)
    for i, left in enumerate(names):
        left_names = set(final_dfs[left]["dataname"])
        for right in names[i + 1:]:
            right_names = set(final_dfs[right]["dataname"])
            overlap = len(left_names & right_names)
            union = len(left_names | right_names)
            rows.append({
                "left": left,
                "right": right,
                "overlap": overlap,
                "jaccard": overlap / union if union else 0.0,
                "left_fraction": overlap / len(left_names) if left_names else 0.0,
                "right_fraction": overlap / len(right_names) if right_names else 0.0,
                "left_unique": len(left_names - right_names),
                "right_unique": len(right_names - left_names),
            })
    return pd.DataFrame(rows)


def logdet_entropy(features: np.ndarray, ridge: float = 1e-6) -> float:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    array = np.asarray(features, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0:
        return float("nan")
    if array.shape[0] == 1:
        cov = np.zeros((array.shape[1], array.shape[1]), dtype=float)
    else:
        cov = np.cov(array, rowvar=False)
        cov = np.atleast_2d(cov)
    cov = cov + ridge * np.eye(cov.shape[0], dtype=float)
    sign, value = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("nan")
    return float(value)


def parse_collect_log(path: Path) -> dict[str, object]:
    text = path.read_text(errors="ignore")
    patterns = {
        "qbc_lo_hi": r"Auto QbC: \[([0-9.]+), ([0-9.]+)\]",
        "rnd_lo_hi": r"Auto RND: \[([0-9.]+), ([0-9.]+)\]",
        "filter_counts": r"Filter Results: Cand=(\d+), Acc=(\d+), Fail=(\d+)",
        "sampled_total": r"Total sampled frames: (\d+)",
        "export_summary": r"Export summary: sampled=(\d+) systems/(\d+) frames, other=(\d+) systems/(\d+) frames",
    }
    result: dict[str, object] = {"workflow_finished": "DPEVA_TAG: WORKFLOW_FINISHED" in text}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match is None:
            continue
        values = match.groups()
        if key in {"qbc_lo_hi", "rnd_lo_hi"}:
            result[key] = tuple(float(value) for value in values)
        else:
            result[key] = tuple(int(value) for value in values)
    return result
```

- [ ] **Step 4: Export module API**

Add this to `src/dpeva/analysis/__init__.py`:

```python
from dpeva.analysis.audit import (
    PoolSummary,
    descriptor_columns,
    internal_nearest_neighbor_summary,
    logdet_entropy,
    nearest_neighbor_summary,
    normalized_entropy,
    pairwise_overlap,
    parse_collect_log,
    pool_name_from_dataname,
    pool_summary,
)
```

- [ ] **Step 5: Run metric tests**

Run:

```bash
pytest tests/unit/analysis/test_audit_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/dpeva/analysis/audit.py src/dpeva/analysis/__init__.py tests/unit/analysis/test_audit_metrics.py
git commit -m "feat: add audit metric primitives"
```

---

### Task 3: Composition Metrics and Collect Artifact Resolution

**Files:**
- Modify: `src/dpeva/analysis/audit.py`
- Modify: `tests/unit/analysis/test_audit_metrics.py`

- [ ] **Step 1: Add failing tests for collect log and composition metrics**

Append to `tests/unit/analysis/test_audit_metrics.py`:

```python
from pathlib import Path

from dpeva.analysis.audit import composition_metrics, parse_collect_log, resolve_collect_artifacts


def test_parse_collect_log_extracts_thresholds_and_counts(tmp_path):
    log = tmp_path / "collection.log"
    log.write_text(
        "\n".join([
            "Auto QbC: [0.1642, 0.4142]",
            "Auto RND: [0.1602, 0.4102]",
            "Filter Results: Cand=18723, Acc=69284, Fail=1955",
            "Total sampled frames: 4829",
            "Export summary: sampled=561 systems/4829 frames, other=1414 systems/85133 frames.",
            "DPEVA_TAG: WORKFLOW_FINISHED",
        ])
    )
    parsed = parse_collect_log(log)
    assert parsed["qbc_lo_hi"] == (0.1642, 0.4142)
    assert parsed["rnd_lo_hi"] == (0.1602, 0.4102)
    assert parsed["filter_counts"] == (18723, 69284, 1955)
    assert parsed["sampled_total"] == (4829,)
    assert parsed["export_summary"] == (561, 4829, 1414, 85133)
    assert parsed["workflow_finished"] is True


def test_resolve_collect_artifacts_uses_default_paths(tmp_path):
    collect = tmp_path / "collect"
    (collect / "dataframe").mkdir(parents=True)
    (collect / "dpdata" / "sampled_dpdata").mkdir(parents=True)
    (collect / "dataframe" / "final_df.csv").write_text("dataname\nsys-0\n")
    (collect / "collection.log").write_text("")
    paths = resolve_collect_artifacts(collect)
    assert paths.final_df_path == collect / "dataframe" / "final_df.csv"
    assert paths.collection_log_path == collect / "collection.log"
    assert paths.sampled_dataset_dir == collect / "dpdata" / "sampled_dpdata"


def test_composition_metrics_compare_formulas():
    target = [{"C": 1, "O": 1}, {"C": 2}, {"Fe": 1}]
    reference = [{"C": 1, "O": 1}, {"C": 1}]
    metrics = composition_metrics(target, reference)
    assert metrics["unique_formulas"] == 3
    assert metrics["frames_unseen_formula_vs_reference"] == 2
    assert metrics["frac_unseen_formula_vs_reference"] == 2 / 3
    assert metrics["composition_l1_to_reference_median"] >= 0.0
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
pytest tests/unit/analysis/test_audit_metrics.py -q
```

Expected: failure because `resolve_collect_artifacts` and `composition_metrics` do not exist.

- [ ] **Step 3: Implement artifact and composition helpers**

Add to `src/dpeva/analysis/audit.py`:

```python
from collections import Counter


@dataclass(frozen=True)
class CollectArtifacts:
    final_df_path: Path
    collection_log_path: Path
    sampled_dataset_dir: Path


def resolve_collect_artifacts(collect_dir: Path) -> CollectArtifacts:
    collect_dir = Path(collect_dir)
    return CollectArtifacts(
        final_df_path=collect_dir / "dataframe" / "final_df.csv",
        collection_log_path=collect_dir / "collection.log",
        sampled_dataset_dir=collect_dir / "dpdata" / "sampled_dpdata",
    )


def _composition_vector(counts: Mapping[str, int], elements: list[str]) -> np.ndarray:
    values = np.asarray([counts.get(element, 0) for element in elements], dtype=float)
    total = float(values.sum())
    if total <= 0:
        return values
    return values / total


def _formula_key(counts: Mapping[str, int], elements: list[str]) -> tuple[int, ...]:
    return tuple(int(counts.get(element, 0)) for element in elements)


def composition_metrics(
    target_counts: list[Mapping[str, int]],
    reference_counts: list[Mapping[str, int]] | None = None,
) -> dict[str, float | int]:
    elements = sorted({element for row in target_counts for element in row})
    if reference_counts:
        elements = sorted(set(elements) | {element for row in reference_counts for element in row})
    target_formulas = [_formula_key(row, elements) for row in target_counts]
    unique_formulas = set(target_formulas)
    metrics: dict[str, float | int] = {
        "unique_formulas": len(unique_formulas),
    }
    atom_counts = np.asarray([sum(row.values()) for row in target_counts], dtype=float)
    if len(atom_counts):
        metrics["natoms_median"] = float(np.median(atom_counts))
        metrics["natoms_p90"] = float(np.percentile(atom_counts, 90))
    if reference_counts:
        reference_formulas = {_formula_key(row, elements) for row in reference_counts}
        unseen = sum(1 for formula in target_formulas if formula not in reference_formulas)
        target_vectors = np.asarray([_composition_vector(row, elements) for row in target_counts])
        reference_vectors = np.asarray([_composition_vector(row, elements) for row in reference_counts])
        nn = NearestNeighbors(n_neighbors=1, metric="manhattan")
        nn.fit(reference_vectors)
        distances = nn.kneighbors(target_vectors, return_distance=True)[0][:, 0]
        metrics.update({
            "frames_unseen_formula_vs_reference": unseen,
            "frac_unseen_formula_vs_reference": unseen / len(target_counts) if target_counts else 0.0,
            "composition_l1_to_reference_median": float(np.median(distances)),
            "composition_l1_to_reference_mean": float(np.mean(distances)),
        })
    return metrics
```

- [ ] **Step 4: Run tests**

Run:

```bash
pytest tests/unit/analysis/test_audit_metrics.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dpeva/analysis/audit.py tests/unit/analysis/test_audit_metrics.py
git commit -m "feat: add audit composition and collect helpers"
```

---

### Task 4: Audit Manager and Output Files

**Files:**
- Create: `src/dpeva/analysis/audit_manager.py`
- Create: `tests/unit/analysis/test_audit_manager.py`
- Modify: `src/dpeva/constants.py`
- Modify: `src/dpeva/analysis/__init__.py`

- [ ] **Step 1: Add output filename constants**

In `src/dpeva/constants.py`, add near dataset analysis filenames:

```python
FILENAME_AUDIT_METRICS_CSV: Final[str] = "audit_metrics.csv"
FILENAME_AUDIT_METRICS_JSON: Final[str] = "audit_metrics.json"
FILENAME_AUDIT_POOL_COUNTS_CSV: Final[str] = "audit_pool_counts.csv"
FILENAME_AUDIT_OVERLAP_CSV: Final[str] = "audit_overlap.csv"
FILENAME_AUDIT_REPORT_MD: Final[str] = "audit_report.md"
```

- [ ] **Step 2: Write failing manager test**

Create `tests/unit/analysis/test_audit_manager.py`:

```python
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dpeva.analysis.audit_manager import AuditAnalysisManager


def _target(name, collect_dir):
    return SimpleNamespace(
        name=name,
        collect_dir=collect_dir,
        dataset_dir=None,
        final_df_path=None,
        desc_dir=None,
        desc_feature_kind="descriptor",
        hdf5_dataset=None,
    )


def test_audit_manager_writes_core_outputs(tmp_path):
    collect = tmp_path / "collect"
    (collect / "dataframe").mkdir(parents=True)
    (collect / "dpdata" / "sampled_dpdata").mkdir(parents=True)
    df = pd.DataFrame({
        "dataname": ["PoolA/sys-0", "PoolA/sys-1", "PoolB/sys-0"],
        "desc_stru_0": [0.0, 1.0, 0.0],
        "desc_stru_1": [0.0, 0.0, 1.0],
    })
    df.to_csv(collect / "dataframe" / "final_df.csv", index=False)
    (collect / "collection.log").write_text("Filter Results: Cand=3, Acc=1, Fail=0\nDPEVA_TAG: WORKFLOW_FINISHED\n")

    manager = AuditAnalysisManager()
    output_dir = tmp_path / "audit"
    manager.analyze(
        output_dir=output_dir,
        reference=None,
        targets=[_target("run_a", collect)],
        metrics=SimpleNamespace(
            basic=True,
            descriptor_nn=False,
            composition=False,
            overlap=True,
            entropy=False,
            advanced_entropy=False,
            entropy_ridge=1e-6,
            entropy_pca_dim=None,
            entropy_bandwidth=None,
        ),
    )

    metrics = pd.read_csv(output_dir / "audit_metrics.csv")
    assert metrics.loc[0, "run"] == "run_a"
    assert metrics.loc[0, "selected_frames"] == 3
    assert metrics.loc[0, "pool_entropy"] > 0.0
    assert (output_dir / "audit_metrics.json").exists()
    assert (output_dir / "audit_pool_counts.csv").exists()
    assert (output_dir / "audit_report.md").exists()
    payload = json.loads((output_dir / "audit_metrics.json").read_text())
    assert payload["runs"][0]["run"] == "run_a"
```

- [ ] **Step 3: Run manager test and confirm failure**

Run:

```bash
pytest tests/unit/analysis/test_audit_manager.py -q
```

Expected: import failure because `AuditAnalysisManager` does not exist.

- [ ] **Step 4: Implement `AuditAnalysisManager`**

Create `src/dpeva/analysis/audit_manager.py` with an implementation that:

- Creates `output_dir`.
- Resolves `collect_dir` defaults.
- Loads `final_df.csv`.
- Computes basic metrics through `pool_summary()`.
- Parses `collection.log` when it exists.
- Optionally computes descriptor NN when both target and reference descriptors are available.
- Optionally computes overlap when at least two targets exist.
- Writes all output files listed in constants.

Use this complete initial structure:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dpeva.analysis.audit import pairwise_overlap, parse_collect_log, pool_summary, resolve_collect_artifacts
from dpeva.constants import (
    FILENAME_AUDIT_METRICS_CSV,
    FILENAME_AUDIT_METRICS_JSON,
    FILENAME_AUDIT_OVERLAP_CSV,
    FILENAME_AUDIT_POOL_COUNTS_CSV,
    FILENAME_AUDIT_REPORT_MD,
)


class AuditAnalysisManager:
    """Run post-collect and dataset audit diagnostics."""

    def analyze(self, output_dir, reference, targets, metrics) -> dict[str, object]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        pool_rows: list[dict[str, object]] = []
        final_dfs: dict[str, pd.DataFrame] = {}

        for target in targets:
            name = str(target.name)
            df = self._load_target_dataframe(target)
            final_dfs[name] = df
            collect_log_path = self._collect_log_path(target)
            row = self._record_for_target(name, df, collect_log_path)
            rows.append(row)
            pool = pool_summary(df)
            pool_rows.append({"run": name, **pool.counts})

        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(output_path / FILENAME_AUDIT_METRICS_CSV, index=False)
        pd.DataFrame(pool_rows).fillna(0).to_csv(output_path / FILENAME_AUDIT_POOL_COUNTS_CSV, index=False)

        overlap_df = None
        if getattr(metrics, "overlap", True) and len(final_dfs) >= 2:
            overlap_df = pairwise_overlap(final_dfs)
            overlap_df.to_csv(output_path / FILENAME_AUDIT_OVERLAP_CSV, index=False)

        payload = {
            "runs": metrics_df.to_dict(orient="records"),
            "overlap": [] if overlap_df is None else overlap_df.to_dict(orient="records"),
        }
        (output_path / FILENAME_AUDIT_METRICS_JSON).write_text(json.dumps(payload, indent=2, default=str))
        self._write_report(output_path, metrics_df, overlap_df)
        return payload

    def _collect_log_path(self, target) -> Path | None:
        collect_dir = getattr(target, "collect_dir", None)
        if collect_dir is None:
            return None
        path = resolve_collect_artifacts(Path(collect_dir)).collection_log_path
        return path if path.exists() else None

    def _load_target_dataframe(self, target) -> pd.DataFrame:
        final_df_path = getattr(target, "final_df_path", None)
        if final_df_path is None and getattr(target, "collect_dir", None) is not None:
            final_df_path = resolve_collect_artifacts(Path(target.collect_dir)).final_df_path
        if final_df_path is None:
            raise ValueError(f"Target '{target.name}' requires final_df_path or collect_dir for the first audit manager implementation")
        final_df_path = Path(final_df_path)
        if not final_df_path.exists():
            raise FileNotFoundError(f"Target '{target.name}' final_df not found: {final_df_path}")
        return pd.read_csv(final_df_path)

    def _record_for_target(self, name: str, df: pd.DataFrame, collect_log_path: Path | None) -> dict[str, object]:
        pool = pool_summary(df)
        record: dict[str, object] = {
            "run": name,
            "selected_frames": int(len(df)),
            "unique_datanames": int(df["dataname"].nunique()),
            "unique_systems": int(df["dataname"].str.rsplit("-", n=1).str[0].nunique()),
            "pool_entropy": pool.entropy,
            "n_pools": pool.n_pools,
        }
        if collect_log_path is not None:
            parsed = parse_collect_log(collect_log_path)
            if "filter_counts" in parsed:
                cand, acc, fail = parsed["filter_counts"]
                record.update({"candidate_frames": cand, "accurate_frames": acc, "failed_frames": fail})
            record["workflow_finished"] = bool(parsed.get("workflow_finished", False))
        return record

    def _write_report(self, output_dir: Path, metrics_df: pd.DataFrame, overlap_df: pd.DataFrame | None) -> None:
        lines = [
            "# DP-EVA Audit Report",
            "",
            "## Metrics",
            "",
            _markdown_table(metrics_df),
        ]
        if overlap_df is not None:
            lines.extend(["", "## Pairwise Overlap", "", _markdown_table(overlap_df)])
        (output_dir / FILENAME_AUDIT_REPORT_MD).write_text("\n".join(lines) + "\n")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    out = df.fillna("")
    headers = [str(col) for col in out.columns]
    rows = [[str(value) for value in row] for row in out.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    return "\n".join([fmt(headers), sep, *(fmt(row) for row in rows)])
```

The first implementation may compute descriptor metrics from `desc_stru_*` columns in `final_df.csv`; loading detached descriptor directories can be added in Task 5.

- [ ] **Step 5: Export manager**

Add to `src/dpeva/analysis/__init__.py`:

```python
from dpeva.analysis.audit_manager import AuditAnalysisManager
```

- [ ] **Step 6: Run manager tests**

Run:

```bash
pytest tests/unit/analysis/test_audit_manager.py tests/unit/analysis/test_audit_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/dpeva/constants.py src/dpeva/analysis/audit_manager.py src/dpeva/analysis/__init__.py tests/unit/analysis/test_audit_manager.py
git commit -m "feat: add audit analysis manager outputs"
```

---

### Task 5: Workflow Routing and Slurm Compatibility

**Files:**
- Modify: `src/dpeva/workflows/analysis.py`
- Modify: `tests/unit/workflows/test_analysis_workflow.py`

- [ ] **Step 1: Write failing workflow routing test**

Add to `tests/unit/workflows/test_analysis_workflow.py`:

```python
from unittest.mock import patch

from dpeva.workflows.analysis import AnalysisWorkflow


def test_analysis_workflow_routes_audit_mode(tmp_path):
    collect = tmp_path / "collect"
    collect.mkdir()
    config = {
        "mode": "audit",
        "output_dir": str(tmp_path / "audit"),
        "audit_targets": [{"name": "sampled", "collect_dir": str(collect)}],
    }

    with patch("dpeva.workflows.analysis.AuditAnalysisManager") as manager_cls:
        workflow = AnalysisWorkflow(config)
        workflow.run()

    manager_cls.return_value.analyze.assert_called_once()
```

- [ ] **Step 2: Run workflow test and confirm failure**

Run:

```bash
pytest tests/unit/workflows/test_analysis_workflow.py::test_analysis_workflow_routes_audit_mode -q
```

Expected: failure because workflow does not route audit mode.

- [ ] **Step 3: Wire manager into workflow**

In `src/dpeva/workflows/analysis.py`, import:

```python
from dpeva.analysis.audit_manager import AuditAnalysisManager
```

In `AnalysisWorkflow.__init__`, add:

```python
self.audit_analysis_manager = AuditAnalysisManager()
```

In `run()`, route:

```python
if self.config.mode == "dataset":
    self._run_dataset_mode()
elif self.config.mode == "audit":
    self._run_audit_mode()
else:
    self._run_model_mode(output_dir)
```

Add:

```python
def _run_audit_mode(self):
    """Run dataset and collect audit analysis."""
    self.logger.info("Running audit analysis mode")
    self.audit_analysis_manager.analyze(
        output_dir=self.config.output_dir,
        reference=self.config.audit_reference,
        targets=self.config.audit_targets,
        metrics=self.config.audit_metrics,
    )
```

- [ ] **Step 4: Run workflow tests**

Run:

```bash
pytest tests/unit/workflows/test_analysis_workflow.py -q
```

Expected: all selected workflow tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dpeva/workflows/analysis.py tests/unit/workflows/test_analysis_workflow.py
git commit -m "feat: route audit mode through analysis workflow"
```

---

### Task 6: Descriptor Directory Loading and Reference Novelty

**Files:**
- Modify: `src/dpeva/analysis/audit_manager.py`
- Modify: `tests/unit/analysis/test_audit_manager.py`

- [ ] **Step 1: Add failing reference novelty test**

Append to `tests/unit/analysis/test_audit_manager.py`:

```python
def test_audit_manager_computes_descriptor_nn_against_reference(tmp_path):
    ref_collect = tmp_path / "ref"
    target_collect = tmp_path / "target"
    for path in [ref_collect, target_collect]:
        (path / "dataframe").mkdir(parents=True)
        (path / "dpdata" / "sampled_dpdata").mkdir(parents=True)
        (path / "collection.log").write_text("")
    pd.DataFrame({
        "dataname": ["Pool/sys-0"],
        "desc_stru_0": [0.0],
        "desc_stru_1": [0.0],
    }).to_csv(ref_collect / "dataframe" / "final_df.csv", index=False)
    pd.DataFrame({
        "dataname": ["Pool/sys-0", "Pool/sys-1"],
        "desc_stru_0": [0.0, 2.0],
        "desc_stru_1": [0.0, 0.0],
    }).to_csv(target_collect / "dataframe" / "final_df.csv", index=False)

    manager = AuditAnalysisManager()
    output_dir = tmp_path / "audit"
    manager.analyze(
        output_dir=output_dir,
        reference=_target("train", ref_collect),
        targets=[_target("sampled", target_collect)],
        metrics=SimpleNamespace(
            basic=True,
            descriptor_nn=True,
            composition=False,
            overlap=False,
            entropy=False,
            advanced_entropy=False,
            entropy_ridge=1e-6,
            entropy_pca_dim=None,
            entropy_bandwidth=None,
        ),
    )
    metrics = pd.read_csv(output_dir / "audit_metrics.csv")
    assert metrics.loc[0, "nn_to_reference_median"] == 1.0
    assert metrics.loc[0, "selected_internal_nn_median"] == 2.0
```

- [ ] **Step 2: Run test and confirm failure if descriptor metrics are missing**

Run:

```bash
pytest tests/unit/analysis/test_audit_manager.py::test_audit_manager_computes_descriptor_nn_against_reference -q
```

Expected: failure until descriptor NN metrics are implemented in the manager.

- [ ] **Step 3: Implement descriptor metric integration**

In `AuditAnalysisManager`, add:

```python
def _descriptor_array_from_dataframe(self, df: pd.DataFrame) -> np.ndarray | None:
    cols = descriptor_columns(df)
    if not cols:
        return None
    return df[cols].to_numpy(dtype=float)
```

In `analyze()`, load reference descriptors from the reference dataframe when `reference` is provided. For each target, add:

```python
target_desc = self._descriptor_array_from_dataframe(df)
if metrics.descriptor_nn and target_desc is not None:
    row.update(internal_nearest_neighbor_summary(target_desc, "selected_internal_nn"))
    if reference_desc is not None:
        row.update(nearest_neighbor_summary(target_desc, reference_desc, "nn_to_reference"))
```

If `metrics.descriptor_nn` is true and no target descriptors exist, raise:

```python
raise ValueError(f"Descriptor metrics requested but no desc_stru_* columns found for target '{name}'")
```

- [ ] **Step 4: Run manager tests**

Run:

```bash
pytest tests/unit/analysis/test_audit_manager.py -q
```

Expected: all manager tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dpeva/analysis/audit_manager.py tests/unit/analysis/test_audit_manager.py
git commit -m "feat: add audit descriptor novelty metrics"
```

---

### Task 7: Entropy Metrics

**Files:**
- Modify: `src/dpeva/analysis/audit_manager.py`
- Modify: `tests/unit/analysis/test_audit_manager.py`

- [ ] **Step 1: Add failing entropy test**

Append to `tests/unit/analysis/test_audit_manager.py`:

```python
def test_audit_manager_reports_logdet_entropy_and_gain(tmp_path):
    ref_collect = tmp_path / "ref_entropy"
    target_collect = tmp_path / "target_entropy"
    for path in [ref_collect, target_collect]:
        (path / "dataframe").mkdir(parents=True)
        (path / "dpdata" / "sampled_dpdata").mkdir(parents=True)
        (path / "collection.log").write_text("")
    pd.DataFrame({
        "dataname": ["Pool/sys-0", "Pool/sys-1"],
        "desc_stru_0": [0.0, 1.0],
        "desc_stru_1": [0.0, 0.0],
    }).to_csv(ref_collect / "dataframe" / "final_df.csv", index=False)
    pd.DataFrame({
        "dataname": ["Pool/sys-2", "Pool/sys-3"],
        "desc_stru_0": [0.0, 1.0],
        "desc_stru_1": [1.0, 1.0],
    }).to_csv(target_collect / "dataframe" / "final_df.csv", index=False)

    manager = AuditAnalysisManager()
    manager.analyze(
        output_dir=tmp_path / "audit_entropy",
        reference=_target("train", ref_collect),
        targets=[_target("sampled", target_collect)],
        metrics=SimpleNamespace(
            basic=True,
            descriptor_nn=True,
            composition=False,
            overlap=False,
            entropy=True,
            advanced_entropy=False,
            entropy_ridge=1e-6,
            entropy_pca_dim=None,
            entropy_bandwidth=None,
        ),
    )
    metrics = pd.read_csv(tmp_path / "audit_entropy" / "audit_metrics.csv")
    assert np.isfinite(metrics.loc[0, "logdet_entropy"])
    assert np.isfinite(metrics.loc[0, "logdet_entropy_gain_vs_reference"])
```

- [ ] **Step 2: Run entropy test and confirm failure**

Run:

```bash
pytest tests/unit/analysis/test_audit_manager.py::test_audit_manager_reports_logdet_entropy_and_gain -q
```

Expected: failure until manager writes entropy columns.

- [ ] **Step 3: Implement entropy integration**

In `AuditAnalysisManager`, when `metrics.entropy` is true and target descriptors exist:

```python
row["logdet_entropy"] = logdet_entropy(target_desc, ridge=metrics.entropy_ridge)
if reference_desc is not None:
    combined = np.vstack([reference_desc, target_desc])
    row["logdet_entropy_gain_vs_reference"] = (
        logdet_entropy(combined, ridge=metrics.entropy_ridge)
        - logdet_entropy(reference_desc, ridge=metrics.entropy_ridge)
    )
```

If `metrics.advanced_entropy` is true, fail explicitly:

```python
raise NotImplementedError(
    "advanced_entropy requires QUESTS-style KDE entropy, which is not implemented in this MVP"
)
```

- [ ] **Step 4: Run entropy and metric tests**

Run:

```bash
pytest tests/unit/analysis/test_audit_manager.py tests/unit/analysis/test_audit_metrics.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/dpeva/analysis/audit_manager.py tests/unit/analysis/test_audit_manager.py
git commit -m "feat: add audit logdet entropy metrics"
```

---

### Task 8: Recipe and Documentation

**Files:**
- Create: `examples/recipes/analysis/config_audit.json`
- Modify: `examples/recipes/README.md`
- Modify: `examples/recipes/analysis/README.md`
- Modify: `docs/guides/cli.md`
- Modify: `docs/guides/configuration.md`
- Modify: `docs/source/api/config/analysis.rst`

- [ ] **Step 1: Add recipe**

Create `examples/recipes/analysis/config_audit.json`:

```json
{
  "mode": "audit",
  "output_dir": "audit",
  "audit_reference": {
    "name": "training",
    "dataset_dir": "./sampled_dpdata",
    "desc_dir": "./desc_train"
  },
  "audit_targets": [
    {
      "name": "collect_run",
      "collect_dir": "./dpeva_uq_result"
    }
  ],
  "audit_metrics": {
    "basic": true,
    "descriptor_nn": true,
    "composition": true,
    "overlap": true,
    "entropy": false,
    "advanced_entropy": false
  }
}
```

- [ ] **Step 2: Update recipe README**

In `examples/recipes/README.md`, add under Analysis:

```markdown
- **Dataset/Collect Audit (`analysis/config_audit.json`)**
  - Inputs: `mode="audit"`, optional `audit_reference`, and one or more `audit_targets`.
  - Outputs: `audit_metrics.csv`, `audit_pool_counts.csv`, `audit_overlap.csv`, `audit_metrics.json`, and `audit_report.md`.
  - Run with either `dpeva audit examples/recipes/analysis/config_audit.json` or `dpeva analysis examples/recipes/analysis/config_audit.json`.
```

- [ ] **Step 3: Update CLI guide**

In `docs/guides/cli.md`, add a subsection after analysis:

```markdown
### audit（训练集/采样集诊断）

`audit` 是 `analysis mode="audit"` 的短入口，用于对训练集、验证集、collect 输出或多个 collect 结果进行 post-collect 诊断。

```bash
dpeva audit examples/recipes/analysis/config_audit.json
```

典型输出包括 `audit_metrics.csv`、`audit_pool_counts.csv`、`audit_overlap.csv`、`audit_metrics.json` 与 `audit_report.md`。默认指标覆盖 pool balance、descriptor nearest-neighbor novelty、内部冗余、composition coverage、collect UQ/filter 摘要和 exact frame overlap。`audit_metrics.entropy=true` 时额外输出 descriptor covariance log-det entropy 和相对 reference 的 entropy gain。
```

- [ ] **Step 4: Update configuration guide**

In `docs/guides/configuration.md`, add a JSON example matching the recipe and describe:

```markdown
- `audit_reference`: 可选参考数据集，通常为已有训练集。
- `audit_targets`: 待诊断对象列表。每个对象至少提供 `dataset_dir`、`collect_dir`、`final_df_path` 或 `desc_dir`。
- `audit_metrics.basic`: 基础帧数、system 数、pool counts、pool entropy、collect log 摘要。
- `audit_metrics.descriptor_nn`: descriptor 最近邻 novelty 和内部冗余。
- `audit_metrics.composition`: 组分 formula 与 composition L1 coverage。
- `audit_metrics.overlap`: 多个 target 间基于 `dataname` 的 exact overlap。
- `audit_metrics.entropy`: descriptor covariance log-det entropy。
- `audit_metrics.advanced_entropy`: QUESTS-style KDE entropy 预留开关，MVP 中显式报错。
```

- [ ] **Step 5: Run doc-adjacent smoke checks**

Run:

```bash
python -m json.tool examples/recipes/analysis/config_audit.json >/dev/null
pytest tests/unit/test_cli.py tests/unit/analysis/test_audit_manager.py tests/unit/analysis/test_audit_metrics.py -q
```

Expected: JSON validates and selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add examples/recipes/analysis/config_audit.json examples/recipes/README.md examples/recipes/analysis/README.md docs/guides/cli.md docs/guides/configuration.md docs/source/api/config/analysis.rst
git commit -m "docs: add audit analysis recipe and guides"
```

---

### Task 9: Final Verification

**Files:**
- All files changed by Tasks 1-8.

- [ ] **Step 1: Run focused tests**

Run:

```bash
pytest tests/unit/analysis/test_audit_metrics.py tests/unit/analysis/test_audit_manager.py tests/unit/workflows/test_analysis_workflow.py tests/unit/test_cli.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run config and docs smoke checks**

Run:

```bash
python -m json.tool examples/recipes/analysis/config_audit.json >/dev/null
python -m py_compile src/dpeva/analysis/audit.py src/dpeva/analysis/audit_manager.py src/dpeva/config.py src/dpeva/cli.py src/dpeva/workflows/analysis.py
```

Expected: both commands exit 0.

- [ ] **Step 3: Run an end-to-end tiny audit command**

Create a temporary fixture under `tmp_path` in a pytest test or run manually with a local scratch directory. The fixture must include one `collect/dataframe/final_df.csv` with `dataname`, `desc_stru_0`, and `desc_stru_1`, plus an empty `collect/collection.log`.

Run:

```bash
dpeva --no-banner audit /path/to/tiny_audit.json
```

Expected:

- `audit_metrics.csv` exists.
- `audit_metrics.json` exists.
- `audit_pool_counts.csv` exists.
- `audit_report.md` exists.
- The log contains `DPEVA_TAG: WORKFLOW_FINISHED`.

- [ ] **Step 4: Verify git diff scope**

Run:

```bash
git diff --stat
git diff -- src/dpeva/analysis/audit.py src/dpeva/analysis/audit_manager.py src/dpeva/config.py src/dpeva/cli.py src/dpeva/workflows/analysis.py
```

Expected: changes are limited to audit analysis implementation, tests, recipes, and docs.

- [ ] **Step 5: Final commit if any changes remain uncommitted**

Run:

```bash
git status --short
```

Expected: only unrelated pre-existing files remain dirty. If audit files remain dirty, commit them with:

```bash
git add src/dpeva/analysis/audit.py src/dpeva/analysis/audit_manager.py src/dpeva/analysis/__init__.py src/dpeva/config.py src/dpeva/cli.py src/dpeva/workflows/analysis.py src/dpeva/constants.py tests/unit/analysis/test_audit_metrics.py tests/unit/analysis/test_audit_manager.py tests/unit/workflows/test_analysis_workflow.py tests/unit/test_cli.py examples/recipes/analysis/config_audit.json examples/recipes/README.md examples/recipes/analysis/README.md docs/guides/cli.md docs/guides/configuration.md docs/source/api/config/analysis.rst
git commit -m "feat: add dataset audit analysis workflow"
```

## Plan Self-Review

Spec coverage:

- CLI alias is covered in Task 1.
- `analysis mode="audit"` is covered in Tasks 1 and 5.
- Basic, descriptor, composition, overlap, and log-det entropy metrics are covered in Tasks 2, 3, 6, and 7.
- Output files and Markdown report are covered in Task 4.
- Recipes and docs are covered in Task 8.
- Verification is covered in Task 9.

Unresolved-slot scan:

- The plan contains no unresolved implementation slots.
- Advanced QUESTS-style KDE entropy is intentionally non-MVP and has an explicit `NotImplementedError` behavior.

Type consistency:

- Config names are consistent across `AuditDatasetConfig`, `AuditMetricsConfig`, `AnalysisConfig`, recipe JSON, and manager method signatures.
- Metric prefix names are consistent across tests and implementation steps: `nn_to_reference`, `selected_internal_nn`, `logdet_entropy`, and `logdet_entropy_gain_vs_reference`.
