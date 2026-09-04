import pytest
from pydantic import ValidationError

from dpeva.run.dataset import (
    DatasetManifest,
    DatasetParent,
    LineageValidationError,
    validate_lineage_counts,
)


def _manifest(**overrides: object) -> DatasetManifest:
    values: dict[str, object] = {
        "dataset_id": "ft2dp-iter11-cumulative",
        "parents": [
            DatasetParent(dataset_id="iter10-cumulative", frame_count=12105),
            DatasetParent(dataset_id="iter11-new", frame_count=4317),
        ],
        "transformation": "merge",
        "frame_count": 16422,
        "removed_frame_count": 0,
        "system_count": 2,
        "type_map": ["Fe", "C", "H", "O"],
        "format": "deepmd/npy/mixed",
    }
    values.update(overrides)
    return DatasetManifest(**values)


def test_iter11_accumulation_reconciles() -> None:
    validate_lineage_counts(_manifest())


def test_omitted_parent_frames_fail() -> None:
    manifest = _manifest(frame_count=4317)

    with pytest.raises(LineageValidationError, match="expected 16422, observed 4317"):
        validate_lineage_counts(manifest)


def test_removed_frames_reconcile_against_parent_total() -> None:
    manifest = _manifest(frame_count=16419, removed_frame_count=3)

    validate_lineage_counts(manifest)


def test_removed_frames_cannot_exceed_parent_total() -> None:
    manifest = _manifest(frame_count=0, removed_frame_count=16423)

    with pytest.raises(LineageValidationError, match="expected -1, observed 0"):
        validate_lineage_counts(manifest)


def test_empty_parent_import_is_an_explicit_root() -> None:
    manifest = _manifest(
        dataset_id="external-import",
        parents=[],
        transformation="import",
        frame_count=42,
    )

    validate_lineage_counts(manifest)


def test_empty_parents_are_only_valid_for_import() -> None:
    with pytest.raises(ValidationError, match="parents"):
        _manifest(parents=[], transformation="merge")


def test_duplicate_parent_references_are_rejected() -> None:
    parent = DatasetParent(dataset_id="same", frame_count=1)

    with pytest.raises(ValidationError, match="duplicate parent dataset_id"):
        _manifest(
            parents=[parent, parent.model_copy()],
            frame_count=2,
        )


def test_negative_counts_are_rejected_at_schema_boundary() -> None:
    with pytest.raises(ValidationError):
        DatasetParent(dataset_id="invalid", frame_count=-1)

    with pytest.raises(ValidationError):
        _manifest(removed_frame_count=-1)


@pytest.mark.parametrize(
    "field",
    [
        "parent.frame_count",
        "frame_count",
        "removed_frame_count",
        "system_count",
    ],
)
@pytest.mark.parametrize("value", ["0", 0.0, True])
def test_count_fields_require_strict_integers(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        if field == "parent.frame_count":
            DatasetParent(dataset_id="invalid", frame_count=value)
        else:
            _manifest(**{field: value})


def test_duplicate_type_map_entries_are_rejected() -> None:
    with pytest.raises(ValidationError, match="type_map entries must be unique"):
        _manifest(type_map=["Fe", "C", "Fe"])


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError, match="extra_field"):
        _manifest(extra_field="must not be silently ignored")
