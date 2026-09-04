"""Versioned dataset lineage records and count validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, model_validator


class DatasetModel(BaseModel):
    """Base model for dataset evidence; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class DatasetParent(DatasetModel):
    """A dataset contributing frames to a derived dataset."""

    dataset_id: str = Field(min_length=1)
    frame_count: StrictInt = Field(ge=0)
    manifest_ref: str | None = None


class DatasetIntersectionSummary(DatasetModel):
    """Machine-readable evidence for overlap handling during integration."""

    method: Literal["not-run", "frame-identity-v1"] = "not-run"
    overlap_frame_count: StrictInt = Field(default=0, ge=0)
    removed_frame_count: StrictInt = Field(default=0, ge=0)
    evidence_ref: str | None = None


class DatasetValidationResult(DatasetModel):
    """The result and rule version of the lineage validation performed."""

    rule_version: Literal["1.0"] = "1.0"
    status: Literal["passed", "failed"]
    counts_reconciled: StrictBool
    sources_declared: StrictBool
    intersections_explained: StrictBool
    type_map_compatible: StrictBool


class DatasetManifest(DatasetModel):
    """Schema-versioned evidence for a dataset transformation."""

    schema_version: Literal["1.0"] = "1.0"
    dataset_id: str = Field(min_length=1)
    parents: list[DatasetParent]
    transformation: Literal["merge", "collect", "label", "clean", "import"]
    frame_count: StrictInt = Field(ge=0)
    removed_frame_count: StrictInt = Field(default=0, ge=0)
    system_count: StrictInt = Field(ge=0)
    type_map: list[str]
    format: str
    source_entries: list[str] = Field(default_factory=list)
    intersection_summary: DatasetIntersectionSummary = Field(
        default_factory=DatasetIntersectionSummary
    )
    content_identity: str | None = None
    content_identity_strength: Literal[
        "none", "structural", "exported-files-sha256"
    ] = "none"
    validation_result: DatasetValidationResult

    @model_validator(mode="after")
    def validate_lineage_shape(self) -> "DatasetManifest":
        if not self.parents and self.transformation != "import":
            raise ValueError("parents must be provided for non-import transformations")

        parent_ids = [parent.dataset_id for parent in self.parents]
        if len(parent_ids) != len(set(parent_ids)):
            raise ValueError("duplicate parent dataset_id references are not allowed")

        if len(self.type_map) != len(set(self.type_map)):
            raise ValueError("type_map entries must be unique")

        return self


class LineageValidationError(ValueError):
    """Raised when observed output counts do not reconcile with lineage."""


def validate_lineage_counts(manifest: DatasetManifest) -> None:
    """Validate the frame-count conservation rule for a dataset manifest.

    Imported roots intentionally have no parents, so their observed count is
    authoritative until a subsequent transformation records parent evidence.
    Derived datasets must account for every parent frame and every removal.
    """

    expected = (
        sum(parent.frame_count for parent in manifest.parents) - manifest.removed_frame_count
        if manifest.parents
        else manifest.frame_count
    )
    result = manifest.validation_result
    summary = manifest.intersection_summary
    sources = set(manifest.source_entries)
    source_ids = {parent.dataset_id for parent in manifest.parents}
    sources_declared = source_ids.issubset(sources) and bool(sources)
    counts_reconciled = expected == manifest.frame_count
    if summary.overlap_frame_count > 0:
        intersections_explained = (
            summary.method != "not-run"
            and bool(summary.evidence_ref)
            and summary.overlap_frame_count == summary.removed_frame_count
            and summary.removed_frame_count == manifest.removed_frame_count
        )
    else:
        intersections_explained = (
            summary.removed_frame_count == manifest.removed_frame_count
            and (
                manifest.removed_frame_count == 0
                or (summary.method != "not-run" and bool(summary.evidence_ref))
            )
        )
    if not counts_reconciled:
        raise LineageValidationError(
            f"lineage frame count mismatch: expected {expected}, observed {manifest.frame_count}"
        )
    if not sources_declared:
        raise LineageValidationError("lineage source entries do not declare every parent dataset")
    if not intersections_explained:
        raise LineageValidationError("intersection/removal evidence is missing or inconsistent")
    if result.status != "passed" or not (
        result.rule_version == "1.0"
        and result.counts_reconciled
        and result.sources_declared
        and result.intersections_explained
        and result.type_map_compatible
    ):
        raise LineageValidationError("dataset validation_result does not record a passed validation")
