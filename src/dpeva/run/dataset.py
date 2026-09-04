"""Versioned dataset lineage records and count validation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class DatasetModel(BaseModel):
    """Base model for dataset evidence; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")


class DatasetParent(DatasetModel):
    """A dataset contributing frames to a derived dataset."""

    dataset_id: str = Field(min_length=1)
    frame_count: StrictInt = Field(ge=0)
    manifest_ref: str | None = None


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
    intersection_summary: dict[str, int] = Field(default_factory=dict)
    content_identity: str | None = None

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

    if not manifest.parents:
        return

    expected = sum(parent.frame_count for parent in manifest.parents) - manifest.removed_frame_count
    if expected != manifest.frame_count:
        raise LineageValidationError(
            f"lineage frame count mismatch: expected {expected}, observed {manifest.frame_count}"
        )
