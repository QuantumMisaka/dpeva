"""Contract tests for the packaged DeepMD capability declaration."""

from __future__ import annotations

import json
import copy
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from dpeva.compatibility.deepmd import (
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMatrix,
    CapabilityUnavailable,
    CapabilityRecord,
    validate_promotion_evidence,
)
from dpeva.compatibility.attestation import CapabilityAttestation


def _key(**updates: str) -> CapabilityKey:
    values = {
        "operation": "eval-desc",
        "backend": "pt-expt",
        "model_family": "DPA4C",
        "artifact": "exportable",
        "data_format": "deepmd/npy",
        "environment": "cpu-periodic",
    }
    values.update(updates)
    return CapabilityKey(**values)


def test_manifest_uses_only_canonical_states() -> None:
    matrix = CapabilityMatrix.load_default()
    assert len(matrix.records) == 17
    assert {record.status for record in matrix.records} <= {
        "supported",
        "experimental",
        "unsupported",
        "blocked-upstream",
    }


def test_periodic_pt_expt_is_experimental_and_non_pbc_is_blocked() -> None:
    matrix = CapabilityMatrix.load_default()
    periodic = matrix.get(_key())
    assert periodic.status == "experimental"
    assert periodic.version_range == ">=3.2,<3.3"
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        matrix.require(_key(environment="cpu-non-pbc"))
    with pytest.raises(CapabilityUnavailable, match="blocked-upstream"):
        matrix.require(_key(operation="inference", environment="cpu-non-pbc"))


def test_experimental_requires_explicit_opt_in() -> None:
    matrix = CapabilityMatrix.load_default()
    with pytest.raises(CapabilityUnavailable, match="experimental"):
        matrix.require(_key())
    assert matrix.require(_key(), allow_experimental=True).status == "experimental"


def test_pt_expt_embed_is_unsupported() -> None:
    matrix = CapabilityMatrix.load_default()
    key = _key(operation="embed")
    with pytest.raises(CapabilityUnavailable, match="unsupported"):
        matrix.require(key, allow_experimental=True)


def test_get_requires_one_exact_key() -> None:
    matrix = CapabilityMatrix.load_default()
    with pytest.raises(CapabilityUnavailable, match="no capability record"):
        matrix.get(_key(operation="does-not-exist"))


def test_key_is_strict_and_immutable() -> None:
    with pytest.raises(ValidationError):
        CapabilityKey(
            operation=1,
            backend="pt",
            model_family="DPA4",
            artifact="checkpoint",
            data_format="deepmd/npy",
            environment="cpu",
        )
    key = _key()
    with pytest.raises(ValidationError):
        key.operation = "train"
    assert hash(key)


def test_record_and_evidence_are_immutable_and_hashable() -> None:
    record = CapabilityMatrix.load_default().get(_key())
    with pytest.raises(ValidationError):
        record.status = "supported"
    assert hash(record)
    evidence = CapabilityEvidence(cpu_contract="tests/contract/deepmd/test.py")
    with pytest.raises(ValidationError):
        evidence.cpu_contract = "tampered"
    assert hash(evidence)


def test_candidate_evaluation_declares_both_model_roles() -> None:
    record = next(
        item
        for item in CapabilityMatrix.load_default().records
        if item.key.operation == "candidate-evaluation"
    )
    assert record.covered_roles == ("regular", "ema")
    with pytest.raises(TypeError):
        record.covered_roles[0] = "ema"  # type: ignore[index]


def test_manifest_evidence_and_sai_case_mapping_is_explicit() -> None:
    from scripts.validation.collect_deepmd_32_qualification import REQUIRED_CASES

    records = CapabilityMatrix.load_default().records
    for record in records:
        if record.status == "supported":
            assert record.required_evidence
        if "sai-v100-qualification" in record.required_evidence:
            if record.verification_status == "implemented":
                assert record.sai_verification_cases
            else:
                assert record.sai_verification_cases is None
            if record.sai_verification_cases:
                assert set(record.sai_verification_cases) <= set(REQUIRED_CASES)
    candidate = next(r for r in records if r.key.operation == "candidate-evaluation")
    assert candidate.verification_status == "planned"
    assert candidate.verification_command is None


def test_attestation_schema_is_strict_and_only_success_can_be_finished() -> None:
    key = _key(operation="test", backend="pt", model_family="DPA4", artifact="checkpoint", environment="cpu")
    with pytest.raises(ValidationError):
        CapabilityAttestation(
            status="finished", returncode=1, capability_key=key,
            verification_command="pytest x::y -q", deepmd_version="DeePMD-kit v3.2.0",
            source="cpu-contract",
        )


def test_malformed_manifest_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8")
    )
    payload["records"][0]["unknown"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        CapabilityMatrix.load(path)


def test_duplicate_keys_are_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8")
    )
    payload["records"].append(payload["records"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate capability key"):
        CapabilityMatrix.load(path)


def test_resource_load_works_outside_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    matrix = CapabilityMatrix.load_default()
    assert matrix.records


def test_manifest_has_no_unapproved_status_or_issue() -> None:
    records = CapabilityMatrix.load_default().records
    non_pbc = [record for record in records if record.key.environment == "cpu-non-pbc"]
    assert {record.key.operation for record in non_pbc} == {"eval-desc", "inference"}
    assert {record.status for record in non_pbc} == {"blocked-upstream"}
    assert {
        record.upstream_issue
        for record in non_pbc
    } == {"https://github.com/deepmodeling/deepmd-kit/issues/6002"}


def test_new_routes_are_explicit_policy_unsupported_records() -> None:
    """These rows are policy declarations, not executable backend routing."""

    records = CapabilityMatrix.load_default().records
    policy = {
        ("train", "dpa-adapt", "DPA4"),
        ("train", "jax", "DPA4"),
        ("train", "tf2", "DPA4"),
    }
    observed = {
        (item.key.operation, item.key.backend, item.key.model_family): item.status
        for item in records
        if (item.key.operation, item.key.backend, item.key.model_family) in policy
    }
    assert set(observed) == policy
    assert set(observed.values()) == {"unsupported"}
    assert "supported" not in {item.status for item in records}


def test_unknown_status_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(
        Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8")
    )
    payload["records"][0]["status"] = "maybe"
    path = tmp_path / "unknown-status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError):
        CapabilityMatrix.load(path)


def test_supported_capabilities_have_complete_evidence() -> None:
    """Every supported row must pass the real promotion gate."""

    matrix = CapabilityMatrix.load_default()
    for record in matrix.records:
        if record.status != "supported":
            continue
        assert validate_promotion_evidence(record, Path.cwd())


def test_supported_mutation_cannot_promote_arbitrary_existing_json(tmp_path: Path) -> None:
    """A future status edit cannot pass merely by pointing at any JSON file."""

    payload = json.loads(Path("src/dpeva/compatibility/deepmd-3.2.json").read_text(encoding="utf-8"))
    payload["records"][0].update({
        "status": "supported",
        "verification_command": "pytest tests/contract/deepmd/test_cli_contract.py::test_pt_test_requires_numeric_output -q",
        "required_evidence": ["cpu-contract"],
        "verification_status": "implemented",
        "evidence_ref": {"cpu_contract": "src/dpeva/compatibility/deepmd-3.2.json"},
        "sai_verification_cases": None,
    })
    arbitrary = tmp_path / "arbitrary.json"
    arbitrary.write_text("{}\n", encoding="utf-8")
    payload["records"][0]["evidence_ref"] = {"cpu_contract": arbitrary.name}
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    record = CapabilityMatrix.load(tmp_path / "manifest.json").records[0]
    assert not validate_promotion_evidence(record, tmp_path)


def test_current_matrix_has_explicit_unqualified_status_distribution() -> None:
    """Do not let an empty supported loop make the qualification gate vacuous."""

    counts = Counter(record.status for record in CapabilityMatrix.load_default().records)
    assert counts == Counter(
        {
            "supported": 0,
            "experimental": 11,
            "unsupported": 4,
            "blocked-upstream": 2,
        }
    )


def test_non_pbc_capabilities_remain_blocked_by_issue_6002() -> None:
    records = [
        record
        for record in CapabilityMatrix.load_default().records
        if record.key.environment == "cpu-non-pbc"
    ]
    assert {record.key.operation for record in records} == {"eval-desc", "inference"}
    assert {record.status for record in records} == {"blocked-upstream"}
    assert {
        record.upstream_issue
        for record in records
    } == {"https://github.com/deepmodeling/deepmd-kit/issues/6002"}


def test_qualification_report_states_current_manifest_distribution() -> None:
    report = Path("docs/reports/2026-09-04-deepmd-3.2-compatibility.md").read_text(
        encoding="utf-8"
    )
    for status, count in (
        ("supported", 0),
        ("experimental", 11),
        ("unsupported", 4),
        ("blocked-upstream", 2),
    ):
        assert f"| `{status}` | {count} |" in report
    assert "#6002" in report
    assert "1126627" in report


def test_manifest_implemented_commands_map_to_collected_pytest_nodes() -> None:
    """Implemented claims must identify real, collectable contract tests."""

    for record in CapabilityMatrix.load_default().records:
        if record.verification_status != "implemented":
            assert record.verification_command is None or record.status == "supported"
            continue
        assert record.verification_command
        command = record.verification_command
        assert command.startswith("pytest ") and "::" in command
        node = command.removeprefix("pytest ").removesuffix(" -q")
        path, test_name = node.split("::", 1)
        assert Path(path).is_file()
        assert f"def {test_name}(" in Path(path).read_text(encoding="utf-8")


def test_promotion_gate_accepts_exact_cpu_machine_evidence(tmp_path: Path) -> None:
    command = "pytest tests/contract/deepmd/test.py::test_case -q"
    key = _key(operation="test", backend="pt", model_family="DPA4", artifact="checkpoint", environment="cpu")
    evidence = tmp_path / "cpu.json"
    evidence.write_text(CapabilityAttestation(
        status="finished", returncode=0, capability_key=key,
        verification_command=command, deepmd_version="DeePMD-kit v3.2.0",
        source="cpu-contract", case="test",
    ).model_dump_json(), encoding="utf-8")
    record = CapabilityRecord(
        key=key, status="supported", version_range=">=3.2,<3.3",
        verification_command=command, required_evidence=("cpu-contract",),
        verification_status="implemented",
        evidence_ref=CapabilityEvidence(cpu_contract="cpu.json"),
    )
    assert validate_promotion_evidence(record, tmp_path)


@pytest.mark.parametrize("mutation", [
    {"status": "failed"},
    {"capability_key": {"operation": "wrong"}},
    {"deepmd_version": "DeePMD-kit v3.2.0b1"},
])
def test_promotion_gate_rejects_non_exact_cpu_evidence(tmp_path: Path, mutation: dict[str, object]) -> None:
    command = "pytest tests/contract/deepmd/test.py::test_case -q"
    key = _key(operation="test", backend="pt", model_family="DPA4", artifact="checkpoint", environment="cpu")
    payload: dict[str, object] = CapabilityAttestation(
        status="finished", returncode=0, capability_key=key,
        verification_command=command, deepmd_version="DeePMD-kit v3.2.0",
        source="cpu-contract", case="test",
    ).model_dump(mode="json")
    payload.update(mutation)
    (tmp_path / "cpu.json").write_text(json.dumps(payload), encoding="utf-8")
    record = CapabilityRecord(
        key=key, status="supported", version_range=">=3.2,<3.3",
        verification_command=command, required_evidence=("cpu-contract",),
        verification_status="implemented",
        evidence_ref=CapabilityEvidence(cpu_contract="cpu.json"),
    )
    assert not validate_promotion_evidence(record, tmp_path)


def test_promotion_gate_requires_sai_job_and_gpu_evidence(tmp_path: Path) -> None:
    command = "pytest tests/contract/deepmd/test.py::test_case -q"
    key = _key(operation="eval-desc", backend="pt-expt", model_family="DPA4C", artifact="exportable", environment="sai-v100")
    cpu = CapabilityAttestation(
        status="finished", returncode=0, capability_key=key,
        verification_command=command, deepmd_version="DeePMD-kit v3.2.0",
        source="cpu-contract", case="test",
    )
    sai = CapabilityAttestation(
        status="finished", returncode=0, capability_key=key,
        verification_command=command, deepmd_version="DeePMD-kit v3.2.0",
        source="sai-v100-qualification", case="pt-test", job_id=123,
        gpu="Tesla V100",
    )
    (tmp_path / "cpu.json").write_text(cpu.model_dump_json(), encoding="utf-8")
    (tmp_path / "sai.json").write_text(sai.model_dump_json(), encoding="utf-8")
    record = CapabilityRecord(
        key=key, status="supported", version_range=">=3.2,<3.3",
        verification_command=command,
        required_evidence=("cpu-contract", "sai-v100-qualification"),
        verification_status="implemented",
        evidence_ref=CapabilityEvidence(cpu_contract="cpu.json", sai_qualification="sai.json"),
        sai_verification_cases=("pt-test",),
    )
    assert validate_promotion_evidence(record, tmp_path)
    (tmp_path / "sai.json").unlink()
    assert not validate_promotion_evidence(record, tmp_path)


def test_sai_aggregate_requires_exact_case_set_and_identity(tmp_path: Path) -> None:
    command = "pytest tests/contract/deepmd/test.py::test_case -q"
    key = _key(operation="test", backend="pt", model_family="DPA4", artifact="checkpoint", environment="cpu")
    attestations = [
        CapabilityAttestation(
            status="finished", returncode=0, capability_key=key,
            verification_command=command, deepmd_version="DeePMD-kit v3.2.0",
            source="sai-v100-qualification", case=case, job_id=123, gpu="Tesla V100",
        ).model_dump(mode="json")
        for case in ("pt-test", "pt-test-ema")
    ]
    aggregate = {
        "schema_version": "1.0", "qualification": "deepmd-3.2-sai-v100",
        "status": "finished", "job_id": "123", "gpu": "Tesla V100",
        "attestations": attestations,
    }
    (tmp_path / "sai.json").write_text(json.dumps(aggregate), encoding="utf-8")
    record = CapabilityRecord(
        key=key, status="supported", version_range=">=3.2,<3.3",
        verification_command=command,
        required_evidence=("sai-v100-qualification",), verification_status="implemented",
        evidence_ref=CapabilityEvidence(sai_qualification="sai.json"),
        sai_verification_cases=("pt-test", "pt-test-ema"),
    )
    assert validate_promotion_evidence(record, tmp_path)
    for mutation in ("missing", "duplicate", "job", "gpu"):
        changed = copy.deepcopy(aggregate)
        if mutation == "missing":
            changed["attestations"].pop()
        elif mutation == "duplicate":
            changed["attestations"][1]["case"] = "pt-test"
        elif mutation == "job":
            changed["attestations"][1]["job_id"] = "999"
        else:
            changed["attestations"][1]["gpu"] = "Tesla A100"
        (tmp_path / "sai.json").write_text(json.dumps(changed), encoding="utf-8")
        assert not validate_promotion_evidence(record, tmp_path), mutation
