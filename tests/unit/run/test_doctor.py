import importlib
import json
import subprocess

import dpeva
from packaging.version import Version
from dpeva.run.doctor import DoctorCheck, DoctorReport, build_doctor_report, probe_deepmd


def test_import_dpeva_does_not_probe_external_commands(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("subprocess called")),
    )

    importlib.reload(dpeva)


def test_probe_deepmd_available_is_structured() -> None:
    def available(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeePMD-kit v3.2.0\n", "")

    check = probe_deepmd(run=available)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "ok",
        "version": "3.2.0",
        "detail": "required >= 3.2.0, < 3.3",
    }


def test_probe_deepmd_missing_is_structured() -> None:
    def missing(*args, **kwargs):
        raise FileNotFoundError("dp")

    check = probe_deepmd(run=missing)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "missing",
        "version": None,
        "detail": "dp executable not found",
    }


def test_probe_deepmd_unparsable_is_structured() -> None:
    def unparsable(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeepMD-kit unknown", "")

    check = probe_deepmd(run=unparsable)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "unknown",
        "version": None,
        "detail": "unparsed version: DeepMD-kit unknown",
    }


def test_probe_deepmd_incompatible_is_structured() -> None:
    def incompatible(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeePMD-kit v3.3.0", "")

    check = probe_deepmd(run=incompatible)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "incompatible",
        "version": "3.3.0",
        "detail": "required >= 3.2.0, < 3.3",
    }


def test_probe_deepmd_nonzero_exit_is_structured_error() -> None:
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, "", "dp: failed to query version")

    check = probe_deepmd(run=failed)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "error",
        "version": None,
        "detail": "dp: failed to query version",
    }


def test_probe_deepmd_below_minimum_is_structured_incompatible() -> None:
    def below_minimum(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeePMD-kit v3.1.9", "")

    check = probe_deepmd(run=below_minimum)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "incompatible",
        "version": "3.1.9",
        "detail": "required >= 3.2.0, < 3.3",
    }


def test_probe_deepmd_accepts_dev_source_release() -> None:
    source_version = "0.1.dev1+g27a18b604"
    assert Version(source_version).is_devrelease

    def source_build(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 0, f"DeePMD-kit v{source_version}", ""
        )

    check = probe_deepmd(run=source_build)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "ok",
        "version": source_version,
        "detail": "required >= 3.2.0, < 3.3",
    }


def test_doctor_report_is_json_serializable() -> None:
    report = build_doctor_report(checks=[])

    assert json.loads(report.model_dump_json()) == {
        "schema_version": "1.0",
        "status": "ok",
        "checks": [],
    }


def test_doctor_report_status_fails_for_non_ok_check() -> None:
    check = DoctorCheck(name="deepmd", status="missing", detail="dp executable not found")

    report = build_doctor_report(checks=[check])

    assert isinstance(report, DoctorReport)
    assert report.status == "failed"
