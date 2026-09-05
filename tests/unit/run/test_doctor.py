import importlib
import json
import subprocess
from types import SimpleNamespace

import dpeva
from packaging.version import Version
from dpeva.run.doctor import (
    DoctorCheck,
    DoctorReport,
    build_doctor_report,
    probe_deepmd,
)


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
        "detail": "runtime envelope >= 3.1.2, < 3.3",
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
        "detail": "runtime envelope >= 3.1.2, < 3.3",
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


def test_probe_deepmd_accepts_retained_legacy_runtime_envelope() -> None:
    def below_minimum(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeePMD-kit v3.1.2", "")

    check = probe_deepmd(run=below_minimum)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "ok",
        "version": "3.1.2",
        "detail": "runtime envelope >= 3.1.2, < 3.3",
    }


def test_probe_deepmd_rejects_legacy_runtime_below_lower_bound() -> None:
    def old_release(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeePMD-kit v3.1.1", "")

    check = probe_deepmd(run=old_release)

    assert check.status == "incompatible"
    assert check.version == "3.1.1"


def test_probe_deepmd_does_not_bypass_bounds_for_prerelease() -> None:
    source_version = "0.1.dev1+g27a18b604"
    assert Version(source_version).is_devrelease

    def source_build(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0], 0, f"DeePMD-kit v{source_version}", ""
        )

    check = probe_deepmd(run=source_build)

    assert check.model_dump() == {
        "name": "deepmd",
        "status": "incompatible",
        "version": source_version,
        "detail": "runtime envelope >= 3.1.2, < 3.3",
    }


def test_probe_deepmd_rejects_future_prerelease_even_below_upper_bound() -> None:
    def future_prerelease(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0, "DeePMD-kit v3.3.0.dev1", "")

    check = probe_deepmd(run=future_prerelease)

    assert check.status == "incompatible"
    assert check.version == "3.3.0.dev1"


def test_doctor_separates_legacy_runtime_from_qualified_lane(monkeypatch) -> None:
    def run(command, **kwargs):
        if command == ["dp", "--version"]:
            return subprocess.CompletedProcess(command, 0, "DeePMD-kit v3.1.2", "")
        return subprocess.CompletedProcess(command, 1, "", "unsupported operation")

    monkeypatch.setattr(
        "dpeva.run.doctor._probe_python_package",
        lambda name, *, required: DoctorCheck(name=name, status="ok", detail="ok", required=required),
    )
    report = build_doctor_report(
        run=run,
        include_optional=False,
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(),
            version=SimpleNamespace(cuda=None),
        ),
        cuda_probe=lambda _: False,
    )

    assert report.status == "ok"
    assert next(check for check in report.checks if check.name == "deepmd").status == "ok"
    qualified = next(check for check in report.checks if check.name == "deepmd.qualified")
    assert qualified.status == "skipped"
    assert qualified.required is False
    assert "3.2 qualification" in qualified.detail
    assert all(
        check.required is False
        for check in report.checks
        if check.name.startswith("deepmd.cli.")
    )


def test_doctor_reports_qualified_32_lane_separately(monkeypatch) -> None:
    def run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "DeePMD-kit v3.2.0", "")

    monkeypatch.setattr(
        "dpeva.run.doctor._probe_python_package",
        lambda name, *, required: DoctorCheck(name=name, status="ok", detail="ok", required=required),
    )
    report = build_doctor_report(
        run=run,
        include_optional=False,
        torch_module=SimpleNamespace(
            cuda=SimpleNamespace(),
            version=SimpleNamespace(cuda=None),
        ),
        cuda_probe=lambda _: False,
    )

    qualified = next(check for check in report.checks if check.name == "deepmd.qualified")
    assert qualified.status == "ok"
    assert qualified.required is False
    assert "3.2" in qualified.detail


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
