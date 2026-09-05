import warnings

from dpeva.run.doctor import DoctorCheck
from dpeva.utils.env_check import check_deepmd_version


def test_check_deepmd_version_is_explicit_deprecated_wrapper(monkeypatch) -> None:
    expected = DoctorCheck(
        name="deepmd",
        status="ok",
        version="3.2.0",
        detail="runtime envelope >= 3.1.2, < 3.3",
    )
    monkeypatch.setattr("dpeva.utils.env_check.probe_deepmd", lambda: expected)

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = check_deepmd_version()

    assert result == expected
    assert len(observed) == 1
    assert issubclass(observed[0].category, DeprecationWarning)


def test_check_deepmd_version_warns_actionably_when_probe_is_not_ok(monkeypatch) -> None:
    expected = DoctorCheck(
        name="deepmd",
        status="missing",
        detail="dp executable not found",
    )
    monkeypatch.setattr("dpeva.utils.env_check.probe_deepmd", lambda: expected)

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = check_deepmd_version()

    assert result == expected
    assert any(issubclass(item.category, UserWarning) for item in observed)
    assert any("dp executable not found" in str(item.message) for item in observed)
