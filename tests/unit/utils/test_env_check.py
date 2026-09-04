import warnings

from dpeva.run.doctor import DoctorCheck
from dpeva.utils.env_check import check_deepmd_version


def test_check_deepmd_version_is_explicit_deprecated_wrapper(monkeypatch) -> None:
    expected = DoctorCheck(
        name="deepmd",
        status="ok",
        version="3.2.0",
        detail="required >= 3.2.0, < 3.3",
    )
    monkeypatch.setattr("dpeva.utils.env_check.probe_deepmd", lambda: expected)

    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        result = check_deepmd_version()

    assert result == expected
    assert len(observed) == 1
    assert issubclass(observed[0].category, DeprecationWarning)
