import warnings

from dpeva.run.doctor import DoctorCheck, probe_deepmd


def check_deepmd_version() -> DoctorCheck:
    """Probe DeepMD explicitly (deprecated; use :func:`probe_deepmd`)."""
    warnings.warn(
        "check_deepmd_version() is deprecated; use probe_deepmd() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    result = probe_deepmd()
    if result.status != "ok":
        warnings.warn(
            f"DeepMD compatibility check is {result.status}: {result.detail}",
            UserWarning,
            stacklevel=2,
        )
    return result
