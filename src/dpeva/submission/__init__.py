__all__ = ["JobConfig", "JobManager", "normalize_slurm_job_name", "ArrayTaskSpec"]


def __getattr__(name):
    if name == "JobConfig":
        from .templates import JobConfig

        return JobConfig
    if name == "JobManager":
        from .manager import JobManager

        return JobManager
    if name == "normalize_slurm_job_name":
        from .names import normalize_slurm_job_name

        return normalize_slurm_job_name
    if name == "ArrayTaskSpec":
        from .array import ArrayTaskSpec

        return ArrayTaskSpec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
