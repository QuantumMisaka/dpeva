from dpeva.submission.names import normalize_slurm_job_name


def test_normalize_slurm_job_name_replaces_path_and_spaces():
    assert normalize_slurm_job_name("fp normal/N_4_0 att0") == "fp-normal-N_4_0-att0"


def test_normalize_slurm_job_name_strips_invalid_symbols():
    assert normalize_slurm_job_name("fp:normal/C64Fe38[0]") == "fp-normal-C64Fe38-0"


def test_normalize_slurm_job_name_uses_fallback_for_empty_result():
    assert normalize_slurm_job_name("///", fallback="dpeva") == "dpeva"


def test_normalize_slurm_job_name_limits_length():
    name = normalize_slurm_job_name("fp-" + "x" * 200, max_length=32)
    assert len(name) == 32
    assert name.startswith("fp-")
