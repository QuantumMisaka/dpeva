import os
import sys

# Ensure dpeva is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../../src")
sys.path.append(src_path)

from dpeva.workflows.active import ActiveWorkflow

# Parameters
project = "stage9-2"
config = {
    # Project Settings
    "project": project,
    "testing_dir": "test-val-npy",
    "testing_head": "results",
    
    # Input Data Settings
    "desc_dir": f"{project}/desc_other",
    "desc_filename": "desc.npy",
    "testdata_dir": f"{project}/other_dpdata",
    "testdata_fmt": "deepmd/npy",
    "testdata_string": "O*",
    
    # UQ Logic Settings
    "uq_select_scheme": "tangent_lo",  # strict, circle_lo, tangent_lo, crossline_lo, loose
    "uq_qbc_trust_lo": 0.12,
    "uq_qbc_trust_hi": 0.22,
    "uq_rnd_rescaled_trust_lo": 0.12,
    "uq_rnd_rescaled_trust_hi": 0.22,
    
    # Sampling Settings
    "num_selection": 100,
    "direct_k": 1,
    "direct_thr_init": 0.5,
    
    # Output Settings
    "root_savedir": "dpeva_uq_post",
    "fig_dpi": 150
}

if __name__ == "__main__":
    workflow = ActiveWorkflow(config)
    workflow.run()
