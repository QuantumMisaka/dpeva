# Integration Data Fixtures

This directory keeps small integration-test datasets in Git.

Large DeePMD pretrained models are external fixtures and are intentionally not
tracked in this repository. Put them in this directory for local/HPC validation,
or point tests to an external cache with environment variables:

- `DPEVA_DPA3_MODEL_PATH`: path to `DPA-3.1-3M.pt`
- `DPEVA_DPA4_MODEL_PATH`: path to `DPA4-Mini-OMat24.pt`

The real DeePMD LLPR/DPOSE validation is opt-in and GPU-node only:

```bash
export DPEVA_RUN_DEEPMD_DPOSE_REAL=1
pytest tests/integration/test_real_deepmd_llpr_dpose.py -q
```

On SAI, prefer the Slurm wrapper:

```bash
sbatch scripts/slurm/run_dpeva_dpa4_llpr_dpose_gpu_test.sh
```
