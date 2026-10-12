# Integration Data Fixtures

This directory keeps small integration-test datasets in Git, together with the
DPA-3.1-3M pretrained model used by the local smoke path.

`DPA-3.1-3M.pt` (47,176,032 bytes) is tracked in this repository since commit
`da7babf`, so a plain clone is enough to run the fixtures. Larger DeePMD
pretrained models (the DPA4 family) remain external fixtures that are
intentionally not tracked: put them in this directory for local/HPC validation,
or point tests to an external cache with environment variables:

- `DPEVA_DPA3_MODEL_PATH`: path to `DPA-3.1-3M.pt`
- `DPEVA_DPA4_MODEL_PATH`: path to `DPA4-Mini-OMat24.pt`

The dataset slices in this directory (`sampled_dpdata/122` with 1 labeled frame,
`other_dpdata_all/amourC/C2` with 5 unlabeled frames) are also the inputs for the
documented smoke closure; see `docs/guides/smoke-demo.md`. Their frame counts are
deliberately tiny and must not be used as a scientific benchmark.

The real DeePMD LLPR/DPOSE validation is opt-in and GPU-node only:

```bash
export DPEVA_RUN_DEEPMD_DPOSE_REAL=1
pytest tests/integration/test_real_deepmd_llpr_dpose.py -q
```

On SAI, prefer the Slurm wrapper:

```bash
sbatch scripts/slurm/run_dpeva_dpa4_llpr_dpose_gpu_test.sh
```
