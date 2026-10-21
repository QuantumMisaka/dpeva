---
title: LMDB CPU Job 1893534 Attestations
status: active
audience: Compatibility Owners / Maintainers
last-updated: 2026-10-21
owner: Compatibility Owner
---

# LMDB CPU job 1893534 attestations

These producer-issued attestations record the first complete exact
`DeePMD-kit v3.2.0` CPU contract for the LMDB data path. The Slurm job
`1893534` completed with `8 passed / 0 skipped` and exit code `0:0`.

The job used the `dpeva-dpa4-320` environment, the protected four-atom
periodic fixture, and a generated `deepmd/lmdb` copy of that fixture. The
contract covers:

- `dp --pt test -s <lmdb> -d ...`, including DP-EVA frame-index attribution of
  the detail rows;
- a two-step `dp --pt train` run whose training and validation data are the
  same LMDB file.

The full raw pytest logs and generated model checkpoint remain on SAI. Only
the machine-readable attestations and the session result are retained here.
The attestation JSON files contain no user, organization, or host paths.

## Files

- `pt-test-lmdb.json`: finished CPU attestation for `test + deepmd/lmdb`.
- `pt-train-lmdb.json`: finished CPU attestation for `train + deepmd/lmdb`.
- `../lmdb-cpu-job-1893534-pytest-session.json`: pytest session result.

The corresponding SAI evidence directory retains the raw logs and generated
fixtures. The runner binds the evidence to the exact DeepMD version; this
directory is not a scientific accuracy claim.
