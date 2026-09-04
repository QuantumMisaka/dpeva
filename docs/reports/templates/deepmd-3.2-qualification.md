---
title: DeepMD-kit 3.2 SAI V100 Qualification
status: draft
audience: Compatibility Owner / Scientific Owner
---

# DeepMD-kit 3.2 SAI V100 Qualification

## Scope

This report records execution evidence for the bounded SAI job. It does not
promote capabilities automatically and does not claim scientific superiority.

## Environment and job identity

- DeepMD version: `qualification.json` → `environment/deepmd-version.json`
- GPU: `qualification.json` → `environment/gpu.json`
- Torch/CUDA: `qualification.json` → `environment/torch-cuda.json`
- JobID and immutable job directory: `qualification.json` / `submission.json`
- Environment lock: `environment/pip-freeze.json`

## Command evidence

The `commands/*.json` records contain argv, UTC start/end timestamps, return
code, declared artifacts, artifact existence and hashes. Any missing command or
artifact is `failed`; `submitted` is not `finished`.

## Fixture and model references

The fixture is a four-atom Fe/C/H/O periodic execution fixture. It is not a
scientific benchmark. Regular and EMA checkpoints are referenced by path and
SHA-256 in `input.json`; they are consumed in place and are not copied.

## Conclusion

Fill this section only after `collect_deepmd_32_qualification.py
--require-complete` succeeds and the Compatibility Owner reviews every case.
