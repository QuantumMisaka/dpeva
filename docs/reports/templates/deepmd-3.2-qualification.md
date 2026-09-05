---
title: DeepMD-kit 3.2 SAI V100 Qualification
status: draft
audience: Compatibility Owner / Scientific Owner
last-updated: 2026-09-06
owner: Compatibility Owner
---

# DeepMD-kit 3.2 SAI V100 Qualification

## Scope

This report records execution evidence for the bounded SAI job. It does not
promote capabilities automatically and does not claim scientific superiority.
Record the launch-bound scope as either `dpa4` or `all`. A scope-less historical
input means `all`; it must not be reinterpreted during submission, execution, or
collection. `dpa4` requires the six regular/EMA DPA4 cases plus their preflight
and environment evidence. `all` additionally requires a genuine DPA4C artifact
and family inspection. DPA4C remains experimental even when that case succeeds.

## Environment and job identity

- Qualification environment: `dpeva-dpa4-320` (exact DeepMD-kit 3.2.0); this
  is distinct from the ordinary development environment `dpeva-dpa4`.
- DeepMD version: `qualification.json` → `environment/deepmd-version.json`
- GPU: `qualification.json` → `environment/gpu.json`
- Torch/CUDA: `qualification.json` → `environment/torch-cuda.json`
- JobID and immutable job directory: `qualification.json` / `submission.json`
- Environment lock: `environment/pip-freeze.json`
- Compute-node preflight: `commands/preflight.json` rehashes the launch
  contract and input/model/script artifacts before scientific commands run.

## Command evidence

The `commands/*.json` records contain argv, UTC start/end timestamps, return
code, declared artifacts, artifact existence and hashes. Any missing command or
artifact is `failed`; `submitted` is not `finished`.

## Fixture and model references

The fixture is a four-atom Fe/C/H/O periodic execution fixture. It is not a
scientific benchmark. Regular and EMA checkpoints are referenced by path and
SHA-256 in `input.json`; they are consumed in place and are not copied. A
`dpa4` run does not require a DPA4C model. An `all` run records the DPA4C model,
head, SHA-256, and successful family probe before its experimental command.

## Conclusion

Fill this section only after `collect_deepmd_32_qualification.py
--require-complete` succeeds and the Compatibility Owner reviews every case.
