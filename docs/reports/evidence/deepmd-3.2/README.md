---
title: DeepMD-kit 3.2 Qualification Evidence
status: active
audience: Compatibility Owners / Maintainers
last-updated: 2026-10-22
owner: Compatibility Owner
---

# DeepMD-kit 3.2 qualification evidence

The sanitized SAI aggregate and CPU contract records in this directory support
the three npy DPA4 PT capabilities referenced by the compatibility manifest,
plus the LMDB `test` capability and the LMDB `train` CPU-contract boundary.
The full raw aggregate remains on SAI and is bound here by SHA-256. Its
DPA4C-labelled command result is deliberately absent because the artifact was
later identified as DPA4 and is not promotion evidence for DPA4C.

LMDB-specific records:

- `lmdb-cpu-job-1893534-attestations/`: exact 3.2.0 CPU attestations for
  `test + deepmd/lmdb` and `train + deepmd/lmdb`; the job completed with
  `8 passed / 0 skipped`.
- `sai-lmdb-ab-job-1893546-public.json`: sanitized 2,604-frame npy/LMDB
  equivalence summary with bit-identical aggregate metrics.
- `sai-lmdb-ab-job-1893546-attestation.json`: one-case SAI promotion
  attestation for `test + deepmd/lmdb`.
- `sai-lmdb-ab-job-1893546-attestation.README.md`: provenance and scope of
  that one-case attestation.
- `sai-lmdb-train-ab-job-1893783-public.json`: 200-step training execution
  A/B summary; the companion README explains why it is not a format
  equivalence or promotion claim.
- `sai-lmdb-train-ab-job-1893783.README.md`: provenance and metric boundary
  for that training smoke.
