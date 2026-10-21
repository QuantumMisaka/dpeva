---
title: LMDB SAI A/B Attestation Provenance
status: active
audience: Compatibility Owners / Maintainers
last-updated: 2026-10-22
owner: Compatibility Owner
---

# LMDB SAI A/B attestation provenance

`sai-lmdb-ab-job-1893546-attestation.json` is the one-case promotion
attestation derived from SAI job `1893546`. The job ran a real
`dp --pt test` comparison of the same 2,604-frame dataset as `deepmd/npy`
and `deepmd/lmdb` on a `4V100` allocation in the exact `dpeva-dpa4-320`
environment.

The attestation uses the existing qualification case name `pt-test` because
the executed command is a `dp test` command. Its `gpu` value records the SAI
partition identity (`SAI 4V100`); no separate hardware-model probe was part
of this bounded A/B job. The sanitized public summary and raw-result SHA-256
are in `sai-lmdb-ab-job-1893546-public.json`; the raw logs and detail files
remain in the immutable SAI evidence directory.

This attestation is scoped to `test + deepmd/lmdb`. It does not promote LMDB
`train`, `fine-tune`, `eval-desc`, or `embed`.
