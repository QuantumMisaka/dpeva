---
title: LMDB SAI Training A/B Job 1893783
status: active
audience: Compatibility Owners / Scientific Owner
last-updated: 2026-10-22
owner: Compatibility Owner
---

# LMDB SAI training A/B job 1893783

This record covers a real 200-step DPA4 training execution comparison on SAI
job `1893783`, using the exact `dpeva-dpa4-320` environment and a single
`4V100` allocation:

- training data: the same 16,485-frame corpus as `deepmd/npy/mixed` and
  `deepmd/lmdb`;
- validation data: the same 2,604-frame npy validation set;
- seed, model configuration, batch size, step count, and output layout are
  identical except for the training-data path;
- both `dp --pt train` commands returned zero and wrote step-200 checkpoints.

The steady-state step-100/200 wall times are close (mixed 34.98/37.71 s,
lmdb 36.46/35.96 s). The first-batch time includes SeZM compilation and is not
used as a format comparison.

This is deliberately an execution qualification, not a bitwise equivalence
claim. A 200-step run has stochastic sampler effects; the force-validation MAE
differed by 13.6% in this run even though the training loss and RMSE were
close. The sanitized JSON records the full metric comparison and marks
`metric_equivalence_claimed=false`. Accordingly, `train + deepmd/lmdb` remains
experimental; only `test + deepmd/lmdb` is promoted.

Raw logs, lcurve files, checkpoints, and `train_ab_result.json` remain in the
immutable SAI evidence directory. The public JSON contains hashes and no user,
organization, or host paths.
