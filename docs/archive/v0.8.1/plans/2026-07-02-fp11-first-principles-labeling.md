---
title: FP11 SAI-1344 First Principles Labeling Implementation Plan
status: archived
audience: Historians / Developers / Operators / AI Agents
last-updated: 2026-07-07
owner: Workflow Owner
---

# FP11 SAI-1344 First Principles Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove SAI-1344 can run FP11 ABACUS labeling, improve DP-EVA's Slurm-native labeling backend where needed, stage FP11 data on SAI-1344, and launch a detached production labeling run from `sampled_dpdata`.

**Architecture:** Treat SAI-1344 as the production execution site and keep the original FP11 directory as the source of truth for input data and configs. First run live Slurm probes to prove whether `rush-gpu` and `flood-gpu` accept single-GPU jobs on `16V100`; then make DP-EVA labeling submit homogeneous task classes as Slurm arrays with normalized job names and SAI-safe environment ordering. Remote execution runs from a dedicated SAI-1344 staging directory under a detached `tmux` session so Slurm jobs and the DP-EVA monitor continue after SSH disconnect.

**Tech Stack:** Python 3.12, DP-EVA CLI, Pydantic v2 configs, dpdata/deepmd npy data, ABACUS `LTSv3.10.1-sm70-auto`, SAI-1344 Slurm, `sbatch --array`, `squeue`, `sacct`, `rsync`, `tmux`, `conda`.

---

## Execution Update 2026-07-06: SAI-1344 Slurm-Native Labeling

Current state: full FP11 SAI-1344 labeling completed through recovery, extract/postprocess, stats generation, and backend reporting. After configured retries, the final stats report is trusted and records `4353` converged frames, `471` failed frames, and `4317` clean frames. No Slurm array rows remain active.

Checklist note: the detailed task checklist below is retained as the implementation trace. The implementation and execution work is reflected by the final evidence in this update; repository cleanup and source commits are handled by the follow-up cleanup pass.

Remote staging:

- Repository: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva`
- FP11 inputs/configs: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11`
- Active config: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/config_gpu_1344.json`
- Work dir: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/labeling_workdir_1344`
- Completed detached recovery monitor: `tmux` session `fp11_1344_recover` exited after final report generation
- Recovery log: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/logs/fp11_recover_1344.log`
- Historical execute workflow log: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/labeling_workdir_1344/labeling_execute.log`
- Historical execute wrapper log: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/logs/fp11_execute_1344.wrapper.log`
- Historical finalizer log: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/logs/fp11_finalize_1344.log`
- Final remote stats: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/labeling_workdir_1344/outputs/labeling_stats_report.json`
- Final remote backend report: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/backend_report_1344.json` and `.md`
- Local backend report copies: `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/backend_report_1344.json` and `.md`

Historical execution evidence as of `2026-07-05 22:53:49 CST`:

- `16V100` single-GPU `rush-gpu` and `flood-gpu` probes were rejected by `sbatch` with `QOSMinGRES`; production therefore uses 4GPU MPI ABACUS fallback.
- Full prepare generated 4829 bundles: 4668 `normal` and 161 `highmem`.
- Attempt 0 submitted highmem job `577510` (`--array=0-160%128`) and normal job `577543` (`--array=0-4667%128`); both completed. Attempt 0 convergence scan reported `Converged: 1029`, `Bad-Converged: 3`, and `Failed: 3797`.
- Attempt 1 submitted highmem job `581292` (`--array=0-160%128`) and normal job `581295` (`--array=0-3635%128`). Highmem completed; normal is still running/pending.
- Current `581295` Slurm summary: `1772 COMPLETED`, `64 RUNNING`, and one pending array range `581295_[1836-3635%128]` with reason `AssocGrpGRES`.
- Abnormal-state query for `581292,581295` returned no `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `NODE_FAIL` records.
- `fp11_1344_finalize` is still waiting for `fp11_1344_execute`; extract/postprocess logs are not final evidence until the execute wrapper ends with `exit_code=0`.
- Interim backend report sidecars are generated at `backend_report_1344.json` and `backend_report_1344.md`; the finalizer will regenerate them after extract/postprocess.
- Tracking retries at `2026-07-05 23:02-23:14 CST` could not refresh the authoritative `login-02` evidence. `c0.sai.ai-4s.com:12022` is TCP-reachable, and local SSH config still points `sai-1344-tmp` to `liuzhaoqing@c0.sai.ai-4s.com:12022` with key `~/.ssh/id_ed25519_sai1_liuzhaoqing`, but the server either closed SSH sessions before command execution or returned `Permission denied` before a command could run. The current `login-01` shell cannot see `/home/pku-jianghong/liuzhaoqing/fp11-sai1344`, and its Slurm accounting is not authoritative for this FP11 run. The last authoritative FP11 state remains the `2026-07-05 22:53:49 CST` `login-02` snapshot above.
- Tracking retry at `2026-07-05 23:18 CST` again found `c0.sai.ai-4s.com:12022` TCP-reachable, but `ssh -o BatchMode=yes sai-1344-tmp` returned `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,keyboard-interactive)`. No new authoritative `login-02` Slurm or file-system evidence was collected.
- Tracking retry at `2026-07-05 23:24 CST` found `c0.sai.ai-4s.com:12022` rejecting TCP connections (`Connection refused`), and `ssh -o BatchMode=yes sai-1344-tmp` failed through `sss_ssh_knownhostsproxy` before command execution. No new authoritative `login-02` Slurm or file-system evidence was collected.
- Access check at `2026-07-06 00:12 CST` confirmed recovery: `c0.sai.ai-4s.com:12022` is TCP-reachable again, and `ssh -o BatchMode=yes sai-1344-tmp hostname` lands on `login-02.mr-sai.ai`. Both detached sessions, `fp11_1344_execute` and `fp11_1344_finalize`, are still present. Attempt 1 finished at `2026-07-05 23:00:18 CST`; the convergence scan reported `Converged: 2595`, `Bad-Converged: 2`, and `Failed: 1200`, after which DP-EVA submitted attempt 2 as `586728` (`highmem-20c61984_attempt_2`) and `586731` (`normal-9c2a6e48_attempt_2`). As of `2026-07-06 00:12 CST`, both attempt-2 arrays are `PENDING` with reason `AssocGrpGRES`; abnormal-state query for `586728,586731` returned no `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `NODE_FAIL` records. `fp11_1344_finalize` is still waiting for `fp11_1344_execute`, so extract/postprocess have not started.
- Recovery action at `2026-07-06 00:53 CST`: authoritative `squeue`/`sacct` evidence showed `581295` was still active even though the DP-EVA monitor had logged `All jobs finished` at `2026-07-05 23:00:18 CST`. At `00:36 CST`, `581295` still had `64 RUNNING` plus a pending array range, and `586728/586731` were only `PENDING`. A manifest overlap check found 449 active `581295` working directories also present in the attempt-2 normal manifest, proving duplicate-write risk if retry arrays later started. Containment stopped the stale `fp11_1344_execute` and `fp11_1344_finalize` tmux sessions, cancelled the not-yet-running retry arrays `586728` and `586731`, and left the original `581295` array running to finish its remaining work. DP-EVA's local and staged remote workflow code now performs a `sacct -X --format=State` active-state confirmation when `squeue` is empty, preventing a repeat of this transient empty-queue false finish. A new detached recovery session `fp11_1344_recover` was launched at `2026-07-06 00:52 CST`; it waits for `581295` to truly finish, processes results, submits only the remaining failed tasks for attempt 2 with the fixed monitor, then runs extract, postprocess, and backend report generation. As of `2026-07-06 00:53:46 CST`, `586728_[0-19%128]` and `586731_[0-1179%128]` are `CANCELLED`, while `581295` reports `2391 COMPLETED`, `64 RUNNING`, and one pending range.
- Recovery log streaming was repaired at `2026-07-06 00:58 CST`: `fp11_1344_recover_wrapper.sh` now runs `conda run --no-capture-output` with `PYTHONUNBUFFERED=1`, so `logs/fp11_recover_1344.log` shows live wait/retry status. The restarted `fp11_1344_recover` session logged `65 Slurm records still active for 581295 after 0 min`, with sample records including the pending range and running array elements.
- Tracking check at `2026-07-06 01:08 CST`: `fp11_1344_recover` is still active and waiting. `sacct -X -j 581295` reports `2437 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log confirms live polling at 5-minute intervals and records `65 Slurm records still active for 581295 after 10 min`, with pending range `581295_[2501-3635%128]`. Final extract/postprocess and regenerated backend report have not started yet.
- Access check at `2026-07-06 01:14 CST`: SSH access to `sai-1344-tmp` is restored and lands on `login-02.mr-sai.ai`; `tmux`, `squeue`, and `sacct` all respond normally. The detached `fp11_1344_recover` session remains active. `sacct -X -j 581295` reports `2491 COMPLETED`, `64 RUNNING`, and one pending range (`581295_[2555-3635%128]`, reason `AssocGrpGRES`), so recovery should continue waiting rather than extracting or submitting retry work.
- Tracking check at `2026-07-06 01:17 CST`: `fp11_1344_recover` is still the only active detached session for this recovery path. `squeue -j 581295` reports pending range `581295_[2574-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2510 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and last recorded `65 Slurm records still active for 581295 after 15 min`. `backend_report_1344.json` and `.md` are still the interim `2026-07-05 21:32` files, and extract/postprocess logs are not present yet.
- Duplicate-write safety check at `2026-07-06 01:18 CST`: `squeue -u liuzhaoqing` shows only the original `581295` FP11 path active; `squeue -j 586728,586731` has no active records, and `sacct -X -j 586728,586731` keeps `586728_[0-19%128]` and `586731_[0-1179%128]` as `CANCELLED by 1478400036`. No premature retry array is currently able to write into overlapping work directories.
- Tracking check at `2026-07-06 01:21 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2586-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2522 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log last recorded `65 Slurm records still active for 581295 after 20 min`. Final extract/postprocess have not started, and `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 01:24 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2606-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2542 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log last recorded `65 Slurm records still active for 581295 after 25 min`. Final extract/postprocess have not started, and `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 01:27 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2613-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2549 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log's latest 5-minute poll is still the `01:23 CST` record (`65 Slurm records still active for 581295 after 25 min`). Final extract/postprocess have not started, and `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 01:29 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2622-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2558 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log last recorded `65 Slurm records still active for 581295 after 30 min`, with pending range `581295_[2616-3635%128]` in its sample. Final extract/postprocess have not started, `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files, and `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036` with no active `squeue` records.
- Tracking check at `2026-07-06 01:32 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2631-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2567 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log's latest 5-minute poll remains the `01:28 CST` record (`65 Slurm records still active for 581295 after 30 min`). Final extract/postprocess have not started, `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files, and `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036` with no active `squeue` records.
- Access retry at `2026-07-06 01:38 CST`: DNS for `c0.sai.ai-4s.com` resolves to `218.84.111.148`, but TCP probes to port `12022` failed and minimal `ssh -o BatchMode=yes sai-1344-tmp "echo ok"` timed out with `Connection to UNKNOWN port 65535 timed out`. No fresh authoritative `login-02` Slurm or file-system evidence was collected in this retry. The last authoritative FP11 state remains the `2026-07-06 01:32 CST` snapshot above; the detached Slurm jobs and remote `tmux` recovery monitor should continue independently of the local SSH session, but their current state is unverified until access recovers.
- Tracking check at `2026-07-06 01:42 CST`: SSH command execution recovered and again lands on `login-02.mr-sai.ai`; `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2713-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2649 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and last recorded `66 Slurm records still active for 581295 after 40 min`, with sample `581295_2490 COMPLETING`, the pending range, and running array elements. Final extract/postprocess have not started, `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files, and the prematurely submitted `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036` with no active `squeue` records.
- Access confirmation at `2026-07-06 01:49 CST`: local DNS resolves `c0.sai.ai-4s.com` to `218.84.111.148`, TCP port `12022` is reachable, and minimal SSH to `sai-1344-tmp` lands on `login-02.mr-sai.ai`; access is recovered. `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2731-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2667 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and last recorded `65 Slurm records still active for 581295 after 50 min`. Final extract/postprocess have not started, `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files, and `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036`.
- Tracking check at `2026-07-06 01:53 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2752-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2688 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 55 min`, with sample pending range `581295_[2755-3635%128]`. Final extract/postprocess outputs are still absent, `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files, and `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036`.
- Tracking check at `2026-07-06 01:57 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[2779-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2715 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log latest visible poll remains `01:53:43 CST` (`65 Slurm records still active for 581295 after 55 min`). Final extract/postprocess outputs are still absent, `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files, and the prematurely submitted retry arrays have no active `squeue` records while `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036`.
- Tracking check at `2026-07-06 01:59 CST`: `squeue -j 581295` reports pending range `581295_[2792-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2728 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 60 min`, with sample pending range `581295_[2784-3635%128]`. Final extract/postprocess outputs are still absent, and `backend_report_1344.json`/`.md` are still the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 02:09 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains the only active detached FP11 recovery session in `tmux`. `squeue -j 581295` reports pending range `581295_[2893-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2829 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 70 min`, with sample pending range `581295_[2872-3635%128]`. Final extract/postprocess outputs are still absent, `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files, and the premature retry arrays have no active `squeue` records while `586728_[0-19%128]` plus `586731_[0-1179%128]` remain `CANCELLED by 1478400036`.
- Tracking check at `2026-07-06 02:12 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports pending range `581295_[2969-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2905 COMPLETED`, `64 RUNNING`, and one pending range. The latest recovery log poll visible in this check remains the `02:08:43 CST` record (`65 Slurm records still active for 581295 after 70 min`). Final extract/postprocess outputs are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files. The premature retry arrays have no active `squeue` records and remain `CANCELLED by 1478400036`; the existing attempt-2 manifests under `array_jobs/*_attempt_2/tasks.json` are still the stale `2026-07-05 23:06` files from the false-finish path and must not be interpreted as a fresh recovery retry submission.
- Tracking check at `2026-07-06 02:15 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports pending range `581295_[3039-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `2975 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 75 min`, with sample pending range `581295_[3023-3635%128]`. Final extract/postprocess outputs are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files. The premature retry arrays `586728` and `586731` have no active `squeue` records and remain `CANCELLED by 1478400036`.
- Tracking check at `2026-07-06 02:20 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports pending range `581295_[3228-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `3164 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 80 min`, with sample pending range `581295_[3202-3635%128]`. Final extract/postprocess outputs are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files. The old retry arrays `586728` and `586731` remain inactive/cancelled, so the recovery path is still correctly waiting on original attempt-1 work.
- Tracking check at `2026-07-06 02:22 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports pending range `581295_[3256-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `3192 COMPLETED`, `64 RUNNING`, and one pending range. The latest recovery log poll visible in this check remains the `02:18:43 CST` record (`65 Slurm records still active for 581295 after 80 min`). Final extract/postprocess outputs are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files. The old retry arrays `586728` and `586731` remain inactive/cancelled, and `squeue -u liuzhaoqing` shows only the original `581295` FP11 array.
- Tracking check at `2026-07-06 02:25 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports pending range `581295_[3264-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `3200 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 85 min`, with sample pending range `581295_[3264-3635%128]`. Final extract/postprocess outputs are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files. The old retry arrays `586728` and `586731` remain inactive/cancelled.
- Access check at `2026-07-06 02:33 CST`: SSH command execution is restored and lands on `login-02.mr-sai.ai`; `tmux ls` shows the detached `fp11_1344_recover` session still alive. `squeue -j 581295` reports pending range `581295_[3341-3635%128]` plus `64 RUNNING`, and `sacct -X -j 581295` reports `3277 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log also wrote a fresh `02:33:43 CST` poll (`65 Slurm records still active for 581295 after 95 min`, sample pending range `581295_[3345-3635%128]`). This confirms both remote access and the detached recovery monitor are available again; the recovery path is still waiting on original attempt-1 work.
- Tracking check at `2026-07-06 02:42 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains the only active detached FP11 recovery session. `squeue -j 581295` reports pending range `581295_[3494-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `3430 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 100 min`, with sample pending range `581295_[3482-3635%128]`. `squeue -u liuzhaoqing` shows only the original `581295` FP11 array; the cancelled retry arrays `586728` and `586731` still have no active `squeue` records and remain `CANCELLED by 1478400036` in `sacct`. Final extract/postprocess outputs are still absent: `ls` only found the interim `backend_report_1344.json`/`.md` files from `2026-07-05 21:32`, while `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are not present yet.
- Tracking check at `2026-07-06 02:49 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active in `tmux`. `squeue -j 581295` reports pending range `581295_[3625-3635%128]` plus `64 RUNNING`; `sacct -X -j 581295` reports `3561 COMPLETED`, `64 RUNNING`, and one pending range. The recovery log remains live and records `65 Slurm records still active for 581295 after 110 min`, with sample pending range `581295_[3620-3635%128]`. This is still the original attempt-1 tail; recovery has not yet reached the post-581295 processing/retry/extract stages.
- Tracking check at `2026-07-06 02:56 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` no longer shows a pending range, but still reports `54 RUNNING`; `sacct -X -j 581295` reports `3582 COMPLETED` and `54 RUNNING`. The recovery log remains live and records `55 Slurm records still active for 581295 after 115 min`, with samples all running. Recovery must continue waiting for these tail tasks before processing results or submitting the fixed attempt-2 retry arrays.
- Tracking check at `2026-07-06 03:03 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports `48 RUNNING`; `sacct -X -j 581295` reports `3588 COMPLETED` and `48 RUNNING`. The recovery log remains live and records `51 Slurm records still active for 581295 after 120 min`, with samples all running. The run is still in the original attempt-1 tail and has not yet reached result processing, fixed retry submission, extract, or postprocess.
- Tracking check at `2026-07-06 03:09 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports `47 RUNNING`; `sacct -X -j 581295` reports `3589 COMPLETED` and `47 RUNNING`. The recovery log remains live and records `48 Slurm records still active for 581295 after 125 min`, with samples all running. Recovery has still not reached result processing, fixed retry submission, extract, or postprocess. The old retry arrays `586728` and `586731` still have no active `squeue` records and remain `CANCELLED by 1478400036`; final extract/postprocess outputs are still absent, with only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` present.
- Tracking check at `2026-07-06 03:19 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports `33 RUNNING`; `sacct -X -j 581295` reports `3603 COMPLETED` and `33 RUNNING`. The recovery log remains live and records `35 Slurm records still active for 581295 after 140 min`, with samples all running. Recovery is still waiting for the original attempt-1 tail before result processing, fixed retry submission, extract, or postprocess. The old retry arrays `586728` and `586731` remain `CANCELLED by 1478400036`, and final extract/postprocess outputs are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Tracking check at `2026-07-06 03:23 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `squeue -j 581295` reports `28 RUNNING`; `sacct -X -j 581295` reports `3608 COMPLETED` and `28 RUNNING`. The abnormal-state query for `581295` returned no `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `NODE_FAIL` records. The latest visible recovery-log poll is still the `03:18 CST` record (`35 Slurm records still active for 581295 after 140 min`). Recovery has not reached result processing, fixed retry submission, extract, or postprocess. The old retry arrays `586728` and `586731` have no active `squeue` records and remain `CANCELLED by 1478400036`; final extract/postprocess outputs are still absent, with only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` present.
- Tracking check at `2026-07-06 03:25 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. The recovery log records `26 Slurm records still active for 581295 after 145 min` at `03:23:43 CST`; a fresh `squeue`/`sacct` check at `03:24:47 CST` reports `24 RUNNING` and `3612 COMPLETED`. Recovery is still in the original attempt-1 tail and has not reached result processing, fixed retry submission, extract, or postprocess.
- Tracking check at `2026-07-06 03:30 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. The recovery log records `20 Slurm records still active for 581295 after 150 min` at `03:28:43 CST`; a fresh `squeue`/`sacct` check at `03:29:41 CST` reports `20 RUNNING` and `3616 COMPLETED`. The abnormal-state query for `581295` remains empty, and `squeue -u liuzhaoqing` shows only the original `581295` FP11 array. Final extract/postprocess outputs are still absent, and recovery has not reached result processing or fixed retry submission.
- Tracking check at `2026-07-06 03:35 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. The recovery log records `14 Slurm records still active for 581295 after 155 min` at `03:33:43 CST`; a fresh `squeue`/`sacct` check at `03:34:40 CST` reports `13 RUNNING` and `3623 COMPLETED`. `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent. Recovery remains in the original attempt-1 tail and has not reached result processing or fixed retry submission.
- Tracking check at `2026-07-06 03:40 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. The recovery log records `8 Slurm records still active for 581295 after 160 min` at `03:38:43 CST`; a fresh `squeue`/`sacct` check at `03:40:24 CST` reports `7 RUNNING` and `3629 COMPLETED`. `squeue -u liuzhaoqing` still shows only the original `581295` FP11 array, and the newest attempt-2 manifests are still the stale `2026-07-05 23:06` files from the false-finish path. Final extract/postprocess outputs are still absent, so recovery has not reached fixed retry submission, extract, or postprocess.
- Tracking check at `2026-07-06 03:45 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. The recovery log records `6 Slurm records still active for 581295 after 165 min` at `03:43:43 CST`; a fresh `squeue`/`sacct` check at `03:45:09 CST` reports `4 RUNNING` and `3632 COMPLETED`. The only active FP11 jobs in `squeue -u liuzhaoqing` remain `581295` array elements, and attempt-2 manifests remain the stale `2026-07-05 23:06` files. Final extract/postprocess outputs are still absent, so recovery has not reached fixed retry submission, extract, or postprocess.
- Tracking check at `2026-07-06 03:50 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. The recovery log records `1 Slurm records still active for 581295 after 170 min` at `03:48:43 CST`, but a fresh `squeue`/`sacct` check at `03:49:59 CST` shows no active `581295` records and `3636 COMPLETED`. `squeue -u liuzhaoqing` shows no active FP11 rows. Recovery has not yet logged result processing or fixed retry submission; it should confirm the clear state on its next poll before processing results. Final extract/postprocess outputs are still absent, and attempt-2 manifests remain the stale `2026-07-05 23:06` files.
- Tracking check at `2026-07-06 03:55 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. `sacct -X -j 581295` now reports `3636 COMPLETED` with no active states, and `squeue -u liuzhaoqing` shows no active FP11 rows. The recovery log confirms `No active Slurm records remain for 581295` at `03:53:43 CST`, then logs `Processing results after true completion of 581295` and `dpeva.labeling.manager: Processing results...` at `03:53:44 CST`. This proves the `sacct` false-finish guard path waited for true completion. Fixed retry submission, extract, postprocess, and final backend report generation have not yet appeared in the log; attempt-2 manifests are still the stale `2026-07-05 23:06` files.
- Tracking check at `2026-07-06 03:58 CST`: recovery processed attempt-1 true completion and reported `Converged: 584, Bad-Converged: 0, Failed: 616`; it then applied attempt-2 parameters to 616 tasks and submitted fresh fixed retry arrays at `03:56 CST`: highmem job `590738` (`fp-highmem-att2`) and normal job `590758` (`fp-normal-att2`). The new attempt-2 manifests replaced the stale false-finish manifests at `2026-07-06 03:56`: `highmem-20c61984_attempt_2/tasks.json` contains 20 tasks and `normal-9c2a6e48_attempt_2/tasks.json` contains 596 tasks. `squeue -u liuzhaoqing` shows highmem elements `590738_0-19` running and normal elements `590758_0-43` running, with `590758_[44-595%128]` pending for `AssocGrpGRES`. Final extract/postprocess outputs are still absent while recovery monitors these two retry arrays.
- Tracking check at `2026-07-06 04:00 CST`: the fresh retry arrays `590738,590758` are being monitored by `fp11_1344_recover`. `squeue -j 590738,590758` and `sacct -X -j 590738,590758` both report `64 RUNNING` plus one pending range, and the abnormal-state query for these two job IDs returns no `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `NODE_FAIL` records.
- Tracking check at `2026-07-06 04:08 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and one pending range `590758_[48-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `4 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` returns no `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, or `NODE_FAIL` records. The recovery log tail still ends at the attempt-2 submission/monitoring records from `03:56 CST`; final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent. `backend_report_1344.json` and `.md` remain the interim `2026-07-05 21:32` files, so they are not final evidence.
- Tracking check at `2026-07-06 04:11 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and the same pending range `590758_[48-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `4 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records; final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 04:14 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[48-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `4 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. `ps` confirms the detached wrapper, `conda run`, and `fp11_1344_recover_after_false_finish.py` process are still alive under the `fp11_1344_recover` session. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 04:16 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[48-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `4 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. `ps` confirms the detached wrapper, `conda run`, and recovery Python process are still alive, with elapsed time over 3 hours. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 04:19 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[48-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `4 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. `ps` confirms the detached wrapper, `conda run`, and recovery Python process are still alive, with elapsed time over 3 hours 20 minutes. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 04:21 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[48-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `4 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. `ps` confirms the detached wrapper, `conda run`, and recovery Python process are still alive, with elapsed time over 3 hours 23 minutes. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Access confirmation at `2026-07-06 04:26 CST`: SSH command execution is restored and lands on `login-02.mr-sai.ai` as `liuzhaoqing`; `tmux ls` shows the detached `fp11_1344_recover` session still alive. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` plus one pending range, while `sacct -X -j 590738,590758` reports `10 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is empty, and the recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records, so extract/postprocess and final backend report generation have not started yet.
- Tracking check at `2026-07-06 04:32 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[60-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `16 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Tracking check at `2026-07-06 04:36 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[66-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `22 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Tracking check at `2026-07-06 04:41 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains active. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[70-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `26 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Tracking check at `2026-07-06 04:45 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux`, the wrapper process, `conda run`, and `fp11_1344_recover_after_false_finish.py` are all still alive under `fp11_1344_recover`. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[84-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `40 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. The recovery log tail still ends at the `03:56 CST` attempt-2 submission/monitoring records. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Tracking check at `2026-07-06 04:50 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux`, the wrapper process, `conda run`, and `fp11_1344_recover_after_false_finish.py` are all still alive under `fp11_1344_recover`. The recovery log now has a fresh `04:46:36 CST` heartbeat reporting `65 jobs still running... (waited 50 mins)`. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[102-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `58 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Tracking check at `2026-07-06 04:55 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux`, the wrapper process, `conda run`, and `fp11_1344_recover_after_false_finish.py` are all still alive under `fp11_1344_recover`. Retry arrays `590738,590758` are still running: `squeue -j 590738,590758` reports `64 RUNNING` and pending range `590758_[108-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `64 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `590738,590758` is still empty. The latest recovery log heartbeat remains the `04:46:36 CST` record reporting `65 jobs still running... (waited 50 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; only the interim `backend_report_1344.json`/`.md` from `2026-07-05 21:32` is present.
- Access confirmation at `2026-07-06 05:03 CST`: SSH command execution is restored and lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive. `ps` confirms the wrapper, `conda run`, and `fp11_1344_recover_after_false_finish.py` are still running with elapsed time over 4 hours. Retry arrays `590738,590758` remain active: `squeue -j 590738,590758` reports highmem rows still running, normal retry rows running, and pending range `590758_[125-595%128]` blocked by `AssocGrpGRES`; `sacct -X -j 590738,590758` reports `81 COMPLETED`, `64 RUNNING`, and one pending range. This confirms access and Slurm query capability have recovered; the recovery path is still monitoring retry work rather than finalizing.
- Tracking check at `2026-07-06 05:13 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux` and the recovery Python process remain alive under `fp11_1344_recover`. Retry arrays `590738,590758` are still active: `squeue -j 590738,590758` reports `64 RUNNING` plus pending range `590758_[132-595%128]` blocked by `AssocGrpGRES`, and a later `sacct -X --format=State` summary reports `97 COMPLETED`, `64 RUNNING`, and one pending range. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The recovery log still has not advanced past the `04:46:36 CST` heartbeat, which is consistent with the monitor remaining inside attempt-2 job waiting. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are absent, while `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 05:17 CST`: SSH still lands on `login-02.mr-sai.ai`, and `fp11_1344_recover` remains alive in `tmux`. Retry arrays `590738,590758` continue to make progress but are not clear yet: `sacct -X --format=State` reports `115 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758` shows pending range `590758_[159-595%128]` blocked by `AssocGrpGRES`. The abnormal-state query still returns only the header. The recovery log has not moved beyond the attempt-2 wait phase, and final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 05:23 CST`: SSH still lands on `login-02.mr-sai.ai`, `fp11_1344_recover` remains alive in `tmux`, and the wrapper/conda/recovery Python processes are still running. Retry arrays `590738,590758` are progressing but still active: `sacct -X --format=State` reports `123 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758` shows pending range `590758_[167-595%128]` blocked by `AssocGrpGRES`. The abnormal-state query still returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Access confirmation at `2026-07-06 05:37 CST`: SSH command execution is restored and lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and `fp11_1344_recover_after_false_finish.py` remain running. Retry arrays `590738,590758` are still active but progressing: `sacct -X -j 590738,590758 --format=State -n -P` reports `192 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590758 -h -t PD -o %i,%T,%R` shows `590758_[236-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 05:44 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and `fp11_1344_recover_after_false_finish.py` are still running. Retry arrays `590738,590758` continue progressing but are still active: `sacct -X -j 590738,590758 --format=State -n -P` reports `275 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[317-595%128]` pending on `AssocGrpGRES` plus highmem elements `590738_5-9` still running. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The recovery log has a fresh heartbeat at `2026-07-06 05:36:36 CST` reporting `65 jobs still running... (waited 100 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 05:52 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and `fp11_1344_recover_after_false_finish.py` are still running. Retry arrays `590738,590758` continue progressing but are still active: `sacct -X -j 590738,590758 --format=State -n -P` reports `316 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[360-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains `2026-07-06 05:36:36 CST` with `65 jobs still running... (waited 100 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent, and `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Access confirmation at `2026-07-06 06:01 CST`: SSH command execution is restored and lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive. Retry arrays `590738,590758` remain active but progressing: `sacct -X -j 590738,590758 --format=State -n -P` reports `326 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[370-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The recovery log is still in the attempt-2 wait phase; final extract/postprocess outputs have not started.
- Tracking check at `2026-07-06 06:07 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive. Retry arrays `590738,590758` remain active but progressing: `sacct -X -j 590738,590758 --format=State -n -P` reports `330 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[374-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 06:14 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still running with elapsed time over 5 hours 15 minutes. Retry arrays `590738,590758` continue progressing but are still active: `sacct -X -j 590738,590758 --format=State -n -P` reports `350 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[394-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains `2026-07-06 05:36:36 CST` with `65 jobs still running... (waited 100 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 06:20 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still running with elapsed time over 5 hours 22 minutes. Retry arrays `590738,590758` continue progressing but are still active: `sacct -X -j 590738,590758 --format=State -n -P` reports `371 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[415-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains `2026-07-06 05:36:36 CST` with `65 jobs still running... (waited 100 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 06:30 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still running with elapsed time over 5 hours 32 minutes. Retry arrays `590738,590758` continue progressing but are still active: `sacct -X -j 590738,590758 --format=State -n -P` reports `382 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[426-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The recovery log has a fresh `2026-07-06 06:26:37 CST` heartbeat reporting `65 jobs still running... (waited 150 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 06:36 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still running with elapsed time over 5 hours 38 minutes. Retry arrays `590738,590758` continue progressing but remain active: `sacct -X -j 590738,590758 --format=State -n -P` reports `392 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[436-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 06:26:37 CST` record reporting `65 jobs still running... (waited 150 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 06:41 CST`: SSH still lands on `login-02.mr-sai.ai`; `tmux ls` shows `fp11_1344_recover` still detached and alive, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still running with elapsed time over 5 hours 43 minutes. Retry arrays `590738,590758` continue progressing but remain active: `sacct -X -j 590738,590758 --format=State -n -P` reports `415 COMPLETED`, `64 RUNNING`, and one pending range, while `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[459-595%128]` pending on `AssocGrpGRES`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 06:26:37 CST` record reporting `65 jobs still running... (waited 150 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Access-restoration check at `2026-07-06 06:50 CST`: BatchMode SSH to `sai-1344-tmp` succeeds again (`dp_eva_access_ok`), landing on `login-02.mr-sai.ai` with remote clock `2026-07-06T06:50:11+08:00`. `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 5 hours 52 minutes. Retry array monitoring remains active rather than complete: `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[476-595%128]` pending on `AssocGrpGRES` plus 64 running array elements; the recovery log still has the latest heartbeat at `2026-07-06 06:26:37 CST` reporting `65 jobs still running... (waited 150 mins)`.
- Tracking check at `2026-07-06 06:56 CST`: SSH still lands on `login-02.mr-sai.ai`, and `tmux ls` still shows the detached `fp11_1344_recover` session. `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 5 hours 57 minutes. Retry arrays `590738,590758` remain active: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `444 COMPLETED`, `64 RUNNING`, and `1 PENDING`; `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[488-595%128]` pending on `AssocGrpGRES` plus running elements. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:02 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:02:09+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 3 minutes. Retry arrays `590738,590758` continue progressing but remain active: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `468 COMPLETED`, `64 RUNNING`, and `1 PENDING`; `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[512-595%128]` pending on `AssocGrpGRES` plus running elements. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:07 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:07:26+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 8 minutes. Retry arrays `590738,590758` remain active: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `476 COMPLETED`, `64 RUNNING`, and `1 PENDING`; `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[520-595%128]` pending on `AssocGrpGRES` plus running elements. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:18 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:17:56+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 20 minutes. Retry arrays `590738,590758` remain active: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `523 COMPLETED`, `64 RUNNING`, and `1 PENDING`; `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[567-595%128]` pending on `AssocGrpGRES` plus running elements. The recovery log has a fresh `2026-07-06 07:16:37 CST` heartbeat reporting `65 jobs still running... (waited 200 mins)`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:25 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:24:33+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 26 minutes. Retry arrays `590738,590758` remain active: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `528 COMPLETED`, `64 RUNNING`, and `1 PENDING`; `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[572-595%128]` pending on `AssocGrpGRES` plus running elements. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 07:16:37 CST` record reporting `65 jobs still running... (waited 200 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:31 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:30:34+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 32 minutes. Retry arrays `590738,590758` remain active: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `537 COMPLETED`, `64 RUNNING`, and `1 PENDING`; `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows `590758_[581-595%128]` pending on `AssocGrpGRES` plus running elements. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 07:16:37 CST` record reporting `65 jobs still running... (waited 200 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:45 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:44:36+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 46 minutes. Retry arrays `590738,590758` are still active but have cleared the pending queue: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `558 COMPLETED` and `58 RUNNING`, with no `PENDING` line. `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows the remaining tail elements running, including `590758_595` through `590758_576` in the head sample and `590758_557` through `590758_524` in the tail sample. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 07:16:37 CST` record reporting `65 jobs still running... (waited 200 mins)`, which is consistent with the workflow's sparse heartbeat cadence. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` remain absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:53 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:53:00+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 6 hours 54 minutes. Retry arrays `590738,590758` remain active with no pending rows: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `566 COMPLETED` and `50 RUNNING`. `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows the remaining normal retry tail running, with head sample `590758_595` through `590758_566` and tail sample down to `590758_525`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 07:16:37 CST` record reporting `65 jobs still running... (waited 200 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 07:59 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T07:59:36+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 7 hours 1 minute. Retry arrays `590738,590758` remain active with no pending rows: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `569 COMPLETED` and `47 RUNNING`. `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows the remaining normal retry tail running, with head sample `590758_595` through `590758_566` and tail sample down to `590758_525`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 07:16:37 CST` record reporting `65 jobs still running... (waited 200 mins)`. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 08:10 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T08:10:10+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 7 hours 13 minutes. Retry arrays `590738,590758` remain active with no pending rows: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `580 COMPLETED` and `36 RUNNING`. `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows the remaining normal retry tail running, with head sample `590758_595` through `590758_564` and tail sample down to `590758_527`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The recovery log has a fresh `2026-07-06 08:06:37 CST` heartbeat reporting `46 jobs still running... (waited 250 mins)`, while the fresher `sacct` sample shows the tail has already dropped to 36 running. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 08:19 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T08:19:42+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 7 hours 22 minutes. Retry arrays `590738,590758` remain active with no pending rows: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `588 COMPLETED` and `28 RUNNING`. `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows the remaining normal retry tail running, with rows from `590758_595` through `590758_554`. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 08:06:37 CST` record reporting `46 jobs still running... (waited 250 mins)`, which is older than the current `sacct` sample. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Tracking check at `2026-07-06 08:32 CST`: SSH still lands on `login-02.mr-sai.ai` with remote clock `2026-07-06T08:32:16+08:00`; `tmux ls` still shows `fp11_1344_recover`, and `ps` confirms the wrapper, `conda run`, and recovery Python process are still alive after about 7 hours 34 minutes. Retry arrays `590738,590758` remain active with no pending rows: `sacct -X -j 590738,590758 --format=State -n -P | sort | uniq -c` reports `608 COMPLETED` and `8 RUNNING`. `squeue -j 590738,590758 -h -o %i,%t,%M,%R` shows the remaining normal retry tail running as `590758_594`, `590758_591`, `590758_590`, `590758_589`, `590758_586`, `590758_585`, `590758_581`, and `590758_576`, with elapsed times from about 54 minutes to 1 hour 5 minutes. The abnormal-state query for `FAILED`, `CANCELLED`, `TIMEOUT`, `OUT_OF_MEMORY`, and `NODE_FAIL` returns only the header. The latest recovery-log heartbeat remains the `2026-07-06 08:06:37 CST` record reporting `46 jobs still running... (waited 250 mins)`, which is older than the current Slurm sample. Final `fp11_extract_1344.log`, `fp11_postprocess_1344.log`, and `labeling_stats_report.json` are still absent; `backend_report_1344.json`/`.md` remain the interim `2026-07-05 21:32` files.
- Final tracking check at `2026-07-06 08:56 CST`: retry arrays `590738,590758` cleared at `08:41 CST`; `sacct -X -j 590738,590758 --format=State -n -P` reported `616 COMPLETED` and `squeue` returned no rows. The recovery log then recorded `All jobs finished`, processed attempt-2 results, and reported `Converged: 145, Bad-Converged: 0, Failed: 471` for the retry set before running final extract/postprocess. Final `labeling_workdir_1344/outputs/labeling_stats_report.json` exists with trusted consistency and global counts `total=4824`, `conv=4353`, `fail=471`, `clean=4317`, and `filt=36`. The `fp11_1344_recover` tmux session and recovery Python process have exited; `tmux ls` returns `no server running`, and `squeue -j 577510,577543,581292,581295,586728,586731,590738,590758` returns no active rows. `backend_report_1344.json`/`.md` were regenerated at `2026-07-06 08:56 CST` after fixing the report script to include recovery-log job ids; the report now lists job ids `577510`, `577543`, `581292`, `581295`, `586728`, `586731`, `590738`, and `590758`, state counts `9242 COMPLETED` plus the two intentionally cancelled false-finish arrays `586728/586731`, submit elapsed seconds `[9.452, 13.286, 4.223, 2.569]`, `array_submission_count=6`, `total_bundle_count=9242`, `array_to_bundle_ratio=0.000649`, and `design_acceptable=True`. The final backend report files were copied back to `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/backend_report_1344.json` and `.md`. Separate `fp11_extract_1344.log` and `fp11_postprocess_1344.log` files were not created on the recovery path; the authoritative final-stage evidence is the recovery log plus the generated stats and backend report artifacts.

Operational commands:

```bash
ssh sai-1344-tmp
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
tmux attach -t fp11_1344_recover
tail -f labeling_workdir_1344/labeling_execute.log
tail -f logs/fp11_recover_1344.log
squeue -j 581295
sacct -X -j 581295 --format=State -n -P | sort | uniq -c
sacct -j 581295 --state=FAILED,CANCELLED,TIMEOUT,OUT_OF_MEMORY,NODE_FAIL --format=JobID,JobName,State,ExitCode,Elapsed -P
```

## Current State And Scope

Current date: 2026-07-04.

Original local FP11 source directory:

```text
/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11
```

Current DP-EVA repository:

```text
/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva
```

SAI-1344 target staging directory:

```text
/home/pku-jianghong/liuzhaoqing/fp11-sai1344
```

The previous partial FP11 attempts produced 580 effectively completed tasks, but this plan intentionally permits a clean SAI-1344 rerun from `sampled_dpdata`, as requested. The completed-task manifest remains useful for comparison but is not a prerequisite for the SAI-1344 run.

The local DP-EVA code already has task-class launcher semantics:

- ordinary tasks: `launcher_mode="abacus"`, `resource_mode="single_gpu"`;
- high-memory tasks: `launcher_mode="mpi_abacus"`, `resource_mode="multi_gpu_mpi"`.

The current Slurm backend is not yet Slurm-native enough for this large run:

- one packed bundle still maps to one `sbatch`;
- there is no generic `JobConfig.array` support;
- there is no manifest-backed Slurm array worker;
- job names are not normalized through a shared helper;
- SAI-1344 requires `source /etc/profile` after all `#SBATCH` lines, and rank-map setup must not be inserted before it.

## Acceptance Criteria

- `rush-gpu` and `flood-gpu` single-GPU probes on SAI-1344 are classified from actual `sbatch`/`sacct` evidence.
- DP-EVA renders Slurm array directives and submits one array per homogeneous labeling task class per attempt.
- DP-EVA normalizes generated Slurm job names with a tested helper.
- `mpi_abacus` task-class environment setup preserves `source /etc/profile` before SAI rank-map setup when that profile line is configured.
- A remote SAI-1344 conda environment imports DP-EVA from the staged repository and validates the FP11 config.
- A small SAI-1344 labeling smoke run uses the array backend and completes with `sacct` evidence.
- The full SAI-1344 FP11 labeling run is launched inside detached `tmux`; reconnecting is optional for the computation and monitor to continue.
- The backend performance report shows array submission count, total bundle count, array throttle, elapsed submit time, Slurm states, and whether the design is acceptable for FP11 production.

## File Structure

Create:

- `src/dpeva/submission/array.py`
  Manifest-backed Slurm array worker utilities.
- `src/dpeva/submission/names.py`
  Slurm job-name normalization.
- `tests/unit/submission/test_slurm_array.py`
  Unit tests for array manifests, worker, `sbatch` parsing, and status summaries.
- `tests/unit/submission/test_slurm_names.py`
  Unit tests for normalized Slurm job names.
- `scripts/fp11_1344_make_config.py`
  Local/remote config patcher for SAI-1344 FP11 execution.
- `scripts/fp11_1344_backend_report.py`
  Summarizes array submission counts, Slurm accounting records, and output counts.

Modify:

- `src/dpeva/config.py`
  Add optional `submission.slurm_array` and `submission.slurm_array_task_limit`.
- `src/dpeva/submission/templates.py`
  Add Slurm array rendering fields to `JobConfig`.
- `src/dpeva/submission/manager.py`
  Add `submit_array`, `parse_sbatch_job_id`, `query_active_slurm_ids`, `parse_sacct_records`, and summary helpers.
- `src/dpeva/submission/__init__.py`
  Export `ArrayTaskSpec` and `normalize_slurm_job_name`.
- `src/dpeva/workflows/labeling.py`
  Route Slurm labeling execution through grouped arrays and preserve SAI profile/rank-map ordering.
- `tests/unit/workflows/test_labeling_workflow.py`
  Add tests for array submission by class, fallback non-array behavior, normalized names, and SAI env ordering.
- `docs/guides/configuration.md`
  Document labeling task classes and array options.
- `docs/guides/slurm.md`
  Document Slurm array behavior, SAI-1344 profile ordering, and monitoring.
- `docs/archive/v0.8.1/plans/2026-07-02-fp11-first-principles-labeling.md`
  This implementation plan.

Remote-only files created under `/home/pku-jianghong/liuzhaoqing/fp11-sai1344`:

- `probes/qos-single-gpu/probe_rush_gpu_g1.sbatch`
- `probes/qos-single-gpu/probe_flood_gpu_g1.sbatch`
- `probes/qos-single-gpu/qos_single_gpu_probe_result.json`
- `dpeva/`
- `fp11/config_gpu_1344.json`
- `fp11/logs/fp11_execute_1344.log`
- `fp11/logs/fp11_execute_1344.pid`
- `fp11/backend_report_1344.json`
- `fp11/backend_report_1344.md`

---

### Task 1: Prove SAI-1344 Single-GPU QOS Behavior

**Files:**
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu/probe_rush_gpu_g1.sbatch`
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu/probe_flood_gpu_g1.sbatch`
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu/qos_single_gpu_probe_result.json`

- [ ] **Step 1: Confirm SAI-1344 login and partition state**

Run from the current SAI login node:

```bash
ssh sai-1344-tmp 'bash -lc "hostname; date +%F_%T; scontrol show partition 16V100 | sed -n \"1,80p\"; sinfo -p 16V100 -o \"%P %a %D %G %l\" | sed -n \"1,20p\""'
```

Expected:

```text
login-02.mr-sai.ai
PartitionName=16V100
AllowQos=...
```

- [ ] **Step 2: Create remote probe directory and input files**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
ROOT=/home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu
rm -rf \"$ROOT\"
mkdir -p \"$ROOT\"
cd \"$ROOT\"
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/INPUT .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/KPT .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/STRU .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/C_ONCV_PBE-1.0.upf .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/C_gga_7au_100Ry_2s2p1d.orb .
sed -i \"s/calculation[[:space:]]\\+relax/calculation             scf/\" INPUT
sed -i \"s/^16 16 1 0 0 0/1 1 1 0 0 0/\" KPT
printf \"probe_dir=%s\\n\" \"$ROOT\"
"'
```

Expected:

```text
probe_dir=/home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu
```

- [ ] **Step 3: Write single-GPU `rush-gpu` probe script**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "cat > /home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu/probe_rush_gpu_g1.sbatch <<'\''EOF'\''
#!/bin/bash
#SBATCH --job-name=fp11-qos-rush-g1
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-gpu
#SBATCH --time=00:05:00
source /etc/profile
module load abacus/LTSv3.10.1-sm70-auto
nvidia-smi -L
abacus > abacus_rush_gpu_g1.out 2>&1
EOF
"'
```

- [ ] **Step 4: Write single-GPU `flood-gpu` probe script**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "cat > /home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu/probe_flood_gpu_g1.sbatch <<'\''EOF'\''
#!/bin/bash
#SBATCH --job-name=fp11-qos-flood-g1
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-gpu
#SBATCH --time=00:05:00
source /etc/profile
module load abacus/LTSv3.10.1-sm70-auto
nvidia-smi -L
abacus > abacus_flood_gpu_g1.out 2>&1
EOF
"'
```

- [ ] **Step 5: Submit both probes**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu
RUSH_OUT=\$(sbatch probe_rush_gpu_g1.sbatch 2>&1 || true)
FLOOD_OUT=\$(sbatch probe_flood_gpu_g1.sbatch 2>&1 || true)
printf \"rush_submit=%s\\n\" \"\$RUSH_OUT\"
printf \"flood_submit=%s\\n\" \"\$FLOOD_OUT\"
RUSH_ID=\$(printf \"%s\\n\" \"\$RUSH_OUT\" | awk \"/Submitted batch job/{print \\\$4}\")
FLOOD_ID=\$(printf \"%s\\n\" \"\$FLOOD_OUT\" | awk \"/Submitted batch job/{print \\\$4}\")
{
  printf \"rush_out=%s\\n\" \"\$RUSH_OUT\"
  printf \"flood_out=%s\\n\" \"\$FLOOD_OUT\"
  printf \"rush_id=%s\\n\" \"\$RUSH_ID\"
  printf \"flood_id=%s\\n\" \"\$FLOOD_ID\"
} > probe_jobids.env
cat probe_jobids.env
"'
```

Expected success form, with numeric job IDs:

```text
rush_submit=Submitted batch job 575631
flood_submit=Submitted batch job 575632
rush_id=575631
flood_id=575632
```

If either output contains `QOSMinGRES`, `Invalid qos`, or `Batch job submission failed`, record that exact output in `qos_single_gpu_probe_result.json` and stop before using that QOS for production.

- [ ] **Step 6: Wait for probes and collect accounting**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu
source probe_jobids.env
if [ -z \"\${rush_id}\" ] || [ -z \"\${flood_id}\" ]; then
  printf \"missing job id; inspect probe_jobids.env\\n\" >&2
  exit 2
fi
while squeue -j \"\${rush_id},\${flood_id}\" --noheader | grep -q .; do
  date +%F_%T
  squeue -j \"\${rush_id},\${flood_id}\" -o \"%.18i %.32j %.8T %.10M %.6D %R\"
  sleep 30
done
sacct -j \"\${rush_id},\${flood_id}\" --format=JobID,JobName,Partition,QOS,State,ExitCode,Elapsed,AllocTRES%80 -P > sacct_probe.psv
cat sacct_probe.psv
"'
```

Expected acceptance for a supported single-GPU QOS:

```text
State=COMPLETED
ExitCode=0:0
AllocTRES contains gres/gpu=1
```

- [ ] **Step 7: Write the probe result JSON**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu
set -a
source probe_jobids.env
set +a
python - <<'\''PY'\''
import json
import os
from datetime import datetime
from pathlib import Path

root = Path.cwd()
rows = []
for line in (root / \"sacct_probe.psv\").read_text().splitlines():
    if not line.strip() or line.startswith(\"JobID|\"):
        continue
    fields = line.split(\"|\")
    if len(fields) >= 8:
        rows.append(
            {
                \"JobID\": fields[0],
                \"JobName\": fields[1],
                \"Partition\": fields[2],
                \"QOS\": fields[3],
                \"State\": fields[4],
                \"ExitCode\": fields[5],
                \"Elapsed\": fields[6],
                \"AllocTRES\": fields[7],
            }
        )

def summarize(job_id: str, submit_out: str) -> dict:
    main = next((row for row in rows if row[\"JobID\"] == job_id), {})
    return {
        \"job_id\": job_id,
        \"gpus_per_node\": 1,
        \"ntasks\": 1,
        \"accepted_by_sbatch\": bool(job_id),
        \"submit_output\": submit_out,
        \"completed\": main.get(\"State\") == \"COMPLETED\",
        \"exit_code\": main.get(\"ExitCode\", \"\"),
        \"alloc_tres\": main.get(\"AllocTRES\", \"\"),
    }

payload = {
    \"generated_at\": datetime.now().isoformat(timespec=\"seconds\"),
    \"partition\": \"16V100\",
    \"rush_gpu\": summarize(os.environ.get(\"rush_id\", \"\"), os.environ.get(\"rush_out\", \"\")),
    \"flood_gpu\": summarize(os.environ.get(\"flood_id\", \"\"), os.environ.get(\"flood_out\", \"\")),
    \"sacct_rows\": rows,
}
(root / \"qos_single_gpu_probe_result.json\").write_text(json.dumps(payload, indent=2) + \"\\n\")
print(json.dumps(payload, indent=2))
PY
"'
```

Expected supported result:

```json
"accepted_by_sbatch": true,
"completed": true,
"exit_code": "0:0"
```

### Task 2: Normalize Slurm Job Names And Preserve SAI Env Ordering

**Files:**
- Create: `src/dpeva/submission/names.py`
- Create: `tests/unit/submission/test_slurm_names.py`
- Modify: `src/dpeva/submission/__init__.py`
- Modify: `src/dpeva/workflows/labeling.py`
- Modify: `tests/unit/workflows/test_labeling_workflow.py`

- [ ] **Step 1: Add failing tests for Slurm job-name normalization**

Create `tests/unit/submission/test_slurm_names.py`:

```python
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
```

- [ ] **Step 2: Implement the name helper**

Create `src/dpeva/submission/names.py`:

```python
"""Slurm-safe naming helpers."""

from __future__ import annotations

import re


def normalize_slurm_job_name(raw: str, fallback: str = "dpeva-job", max_length: int = 128) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(raw)).strip("-._")
    value = re.sub(r"-{2,}", "-", value)
    if not value:
        value = fallback
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    return value[:max_length]
```

- [ ] **Step 3: Export the name helper**

Modify `src/dpeva/submission/__init__.py` to:

```python
from .templates import JobConfig
from .manager import JobManager
from .names import normalize_slurm_job_name

__all__ = ["JobConfig", "JobManager", "normalize_slurm_job_name"]
```

- [ ] **Step 4: Add a failing test for SAI profile before rank-map**

Append to `tests/unit/workflows/test_labeling_workflow.py`:

```python
@patch("dpeva.workflows.labeling.LabelingManager")
def test_mpi_abacus_env_keeps_profile_before_rank_map(MockManager, tmp_path):
    config = LabelingConfig(
        work_dir=str(tmp_path),
        input_data_path=str(tmp_path / "data"),
        submission={
            "backend": "slurm",
            "env_setup": [
                "source /etc/profile",
                "module load abacus/LTSv3.10.1-sm70-auto",
            ],
            "slurm_config": {"partition": "16V100", "qos": "rush-gpu"},
        },
        dft_params={},
        attempt_params=[],
        pp_dir="/tmp/pp",
        orb_dir="/tmp/orb",
        labeling_task_classes=[
            {
                "name": "highmem",
                "selector": {"min_atoms": 181},
                "launcher_mode": "mpi_abacus",
                "resource_mode": "multi_gpu_mpi",
            }
        ],
    )
    workflow = LabelingWorkflow(config)

    env_setup = workflow._env_setup_for_class(config.labeling_task_classes[0], "mpi_abacus")

    lines = env_setup.splitlines()
    assert lines.index("source /etc/profile") < lines.index(
        "source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash"
    )
```

- [ ] **Step 5: Modify `_env_setup_for_class` ordering**

In `src/dpeva/workflows/labeling.py`, replace the `mpi_abacus` branch in `_env_setup_for_class` with:

```python
        elif launcher_mode == "mpi_abacus":
            if not any("mps_mapping.d" in line for line in lines):
                profile_idx = next(
                    (idx for idx, line in enumerate(lines) if line.strip() == "source /etc/profile"),
                    None,
                )
                insert_idx = profile_idx + 1 if profile_idx is not None else 0
                lines.insert(insert_idx, SAI_RANK_MAP_SETUP)
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
pytest tests/unit/submission/test_slurm_names.py tests/unit/workflows/test_labeling_workflow.py::test_mpi_abacus_env_keeps_profile_before_rank_map -q
```

Expected:

```text
5 passed
```

- [ ] **Step 7: Commit**

Run:

```bash
git add src/dpeva/submission/names.py src/dpeva/submission/__init__.py src/dpeva/workflows/labeling.py tests/unit/submission/test_slurm_names.py tests/unit/workflows/test_labeling_workflow.py
git commit -m "feat: normalize slurm job names and order sai env setup"
```

### Task 3: Add Generic Slurm Array Submission Primitives

**Files:**
- Create: `src/dpeva/submission/array.py`
- Modify: `src/dpeva/submission/templates.py`
- Modify: `src/dpeva/submission/manager.py`
- Modify: `src/dpeva/submission/__init__.py`
- Create: `tests/unit/submission/test_slurm_array.py`
- Modify: `tests/unit/submission/test_job_manager.py`

- [ ] **Step 1: Add failing tests for Slurm array rendering**

Append to `tests/unit/submission/test_job_manager.py`:

```python
def test_generate_script_slurm_array_with_limit(tmp_path):
    config = JobConfig(
        job_name="array_job",
        command="echo $SLURM_ARRAY_TASK_ID",
        partition="debug",
        array="0-7",
        array_task_limit=3,
        output_log="logs/%A_%a.out",
        error_log="logs/%A_%a.err",
    )
    manager = JobManager(mode="slurm")
    output_path = tmp_path / "array.slurm"

    manager.generate_script(config, str(output_path))

    content = output_path.read_text()
    assert "#SBATCH --array=0-7%3" in content
    assert "#SBATCH -o logs/%A_%a.out" in content
    assert "#SBATCH -e logs/%A_%a.err" in content
    assert "echo $SLURM_ARRAY_TASK_ID" in content


def test_generate_script_slurm_array_without_limit(tmp_path):
    config = JobConfig(job_name="array_job", command="hostname", array="1,3,5")
    manager = JobManager(mode="slurm")
    output_path = tmp_path / "array.slurm"

    manager.generate_script(config, str(output_path))

    assert "#SBATCH --array=1,3,5" in output_path.read_text()
```

- [ ] **Step 2: Add array fields to `JobConfig`**

In `src/dpeva/submission/templates.py`, add fields after `custom_headers`:

```python
    array: Optional[str] = None
    array_task_limit: Optional[int] = None
```

Add this method inside `JobConfig`:

```python
    def _array_expression(self) -> Optional[str]:
        if not self.array:
            return None
        if self.array_task_limit is None:
            return self.array
        if self.array_task_limit < 1:
            raise ValueError("array_task_limit must be >= 1")
        return f"{self.array}%{self.array_task_limit}"
```

In `to_dict()`, insert this block after the existing `if self.error_log:` block:

```python
        array_expr = self._array_expression()
        if array_expr:
            optional_params.append(f"#SBATCH --array={array_expr}")
```

- [ ] **Step 3: Create array manifest tests**

Create `tests/unit/submission/test_slurm_array.py`:

```python
from unittest.mock import patch

import pytest

from dpeva.submission.array import (
    ArrayTaskSpec,
    build_array_command,
    load_array_manifest,
    run_array_task,
    write_array_manifest,
)
from dpeva.submission.manager import JobManager
from dpeva.submission.templates import JobConfig


def test_write_and_load_array_manifest(tmp_path):
    manifest_path = tmp_path / "tasks.json"
    tasks = [
        ArrayTaskSpec(index=0, name="bundle0", working_dir=tmp_path / "a", argv=["python", "-V"]),
        ArrayTaskSpec(index=1, name="bundle1", working_dir=tmp_path / "b", argv=["python", "-V"]),
    ]

    write_array_manifest(tasks, manifest_path)
    loaded = load_array_manifest(manifest_path)

    assert [task.index for task in loaded] == [0, 1]
    assert loaded[0].name == "bundle0"
    assert loaded[1].working_dir == tmp_path / "b"


def test_write_array_manifest_rejects_non_contiguous_indices(tmp_path):
    with pytest.raises(ValueError, match="contiguous"):
        write_array_manifest(
            [
                ArrayTaskSpec(index=0, name="a", working_dir=tmp_path, argv=["true"]),
                ArrayTaskSpec(index=2, name="b", working_dir=tmp_path, argv=["true"]),
            ],
            tmp_path / "tasks.json",
        )


def test_run_array_task_uses_slurm_array_task_id(tmp_path, monkeypatch):
    manifest_path = tmp_path / "tasks.json"
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    marker = work_dir / "ran.txt"
    script = work_dir / "worker.py"
    script.write_text("from pathlib import Path\nPath('ran.txt').write_text('ok')\n")
    write_array_manifest(
        [ArrayTaskSpec(index=0, name="bundle0", working_dir=work_dir, argv=["python", str(script)])],
        manifest_path,
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")

    run_array_task(manifest_path)

    assert marker.read_text() == "ok"


def test_build_array_command_uses_module_worker(tmp_path):
    command = build_array_command(tmp_path / "tasks.json")
    assert " -m dpeva.submission.array " in command
    assert str(tmp_path / "tasks.json") in command


def test_submit_array_writes_manifest_and_single_script(tmp_path):
    manager = JobManager(mode="slurm")
    tasks = [
        ArrayTaskSpec(index=0, name="a", working_dir=tmp_path / "a", argv=["python", "-V"]),
        ArrayTaskSpec(index=1, name="b", working_dir=tmp_path / "b", argv=["python", "-V"]),
    ]
    job_config = JobConfig(job_name="label_array", command="")
    manifest_path = tmp_path / "array" / "tasks.json"
    script_path = tmp_path / "array" / "submit.slurm"

    with patch.object(manager, "submit", return_value="Submitted batch job 24680") as mock_submit:
        job_id = manager.submit_array(
            tasks=tasks,
            job_config=job_config,
            manifest_path=str(manifest_path),
            script_path=str(script_path),
            working_dir=str(tmp_path),
            array_task_limit=1,
        )

    assert job_id == "24680"
    assert manifest_path.exists()
    assert script_path.exists()
    assert "#SBATCH --array=0-1%1" in script_path.read_text()
    mock_submit.assert_called_once_with(str(script_path), working_dir=str(tmp_path))
```

- [ ] **Step 4: Implement array manifest worker**

Create `src/dpeva/submission/array.py`:

```python
"""Manifest-based Slurm array worker utilities."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union


@dataclass(frozen=True)
class ArrayTaskSpec:
    index: int
    name: str
    working_dir: Path
    argv: Sequence[str]

    def to_json_dict(self) -> dict:
        if self.index < 0:
            raise ValueError("array task index must be >= 0")
        if not self.argv:
            raise ValueError("array task argv must not be empty")
        return {
            "index": self.index,
            "name": self.name,
            "working_dir": str(Path(self.working_dir).resolve()),
            "argv": [str(item) for item in self.argv],
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "ArrayTaskSpec":
        return cls(
            index=int(data["index"]),
            name=str(data["name"]),
            working_dir=Path(data["working_dir"]),
            argv=[str(item) for item in data["argv"]],
        )


def _validate_contiguous(tasks: List[ArrayTaskSpec]) -> None:
    actual = [task.index for task in tasks]
    expected = list(range(len(tasks)))
    if actual != expected:
        raise ValueError(f"array task indices must be contiguous from 0; got {actual}")


def write_array_manifest(tasks: Iterable[ArrayTaskSpec], manifest_path: Union[str, Path]) -> Path:
    manifest = Path(manifest_path)
    task_list = list(tasks)
    _validate_contiguous(task_list)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([task.to_json_dict() for task in task_list], indent=2) + "\n")
    return manifest


def load_array_manifest(manifest_path: Union[str, Path]) -> List[ArrayTaskSpec]:
    tasks = [ArrayTaskSpec.from_json_dict(item) for item in json.loads(Path(manifest_path).read_text())]
    _validate_contiguous(tasks)
    return tasks


def get_array_task(tasks: List[ArrayTaskSpec], task_id: Optional[int] = None) -> ArrayTaskSpec:
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise RuntimeError("SLURM_ARRAY_TASK_ID is not set")
        task_id = int(raw)
    for task in tasks:
        if task.index == task_id:
            return task
    raise IndexError(f"No array task with index {task_id}")


def run_array_task(manifest_path: Union[str, Path], task_id: Optional[int] = None) -> None:
    task = get_array_task(load_array_manifest(manifest_path), task_id=task_id)
    work_dir = Path(task.working_dir)
    if not work_dir.exists():
        raise FileNotFoundError(f"Array task working directory not found: {work_dir}")
    print(f"DPEVA array task {task.index}: {task.name}")
    print(f"Working directory: {work_dir}")
    print("Command: " + " ".join(shlex.quote(arg) for arg in task.argv))
    subprocess.run(list(task.argv), cwd=str(work_dir), check=True)


def build_array_command(manifest_path: Union[str, Path]) -> str:
    return f"{shlex.quote(sys.executable)} -m dpeva.submission.array {shlex.quote(str(Path(manifest_path).resolve()))}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("Usage: python -m dpeva.submission.array manifest.json", file=sys.stderr)
        return 2
    run_array_task(args[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Add `submit_array` to `JobManager`**

In `src/dpeva/submission/manager.py`, import:

```python
import re
from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional

from .array import ArrayTaskSpec, build_array_command, write_array_manifest
```

Add these dataclasses below the imports:

```python
SUCCESS_STATES = {"COMPLETED"}


@dataclass(frozen=True)
class SlurmJobRecord:
    job_id: str
    state: str
    exit_code: str
    array_task_id: Optional[int] = None


@dataclass(frozen=True)
class SlurmArraySummary:
    array_job_id: str
    expected_count: int
    completed: int
    failed: int
    missing_task_ids: List[int]
    failed_task_ids: List[int]

    @property
    def ok(self) -> bool:
        return self.failed == 0 and not self.missing_task_ids and self.completed == self.expected_count
```

Add these methods to `JobManager`:

```python
    @staticmethod
    def parse_sbatch_job_id(output: str) -> str:
        match = re.search(r"\b(\d+)\b", output)
        if not match:
            raise ValueError(f"Could not parse Slurm job id from output: {output!r}")
        return match.group(1)

    def submit_array(
        self,
        tasks: Iterable[ArrayTaskSpec],
        job_config: JobConfig,
        manifest_path: str,
        script_path: str,
        working_dir: str = ".",
        array_task_limit: Optional[int] = None,
    ) -> str:
        task_list = list(tasks)
        if not task_list:
            raise ValueError("Cannot submit an empty Slurm array")
        manifest = write_array_manifest(task_list, manifest_path)
        job_config.command = build_array_command(manifest)
        job_config.array = f"0-{len(task_list) - 1}"
        if array_task_limit is not None:
            job_config.array_task_limit = array_task_limit
        if job_config.output_log is None:
            job_config.output_log = "slurm-%A_%a.out"
        if job_config.error_log is None:
            job_config.error_log = "slurm-%A_%a.err"
        self.generate_script(job_config, script_path)
        return self.parse_sbatch_job_id(self.submit(script_path, working_dir=str(working_dir)))
```

- [ ] **Step 6: Export array helpers**

Modify `src/dpeva/submission/__init__.py` to:

```python
from .templates import JobConfig
from .manager import JobManager
from .array import ArrayTaskSpec
from .names import normalize_slurm_job_name

__all__ = ["JobConfig", "JobManager", "ArrayTaskSpec", "normalize_slurm_job_name"]
```

- [ ] **Step 7: Run submission tests**

Run:

```bash
pytest tests/unit/submission/test_job_manager.py tests/unit/submission/test_slurm_array.py tests/unit/submission/test_slurm_names.py -q
```

Expected:

```text
passed
```

- [ ] **Step 8: Commit**

Run:

```bash
git add src/dpeva/submission tests/unit/submission
git commit -m "feat: add slurm array submission primitives"
```

### Task 4: Route Labeling Slurm Execution Through Arrays

**Files:**
- Modify: `src/dpeva/config.py`
- Modify: `src/dpeva/workflows/labeling.py`
- Modify: `tests/unit/workflows/test_labeling_workflow.py`

- [ ] **Step 1: Add submission config fields**

Modify `SubmissionConfig` in `src/dpeva/config.py`:

```python
    slurm_array: bool = Field(
        False,
        description="When true, workflows that support homogeneous Slurm arrays may submit arrays instead of one job per task.",
    )
    slurm_array_task_limit: Optional[int] = Field(
        None,
        gt=0,
        description="Optional Slurm array throttle, rendered as --array=0-N%limit.",
    )
```

- [ ] **Step 2: Add failing test for class-grouped array submission**

Append to `tests/unit/workflows/test_labeling_workflow.py`:

```python
@patch("dpeva.workflows.labeling.LabelingManager")
def test_submit_job_dirs_uses_arrays_grouped_by_task_class(MockManager, tmp_path):
    config = LabelingConfig(
        work_dir=str(tmp_path),
        input_data_path=str(tmp_path / "data"),
        submission={
            "backend": "slurm",
            "slurm_array": True,
            "slurm_array_task_limit": 2,
            "env_setup": ["source /etc/profile", "module load abacus/LTSv3.10.1-sm70-auto"],
            "slurm_config": {"partition": "16V100", "qos": "flood-gpu", "walltime": "02:00:00"},
        },
        dft_params={},
        attempt_params=[],
        pp_dir="/tmp/pp",
        orb_dir="/tmp/orb",
        labeling_task_classes=[
            {
                "name": "normal",
                "selector": {"max_atoms": 180},
                "launcher_mode": "abacus",
                "resource_mode": "single_gpu",
                "slurm_config": {"ntasks": 1, "gpus_per_node": 1, "qos": "flood-gpu"},
            },
            {
                "name": "highmem",
                "selector": {"min_atoms": 181},
                "launcher_mode": "mpi_abacus",
                "resource_mode": "multi_gpu_mpi",
                "slurm_config": {"ntasks": 4, "gpus_per_node": 4, "qos": "rush-gpu"},
            },
        ],
    )
    normal_0 = tmp_path / "inputs" / "normal" / "N_1_0"
    normal_1 = tmp_path / "inputs" / "normal" / "N_1_1"
    high_0 = tmp_path / "inputs" / "highmem" / "N_1_0"
    for path in (normal_0, normal_1, high_0):
        path.mkdir(parents=True)
        (path / ".dpeva_job_class.json").write_text('{"task_class": "' + path.parent.name + '"}')
    manager = MockManager.return_value
    manager.generate_runner_script.side_effect = ["print('n0')", "print('n1')", "print('h0')"]
    workflow = LabelingWorkflow(config)
    workflow.job_manager.submit_array = MagicMock(side_effect=["100", "200"])

    job_ids = workflow._submit_job_dirs([normal_0, normal_1, high_0], attempt=0)

    assert job_ids == ["100", "200"]
    normal_call = workflow.job_manager.submit_array.call_args_list[0]
    high_call = workflow.job_manager.submit_array.call_args_list[1]
    assert len(normal_call.kwargs["tasks"]) == 2
    assert normal_call.kwargs["job_config"].job_name == "fp-normal-att0"
    assert normal_call.kwargs["job_config"].gpus_per_node == 1
    assert normal_call.kwargs["array_task_limit"] == 2
    assert len(high_call.kwargs["tasks"]) == 1
    assert high_call.kwargs["job_config"].job_name == "fp-highmem-att0"
    assert high_call.kwargs["job_config"].gpus_per_node == 4
```

- [ ] **Step 3: Add array grouping helpers to `LabelingWorkflow`**

In `src/dpeva/workflows/labeling.py`, import:

```python
import sys
from dpeva.submission.array import ArrayTaskSpec
from dpeva.submission.names import normalize_slurm_job_name
```

Add this method to `LabelingWorkflow`:

```python
    def _submit_job_dirs_as_arrays(self, active_job_dirs: List[Path], attempt: int) -> List[str]:
        grouped: Dict[str, List[Path]] = {}
        class_config_by_key: Dict[str, Any] = {}
        for job_dir in active_job_dirs:
            class_config = self._task_class_config_for_job_dir(job_dir)
            class_name = self._class_value(class_config, "name", "default")
            grouped.setdefault(class_name, []).append(job_dir)
            class_config_by_key[class_name] = class_config

        job_ids: List[str] = []
        for class_name, job_dirs in sorted(grouped.items()):
            class_config = class_config_by_key[class_name]
            launcher_mode = self._class_value(class_config, "launcher_mode", "auto")
            slurm_conf = self._slurm_config_for_class(class_config)
            env_setup = self._env_setup_for_class(class_config, launcher_mode)
            tasks: List[ArrayTaskSpec] = []
            for idx, job_dir in enumerate(sorted(job_dirs, key=lambda path: str(path))):
                runner_content = self.manager.generate_runner_script(job_dir, launcher_mode=launcher_mode)
                runner_path = job_dir / "run_batch.py"
                runner_path.write_text(runner_content, encoding="utf-8")
                tasks.append(
                    ArrayTaskSpec(
                        index=idx,
                        name=job_dir.name,
                        working_dir=job_dir,
                        argv=[sys.executable, "-u", str(runner_path.resolve())],
                    )
                )
            array_root = Path(self.config.work_dir) / "array_jobs" / f"{class_name}_attempt_{attempt}"
            job_config = JobConfig(
                command="",
                job_name=normalize_slurm_job_name(f"fp-{class_name}-att{attempt}"),
                partition=slurm_conf.get("partition"),
                qos=slurm_conf.get("qos"),
                nodes=slurm_conf.get("nodes", 1),
                ntasks=slurm_conf.get("ntasks", 1),
                gpus_per_node=slurm_conf.get("gpus_per_node"),
                cpus_per_task=slurm_conf.get("cpus_per_task"),
                walltime=slurm_conf.get("walltime", "24:00:00"),
                env_setup=env_setup,
            )
            try:
                job_id = self.job_manager.submit_array(
                    tasks=tasks,
                    job_config=job_config,
                    manifest_path=str(array_root / "tasks.json"),
                    script_path=str(array_root / "submit_array.slurm"),
                    working_dir=str(array_root),
                    array_task_limit=self.config.submission.slurm_array_task_limit,
                )
                job_ids.append(job_id)
            except Exception as e:
                logger.error(f"Failed to submit array for class {class_name}: {e}")
        return job_ids
```

- [ ] **Step 4: Route `_submit_job_dirs` through array mode**

At the top of `_submit_job_dirs` in `src/dpeva/workflows/labeling.py`, add:

```python
        if self.config.submission.backend == "slurm" and self.config.submission.slurm_array:
            return self._submit_job_dirs_as_arrays(active_job_dirs, attempt)
```

- [ ] **Step 5: Run labeling workflow tests**

Run:

```bash
pytest tests/unit/workflows/test_labeling_workflow.py -q
```

Expected:

```text
passed
```

- [ ] **Step 6: Commit**

Run:

```bash
git add src/dpeva/config.py src/dpeva/workflows/labeling.py tests/unit/workflows/test_labeling_workflow.py
git commit -m "feat: submit labeling bundles as slurm arrays"
```

### Task 5: Build The SAI-1344 FP11 Config Patcher

**Files:**
- Create: `scripts/fp11_1344_make_config.py`
- Test: `tests/unit/scripts/test_fp11_1344_make_config.py`

- [ ] **Step 1: Add config patcher tests**

Create `tests/unit/scripts/test_fp11_1344_make_config.py`:

```python
import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "fp11_1344_make_config.py"


def load_module():
    spec = importlib.util.spec_from_file_location("fp11_1344_make_config", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_patch_config_for_1344_uses_single_gpu_flood_when_probe_passed():
    module = load_module()
    config = {
        "work_dir": "labeling_workdir_normal_g1",
        "submission": {
            "backend": "slurm",
            "env_setup": ["module load abacus/LTSv3.10.1-sm70-auto"],
            "slurm_config": {"partition": "4V100", "qos": "flood-1o2gpu"},
        },
        "labeling_task_classes": [
            {
                "name": "normal",
                "launcher_mode": "abacus",
                "resource_mode": "single_gpu",
                "slurm_config": {"ntasks": 1, "gpus_per_node": 1, "qos": "flood-1o2gpu"},
            },
            {
                "name": "highmem",
                "launcher_mode": "mpi_abacus",
                "resource_mode": "multi_gpu_mpi",
                "slurm_config": {"ntasks": 4, "gpus_per_node": 4, "qos": "flood-gpu"},
            },
        ],
    }
    probe = {
        "rush_gpu": {"accepted_by_sbatch": True, "completed": True, "exit_code": "0:0"},
        "flood_gpu": {"accepted_by_sbatch": True, "completed": True, "exit_code": "0:0"},
    }

    patched = module.patch_config_for_1344(config, probe)

    assert patched["work_dir"] == "labeling_workdir_1344"
    assert patched["submission"]["slurm_array"] is True
    assert patched["submission"]["slurm_array_task_limit"] == 128
    assert patched["submission"]["slurm_config"]["partition"] == "16V100"
    assert patched["submission"]["slurm_config"]["qos"] == "flood-gpu"
    assert patched["submission"]["env_setup"][0] == "source /etc/profile"
    normal = patched["labeling_task_classes"][0]
    assert normal["slurm_config"]["partition"] == "16V100"
    assert normal["slurm_config"]["qos"] == "flood-gpu"
    assert normal["slurm_config"]["gpus_per_node"] == 1
    assert normal["launcher_mode"] == "abacus"


def test_patch_config_for_1344_rejects_failed_single_gpu_probe():
    module = load_module()
    probe = {"flood_gpu": {"accepted_by_sbatch": False, "completed": False, "exit_code": ""}}
    try:
        module.patch_config_for_1344({"submission": {}, "labeling_task_classes": []}, probe)
    except ValueError as exc:
        assert "flood-gpu single-GPU probe did not pass" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Implement the config patcher**

Create `scripts/fp11_1344_make_config.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _probe_passed(probe: dict[str, Any], key: str) -> bool:
    item = probe.get(key, {})
    return bool(item.get("accepted_by_sbatch") and item.get("completed") and item.get("exit_code") == "0:0")


def _ensure_profile(env_setup: Any) -> list[str]:
    if isinstance(env_setup, str):
        lines = [line for line in env_setup.splitlines() if line.strip()]
    elif isinstance(env_setup, list):
        lines = [str(line) for line in env_setup if str(line).strip()]
    else:
        lines = []
    if "source /etc/profile" not in lines:
        lines.insert(0, "source /etc/profile")
    if not any("module load abacus/LTSv3.10.1-sm70-auto" in line for line in lines):
        lines.append("module load abacus/LTSv3.10.1-sm70-auto")
    return lines


def patch_config_for_1344(config: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    if not _probe_passed(probe, "flood_gpu"):
        raise ValueError("flood-gpu single-GPU probe did not pass; do not create production single-GPU config")

    patched = copy.deepcopy(config)
    patched["work_dir"] = "labeling_workdir_1344"
    submission = patched.setdefault("submission", {})
    submission["backend"] = "slurm"
    submission["slurm_array"] = True
    submission["slurm_array_task_limit"] = int(submission.get("slurm_array_task_limit") or 128)
    submission["env_setup"] = _ensure_profile(submission.get("env_setup", []))
    slurm = submission.setdefault("slurm_config", {})
    slurm.update(
        {
            "partition": "16V100",
            "qos": "flood-gpu",
            "ntasks": 1,
            "gpus_per_node": 1,
            "walltime": "04:00:00",
        }
    )

    task_classes = patched.setdefault("labeling_task_classes", [])
    if not task_classes:
        task_classes.extend(
            [
                {
                    "name": "normal",
                    "selector": {"max_atoms": 180},
                    "tasks_per_job": 1,
                    "launcher_mode": "abacus",
                    "resource_mode": "single_gpu",
                    "slurm_config": {},
                },
                {
                    "name": "highmem",
                    "selector": {"min_atoms": 181},
                    "tasks_per_job": 1,
                    "launcher_mode": "mpi_abacus",
                    "resource_mode": "multi_gpu_mpi",
                    "slurm_config": {},
                },
            ]
        )
    for task_class in task_classes:
        class_slurm = task_class.setdefault("slurm_config", {})
        class_slurm["partition"] = "16V100"
        if task_class.get("resource_mode") == "multi_gpu_mpi":
            task_class["launcher_mode"] = "mpi_abacus"
            class_slurm.update({"ntasks": 4, "gpus_per_node": 4, "qos": "rush-gpu", "walltime": "04:00:00"})
        else:
            task_class["launcher_mode"] = "abacus"
            task_class["resource_mode"] = "single_gpu"
            class_slurm.update({"ntasks": 1, "gpus_per_node": 1, "qos": "flood-gpu", "walltime": "04:00:00"})
    return patched


def main() -> int:
    root = Path.cwd()
    source = root / "config_gpu.json"
    probe_path = root / "probes" / "qos-single-gpu" / "qos_single_gpu_probe_result.json"
    output = root / "config_gpu_1344.json"
    config = json.loads(source.read_text(encoding="utf-8"))
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    output.write_text(json.dumps(patch_config_for_1344(config, probe), indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Run patcher tests**

Run:

```bash
pytest tests/unit/scripts/test_fp11_1344_make_config.py -q
```

Expected:

```text
2 passed
```

- [ ] **Step 4: Commit**

Run:

```bash
git add scripts/fp11_1344_make_config.py tests/unit/scripts/test_fp11_1344_make_config.py
git commit -m "feat: add fp11 sai-1344 config patcher"
```

### Task 6: Transfer Repository, Data, And PP/ORB To SAI-1344

**Files:**
- Create remote directory: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva`
- Create remote directory: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11`
- Create remote file: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/config_gpu_1344.json`

- [ ] **Step 1: Transfer the DP-EVA repository**

Run from `/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva`:

```bash
rsync -a --delete --info=progress2 \
  --exclude .git \
  --exclude build \
  --exclude __pycache__ \
  --exclude .pytest_cache \
  /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/ \
  sai-1344-tmp:/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva/
```

Expected:

```text
sent ... bytes
```

- [ ] **Step 2: Transfer FP11 input data and helper files**

Run:

```bash
rsync -a --info=progress2 \
  /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/sampled_dpdata \
  /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/config_gpu.json \
  /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_config_audit.py \
  /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/fp11_benchmark_summary.json \
  sai-1344-tmp:/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/
```

Expected:

```text
sampled_dpdata/
config_gpu.json
```

- [ ] **Step 3: Transfer PP/ORB files**

Run:

```bash
rsync -a --info=progress2 \
  /home/pku-jianghong/liuzhaoqing/PP_ORB/ \
  sai-1344-tmp:/home/pku-jianghong/liuzhaoqing/PP_ORB/
```

Expected:

```text
PP/
ORB/
```

- [ ] **Step 4: Transfer probe result into FP11 staging**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
mkdir -p /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/probes/qos-single-gpu
cp /home/pku-jianghong/liuzhaoqing/fp11-sai1344/probes/qos-single-gpu/qos_single_gpu_probe_result.json /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/probes/qos-single-gpu/
"'
```

- [ ] **Step 5: Create SAI-1344 config**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
PYTHONPATH=/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva/src python /home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva/scripts/fp11_1344_make_config.py
python -m json.tool config_gpu_1344.json >/dev/null
grep -n \"slurm_array\" config_gpu_1344.json
"'
```

Expected:

```text
/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/config_gpu_1344.json
```

### Task 7: Build Or Validate Remote Conda Environment

**Files:**
- Use remote conda env: `dpeva-dpa4`

- [ ] **Step 1: Check whether `dpeva-dpa4` exists on SAI-1344**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "source /etc/profile; conda env list | sed -n \"1,120p\"; conda run -n dpeva-dpa4 python -c \"import sys; print(sys.executable)\" 2>/dev/null || true"'
```

Expected if present:

```text
dpeva-dpa4
.../envs/dpeva-dpa4/bin/python
```

- [ ] **Step 2: Create `dpeva-dpa4` if missing**

Run only if Step 1 did not find the environment:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
source /etc/profile
conda create -y -n dpeva-dpa4 python=3.12
conda run -n dpeva-dpa4 python -m pip install --upgrade pip
conda run -n dpeva-dpa4 python -m pip install -e /home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva
"'
```

Expected:

```text
Successfully installed dpeva
```

- [ ] **Step 3: Upgrade editable DP-EVA install from staged repository**

Run whether the environment existed or was newly created:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
source /etc/profile
conda run -n dpeva-dpa4 python -m pip install -e /home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva
conda run -n dpeva-dpa4 python -c \"import dpeva, inspect; print(inspect.getfile(dpeva))\"
"'
```

Expected:

```text
/home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva/src/dpeva/__init__.py
```

- [ ] **Step 4: Validate SAI-1344 FP11 config**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 python - <<'\''PY'\''
from dpeva.cli import load_and_resolve_config
from dpeva.config import LabelingConfig
cfg = LabelingConfig(**load_and_resolve_config(\"config_gpu_1344.json\"))
print(cfg.work_dir)
print(cfg.submission.slurm_array)
print(cfg.submission.slurm_array_task_limit)
print(cfg.submission.slurm_config)
PY
"'
```

Expected:

```text
labeling_workdir_1344
True
128
```

### Task 8: Run A Small SAI-1344 Array Smoke Test

**Files:**
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/config_gpu_1344_smoke.json`
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/sampled_dpdata_smoke`
- Read remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/labeling_workdir_1344_smoke`

- [ ] **Step 1: Create a deterministic 4-frame smoke dataset**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
conda run -n dpeva-dpa4 python - <<'\''PY'\''
import json
import shutil
from pathlib import Path

root = Path.cwd()
src = root / \"sampled_dpdata\"
dst = root / \"sampled_dpdata_smoke\"
if dst.exists():
    shutil.rmtree(dst)
selected = []
for type_raw in sorted(src.rglob(\"type.raw\")):
    system = type_raw.parent
    atoms = len(type_raw.read_text().split())
    if atoms <= 120:
        selected.append(system)
    if len(selected) == 4:
        break
for system in selected:
    target = dst / system.relative_to(src)
    shutil.copytree(system, target)
cfg = json.loads((root / \"config_gpu_1344.json\").read_text())
cfg[\"input_data_path\"] = \"sampled_dpdata_smoke\"
cfg[\"work_dir\"] = \"labeling_workdir_1344_smoke\"
cfg[\"tasks_per_job\"] = 1
for item in cfg.get(\"labeling_task_classes\", []):
    item[\"tasks_per_job\"] = 1
cfg[\"integration_enabled\"] = False
cfg[\"existing_training_data_path\"] = None
cfg[\"merged_training_data_path\"] = None
(root / \"config_gpu_1344_smoke.json\").write_text(json.dumps(cfg, indent=2) + \"\\n\")
print(\"systems\", len(selected))
print(root / \"config_gpu_1344_smoke.json\")
PY
"'
```

Expected:

```text
systems 4
config_gpu_1344_smoke.json
```

- [ ] **Step 2: Prepare smoke tasks**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 dpeva label config_gpu_1344_smoke.json --stage prepare
find labeling_workdir_1344_smoke/inputs -maxdepth 3 -type d -name \"N_*\" | sort
"'
```

Expected:

```text
N_1_
```

- [ ] **Step 3: Execute smoke tasks with array backend**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 dpeva label config_gpu_1344_smoke.json --stage execute
"'
```

Expected:

```text
Submitted batch job 575700
All jobs finished.
```

- [ ] **Step 4: Verify smoke used Slurm array artifacts**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
find labeling_workdir_1344_smoke/array_jobs -type f | sort
grep -R \"#SBATCH --array\" -n labeling_workdir_1344_smoke/array_jobs
"'
```

Expected:

```text
tasks.json
submit_array.slurm
#SBATCH --array=0-
```

- [ ] **Step 5: Extract and postprocess smoke results**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 dpeva label config_gpu_1344_smoke.json --stage extract
conda run -n dpeva-dpa4 dpeva label config_gpu_1344_smoke.json --stage postprocess
test -s labeling_workdir_1344_smoke/outputs/labeling_stats_report.json
"'
```

Expected:

```text
Extraction summary:
```

### Task 9: Launch Detached Full FP11 Labeling On SAI-1344

**Files:**
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/logs/fp11_execute_1344.log`
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/logs/fp11_execute_1344.pid`

- [ ] **Step 1: Prepare full FP11 tasks**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 dpeva label config_gpu_1344.json --stage prepare
find labeling_workdir_1344/inputs -maxdepth 3 -type d -name \"N_*\" | wc -l
"'
```

Expected:

```text
4829
```

If `tasks_per_job` groups multiple tasks per bundle, expected bundle count is less than 4829. Record the actual count in `backend_report_1344.md`.

- [ ] **Step 2: Launch detached execute monitor with tmux**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
mkdir -p logs
tmux has-session -t fp11_1344 2>/dev/null && tmux kill-session -t fp11_1344
tmux new-session -d -s fp11_1344 '\''cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11 && source /etc/profile && conda run -n dpeva-dpa4 dpeva label config_gpu_1344.json --stage execute > logs/fp11_execute_1344.log 2>&1'\''
tmux list-sessions | grep fp11_1344
"'
```

Expected:

```text
fp11_1344:
```

- [ ] **Step 3: Verify the monitor survives SSH disconnect**

Open a new SSH command after closing the previous connection:

```bash
ssh sai-1344-tmp 'bash -lc "tmux list-sessions | grep fp11_1344; tail -n 40 /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/logs/fp11_execute_1344.log"'
```

Expected:

```text
Submitting
Submitted batch job
Monitoring
```

- [ ] **Step 4: Monitor Slurm state without attaching**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "squeue -u $USER -o \"%.18i %.32j %.8T %.10M %.6D %R\" | grep -E \"fp-(normal|highmem)-att\" || true"'
```

Expected while running or pending:

```text
fp-normal-att0
fp-highmem-att0
```

- [ ] **Step 5: Attach only when interactive inspection is needed**

Run:

```bash
ssh -t sai-1344-tmp 'tmux attach -t fp11_1344'
```

Detach from tmux with:

```text
Ctrl-b d
```

### Task 10: Extract, Postprocess, And Produce Backend Report

**Files:**
- Create: `scripts/fp11_1344_backend_report.py`
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/backend_report_1344.json`
- Create remote: `/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/backend_report_1344.md`
- Modify: `docs/guides/slurm.md`
- Modify: `docs/guides/configuration.md`

- [ ] **Step 1: Add backend report script**

Create `scripts/fp11_1344_backend_report.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    return result.stdout


def main() -> int:
    root = Path.cwd()
    work_dir = root / "labeling_workdir_1344"
    array_scripts = sorted(work_dir.glob("array_jobs/*/submit_array.slurm"))
    manifests = sorted(work_dir.glob("array_jobs/*/tasks.json"))
    task_count = 0
    for manifest in manifests:
        task_count += len(json.loads(manifest.read_text()))
    job_names = []
    for script in array_scripts:
        for line in script.read_text().splitlines():
            if line.startswith("#SBATCH -J "):
                job_names.append(line.replace("#SBATCH -J ", "").strip())
    sacct = run(["sacct", "-X", "--format=JobID,JobName%30,State,ExitCode,Elapsed", "-S", "2026-07-04T00:00:00", "-P"])
    stats_path = work_dir / "outputs" / "labeling_stats_report.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    payload = {
        "work_dir": str(work_dir),
        "array_script_count": len(array_scripts),
        "manifest_count": len(manifests),
        "array_task_count": task_count,
        "job_names": job_names,
        "stats_report_exists": stats_path.exists(),
        "stats_global": stats.get("global", {}),
        "sacct_tail": sacct.splitlines()[-200:],
    }
    (root / "backend_report_1344.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# FP11 SAI-1344 Backend Report",
        "",
        f"- Work dir: `{work_dir}`",
        f"- Array scripts: {len(array_scripts)}",
        f"- Array manifests: {len(manifests)}",
        f"- Array tasks: {task_count}",
        f"- Job names: {', '.join(job_names) if job_names else 'none'}",
        f"- Stats report exists: {stats_path.exists()}",
        "",
        "## Accounting Tail",
        "",
        "```text",
        *payload["sacct_tail"],
        "```",
    ]
    (root / "backend_report_1344.md").write_text("\n".join(lines) + "\n")
    print(root / "backend_report_1344.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run extract and postprocess after execute finishes**

Run after `logs/fp11_execute_1344.log` shows `All active tasks converged.` or after all attempts finish:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 dpeva label config_gpu_1344.json --stage extract
conda run -n dpeva-dpa4 dpeva label config_gpu_1344.json --stage postprocess
"'
```

Expected:

```text
Extraction summary:
```

- [ ] **Step 3: Generate backend report**

Run:

```bash
ssh sai-1344-tmp 'bash -lc "set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11
source /etc/profile
conda run -n dpeva-dpa4 python /home/pku-jianghong/liuzhaoqing/fp11-sai1344/dpeva/scripts/fp11_1344_backend_report.py
sed -n \"1,160p\" backend_report_1344.md
"'
```

Expected:

```text
# FP11 SAI-1344 Backend Report
- Array scripts:
- Array tasks:
```

- [ ] **Step 4: Copy report back to local repo workspace**

Run:

```bash
rsync -a \
  sai-1344-tmp:/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/backend_report_1344.json \
  sai-1344-tmp:/home/pku-jianghong/liuzhaoqing/fp11-sai1344/fp11/backend_report_1344.md \
  /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/iter10-dpa4/fp11/
```

Expected:

```text
backend_report_1344.json
backend_report_1344.md
```

- [ ] **Step 5: Update Slurm docs**

In `docs/guides/slurm.md`, add this section after "5.2 Monitor completion marker":

```markdown
### 5.3 Labeling Slurm Arrays

Labeling can submit packed ABACUS bundles as Slurm arrays when `submission.slurm_array=true`.
DP-EVA writes one array manifest per homogeneous task class and attempt under
`labeling_workdir_1344/array_jobs/normal_attempt_0/tasks.json`, then submits one
`submit_array.slurm` script for that group.

On SAI-1344, include `source /etc/profile` in `submission.env_setup` before module loads:

```json
"env_setup": [
  "source /etc/profile",
  "module load abacus/LTSv3.10.1-sm70-auto"
]
```

The generated Slurm script renders all `#SBATCH` lines before `env_setup`, so
`source /etc/profile` remains after scheduler directives, as required by SAI-1344.
For `launcher_mode="mpi_abacus"`, DP-EVA inserts the SAI rank-map line after
`source /etc/profile` when that profile line is present.
```

- [ ] **Step 6: Update configuration docs**

In `docs/guides/configuration.md`, extend the SAI ABACUS labeling example with:

```json
"submission": {
  "backend": "slurm",
  "slurm_array": true,
  "slurm_array_task_limit": 128,
  "env_setup": [
    "source /etc/profile",
    "module load abacus/LTSv3.10.1-sm70-auto"
  ],
  "slurm_config": {
    "partition": "16V100",
    "qos": "flood-gpu",
    "walltime": "04:00:00"
  }
}
```

- [ ] **Step 7: Run local verification**

Run:

```bash
pytest tests/unit/submission tests/unit/workflows/test_labeling_workflow.py tests/unit/scripts/test_fp11_1344_make_config.py -q
```

Expected:

```text
passed
```

- [ ] **Step 8: Commit**

Run:

```bash
git add scripts/fp11_1344_backend_report.py docs/guides/slurm.md docs/guides/configuration.md
git commit -m "docs: document sai-1344 labeling array execution"
```

## Self-Review

Spec coverage:

- Single-GPU `flood-gpu` and `rush-gpu` validation is covered by Task 1 with real `sbatch` and `sacct` evidence.
- Remote FP11 transfer, remote conda validation/rebuild, and staged config generation are covered by Tasks 6 and 7.
- Detached long-running execution is covered by Task 9 through `tmux`.
- Slurm-native backend design, array support, and job-name normalization are covered by Tasks 2, 3, and 4.
- Backend performance and elegance are judged by Task 10 with concrete array counts, job names, accounting, and output reports.

Placeholder scan:

- This plan does not rely on open-ended implementation notes. Every code change includes exact test or implementation snippets.

Type consistency:

- `submission.slurm_array` and `submission.slurm_array_task_limit` are defined on `SubmissionConfig` and used from `self.config.submission`.
- `ArrayTaskSpec` fields are `index`, `name`, `working_dir`, and `argv` throughout.
- `normalize_slurm_job_name` is exported from `dpeva.submission` and imported by `LabelingWorkflow`.
