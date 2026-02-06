# DP-EVA Slurm Workflow Automation Design Document

## 1. Context and Problem Statement

### Current Status
DP-EVA currently supports two backends:
1.  **Local**: Executes commands directly (blocking/synchronous).
2.  **Slurm**: Generates batch scripts and submits them via `sbatch` (fire-and-forget/asynchronous).

### The Gap
While the **Local** backend supports synchronous execution, the **Slurm** backend currently lacks a mechanism to track job completion.
- `CollectionWorkflow`: Submits a wrapper script and exits.
- `TrainingWorkflow` / `InferenceWorkflow`: Submits multiple batch jobs and exits immediately.
- Explicit warning in code: `"Blocking wait is not yet implemented for Slurm backend."`

### Objective
Enable **automatic workflow chaining** on Slurm (e.g., `Train -> Wait -> Infer + Feature -> Wait -> Collect`) by implementing a mechanism to **monitor and wait** for Slurm jobs to complete.

---

## 2. System Analysis

### 2.1 Workflow Slurm Support Matrix

Based on a comprehensive code review of `src/dpeva`, here is the current support status:

| Workflow | Backend Support | Submission Logic | Monitoring Status |
| :--- | :--- | :--- | :--- |
| **Collection** | ✅ Full | Uses `_submit_to_slurm` to submit a self-wrapper. | ❌ None (Fire & Forget) |
| **Training** | ✅ Full | `ParallelTrainer` submits N jobs (one per model). | ❌ None (Fire & Forget) |
| **Inference** | ✅ Full | Iterates models and submits N inference jobs. | ❌ None (Fire & Forget) |
| **Feature** | ✅ Full | `DescriptorGenerator` supports CLI (`dp eval-desc`) & Python modes on Slurm. | ❌ None (Fire & Forget) |

### 2.2 Functional Coupling & Data Flow

The workflows are tightly coupled via the file system. Automation requires ensuring that **Output X** is fully written before **Input Y** reads it.

1.  **Feature Generation** (`feature.py`)
    *   **Input**: Raw Data (`type.raw`, coords) + Model
    *   **Output**: Descriptor Files (`*.npy`)
    *   **Downstream**: *CollectionWorkflow* (Requires descriptors to be fully written)

2.  **Inference** (`infer.py`)
    *   **Input**: Test Data + Models (`*.pb`/`*.pt`)
    *   **Output**: Prediction Results (Energy/Force text files)
    *   **Downstream**: *CollectionWorkflow* (Requires prediction results to compute UQ)

3.  **Collection** (`collect.py`)
    *   **Input**: Prediction Results + Descriptors
    *   **Output**: Sampled Dataset (`sampled_dpdata`)
    *   **Downstream**: *TrainingWorkflow* (Requires sampled data for labeling/training)

4.  **Training** (`train.py`)
    *   **Input**: Sampled Dataset (Labeled) + Base Model
    *   **Output**: New Models (`frozen_model.pb`)
    *   **Downstream**: *InferenceWorkflow* (Requires new models)

---

## 3. Technical Implementation Plan

### 3.1. Enhanced Submission Manager (`dpeva.submission`)

The `JobManager` must provide structured feedback about submitted jobs.

**A. Regex for Job ID**
Slurm `sbatch` typically outputs: `Submitted batch job 123456`.
Regex: `r"Submitted batch job (\d+)"`

**B. Data Structures**
```python
@dataclass
class SubmissionResult:
    job_id: str          # Slurm Job ID (or PID for local)
    backend: str         # 'slurm' or 'local'
    log_file: str        # Path to the main log file (stdout)
    workspace: str       # Working directory
```

**C. API Changes**
Change `JobManager.submit()` signature to return `SubmissionResult` instead of `str`.

### 3.2. Monitoring System (`dpeva.monitor`)

We will implement a unified `JobMonitor` interface.

#### Strategy: Slurm Job Monitor (Primary)
Use `squeue` to query job status. This is the most robust method.

*   **Command**: `squeue -h -j <job_id> -o "%t"`
    *   `-h`: No header
    *   `-j`: Job ID
    *   `-o "%t"`: Output state compact form (e.g., `R`, `PD`, `CG`, `CD`)
*   **State Logic**:
    *   **Running/Pending**: `PD` (Pending), `R` (Running), `CG` (Completing), `CF` (Configuring). -> **WAIT**
    *   **Success**: Output is empty (Job gone from queue) OR state is `CD` (Completed). -> **DONE**
    *   **Failure**: State is `F` (Failed), `TO` (Timeout), `NF` (Node Fail). -> **ERROR**
*   **Fallback**: If `squeue` returns "Invalid job id specified", and we just submitted it, it usually means it finished very quickly or failed instantly. We check `sacct` or assume finished if reasonable time passed.

### 3.3. Workflow Integration Points

Each workflow class will support a `wait=True` argument in its `run()` method.

#### 1. `CollectionWorkflow.run()`
```python
if self.backend == "slurm":
    result = self._submit_to_slurm() # Returns SubmissionResult
    if wait:
        monitor = SlurmJobMonitor(result.job_id)
        monitor.wait()
    return
```

#### 2. `TrainingWorkflow` & `InferenceWorkflow`
These submit **multiple** jobs.
```python
# In ParallelTrainer.train()
job_ids = []
for i in range(num_models):
    res = self.job_manager.submit(...)
    job_ids.append(res.job_id)

if backend == "slurm" and blocking:
    self.logger.info(f"Waiting for {len(job_ids)} training jobs...")
    monitor = SlurmBatchMonitor(job_ids) # Monitor list of IDs
    monitor.wait()
```

---

## 4. Development Steps

1.  **Step 1: Core Upgrade**
    *   Modify `JobManager.submit` to parse and return `SubmissionResult`.
    *   Implement `dpeva.monitor.SlurmJobMonitor`.

2.  **Step 2: Workflow Adaptation**
    *   Update `CollectionWorkflow` to use `SlurmJobMonitor`.
    *   Update `ParallelTrainer` to collect IDs and wait.
    *   Update `InferenceWorkflow` to collect IDs and wait.
    *   Update `DescriptorGenerator` to return ID and wait.

3.  **Step 3: Verification**
    *   Create a test script that submits a dummy "sleep 10" job via `JobManager` and verifies that `monitor.wait()` blocks for ~10 seconds.
