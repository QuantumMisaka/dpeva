---
title: DeepMD-kit SAI Build Guide
status: active
audience: Developers / Platform Maintainers
last-updated: 2026-07-01
owner: Platform Maintainers
---

# SAI 服务器 DeepMD-kit 安装配置指南

本页记录在本项目 SAI 服务器环境中，将 `dpeva-dpa4` Conda 环境里的
`deepmd-kit` 与仓库内 `test/deepmd-kit` 源码同步的构建、安装和验证方法。

适用场景：

- 需要从 `test/deepmd-kit` 本地源码安装 DeepMD-kit，而不是使用 PyPI/Conda
  预构建版本。
- 需要保留 `dpeva-dpa4` 环境中的 `torch 2.11.0+cu126`。
- 需要 CUDA 12.6.3 和 PyTorch 后端，并在 V100 GPU 上实际验证 `dp embed`
  输出 `descriptor`、`atomic_feature`、`structural_feature`。

## 1. 方法来源

本安装方法来自本仓库内的可审计来源，而不是外部临时命令：

- `test/deepmd-kit/pyproject.toml`：确认 DeepMD-kit 使用
  `scikit-build-core`/CMake 构建，并支持通过环境变量控制后端。
- `test/deepmd-kit/CMakeLists.txt` 与构建日志：确认 CUDA toolkit、Torch
  库和 PyTorch 后端由 CMake 探测。
- `test/deepmd-kit/doc/inference/embedding.md`：确认 `dp embed` 的 CLI 用法和
  HDF5 输出数据集。
- `test/deepmd-kit/source/tests/pt/model/test_embedding.py`：提供可运行的
  SeZM/DPA4 embedding 测试模型构造方式。
- SAI 平台规则：登录节点只做构建、提交和轻量检查；GPU 可用性必须通过
  Slurm 分配计算节点验证。

## 2. 当前已验证基线

记录日期：2026-07-01。

| 项目 | 已验证值 |
|---|---|
| Conda 环境 | `dpeva-dpa4` |
| DeepMD-kit 源码 | `test/deepmd-kit` |
| 源码提交 | `73de44b1f94471b2e3bdb6b11f57b34d7bc791bb` |
| 源码 describe | `v3.2.0b0-67-g73de44b1` |
| 安装版本 | `3.2.0b1.dev67+g73de44b1f` |
| CUDA toolkit | `/opt/devtools/nvidia/cuda-12.6.3` |
| NVCC | `12.6.85` |
| Torch | `2.11.0+cu126` |
| Torch CUDA | `12.6` |
| GPU 架构 | V100 / `sm_70` |
| SeZM/DPA4 邻居表依赖 | `vesin 0.5.8`, `vesin-torch 0.5.8` |

## 3. 前置检查

从项目根目录执行：

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva
source scripts/env/dpeva-dpa4.env

python -W ignore -c 'import torch; print(torch.__version__); print(torch.version.cuda)'
git -C test/deepmd-kit log -1 --pretty='commit %H%nsubject %s'
git -C test/deepmd-kit describe --tags --always --dirty
nvcc --version
```

预期要点：

- `torch` 为 `2.11.0+cu126`。
- `torch.version.cuda` 为 `12.6`。
- `nvcc` 来自 `/opt/devtools/nvidia/cuda-12.6.3/bin/nvcc`。
- `test/deepmd-kit` 工作树处于预期提交；若带 `-dirty`，先确认本地改动是否
  应进入安装版本。

## 4. 构建 wheel

使用 wheel 构建而不是直接 editable 安装，便于保留构建产物和日志。关键点是
`--no-build-isolation --no-deps`：构建使用当前 `dpeva-dpa4` 环境里的 Torch，
避免 pip 重新解析或替换核心 CUDA/Torch 依赖。

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva
source scripts/env/dpeva-dpa4.env

STAMP=$(date +%Y%m%d-%H%M%S)
LOGDIR="$PWD/logs/deepmd-dpeva-dpa4-$STAMP"
mkdir -p "$LOGDIR/wheelhouse"

git -C test/deepmd-kit log -1 --pretty='commit %H%nsubject %s' | tee "$LOGDIR/build.log"
git -C test/deepmd-kit describe --tags --always --dirty | tee -a "$LOGDIR/build.log"
python -W ignore -c 'import torch; print("torch", torch.__version__); print("torch_cuda", torch.version.cuda)' | tee -a "$LOGDIR/build.log"
nvcc --version | sed -n '1,6p' | tee -a "$LOGDIR/build.log"

export MAX_JOBS=4
export CMAKE_BUILD_PARALLEL_LEVEL=4
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=70"
export DP_VARIANT=cuda
export DP_ENABLE_PYTORCH=1
export DP_ENABLE_TENSORFLOW=0
export DP_ENABLE_PADDLE=0
export DP_ENABLE_JAX=0

python -m pip wheel \
  --no-build-isolation \
  --no-deps \
  --verbose \
  --wheel-dir "$LOGDIR/wheelhouse" \
  test/deepmd-kit 2>&1 | tee -a "$LOGDIR/build.log"
```

构建日志中应出现类似信息：

```text
Found CUDAToolkit: /opt/devtools/nvidia/cuda-12.6.3
Found CUDA in /opt/devtools/nvidia/cuda-12.6.3/bin, build nv GPU support
PyTorch: CUDA detected: 12.6
PyTorch: CUDA nvcc is: /opt/devtools/nvidia/cuda-12.6.3/bin/nvcc
Found Torch: .../.conda/envs/dpeva-dpa4/.../torch/lib/libtorch.so
Added CUDA NVCC flags for: -gencode;arch=compute_70,code=sm_70
Enabled backends:
  - PyTorch
Created deepmd_kit-...whl
```

## 5. 安装 wheel

```bash
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva
source scripts/env/dpeva-dpa4.env

LOGDIR=/path/to/logs/deepmd-dpeva-dpa4-YYYYmmdd-HHMMSS
python -m pip install --no-deps --force-reinstall \
  "$LOGDIR"/wheelhouse/deepmd_kit-*.whl 2>&1 | tee "$LOGDIR/install.log"
```

安装后核对：

```bash
python -W ignore - <<'PY'
import importlib.metadata as md
import deepmd
import torch

print("deepmd-kit", md.version("deepmd-kit"))
print("deepmd.__version__", getattr(deepmd, "__version__", "NO_ATTR"))
print("deepmd.__file__", deepmd.__file__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
PY

dp --version
dp --pt -h | grep -E 'embed|eval-desc'
dp --pt embed -h
```

`dp --pt -h` 必须列出 `embed` 子命令。

## 6. 安装 SeZM/DPA4 embedding 运行依赖

`dp embed` 对 SeZM/DPA4 模型会走内建邻居表路径。当前 DeepMD-kit 源码要求
`nvalchemiops` 或 `vesin` 至少一个可导入；本环境采用 `vesin[torch]`。

先 dry-run，确认不会替换 Torch：

```bash
source scripts/env/dpeva-dpa4.env
python -m pip install --dry-run 'vesin[torch]'
```

确认只会新增 `vesin` 与 `vesin-torch` 后安装：

```bash
python -m pip install 'vesin[torch]'
```

核对：

```bash
python - <<'PY'
for name in ("vesin", "vesin.torch", "nvalchemiops"):
    try:
        mod = __import__(name)
        print(name, "OK", getattr(mod, "__version__", "NO_VERSION"))
    except Exception as exc:
        print(name, "MISSING", type(exc).__name__, exc)
PY
```

`nvalchemiops` 可缺失；只要 `vesin` 和 `vesin.torch` 可导入，SeZM/DPA4
embedding 的邻居表构建即可运行。

## 7. CPU 侧 `dp embed` smoke test

建议用 `test/deepmd-kit/source/tests/pt/model/test_embedding.py` 中的小型 SeZM
配置生成临时 checkpoint，再用 DeePMD system 目录测试 CLI。测试产物可放在本次
`LOGDIR/embed-fixture/`。

最小验收标准：

- `dp --pt embed` 退出码为 0。
- HDF5 中有 `descriptor`、`atomic_feature`、`structural_feature`。
- 三个 dataset 的 dtype 与 `--dtype fp32` 一致，且数值全部 finite。

示例 HDF5 读回检查：

```bash
python - <<'PY'
import h5py
import numpy as np

path = "embedding.hdf5"
with h5py.File(path, "r") as f:
    keys = []
    def walk(name, obj):
        if hasattr(obj, "shape"):
            arr = obj[()]
            keys.append(name)
            print(name, obj.shape, obj.dtype, bool(np.isfinite(arr).all()))
    f.visititems(walk)
    present = {k.split("/")[-1] for k in keys}
    missing = {"descriptor", "atomic_feature", "structural_feature"} - present
    if missing:
        raise SystemExit(f"missing datasets: {sorted(missing)}")
PY
```

## 8. GPU 验证

登录节点没有可用 NVIDIA driver，不能作为 GPU 可用性证据。必须通过 Slurm
分配 GPU 计算节点验证。

单卡快速验证优先使用支持 1 GPU 的分区/QOS，例如：

```bash
srun \
  --partition=4V100 \
  --qos=rush-1o2gpu \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --time=00:10:00 \
  bash -lc 'source scripts/env/dpeva-dpa4.env && python - <<PY
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
PY'
```

若单卡资源排队过久，`8V100V0` 可能要求每节点 GPU 数量按 4 的倍数申请；此时
可用 `--gpus-per-node=4` 跑同一单进程验证。不要在同一时间保留多个等价验证作业；
切换分区前先取消旧的等待作业。

完整 GPU 验证需要同时检查版本、CUDA 可见性和 `dp embed` 输出：

```bash
srun --partition=8V100V0 --nodes=1 --ntasks=1 --gpus-per-node=4 --time=00:10:00 bash -lc '
set -euo pipefail
cd /home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva
LOGDIR=/path/to/logs/deepmd-dpeva-dpa4-YYYYmmdd-HHMMSS
source scripts/env/dpeva-dpa4.env
export OMP_NUM_THREADS=1

hostname
nvidia-smi -L
python -W ignore - <<PY
import importlib.metadata as md
import deepmd
import torch
import vesin
import vesin.torch

print("deepmd-kit", md.version("deepmd-kit"))
print("deepmd_version", getattr(deepmd, "__version__", "NO_ATTR"))
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vesin", getattr(vesin, "__version__", "NO_VERSION"))
print("vesin_torch", getattr(vesin.torch, "__version__", "NO_VERSION"))
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
print("cuda_device_name", torch.cuda.get_device_name(0))
PY

out="${SLURM_TMPDIR:-/tmp}/dpeva-dpa4-sezm-embed-${SLURM_JOB_ID}.hdf5"
export out
rm -f "$out"
dp --pt embed \
  -m "$LOGDIR/embed-fixture/sezm_embed.pt" \
  -s "$LOGDIR/embed-fixture/system" \
  -o "$out" \
  --dtype fp32

python - <<PY
import os
import h5py
import numpy as np

out = os.environ["out"]
print("hdf5_path", out)
with h5py.File(out, "r") as f:
    keys = []
    def walk(name, obj):
        if hasattr(obj, "shape"):
            arr = obj[()]
            keys.append(name)
            print("dataset", name, obj.shape, obj.dtype, bool(np.isfinite(arr).all()))
    f.visititems(walk)
    present = {k.split("/")[-1] for k in keys}
    missing = {"descriptor", "atomic_feature", "structural_feature"} - present
    if missing:
        raise SystemExit("missing datasets: " + ", ".join(sorted(missing)))
PY
'
```

任务完成后用 Slurm 再确认退出状态：

```bash
sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,NodeList -P
```

## 9. 常见问题

### 9.1 `dp embed` 报 `DPA4/SeZM model supports ... but got ener`

该错误说明测试模型不是当前 `dp embed` SeZM/DPA4 路径支持的 fitting type。不要修改
DeepMD-kit 代码绕过错误；换用同版本测试中构造的 SeZM/DPA4 checkpoint，或使用
真实的 DPA4/SeZM 训练 checkpoint。

### 9.2 `SeZM neighbor-list construction requires either 'nvalchemiops' or 'vesin'`

安装 `vesin[torch]`。安装前先 `--dry-run`，确认不会替换当前 `torch`。

### 9.3 `QOSMinGRES`

资源形态不符合 SAI QOS/分区策略。1 GPU 作业使用支持单卡的 QOS，例如
`4V100 + rush-1o2gpu`；如果在 `8V100V0` 上验证，通常按每节点 4 GPU 申请。

### 9.4 登录节点 `nvidia-smi` 失败

登录节点没有 NVIDIA driver 或不可访问 GPU 是正常现象。GPU 可用性结论必须来自
Slurm 分配的计算节点。

## 10. 维护原则

- 每次升级 `test/deepmd-kit` 后，都重新记录源码提交、wheel 版本、Torch/CUDA 版本
  和 GPU smoke test job id。
- 不要把 `pip install deepmd-kit[...]` 作为本环境升级方式；本项目环境要求以
  `test/deepmd-kit` 源码和当前 `dpeva-dpa4` Torch/CUDA 栈为准。
- 不要依赖交互式 shell 的隐式状态；Slurm 脚本中通过 `source scripts/env/dpeva-dpa4.env`
  显式加载环境。
- 构建日志、wheel、安装日志和验证 fixture 应保留在 `logs/deepmd-dpeva-dpa4-*` 下，
  便于复盘。
