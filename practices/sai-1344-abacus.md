# 在 sai-1344 试用集群提交并完成 ABACUS 计算任务

本文记录在 SAI 1344 试用集群上提交 ABACUS 任务的可复用流程，并附带一次已完成的探针任务结果。

适用时段：SAI 2026-07 机房迁移和 1344 超节点试用集群过渡期。2026-08-01 后使用本文前，应先确认 SAI 是否已有新的正式入口、Slurm 配置或模板更新。

## 1. 登录 1344 试用集群

如果当前已经在旧集群登录节点上，可直接跳转到 1344 试用集群：

```bash
ssh sai-1344-tmp
```

也可以从外部直接连接新集群入口：

```bash
ssh -p 12022 <username>@c0.sai.ai-4s.com
```

从 1344 试用集群反向进入旧集群：

```bash
ssh sai-legacy
```

本次实测中，`sai-1344-tmp` 和 `c0.sai.ai-4s.com:12022` 都连接到 `login-02.mr-sai.ai`。

## 2. 识别 Slurm 与分区状态

进入 1344 后先检查队列和分区：

```bash
squeue
scontrol show partition 16V100
scontrol show config
```

本次实测中，`16V100` 分区可用，`scontrol show partition 16V100` 显示：

- `PartitionName=16V100`
- `AllowQos=huge-gpu,flood-gpu,rush-gpu,ultimate-gpu`
- `Nodes=16v100n[01-29,32-84]`
- `TotalNodes=82`
- `TRES=... gres/gpu=1312`
- `JobDefaults=DefCpuPerGPU=8,DefMemPerGPU=16000`

注意：本次非交互 SSH 环境中 `gpu` 定制命令不可用，返回 `command not found`。此时可用 `squeue`、`scontrol show partition` 和 `sinfo` 代替。

## 3. 准备 ABACUS 输入目录

建议先做一个很小的探针任务，确认 SSH、Slurm、module、MPI 和 ABACUS 都能跑通，再提交正式计算。

本次探针复用了集群上的 graphene ABACUS 示例：

```bash
mkdir -p ~/sai-1344-abacus-probe
cd ~/sai-1344-abacus-probe

cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/INPUT .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/KPT .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/STRU .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/C_ONCV_PBE-1.0.upf .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/C_gga_7au_100Ry_2s2p1d.orb .
```

为了让探针尽快结束，可把示例从结构优化改成 SCF，并降低 K 点：

```bash
sed -i 's/calculation[[:space:]]\+relax/calculation             scf/' INPUT
sed -i 's/^16 16 1 0 0 0/1 1 1 0 0 0/' KPT
```

正式任务应按实际体系和精度需求修改 `INPUT`、`KPT`、`STRU`、赝势和轨道文件，不要沿用探针里的低精度设置。

## 4. 准备 sbatch 脚本

从系统模板复制 ABACUS 脚本：

```bash
cp /opt/sbatch_examples/gpu_abacus.sbatch ./abacus_16v100.sbatch
```

在当前 1344 试用集群上，模板需要显式初始化系统 profile，否则 batch 作业中可能出现：

```text
module: command not found
mpirun: command not found
```

可用脚本如下：

```bash
#!/bin/bash
#SBATCH --job-name=ABACUS_PROBE
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=4          # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=4   # Specify the GPUs-per-node
#SBATCH --qos=rush-gpu
source /etc/profile

# Do not modify CUDA-MPS and Rank-Map settings unless you know what you are doing.
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash

nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
module load abacus/LTSv3.10.1-sm70-auto
mpirun -np $SLURM_NTASKS --map-by $MAP_OPT abacus

exit
```

关键点：

- `source /etc/profile` 必须放在所有 `#SBATCH` 行之后。若放在 `#SBATCH` 行之前，Slurm 会停止解析后续资源指令，可能导致 `QOSMinGRES`。
- `16V100` 分区支持的 QOS 包括 `rush-gpu`、`huge-gpu`、`flood-gpu`、`ultimate-gpu`。
- 除特殊小卡数 QOS 外，GPU 任务保持每节点 GPU 数为 4 的倍数。1344 的 `16V100` 节点每节点 16 GPU；探针用 4 GPU 是合法的小任务。
- 不要手动指定 GPU 任务的 `--mem` 或 `--cpus-per-task`。系统会按每 GPU 默认 CPU 和内存策略分配。

如果要做 1 节点 16 GPU 正式任务，可相应调整：

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=16
#SBATCH --qos=huge-gpu
```

`--ntasks` 通常按 `Nodes * GPUs-per-node * Ranks-per-GPU` 设置。若每个 GPU 运行 1 个 MPI rank，则 `--ntasks` 等于 GPU 总数。

## 5. 提交任务

在 ABACUS 输入目录中执行：

```bash
sbatch abacus_16v100.sbatch
```

提交成功会返回类似：

```text
Submitted batch job 575631
```

## 6. 查看任务状态

查看队列状态：

```bash
squeue -j <jobid>
```

查看详细调度信息：

```bash
scontrol show job <jobid>
```

查看历史和退出码：

```bash
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList%30
```

常见状态解释：

- `PD`：排队中，重点看 `Reason`，例如资源不足或配额限制。
- `R`：运行中，`NodeList` 显示运行节点。
- `COMPLETED` 且 `ExitCode=0:0`：Slurm 任务正常结束。
- `FAILED` 且 `ExitCode=127:0`：通常是命令不存在，例如 `module` 或 `mpirun` 未初始化。

## 7. 检查 ABACUS 输出

ABACUS 正常运行后，会在工作目录中生成：

```text
OUT.ABACUS/
abacus.json
time.json
nvdmon_job-<jobid>.log
slurm-<jobid>.out
```

常用检查命令：

```bash
tail -n 80 slurm-<jobid>.out
tail -n 80 OUT.ABACUS/running_scf.log
ls -lh OUT.ABACUS
```

正常完成时，`slurm-<jobid>.out` 末尾会有类似：

```text
START  Time  : Sat Jul  4 12:01:13 2026
FINISH Time  : Sat Jul  4 12:01:15 2026
TOTAL  Time  : 2
SEE INFORMATION IN : OUT.ABACUS/
```

`OUT.ABACUS/` 中应包含 `running_scf.log`、`warning.log`、`kpoints`、`istate.info` 等文件。SCF 任务还可能生成电荷密度 restart 文件。

## 8. 本次探针实测记录

探针目录：

```text
/home/pku-jianghong/liuzhaoqing/sai-1344-abacus-probe-20260704-114002
```

第一次提交：

- JobID：`575584`
- 结果：`FAILED`
- 退出码：`127:0`
- 原因：batch 环境中 `module` 和 `mpirun` 未初始化
- 日志关键行：

```text
/var/spool/slurmd/job575584/slurm_script: line 14: module: command not found
/var/spool/slurmd/job575584/slurm_script: line 15: mpirun: command not found
```

修正：在全部 `#SBATCH` 行之后加入 `source /etc/profile`。

第二次错误尝试：

- 现象：`sbatch: error: QOSMinGRES`
- 原因：把 `source /etc/profile` 插入到了 `#SBATCH` 指令之前，导致 Slurm 不再解析后续 `#SBATCH` 资源行。
- 处理：把 `source /etc/profile` 移到所有 `#SBATCH` 行之后。

最终成功提交：

- JobID：`575631`
- 状态：`COMPLETED`
- 退出码：`0:0`
- 分区：`16V100`
- 节点：`16v100n18`
- 资源：1 节点，4 GPU，Slurm 实际分配 32 CPU 和 62.50 GiB 内存
- Slurm 运行时间：9 秒
- ABACUS 计算时间：约 2 秒

成功任务使用的脚本：

```text
/home/pku-jianghong/liuzhaoqing/sai-1344-abacus-probe-20260704-114002/probe_abacus_fixed.sbatch
```

## 9. 迁移期 IO 注意事项

1344 试用集群处于迁移过渡期时，登录节点只用于 SSH、编辑、任务提交管理、轻量安装和数据传输。不要在登录节点上运行 ABACUS、批量解压、扫描海量小文件或做训练数据随机读取。

正式计算建议：

- 输入和输出集中放在任务工作目录中。
- 大量小文件先打包整理，减少元数据压力。
- 中间临时数据优先使用计算节点本地 SSD、`$CACHE_LOCAL` 或 `/tmp`。
- 旧数据通过 NFS-over-RDMA 挂载恢复访问时，避免在旧数据挂载上做重度 IO。

## 10. 快速复用命令

以下命令适合在 1344 登录节点上快速创建一个 ABACUS 探针：

```bash
mkdir -p ~/sai-1344-abacus-probe
cd ~/sai-1344-abacus-probe

cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/INPUT .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/KPT .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/STRU .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/C_ONCV_PBE-1.0.upf .
cp /opt/apps/vaspkit.1.5.1/examples/ABACUS/elastic/graphene-relax/C_gga_7au_100Ry_2s2p1d.orb .

sed -i 's/calculation[[:space:]]\+relax/calculation             scf/' INPUT
sed -i 's/^16 16 1 0 0 0/1 1 1 0 0 0/' KPT

cat > abacus_16v100.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=ABACUS_PROBE
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4
#SBATCH --qos=rush-gpu
source /etc/profile

source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash

nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
module load abacus/LTSv3.10.1-sm70-auto
mpirun -np $SLURM_NTASKS --map-by $MAP_OPT abacus

exit
EOF

sbatch abacus_16v100.sbatch
```
