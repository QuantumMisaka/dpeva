#!/bin/bash
#SBATCH --job-name=DPA3_mGPU_Train
#SBATCH --partition=4V100PX
#SBATCH -w 4v100bkn01
#SBATCH --nodes=1
#SBATCH --ntasks=1          # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=1   # Specify the GPUs-per-node
#SBATCH --qos=rush-1o2gpu     # Depending on your needs [Priority: rush-4gpu = rush-8gpu > improper-gpu > huge-gpu]

# ⚠ DO NOT modify [CUDA-MPS] and [Rank-Map] settings unless you know what you are doing.
#source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash

export OMP_NUM_THREADS=2

# Below are executing commands
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
source /opt/envs/deepmd3.1.2.env
export DP_INTERFACE_PREC=high
    # For PT backend parallelization (DDP), using `torchrun` and set `--ntasks=1` instead of `--ntasks=xx` and do not use mps_mapping


TESTDATA="../../sampled_dpdata"
dp --pt test -s $TESTDATA -m model.ckpt.pt -d results 2>&1 | tee test.log

