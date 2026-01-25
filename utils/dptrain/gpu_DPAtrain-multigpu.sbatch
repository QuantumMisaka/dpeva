#!/bin/bash
#SBATCH --job-name=DPA3_mGPU_Train
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1          # Nodes * GPUs-per-node * Ranks-per-GPU
#SBATCH --gpus-per-node=4   # Specify the GPUs-per-node
#SBATCH --qos=huge-gpu     # Depending on your needs [Priority: rush-4gpu = rush-8gpu > improper-gpu > huge-gpu]

# ⚠ DO NOT modify [CUDA-MPS] and [Rank-Map] settings unless you know what you are doing.
#source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash

export OMP_NUM_THREADS=2

export BASE_MODEL=/home/pku-jianghong/liuzhaoqing/WORK/FT2DP-DPEVA/iter1/model/DPA-3.1-3M.pt

# Below are executing commands
nvidia-smi dmon -s pucvmte -o T > nvdmon_job-$SLURM_JOB_ID.log &
source /opt/envs/deepmd3.1.2.env
export DP_INTERFACE_PREC=high
torchrun --nproc_per_node=$((SLURM_NTASKS*SLURM_GPUS_ON_NODE)) --no-python --rdzv_backend=c10d --rdzv_endpoint=localhost:0 dp --pt train input.json --skip-neighbor-stat --finetune $BASE_MODEL 
    # For PT backend parallelization (DDP), using `torchrun` and set `--ntasks=1` instead of `--ntasks=xx` and do not use mps_mapping

#mpirun -np $SLURM_NTASKS -map-by $MAP_OPT dp train input.json
#dp freeze -o graph.pb
#dp compress -i graph.pb -o compressed.pb

dp --pt freeze

