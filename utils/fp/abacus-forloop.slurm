#!/bin/bash
#SBATCH -J abacus-forloop
#SBATCH -p formal-6226R
#SBATCH -t 120:00:00
#SBATCH -n 32
#SBATCH -N 1
#SBATCH -o abacus-forloop.out
#SBATCH -e abacus-forloop.err

# Koushare server 
module load gcc/9.3.0
#source /mnt/sg001/opt/intel/oneapi2024/setvars.sh
source $HOME/env_scripts/abacus_env_icx.sh

# OMP abacus have better performace than MPI - 2023.9.1
export OMP_NUM_THREADS=4
NP=`expr $SLURM_NTASKS / $OMP_NUM_THREADS`

#echo "change mincpus"
#scontrol update JobId=$SLURM_JOB_ID MinCPUsNode=4

# doing job
touch JobProcessing.state
echo `date` >> JobProcessing.state 

ROOTDIR=$(pwd)
exclude=("CONVERGED")
for dir in */
do
    if [[ " ${exclude[@]} " =~ " ${dir%/} " ]]
    then
            continue
    fi
    cd $dir
    echo $dir
    mpirun -np $NP abacus | tee abacus.out
    cd $ROOTDIR
done

echo `date` >> $HOME/finish
echo `pwd` >> $HOME/finish
echo `date` >> JobProcessing.state
