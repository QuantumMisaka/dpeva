#!/bin/bash
#SBATCH -J desc_all
#SBATCH -p amd
##SBATCH -t 48:00:00
#SBATCH --ntasks=1
#SBATCH --cores-per-socket=1
#SBATCH --cpus-per-task=16
#SBATCH -o desc_genall.out
#SBATCH -e desc_genall.err

# use calc_desc.py to calculate descriptors for deepmd/npy in all directories
# usage: bash desc_all.sh

if [[ -z $SLURM_CPUS_PER_TASK ]]
then
    SLURM_CPUS_PER_TASK=4
fi
data_dirs=$(ls -d ./data-clean-v2-7-20873-npy/*)
    
done
export OMP_NUM_THREADS=`expr $SLURM_CPUS_PER_TASK \* 4`
desc_dir="./descriptors"
for dir in $data_dirs
do
    echo "deal with $dir"
    dirname=$(basename "$dir")
    if [[ -e "${desc_dir}/${dirname}/desc.npy" ]]
    then
        echo "desc for ${dir} already exists"
        continue
    else
        python calc_desc.py $dir $desc_dir
    fi
done