#!/bin/bash

ROOTDIR=$(pwd)
for project in */
do
    for jobdir in ${project}/*/N_*
    do
        cp abacus-forloop.slurm ${jobdir}
        cp reprepare.py ${jobdir}
        cd ${jobdir}
        python reprepare.py
        sbatch abacus-forloop.slurm
        cd ${ROOTDIR}
    done
done