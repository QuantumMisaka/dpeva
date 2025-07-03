#!/bin/bash

ROOTDIR=$(pwd)
for project in */
do
    for jobdir in ${project}/*/
    do
        cp abacus-forloop.slurm ${jobdir}
        cd ${jobdir}
        python subjob_dist.py
        for subjobdir in N_*/
        do
            cp abacus-forloop.slurm ${subjobdir}
            cd ${subjobdir}
            sbatch abacus-forloop.slurm
            cd ${jobdir}
        done
        cd ${ROOTDIR}
    done
done