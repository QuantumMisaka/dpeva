#!/bin/bash

ROOTDIR=$(pwd)
for project in */
do
    for jobdir in ${project}/*/
    do
        cd ${jobdir}
        python ${ROOTDIR}/subjob_dist.py
        for subjobdir in N_*/
        do
            cp ${ROOTDIR}/abacus-forloop.slurm ${subjobdir}
            cp ${ROOTDIR}/reprepare.py ${subjobdir}
            cd ${subjobdir}
            # python reprepare.py
            sbatch abacus-forloop.slurm
            cd ${jobdir}
        done
        cd ${ROOTDIR}
    done
done