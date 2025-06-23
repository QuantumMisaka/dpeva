#!/bin/bash

ROOTDIR=$(pwd)
for project in */
do
    for jobdir in ${project}/*/
    do
        cp check_conv.sh ${jobdir}
        cd ${jobdir}
        sh check_conv.sh
        cd ${ROOTDIR}
    done
done