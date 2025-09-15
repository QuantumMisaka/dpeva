#!/bin/bash

ROOTDIR=`pwd`
TASK="trn" 
#TASK="val-npy"

scripts="${ROOTDIR}/test-${TASK}.sh"

for i in $(ls -d [0-9]/)
do
    mkdir $i/test-${TASK}
    cd $i/test-${TASK}
    cp $scripts .
    cp ../model.ckpt.pt .
    nohup sh $scripts &
    cd $ROOTDIR
done
