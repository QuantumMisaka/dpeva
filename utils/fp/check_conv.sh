#!/bin/bash
conv_dir="./CONVERGED"
if [ ! -d $conv_dir ]
then
    mkdir -p $conv_dir
fi

for dir in */N_*/
do
        if grep -q "CONV" $dir/abacus.out
        then
                echo "calculation $dir did not converge !"
        else
                mv $dir $conv_dir
        fi
done