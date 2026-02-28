#!/bin/bash
conv_dir="./CONVERGED"
if [ ! -d $conv_dir ]
then
    mkdir -p $conv_dir
fi

for dir in $(ls -d ./N_*/*/)
do
        # Default to not converged
        is_converged=0

        # Check 1: abacus.out failure
        if grep -q "CONVERGENCE HAS NOT BEEN ACHIEVED" $dir/abacus.out 2>/dev/null
        then
                is_converged=0
        # Check 2: running_scf.log failure
        elif grep -q "convergence has not been achieved" $dir/OUT.*/running_scf.log 2>/dev/null
        then
                is_converged=0
        # Check 3: running_scf.log success
        elif grep -q "charge density convergence is achieved" $dir/OUT.*/running_scf.log 2>/dev/null
        then
                is_converged=1
        fi

        if [ $is_converged -eq 1 ]
        then
                mv $dir $conv_dir
        else
                echo "calculation $dir did not converge !"
        fi
done