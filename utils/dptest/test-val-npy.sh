#!/bin/bash
TESTDATA="../../other_dpdata"
export OMP_NUM_THREADS=24
export DP_INTER_OP_PARALLELISM_THREADS=12
export DP_INTRA_OP_PARALLELISM_THREADS=24
#dp --pt train input.json --finetune dpa-2.3.1-m.pt --model-branch MP_traj_v024_alldata_mixu 2>&1 | tee train.log
dp --pt test -s $TESTDATA -m model.ckpt.pt -d results --head Hybrid_Perovskite  2>&1 | tee test.log
