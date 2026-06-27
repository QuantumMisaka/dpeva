#!/bin/bash
#SBATCH -J dpa4_embed_216
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -p 4V100
#SBATCH -o direct_embed_216.log
#SBATCH -e direct_embed_216.err
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --open-mode=truncate

set -euo pipefail

REPO="/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva"
CASE_ROOT="${REPO}/practices/dpeva-dpa4-test/dpa4-dpeva-test"
DATA="${CASE_ROOT}/issue_dpa4_plus_eval_desc_oom/sampled_dpdata/216"
MODEL="${CASE_ROOT}/DPA4-Plus-OMat24-16M.pt"
OUT="${CASE_ROOT}/embed_validation/direct_embed_216/embedding.hdf5"

cd "${CASE_ROOT}/embed_validation/direct_embed_216"
nvidia-smi dmon -s pucvmte -o T > "nvdmon_job-${SLURM_JOB_ID}.log" &
source "${REPO}/scripts/env/dpeva-dpa4-embed-test.env"
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export DP_INTERFACE_PREC=high
export OMP_NUM_THREADS=2

rm -f "${OUT}"
dp --pt embed -s "${DATA}" -m "${MODEL}" -o "${OUT}" --dtype fp32

python - <<'PY'
import h5py
from dpeva.io.collection import CollectionIOManager

path = "/home/pku-jianghong/liuzhaoqing/work/ft2dp-dpeva/dpeva/practices/dpeva-dpa4-test/dpa4-dpeva-test/embed_validation/direct_embed_216/embedding.hdf5"
with h5py.File(path, "r") as h5:
    groups = sorted(h5.keys())
    assert groups, "no HDF5 groups written"
    group = h5[groups[0]]
    for name in ("descriptor", "atomic_feature", "structural_feature", "atom_types"):
        assert name in group, f"missing dataset {name}"
    descriptor_shape = tuple(group["descriptor"].shape)
    atomic_shape = tuple(group["atomic_feature"].shape)
    structural_shape = tuple(group["structural_feature"].shape)
    atom_types_shape = tuple(group["atom_types"].shape)
    assert descriptor_shape[0] == 59, descriptor_shape
    assert descriptor_shape[1] == 216, descriptor_shape
    assert atom_types_shape == descriptor_shape[:2], atom_types_shape
    assert structural_shape[0] == descriptor_shape[0], structural_shape
    print("HDF5_GROUP", groups[0])
    print("DESCRIPTOR_SHAPE", descriptor_shape)
    print("ATOMIC_FEATURE_SHAPE", atomic_shape)
    print("STRUCTURAL_FEATURE_SHAPE", structural_shape)
    print("ATOM_TYPES_SHAPE", atom_types_shape)

manager = CollectionIOManager(".", "embed_validation_readback")
names, desc = manager.load_descriptors(path, "embed descriptors")
assert len(names) == 59, len(names)
assert desc.shape[0] == 59, desc.shape
print("DPEVA_DESCRIPTOR_READBACK", len(names), desc.shape)
PY

echo "DPEVA_TAG: DIRECT_EMBED_216_FINISHED"
