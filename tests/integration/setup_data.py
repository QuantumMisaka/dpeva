
import os
import shutil
import json
import numpy as np
from pathlib import Path

# Source and Destination Paths
# Try to find source relative to project root, fallback to None
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT_CANDIDATE = PROJECT_ROOT / "test" / "test-for-multiple-datapool"
SRC_ROOT = SRC_ROOT_CANDIDATE if SRC_ROOT_CANDIDATE.exists() else None

DST_ROOT = Path(__file__).parent / "data"

def setup_integration_data():
    if not SRC_ROOT:
        print(f"Source data directory not found at {SRC_ROOT_CANDIDATE}. Skipping data setup.")
        print("Ensure 'test/' directory exists or manually populate 'tests/integration/data'.")
        return

    if DST_ROOT.exists():
        print(f"Cleaning existing data at {DST_ROOT}...")
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    print("Copying base model and input...")
    # Copy Model
    shutil.copy2(SRC_ROOT / "0/DPA-3.1-3M.pt", DST_ROOT / "DPA-3.1-3M.pt")
    # Copy Input
    shutil.copy2(SRC_ROOT / "0/input.json", DST_ROOT / "input.json")

    print("Copying candidate pool (subset)...")
    # Copy Candidate Pool (subset of amourC/C2)
    # We only need a few frames for testing
    cand_src = SRC_ROOT / "other_dpdata_all/amourC/C2"
    cand_dst = DST_ROOT / "other_dpdata_all/amourC/C2"
    cand_dst.mkdir(parents=True, exist_ok=True)
    
    # Copy type maps
    for f in ["type.raw", "type_map.raw"]:
        if (cand_src / f).exists():
            shutil.copy2(cand_src / f, cand_dst / f)
            
    # Slice set.000
    if (cand_src / "set.000").exists():
        (cand_dst / "set.000").mkdir(exist_ok=True)
        # Take first 5 frames
        _slice_npy(cand_src / "set.000", cand_dst / "set.000", 5, ["coord.npy", "box.npy"])

    print("Copying training data (subset)...")
    # Copy Training Data (subset of sampled_dpdata/122)
    train_src = SRC_ROOT / "sampled_dpdata/122"
    train_dst = DST_ROOT / "sampled_dpdata/122"
    train_dst.mkdir(parents=True, exist_ok=True)
    
    for f in ["type.raw", "type_map.raw"]:
        if (train_src / f).exists():
            shutil.copy2(train_src / f, train_dst / f)
            
    if (train_src / "set.000").exists():
        (train_dst / "set.000").mkdir(exist_ok=True)
        # Take first 2 frames
        _slice_npy(train_src / "set.000", train_dst / "set.000", 2, 
                   ["coord.npy", "box.npy", "energy.npy", "force.npy", "virial.npy", "spin.npy", "real_atom_types.npy"])

    print("Data setup complete.")

def _slice_npy(src_dir, dst_dir, frames, files):
    for f in files:
        src = src_dir / f
        dst = dst_dir / f
        if src.exists():
            arr = np.load(src)
            if len(arr) > frames:
                arr = arr[:frames]
            np.save(dst, arr)

if __name__ == "__main__":
    setup_integration_data()
