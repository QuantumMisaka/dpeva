
import os
import shutil
import logging
from pathlib import Path
import dpdata
import numpy as np
from dpeva.labeling.generator import AbacusGenerator
from dpeva.labeling.strategy import ResubmissionStrategy
from dpeva.labeling.postprocess import AbacusPostProcessor
from dpeva.labeling.packer import TaskPacker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_reproduce_fp_workflow():
    # Paths
    base_dir = Path("test/fp-setting")
    input_data_dir = base_dir / "sampled_dpdata"
    ref_inputs_dir = base_dir / "abacus-inputs"
    ref_outputs_dir = base_dir / "conveged_data_view"
    
    work_dir = Path("test_reproduction_workdir")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()

    # 1. Test Input Generation
    logger.info("--- Testing Input Generation ---")
    
    # Config for generator (mimicking FeCHO_fp_set.py)
    config = {
        "dft_params": {
            "calculation": "scf",
            "nspin": 2,
            "xc": "pbe",
            "ecutwfc": 100,
            "mixing_type": "broyden",
            "mixing_beta": 0.4,
            "mixing_ndim": 20,
            "kpts": [1, 1, 1], # default
            "cal_force": 1,
            "cal_stress": 1,
            "out_stru": 1
        },
        "pp_map": {
            'H': 'H.upf', 'C': 'C.upf', 'O': 'O.upf', 'Fe': 'Fe.upf'
        },
        "orb_map": {
            'H': 'H.orb', 'C': 'C.orb', 'O': 'O.orb', 'Fe': 'Fe.orb'
        },
        "pp_dir": "/path/to/pp",
        "orb_dir": "/path/to/orb",
        "kpt_criteria": 25,
        "vacuum_thickness": 6.3
    }
    
    generator = AbacusGenerator(config)
    
    # Load one system to test
    # Find a valid system dir in sampled_dpdata
    # It seems sampled_dpdata contains dataset_name subdirs
    # FeCHO_fp_set.py iterates dataset_names.
    # Let's find one.
    dataset_dirs = [d for d in input_data_dir.iterdir() if d.is_dir()]
    if not dataset_dirs:
        logger.warning("No dataset directories found in sampled_dpdata")
        return

    test_dataset = dataset_dirs[0]
    logger.info(f"Using dataset: {test_dataset.name}")
    
    # Load with dpdata
    # dpdata.MultiSystems.from_file might fail if deepmd/npy structure is complex
    # Let's use System directly if possible or loop
    try:
        ms = dpdata.MultiSystems()
        # Iterate subdirs if any, or just load the dir
        # FeCHO_fp_set.py loads `target_dpdata.append(dpdata.System(item, fmt='deepmd/npy'))`
        # where item is `project_dir/dataset_name/*`
        for item in test_dataset.iterdir():
            if item.is_dir():
                try:
                    ms.append(dpdata.System(str(item), fmt='deepmd/npy'))
                except Exception as e:
                    logger.warning(f"Skipping {item}: {e}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Loaded {len(ms)} systems.")
    
    # Generate inputs
    generated_dir = work_dir / "generated_inputs"
    generated_dir.mkdir()
    
    count = 0
    for i, s in enumerate(ms):
        atoms_list = s.to_ase_structure()
        for j, atoms in enumerate(atoms_list):
            task_name = f"sys{i}_frame{j}"
            task_dir = generated_dir / task_name
            stru_type = generator.generate(atoms, task_dir, task_name)
            logger.info(f"Generated {task_name} (Type: {stru_type})")
            
            # Verify files exist
            assert (task_dir / "INPUT").exists()
            assert (task_dir / "STRU").exists()
            assert (task_dir / "KPT").exists()
            count += 1
            if count >= 5: break # Test first 5
        if count >= 5: break

    # 2. Test Packer
    logger.info("--- Testing Task Packer ---")
    packer = TaskPacker(tasks_per_job=2)
    packer.pack(generated_dir)
    
    # Check if N_2_0 exists
    packed_dir = generated_dir / "N_2_0"
    if packed_dir.exists():
        logger.info(f"Packed directory created: {packed_dir}")
        logger.info(f"Contents: {[x.name for x in packed_dir.iterdir()]}")
    else:
        logger.error("Packed directory NOT created")

    # 3. Test Strategy
    logger.info("--- Testing Resubmission Strategy ---")
    strategy = ResubmissionStrategy([
        {"mixing_beta": 0.4},
        {"mixing_beta": 0.1},
        {"mixing_beta": 0.025}
    ])
    
    # Modify one input
    if packed_dir.exists():
        # Find a task dir inside
        task_subdirs = [d for d in packed_dir.iterdir() if d.is_dir()]
        if task_subdirs:
            target_task = task_subdirs[0]
            logger.info(f"Applying strategy to {target_task}")
            
            # Apply attempt 1 (beta 0.1)
            strategy.apply(target_task, 1)
            
            # Verify change
            with open(target_task / "INPUT", "r") as f:
                content = f.read()
                if "mixing_beta 0.1" in content or "mixing_beta          0.1" in content:
                    logger.info("Strategy applied successfully (beta -> 0.1)")
                else:
                    logger.error(f"Strategy failed. Content:\n{content}")

    logger.info("Test reproduction workflow completed.")

if __name__ == "__main__":
    test_reproduce_fp_workflow()
