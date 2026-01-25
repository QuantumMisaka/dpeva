import os
import json
import shutil

# Update: 2025-03-01

# Json template
input_json_path = './input.json'
with open(input_json_path, 'r') as f:
    original_json = json.load(f)

laststage = "stage0"
base_model = "DPA-3.1-3M.pt"
ROOTDIR=os.getcwd()


# Defined Parameters
seeds = [
        19090, 
        42, 
        10032, 
        2933]  
training_seeds = [
                19090, 
                42, 
                10032, 
                2933
                    ]
finetune_heads = [
                "Hybrid_Perovskite", 
                "RANDOM", 
                "RANDOM", 
                "RANDOM"
                ]

# length check
if len(seeds) != 4 or len(training_seeds) != 4 or len(finetune_heads) != 4:
    raise ValueError("seeds, training_seeds and finetune_heads Should contain 4 values。")

for i in range(4):
    if not os.path.exists(base_model):
        raise FileNotFoundError(f"{base_model} does not exist.")
    folder_name = f"{ROOTDIR}/{i}"
    os.makedirs(folder_name, exist_ok=True)
    modified_json = json.loads(json.dumps(original_json))
    modified_json["model"]["fitting_net"]["seed"] = seeds[i]
    modified_json["training"]["seed"] = training_seeds[i]
    modified_json["model"]["finetune_head"] = finetune_heads[i]

    with open(os.path.join(folder_name, "input.json"), "w") as f:
        json.dump(modified_json, f, indent=4)
    
    job_script = "gpu_DPAtrain-multigpu.sbatch"
    shutil.copy(base_model, folder_name)
    shutil.copy(job_script, folder_name)
    os.chdir(folder_name)
    os.system(f"sbatch {job_script}")
    os.chdir(ROOTDIR)

