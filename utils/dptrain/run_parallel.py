import os
import json
import shutil

# Update: 2025-03-01

# Json template
input_json_path = './input.json'
with open(input_json_path, 'r') as f:
    original_json = json.load(f)
running_way = "cont" # init or cont

laststage = "stage8"
base_models = [
    f"./{laststage}-fit0.pt",
    f"./{laststage}-fit1.pt",
    f"./{laststage}-fit2.pt",
    f"./{laststage}-fit3.pt",
]
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
if running_way == "init":
    finetune_heads = [
                    "Target_FTS", 
                    "RANDOM", 
                    "RANDOM", 
                    "RANDOM"
                    ]
elif running_way == "cont":
    finetune_heads = [
                    "Target_FTS", 
                    "Target_FTS", 
                    "Target_FTS", 
                    "Target_FTS", 
                    ] 
else:
    raise ValueError("running_way should be 'init' or 'cont'")

# length check
if len(seeds) != 4 or len(training_seeds) != 4 or len(finetune_heads) != 4:
    raise ValueError("seeds, training_seeds and finetune_heads Should contain 4 values。")

for i in range(4):
    if running_way == "init":
        base_model = base_models[0]
    elif running_way == "cont":
        base_model = base_models[i]
    else:
        raise ValueError("running_way should be 'init' or 'cont'")
    # check if base_model exists
    if not os.path.exists(base_model):
        raise FileNotFoundError(f"{base_model} does not exist.")
    folder_name = f"{ROOTDIR}/{i}"
    os.makedirs(folder_name, exist_ok=True)
    training_setting = f'''#!/bin/bash
export OMP_NUM_THREADS=12
export DP_INTER_OP_PARALLELISM_THREADS=6
export DP_INTRA_OP_PARALLELISM_THREADS=12
dp --pt train input.json --finetune {base_model} 2>&1 | tee train.log
'''
    modified_json = json.loads(json.dumps(original_json))
    modified_json["model"]["fitting_net"]["seed"] = seeds[i]
    modified_json["training"]["seed"] = training_seeds[i]
    modified_json["model"]["finetune_head"] = finetune_heads[i]

    with open(os.path.join(folder_name, "input.json"), "w") as f:
        json.dump(modified_json, f, indent=4)
    with open(os.path.join(folder_name, "train.sh"), "w") as f:
        f.write(training_setting)
    
    shutil.copy(base_model, folder_name)
    os.chdir(folder_name)
    os.system("nohup sh train.sh &")
    os.chdir(ROOTDIR)
