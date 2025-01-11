import dpdata
from deepmd.infer.deep_pot import DeepPot
#from deepmd.calculator import DP
import numpy as np
import os
#from multiprocessing import pool
import gc
import glob
import logging
import torch
import warnings

datadir = "./sampled-data-direct-10p-npy"
modelpath = "./model.ckpt.pt"
savedir = "descriptors"

omp = 16
#proc = 4
os.environ['OMP_NUM_THREADS'] = f'{omp}'

def descriptor_from_model(sys: dpdata.LabeledSystem, model:DeepPot):
    coords = sys.data["coords"]
    cells = sys.data["cells"]
    model_type_map = model.get_type_map()
    type_trans = np.array([model_type_map.index(i) for i in sys.data['atom_names']])
    atypes = list(type_trans[sys.data['atom_types']])
    predict = model.eval_descriptor(coords, cells, atypes)
    return predict
#alldata = dpdata.MultiSystems.from_dir(datadir,datakey,fmt="deepmd/npy")
all_set_directories = glob.glob(os.path.join(
    datadir, '**', 'set.*'), recursive=True)
all_directories = set()
for directory in all_set_directories:
    coord_path = os.path.join(directory, 'coord.npy')
    if os.path.exists(coord_path):
        all_directories.add(os.path.dirname(directory))
all_directories = list(all_directories)
all_directories.sort()

# 配置日志格式，包含时间戳、日志级别和日志信息
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间戳格式
)


logging.info("Start Generating Descriptors")

if not os.path.exists(savedir):
    os.mkdir(savedir)

with open("running", "w") as fo:
    for onedir in all_directories:
        onedata = dpdata.LabeledSystem(onedir, fmt="deepmd/npy")
        key = onedata.short_name
        save_key = f"{savedir}/{key}"
        logging.info(f"Generating descriptors for {key}")
        if os.path.exists(save_key):
            if os.path.exists(f"{save_key}/desc.npy"):
                logging.info(f"Descriptors for {key} already exist, skip")
                continue
        model = DeepPot(modelpath, head="Target_FTS")
        try:
            desc = descriptor_from_model(onedata, model)
        except torch.cuda.OutOfMemoryError:
            warnings.warn(f"CUDA OOM, Skipping {key}")
            continue
        except Exception as e:
            raise e
            
        logging.info(f"Descriptors for {key} generated")
        os.mkdir(save_key)
        np.save(f"{savedir}/{key}/desc.npy", desc)
        logging.info(f"Descriptors for {key} saved")
        del onedata, model, desc
        torch.cuda.empty_cache()

logging.info("All Done !!!")
os.system("mv running done")