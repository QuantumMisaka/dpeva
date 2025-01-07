import dpdata
from deepmd.infer.deep_pot import DeepPot
#from deepmd.calculator import DP
import numpy as np
import os
import time
import gc
import sys
import logging

modelpath = "./FeCHO-dpa231-v2-7-3heads-100w.pt"
onedir = sys.argv[1]
if len(sys.argv) < 3:
    savedir = "descriptors"
else:
    savedir= sys.argv[2]

omp = 16
os.environ['OMP_NUM_THREADS'] = f'{omp}'

def descriptor_from_model(sys: dpdata.LabeledSystem, model:DeepPot):
    coords = sys.data["coords"]
    cells = sys.data["cells"]
    model_type_map = model.get_type_map()
    type_trans = np.array([model_type_map.index(i) for i in sys.data['atom_names']])
    atypes = list(type_trans[sys.data['atom_types']])
    predict = model.eval_descriptor(coords, cells, atypes)
    return predict

# 配置日志格式，包含时间戳、日志级别和日志信息
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format='%(asctime)s - %(levelname)s - %(message)s',  # 定义日志格式
    datefmt='%Y-%m-%d %H:%M:%S'  # 定义时间戳格式
)


# generate descriptor for alldata
# desc_dict = {}
logging.info("Start Generating Descriptors")

if not os.path.exists(savedir):
    os.mkdir(savedir)

onedata = dpdata.LabeledSystem(onedir, fmt="deepmd/npy")
key = onedata.short_name
save_key = f"{savedir}/{key}"
logging.info(f"Generating descriptors for {key}")
if os.path.exists(save_key):
    if os.path.exists(f"{save_key}/desc.npy"):
        logging.info(f"Descriptors for {key} already exist, skip")
        sys.exit(0)
else:
    model = DeepPot(modelpath, head="Target_FTS")
    desc = descriptor_from_model(onedata, model)
    #desc_dict[key] = desc
    os.mkdir(save_key)
    np.save(f"{savedir}/{key}/desc.npy", desc)
logging.info(f"Descriptors for {key} Done")
