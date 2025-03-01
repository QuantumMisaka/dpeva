import dpdata
from deepmd.infer.deep_pot import DeepPot
#from deepmd.calculator import DP
import numpy as np
import os
import logging
from torch.cuda import empty_cache

datadir = "./sampled-data-direct-10p-npy"
format = "deepmd/npy" # default
modelpath = "./model.ckpt.pt"
savedir = "descriptors"
data_string = "O*" # for dpdata.MultiSystems.from_dir

omp = 16
batch_size = 4
os.environ['OMP_NUM_THREADS'] = f'{omp}'

def descriptor_from_model(sys: dpdata.System, model:DeepPot) -> np.ndarray:
    coords = sys.data["coords"]
    cells = sys.data["cells"]
    model_type_map = model.get_type_map()
    type_trans = np.array([model_type_map.index(i) for i in sys.data['atom_names']])
    atypes = list(type_trans[sys.data['atom_types']])
    predict = model.eval_descriptor(coords, cells, atypes)
    return predict
#alldata = dpdata.MultiSystems.from_file(datadir, fmt=format)
alldata = dpdata.MultiSystems.from_dir(datadir, data_string, fmt=format)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Start Generating Descriptors")

if not os.path.exists(savedir):
    os.mkdir(savedir)

with open("running", "w") as fo:
    for onedata in alldata:
        onedata: dpdata.System
        key = onedata.short_name
        save_key = f"{savedir}/{key}"
        logging.info(f"Generating descriptors for {key}")
        if os.path.exists(save_key):
            if os.path.exists(f"{save_key}/desc.npy"):
                logging.info(f"Descriptors for {key} already exist, skip")
                continue
        model = DeepPot(modelpath, head="Target_FTS")
        #desc = descriptor_from_model(onedata, model)
        # use for-loop to avoid OOM
        desc_list = []
        for i in range(0, len(onedata), batch_size):
            batch = onedata[i:i + batch_size]  
            desc_batch = descriptor_from_model(batch, model)
            desc_list.append(desc_batch)
        desc = np.concatenate(desc_list, axis=0)
        logging.info(f"Descriptors for {key} generated")
        os.mkdir(save_key)
        np.save(f"{savedir}/{key}/desc.npy", desc)
        logging.info(f"Descriptors for {key} saved")
        del onedata, model, desc, desc_list
        empty_cache()

logging.info("All Done !!!")