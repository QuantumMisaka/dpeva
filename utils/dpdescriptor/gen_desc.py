# generated desc for deepmd/npy/mixed (and deepmd/npy)
import dpdata
from deepmd.infer.deep_pot import DeepPot
#from deepmd.calculator import DP
import numpy as np
import os
import logging
import time
import glob
from torch.cuda import empty_cache

datadir = "./sampled-data"
format = "deepmd/npy/mixed" # default
modelpath = "./model.ckpt.pt"
savedir = "descriptors"
head = None # multi head for LAM

omp = 16
batch_size = 4000
os.environ['OMP_NUM_THREADS'] = f'{omp}'

def descriptor_from_model(sys: dpdata.System, model:DeepPot) -> np.ndarray:
    coords = sys.data["coords"]
    cells = sys.data["cells"]
    model_type_map = model.get_type_map()
    type_trans = np.array([model_type_map.index(i) for i in sys.data['atom_names']])
    atypes = list(type_trans[sys.data['atom_types']])
    predict = model.eval_descriptor(coords, cells, atypes)
    return predict

# init
# mixed type need to be read-in iteratively in desc-gen
# perhaps npy data can be dealed with in same manner
# alldata = dpdata.MultiSystems() 

model = DeepPot(modelpath, head=head)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("Start Generating Descriptors")

if not os.path.exists(savedir):
    os.mkdir(savedir)

with open("running", "w") as fo:
    starting_time = time.perf_counter()
    for item in sorted(glob.glob(f"{datadir}/*")):
        key = os.path.split(item)[-1]
        save_key = f"{savedir}/{key}"
        logging.info(f"Generating descriptors for {key} system")
        if os.path.exists(save_key):
            if os.path.exists(f"{save_key}/desc.npy"):
                logging.info(f"Descriptors for {key} already exist, skip")
                continue
        onedata = dpdata.MultiSystems.from_file(item, fmt=format)
        # use for-loop to avoid OOM in old ver
        desc_list = []
        for onesys in onedata:
            onesys: dpdata.System
            for i in range(0, len(onesys), batch_size):
                batch = onesys[i:i + batch_size]  
                desc_batch = descriptor_from_model(batch, model)
                desc_list.append(desc_batch)
        desc = np.concatenate(desc_list, axis=0)
        logging.info(f"Descriptors for {key} generated")
        os.mkdir(save_key)
        np.save(f"{savedir}/{key}/desc.npy", desc)
        logging.info(f"Descriptors for {key} saved")
        del onedata, desc, desc_list
        empty_cache()
    ending_time = time.perf_counter()
    fo.write(f"DONE in {ending_time - starting_time} sec !")

logging.info("All Done !!!")
os.system("mv running done")