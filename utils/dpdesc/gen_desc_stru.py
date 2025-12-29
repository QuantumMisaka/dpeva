# generated desc_stru for deepmd/npy/mixed (and deepmd/npy)
import dpdata
from deepmd.infer.deep_pot import DeepPot
#from deepmd.calculator import DP
import numpy as np
import os
import logging
import sys
import time
import glob
from torch.cuda import empty_cache

datadir = sys.argv[1]
format = "deepmd/npy" # default
modelpath = sys.argv[2]
savedir = f"desc-{modelpath.split(".")[0]}-{datadir}"
head = "OC20M" # multi head for LAM

omp = 24
os.environ['OMP_NUM_THREADS'] = f'{omp}'
batch_size = 1000  
# batch_size can be as large as possible, but should all in one node
# if any problem encountered, set batch_size to 1

# notice: DeepPot.eval_descriptor have a parameter "mixed_type"
def descriptor_from_model(sys: dpdata.System, model:DeepPot, nopbc=False) -> np.ndarray:
    coords = sys.data["coords"]
    cells = sys.data["cells"]
    if nopbc:
        cells = None
    model_type_map = model.get_type_map()
    type_trans = np.array([model_type_map.index(i) for i in sys.data['atom_names']])
    atypes = list(type_trans[sys.data['atom_types']])
    predict = model.eval_descriptor(coords, cells, atypes)
    return predict

def get_desc_by_batch(sys: dpdata.System, model:DeepPot, batch_size: int, nopbc=False) -> list:
    desc_list = []
    for i in range(0, len(sys), batch_size):
        batch = sys[i:i + batch_size]  
        desc_batch = descriptor_from_model(batch, model, nopbc=nopbc)
        desc_list.append(desc_batch)
    return desc_list

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
    os.makedirs(savedir)

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
        if format == "deepmd/npy/mixed":
            onedata = dpdata.MultiSystems.from_file(item, fmt=format)
        else:
            onedata = dpdata.System(item, fmt=format)
        # use for-loop to avoid OOM in old ver
        desc_list = []
        if format == "deepmd/npy/mixed":
            for onesys in onedata:
                nopbc = onesys.data.get('nopbc', False)
                one_desc_list = get_desc_by_batch(onesys, model, batch_size, nopbc)
                desc_list.extend(one_desc_list)
        else:
            nopbc = onedata.data.get('nopbc', False)
            desc_list = get_desc_by_batch(onedata, model, batch_size, nopbc)
        desc = np.concatenate(desc_list, axis=0)
        desc_stru = np.mean(desc, axis=1)
        logging.info(f"Descriptors STRU for {key} generated")
        os.mkdir(save_key)
        np.save(f"{savedir}/{key}/desc_stru.npy", desc_stru)
        logging.info(f"Descriptors STRU for {key} saved")
        del onedata, desc, desc_stru
        empty_cache()
    ending_time = time.perf_counter()
    fo.write(f"DONE in {ending_time - starting_time} sec !")

logging.info("All Done !!!")
os.system("mv running done")