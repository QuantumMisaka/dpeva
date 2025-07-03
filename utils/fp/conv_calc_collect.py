# read stru from abacus/scf and transfer it to deepmd/npy for training
# update by JamesMisaka in 2025-06-28

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w',
    filename="UQ-DIRECT-selection.log",
)

import numpy as np
import os
import sys
from ase.io.abacus import write_input, write_abacus
from copy import deepcopy
from random import randint
import dpdata
import glob
from tqdm import tqdm
import numpy as np


dataset_names = [
    "./alex-2d-1d-FeCOH/",
    "./alex-3d-FeCOH/",
    "./amourC/",
    "./C-GAP2020/",
    "./DeepCNT/",
    "./Fe-C-ActMat2024/",
    "./Fe-O-npj2025",
    "./FeCHO-zpliu/",
    "./matpes-FeCOH/",
    "./mptrj-FeCOH/",
    "./oc2m-FeCOH/",
    "./oc22-FeCOH/",
    "./omat24-FeCOH/"
]

datatype_names = [
    "bulk",
    "layer",
    "string",
    "cluster",
    "cubic_cluster"
]

conv_dir = "CONVERGED"

alldata_conv = dpdata.MultiSystems()
alldata_noconv = dpdata.MultiSystems()


for dataset in dataset_names:
    for datatype in datatype_names:
        target_dir = os.path.join(dataset, datatype)
        if os.path.exists(target_dir):
            logging.info(f"collecting {target_dir}")
            for dir in sorted(glob.glob(f"{target_dir}/{conv_dir}/*/")):
                onedata = dpdata.LabeledSystem(dir, fmt="abacus/scf")
                alldata_conv.append(onedata)
            for dir in sorted(glob.glob(f"{target_dir}/*/")):
                if conv_dir not in dir:
                    onedata = dpdata.System(dir, fmt="abacus/scf")
                    alldata_noconv.append(onedata)
logging.info("collecting done")
alldata_conv.to_deepmd_npy("./dpeva-fp-conv/")
alldata_noconv.to_deepmd_npy("./dpeva-fp-noconv/")