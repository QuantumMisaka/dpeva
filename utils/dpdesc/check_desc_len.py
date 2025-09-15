# check if the length of target sys is the same of desc

import dpdata
import numpy as np
import os
import glob
import sys
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

desc_dir = sys.argv[1]
data_dir = sys.argv[2]
format = "deepmd/npy/mixed" # default

desc_paths = sorted(glob.glob(f"{desc_dir}/*"))

for desc_path in desc_paths:
    desc_name = desc_path.split("/")[-1]
    desc_nframe = np.load(f"{desc_path}/desc.npy").shape[0]
    if format == "deepmd/npy/mixed":
        onedata = dpdata.MultiSystems.from_file(f"{data_dir}/{desc_name}", fmt="deepmd/npy/mixed")
    elif format == "deepmd/npy":
        onedata = dpdata.System(f"{data_dir}/{desc_name}", fmt="deepmd/npy")
    else:
        logging.error(f"Unsupported format: {format}")
        break
    data_nframe = onedata.get_nframes()
    if data_nframe != desc_nframe:
        logging.error(f"data {desc_name} has {data_nframe} frames BUT {desc_nframe} in desc.npy")
    else:
        logging.info(f"data {desc_name} has {data_nframe} frames as same in desc.npy")