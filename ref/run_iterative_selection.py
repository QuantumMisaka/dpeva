import os
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import argparse
from random import randint
from multiprocessing import cpu_count

import torch
from torch.optim import Adam

from schnetpack import AtomsLoader, AtomsData
from schnetpack.environment import TorchEnvironmentProvider
from schnetpack.train import build_mse_loss

from mlsuite.novelty_models.random_network_distillation import RandomNetworkDistillation
from mlsuite.utils.torch_utils import weight_reset

from inp_helper import get_target_model, build_kld_loss, build_cossim_loss, build_ce_loss, build_kld_nov_score

PROJECT_PATH = "/home/jfinkbeiner/MasterThesis/projects/novelty_methods"

def determine_argmax(db_name, property_):
    from ase.db import connect
    max_val = -np.inf
    argmax = 0
    # print(db_name)
    with connect(db_name) as db:
        # print(len(db))
        for row in db.select():
            val = row.data.get(property_)
            if val is None:
                val = row.get(property_)
            # print(val)
            if isinstance(val, np.ndarray):
                val = np.max(val)
            if val > max_val:
                max_val = val
                argmax = row.id
    return argmax


def iter_to_num_epochs(iteration):
    return int(np.sqrt(128 / iteration) * 250)

def main(args):

    system = "NaCl"
    rcut = 6.0 # LiCl: 5.5, NaCl: 6.0, KCl: 7.0
    model_type = "large-ext"
    ensemble = "npt"
    learning_rate = 1e-3
    data_run = args.data_run
    samples_per_iter = 1
    nov_func_name = "cossim"


    if ensemble=="nvt":
        db_num_samples = 2048
    elif ensemble=="npt":
        db_num_samples = 4096
    else:
        raise ValueError

    # if system == "Ar":
    #     elements = [18]
    # elif system == "NaCl":
    #     elements = [11, 17]
    # else:
    #     raise ValueError

    model = get_target_model(model_type, rcut)
    optimizer = Adam
    optimizer_kwargs = dict(lr=learning_rate)
    buffer_db_name = os.path.join(PROJECT_PATH, f"data/{system}/{system}_{ensemble}_run{int(data_run)}.db")
    # buffer_db_name = os.path.join(PROJECT_PATH, f"data/{system}/schnet_dbs/{system}_sss_energies_{ensemble}_{int(db_num_samples)}samples_run{int(data_run)}.db")
    # buffer_db_name = "/auto.eland/home/jfinkbeiner/MasterThesis/projects/classical_proofs/data/NaCl/simulations/NaCl_nvt_run1_rcut7.5_a900605e/LAMMPS_SIMS_e0959656.db"

    # crit = torch.nn.KLDivLoss(reduction='none', log_target=True) # TODO correct output shape ?

    if nov_func_name == "kld":
        loss_fn = build_ce_loss(["target"])
        novelty_func = build_kld_nov_score()
    elif nov_func_name == "mse":
        loss_fn = build_mse_loss(["target"])
        crit = torch.nn.MSELoss(reduction='none')
        novelty_func = lambda target, pred: torch.sum(crit(pred, target), dim=-1)
    elif nov_func_name == "cossim":
        loss_fn = build_cossim_loss(["target"])
        crit = torch.nn.CosineSimilarity(dim=2)
        novelty_func = lambda target, pred: -(crit(target, pred)-1) 
    else:
        raise ValueError(f"unknown `nov_func_name`, got '{novelty_func}'")

    
    # TODO change back
    model_path = os.path.join(PROJECT_PATH, "test_distribs", system, f"{system}_{ensemble}_{nov_func_name}_{model_type}_run{int(data_run)}_periter{samples_per_iter}_{randint(16**7, 16**8 - 1):x}")
    # model_path = os.path.join(PROJECT_PATH, "analysis", system, f"{system}_{ensemble}_{nov_func_name}_{model_type}_run{int(data_run)}_periter{samples_per_iter}_{randint(16**7, 16**8 - 1):x}")
    save_filename = os.path.join(model_path, f"selected_samples.json")

    rnd = RandomNetworkDistillation(
        model_path=model_path,
        model=model,
        cutoff=rcut,
        # buffer_db_name=buffer_db_name,
        loss_fn=loss_fn,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        train_batch_size=8,
        eval_batch_size=128,
        epochs_per_iter=iter_to_num_epochs,
        samples_per_iter=samples_per_iter,
        max_iter=128, # TODO
        num_workers=cpu_count(),
        device=torch.device("cuda"),
        novelty_func=novelty_func,
        save_distrbs_interval=1, # TODO
    )

    # init_subset = [4000]
    init_subset = [determine_argmax(buffer_db_name, "energies")-1] # TODO tmp inedx fix
    # init_subset = [150385-147249+1]
    target_subset=None

    # print(init_subset)
    # from ase.db import connect
    # with connect(buffer_db_name) as db:
    #    print(db.get(init_subset[0]-1).energy)
    #    print(db.get(init_subset[0]).energy)
    #    print(db.get(init_subset[0]+1).energy)

    # sys.exit()

    iter_selections = rnd.select_iteratively(
        buffer_db_name=buffer_db_name, 
        init_subset=init_subset.copy(), 
        target_subset=target_subset, 
        return_iter_selections=True,
        save_selection_filename=save_filename,
    )

    with open(save_filename, "w") as jsonfile:
        json.dump(iter_selections, jsonfile)


def try_argmax():

    buffer_db_name = os.path.join("/home/jfinkbeiner/MasterThesis/projects/classical_proofs/data/Ar/Ar.db")
    name = "sss-energies-npt-4096samples-run3"
    
    from ase.db import connect
    property_ = "energies"
    max_val = -np.inf
    argmax = 0
#     print(buffer_db_name)
    with connect(buffer_db_name) as db:
#         print(db.get(1))
#         print(len(db))
        for row in db.select(name=name):
            val = row.data.get(property_)
            if val is None:
                val = row.get(property_)
#             print(val)
            if isinstance(val, np.ndarray):
                val = np.max(val)
            if val > max_val:
                argmax = row.id
                max_val = val
            print(row.id, val, row.energy)
    print(argmax)
    return argmax



if __name__ == "__main__":
    # try_argmax()

    parser = argparse.ArgumentParser(description='Select configurations via the Random Network Distillation method.')
    parser.add_argument('--data_run', type=int, default=1, help='which data-creation-run to use.')
    args = parser.parse_args()

    main(args)
