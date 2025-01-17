# last modified: 2025-01-17
import numpy as np

import matplotlib.pyplot as plt
import dpdata
import glob
import seaborn as sns
import pandas as pd
import os
import logging

from dpeva.uncertain.rnd import RandomNetworkDistillation
from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from sklearn.decomposition import PCA

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# load data
desc_string_trn = 'descriptors-stage1-trn/*/desc.npy'
desc_string_pool = 'descriptors-stage1-val/*/desc.npy'
#data_trn_path = "data-trn-stage1"
data_pool_path = "data-val-stage1"
dpdata_string = "O*"

# setting of rnd
output_dim = 80  # output dimension
hidden_dim = 160  # hidden dimension
num_residual_blocks = 2  # number of residual blocks
device = "cuda"
num_iter = 100  # number of selection iterations
sample_per_iter = 1  # number of samples per iteration
train_mask = 0.4  # mask ratio for training data
distance_metric = "cossim"
rnd_train_batchsize=4096
rnd_eval_batchsize=4096
rnd_train_numb_batch = lambda x:int(np.sqrt(256 / (x + 1)) * 500)
rnd_train_lr = 1e-3
rnd_train_decay_gamma = 0.95
rnd_train_decay_ratio = 100
rnd_train_disp_freq = 500
rnd_train_save_freq = 2000
rnd_eval_disp_freq = 1
direct_cluster_ratio = 0.2
direct_k = 2
direct_thr_init = 0.05


# read descriptors/*/desc.npy data
logger.info("Reading descriptor results from training data...")
desc_trn_names = []
desc_trn = []
iter_list = sorted(glob.glob(desc_string_trn))
for f in iter_list:
    # extract dirname of desc.npy from descriptors/*
    directory, _ = os.path.split(f)
    _, keyname = os.path.split(directory)
    desc_trn_names.append(keyname)
    one_desc = np.load(f) # nframe, natoms, ndesc
    for i in range(len(one_desc)):
        desc_trn.append(one_desc[i])
logger.info(f"Training data have {np.shape(np.concatenate(desc_trn, axis=0))} size as (natoms, ndesc) with {len(desc_trn)} frames")

# read descriptors/*/desc.npy data
logger.info("Reading descriptor results from pool data...")
desc_pool_names = []
desc_pool_keys = []
desc_pool = []
iter_list = sorted(glob.glob(desc_string_pool))
for f in iter_list:
    # extract dirname of desc.npy from descriptors/*
    directory, _ = os.path.split(f)
    _, keyname = os.path.split(directory)
    desc_pool_names.append(keyname)
    one_desc = np.load(f) # nframe, natoms, ndesc
    for i in range(len(one_desc)):
        desc_pool_keys.append(f"{keyname}-{i}")
        desc_pool.append(one_desc[i])
len(desc_pool), np.shape(np.concatenate(desc_pool, axis=0))
logger.info(f"Pool data have {np.shape(np.concatenate(desc_pool, axis=0))} size as (natoms, ndesc) with {len(desc_pool)} frames ")

# check the dimension of descriptor
assert desc_trn[0].shape[-1] == desc_pool[0].shape[-1], "The descriptor dimension of training and pool data should be the same."

# read dpdata of selection pool
logger.info(f"Reading pool dpdata from {data_pool_path}...")
dpdata_pool =  dpdata.MultiSystems.from_dir(data_pool_path, dpdata_string, fmt="deepmd/npy")

dpdata_pool_dict = {}
for lbsys in dpdata_pool:
    dpdata_pool_dict[lbsys.short_name] = lbsys
        
# initialization of RND    
input_dim = desc_trn[0].shape[-1]  # input dimension
rnd = RandomNetworkDistillation(
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_dim=hidden_dim, 
            num_residual_blocks=num_residual_blocks, 
            distance_metric=distance_metric, 
            device=device)

# iteratively use RND to choose the next data point
# maintain a selected index for the selected data points
RND_global_select_indices = np.array([], dtype=int)
RND_local_select_indices = np.array([], dtype=int)

# mean pooling in pool_desc_list in natom dimension
pool_desc_stru_list = [np.mean(desc, axis=0).reshape(1,desc.shape[-1]) for desc in desc_pool]
pool_desc_stru = np.concatenate(pool_desc_stru_list, axis=0)

# direct selection for pool desc
num_stru_pool = len(pool_desc_stru)
DIRECT_sampler = DIRECTSampler(
    structure_encoder=None,
    clustering=BirchClustering(n=int(num_stru_pool * 0.2), threshold_init=0.05),
    select_k_from_clusters=SelectKFromClusters(k=2),
)
DIRECT_selection = DIRECT_sampler.fit_transform(pool_desc_stru)
DIRECT_selected_indices = DIRECT_selection["selected_indexes"]
pool_desc_stru_selected = pool_desc_stru[DIRECT_selected_indices] 
# the order for DIRECT_select_indices and pool_desc_stru_selected are the same

for iter_ind in range(num_iter):
    logger.info(f"Selecting data point {iter_ind + 1}/{num_iter}...")
    num_batches = rnd_train_numb_batch(iter_ind)  # number of batches 
    decay_steps = num_batches // rnd_train_decay_ratio  # decay steps
    
    # dataset process and selection
    train_array = np.concatenate(desc_trn, axis=0)
    train_index = np.random.choice(
        range(len(train_array)), int((1-train_mask)*len(train_array)), replace=False)
    train_desc_raw = train_array[train_index]
    # add the selected data points to the training set
    if len(RND_global_select_indices) == 0:
        train_desc = train_desc_raw.copy()
    else:
        pool_desc_added = np.concatenate([desc_pool[int(i)] for i in RND_global_select_indices], axis=0)
        train_desc = np.concatenate([train_desc_raw, pool_desc_added], axis=0)
    # generate target desc data which purge the selected data points

    
    logger.info(f"Training data shape: {train_desc.shape}, target data shape: {pool_desc_stru_selected.shape}")
    
    rnd.train(train_desc, 
          num_batches=num_batches,
          batch_size=rnd_train_batchsize,
          initial_lr=rnd_train_lr,
          gamma=rnd_train_decay_gamma,
          decay_steps=decay_steps,
          disp_freq=rnd_train_disp_freq,
          save_freq=rnd_train_save_freq)
    
    # calculate intrinsic reward
    intrinsic_rewards = rnd.eval_intrinsic_rewards(
                    pool_desc_stru_selected, 
                    batch_size=rnd_eval_batchsize, 
                    disp_freq=rnd_eval_disp_freq)
    
    # mask the selected data points
    intrinsic_rewards[list(RND_local_select_indices)] = 0 # or -np.inf
    
    # plot the distribution of intrinsic rewards
    intrinsic_rewards_savedir = "rnd_rewards"
    if not os.path.exists(intrinsic_rewards_savedir):
        os.makedirs(intrinsic_rewards_savedir)
    plt.figure(figsize=(10, 6))
    sns.histplot(intrinsic_rewards, color="blue", label="Intrinsic Reward Distribution", bins=100, kde=True)
    plt.title("Distribution of Intrinsic Rewards by Histogram")
    plt.xlabel("Intrinsic Reward Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{intrinsic_rewards_savedir}/rewards_{iter_ind}.png", dpi=100)
    plt.close() # close the plot

    # select data points by the maximum intrinsic reward, give local indices
    select_ind_iter_local = np.argsort(intrinsic_rewards)[-sample_per_iter:]
    select_ind_iter_global = DIRECT_selected_indices[select_ind_iter_local]

    logger.info(f"Selected data index: {select_ind_iter_local}/{select_ind_iter_global} in DIRECT_sampled/raw pool data with intrinsic rewards: {intrinsic_rewards[select_ind_iter_local]}, which has the system key {[desc_pool_keys[ind] for ind in select_ind_iter_global]}")

    # add the selected index to the selected index array
    RND_local_select_indices = np.concatenate((RND_local_select_indices, select_ind_iter_local))
    RND_global_select_indices = np.concatenate((RND_global_select_indices, select_ind_iter_global))


# save the selected data indice
select_index_array = np.array(RND_global_select_indices)
save_file = "selected_index.npy"
np.save(save_file, select_index_array)
logger.info(f"Selected data index saved to {save_file}")

# PCA reduction for visualization
logger.info("PCA reduction for visualization...")
pca = PCA(n_components=2) 
desc_trn_stru = np.concatenate([np.mean(desc, axis=0).reshape(1,desc.shape[-1]) for desc in desc_trn], axis=0)
desc_pool_stru = np.concatenate([np.mean(desc, axis=0).reshape(1,desc.shape[-1]) for desc in desc_pool], axis=0)
pca_result = pca.fit_transform(desc_trn_stru) # PCA reduction for all data points
trn_points = pca.transform(desc_trn_stru) 
pool_points = pca.transform(desc_pool_stru) 
select_points = pca.transform([desc_pool_stru[i] for i in range(len(desc_pool_stru)) if i in RND_global_select_indices])


plt.figure(figsize=(12, 8))

sns.scatterplot(
    x=trn_points[:, 0], y=trn_points[:, 1],
    color="purple", alpha=0.6, label="Training Points", 
)
sns.scatterplot(
    x=pool_points[:, 0], y=pool_points[:, 1],
    color="blue", alpha=0.4, label="Pool Points",)
sns.scatterplot(
    x=select_points[:, 0], y=select_points[:, 1],
    color="red", label="Unique Points", s=80, edgecolor="black"
)

plt.title("PCA Visualization of Dataset with Uniqueness Coloring")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()

pca_fig_name = "PCA_visualization.png"

plt.savefig(pca_fig_name, dpi=200)

plt.close()

logger.info(f"PCA visualization saved to {pca_fig_name}")

# choose the selected data points in corresponding dpdata
logger.info("Saving selected data points in dpdata format...")
selected_dpdata = dpdata.MultiSystems()
selected_string = ""
for i in RND_global_select_indices:
    keyname, ind = desc_pool_keys[i].split("-")
    # save keyname-ind pairs
    selected_string += f"{keyname}-{ind}\n"
    selected_dpdata.append(dpdata_pool_dict[keyname][int(ind)])
with open("selected_data.txt", "w") as f:
    f.write(selected_string)
selected_dpdata.to_deepmd_npy("selected_data")

logger.info("DONE!")
