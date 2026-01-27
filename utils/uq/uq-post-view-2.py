import logging
# parameters
# testing setting
project = "."
uq_select_scheme = "tangent_lo"  # strict, circle_lo, tangent_lo, crossline_lo, loose
testing_dir = "test_val"
testing_head = "results"
# descriptor loading setting
desc_dir = f"{project}/desc_pool"
# testdata setting
testdata_dir = f"{project}/other_dpdata_test"
testdata_fmt = "deepmd/npy"
# figure setting
kde_bw_adjust = 0.5
fig_dpi = 150
# save setting
root_savedir = "dpeva_uq_post"
view_savedir = f"./{project}/{root_savedir}/view"
dpdata_savedir = f"./{project}/{root_savedir}/dpdata"
df_savedir = f"./{project}/{root_savedir}/dataframe"
# selection setting
uq_qbc_trust_lo = 0.15
uq_qbc_trust_hi = 0.40
uq_rnd_rescaled_trust_lo = uq_qbc_trust_lo
uq_rnd_rescaled_trust_hi = uq_qbc_trust_hi
num_selection = 100
direct_k = 1
direct_thr_init = 0.5

# log setting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    filemode='w',
    filename="UQ-DIRECT-selection.log",
)

logger = logging.getLogger(__name__)

import dpdata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import os
import shutil
import glob

from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from dpeva.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
from dpeva.io.dataproc import DPTestResults

plt.rcParams['xtick.direction'] = 'in'  # set the direction of xticks inside
plt.rcParams['ytick.direction'] = 'in'  # set the direction of yticks inside
plt.rcParams['font.size'] = 10  # set the font size

# check and make the directories
logger.info(f"Initializing selection in {project} ---")
if os.path.exists(project) == False:
    logger.error(f"Project directory {project} not found!")
    raise ValueError(f"Project directory {project} not found!")
if os.path.exists(f"{project}/0/{testing_dir}") == False:
    logger.error(f"Testing directory {testing_dir} not found!")
    raise ValueError(f"Testing directory {testing_dir} not found!")
if os.path.exists(desc_dir) == False:
    logger.error(f"Descriptor directory {desc_dir} not found!")
    raise ValueError(f"Descriptor directory {desc_dir} not found!")
if os.path.exists(testdata_dir) == False:
    logger.error(f"Testdata directory {testdata_dir} not found!")
    raise ValueError(f"Testdata directory {testdata_dir} not found!")
uq_options = ["strict", 
              "circle_lo", 
              "crossline_lo", 
              "tangent_lo",
              "loose"]
if uq_select_scheme not in uq_options:
    logger.error(f"UQ selection scheme {uq_select_scheme} not supported! Please choose from {uq_options}.")
    raise ValueError(f"UQ selection scheme {uq_select_scheme} not supported! Please choose from {uq_options}.")
if ((uq_qbc_trust_lo >= uq_qbc_trust_hi) 
    or (uq_rnd_rescaled_trust_lo >= uq_rnd_rescaled_trust_hi)):
    raise ValueError("Low trust threshold should be lower than High trust thershold !")

if os.path.exists(view_savedir) == False:
    os.makedirs(view_savedir)
if os.path.exists(dpdata_savedir) == False:
    os.makedirs(dpdata_savedir)
if os.path.exists(df_savedir) == False:
    os.makedirs(df_savedir)

# load the test results
logger.info("Loading the test results")
dp_test_results_0 = DPTestResults(f"./{project}/0/{testing_dir}/{testing_head}")
dp_test_results_1 = DPTestResults(f"./{project}/1/{testing_dir}/{testing_head}")
dp_test_results_2 = DPTestResults(f"./{project}/2/{testing_dir}/{testing_head}")
dp_test_results_3 = DPTestResults(f"./{project}/3/{testing_dir}/{testing_head}")

# deal with force difference between 0 head prediction and existing label, only for view
logger.info("Dealing with force difference between 0 head prediction and existing label")
has_ground_truth = dp_test_results_0.has_ground_truth
if has_ground_truth:
    diff_f_0 = np.sqrt(dp_test_results_0.diff_fx**2 + dp_test_results_0.diff_fy**2 + dp_test_results_0.diff_fz**2)
    # map diff_f_0 to each structures with max force diff
    index = 0
    diff_maxf_0_frame = []
    diff_rmsf_0_frame = []
    for item in dp_test_results_0.dataname_list:
        natom = item[2]
        diff_f_0_item = diff_f_0[index:index + natom]
        diff_maxf_0_frame.append(np.max(diff_f_0_item))
        diff_rmsf_0_frame.append(np.sqrt(np.mean(diff_f_0_item**2)))
        index += natom
    diff_maxf_0_frame = np.array(diff_maxf_0_frame)
    diff_rmsf_0_frame = np.array(diff_rmsf_0_frame)
else:
    logger.info("Ground truth not found (all zeros), skipping force difference calculation.")
    diff_maxf_0_frame = None
    diff_rmsf_0_frame = None

# deal with atomic force and average 1, 2, 3
logger.info("Dealing with atomic force and average 1, 2, 3")
fx_0 = dp_test_results_0.data_f['pred_fx']
fy_0 = dp_test_results_0.data_f['pred_fy']
fz_0 = dp_test_results_0.data_f['pred_fz']
fx_1 = dp_test_results_1.data_f['pred_fx']
fy_1 = dp_test_results_1.data_f['pred_fy']
fz_1 = dp_test_results_1.data_f['pred_fz']
fx_2 = dp_test_results_2.data_f['pred_fx']
fy_2 = dp_test_results_2.data_f['pred_fy']
fz_2 = dp_test_results_2.data_f['pred_fz']
fx_3 = dp_test_results_3.data_f['pred_fx']
fy_3 = dp_test_results_3.data_f['pred_fy']
fz_3 = dp_test_results_3.data_f['pred_fz'] 
fx_expt = np.mean((fx_1, fx_2, fx_3), axis=0)
fy_expt = np.mean((fy_1, fy_2, fy_3), axis=0)
fz_expt = np.mean((fz_1, fz_2, fz_3), axis=0)

# deal with atomic force UQ
# use DPGEN formula: 
# \epsilon_t=\max _i \sqrt{\left\langle\left\|F_{w, i}\left(\mathcal{R}_t\right)-\left\langle F_{w, i}\left(\mathcal{R}_t\right)\right\rangle\right\|^2\right\rangle}
# starting from atomic force
logger.info("Dealing with atomic force UQ by DPGEN formula, in QbC and RND-like")

# deal with QbC force UQ
logger.info("Dealing with QbC force UQ")
fx_qbc_square_diff =  np.mean(((fx_1 - fx_expt)**2, (fx_2 - fx_expt)**2, (fx_3 - fx_expt)**2), axis=0) 
fy_qbc_square_diff =  np.mean(((fy_1 - fy_expt)**2, (fy_2 - fy_expt)**2, (fy_3 - fy_expt)**2), axis=0) 
fz_qbc_square_diff =  np.mean(((fz_1 - fz_expt)**2, (fz_2 - fz_expt)**2, (fz_3 - fz_expt)**2), axis=0) 
f_qbc_stddiff = np.sqrt(fx_qbc_square_diff + fy_qbc_square_diff + fz_qbc_square_diff)

# assign atomic force stddiff to each structure and get UQ by max atomic force diff
index = 0
uq_qbc_for_list = []
for item in dp_test_results_0.dataname_list:
    natom = item[2]
    f_qbc_stddiff_item = f_qbc_stddiff[index:index + natom]
    uq_qbc_for_list.append(np.max(f_qbc_stddiff_item))
    index += natom
uq_qbc_for = np.array(uq_qbc_for_list)

# deal with RND-like force UQ
logger.info("Dealing with RND-like force UQ")
fx_rnd_square_diff = np.mean(((fx_1 - fx_0)**2, (fx_2 - fx_0)**2, (fx_3 - fx_0)**2), axis=0) 
fy_rnd_square_diff = np.mean(((fy_1 - fy_0)**2, (fy_2 - fy_0)**2, (fy_3 - fy_0)**2), axis=0) 
fz_rnd_square_diff = np.mean(((fz_1 - fz_0)**2, (fz_2 - fz_0)**2, (fz_3 - fz_0)**2), axis=0) 
f_rnd_stddiff = np.sqrt(fx_rnd_square_diff + fy_rnd_square_diff + fz_rnd_square_diff)

# assign atomic force stddiff to each structure and get UQ by max atomic force diff
index = 0
uq_rnd_for_list = []
for item in dp_test_results_0.dataname_list:
    natom = item[2]
    f_rnd_stddiff_item = f_rnd_stddiff[index:index + natom]
    uq_rnd_for_list.append(np.max(f_rnd_stddiff_item))
    index += natom
uq_rnd_for = np.array(uq_rnd_for_list)

# align RND to QbC by RobustScaler (Median and IQR)
# This is more robust to outliers than Z-Score (Mean and Std)
logger.info("Aligning UQ-RND to UQ-QbC by RobustScaler (Median/IQR alignment)")
from sklearn.preprocessing import RobustScaler
scaler_qbc_for = RobustScaler()
scaler_rnd_for = RobustScaler()
uq_qbc_for_scaled = scaler_qbc_for.fit_transform(uq_qbc_for.reshape(-1,1)).flatten()
uq_rnd_for_scaled = scaler_rnd_for.fit_transform(uq_rnd_for.reshape(-1,1)).flatten()
uq_rnd_for_rescaled = scaler_qbc_for.inverse_transform(uq_rnd_for_scaled.reshape(-1,1)).flatten()

# simplify the variables
uq_x_lo = uq_qbc_trust_lo
uq_y_lo = uq_rnd_rescaled_trust_lo
uq_x_hi = uq_qbc_trust_hi
uq_y_hi = uq_rnd_rescaled_trust_hi

# Filter UQ values for visualization
# Only keep values in [0, 2] for visualization purposes as requested
def filter_uq_for_viz(data, name="UQ"):
    """Filter UQ data to be within [0, 2] and warn if truncation occurs."""
    mask = (data >= 0) & (data <= 2.0)
    truncated_count = len(data) - np.sum(mask)
    if truncated_count > 0:
        logger.warning(f"{name}: Truncating {truncated_count} values outside [0, 2] for visualization.")
    return data[mask], mask

# Stats for UQ variables
logger.info("Calculating statistics for UQ variables (QbC, RND, RND_rescaled)")
df_uq_stats = pd.DataFrame({
    "UQ_QbC": uq_qbc_for,
    "UQ_RND": uq_rnd_for,
    "UQ_RND_rescaled": uq_rnd_for_rescaled
})
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)
stats_desc = df_uq_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
logger.info(f"UQ Statistics:\n{stats_desc}")

uq_qbc_for_viz, mask_qbc = filter_uq_for_viz(uq_qbc_for, "UQ-QbC")
uq_rnd_for_viz, mask_rnd = filter_uq_for_viz(uq_rnd_for, "UQ-RND")
uq_rnd_for_rescaled_viz, mask_rnd_rescaled = filter_uq_for_viz(uq_rnd_for_rescaled, "UQ-RND-rescaled")

# plot and save the figures of UQ-force
# pass
logger.info("Plotting and saving the figures of UQ-force")
plt.figure(figsize=(8, 6))
if len(uq_qbc_for_viz) > 0:
    sns.kdeplot(uq_qbc_for_viz, color="blue", label="UQ-QbC", bw_adjust=0.5)
if len(uq_rnd_for_viz) > 0:
    sns.kdeplot(uq_rnd_for_viz, color="red", label="UQ-RND", bw_adjust=0.5)
plt.title("Distribution of UQ-force by KDEplot (Truncated [0, 2])")
plt.xlabel("UQ Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig(f"./{view_savedir}/UQ-force.png", dpi=fig_dpi)
plt.close()

# plot and save the figures of UQ-force rescaled
# pass
logger.info("Plotting and saving the figures of UQ-force rescaled")
plt.figure(figsize=(8, 6))
if len(uq_qbc_for_viz) > 0:
    sns.kdeplot(uq_qbc_for_viz, color="blue", label="UQ-QbC", bw_adjust=0.5)
if len(uq_rnd_for_rescaled_viz) > 0:
    sns.kdeplot(uq_rnd_for_rescaled_viz, color="red", label="UQ-RND-rescaled", bw_adjust=0.5)
plt.title("Distribution of UQ-force by KDEplot (Truncated [0, 2])")
plt.xlabel("UQ Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig(f"./{view_savedir}/UQ-force-rescaled.png", dpi=fig_dpi)
plt.close()

# plot and save the figures of UQ-QbC-force with UQ trust range
# pass
logger.info("Plotting and saving the figures of UQ-QbC-force with UQ trust range")
plt.figure(figsize=(8, 6))
if len(uq_qbc_for_viz) > 0:
    sns.kdeplot(uq_qbc_for_viz, color="blue", bw_adjust=0.5)
plt.title("Distribution of UQ-QbC-force by KDEplot (Truncated [0, 2])")
plt.xlabel("UQ-QbC Value")
plt.ylabel("Density")
plt.grid(True)
plt.axvline(uq_x_lo, color='purple', linestyle='--', linewidth=1)
plt.axvline(uq_x_hi, color='purple', linestyle='--', linewidth=1)
# show the UQ trust range
# Use viz min/max for span
viz_min = np.min(uq_qbc_for_viz) if len(uq_qbc_for_viz) > 0 else 0
viz_max = np.max(uq_qbc_for_viz) if len(uq_qbc_for_viz) > 0 else 2
plt.axvspan(viz_min, uq_x_lo, alpha=0.1, color='green')
plt.axvspan(uq_x_lo, uq_x_hi, alpha=0.1, color='yellow')
plt.axvspan(uq_x_hi, viz_max, alpha=0.1, color='red')
plt.savefig(f"./{view_savedir}/UQ-QbC-force.png", dpi=fig_dpi)
plt.close()

# plot and save the figures of UQ-RND-force-rescaled with UQ trust range
# pass
logger.info("Plotting and saving the figures of UQ-RND-force-rescaled with UQ trust range")
plt.figure(figsize=(8, 6))
if len(uq_rnd_for_rescaled_viz) > 0:
    sns.kdeplot(uq_rnd_for_rescaled_viz, color="blue", bw_adjust=0.5)
plt.title("Distribution of UQ-RND-force-rescaled by KDEplot (Truncated [0, 2])")
plt.xlabel("UQ-RND-rescaled Value")
plt.ylabel("Density")
plt.grid(True)
plt.axvline(uq_x_lo, color='purple', linestyle='--', linewidth=1)
plt.axvline(uq_x_hi, color='purple', linestyle='--', linewidth=1)
# show the UQ trust range
viz_min_rnd = np.min(uq_rnd_for_rescaled_viz) if len(uq_rnd_for_rescaled_viz) > 0 else 0
viz_max_rnd = np.max(uq_rnd_for_rescaled_viz) if len(uq_rnd_for_rescaled_viz) > 0 else 2
plt.axvspan(viz_min_rnd, uq_x_lo, alpha=0.1, color='green')
plt.axvspan(uq_x_lo, uq_x_hi, alpha=0.1, color='yellow')
plt.axvspan(uq_x_hi, viz_max_rnd, alpha=0.1, color='red')
plt.savefig(f"./{view_savedir}/UQ-RND-force-rescaled.png", dpi=fig_dpi)
plt.close()


# plot and save the figures of UQ-force vs force diff
# pass
if has_ground_truth:
    logger.info("Plotting and saving the figures of UQ-force vs force diff")
    plt.figure(figsize=(8, 6))
    
    # Filter for scatter plots: we need corresponding Y values
    mask_qbc_scatter = (uq_qbc_for >= 0) & (uq_qbc_for <= 2.0)
    plt.scatter(uq_qbc_for[mask_qbc_scatter], diff_maxf_0_frame[mask_qbc_scatter], color="blue", label="QbC",s=20)
    
    mask_rnd_scatter = (uq_rnd_for >= 0) & (uq_rnd_for <= 2.0)
    plt.scatter(uq_rnd_for[mask_rnd_scatter], diff_maxf_0_frame[mask_rnd_scatter], color="red", label="RND",s=20)
    
    plt.title("UQ vs Force Diff (Truncated [0, 2])")
    plt.xlabel("UQ Value")
    plt.ylabel("True Max Force Diff")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{view_savedir}/UQ-force-fdiff-parity.png", dpi=fig_dpi)
    plt.close()

    # plot and save the figures of UQ-force-rescaled vs force diff
    # pass
    logger.info("Plotting and saving the figures of UQ-force-rescaled vs force diff")
    plt.figure(figsize=(8, 6))
    
    # We already have mask_qbc_scatter
    plt.scatter(uq_qbc_for[mask_qbc_scatter], diff_maxf_0_frame[mask_qbc_scatter], color="blue", label="QbC", s=20)
    
    mask_rnd_rescaled_scatter = (uq_rnd_for_rescaled >= 0) & (uq_rnd_for_rescaled <= 2.0)
    plt.scatter(uq_rnd_for_rescaled[mask_rnd_rescaled_scatter], diff_maxf_0_frame[mask_rnd_rescaled_scatter], color="red", label="RND-rescaled", s=20)
    
    plt.title("UQ vs Force Diff (Truncated [0, 2])")
    plt.xlabel("UQ Value")
    plt.ylabel("True Max Force Diff")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{view_savedir}/UQ-force-rescaled-fdiff-parity.png", dpi=fig_dpi)
    plt.close()

# difference between UQ-qbc and UQ-rnd-rescaled
logger.info("Calculating the difference between UQ-qbc and UQ-rnd-rescaled")
uq_diff_for_scaled_to_qbc = np.abs(uq_rnd_for_rescaled - uq_qbc_for)

# plot and save the figures of UQ-force-diff vs UQ-force
# pass
logger.info("Plotting and saving the figures of UQ-diff vs UQ")
plt.figure(figsize=(8, 6))
plt.scatter(uq_diff_for_scaled_to_qbc, uq_qbc_for, color="blue", label="UQ-qbc-for", s=20)
plt.scatter(uq_diff_for_scaled_to_qbc, uq_rnd_for_rescaled, color="red", label="UQ-rnd-for-rescaled", s=20)
plt.title("UQ-diff vs UQ")
plt.xlabel("UQ-diff Value")
plt.ylabel("UQ Value")
plt.legend()
plt.grid(True)
plt.savefig(f"./{view_savedir}/UQ-diff-UQ-parity.png", dpi=fig_dpi)
plt.close()

# plot and save the figures of UQ-force-diff vs force diff
# pass
if has_ground_truth:
    logger.info("Plotting and saving the figures of UQ-diff vs force diff")
    plt.figure(figsize=(8, 6))
    plt.scatter(uq_diff_for_scaled_to_qbc, diff_maxf_0_frame, color="blue", label="UQ-diff-force", s=20)
    plt.title("UQ-diff vs Force Diff")
    plt.xlabel("UQ-diff Value")
    plt.ylabel("True Max Force Diff")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{view_savedir}/UQ-diff-fdiff-parity.png", dpi=fig_dpi)
    plt.close()

    # plot and save the sns.scatterplot figures of UQ-qbc-force and UQ-rnd-force-rescaled vs force diff
    logger.info("Plotting and saving the figures of UQ-qbc-force and UQ-rnd-force-rescaled vs force diff")
    if has_ground_truth:
        df_uq_maxfor = pd.DataFrame({"UQ-QbC": uq_qbc_for, "UQ-RND-rescaled": uq_rnd_for_rescaled, "Max Force Diff": diff_maxf_0_frame})
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_uq_maxfor, 
                        x="UQ-QbC", 
                        y="UQ-RND-rescaled", 
                        hue="Max Force Diff", 
                        palette="Reds",
                        alpha=0.8,
                        s=60)
        plt.title("UQ-QbC and UQ-RND vs Max Force Diff", fontsize=14)
        plt.grid(True)
        plt.xlabel("UQ-QbC Value", fontsize=12)
        plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
        plt.legend(title="Max Force Diff", fontsize=10)
        plt.grid(True)
        # set the xticks and yticks
        ax = plt.gca()
        # Adaptive locator: use MultipleLocator(0.1) only if range is small enough
        if df_uq_maxfor["UQ-QbC"].max() < 5 and df_uq_maxfor["UQ-RND-rescaled"].max() < 5:
            x_major_locator = mtick.MultipleLocator(0.1)
            y_major_locator = mtick.MultipleLocator(0.1)
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
        # plot the UQ region of trust inside the plot
        plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
        plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)
        if uq_select_scheme == "strict":
            plt.plot([uq_x_lo, uq_x_lo], [uq_y_lo, uq_y_hi], color='purple', linestyle='--', linewidth=2)
            plt.plot([uq_x_lo, uq_x_hi], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
        elif uq_select_scheme == "circle_lo":
            # ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 >= (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
            center = (uq_x_hi, uq_y_hi)
            radius = np.sqrt((uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
            theta = np.linspace(np.pi, 1.5*np.pi, 100)
            x_val = center[0] + radius * np.cos(theta)
            y_val = center[1] + radius * np.sin(theta)
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif uq_select_scheme == "tangent_lo":
            # ((uq_x-uq_x_hi)*(uq_x_hi-uq_x_lo) + (uq_y-uq_y_hi)*(uq_y_hi-uq_y_lo) >= 0)
            x_val = np.linspace(0, uq_x_hi, 100)
            y_val = - (uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
            x_val = x_val[y_val < uq_y_hi]
            y_val = y_val[y_val < uq_y_hi]
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif uq_select_scheme == "crossline_lo":
            def balance_linear_func(x_lo, x_hi, y_lo, y_hi, x_range=(0,10), num_points=20):
                # y1 = (y_hi*x_lo - (y_hi - y_lo)*x)/x_lo
                # y2 = (y_lo*x_hi - y_lo*x)/(x_hi - x_lo)
                x_val = np.linspace(x_range[0], x_range[1], num_points)
                delta_y = y_hi - y_lo
                delta_x = x_hi - x_lo
                y1 = (y_hi * x_lo - delta_y * x_val) / x_lo
                y2 = (y_lo * x_hi - y_lo * x_val) / delta_x
                y = np.max((y1, y2), axis=0)
                return x_val, y
            x_val, y_val = balance_linear_func(
            uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi, (0, uq_x_hi), 100)
            # filter the x_val and y_val in the range of uq_x_lo and uq_x_hi
            x_val = x_val[y_val < uq_y_hi]
            y_val = y_val[y_val < uq_y_hi]
            plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
        elif uq_select_scheme == "loose":
            plt.plot([uq_x_lo, uq_x_lo], [0, uq_y_lo], color='purple', linestyle='--', linewidth=2)
            plt.plot([0, uq_x_lo], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
        else:
            raise ValueError(f"UQ selection scheme {uq_select_scheme} not supported!")
            
        plt.savefig(f"./{view_savedir}/UQ-force-qbc-rnd-fdiff-scatter.png", dpi=fig_dpi)
        plt.close()

# deal with specific selection
logger.info("Dealing with Selection in Target dpdata")
datanames_ind_list = [f"{i[0]}-{i[1]}" for i in dp_test_results_0.dataname_list]
data_dict_uq = {"dataname": datanames_ind_list, 
             "uq_qbc_for": uq_qbc_for, 
             "uq_rnd_for_rescaled": uq_rnd_for_rescaled,
             "uq_rnd_for": uq_rnd_for,
             }
if has_ground_truth:
    data_dict_uq["diff_maxf_0_frame"] = diff_maxf_0_frame
df_uq = pd.DataFrame(data_dict_uq)

# load target testing data
logger.info(f"Loading the target testing data from {testdata_dir}")
# Custom loading to handle unlabelled data (missing energy/force)
# Use a list instead of MultiSystems to prevent automatic merging of systems with same formula,
# ensuring that we preserve the exact directory-to-system mapping and short_names.
test_data = [] 
testdata_string = f'{testdata_dir}/*'
found_dirs = sorted(glob.glob(testdata_string))
logger.info(f"Searching for test data in {testdata_string}, found {len(found_dirs)} directories")

for d in found_dirs:
    if not os.path.isdir(d):
        continue
    try:
        # Try loading as LabeledSystem first
        sys = dpdata.LabeledSystem(d, fmt=testdata_fmt)
    except Exception as e:
        # If it fails (e.g. missing energies), try loading as System (unlabelled)
        # logger.warning(f"Failed to load {d} as LabeledSystem: {e}. Trying as unlabelled System.")
        try:
            sys = dpdata.System(d, fmt=testdata_fmt)
        except Exception as e2:
            logger.error(f"Failed to load {d} as System: {e2}")
            continue
    
    # Verify short_name matches directory basename to ensure correspondence
    expected_name = os.path.basename(d)
    if sys.short_name != expected_name:
        logger.warning(f"System short_name '{sys.short_name}' does not match directory name '{expected_name}'. Correspondence might be broken!")
    
    test_data.append(sys)

# load descriptors/*.npy data
logger.info(f"Loading the target descriptors from {desc_dir}")
desc_string_test = f'{desc_dir}/*.npy'
desc_datanames = []
desc_stru = []
desc_iter_list = sorted(glob.glob(desc_string_test))
for f in desc_iter_list:
    # extract dataname of desc.npy from descriptors/*.npy
    dataname, _ = os.path.splitext(os.path.basename(f))
    one_desc = np.load(f) # nframe, natoms, ndesc
    
    # L2 Normalization of descriptors
    # one_desc shape: (nframe, natoms, ndesc)
    # We normalize along the last axis (ndesc) for each atom
    # Avoid division by zero by adding a small epsilon
    one_desc_modulo = np.linalg.norm(one_desc, axis=2, keepdims=True)
    one_desc_norm = one_desc / (one_desc_modulo + 1e-12)
    
    for i in range(len(one_desc)):
        desc_dataname = f"{dataname}-{i}"
        desc_datanames.append(desc_dataname)
    
    # Vectorized computation for structure descriptors
    # 1. Mean pooling of normalized atomic descriptors -> structure descriptor
    # one_desc_norm shape: (n_frames, n_atoms, n_desc)
    # axis=1 is the atom dimension
    # one_desc_stru shape: (n_frames, n_desc)
    one_desc_stru = np.mean(one_desc_norm, axis=1)

    # 2. L2 Normalization of structure descriptors (Structure Level)
    # Norm along axis 1 (descriptor dimension)
    # stru_modulo shape: (n_frames, 1)
    stru_modulo = np.linalg.norm(one_desc_stru, axis=1, keepdims=True)
    # Avoid division by zero
    one_desc_stru_final = one_desc_stru / (stru_modulo + 1e-12)
    
    desc_stru.append(one_desc_stru_final)
desc_stru = np.concatenate(desc_stru, axis=0)

logger.info(f"Collecting data to dataframe and do UQ selection")
df_desc = pd.DataFrame(desc_stru, 
                       columns=[f"desc_stru_{i}" for i in range(desc_stru.shape[1])])
df_desc["dataname"] = desc_datanames
df_uq_desc = pd.merge(df_uq, df_desc, on="dataname")
# save the dataframe
logger.info(f"Save df_uq_desc dataframe to {df_savedir}/df_uq_desc.csv")
df_uq_desc.to_csv(f"{df_savedir}/df_uq_desc.csv", index=True)

# simplify
uq_x = df_uq["uq_qbc_for"]
uq_y = df_uq["uq_rnd_for_rescaled"]

# uq selection
if uq_select_scheme == "strict":
    # strict selection: QbC and RND-like are both trustable
    df_uq_desc_candidate = df_uq_desc[
        (uq_x >= uq_x_lo)
        & (uq_x <= uq_x_hi) 
        & (uq_y >= uq_y_lo) 
        & (uq_y <= uq_y_hi)
    ]
    df_uq_accurate = df_uq[
        ((uq_x < uq_x_lo) &
        (uq_y < uq_y_hi)) |
        ((uq_x< uq_x_hi) &
        (uq_y < uq_y_lo))
    ]
    df_uq_failed = df_uq[
        (uq_x > uq_x_hi) |
        (uq_y > uq_y_hi) 
    ]
elif uq_select_scheme == "circle_lo":
    # balance selection: QbC and RND-like are trustable in a circle balance way
    df_uq_desc_candidate = df_uq_desc[
        ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
        ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 <= (uq_x_hi - uq_x_lo)**2 + (uq_y_hi - uq_y_lo)**2)
    ]
    df_uq_accurate = df_uq[
        ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 > (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2) & ((uq_x < uq_x_hi) & (uq_y < uq_y_hi))
        ]
    df_uq_failed = df_uq[
        (uq_x > uq_x_hi) |
        (uq_y > uq_y_hi) 
    ]
elif uq_select_scheme == "tangent_lo":
    # balance selection: QbC and RND-like are trustable in a tangent-circle balance way
    df_uq_desc_candidate = df_uq_desc[
        ((uq_x <= uq_x_hi) & (uq_y <= uq_y_hi)) &
        ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) <= 0)
    ]
    df_uq_accurate = df_uq[
        ((uq_x-uq_x_lo)*(uq_x_lo-uq_x_hi) + (uq_y-uq_y_lo)*(uq_y_lo-uq_y_hi) > 0) 
        & ((uq_x < uq_x_hi) & (uq_y < uq_y_hi)) 
        ]
    df_uq_failed = df_uq[
        (uq_x > uq_x_hi) |
        (uq_y > uq_y_hi)
    ]
    
elif uq_select_scheme == "crossline_lo":
    # balance selection: QbC and RND-like are trustable in a croseline balance way
    df_uq_desc_candidate = df_uq_desc[
        (uq_x <= uq_x_hi) & 
        (uq_y <= uq_y_hi) &
        (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x >= uq_x_lo * uq_y_hi) &
        (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y >= uq_x_hi * uq_y_lo)
    ]
    df_uq_accurate = df_uq[
        (uq_x_lo * uq_y + (uq_y_hi - uq_y_lo) * uq_x < uq_x_lo * uq_y_hi) |
        (uq_x * uq_y_lo + (uq_x_hi - uq_x_lo) * uq_y < uq_x_hi * uq_y_lo)
    ]
    df_uq_failed = df_uq[
        (uq_x > uq_x_hi) |
        (uq_y > uq_y_hi) 
    ]
elif uq_select_scheme == "loose":
    # loose selection: QbC or RND-like is either trustable
    df_uq_desc_candidate = df_uq_desc[
        ((uq_x >= uq_x_lo)
        & (uq_x <= uq_x_hi)) 
        | ((uq_y >= uq_y_lo) 
        & (uq_y <= uq_y_hi))
    ]
    df_uq_accurate = df_uq[
        (uq_x < uq_x_lo) &
        (uq_y < uq_y_lo)
    ]
    df_uq_failed = df_uq[
        (uq_x > uq_x_hi) |
        (uq_y > uq_y_hi) 
    ]
else:
    raise ValueError(f"UQ selection scheme {uq_select_scheme} not supported!")

# UQ selection information
logger.info(f"UQ scheme: {uq_select_scheme} between QbC and RND-like")
logger.info(f"UQ selection information : {uq_select_scheme}")
logger.info(f"Total number of structures: {len(df_uq_desc)}")
logger.info(f"Accurate structures: {len(df_uq_accurate)}, Precentage: {len(df_uq_accurate) / len(df_uq_desc) * 100:.2f}%")
logger.info(f"Candidate structures: {len(df_uq_desc_candidate)}, Precentage: {len(df_uq_desc_candidate) / len(df_uq_desc) * 100:.2f}%")
logger.info(f"Failed structures: {len(df_uq_failed)}, Precentage: {len(df_uq_failed) / len(df_uq_desc) * 100:.2f}%")
# store the selection information in df_uq dataframe
df_uq['uq_identity'] = np.where(df_uq['dataname'].isin(df_uq_desc_candidate['dataname']), 'candidate',
                                np.where(df_uq['dataname'].isin(df_uq_accurate['dataname']), 'accurate', 'failed'))
# save the dataframe
logger.info(f"Save df_uq dataframe to {df_savedir}/df_uq.csv after UQ selection and identication")
df_uq.to_csv(f"{df_savedir}/df_uq.csv", index=True)
# save the dataframe
logger.info(f"Save df_uq_desc_candidate dataframe to {df_savedir}/df_uq_desc_sampled-UQ.csv")
df_uq_desc_candidate.to_csv(f"{df_savedir}/df_uq_desc_sampled-UQ.csv", index=True)

# Visualization of UQ selection in scatter
# Two-dim scatter plot
logger.info("Plotting and saving the figure of UQ-identity in QbC-RND 2D space")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_uq, 
                x="uq_qbc_for", 
                y="uq_rnd_for_rescaled", 
                hue="uq_identity", 
                palette={f"candidate": "orange", 
                         f"accurate": "green", 
                         f"failed": "red"},
                alpha=0.5,
                s=60)
plt.title("UQ QbC+RND Selection View", fontsize=14)
plt.grid(True)
plt.xlabel("UQ-QbC Value", fontsize=12)
plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
plt.legend(title="Identity", fontsize=10)

# set the xticks and yticks
ax = plt.gca()
# Adaptive locator
if df_uq["uq_qbc_for"].max() < 5 and df_uq["uq_rnd_for_rescaled"].max() < 5:
    x_major_locator = mtick.MultipleLocator(0.1)
    y_major_locator = mtick.MultipleLocator(0.1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
# plot the UQ region of trust inside the plot
plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)
if uq_select_scheme == "strict":
    plt.plot([uq_x_lo, uq_x_lo], [uq_y_lo, uq_y_hi], color='purple', linestyle='--', linewidth=2)
    plt.plot([uq_x_lo, uq_x_hi], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
elif uq_select_scheme == "circle_lo":
    # ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 >= (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
    center = (uq_x_hi, uq_y_hi)
    radius = np.sqrt((uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
    theta = np.linspace(np.pi, 1.5*np.pi, 100)
    x_val = center[0] + radius * np.cos(theta)
    y_val = center[1] + radius * np.sin(theta)
    plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
elif uq_select_scheme == "tangent_lo":
    # ((uq_x-uq_x_hi)*(uq_x_hi-uq_x_lo) + (uq_y-uq_y_hi)*(uq_y_hi-uq_y_lo) >= 0)
    x_val = np.linspace(0, uq_x_hi, 100)
    y_val = - (uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
    x_val = x_val[y_val < uq_y_hi]
    y_val = y_val[y_val < uq_y_hi]
    plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
elif uq_select_scheme == "crossline_lo":
    def balance_linear_func(x_lo, x_hi, y_lo, y_hi, x_range=(0,10), num_points=20):
        # y1 = (y_hi*x_lo - (y_hi - y_lo)*x)/x_lo
        # y2 = (y_lo*x_hi - y_lo*x)/(x_hi - x_lo)
        x_val = np.linspace(x_range[0], x_range[1], num_points)
        delta_y = y_hi - y_lo
        delta_x = x_hi - x_lo
        y1 = (y_hi * x_lo - delta_y * x_val) / x_lo
        y2 = (y_lo * x_hi - y_lo * x_val) / delta_x
        y = np.max((y1, y2), axis=0)
        return x_val, y
    x_val, y_val = balance_linear_func(
    uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi, (0, uq_x_hi), 100)
    # filter the x_val and y_val in the range of uq_x_lo and uq_x_hi
    x_val = x_val[y_val < uq_y_hi]
    y_val = y_val[y_val < uq_y_hi]
    plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
elif uq_select_scheme == "loose":
    plt.plot([uq_x_lo, uq_x_lo], [0, uq_y_lo], color='purple', linestyle='--', linewidth=2)
    plt.plot([0, uq_x_lo], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
else:
    raise ValueError(f"UQ selection scheme {uq_select_scheme} not supported!")
    
plt.savefig(f"./{view_savedir}/UQ-force-qbc-rnd-identity-scatter.png", dpi=fig_dpi)
plt.close()

# Plot truncated version of UQ-identity scatter
logger.info("Plotting and saving the figure of UQ-identity in QbC-RND 2D space (Truncated [0, 2])")
df_uq_viz = df_uq[(df_uq["uq_qbc_for"] >= 0) & (df_uq["uq_qbc_for"] <= 2) & 
                  (df_uq["uq_rnd_for_rescaled"] >= 0) & (df_uq["uq_rnd_for_rescaled"] <= 2)]
truncated_count_scatter = len(df_uq) - len(df_uq_viz)
if truncated_count_scatter > 0:
    logger.warning(f"UQ-identity-scatter: Truncating {truncated_count_scatter} structures outside [0, 2] for visualization.")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_uq_viz, 
                x="uq_qbc_for", 
                y="uq_rnd_for_rescaled", 
                hue="uq_identity", 
                palette={f"candidate": "orange", 
                         f"accurate": "green", 
                         f"failed": "red"},
                alpha=0.5,
                s=60)
plt.title("UQ QbC+RND Selection View (Truncated [0, 2])", fontsize=14)
plt.grid(True)
plt.xlabel("UQ-QbC Value", fontsize=12)
plt.ylabel("UQ-RND-rescaled Value", fontsize=12)
plt.legend(title="Identity", fontsize=10)

# set the xticks and yticks
ax = plt.gca()
# Adaptive locator: since we truncated to [0, 2], we can safely use 0.1
x_major_locator = mtick.MultipleLocator(0.1)
y_major_locator = mtick.MultipleLocator(0.1)
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)

# plot the UQ region of trust inside the plot
plt.plot([0, uq_x_hi], [uq_y_hi, uq_y_hi], color='black', linestyle='--', linewidth=2)
plt.plot([uq_x_hi, uq_x_hi], [0, uq_y_hi], color='black', linestyle='--', linewidth=2)
if uq_select_scheme == "strict":
    plt.plot([uq_x_lo, uq_x_lo], [uq_y_lo, uq_y_hi], color='purple', linestyle='--', linewidth=2)
    plt.plot([uq_x_lo, uq_x_hi], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
elif uq_select_scheme == "circle_lo":
    # ((uq_x-uq_x_hi)** 2 + (uq_y-uq_y_hi)**2 >= (uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
    center = (uq_x_hi, uq_y_hi)
    radius = np.sqrt((uq_x_lo - uq_x_hi)**2 + (uq_y_lo - uq_y_hi)**2)
    theta = np.linspace(np.pi, 1.5*np.pi, 100)
    x_val = center[0] + radius * np.cos(theta)
    y_val = center[1] + radius * np.sin(theta)
    plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
elif uq_select_scheme == "tangent_lo":
    # ((uq_x-uq_x_hi)*(uq_x_hi-uq_x_lo) + (uq_y-uq_y_hi)*(uq_y_hi-uq_y_lo) >= 0)
    x_val = np.linspace(0, uq_x_hi, 100)
    y_val = - (uq_y_hi - uq_y_lo) / (uq_x_hi - uq_x_lo) * (x_val - uq_x_lo) + uq_y_lo
    x_val = x_val[y_val < uq_y_hi]
    y_val = y_val[y_val < uq_y_hi]
    plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
elif uq_select_scheme == "crossline_lo":
    def balance_linear_func(x_lo, x_hi, y_lo, y_hi, x_range=(0,10), num_points=20):
        # y1 = (y_hi*x_lo - (y_hi - y_lo)*x)/x_lo
        # y2 = (y_lo*x_hi - y_lo*x)/(x_hi - x_lo)
        x_val = np.linspace(x_range[0], x_range[1], num_points)
        delta_y = y_hi - y_lo
        delta_x = x_hi - x_lo
        y1 = (y_hi * x_lo - delta_y * x_val) / x_lo
        y2 = (y_lo * x_hi - y_lo * x_val) / delta_x
        y = np.max((y1, y2), axis=0)
        return x_val, y
    x_val, y_val = balance_linear_func(
    uq_x_lo, uq_x_hi, uq_y_lo, uq_y_hi, (0, uq_x_hi), 100)
    # filter the x_val and y_val in the range of uq_x_lo and uq_x_hi
    x_val = x_val[y_val < uq_y_hi]
    y_val = y_val[y_val < uq_y_hi]
    plt.plot(x_val, y_val, color="purple", linestyle="--", linewidth=2)
elif uq_select_scheme == "loose":
    plt.plot([uq_x_lo, uq_x_lo], [0, uq_y_lo], color='purple', linestyle='--', linewidth=2)
    plt.plot([0, uq_x_lo], [uq_y_lo, uq_y_lo], color='purple', linestyle='--', linewidth=2)
else:
    raise ValueError(f"UQ selection scheme {uq_select_scheme} not supported!")
    
plt.savefig(f"./{view_savedir}/UQ-force-qbc-rnd-identity-scatter-truncated.png", dpi=fig_dpi)
plt.close()
# one dim scatter plot for each UQ
# for QbC
# passed
if has_ground_truth:
    logger.info("Plotting and saving the figure of UQ-Candidate in QbC space against Max Force Diff")
    plt.figure(figsize=(8, 6))
    plt.scatter(uq_qbc_for, diff_maxf_0_frame, color="blue", label="UQ-QbC", s=20)
    plt.scatter(df_uq_desc_candidate["uq_qbc_for"], df_uq_desc_candidate["diff_maxf_0_frame"], color="orange", label="Candidate", s=20)
    plt.title("UQ vs Force Diff")
    plt.xlabel("UQ Value")
    plt.ylabel("True Max Force Diff")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{view_savedir}/UQ-QbC-Candidate-fdiff-parity.png", dpi=fig_dpi)
    plt.close()

    # for RND
    # passed
    logger.info("Plotting and saving the figure of UQ-Candidate in RND-rescaled space against Max Force Diff")
    plt.figure(figsize=(8, 6))
    plt.scatter(uq_rnd_for_rescaled, diff_maxf_0_frame, color="blue", label="UQ-RND-rescaled", s=20)
    plt.scatter(df_uq_desc_candidate["uq_rnd_for_rescaled"], df_uq_desc_candidate["diff_maxf_0_frame"], color="orange", label="Candidate", s=20)
    plt.title("UQ vs Force Diff")
    plt.xlabel("UQ Value")
    plt.ylabel("True Max Force Diff")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"./{view_savedir}/UQ-RND-Candidate-fdiff-parity.png", dpi=fig_dpi)
    plt.close()


# DIRECT selection on UQ data
logger.info(f"Doing DIRECT Selection on UQ-selected data")

# Check if there are any candidates after UQ selection
if len(df_uq_desc_candidate) == 0:
    logger.warning("No structures selected by UQ scheme! Skipping DIRECT selection and further processing.")
    # Exit or handle gracefully? 
    # Since subsequent steps depend on DIRECT selection results, we should stop here.
    # We can save an empty results file if needed, or just exit.
    logger.info("All Done (No candidates found).")
    import sys
    sys.exit(0)

DIRECT_sampler = DIRECTSampler(
    structure_encoder=None,
    clustering=BirchClustering(
        n=num_selection // direct_k, 
    threshold_init=direct_thr_init),
    select_k_from_clusters=SelectKFromClusters(k=direct_k),
)
desc_features = [f"desc_stru_{i}" for i in range(desc_stru.shape[1])]
DIRECT_selection = DIRECT_sampler.fit_transform(df_uq_desc_candidate[desc_features].values)
DIRECT_selected_indices = DIRECT_selection["selected_indices"]
explained_variance = DIRECT_sampler.pca.pca.explained_variance_
selected_PC_dim = len([e for e in explained_variance if e > 1])
DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:selected_PC_dim]
all_features = DIRECT_selection["PCAfeatures_unweighted"]

df_uq_desc_selected_final = df_uq_desc_candidate.iloc[DIRECT_selected_indices]
# save the dataframe
logger.info(f"Saving df_uq_desc_selected_final dataframe to {df_savedir}/df_uq_desc_sampled-final.csv")
df_uq_desc_selected_final.to_csv(f"{df_savedir}/df_uq_desc_sampled-final.csv", index=True)

# Visualization of DIRECT results
# Explained Variance
logger.info(f"Visualization of DIRECT results compared with Random")
plt.figure(figsize=(8, 6))
plt.plot(
    range(1, selected_PC_dim+6+1),
    explained_variance[:selected_PC_dim+6],
    "o-",
)
plt.xlabel(r"i$^{\mathrm{th}}$ PC", size=12)
plt.ylabel("Explained variance", size=12)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.savefig(f"{view_savedir}/explained_variance.png", dpi=150)
plt.close()
# PCA feature converage
def plot_PCAfeature_coverage(all_features, selected_indices, method="DIRECT", dpi=150, savedir='.'):
    plt.plot(figsize=(8, 6))
    selected_features = all_features[selected_indices]
    plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.6, label=f"All {len(all_features):,} structures")
    plt.plot(
        selected_features[:, 0],
        selected_features[:, 1],
        "*",
        alpha=0.6,
        label=f"{method} sampled {len(selected_features):,}",
    )
    plt.legend(frameon=False, fontsize=10, reverse=True)
    plt.ylabel("PC 2", size=12)
    plt.xlabel("PC 1", size=12)
    plt.savefig(f"{savedir}/{method}_PCA_feature_coverage.png", dpi=dpi)
    plt.close()
# DIRECT visualization
plot_PCAfeature_coverage(all_features,
                         DIRECT_selected_indices, 
                         dpi=fig_dpi, 
                         savedir=view_savedir)
# Simulate a manual selection by random
np.random.seed(42)
manual_selection_index = np.random.choice(len(all_features), 
                                          num_selection, 
                                          replace=False)
plot_PCAfeature_coverage(all_features,
                         manual_selection_index, 
                         "Random", 
                         dpi=fig_dpi, 
                         savedir=view_savedir)

# feature converage score comparsion
def calculate_feature_coverage_score(all_features, selected_indices, n_bins=100):
    selected_features = all_features[selected_indices]
    n_all = np.count_nonzero(
        np.histogram(all_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
    )
    n_select = np.count_nonzero(
        np.histogram(selected_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
    )
    return n_select / n_all
def calculate_all_FCS(all_features, selected_indices, b_bins=100):
    select_scores = [
        calculate_feature_coverage_score(all_features[:, i], selected_indices, n_bins=b_bins)
        for i in range(all_features.shape[1])
    ]
    return select_scores

all_features = DIRECT_selection["PCAfeatures_unweighted"]
scores_DIRECT = calculate_all_FCS(all_features, DIRECT_selection["selected_indices"], b_bins=100)
scores_MS = calculate_all_FCS(all_features, manual_selection_index, b_bins=100)
# plot the feature converage score comparsion
x = np.arange(len(scores_DIRECT))
x_ticks = [f"PC {n+1}" for n in range(len(x))]
plt.figure(figsize=(15, 4))
plt.bar(
    x + 0.6,
    scores_DIRECT,
    width=0.3,
    label=rf"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_DIRECT):.3f}",
)
plt.bar(
    x + 0.3,
    scores_MS,
    width=0.3,
    label=rf"Random, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_MS):.3f}",
)
plt.xticks(x + 0.45, x_ticks, size=12)
plt.yticks(np.linspace(0, 1.0, 6), size=12)
plt.ylabel("Coverage score", size=12)
plt.legend(shadow=True, loc="lower right", fontsize=12)
plt.savefig(f"{view_savedir}/coverage_score.png", dpi=fig_dpi)

# visualization of final selection results in PCA space
logger.info(f"Visualization of final selection results in PCA space")
X = df_uq_desc[desc_features].values
pca = PCA(n_components=2)
PCs_alldata = pca.fit_transform(X)
PCs_df = pd.DataFrame(data = PCs_alldata, columns = ['PC1', 'PC2'])
df_alldataPC_visual = pd.concat([df_uq, PCs_df], axis = 1)
# save the dataframe
df_alldataPC_visual.to_csv(f"{df_savedir}/final_df.csv", index=True)

# global indices of selected structures in original dataframe
UQ_selected_indices_global = df_uq_desc_candidate.index
final_selected_indices_global = df_uq_desc_selected_final.index
num_all = len(df_uq.index)
num_selected_UQ = len(UQ_selected_indices_global)
num_selected_final = len(final_selected_indices_global)
# first, plot all structures in the feature space
plt.figure(figsize=(10, 8))
plt.scatter(PCs_alldata[:, 0], 
            PCs_alldata[:, 1], 
            marker="*", 
            color="gray", 
            label=f"All {num_all} structures", 
            alpha=0.7,
            s=15)
# second, plot the selected structures only by UQ
plt.scatter(PCs_alldata[UQ_selected_indices_global, 0], 
            PCs_alldata[UQ_selected_indices_global, 1], 
            marker="*", 
            color="blue", 
            label=f"UQ sampled {num_selected_UQ}", 
            alpha=0.7,
            s=30)
# third, plot the selected structures by UQ-DIRECT
plt.scatter(PCs_alldata[final_selected_indices_global, 0], 
            PCs_alldata[final_selected_indices_global, 1], 
            marker="*", 
            color="red", 
            label=f"UQ-DIRECT sampled {num_selected_final}", 
            s=30)
plt.title(f"PCA of UQ-DIRECT sampling", fontsize=14)
plt.xlabel("PC1", size=12)
plt.ylabel("PC2", size=12)
plt.legend(frameon=False, fontsize=12, reverse=True)
logger.info(f"Saving the PCA view of UQ-DIRECT sampling to {view_savedir}/Final_sampled_PCAview.png")
plt.savefig(f"{view_savedir}/Final_sampled_PCAview.png", dpi=fig_dpi)

# selection of dpdata
logger.info(f"Sampling dpdata based on selected indices")
sampled_datanames = df_uq_desc.iloc[final_selected_indices_global]['dataname'].to_list()
sampled_dpdata = dpdata.MultiSystems()
other_dpdata = dpdata.MultiSystems()
for lbsys in test_data:
    for ind, sys in enumerate(lbsys):
        dataname_sys = f"{sys.short_name}-{ind}"
        if dataname_sys in sampled_datanames:
            sampled_dpdata.append(sys)
        else:
            other_dpdata.append(sys)
logger.info(f'Sampled dpdata: {sampled_dpdata}')
logger.info(f'Other dpdata: {other_dpdata}')
logger.info(f"Dumping sampled and other dpdata to {dpdata_savedir}")
# remove existing dpdata dir
if os.path.exists(f"{dpdata_savedir}/sampled_dpdata"):
    shutil.rmtree(f"{dpdata_savedir}/sampled_dpdata")
if os.path.exists(f"{dpdata_savedir}/other_dpdata"):
    shutil.rmtree(f"{dpdata_savedir}/other_dpdata")
sampled_dpdata.to_deepmd_npy(f"{dpdata_savedir}/sampled_dpdata")
other_dpdata.to_deepmd_npy(f"{dpdata_savedir}/other_dpdata")

logger.info("All Done!")