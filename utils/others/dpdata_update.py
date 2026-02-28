import dpdata
from copy import deepcopy

# load data
data_train_string = "target-data-0"
data_valid_string = "pool-data-0"
data_select_string = "select-data-0"
train_update_string = "target-data-1"
valid_update_string = "pool-data-1"
dpdata_string = "O*"

dpdata_train = dpdata.MultiSystems.from_dir(data_train_string, dpdata_string, fmt="deepmd/npy")
dpdata_valid = dpdata.MultiSystems.from_dir(data_valid_string, dpdata_string, fmt="deepmd/npy")
dpdata_select = dpdata.MultiSystems.from_dir(data_select_string, dpdata_string, fmt="deepmd/npy")

# info
print("Target Data:", dpdata_train)
print("Pool Data:", dpdata_valid)

dpdata_train_update = deepcopy(dpdata_train)
dpdata_valid_update = dpdata.MultiSystems()

# add the selected data to the training data
# and remove the selected data from the validation data
select_data_list = []
for lbsys in dpdata_select:
    dpdata_train_update.append(lbsys)
    for sys in lbsys:
        select_data_list.append(sys.data["energies"])
for lbsys in dpdata_valid:
    for sys in lbsys:
        if sys.data['energies'] not in select_data_list:
            dpdata_valid_update.append(sys)
# info
print("Selected Data:", dpdata_select)
print("Updated Target Data:", dpdata_train_update)
print("Updated Pool Data:", dpdata_valid_update)

# save
dpdata_train_update.to_deepmd_npy(train_update_string)
dpdata_valid_update.to_deepmd_npy(valid_update_string)
dpdata_train_update.to_deepmd_npy_mixed(f"{train_update_string}-mixed")
dpdata_valid_update.to_deepmd_npy_mixed(f"{valid_update_string}-mixed")
