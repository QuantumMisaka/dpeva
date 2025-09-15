import numpy as np
import matplotlib.pyplot as plt
import dpdata
import glob
import seaborn as sns
import pandas as pd
import os
from sklearn.decomposition import PCA


# load data
desc_string = 'descriptors-30656/*/desc.npy'
dpdata_name = "FeCHO-clean-30656"
dpdata_path = "./clean-30656"
dpdata_string = "C*"

# PCA setting
ndim = 20
save_name = f"PCA-desc-stru-{ndim}dim"

# define used function
def extract_elements_array(data: dpdata.LabeledSystem) -> list:
    '''extract elements array from dpdata for draw PCA'''
    types = data.data['atom_types']
    names = data.data['atom_names']
    ele_array = [names[ind] for ind in types]
    return ele_array

# 
if os.path.exists(f'{save_name}.pickle'):
    print(f"{save_name}.pickle already exists, skip PCA.")
    print(f"Data loaded from {save_name}.pickle")
    df_desc = pd.read_pickle(f'{save_name}.pickle')
else:
    # read descriptors/*/desc.npy data
    print("Reading descriptor results...")
    desc_keys = []
    all_desc_stru = []
    for f in glob.glob(desc_string):
        # extract dirname of desc.npy from descriptors/*
        directory, _ = os.path.split(f)
        _, keyname = os.path.split(directory)
        desc_keys.append(keyname)
        one_desc = np.load(f) # nframe, natoms, ndesc
        # do average in natoms dimension
        one_desc_stru = np.mean(one_desc, axis=1)
        all_desc_stru.append(one_desc_stru)
    all_desc_stru = np.concatenate(all_desc_stru, axis=0)

    # read dpdata for element type information 
    print("Reading corresponding dpdata...")
    alldata =  dpdata.MultiSystems.from_dir(dpdata_path, dpdata_string, fmt="deepmd/npy")

    alldata_dict = {}
    for lbsys in alldata:
        alldata_dict[lbsys.short_name] = lbsys
        
    # get list of system name
    sys_list = []
    for keyname in desc_keys:
        target_sys = alldata_dict[keyname]
        for ind in range(target_sys.get_nframes()):
            sys_list.append(f"{keyname}-{ind}")
        
    # get element ratio
    element_ratio_dict = {}
    element_names = alldata[0].get_atom_names()
    for element in element_names:
        ratio_for_ele = []
        for keyname in desc_keys:
            target_sys = alldata_dict[keyname]
            ratio = target_sys.get_atom_numbs()[target_sys.get_atom_names().index(element)] / np.sum(target_sys.get_atom_numbs())
            ratio_for_ele.extend([ratio] * target_sys.get_nframes())
        element_ratio_dict[element] = ratio_for_ele

    # do PCA, most time consuming step
    pdf = pd.DataFrame(all_desc_stru)

    print("Doing PCA...")
    pca = PCA(
        n_components=ndim,
    )

    embedding = pca.fit_transform(pdf)
    embedding_np = embedding[:, :2]
    print("PCA done.")

    # get formation energy of each stru
    # 生成能字典
    elements_ref_ene = {
        "C": -155.07351,
        "Fe": -3220.20451,
        "H": -15.849995,
        "O": -432.63044825,
    }

    # 根据dataname得到生成能
    def get_ref_ene(dataname, elements_ref_ene=elements_ref_ene):
        ref_ene = 0
        ene_list = list(elements_ref_ene.keys())
        for ele in ene_list:
            dataname = dataname.replace(ele, f" {ele},")
        ene_string_dict = dataname.strip().split(" ") # O,2 as example
        natom = 0
        for ind, ele_string in enumerate(ene_string_dict):
            ele_list = ele_string.split(',')
            ele = ele_list[0]
            num = ele_list[1]
            natom += eval(num)
            ref_ene += eval(num) * elements_ref_ene[ele]
        ref_ene /= natom # 归一化到ev-per-atom
        return ref_ene
    
    form_ene_list = []
    for keyname in desc_keys:
        target_sys = alldata_dict[keyname]
        for ene in target_sys.data['energies']:
            form_ene_list.append(ene/np.sum(target_sys.get_atom_numbs()) - get_ref_ene(target_sys.short_name))
    
    # to pandas
    df_desc = pd.DataFrame(embedding_np, columns=['Dim1','Dim2'])
    df_desc['sys_name'] = sys_list
    df_desc['E_Form'] = form_ene_list
    for ele,ratio_for_ele in element_ratio_dict.items():
        df_desc[f'ratio_{ele}'] = ratio_for_ele

    df_desc.to_pickle(f'{save_name}.pickle')
    print(f"Data saved as {save_name}.pickle")


# draw graph
print("Drawing graph...")
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Dim1', y='Dim2', hue='ratio_Fe', data=df_desc, palette='viridis', s=100, alpha=0.7)
plt.title(f'PCA of {dpdata_name} dataset stru by ndim {ndim}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Fe ratio')
plt.grid(True)


plt.savefig(f'{save_name}.png',dpi=200)
print(f"Graph saved as {save_name}.png")

print("All done.")
