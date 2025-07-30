# read stru from dpdata/npy and transfer it to dir/abacus-input(INPUT, STRU, KPT)
# update by JamesMisaka in 2025-0609
# initialize the abacus calc

import numpy as np
import os
import sys
from ase.io.abacus import write_input, write_abacus
from ase import Atoms
from typing import List
from copy import deepcopy
from random import randint
import dpdata
import glob
from tqdm import tqdm
import numpy as np

treat_stru = True
project_dir = "input_dpdata"
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


# use the newest APNS-2025 PP-ORB_DZP
pp = {
      'H': 'H.upf',
      'C': 'C.upf',
      'O': 'O.upf',
      'Fe': 'Fe_ONCV_PBE-1.2.upf',
      }
basis = {
         'H': 'H_gga_6au_100Ry_2s1p.orb',
         'C': 'C_gga_8au_100Ry_2s2p1d.orb',
         'O': 'O_gga_6au_100Ry_2s2p1d.orb',
         'Fe': 'Fe_gga_7au_100Ry_4s2p2d1f.orb',
         }

# Judgement
vaccum = 6.3 # Ang, the lowest vaccum thickness
cubic_symprec_decimal = 0 # decimal for symprec in cubic_cluster
# in the first example, decimal = 0 is the same of decimal = 1.
cubic_min_length = 9.8 # Ang, the minimum length of cubic_cluster cell
cubic_tol = 6.3 # if lattice length > cubic_min_length + cubic_tol, then the stru should be cubic_cluster even without vaccum

lib_dir = "/mnt/sg001/home/fz_pku_jh/PP_ORB/abacus"
pseudo_dir = f"{lib_dir}/PP-2025"
basis_dir = f"{lib_dir}/ORB-2025"
kpts = [1,1,1]
kpt_criteria = 25
merge_traj = True
# kspacing_basic = [0.14, 0.14, 0.14]
# k*a ~ 25 Ang <=> kspacing = 0.14 (Ry in ABACUS)
# but here we use the generated kpoints in code
basic_input = {
    'calculation': 'scf',
    'nspin': 2,
    'xc': 'pbe',
    'ecutwfc': 100,
    'ks_solver': 'genelpa',
    'symmetry': 0,
    'vdw_method': 'none',
    'smearing_method': 'gaussian',
    'smearing_sigma': 0.004,
    'basis_type': 'lcao',
    'mixing_type': 'broyden',
    'mixing_beta': 0.4,
    'mixing_gg0': 1.0,
    'mixing_ndim': 20,
    'scf_thr': 1e-7,
    'scf_nmax': 300,
    'kpts': kpts,
    'pp': pp,
    'basis': basis,
    'pseudo_dir': pseudo_dir,
    'basis_dir': basis_dir,
    'cal_force': 1,
    'cal_stress': 1,
    'init_chg': 'atomic',
    'out_stru': 1,
    'out_chg': 0,
    'out_bandgap': 1,
    'onsite_radius': 3.0,
}

ROOTDIR = os.getcwd()

def judge_vaccum(atoms, vac=6):
    '''judge vaccum exist or not in 3 dim by tracking the vaccum slab

    :params: atoms : Atoms object
    :params vac: minimum vaccum thickness
    :returns: [bool,bool,bool] along x,y,z for have vaccum or not
    '''
    atoms_foruse = atoms.copy()
    atoms_foruse.center()
    vaccum_status = []
    for dim in range(3):
        dim_pos = atoms_foruse.positions[:,dim]
        dim_pos_gap = max(dim_pos) - min(dim_pos)
        # dim_lat = atoms_foruse.cell.cellpar()[dim]
        # should consider Cartesian coordinate of nearest
        dim_lat = atoms_foruse.cell[dim,dim]
        if dim_pos_gap > dim_lat - vac:
            vaccum_status.append(False)
        else:
            vaccum_status.append(True)
    return vaccum_status


def set_magmom_for_Atoms(atoms, mag_ele=[], mag_num=[]):
    """Set Atoms Object magmom by element
    
    Args:
        atoms: (atoms) Atoms object
        mag_ele (list): element list
        mag_num (list): magmom list
    """
    # init magmom can only be set to intermediate images
    init_magmom = atoms.get_initial_magnetic_moments()
    if len(mag_ele) != len(mag_num):
        raise SyntaxWarning("mag_ele and mag_num have different length")
    for mag_pair in zip(mag_ele, mag_num):
        ele_ind = [atom.index for atom in atoms if atom.symbol == mag_pair[0]]
        init_magmom[ele_ind] = mag_pair[1]
    atoms.set_initial_magnetic_moments(init_magmom)
    # print(f"---- Set initial magmom for {mag_ele} to {mag_num} ----")

def set_kpoints(atoms, criteria=25, vaccum_status=[False,False,False], cluster=False):
    '''set KPOINTS for various basic shape
    
    Args:
        atoms: Atoms object
        criteria: minimum for kpoints * lattice
        vac: minimum vaccum thickness
        cluster: if True, set kpoints to [1,1,1]
    '''
    kpoints = [1,1,1]
    if cluster:
        return kpoints
    for dim in range(3):
        if vaccum_status[dim]:
            kpoints[dim] = 1
        else:
            kpoints[dim] = int(criteria / atoms.cell.cellpar()[dim]) + 1
    return kpoints

def write_abacus_kpts(filename='KPT', kpoints=[1,1,1]):
    '''write KPOINTS for abacus

    Args:
        filename: KPT
        kpoints: [k1,k2,k3]
    '''
    kptstring = f'K_POINTS\n0\nGamma\n'
    with open(filename, 'w') as fo:
        kptstring += f'{kpoints[0]} {kpoints[1]} {kpoints[2]} 0 0 0\n'
        fo.write(kptstring)

def swap_crystal_lattice(structure: Atoms, swap_indices: List[int] = [1, 2]) -> Atoms:
    """
    Swap the lattice vector of a crystal structure by ASE.
    """
    # swap the cell for lattice
    old_cellpar = structure.get_cell().cellpar()
    new_cellpar = old_cellpar.copy()
    # a,b,c
    new_cellpar[swap_indices[0]] = old_cellpar[swap_indices[1]]
    new_cellpar[swap_indices[1]] = old_cellpar[swap_indices[0]]
    # alpha beta gamma
    new_cellpar[swap_indices[0] + 3] = old_cellpar[swap_indices[1] + 3]
    new_cellpar[swap_indices[1] + 3] = old_cellpar[swap_indices[0] + 3]
    new_structure = structure.copy()
    new_structure.set_cell(new_cellpar)
    # swap the scaled positions
    old_scaled_positions = structure.get_scaled_positions()
    new_positions = old_scaled_positions.copy()
    new_positions[:, swap_indices[0]] = old_scaled_positions[:, swap_indices[1]]
    new_positions[:, swap_indices[1]] = old_scaled_positions[:, swap_indices[0]]
    # set the new scaled positions after the setting of cell
    new_structure.set_scaled_positions(new_positions)
    return new_structure


def sampled_dpdata_to_abacus(dataset_name, project_dir, vaccum=6.18, kpt_criteria=25, merge_traj=True):
    '''transfer sampled stru from dpdata/npy to abacus-input
    Args:
        dataset_name: name of dataset
        project_dir: dir to store abacus-input
        vac: minimum vaccum thickness
        kpt_criteria: minimum for kpoints * lattice
        merge_traj: if True, merge traj to one file
    '''
    # read all stru in dpdata/npy format as System in MultiSystems
    target_dpdata_dir = f"./{dataset_name}/sampled_dpdata/"
    target_dpdata = dpdata.MultiSystems()
    for item in sorted(glob.glob(f'{target_dpdata_dir}/*')):
        target_dpdata.append(dpdata.System(item, fmt='deepmd/npy'))

    for one_systems in tqdm(target_dpdata):
        sysname = one_systems.short_name
        one_stru_list = one_systems.to_ase_structure()
        for ind, stru in enumerate(one_stru_list):
            # preprocessing
            stru.wrap()
            stru.center()
            # the order of center and wrap may fail in some stru
            # so we need to do some additional preprocessing
            # in the center of cell, there must be atoms existing
            scal_coords_now = stru.get_scaled_positions()
            scal_coords_trans_key = False
            for dim in range(3):
                if np.sum((scal_coords_now[:,dim] > 0.25) & (scal_coords_now[:,dim] < 0.75)) / scal_coords_now.shape[0] < 0.45:
                    # indicating that the stru is not in the center of cell
                    higher_mask = (scal_coords_now[:,dim] > 0.67) # give 0.5-0.75 some tolerance
                    scal_coords_now[higher_mask, dim] -= 1.0
                    scal_coords_trans_key = True
                else:
                    continue
            if scal_coords_trans_key:
                stru.set_scaled_positions(scal_coords_now)
                stru.center()
            
            stru_root_name = f"{project_dir}/{dataset_name}/"
            vaccum_state = judge_vaccum(stru, vac=vaccum)
            input_parameters = deepcopy(basic_input)
            
            # give different systems different input files accordingly
            if sum(vaccum_state) == 3:
                stru_type = "cluster"
            # notice: some special cluster may have vaccum layer in a not enough large scale
            # to notify these, define a special case of cluster where cell is cubic and having at least one vaccum layer
            # standard cubic cell: 
            # [[ a.  0.  0.]]
            # [[ 0.  a.  0.]]
            # [[ 0.  0.  a.]]
            elif (np.min(stru.cell.cellpar()[:3]) >= cubic_min_length) \
            and (np.round(stru.cell[0,0], decimals=cubic_symprec_decimal) \
                == np.round(stru.cell[1,1], decimals=cubic_symprec_decimal) \
                == np.round(stru.cell[2,2], decimals=cubic_symprec_decimal)) \
            and (np.round(np.min(stru.cell), decimals=cubic_symprec_decimal) == 0.0) \
            and (sum(vaccum_state) != 0):
                stru_type = "cubic_cluster"
            elif (np.min(stru.cell.cellpar()[:3]) >= cubic_min_length + cubic_tol) \
            and (np.round(stru.cell[0,0], decimals=cubic_symprec_decimal) \
                == np.round(stru.cell[1,1], decimals=cubic_symprec_decimal) \
                == np.round(stru.cell[2,2], decimals=cubic_symprec_decimal)) \
            and (np.round(np.min(stru.cell), decimals=cubic_symprec_decimal) == 0.0):
                stru_type = "cubic_cluster"
            # then turn back to common case
            elif sum(vaccum_state) == 0:
                stru_type = "bulk"
            elif sum(vaccum_state) == 2:
                stru_type = "string"
            elif sum(vaccum_state) == 1:
                stru_type = "layer"
                # set dipole correction only for layer
                # for most cases not needed but can added
                # but for Fe-O surface the dipole correction is needed
                vaccum_dim = vaccum_state.index(True)
                # added in 0729
                if vaccum_dim == 2:
                    # change Z-axis vacuum to Y-axis vacuum
                    stru = swap_crystal_lattice(stru, [1, 2])
                    vaccum_dim = 1
                    vaccum_state[1] = True
                    vaccum_state[2] = False
                input_parameters.update(
                    {
                        'efield_flag': 1,
                        'dip_cor_flag': 1,
                        'efield_dir': vaccum_dim,
                    }
                )
            else:
                raise SyntaxWarning("vaccum_state is not correct")

            stru_dir_name = f"{stru_root_name}/{stru_type}/{sysname}-{ind}"
            if not os.path.exists(stru_dir_name):
                os.makedirs(stru_dir_name)

            # deal with kpt generated in code
            if (stru_type == "cluster") \
                or (stru_type == "cubic_cluster"):
                cluster_identify = True
            else:
                cluster_identify = False
            kpoints = set_kpoints(stru, criteria=kpt_criteria, vaccum_status=vaccum_state, cluster=cluster_identify)
            input_parameters.update(
                {
                    'kpts': kpoints
                }
            )
            # deal with gamma-only if it is true
            if (sum(vaccum_state) == 3) or (kpoints == [1,1,1]) or (cluster_identify == True):
                input_parameters.update(
                    {
                        'gamma_only': 1
                    }
                )
            if treat_stru:
                if "Fe" in str(stru.symbols):
                    if treat_stru:
                        set_magmom_for_Atoms(stru, mag_ele=["H","C","O", "Fe"], mag_num=[1,1,1,5])
                else:
                    if treat_stru:
                        set_magmom_for_Atoms(stru, mag_ele=["H","C","O"], mag_num=[1,1,1])
                write_input(open(f"{stru_dir_name}/INPUT", 'w'), parameters=input_parameters)
                write_abacus(open(f"{stru_dir_name}/STRU", 'w'), stru, pp=input_parameters['pp'], basis=input_parameters['basis'])
                with open(f"{stru_dir_name}/INPUT", "a") as fw:
                    # fw.write(f"pseudo_dir     {pseudo_dir}\n")
                    fw.write(f"orbital_dir    {basis_dir}\n")
                write_abacus_kpts(f"{stru_dir_name}/KPT", kpoints=kpoints)
                # write backup cif & extxyz for visualization
                stru.write(f"{stru_dir_name}/{sysname}-{ind}.cif")
                stru.write(f"{stru_dir_name}/{sysname}-{ind}.extxyz")
                
    # get a merged extxyz for visualization
    if merge_traj:
        for type_dir in glob.glob(f"{stru_root_name}/*"):
            with open(f"{type_dir}/merged.extxyz", "w") as fw:
                for extxyz_filename in glob.glob(f"{type_dir}/*/*.extxyz"):
                    with open(extxyz_filename, "r", encoding='utf-8') as fr:
                        fw.write(fr.read())

    print(f"---- Finish dealing {len(target_dpdata)} Systems of structure ----")
    print(f"---- Save Structure and ABACUS Inputs in {project_dir}/{dataset_name} ----")
    
if __name__ == "__main__":
    for dataset_name in dataset_names:
        sampled_dpdata_to_abacus(dataset_name, f"{ROOTDIR}/{project_dir}", vaccum=vaccum, kpt_criteria=kpt_criteria, merge_traj=merge_traj)
    print("---- Finish all ----")
