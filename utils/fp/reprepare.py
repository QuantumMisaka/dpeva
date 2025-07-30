# update by JamesMisaka in 2025-0609
# reprepare the abacus calc, run in abacus-forloop project dir

import numpy as np
import os
from ase.io.abacus import write_input, write_abacus
from ase.io import read
import glob
import numpy as np
from ase import Atoms
from typing import List
from copy import deepcopy

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

# Judgement, global
vaccum = 6.3 # Ang, the lowest vaccum thickness
cubic_symprec_decimal = 0 # decimal for symprec in cubic_cluster
# in the first example, decimal = 0 is the same of decimal = 1.
cubic_min_length = 9.8 # Ang, the minimum length of cubic_cluster cell
cubic_tol = 6.3 # if lattice length > cubic_min_length + cubic_tol, then the stru should be cubic_cluster even without vaccum

lib_dir = "/mnt/sg001/home/fz_pku_jh/PP_ORB/abacus"
pseudo_dir = f"{lib_dir}/PP-2025"
basis_dir = f"{lib_dir}/ORB-2025"
kpts = [1,1,1] # default, do not change
kpt_criteria = 25 # need to adjust
change_input = True
change_kpt = True  # only useful if change_input=True
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
    'scf_os_stop': 1,
    'scf_os_ndim': 80,
    'kpts': kpts,
    'pp': pp,
    'basis': basis,
    'pseudo_dir': pseudo_dir,
    'basis_dir': basis_dir,
    'cal_force': 1,
    'cal_stress': 1,
    'init_chg': 'atomic',
    'out_stru': 1,
    'out_mul': 1,
    'out_chg': 0,
    'out_bandgap': 1,
    'onsite_radius': 3.0,
}
# optional parameter
# 

ROOTDIR = os.getcwd()

# if INPUT file is changed, everything need to be re-prepared
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


def main():
    """Main function, run after check conv and leave only non-conv here for reprepare"""
    for dir in sorted(glob.glob(f"{ROOTDIR}/*")):
        if not os.path.isdir(dir):
            continue
        if "CONVERGED" in os.path.basename(dir):
            continue
        print(f"Processing {dir}")
        # read in abacus STRU
        stru = read(f"{dir}/STRU", format='abacus')
        # set magmom for atoms
        init_magmom = stru.get_initial_magnetic_moments()
        mag_eles = ["Fe", "O", "H", "C"]
        for ele in mag_eles:
            ind = [atom.index for atom in stru if atom.symbol == ele]
            if ele == 'Fe':
                magmom = np.array([(1)**n * 5 for n in range(0, len(ind))]).reshape(len(ind),1)
                # self-consistent FM is better than some AFM to mag-ground state
            else:
                magmom = np.array([(1)**n * 1 for n in range(0, len(ind))]).reshape(len(ind),1)
            init_magmom[ind] = magmom
        stru.set_initial_magnetic_moments(init_magmom)
        input_parameters = deepcopy(basic_input)
        # write input
        if change_input:
            print("Notice: INPUT file is changed !")
             # give different systems different input files accordingly
            vaccum_state = judge_vaccum(stru, vac=vaccum)
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
            # deal with kpt
            if change_kpt:
                print("Notice: KPT file is also changed in INPUT changing !")
                if (stru_type == "cluster") or (stru_type == "cubic_cluster"):
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
                write_abacus_kpts(f"{dir}/KPT", kpoints=kpoints)
            write_input(open(f"{dir}/INPUT", 'w'), parameters=input_parameters)

        # write stru
        write_abacus(open(f"{dir}/STRU", 'w'), stru, pp=basic_input['pp'], basis=basic_input['basis'])
        with open(f"{dir}/INPUT", "a") as fw:
            fw.write(f"orbital_dir    {basis_dir}\n")


if __name__ == "__main__":
    main()

