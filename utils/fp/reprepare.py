# update by JamesMisaka in 2025-0609
# reprepare the abacus calc, run in abacus-forloop project dir

import numpy as np
import os
from ase.io.abacus import write_input, write_abacus
from ase.io import read
import glob
import numpy as np

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
    'scf_os_stop': 1,
    'scf_os_ndim': 80,
    'kpts': kpts,
    'pp': pp,
    'basis': basis,
    'pseudo_dir': pseudo_dir,
    'basis_dir': basis_dir,
    'cal_force': 1,
    'cal_stress': 1,
    'init_chg': 'auto',
    'out_stru': 1,
    'out_mul': 1,
    'out_chg': 0,
    'out_bandgap': 1,
}
# optional parameter
# onsite_radius 3.0

ROOTDIR = os.getcwd()

def main():
    """Main function, run after check conv and leave only non-conv here"""
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
        mag_eles = ['Fe',  'O', "H"]
        for ele in mag_eles:
            ind = [atom.index for atom in stru if atom.symbol == ele]
            if ele == 'Fe':
                magmom = np.array([(1)**n * 5 for n in range(0, len(ind))]).reshape(len(ind),1)
                # self-consistent FM is better than some AFM to mag-ground state
            else:
                magmom = np.array([(1)**n * 2 for n in range(0, len(ind))]).reshape(len(ind),1)
            init_magmom[ind] = magmom
        stru.set_initial_magnetic_moments(init_magmom)
        # write input
        write_input(open(f"{dir}/INPUT", 'w'), parameters=basic_input)
        write_abacus(open(f"{dir}/STRU", 'w'), stru, pp=basic_input['pp'], basis=basic_input['basis'])
        with open(f"{dir}/INPUT", "a") as fw:
            fw.write(f"orbital_dir    {basis_dir}\n")


if __name__ == "__main__":
    main()

