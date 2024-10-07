# create instance
from water import water
from src import Pyscf_helper

molecule = water
spin = 0
basis_set = 'sto-3g'

scf_water_init = Pyscf_helper()
scf_water = scf_water_init.pyscf_scf(molecule, spin, basis_set)

cis_water = scf_water_init.configuration_interaction_singles(mean_field ,1, 1)