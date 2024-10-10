import pyscf
import matplotlib.pyplot as plt
import numpy as np
import json

from pyscf import gto, scf, tdscf
from geometry.water_geo import water
from Pyscf_helper import Pyscf_helper
from Active_space_helper import Active_space_helper

molecule = water
spin = 0
basis_set = 'sto-3g'

# Create an instance Initialize Pyscf_helper
scf_water_init = Pyscf_helper()

# Run SCF calculation
scf_water = scf_water_init.pyscf_scf(molecule, spin, basis_set)

# Run configuration interaction singles
n_singlets = 1  # Example value for number of singlets
n_triplets = 1  # Example value for number of triplets
cis_singlet_E, density_singlet, cis_triplet_E, density_triplet = scf_water_init.configuration_interaction_singles(n_singlets, n_triplets)

# Print results
print("CIS Singlet Energy:", cis_singlet_E)
print("CIS Triplet Energy:", cis_triplet_E)

active_space_type = "Increasing both occupied and virtual orbital" # "Increasing occupied orbital", "Increasing virtual orbital"
orbital_type = "natural orbitals" # "canonical orbitals"

active_space_init = Active_space_helper()
active_space = generate_active_space(self, C, active_space_type, n_electrons, mo_occ)

calculate_active_space = calculate_in_active_space(active_space)