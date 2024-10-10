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