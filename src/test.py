import pyscf
import matplotlib.pyplot as plt
import numpy as np
import json

from pyscf import gto, scf, tdscf
from geometry.tetracene_geo import tetracene
from helper import *

# Define molecule and calculation parameters
molecule = tetracene
spin = 0
basis_set = 'sto-3g'

# Initialize Pyscf_helper
scf_water_init = Pyscf_helper()

# Run SCF calculation
scf_data = scf_water_init.pyscf_scf(molecule, spin, basis_set)

# Get the mean_field object from Pyscf_helper
mean_field = scf_water_init.mean_field  # This was missing

# Run configuration interaction singles
n_singlets = 1  # Example value for number of singlets
n_triplets = 1  # Example value for number of triplets
cis_singlet_E, cis_triplet_E, density_singlet, density_triplet = scf_water_init.configuration_interaction_singles(n_singlets, n_triplets)

# Print results
print("CIS Singlet Energy:", cis_singlet_E)
print("CIS Triplet Energy:", cis_triplet_E)

# Active space parameters
active_space_type = "Increasing both occupied and virtual orbital"  # Other options: "Increasing occupied orbital", "Increasing virtual orbital"

# Initialize Active_space_helper
active_space_helper = Active_space_helper()

# Get required data from SCF results
C = scf_data['coeff']  # Molecular orbital coefficients
mo_occ = scf_data['mo_occ']  # Molecular orbital occupations
overlap = scf_data['overlap']  # Overlap matrix
n_electrons = scf_data['n_electrons']  # Number of electrons

# Generate active space and calculate D_A
occupied, active, virtual, n_active, n_virtual, n_occupied, D_A = active_space_helper.generate_active_space(C, active_space_type, overlap, n_electrons, mo_occ)

# Calculate embedding potential (use the mean_field from the SCF calculation)
Vemb, D_A, n_occ, n_act, n_vir, verror = active_space_helper.calculate_embedding_potential(occupied, active, virtual, mo_occ, overlap, mean_field)

# Define energy lists and active sizes for plotting
active_sizes = list(range(n_act))  # Example active sizes, customize as needed
elists = []  # List to store energies (you'll need to fill this)
elistt = []  # List to store triplet energies (you'll need to fill this)

# Plot energy of singlet and triplet vs active space size
e_s_cis = cis_singlet_E  # Singlet CIS energy from earlier
e_t_cis = cis_triplet_E  # Triplet CIS energy from earlier

# Initialize the plot
plot = Plot()
plot.plot(active_sizes, elists, e_s_cis, e_t_cis, elistt)
