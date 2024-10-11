import pyscf
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, tdscf
from geometry.water_geo import water
from helper import *

# Define molecule and calculation parameters
molecule = water
spin = 0
basis_set = 'sto-3g'

# Initialize Pyscf_helper
scf_water_init = Pyscf_helper()

# Run SCF calculation with error handling and debug prints
try:
    scf_data = scf_water_init.pyscf_scf(molecule, spin, basis_set)
    print("SCF Data keys:", scf_data.keys())  # Check what data is available
except Exception as e:
    print(f"Error during SCF calculation: {e}")
    exit()

# Get the mean_field object from Pyscf_helper
mean_field = scf_water_init.mean_field 

# Run configuration interaction singles
n_singlets = 1  # Example value for number of singlets
n_triplets = 1  # Example value for number of triplets
try:
    cis_singlet_E, cis_triplet_E, density_singlet, density_triplet = scf_water_init.configuration_interaction_singles(n_singlets, n_triplets)
except Exception as e:
    print(f"Error during CIS calculation: {e}")
    exit()

# Print results
print(f"CIS Singlet Energy: {cis_singlet_E:.4f} eV")
print(f"CIS Triplet Energy: {cis_triplet_E:.4f} eV")

# Active space parameters
active_space_type = "Increasing both occupied and virtual orbital"  # Other options: "Increasing occupied orbital", "Increasing virtual orbital"

# Initialize Active_space_helper
active_space_helper = Active_space_helper()

# Get required data from SCF results
try:
    C = scf_data['coeff']  # Molecular orbital coefficients
    mo_occ = scf_data['mo_occ']  # Molecular orbital occupations
    overlap = scf_data['overlap']  # Overlap matrix
    n_electrons = scf_data['n_electrons']  # Number of electrons

    # Check if the retrieved arrays are empty
    if len(mo_occ) == 0 or C.size == 0:
        raise ValueError("SCF calculation did not generate valid molecular orbitals or occupancy.")
    
    print("Molecular Occupation (mo_occ):", mo_occ)
    print("Coefficient matrix shape (C):", C.shape)

except KeyError as e:
    print(f"Missing data in SCF results: {e}")
    exit()
except ValueError as e:
    print(e)
    exit()

# Generate active space and calculate D_A
try:
    occupied, active, virtual, n_active, n_virtual, n_occupied, D_A = active_space_helper.generate_active_space(C, active_space_type, overlap, n_electrons, mo_occ)
except Exception as e:
    print(f"Error generating active space: {e}")
    exit()

# Calculate embedding potential (use the mean_field from the SCF calculation)
try:
    Vemb, D_A, n_occ, n_act, n_vir, verror = active_space_helper.calculate_embedding_potential(occupied, active, virtual, mo_occ, overlap, mean_field)
except Exception as e:
    print(f"Error calculating embedding potential: {e}")
    exit()

# Calculate in active space (pass mol and other variables properly)
try:
    elists, elistt = active_space_helper.calculate_in_active_space(mean_field.mol, n_act, scf_data['core_hamiltonian'], Vemb, D_A)
except Exception as e:
    print(f"Error in active space calculation: {e}")
    exit()

# Plot energy of singlet and triplet vs active space size
e_s_cis = cis_singlet_E  # Singlet CIS energy from earlier
e_t_cis = cis_triplet_E  # Triplet CIS energy from earlier

# Initialize the plot
plot = Plot()
plot.plot(n_act, elists, e_s_cis, e_t_cis, elistt)
