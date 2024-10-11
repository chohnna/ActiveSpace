import pyscf
import matplotlib.pyplot as plt
import numpy as np
from pyscf import gto, scf, tdscf


class Pyscf_helper:
    # Constructor
    def __init__(self):
        # Instance variables
        self.core_hamiltonian = None
        self.mo_energy = None
        self.n_orbitals = None
        self.n_electrons = None
        self.mo_occ = None
        self.overlap = None
        self.coeff = None
        self.fock_matrix = None
        self.density_matrix = None
        self.mean_field = None

    def pyscf_scf(self, molecule, spin, basis_set):
        """
        Perform SCF (Self-Consistent Field) calculation on the given molecule.
        """
        mol = gto.Mole()
        mol.atom = molecule
        mol.basis = basis_set
        mol.spin = spin
        mol.build()

        self.mean_field = scf.RHF(mol).run(verbose=4)
        self.mean_field.analyze()

        # Assign the values to instance variables
        self.core_hamiltonian = self.mean_field.get_hcore()
        self.mo_energy = self.mean_field.mo_energy
        self.mo_occ = self.mean_field.get_occ(self.mo_energy)
        self.overlap = self.mean_field.get_ovlp()
        self.coeff = self.mean_field.mo_coeff
        self.fock_matrix = self.mean_field.get_fock()
        self.density_matrix = self.mean_field.make_rdm1()
        self.n_electrons = 2 * round(0.5 * np.trace(self.density_matrix @ self.overlap))
        self.n_orbitals = len(self.mo_occ)

        # Return a dictionary instead of a tuple
        return {
            'core_hamiltonian': self.core_hamiltonian,
            'mo_energy': self.mo_energy,
            'n_orbitals': self.n_orbitals,
            'n_electrons': self.n_electrons,
            'mo_occ': self.mo_occ,
            'overlap': self.overlap,
            'coeff': self.coeff,
            'fock_matrix': self.fock_matrix,
            'density_matrix': self.density_matrix
        }

    def tda_density_matrix(self, td, state_id):
        """
        Taking the TDA amplitudes as the CIS coefficients, calculate the density
        matrix (in AO basis) of the excited states.
        """
        cis_t1 = td.xy[state_id][0]
        dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
        dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())
        mf = td._scf
        dm = np.diag(mf.mo_occ)
        nocc = cis_t1.shape[0]
        dm[:nocc, :nocc] += dm_oo * 2
        dm[nocc:, nocc:] += dm_vv * 2
        mo = mf.mo_coeff
        dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
        return dm

    def configuration_interaction_singles(self, n_singlets, n_triplets):
        density_singlet = 0
        density_triplet = 0

        # Compute singlets
        singlet_excitation = tdscf.TDA(self.mean_field)
        singlet_excitation.singlet = True
        singlet_excitation = singlet_excitation.run(nstates=n_singlets)
        singlet_excitation.analyze()
        cis_singlet_E = min(singlet_excitation.kernel()[0])
        for i in range(singlet_excitation.nroots):
            density_singlet += self.tda_density_matrix(singlet_excitation, i)

        # Compute triplets
        triplet_excitation = tdscf.TDA(self.mean_field)
        triplet_excitation.singlet = False
        triplet_excitation = triplet_excitation.run(nstates=n_triplets)
        triplet_excitation.analyze()
        cis_triplet_E = min(triplet_excitation.kernel()[0])
        for i in range(triplet_excitation.nroots):
            density_triplet += self.tda_density_matrix(triplet_excitation, i)

        return cis_singlet_E, cis_triplet_E, density_singlet, density_triplet

class Active_space_helper:
    def __init__(self) -> None:
        pass
class Active_space_helper:
    def __init__(self) -> None:
        pass

class Active_space_helper:
    def __init__(self) -> None:
        pass

class Active_space_helper:
    def __init__(self) -> None:
        pass

    def generate_active_space(self, C, active_space_type, overlap, n_electrons, mo_occ):
        """
        Generate active space based on the type of active space selection.
        """
        HOMO_index = np.where(mo_occ == 2)[0][-1]  # Highest occupied molecular orbital index
        LUMO_index = HOMO_index + 1  # Lowest unoccupied molecular orbital index
        n_orbitals = len(mo_occ)
        n_columns = C.shape[1]  # Number of columns in C (molecular orbitals)

        active_list = []
        virtual_list = []
        double_occupied_list = []

        for i in range(min(HOMO_index, LUMO_index)):
            if active_space_type == "Increasing both occupied and virtual orbital":
                double_occupied_list = list(range(HOMO_index - i))
                active_list = list(range(HOMO_index - i, LUMO_index + i + 1))
                virtual_list = list(range(LUMO_index + i + 1, n_orbitals))

            elif active_space_type == "Increasing occupied orbital":
                double_occupied_list = list(range(HOMO_index - i))
                active_list = list(range(HOMO_index - i, LUMO_index))
                virtual_list = list(range(LUMO_index + 1, n_orbitals))

            elif active_space_type == "Increasing virtual orbital":
                active_list = list(range(0, n_orbitals - i))
                virtual_list = list(range(n_orbitals - i, n_orbitals))

            # Ensure indices are within bounds of C
            active_list = [a for a in active_list if a < n_columns and a >= 0]
            virtual_list = [v for v in virtual_list if v < n_columns and v >= 0]
            double_occupied_list = [d for d in double_occupied_list if d < n_columns and d >= 0]

            # Skip if no valid orbitals are selected or if index is 0
            if len(active_list) == 0 or len(virtual_list) == 0 or len(double_occupied_list) == 0:
                print("Skipping active space generation: No valid orbitals selected.")
                continue

            # Create orbitals from the coefficients
            occupied = C[:, double_occupied_list] if double_occupied_list else np.array([])  # Handle empty arrays
            active = C[:, active_list] if active_list else np.array([])
            virtual = C[:, virtual_list] if virtual_list else np.array([])

            # Ensure electron counts match the total electron count (Nalpha + Nbeta)
            n_occ_electrons = 2 * len(double_occupied_list)  # Occupied orbitals contribute 2 electrons each
            n_act_electrons = len(active_list)  # Active orbitals contribute 1 electron each
            total_electrons = n_occ_electrons + n_act_electrons

            if total_electrons != n_electrons:
                print(f"Skipping active space: Electron count {total_electrons} does not match expected {n_electrons}.")
                continue

            # Print debug information
            print(f"Occupied orbitals: {occupied.shape}")
            print(f"Active orbitals: {active.shape}")
            print(f"Virtual orbitals: {virtual.shape}")

            # Break if any of the orbital lists are empty, indicating no valid orbitals
            if occupied.size == 0 or active.size == 0 or virtual.size == 0:
                print("Skipping calculation: One of the orbital lists is empty.")
                break

        # Return active space information
        n_active_orbitals = len(active_list)
        return active, virtual, occupied, len(active), len(virtual_list), len(double_occupied_list), n_active_orbitals

    def calculate_embedding_potential(self, occupied, active, virtual, mo_occ, overlap, mean_field):
        """
        Calculate the embedding potential Vemb and related quantities.
        Handle cases where arrays might be empty and electron count issues.
        """
        try:
            # Number of occupied, active, and virtual orbitals
            n_occ = occupied.shape[1] if occupied.size > 0 else 0
            n_act = active.shape[1] if active.size > 0 else 0
            n_vir = virtual.shape[1] if virtual.size > 0 else 0

            # Skip calculation if all orbital spaces are empty
            if n_occ == 0 and n_act == 0 and n_vir == 0:
                print("Skipping embedding potential calculation: No valid orbitals.")
                return None, None, 0, 0, 0, None

            # Print debug info
            print(f"Number of occupied orbitals: {n_occ}")
            print(f"Number of active orbitals: {n_act}")
            print(f"Number of virtual orbitals: {n_vir}")

            # Density matrices (Handle empty arrays)
            D_O = np.dot(occupied * mo_occ[:n_occ], occupied.conj().T) if n_occ > 0 else np.zeros((overlap.shape[0], overlap.shape[1]))
            D_A = np.dot(active * mo_occ[n_occ:n_occ + n_act], active.conj().T) if n_act > 0 else np.zeros((overlap.shape[0], overlap.shape[1]))
            D_C = np.dot(virtual * mo_occ[-n_vir:], virtual.conj().T) if n_vir > 0 else np.zeros((overlap.shape[0], overlap.shape[1]))

            # Total density matrix
            D_tot = D_O + D_A + D_C

            # Projectors
            P_c = np.dot(virtual, virtual.conj().T) if n_vir > 0 else np.zeros_like(overlap)
            P_o = np.dot(occupied, occupied.conj().T) if n_occ > 0 else np.zeros_like(overlap)

            # Projector P
            P = overlap @ P_c @ overlap + overlap @ P_o @ overlap

            # Embedding calculations
            mu = 1.0e6
            Vsys = mean_field.get_veff(dm=D_tot)
            Vact = mean_field.get_veff(dm=D_A)
            Venv = mean_field.get_veff(dm=D_C)

            # Embedding potential
            Vemb = Vsys - Vact + (mu * P)
            verror = Vsys - Vact  # Can be used for error estimation or analysis

            return Vemb, D_A, n_occ, n_act, n_vir, verror

        except Exception as e:
            print(f"Error calculating embedding potential: {e}")
            return None, None, 0, 0, 0, None


    def calculate_in_active_space(self, mol, n_act, H_core, Vemb, D_A):
        """
        Placeholder for calculation of density matrix in active space.
        """
        elists = []
        elistt = []

        emb_mf = scf.RHF(mol)
        mol.nelectron = n_act
        mol.build()

        emb_mf.verbose = 4
        emb_mf.get_hcore = lambda *args: H_core + Vemb
        emb_mf.max_cycle = 200
        e_hf_act = emb_mf.kernel(dm0=D_A)
        print('ehfact', e_hf_act)

        emb_tda = tdscf.TDA(emb_mf)
        emb_tda.nstates = 3
        e = min(emb_tda.kernel()[0])

        emb_tda_t = tdscf.TDA(emb_mf)
        emb_tda_t.nstates = 3
        emb_tda_t.singlet = False
        e_t = min(emb_tda_t.kernel()[0])

        elists.append(e)
        elistt.append(e_t)
        return elists, elistt


class Plot:
    def __init__(self):
        pass

    def plot(self, active_sizes, elists, e_s_cis, e_t_cis, elistt):
        """
        Plot excitation energies for singlets and triplets based on active space size.
        """
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(active_sizes, elists, marker='o', linestyle='-', label='CIS_act')
        plt.axhline(y=e_s_cis, color='blue', linestyle='--', label='CIS')
        plt.xlabel('# of orbitals in active space')
        plt.ylabel('Excited State energies (eV)')
        plt.title('Excitation energy of active space; Singlets')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.plot(active_sizes, elistt, marker='o', linestyle='-', label='CIS_act')
        plt.axhline(y=e_t_cis, color='blue', linestyle='--', label='CIS')
        plt.xlabel('# of orbitals in active space')
        plt.ylabel('Excited State energies (eV)')
        plt.title('Excitation energy of active space; Triplets')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        file_name = "no_HL_plots.png"
        plt.savefig(file_name)
        plt.show()
