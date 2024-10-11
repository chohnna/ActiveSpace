class Pyscf_helper:
    # Constructor
    def __init__(self):
        # Instance variables
        self.mf = None
        self.mol = None
        self.h = None
        self.g = None
        self.n_orb = None
        self.C = None
        self.S = None
        self.J = None
        self.K = None
        self.F = None

    # Methods
    def pyscf_scf(self, molecule, spin, basis_set):
        mol = gto.Mole()
        mol.atom = molecule
        mol.basis = basis_set
        mol.spin = spin
        mol.build()

        mean_field = scf.RHF(mol).run(verbose=4)
        mean_field.analyze()
        self.mf = mean_field

        core_hamiltonian = mean_field.get_hcore()
        mo_energy = mean_field.mo_energy
        mo_occ = mean_field.get_occ(mo_energy)
        overlap = mean_field.get_ovlp()
        coeff = mean_field.mo_coeff
        fock_matrix = mean_field.get_fock()
        density = mean_field.make_rdm1()
        n_electrons = 2*round(0.5 * np.trace(density@overlap))
        n_orb = len(mo_occ)

        return core_hamiltonian, mo_energy, n_orb, n_electrons, mo_occ, overlap, coeff, fock_matrix, density
    
    def configuration_interaction_singles(self, n_singlets, n_triplets):
        density_singlet = 0
        density_triplet = 0

        # Compute singlets
        mytd_singlet = tdscf.TDA(self.mf)
        mytd_singlet.singlet = True
        mytd_singlet = mytd_singlet.run(nstates=n_singlets)
        mytd_singlet.analyze()
        cis_singlet_E = min(mytd_singlet.kernel()[0])
        for i in range(mytd_singlet.nroots):
            density_singlet += self.tda_density_matrix(mytd_singlet, i)
        
        # Compute triplets
        mytd_triplet = tdscf.TDA(self.mf)
        mytd_triplet.singlet = False
        mytd_triplet = mytd_triplet.run(nstates=n_triplets)
        mytd_triplet.analyze()
        cis_triplet_E = min(mytd_triplet.kernel()[0])
        for i in range(mytd_triplet.nroots):
            density_triplet += self.tda_density_matrix(mytd_triplet, i)
        
        return cis_singlet_E, density_singlet, cis_triplet_E, density_triplet
    

class Active_space_helper:
    def __init__(self) -> None:
        pass
    
    def get_state_averaged_rdm(self, orbital_type, coeff, reduced_density, overlap, n_electrons, n_singlets, n_triplets):
        state_averaged_reduced_density = reduced_density / (n_singlets + n_triplets + 1) 
        
        if orbital_type == "natural orbitals": 
            natural_density = coeff.T @ overlap @ reduced_density @ overlap @ coeff
            # np.savetxt("no_avg_rdm", reduced_density, fmt='%1.13f')
            D_evals, D_evecs = np.linalg.eigh(natural_density)
            sorted_list = np.argsort(D_evals)[::-1]
            D_evals = D_evals[sorted_list] 
            # np.savetxt("Density matrix eigenvalues and eigenvectors", D_evals, fmt='%1.13f')
            D_evecs = D_evecs[:,sorted_list]
            C = coeff @ D_evecs        
            return C, reduced_density, D_evals, D_evecs
        elif orbital_type == "canonical orbitals":
            C = coeff
            return C, reduced_density

    def tda_density_matrix(self, td, state_id):
        '''
        Taking the TDA amplitudes as the CIS coefficients, calculate the density
        matrix (in AO basis) of the excited states.
        '''
        cis_t1 = td.xy[state_id][0]
        dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
        dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())
        # The ground state density matrix in mo_basis
        mf = td._scf
        dm = np.diag(mf.mo_occ)
        # Add CIS contribution
        nocc = cis_t1.shape[0]
        # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
        # include the spin-down contribution
        dm[:nocc, :nocc] += dm_oo * 2
        dm[nocc:, nocc:] += dm_vv * 2
        # Transform density matrix to AO basis
        mo = mf.mo_coeff
        dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
        return dm
    
    def generate_active_space(self, C, active_space_type, n_electrons, mo_occ):
        HOMO_index = np.where(mo_occ == 2)[0][-1]
        LUMO_index = HOMO_index + 1
        smaller_index = min(HOMO_index, LUMO_index)
        n_orbital = len(mo_occ)

        active_list = []
        virtual_list = []
        double_occupied_list = []

        if active_space_type = "Increasing both occupied and virtual orbital"
            for i in smaller_index:
            
                double_occupied_list = list(range(HOMO_index-i))
                active_list = list(range(HOMO_index-i, LUMO_index+i+1))
                virtual_list = list(range(LUMO_index+i+1, n_orbital))

                occupied = C[:, double_occupied_list]
                occ_occ = mo_occ[double_occupied_list]
                occ_occ = np.array(occ_occ)

                active = C[:, active_list]
                act_occ = mo_occ[active_list]
                act_occ = np.array(act_occ)
                    
                virtual = C[:, virtual_list]
                vir_occ = mo_occ[virtual_list]
                vir_occ = np.array(vir_occ)

                nact = len(active)
                nvir = len(virtual_list)
                nocc = len(occupied)

            n_active_orbital = len(active_list)
            return active, virtual, occupied, nact, nvir, nocc, n_active_orbital

        if active_space_type = "Increasing occupied orbital"
            for i in smaller_index:
                n_active_orbital = len(active_list)
                double_occupied_list = list(range(HOMO_index-i))
                active_list = list(range(HOMO_index-i, LUMO_index))
                virtual_list = list(range(LUMO_index + 1, n_orbital))

                active = C[:, active_list]
                act_occ = D_evals[active_list]
                act_occ = np.array(act_occ)

                Cvir = NO[:, virtual_list]
                vir_occ = D_evals[virtual_list]
                vir_occ = np.array(vir_occ)

                nact = len(active_list)
                nvir = len(virtual_list)
            n_active_orbital = len(active_list)
            return active, virtual, occupied, nact, nvir, nocc, n_active_orbital

        if active_space_type = "Increasing virtual orbital"
            #allvir
            w = round(0.5 * n_eletrons)
            print('w = ',w)
            mo_occ = mo_occ[::-1]
            for i in range(1, w):
                active_sizes.append(i)

                active_list = list(range(0, num_orbitals-w+i))
                virtual_list = list(range(num_orbitals-w+i, num_orbitals))

                Cact = NO[:, active_list]
                act_occ =  D_evals[active_list]
                act_occ = np.array(act_occ)

                Cvir = NO[:, virtual_list]
                vir_occ = D_evals[virtual_list]
                vir_occ = np.array(vir_occ)

            #molden.from_mo(mol, 'ttc_act.molden',Cact)
                nact = len(active_list)
                nvir = len(virtual_list)

        n_active_orbital = len(active_list)
        return active, virtual, occupied, nact, nvir, nocc, n_active_orbital

        return LUMO_index, HOMO_index, act_occ, vir_occ
        

    def calculate_in_active_space(self, orbital_type, coeff, reduced_density, overlap, n_electrons): 
        if orbital_type = "natural orbitals"
        

        if orbital_type = "canonical orbitals"



class plot:
    def __init__(self):
        

    def plot(self, n_active_orbital,  ):
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(active_sizes, elists_no, marker='o', linestyle='-', label='CIS_act')
        plt.axhline(y=e_s_cis, color='blue', linestyle='--', label='CIS')
        # plt.axhline(y=e_s, color='black', linestyle='--', label='EOM-CCSD')
        # plt.axhline(y=e_dft_s, color='red', linestyle='--', label='TDDFT')
        plt.xlabel('# of orbitals in active space')
        plt.ylabel('Excited State energies (eV)')
        plt.title('Excitation energy of active space; Singlets')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.plot(active_sizes_can, elistt_no, marker='o', linestyle='-', label='CIS_act')
        plt.axhline(y=e_t_cis, color='blue', linestyle='--', label='CIS')
        # plt.axhline(y=e_t, color='black', linestyle='--', label='EOM-CCSD')
        # plt.axhline(y=e_dft_t, color='red', linestyle='--', label='TDDFT')
        plt.xlabel('# of orbitals in active space')
        plt.ylabel('Excited State energies (eV)')
        plt.title('Excitation energy of active space; Triplets')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        file_name = "no_HL_plots.png"
        plt.savefig(file_name)
        plt.show()