
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
                n_active_orbital = len(active_list)
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

                return active, virtual, occupied, nact, nvir, nocc

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

        return LUMO_index, HOMO_index, act_occ, vir_occ
        

    def calculate_in_active_space(self, orbital_type, coeff, reduced_density, overlap, n_electrons): 
        if orbital_type = "natural orbitals"
        

        if orbital_type = "canonical orbitals"


