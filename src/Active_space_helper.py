
class Active_space_helper:
    def __init__(self) -> None:
        pass
    
    def get_state_averaged_rdm(self, orbital_type, coeff, reduced_density, overlap, n_electrons, n_singlets, n_triplets):
        if orbital_type = "natural orbitals" 
        

        elif orbital_type = "canonical orbitals"
            state_averaged_reduced_density = reduced_density / (n_singlets + n_triplets + 1)
            canonical_reduced_density = coeff.T @ overlap @ reduced_density @ overlap @ coeff
            # np.savetxt("no_avg_rdm", reduced_density, fmt='%1.13f')
            D_evals, D_evecs = np.linalg.eigh(reduced_density)
            sorted_list = np.argsort(D_evals)[::-1]
            D_evals = D_evals[sorted_list] 
            # np.savetxt("Density matrix eigenvalues and eigenvectors", D_evals, fmt='%1.13f')
            D_evecs = D_evecs[:,sorted_list]
            NO = coeff @ D_evecs
        return reduced_density, D_evals, D_evecs

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
    
    def generate_active_space(self, active_space_type, n_electrons, mo_occ):
        HOMO_index = np.where(mo_occ == 2)[0][-1]
        LUMO_index = HOMO_index + 1

        active_list = []
        double_occupied_list = []

        active_sizes = []

        if active_space_type = "Increasing both occupied and virtual orbital"
            w = round(num_orbitals - 0.5 * n_elec)
            print('w = ', w)
            for i in range(0, w -1):
                active_sizes.append(2*i)

                occ_list = list(range(0,HOMO_index-i))
                act_list = list(range(HOMO_index - i, LUMO_index+i+1))
                vir_list = list(range(LUMO_index + i +1, num_orbitals))

                print('actlist',act_list)
                act_array = np.array(act_list)

                Cocc = C[:, occ_list]
                occ_occ = mo_occ[occ_list]
                occ_occ = np.array(occ_occ)

                Cact = C[:, act_list]
                act_occ = mo_occ[act_list]
                act_occ = np.array(act_occ)

            #molden.from_mo(mol, 'ttc_act.molden',Cact)

                Cvir = C[:, vir_list]
                vir_occ = mo_occ[vir_list]
                vir_occ = np.array(vir_occ)

                nact = len(act_list)
                nvir = len(vir_list)
                nocc = len(occ_list)

        if active_space_type = "Increasing occupied orbital"
            w = round(num_orbitals - 0.5 * n_elec)
            print('w = ', w)
            mo_occ = mo_occ[::-1]
            for i in range(1, w):
                active_sizes.append(i)

                act_list = list(range(w - i, num_orbitals))
                vir_list = list(range(0, w -i))

                act_array = np.array(act_list)

                Cact = NO[:, act_list]
                act_occ = D_evals[act_list]
                act_occ = np.array(act_occ)

                Cvir = NO[:, vir_list]
                vir_occ = D_evals[vir_list]
                vir_occ = np.array(vir_occ)

            #molden.from_mo(mol, 'ttc_act.molden',Cact)
                nact = len(act_list)
                nvir = len(vir_list)

        if active_space_type = "Increasing virtual orbital"
            #allvir
            w = round(0.5 * n_eletrons)
            print('w = ',w)
            mo_occ = mo_occ[::-1]
            for i in range(1, w):
                active_sizes.append(i)

                act_list = list(range(0, num_orbitals-w+i))
                vir_list = list(range(num_orbitals-w+i, num_orbitals))

                act_array = np.array(act_list)

                Cact = NO[:, act_list]
                act_occ =  D_evals[act_list]
                act_occ = np.array(act_occ)

                Cvir = NO[:, vir_list]
                vir_occ = D_evals[vir_list]
                vir_occ = np.array(vir_occ)

            #molden.from_mo(mol, 'ttc_act.molden',Cact)
                nact = len(act_list)
                nvir = len(vir_list)

        return LUMO_index, HOMO_index, act_occ, vir_occ
        

    def calculate_in_active_space(self, orbital_type, coeff, reduced_density, overlap, n_electrons): 
        if orbital_type = "natural orbitals"
        

        if orbital_type = "canonical orbitals"