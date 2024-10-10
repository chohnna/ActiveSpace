        


class Active_space_helper:
    def __init__(self) -> None:
        pass
    
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
    
    def generate_active_space(self, active_space_type, n_electrons):
        if active_space_type = "Increasing HOMO and LUMO"
    
        if active_space_type = "Increasing HOMO"
    
        if active_space_type = "Increasing LUMO"
    
        

    def calculate_in_active_space(self, orbital_type, rdm, C, S, n_electrons): 
        if orbital_type = "natural orbitals"


        if orbital_type = "canonical orbitals"