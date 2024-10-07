import pyscf
from pyscf import gto, scf, tdscf

import matplotlib.pyplot as plt
import numpy as np
import json

class Pyscf_helper:
    # constructers
    def __init__(self):
        # instance variables
        self.mf  = None
        self.mol = None
        self.h = None
        self.g      = None
        self.n_orb  = None
        self.C      = None
        self.S      = None
        self.J      = None
        self.K      = None
        self.F = None

    # methods
        
    def tda_denisty_matrix(td, state_id):
    # ‘’'
    # Taking the TDA amplitudes as the CIS coefficients, calculate the density
    # matrix (in AO basis) of the excited states
    # ‘’'
        cis_t1 = td.xy[state_id][0]
        dm_oo =-np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
        dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())
        # The ground state density matrix in mo_basis
        mf = td._scf
        dm = np.diag(mf.mo_occ)
        # Add CIS contribution
        nocc = cis_t1.shape[0]
        # Note that dm_oo and dm_vv correspond to spin-up contribution. “*2” to
        # include the spin-down contribution
        dm[:nocc,:nocc] += dm_oo * 2
        dm[nocc:,nocc:] += dm_vv * 2
        # Transform density matrix to AO basis
        mo = mf.mo_coeff
        dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
        return dm

    def pyscf_scf(self, molecule, spin, basis_set):
        mol = gto.Mole()
        mol.atom = molecule
        mol.basis = basis_set
        mol.spin = spin
        mol.build()

        mean_field = scf.RHF(mol).run(verbose = 4)
        mean_field.analyze()

        core_hamiltornian = mean_field.get_hcore()
        mo_energy = mean_field.mo_energy
        mo_occ = mean_field.get_occ(mo_energy)
        overlap = mean_field.get_ovlp()
        coeff = mean_field.mo_coeff
        fock_matrix = mean_field.get_fock()
        density = mean_field.make_rdm1()
        
        return core_hamiltornian, mo_energy, mo_occ, overlap, coeff, fock_matrix, density
    
    def configuration_interaction_singles(Pyscf_helper, n_singlets, n_triplets):
        # compute singlets
        mytd = tdscf.TDA(mean_field)
        mytd.singlet = True
        mytd = mytd.run(nstates=n_singlets)
        mytd.analyze()
        cis_singlet_E = min(mytd.kernel()[0])
        for i in range(mytd.nroots):
            density += tda_denisty_matrix(mytd, i)
        # compute triplets
        mytd = tdscf.TDA(mean_field)
        mytd.singlet = False
        mytd = mytd.run(nstates=n_triplets)
        mytd.analyze()
        cis_triplet_E = min(mytd.kernel()[0])
        for i in range(mytd.nroots):
            D += tda_denisty_matrix(mytd, i)

