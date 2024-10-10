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

        core_hamiltonian = mean_field.get_hcore()
        mo_energy = mean_field.mo_energy
        mo_occ = mean_field.get_occ(mo_energy)
        overlap = mean_field.get_ovlp()
        coeff = mean_field.mo_coeff
        fock_matrix = mean_field.get_fock()
        density = mean_field.make_rdm1()
        n_electrons = 2*round(0.5 * np.trace(density@overlap))
        
        self.mf = mean_field

        return core_hamiltonian, mo_energy, mo_occ, overlap, coeff, fock_matrix, density
    
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

