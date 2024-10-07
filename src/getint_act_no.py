import pyscf
from pyscf import gto, scf, tdscf, cc, tddft, ao2mo
from pyscf.tools import molden
import matplotlib.pyplot as plt
import numpy as np
import scipy

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

# Create the molecule
mol = gto.Mole()
mol.atom = '''
H       -3.4261000000     -2.2404000000      5.4884000000
H       -5.6274000000     -1.0770000000      5.2147000000
C       -3.6535000000     -1.7327000000      4.5516000000
H       -1.7671000000     -2.2370000000      3.6639000000
C       -4.9073000000     -1.0688000000      4.3947000000
H       -6.1631000000      0.0964000000      3.1014000000
C       -2.7258000000     -1.7321000000      3.5406000000
H       -0.3003000000      1.0832000000     -5.2357000000
C       -5.2098000000     -0.4190000000      3.2249000000
C       -2.9961000000     -1.0636000000      2.3073000000
H       -1.1030000000     -1.5329000000      1.3977000000
H       -0.4270000000     -0.8029000000     -0.8566000000
H        0.2361000000     -0.0979000000     -3.1273000000
C       -1.0193000000      1.0730000000     -4.4150000000
H       -2.4988000000      2.2519000000     -5.5034000000
C       -4.2740000000     -0.3924000000      2.1445000000
H       -5.5015000000      0.7944000000      0.8310000000
C       -2.0613000000     -1.0272000000      1.2718000000
C       -1.3820000000     -0.2895000000     -0.9772000000
C       -0.7171000000      0.4180000000     -3.2476000000
C       -2.2720000000      1.7395000000     -4.5690000000
H       -4.1576000000      2.2412000000     -3.6787000000
C       -4.5463000000      0.2817000000      0.9534000000
C       -2.3243000000     -0.3402000000      0.0704000000
C       -1.6528000000      0.3874000000     -2.1670000000
C       -3.1998000000      1.7341000000     -3.5584000000
C       -3.6044000000      0.3309000000     -0.0943000000
C       -2.9302000000      1.0591000000     -2.3292000000
C       -3.8665000000      1.0187000000     -1.2955000000
H       -4.8243000000      1.5256000000     -1.4217000000
'''
mol.basis = 'sto-3g'
mol.spin = 0
mol.build()
mf = scf.RHF(mol).run(verbose = 4)
mf.analyze()

H_core = mf.get_hcore()
mo_energy = mf.mo_energy
mo_occ = mf.get_occ(mo_energy)
S = mf.get_ovlp()
C = mf.mo_coeff
F = mf.get_fock()
D = mf.make_rdm1()

X = scipy.linalg.sqrtm(S)
Xinv = scipy.linalg.inv(X)

nelec = np.trace(S@D)
norb = len(C)
print("number of orbitals: %12.8f" %norb)
print("number of electrons: %12.8f" %np.trace(S@D))

# Hartree to eV conversion factor
ev_conversion_factor = 27.2114079527
ev_conversion_factor = float(ev_conversion_factor)

n_triplets = 1
n_singlets = 1

# compute singlets
mytd = tdscf.TDA(mf)
mytd.singlet = True
mytd = mytd.run(nstates=n_singlets)
mytd.analyze()
e_s_cis = min(mytd.kernel()[0])
for i in range(mytd.nroots):
    D += tda_denisty_matrix(mytd, i)
# compute triplets
mytd = tdscf.TDA(mf)
mytd.singlet = False
mytd = mytd.run(nstates=n_triplets)
mytd.analyze()
e_t_cis = min(mytd.kernel()[0])
for i in range(mytd.nroots):
    D += tda_denisty_matrix(mytd, i)

e_s_cis_eV = ev_conversion_factor*e_s_cis
e_t_cis_eV = ev_conversion_factor*e_t_cis

#2. state averaged 1RDM
rdm = D / (n_singlets + n_triplets + 1)
n_elec = 2*round(0.5 * np.trace(rdm@S))
print('Number of electrons; np.trace(rdm)', n_elec) 

rdm = C.T @ S @ rdm @ S @ C

#3. get natural orbital, diagonalize rdm
np.savetxt("no_avg_rdm", rdm, fmt='%1.13f')
D_evals, D_evecs = np.linalg.eigh(rdm)
sorted_list = np.argsort(D_evals)[::-1]
D_evals = D_evals[sorted_list] 
np.savetxt("D_eval_eigen", D_evals, fmt='%1.13f')

D_evecs = D_evecs[:,sorted_list]
NO = C @ D_evecs

#pick n most partially occupied orbiatls to be the active space
print('Number of electrons; np.trace(XrdmX)', np.trace(rdm)) 
np.savetxt("../data/NO_D_evals", D_evals, fmt='%1.13f')
np.savetxt("../data/NO_D_evec", D_evecs, fmt='%1.13f')

occ_list = []
act_list = []
doc_list = []

active_sizes = []

elists = []
elistt = []
ehf = []

# Calculate HOMO and LUMO indices from mo_occ
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

for i in range(0,norb-lumo_index):

    occ_list = list(range(0,homo_index-i))
    act_list = list(range(homo_index - i, lumo_index+i+1))
    vir_list = list(range(lumo_index + i +1, norb))

    nact = len(act_list) 
    nvir = len(vir_list)
    ncore = len(occ_list)
    nocc = nact + ncore
    norb = nocc + nvir
    
    Cocc = NO[:, :ncore]
    occ_occ = mo_occ[ncore]
    print('cocc.shape: ', Cocc.shape)
    # occ_occ = mo_occ[occ_list]
    # occ_occ = np.array(occ_occ)

    Cact = NO[:, ncore:nocc]
    act_occ = mo_occ[ncore:nocc]
    print('cact.shape: ', Cact.shape)
    # act_occ = mo_occ[act_list]
    # act_occ = np.array(act_occ)

    Cvir = NO[:, nocc:norb]
    print('cvir.shape: ', Cvir.shape)
    # vir_occ = mo_occ[vir_list]
    # vir_occ = np.array(vir_occ)

    D_O = Cocc * occ_occ @ Cocc.T
    D_A = Cact * act_occ @ Cact.T

    C = mf.mo_coeff
    h0 = mf.energy_nuc()
    H_core = mf.get_hcore(mol)

    j,k = mf.get_jk(mol, D_O, hermi =1)
    h0 += np.trace(D_O @ (H_core + 0.5*j - 0.25*k))
    #h0 nicole was -656 something
    # Rotate 1electron terms to active space
    h = Cact.T @ H_core @ Cact
    j = Cact.T @ j @ Cact;
    k = Cact.T @ k @ Cact;

    h1 = h + j - .5*k;

    print('number of electrons in active space: ', round(np.trace(S@D_A)))
    # veff = mf.get_veff(mol, D_O)

    # h0 += np.einsum('ij,ji', D_O, H_core).real
    # h0 += np.einsum('ij,ji', D_O, veff).real * .5

    # h1 = reduce(np.dot, (Cact.conj().T, H_core + veff, Cact))

    h2 = ao2mo.kernel(mol, Cact, aosym="s4", compact = False)
    h2.shape = (nact,nact,nact,nact)

    print('nact',nact)
    print('h0.shape', h0.shape)
    print('h1.shape', h1.shape)    
    print('h2.shape', h2.shape)

    np.save('../data/NO_h0_' + str(2*i+2) + '.npy',h0)
    np.save('../data/NO_h1_' + str(2*i+2) + '.npy',h1)
    np.save('../data/NO_h2_' + str(2*i+2) + '.npy',h2)

    print('h0,h1,h2 saved as: ', str(2*i+2))
