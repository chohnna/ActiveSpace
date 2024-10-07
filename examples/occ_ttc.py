import pyscf 
from pyscf import gto, scf, tdscf, cc, tddft
from pyscf.tools import molden
import matplotlib.pyplot as plt

from functools import reduce
import numpy as np
import scipy

def print_matrix_as_integers(mat):
    for row in mat:
        row_str = " ".join(str(int(elem)) for elem in row)
        print(row_str)

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

# Perform Restricted Hartree-Fock calculation
mf = scf.RHF(mol).run(verbose = 4)
mf.analyze()

H_core = mf.get_hcore()
e_hf = mf.kernel()

F = mf.get_fock()
J, K = mf.get_jk()
D = mf.make_rdm1()
S = mf.get_ovlp()

Vsys = mf.get_veff()
mo_energy = mf.mo_energy
mo_occ = mf.get_occ(mo_energy)
C = mf.mo_coeff

np.savetxt("C", C, fmt='%1.13f')
np.savetxt("D", D, fmt='%1.13f')
np.savetxt("S", S, fmt='%1.13f')
np.savetxt("F", S, fmt='%1.13f')

# #canonicalize
# e, v = np.linalg.eigh(S)
# X = np.dot(v*np.sqrt(e), v.T.conj())
# Xinv = scipy.linalg.inv(X)

# #canonicalize
# F_mo = C.T @ F @ C
# e, v = np.linalg.eigh(F_mo)
# C_bar = C @ v
# #orthogonalize
# C_bar = X @ C
# F_bar = Xinv.T @ F @ Xinv
# D_bar = mf.make_rdm1(C_bar, mo_occ) 

# np.savetxt("C_bar", C_bar, fmt='%1.13f')
# np.savetxt("D_bar", D_bar, fmt='%1.13f')
# np.savetxt("F_bar", F_bar, fmt='%1.13f')
# np.savetxt("F_bar_mo", C_bar.T @ F_bar @ C_bar, fmt='%1.13f')
# np.savetxt("D_bar_mo", C_bar.T @ D_bar @ C_bar, fmt='%1.13f')


num_orbitals = len(C)
print("number of orbitals: %12.8f" %num_orbitals)
print("number of electrons: %12.8f" %np.trace(S@D))
molden.from_mo(mol, 'ttc_hf.molden', C)

# Hartree to eV conversion factor
ev_conversion_factor = 27.2114079527
ev_conversion_factor = float(ev_conversion_factor)

#eom-ccsd
mycc = cc.CCSD(mf).run()
e_s, c_s = mycc.eomee_ccsd_singlet(nroots=1)
e_t, c_t = mycc.eomee_ccsd_triplet(nroots=1)
print('eomcc singlet, triplet', e_s, e_t)
e_s = e_s * ev_conversion_factor
e_t = e_t * ev_conversion_factor
print('eomcc singlet,triplet eV', e_s, e_t)

#tddft
mytd = mol.RKS().run().TDDFT().run()
e_dft_s = min(mytd.kernel()[0])
mytd.singlet = False
e_dft_t = min(mytd.kernel()[0])
print('tddcf singlet, triplet', e_dft_s, e_dft_t)
e_dft_s = e_dft_s * ev_conversion_factor
e_dft_t = e_dft_t * ev_conversion_factor
print('tddft singlet,triplet eV', e_dft_s, e_dft_t)

#TDA
tda_h2o = tdscf.TDA(mf)
tda_h2o.nstates = 1
e_s_cis = min(tda_h2o.kernel()[0])

tda_h2o = tdscf.TDA(mf)
tda_h2o.nstates = 1
tda_h2o.singlet = False
e_t_cis = min(tda_h2o.kernel()[0])
                  
print('tda singlet, triplet', e_s_cis, e_t_cis)
e_s_cis = e_s_cis * ev_conversion_factor
e_t_cis = e_t_cis * ev_conversion_factor
print('cis singlet,triplet eV', e_s_cis, e_t_cis)

act_list = []
doc_list = []

# Calculate HOMO and LUMO indices from mo_occ
homo_index = np.where(mo_occ == 2)[0][-1]
lumo_index = homo_index + 1

print('homoidx',homo_index)
print('lumoidx',lumo_index)

#active_sizes = len(lumo_index,num_orbitals)
#active_sizes = list(range(0,active_sizes))

active_sizes = []
elists = []
elistt = []
hf = []

for i in range(lumo_index+1,num_orbitals+1):

    active_sizes.append(i)

    act_list = list(range(0, i))
    vir_list = list(range(i, num_orbitals))

    act_array = np.array(act_list)

    Cact = C[:, act_list]
    act_occ = mo_occ[act_list]
    act_occ = np.array(act_occ)

   #molden.from_mo(mol, 'ttc_act.molden',Cact)

    Cvir = C[:, vir_list]
    vir_occ = mo_occ[vir_list]
    vir_occ = np.array(vir_occ)
    nact = len(act_list)
    nvir = len(vir_list)

    D_A = np.dot(Cact*act_occ, Cact.conj().T)
    #molden.from_mo(mol, 'ttc_D_A%4i.molden' %(nact), D_A)
    D_C = np.dot(Cvir*vir_occ, Cvir.conj().T)
    D_tot = D_A +D_C
    # D_C = np.dot(Cvir, Cvir.conj().T)
    P_c = np.dot(Cvir, Cvir.conj().T)

    #projector
    P = S @ P_c @ S
    mu = 1.0e6

    Vsys = mf.get_veff(dm=D_tot)
    Vact = mf.get_veff(dm=D_A) 
    Venv = mf.get_veff(dm=D_C)

    #new fock ao
    Vemb = Vsys - Vact + (mu * P)
    verror = Vsys - Vact


    n_act = 2*round(0.5 * np.trace(S@D_A))

    print('Number of Active orbitals: ', nact)
    print('Number of Virtual orbitals: ', nvir) 
    print('Number of electrons in active space',n_act) 

    emb_mf = scf.RHF(mol)
    mol.nelectron = n_act
    mol.build()

    emb_mf.verbose = 4
    emb_mf.get_hcore = lambda *args: H_core + Vemb
    emb_mf.max_cycle = 200
    e_hf_act = emb_mf.kernel(dm0=D_A)
    print('ehfact',e_hf_act)

    emb_tda = tdscf.TDA(emb_mf)
    emb_tda.nstates = 3
    e = min(emb_tda.kernel()[0])
    e_ev = e * ev_conversion_factor

    emb_tda_t = tdscf.TDA(emb_mf)
    emb_tda_t.nstates = 3
    emb_tda_t.singlet = False
    e_t = min(emb_tda_t.kernel()[0])
    e_t_ev = e_t * ev_conversion_factor

    elists.append(e_ev)
    elistt.append(e_t_ev)

elists = np.asarray(elists)
np.savetxt("occ_singlet", elists, fmt='%1.13f')

elistt = np.asarray(elistt)
np.savetxt("occ_triplet", elistt, fmt='%1.13f')

active_sizes = [i - lumo_index for i in active_sizes ]

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(active_sizes, elists, marker='o', linestyle='-', label='CIS_act')
plt.axhline(y=e_s_cis, color='blue', linestyle='--', label='CIS')
plt.axhline(y=e_s, color='black', linestyle='--', label='EOM-CCSD')
plt.axhline(y=e_dft_s, color='red', linestyle='--', label='TDDFT')
plt.xlabel('# of unoccupied orbitals in active space')
plt.ylabel('Excited State energies (eV)')
plt.title('Excitation energy of active space; Singlets')
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
plt.plot(active_sizes, elistt, marker='o', linestyle='-', label='CIS_act')
plt.axhline(y=e_t_cis, color='blue', linestyle='--', label='CIS')
plt.axhline(y=e_t, color='black', linestyle='--', label='EOM-CCSD')
plt.axhline(y=e_dft_t, color='red', linestyle='--', label='TDDFT')
plt.xlabel('# of unoccupied orbitals in active space')
plt.ylabel('Excited State energies (eV)')
plt.title('Excitation energy of active space; Triplets')
plt.grid(True)
plt.legend()
plt.tight_layout()

file_name = "ttc_allocc_plots.png"
plt.savefig(file_name)
plt.show()
