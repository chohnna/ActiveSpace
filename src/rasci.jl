using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using Printf
using Arpack
using NPZ
using FileIO
using Plots

# Arrays to store data for plotting
energies = Float64[]
num_orbitals = Int64[]
excitation_energy_i = Float64[]

norb = 102
nelec = 60

println("norb = ", norb)
println("number of electrons: ", nelec)

homo_index = 59
lumo_index = 60

# for i in 1:norb - lumo_index
for i in 1:norb - lumo_index
    # Load data from npy files
    h0 = npzread("../data/h0_$(i-1).npy")
    h1 = npzread("../data/h1_$(i-1).npy")
    h2 = npzread("../data/h2_$(i-1).npy")

    norbs = 2 * (i)
    n_elec_a = i
    n_elec_b = i
    norbs_ras1 = i
    norbs_ras2 = 0
    norbs_ras3 = i
    n_holes = 1
    n_particles = 1

    # get h0, h1, h2 from pyscf or elsewhere and create ints
    ints = InCoreInts(h0, h1, h2)	

    # to use RASCI, we need to define the number of orbitals, electrons, number of orbitals in each RAS subspace (ras1, ras2, ras3), maximum number of holes allowed in ras1, and maximum number of particle excitations allowed in ras3
    ansatz = RASCIAnsatz(norbs, n_elec_a, n_elec_b, (norbs_ras1, norbs_ras2, norbs_ras3), max_h=n_holes, max_p=n_particles)

    # We define some solver settings - default uses Arpack.jl
    solver = SolverSettings(nroots=5, tol=1e-6, maxiter=100)

    println("solver", solver)

    # we can now solve our Ansatz and get energies and vectors from solution
    solution = solve(ints, ansatz, solver)
    
    display(solution)

    e = solution.energies
    v = solution.vectors

    for i in 2:length(e)
        excitation_energy = e[i] - e[1]
        println("Excitation energy $i: ", excitation_energy)
        push!(excitation_energy_i, excitation_energy)
    end


    # Save data for plotting
    push!(energies, e[1])
    push!(num_orbitals, norbs)

    # # Compute 1RDM (not used for plotting)
    # rdm1a, rdm1b = compute_1rdm(solution, root=4)

    println("e saved as: $i")
end

# Save results using JLD2
FileIO.save("../data/e", "e", energies)
FileIO.save("../data/ex1", "ex1", excitation_energy_i)

# Plot the energies vs. number of orbitals
plt1 = plot(num_orbitals, energies, xlabel="Number of Orbitals", ylabel="Energy", label="Energy vs. Num Orbitals", legend=:topleft)
plt2 = plot(num_orbitals[2:end], excitation_energy_i, xlabel="Number of Orbitals", ylabel="Excitation Energy", label="Excitation Energy vs. Num Orbitals", legend=:topleft)

plot(plt1, plt2, layout=(2,1))

savefig("../data/combined_plot.png")
