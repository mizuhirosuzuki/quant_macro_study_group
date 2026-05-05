using Distributions
using Random

# Build the joint Markov transition matrices for Krusell & Smith (1998):
#   pi_z[iz, izz]                      — aggregate state z -> z'
#   pi_e[2(iz-1)+ie, 2(izz-1)+iee]     — joint (z, e) -> (z', e')
#
# Calibration follows KS (1998): expected duration of 8 quarters in each
# aggregate state, and idiosyncratic transitions tied to the unemployment
# rates ugrid = [u_b, u_g] in the bad / good states.
function ks_markov(ugrid::AbstractVector, Ne::Integer, Nz::Integer)
    pi_e = zeros(Ne * Nz, Ne * Nz)
    pi_z = zeros(Nz, Nz)

    # (1) aggregate transition pi_z[iz, izz]
    Az = [8 0 0 0; 0 8 0 0; 1 0 0 1; 0 1 1 0]
    bz = [7; 7; 1; 1]
    xz = Az^-1 * bz

    pi_z[1, 1] = xz[1]
    pi_z[2, 2] = xz[2]
    pi_z[1, 2] = xz[3]
    pi_z[2, 1] = xz[4]

    # (2) joint transition pi_e[(z,e), (z',e')]
    Ap = zeros((Ne * Nz)^2, (Ne * Nz)^2)

    Ap[1, 11] = 1.5
    Ap[2, 1]  = 2.5
    Ap[3, 1]  = -1.25 * xz[4]
    Ap[3, 9]  = xz[2]
    Ap[4, 3]  = xz[1]
    Ap[4, 11] = -0.75 * xz[3]

    Ap[5, 11]  = 1.0; Ap[5, 12]  = 1.0
    Ap[6, 15]  = 1.0; Ap[6, 16]  = 1.0
    Ap[7, 9]   = 1.0; Ap[7, 10]  = 1.0
    Ap[8, 13]  = 1.0; Ap[8, 14]  = 1.0
    Ap[9, 3]   = 1.0; Ap[9, 4]   = 1.0
    Ap[10, 7]  = 1.0; Ap[10, 8]  = 1.0
    Ap[11, 1]  = 1.0; Ap[11, 2]  = 1.0
    Ap[12, 5]  = 1.0; Ap[12, 6]  = 1.0

    Ap[13, 11] = ugrid[2]; Ap[13, 15] = 1 - ugrid[2]
    Ap[14, 9]  = ugrid[2]; Ap[14, 13] = 1 - ugrid[2]
    Ap[15, 3]  = ugrid[1]; Ap[15, 7]  = 1 - ugrid[1]
    Ap[16, 1]  = ugrid[1]; Ap[16, 5]  = 1 - ugrid[1]

    bp = [0.5 * xz[1]; 1.5 * xz[2]; 0.0; 0.0;
          xz[1]; xz[1]; xz[4]; xz[4];
          xz[3]; xz[3]; xz[2]; xz[2];
          ugrid[2] * xz[1]; ugrid[1] * xz[4];
          ugrid[2] * xz[3]; ugrid[1] * xz[2]]
    xp = Ap^-1 * bp

    for i in 1:Ne * Nz, j in 1:Ne * Nz
        pi_e[i, j] = xp[(Nz * Ne) * (i - 1) + j]
    end

    return pi_z, pi_e
end

# Simulate a path of length T_sim for the aggregate state index (1 = bad,
# 2 = good). The chain starts in the bad state.
function simulate_aggregate_shock(zgrid::AbstractVector, pi_z::AbstractMatrix,
                                  T_sim::Integer; rng = Random.default_rng())
    e = rand(rng, Uniform(0, 1), T_sim)
    z_sim_index = zeros(T_sim)
    z_sim       = zeros(T_sim)
    z_sim_index[1] = 2
    z_sim[1]       = zgrid[Int(z_sim_index[1])]

    pi_z_hat = cumsum(pi_z, dims = 2)

    for t in 1:T_sim - 1
        row = pi_z_hat[Int(z_sim_index[t]), :]
        z_sim_index[t + 1] = sum(e[t] .- row .>= 0)
        z_sim_index[t + 1] = min(Int(z_sim_index[t + 1] + 1), 2)
        z_sim[t + 1] = zgrid[Int(z_sim_index[t + 1])]
    end

    return z_sim_index, z_sim
end
