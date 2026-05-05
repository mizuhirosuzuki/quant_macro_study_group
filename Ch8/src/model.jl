# Container for primitives, grids, and shock processes used across the
# Krusell & Smith (1998) algorithm.
struct Model{TI<:Integer, TF<:AbstractFloat}
    # primitives
    alpha::TF        # capital share
    delta::TF        # depreciation
    beta::TF         # discount factor
    lbar::TF         # labor supply per employed agent

    # idiosyncratic / aggregate shock grids
    Ne::TI
    egrid::Vector{TF}     # idiosyncratic employment status
    Nz::TI
    zgrid::Vector{TF}     # aggregate productivity {z_b, z_g}
    Nu::TI
    ugrid::Vector{TF}     # state-contingent unemployment rate {u_b, u_g}
    pi_z::Matrix{TF}      # P(z' | z)
    pi_e::Matrix{TF}      # P((z', e') | (z, e))

    # aggregate-capital grid (forecast state)
    Nm::TI
    mmin::TF
    mmax::TF
    mgrid::Vector{TF}

    # individual-capital grid (decision and distribution)
    Nk::TI
    Nk_dist::TI
    kmin::TF
    kmax::TF
    kgrid::Vector{TF}      # log-spaced grid for VFI
    kgrid_dist::Vector{TF} # uniform grid for the Young-style histogram

    # simulation horizon
    T_sim::TI
    T0::TI                  # burn-in length
end

# Build a Model with KS-style defaults. All parameters are keyword args so a
# caller can override any single value (e.g. Construct(beta = 0.995)).
function Construct(;
    alpha::Real = 0.36,
    delta::Real = 0.025,
    beta::Real  = 0.99,
    lbar::Real  = 0.3271,

    Ne::Integer = 2,
    egrid       = [0.0001, 1.0],
    Nz::Integer = 2,
    zgrid       = [0.99, 1.01],
    Nu::Integer = 2,
    ugrid       = [0.1, 0.04],

    Nm::Integer       = 5,
    mmin::Real        = 9.0,
    mmax::Real        = 14.0,
    Nk::Integer       = 35,
    Nk_dist::Integer  = 1000,
    kmin::Real        = 0.0,
    kmax::Real        = 75.0,

    T_sim::Integer = 6000,
    T0::Integer    = 1000,
)
    pi_z, pi_e = ks_markov(ugrid, Ne, Nz)

    mgrid      = collect(range(mmin, mmax, length = Nm))
    # log-spaced grid concentrates points near the borrowing constraint
    kgrid      = exp.(range(0, log(kmax - kmin + 1), length = Nk)) .+ (-1 + kmin)
    kgrid_dist = collect(range(kmin, kmax, length = Nk_dist))

    return Model(
        float(alpha), float(delta), float(beta), float(lbar),
        Ne, Vector{Float64}(egrid), Nz, Vector{Float64}(zgrid),
        Nu, Vector{Float64}(ugrid), pi_z, pi_e,
        Nm, float(mmin), float(mmax), mgrid,
        Nk, Nk_dist, float(kmin), float(kmax), kgrid, kgrid_dist,
        T_sim, T0,
    )
end

# Factor prices given aggregate state (zgrid index iz) and aggregate capital m.
function factor_prices(m::Model, iz::Integer, mbar::Real)
    z = m.zgrid[iz]
    L = (1 - m.ugrid[iz]) * m.lbar
    r = m.alpha       * z * mbar^(m.alpha - 1) * L^(1 - m.alpha) - m.delta
    w = (1 - m.alpha) * z * mbar^m.alpha       * L^(-m.alpha)
    return r, w
end
