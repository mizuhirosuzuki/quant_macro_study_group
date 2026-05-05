using Interpolations

# Aggregate capital implied by a (Nk_dist × Ne) cross-sectional distribution.
@inline function aggregate_capital(kgrid_dist::AbstractVector, mu::AbstractMatrix)
    total = 0.0
    @inbounds for ik in 1:size(mu, 1)
        s = 0.0
        for ie in 1:size(mu, 2)
            s += mu[ik, ie]
        end
        total += kgrid_dist[ik] * s
    end
    return total
end

# Build a cubic interpolant of a length-Nk policy slice over the kgrid.
# We exploit the log-spaced grid construction (kgrid_i = exp((i-1)/(Nk-1) ·
# u_max) + kmin - 1) to fit on the uniform u-axis u = log(k − kmin + 1), then
# query at the pre-transformed `u_kgrid_dist`. Returns the same vector shape
# as the original `Spline1D(kgrid, .., k=3).(kgrid_dist)`.
@inline function _interp_policy_to_dist!(out::AbstractVector,
                                          values::AbstractVector,
                                          u_axis::AbstractRange,
                                          u_kgrid_dist::AbstractVector)
    extp = extrapolate(
        Interpolations.scale(
            interpolate(values, BSpline(Cubic(Line(OnGrid())))),
            u_axis,
        ),
        Line(),
    )
    @inbounds for ik in eachindex(u_kgrid_dist)
        out[ik] = extp(u_kgrid_dist[ik])
    end
    return nothing
end

# Simulate the equilibrium aggregate-capital path given a household policy
# k'(k, e, z, m) and a draw of aggregate-shock indices `z_sim_index`.
#
# Hot-loop optimizations:
#   - Linear interpolation in the m-direction is done inline (one bracket per
#     time step, no Spline1D allocations).
#   - Cubic interpolation in the k-direction uses Interpolations.jl on the
#     transformed uniform u-axis — allocation-free queries.
#   - Cross-sectional distributions are kept in two flip-flop buffers
#     (`mu_curr`, `mu_next`) instead of a (T_sim × Nk_dist × Ne) history.
function simulate_capital_path(mod::Model, g::AbstractArray{Float64,4},
                               z_sim_index::AbstractVector,
                               mu0::AbstractMatrix{<:Real})
    kbar_sim   = zeros(mod.T_sim)
    mu_curr    = Matrix{Float64}(copy(mu0))
    mu_next    = zeros(mod.Nk_dist, mod.Ne)
    g_dist_pre = zeros(mod.Nk, mod.Ne)
    g_dist     = zeros(mod.Nk_dist, mod.Ne)

    u_axis       = range(0.0, log(mod.kmax - mod.kmin + 1.0), length = mod.Nk)
    u_kgrid_dist = [log(k - mod.kmin + 1.0) for k in mod.kgrid_dist]

    kbar_sim[1] = aggregate_capital(mod.kgrid_dist, mu_curr)

    for t in 1:mod.T_sim - 1
        iz  = Int(z_sim_index[t])
        izz = Int(z_sim_index[t + 1])

        # (a) linear interpolation in the m-direction at m = kbar_sim[t].
        mq = kbar_sim[t]
        jm = clamp(searchsortedlast(mod.mgrid, mq), 1, mod.Nm - 1)
        wm = (mq - mod.mgrid[jm]) / (mod.mgrid[jm + 1] - mod.mgrid[jm])
        @inbounds for ie in 1:mod.Ne, ik in 1:mod.Nk
            g_dist_pre[ik, ie] =
                (1 - wm) * g[ik, ie, iz, jm] + wm * g[ik, ie, iz, jm + 1]
        end

        # (b) cubic interpolation in the k-direction onto the fine histogram
        #     grid `kgrid_dist`, evaluated allocation-free in u-space.
        for ie in 1:mod.Ne
            _interp_policy_to_dist!(view(g_dist, :, ie),
                                     view(g_dist_pre, :, ie),
                                     u_axis, u_kgrid_dist)
        end

        # (c) Young-style histogram update with conditional e-transition
        #     P(e' | e, z, z') derived from `pi_e`.
        fill!(mu_next, 0.0)
        @inbounds for ik in 1:mod.Nk_dist, ie in 1:mod.Ne
            kp  = g_dist[ik, ie]
            idx = max(searchsortedlast(mod.kgrid_dist, kp), 1)
            wj  = (mod.kgrid_dist[idx + 1] - kp) /
                  (mod.kgrid_dist[idx + 1] - mod.kgrid_dist[idx])

            denom = mod.pi_e[2 * (iz - 1) + ie, 2 * (izz - 1) + 1] +
                    mod.pi_e[2 * (iz - 1) + ie, 2 * (izz - 1) + 2]
            mass_curr = mu_curr[ik, ie]
            for iee in 1:mod.Ne
                pee  = mod.pi_e[2 * (iz - 1) + ie, 2 * (izz - 1) + iee] / denom
                mass = pee * mass_curr
                mu_next[idx,     iee] += wj       * mass
                mu_next[idx + 1, iee] += (1 - wj) * mass
            end
        end

        kbar_sim[t + 1] = aggregate_capital(mod.kgrid_dist, mu_next)
        mu_curr, mu_next = mu_next, mu_curr   # flip
    end

    return kbar_sim[mod.T0 + 1:end]
end

# Approximate stationary distribution under the assumption that the aggregate
# state is fixed at the good state. Used as an optional warm start; the basic
# algorithm uses a uniform mu0.
function stationary_distribution(g::AbstractArray{Float64,4}, mod::Model,
                                  mu0::AbstractMatrix{<:Real};
                                  tol::Real = 1e-8, maxit::Integer = 5000)
    g_dist       = zeros(mod.Nk_dist, mod.Ne)
    u_axis       = range(0.0, log(mod.kmax - mod.kmin + 1.0), length = mod.Nk)
    u_kgrid_dist = [log(k - mod.kmin + 1.0) for k in mod.kgrid_dist]
    @inbounds for ie in 1:mod.Ne
        _interp_policy_to_dist!(view(g_dist, :, ie),
                                 view(g, :, ie, 2, lastindex(g, 4)),
                                 u_axis, u_kgrid_dist)
    end

    # P(e' | e) conditional on (z, z') = (good, good)
    P = mod.pi_e[3:4, 3:4] ./ sum(mod.pi_e[3:4, 3:4], dims = 2)

    mu       = Matrix{Float64}(copy(mu0))
    Tmu      = zeros(mod.Nk_dist, mod.Ne)
    distance = Inf
    iter     = 0
    while iter < maxit && distance > tol
        fill!(Tmu, 0.0)
        @inbounds for ik in 1:mod.Nk_dist, ie in 1:mod.Ne
            kp  = g_dist[ik, ie]
            idx = max(searchsortedlast(mod.kgrid_dist, kp), 1)
            wj  = (mod.kgrid_dist[idx + 1] - kp) /
                  (mod.kgrid_dist[idx + 1] - mod.kgrid_dist[idx])
            for iee in 1:mod.Ne
                Tmu[idx,     iee] += wj       * P[ie, iee] * mu[ik, ie]
                Tmu[idx + 1, iee] += (1 - wj) * P[ie, iee] * mu[ik, ie]
            end
        end
        distance = maximum(abs.(Tmu .- mu))
        iter += 1
        mu, Tmu = Tmu, mu
    end
    if distance > tol
        error("stationary distribution did not converge in $maxit iterations")
    end
    return mu
end
