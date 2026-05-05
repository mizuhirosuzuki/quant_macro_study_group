using Interpolations

# E[V(k', e', z', m')] reduced over (e', z') given current (e, z), written
# into a preallocated (Nk × Nm) buffer `ev` (zeroed inside).
function expected_value!(ev::AbstractMatrix{Float64},
                          m::Model, ie::Integer, iz::Integer,
                          v::AbstractArray{<:Real,4})
    fill!(ev, 0.0)
    @inbounds for izz in 1:m.Nz
        pz    = m.pi_z[iz, izz]
        denom = m.pi_e[2 * (iz - 1) + ie, 2 * (izz - 1) + 1] +
                m.pi_e[2 * (iz - 1) + ie, 2 * (izz - 1) + 2]
        for iee in 1:m.Ne
            pee  = m.pi_e[2 * (iz - 1) + ie, 2 * (izz - 1) + iee] / denom
            coef = pz * pee
            for im in 1:m.Nm, ik in 1:m.Nk
                ev[ik, im] += coef * v[ik, iee, izz, im]
            end
        end
    end
    return ev
end

# Cubic interpolant of EV over (k', m'), evaluated allocation-free.
#
# The kgrid is constructed log-spaced as kgrid_i = exp((i-1)/(Nk-1) · u_max) +
# kmin - 1, so the change of variables u = log(k - kmin + 1) maps it to a
# uniform grid in u. We build a uniform-axis BSpline cubic in (u, m) and
# transform queries inline via the wrapper struct.
struct UScaledSpline{S}
    s::S            # Interpolations extrapolation in (u, m) space
    kmin::Float64
end
@inline (us::UScaledSpline)(k::Real, m::Real) = us.s(log(k - us.kmin + 1.0), m)

# Pre-build the (k', m') interpolant for each (ie, iz). The result depends
# only on those two indices, so doing it once here saves Nm× redundant builds
# inside the bellman_update! sweep.
function build_value_splines(mod::Model, v::AbstractArray{<:Real,4})
    u_max  = log(mod.kmax - mod.kmin + 1.0)
    u_axis = range(0.0, u_max, length = mod.Nk)
    m_axis = range(mod.mmin, mod.mmax, length = mod.Nm)

    # construct a prototype to capture the concrete extrapolation type
    ev_proto = zeros(mod.Nk, mod.Nm)
    expected_value!(ev_proto, mod, 1, 1, v)
    proto_extp = extrapolate(
        Interpolations.scale(interpolate(ev_proto, BSpline(Cubic(Line(OnGrid())))),
              u_axis, m_axis),
        Line(),
    )
    proto = UScaledSpline(proto_extp, float(mod.kmin))

    splines = Matrix{typeof(proto)}(undef, mod.Ne, mod.Nz)
    splines[1, 1] = proto
    for iz in 1:mod.Nz, ie in 1:mod.Ne
        (iz == 1 && ie == 1) && continue
        ev = zeros(mod.Nk, mod.Nm)
        expected_value!(ev, mod, ie, iz, v)
        extp = extrapolate(
            Interpolations.scale(interpolate(ev, BSpline(Cubic(Line(OnGrid())))),
                  u_axis, m_axis),
            Line(),
        )
        splines[ie, iz] = UScaledSpline(extp, float(mod.kmin))
    end
    return splines
end

# Allocation-free golden-section minimization of the negated Bellman RHS
#     f(k') = -log(income - k') - β V(k', m')
# over k' ∈ [a, b]. Returns the minimizer x* and the minimum value f(x*).
#
# The bracket size shrinks by 1/φ ≈ 0.618 per iteration; the iteration count
# is set so the final bracket width is below `tol`. We never evaluate at the
# endpoints, so income - k' > 0 stays satisfied as long as b ≤ income (the
# caller passes b = min(income, kmax)).
@inline function gss_bellman(mprime::Float64, income::Float64,
                              v_spline, beta::Float64,
                              a::Float64, b::Float64;
                              tol::Float64 = 1e-8)
    invphi  = 0.6180339887498949   # 1/φ
    invphi2 = 0.3819660112501051   # 1/φ²

    h = b - a
    if h <= tol
        x = 0.5 * (a + b)
        return x, -log(income - x) - beta * v_spline(x, mprime)
    end

    n = ceil(Int, log(tol / h) / log(invphi))

    c  = a + invphi2 * h
    d  = a + invphi  * h
    yc = -log(income - c) - beta * v_spline(c, mprime)
    yd = -log(income - d) - beta * v_spline(d, mprime)

    for _ in 1:n - 1
        if yc < yd
            b  = d
            d  = c
            yd = yc
            h  = invphi * h
            c  = a + invphi2 * h
            yc = -log(income - c) - beta * v_spline(c, mprime)
        else
            a  = c
            c  = d
            yc = yd
            h  = invphi * h
            d  = a + invphi * h
            yd = -log(income - d) - beta * v_spline(d, mprime)
        end
    end

    return yc < yd ? (c, yc) : (d, yd)
end

# One Bellman update T·v over the full state grid (k, e, z, m). Splines are
# built outside (once per (ie, iz)) and shared across the threaded sweep over
# `im`. `m_update[iz, :] = [b0 b1]` (iz=1, bad) / `[a0 a1]` (iz=2, good).
function bellman_update!(Tv::AbstractArray{Float64,4}, g::AbstractArray{Float64,4},
                          c::AbstractArray{Float64,4},
                          mod::Model, v::AbstractArray{Float64,4},
                          m_update::AbstractMatrix{<:Real})
    splines = build_value_splines(mod, v)

    Threads.@threads for im in 1:mod.Nm
        mnow = mod.mgrid[im]
        @inbounds for iz in 1:mod.Nz
            r, w = factor_prices(mod, iz, mnow)
            mprime = exp(m_update[iz, 1] + m_update[iz, 2] * log(mnow))
            for ie in 1:mod.Ne
                v_spline = splines[ie, iz]
                enow = mod.egrid[ie]
                for ik in 1:mod.Nk
                    know   = mod.kgrid[ik]
                    income = (1 + r) * know + w * enow * mod.lbar
                    ub     = min(income, mod.kmax)

                    kp_star, neg_v = gss_bellman(mprime, income, v_spline,
                                                  mod.beta, mod.kmin, ub)
                    g[ik, ie, iz, im]  = kp_star
                    Tv[ik, ie, iz, im] = -neg_v
                    c[ik, ie, iz, im]  = income - kp_star
                end
            end
        end
    end
    return nothing
end

# Iterate the Bellman operator until convergence given a candidate forecast
# rule (a0, a1, b0, b1). Returns (V, k'(k,e,z,m), c(k,e,z,m)).
function solve_household(mod::Model, a0::Real, a1::Real, b0::Real, b1::Real,
                          v0::AbstractArray{<:Real,4};
                          tol::Real = 1e-7, maxit::Integer = 5000)
    v  = copy(v0)
    Tv = similar(v)
    g  = similar(v)
    c  = similar(v)
    m_update = [b0 b1; a0 a1]   # row 1 = bad, row 2 = good

    diff = 1.0
    iter = 0
    while diff > tol && iter < maxit
        bellman_update!(Tv, g, c, mod, v, m_update)
        diff = maximum(abs.(Tv .- v))
        iter += 1
        v .= Tv
    end
    return v, g, c
end
