# Accidental bequest per alive agent in steady state:
#   q = (1+r) · Σ_{j,e,a} (1-s_j) · afun(j,e,a) · mea(j,e,a) / Σ mea.
function bequest_per_capita(m::Model, r, afun, mea)
    num = 0.0
    for jc in 1:m.Nj, ec in 1:m.Ne, ac in 1:m.Na
        num += (1 - m.s[jc]) * afun[jc, ec, ac] * mea[jc, ec, ac]
    end
    return (1 + r) * num / sum(mea)
end

# Solve the stationary equilibrium by damped fixed-point iteration on (K, q).
# Each outer step:  (1) update prices from K, (2) solve the life-cycle
# household problem with cash-on-hand augmented by q, (3) roll the distribution
# forward, (4) compare A(K, q) to K and bequest-per-capita to q.
function solve_value_function(m::Model, grids, capital_grid_translations;
                               K0::Real = 7.0,
                               q0::Real = 0.0,
                               adjq::Real = 0.2,
                               verbose::Bool = true)

    tau = m.rho * sum(m.meaJ[m.Njw+1:m.Nj]) / sum(m.meaJ[1:m.Njw])

    vfun  = zeros(m.Nj, m.Ne, m.Na)
    afunG = zeros(Int64, m.Nj, m.Ne, m.Na)
    afun  = zeros(m.Nj, m.Ne, m.Na)

    K = float(K0); q = float(q0); r = 0.0; w = 0.0
    mea = zeros(m.Nj, m.Ne, m.Na)

    for iter in 1:m.maxiter
        r = m.alpha * (K / m.L)^(m.alpha - 1) - m.delta
        w = (1 - m.alpha) * (K / m.L)^m.alpha

        yvec = income_grid(m, w, tau, m.rho; q = q)
        solve_household_period!(vfun, afunG, afun,
                                 m, grids, capital_grid_translations,
                                 vfun, r, yvec)

        mea = stationary_distribution(m, afunG, capital_grid_translations)

        errm = abs(sum(mea) - 1)
        if errm > 1e-4
            verbose && println("distribution mass drift: $errm")
            break
        end
        mea_maxA = sum(@view mea[:, :, m.Na])
        if mea_maxA > 1e-3 && verbose
            println("mass at top asset grid is large: $mea_maxA")
        end

        A     = sum(afun .* mea)
        q_new = bequest_per_capita(m, r, afun, mea)

        errK = abs(K - A)
        errq = abs(q - q_new)
        if errK < m.tol && errq < m.tol
            verbose && println("converged at iter $iter: K=$K q=$q")
            break
        end
        K += m.adjK * (A - K)
        q += adjq * (q_new - q)
        verbose && println("$iter errK=$errK errq=$errq  K=$K q=$q")
    end

    return (; vfun, afun, afunG, mea, r, w, K, q)
end
