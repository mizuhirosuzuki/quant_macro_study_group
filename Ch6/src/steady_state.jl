# Solve the stationary equilibrium by damped fixed-point iteration on K.
# Each outer step:  (1) update prices from K, (2) solve the life-cycle
# household problem, (3) roll the distribution forward, (4) compare A(K) to K.
function solve_value_function(m::Model, grids, capital_grid_translations;
                               K0::Real = 7.0, verbose::Bool = true)

    tau = m.rho * sum(m.meaJ[m.Njw+1:m.Nj]) / sum(m.meaJ[1:m.Njw])

    vfun  = zeros(m.Nj, m.Ne, m.Na)
    afunG = zeros(Int64, m.Nj, m.Ne, m.Na)
    afun  = zeros(m.Nj, m.Ne, m.Na)

    K = float(K0); r = 0.0; w = 0.0
    mea = zeros(m.Nj, m.Ne, m.Na)

    for iter in 1:m.maxiter
        r = m.alpha * (K / m.L)^(m.alpha - 1) - m.delta
        w = (1 - m.alpha) * (K / m.L)^m.alpha

        yvec = income_grid(m, w, tau, m.rho)
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

        A = sum(afun .* mea)
        errK = abs(K - A)
        if errK < m.tol
            verbose && println("K converged at iter $iter")
            break
        end
        K += m.adjK * (A - K)
        verbose && println("$iter errK = $errK")
    end

    return (; vfun, afun, afunG, mea, r, w, K)
end
