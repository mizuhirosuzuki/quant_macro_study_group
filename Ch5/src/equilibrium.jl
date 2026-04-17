function _solve_VFI_dispatch(method, m, r0, K0, w0, grids, capital_grid_translations; tau=0)
    if method == :grid_search
        return solve_VFI(m, r0, K0, w0, grids, capital_grid_translations; tau=tau)
    elseif method == :continuous
        return solve_VFI_continuous(m, r0, K0, w0, grids; tau=tau)
    else
        error("Unknown VFI method: $method. Use :grid_search or :continuous.")
    end
end

# Given a candidate interest rate, solve the household problem and the
# stationary distribution, then return capital demand (K0), supply (K1)
# and the household-side artifacts.
function compute_capital_at_rate(m::Model, rate0; tau=0, method=:grid_search)
    K0 = calculate_K0(m, rate0)
    w0 = calculate_w0(m, rate0)

    grids = generate_capital_grid(m, rate0, w0)
    grid1 = grids[1]
    capital_grid_translations = translate_capital_grid(m, grids)

    kfun, kfunG, v = _solve_VFI_dispatch(
        method, m, rate0, K0, w0, grids, capital_grid_translations; tau=tau,
    )

    if method == :continuous
        # Use continuous policy directly (no gridk2 snapping)
        mea0 = derive_stationary_distribution_continuous(m, kfun, grid1)
    else
        mea0 = derive_stationary_distribution(m, kfunG, capital_grid_translations)
    end
    K1 = calculate_capital_supply(mea0, kfun)

    return (; K0, K1, w0, mea0, grid1, v)
end

function solve_stationary_equilibrium(m::Model; tau=0, adj=0.001, rate0=0.02, method=:grid_search)

    res = nothing
    while true
        res = compute_capital_at_rate(m, rate0; tau=tau, method=method)

        println(
            "r0: ", round(rate0, digits=4),
            ", K0: ", round(res.K0, digits=3),
            ", K1: ", round(res.K1, digits=3),
        )

        # stop if capital demand < capital supply
        res.K0 < res.K1 && break

        rate0 += adj
    end

    return (; rate0, K1=res.K1, w0=res.w0, mea0=res.mea0, grid1=res.grid1, v=res.v)
end

# Solve for the equilibrium interest rate using bisection on the excess
# capital demand function f(r) = K_demand(r) - K_supply(r). The bracket
# [r_low, r_high] must have f(r_low) > 0 and f(r_high) < 0.
function solve_stationary_equilibrium_root(
    m::Model;
    tau=0,
    r_low=0.01,
    r_high=0.04,
    tol=1e-5,
    max_iter=50,
    method=:grid_search,
)
    res_low = compute_capital_at_rate(m, r_low; tau=tau, method=method)
    res_high = compute_capital_at_rate(m, r_high; tau=tau, method=method)

    excess_low = res_low.K0 - res_low.K1
    excess_high = res_high.K0 - res_high.K1

    println("bracket check: excess(r_low=$r_low)=$(round(excess_low, digits=4)), excess(r_high=$r_high)=$(round(excess_high, digits=4))")

    if excess_low * excess_high > 0
        error(
            "Bracket [$r_low, $r_high] does not contain a root. " *
            "excess_low=$excess_low, excess_high=$excess_high. " *
            "Widen the bracket so demand > supply at r_low and supply > demand at r_high."
        )
    end

    rate_mid = (r_low + r_high) / 2
    res = res_low

    for iter in 1:max_iter
        rate_mid = (r_low + r_high) / 2
        res = compute_capital_at_rate(m, rate_mid; tau=tau, method=method)
        excess = res.K0 - res.K1

        println(
            "iter ", iter,
            ": r=", round(rate_mid, digits=5),
            ", K0=", round(res.K0, digits=3),
            ", K1=", round(res.K1, digits=3),
            ", excess=", round(excess, digits=5),
        )

        if abs(r_high - r_low) < tol
            break
        end

        if excess > 0
            # demand exceeds supply → equilibrium r is higher
            r_low = rate_mid
        else
            # supply exceeds demand → equilibrium r is lower
            r_high = rate_mid
        end
    end

    return (; rate0=rate_mid, K1=res.K1, w0=res.w0, mea0=res.mea0, grid1=res.grid1, v=res.v)
end

function derive_capital_supply_curve(m::Model, r0_grid; tau=0, method=:grid_search)

    K0_grid = zeros(length(r0_grid))
    K1_grid = zeros(length(r0_grid))

    for (i, rate0) in enumerate(r0_grid)
        res = compute_capital_at_rate(m, rate0; tau=tau, method=method)
        K0_grid[i] = res.K0
        K1_grid[i] = res.K1
    end

    return (; K0_grid, K1_grid)
end
