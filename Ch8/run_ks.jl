using Random
using Printf
using Statistics

const SRC = joinpath(@__DIR__, "src")
include(joinpath(SRC, "markov.jl"))
include(joinpath(SRC, "model.jl"))
include(joinpath(SRC, "household.jl"))
include(joinpath(SRC, "simulation.jl"))
include(joinpath(SRC, "regression.jl"))


# Outer loop on the perceived forecast rule for log m'.
function solve_ks(mod::Model;
                  a0::Real = 0.095, a1::Real = 0.962,
                  b0::Real = 0.085, b1::Real = 0.965,
                  tol::Real = 1e-7, maxit::Integer = 50_001,
                  lambda::Real = 0.8,
                  seed::Integer = 1234,
                  verbose::Bool = true)
    v0  = ones(mod.Nk, mod.Ne, mod.Nz, mod.Nm)
    mu0 = ones(mod.Nk_dist, mod.Ne) ./ (mod.Nk_dist * mod.Ne)
    z_sim_index, _ = simulate_aggregate_shock(mod.zgrid, mod.pi_z, mod.T_sim;
                                              rng = MersenneTwister(seed))

    diff_params = 1.0
    iter = 0
    v = v0
    g = similar(v0)
    while diff_params > tol && iter < maxit
        iter += 1
        v, g, _  = solve_household(mod, a0, a1, b0, b1, v0)
        kbar_sim = simulate_capital_path(mod, g, z_sim_index, mu0)
        a0_hat, a1_hat, b0_hat, b1_hat = ols_forecast_rule(mod, kbar_sim, z_sim_index)

        diff_params = max(abs(a0 - a0_hat), abs(a1 - a1_hat),
                          abs(b0 - b0_hat), abs(b1 - b1_hat))

        if verbose
            @printf("%4d  a=[% .6f, % .6f]  b=[% .6f, % .6f]  diff=%.3e\n",
                    iter, a0, a1, b0, b1, diff_params)
            flush(stdout)
        end

        a0 = lambda * a0 + (1 - lambda) * a0_hat
        a1 = lambda * a1 + (1 - lambda) * a1_hat
        b0 = lambda * b0 + (1 - lambda) * b0_hat
        b1 = lambda * b1 + (1 - lambda) * b1_hat
        v0 = v
    end

    return (a0 = a0, a1 = a1, b0 = b0, b1 = b1,
            v = v, g = g,
            z_sim_index = z_sim_index, mu0 = mu0, iter = iter)
end

# Forecast vs simulated path under the converged rule (den Haan diagnostic).
function den_haan_test(mod::Model, kbar_sim::AbstractVector,
                       z_sim_index::AbstractVector,
                       a0::Real, a1::Real, b0::Real, b1::Real)
    T_post = mod.T_sim - mod.T0
    kbar_forecast = zeros(T_post)
    kbar_forecast[1] = kbar_sim[1]
    for t in 1:T_post - 1
        if z_sim_index[mod.T0 + t] == 1.0
            kbar_forecast[t + 1] = exp(b0 + b1 * log(kbar_forecast[t]))
        else
            kbar_forecast[t + 1] = exp(a0 + a1 * log(kbar_forecast[t]))
        end
    end
    errors = abs.((kbar_sim .- kbar_forecast) ./ kbar_forecast)
    return kbar_forecast, mean(errors) * 100, maximum(errors) * 100
end

function main()
    mod = Construct()
    println("=== Krusell & Smith (1998) ===")
    println("[iter, a0, a1, b0, b1, diff]")
    flush(stdout)

    @time res = solve_ks(mod)

    # Diagnostics at the converged forecast rule.
    kbar_sim = simulate_capital_path(mod, res.g, res.z_sim_index, res.mu0)
    a0h, a1h, b0h, b1h = ols_forecast_rule(mod, kbar_sim, res.z_sim_index)
    R2g, R2b, sig2g, sig2b = ols_fit_diagnostics(mod, kbar_sim, res.z_sim_index,
                                                 a0h, a1h, b0h, b1h)
    println()
    println("R^2  good = ", R2g)
    println("R^2  bad  = ", R2b)
    println("sigma good = ", sqrt(sig2g) * 100, " %")
    println("sigma bad  = ", sqrt(sig2b) * 100, " %")

    _, mean_err, max_err = den_haan_test(mod, kbar_sim, res.z_sim_index,
                                         a0h, a1h, b0h, b1h)
    println("den Haan mean error = ", mean_err, " %")
    println("den Haan max  error = ", max_err,  " %")

    return res
end

# Run when executed as a script: `julia run_ks.jl`.
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
