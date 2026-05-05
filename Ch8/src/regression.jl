using LinearAlgebra

# Stack regressors / regressands for the two aggregate states. The simulated
# series start at t = T0 + 1 (the burn-in is already dropped in
# `simulate_capital_path`), but the original aggregate-shock vector covers all
# T_sim periods, so we slice it the same way before splitting by state.
function _split_by_state(mod::Model, kbar_sim::AbstractVector,
                         z_sim_index::AbstractVector)
    z_post = z_sim_index[mod.T0 + 1:end - 1]
    ind_b  = findall(z_post .== 1.0)
    ind_g  = findall(z_post .== 2.0)

    Xb = [ones(length(ind_b)) log.(kbar_sim[ind_b])]
    Xg = [ones(length(ind_g)) log.(kbar_sim[ind_g])]
    Yb = log.(kbar_sim[ind_b .+ 1])
    Yg = log.(kbar_sim[ind_g .+ 1])
    return Xb, Yb, Xg, Yg
end

# Estimate the log-linear forecast rule
#     log m'_{t+1} = a0 + a1 log m_t   in the good state,
#     log m'_{t+1} = b0 + b1 log m_t   in the bad state,
# from a simulated path of aggregate capital.
function ols_forecast_rule(mod::Model, kbar_sim::AbstractVector,
                           z_sim_index::AbstractVector)
    Xb, Yb, Xg, Yg = _split_by_state(mod, kbar_sim, z_sim_index)
    a0, a1 = (Xg' * Xg) \ (Xg' * Yg)
    b0, b1 = (Xb' * Xb) \ (Xb' * Yb)
    return a0, a1, b0, b1
end

# R² and residual variance of the forecast rule, by aggregate state.
function ols_fit_diagnostics(mod::Model, kbar_sim::AbstractVector,
                             z_sim_index::AbstractVector,
                             a0::Real, a1::Real, b0::Real, b1::Real)
    Xb, Yb, Xg, Yg = _split_by_state(mod, kbar_sim, z_sim_index)

    a = [a0; a1]
    b = [b0; b1]

    eb = Yb .- Xb * b
    eg = Yg .- Xg * a
    sig2_bad  = (eb' * eb) / length(Yb)
    sig2_good = (eg' * eg) / length(Yg)

    Mb = I(length(Yb)) - ones(length(Yb), length(Yb)) ./ length(Yb)
    Mg = I(length(Yg)) - ones(length(Yg), length(Yg)) ./ length(Yg)

    R2_bad  = (b' * Xb' * Mb * Xb * b) / (Yb' * Mb * Yb)
    R2_good = (a' * Xg' * Mg * Xg * a) / (Yg' * Mg * Yg)

    return R2_good, R2_bad, sig2_good, sig2_bad
end
