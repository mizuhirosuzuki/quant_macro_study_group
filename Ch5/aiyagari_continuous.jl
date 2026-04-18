# ======================================================= #
# Model of Aiyagari (1994)                                #
# Continuous-control version (golden section search)      #
# ======================================================= #

# load source files
include("src/tauchen.jl")
include("src/model.jl")
include("src/grid.jl")
include("src/vfi.jl")
include("src/vfi_continuous.jl")
include("src/distribution.jl")
include("src/equilibrium.jl")
include("src/transition.jl")
include("src/transition_continuous.jl")

using Plots
using Statistics
using LaTeXStrings

# ===================== #
#  CREATE MODEL         #
# ===================== #

m = create_model(b=3.0, maxK=40.0)

# =============================== #
#  SOLVE STATIONARY EQUILIBRIUM   #
# =============================== #

res = solve_stationary_equilibrium(m; adj=0.0001, rate0=0.026, method=:continuous)

# plot density
y = vec(sum(res.mea0, dims=1))
window = 20
y_smooth = [mean(y[max(1, i - window ÷ 2):min(length(y), i + window ÷ 2)]) for i in 1:length(y)]
plot(res.grid1, y, label="density", linewidth=1)
plot!(res.grid1, y_smooth, label="smoothed density", linewidth=3)
savefig("images/fig_density_continuous.pdf")

# ============================== #
#  CAPITAL SUPPLY/DEMAND CURVES  #
# ============================== #

r0_grid = 0.026:0.001:0.030
K_grids = derive_capital_supply_curve(m, r0_grid; method=:continuous)

plot(K_grids.K1_grid, r0_grid, label="capital supply")
plot!(K_grids.K0_grid, r0_grid, label="capital demand")
xlims!(6.8, 8)
ylims!(0.026, 0.030)
savefig("images/fig_aiyagari_supply_demand_continuous.pdf")

# =============================== #
#  TRANSITION DYNAMICS            #
# =============================== #

m_transition = create_model(b=0.0, maxK=30.0)

tau = 0.1

# initial and final steady states
res_SS0 = solve_stationary_equilibrium(m_transition; adj=0.0002, rate0=0.023, method=:continuous)
res_SS1 = solve_stationary_equilibrium(m_transition; tau=tau, adj=0.0002, rate0=0.025, method=:continuous)

K_SS0 = res_SS0.K1
K_SS1 = res_SS1.K1
v_SS1 = res_SS1.v
mea_SS0 = res_SS0.mea0

# compute transition
KT_transition, KT0_iteration_history, meaT = derive_transition(
    m_transition, K_SS0, K_SS1, v_SS1, mea_SS0, tau; method=:continuous,
)

NT = m_transition.NT
rT_transition = m_transition.alpha .* ((KT_transition ./ m_transition.labor) .^ (m_transition.alpha .- 1)) .- m_transition.delta

plot(1:NT, KT_transition, label="capital transition", linewidth=3)
xlabel!("time")
ylabel!("capital")
savefig("images/fig_transition_K_continuous.pdf")

plot(1:NT, rT_transition, label="interest rate transition", linewidth=3)
xlabel!("time")
ylabel!("interest rate")
savefig("images/fig_transition_r_continuous.pdf")
