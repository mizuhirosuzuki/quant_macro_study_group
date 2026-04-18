using Plots

const SRC = joinpath(@__DIR__, "src")
include(joinpath(SRC, "model.jl"))
include(joinpath(SRC, "grids.jl"))
include(joinpath(SRC, "household.jl"))
include(joinpath(SRC, "distribution.jl"))
include(joinpath(SRC, "steady_state.jl"))
include(joinpath(SRC, "transition.jl"))
include(joinpath(SRC, "stats.jl"))
include(joinpath(SRC, "transition_plots.jl"))
include(joinpath(SRC, "welfare.jl"))

# ================ #
#  SET PARAMETERS  #
# ================ #
const ALPHA   = 0.40
const DELTA   = 0.08
const BETA    = 0.98

const MAXITER = 2000
const TOL     = 1e-3
const ADJK    = 0.2

const NJ      = 61
const NJW     = 45
const NA      = 201
const NA2     = 8001
const NE      = 2

const MAXA    = 25.0
const CURVA   = 1.2

const NT      = 100

const GRIDE   = [1.0 - 0.3, 1.0 + 0.3]
const PE      = [0.8  0.2;
                 0.2  0.8]
const MEAJ    = fill(1.0 / NJ, NJ)
const L_AGG   = sum(MEAJ[1:NJW])

const IMGDIR  = joinpath(@__DIR__, "images")

make_model(rho::Real) = Model(
    ALPHA, DELTA, BETA, float(rho),
    MAXITER, TOL, ADJK,
    NJ, NJW, NA, NA2, NE,
    MAXA, CURVA,
    MEAJ, L_AGG, GRIDE, PE,
    0.0, NT,
)

# ======================== #
#  INITIAL STEADY STATE    #
# ======================== #
const RHO0 = 0.50
const RHO1 = 0.25

m_ss0 = make_model(RHO0)
m_ss1 = make_model(RHO1)

grids                     = generate_capital_grid(m_ss0)
capital_grid_translations = translate_capital_grid(m_ss0, grids)

println("Solving initial steady state (rho=$(RHO0))...")
res_ss0 = solve_value_function(m_ss0, grids, capital_grid_translations)

println("Solving final steady state   (rho=$(RHO1))...")
res_ss1 = solve_value_function(m_ss1, grids, capital_grid_translations)

K_SS0, mea_SS0, vfun_SS0 = res_ss0.K, res_ss0.mea, res_ss0.vfun
K_SS1, mea_SS1, vfun_SS1 = res_ss1.K, res_ss1.mea, res_ss1.vfun

# Life-cycle stats + plots for the initial SS
stats_ss0 = compute_lifecycle_stats(m_ss0, grids, res_ss0)
plot_lifecycle(m_ss0, grids, stats_ss0; outdir=IMGDIR)

# ==================== #
#  TRANSITION          #
# ==================== #
println("Computing transition (rho $(RHO0) -> $(RHO1))...")
res_transition = compute_transition(
    m_ss0, grids, capital_grid_translations,
    K_SS0, K_SS1, RHO0, RHO1,
    vfun_SS1, mea_SS0,
)

# ======================== #
#  TRANSITION DYNAMICS     #
# ======================== #
rT, wT = compute_price_paths(m_ss0, res_transition.KT0)

plot_transition_capital(m_ss0, res_transition.KT0;
                        K_SS0 = K_SS0, K_SS1 = K_SS1, outdir = IMGDIR)
plot_transition_interest(m_ss0, rT; outdir = IMGDIR)

# ======================== #
#  WELFARE (CEV)           #
# ======================== #
betaJ    = life_cycle_discount(m_ss0)
welf0    = compute_welfare_existing(m_ss0, res_transition.vfun_TR0, vfun_SS0, betaJ)
welf0_JE = average_welfare_existing(m_ss0, welf0, mea_SS0)
welfTR   = compute_welfare_newborn(m_ss0, res_transition.vfun_TR, vfun_SS0, betaJ)

plot_welfare_newborn_cohorts(m_ss0, welfTR; outdir = IMGDIR)
plot_welfare_by_age(m_ss0, welf0_JE; outdir = IMGDIR)
plot_welfare_by_asset(m_ss0, grids, welf0; jc = 21, outdir = IMGDIR)

println("Done. K_SS0=$(K_SS0)  K_SS1=$(K_SS1)  final KT[NT]=$(res_transition.KT0[end])")
