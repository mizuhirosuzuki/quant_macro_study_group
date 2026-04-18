struct Model{TI<:Integer, TF<:AbstractFloat}
    alpha::TF         # capital share
    delta::TF         # depreciation rate
    beta::TF          # discount factor
    rho::TF           # social-security replacement rate

    maxiter::TI       # max outer (capital) iterations
    tol::TF           # tolerance for outer iteration
    adjK::TF          # dampening for capital updates

    Nj::TI            # total ages (e.g. 20-80 -> 61)
    Njw::TI           # working ages (1..Njw); retire at Njw+1
    Na::TI            # size of asset state grid
    Na2::TI           # size of asset choice grid
    Ne::TI            # productivity grid size

    maxA::TF          # upper bound on assets
    curvA::TF         # grid curvature (1 = uniform)

    meaJ::Vector{TF}  # age distribution (sums to 1)
    L::TF             # aggregate labor supply
    gride::Vector{TF} # productivity grid (mean ≈ 1)
    Pe::Matrix{TF}    # productivity transition matrix

    b::TF             # ad-hoc borrowing limit (0 = none)
    NT::TI            # transition horizon
end
