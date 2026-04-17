struct Model{TI<:Integer, TF<:AbstractFloat}

    mu::TF                # risk aversion (=3 baseline)
    beta::TF              # subjective discount factor
    delta::TF             # depreciation
    alpha::TF             # capital's share of income
    b::TF                 # borrowing limit
    Nl::TI                # number of discretized states
    s::Array{TF,1}        # (exponentialed) discretized states of log labor earnings
    prob::Array{TF,2}     # transition matrix of the Markov chain
    labor::TF             # aggregate labor supply

    Nk::TI                # grid size for capital (state variable)
    maxK::TF              # maximum value of capital grid
    curvK::TF             # a parameter controlling grid width

    Nk2::TI               # grid size for capital (control variable)

    NT::TI                # transition period

end

function create_model(;
    mu    = 3.0,
    beta  = 0.96,
    delta = 0.08,
    alpha = 0.36,
    b     = 3.0,
    Nl    = 7,
    rho   = 0.6,
    sig   = 0.4,
    M     = 2.0,
    Nk    = 300,
    maxK  = 30.0,
    curvK = 2.0,
    Nk2   = 800,
    NT    = 200,
)
    logs, prob, invdist = tauchen(Nl, rho, sig, M)
    s = exp.(logs)
    labor = s' * invdist

    return Model(
        mu, beta, delta, alpha, b,
        Nl, s, prob, labor,
        Nk, maxK, curvK, Nk2, NT,
    )
end

function calculate_K0(m::Model, r)
    return m.labor * (m.alpha / (r + m.delta))^(1 / (1 - m.alpha))
end

function calculate_w0(m::Model, r)
    return (1 - m.alpha) * ((m.alpha / (r + m.delta))^m.alpha)^(1 / (1 - m.alpha))
end
