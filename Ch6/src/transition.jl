# Factor prices implied by the aggregate production function at each date.
function compute_price_paths(m::Model, KT)
    rT = zeros(m.NT)
    wT = zeros(m.NT)
    for tc in 1:m.NT
        rT[tc] = m.alpha * (KT[tc] / m.L)^(m.alpha - 1) - m.delta
        wT[tc] = (1 - m.alpha) * (KT[tc] / m.L)^m.alpha
    end
    return rT, wT
end

# Bequest per alive agent along the transition. The bequest realised at tc
# comes from agents who chose a' at tc-1 and die between tc-1 and tc. Their
# wealth (1+r_tc)·a' enters the redistribution pool. qT[1] is pinned to the
# initial-SS value (the reform-date bequest is predetermined).
function compute_bequest_path(m::Model, rT, afunT, meaT, q_SS0)
    qT = zeros(m.NT)
    qT[1] = q_SS0
    for tc in 2:m.NT
        num = 0.0
        for jc in 1:m.Nj, ec in 1:m.Ne, ac in 1:m.Na
            num += (1 - m.s[jc]) * afunT[tc-1, jc, ec, ac] *
                   meaT[tc-1, jc, ec, ac]
        end
        qT[tc] = (1 + rT[tc]) * num / sum(@view meaT[tc, :, :, :])
    end
    return qT
end

# Path of replacement rate (linear phase-in over TT periods) and the
# corresponding balanced-budget tax rate.
function build_policy_paths(m::Model, rho_init, rho_final; TT=25)
    rhoT = zeros(m.NT)
    for tc in 1:TT
        rhoT[tc] = rho_init + (rho_final - rho_init) * (tc - 1) / (TT - 1)
    end
    rhoT[TT+1:m.NT] .= rho_final

    retired = sum(m.meaJ[m.Njw+1:m.Nj])
    working = sum(m.meaJ[1:m.Njw])
    tauT = [rhoT[tc] * retired / working for tc in 1:m.NT]

    return rhoT, tauT
end

# Linear initial guess: x_0 at tc=1, x_1 from tc=NT0 onwards.
function _linear_path(m::Model, x0, x1; NT0=30)
    path = fill(float(x1), m.NT)
    inc  = (x1 - x0) / (NT0 - 1)
    for tc in 1:NT0
        path[tc] = x0 + inc * (tc - 1)
    end
    return path
end

initial_guess_KT(m::Model, K_SS0, K_SS1; NT0=30) =
    _linear_path(m, K_SS0, K_SS1; NT0=NT0)
initial_guess_qT(m::Model, q_SS0, q_SS1; NT0=30) =
    _linear_path(m, q_SS0, q_SS1; NT0=NT0)

# Solve the household problem period by period, tc = NT -> 1, given the
# paths KT, qT and terminal continuation value vfun_SS1. Buffers for the
# per-period (vfun, afunG, afun) are swapped with the continuation slot to
# avoid allocation in the hot loop.
function backward_vfi_transition(m::Model, grids, capital_grid_translations,
                                  KT0, qT, rhoT, tauT, vfun_SS1)
    afunGT   = zeros(Int64, m.NT, m.Nj, m.Ne, m.Na)
    afunT    = zeros(m.NT, m.Nj, m.Ne, m.Na)
    vfun_TR  = zeros(m.NT, m.Ne)
    vfun_TR0 = zeros(m.Nj, m.Ne, m.Na)

    vfun_next = copy(vfun_SS1)
    vfun1     = zeros(m.Nj, m.Ne, m.Na)
    afunG     = zeros(Int64, m.Nj, m.Ne, m.Na)
    afun      = zeros(m.Nj, m.Ne, m.Na)

    for tc in m.NT:-1:1
        r = m.alpha * (KT0[tc] / m.L)^(m.alpha - 1) - m.delta
        w = (1 - m.alpha) * (KT0[tc] / m.L)^m.alpha
        yvec = income_grid(m, w, tauT[tc], rhoT[tc]; q = qT[tc])

        solve_household_period!(vfun1, afunG, afun, m, grids,
                                 capital_grid_translations, vfun_next, r, yvec)

        afunGT[tc, :, :, :] .= afunG
        afunT[tc, :, :, :]  .= afun
        vfun_TR[tc, :] = vfun1[1, :, 1]
        if tc == 1
            vfun_TR0 .= vfun1
        end

        vfun_next, vfun1 = vfun1, vfun_next
    end
    return afunGT, afunT, vfun_TR, vfun_TR0
end

# Forward simulation of the cross-sectional distribution over the horizon.
function forward_distribution_transition(m::Model, capital_grid_translations,
                                          afunGT, mea_SS0)
    meaT = zeros(m.NT, m.Nj, m.Ne, m.Na)
    meaT[1, :, :, :] .= mea_SS0
    mea_old = copy(mea_SS0)
    mea_new = zeros(m.Nj, m.Ne, m.Na)

    errm = sum(mea_old) - 1
    errm > 1e-4 && error("initial distribution mass drift: $errm")

    for tc in 1:m.NT-1
        afunG = @view afunGT[tc, :, :, :]
        advance_distribution!(mea_new, mea_old, m, afunG, capital_grid_translations)

        errm = abs(sum(mea_new) - 1)
        errm > 1e-4 && error("distribution mass drift at tc=$tc: $errm")

        mea_maxA = sum(@view mea_new[:, :, m.Na])
        if mea_maxA > 1e-3
            @warn "mass at top asset grid is large" mea_maxA tc
        end

        meaT[tc+1, :, :, :] .= mea_new
        mea_old, mea_new = mea_new, mea_old
    end
    return meaT
end

# Aggregate saving at each tc gives next period's capital: K_{t+1} = Σ mea_t·a'_t
# (including the forfeited savings of agents who will die; the (1+r) return on
# the forfeited share is what shows up as the bequest q_{t+1}).
function aggregate_capital_path(m::Model, afunT, meaT, KT0)
    KT1 = zeros(m.NT)
    errKT = zeros(m.NT)
    KT1[1] = KT0[1]
    @inbounds for tc in 1:m.NT-1
        s = 0.0
        for jc in 1:m.Nj, ec in 1:m.Ne, ac in 1:m.Na
            s += meaT[tc, jc, ec, ac] * afunT[tc, jc, ec, ac]
        end
        KT1[tc+1] = s
    end
    for tc in 1:m.NT
        errKT[tc] = abs(KT1[tc] - KT0[tc])
    end
    return KT1, errKT
end

# Outer loop: damped iteration on both the capital path and the bequest path.
function compute_transition(m::Model, grids, capital_grid_translations,
                             K_SS0, K_SS1, q_SS0, q_SS1,
                             rho0, rho1,
                             vfun_SS1, mea_SS0;
                             maxiterTR::Integer = 300,
                             errKTol::Real     = 1e-4,
                             errqTol::Real     = 1e-5,
                             TT::Integer       = 25,
                             NT0::Integer      = 30,
                             adjq::Real        = 0.2,
                             verbose::Bool     = true)

    rhoT, tauT = build_policy_paths(m, rho0, rho1; TT=TT)
    KT0 = initial_guess_KT(m, K_SS0, K_SS1; NT0=NT0)
    qT  = initial_guess_qT(m, q_SS0, q_SS1; NT0=NT0)
    qT[1] = q_SS0  # reform-date bequest is predetermined

    vfun_TR  = zeros(m.NT, m.Ne)
    vfun_TR0 = zeros(m.Nj, m.Ne, m.Na)
    errKvec  = zeros(maxiterTR)
    errqvec  = zeros(maxiterTR)
    errK     = 1.0
    errq     = 1.0
    iterTR   = 1

    while ((errK > errKTol) || (errq > errqTol)) && (iterTR <= maxiterTR)
        afunGT, afunT, vfun_TR, vfun_TR0 = backward_vfi_transition(
            m, grids, capital_grid_translations, KT0, qT, rhoT, tauT, vfun_SS1)

        meaT = forward_distribution_transition(
            m, capital_grid_translations, afunGT, mea_SS0)

        KT1, errKT = aggregate_capital_path(m, afunT, meaT, KT0)
        rT, _      = compute_price_paths(m, KT0)
        qT1        = compute_bequest_path(m, rT, afunT, meaT, q_SS0)

        errK = maximum(errKT)
        errq = maximum(abs.(qT1 .- qT))
        errKvec[iterTR] = errK
        errqvec[iterTR] = errq

        if errK > errKTol
            for tc in 2:m.NT
                KT0[tc] += m.adjK * (KT1[tc] - KT0[tc])
            end
        end
        if errq > errqTol
            for tc in 2:m.NT
                qT[tc]  += adjq * (qT1[tc] - qT[tc])
            end
        end

        verbose && println("iterTR = $iterTR, errK = $errK, errq = $errq")
        flush(stdout)
        iterTR += 1
    end

    if iterTR > maxiterTR && (errK > errKTol || errq > errqTol)
        @warn "transition did not converge" iterTR errK errq
    end

    return (; KT0, qT, vfun_TR, vfun_TR0, errKvec, errqvec, rhoT, tauT)
end
