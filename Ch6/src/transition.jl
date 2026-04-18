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

# Linear initial guess: K_SS0 at tc=1, K_SS1 from tc=NT0 onwards.
function initial_guess_KT(m::Model, K_SS0, K_SS1; NT0=30)
    KT0 = fill(K_SS1, m.NT)
    intK = (K_SS1 - K_SS0) / (NT0 - 1)
    for tc in 1:NT0
        KT0[tc] = K_SS0 + intK * (tc - 1)
    end
    return KT0
end

# Solve the household problem period by period, tc = NT -> 1, given the
# capital path KT0 and terminal continuation value vfun_SS1. Buffers for
# per-period (vfun, afunG, afun) are swapped with the continuation slot to
# avoid allocation in the hot loop.
function backward_vfi_transition(m::Model, grids, capital_grid_translations,
                                  KT0, rhoT, tauT, vfun_SS1)
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
        yvec = income_grid(m, w, tauT[tc], rhoT[tc])

        solve_household_period!(vfun1, afunG, afun, m, grids,
                                 capital_grid_translations, vfun_next, r, yvec)

        afunGT[tc, :, :, :] .= afunG
        afunT[tc, :, :, :]  .= afun
        vfun_TR[tc, :] = vfun1[1, :, 1]
        if tc == 1
            vfun_TR0 .= vfun1
        end

        vfun_next, vfun1 = vfun1, vfun_next  # continuation for tc-1 is what we just wrote
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

# Aggregate saving at each tc gives next period's capital: K_{t+1} = Σ mea_t · a'_t.
function aggregate_capital_path(m::Model, afunT, meaT, KT0)
    KT1 = zeros(m.NT)
    errKT = zeros(m.NT)
    KT1[1] = KT0[1]                                   # predetermined
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

# Outer loop: damped iteration on the capital path.
function compute_transition(m::Model, grids, capital_grid_translations,
                             K_SS0, K_SS1, rho0, rho1,
                             vfun_SS1, mea_SS0;
                             maxiterTR::Integer = 300,
                             errKTol::Real     = 1e-4,
                             TT::Integer       = 25,
                             NT0::Integer      = 30,
                             verbose::Bool     = true)

    rhoT, tauT = build_policy_paths(m, rho0, rho1; TT=TT)
    KT0 = initial_guess_KT(m, K_SS0, K_SS1; NT0=NT0)

    vfun_TR  = zeros(m.NT, m.Ne)
    vfun_TR0 = zeros(m.Nj, m.Ne, m.Na)
    errKvec  = zeros(maxiterTR)
    errK     = 1.0
    iterTR   = 1

    while (errK > errKTol) && (iterTR <= maxiterTR)
        afunGT, afunT, vfun_TR, vfun_TR0 = backward_vfi_transition(
            m, grids, capital_grid_translations, KT0, rhoT, tauT, vfun_SS1)

        meaT = forward_distribution_transition(
            m, capital_grid_translations, afunGT, mea_SS0)

        KT1, errKT = aggregate_capital_path(m, afunT, meaT, KT0)

        errK = maximum(errKT)
        errKvec[iterTR] = errK

        if errK > errKTol
            for tc in 2:m.NT
                KT0[tc] += m.adjK * (KT1[tc] - KT0[tc])
            end
        end

        verbose && println("iterTR = $iterTR, errK = $errK")
        flush(stdout)
        iterTR += 1
    end

    if iterTR > maxiterTR && errK > errKTol
        @warn "transition did not converge" iterTR errK
    end

    return (; KT0, vfun_TR, vfun_TR0, errKvec, rhoT, tauT)
end
