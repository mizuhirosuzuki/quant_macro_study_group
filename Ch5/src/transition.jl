function make_initial_guesses(m::Model, K_SS0, K_SS1, tau)

    KT0 = K_SS1 .* ones(m.NT)
    NT0 = m.NT

    intK = (K_SS1 - K_SS0) / (NT0 - 1)

    for tc in 1:NT0
        KT0[tc] = K_SS0 + intK * (tc - 1)
    end

    rT0 = zeros(m.NT)
    for tc in 1:m.NT
        rT0[tc] = m.alpha * ((KT0[tc] / m.labor)^(m.alpha - 1)) - m.delta
    end

    TT0 = KT0 .* rT0 .* tau

    return TT0, KT0, rT0
end

function compute_value_function_backwards!(
    vfun0, kfunGT, kfunT, m::Model, tc, TT0, KT0, rT0,
    grids, capital_grid_translations, tau,
)
    gridk, gridk2 = grids
    kc1vec, kc2vec, prk1vec, prk2vec = capital_grid_translations

    r0 = rT0[tc]
    T0 = TT0[tc]
    wage = calculate_w0(m, r0)

    kfunG = zeros(m.Nl, m.Nk)
    vfun1 = zeros(m.Nl, m.Nk)
    kfun = zeros(m.Nl, m.Nk)

    for kc in 1:m.Nk
        for lc in 1:m.Nl

            vtemp = -1000000 .* ones(m.Nk2)
            kccmax = m.Nk2

            for kcc in 1:m.Nk2

                cons = (
                    m.s[lc] * wage
                    + (1 + r0 * (1 - tau)) * gridk[kc]
                    - gridk2[kcc]
                    + T0
                )

                if cons <= 0.0
                    kccmax = kcc - 1
                    break
                end

                util = (cons^(1.0 - m.mu)) / (1.0 - m.mu)

                kcc1 = Int(kc1vec[kcc])
                kcc2 = Int(kc2vec[kcc])

                vpr = 0.0
                for lcc in 1:m.Nl
                    vpr += m.prob[lc, lcc] * (
                        prk1vec[kcc] * vfun0[lcc, kcc1]
                        + prk2vec[kcc] * vfun0[lcc, kcc2]
                    )
                end

                vtemp[kcc] = util + m.beta * vpr

            end

            t1, t2 = findmax(vtemp[1:kccmax])
            vfun1[lc, kc] = t1
            kfunG[lc, kc] = t2
            kfun[lc, kc] = gridk2[t2]

        end
    end

    vfun0[:, :] .= vfun1
    kfunGT[tc, :, :] .= kfunG
    kfunT[tc, :, :] .= kfun

    return nothing
end

function compute_all_value_function_backwards(
    m::Model, TT0, KT0, rT0, v_SS1, grids, capital_grid_translations, tau,
)
    kfunGT = zeros(m.NT, m.Nl, m.Nk)
    kfunT = similar(kfunGT)

    vfun0 = copy(v_SS1)

    for tc in m.NT:-1:1
        compute_value_function_backwards!(
            vfun0, kfunGT, kfunT, m, tc, TT0, KT0, rT0,
            grids, capital_grid_translations, tau,
        )
    end

    return kfunGT, kfunT
end

function compute_distribution_t(m::Model, kfunGT, capital_grid_translations, mea_SS0)

    kc1vec, kc2vec, prk1vec, prk2vec = capital_grid_translations

    meaT = zeros(m.NT, m.Nl, m.Nk)
    meaT[1, :, :] .= copy(mea_SS0)

    mea0 = mea_SS0

    for tc in 1:m.NT-1

        kfunG = copy(kfunGT[tc, :, :])
        mea1 = zeros(m.Nl, m.Nk)

        for kc in 1:m.Nk
            for lc in 1:m.Nl

                kcc = Int(kfunG[lc, kc])

                kcc1 = Int(kc1vec[kcc])
                kcc2 = Int(kc2vec[kcc])

                for lcc in 1:m.Nl
                    mea1[lcc, kcc1] += m.prob[lc, lcc] * prk1vec[kcc] * mea0[lc, kc]
                    mea1[lcc, kcc2] += m.prob[lc, lcc] * prk2vec[kcc] * mea0[lc, kc]
                end
            end
        end

        meaT[tc+1, :, :] = copy(mea1)
        mea0 = copy(mea1)

    end

    return meaT
end

function compute_capital(m::Model, KT0, kfunT, meaT)

    KT1 = zeros(m.NT)
    KT1[1] = KT0[1]

    for tc in 1:m.NT-1
        kfun = copy(kfunT[tc, :, :])
        mea0 = meaT[tc, :, :]
        KT1[tc+1] = sum(mea0 .* kfun)
    end

    return KT1
end

function update_variables!(TT0, KT0, rT0, m::Model, KT1, K_SS0, K_SS1, tau; adjK=0.04)
    KT0 .+= adjK .* (KT1 - KT0)
    KT0 .= (KT0 .- KT0[end]) ./ (KT0[1] - KT0[end]) .* (K_SS0 - K_SS1) .+ K_SS1
    rT0[:] .= m.alpha .* ((KT0 ./ m.labor) .^ (m.alpha - 1)) .- m.delta
    TT0[:] .= KT0 .* rT0 .* tau

    return nothing
end

function _backward_VFI_dispatch(method, m, TT0, KT0, rT0, v_SS1, grids, capital_grid_translations, tau)
    if method == :grid_search
        return compute_all_value_function_backwards(
            m, TT0, KT0, rT0, v_SS1, grids, capital_grid_translations, tau,
        )
    elseif method == :continuous
        return compute_all_value_function_backwards_continuous(
            m, TT0, KT0, rT0, v_SS1, grids, tau,
        )
    else
        error("Unknown VFI method: $method. Use :grid_search or :continuous.")
    end
end

function _distribution_t_dispatch(method, m, kfunGT, kfunT, grids, capital_grid_translations, mea_SS0)
    if method == :grid_search
        return compute_distribution_t(m, kfunGT, capital_grid_translations, mea_SS0)
    elseif method == :continuous
        # Use continuous policy directly (no gridk2 snapping)
        return compute_distribution_t_continuous(m, kfunT, grids[1], mea_SS0)
    else
        error("Unknown VFI method: $method. Use :grid_search or :continuous.")
    end
end

function derive_transition(m::Model, K_SS0, K_SS1, v_SS1, mea_SS0, tau; method=:grid_search)

    grids = generate_capital_grid(m)
    capital_grid_translations = translate_capital_grid(m, grids)

    errKTol = 1e-3
    errK = 1.0
    maxiterTR = 100
    iterTR = 1

    KT0_iteration_history = zeros(m.NT, maxiterTR)

    TT0, KT0, rT0 = make_initial_guesses(m, K_SS0, K_SS1, tau)

    meaT = zeros(m.NT, m.Nl, m.Nk)

    while (errK > errKTol) && (iterTR < maxiterTR)

        kfunGT, kfunT = _backward_VFI_dispatch(
            method, m, TT0, KT0, rT0, v_SS1, grids, capital_grid_translations, tau,
        )

        meaT = _distribution_t_dispatch(
            method, m, kfunGT, kfunT, grids, capital_grid_translations, mea_SS0,
        )

        KT1 = compute_capital(m, KT0, kfunT, meaT)

        errK = maximum(abs.(KT1 - KT0))
        if errK > errKTol
            update_variables!(TT0, KT0, rT0, m, KT1, K_SS0, K_SS1, tau)
        end

        KT0_iteration_history[:, iterTR] = KT0

        iterTR += 1

    end

    return KT0, KT0_iteration_history, meaT
end
