function compute_value_function_backwards_continuous!(
    vfun0, kfunGT, kfunT, m::Model, tc, TT0, KT0, rT0, grids, tau,
)
    gridk, gridk2 = grids

    r0 = rT0[tc]
    T0 = TT0[tc]
    wage = calculate_w0(m, r0)

    vfun1 = zeros(m.Nl, m.Nk)
    kfun = zeros(m.Nl, m.Nk)
    Ev = zeros(m.Nl, m.Nk)

    minK = gridk2[1]
    mu = m.mu
    beta = m.beta

    # precompute expectation E[v(k, l') | l] once for this period
    compute_expected_value!(Ev, m, vfun0)

    for kc in 1:m.Nk
        for lc in 1:m.Nl

            cash_on_hand = (
                m.s[lc] * wage
                + (1 + r0 * (1 - tau)) * gridk[kc]
                + T0
            )

            kp_max = min(cash_on_hand - 1e-10, gridk2[end])

            if kp_max <= minK
                vfun1[lc, kc] = -1e10
                kfun[lc, kc] = minK
                continue
            end

            Ev_row = @view Ev[lc, :]
            kp_opt, neg_v_opt = golden_section_bellman(
                minK, kp_max, cash_on_hand, mu, beta, gridk, Ev_row,
            )
            vfun1[lc, kc] = -neg_v_opt
            kfun[lc, kc] = kp_opt

        end
    end

    vfun0[:, :] .= vfun1
    # kfunGT is left as zeros — the continuous distribution iteration
    # uses kfunT directly, bypassing the gridk2 snapping detour.
    kfunT[tc, :, :] .= kfun

    return nothing
end

function compute_all_value_function_backwards_continuous(
    m::Model, TT0, KT0, rT0, v_SS1, grids, tau,
)
    kfunGT = zeros(m.NT, m.Nl, m.Nk)
    kfunT = similar(kfunGT)

    vfun0 = copy(v_SS1)

    for tc in m.NT:-1:1
        compute_value_function_backwards_continuous!(
            vfun0, kfunGT, kfunT, m, tc, TT0, KT0, rT0, grids, tau,
        )
    end

    return kfunGT, kfunT
end

# Forward distribution iteration for the continuous policy: interpolation
# weights are computed directly on gridk from the continuous kfun values.
function compute_distribution_t_continuous(m::Model, kfunT, gridk, mea_SS0)

    meaT = zeros(m.NT, m.Nl, m.Nk)
    meaT[1, :, :] .= copy(mea_SS0)

    mea0 = copy(mea_SS0)

    for tc in 1:m.NT-1

        kfun_t = @view kfunT[tc, :, :]
        mea1 = zeros(m.Nl, m.Nk)

        for kc in 1:m.Nk
            for lc in 1:m.Nl

                kcc1, kcc2, w1, w2 = interp_weights_on_gridk(kfun_t[lc, kc], gridk)

                for lcc in 1:m.Nl
                    mea1[lcc, kcc1] += m.prob[lc, lcc] * w1 * mea0[lc, kc]
                    mea1[lcc, kcc2] += m.prob[lc, lcc] * w2 * mea0[lc, kc]
                end
            end
        end

        meaT[tc+1, :, :] = copy(mea1)
        mea0 = copy(mea1)

    end

    return meaT
end
