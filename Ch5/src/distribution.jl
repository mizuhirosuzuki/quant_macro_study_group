function derive_stationary_distribution(m::Model, kfunG, capital_grid_translations)

    kc1vec, kc2vec, prk1vec, prk2vec = capital_grid_translations

    mea0 = ones(m.Nl, m.Nk) / (m.Nl * m.Nk)
    mea1 = zeros(m.Nl, m.Nk)
    err = 1.0
    errTol = 0.00001
    maxiter = 2000
    iter = 1

    while (err > errTol) & (iter < maxiter)

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

        err = maximum(abs.(mea1 - mea0))
        mea0 = copy(mea1)
        iter += 1
        mea1 = zeros(m.Nl, m.Nk)

    end

    if iter == maxiter
        println("WARNING!! INVARIANT DIST: iteration reached max: iter=$iter, err=$err")
    end

    return mea0
end

# Compute interpolation weights for a continuous k' value directly onto
# the state grid `gridk`. Returns (kcc1, kcc2, w1, w2) such that
# k' = w1 * gridk[kcc1] + w2 * gridk[kcc2] (when k' is inside the grid).
@inline function interp_weights_on_gridk(kp, gridk)
    Nk = length(gridk)
    if kp <= gridk[1]
        return 1, 1, 1.0, 0.0
    elseif kp >= gridk[Nk]
        return Nk, Nk, 1.0, 0.0
    end
    kcc1 = searchsortedlast(gridk, kp)
    kcc2 = kcc1 + 1
    w2 = (kp - gridk[kcc1]) / (gridk[kcc2] - gridk[kcc1])
    return kcc1, kcc2, 1 - w2, w2
end

# Stationary distribution for a continuous policy: weights for next-period
# capital are computed directly on `gridk` (no `gridk2` snapping).
function derive_stationary_distribution_continuous(m::Model, kfun, gridk)

    mea0 = ones(m.Nl, m.Nk) / (m.Nl * m.Nk)
    mea1 = zeros(m.Nl, m.Nk)
    err = 1.0
    errTol = 0.00001
    maxiter = 2000
    iter = 1

    while (err > errTol) & (iter < maxiter)

        for kc in 1:m.Nk
            for lc in 1:m.Nl

                kcc1, kcc2, w1, w2 = interp_weights_on_gridk(kfun[lc, kc], gridk)

                for lcc in 1:m.Nl
                    mea1[lcc, kcc1] += m.prob[lc, lcc] * w1 * mea0[lc, kc]
                    mea1[lcc, kcc2] += m.prob[lc, lcc] * w2 * mea0[lc, kc]
                end
            end
        end

        err = maximum(abs.(mea1 - mea0))
        mea0 = copy(mea1)
        iter += 1
        mea1 = zeros(m.Nl, m.Nk)

    end

    if iter == maxiter
        println("WARNING!! INVARIANT DIST (continuous): iteration reached max: iter=$iter, err=$err")
    end

    return mea0
end

function calculate_capital_supply(mea0, kfun)
    return sum(mea0 .* kfun)
end
