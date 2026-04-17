function solve_VFI(m::Model, r0, K0, wage, grids, capital_grid_translations; tau=0)

    gridk, gridk2 = grids
    kc1vec, kc2vec, prk1vec, prk2vec = capital_grid_translations

    kfunG = zeros(m.Nl, m.Nk)
    kfun = similar(kfunG)
    v = zeros(m.Nl, m.Nk)
    tv = similar(kfunG)
    kfunG_old = zeros(m.Nl, m.Nk)

    err = 10
    maxiter = 2000
    iter = 1

    while (err > 0) & (iter < maxiter)

        for kc in 1:m.Nk
            for lc in 1:m.Nl

                kccmax = m.Nk2
                vtemp = -1000000 .* ones(m.Nk2)

                for kcc in 1:m.Nk2

                    cons = (
                        m.s[lc] * wage
                        + (1 + r0 * (1 - tau)) * gridk[kc]
                        - gridk2[kcc]
                        + tau * r0 * K0
                    )

                    if cons <= 0.0
                        kccmax = kcc - 1
                        break
                    end

                    util = (cons^(1 - m.mu)) / (1 - m.mu)

                    kcc1 = Int(kc1vec[kcc])
                    kcc2 = Int(kc2vec[kcc])

                    vpr = 0.0
                    for lcc in 1:m.Nl
                        vpr += m.prob[lc, lcc] * (
                            prk1vec[kcc] * v[lcc, kcc1]
                            + prk2vec[kcc] * v[lcc, kcc2]
                        )
                    end

                    vtemp[kcc] = util + m.beta * vpr

                end

                t1, t2 = findmax(vtemp[1:kccmax])
                tv[lc, kc] = t1
                kfunG[lc, kc] = t2
                kfun[lc, kc] = gridk2[t2]

            end
        end

        v = copy(tv)
        err = maximum(abs.(kfunG - kfunG_old))
        kfunG_old = copy(kfunG)
        iter += 1

    end

    if iter == maxiter
        println("WARNING!! VFI: iteration reached max: iter=$iter, err=$err")
    end

    return kfun, kfunG, v
end
