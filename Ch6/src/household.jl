function income_grid(m::Model, w, tau, rho)
    yvec = zeros(m.Nj, m.Ne)
    for jc in 1:m.Njw, ec in 1:m.Ne
        yvec[jc, ec] = (1 - tau) * w * m.gride[ec]
    end
    yvec[m.Njw+1:m.Nj, :] .= rho * w
    return yvec
end

# Backward induction over ages at a single date, writing into preallocated
# output arrays. `vfun_next` is the continuation value (next period's value
# function for the transition; the same array for stationary iteration —
# aliasing is safe because the loop reads age jc+1 only after writing it).
#
# Speed tricks inside the inner search:
#   monotonicity — optimal acc*(ac) is non-decreasing in ac, so scan starts
#                  from the previous ac's optimum (acc_lo).
#   concavity   — once the candidate value stops rising we break.
#   threads     — ec slices of the output arrays are disjoint.
function solve_household_period!(vfun, afunG, afun,
                                 m::Model, grids, capital_grid_translations,
                                 vfun_next, r, yvec)
    grida, grida2 = grids
    ac1vec, ac2vec, pra1vec, pra2vec = capital_grid_translations

    jc = m.Nj
    @inbounds for ec in 1:m.Ne, ac in 1:m.Na
        c = yvec[jc, ec] + (1 + r) * grida[ac]
        vfun[jc, ec, ac]  = log(c)
        afunG[jc, ec, ac] = 1
        afun[jc, ec, ac]  = grida2[1]
    end

    for jc in m.Nj-1:-1:1
        Threads.@threads for ec in 1:m.Ne
            y = yvec[jc, ec]
            acc_lo = 1
            @inbounds for ac in 1:m.Na
                cash = y + (1 + r) * grida[ac]
                best_val = -Inf
                best_idx = acc_lo
                for acc in acc_lo:m.Na2
                    c = cash - grida2[acc]
                    c <= 0.0 && break
                    acc1 = ac1vec[acc]; acc2 = ac2vec[acc]
                    vpr = 0.0
                    for ecc in 1:m.Ne
                        vpr += m.Pe[ec, ecc] * (
                            pra1vec[acc] * vfun_next[jc+1, ecc, acc1]
                          + pra2vec[acc] * vfun_next[jc+1, ecc, acc2])
                    end
                    v = log(c) + m.beta * vpr
                    if v > best_val
                        best_val = v
                        best_idx = acc
                    else
                        break
                    end
                end
                vfun[jc, ec, ac]  = best_val
                afunG[jc, ec, ac] = best_idx
                afun[jc, ec, ac]  = grida2[best_idx]
                acc_lo = best_idx
            end
        end
    end
    return nothing
end

function solve_household_period(m::Model, grids, capital_grid_translations,
                                vfun_next, r, yvec)
    vfun  = zeros(m.Nj, m.Ne, m.Na)
    afunG = zeros(Int64, m.Nj, m.Ne, m.Na)
    afun  = zeros(m.Nj, m.Ne, m.Na)
    solve_household_period!(vfun, afunG, afun, m, grids,
                             capital_grid_translations, vfun_next, r, yvec)
    return vfun, afunG, afun
end
