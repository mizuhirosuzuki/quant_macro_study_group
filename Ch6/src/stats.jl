# Cross-sectional and life-cycle statistics from a steady-state solution `res`.
function compute_lifecycle_stats(m::Model, grids, res)
    grida, grida2 = grids
    tau = m.rho * sum(m.meaJ[m.Njw+1:m.Nj]) / sum(m.meaJ[1:m.Njw])
    yvec = income_grid(m, res.w, tau, m.rho)

    afunJ = zeros(m.Nj)
    for jc in 1:m.Nj
        s = 0.0
        for ac in 1:m.Na
            s += grida[ac] * sum(@view res.mea[jc, :, ac])
        end
        afunJ[jc] = s / m.meaJ[jc]
    end

    afunJE = zeros(m.Nj, m.Ne)
    for jc in 1:m.Nj, ec in 1:m.Ne
        mass = sum(@view res.mea[jc, ec, :])
        s = 0.0
        for ac in 1:m.Na
            s += grida[ac] * res.mea[jc, ec, ac]
        end
        afunJE[jc, ec] = s / mass
    end

    cfun = zeros(m.Nj, m.Ne, m.Na)
    sfun = zeros(m.Nj, m.Ne, m.Na)
    srat = zeros(m.Nj, m.Ne, m.Na)
    for jc in 1:m.Nj, ec in 1:m.Ne, ac in 1:m.Na
        acc = res.afunG[jc, ec, ac]
        y   = yvec[jc, ec]
        c   = y + (1 + res.r) * grida[ac] - grida2[acc]
        inc = y + res.r * grida[ac]
        cfun[jc, ec, ac] = c
        sfun[jc, ec, ac] = inc - c
        srat[jc, ec, ac] = (inc - c) / inc
    end

    cfunJ = [sum(@view(res.mea[jc, :, :]) .* @view(cfun[jc, :, :])) /
             sum(@view res.mea[jc, :, :]) for jc in 1:m.Nj]

    sfunJE = zeros(m.Nj, m.Ne)
    sratJE = zeros(m.Nj, m.Ne)
    for jc in 1:m.Nj, ec in 1:m.Ne
        mass = sum(@view res.mea[jc, ec, :])
        sfunJE[jc, ec] = sum(@view(res.mea[jc, ec, :]) .* @view(sfun[jc, ec, :])) / mass
        sratJE[jc, ec] = sum(@view(res.mea[jc, ec, :]) .* @view(srat[jc, ec, :])) / mass
    end

    return (; afunJ, afunJE, cfun, sfun, srat, cfunJ, sfunJE, sratJE, yvec)
end

function plot_lifecycle(m::Model, grids, stats; outdir::AbstractString, age_start::Int=20,
                        ref_age::Int=21)
    isdir(outdir) || mkpath(outdir)
    grida = grids[1]
    ages  = collect(age_start:age_start + m.Nj - 1)
    norm  = 1 / stats.cfunJ[1]
    zerovec = zeros(m.Na)

    p1 = plot(ages, norm .* stats.afunJE[:, 2], ls=:solid,   lc=:black, lw=3, label="high", legend=:topleft)
    plot!(p1, ages, norm .* stats.afunJE[:, 1], ls=:dashdot, lc=:black, lw=3, label="low")
    xlims!(p1, age_start, age_start + m.Nj - 1)
    title!(p1, "Asset (Stock) By Age and Prod"); xlabel!(p1, "Age"); ylabel!(p1, "Asset")
    savefig(p1, joinpath(outdir, "fig_olg2_a.pdf"))

    p2 = plot(grida, stats.srat[ref_age, 2, :], ls=:solid,   lc=:black, lw=3, label="high")
    plot!(p2, grida, stats.srat[ref_age, 1, :], ls=:dashdot, lc=:black, lw=3, label="low")
    plot!(p2, grida, zerovec, ls=:dot, lc=:black, lw=1, label="")
    xlims!(p2, 0.0, 20.0)
    title!(p2, "Saving Rate By Asset and Prod"); xlabel!(p2, "Asset"); ylabel!(p2, "Saving Rate")
    savefig(p2, joinpath(outdir, "fig_olg2_s.pdf"))

    p3 = plot(grida, stats.sfun[ref_age, 2, :], ls=:solid,   lc=:black, lw=3, label="high")
    plot!(p3, grida, stats.sfun[ref_age, 1, :], ls=:dashdot, lc=:black, lw=3, label="low")
    plot!(p3, grida, zerovec, ls=:dot, lc=:black, lw=1, label="")
    xlims!(p3, 0.0, 20.0)
    title!(p3, "Saving Level By Asset and Prod"); xlabel!(p3, "Asset"); ylabel!(p3, "Saving (level)")
    savefig(p3, joinpath(outdir, "fig_olg2_slevel.pdf"))

    return (p1, p2, p3)
end

