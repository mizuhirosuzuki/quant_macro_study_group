# Remaining-life discount factor: betaJ[jc] = Σ_{k=0..Nj-jc} β^k.
# Dividing a utility gap by betaJ[jc] converts it into a per-period consumption
# equivalent for a log-utility agent.
function life_cycle_discount(m::Model)
    betaJ = zeros(m.Nj)
    for jc in 1:m.Nj
        s = 0.0
        for ic in jc:m.Nj
            s += m.beta^(ic - jc)
        end
        betaJ[jc] = s
    end
    return betaJ
end

# CEV at t=0 by (age, productivity, asset) for households already alive when
# the reform hits. Positive = reform is welfare-improving for that cell.
function compute_welfare_existing(m::Model, vfun_TR0, vfun_SS0,
                                  betaJ = life_cycle_discount(m))
    welf = zeros(m.Nj, m.Ne, m.Na)
    for jc in 1:m.Nj, ec in 1:m.Ne, ac in 1:m.Na
        welf[jc, ec, ac] =
            exp((vfun_TR0[jc, ec, ac] - vfun_SS0[jc, ec, ac]) / betaJ[jc]) - 1
    end
    return welf
end

# Mass-weighted average CEV by (age, productivity), weighting by the initial
# SS distribution.
function average_welfare_existing(m::Model, welf, mea_SS0)
    out = zeros(m.Nj, m.Ne)
    for jc in 1:m.Nj, ec in 1:m.Ne
        mass = sum(@view mea_SS0[jc, ec, :])
        out[jc, ec] =
            sum(@view(welf[jc, ec, :]) .* @view(mea_SS0[jc, ec, :])) / mass
    end
    return out
end

# CEV for a newborn cohort entering at date tc, by productivity — relative to
# being born into the initial SS.
function compute_welfare_newborn(m::Model, vfun_TR, vfun_SS0,
                                  betaJ = life_cycle_discount(m))
    welfTR = zeros(m.NT, m.Ne)
    for tc in 1:m.NT, ec in 1:m.Ne
        welfTR[tc, ec] =
            exp((vfun_TR[tc, ec] - vfun_SS0[1, ec, 1]) / betaJ[1]) - 1
    end
    return welfTR
end

# Plots ---------------------------------------------------------------

function plot_welfare_newborn_cohorts(m::Model, welfTR; outdir::AbstractString,
                                       filename::AbstractString = "fig_welf_tr.pdf",
                                       xlimits = (1, 60),
                                       ylimits = (0.01, 0.08))
    isdir(outdir) || mkpath(outdir)
    ts = collect(1:m.NT)
    p = plot(ts, welfTR[:, 1], ls=:dash,  c=:black, lw=3, label="low", legend=:bottomright)
    plot!(p, ts, welfTR[:, 2], ls=:solid, c=:black, lw=3, label="high")
    xlims!(p, xlimits...)
    ylims!(p, ylimits...)
    xlabel!(p, "Cohort"); ylabel!(p, "CEV")
    savefig(p, joinpath(outdir, filename))
    return p
end

function plot_welfare_by_age(m::Model, welf0_JE; outdir::AbstractString,
                              age_start::Int = 20,
                              filename::AbstractString = "fig_welf_0.pdf",
                              ylimits = (-0.1, 0.04))
    isdir(outdir) || mkpath(outdir)
    ages = collect(age_start:age_start + m.Nj - 1)
    p = plot(ages, welf0_JE[:, 1], ls=:dash,  c=:black, lw=3, label="low", legend=:bottomright)
    plot!(p, ages, welf0_JE[:, 2], ls=:solid, c=:black, lw=3, label="high")
    ylims!(p, ylimits...)
    xlabel!(p, "Age"); ylabel!(p, "CEV")
    savefig(p, joinpath(outdir, filename))
    return p
end

function plot_welfare_by_asset(::Model, grids, welf; jc::Int = 21,
                                outdir::AbstractString,
                                filename::AbstractString = "fig_welf_aged_40.pdf")
    isdir(outdir) || mkpath(outdir)
    grida = grids[1]
    p = plot(grida, welf[jc, 1, :], ls=:dash,  c=:black, lw=3, label="low", legend=:bottomright)
    plot!(p, grida, welf[jc, 2, :], ls=:solid, c=:black, lw=3, label="high")
    xlabel!(p, "Asset"); ylabel!(p, "CEV")
    savefig(p, joinpath(outdir, filename))
    return p
end
