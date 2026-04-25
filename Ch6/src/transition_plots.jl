# Capital path normalised by K_SS0, with red markers at t=1 and t=NT (the two
# steady states).
function plot_transition_capital(m::Model, KT; K_SS0, K_SS1, outdir::AbstractString,
                                  suffix::AbstractString = "")
    isdir(outdir) || mkpath(outdir)
    ts = collect(1:m.NT)
    p = plot([1], [K_SS0 / K_SS0], mc=:red, markershape=:circle, lw=2, legend=false)
    plot!(p, ts, KT ./ K_SS0, color=:blue, lw=2)
    plot!(p, [m.NT], [K_SS1 / K_SS0], mc=:red, markershape=:circle, lw=2)
    title!(p, "Capital"); xlabel!(p, "Time")
    xlims!(p, 1 - 0.9, m.NT + 0.9)
    savefig(p, joinpath(outdir, "fig_olg2_tr_k$(suffix).pdf"))
    return p
end

# Interest rate path, red markers at the endpoints (approximate SS rates).
function plot_transition_interest(m::Model, rT; outdir::AbstractString,
                                   suffix::AbstractString = "")
    isdir(outdir) || mkpath(outdir)
    ts = collect(1:m.NT)
    p = plot([1], [rT[1]], mc=:red, markershape=:circle, lw=2, legend=false)
    plot!(p, ts, rT, color=:blue, lw=2)
    plot!(p, [m.NT], [rT[m.NT]], mc=:red, markershape=:circle, lw=2)
    title!(p, "Interest Rate"); xlabel!(p, "Time")
    xlims!(p, 1 - 0.9, m.NT + 0.9)
    savefig(p, joinpath(outdir, "fig_olg2_tr_r$(suffix).pdf"))
    return p
end
