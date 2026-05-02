using Interpolations
using NLsolve
using LinearAlgebra: dot

"""
最適コミットメント政策（commitment policy）の時間反復法による求解。

教科書 7.6 節のアルゴリズムを実装。
状態変数: (φ_EE,-1, φ_PC,-1, s_i)（過去のラグランジュ乗数 + 外生ショック）
方策関数: ς_y, ς_π, ς_rn, ς_φEE, ς_φPC

モデル方程式（discNK の規約: ショックは IS に +g として加わる）:
- IS:  y = E[y'] - σ(r_n - E[π']) + g
- PC:  π = κ y + β E[π']
- FOC π: π + φ_PC - φ_PC,-1 - σ·β⁻¹·φ_EE,-1 = 0
- FOC y: λ y + φ_EE - β⁻¹·φ_EE,-1 - κ·φ_PC = 0
- ZLB: r_n ≥ 0, φ_EE ≥ 0（ZLB の乗数）, 相補スラックネス
"""

# -- 期待値計算 -------------------------------------------------------------

"""
    build_interpolators(grid_φ_EE, grid_φ_PC, ς_y_old, ς_π_old)
        -> (itp_y::Vector, itp_π::Vector)

各ショック状態 j ごとに 2D 双線形補間オブジェクトを作成。範囲外は `Flat()` 外挿。
"""
function build_interpolators(grid_φ_EE::AbstractVector, grid_φ_PC::AbstractVector,
                              ς_y_old::AbstractArray{<:Real,3}, ς_π_old::AbstractArray{<:Real,3})
    Ns = size(ς_y_old, 3)
    itp_y = [extrapolate(interpolate((grid_φ_EE, grid_φ_PC), ς_y_old[:,:,j], Gridded(Linear())), Flat()) for j in 1:Ns]
    itp_π = [extrapolate(interpolate((grid_φ_EE, grid_φ_PC), ς_π_old[:,:,j], Gridded(Linear())), Flat()) for j in 1:Ns]
    return itp_y, itp_π
end

"""
    expected_values(itp_y, itp_π, Ps, i, φ_EE, φ_PC) -> (ye, pe)

現在ショック i のもとで、来期の期待値を計算:
ye = Σ_j p_ij · ς_y(φ_EE, φ_PC, s_j),  pe = 同様。
"""
function expected_values(itp_y::AbstractVector, itp_π::AbstractVector,
                         Ps::AbstractMatrix, i::Integer, φ_EE::Real, φ_PC::Real)
    Ns = length(itp_y)
    ye = 0.0
    pe = 0.0
    @inbounds for j in 1:Ns
        ye += Ps[i, j] * itp_y[j](φ_EE, φ_PC)
        pe += Ps[i, j] * itp_π[j](φ_EE, φ_PC)
    end
    return ye, pe
end

# -- Case A: ZLB 非バインド (φ_EE = 0) -------------------------------------

"""
    commitment_unconstrained(m, itp_y, itp_π, Ps, i, g, φ_EE_lag, φ_PC_lag)
        -> (y, π, r_n, φ_EE, φ_PC)

φ_EE = 0 を仮定。FOC y より y = (β⁻¹·φ_EE,-1 + κ·φ_PC) / λ、
FOC π より π = φ_PC,-1 + σ·β⁻¹·φ_EE,-1 - φ_PC。これらを PC に代入すると
φ_PC についての 1 次元非線形方程式に帰着する:

    φ_PC·(1 + κ²/λ) - φ_PC,-1 - β⁻¹·φ_EE,-1·(σ - κ/λ) + β·π_e(0, φ_PC; i) = 0

求めた φ_PC から y, π を逆算し、IS から r_n を求める。
r_n が負になっていたら呼び出し側で Case B に切り替える。
"""
function commitment_unconstrained(m::Model, itp_y, itp_π, Ps::AbstractMatrix, i::Integer,
                                   g::Real, φ_EE_lag::Real, φ_PC_lag::Real;
                                   x0::Real=0.0, ftol::Real=1e-10)
    function res!(r, x)
        _, pe = expected_values(itp_y, itp_π, Ps, i, 0.0, x[1])
        r[1] = x[1]*(1 + m.kap^2/m.lam) - φ_PC_lag - φ_EE_lag/m.bet*(m.sig - m.kap/m.lam) + m.bet*pe
    end
    sol = nlsolve(res!, [float(x0)]; ftol=ftol, autodiff=:forward)
    φ_PC = sol.zero[1]

    y = (φ_EE_lag/m.bet + m.kap*φ_PC) / m.lam
    π = φ_PC_lag + m.sig*φ_EE_lag/m.bet - φ_PC
    ye, pe = expected_values(itp_y, itp_π, Ps, i, 0.0, φ_PC)
    r = (1/m.sig)*(ye - y + g) + pe
    return y, π, r, 0.0, φ_PC
end

# -- Case B: ZLB バインド (r_n = 0) ----------------------------------------

"""
    commitment_zlb(m, itp_y, itp_π, Ps, i, g, φ_EE_lag, φ_PC_lag, x0)
        -> (y, π, r_n, φ_EE, φ_PC)

r_n = 0 を仮定。IS より y = y_e + σ·π_e + g、PC より π = κy + β·π_e。
未知の (φ_EE, φ_PC) について以下の 2 次元非線形系を NLsolve.jl で解く:

    res1 = φ_PC - φ_PC,-1 - σ·β⁻¹·φ_EE,-1 + κ·y_e + (κσ + β)·π_e + κg
    res2 = φ_EE - β⁻¹·φ_EE,-1 - κ·φ_PC + λ·(y_e + σ·π_e + g)
"""
function commitment_zlb(m::Model, itp_y, itp_π, Ps::AbstractMatrix, i::Integer,
                         g::Real, φ_EE_lag::Real, φ_PC_lag::Real,
                         x0::AbstractVector; ftol=1e-10)
    function residual!(res, x)
        φ_EE, φ_PC = x[1], x[2]
        ye, pe = expected_values(itp_y, itp_π, Ps, i, φ_EE, φ_PC)
        res[1] = φ_PC - φ_PC_lag - m.sig*φ_EE_lag/m.bet + m.kap*ye + (m.kap*m.sig + m.bet)*pe + m.kap*g
        res[2] = φ_EE - φ_EE_lag/m.bet - m.kap*φ_PC + m.lam*(ye + m.sig*pe + g)
    end
    sol = nlsolve(residual!, collect(x0); ftol=ftol, autodiff=:forward)
    φ_EE = sol.zero[1]
    φ_PC = sol.zero[2]
    ye, pe = expected_values(itp_y, itp_π, Ps, i, φ_EE, φ_PC)
    y = ye + m.sig*pe + g
    π = m.kap*y + m.bet*pe
    return y, π, 0.0, φ_EE, φ_PC
end

# -- 場合分けと反復 --------------------------------------------------------

"""
    commitment_step(m, itp_y, itp_π, Ps, i, g, φ_EE_lag, φ_PC_lag, x0_zlb)
        -> (y, π, r_n, φ_EE, φ_PC)

まず Case A（ZLB 非バインド）を試し、r_n < 0 なら Case B に切り替える。
"""
function commitment_step(m::Model, itp_y, itp_π, Ps::AbstractMatrix, i::Integer,
                         g::Real, φ_EE_lag::Real, φ_PC_lag::Real,
                         x0_zlb::AbstractVector;
                         x0_unc::Real=0.0)
    y, π, r, φ_EE, φ_PC = commitment_unconstrained(m, itp_y, itp_π, Ps, i, g, φ_EE_lag, φ_PC_lag; x0=x0_unc)
    if r < 0
        y, π, r, φ_EE, φ_PC = commitment_zlb(m, itp_y, itp_π, Ps, i, g, φ_EE_lag, φ_PC_lag, x0_zlb)
    end
    return y, π, r, φ_EE, φ_PC
end

"""
    solve_commitment(m; Gs, Ps, grid_φ_EE, grid_φ_PC,
                     verbose=false, max_outer=300, tol=1e-6)
        -> NamedTuple

コミットメント政策関数の時間反復による解。

引数:
- `Gs::AbstractVector{<:Real}` : 長さ Ns のショック値グリッド (g; discNK 規約で IS に +g として加わる)
- `Ps::AbstractMatrix` : Ns × Ns 遷移行列
- `grid_φ_EE`, `grid_φ_PC` : 内生状態 (過去の Lagrange 乗数) のグリッド
- `tol` : 政策関数の収束基準 (max-norm)

戻り値の NamedTuple は各方策関数を `(N1, N2, Ns)` の 3 次元配列で保持。
"""
function solve_commitment(m::Model; Gs::AbstractVector, Ps::AbstractMatrix,
                          grid_φ_EE::AbstractVector, grid_φ_PC::AbstractVector,
                          verbose::Bool=false, max_outer::Int=300, tol::Float64=1e-6)
    Ns = length(Gs)
    @assert size(Ps) == (Ns, Ns)
    N1 = length(grid_φ_EE)
    N2 = length(grid_φ_PC)

    ς_y    = zeros(N1, N2, Ns)
    ς_π    = zeros(N1, N2, Ns)
    ς_r    = zeros(N1, N2, Ns)
    ς_φEE  = zeros(N1, N2, Ns)
    ς_φPC  = zeros(N1, N2, Ns)

    ς_y_new   = similar(ς_y)
    ς_π_new   = similar(ς_π)
    ς_r_new   = similar(ς_r)
    ς_φEE_new = similar(ς_φEE)
    ς_φPC_new = similar(ς_φPC)

    diff = Inf
    iter = 0
    while diff > tol && iter < max_outer
        iter += 1
        itp_y, itp_π = build_interpolators(grid_φ_EE, grid_φ_PC, ς_y, ς_π)

        for k in 1:N1, l in 1:N2, i in 1:Ns
            φ_EE_lag = grid_φ_EE[k]
            φ_PC_lag = grid_φ_PC[l]
            g = Gs[i]
            x0_zlb = [ς_φEE[k,l,i], ς_φPC[k,l,i]]
            x0_unc = ς_φPC[k,l,i]
            y, π, r, φ_EE, φ_PC = commitment_step(m, itp_y, itp_π, Ps, i, g, φ_EE_lag, φ_PC_lag, x0_zlb; x0_unc=x0_unc)
            ς_y_new[k,l,i]   = y
            ς_π_new[k,l,i]   = π
            ς_r_new[k,l,i]   = r
            ς_φEE_new[k,l,i] = φ_EE
            ς_φPC_new[k,l,i] = φ_PC
        end

        diff = max(
            maximum(abs.(ς_y_new   .- ς_y)),
            maximum(abs.(ς_π_new   .- ς_π)),
            maximum(abs.(ς_r_new   .- ς_r)),
            maximum(abs.(ς_φEE_new .- ς_φEE)),
            maximum(abs.(ς_φPC_new .- ς_φPC)),
        )
        if verbose && (iter % 10 == 0 || iter == 1)
            println("commit iter=$iter, diff=$diff")
        end
        ς_y    .= ς_y_new
        ς_π    .= ς_π_new
        ς_r    .= ς_r_new
        ς_φEE  .= ς_φEE_new
        ς_φPC  .= ς_φPC_new
    end

    if verbose
        println("commitment converged in $iter iterations (final diff=$diff)")
    end

    return (y=ς_y, π=ς_π, r_n=ς_r, φ_EE=ς_φEE, φ_PC=ς_φPC,
            grid_φ_EE=collect(grid_φ_EE), grid_φ_PC=collect(grid_φ_PC),
            Gs=collect(Gs), Ps=Matrix(Ps), iter=iter, diff=diff)
end
