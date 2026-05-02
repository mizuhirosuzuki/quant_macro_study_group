using LinearAlgebra: dot

"""
最適裁量政策（discretionary policy）の時間反復法による求解。

discNK.ipynb の `ti()` を以下のように分解した:
- `expectations` : 古い政策関数とショック遷移確率から (E[y'], E[π']) を計算
- `discretionary_unconstrained` : ZLB が拘束しないと仮定した最適化
- `discretionary_zlb` : ZLB が拘束 (r=0) すると仮定した最適化
- `discretionary_step` : 上記2つを場合分けで使い分ける
- `solve_discretionary` : すべての状態に対して反復し政策関数の収束を確認
"""

"""
    expectations(Ps_row, yvec, pvec) -> (ye, pe)

遷移確率 `Ps_row`（長さ Ns の確率ベクトル）と古い政策関数の値ベクトル
`yvec`, `pvec`（長さ Ns）から、来期の期待値 (E[y'], E[π']) を計算する。
"""
function expectations(Ps_row::AbstractVector, yvec::AbstractVector, pvec::AbstractVector)
    ye  = dot(Ps_row, yvec)
    pe  = dot(Ps_row, pvec)
    return ye, pe
end

"""
    discretionary_unconstrained(m, ye, pe, g, u) -> (y, π, r_n)

ZLB が拘束しないと仮定したときの裁量政策の解。
discNK.ipynb の最適化部分に対応。

最適化条件 (FOC): π = -(λ/κ) y, 即ち y = -(κ/λ) π
これと PC: π = κy + β·E[π'] + u を連立すると:
    π = (β·E[π'] + u) / (1 + κ²/λ)
"""
function discretionary_unconstrained(m::Model, ye::Real, pe::Real, g::Real, u::Real)
    p0 = (m.bet * pe + u) / (1 + m.kap^2 / m.lam)
    y0 = -(m.kap / m.lam) * p0
    r0 = (1 / m.sig) * (ye - y0 + g) + pe
    return y0, p0, r0
end

"""
    discretionary_zlb(m, ye, pe, g, u) -> (y, π, r_n)

ZLB が拘束 (r_n = 0) すると仮定したときの裁量政策の解。
IS 式に r_n = 0 を代入して y を求め、PC 式から π を求める。
"""
function discretionary_zlb(m::Model, ye::Real, pe::Real, g::Real, u::Real)
    r0 = 0.0
    y0 = ye - m.sig * (r0 - pe) + g
    p0 = m.kap * y0 + m.bet * pe + u
    return y0, p0, r0
end

"""
    discretionary_step(m, ye, pe, g, u) -> (y, π, r_n)

非拘束解を試し、r_n < 0 なら ZLB 拘束解に切り替える。
"""
function discretionary_step(m::Model, ye::Real, pe::Real, g::Real, u::Real)
    y0, p0, r0 = discretionary_unconstrained(m, ye, pe, g, u)
    if r0 < 0
        y0, p0, r0 = discretionary_zlb(m, ye, pe, g, u)
    end
    return y0, p0, r0
end

"""
    policy_diff(new, old) -> Float64

新旧の政策関数ベクトル間の最大絶対誤差。
"""
function policy_diff(new::AbstractArray, old::AbstractArray)
    return maximum(abs.(new .- old))
end

"""
    solve_discretionary(m; Gs, Ps, verbose=false) -> (yvec, pvec, rvec, iter)

任意のショックグリッド `Gs`（Ns×2 行列、列1=g, 列2=u）と遷移行列 `Ps` (Ns×Ns)
に対して、裁量政策を時間反復法で解く。

戻り値:
- `yvec`, `pvec`, `rvec`: 各状態における y, π, r_n の値（長さ Ns）
- `iter`: 収束に要した反復回数
"""
function solve_discretionary(m::Model; Gs::AbstractMatrix, Ps::AbstractMatrix,
                              verbose::Bool=false)
    Ns = size(Gs, 1)
    @assert size(Gs, 2) == 2 "Gs must be Ns x 2 (columns: g, u)"
    @assert size(Ps) == (Ns, Ns)

    yvec0 = zeros(Ns); yvec1 = zeros(Ns)
    pvec0 = zeros(Ns); pvec1 = zeros(Ns)
    rvec0 = zeros(Ns); rvec1 = zeros(Ns)

    diff = Inf
    iter = 0
    while diff > m.tol && iter < m.maxiter
        iter += 1
        for is in 1:Ns
            g0 = Gs[is, 1]
            u0 = Gs[is, 2]
            ye, pe = expectations(view(Ps, is, :), yvec0, pvec0)
            y0, p0, r0 = discretionary_step(m, ye, pe, g0, u0)
            yvec1[is] = y0
            pvec1[is] = p0
            rvec1[is] = r0
        end
        diff = max(policy_diff(yvec1, yvec0),
                   policy_diff(pvec1, pvec0),
                   policy_diff(rvec1, rvec0))
        if verbose && (iter % 50 == 0 || iter == 1)
            println("iter=$iter, diff=$diff")
        end
        yvec0 = copy(yvec1)
        pvec0 = copy(pvec1)
        rvec0 = copy(rvec1)
    end

    if verbose
        println("converged in $iter iterations (diff=$diff)")
    end
    return yvec0, pvec0, rvec0, iter
end
