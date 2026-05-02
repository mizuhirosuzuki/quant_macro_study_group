using Distributions

"""
標準正規分布の累積分布関数。Tauchen 法のヘルパ。
"""
function cdf_normal(x::Real)
    return cdf(Normal(0, 1), x)
end

"""
    tauchen(N, mu, rho, sigma, m) -> (Z, Zprob)

AR(1) 過程 z' = (1-ρ)μ + ρz + ε, ε~N(0, σ²) を Tauchen 法で N 点に離散化。

戻り値:
- `Z`: 離散グリッド（長さ N）
- `Zprob`: 遷移確率行列（N×N、行和=1）
"""
function tauchen(N::Integer, mu::Real, rho::Real, sigma::Real, m::Real)
    Zprob = zeros(N, N)
    c = (1 - rho) * mu

    zmax = m * sqrt(sigma^2 / (1 - rho^2))
    zmin = -zmax
    w = (zmax - zmin) / (N - 1)

    Z = collect(range(zmin, zmax, length=N))
    Z .+= mu

    for j in 1:N
        for k in 1:N
            if k == 1
                Zprob[j, k] = cdf_normal((Z[k] - c - rho*Z[j] + w/2) / sigma)
            elseif k == N
                Zprob[j, k] = 1 - cdf_normal((Z[k] - c - rho*Z[j] - w/2) / sigma)
            else
                Zprob[j, k] = cdf_normal((Z[k] - c - rho*Z[j] + w/2) / sigma) -
                              cdf_normal((Z[k] - c - rho*Z[j] - w/2) / sigma)
            end
        end
    end

    return Z, Zprob
end

"""
    build_joint_shock_grid(Gg, Pg, Gu, Pu) -> (Gs, Ps)

discNK.ipynb と同じ規約で、(g, u) の2変数ショックを1次元の状態 is に集約する。
インデックス規約: is = m.Nu * (ig - 1) + iu
- `Gs[is, 1] = Gg[ig]`, `Gs[is, 2] = Gu[iu]`
- `Ps = kron(Pg, Pu)`
"""
function build_joint_shock_grid(Gg::AbstractVector, Pg::AbstractMatrix,
                                Gu::AbstractVector, Pu::AbstractMatrix)
    Ng, Nu = length(Gg), length(Gu)
    Ns = Ng * Nu
    Gs = zeros(Ns, 2)
    for ig in 1:Ng, iu in 1:Nu
        is = Nu * (ig - 1) + iu
        Gs[is, 1] = Gg[ig]
        Gs[is, 2] = Gu[iu]
    end
    Ps = kron(Pg, Pu)
    return Gs, Ps
end

"""
    build_two_state_chain(sH, sL, pH, pL) -> (Gs, Ps)

2状態 Markov 連鎖。状態 1 = 平常時 (sH)、状態 2 = 危機 (sL)。
- pH: 平常から危機へ遷移する確率
- pL: 危機が継続する確率

戻り値:
- `Gs`: 長さ2のベクトル [sH, sL]
- `Ps`: 2×2 遷移行列。Ps[i, j] = i → j の確率
"""
function build_two_state_chain(sH::Real, sL::Real, pH::Real, pL::Real)
    Gs = [sH, sL]
    Ps = [1 - pH    pH;
          1 - pL    pL]
    return Gs, Ps
end
