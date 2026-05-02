using Interpolations

"""
決定論的なショック系列に対する裁量・コミット両政策の経路シミュレーション。
教科書 図7.4 を再現するためのもの。
"""

"""
    simulate_discretionary(yvec, pvec, rvec, shock_idx) -> NamedTuple

裁量政策の方策関数 (yvec, pvec, rvec; 各 Ns ベクトル) と
ショックインデックス系列 `shock_idx::Vector{Int}` から
時系列を構成する。裁量では内生状態がないので単純なルックアップ。
"""
function simulate_discretionary(yvec::AbstractVector, pvec::AbstractVector,
                                 rvec::AbstractVector, shock_idx::AbstractVector{<:Integer})
    T = length(shock_idx)
    y = [yvec[shock_idx[t]] for t in 1:T]
    π = [pvec[shock_idx[t]] for t in 1:T]
    r = [rvec[shock_idx[t]] for t in 1:T]
    return (y=y, π=π, r_n=r)
end

"""
    simulate_commitment(commit, shock_idx; φ_EE0=0.0, φ_PC0=0.0) -> NamedTuple

コミット政策の方策関数（`solve_commitment` の戻り値）から、
内生状態 (φ_EE,-1, φ_PC,-1) を初期値 `(φ_EE0, φ_PC0)` で開始し、
決定論的ショック系列 `shock_idx` のもとで前向きに更新したパスを返す。

各 t において:
1. 現在の (φ_EE,-1, φ_PC,-1, s_t) で各方策関数を 2D 補間して y_t, π_t, r_n_t, φ_EE_t, φ_PC_t を取得
2. 状態更新: (φ_EE,-1, φ_PC,-1) ← (φ_EE_t, φ_PC_t)
"""
function simulate_commitment(commit::NamedTuple, shock_idx::AbstractVector{<:Integer};
                              φ_EE0::Real=0.0, φ_PC0::Real=0.0)
    grid_φ_EE = commit.grid_φ_EE
    grid_φ_PC = commit.grid_φ_PC
    Ns = length(commit.Gs)

    # 5 系列 × Ns 補間器
    function build_itp_set(arr3d::AbstractArray{<:Real,3})
        return [extrapolate(interpolate((grid_φ_EE, grid_φ_PC), arr3d[:,:,j], Gridded(Linear())), Flat())
                for j in 1:Ns]
    end
    itp_y    = build_itp_set(commit.y)
    itp_π    = build_itp_set(commit.π)
    itp_r    = build_itp_set(commit.r_n)
    itp_φEE  = build_itp_set(commit.φ_EE)
    itp_φPC  = build_itp_set(commit.φ_PC)

    T = length(shock_idx)
    y    = zeros(T)
    π    = zeros(T)
    r_n  = zeros(T)
    φ_EE = zeros(T)
    φ_PC = zeros(T)

    φ_EE_lag = float(φ_EE0)
    φ_PC_lag = float(φ_PC0)
    for t in 1:T
        i = shock_idx[t]
        y[t]    = itp_y[i](φ_EE_lag, φ_PC_lag)
        π[t]    = itp_π[i](φ_EE_lag, φ_PC_lag)
        r_n[t]  = itp_r[i](φ_EE_lag, φ_PC_lag)
        φ_EE[t] = itp_φEE[i](φ_EE_lag, φ_PC_lag)
        φ_PC[t] = itp_φPC[i](φ_EE_lag, φ_PC_lag)
        φ_EE_lag = φ_EE[t]
        φ_PC_lag = φ_PC[t]
    end
    return (y=y, π=π, r_n=r_n, φ_EE=φ_EE, φ_PC=φ_PC)
end

"""
    crisis_shock_path(T, t_switch; H_idx=1, L_idx=2) -> Vector{Int}

教科書 図7.4 のショック系列を構成: t = 1..t_switch-1 の間は危機状態 (L)、
t ≥ t_switch は平常状態 (H)。デフォルト規約は `build_two_state_chain` と整合的に
状態 1 = H, 状態 2 = L。
"""
function crisis_shock_path(T::Integer, t_switch::Integer; H_idx::Integer=1, L_idx::Integer=2)
    return Int[t < t_switch ? L_idx : H_idx for t in 1:T]
end
