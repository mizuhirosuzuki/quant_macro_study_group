"""
NewKeynesianモデルのパラメータを格納する構造体。
discNK.ipynb のオリジナル `Model` と互換だが、
2状態モデル（マークアップショック u を使わないケース）にも対応するために
sigu, rhou, Nu などはオプション扱いとして 0/1 にしておけば無視できる。
"""
struct Model{TI<:Integer, TF<:AbstractFloat}
    rstar::TF   # pH=0 のときの定常名目金利（％, 四半期）
    bet::TF     # 割引率
    sig::TF     # IS の実質金利弾力性
    alp::TF     # 価格据え置き企業の比率（Calvo パラメータ）
    the::TF     # 需要の価格弾力性
    ome::TF     # 限界費用の出力弾力性
    kap::TF     # フィリップス曲線の傾き
    lam::TF     # 損失関数における出力ギャップの重み
    rhou::TF    # マークアップショックの AR 係数
    rhog::TF    # 実質金利ショックの AR 係数
    sigu::TF    # マークアップショックの標準偏差
    sigg::TF    # 実質金利ショックの標準偏差
    Nu::TI      # マークアップショックグリッド数
    Ng::TI      # 実質金利ショックグリッド数
    maxiter::TI # 反復回数の最大値
    tol::TF     # 収束基準
end

"""
    default_model(; kwargs...)

discNK.ipynb のカリブレーション値をデフォルトとする `Model` を返すヘルパ。
キーワード引数で個別に上書き可能。
"""
function default_model(;
    rstar::Float64 = 3.5/4,
    sig::Float64   = 6.25,
    alp::Float64   = 0.66,
    the::Float64   = 7.66,
    ome::Float64   = 0.47,
    lam::Float64   = 0.048/16,
    rhou::Float64  = 0.0,
    rhog::Float64  = 0.8,
    sigu::Float64  = 0.154,
    sigg::Float64  = 1.524,
    Nu::Int        = 31,
    Ng::Int        = 31,
    maxiter::Int   = 2000,
    tol::Float64   = 1e-5,
)
    bet = 1/(1+rstar/100)
    kap = (1-alp)*(1-alp*bet)/alp * (1/sig + ome) / (1 + ome*the)
    return Model(rstar, bet, sig, alp, the, ome, kap, lam,
                 rhou, rhog, sigu, sigg, Nu, Ng, maxiter, tol)
end
