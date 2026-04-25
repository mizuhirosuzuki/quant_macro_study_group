# Stationary age distribution implied by survival probabilities s:
#   meaJ[1] ∝ 1,  meaJ[j+1] = s[j] · meaJ[j],  ∑ meaJ = 1.
# Using this for a Model makes the forward-distribution update mass-preserving.
function stationary_age_distribution(s::AbstractVector)
    Nj = length(s)
    w  = ones(eltype(s), Nj)
    for j in 2:Nj
        w[j] = w[j-1] * s[j-1]
    end
    return w ./ sum(w)
end

# Stationary OLG distribution given policy `afunG`. Single pass over ages:
# newborns at (age 1, zero assets), each age jc is read after being written
# (so we can use one array in place). Mass at age j+1 is scaled by s[j] to
# reflect survival.
function stationary_distribution(m::Model, afunG, capital_grid_translations)
    ac1vec, ac2vec, pra1vec, pra2vec = capital_grid_translations
    mea = zeros(m.Nj, m.Ne, m.Na)
    mea[1, :, 1] .= m.meaJ[1] / m.Ne

    @inbounds for jc in 1:m.Nj-1
        sj = m.s[jc]
        for ec in 1:m.Ne, ac in 1:m.Na
            meaX = mea[jc, ec, ac]
            meaX == 0.0 && continue
            acc  = afunG[jc, ec, ac]
            acc1 = ac1vec[acc]; acc2 = ac2vec[acc]
            pra1 = pra1vec[acc]; pra2 = pra2vec[acc]
            for ecc in 1:m.Ne
                p = m.Pe[ec, ecc] * meaX * sj
                mea[jc+1, ecc, acc1] += p * pra1
                mea[jc+1, ecc, acc2] += p * pra2
            end
        end
    end
    return mea
end

# Advance the distribution one time step: mea_new ← T(mea_old; afunG).
# Age jc at time t becomes age jc+1 at time t+1 with probability s[jc];
# newborns enter at age 1, zero assets, with mass meaJ[1].
function advance_distribution!(mea_new, mea_old,
                               m::Model, afunG, capital_grid_translations)
    ac1vec, ac2vec, pra1vec, pra2vec = capital_grid_translations
    fill!(mea_new, 0.0)
    mea_new[1, :, 1] .= m.meaJ[1] / m.Ne

    @inbounds for jc in 1:m.Nj-1
        sj = m.s[jc]
        for ec in 1:m.Ne, ac in 1:m.Na
            meaX = mea_old[jc, ec, ac]
            meaX == 0.0 && continue
            acc  = afunG[jc, ec, ac]
            acc1 = ac1vec[acc]; acc2 = ac2vec[acc]
            pra1 = pra1vec[acc]; pra2 = pra2vec[acc]
            for ecc in 1:m.Ne
                p = m.Pe[ec, ecc] * meaX * sj
                mea_new[jc+1, ecc, acc1] += p * pra1
                mea_new[jc+1, ecc, acc2] += p * pra2
            end
        end
    end
    return nothing
end
