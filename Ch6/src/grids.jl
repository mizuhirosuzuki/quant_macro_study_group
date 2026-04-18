function _power_grid(minA::Real, maxA::Real, N::Integer, curv::Real)
    g = zeros(N)
    g[1] = minA
    for i in 2:N
        g[i] = minA + (maxA - minA) * ((i - 1)/(N - 1))^curv
    end
    return g
end

# State (grida, size Na) and choice (grida2, size Na2) grids. No borrowing.
function generate_capital_grid(m::Model)
    minA = 0.0
    grida  = _power_grid(minA, m.maxA, m.Na,  m.curvA)
    grida2 = _power_grid(minA, m.maxA, m.Na2, m.curvA)
    return grida, grida2
end

# For each grida2 point, find the bracketing pair of grida points and the
# Young-style linear-interpolation weights. Used by both the backward Bellman
# (to value off-grid choices) and the forward distribution update.
function translate_capital_grid(m::Model, grids)
    grida, grida2 = grids
    ac1vec  = zeros(Int64, m.Na2)
    ac2vec  = zeros(Int64, m.Na2)
    pra1vec = zeros(m.Na2)
    pra2vec = zeros(m.Na2)

    for ac in 1:m.Na2
        xx = grida2[ac]
        if xx >= grida[m.Na]
            ac1vec[ac]  = m.Na
            ac2vec[ac]  = m.Na
            pra1vec[ac] = 1.0
            pra2vec[ac] = 0.0
        else
            ind = 1
            while xx > grida[ind+1]
                ind += 1
                ind + 1 >= m.Na && break
            end
            ac1vec[ac] = ind
            if ind < m.Na
                ac2vec[ac] = ind + 1
                dA = (xx - grida[ind])/(grida[ind+1] - grida[ind])
                pra1vec[ac] = 1.0 - dA
                pra2vec[ac] = dA
            else
                ac2vec[ac]  = ind
                pra1vec[ac] = 1.0
                pra2vec[ac] = 0.0
            end
        end
    end
    return ac1vec, ac2vec, pra1vec, pra2vec
end
