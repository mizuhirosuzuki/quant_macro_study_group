function generate_capital_grid(m::Model, r=nothing, wage=nothing)

    if (r === nothing) && (wage === nothing)
        phi = 0.0
    else
        if r <= 0.0
            phi = m.b
        else
            phi = min(m.b, wage * m.s[1] / r)
        end
    end

    minK = -phi

    gridk = zeros(m.Nk)
    gridk[1] = minK
    for kc in 2:m.Nk
        gridk[kc] = gridk[1] + (m.maxK - minK) * ((kc - 1) / (m.Nk - 1))^m.curvK
    end

    gridk2 = zeros(m.Nk2)
    gridk2[1] = minK
    for kc in 2:m.Nk2
        gridk2[kc] = gridk2[1] + (m.maxK - minK) * ((kc - 1) / (m.Nk2 - 1))^m.curvK
    end

    return gridk, gridk2
end

function translate_capital_grid(m::Model, grids)

    gridk, gridk2 = grids

    kc1vec = zeros(m.Nk2)
    kc2vec = zeros(m.Nk2)
    prk1vec = zeros(m.Nk2)
    prk2vec = zeros(m.Nk2)

    for kc in 1:m.Nk2

        xx = gridk2[kc]

        if xx >= gridk[m.Nk]

            kc1vec[kc] = m.Nk
            kc2vec[kc] = m.Nk
            prk1vec[kc] = 1.0
            prk2vec[kc] = 0.0

        else

            ind = 1
            while xx > gridk[ind+1]
                ind += 1
                if ind + 1 >= m.Nk
                    break
                end
            end

            kc1vec[kc] = ind

            if ind < m.Nk
                kc2vec[kc] = ind + 1
                dK = (xx - gridk[ind]) / (gridk[ind+1] - gridk[ind])
                prk1vec[kc] = 1.0 - dK
                prk2vec[kc] = dK
            else
                kc2vec[kc] = ind
                prk1vec[kc] = 1.0
                prk2vec[kc] = 0.0
            end
        end
    end

    return kc1vec, kc2vec, prk1vec, prk2vec
end
