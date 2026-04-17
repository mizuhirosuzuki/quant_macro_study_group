@inline function interp_value(k_prime, gridk, v_row)
    Nk = length(gridk)
    if k_prime <= gridk[1]
        return v_row[1]
    elseif k_prime >= gridk[Nk]
        return v_row[Nk]
    end
    ind = searchsortedlast(gridk, k_prime)
    weight = (k_prime - gridk[ind]) / (gridk[ind+1] - gridk[ind])
    return (1 - weight) * v_row[ind] + weight * v_row[ind+1]
end

@inline function neg_bellman(kp, cash_on_hand, mu, beta, gridk, Ev_row)
    cons = cash_on_hand - kp
    if cons <= 0.0
        return 1e10
    end
    util = (cons^(1 - mu)) / (1 - mu)
    vpr = interp_value(kp, gridk, Ev_row)
    return -(util + beta * vpr)
end

function golden_section_search(f, a, b; tol=1e-6)
    gr = (sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = f(c)
    fd = f(d)
    while abs(b - a) > tol
        if fc < fd
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = f(c)
        else
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = f(d)
        end
    end
    x_opt = (a + b) / 2
    return x_opt, f(x_opt)
end

# Specialized fused golden-section + Bellman search.
# Avoids closure boxing for the hot inner loop in VFI.
function golden_section_bellman(a, b, cash_on_hand, mu, beta, gridk, Ev_row; tol=1e-6)
    gr = (sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc = neg_bellman(c, cash_on_hand, mu, beta, gridk, Ev_row)
    fd = neg_bellman(d, cash_on_hand, mu, beta, gridk, Ev_row)
    while abs(b - a) > tol
        if fc < fd
            b = d
            d = c
            fd = fc
            c = b - (b - a) / gr
            fc = neg_bellman(c, cash_on_hand, mu, beta, gridk, Ev_row)
        else
            a = c
            c = d
            fc = fd
            d = a + (b - a) / gr
            fd = neg_bellman(d, cash_on_hand, mu, beta, gridk, Ev_row)
        end
    end
    x_opt = (a + b) / 2
    f_opt = neg_bellman(x_opt, cash_on_hand, mu, beta, gridk, Ev_row)
    return x_opt, f_opt
end

function find_nearest_grid_index(k_prime, gridk2)
    N = length(gridk2)
    idx = searchsortedfirst(gridk2, k_prime)
    if idx == 1
        return 1
    elseif idx > N
        return N
    else
        return abs(gridk2[idx] - k_prime) < abs(gridk2[idx-1] - k_prime) ? idx : idx - 1
    end
end

# Ev[lc, k] = E[v(k, l') | l = lc] = sum_lcc prob[lc, lcc] * v[lcc, k]
function compute_expected_value!(Ev, m::Model, v)
    fill!(Ev, 0.0)
    for lc in 1:m.Nl
        for k in 1:m.Nk
            acc = 0.0
            for lcc in 1:m.Nl
                acc += m.prob[lc, lcc] * v[lcc, k]
            end
            Ev[lc, k] = acc
        end
    end
    return nothing
end

function solve_VFI_continuous(m::Model, r0, K0, wage, grids; tau=0)

    gridk, gridk2 = grids

    kfunG = zeros(m.Nl, m.Nk)
    kfun = zeros(m.Nl, m.Nk)
    v = zeros(m.Nl, m.Nk)
    tv = zeros(m.Nl, m.Nk)
    Ev = zeros(m.Nl, m.Nk)

    err = 10.0
    maxiter = 2000
    iter = 1
    errTol = 1e-5

    minK = gridk2[1]
    mu = m.mu
    beta = m.beta

    while (err > errTol) & (iter < maxiter)

        # precompute expectation once per VFI iteration
        compute_expected_value!(Ev, m, v)

        for kc in 1:m.Nk
            for lc in 1:m.Nl

                cash_on_hand = (
                    m.s[lc] * wage
                    + (1 + r0 * (1 - tau)) * gridk[kc]
                    + tau * r0 * K0
                )

                kp_max = min(cash_on_hand - 1e-10, gridk2[end])

                if kp_max <= minK
                    tv[lc, kc] = -1e10
                    kfun[lc, kc] = minK
                    continue
                end

                Ev_row = @view Ev[lc, :]
                kp_opt, neg_v_opt = golden_section_bellman(
                    minK, kp_max, cash_on_hand, mu, beta, gridk, Ev_row,
                )
                tv[lc, kc] = -neg_v_opt
                kfun[lc, kc] = kp_opt
                # kfunG is left as zeros — the continuous distribution
                # iteration uses kfun directly, bypassing gridk2 snapping.

            end
        end

        err = maximum(abs.(tv - v))
        # swap buffer pointers — next iteration will overwrite tv anyway
        v, tv = tv, v
        iter += 1

    end

    if iter == maxiter
        println("WARNING!! VFI continuous: iteration reached max: iter=$iter, err=$err")
    end

    return kfun, kfunG, v
end
