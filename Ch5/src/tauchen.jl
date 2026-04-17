using Distributions

function tauchen(N, rho, sigma, m)
    Zprob = zeros(N, N)
    d = Normal(0, 1)

    zmax = m * sqrt(sigma^2 / (1 - rho^2))
    zmin = -zmax
    w = (zmax - zmin) / (N - 1)

    Z = collect(range(zmin, zmax, length=N))

    for j in 1:N
        for k in 1:N
            if k == 1
                Zprob[j, k] = cdf(d, (Z[k] - rho * Z[j] + w / 2) / sigma)
            elseif k == N
                Zprob[j, k] = 1 - cdf(d, (Z[k] - rho * Z[j] - w / 2) / sigma)
            else
                Zprob[j, k] = (
                    cdf(d, (Z[k] - rho * Z[j] + w / 2) / sigma)
                    - cdf(d, (Z[k] - rho * Z[j] - w / 2) / sigma)
                )
            end
        end
    end

    dist0 = (1 / N) .* ones(N)
    dist1 = copy(dist0)
    err = 1.0
    errtol = 1e-8
    while err > errtol
        dist1 = Zprob' * dist0
        err = sum(abs.(dist0 - dist1))
        dist0 = copy(dist1)
    end

    Zinv = copy(dist1)

    return Z, Zprob, Zinv
end
