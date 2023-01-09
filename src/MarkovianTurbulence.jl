module MarkovianTurbulence

using ProgressBars

export lorenz!, rk4
# generate data
function lorenz!(ṡ, s)
    ṡ[1] = 10.0 * (s[2] - s[1])
    ṡ[2] = s[1] * (28.0 - s[3]) - s[2]
    ṡ[3] = s[1] * s[2] - (8 / 3) * s[3]
    return nothing
end

function rk4(f, s, dt)
    ls = length(s)
    k1 = zeros(ls)
    k2 = zeros(ls)
    k3 = zeros(ls)
    k4 = zeros(ls)
    f(k1, s)
    f(k2, s + k1 * dt / 2)
    f(k3, s + k2 * dt / 2)
    f(k4, s + k3 * dt)
    return s + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
end

export autocovariance, autocorrelation

function autocovariance(x; timesteps=length(x))
    μ = mean(x)
    autocor = zeros(timesteps)
    for i in ProgressBar(1:timesteps)
        autocor[i] = mean(x[i:end] .* x[1:end-i+1]) - μ^2
    end
    return autocor
end

function autocorrelation(x; timesteps=length(x))
    μ = mean(x)
    autocor = autocovariance(x; timesteps=timesteps)
    return autocor ./ μ^2
end

function autocovariance(g⃗, Q, timelist)
    autocov = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = eigen(Q)
    p = steady_state(Q)
    v1 = V \ (p .* g⃗)
    w1 = g⃗' * V
    μ = sum(p .* g⃗)
    for i in ProgressBar(eachindex(timelist))
        autocov[i] = real(w1 * (exp.(Λ .* tlist[i]) .* v1)) - μ^2
    end
    return autocov
end

function autocorrelation(g⃗, Q, timelist)
    autocor = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = eigen(Q)
    p = steady_state(Q)
    v1 = V \ (p .* g⃗)
    w1 = g⃗' * V
    μ = sum(p .* g⃗)
    for i in ProgressBar(eachindex(timelist))
        autocor[i] = real(w1 * (exp.(Λ .* tlist[i]) .* v1)) / μ^2 - 1
    end
    return autocor
end

end # module MarkovianTurbulence
