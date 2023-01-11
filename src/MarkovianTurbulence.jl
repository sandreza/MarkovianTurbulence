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

export RandomGeneratorMatrix
using MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, count_operator
using Distributions, LinearAlgebra
struct RandomGeneratorMatrix{S,T,V}
    dt::S
    erlang_distributions::T
    binomial_distributions::V
end

function RandomGeneratorMatrix(markov_chain, number_of_states; dt=1)
    ht = holding_times(markov_chain, number_of_states; dt=dt)
    erlang_distributions = Erlang{Float64}[]
    for i in 1:number_of_states
        ht_local = ht[i]
        if length(ht_local) == 0
            push!(erlang_distributions, Erlang(1, 0))
        else
            push!(erlang_distributions, Erlang(length(ht_local), mean(ht_local) / length(ht_local)))
        end
    end
    # off-diagonal probabilities 
    count_matrix = count_operator(markov_chain, number_of_states)
    count_matrix = count_matrix - Diagonal(count_matrix)
    Ntotes = sum(count_matrix, dims=1)
    pmatrix = count_matrix ./ Ntotes
    # NEED ERROR HANDLING HERE, NEED BETTER ERROR HANDLING
    for (i, csum) in enumerate(Ntotes)
        if csum == 0
            # choice 1
            # random_Q[:, i] .= 0 
            # choice 2
            pmatrix[:, i] .= 0
            pmatrix[i, i] = 1.0
        end
    end
    binomial_distributions = Binomial{Float64}[]
    for j in 1:number_of_states, i in 1:number_of_states
        if Ntotes[i] == 0
            push!(binomial_distributions, Binomial(1, 0))
        else
            push!(binomial_distributions, Binomial(Ntotes[j], pmatrix[i, j]))
        end
    end
    return RandomGeneratorMatrix(dt, erlang_distributions, binomial_distributions)
end

RandomGeneratorMatrix(markov_chain; dt=1) = RandomGeneratorMatrix(markov_chain, maximum(markov_chain); dt=dt)

import Base: rand
function rand(Q::RandomGeneratorMatrix)
    (; dt, erlang_distributions, binomial_distributions) = Q
    n_states = length(erlang_distributions)
    random_Q = zeros(n_states, n_states)
    scaling = reshape(1 ./ rand.(erlang_distributions), (1, n_states))
    random_Q[:] .= rand.(binomial_distributions)
    # need error handling here the same as before
    column_sum = sum(random_Q, dims=1)
    random_Q .= random_Q ./ column_sum
    random_Q -= I
    random_Q .*= scaling
    # error handling
    for (i, csum) in enumerate(column_sum)
        if csum == 0
            # choice 1
            # random_Q[:, i] .= 0 
            # choice 2
            random_Q[:, i] .= 1 / (length(column_sum) - 1) / dt
            random_Q[i, i] = -1 / dt
        end
    end
    return random_Q
end

rand(Q::RandomGeneratorMatrix, N::Int) = [rand(Q) for i in 1:N]


end # module MarkovianTurbulence
