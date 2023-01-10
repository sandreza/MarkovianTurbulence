using GLMakie
using MarkovChainHammer, MarkovianTurbulence
using Distributions

using ProgressBars, LinearAlgebra, Statistics, Random

import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, count_operator
import MarkovChainHammer.Trajectory: generate
import MarkovChainHammer.Utils: histogram

fixed_points = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
markov_states = fixed_points

initial_condition = [14.0, 20.0, 27.0]
dt = 0.005
iterations = 10^4

timeseries = zeros(3, iterations)
markov_chain = zeros(Int, iterations)
timeseries[:, 1] .= initial_condition

markov_index = argmin([norm(initial_condition - markov_state) for markov_state in markov_states])
markov_chain[1] = markov_index
for i in ProgressBar(2:iterations)
    # take one timestep forward via Runge-Kutta 4
    state = rk4(lorenz!, timeseries[:, i-1], dt)
    timeseries[:, i] .= state
    # partition state space according to most similar markov state
    local markov_index = argmin([norm(state - markov_state) for markov_state in markov_states])
    markov_chain[i] = markov_index
end

##
function symmetry13(state)
    if state == 1
        return 3
    elseif state == 2
        return 2
    else
        return 1
    end
end
s_markov_chain = symmetry13.(markov_chain)
omarkov_chain = copy(markov_chain)
markov_chain = [omarkov_chain... s_markov_chain...]
## construct transition matrix
Q = generator(markov_chain; dt=dt)
p = steady_state(Q)
ht = holding_times(markov_chain; dt=dt)
##
# holding times
erlang_distributions = []
for i in eachindex(ht)
    ht_local = ht[i] 
    if length(ht) == 0
        push!(erlang_distributions, Erlang(1, 0))
    end
    push!(erlang_distributions, Erlang(length(ht_local), mean(ht_local)/length(ht_local)))
end
# off-diagonal probabilities 
count_matrix = count_operator(markov_chain)
count_matrix = count_matrix - Diagonal(count_matrix)
Ntotes = sum(count_matrix, dims = 1)
pmatrix = count_matrix ./ Ntotes
binomial_distributions = []
for j in eachindex(ht), i in eachindex(ht)
    if Ntotes[i]== 0
        push!(binomial_distributions, Binomial(1, 0))
    end
    push!(binomial_distributions, Binomial(Ntotes[j], pmatrix[i,j]))
end
## random generator
Qs = []
for i in ProgressBar(1:1000)
    random_Q = similar(Q)
    scaling = reshape(1 ./ rand.(erlang_distributions), (1, size(Q)[1]))
    random_Q[:] .= rand.(binomial_distributions)
    # need error handling here the same as before
    column_sum = sum(random_Q, dims=1)
    random_Q .= random_Q ./ column_sum
    random_Q -= I
    random_Q .*= scaling
    for (i, csum) in enumerate(column_sum)
        if csum == 0
            # random_Q[:, i] .= 0 # choice 1
            # choice 2
            random_Q[:, i] .= 1 / (length(column_sum) - 1) / dt_days
            random_Q[i, i] = -1 / dt_days
        end
    end

    push!(Qs, random_Q)
end

##
Q
Qs[1]
mean(Qs)
##
Λs = eigvals.(Qs)
##
distΛ2 = [1 ./ real(Λ[1]) for Λ in Λs]
distΛ3 = [1 ./ real(Λ[end-1]) for Λ in Λs]
x2, y2 = histogram(distΛ2, bins = 10)
x3, y3 = histogram(distΛ3, bins= 10)
fig = Figure() 
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1, 2])
barplot!(ax1, x2, y2, color = :blue)
barplot!(ax2, x3, y3, color=:blue)
display(fig)

##
fig = Figure() 
ax = Axis(fig[1,1])
ylims!(ax, 0, 4)
N = 100 # N is at most 100
for i in 1:N
    scatter!(ax, -1 ./ real(Λs[i]), color = (:black, 1))
end
display(fig)