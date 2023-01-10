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
iterations = 2 * 10^7

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
# markov_chain = [omarkov_chain... s_markov_chain...]
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
    push!(erlang_distributions, Erlang(length(ht_local), mean(ht_local) / length(ht_local)))
end
# off-diagonal probabilities 
count_matrix = count_operator(markov_chain)
count_matrix = count_matrix - Diagonal(count_matrix)
Ntotes = sum(count_matrix, dims=1)
pmatrix = count_matrix ./ Ntotes
binomial_distributions = []
for j in eachindex(ht), i in eachindex(ht)
    if Ntotes[i] == 0
        push!(binomial_distributions, Binomial(1, 0))
    end
    push!(binomial_distributions, Binomial(Ntotes[j], pmatrix[i, j]))
end
##


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
            # choice 1
            # random_Q[:, i] .= 0 
            # choice 2
            random_Q[:, i] .= 1 / (length(column_sum) - 1) / dt
            random_Q[i, i] = -1 / dt
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
x2, y2 = histogram(distΛ2, bins=10)
x3, y3 = histogram(distΛ3, bins=10)
fig = Figure()
ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
barplot!(ax1, x2, y2, color=:blue)
barplot!(ax2, x3, y3, color=:blue)
display(fig)

##
fig = Figure()
ax = Axis(fig[1, 1])
ylims!(ax, 0, 4)
N = 100 # N is at most 100
for i in 1:N
    scatter!(ax, -1 ./ real(Λs[i]), color=(:black, 1))
end
display(fig)


##
# Abstraction prototype 

struct RandomGeneratorMatrix{S,T,V}
    dt::S
    erlang_distributions::T
    binomial_distributions::V
end

function RandomGeneratorMatrix(markov_chain, number_of_states; dt=1)
    ht = holding_times(markov_chain, number_of_states; dt=dt)
    erlang_distributions = Erlang{Float64}[]
    for i in eachindex(ht)
        ht_local = ht[i]
        if length(ht) == 0
            push!(erlang_distributions, Erlang(1, 0))
        end
        push!(erlang_distributions, Erlang(length(ht_local), mean(ht_local) / length(ht_local)))
    end
    # off-diagonal probabilities 
    count_matrix = count_operator(markov_chain, number_of_states)
    count_matrix = count_matrix - Diagonal(count_matrix)
    Ntotes = sum(count_matrix, dims=1)
    pmatrix = count_matrix ./ Ntotes
    binomial_distributions = Binomial{Float64}[]
    for j in eachindex(ht), i in eachindex(ht)
        if Ntotes[i] == 0
            push!(binomial_distributions, Binomial(1, 0))
        end
        push!(binomial_distributions, Binomial(Ntotes[j], pmatrix[i, j]))
    end
    return RandomGeneratorMatrix(dt, erlang_distributions, binomial_distributions)
end

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

##
Q1 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^4)], 3; dt=dt)
Q2 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^5)], 3; dt=dt)
Q3 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^6)], 3; dt=dt)
num_samples = 100000
Q1s = rand(Q1, num_samples)
Q2s = rand(Q2, num_samples)
Q3s = rand(Q3, num_samples)

# Λ1s = eigvals.(Q1s)
# Λ2s = eigvals.(Q2s)
# Λ3s = eigvals.(Q3s)

# observable1(Q) = -real.(eigvals(Q))[2]
# observable2(Q) = -real.(eigvals(Q))[1]
observable1(Q) = Q[1, 2]
observable2(Q) = Q[2, 1]
obs1 = [observable1.(Qs) for Qs in [Q1s, Q2s, Q3s]]
obs2 = [observable2.(Qs) for Qs in [Q1s, Q2s, Q3s]]
##Q1 
Nbins = 1000
xrange = quantile.(Ref(obs1[1]), (0.0001, 0.9999))
xy1 = [histogram(obs1[i], bins=Nbins, custom_range=xrange) for i in 1:3]
xrange = quantile.(Ref(obs2[1]), (0.0001, 0.9999))
xy2 = [histogram(obs2[i], bins=Nbins, custom_range=xrange) for i in 1:3]
##
Q = generator(markov_chain[10^6+1:end]; dt=dt)
best1 = observable1(Q)
best2 = observable2(Q)
##
Qemp1 = generator(markov_chain[1:floor(Int, 2 * 10^4)]; dt=dt)
emp1_1 = observable1(Qemp1)
emp1_2 = observable2(Qemp1)

Qemp2 = generator(markov_chain[1:floor(Int, 2 * 10^5)]; dt=dt)
emp2_1 = observable1(Qemp2)
emp2_2 = observable2(Qemp2)

Qemp3 = generator(markov_chain[1:floor(Int, 2 * 10^6)]; dt=dt)
emp3_1 = observable1(Qemp3)
emp3_2 = observable2(Qemp3)

emplist1 = [emp1_1, emp2_1, emp3_1]
emplist2 = [emp1_2, emp2_2, emp3_2]
##
fig = Figure(resolution=(2225, 1284))
labelsize = 40
options = (; xlabel="eigenvalue", ylabel="probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)

ax1s = [Axis(fig[i, 1]; title="Eigenvalue 1: T = 1e$(i+1) samples", options...) for i in 1:3]
for (i, ax) in enumerate(ax1s)
    barplot!(ax, xy1[i]..., color=:blue)
    vlines!(ax, best1; color=:red, linewidth=4, label="best empirical")
    vlines!(ax1s[i], emplist1[i]; color=:orange, linewidth=4, label="empirical")
end

axislegend(ax1s[1], position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)

ax2s = [Axis(fig[i, 2]; title="Eigenvalue 2: N = 10^$(i+3) samples", options...) for i in 1:3]
for (i, ax) in enumerate(ax2s)
    barplot!(ax, xy2[i]..., color=:blue)
    vlines!(ax, best2; color=:red, linewidth=4, label="best empirical")
    vlines!(ax2s[i], emplist2[i]; color=:orange, linewidth=4, label="empirical")
end

display(fig)

## 
empirical_choice = [-1 / eigvals(generator(markov_chain[1:floor(Int, 10^i)]; dt=dt))[2] for i in 4:0.1:7]


##
# Q1 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^4)], 3; dt=dt)
# Q2 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^5)], 3; dt=dt)
# Q3 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^6)], 3; dt=dt)
Q1 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^(4+0))], 3; dt=dt)
Q2 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^(5+0))], 3; dt=dt)
Q3 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^(6+0))], 3; dt=dt)
Q4 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^(6 + 1))], 3; dt=dt)
num_samples = 100000
Q1s = rand(Q1, num_samples)
Q2s = rand(Q2, num_samples)
Q3s = rand(Q3, num_samples)
Q4s = rand(Q4, num_samples)
Qs = [Q1s, Q2s, Q3s, Q4s]

Q = generator(markov_chain[10^6+1:end]; dt=dt)
##
observables = [Q -> Q[i] for i in 1:9]
obs = [observable.(Q) for observable in observables, Q in Qs]
best_empirical = [observable(Q) for observable in observables]
##
Nbins = 500
xys = []
for i in 1:9
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]..., obs[i, 3]...]), (0.0001, 0.9999))
    xy = [histogram(obs[i, j], bins=Nbins, custom_range=xrange) for j in eachindex(Qs)]
    push!(xys, xy)
end

##
fig = Figure(resolution=(3000, 1500))
labelsize = 40
options = (; xlabel=" ", ylabel="probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
titlenames = ["Q₁₁", "Q₂₁", "Q₃₁", "Q₁₂", "Q₂₂", "Q₃₂", "Q₁₃", "Q₂₃", "Q₃₃"]

axs = []
for i in 1:9
    ii = (i - 1) % 3 + 1
    jj = (i - 1) ÷ 3 + 1
    ax = Axis(fig[ii, jj]; title=titlenames[i], options...)
    push!(axs, ax)
    barplot!(ax, xys[i][1]..., color=(:blue, 0.75), label="T=100")
    barplot!(ax, xys[i][2]..., color=(:orange, 0.75), label="T=1000")
    barplot!(ax, xys[i][3]..., color=(:black, 0.75), label="T=10000")
    barplot!(ax, xys[i][4]..., color=(:red,   0.75), label="T=100000")
    ylims!(ax, (-0.01, 0.1))
end

# xlims!(axs[1])

axislegend(axs[5], position=:lt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)

