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
Q1 = RandomGeneratorMatrix(markov_chain[1:floor(Int, 2 * 10^4)], 3; dt=dt)
Q2 = RandomGeneratorMatrix(markov_chain[floor(Int, 2 * 10^4)+1:floor(Int, 2 * 10^5)], 3; dt=dt)
Q3 = RandomGeneratorMatrix(markov_chain[floor(Int, 2 * 10^5)+1:floor(Int, 2 * 10^6)], 3; dt=dt)
Q4 = RandomGeneratorMatrix(markov_chain[floor(Int, 2 * 10^6)+1:floor(Int, 2 * 10^7)], 3; dt=dt)
num_samples = 100000
Q1s = rand(Q1, num_samples)
Q2s = rand(Q2, num_samples)
Q3s = rand(Q3, num_samples)
Q4s = rand(Q4, num_samples)
Qs = [Q1s, Q2s, Q3s, Q4s]

Q = generator(markov_chain[1:end]; dt=dt)
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
    ylims!(ax, (-0.001, 0.1))
end

xlims!(axs[1], (-1.5, -0.7))
xlims!(axs[2], (0.25, 0.65))
xlims!(axs[3], (0.4, 0.85))
xlims!(axs[4], (1.0, 2.75))
xlims!(axs[5], (-6, -3))
xlims!(axs[6], (1.5, 3.5))
xlims!(axs[7], (0.5, 1.2))
xlims!(axs[8], (0.4, 1.1))
xlims!(axs[9], (-2.25, -0.9))

axislegend(axs[5], position=:lt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)
##
save("lorenz_random_entries.png", fig)

