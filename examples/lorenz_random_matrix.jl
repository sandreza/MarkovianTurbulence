using GLMakie
using MarkovChainHammer, MarkovianTurbulence
using Distributions

using ProgressBars, LinearAlgebra, Statistics, Random

import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, count_operator
import MarkovChainHammer.Trajectory: generate
import MarkovChainHammer.Utils: histogram

Random.seed!(12345)

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
sQ = generator(s_markov_chain; dt=dt)
# markov_chain = [omarkov_chain... s_markov_chain...]
## construct transition matrix
Q = generator(markov_chain; dt=dt)
p = steady_state(Q)
ht = holding_times(markov_chain; dt=dt)
##
Q1 = RandomGeneratorMatrix2(markov_chain[1:floor(Int, 2 * 10^4)], 3; dt=dt)
Q2 = RandomGeneratorMatrix2(markov_chain[floor(Int, 2 * 10^4)+1:floor(Int, 2 * 10^5)], 3; dt=dt)
Q3 = RandomGeneratorMatrix2(markov_chain[floor(Int, 2 * 10^5)+1:floor(Int, 2 * 10^6)], 3; dt=dt)
Q4 = RandomGeneratorMatrix2(markov_chain[floor(Int, 2 * 10^6)+1:floor(Int, 2 * 10^7)], 3; dt=dt)
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
options = (; xlabel=" ", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
titlenames = ["Q₁₁", "Q₂₁", "Q₃₁", "Q₁₂", "Q₂₂", "Q₃₂", "Q₁₃", "Q₂₃", "Q₃₃"]
# https://docs.makie.org/v0.19/api/index.html#Axis 
# https://juliagraphics.github.io/Colors.jl/latest/namedcolors/
spine_colors = [:red, :blue, :orange]
time_pdf_colors = [:blue, :orange, :black, :red]
time_pdf_colors = [:cyan4, :darkslateblue, :gold4, :black]
time_pdf_labels = ["T=100", "T=1000", "T=10000", "T=100000"]
opacities = [0.75, 0.75, 0.75, 0.75] .* 0.75
axs = []
for i in 1:9
    ii = (i - 1) % 3 + 1
    jj = (i - 1) ÷ 3 + 1
    # change spine colors
    spinecolor = (; bottomspinecolor=spine_colors[jj], topspinecolor=spine_colors[jj], leftspinecolor=spine_colors[ii], rightspinecolor=spine_colors[ii])
    othercolor = (; titlecolor=spine_colors[jj], xgridcolor=spine_colors[jj], ygridcolor=spine_colors[jj], xtickcolor=spine_colors[jj], ytickcolor=spine_colors[jj], xticklabelcolor=spine_colors[jj], yticklabelcolor=spine_colors[jj])
    ax = Axis(fig[ii, jj]; title=titlenames[i], othercolor..., spinewidth=10, spinecolor..., options..., xgridvisible=false, ygridvisible=false)
    push!(axs, ax)
    for j in 1:4
        barplot!(ax, xys[i][j]..., color=(time_pdf_colors[j], opacities[j]), label=time_pdf_labels[j], gap=0.0)
    end
    if jj > 1
        hideydecorations!(ax)
    end
    ylims!(ax, (-0.001, 0.1))
end

xlims!(axs[1], (-1.5, -0.7));
xlims!(axs[2], (0.25, 0.65))
xlims!(axs[3], (0.4, 0.85))
xlims!(axs[4], (1.0, 2.75))
xlims!(axs[5], (-6, -3))
xlims!(axs[6], (1.5, 3.5))
xlims!(axs[7], (0.4, 1.2))
xlims!(axs[8], (0.4, 1.1))
xlims!(axs[9], (-2.25, -0.9))
[ylims!(axs[i], (-0.001, 0.075)) for i in [1, 4, 7]]
[ylims!(axs[i], (-0.001, 0.085)) for i in [1 + 1, 4 + 1, 7 + 1]]
[ylims!(axs[i], (-0.001, 0.085)) for i in [1 + 2, 4 + 2, 7 + 2]]

axislegend(axs[5], position=:lt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)
##
save("lorenz_random_entries.png", fig)

