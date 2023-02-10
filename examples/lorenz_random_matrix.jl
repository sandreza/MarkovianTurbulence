using GLMakie
using MarkovChainHammer, MarkovianTurbulence, MarkovChainHammer.BayesianMatrix
using Distributions

using ProgressBars, LinearAlgebra, Statistics, Random

import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, count_operator
import MarkovChainHammer.Trajectory: generate
import MarkovChainHammer.Utils: histogram

Random.seed!(12345)

fixed_points = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
markov_states = fixed_points

embedding = StateEmbedding(fixed_points)

initial_condition = [14.0, 20.0, 27.0]
dt = 0.005
iterations = 2 * 10^7

timeseries = zeros(3, iterations)
markov_chain = zeros(Int, iterations)
timeseries[:, 1] .= initial_condition

markov_index = embedding(initial_condition)
markov_chain[1] = markov_index
for i in ProgressBar(2:iterations)
    # take one timestep forward via Runge-Kutta 4
    state = rk4(lorenz!, timeseries[:, i-1], dt)
    timeseries[:, i] .= state
    # partition state space according to most similar markov state
    markov_index = embedding(state)
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
Q̃ = Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p))
noise_Q̃ = Symmetric((Q̃ + Q̃') / 2)
drift_Q̃ = (Q̃ - Q̃') / 2
ht = holding_times(markov_chain; dt=dt)
##
prior = MarkovChainHammer.BayesianMatrix.uninformative_prior(3)
#=
Q1 = BayesianGenerator(markov_chain[1:floor(Int, 2 * 10^4)], prior; dt=dt)
Q2 = BayesianGenerator(markov_chain[floor(Int, 2 * 10^4)+1:floor(Int, 2 * 10^5)], prior; dt=dt)
Q3 = BayesianGenerator(markov_chain[floor(Int, 2 * 10^5)+1:floor(Int, 2 * 10^6)], prior; dt=dt)
Q4 = BayesianGenerator(markov_chain[floor(Int, 2 * 10^6)+1:floor(Int, 2 * 10^7)], prior; dt=dt)
=#
# Q1 = BayesianGenerator(markov_chain[1:floor(Int, 2 * 10^3)], prior; dt=dt)
Q1 = BayesianGenerator(markov_chain[1:10^5], prior; dt=dt)
Q2 = BayesianGenerator(markov_chain[10^5+1:2*10^5], prior; dt=dt)

Q12 = BayesianGenerator(markov_chain[10^5+1:2*10^5], Q1.posterior; dt=dt)
Q3 = BayesianGenerator(markov_chain[1:2*10^5], prior; dt=dt)


Q2 = BayesianGenerator(markov_chain[1+2*10^7-10^5:2*10^7], prior; dt=dt)
Q3 = BayesianGenerator(markov_chain[1:2*10^6], prior; dt=dt)
Q4 = BayesianGenerator(markov_chain[2*10^7+1 - 2*10^6:2*10^7], prior; dt=dt)
Q5 = BayesianGenerator(markov_chain, prior; dt=dt)
Q̄ = mean(Q5)
p̄ = steady_state(Q̄)

Q_symmetrized = BayesianGenerator(s_markov_chain, Q5.posterior; dt=dt)
num_samples = 100000
Q1s = rand(Q1, num_samples)
Q2s = rand(Q2, num_samples)
Q3s = rand(Q3, num_samples)
Q4s = rand(Q4, num_samples)
Q5s = rand(Q5, num_samples)
Qs = [Q1s, Q2s, Q3s, Q4s, Q5s]

Q = generator(markov_chain[1:end]; dt=dt)
##
observables = [Q -> Q[i] for i in 1:9]
obs = [observable.(Q) for observable in observables, Q in Qs]
best_empirical = [observable(Q) for observable in observables]
##
Nbins = 500
xys = []
for i in 1:9
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]..., obs[i, 3]...]), (0.001, 0.999))
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
time_pdf_colors = [:cyan4, :darkslateblue, :gold4, :orchid, :black]
# time_pdf_labels = ["T=100", "T=1000", "T=10000", "T=100000"]
time_pdf_labels = ["T ∈ [0, 10³]", "T ∈ [10⁷ - 10³, 10⁷]", "T ∈ [0, 10⁴]", "T ∈ [10⁷ - 10⁴, 10⁷]", "T ∈ [0, 10⁷]"]
opacities = [0.75, 0.75, 0.75, 0.75, 0.5] .* 0.75
axs = []
for i in 1:9
    ii = (i - 1) % 3 + 1
    jj = (i - 1) ÷ 3 + 1
    # change spine colors
    spinecolor = (; bottomspinecolor=spine_colors[jj], topspinecolor=spine_colors[jj], leftspinecolor=spine_colors[ii], rightspinecolor=spine_colors[ii])
    othercolor = (; titlecolor=spine_colors[jj], xgridcolor=spine_colors[jj], ygridcolor=spine_colors[jj], xtickcolor=spine_colors[jj], ytickcolor=spine_colors[jj], xticklabelcolor=spine_colors[jj], yticklabelcolor=:black)
    ax = Axis(fig[ii, jj]; title=titlenames[i], othercolor..., spinewidth=3, spinecolor..., options..., xgridvisible=false, ygridvisible=false)
    push!(axs, ax)
    for j in 1:5
        barplot!(ax, xys[i][j]..., color=(time_pdf_colors[j], opacities[j]), label=time_pdf_labels[j], gap=0.0)
    end
    # vlines!(ax, Q[i], color=(:orchid3, 1.0), linewidth=10, label="Best empirical")
    if jj > 1
        hideydecorations!(ax)
    end
    # ylims!(ax, (-0.001, 0.1))
end

xlims!(axs[1], (-1.3, -1.0));
xlims!(axs[2], (0.4, 0.65))
xlims!(axs[3], (0.5, 0.8))
xlims!(axs[4], (1.55, 2.25))
xlims!(axs[5], (-4.5, -3))
xlims!(axs[6], (1.3, 2.5))
xlims!(axs[7], (0.5, 0.75))
xlims!(axs[8], (0.35, 0.7))
xlims!(axs[9], (-1.5, -0.9))

[ylims!(axs[i], (-0.000, 0.08)) for i in [1, 4, 7]]
[ylims!(axs[i], (-0.000, 0.08)) for i in [1 + 1, 4 + 1, 7 + 1]]
[ylims!(axs[i], (-0.000, 0.08)) for i in [1 + 2, 4 + 2, 7 + 2]]

axislegend(axs[5], position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)
##
save("lorenz_random_entries.png", fig)

##
primitive_labels = ["x", "y", "z"]
observables = []
labels = []
for i in 1:3
    push!(observables, u -> u[i])
    push!(labels, primitive_labels[i])
end
for i in 1:3
    for j in i:3
        push!(observables, u -> u[i] * u[j])
        push!(labels, primitive_labels[i] * primitive_labels[j])
    end
end
for i in 1:3
    for j in i:3
        for k in j:3
            push!(observables, u -> u[i] * u[j] * u[k])
            push!(labels, primitive_labels[i] * primitive_labels[j] * primitive_labels[k])
        end
    end
end
for i in 1:3
    for j in i:3
        for k in j:3
            for l in k:3
                push!(observables, u -> u[i] * u[j] * u[k] * u[l])
                push!(labels, primitive_labels[i] * primitive_labels[j] * primitive_labels[k] * primitive_labels[l])
            end
        end
    end
end

ensemble_mean = Float64[]
temporal_mean = Float64[]
for i in eachindex(labels)
    g = observables[i]
    push!(ensemble_mean, sum(g.(markov_states) .* p̄))
    push!(temporal_mean, mean([g(timeseries[:,i]) for i in 1:iterations]))
    println(" ensemble: ⟨$(labels[i])⟩ = $(ensemble_mean[i])")
    println(" temporal: ⟨$(labels[i])⟩ = $(temporal_mean[i])")
    println("--------------------------------------------")
end
##
using Printf
averages_symbol = [" \$\\langle $(labels[i]) \\rangle\$ &" for i in 1:9]
averages_string = prod(averages_symbol)
ensemble_symbol = [@sprintf("%.1e", ensemble_mean[i]) * " & " for i in 1:9]
ensemble_string = prod(ensemble_symbol)
temporal_symbol = [@sprintf("%.1e", temporal_mean[i]) * " & " for i in 1:9]
temporal_string = prod(temporal_symbol)

abc = open("example.txt", "w")
write(abc, averages_string)
write(abc, "\n")
write(abc, ensemble_string)
write(abc, "\n")
write(abc, temporal_string)
close(abc)


# averages_symbol = [" \$\\langle \\mathscr{$(labels[i])} \\rangle\$ &" for i in 10:19]
averages_symbol = [" \$\\langle $(labels[i]) \\rangle\$ &" for i in 10:19]
averages_string = prod(averages_symbol)
ensemble_symbol = [@sprintf("%.1e", ensemble_mean[i]) * " & " for i in 10:19]
ensemble_string = prod(ensemble_symbol)
temporal_symbol = [@sprintf("%.1e", temporal_mean[i]) * " & " for i in 10:19]
temporal_string = prod(temporal_symbol)

abc = open("example2.txt", "w")
write(abc, averages_string)
write(abc, "\n")
write(abc, ensemble_string)
write(abc, "\n")
write(abc, temporal_string)
close(abc)
