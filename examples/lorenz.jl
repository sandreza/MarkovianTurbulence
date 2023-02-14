using GLMakie
using MarkovChainHammer, MarkovianTurbulence

using ProgressBars, LinearAlgebra, Statistics, Random

import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius
import MarkovChainHammer.Utils: histogram, autocovariance

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

## construct transition matrix
Q = generator(markov_chain; dt=dt)
p = steady_state(Q)
## construct several transfer operators 
Ps = [mean([perron_frobenius(markov_chain[i:j:end], 3) for i in 1:j]) for j in 10:10:800]
## Histogram Figure 
reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates
markovs = []
rtimeseriess = []
for i in ProgressBar(1:3)
    current_reaction_coordinate = reaction_coordinates[i]
    markov = [current_reaction_coordinate(markov_state) for markov_state in markov_states]
    rtimeseries = [current_reaction_coordinate(timeseries[:, i]) for i in 1:iterations]
    push!(markovs, markov)
    push!(rtimeseriess, rtimeseries)
end
##
hfig = Figure(resolution=(1800, 1500))
xfig = hfig[1, 1] = GridLayout()
yfig = hfig[2, 1] = GridLayout()
zfig = hfig[3, 1] = GridLayout()
subfigs = [xfig, yfig, zfig]
colors = [:red, :blue, :orange]
labels = ["x", "y", "z"]

# reaction_coordinate(u) = real(iV[1, argmin([norm(u - s) for s in markov_states])]) # u[3] # 
kwargs = (; ylabel="Probability", titlesize=30, ylabelsize=40, xgridstyle=:dash, ygridstyle=:dash, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20,
    xticklabelsize=40, yticklabelsize=40)
bins1 = 19
bins2 = 19

for i in 1:3
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = markovs[i] # [current_reaction_coordinate(markov_state) for markov_state in markov_states]
    rtimeseries = rtimeseriess[i] # [current_reaction_coordinate(timeseries[:, i]) for i in 1:iterations]
    xs_m, ys_m = histogram(markov, normalization=p, bins=bins1, custom_range=extrema(rtimeseries))
    xs_t, ys_t = histogram(rtimeseries, bins=bins2, custom_range=extrema(rtimeseries))


    ax1 = Axis(subfig[1, 1]; title="Markov Chain Histogram, " * labels[i], kwargs...)
    ax2 = Axis(subfig[1, 2]; title="Timeseries Histogram, " * labels[i], kwargs...)

    for ax in [ax1, ax2]
        x_min = minimum([minimum(xs_m), minimum(xs_t)])
        x_max = maximum([maximum(xs_m), maximum(xs_t)])
        y_min = minimum([minimum(ys_m), minimum(ys_t)])
        y_max = maximum([maximum(ys_m), maximum(ys_t)])
        if i < 3
            xlims!(ax, (x_min, x_max))
            ylims!(ax, (y_min, y_max))
        else
            xlims!(ax, (-5, x_max))
            ylims!(ax, (y_min, y_max))
        end
    end

    barplot!(ax1, xs_m, ys_m, color=:purple)
    barplot!(ax2, xs_t, ys_t, color=:black)
    hideydecorations!(ax2, grid=false)


end
display(hfig)
##
save("lorenz_histogram.png", hfig)

## Lorenz autocorrelations
perron_frobenius_1 = exp(Q * dt)
auto_fig = Figure(resolution=(3000, 2000))
xfig = auto_fig[1, 1] = GridLayout()
yfig = auto_fig[2, 1] = GridLayout()
zfig = auto_fig[3, 1] = GridLayout()

xfig2 = auto_fig[1, 2] = GridLayout()
yfig2 = auto_fig[2, 2] = GridLayout()
zfig2 = auto_fig[3, 2] = GridLayout()

subfigs = [xfig, yfig, zfig, xfig2, yfig2, zfig2]
colors = [:red, :blue, :orange]

labels = ["x", "y", "z"]
reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates

# labels = [labels..., "x > 0", "y > 0", "z > 5"]
# reaction_coordinates = [reaction_coordinates..., u -> u[1] > 0, u -> u[2] > 0, u -> u[3] > 5]
labels = [labels..., "ℰ(s)==1", "sign(x)", "ℰ(s)==2"]
reaction_coordinates = [reaction_coordinates..., u -> argmin([norm(u - markov_state) for markov_state in markov_states]) == 1, u -> sign(u[1]), u -> argmin([norm(u - markov_state) for markov_state in markov_states]) == 2]

kwargs = (; ylabel="Autocorrelation", titlesize=30, ylabelsize=40,
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xlabel="Time",
    xticklabelsize=40, yticklabelsize=40, xlabelsize=40)

# ctep = ContinuousTimeEmpiricalProcess(markov_chain)
# generated_chain = generate(ctep, 10^4, markov_chain[1])

axs = []
for i in ProgressBar(1:6)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = [current_reaction_coordinate(markov_state) for markov_state in markov_states]
    iterations_used = floor(Int, iterations / 100)
    rtimeseries = [current_reaction_coordinate(timeseries[:, i]) for i in 1:iterations_used]

    total = 800
    auto_correlation_timeseries = zeros(total)
    for s in ProgressBar(0:total-1)
        auto_correlation_timeseries[s+1] = mean(rtimeseries[s+1:end] .* rtimeseries[1:end-s])
    end
    auto_correlation_timeseries .-= mean(rtimeseries)^2
    auto_correlation_timeseries .*= 1 / auto_correlation_timeseries[1]

    auto_correlation_snapshots = zeros(total)

    markov = [current_reaction_coordinate(markov_state) for markov_state in markov_states]

    Pτ = perron_frobenius_1 * 0 + I
    for i in 0:total-1
        auto_correlation_snapshots[i+1] = sum(markov' * Pτ * (p .* markov))
        Pτ *= perron_frobenius_1
    end
    auto_correlation_snapshots .= auto_correlation_snapshots .- sum(markov .* p)^2
    auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1]


    ax1 = Axis(subfig[1, 1]; title="Observable:  " * labels[i], kwargs...)
    l1 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_timeseries[:], color=:black, label="Timeseries", linewidth=7)
    l2 = lines!(ax1, dt .* collect(0:total-1), auto_correlation_snapshots[:], color=(:purple, 0.5), label="Generator", linewidth=7)
    # autocorrelation_perron_frobenius = autocovariance(markov, Ps, 79)
    # autocorrelation_perron_frobenius = autocorrelation_perron_frobenius / autocorrelation_perron_frobenius[1]
    # l3 = scatter!(ax1, dt .* collect(0:10:total-1), autocorrelation_perron_frobenius, color=(:green, 0.5), markersize=20, label="Transfer Operators")

    # generated_states = [current_reaction_coordinate(markov_states[link]) for link in generated_chain]
    # autocorrelation_generated = autocovariance(generated_states; timesteps = 800)
    # l4 = lines!(ax1, dt .* collect(0:total-1), autocorrelation_generated / autocorrelation_generated[1], color=(:blue, 0.5), label="Model", linewidth=5)
    # axislegend(ax1, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
    display(auto_fig)
    push!(axs, ax1)
end
axislegend(axs[1], position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=80, labelsize=40)
display(auto_fig)

##
save("lorenz_autocorrelation.png", auto_fig)

## Holding Times Figure
ht = holding_times(markov_chain, maximum(markov_chain); dt=dt)
bins = [5, 20, 100]
color_choices = [:red, :blue, :orange] # same convention as before
index_names = ["Negative Lobe", "Origin", "Positive Lobe"]
hi = 1 #holding index
bin_index = 1 # bin index
labelsize = 40
options = (; xlabel="Time", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution=(2800, 1800))
for hi in 1:3, bin_index in 1:3
    ax = Axis(fig[hi, bin_index]; title=index_names[hi] * " Holding Times" * ", " * string(bins[bin_index]) * " Bins", options...)
    holding_time_index = hi

    holding_time_limits = (0, ceil(Int, maximum(ht[holding_time_index])))
    holding_time, holding_time_probability = histogram(ht[holding_time_index]; bins=bins[bin_index], custom_range=holding_time_limits)

    barplot!(ax, holding_time, holding_time_probability, color=(color_choices[hi], 0.5), gap=0.0, label="Data")
    λ = 1 / mean(ht[holding_time_index])

    Δholding_time = holding_time[2] - holding_time[1]
    exponential_distribution = @. (exp(-λ * (holding_time - 0.5 * Δholding_time)) - exp(-λ * (holding_time + 0.5 * Δholding_time)))
    lines!(ax, holding_time, exponential_distribution, color=:black, linewidth=3)
    scatter!(ax, holding_time, exponential_distribution, color=(:black, 0.5), markersize=20, label="Exponential")
    axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
end
display(fig)
##
save("lorenz_holding_times.png", fig)

##
# Markov Chain Embedding 
nsteps = 2100
color_choices = [:red, :blue, :orange]
colorlist = [color_choices[markov_chain[i]] for i in 1:nsteps]
tlist = collect(0:dt:dt*(iterations-1))
labelsize = 30
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize, xgridvisible=false, ygridvisible=false)

fig = Figure(resolution=(1500, 1000))
embedding_fig = fig[2, 1] = GridLayout()
dynamics_fig = fig[1, 1] = GridLayout()

ax = Axis(embedding_fig[1, 1]; title="Markov Chain Embedding", xlabel="Time", ylabel="State", options...)
scatter!(ax, tlist[1:nsteps], markov_chain[1:nsteps], color=colorlist)
ax.yticks = ([1, 2, 3], ["Left Lobe", "Origin", "Right Lobe"])

dynamics_labels = ["x", "y", "z"]
axslist = []
for i in 1:3
    if i == 1
        ax_x = Axis(dynamics_fig[i, 1]; title="Dynamics", xlabel="Time", ylabel=dynamics_labels[i], options...)
    else
        ax_x = Axis(dynamics_fig[i, 1]; xlabel="Time", ylabel=dynamics_labels[i], options...)
    end
    lines!(ax_x, tlist[1:nsteps], timeseries[i, 1:nsteps], color=colorlist, linewidth=5)
    push!(axslist, ax_x)
end
hidexdecorations!(axslist[1])
hidexdecorations!(axslist[2])
##
display(fig)
##
save("lorenz_dynamics_embedding.png", fig)

## 
## construct ensemble and temporal averages of first and second moments
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

for i in eachindex(labels)
    g = observables[i]
    g_ensemble = sum(g.(markov_states) .* p)
    g_temporal = mean([g(timeseries[:, i]) for i in 1:iterations])
    println(" ensemble: ⟨$(labels[i])⟩ = $g_ensemble")
    println(" temporal: ⟨$(labels[i])⟩ = $g_temporal")
    println("--------------------------------------------")
end

observable(u) = u[1] * u[2] * u[3] # ⟨xyz⟩ triple correlation
# ensemble and temporal average
g_ensemble = sum(observable.(markov_states) .* p)
g_temporal = mean([observable(timeseries[:, i]) for i in 1:iterations])
println("The ensemble average   ⟨xyz⟩  is $(g_ensemble)")
println("The timeseries average ⟨xyz⟩  is $(g_temporal)")
