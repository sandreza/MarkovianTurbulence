using GLMakie
using MarkovChainHammer, MarkovianTurbulence

using ProgressBars, LinearAlgebra, Statistics, Random

import MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, perron_frobenius
import MarkovChainHammer.Utils: histogram, autocovariance

##
@info "random partition"
ic= [8.899951468086583, 2.239908347694721, 34.31187951584054]# initial condition from Viswanath
dt = 0.1
subdiv = 2^14
timeseries = zeros(3, subdiv)
timeseries[:, 1] .= ic

for i in ProgressBar(2:subdiv)
    # take one timestep forward via Runge-Kutta 4
    state = rk4(lorenz!, timeseries[:, i-1], dt)
    timeseries[:, i] .= state
end

n_markov_states = 128
skip = round(Int, subdiv / n_markov_states)
markov_states_random = [timeseries[:, i] for i in 1:skip:subdiv]
numstates = length(markov_states_random)

scatter(timeseries[:,  1:skip:subdiv])
##
embedding = StateEmbedding(markov_states_random)

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

markov_chain_random = copy(markov_chain)
## construct transition matrix
Q = generator(markov_chain; dt=dt)
p = steady_state(Q)
p_random = copy(p)

##
reaction_coordinates = [u -> u[i] for i in 1:3] # define anonymous functions for reaction coordinates
markovs = []
rtimeseriess = []
for i in ProgressBar(1:3)
    current_reaction_coordinate = reaction_coordinates[i]
    markov = [current_reaction_coordinate(markov_state) for markov_state in markov_states_random]
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
kwargs = (; ylabel="Probability", titlesize=40, ylabelsize=40, xgridstyle=:dash, ygridstyle=:dash, xtickalign=1,
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
# labels = [labels..., "ℰ(s)==1", "sign(x)", "ℰ(s)==2"]
labels = ["g¹", "g²", "g³", "g⁴", "g⁵", "g⁶"]
reaction_coordinates = [reaction_coordinates..., u -> argmin([norm(u - markov_state) for markov_state in markov_states_random]) == 1, u -> sign(u[1]), u -> argmin([norm(u - markov_state) for markov_state in markov_states_random]) == 2]

kwargs = (; ylabel="Autocorrelation", titlesize=50, ylabelsize=40,
    xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, xtickalign=1,
    xticksize=20, ytickalign=1, yticksize=20, xlabel="Time",
    xticklabelsize=40, yticklabelsize=40, xlabelsize=40)

# ctep = ContinuousTimeEmpiricalProcess(markov_chain)
# generated_chain = generate(ctep, 10^4, markov_chain[1])
random_auto = []
axs = []
for i in ProgressBar(1:6)
    current_reaction_coordinate = reaction_coordinates[i]
    subfig = subfigs[i]

    markov = [current_reaction_coordinate(markov_state) for markov_state in markov_states_random]
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

    markov = [current_reaction_coordinate(markov_state) for markov_state in markov_states_random]

    Pτ = perron_frobenius_1 * 0 + I
    for i in 0:total-1
        auto_correlation_snapshots[i+1] = sum(markov' * Pτ * (p .* markov))
        Pτ *= perron_frobenius_1
    end
    auto_correlation_snapshots .= auto_correlation_snapshots .- sum(markov .* p)^2
    auto_correlation_snapshots .*= 1.0 / auto_correlation_snapshots[1]

    push!(random_auto, auto_correlation_snapshots)

    ax1 = Axis(subfig[1, 1]; title="  " * labels[i], kwargs...)
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

for i in eachindex(labels)
    g = observables[i]
    g_ensemble = sum(g.(markov_states_random) .* p_random)
    # g_temporal = mean([g(timeseries[:, i]) for i in 1:iterations])
    println(" ensemble: ⟨$(labels[i])⟩ = $g_ensemble")
    # println(" temporal: ⟨$(labels[i])⟩ = $g_temporal")
    println("--------------------------------------------")
end

#=
observable(u) = u[1] * u[2] * u[3] # ⟨xyz⟩ triple correlation
# ensemble and temporal average
g_ensemble = sum(observable.(markov_states_random) .* p)
g_temporal = mean([observable(timeseries[:, i]) for i in 1:iterations])
println("The ensemble average   ⟨xyz⟩  is $(g_ensemble)")
println("The timeseries average ⟨xyz⟩  is $(g_temporal)")
=#
##
inds= 1:100:iterations
fig = Figure(resolution=(1500, 1000))
axs = []
for i in 1:3
    ax = LScene(fig[1, i]; show_axis=false)
    push!(axs, ax)
end
for j in 1:3
    ax = LScene(fig[2, j]; show_axis=false)
    push!(axs, ax)
end

# scatter!(ax, Tuple.(markov_states), color=:black, markersize=5.0)
θ₀ = -π/5 
δ = π/7
cmapa = RGBAf.(to_colormap(:glasbey_hv_n256))
colors = [(cmapa[markov_chain[i]], 0.5) for i in inds]
for (i, ax) in enumerate(axs)
    scatter!(ax, Tuple.(markov_states_random), color=:black, markersize=10.0)
    scatter!(ax, timeseries[:, inds] , color=colors, markersize = 5.0)
    scatter!(ax, Tuple.(markov_states_random), color=:black, markersize=10.0)
    rotate_cam!(ax.scene, (0, θ₀ + (i-1) * δ, 0))
end
display(fig)