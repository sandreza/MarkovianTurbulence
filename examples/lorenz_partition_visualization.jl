# for visualization purposes we generate less data than what we use quantitiatively
using GLMakie, MarkovianTurbulence
using MarkovChainHammer

using ProgressBars, LinearAlgebra, Statistics, Random
using GLMakie

using ProgressBars, LinearAlgebra, Statistics
using MarkovianTurbulence: lorenz!, rk4
import MarkovChainHammer.TransitionMatrix: generator

fixed_points = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
markov_states = fixed_points

timeseries = Vector{Float64}[]
markov_chain = Int64[]
initial_condition = [14.0, 20.0, 27.0]
push!(timeseries, initial_condition)
dt = 0.01
iterations = 10^6

markov_index = argmin([norm(initial_condition - markov_state) for markov_state in markov_states])
push!(markov_chain, markov_index)
for i in ProgressBar(2:iterations)
    # take one timestep forward via Runge-Kutta 4
    local state = rk4(lorenz!, timeseries[i-1], dt)
    push!(timeseries, state)
    # partition state space according to most similar markov state
    local markov_index = argmin([norm(state - markov_state) for markov_state in markov_states])
    push!(markov_chain, markov_index)
end

##
# create colors for the plot
colors = []
# non-custom, see https://docs.juliaplots.org/latest/generated/colorschemes/
color_choices = [:red, :blue, :orange]

for i in eachindex(timeseries)
    push!(colors, color_choices[markov_chain[i]])
end
tuple_timeseries = Tuple.(timeseries)

# everything is done for plotting
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
for (i, ax) in enumerate(axs)
    lines!(ax, tuple_timeseries, color=colors)
    scatter!(ax, Tuple.(markov_states), color=color_choices, markersize=30.0)
    rotate_cam!(ax.scene, (0, θ₀ + (i-1) * δ, 0))
end

save("lorenz_partition_visualization.png", fig)
