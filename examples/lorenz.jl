using GLMakie
using MarkovChainHammer

using ProgressBars, LinearAlgebra, Statistics, Random
using GLMakie

using ProgressBars, LinearAlgebra, Statistics
import MarkovChainHammer.TransitionMatrix: generator

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

fixed_points = [[-sqrt(72), -sqrt(72), 27], [0.0, 0.0, 0.0], [sqrt(72), sqrt(72), 27]]
markov_states = fixed_points

timeseries = Vector{Float64}[]
markov_chain = Int64[]
initial_condition = [14.0, 20.0, 27.0]
push!(timeseries, initial_condition)
dt = 1.5586522107162 / 64
iterations = 1000000

markov_index = argmin([norm(initial_condition - markov_state) for markov_state in markov_states])
push!(markov_chain, markov_index)
for i in ProgressBar(2:iterations)
    local state = rk4(lorenz!, timeseries[i-1], dt)
    push!(timeseries, state)
    # partition state space according to most similar markov state
    markov_index = argmin([norm(state - markov_state) for markov_state in markov_states])
    push!(markov_chain, markov_index)
end

## construct transition matrix
pQ = generator(markov_chain; dt=dt)

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
fig = Figure(resolution=(1000, 700))
ax = LScene(fig[1:2, 1:2]; show_axis=false)
# scatter!(ax, Tuple.(markov_states), color=:black, markersize=5.0)
lines!(ax, tuple_timeseries, color=colors)

rotate_cam!(ax.scene, (0, -π / 4, 0))
display(fig)

last_time_index = minimum([60 * 15 * 2, length(timeseries)])
time_indices = 1:last_time_index

display(fig)

function change_function(time_index)
    phase = 2π / (60 * 15)
    rotate_cam!(ax.scene, (0, phase, 0))
end
