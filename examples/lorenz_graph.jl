using Random, LinearAlgebra
using Graphs
using GLMakie, GraphMakie

# for visualization purposes we generate less data than what we use quantitiatively
##
using GLMakie, MarkovianTurbulence
using MarkovChainHammer
using ProgressBars, LinearAlgebra, Statistics, Random
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

Q = generator(markov_chain, 3; dt=dt)

##

cmap = :balance # :Blues_9
cmapa = to_colormap(cmap);
# cmap = vcat(cmapa[1:15], fill(RGBAf(0, 0, 0, 0), 10), cmapa[25:end])

fig = Figure(resolution=(1506, 1076))
ax = Axis(fig[1, 1]; title="Transition Probability", titlesize=30)
ax_Q = Axis(fig[1, 2]; title="Transition Rate", titlesize=30)

dt_slider = Slider(fig[2, 1:2], range=0:0.01:2, startvalue=0)
dt = dt_slider.value

# Q = uniform_phase(4)
T = @lift exp(Q * $dt)

g = DiGraph(exp(Q))
g_Q = DiGraph(Q)

elabels = string.([Q[i] for i in 1:ne(g)])

elabels_T = @lift string.([$T[i] for i in 1:ne(g)])

edge_color = @lift [RGBAf(cmapa[4].r, cmapa[4].g, cmapa[4].b, $T[i] / 0.44) for i in 1:ne(g)]
edge_width = [4.0 for i in 1:ne(g)]
arrow_size = [30.0 for i in 1:ne(g)]
node_labels = repr.(1:nv(g))

# edge_color_Q = [RGBAf(cmapa[4].r, cmapa[4].g, cmapa[4].b, 1.0) for i in 1:ne(g_Q)]
edge_color_Q = [:red, :red, :red, :blue, :blue, :blue, :orange, :orange, :orange]
edge_width_Q = [4.0 for i in 1:ne(g_Q)]
arrow_size_Q = [30.0 for i in 1:ne(g_Q)]
node_labels_Q = repr.(1:nv(g_Q))
node_size = 20.0

# obs_string = @lift("Transition Probability at time t = " * string($dt) )
p = graphplot!(ax, g, elabels=elabels_T, edge_color=edge_color, edge_width=edge_width,
    arrow_size=arrow_size, node_size=node_size,
    nlabels=node_labels, nlabels_textsize=50.0)

offsets = 0.15 * (p[:node_pos][] .- p[:node_pos][][1])
offsets[1] = Point2f(0, 0.3)
p.nlabels_offset[] = offsets
autolimits!(ax)
hidedecorations!(ax)

p_Q = graphplot!(ax_Q, g_Q, elabels=elabels, edge_color=edge_color_Q, edge_width=edge_width_Q,
    arrow_size=arrow_size_Q, node_size=node_size,
    nlabels=node_labels_Q, nlabels_textsize=50.0)

offsets = 0.15 * (p_Q[:node_pos][] .- p_Q[:node_pos][][1])
offsets[1] = Point2f(0, 0.3)
p_Q.nlabels_offset[] = offsets
autolimits!(ax_Q)

hidedecorations!(ax_Q)
display(fig)