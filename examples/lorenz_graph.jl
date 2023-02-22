using Random, LinearAlgebra
using Graphs
using GLMakie, GraphMakie
using MarkovianTurbulence

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
embedding = StateEmbedding(fixed_points)

initial_condition = [14.0, 20.0, 27.0]
dt = 0.005
iterations = 2*10^7

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

Q = generator(markov_chain, 3; dt=dt)

##
fig = Figure(resolution=(2000, 1500))
ax_Q = Axis(fig[1, 1]; title="Generator", titlesize=30)

# Generator
g_Q = DiGraph(Q')
edge_color_Q = [:red, :red, :red, :blue, :blue, :blue, :orange, :orange, :orange]
elabels = [string(Q[i])[1:5] for i in 1:ne(g_Q)]
elabels_color = edge_color_Q
node_color = [:red, :blue, :orange]
edge_attr = (; linestyle=[:dot, :dash, :dash, :dash, :dot, :dash, :dash, :dash, :dot])
elabels_fontsize = 30
nlabels_fontsize = 30
node_size = 30.0

edge_width_Q = [2.0 for i in 1:ne(g_Q)]
arrow_size_Q = [20.0 for i in 1:ne(g_Q)]
node_labels_Q = repr.(1:nv(g_Q))
node_labels_Q = ["Negative Wing", "Origin", "Positive Wing"]


p_Q = graphplot!(ax_Q, g_Q, elabels=elabels, elabels_color=elabels_color,
    elabels_fontsize=elabels_fontsize, edge_color=edge_color_Q,
    edge_width=edge_width_Q, node_color=node_color,
    arrow_size=arrow_size_Q, node_size=node_size,
    nlabels=node_labels_Q, nlabels_fontsize=nlabels_fontsize)

offsets = 0.0 * (p_Q[:node_pos][] .- p_Q[:node_pos][][1])
offsets[2] = Point2f(-0.5, -0.5)
p_Q.nlabels_offset[] = offsets
autolimits!(ax_Q)
hidedecorations!(ax_Q)

# t = 0.1 Transfer operator
dt = 0.005
dt_strings = ["0.1", "0.5", "1.5"]
dt_vals = parse.(Float64, dt_strings)
for (ii, τ) in enumerate(dt_vals)
    j = (ii) % 2 + 1
    i = (ii) ÷ 2 + 1
    ax_T1 = Axis(fig[i, j]; title="Transfer Operator for τ = " * dt_strings[ii], titlesize=30)
    skip = round(Int, τ/dt)
    # T1 = mean([perron_frobenius(markov_chain[i:skip:end], 3) for i in 1:skip])
    T1 = exp(Q * τ)
    g_T1 = DiGraph(T1)
    elabels = [string(T1[i])[1:5] for i in 1:ne(g_T1)]
    edge_color_T1 = [(edge_color, T1[i] / 0.5) for (i, edge_color) in enumerate(edge_color_Q)]
    elabels_color = edge_color_Q
    node_color = [:red, :blue, :orange]
    edge_attr = (; linestyle=[:dot, :dash, :dash, :dash, :dot, :dash, :dash, :dash, :dot])
    edge_width_Q = [2.0 for i in 1:ne(g_T1)]
    arrow_size_Q = [20.0 for i in 1:ne(g_T1)]
    node_labels_Q = repr.(1:nv(g_Q))
    node_labels_Q = ["Negative Wing", "Origin", "Positive Wing"]

    p_Q = graphplot!(ax_T1, g_T1, elabels=elabels, elabels_color=elabels_color,
        elabels_fontsize=elabels_fontsize, edge_color=edge_color_T1,
        edge_width=edge_width_Q, node_color=node_color,
        arrow_size=arrow_size_Q, node_size=node_size,
        nlabels=node_labels_Q, nlabels_fontsize=nlabels_fontsize)

    offsets = 0.0 * (p_Q[:node_pos][] .- p_Q[:node_pos][][1])
    offsets[2] = Point2f(-0.5, -0.5)
    p_Q.nlabels_offset[] = offsets
    autolimits!(ax_T1)
    hidedecorations!(ax_T1)
end

display(fig)
##
save("lorenz_graph.png", fig)