using HDF5, Statistics
using MarkovChainHammer, LinearAlgebra, GLMakie
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy
import MarkovChainHammer.Utils: histogram

nstates = 100
data_directory = pwd() * "/data/"
file_name = "markov_model_extreme_nstate_100.h5"
hfile = h5open(data_directory * file_name, "r")

jump_factor = 5 # forgot to save
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
markov_states = typeof(read(hfile["markov state 1"]))[]
for i in 1:100
    push!(markov_states, read(hfile["markov state $i"]))
end
close(hfile)
file_name = "markov_model_even_time_nstate_100_extreme.h5"
hfile = h5open(data_directory * file_name, "r")
markov_chain = read(hfile["markov embedding"])
time_in_days = (0:length(markov_chain)-1) .* dt_days

ht = holding_times(markov_chain, nstates; dt=dt_days)

Q = generator(markov_chain; dt=dt_days)
p = steady_state(Q)
index_ordering = reverse(sortperm(p)) # order indices by probability
mean_holding_time = [-1 / Q[i, i] for i in eachindex(p)][index_ordering]
entropy(p)
connectivity_out = sum(Q .> 0, dims=1)[index_ordering]
connectivity_in = sum(Q .> 0, dims=2)[index_ordering]
Λ, V = eigen(Q)
timescales = -1 ./ real.(Λ)
time_in_days = (0:length(markov_chain)-1) .* dt_days

## 
# subnetwork markov chain 
sub_markov_chain = copy(markov_chain)
sub_markov_chain[markov_chain.>10] .= 11
Q_sub = generator(sub_markov_chain; dt=dt_days)
P_sub = perron_frobenius(sub_markov_chain)
p_sub = steady_state(Q_sub)
ht_sub = holding_times(sub_markov_chain, 11; dt=dt_days)
##
using Graphs, GraphMakie
opacity_matrix = abs.(Q_sub ./ reshape([Q_sub[i, i] for i in 1:11], (1, 11)))
# opacity_matrix = abs.(opacity_matrix .* reshape(p_sub, 1, 11))
opacity_matrix = opacity_matrix ./ maximum(opacity_matrix[1:10, 1:10] - I)
opacity_list = opacity_matrix[opacity_matrix.>eps(1.0)]
fig = Figure(resolution = (1500, 1500))
ax = Axis(fig[1, 1]; title="Subgraph for Extreme States", titlesize=30)
g_Q_sub = DiGraph(Q_sub')
# elabels = [string(Q_sub[i])[1:5] for i in 1:ne(g_Q_sub)]
edge_color_matrix = [:red for i in 1:11, j in 1:11]
edge_color_matrix[end, :] .= :blue
edge_color_Q = edge_color_matrix[opacity_matrix.>eps(1.0)]
edge_color_Q = [(edge_color, minimum([opacity_list[i] * 0.5, 1])) for (i, edge_color) in enumerate(edge_color_Q)]
elabels_color = edge_color_Q
edge_width_Q = [2.0 for i in 1:ne(g_Q_sub)]
arrow_size_Q = [20.0 for i in 1:ne(g_Q_sub)]
node_labels_Q = repr.(1:nv(g_Q_sub))
nlabels_fontsize  = 40
node_size = 20
# elabels_fontsize=elabels_fontsize, node_color=node_color,node_size=node_size, nlabels_fontsize=nlabels_fontsize
p_Q = graphplot!(ax, g_Q_sub, elabels_color=elabels_color,
    edge_color=edge_color_Q,
    edge_width=edge_width_Q,node_size=node_size,
    arrow_size=arrow_size_Q, node_color =[edge_color_matrix[i,i] for i in 1:11],
    nlabels=node_labels_Q, nlabels_fontsize=nlabels_fontsize)
display(fig)