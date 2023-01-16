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
Φ = read(hfile["geopotential"])
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
htextreme = holding_times((markov_chain .< 11) .+ 1, 2; dt=dt_days)
stats = (; max=maximum(htextreme[2]), mean=mean(htextreme[2]), min=minimum(htextreme[2]), median=median(htextreme[2]))

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

close(hfile)

# subnetwork markov chain 
sub_markov_chain = copy(markov_chain)
sub_markov_chain[markov_chain.>10] .= 11 # lump all states > 10 into one state, this corresponds to treating all non-extreme states the same
Q_sub = generator(sub_markov_chain; dt=dt_days)
Ps_sub = [mean([perron_frobenius(sub_markov_chain[i:j:end], 11) for i in 1:j]) for j in 1:length(tlist)]
P_sub = perron_frobenius(sub_markov_chain)
p_sub = steady_state(Q_sub)
ht_sub = holding_times(sub_markov_chain, 11; dt=dt_days)

function temperature(markov_state, Φ)
    ρ = markov_state[:, :, 1]
    ρu = markov_state[:, :, 2]
    ρv = markov_state[:, :, 3]
    ρw = markov_state[:, :, 4]
    ρe = markov_state[:, :, 5]
    γ = 1.4
    R_d = 287.0
    T = (γ - 1) .* (ρe .- 0.5 .* (ρu .^ 2 .+ ρv .^ 2 .+ ρw .^ 2) ./ ρ .- ρ .* Φ) ./ (ρ .* R_d)
    return T
end
function temperature2(markov_state, Φ)
    ρ = markov_state[:, 1]
    ρu = markov_state[:, 2]
    ρv = markov_state[:, 3]
    ρw = markov_state[:, 4]
    ρe = markov_state[:, 5]
    γ = 1.4
    R_d = 287.0
    T = (γ - 1) .* (ρe .- 0.5 .* (ρu .^ 2 .+ ρv .^ 2 .+ ρw .^ 2) ./ ρ .- ρ .* Φ) ./ (ρ .* R_d)
    return T
end

filename = "observables_test_2.h5"
data_directory = pwd() * "/data/"
ofile = h5open(data_directory * filename, "r")
observables = read(ofile["observables"])
prognostic_observables = observables[:, 7:end]
dt = read(ofile["dt"])
close(ofile)
Tlist = temperature2(prognostic_observables, Φ[1, 1])

tlist = collect(time_in_days)[1:200] # [1:327]

autocovariance_t2 = autocovariance(Tlist .> 290; timesteps=length(tlist))

htextreme = holding_times((Tlist .> 290) .+ 1, 2; dt=dt_days)
stats = (; max=maximum(htextreme[2]), mean=mean(htextreme[2]), min=minimum(htextreme[2]), median=median(htextreme[2]))
##
# perron-frobenius operator for each time t 
Ps = [mean([perron_frobenius(markov_chain[i:j:end], nstates) for i in 1:j]) for j in 1:length(tlist)]
##
function autocovariance(observable, Ps::Vector{Matrix{Float64}}, steps)
    autocor = zeros(steps + 1)
    p = steady_state(Ps[1])
    μ² = sum(observable .* p)^2
    autocor[1] = observable' * (observable .* p) - μ²
    for i in 1:steps
        # p = steady_state(Ps[i])
        # μ² = sum(observable .* p)^2
        autocor[i+1] = observable' * Ps[i] * (observable .* p) - μ²
    end
    return autocor
end
autocovariance_m2 = autocovariance(observable_m, Ps, length(tlist) - 1)
autocovariance_m_sub = autocovariance(observable_m[1:11], Q_sub, tlist)
autocovariance_m_sub2 = autocovariance(observable_m[1:11], Ps_sub, length(tlist) - 1)
##
autocovariance_t = autocovariance(markov_chain .< 11; timesteps=length(tlist))
observable_m = zeros(100)
observable_m[1:10] .= 1
autocovariance_m = autocovariance(observable_m, Q, tlist)
autocovariance_random = autocovariance(randn(100), Q, tlist)
autocovariance_random = autocovariance_random ./ maximum(autocovariance_random) * maximum(autocovariance_m)

tmpt = holding_times((markov_chain .< 11) .+ 1; dt=dt_days)
Λsmall = mean.(tmpt)[2]
##
fig = Figure(resolution=(1500, 1000))
labelsize = 40
fixmeratio = (1 / autocovariance_t2[1]) * autocovariance_t[1] # SET EQUAL TO 1 LATER
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
ax = Axis(fig[1, 1]; title="Extreme State", titlesize=30, xlabel="τ [days]", ylabel="Autocovariance", xgridvisible=false, ygridvisible=false, options...)
lines!(ax, tlist, autocovariance_t, color=:black, linewidth=5, label="Markov Embedding")
lines!(ax, tlist, autocovariance_t2 * fixmeratio, color=:blue, linewidth=5, label="FIX ME LATER Timeseries")
scatter!(ax, tlist, autocovariance_m, color=(:purple, 0.5), markersize=20, label="Generator")
scatter!(ax, tlist, autocovariance_m2, color=(:green, 0.5), markersize=20, label="Perron-Frobenius at each time τ")
lines!(ax, tlist, autocovariance_random, color=:red, linewidth=5, label="Random Vector")
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
# inset
shiftx = 400
shifty = 500
widthx = 300
widthy = 300
bboxsizes = 20
ax2 = Axis(fig, bbox=BBox(shiftx, shiftx + widthx, shifty, widthy + shifty), title="zoom", titlesize=bboxsizes * 2, xlabel="τ [days]", xlabelsize=bboxsizes, xticklabelsize=bboxsizes, xgridvisible=false, ygridvisible=false)
endind = 20
lines!(ax2, tlist[1:endind], autocovariance_t2[1:endind] * fixmeratio, color=:blue, linewidth=5)
scatter!(ax2, tlist[1:endind], autocovariance_m[1:endind], color=(:purple, 0.5), markersize=20)
scatter!(ax2, tlist[1:endind], autocovariance_m2[1:endind], color=(:green, 0.5), markersize=20)
hideydecorations!(ax2)
# limits!(ax2, -3.1, -1.9, -0.05, 0.05)
# ax2.yticks = [-0.05, 0, 0.05]
# ax2.xticks = [-3, -2.5, -2]
# translate!(ax2.scene, 0, 0, 100);
display(fig)
save("held_suarez_extreme_correlation_n100.png", fig)

##
using Graphs, GraphMakie
opacity_matrix = abs.(Q_sub ./ reshape([Q_sub[i, i] for i in 1:11], (1, 11)))
opacity_matrix = opacity_matrix ./ maximum(opacity_matrix[1:10, 1:10] - I)
opacity_list = opacity_matrix[opacity_matrix.>eps(1.0)]
fig = Figure(resolution=(1500, 1500))
ax = Axis(fig[1, 1]; title="Subgraph for Extreme States", titlesize=30)
g_Q_sub = DiGraph(Q_sub')

edge_color_matrix = [:red for i in 1:11, j in 1:11]
edge_color_matrix[end, :] .= :blue
edge_color_Q = edge_color_matrix[opacity_matrix.>eps(1.0)]
edge_color_Q = [(edge_color, minimum([opacity_list[i] * 0.5, 1])) for (i, edge_color) in enumerate(edge_color_Q)]
elabels_color = edge_color_Q
edge_width_Q = [2.0 for i in 1:ne(g_Q_sub)]
arrow_size_Q = [20.0 for i in 1:ne(g_Q_sub)]
node_labels_Q = repr.(1:nv(g_Q_sub))
nlabels_fontsize = 40
node_size = 20
# elabels_fontsize=elabels_fontsize, node_color=node_color,node_size=node_size, nlabels_fontsize=nlabels_fontsize
p_Q = graphplot!(ax, g_Q_sub, elabels_color=elabels_color,
    edge_color=edge_color_Q,
    edge_width=edge_width_Q, node_size=node_size,
    arrow_size=arrow_size_Q, node_color=[edge_color_matrix[i, i] for i in 1:11],
    nlabels=node_labels_Q, nlabels_fontsize=nlabels_fontsize)
display(fig)
##
save("held_suarez_extreme_graph_n100.png", fig)