using HDF5, Statistics, MarkovianTurbulence
using MarkovChainHammer, LinearAlgebra, GLMakie
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy
import MarkovChainHammer.Utils: histogram, autocovariance

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

function temperature_timeseries(markov_state, Φ)
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

function observable_choice(markov_state, Φ)
    ρ = markov_state[1,1, 1]
    ρu = markov_state[1,1, 2]
    ρv = markov_state[1,1, 3]
    ρw = markov_state[1,1, 4]
    ρe = markov_state[1,1, 5]
    γ = 1.4
    R_d = 287.0
    T = (γ - 1) .* (ρe .- 0.5 .* (ρu .^ 2 .+ ρv .^ 2 .+ ρw .^ 2) ./ ρ .- ρ .* Φ[1,1]) ./ (ρ .* R_d)
    return T
end
##

# Extreme States
nstates = 100
data_directory = pwd() * "/data/"
file_name1 = "markov_model_extreme_nstate_100.h5"
file_name = "p3_markov_model_even_time_nstate_100_extreme.h5"
hfile = h5open(data_directory * file_name, "r")
hfilegeo = h5open(data_directory * file_name1, "r")

jump_factor = 5 # forgot to save
Φ = read(hfilegeo["geopotential"])
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
extreme_markov_states = typeof(read(hfilegeo["markov state 1"]))[]
for i in 1:100
    push!(extreme_markov_states, read(hfilegeo["markov state $i"]))
end

markov_chain = read(hfile["markov embedding"])
Q = BayesianGenerator(markov_chain; dt=dt_days)
Q_extreme = mean(Q)
p_extreme = steady_state(Q_extreme)

time_in_days = (0:length(markov_chain)-1) .* dt_days
ht = holding_times(markov_chain, nstates; dt=dt_days)
close(hfile)

temperature_special_partition = observable_choice.(extreme_markov_states, Ref(Φ))

# Observables 
filename = "observables_test_2.h5"
data_directory = pwd() * "/data/"
ofile = h5open(data_directory * filename, "r")
observables = read(ofile["observables"])
prognostic_observables = observables[:, 7:end]
dt = read(ofile["dt"])
close(ofile)
Tlist = temperature_timeseries(prognostic_observables, Φ[1, 1])

## 400 State 

data_directory = pwd() * "/data/"
file_name = "markov_model_even_time_nstate_400.h5"
hfile1 = h5open(data_directory * file_name, "r")
file_name = "part2_markov_model_even_time_nstate_400.h5"
hfile2 = h5open(data_directory * file_name, "r")

markov_chain = [read(hfile1["markov embedding 2"])..., read(hfile2["markov embedding 2"])...]

hfile_markov = hfile1 # grab markov states from first file
hfile = hfile1 # grab other data from the file here

jump_factor = 5 # forgot to save
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
markov_embedding_0p5 = read(hfile["markov embedding 0p5"]) # L1/2
markov_embedding_1 = read(hfile["markov embedding 1"]) # L¹
markov_embedding_2 = read(hfile["markov embedding 2"]) # L²
markov_embedding_2_2 = read(hfile2["markov embedding 2"]) # L²
markov_states = []
for i in 1:400
    push!(markov_states, read(hfile_markov["markov state $i"]))
end
time_in_days = (0:length(markov_embedding_2)-1) .* dt_days

temperature_topology_partition = observable_choice.(markov_states, Ref(Φ))
# α + number_of_exits, β + sum(ht_data[i])
# prior distribution is 1 exit and means Δt
prior = GeneratorParameterDistributions(400; α=1, β=dt_days, αs=ones(399) * 1e-4)
Q = BayesianGenerator(markov_embedding_2, prior; dt=dt_days)
Q = BayesianGenerator(markov_embedding_2_2, Q.posterior; dt=dt_days)
ht1 = holding_times(markov_embedding_2, dt=dt_days)
ht2 = holding_times(markov_embedding_2_2, dt=dt_days)
ht_12 = [[ht1[i]..., ht2[i]...] for i in 1:400]

Q_400 = mean(Q)
p_400 = steady_state(Q_400)

## Check the observables
bins = [105, 20]
bin_index = 1
Tlimits = (minimum(Tlist), 290) # maximum(Tlist)
Tlimits = (0, 1.1)

xs_t, ys_t = histogram(Tlist .> 290; bins=bins[bin_index], custom_range= Tlimits)
xs_extreme, ys_extreme = histogram(temperature_special_partition .> 290; bins=bins[bin_index], custom_range=Tlimits, normalization = p_extreme)
xs_400, ys_400 = histogram(temperature_topology_partition .> 290; bins=bins[bin_index], custom_range=Tlimits, normalization=p_400)

labelsize = 40
options = (; xlabel="Observable", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution=(2500, 1250))
ax1 = Axis(fig[1, 1]; title = "Timeseries", titlesize=labelsize, options...)
ax2 = Axis(fig[1, 2]; title = "Observable Partition", titlesize=labelsize, options...)
ax3 = Axis(fig[1, 3]; title = "Generic Partition", titlesize=labelsize, options...)
barplot!(ax1, xs_t, ys_t, color=(:red, 0.5))
barplot!(ax2, xs_extreme, ys_extreme, color=(:blue, 0.5))
barplot!(ax3, xs_400, ys_400, color=(:blue, 0.5))
for ax in [ax1, ax2, ax3]
    xlims!(ax, (0.95, 1.05))
    ylims!(ax, (0, 0.08))
end


bin_index = 2
Tlimits = (minimum(Tlist), maximum(Tlist))
xs_t, ys_t = histogram(Tlist; bins=bins[bin_index], custom_range=Tlimits)
xs_extreme, ys_extreme = histogram(temperature_special_partition; bins=bins[bin_index], custom_range=Tlimits, normalization=p_extreme)
xs_400, ys_400 = histogram(temperature_topology_partition; bins=bins[bin_index], custom_range=Tlimits, normalization=p_400)

options = (; xlabel="Temperature", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
ax4 = Axis(fig[2, 1]; titlesize=labelsize, options...)
ax5 = Axis(fig[2, 2]; titlesize=labelsize, options...)
ax6 = Axis(fig[2, 3]; titlesize=labelsize, options...)
barplot!(ax4, xs_t, ys_t, color=(:red, 0.5))
barplot!(ax5, xs_extreme, ys_extreme, color=(:blue, 0.5))
barplot!(ax6, xs_400, ys_400, color=(:blue, 0.5))
for ax in [ax4, ax5, ax6]
    # xlims!(ax, (0.95, 1.05))
    ylims!(ax, (0, 0.3))
end

display(fig)
##
save("held_suarez_observable_comparison.png", fig)