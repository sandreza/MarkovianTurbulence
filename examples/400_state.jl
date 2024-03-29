using HDF5, Statistics
using MarkovChainHammer, LinearAlgebra, GLMakie
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: count_operator, steady_state, entropy
import MarkovChainHammer.Utils: histogram

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

# α + number_of_exits, β + sum(ht_data[i])
# prior distribution is 1 exit and means Δt
prior = GeneratorParameterDistributions(400; α=1, β=dt_days, αs=ones(399) * 1e-4)
Q = BayesianGenerator(markov_embedding_2, prior; dt=dt_days)
Q = BayesianGenerator(markov_embedding_2_2, Q.posterior; dt=dt_days)
ht1 = holding_times(markov_embedding_2, dt = dt_days)
ht2 = holding_times(markov_embedding_2_2, dt = dt_days)
ht_12 = [[ht1[i]..., ht2[i]...] for i in 1:400]

Q = mean(Q)
p = steady_state(Q)
Q̃ = Diagonal(1 ./ sqrt.(p)) * Q * Diagonal(sqrt.(p))
symmetric_Q̃ = Symmetric((Q̃ + Q̃') / 2)
antisymmetric_Q̃ = (Q̃ - Q̃')/2
sQ = Diagonal(sqrt.(p)) * symmetric_Q̃ * Diagonal(1 ./ sqrt.(p))
aQ = Diagonal(sqrt.(p)) * antisymmetric_Q̃ * Diagonal(1 ./ sqrt.(p))
index_ordering = reverse(sortperm(p)) # order indices by probability, 129 -> 1, 272 -> 2, 310 -> 397, 196 -> 400
mean_holding_time = [-1 / Q[i, i] for i in eachindex(p)][index_ordering]
scaled_entropy(p)
connectivity_out = sum(Q .> 1e-4, dims=1)[index_ordering]
connectivity_in = sum(Q .> 1e-4, dims=2)[index_ordering]
Λ, V = eigen(Q)
timescales = -1 ./ real.(Λ)
imtimescales_bool = abs.(imag.(Λ)) .> eps(1e8)
imtimescales = 1 ./ imag.(Λ[imtimescales_bool]) * 2π
time_in_days = (0:length(markov_embedding_2)-1) .* dt_days
close(hfile1)
close(hfile2)
##
# Timescales captured 
fig = Figure(resolution=(2 * 1400, 2 * 900))
labelsize = 40
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)

ax11 = Axis(fig[1, 1]; title="Decorrelation Timescales", ylabel="Time [days]", xlabel="Eigenvalue Index", options...)
colors = [:blue for i in 1:399]
# colors[imtimescales[1:end-1]] .= :red
# colors = reverse(colors)
colors[1:8] .= :red
scatter!(ax11, 1:length(timescales)-1, reverse(timescales[1:end-1]), color=colors, markersize=20.0)
ylims!(ax11, 0, 1.7)

ax12 = Axis(fig[1, 2]; title="Cell Probabilities", ylabel="Probability", xlabel="Cell Index", options...)
scatter!(ax12, p[index_ordering], markersize=20.0, color=:blue, label="Empirical")
scatter!(ax12, ones(length(p)) ./ length(p), color=(:black, 0.15), markersize=20.0, label="Uniform")
axislegend(ax12, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
ylims!(ax12, -0.001, 0.031)

ax21 = Axis(fig[2, 1]; title="Connectivity", ylabel="# of Cells", xlabel="Cell Index", options...)
scatter!(ax21, 1:length(timescales), connectivity_out, color=(:blue, 1.0), markersize=30.0, label="Out")
scatter!(ax21, 1:length(timescales), connectivity_in, color=(:red, 0.5), markersize=20.0, marker=:diamond, label="In")
axislegend(ax21, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
ylims!(ax21, -10, 260)

ax22 = Axis(fig[2, 2]; title="Average Holding Time", ylabel="Time [days]", xlabel="Cell Index", options...)
scatter!(ax22, 1:length(timescales), mean_holding_time, color=:blue, markersize=20.0)
ylims!(ax22, 0, 1.5)

display(fig)
##
save("held_suarez_generator_properties_" * string(length(p)) * ".png", fig)

##
fig = Figure(resolution=(2 * 1400, 2 * 900))
labelsize = 40
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)

ax11 = Axis(fig[1, 1]; title="Oscillatory Timescales", ylabel="Time [days]", xlabel="Eigenvalue Index", options...)
colors = [:blue for i in 1:400]
# colors[imtimescales[1:end-1]] .= :red
# colors = reverse(colors)
colors[1:8] .= :red
scatter!(ax11, collect(1:length(timescales))[imtimescales_bool], abs.(reverse(imtimescales)), color=colors[imtimescales_bool], markersize=20.0)
display(fig)
extrema(abs.(reverse(imtimescales)))
##
save("held_suarez_generator_oscillatory_timescales_" * string(length(p)) * ".png", fig)
##
# Holding times
ht = ht_12
ordered_indices = index_ordering
bins = [5, 20, 100]
color_choices = [:red, :blue, :orange] # same convention as before
index_names = ["Cell " * string(i) for i in 1:length(p)]
hi = 1 #holding index
bin_index = 1 # bin index
labelsize = 40
options = (; xlabel="Time [days]", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution=(2800, 1800))
for hi in 1:3, bin_index in [1, 2, 3]
    ax = Axis(fig[hi, bin_index]; title=index_names[hi] * " Holding Times" * ", " * string(bins[bin_index]) * " Bins", options...)
    holding_time_index = ordered_indices[hi]

    holding_time_limits = (0, ceil(Int, maximum(ht[holding_time_index])))
    holding_time, holding_time_probability = histogram(ht[holding_time_index]; bins=bins[bin_index], custom_range=holding_time_limits)

    barplot!(ax, holding_time, holding_time_probability, color=(color_choices[hi], 0.5), gap=0.0, label="Data")
    λ = 1 / mean(ht[holding_time_index])

    Δholding_time = holding_time[2] - holding_time[1]
    exponential_distribution = @. (exp(-λ * (holding_time - 0.5 * Δholding_time)) - exp(-λ * (holding_time + 0.5 * Δholding_time)))
    lines!(ax, holding_time, exponential_distribution, color=(:black, 0.5), linewidth=3)
    scatter!(ax, holding_time, exponential_distribution, color=(:black, 0.5), markersize=20, label="Exponential")
    axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
end
display(fig)
##
save("held_suarez_holding_times_" * string(length(p)) * ".png", fig)

## 
# index ordered_indices_2[1] should become index 1
# thus need to permsort to find the conversion
conversion = sortperm(index_ordering)
embedding_ordered = [conversion[markov_index] for markov_index in markov_embedding_2]
##
indices = 1:1200
fig = Figure(resolution=(1700, 1000))
labelsize = 40

options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
ax = Axis(fig[1, 1]; title="Partition Dynamics", xlabel="Time [days]", ylabel="Cell Index", options...)
scatter!(ax, time_in_days[indices], embedding_ordered[indices], color=:black)
xlims!(ax, (0, 33))
ylims!(ax, -1, 401)
display(fig)
##
save("held_suarez_embedding_" * string(length(p)) * ".png", fig)