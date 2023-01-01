using HDF5
using MarkovChainHammer
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy
import MarkovChainHammer.Utils: histogram

data_directory = pwd() * "/data/"
file_name = "markov_model_even_time_nstate_100.h5"
hfile = h5open(data_directory * file_name, "r")

jump_factor = 5
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
markov_embedding_0p5 = read(hfile["markov embedding 0p5"])
markov_embedding_1 = read(hfile["markov embedding 1"])
markov_embedding_2 = read(hfile["markov embedding 2"])

ht_0p5 = holding_times(markov_embedding_0p5, maximum(markov_embedding_0p5); dt=dt_days)
ht_1 = holding_times(markov_embedding_1, maximum(markov_embedding_1); dt=dt_days)
ht_2 = holding_times(markov_embedding_2, maximum(markov_embedding_2); dt=dt_days)

ordered_indices_0p5 = reverse(sortperm(length.(ht_0p5)))
ordered_indices_1 = reverse(sortperm(length.(ht_1)))
ordered_indices_2 = reverse(sortperm(length.(ht_2)))

Q = generator(markov_embedding_2; dt=dt_days)
p = steady_state(Q)
entropy(p)
Λ, V = eigen(Q)
timescales = -1 ./ real.(Λ)
##
# Timescales captured 
fig = Figure(resolution=(2 * 1400, 900))
labelsize = 40
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)

ax11 = Axis(fig[1, 1]; title="Decorrelation Timescales", ylabel="Time [days]", xlabel="Eigenvalue Index", options...)
scatter!(ax11, 1:length(timescales)-1, timescales[1:end-1], color = :blue, markersize=20.0)
ylims!(ax11, 0, 2)

ax12 = Axis(fig[1, 2]; title="State Probabilities", ylabel="Probability", xlabel="State Index", options...)
scatter!(ax12, reverse(sort(p)), markersize=20.0, color = :blue, label= "Empirical")
scatter!(ax12, ones(length(p)) ./ length(p), color=:black, markersize=20.0, label="Uniform")
axislegend(ax12, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
ylims!(ax12, -0.01, 0.1)

fig
##
# Holding times
ht = ht_2
ordered_indices = ordered_indices_2
bins = [5, 20, 100]
color_choices = [:red, :blue, :orange] # same convention as before
index_names = ["State " * string(i) for i in 1:length(p)]
hi = 1 #holding index
bin_index = 1 # bin index
labelsize = 40
options = (; xlabel="Time [days]", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution=(2800, 1800))
for hi in 1:3, bin_index in [1,2, 3]
    ax = Axis(fig[hi, bin_index]; title=index_names[hi] * " Holding Times " * ", " * string(bins[bin_index]) * " Bins", options...)
    holding_time_index = ordered_indices[hi]

    holding_time_limits = (0, ceil(Int, maximum(ht[holding_time_index])))
    holding_time, holding_time_probability = histogram(ht[holding_time_index]; bins=bins[bin_index], custom_range=holding_time_limits)

    barplot!(ax, holding_time, holding_time_probability, color=color_choices[hi], label="Data")
    λ = 1 / mean(ht[holding_time_index])

    Δholding_time = holding_time[2] - holding_time[1]
    exponential_distribution = @. (exp(-λ * (holding_time - 0.5 * Δholding_time)) - exp(-λ * (holding_time + 0.5 * Δholding_time)))
    lines!(ax, holding_time, exponential_distribution, color=:black, linewidth=3)
    scatter!(ax, holding_time, exponential_distribution, color=:black, markersize=20, label="Exponential")
    axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
end
display(fig)

# save("held_suarez_holding_times_100.png", fig)
