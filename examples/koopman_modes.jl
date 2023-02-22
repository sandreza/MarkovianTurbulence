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
ht1 = holding_times(markov_embedding_2, dt=dt_days)
ht2 = holding_times(markov_embedding_2_2, dt=dt_days)
ht_12 = [[ht1[i]..., ht2[i]...] for i in 1:400]

Q̅ = mean(Q)
p̅ = steady_state(Q̅)
Q̃ = Diagonal(1 ./ sqrt.(p̅)) * Q̅ * Diagonal(sqrt.(p̅))
symmetric_Q̃ = Symmetric((Q̃ + Q̃') / 2)
antisymmetric_Q̃ = (Q̃ - Q̃') / 2
sQ = Diagonal(sqrt.(p̅)) * symmetric_Q̃ * Diagonal(1 ./ sqrt.(p̅))
aQ = Diagonal(sqrt.(p̅)) * antisymmetric_Q̃ * Diagonal(1 ./ sqrt.(p̅))
index_ordering = reverse(sortperm(p̅)) # order indices by probability
mean_holding_time = [-1 / Q̅[i, i] for i in eachindex(p)][index_ordering]
entropy(p̅)
connectivity_out = sum(Q̅ .> 1e-4, dims=1)[index_ordering]
connectivity_in = sum(Q̅ .> 1e-4, dims=2)[index_ordering]
Λ, V = eigen(Q̅)
timescales = real.(-1 ./ Λ)
imtimescales = abs.(imag.(-1 ./ Λ)) .> eps(1e8)
time_in_days = (0:length(markov_embedding_2)-1) .* dt_days
close(hfile1)
close(hfile2)
##
function autocovariance2(g⃗, Q::Eigen, timelist; progress=false)
    @assert all(real.(Q.values[1:end-1]) .< 0) "Did not pass an ergodic generator matrix"

    autocov = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = Q
    p = real.(V[:, end] ./ sum(V[:, end]))
    v1 = V \ (p .* g⃗)
    w1 = g⃗' * V
    μ = sum(p .* g⃗)
    progress ? iter = ProgressBar(eachindex(timelist)) : iter = eachindex(timelist)
    for i in iter
        autocov[i] = real(w1 * (exp.(Λ .* timelist[i]) .* v1) - μ^2)
    end
    return autocov
end
##
using GLMakie
import MarkovChainHammer.TransitionMatrix: koopman_modes
import MarkovChainHammer.Utils: autocovariance
Qchoice = mean(Q)
𝒦 = koopman_modes(Qchoice)
EigenQ = eigen(Qchoice)
Λ, V = EigenQ
timescales = -1 ./ Λ
index1 = 50 # minimum([argmax(imag.(-1 ./ Λ)), 399])
index2 = 395
fastest = [real(𝒦[index1, markov_index]) for markov_index in markov_embedding_2]
slowest = [real(𝒦[index2, markov_index]) for markov_index in markov_embedding_2]
steps = 1200
tmp_fast = autocovariance(fastest[1:1200000]; timesteps=steps)
tmp_slow = autocovariance(slowest[1:1200000]; timesteps=steps)
tlist = time_in_days[1:steps]

tmp_fast1 = autocovariance2(real(𝒦[index1, :]), EigenQ, tlist)
tmp_slow2 = autocovariance2(real(𝒦[index2, :]), EigenQ, tlist)
println("slowest ", 2π/imag(Λ[index2])) 
##
fig = Figure(resolution=(2400, 1400))
labelsize = 40
options = (; titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
common_options = (; linewidth=7)
ax11 = Axis(fig[1, 1]; title="Koopman Mode Time series", xlabel="Time [days]", options...)
lines!(ax11, tlist, fastest[1:steps]; color=:red, common_options..., label="Mode 351")
lines!(ax11, tlist, slowest[1:steps]; color=:blue, common_options..., label="Mode 6")
axislegend(ax11, position=:lb, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
ax12 = Axis(fig[1, 2]; title="Koopman Mode Decorrelation", xlabel="Time [days]", options...)

lines!(ax12, tlist, tmp_fast / tmp_fast[1]; color=(:red, 0.25), label = "Mode 6 Time series", common_options...)
lines!(ax12, tlist, tmp_slow / tmp_slow[1]; color=(:blue, 0.25), label = "Mode 351 Time series", common_options...)
lines!(ax12, tlist, tmp_slow2 / tmp_slow2[1]; color=(:blue, 1.0), linestyle=:dot, label = "Mode 6 Ensemble", common_options...)
lines!(ax12, tlist, tmp_fast1 / tmp_fast1[1]; color=(:red, 1.0), linestyle=:dot, label="Mode 351 Ensemble", common_options...)
axislegend(ax12, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
# lines!(ax12, tlist, exp.(-0.3 .* tlist) .* cos.(2π/13 * tlist); color=(:purple, 0.5), linestyle=:dash, linewidth = 10)
xlims!(ax12, (0, tlist[end]))

g⃗1 = real(𝒦[index1, index_ordering])
g⃗2 = real(𝒦[index2, index_ordering])
g⃗1 = g⃗1 / sqrt(sum(g⃗1 .^2 .* p̅ ) - sum(g⃗1 .* p̅ )^2)
g⃗2 = g⃗2 / sqrt(sum(g⃗2 .^2 .* p̅ ) - sum(g⃗2 .* p̅ )^2)
ax21 = Axis(fig[2, 1]; title="Koopman Mode 351", xlabel= "Partition Index", ylabel="Mode Amplitude", options...)
scatter!(ax21, g⃗1, color=:red)
ax22 = Axis(fig[2, 2]; title="Koopman Mode 6",  xlabel= "Partition Index", ylabel="Mode Amplitude", options...)
scatter!(ax22, g⃗2, color=:blue)
display(fig)
##
save("held_suarez_koopman_modes.png", fig)
##

