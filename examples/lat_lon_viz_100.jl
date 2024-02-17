using GLMakie
using HDF5
using JLD2
using Statistics

using HDF5, Statistics
using MarkovChainHammer, LinearAlgebra, GLMakie
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: count_operator, steady_state, entropy
import MarkovChainHammer.Utils: histogram

data_directory = pwd() * "/data/"
file_name = "markov_model_even_time_nstate_100.h5"
hfile1 = h5open(data_directory * file_name, "r")

markov_chain = read(hfile1["markov embedding 2"])

hfile_markov = hfile1 # grab markov states from first file
hfile = hfile1 # grab other data from the file here

jump_factor = 5 # forgot to save
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
markov_embedding_0p5 = read(hfile["markov embedding 0p5"]) # L1/2
markov_embedding_1 = read(hfile["markov embedding 1"]) # L¹
markov_embedding_2 = read(hfile["markov embedding 2"]) # L²
markov_states = []
for i in 1:100
    push!(markov_states, read(hfile_markov["markov state $i"]))
end
time_in_days = (0:length(markov_embedding_2)-1) .* dt_days

# α + number_of_exits, β + sum(ht_data[i])
# prior distribution is 1 exit and means Δt
Q = BayesianGenerator(markov_embedding_2; dt=dt_days)

Q = mean(Q)
p = steady_state(Q)
index_ordering = reverse(sortperm(p)) # order indices by probability
close(hfile)
##
hfile = h5open("data/viz_fields_100.h5", "r")
rlist = read(hfile["rlist"])
θlist = read(hfile["thetalist"])
ϕlist = read(hfile["philist"])


state = read(hfile["surface field 1"])

heatmap(θlist, ϕlist, state[:, :, 5],  colormap=:plasma, interpolate = true)

##
Λ, V = eigen(Q)
p = steady_state(Q)
p[index_ordering[3]]
##
#=
fig = Figure()
ax1 = Axis(fig[1,1]; title = "Mode 1 Average")
ax2 = Axis(fig[1,2]; title = "Mode 2 Average")
heatmap!(ax1, ϕlist, -pressure, real.(zonal_state2), colormap=:balance, colorrange = (-tmp,tmp), interpolate=true)
heatmap!(ax2, ϕlist, -pressure, imag.(zonal_state2), colormap=:balance, colorrange = (-tmp,tmp), interpolate=true)
=#
##
λ = collect(θlist) / 2π * 360
ϕ = ϕlist / π * 180 .- 90

field_index = 5 # 1: u, 2: v, 3: w, 4: p, 5: T
labelsize = 40
fig = Figure(resolution=(2700, 1400); )
markov_indices = [1, 10, 50, 100]
colormap = :afmhot
options = (; titlesize=labelsize) 
for (ii,i) in enumerate([0, 2, 4, 6])
    ax1 = Axis(fig[ii, 1]; title="Temperature Markov State $(markov_indices[ii])", options...)
    ax2 = Axis(fig[ii, 2]; title="Koopman Mode $(i): Real", options...)
    ax3 = Axis(fig[ii, 3]; title="Koopman Mode $(i): Imaginary", options...)
    pλ = -V[:, 100-i]

    zonal_state2 = read(hfile["surface field 1"])[:, :, field_index] * pλ[1]
    for j in 2:100
        zonal_state2 = zonal_state2 .+ read(hfile["surface field $j"])[:, :, field_index] * pλ[j]
    end
    c2 = quantile(real.(zonal_state2)[:], 0.99)
    c1 = quantile(real.(zonal_state2)[:], 0.01)
    lower, upper = extrema(read(hfile["surface field $(index_ordering[markov_indices[1]])"])[:, :, field_index][:])
    heatmap!(ax1, λ, ϕ, read(hfile["surface field $(index_ordering[markov_indices[ii]])"])[:, :, field_index], colormap=colormap, colorrange=(lower, upper), interpolate = true)
    heatmap!(ax2, λ, ϕ, real.(zonal_state2), colormap=colormap, colorrange=(c1, c2), interpolate=true)
    heatmap!(ax3, λ, ϕ, imag.(zonal_state2), colormap=colormap, colorrange=(c1, c2), interpolate=true)
    for ax in [ax1, ax2, ax3]
        ax.limits = (extrema(λ)..., extrema(ϕ)...)

        ax.xlabel = "Longitude [ᵒ]"
        ax.ylabel = "Latitude [ᵒ]"
        ax.xlabelsize = 25 + 5
        ax.ylabelsize = 25 + 5
        ax.xticklabelsize = 20 + 5
        ax.yticklabelsize = 15# 20 + 5
        ax.yticks[] = ([-80, -60, -30, 0, 30, 60, 80], ["80S", "60S", "30S", "0", "30N", "60N", "80N"])
        ax.xticks[] = ([-160, -120, -80, -40, 0, 40, 80, 120, 160], ["160W", "120W", "80W", "40W", "0", "40E", "80E", "120E", "160E"])
    end
end
display(fig)
close(hfile)
##
save("held_suarez_temperature_koopman_mode_amplitudes_100.png", fig)