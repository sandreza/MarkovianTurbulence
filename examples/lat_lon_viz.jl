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

Q = mean(Q)
p = steady_state(Q)
close(hfile)
close(hfile2)
##
#=
function mean_variables(state, x⃗, Φ)
    ρ = state[1]
    ρu⃗ = SVector(state[2], state[3], state[4])
    ρe = state[5]
    x = x⃗[1]
    y = x⃗[2]
    z = x⃗[3]
    # spherical vectors
    r⃗ = SVector(x, y, z)
    ϕ⃗ = SVector(x * z, y * z, -(x^2 + y^2))
    λ⃗ = SVector(-y, x, 0)
    # normalize (using nested functions gives error)
    r⃗_norm = sqrt(r⃗' * r⃗)
    r⃗_norm = r⃗_norm ≈ 0.0 ? 1.0 : r⃗_norm
    ϕ⃗_norm = sqrt(ϕ⃗' * ϕ⃗)
    ϕ⃗_norm = ϕ⃗_norm ≈ 0.0 ? 1.0 : ϕ⃗_norm
    λ⃗_norm = sqrt(λ⃗' * λ⃗)
    λ⃗_norm = λ⃗_norm ≈ 0.0 ? 1.0 : λ⃗_norm
    u⃗ = ρu⃗ / ρ
    u = (λ⃗' * u⃗) / λ⃗_norm
    v = (ϕ⃗' * u⃗) / ϕ⃗_norm
    w = (r⃗' * u⃗) / r⃗_norm
    γ = 1.4
    p = (γ - 1) * (ρe - 0.5 * ρ * u⃗' * u⃗ - ρ * Φ)
    T = p / (ρ * 287)
    return [u, v, w, p, T]
end
=#
jlfile = jldopen("data/TraditionalSmallHeldSuarezStatisticsConsistent_Nev4_Neh6_Nq1_7_Nq2_7_Nq3_7_X_80.0.jld2")
uzonal = jlfile["firstmoment"]["u"]

hfile = h5open("data/viz_fields.h5", "r")
rlist = read(hfile["rlist"])
θlist = read(hfile["thetalist"])
ϕlist = read(hfile["philist"])


state = read(hfile["surface field 1"])

heatmap(θlist, ϕlist, state[:, :, 5],  colormap=:plasma, interpolate = true)

##
Λ, V = eigen(Q)
p = steady_state(Q)
p[index_ordering[3]]
zonal_state = read(hfile["zonal mean zonal wind 1"])[1, :, :] * p[1]
for i in 2:400
    zonal_state = zonal_state .+ read(hfile["zonal mean zonal wind $i"])[1, :, :] * p[i]
end
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
fig = Figure(resolution=(2700, 1400))
markov_indices = [1, 10, 100, 400]
colormap = :afmhot
labelsize = 40
options = (; titlesize=labelsize) 
for (ii,i) in enumerate([0, 2, 4, 6])
    ax1 = Axis(fig[ii, 1]; title="Temperature Markov State $(markov_indices[ii])", options...)
    ax2 = Axis(fig[ii, 2]; title="Mode $(i) Amplitude: Real", options...)
    ax3 = Axis(fig[ii, 3]; title="Mode $(i) Amplitude: Imaginary", options...)
    pλ = -V[:, 400-i]

    zonal_state2 = read(hfile["surface field 1"])[:, :, field_index] * pλ[1]
    for j in 2:400
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
        ax.yticklabelsize = 20 + 5
        ax.yticks[] = ([-80, -60, -30, 0, 30, 60, 80], ["80S", "60S", "30S", "0", "30N", "60N", "80N"])
        ax.xticks[] = ([-160, -120, -80, -40, 0, 40, 80, 120, 160], ["160W", "120W", "80W", "40W", "0", "40E", "80E", "120E", "160E"])
    end
end
display(fig)
##
save("held_suarez_temperature_koopman_mode_amplitudes.png", fig)
##
zonal_state2 = mean(uzonal, dims=1)[1, :, :]
pressure = read(hfile["pressure 1"])[1, 90, :]
##
fig = Figure()
ax1 = Axis(fig[1,1]; title = "Ensemble Average")
ax2 = Axis(fig[1,2]; title = "Time Average")
ax3 = Axis(fig[1, 3]; title = "Difference")
heatmap!(ax1, ϕlist, -pressure, zonal_state, colormap=:balance, colorrange=(-30, 30), interpolate = true)
heatmap!(ax2, ϕlist, -pressure, zonal_state2, colormap=:balance, colorrange=(-30, 30), interpolate=true)
heatmap!(ax3, ϕlist, -pressure, zonal_state2-zonal_state, colormap=:balance, colorrange=(-5, 5), interpolate=true)
display(fig)
##
using Random
include("contour_heatmap.jl")
##
contour_levels = collect(-8:4:28)
fig = Figure(resolution=(3000, 1000))
latitudes = ϕlist / π * 180 .- 90
labelsize = 50
markersize = 40
ax1 = Axis(fig[1,1]; title = "Ensemble Average", titlesize = 40)
contour_heatmap!(ax1, latitudes, pressure, zonal_state, contour_levels, (-40, 40), colormap=:balance, add_labels=true, labelsize = labelsize, markersize = markersize)
ax2 = Axis(fig[1, 2]; title="Time Average", titlesize = 40)
contour_heatmap!(ax2, latitudes, pressure, zonal_state2, contour_levels, (-40, 40), colormap=:balance, add_labels=true, labelsize = labelsize, markersize = markersize)
ax3 = Axis(fig[1, 3]; title="Difference", titlesize = 40)
contour_levels_Δ = collect(1:0.25:3)
contour_heatmap!(ax3, latitudes, pressure, abs.(zonal_state2 - zonal_state), contour_levels_Δ, (0, 4), colormap=:thermometer, add_labels=true, labelsize = labelsize, markersize = markersize)
hideydecorations!(ax2)
hideydecorations!(ax3)
display(fig)

##
save("held_suarez_ensemble_average_zonal_wind.png", fig)