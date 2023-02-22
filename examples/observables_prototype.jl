using MarkovianTurbulence
using HDF5, Statistics, ProgressBars
using MarkovChainHammer, LinearAlgebra, GLMakie
using StaticArrays
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy, koopman_modes
import MarkovChainHammer.Utils: autocovariance
import MarkovChainHammer.Utils: histogram

include("sphere_utils.jl")

filename = "observables_test_2.h5"
data_directory = pwd() * "/data/"
ofile = h5open(data_directory * filename, "r")
observables = read(ofile["observables"])
prognostic_observables = observables[:, 7:end]
dt = read(ofile["dt"])
close(ofile)

filename = "gridinfo.h5"
data_directory = pwd() * "/data/"
ofile = h5open(data_directory * filename, "r")
x = read(ofile["x"])
y = read(ofile["y"])
z = read(ofile["z"])
close(ofile)

file_name1 = "markov_model_extreme_nstate_100.h5"
hfilegeo = h5open(data_directory * file_name1, "r")
Φ = read(hfilegeo["geopotential"])[1,1]
close(hfilegeo)

dt_days = dt * 80 / 86400
time_in_days = (0:length(observables[:, 1])-1) .* dt_days

##
placeholder_t = zeros(4, length(observables[:, 1]))
for i in ProgressBar(eachindex(observables[:,1]))
    ρ, ρu, ρv, ρw, ρe = observables[i, 1+6:5+6]
    w, v, u = to_sphere(ρ, ρu, ρv, ρw, x[1,1], y[1,1], z[1,1])
    R = 287 
    γ = 1.4
    p = (γ - 1) * (ρe - 0.5 * ρ * (u^2 + v^2 + w^2) - ρ * Φ)
    T = p / (ρ * R)
    placeholder_t[1, i] = ρ
    placeholder_t[2, i] = u
    placeholder_t[3, i] = v
    placeholder_t[4, i] = T
end
g⃗_t = [placeholder_t[1, :], placeholder_t[2, :], placeholder_t[3, :], placeholder_t[4, :]]
##
placeholder_m = zeros(4, length(markov_states))
for i in eachindex(markov_states)
    ρ, ρu, ρv, ρw, ρe = markov_states[i][1,1,1:5]
    w, v, u = to_sphere(ρ, ρu, ρv, ρw, x[1,1], y[1,1], z[1,1])
    R = 287
    γ = 1.4
    p = (γ - 1) * (ρe - 0.5 * ρ * (u^2 + v^2 + w^2) - ρ * Φ)
    T = p / (ρ * R)
    placeholder_m[1,i] = ρ
    placeholder_m[2,i] = u
    placeholder_m[3,i] = v
    placeholder_m[4,i] = T
end
g⃗_m = [placeholder_m[1, :], placeholder_m[2, :], placeholder_m[3, :], placeholder_m[4, :]]
#=
indexchoices = [2, 3, 4, 5]
observable_function(x; indexchoice) = x[1, 1, indexchoice]
g⃗_t = [observables[1:end, indexchoice+6]  for indexchoice in indexchoices]
g⃗_m = [observable_function.(markov_states; indexchoice=indexchoice) for indexchoice in indexchoices]
=#
##
μ⃗_t = mean.(g⃗_t)
σ⃗_t = std.(g⃗_t)
μ⃗_m = [sum(g⃗_m[i] .* p) for i in eachindex(g⃗_m)]
σ⃗_m = [sqrt(sum((g⃗_m[i] .- μ⃗_m[i]) .^ 2 .* p)) for i in eachindex(g⃗_m)]

tlist = collect(time_in_days)[1:327]
autocovariance_t = [autocovariance(observable; timesteps=length(tlist)) for observable in g⃗_t]
autocovariance_m = [autocovariance(observable, Q, tlist) for observable in g⃗_m]
##
# autocorrelation 
fig = Figure(resolution = (2000,1000))
labelsize = 40
axis_options = (; xlabel="Time (days)", ylabel="Autocovariance", xgridstyle=:dash, ygridstyle=:dash, ygridwidth=5, xgridwidth=5, titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
common_options = (; linewidth=5)
titlelabels = ["g¹", "g²", "g³", "g⁴"]
for i in 1:4
    ii = (i - 1) % 2 + 1
    jj = (i - 1) ÷ 2 + 1
    ax = Axis(fig[ii, jj]; title = titlelabels[i], axis_options...)
    # lines!(ax, tlist, autocovariance_t[i][1] * exp.(-tlist / timescales[end-1]); color=:blue, linestyle=:dot, label = "markov upper bound", common_options...)
    # lines!(ax, tlist, autocovariance_t[i][1] * exp.(-tlist / timescales[1]); color=:red, linestyle=:dot, label = "markov lower bound", common_options...)
    lines!(ax, tlist, autocovariance_t[i]; color=:black, label = "Time series", common_options...)
    lines!(ax, tlist, autocovariance_m[i]; color=(:purple, 0.5), label = "Ensemble", common_options...)
    xlims!(ax, (0,10))
    if i == 1
        axislegend(ax; position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
    end
end
display(fig)
##
nstates = 400
save("held_suarez_autocovariance_n" * string(nstates) * ".png", fig)

##
# steady state 
labelsize = 40
options = (; xlabel="Observable", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution=(2500, 1250))
index = 2
observable_timeseries = g⃗_t[index]
observabe_steady_state = g⃗_m[index]

bin_options = [5, 10, 20, 400]
for (i, bins) in enumerate(bin_options)
    ii = (i - 1) % 2 + 1
    jj = (i - 1) ÷ 2 + 1

    ax = Axis(fig[ii, jj]; title="Bins = $bins", titlesize=labelsize, options...)
    xs_m, ys_m = histogram(observabe_steady_state, normalization=p, bins=bins, custom_range=extrema(observable_timeseries))
    xs_t, ys_t = histogram(observable_timeseries, bins=bins, custom_range=extrema(observable_timeseries))

    barplot!(ax, xs_m, ys_m, color=(:red, 0.5), label="Ensemble")
    barplot!(ax, xs_t, ys_t, color=(:blue, 0.5), label="Timeseries")
    axislegend(ax; position=:lt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=labelsize)


    xlims!(ax, extrema(observable_timeseries))
    t1 = extrema(ys_m)
    t2 = extrema(ys_t)
    ylims!(ax, (0, 1.1 * maximum([t1[2], t2[2]])))
end

display(fig)
##
save("held_suarez_steady_state_n" * string(nstates) * ".png", fig)