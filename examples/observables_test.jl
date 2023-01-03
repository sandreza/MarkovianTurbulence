using HDF5, Statistics
using MarkovChainHammer, LinearAlgebra, GLMakie
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy
import MarkovChainHammer.Utils: histogram

filename = "observables_test_1.h5"
data_directory = pwd() * "/data/"
ofile = h5open(data_directory * filename, "r")
observables = read(ofile["observables"])
dt = read(ofile["dt"])
close(ofile)

dt_days = dt * 80 / 86400

time_in_days = (0:length(observables[:, 1])-1) .* dt_days

scatter(observables[1:1000, 1])
##
i = 1

indexchoice = 5
μ = mean(observables[1:end, indexchoice])

timesteps = 1000
autocor = zeros(timesteps)
for i in 1:timesteps
    autocor[i] = mean(observables[i:end, indexchoice] .* observables[1:end-i+1, indexchoice])/μ^2 - 1
end

fig = Figure()
ax = Axis(fig[1, 1])
tlist = time_in_days[1:300]
autocr = autocor[1:300]
scatter!(ax, tlist, autocr)
scatter!(ax, tlist, autocr[1] .* exp.(-tlist / 0.75))
display(fig)