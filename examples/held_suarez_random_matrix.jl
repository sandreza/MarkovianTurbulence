using HDF5, Statistics, MarkovianTurbulence
using MarkovChainHammer, LinearAlgebra, GLMakie
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy
import MarkovChainHammer.Utils: histogram

data_directory = pwd() * "/data/"
nstates = 400
file_name = "markov_model_even_time_nstate_" * string(nstates) * ".h5"
hfile = h5open(data_directory * file_name, "r")

jump_factor = 5 # forgot to save
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
unordered_markov_chain = read(hfile["markov embedding 2"]) # L²
time_in_days = (0:length(markov_chain)-1) .* dt_days

unordered_Q = generator(unordered_markov_chain; dt=dt_days)
unordered_p = steady_state(unordered_Q)
index_ordering = reverse(sortperm(unordered_p)) # order indices by probability
ordering_operator = sortperm(index_ordering)
markov_chain = [ordering_operator[unordered_markov_chain[i]] for i in eachindex(unordered_markov_chain)]
Q = generator(markov_chain; dt=dt_days)
mean_holding_time = [-1 / Q[i, i] for i in eachindex(p)]
p = steady_state(Q)
entropy(p)
connectivity_out = sum(Q .> 0, dims=1)
connectivity_in = sum(Q .> 0, dims=2)
Λ, V = eigen(Q)
timescales = -1 ./ real.(Λ)

Nfull = floor(Int, length(markov_chain))
N2 = floor(Int, Nfull / 2)
N10 = floor(Int, Nfull / 10)
N100 = floor(Int, Nfull / 100)
Q1 = RandomGeneratorMatrix(markov_chain[N10+1:end], nstates; dt=dt_days)
Q10 = RandomGeneratorMatrix(markov_chain[N100+1:N10], nstates; dt=dt_days)
Q100 = RandomGeneratorMatrix(markov_chain[1:N100], nstates; dt=dt_days)
Q2_p1 = RandomGeneratorMatrix(markov_chain[N2+1:end], nstates; dt=dt_days)
Q2_p2 = RandomGeneratorMatrix(markov_chain[1:N2], nstates; dt=dt_days)

Nrandom_arrays = 10000
tic = time()
Q1s = rand(Q1, Nrandom_arrays)
Q10s = rand(Q10, Nrandom_arrays)
Q100s = rand(Q100, Nrandom_arrays)
Q2_p1s = rand(Q2_p1, Nrandom_arrays)
Q2_p2s = rand(Q2_p2, Nrandom_arrays)
toc = time()
println("The amount of time in minutes to generate the random arrays are $( (toc - tic) / 60)")
Q̅ = mean(Q1s)
p̅ = steady_state(Q̅)

Qs = [Q100s, Q10s, Q2_p1s, Q2_p2s]
# Q1s, 
##
observables = [Q -> -1 / Q[i, i] for i in 1:9]
obs = [observable.(Q) for observable in observables, Q in Qs]
best_empirical = [observable(Q) for observable in observables]
##
Nbins = 1000
xys = []
for i in 1:9
    xrange = quantile.(Ref([obs[i, 4]..., obs[i, 3]..., obs[i, 2]...]), (0.0001, 0.9999))
    xy = [histogram(obs[i, j], bins=Nbins, custom_range=(0, 3)) for j in eachindex(Qs)]
    push!(xys, xy)
end
##
fig = Figure(resolution=(3000, 1500))
labelsize = 40
options = (; xlabel=" ", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
titlenames = ["1/Q₁₁", "1/Q₂₂", "1/Q₃₃", "1/Q₄₄", "1/Q₅₅", "1/Q₆₆", "1/Q₇₇", "1/Q₈₈", "1/Q₉₉"]
# https://docs.makie.org/v0.19/api/index.html#Axis 
# https://juliagraphics.github.io/Colors.jl/latest/namedcolors/
spine_colors = [:red, :blue, :orange]
time_pdf_colors = [:blue, :orange, :black, :red]
time_pdf_colors = [:gray, :maroon, :gold4, :dodgerblue]
time_pdf_labels = ["1 year", "10 year", "50 year (part 1)", "50 year (part 2)"]
opacities = [0.75, 0.75, 0.75, 0.75] .* 0.75
axs = []
for i in 1:9
    ii = (i - 1) % 3 + 1
    jj = (i - 1) ÷ 3 + 1
    # change spine colors
    spinecolor = (; bottomspinecolor=spine_colors[jj], topspinecolor=spine_colors[jj], leftspinecolor=spine_colors[ii], rightspinecolor=spine_colors[ii])
    othercolor = (; titlecolor=spine_colors[jj], xgridcolor=spine_colors[jj], ygridcolor=spine_colors[jj], xtickcolor=spine_colors[jj], ytickcolor=spine_colors[jj], xticklabelcolor=spine_colors[jj], yticklabelcolor=spine_colors[jj])
    ax = Axis(fig[ii, jj]; title=titlenames[i], spinewidth=10, options..., xgridvisible=false, ygridvisible=false)
    push!(axs, ax)
    for j in 1:4
        barplot!(ax, xys[i][j]..., color=(time_pdf_colors[j], opacities[j]), label=time_pdf_labels[j], gap=0.0)
    end
    if jj > 1
        hideydecorations!(ax)
    end
    xlims!(ax, (0.25, 3))
    ylims!(ax, (0, 0.03))
end
axislegend(axs[5], position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)
##
save("held_suarez_random_entries.png", fig)

##
eigenvalue_indices = 1:2:18
Qs = [Q2_p1s, Q2_p2s]
observables = [Q -> -1 / real(eigvals(Q)[end-i]) for i in eigenvalue_indices]
tic = time()
obs = [observable.(Q[1:1000]) for observable in observables, Q in Qs]
toc = time() 
println("the amount of time spent is ", toc - tic)
best_empirical = [observable(Q) for observable in observables]
##
Nbins = 20
xys = []
p = 0.001
for i in 1:9
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]...]), (p*0.1, 1-p))
    xy = [histogram(obs[i, j], bins=Nbins, custom_range=xrange) for j in eachindex(Qs)]
    push!(xys, xy)
end
##
fig = Figure(resolution=(3000, 1500))
labelsize = 40
options = (; xlabel=" ", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
titlenames = ["-1 / real(λᵢ) for i=" * string(i+1) for i in eigenvalue_indices ]
# https://docs.makie.org/v0.19/api/index.html#Axis 
# https://juliagraphics.github.io/Colors.jl/latest/namedcolors/
time_pdf_colors = [:gold4, :dodgerblue]
time_pdf_labels = ["50 year (part 1)", "50 year (part 2)"]
opacities = [0.75, 0.75] .* 0.75
axs = []
for i in 1:9
    ii = (i - 1) % 3 + 1
    jj = (i - 1) ÷ 3 + 1
    # change spine colors
    spinecolor = (; bottomspinecolor=spine_colors[jj], topspinecolor=spine_colors[jj], leftspinecolor=spine_colors[ii], rightspinecolor=spine_colors[ii])
    othercolor = (; titlecolor=spine_colors[jj], xgridcolor=spine_colors[jj], ygridcolor=spine_colors[jj], xtickcolor=spine_colors[jj], ytickcolor=spine_colors[jj], xticklabelcolor=spine_colors[jj], yticklabelcolor=spine_colors[jj])
    ax = Axis(fig[ii, jj]; title=titlenames[i], spinewidth=10, options..., xgridvisible=false, ygridvisible=false)
    push!(axs, ax)
    for j in 1:2
        barplot!(ax, xys[i][j]..., color=(time_pdf_colors[j], opacities[j]), label=time_pdf_labels[j], gap=0.0)
    end
    if jj > 1
        hideydecorations!(ax)
    end
    # xlims!(ax, (0.25, 3))
    # ylims!(ax, (0, 0.03))
end
for i in 4:9 
    xlims!(axs[i], (1.1, 1.9))
end
axislegend(axs[5], position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)
##
save("held_suarez_eigenvalue_scales.png", fig)