using HDF5, Statistics, MarkovianTurbulence, ProgressBars
using MarkovChainHammer, LinearAlgebra, GLMakie, Random
using MarkovChainHammer.BayesianMatrix
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy
import MarkovChainHammer.Utils: histogram

#=
Random.seed!(12345)
data_directory = pwd() * "/data/"
nstates = 100
file_name = "markov_model_even_time_nstate_" * string(nstates) * ".h5"
hfile = h5open(data_directory * file_name, "r")

jump_factor = 5 # forgot to save
dt = read(hfile["dt"]) * read(hfile["small planet factor"]) * jump_factor
dt_days = dt / 86400
unordered_markov_chain = read(hfile["markov embedding 2"]) # L²
=#
nstates = 400
data_directory = pwd() * "/data/"
file_name = "markov_model_even_time_nstate_400.h5"
hfile1 = h5open(data_directory * file_name, "r")
file_name = "part2_markov_model_even_time_nstate_400.h5"
hfile2 = h5open(data_directory * file_name, "r")
jump_factor = 5 # forgot to save
dt = read(hfile1["dt"]) * read(hfile1["small planet factor"]) * jump_factor
dt_days = dt / 86400
unordered_markov_chain_1 = read(hfile1["markov embedding 2"])
unordered_markov_chain_2 = read(hfile2["markov embedding 2"])
time_in_days = (0:2*length(unordered_markov_chain_1)-1) .* dt_days
close(hfile1)
close(hfile2)

prior = GeneratorParameterDistributions(400; α=1, β=dt_days, αs=ones(399) * 1e-4)
Q1 = BayesianGenerator(unordered_markov_chain_1, prior; dt=dt_days)
Q2 = BayesianGenerator(unordered_markov_chain_2, prior; dt=dt_days)
Q21 = BayesianGenerator(unordered_markov_chain_2, Q1.posterior; dt=dt_days)
unordered_Q = mean(Q21)

unordered_p = steady_state(unordered_Q)
index_ordering = reverse(sortperm(unordered_p)) # order indices by probability
ordering_operator = sortperm(index_ordering)
markov_chain_1 = [ordering_operator[unordered_markov_chain_1[i]] for i in eachindex(unordered_markov_chain_1)]
markov_chain_2 = [ordering_operator[unordered_markov_chain_2[i]] for i in eachindex(unordered_markov_chain_2)]

Q1 = BayesianGenerator(markov_chain_1, prior; dt=dt_days)
Q2 = BayesianGenerator(markov_chain_2, prior; dt=dt_days)
Q21 = BayesianGenerator(markov_chain_2, Q1.posterior; dt=dt_days)
Q = mean(Q21)
mean_holding_time = [-1 / Q[i, i] for i in eachindex(unordered_p)]
p = steady_state(Q)
entropy(p)
connectivity_out = sum(Q .> eps(100.0), dims=1)
connectivity_in = sum(Q .> eps(100.0), dims=2)
Λ, V = eigen(Q)
timescales = -1 ./ real.(Λ)
##
Nfull = floor(Int, length(markov_chain_1))
N2 = floor(Int, Nfull / 2)
N10 = floor(Int, Nfull / 10)
N100 = floor(Int, Nfull / 100)
Qtotal = Q21
Q10 = BayesianGenerator(markov_chain_1[N100+1:N10], prior; dt=dt_days)
Q100 = BayesianGenerator(markov_chain_1[1:N100], prior; dt=dt_days)
Nrandom_arrays = 1000
#=
tic = Base.time()
Q10s = rand(Q10, Nrandom_arrays)
Q100s = rand(Q100, Nrandom_arrays)
Q1s = rand(Q1, Nrandom_arrays)
Q2s = rand(Q2, Nrandom_arrays)
toc = Base.time()
Q̅ = mean(Q1s);
=#
Q̄ = mean(Q21)
p̅ = steady_state(Q̄);
Qs = [Q100, Q10, Q1, Q2];
# println("The amount of time in minutes to generate the random arrays are $( (toc - tic) / 60)")
# Q1s, 
##
observables = [Q -> -1 / Q[i, i] for i in 1:9]
# obs = [observable.(Q) for observable in observables, Q in Qs]
obs =  [zeros(Nrandom_arrays) for i in 1:9, j in 1:4] 
tic = Base.time()
for j in ProgressBar(1:4)
    Qrandom = Qs[j]
    for k in ProgressBar(1:Nrandom_arrays)
        Q̃ = rand(Qrandom)
        for i in 1:9
            @inbounds obs[i, j][k] = observables[i](Q̃)
        end
    end
end
toc = Base.time()
println("the time is ", toc - tic)

##
best_empirical = [observable(Q) for observable in observables]
##
Nbins = 100
xys = []
for i in 1:9
    xrange = quantile.(Ref([obs[i, 4]..., obs[i, 3]..., obs[i, 2]...]), (0.0001, 0.9999))
    xy = [histogram(obs[i, j], bins=Nbins, custom_range=(0, 3)) for j in eachindex(Qs)]
    push!(xys, xy)
end
##
fig = Figure(resolution=(3000, 1500))
labelsize = 40
options = (; xlabel="Days", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
titlenames = ["1/Q₁₁", "1/Q₂₂", "1/Q₃₃", "1/Q₄₄", "1/Q₅₅", "1/Q₆₆", "1/Q₇₇", "1/Q₈₈", "1/Q₉₉"]
# https://docs.makie.org/v0.19/api/index.html#Axis 
# https://juliagraphics.github.io/Colors.jl/latest/namedcolors/
spine_colors = [:red, :blue, :orange]
time_pdf_colors = [:blue, :orange, :black, :red]
time_pdf_colors = [:gray, :maroon, :gold4, :dodgerblue]
time_pdf_labels = ["1 year", "10 year", "100 year (part 1)", "100 year (part 2)"]
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
    # ylims!(ax, (0, 0.03))
end
axislegend(axs[5], position=:rt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=40)
display(fig)
##
save("held_suarez_random_entries_n" * string(nstates) * ".png", fig)

##
eigenvalue_indices = 1:2:18
observables = [Λ -> real(-1 / Λ[end-i]) for i in eigenvalue_indices]
obs = [zeros(Nrandom_arrays) for i in 1:9, j in 1:3]
Qs = [Q1, Q2, Qtotal];
tic = Base.time()
for j in ProgressBar(1:3)
    Qrandom = Qs[j]
    for k in ProgressBar(1:Nrandom_arrays)
        Q̃ = rand(Qrandom)
        Λ = eigvals(Q̃)
        for i in 1:9
            @inbounds obs[i, j][k] = observables[i](Λ)
        end
    end
end
toc = Base.time()
println("the time for eigenvalues is  ", toc - tic)
##
#=

Q_totals = rand(Qtotal, Nrandom_arrays)
Qs = [Q1s, Q2s, Q_totals]
tic = time()
obs = [observable.(Q) for observable in observables, Q in Qs]
toc = time()
println("the amount of time spent is ", toc - tic)
=#
#=
using JLD2
obs = jldopen("10k_eigenvalues.jld2")["obs"]
=#
best_empirical = [observable(Q) for observable in observables]
##
Nbins = 100
xys = []
p = 0.001
for i in 1:9
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]..., obs[i, 3]...]), (p, 1 - p*10))
    # xrange = (1, xrange[2])
    #=
    if nstates > 100
        if i > 3 
            xrange = (1.1, 1.75)
        end  
    else 
        if i > 5 
            xrange = (1.1, 1.75)
        end
    end
    =#

    xy = [histogram(obs[i, j], bins=Nbins, custom_range=xrange) for j in 1:3]
    push!(xys, xy)
end
##
fig = Figure(resolution=(3000, 1500))
labelsize = 40
options = (; xlabel="Days", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
titlenames = ["-real(1 / λᵢ) for i=" * string(i + 1) for i in eigenvalue_indices]
# https://docs.makie.org/v0.19/api/index.html#Axis 
# https://juliagraphics.github.io/Colors.jl/latest/namedcolors/
time_pdf_colors = [:gold4, :orchid, :black]
time_pdf_labels = ["100 year (part 1)", "100 year (part 2)", "200 year"]
spine_colors = [:red, :blue, :orange]
opacities = [0.75, 0.75, 0.75] .* 0.75
axs = []
for i in 1:9
    ii = (i - 1) % 3 + 1
    jj = (i - 1) ÷ 3 + 1
    # change spine colors
    spinecolor = (; bottomspinecolor=spine_colors[jj], topspinecolor=spine_colors[jj], leftspinecolor=spine_colors[ii], rightspinecolor=spine_colors[ii])
    othercolor = (; titlecolor=spine_colors[jj], xgridcolor=spine_colors[jj], ygridcolor=spine_colors[jj], xtickcolor=spine_colors[jj], ytickcolor=spine_colors[jj], xticklabelcolor=spine_colors[jj], yticklabelcolor=spine_colors[jj])
    ax = Axis(fig[ii, jj]; title=titlenames[i], spinewidth=5, options..., xgridvisible=false, ygridvisible=false)
    push!(axs, ax)
    for j in 1:3
        barplot!(ax, xys[i][j]..., color=(time_pdf_colors[j], opacities[j]), label=time_pdf_labels[j], gap=0.0)
    end
    vlines!(ax, best_empirical[i], color=(:cyan4, 1.0), linewidth=10, label="200 year (point)")
    if jj > 1
        hideydecorations!(ax)
    end
    # xlims!(ax, (0.25, 3))
    ylims!(ax, (0, 0.1))
end
# xlims!(axs[i], (1.1, 1.9))
p = 0.002
for i in 1:9
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]..., obs[i, 3]...]), (p, 1 - p*10))
    xlims!(axs[i], (0.9, xrange[2]))
end
for i in 2:4
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]..., obs[i, 3]...]), (p, 1 - p*10))
    xlims!(axs[i], xrange)
end
for i in 5:9
    xrange = quantile.(Ref([obs[i, 1]..., obs[i, 2]..., obs[i, 3]...]), (p, 1 - p * 10))
    xlims!(axs[i], (0.95, xrange[2]))
end
#=
for i in 4:9 
    xlims!(axs[i], (1.1, 1.9))
end
=#
axislegend(axs[5], position=:lt, framecolor=(:grey, 0.5), patchsize=(30, 30), markersize=100, labelsize=30)
display(fig)
##
save("held_suarez_eigenvalue_scales_n" * string(nstates) * ".png", fig)

#=
saving observables
using HDF5
hfile = h5open("data/held_suarez_eigenvalue_observable_save.h5", "w")
matrix_observables = zeros(size(obs)..., size(obs[1])...)
for i in 1:9, j in 1:2 
    matrix_observables[i, j, :] .= obs[i,j]
end
hfile["observables"] = matrix_observables 
close(hfile)
=#
##
#=
Q_totals = rand(Qtotal, 10)
eigenlist = [eigen(Q) for Q in Q_totals]
eigenvalues = [eigenlist[i].values for i in eachindex(eigenlist)]
eigenvectors = [eigenlist[i].vectors[:, end-1] for i in eachindex(eigenlist)]
[scatter!(real.(eigenvectors[i]), imag.(eigenvectors[i])) for i in 1:10]
##
n = 10
periodic = zeros(n, n)
for i in 1:n
    periodic[i, i] = -1
    periodic[i%n+1, i] = 1
end
anywhere = zeros(n, n)
anywhere[:] .= 1 / (n - 1)
anywhere[diagind(anywhere)] .= -1
lA, VA = eigen(anywhere)
lP, VP = eigen(periodic)

scatter(real.(VP[:, end-1]), imag.((VP[:, end-1])))
scatter([real(c)^2 + imag(c)^2 for c in VA[:, end-1]], [atan(imag(c), real(c)) for c in VA[:, end-1]])
=#
