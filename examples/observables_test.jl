using HDF5, Statistics, ProgressBars
using MarkovChainHammer, LinearAlgebra, GLMakie
import MarkovChainHammer.TransitionMatrix: generator, holding_times, perron_frobenius
import MarkovChainHammer.TransitionMatrix: steady_state, entropy, koopman_modes
import MarkovChainHammer.Utils: histogram

filename = "observables_test_2.h5"
data_directory = pwd() * "/data/"
ofile = h5open(data_directory * filename, "r")
observables = read(ofile["observables"])
dt = read(ofile["dt"])
close(ofile)

dt_days = dt * 80 / 86400

time_in_days = (0:length(observables[:, 1])-1) .* dt_days

scatter(observables[1:1000, 1])
##
function autocovariance(x; timesteps=length(x))
    μ = mean(x)
    autocor = zeros(timesteps)
    for i in 1:timesteps
        autocor[i] = mean(x[i:end] .* x[1:end-i+1]) - μ^2
    end
    return autocor
end

function autocorrelation(x; timesteps=length(x))
    μ = mean(x)
    autocor = autocovariance(x; timesteps=timesteps)
    return autocor ./ μ^2
end

function autocorrelation(g⃗, Q, timelist)
    autocor = zeros(length(timelist))
    # Q  = V Λ V⁻¹
    Λ, V = eigen(Q)
    p = steady_state(Q)
    v1 = V \ (p .* g⃗)
    w1 = g⃗' * V
    μ = sum(p .* g⃗)
    for i in ProgressBar(eachindex(timelist))
        autocor[i] = real(w1 * (exp.(Λ .* tlist[i]) .* v1)) / μ^2 - 1
    end
    return autocor
end

indexchoice = 2
g⃗_t = observables[1:end, indexchoice+6]
μ = mean(g⃗_t)

timesteps = 12000
autocor = autocorrelation(g⃗_t[1:timesteps])
# make sure 100_state has been loaded
observable_function(x) = x[1, 1, indexchoice]
g⃗ = observable_function.(markov_states)
tlist = collect(time_in_days)
modelval = zeros(length(tlist[1:400]))
for i in ProgressBar(eachindex(tlist[1:400]))
    modelval[i] = g⃗' * exp(Q * tlist[i]) * (p .* g⃗) / sum(p .* g⃗)^2 - 1
end

modelval2 = zeros(length(tlist[1:400]))
# Q  = V Λ V⁻¹
Λ, V = eigen(Q)
v1 = V \ (p .* g⃗)
w1 = g⃗' * V
μ = sum(p .* g⃗)
for i in ProgressBar(eachindex(tlist[1:400]))
    modelval2[i] = real(w1 * (exp.(Λ .* tlist[i]) .* v1)) / μ^2 - 1
end

# V⁻¹ exp(Λ t) V¹
# (V⁻¹)^T g⃗ and V * (p .* g⃗)

μ_m = sum(g⃗ .* p)
μ̂_m = mean(g⃗)
μ_t = mean(g⃗_t)

σ_m = sqrt(sum((g⃗ .^ 2) .* p) - μ_m^2)
σ_t = std(g⃗_t)
σ̂_m = std(g⃗)

##

fig = Figure()
ax = Axis(fig[1, 1])
tlist = time_in_days[1:400]
autocr = autocor[1:400]
automod = modelval[1:400]
scatter!(ax, tlist, autocr / autocr[1], color=:black)
scatter!(ax, tlist, automod / automod[1], color=:purple)
lines!(ax, tlist, exp.(-tlist / timescales[end-1]), color=:red, linestyle=:dash)
lines!(ax, tlist, exp.(-tlist / timescales[1]), color=:blue, linestyle=:dash)
display(fig)

##
fig = Figure()
ax_m = Axis(fig[1, 1])
ax_t = Axis(fig[1, 2])
rtimeseries = g⃗_t
bins = 400
xs_m, ys_m = histogram(g⃗, normalization=p, bins=bins, custom_range=extrema(rtimeseries))
xs_t, ys_t = histogram(rtimeseries, bins=bins, custom_range=extrema(rtimeseries))

barplot!(ax_m, xs_m, ys_m, color=:red)
barplot!(ax_t, xs_t, ys_t, color=:blue)
for ax in [ax_m, ax_t]
    xlims!(ax, extrema(rtimeseries))
    t1 = extrema(ys_m)
    t2 = extrema(ys_t)
    ylims!(ax, (0, 1.1 * maximum([t1[2], t2[2]])))
end
display(fig)

##
# Try together
labelsize = 40
options = (; xlabel="Observable", ylabel="Probability", titlesize=labelsize, ylabelsize=labelsize, xlabelsize=labelsize, xticklabelsize=labelsize, yticklabelsize=labelsize)
fig = Figure(resolution=(2500, 1250))
rtimeseries = g⃗_t
bin_options = [5, 10, 20, 400]
for (i, bins) in enumerate(bin_options)
    ii = (i - 1) % 2 + 1
    jj = (i - 1) ÷ 2 + 1

    ax = Axis(fig[ii, jj]; title="Bins = $bins", titlesize=labelsize, options...)
    xs_m, ys_m = histogram(g⃗, normalization=p, bins=bins, custom_range=extrema(rtimeseries))
    xs_t, ys_t = histogram(rtimeseries, bins=bins, custom_range=extrema(rtimeseries))

    barplot!(ax, xs_m, ys_m, color=(:red, 0.5), label="markov model")
    barplot!(ax, xs_t, ys_t, color=(:blue, 0.5), label="timeseries")
    axislegend(ax; position=:lt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=labelsize)


    xlims!(ax, extrema(rtimeseries))
    t1 = extrema(ys_m)
    t2 = extrema(ys_t)
    ylims!(ax, (0, 1.1 * maximum([t1[2], t2[2]])))
end

display(fig)

##
projected_g⃗ = [g⃗[markov_index] for markov_index in markov_embedding_2]
tmp = autocorrelation(projected_g⃗; timesteps=300)
##
fig = Figure()
ax = Axis(fig[1, 1])
tlist = time_in_days[1:300]
autocr = autocor[1:300]
automod = modelval[1:300]
scatter!(ax, tlist, autocr / autocr[1], color=:black)
scatter!(ax, tlist, automod / automod[1], color=:purple)
scatter!(ax, tlist, tmp / tmp[1], color=:orange)
display(fig)

##
import MarkovChainHammer.TransitionMatrix: koopman_modes
𝒦 = koopman_modes(Q)
fastest = [real(𝒦[1, markov_index]) for markov_index in markov_embedding_2]
slowest = [real(𝒦[end-1, markov_index]) for markov_index in markov_embedding_2]

tmp_fast = autocorrelation(fastest; timesteps=1000)
tmp_slow = autocorrelation(slowest; timesteps=1000)
##
all_modes = zeros(1000, 100)
for i in ProgressBar(1:100)
    tmp = [real(𝒦[i, markov_index]) for markov_index in markov_embedding_2]
    all_modes[:, i] .= autocorrelation(tmp; timesteps=1000)
end
##
fig = Figure()
ax = Axis(fig[1, 1])
tlist = time_in_days[1:1000]
common_options = (; linewidth=3)
lines!(ax, tlist, tmp_fast / tmp_fast[1]; color=:red, common_options...)
lines!(ax, tlist, tmp_slow / tmp_slow[1]; color=:blue, common_options...)
lines!(ax, tlist, exp.(-tlist / timescales[end-1]); color=:blue, linestyle=:dot, common_options...)
lines!(ax, tlist, exp.(-tlist / timescales[1]); color=:red, linestyle=:dot, common_options...)
lines!(ax, tlist, 0.6 * exp.(-tlist / 4); color=:purple, linestyle=:dot, common_options...)
display(fig)

##
fig = Figure()

for i in 1:5
    ax = Axis(fig[1, i])
    v = real(𝒦[end-i, index_ordering])
    v /= maximum(abs.(v))
    scatter!(ax, v)

    ax = Axis(fig[2, i])
    v = real(𝒦[i, index_ordering])
    v /= maximum(abs.(v))
    scatter!(ax, v)

end

display(fig)

##
fig = Figure(resolution=(3000, 1000))

mode_index = collect(1:4:100)
mode_index = collect(1:16:400)
mode_index[2] = 2
mode_index[end] = 100
for i in 1:25
    jj = (i - 1) ÷ 5 + 1
    ii = (i - 1) % 5 + 1
    mi = mode_index[i] # mode index, i.e. mi
    ax = Axis(fig[ii, jj]; title="Mode $mi")
    v = real(𝒦[end-mi+1, index_ordering])
    sv = sortperm(v)
    v = v[sv]
    v /= maximum(abs.(v))
    w = real(V[index_ordering, end-mi+1])
    w /= maximum(abs.(w))
    w = w[sv]
    scatter!(ax, v, markersize=5, color=:black)
    scatter!(ax, w, markersize=5, color=:red)
    ylims!(ax, (-1.1, 1.1))
end

display(fig)

##
sort(real(𝒦[end-1, :]))
##
fig = Figure()
ax = Axis(fig[1, 1])
tlist = time_in_days[1:1000]
common_options = (; linewidth=3)
for i in 1:100
    normalization = all_modes[1, i]
    abs(normalization) > eps(100.0) ? nothing : (normalization = 1)
    lines!(ax, tlist, all_modes[:, i] / normalization; common_options...)
end
display(fig)


##
using MarkovChainHammer.Trajectory: generate
using MarkovChainHammer.TransitionMatrix: count_operator
using Distributions
Q
Δt = 0.005
n = 10^5
Qs = typeof(Q)[]
for i in ProgressBar(1:1000)
    markov_series = generate(Q, n; dt=Δt)
    push!(Qs, generator(markov_series, 3; dt=Δt))
end

observable(x) = -x[1, 1]
observables = observable.(Qs)

hist(observables, bins=10, normed=true, color=:red, alpha=0.5)

std(observables) / mean(observables)

tmp = [Q[i, i] for i in 1:3]
probs = -Q ./ reshape(tmp, 1, 3)

tmp


ht_dis = []
for i in 1:3
    μ = log(-tmp[i])
    σ = 20 / sqrt(1 + n)
    D = LogNormal(μ, σ)
    push!(ht_dis, D)
end
std(ht_dis[2])
mean(ht_dis[2])
diags = copy(ht_dis)

co = count_operator(markov_chain)

tmp = co

random_Q = zeros(3,3)
λ2s = Float64[]
λ1s = Float64[]
for j in ProgressBar(1:1000)
    for i in 1:9
        μ = log(1+abs(tmp[i]))
        ii = (i-1) ÷ 3 + 1
        p = (tmp[i]) / (1+sum(co[:, ii])) # weight by empirical probability of being in state, tmp[i] +1 or not has different effect. No observations = 0 probability vs 1 probability
        σ = log(1 + p * 100 / sqrt(1 + abs(tmp[i])) )
        random_Q[i] = rand(LogNormal(μ, σ))
    end
    random_Q = random_Q ./ sum(random_Q, dims=1)
    random_Q = (random_Q-I ) / dt
    λs = eigvals(random_Q)
    push!(λ1s,λs[1])
    push!(λ2s,λs[2])
end

hist(λ1s)
eigvals(Q)
std(λ1s) 