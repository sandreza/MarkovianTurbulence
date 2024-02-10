include("lorenz_periodic.jl")
include("lorenz_random_points.jl")
##
inds= 1:10:iterations
fig = Figure(resolution=(1500, 1000))
axs = []
θ₀ = -π/5 
δ = -π/5
colors_ab = [(cmapa[markov_chain_ab[i]], 0.9) for i in inds]
ms = 12.0
ga = GridLayout(fig[1,1])
for i in 1:3
    ax = LScene(ga[1, i]; show_axis=false)
    scatter!(ax, Tuple.(markov_states_ab), color=:black, markersize=ms)
    scatter!(ax, timeseries[:, inds] , color=colors_ab, markersize = 5.0)
    scatter!(ax, Tuple.(markov_states_ab), color=:black, markersize=ms)
    rotate_cam!(ax.scene, (0, θ₀ + (i-1) * δ, 0))
    push!(axs, ax)
end
colors_random = [(cmapa[markov_chain_random[i]], 0.5) for i in inds]
for j in 1:3
    ax = LScene(ga[2, j]; show_axis=false)
    scatter!(ax, Tuple.(markov_states_random), color=:black, markersize=ms)
    scatter!(ax, timeseries[:, inds] , color=colors_random, markersize = 5.0)
    scatter!(ax, Tuple.(markov_states_random), color=:black, markersize=ms)
    rotate_cam!(ax.scene, (0, θ₀ + (j-1) * δ, 0))
    push!(axs, ax)
end
colgap!(ga, 0)
rowgap!(ga, -200)

display(fig)
##
save("lorenz_random_periodic_together.png", fig)

##
using LaTeXStrings
labels = [L"g^{[1]}", L"g^{[2]}", L"g^{[3]}"]
# labels = [L"$g^1$", L"g^2", L"g^3"]
op = 0.75
lw = 5
fs = 30
random_auto
ab_auto
timeseries_auto
fig = Figure(resolution = (1600, 425), fontsize = fs) 
total = 800
ts = dt .* collect(0:total-1)
for i in 1:3
    ii = (i-1) ÷ 3 + 1
    jj = (i-1) % 3 + 1
    ax = Axis(fig[ii, jj], title= labels[i], ylabel="Autocorrelation", xlabel = "Time", titlesize = 40, xgridstyle=:dash, ygridstyle=:dash,)
    lines!(ax, ts, ab_auto[i], color = :black, linewidth = lw, label = "Timeseries" )
    lines!(ax, ts, timeseries_auto[i], color = (:IndianRed, op), linewidth = lw, label = "AB Generator")
    lines!(ax, ts, random_auto[i], color = (:SteelBlue, op), linewidth = lw, label = "Sampling Generator")
    if i == 1
        axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=40, labelsize=fs)
    end
end
display(fig)
save("lorenz_random_periodic_together_auto.png", fig)