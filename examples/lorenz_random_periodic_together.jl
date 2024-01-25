# include("lorenz_periodic.jl")
# include("lorenz_random_points.jl")

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