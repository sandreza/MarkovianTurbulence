# Note need extreme state to be loaded

using Clustering

Tmarkovchain = sub_markov_chain # (Tlist .> 290) .+ 1

lag = 6
embedding_dimension = 10
tuple_list = []
mclength = length(Tmarkovchain)
embedded_chain = zeros(embedding_dimension, mclength - lag * (embedding_dimension - 1))
for i in 1:embedding_dimension
    index_start = lag * (i - 1) + 1
    index_end = lag * (embedding_dimension - i)
    embedded_chain[i, :] .= Tmarkovchain[index_start:end-index_end]
    push!(tuple_list, (index_start, index_end))
end
ncluster = 200# embedding_dimension * 11
kmeansr = kmeans(embedded_chain, ncluster)
Q = generator(kmeansr.assignments, ncluster; dt=dt_days)
ll, vv = eigen(Q);
kmodes = inv(vv)
ll
##
p = steady_state(Q)
observable_m = 1 * real.(kmodes[end-12, :]) + 3.5 * real.(kmodes[end-1, :])# p .< (0.2 / ncluster)
# sum(p[observable_m])
autoTQ = autocovariance(kmeansr.centers[10, :], Q, tlist)
autoT = autocovariance(Tmarkovchain; timesteps=length(tlist))
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, tlist, autoT, color=:red)
scatter!(ax, tlist, autoT[1] * exp.(tlist ./ real(ll[end-1])), color=:blue)
scatter!(ax, tlist, autoT[1] * autoTQ / autoTQ[1], color=:green)
display(fig)

##
ht_extreme = holding_times((Tlist .> 290) .+ 1)
simulated_chain = []
push!(simulated_chain, 1)
for i in ProgressBar(1:3000)
    current_state = Int(simulated_chain[end])
    htempirical = rand(ht_extreme[current_state])
    for i in 1:htempirical
        push!(simulated_chain, current_state)
    end
    simulated_chain = vcat(simulated_chain...)
    if current_state == 1
        push!(simulated_chain, 2)
    else
        push!(simulated_chain, 1)
    end
end

##
ctmep = ContinuousTimeEmpiricalProcess(markov_chain)
simulated_chain = generate(ctmep, 20000, 1)
autocor = autocovariance((simulated_chain .< 11 .+ 1); timesteps=length(tlist))
##
ctmep = ContinuousTimeEmpiricalProcess((markov_chain .< 11) .+ 1)
simulated_chain = generate(ctmep, 10000, 1)
autocor2 = autocovariance(simulated_chain; timesteps=length(tlist))
##
scatter(autocor2, color=:red)
scatter!(autocor, color=:blue)
##
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, tlist, autoT, color=:red)
scatter!(ax, tlist, autoT[1] * exp.(tlist ./ real(ll[end-1])), color=:blue)
scatter!(ax, tlist, autoT[1] * autoTQ / autoTQ[1], color=:green)
scatter!(ax, tlist, autoT[1] * autocor / autocor[1], color=:yellow)
display(fig)