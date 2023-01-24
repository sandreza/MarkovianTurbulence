using MarkovChainHammer, Distributions, GLMakie, Random, Statistics

## Prototype a Bayesian Random Matrix
xrange = range(0, 5, length=100)
xvals = [x for x in xrange]
likelihood = Exponential(1 / 3)
num_obs = 1000
samples = rand(likelihood, num_obs)
c_samples = cumsum(samples)
plot_c_samples = [0, c_samples...]

μ = 3
σ² = 1.0
α₀ = μ^2 / σ²
β₀ = μ / σ²
fig = Figure()
sl1 = Slider(fig[2, 1], range=0:num_obs, startvalue=0)
obs = sl1.value
α = @lift α₀ + $obs
β = @lift β₀ + plot_c_samples[$obs+1]
θ = @lift 1 / $β
dist = @lift Gamma($α, $θ)
mean_value = @lift "μ = " * string(mean($dist))
prob = @lift [pdf($dist, x) for x in xrange]

ax = Axis(fig[1, 1]; title=mean_value)

lines!(ax, xvals, prob)
xlims!(0, 5)
ylims!(0, 3)

display(fig)

##
# Now we need the posterior predictive distribution
α = 2
β = 2
n = 10
αp = α + n # length(holding_times)
βp = β + plot_c_samples[n+1] # sum(holding_times)

μ = 0 # lomax
σ = βp / αp
ξ = 1 / αp
posterior_predictive_exponential_gamma = GeneralizedPareto(μ, σ, ξ)
likelihood = Exponential(1 / 3)
tmp = [pdf(posterior_predictive_exponential_gamma, x) for x in xrange]
tmp2 = [pdf(likelihood, x) for x in xrange]
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, tmp)
lines!(ax, tmp2, color=:red, linewidth=3)
display(fig)

mean(posterior_predictive_exponential_gamma)

##
# the marginal distributions of the posterior are Beta distributions with particular parameters
p = [0.1, 0.2, 0.3, 0.4]
n = 1
likelihood = Multinomial(n, p)
num_obs = 1000
samples = rand(likelihood, num_obs)
c_samples = cumsum(samples, dims=2)
plot_c_samples = zeros(Int64, length(p), num_obs + 1)
plot_c_samples[:, 2:end] .= c_samples

α = [1, 1, 1, 1] .+ plot_c_samples[:, end]
dist = Dirichlet(α)

# DirichletMultinomial does exist 
posterior_samples = []
β1 = []
β2 = []
β3 = []
β4 = []
βs = [β1, β2, β3, β4]
# α₀ const is how seriously to take the data
for i in 1:num_obs+1
    α = 2 .* [1, 2, 3, 4] .+ plot_c_samples[:, i]
    dist = Dirichlet(α)
    for i in 1:4
        push!(βs[i], Beta(α[i], sum(α) - α[i]))
    end
    # push!(posterior_samples, rand(dist, 100))
end

xrange = range(0, 1, length=101)
fig = Figure()
sl1 = Slider(fig[3, 1:2], range=0:num_obs, startvalue=0)
obs = sl1.value
for i in 1:4
    ii = (i - 1) ÷ 2 + 1
    jj = (i - 1) % 2 + 1

    dist = @lift βs[i][$obs+1]
    prob = @lift [pdf($dist, x) for x in xrange]
    mean_value = @lift string(i) * "μ = " * string(mean($dist))
    ax = Axis(fig[ii, jj]; title=mean_value)
    # dist = @lift posterior_samples[$obs+1][i, :]
    # hist!(ax11, dist, bins=10)
    lines!(ax, xrange, prob)
    xlims!(ax, (0, 1))
    ylims!(ax, (0, 10))
end
display(fig)
p̂ = plot_c_samples[:, end] / sum(plot_c_samples[:, end])
##
# We have the likelihood functions, and we have the prior distributions 


##
p = [0.1, 0.2, 0.3, 0.4]
α = [1, 2, 3, 4] # * (eps(1.0))
αp = α .+ plot_c_samples[:, end] # length.(holding_times)
n = size(plot_c_samples)[2] # choose as length.(holding_times),  is the appropriate n the number of observations or is it something else like the sum of alpha
# n = sum(αp) has roughly twice the variance as just the multinomial distribution
# for n = 0, need some kind of prior predictive distribution, n is god given here. 
# We can do conservative estimates or nonconservative estimates. n =1 with p drawn from the dirichlet distribution is one option
# another option is just to take the estimate from the random dirichlet distribution. This isn't a crazy thing that we need to know how to handle 
# We need to figure out what happens when we don' thave any observations of a given column. Since we are uncertain, it seems more conservative to 
# assume that the probalities are drawn from the binomial n=1 distribution so that we have a DirichletMultinomial(n, α) distribution where
# α characterizes the prior distribution. The question is one of sparse connectivity or full connectivity of the steady distribution. 
# it seems like the sparse connectivity is the more conservative estimate.
# if we don't observe something it must be poorly connected to go in, or have many ways of exiting. On the other hand, as soon as we make one observation, 
# there is a large discrepancy between something that is chosen uniformly and what is done by the DirichletMultinomial distribution.
# Thus we have settled on letting it be the Dirichlet Multinomial distribution with one observation
# n is the number of observations, this will be the sum of the count operator without the diagonal entries
posterior_predictive_multinomial_dirichlet = DirichletMultinomial(n, αp)
likelihood = Multinomial(n, plot_c_samples[:, end] / sum(plot_c_samples[:, end]))
prior = Dirichlet(α)

mean(posterior_predictive_multinomial_dirichlet)
mean(likelihood)

var(posterior_predictive_multinomial_dirichlet)
var(likelihood)

rand(posterior_predictive_multinomial_dirichlet, 1000)
rand(likelihood, 10)

## Now for abstractions
# Prior Abstraction 
abstract type AbstractPriorDistributionParameters end

# probably use the same abstraction here: RENAME HOLDING TIMES TO RATES
# GeneratorParameterDistributions and have abstract distribution parameters
struct GeneratorPriorParameters{H,P} <: AbstractPriorDistributionParameters
    mean_rates::H
    exit_probabilities::P
end

function GeneratorPriorParameters(number_of_states::Int, inverse_time_scale::Float64)
    α = 2 / inverse_time_scale
    β = 2
    θ = 1 / β
    mean_rates = [Gamma(α, θ) for i in 1:number_of_states]
    α⃗ = ones(number_of_states - 1)
    exit_probabilities = [Dirichlet(α⃗) for i in 1:number_of_states]
    return GeneratorPriorParameters(mean_rates, exit_probabilities)
end

tmp = GeneratorPriorParameters(3, 10.0)

rand.(tmp.mean_rates)
rand.(tmp.exit_probabilities)

## Predictive Posterior Abstraction 
struct BayesianGenerator{PB,D,PA,PP}
    prior::PB
    data::D
    posterior::PA
    predictive_posterior::PP
end

using MarkovChainHammer.TransitionMatrix: generator, holding_times, steady_state, count_operator
##
# GeneratorParameterDistributions and have abstract distribution parameters
abstract type AbstractGeneratorParameterDistributions end

# for both the prior and the posterior
struct GeneratorParameterDistributions{H,P} <: AbstractGeneratorParameterDistributions
    mean_rates::H
    exit_probabilities::P
end

# account for finite sampling effects
struct GeneratorPredictiveDistributions{H,P} <: AbstractGeneratorParameterDistributions
    holding_times::H
    exit_counts::P
end

function BayesianGenerator(data, prior::GeneratorParameterDistributions; dt=1)
    number_of_states = length(prior.mean_rates)
    ht_data = holding_times(data, number_of_states; dt=dt)
    p_data = Int.(count_operator(data, number_of_states))
    p_data = p_data - Diagonal(diag(p_data))

    posterior_rates = Vector{Gamma{Float64}}(undef, number_of_states)
    posterior_exit_probabilities = Vector{Dirichlet{Float64,Vector{Float64},Float64}}(undef, number_of_states)
    predictive_holding_times = Vector{GeneralizedPareto{Float64}}(undef, number_of_states)
    predictive_exit_counts = Vector{DirichletMultinomial{Float64}}(undef, number_of_states)

    for i in 1:number_of_states
        number_of_exits = length(ht_data[i])
        α, θ = params(prior.mean_rates[i])
        # first handle the rates
        # posterior
        β = 1 / θ
        α_new = α + number_of_exits
        if number_of_exits > 0
            β_new = β + sum(ht_data[i])
        else
            @warn "no data for state $i, falling back on prior for posterior distribution"
            β_new = β
        end
        θ_new = 1 / β_new
        posterior_rates[i] = Gamma(α_new, θ_new)
        # predictive posterior 
        μ = 0 # lomax
        σ = β_new / α_new
        ξ = 1 / α_new
        predictive_holding_times[i] = GeneralizedPareto(μ, σ, ξ)

        # next the exit probabilities
        # posterior
        α⃗ = params(prior.exit_probabilities[i])[1]
        α⃗_new = α⃗ + p_data[i, [1:i-1..., i+1:number_of_states...]]
        posterior_exit_probabilities[i] = Dirichlet(α⃗_new)
        # predictive
        if number_of_exits > 0
            n = length(ht_data[i])
        else
            @warn "no data for state $i, falling back on DirichletMultinomial with one observation for predictive distribution"
            n = 1
        end
        predictive_exit_counts[i] = DirichletMultinomial(n, α⃗_new)
    end
    posterior = GeneratorParameterDistributions(posterior_rates, posterior_exit_probabilities)
    predictive = GeneratorPredictiveDistributions(predictive_holding_times, predictive_exit_counts)
    return BayesianGenerator(prior, data, posterior, predictive)
end

function construct_generator(rates, exit_probabilities)
    number_of_states = length(rates)
    generator = zeros(number_of_states, number_of_states)
    generator -= I
    for i in 1:number_of_states
        generator[[1:i-1..., i+1:number_of_states...], i] .= exit_probabilities[i]
        generator[:, i] *= rates[i]
    end

    return generator
end
##
function prior_parameter_distribution(number_of_states::Int, time_rates::Float64)
    α = 2 * time_rates
    β = 2
    θ = 1 / β
    rates = [Gamma(α, θ) for i in 1:number_of_states]
    α⃗ = ones(number_of_states - 1)
    exit_probabilities = [Dirichlet(α⃗) for i in 1:number_of_states]
    return GeneratorParameterDistributions(rates, exit_probabilities)
end

function prior_parameter_distribution(number_of_states::Int, time_rates::Float64)
    α = 2 * time_rates
    β = 2
    θ = 1 / β
    rates = [Gamma(α, θ) for i in 1:number_of_states]
    α⃗ = ones(number_of_states - 1)
    exit_probabilities = [Dirichlet(α⃗) for i in 1:number_of_states]
    return GeneratorParameterDistributions(rates, exit_probabilities)
end

function prior_parameter_distribution(number_of_states::Int, time_rates::Float64; α=2, β=2, α⃗=ones(number_of_states - 1))
    α = α * time_rates
    β = β
    θ = 1 / β
    rates = [Gamma(α, θ) for i in 1:number_of_states]
    exit_probabilities = [Dirichlet(α⃗) for i in 1:number_of_states]
    return GeneratorParameterDistributions(rates, exit_probabilities)
end
##
number_of_states = 3
rates = 1.0
prior = prior_parameter_distribution(3, 1.0)

data = markov_chain[1:1]
tmp = BayesianGenerator(data, prior; dt=dt)

rates = mean.(tmp.posterior.mean_rates)
exit_probabilities = mean.(tmp.posterior.exit_probabilities)
Q1 = construct_generator(rates, exit_probabilities)
rates = 1 ./ mean.(tmp.predictive_posterior.holding_times)
exit_counts = mean.(tmp.predictive_posterior.exit_counts)
exit_probabilities = [exit_counts[i] / sum(exit_counts[i]) for i in eachindex(exit_counts)]
Q3 = construct_generator(rates, exit_probabilities)
Q2 = generator(data, 3; dt=dt)

Q1
Q2
Q3

##
import Base: rand
function rand(Q::GeneratorParameterDistributions)
    rates = rand.(Q.mean_rates)
    exit_probabilities = rand.(Q.exit_probabilities)
    return construct_generator(rates, exit_probabilities)
end
rand(Q::GeneratorParameterDistributions, n::Int) = [rand(Q) for i in 1:n]
function rand(Q::GeneratorPredictiveDistributions)
    rates = 1 ./ rand.(Q.holding_times)
    exit_counts = rand.(Q.exit_counts)
    exit_probabilities = [exit_counts[i] / sum(exit_counts[i]) for i in eachindex(exit_counts)]
    return construct_generator(rates, exit_probabilities)
end
rand(Q::GeneratorPredictiveDistributions, n::Int) = [rand(Q) for i in 1:n]

function rand(Q::BayesianGenerator)
    return rand(Q.posterior)
end
rand(Q::BayesianGenerator, n::Int) = [rand(Q) for i in 1:n]


##
Q1 = BayesianGenerator(markov_chain[1:floor(Int, 2 * 10^4)], prior; dt=dt)
Q2 = BayesianGenerator(markov_chain[floor(Int, 2 * 10^4)+1:floor(Int, 2 * 10^5)], prior; dt=dt)
Q3 = BayesianGenerator(markov_chain[floor(Int, 2 * 10^5)+1:floor(Int, 2 * 10^6)], prior; dt=dt)
Q4 = BayesianGenerator(markov_chain[floor(Int, 2 * 10^6)+1:floor(Int, 2 * 10^7)], prior; dt=dt)

rand(Q1)

##
states = 100
prior_Q = prior_parameter_distribution(states, 1.0; α=100, β=100, α⃗=ones(states - 1) * 0.001)
Λ, V = eigen(rand(Q))
tmp = [eigen(rand(Q)) for i in 1:100]
##
fig = Figure()
ax = Axis(fig[1, 1])
scatter!(ax, real.(tmp[1].values), imag.(tmp[1].values), color=(:red, 0.1))
for i in eachindex(tmp)
    scatter!(ax, real.(tmp[i].values), imag.(tmp[i].values), color=(:red, 0.1))
end

ax2 = Axis(fig[1, 2])
scatter!(ax2, reverse(real.(-1 ./ tmp[1].values)[1:end-1]), color=(:blue, 0.1))
for i in eachindex(tmp)
    scatter!(ax2, reverse(real.(-1 ./ tmp[i].values)[1:end-1]), color=(:blue, 0.1))
end
ylims!(ax2, (0, 2))

ax3 = Axis(fig[2, 1])
for i in eachindex(tmp)
    p = real.(tmp[i].vectors[:, end])
    p /= sum(p)
    scatter!(ax3, reverse(sort(p)), color=(:blue, 0.1))
end
ylims!(ax3, 0, 0.05)
display(fig)

#
hexbin(real.(tmp[1].values), imag.(tmp[1].values), cellsize=0.1, colormap=:plasma)
xs = Float64[]
ys = Float64[]
for i in eachindex(tmp)
    xs = [xs..., real.(tmp[i].values)...]
    ys = [ys..., imag.(tmp[i].values)...]
end
##
hexbin(xs, ys, cellsize=0.04, colormap=:plasma)

##
states = length(union(markov_chain))
prior_Q = prior_parameter_distribution(states, 1.0; α=1.0, β=1.0, α⃗=ones(states - 1) * 0.1)
posterior_similar_Q = prior_parameter_distribution(states, 1.35; α=20.0, β=20.0, α⃗=ones(states - 1)*10 )
#  1.35; α=20.0, β=20.0, α⃗=ones(states - 1)*10 , similar to posterior
# increasing certainty about holding times is what does the trick

##
Qtotal = BayesianGenerator(markov_chain, prior_Q; dt=dt_days)

fig = Figure()
ax = Axis(fig[1, 1]; title = "Decorrelation Times", xlabel = "eigenvalue index", ylabel = "Decorrelation Time (days)")

for i in 1:1
    scatter!(ax, reverse(sort(real.(-1 ./ eigen(rand(Qtotal)).values)[1:end-1])), color = (:blue, 0.3), label = "posterior")
    scatter!(ax, reverse(sort(real.(-1 ./ eigen(rand(prior_Q)).values)[1:end-1])), color = (:green, 0.3), label = "prior")
end
for i in 2:30
    scatter!(ax, reverse(sort(real.(-1 ./ eigen(rand(Qtotal)).values)[1:end-1])), color=(:blue, 0.1))
    scatter!(ax, reverse(sort(real.(-1 ./ eigen(rand(prior_Q)).values)[1:end-1])), color=(:green, 0.1))
end
scatter!(ax, reverse(sort(real.(-1 ./ eigen(Q).values)[1:end-1])), color=(:red, 0.5), label="empirical", markersize = 15)
axislegend(ax, position=:rt, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=100, labelsize=40)
ylims!(ax, (0, 3))
display(fig)
##
Nfull = floor(Int, length(markov_chain))
N2 = floor(Int, Nfull / 2)
N10 = floor(Int, Nfull / 10)
N100 = floor(Int, Nfull / 100)

Q1 = BayesianGenerator(markov_chain[N10+1:end], prior_Q; dt=dt_days)
Q10 = BayesianGenerator(markov_chain[N100+1:N10], prior_Q; dt=dt_days)
Q100 = BayesianGenerator(markov_chain[1:N100], prior_Q; dt=dt_days)
Q2_p1 = BayesianGenerator(markov_chain[N2+1:end], prior_Q; dt=dt_days)
Q2_p2 = BayesianGenerator(markov_chain[1:N2], prior_Q; dt=dt_days)