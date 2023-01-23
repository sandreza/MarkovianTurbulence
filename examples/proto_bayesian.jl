using MarkovChainHammer, Distributions, GLMakie, Random, Statistics

## Prototype a Bayesian Random Matrix
xrange = range(0, 5, length=100)
xvals = [x for x in xrange]
likelihood = Exponential(1 / 3)
num_obs = 1000
samples = rand(likelihood, num_obs)
c_samples = cumsum(samples)
plot_c_samples = [0, c_samples...]

fig = Figure()
sl1 = Slider(fig[2, 1], range=0:num_obs, startvalue=0)
obs = sl1.value
α = @lift 2 + $obs
β = @lift 2 + plot_c_samples[$obs+1]
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

function BayesianGenerator(data, prior::GeneratorPriorParameters; dt=1)
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
        σ =  β_new / α_new
        ξ = 1/α_new
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
prior = GeneratorPriorParameters(3, 1.0)
data = markov_chain[1:100000]
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