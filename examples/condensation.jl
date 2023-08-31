using MarkovChainHammer, LinearAlgebra
using MarkovChainHammer.TransitionMatrix: perron_frobenius, generator, steady_state
using MarkovChainHammer.Trajectory: generate

# Define the transition matrix
states = 12
Q = zeros(states, states)
for i in 1:states
    if i < states
        Q[i, i] = -1
        Q[i+1, i] = 1
    end
end
E = zeros(3, states)
E1 = [1, 2, 3, 6, 7, 9, 11, 12]
E2 = [4, 5, 10]
E3 = [8]
[E[1, j] = 1.0 for j in E1]
[E[2, j] = 1.0 for j in E2]
[E[3, j] = 1.0 for j in E3]

E⁺ = pinv(E)
Q1 = E * Q * E⁺

E⁺2 = pinv(E)
E⁺2[1, 1] = 0
E⁺2[end, 1] = 0 
E⁺2 = E⁺2 ./ sum(E⁺2, dims=1)
Q2 = E * Q * E⁺2

ms = [1, 1, 1, 2, 2, 1, 1, 3, 1, 2, 1, 1]
Qtrue = generator(ms)
Q2 - Qtrue
##
E = zeros(3, states)
E1 = [2, 3, 7, 11]
E2 = [4, 5, 6, 10, 12]
E3 = [1, 8, 9]
[E[1, j] = 1.0 for j in E1]
[E[2, j] = 1.0 for j in E2]
[E[3, j] = 1.0 for j in E3]

E⁺2 = pinv(E)
# E⁺2[1, :] .= 0
# E⁺2[end, :] .= 0 
E⁺2 = E⁺2 ./ sum(E⁺2, dims=1)
Q2 = E * Q * E⁺2
EQ = E * Q

ms = [3, 1, 1, 2, 2, 2, 1, 3, 3, 2, 1, 2]
all([all([ms[i] == 1 for i in E1]), all([ms[i] == 2 for i in E2]), all([ms[i] == 3 for i in E3])])
Qtrue = generator(ms)
Q2 - Qtrue

##
using Random
Random.seed!(1234)
ms = [1, 2, 3, 1, 2]
# ms = rand([1,2,3], 10) # rand([1,2,3,4, 5, 6, 7], 10) 
ss = 1
ms_seed = Int.([ones(ss)...,  (2 * ones(ss))..., (3 * ones(ss))...])
ms = Int64[]
for i in 1:20
    [push!(ms, ms_seed[i]) for i in eachindex(ms_seed)]
end
# ms = [ms..., 1]
# ms = [1, 2, 3, 1, 1] # [1, 1, 2, 2, 3, 2,1, 2, 3, 3, 1,3, 3, 1]
# ms = [1, 2, 3, 1, 2, 3, 1, 2, 3]
# ms = [1, 2, 3, 4, 5, 6, 7, 8]
# ms = [1, 1, 1, 2, 2, 1, 1, 3, 1, 2, 1, 1]
states = length(ms)
Q = zeros(states, states)
P = zeros(states, states)
for i in 1:states
    if i < states
        Q[i, i] = -1
        Q[i+1, i] = 1
        P[i+1, i] = 1
    end
end
# Q[end, end] = -1
# Q[1:end-1, end] .= 1/(states-1)


E = zeros(maximum(ms), length(ms))
[E[ms[i], i] = 1 for i in 1:length(ms)]
E⁺ = pinv(E)
# E⁺[1, :] .= 0
E⁺[end, :] .= 0 
E⁺ = E⁺ ./ sum(E⁺, dims=1)
Q2 = E * Q * E⁺
P2 = E * P * E⁺
EQ = E * Q
(pinv(E') * Q' * E')'

Q2
Qtrue = generator(ms)
display(Q2)
display(Qtrue)
display(Q2 - Qtrue)
##
# case 1: end on a state you haven't seen before 
# case 2: end on a state that you've seen before 
# case 1: usual MPI
# ms = generate(Q2, 1000); 
# ms = [2  2  1 3 4 4 4 4 3 3 3 3 3 5]

##
# LX = Y => L = Y * pinv(X * X') * X'
X = E[:, 1:end-1]
Y = E[:, 2:end]
# L = Y * pinv(X' * X) * X'
Pdmd = Y * pinv(X)
Pmch = perron_frobenius(ms)
Qmch = generator(ms)
# norm((X' * X) * Diagonal(sum(X' * X, dims = 1)[:] .^(-2))- pinv(X' * X))
# Y * (X' * X) * Diagonal(sum(X' * X, dims = 1)[:] .^(-2)) * X'
G = X * X'
A = X * Y'
Pedmd = (pinv(G) * A)'
Gnh = inv(sqrt.(G))
Gh = sqrt.(G)
U1, Σ, U2 = svd(Gnh * Y * X' * Gnh)
U1 * Diagonal(Σ) * U2' - Gnh * Y * X' * Gnh
Pmpdmd = (Gnh * U2 * U1' * Gh)'
p = steady_state(Qmch)
Punitary = exp(log(Pmch)  -Diagonal(p) *log(Pmch)' * inv(Diagonal(p))/2)


eigen(Qmch)
eigen((Qmch  -Diagonal(p) *Qmch' * inv(Diagonal(p)))/2)
Pim = exp((Qmch  -Diagonal(p) *Qmch' * inv(Diagonal(p)))/2)
