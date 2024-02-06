using MarkovChainHammer, GLMakie, LinearAlgebra

function periodic_generator(ω, N)
    Q = zeros(N, N)
    T = 2π / ω
    τ = T / N # time spend in a partition
    for i in 1:N
        Q[i, i] = - 1/τ
        Q[i%N + 1, i] = 1/τ
    end
    return Q
end

function periodic_perron_frobenius(N)
    P = zeros(N, N)
    for i in 1:N
        P[i%N + 1, i] = 1
    end
    return P
end

ω = 1
N = 16
Δt = 2π/ (ω * N)
Q = periodic_generator(ω, N)
eigvals(Q)
P = periodic_perron_frobenius(N)
all(P .≈ I + Q * Δt)

Λₚ, V = eigen(P)
λₚ = [exp(im * Δt * k) for k in 0:N-1]
lnλₚ = [im * Δt * k for k in 0:N-1]
Λₙ, V = eigen(Q)
λₙ = [(exp(im * Δt * k) - 1) / Δt for k in 0:N-1]
##
fig = Figure()
ms = 50
op = 0.5
ax = Axis(fig[1,1])
scatter!(ax, real(Λₚ), imag(Λₚ), color = (:blue, op), markersize = ms)
scatter!(ax, real(λₚ), imag(λₚ), color = (:red, op), markersize = ms)
ax = Axis(fig[1,2])
scatter!(ax, real(Λₙ), imag(Λₙ), color = (:red, op), markersize = ms)
scatter!(ax, real(λₙ), imag(λₙ), color = (:blue, op), markersize = ms)
display(fig)
##
function arctan(y, x)
    if y ≥ 0
        atan(y, x)
    else
        atan(y, x) + 2π
    end   
end
## 
M = 4
Δt = 2π/M
ts = Δt * collect(0:M) 
xs = cos.(ts)
ys = sin.(ts)
C(x,y) = (round(Int, arctan(y, x) / 2π * M))% M + 1
C.(xs, ys)