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
# Dual Limit
L = 1000
N = 4
M = L*4

Δt = 2π/M
dt = 2π/N
ts = Δt * collect(0:M+L-1) 
xs = cos.(ts)
ys = sin.(ts)
C(x,y) = (floor(Int, arctan(y, x) / 2π * N))% N + 1
chain = C.(xs, ys)

pf = perron_frobenius(chain[1:M+1])
ge = generator(chain; dt = dt )
pf =  I + dt * ge - pf
##
# Δt → 0 limit. 
lw = 10
color_choices = [:red, :blue, :cyan, :orange]
labels = ["1", "2", "3", "4"]
fig = Figure(resolution = (530, 485), fontsize = 40)
ax = Axis(fig[1,1])
for i in 1:N
    inds = 1+L*(i-1):L *i
    lines!(ax, xs[inds], ys[inds], color = color_choices[i], linewidth = lw, label = labels[i])
end
axislegend(ax, position=:cc, framecolor=(:grey, 0.5), patchsize=(50, 50), markersize=80, labelsize=40)
display(fig)
save("periodic_trajectory.png", fig)


