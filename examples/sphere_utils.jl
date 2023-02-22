using StaticArrays

rad(x,y,z) = sqrt(x^2 + y^2 + z^2)
lat(x,y,z) = asin(z/rad(x,y,z)) # ϕ ∈ [-π/2, π/2] 
lon(x,y,z) = atan(y,x) # λ ∈ [-π, π) 

r̂ⁿᵒʳᵐ(x,y,z) = norm(@SVector[x,y,z]) ≈ 0 ? 1 : norm(@SVector[x, y, z])^(-1)
ϕ̂ⁿᵒʳᵐ(x,y,z) = norm(@SVector[x,y,0]) ≈ 0 ? 1 : (norm(@SVector[x, y, z]) * norm(@SVector[x, y, 0]))^(-1)
λ̂ⁿᵒʳᵐ(x,y,z) = norm(@SVector[x,y,0]) ≈ 0 ? 1 : norm(@SVector[x, y, 0])^(-1)

r̂(x,y,z) = r̂ⁿᵒʳᵐ(x,y,z) * @SVector([x, y, z])
ϕ̂(x,y,z) = ϕ̂ⁿᵒʳᵐ(x,y,z) * @SVector([x*z, y*z, -(x^2 + y^2)])
λ̂(x,y,z) = λ̂ⁿᵒʳᵐ(x,y,z) * @SVector([-y, x, 0] )


function to_sphere(u, v, w, x, y, z)
    u⃗ = @SVector([u, v, w])
    r̃ = r̂(x,y,z)
    ϕ̃ = ϕ̂(x,y,z)
    λ̃ = λ̂(x,y,z)
    return u⃗ ⋅ r̃, u⃗ ⋅ ϕ̃, u⃗ ⋅ λ̃ 
end

"""
vertical velocity 
meriodional velocity
then zonal velocity
"""
function to_sphere(ρ, ρu, ρv, ρw, x, y, z)
    u⃗ = @SVector([ρu, ρv, ρw]) / ρ
    r̃ = r̂(x, y, z)
    ϕ̃ = ϕ̂(x, y, z)
    λ̃ = λ̂(x, y, z)
    return u⃗ ⋅ r̃, u⃗ ⋅ ϕ̃, u⃗ ⋅ λ̃
end

lat(x[1,1], y[1,1], z[1,1])  / π * 180
lon(x[1,1], y[1,1], z[1,1])  / π * 180
rad(x[1,1], y[1,1], z[1,1])