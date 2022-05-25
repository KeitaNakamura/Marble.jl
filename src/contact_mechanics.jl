struct CoulombFriction
    μ::Float64
    ϕ::Float64
    c::Float64
    separation::Bool
end

"""
    CoulombFriction(; parameters...)

Frictional contact using Mohr-Coulomb criterion.

# Parameters
* `μ`: friction coefficient (use `μ` or `ϕ`)
* `ϕ`: friction angle (radian)
* `c`: cohesion (default: `0`)
* `separation`: `true` or `false` (default: `false`).

If `separation` is `true`, continuum body can leave from the boundary surface.
"""
function CoulombFriction(; μ::Union{Real, Nothing} = nothing, ϕ::Union{Real, Nothing} = nothing, c::Real = 0, separation::Bool = false)
    ( isnothing(μ) &&  isnothing(ϕ)) && throw(ArgumentError("both `μ` and `ϕ` are not found"))
    (!isnothing(μ) && !isnothing(ϕ)) && throw(ArgumentError("both `μ` and `ϕ` are used, choose only one parameter"))
    isnothing(ϕ) && (ϕ = atan(μ))
    isnothing(μ) && (μ =  tan(ϕ))
    CoulombFriction(μ, ϕ, c, separation)
end

"""
    CoulombFriction(:sticky)

This is the same as the `CoulombFriction(; μ = Inf, separation = false)`.

---

    CoulombFriction(:slip; separation = false)

This is the same as the `CoulombFriction(; μ = 0, separation = false)`.
"""
function CoulombFriction(cond::Symbol; separation = false)
    cond == :sticky && return CoulombFriction(; μ = Inf, separation = false)
    cond == :slip   && return CoulombFriction(; μ = 0, separation)
    throw(ArgumentError("Use `:sticky` or `:slip` for contact condition"))
end

issticky(cond::CoulombFriction) = isinf(cond.μ) && !cond.separation
isslip(cond::CoulombFriction) = iszero(cond.μ) && iszero(cond.c)

"""
    contacted(::CoulombFriction, v::Vec, n::Vec)

Compute velocity `v` caused by contact.
The other quantities, which are equivalent to velocity such as momentum and force, are also available.
`n` is the unit vector normal to the surface.

# Examples
```jldoctest
julia> cond = CoulombFriction(:slip, separation = false);

julia> v = Vec(1.0, -1.0); n = Vec(0.0, 1.0);

julia> v + contacted(cond, v, n)
2-element Vec{2, Float64}:
 1.0
 0.0
```
"""
function contacted(cond::CoulombFriction, v::Vec{dim, T}, n::Union{Vec{dim, T}, Vec{dim, Int}})::Vec{dim, T} where {dim, T}
    v_sticky = -v # contact force for sticky contact
    issticky(cond) && return v_sticky
    d = v_sticky ⋅ n
    vn = d * n
    isslip(cond) && return ifelse(d > 0 || !cond.separation, vn, zero(vn))
    vt = v_sticky - vn
    if d > 0
        μ = T(cond.μ)
        c = T(cond.c)
        return vn + min(1, (c + μ*norm(vn))/norm(vt)) * vt # put `norm(vt)` inside of `min` to handle with deviding zero
    else
        return ifelse(!cond.separation, vn, zero(vn))
    end
end

function contacted(cond::CoulombFriction, v::Vec, n::Vec)
    contacted(cond, promote(v, n)...)
end
