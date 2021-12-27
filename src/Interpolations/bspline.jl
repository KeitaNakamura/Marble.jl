"""
    BSpline{order}()
    LinearBSpline()
    QuadraticBSpline()
    CubicBSpline()

Create B-spline kernel.

# Examples
```jldoctest
julia> f = LinearBSpline()
LinearBSpline()

julia> Poingr.value(f, Vec(0.5, 0.5))
0.25
```
"""
struct BSpline{order} <: Kernel
    function BSpline{order}() where {order}
        new{order::Int}()
    end
end

const LinearBSpline    = BSpline{1}
const QuadraticBSpline = BSpline{2}
const CubicBSpline     = BSpline{3}

support_length(::BSpline{1}) = 1.0
support_length(::BSpline{2}) = 1.5
support_length(::BSpline{3}) = 2.0
support_length(::BSpline{4}) = 2.5
active_length(bspline::BSpline) = support_length(bspline) # for sparsity pattern

@pure nnodes(bspline::BSpline, ::Val{dim}) where {dim} = prod(nfill(Int(2*support_length(bspline)), Val(dim)))


fract(x) = x - floor(x)

# `x` must be normalized by `dx`
function Base.values(::BSpline{1}, x::T) where {T <: Real}
    ξ = fract(x)
    Vec{2, T}(1-ξ, ξ)
end

function Base.values(::BSpline{2}, x::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    @. $V(0.5, -1.0, 0.5) * ξ^2 +
       $V(-1.5, 0.0, 1.5) * ξ +
       $V(1.125, 0.75, 1.125)
end

function Base.values(::BSpline{3}, x::T) where {T <: Real}
    V = Vec{4, T}
    x′ = fract(x)
    ξ = x′ .- V(-1, 0, 1, 2)
    ξ² = ξ .* ξ
    ξ³ = ξ² .* ξ
    @. $V(-1/6, 0.5, -0.5, 1/6) * ξ³ +
       $V(1, -1, -1, 1) * ξ² +
       $V(-2, 0, 0, 2) * ξ +
       $V(4/3, 2/3, 2/3, 4/3)
end

@generated function Base.values(bspline::BSpline, x::Vec{dim}) where {dim}
    exps = [:(values(bspline, x[$i])) for i in 1:dim]
    quote
        otimes($(exps...))
    end
end

function value(::BSpline{1}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? 1 - ξ : zero(ξ)
end

function value(::BSpline{2}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (3 - 4ξ^2) / 4 :
    ξ < 1.5 ? (3 - 2ξ)^2 / 8 : zero(ξ)
end
@inline function value(::BSpline{2}, ξ::Float64)
    ξ = abs(ξ)
    ξ < 0.5 ? 0.75 - ξ^2         :
    ξ < 1.5 ? 0.125 * (3 - 2ξ)^2 : zero(ξ)
end

function value(::BSpline{3}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
end
@inline function value(::BSpline{3}, ξ::Float64)
    ξ = abs(ξ)
    ξ < 1 ? ξ^3/2 - ξ^2 + 2/3 :
    ξ < 2 ? (2 - ξ)^3 / 6     : zero(ξ)
end

function value(::BSpline{4}, ξ::Real)
    ξ = abs(ξ)
    ξ < 0.5 ? (48ξ^4 - 120ξ^2 + 115) / 192 :
    ξ < 1.5 ? -(16ξ^4 - 80ξ^3 + 120ξ^2 - 20ξ - 55) / 96 :
    ξ < 2.5 ? (5 - 2ξ)^4 / 384 : zero(ξ)
end

@generated function value(bspline::BSpline, ξ::Vec{dim}) where {dim}
    exps = [:(value(bspline, ξ[$i])) for i in 1:dim]
    quote
        @_inline_meta
        *($(exps...))
    end
end

function value(spline::BSpline{1}, ξ::Real, pos::NodePosition)::typeof(ξ)
    value(spline, ξ)
end

function value(spline::BSpline{2}, ξ::Real, pos::NodePosition)::typeof(ξ)
    if nthfrombound(pos) == 0
        ξ = abs(ξ)
        ξ < 0.5 ? (3 - 4ξ^2) / 3 :
        ξ < 1.5 ? (3 - 2ξ)^2 / 6 : zero(ξ)
    elseif nthfrombound(pos) == 1
        ξ = dirfrombound(pos) * ξ
        ξ < -1   ? zero(ξ)                 :
        ξ < -0.5 ? 4(1 + ξ)^2 / 3          :
        ξ <  0.5 ? -(28ξ^2 - 4ξ - 17) / 24 :
        ξ <  1.5 ? (3 - 2ξ)^2 / 8          : zero(ξ)
    else
        value(spline, ξ)
    end
end

function value(spline::BSpline{3}, ξ::Real, pos::NodePosition)::typeof(ξ)
    if nthfrombound(pos) == 0
        ξ = abs(ξ)
        ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 4 :
        ξ < 2 ? (2 - ξ)^3 / 4         : zero(ξ)
    elseif nthfrombound(pos) == 1
        ξ = dirfrombound(pos) * ξ
        ξ < -1 ? zero(ξ)                      :
        ξ <  0 ? (1 + ξ)^2 * (7 - 11ξ) / 12   :
        ξ <  1 ? (7ξ^3 - 15ξ^2 + 3ξ + 7) / 12 :
        ξ <  2 ? (2 - ξ)^3 / 6                : zero(ξ)
    else
        value(spline, ξ)
    end
end

@inline function value(bspline::BSpline, ξ::Vec{dim}, pos::NTuple{dim, NodePosition}) where {dim}
    prod(value.(Ref(bspline), ξ, pos))
end


struct BSplineValues{order, dim, T, L} <: MPValues{dim, T}
    F::BSpline{order}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    x::Base.RefValue{Vec{dim, T}}
    gridindices::MVector{L, Index{dim}}
    len::Base.RefValue{Int}
end

function BSplineValues{order, dim, T, L}() where {order, dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    gridindices = MVector{L, Index{dim}}(undef)
    x = Ref(zero(Vec{dim, T}))
    BSplineValues(BSpline{order}(), N, ∇N, x, gridindices, Ref(0))
end

function MPValues{dim, T}(F::BSpline{order}) where {order, dim, T}
    L = nnodes(F, Val(dim))
    BSplineValues{order, dim, T, L}()
end

function update!(mpvalues::BSplineValues{<: Any, dim}, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = mpvalues.F
    mpvalues.N .= zero(mpvalues.N)
    mpvalues.∇N .= zero(mpvalues.∇N)
    mpvalues.x[] = x
    update_gridindices!(mpvalues, grid, x, spat)
    dx⁻¹ = gridsteps_inv(grid)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        mpvalues.∇N[i], mpvalues.N[i] = gradient(x, :all) do x
            @_inline_meta
            ξ = (x - xᵢ) .* dx⁻¹
            value(F, ξ, node_position(grid, I))
        end
    end
    mpvalues
end

struct BSplineValue{dim, T}
    N::T
    ∇N::Vec{dim, T}
    x::Vec{dim, T}
    index::Index{dim}
end

@inline function Base.getindex(mpvalues::BSplineValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.x[], mpvalues.gridindices[i])
end
