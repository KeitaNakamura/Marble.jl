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

julia> Metale.value(f, Vec(0.5, 0.5))
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

getsupportlength(::BSpline{1}) = 1.0
getsupportlength(::BSpline{2}) = 1.5
getsupportlength(::BSpline{3}) = 2.0
getsupportlength(::BSpline{4}) = 2.5

@pure function getnnodes(bspline::BSpline, ::Val{dim})::Int where {dim}
    (2*getsupportlength(bspline))^dim
end


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
    @. $V(0.5,-1.0,0.5)*ξ^2 + $V(-1.5,0.0,1.5)*ξ + $V(1.125, 0.75, 1.125)
end

function Base.values(::BSpline{3}, x::T) where {T <: Real}
    V = Vec{4, T}
    x′ = fract(x)
    ξ = x′ .- V(-1, 0, 1, 2)
    ξ² = ξ .* ξ
    ξ³ = ξ² .* ξ
    @. $V(-1/6,0.5,-0.5,1/6)*ξ³ + $V(1,-1,-1,1)*ξ² + $V(-2,0,0,2)*ξ + $V(4/3,2/3,2/3,4/3)
end

@generated function Base.values(bspline::BSpline, x::Vec{dim}) where {dim}
    exps = [:(values(bspline, x[$i])) for i in 1:dim]
    quote
        @_inline_meta
        Tuple(otimes($(exps...)))
    end
end

# `x` must be normalized by `dx`
function values_gradients(::BSpline{1}, x::T) where {T <: Real}
    V = Vec{2, T}
    ξ = fract(x)
    V(1-ξ, ξ), V(-1, 1)
end

function values_gradients(::BSpline{2}, x::T) where {T <: Real}
    V = Vec{3, T}
    x′ = fract(x - T(0.5))
    ξ = x′ .- V(-0.5, 0.5, 1.5)
    vals = @. $V(0.5,-1.0,0.5)*ξ^2 + $V(-1.5,0.0,1.5)*ξ + $V(1.125,0.75,1.125)
    grads = @. $V(1.0,-2.0,1.0)*ξ + $V(-1.5,0.0,1.5)
    vals, grads
end

function values_gradients(::BSpline{3}, x::T) where {T <: Real}
    V = Vec{4, T}
    x′ = fract(x)
    ξ = x′ .- V(-1, 0, 1, 2)
    ξ² = ξ .* ξ
    ξ³ = ξ² .* ξ
    vals = @. $V(-1/6,0.5,-0.5,1/6)*ξ³ + $V(1,-1,-1,1)*ξ² + $V(-2,0,0,2)*ξ + $V(4/3,2/3,2/3,4/3)
    grads = @. $V(-0.5,1.5,-1.5,0.5)*ξ² + $V(2,-2,-2,2)*ξ + $V(-2,0,0,2)
    vals, grads
end

@generated function values_gradients(bspline::BSpline, x::Vec{dim}) where {dim}
    exps = [:(values_gradients(bspline, x[$i])) for i in 1:dim]
    derivs = map(1:dim) do i
        x = [d == i ? :(grads[$d]) : :(vals[$d]) for d in 1:dim]
        :(Tuple(otimes($(x...))))
    end
    quote
        @_inline_meta
        vals_grads = tuple($(exps...))
        vals = getindex.(vals_grads, 1)
        grads = getindex.(vals_grads, 2)
        Tuple(otimes(vals...)), Vec{dim}.($(derivs...))
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

function value(::BSpline{3}, ξ::Real)
    ξ = abs(ξ)
    ξ < 1 ? (3ξ^3 - 6ξ^2 + 4) / 6 :
    ξ < 2 ? (2 - ξ)^3 / 6         : zero(ξ)
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


struct BSplineValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    xp::Vec{dim, T}
end

mutable struct BSplineValues{order, dim, T, L} <: MPValues{dim, T, BSplineValue{dim, T}}
    F::BSpline{order}
    N::MVector{L, T}
    ∇N::MVector{L, Vec{dim, T}}
    gridindices::MVector{L, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

function BSplineValues{order, dim, T, L}() where {order, dim, T, L}
    N = MVector{L, T}(undef)
    ∇N = MVector{L, Vec{dim, T}}(undef)
    gridindices = MVector{L, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    BSplineValues(BSpline{order}(), N, ∇N, gridindices, xp, 0)
end

function MPValues{dim, T}(F::BSpline{order}) where {order, dim, T}
    L = getnnodes(F, Val(dim))
    BSplineValues{order, dim, T, L}()
end

function update!(mpvalues::BSplineValues{<: Any, dim}, grid::Grid{dim}, xp::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = mpvalues.F
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    mpvalues.xp = xp
    dx⁻¹ = gridsteps_inv(grid)
    update_active_gridindices!(mpvalues, neighboring_nodes(grid, xp, getsupportlength(F)), spat)
    @inbounds @simd for i in 1:length(mpvalues)
        I = gridindices(mpvalues, i)
        xi = grid[I]
        mpvalues.∇N[i], mpvalues.N[i] = gradient(xp, :all) do xp
            @_inline_meta
            ξ = (xp - xi) .* dx⁻¹
            value(F, ξ, node_position(grid, I))
        end
    end
    mpvalues
end

@inline function Base.getindex(mpvalues::BSplineValues, i::Int)
    @_propagate_inbounds_meta
    BSplineValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.xp)
end
