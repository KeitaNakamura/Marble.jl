abstract type Interpolation{dim, T} <: AbstractVector{ScalarVector{T, dim}} end

"""
    construct(::ShapeFunction)
    construct(::Type{T}, ::ShapeFunction)

Construct interpolation object for `ShapeFunction`.

# Examples
```jldoctest
julia> it = construct(QuadraticBSpline(dim = 2));

julia> reinit!(it, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(it)
1.0

julia> sum(∇, it)
2-element Tensor{1,2,Float64,2}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
construct(F::ShapeFunction{dim}) where {dim} = construct(Float64, F)::Interpolation{dim}

"""
    reinit!(::Interpolation, grid::AbstractGrid, x::Vec)
    reinit!(::Interpolation, grid::AbstractGrid, indices::AbstractArray, x::Vec)

Reinitialize interpolation object at `x` with each `grid` node.

# Examples
```jldoctest
julia> it = construct(QuadraticBSpline(dim = 2));

julia> reinit!(it, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(it)
1.0

julia> reinit!(it, Grid(0:3, 0:3), CartesianIndices((1:2, 1:2)), Vec(1, 1));

julia> sum(it)
0.765625
```
"""
reinit!

function reinit!(it::Interpolation, grid::AbstractGrid, x::Vec)
    reinit!(it, grid, eachindex(grid), x)
end

Base.size(it::Interpolation) = (length(it.N),)

function Base.getindex(it::Interpolation, i::Int)
    @_propagate_inbounds_meta
    ScalarVector(it.N[i], it.dN[i])
end


struct VectorValue{dim, T, IT <: Interpolation{dim, T}, M} <: AbstractVector{VectorTensor{dim, T, M}}
    Ni::IT
end

function VectorValue(Ni::Interpolation{dim, T}) where {dim, T}
    VectorValue{dim, T, typeof(Ni), dim^2}(Ni)
end

Base.parent(it::VectorValue) = it.Ni
Base.length(it::VectorValue{dim}) where {dim} = dim * length(it.Ni)

function Base.getindex(it::VectorValue{dim}, j::Int) where {dim}
    @boundscheck checkbounds(it, j)
    i, d = divrem(j - 1, dim) .+ 1
    @inbounds begin
        ei = eᵢ(Vec{dim, Int}, d)
        N = it.Ni[i]
    end
    VectorTensor(ei * N, ei ⊗ ∇(N))
end
