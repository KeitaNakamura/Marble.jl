abstract type ShapeFunction end
abstract type ShapeValues{dim, T} <: AbstractVector{T} end

Base.size(x::ShapeValues) = (x.len[],)

"""
    Poingr.ShapeValues(::ShapeFunction)
    Poingr.ShapeValues(::Type{T}, ::ShapeFunction)

Construct object storing value of `ShapeFunction`.

# Examples
```jldoctest
julia> sv = Poingr.ShapeValues(QuadraticBSpline{2}());

julia> update!(sv, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(sv.N)
1.0

julia> sum(sv.∇N)
2-element Vec{2, Float64}:
 5.551115123125783e-17
 5.551115123125783e-17
```
"""
ShapeValues{dim}(F::ShapeFunction) where {dim} = ShapeValues{dim, Float64}(F)

"""
    update!(::ShapeValues, grid::Grid, x::Vec)
    update!(::ShapeValues, grid::Grid, indices::AbstractArray, x::Vec)

Update value of shape function at `x` with each `grid` node.

# Examples
```jldoctest
julia> sv = Poingr.ShapeValues(QuadraticBSpline{2}());

julia> update!(sv, Grid(0:3, 0:3), Vec(1, 1));

julia> sum(sv.N)
1.0

julia> update!(sv, Grid(0:3, 0:3), Vec(1, 1), CartesianIndices((1:2, 1:2)));

julia> sum(sv.N)
0.765625
```
"""
update!

update!(it::ShapeValues, grid, x::Vec) = update!(it, grid, x, trues(size(grid)))

function update_gridindices!(it::ShapeValues, grid, x::Vec{dim}, spat::BitArray{dim}) where {dim}
    inds = neighboring_nodes(grid, x, support_length(it.F))
    count = 0
    @inbounds for I in inds
        i = LinearIndices(grid)[I]
        if spat[i]
            @assert count != length(it.inds)
            it.inds[count+=1] = Index(i, I)
        end
    end
    it.len[] = count
    it
end
