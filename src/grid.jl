"""
    Grid([::Type{NodeState}], [::Interpolation], axes::AbstractVector...)

Construct `Grid` by `axes`.

# Examples
```jldoctest
julia> Grid(range(0, 3, step = 1.0), range(1, 4, step = 1.0))
4×4 Grid{2, Float64, Nothing, Nothing, Metale.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
 [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]
 [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]
 [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]
 [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]
```
"""
struct Grid{dim, T, F <: Union{Nothing, Interpolation}, Node, State <: SpArray{Node, dim}, CS <: CoordinateSystem} <: AbstractArray{Vec{dim, T}, dim}
    interpolation::F
    axes::NTuple{dim, Vector{T}}
    gridsteps::NTuple{dim, T}
    gridsteps_inv::NTuple{dim, T}
    state::State
    coordinate_system::CS
end

Base.size(x::Grid) = map(length, gridaxes(x))
gridsteps(x::Grid) = x.gridsteps
gridsteps(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps(x)[i])
gridsteps_inv(x::Grid) = x.gridsteps_inv
gridsteps_inv(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridsteps_inv(x)[i])
gridaxes(x::Grid) = x.axes
gridaxes(x::Grid, i::Int) = (@_propagate_inbounds_meta; gridaxes(x)[i])
gridorigin(x::Grid) = Vec(map(first, gridaxes(x)))

check_interpolation(::Grid{<: Any, <: Any, Nothing}) = throw(ArgumentError("`Grid` must include the information of interpolation, see help `?Grid` for more details."))
check_interpolation(::Grid{<: Any, <: Any, <: Interpolation}) = nothing

function Grid(::Type{Node}, interp, axes::NTuple{dim, AbstractVector}; coordinate_system = nothing) where {Node, dim}
    state = SpArray(StructVector{Node}(undef, 0), SpPattern(map(length, axes)))
    dx = map(step, axes)
    dx⁻¹ = inv.(dx)

    T = promote_type(eltype.(axes)...)
    T = ifelse(T <: Integer, Float64, T) # use Float64 by default

    Grid(
        interp,
        map(Array{T}, axes),
        T.(dx),
        T.(dx⁻¹),
        state,
        get_coordinate_system(coordinate_system, Val(dim)),
    )
end
function Grid(interp::Interpolation, axes::NTuple{dim, AbstractVector}; kwargs...) where {dim}
    T = promote_type(map(eltype, axes)...)
    Node = default_nodestate_type(interp, Val(dim), Val(T))
    Grid(Node, interp, axes; kwargs...)
end
Grid(axes::Tuple{Vararg{AbstractVector}}; kwargs...) = Grid(Nothing, nothing, axes; kwargs...)

# `interp` must be given if Node is given
Grid(Node::Type, interp, axes::AbstractVector...; kwargs...) = Grid(Node, interp, axes; kwargs...)
Grid(interp::Interpolation, axes::AbstractVector...; kwargs...) = Grid(interp, axes; kwargs...)
Grid(axes::AbstractVector...; kwargs...) = Grid(Nothing, nothing, axes; kwargs...)

@inline function Base.getindex(grid::Grid{dim}, i::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(grid, i...)
    @inbounds Vec(map(getindex, grid.axes, i))
end

"""
    Metale.neighboring_nodes(grid, x::Vec, h)

Return `CartesianIndices` storing neighboring node indices around `x`.
`h` is a range for searching and its unit is `gridsteps` `dx`.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0)
6-element Grid{1, Float64, Nothing, Nothing, Metale.SpArray{Nothing, 1, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, Metale.OneDimensional}:
 [0.0]
 [1.0]
 [2.0]
 [3.0]
 [4.0]
 [5.0]

julia> Metale.neighboring_nodes(grid, Vec(1.5), 1)
2-element CartesianIndices{1, Tuple{UnitRange{Int64}}}:
 CartesianIndex(2,)
 CartesianIndex(3,)

julia> Metale.neighboring_nodes(grid, Vec(1.5), 2)
4-element CartesianIndices{1, Tuple{UnitRange{Int64}}}:
 CartesianIndex(1,)
 CartesianIndex(2,)
 CartesianIndex(3,)
 CartesianIndex(4,)
```
"""
@inline function neighboring_nodes(grid::Grid{dim}, x::Vec{dim}, h) where {dim}
    dx⁻¹ = gridsteps_inv(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) .* dx⁻¹)
    T = eltype(ξ)
    all(@. zero(T) ≤ ξ ≤ T($size(grid)-1)) || return CartesianIndices(nfill(1:0, Val(dim)))
    # To handle zero division in nodal calculations such as fᵢ/mᵢ, we use a bit small `h`.
    # This means `neighboring_nodes` doesn't include bounds of range.
    _neighboring_nodes(size(grid), ξ, @. T(h) - sqrt(eps(T)))
end
@inline function _neighboring_nodes(dims::Dims, ξ, h)
    imin = Tuple(@. max(unsafe_trunc(Int,  ceil(ξ - h)) + 1, 1))
    imax = Tuple(@. min(unsafe_trunc(Int, floor(ξ + h)) + 1, dims))
    CartesianIndices(@. UnitRange(imin, imax))
end


"""
    Metale.neighboring_cells(grid, x::Vec, h::Int)
    Metale.neighboring_cells(grid, cellindex::CartesianIndex, h::Int)

Return `CartesianIndices` storing neighboring cell indices around `x`.
`h` is number of outer cells around cell where `x` locates.
In 1D, for example, the searching range becomes `x ± h*dx`.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0, 0.0:1.0:5.0)
6×6 Grid{2, Float64, Nothing, Nothing, Metale.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> x = Vec(1.5, 1.5);

julia> Metale.neighboring_cells(grid, x, 1)
3×3 CartesianIndices{2, Tuple{UnitRange{Int64}, UnitRange{Int64}}}:
 CartesianIndex(1, 1)  CartesianIndex(1, 2)  CartesianIndex(1, 3)
 CartesianIndex(2, 1)  CartesianIndex(2, 2)  CartesianIndex(2, 3)
 CartesianIndex(3, 1)  CartesianIndex(3, 2)  CartesianIndex(3, 3)

julia> Metale.neighboring_cells(grid, Metale.whichcell(grid, x), 1) == ans
true
```
"""
function neighboring_cells(grid::Grid{dim}, cellindex::CartesianIndex{dim}, h::Int) where {dim}
    inds = CartesianIndices(size(grid) .- 1)
    @boundscheck checkbounds(inds, cellindex)
    u = oneunit(cellindex)
    inds ∩ (cellindex-h*u:cellindex+h*u)
end

@inline function neighboring_cells(grid::Grid, x::Vec, h::Int)
    neighboring_cells(grid, whichcell(grid, x), h)
end

function neighboring_blocks(grid::Grid{dim}, blockindex::CartesianIndex{dim}, h::Int) where {dim}
    inds = CartesianIndices(blocksize(grid))
    @boundscheck checkbounds(inds, blockindex)
    u = oneunit(blockindex)
    inds ∩ (blockindex-h*u:blockindex+h*u)
end

@inline function neighboring_blocks(grid::Grid, x::Vec, h::Int)
    neighboring_blocks(grid, whichblock(grid, x), h)
end

"""
    Metale.whichcell(grid, x::Vec)

Return cell index where `x` locates.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:5.0, 0.0:1.0:5.0)
6×6 Grid{2, Float64, Nothing, Nothing, Metale.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
 [0.0, 0.0]  [0.0, 1.0]  [0.0, 2.0]  [0.0, 3.0]  [0.0, 4.0]  [0.0, 5.0]
 [1.0, 0.0]  [1.0, 1.0]  [1.0, 2.0]  [1.0, 3.0]  [1.0, 4.0]  [1.0, 5.0]
 [2.0, 0.0]  [2.0, 1.0]  [2.0, 2.0]  [2.0, 3.0]  [2.0, 4.0]  [2.0, 5.0]
 [3.0, 0.0]  [3.0, 1.0]  [3.0, 2.0]  [3.0, 3.0]  [3.0, 4.0]  [3.0, 5.0]
 [4.0, 0.0]  [4.0, 1.0]  [4.0, 2.0]  [4.0, 3.0]  [4.0, 4.0]  [4.0, 5.0]
 [5.0, 0.0]  [5.0, 1.0]  [5.0, 2.0]  [5.0, 3.0]  [5.0, 4.0]  [5.0, 5.0]

julia> Metale.whichcell(grid, Vec(1.5, 1.5))
CartesianIndex(2, 2)
```
"""
@inline function whichcell(grid::Grid{dim}, x::Vec{dim}) where {dim}
    dx⁻¹ = gridsteps_inv(grid)
    xmin = gridorigin(grid)
    ξ = Tuple((x - xmin) .* dx⁻¹)
    all(@. 0 ≤ ξ ≤ $size(grid)-1) || return nothing
    CartesianIndex(@. unsafe_trunc(Int, floor(ξ)) + 1)
end

"""
    Metale.whichblock(grid, x::Vec)

Return block index where `x` locates.
The unit block size is `2^$BLOCK_UNIT` cells.

# Examples
```jldoctest
julia> grid = Grid(0.0:1.0:10.0, 0.0:1.0:10.0)
11×11 Grid{2, Float64, Nothing, Nothing, Metale.SpArray{Nothing, 2, StructArrays.StructVector{Nothing, NamedTuple{(), Tuple{}}, Int64}}, PlaneStrain}:
 [0.0, 0.0]   [0.0, 1.0]   [0.0, 2.0]   …  [0.0, 9.0]   [0.0, 10.0]
 [1.0, 0.0]   [1.0, 1.0]   [1.0, 2.0]      [1.0, 9.0]   [1.0, 10.0]
 [2.0, 0.0]   [2.0, 1.0]   [2.0, 2.0]      [2.0, 9.0]   [2.0, 10.0]
 [3.0, 0.0]   [3.0, 1.0]   [3.0, 2.0]      [3.0, 9.0]   [3.0, 10.0]
 [4.0, 0.0]   [4.0, 1.0]   [4.0, 2.0]      [4.0, 9.0]   [4.0, 10.0]
 [5.0, 0.0]   [5.0, 1.0]   [5.0, 2.0]   …  [5.0, 9.0]   [5.0, 10.0]
 [6.0, 0.0]   [6.0, 1.0]   [6.0, 2.0]      [6.0, 9.0]   [6.0, 10.0]
 [7.0, 0.0]   [7.0, 1.0]   [7.0, 2.0]      [7.0, 9.0]   [7.0, 10.0]
 [8.0, 0.0]   [8.0, 1.0]   [8.0, 2.0]      [8.0, 9.0]   [8.0, 10.0]
 [9.0, 0.0]   [9.0, 1.0]   [9.0, 2.0]      [9.0, 9.0]   [9.0, 10.0]
 [10.0, 0.0]  [10.0, 1.0]  [10.0, 2.0]  …  [10.0, 9.0]  [10.0, 10.0]

julia> Metale.whichblock(grid, Vec(8.5, 1.5))
CartesianIndex(2, 1)
```
"""
@inline function whichblock(grid::Grid, x::Vec)
    I = whichcell(grid, x)
    I === nothing && return nothing
    CartesianIndex(@. ($Tuple(I)-1) >> BLOCK_UNIT + 1)
end

blocksize(grid::Grid) = (ncells = size(grid) .- 1; @. (ncells - 1) >> BLOCK_UNIT + 1)

struct BlockStepIndices{N} <: AbstractArray{CartesianIndex{N}, N}
    inds::NTuple{N, StepRange{Int, Int}}
end
Base.size(x::BlockStepIndices) = map(length, x.inds)
Base.getindex(x::BlockStepIndices{N}, i::Vararg{Int, N}) where {N} = (@_propagate_inbounds_meta; CartesianIndex(map(getindex, x.inds, i)))

function threadsafe_blocks(gridsize::NTuple{dim, Int}) where {dim}
    ncells = gridsize .- 1
    starts = SArray{NTuple{dim, 2}}(Iterators.product(nfill((1,2), Val(dim))...)...)
    nblocks = @. (ncells - 1) >> BLOCK_UNIT + 1
    vec(map(st -> BlockStepIndices(StepRange.(st, 2, nblocks)), starts))
end


struct Boundaries{dim} <: AbstractArray{Tuple{CartesianIndex{dim}, Vec{dim, Int}}, dim}
    inds::CartesianIndices{dim}
    n::Vec{dim, Int}
end
Base.IndexStyle(::Type{<: Boundaries}) = IndexCartesian()
Base.size(x::Boundaries) = size(x.inds)
Base.getindex(x::Boundaries{dim}, I::Vararg{Int, dim}) where {dim} = (@_propagate_inbounds_meta; (x.inds[I...], x.n))

function _boundaries(grid::AbstractArray{<: Any, dim}, which::String) where {dim}
    if     which[2] == 'x'; axis = 1
    elseif which[2] == 'y'; axis = 2
    elseif which[2] == 'z'; axis = 3
    else error("invalid bound name")
    end

    if     which[1] == '-'; index = firstindex(grid, axis); dir =  1
    elseif which[1] == '+'; index =  lastindex(grid, axis); dir = -1
    else error("invalid bound name")
    end

    inds = CartesianIndices(ntuple(d -> d==axis ? (index:index) : axes(grid, d), Val(dim)))
    n = Vec(ntuple(d -> ifelse(d==axis, dir, 0), Val(dim)))

    Boundaries(inds, n)
end
function boundaries(grid::AbstractArray, which::Vararg{String, N}) where {N}
    Iterators.flatten(ntuple(i -> _boundaries(grid, which[i]), Val(N)))
end
