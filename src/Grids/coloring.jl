struct DashedUnitRange{T} <: AbstractVector{UnitRange{T}}
    parent::UnitRange{T}
    on::Int
    off::Int
end

Base.parent(dashed::DashedUnitRange) = dashed.parent
Base.size(dashed::DashedUnitRange) = (length(dashedstartinds(dashed)),)

function dashedstartinds(dashed)
    p = parent(dashed)
    firstindex(p):dashed.on+dashed.off:lastindex(p)
end

function Base.getindex(dashed::DashedUnitRange, i::Int)
    startinds = dashedstartinds(dashed)
    @boundscheck checkbounds(startinds, i)
    @inbounds begin
        start = startinds[i]
        p = parent(dashed)
        p[start:min(start+dashed.on-1, lastindex(p))]
    end
end


struct DashedCartesianIndices{dim} <: AbstractArray{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}, dim}
    axes::NTuple{dim, DashedUnitRange{Int}}
end

Base.size(inds::DashedCartesianIndices) = map(length, inds.axes)
@inline function Base.getindex(inds::DashedCartesianIndices{dim}, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(inds, I...)
    @inbounds CartesianIndices(Coordinate(inds.axes)[I...])
end

function coloringcells(grid::AbstractGrid, len::Int)
    ColoredBlocks(size(grid).-1, len)
end


struct ColoredBlocks{dim} <: AbstractVector{DashedCartesianIndices{dim}}
    colors::Vector{DashedCartesianIndices{dim}}
    len::Int
end

Base.size(x::ColoredBlocks) = size(x.colors)
@inline function Base.getindex(x::ColoredBlocks, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.colors[i]
end

function ColoredBlocks(dims::NTuple{dim, Int}, len::Int) where {dim}
    f(I...) = DashedCartesianIndices(DashedUnitRange.(UnitRange.(I, dims), len, len))
    iter = Iterators.ProductIterator(ntuple(d -> (1, len+1), Val(dim)))
    colors = vec([f(I...) for I in iter])
    ColoredBlocks(colors, len)
end
