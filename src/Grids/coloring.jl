struct DashedCartesianIndices{dim} <: AbstractArray{CartesianIndices{dim, NTuple{dim, UnitRange{Int}}}, dim}
    axes::NTuple{dim, DashedUnitRange{Int}}
end

Base.size(inds::DashedCartesianIndices) = map(length, inds.axes)
function Base.getindex(inds::DashedCartesianIndices{dim}, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(inds, I...)
    CartesianIndices(Coordinate(inds.axes)[I...])
end

function coloringcells(grid::AbstractGrid, len::Int)
    f(i, j) = DashedCartesianIndices(DashedUnitRange.(UnitRange.((i, j), size(grid).-1), len, len))
    [f(i_start, j_start) for j_start in (1, len+1) for i_start in (1, len+1)]
end
