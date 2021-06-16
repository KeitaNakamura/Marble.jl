struct GridState{dim, T, V, C} <: AbstractArray{T, dim}
    sp::SparseArray{T, dim, V}
    dofindices::PointToDofIndices
    pointsincell::Array{Vector{Int}, dim}
    colors::Vector{C}
end

function gridstate(::Type{T}, dofmap::DofMap, dofindices::PointToDofIndices, pointsincell, colors) where {T}
    GridState(SparseArray{T}(dofmap), dofindices, pointsincell, colors)
end

Base.size(A::GridState) = size(A.sp)
nonzeros(A::GridState) = nonzeros(A.sp)
nzindices(A::GridState) = nzindices(A.sp)
dofindices(A::GridState) = A.dofindices
nnz(A::GridState) = nnz(A.sp)

totalnorm(A::GridState) = norm(flatview(nonzeros(A.sp)))
Base.zero(A::GridState) = GridState(zero(A.sp), dofindices(A))

@inline function Base.getindex(A::GridState{dim, T}, I::Vararg{Int, dim}) where {dim, T}
    @boundscheck checkbounds(A, I...)
    @inbounds A.sp[I...]
end

@inline function Base.setindex!(A::GridState{dim}, v, I::Vararg{Int, dim}) where {dim}
    @boundscheck checkbounds(A, I...)
    @inbounds A.sp[I...] = v
    A
end

zeros!(v::AbstractVector{T}, n) where {T} = (resize!(v, n); fill!(v, zero(T)); v)
zeros!(v) = (fill!(v, zero(eltype(v))); v)
zeros!(A::GridState) = (zeros!(nonzeros(A), nnz(A)); A)
Base.resize!(A::GridState) = (resize!(nonzeros(A), nnz(A)); A)
