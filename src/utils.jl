struct Index{dim}
    i::Int
    I::CartesianIndex{dim}
end
@inline Index(grid::AbstractArray, i) = (@_propagate_inbounds_meta; Index(size(grid), i))
@inline Index(dims::Dims, i::Int) = (@_propagate_inbounds_meta; Index(i, CartesianIndices(dims)[i]))
@inline Index(dims::Dims, I::CartesianIndex) = (@_propagate_inbounds_meta; Index(LinearIndices(dims)[I], I))
@inline Index(dims::Dims, I::Dims) = (@_propagate_inbounds_meta; Index(LinearIndices(dims)[I...], CartesianIndex(I)))
@inline Base.checkindex(::Type{Bool}, inds::AbstractUnitRange, i::Index) = checkindex(Bool, inds, i.i)
@inline _to_indices(::IndexLinear, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (I[1].i, Base.tail(I)...))
@inline _to_indices(::IndexCartesian, A, inds, I::Tuple{Index, Vararg{Any}}) = to_indices(A, inds, (Tuple(I[1].I)..., Base.tail(I)...))
@inline Base.to_indices(A, inds, I::Tuple{Index, Vararg{Any}}) = _to_indices(IndexStyle(A), A, inds, I)


interpolate(current::AbstractVector, prev::AbstractVector, α::Real) = broadcast((c,p) -> (1-α)*p + α*c, current, prev)
@generated function interpolate(x::V, y::V, α::Real) where {names, T, V <: StructVector{T, <: NamedTuple{names}}}
    exps = [:(interpolate(x.$name, y.$name, α)) for name in names]
    quote
        StructVector{T}(($(exps...),))
    end
end


@generated function Base.popat!(x::StructVector{T, <: NamedTuple{names}}, i::Int) where {T, names}
    exps = [:(popat!(x.$name, i)) for name in names]
    quote
        @boundscheck checkbounds(x, i)
        @inbounds StructArrays.createinstance(T, $(exps...))
    end
end
@generated function Base.deleteat!(x::StructVector{T, <: NamedTuple{names}}, i::Int) where {T, names}
    exps = [:(deleteat!(x.$name, i)) for name in names]
    quote
        @boundscheck checkbounds(x, i)
        @inbounds begin
            $(exps...)
        end
        x
    end
end


function Tensor3D(x::SecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[1,2], x[2,2], z, z, z, z)
end

function Tensor3D(x::SymmetricSecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SymmetricSecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[2,2], z, z)
end

function Tensor2D(x::SecondOrderTensor{3,T}) where {T}
    @inbounds SecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,1], x[2,2])
end

function Tensor2D(x::SymmetricSecondOrderTensor{3,T}) where {T}
    @inbounds SymmetricSecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,2])
end

function Tensor2D(x::FourthOrderTensor{3,T}) where {T}
    @inbounds FourthOrderTensor{2,T}((i,j,k,l) -> @inbounds(x[i,j,k,l]))
end

function Tensor2D(x::SymmetricFourthOrderTensor{3,T}) where {T}
    @inbounds SymmetricFourthOrderTensor{2,T}((i,j,k,l) -> @inbounds(x[i,j,k,l]))
end
