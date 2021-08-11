struct Mask{dim} <: AbstractArray{Bool, dim}
    indices::Array{Int, dim}
end

Mask(dims::Tuple{Vararg{Int}}) = Mask(fill(-1, dims))
Mask(dims::Int...) = Mask(dims)

Base.size(mask::Mask) = size(mask.indices)
Base.IndexStyle(::Type{<: Mask}) = IndexLinear()

@inline Base.getindex(mask::Mask, i::Int) = (@_propagate_inbounds_meta; mask.indices[i] !== -1)
@inline Base.setindex!(mask::Mask, v, i::Int) = (@_propagate_inbounds_meta; mask.indices[i] = ifelse(convert(Bool, v), 0, -1))

Base.fill!(mask::Mask, v) = (fill!(mask.indices, ifelse(convert(Bool, v), 0, -1)); mask)

function reinit!(mask::Mask)
    count = 0
    for i in eachindex(mask)
        @inbounds mask.indices[i] = (mask[i] ? count += 1 : -1)
    end
    count
end


struct MaskedArray{T, dim, V <: AbstractVector{T}} <: AbstractArray{T, dim}
    data::V
    mask::Mask{dim}
end

Base.IndexStyle(::Type{<: MaskedArray}) = IndexLinear()
Base.size(x::MaskedArray) = size(x.mask)

Base.propertynames(x::MaskedArray) = (:mask, :data, propertynames(x.data)...)
function Base.getproperty(x::MaskedArray, name::Symbol)
    name == :mask && return getfield(x, :mask)
    name == :data && return getfield(x, :data)
    MaskedArray(getproperty(getfield(x, :data), name), getfield(x, :mask))
end

@inline function Base.getindex(x::MaskedArray, i::Int)
    @boundscheck checkbounds(x, i)
    mask = x.mask
    @inbounds begin
        # mask[i] || throw(UndefRefError())
        # x.data[mask.indices[i]]
        mask[i] ? x.data[mask.indices[i]] : zero(eltype(x))
    end
end
@inline function Base.setindex!(x::MaskedArray, v, i::Int)
    @boundscheck checkbounds(x, i)
    mask = x.mask
    @inbounds begin
        # mask[i] || throw(UndefRefError())
        mask[i] || return x
        x.data[mask.indices[i]] = v
    end
    x
end

function reinit!(x::MaskedArray{T}) where {T}
    n = reinit!(x.mask)
    resize!(x.data, n)
    reinit!(x.data)
    x
end


_extract_masks(masks::Tuple, x::Any) = masks
_extract_masks(masks::Tuple, x::MaskedArray) = (masks..., x.mask)
_extract_masks(masks::Tuple, x::Any, y...) = _extract_masks(masks, y...)
_extract_masks(masks::Tuple, x::MaskedArray, y...) = _extract_masks((masks..., x.mask), y...)
extract_masks(args...) = _extract_masks((), args...)
allsame(args) = all(x -> x === args[1], args)
getdata(x::MaskedArray) = x.data
getdata(x::Any) = x
Broadcast.BroadcastStyle(::Type{<: MaskedArray}) = Broadcast.ArrayStyle{MaskedArray}()
function _copyto!(f, dest::MaskedArray, args...)
    @assert allsame(extract_masks(dest, args...))
    broadcast!(f, getdata(dest), map(getdata, args)...)
end
function Base.copyto!(dest::MaskedArray, bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{MaskedArray}})
    bcf = Broadcast.flatten(bc)
    _copyto!(bcf.f, dest, bcf.args...)
    dest
end


struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowMaskedArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::A
end
Base.size(x::ShowMaskedArray) = size(x.parent)
Base.axes(x::ShowMaskedArray) = axes(x.parent)
@inline function Base.getindex(x::ShowMaskedArray, i::Int...)
    @_propagate_inbounds_meta
    p = x.parent
    p.mask[i...] ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::MaskedArray) = ShowMaskedArray(x)

Base.summary(io::IO, x::ShowMaskedArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::MaskedArray) = show(io, mime, ShowMaskedArray(x))
Base.show(io::IO, x::MaskedArray) = show(io, ShowMaskedArray(x))
