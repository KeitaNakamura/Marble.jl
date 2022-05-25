struct SpPattern{dim} <: AbstractArray{Bool, dim}
    indices::Array{Int, dim}
    mask::Array{Bool, dim}
end

SpPattern(dims::Tuple{Vararg{Int}}) = SpPattern(fill(-1, dims), fill(false, dims))
SpPattern(dims::Int...) = SpPattern(dims)

Base.size(spat::SpPattern) = size(spat.indices)
Base.IndexStyle(::Type{<: SpPattern}) = IndexLinear()

@inline Base.getindex(spat::SpPattern, i::Int) = (@_propagate_inbounds_meta; spat.mask[i])
@inline Base.setindex!(spat::SpPattern, v, i::Int) = (@_propagate_inbounds_meta; spat.mask[i] = convert(Bool, v))

Base.fill!(spat::SpPattern, v) = (fill!(spat.mask, v); spat)

function reinit!(spat::SpPattern)
    count = 0
    @inbounds for i in eachindex(spat)
        spat.indices[i] = (spat[i] ? count += 1 : -1)
    end
    count
end

Base.copy(spat::SpPattern) = SpPattern(copy(spat.indices), copy(spat.mask))

Broadcast.BroadcastStyle(::Type{<: SpPattern}) = ArrayStyle{SpPattern}()
Base.similar(bc::Broadcasted{ArrayStyle{SpPattern}}, ::Type{Bool}) = SpPattern(size(bc))


struct SpArray{T, dim, V <: AbstractVector{T}} <: AbstractArray{T, dim}
    data::V
    spat::SpPattern{dim}
end

function SpArray{T}(dims::Tuple{Vararg{Int}}) where {T}
    data = Vector{T}(undef, prod(dims))
    spat = SpPattern(dims)
    SpArray(data, spat)
end
SpArray{T}(dims::Int...) where {T} = SpArray{T}(dims)

Base.IndexStyle(::Type{<: SpArray}) = IndexLinear()
Base.size(x::SpArray) = size(x.spat)

Base.propertynames(x::SpArray{<: Any, <: Any, <: StructVector}) = (:data, :spat, propertynames(x.data)...)
function Base.getproperty(x::SpArray{<: Any, <: Any, <: StructVector}, name::Symbol)
    name == :data && return getfield(x, :data)
    name == :spat && return getfield(x, :spat)
    SpArray(getproperty(getfield(x, :data), name), getfield(x, :spat))
end

@inline function Base.getindex(x::SpArray, i::Int)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    index = spat.indices[i]
    @inbounds index !== -1 ? x.data[index] : zero_recursive(eltype(x))
end
@inline function Base.setindex!(x::SpArray, v, i::Int)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    @inbounds begin
        # spat[i] || throw(UndefRefError()) # cannot use this because `@. A[indices] = A[indices]` doesn't work well yet
        index = spat.indices[i]
        index === -1 && return x
        x.data[index] = v
    end
    x
end

@inline function unsafe_add!(x::SpArray, i, v)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    @inbounds begin
        index = spat.indices[i]
        x.data[index] += v # don't check if `index == -1`
    end
    x
end
@inline function unsafe_add!(x::AbstractArray, i, v)
    @boundscheck checkbounds(x, i)
    @inbounds x[i] += v
    x
end

fillzero!(x::SpArray) = (fillzero!(x.data); x)

function reinit!(x::SpArray)
    n = reinit!(x.spat)
    resize!(x.data, n)
    x
end
reinit!(x::SpArray{Nothing}) = x # for Grid without NodeState type


Broadcast.BroadcastStyle(::Type{<: SpArray}) = ArrayStyle{SpArray}()

__extract_spats(spats::Tuple, x::Any) = spats
__extract_spats(spats::Tuple, x::AbstractArray) = (spats..., nothing)
_extract_spats(spats::Tuple, x::AbstractArray) = __extract_spats(spats, broadcastable(x)) # handle Tensor
_extract_spats(spats::Tuple, x::SpArray) = (spats..., x.spat)
_extract_spats(spats::Tuple, x::Any) = spats
extract_spats(spats::Tuple, args::Tuple{}) = spats
extract_spats(spats::Tuple, args::Tuple) = extract_spats(_extract_spats(spats, args[1]), Base.tail(args))
identical_spat(args...) = (spats = extract_spats((), args); all(x -> x === spats[1], spats))

getdata(x::SpArray) = x.data
getdata(x::Any) = x

getspat(x::SpArray) = x.spat
getspat(x::AbstractArray) = ifelse(broadcastable(x) isa AbstractArray, true, false) # handle Tensor
getspat(x::Any) = false

function Base.similar(bc::Broadcasted{ArrayStyle{SpArray}}, ::Type{ElType}) where {ElType}
    spat = broadcast(|, getspat.(bc.args)...)
    reinit!(SpArray(Vector{ElType}(undef, length(bc)), spat))
end

function Base.copyto!(dest::SpArray, bc::Broadcasted{ArrayStyle{SpArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.flatten(bc)
    if identical_spat(dest, bc′.args...)
        broadcast!(bc′.f, getdata(dest), map(getdata, bc′.args)...)
    else
        copyto!(dest, convert(Broadcasted{Nothing}, bc′))
    end
    dest
end

function Base.copyto!(dest::SpArray, bc::Broadcasted{ThreadedStyle})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.flatten(bc.args[1])
    if identical_spat(dest, bc′.args...)
        _copyto!(getdata(dest), broadcasted(dot_threads, broadcasted(bc′.f, map(getdata, bc′.args)...)))
    else
        _copyto!(dest, broadcasted(dot_threads, bc′))
    end
    dest
end


struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowSpArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::A
end
Base.size(x::ShowSpArray) = size(x.parent)
Base.axes(x::ShowSpArray) = axes(x.parent)
@inline function Base.getindex(x::ShowSpArray, i::Int...)
    @_propagate_inbounds_meta
    p = x.parent
    p.spat[i...] ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::SpArray) = ShowSpArray(x)

Base.summary(io::IO, x::ShowSpArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::SpArray) = show(io, mime, ShowSpArray(x))
Base.show(io::IO, x::SpArray) = show(io, ShowSpArray(x))
