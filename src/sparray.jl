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

# handle `StructVector`
Base.propertynames(x::SpArray{<: Any, <: Any, <: StructVector}) = (:data, :spat, propertynames(x.data)...)
function Base.getproperty(x::SpArray{<: Any, <: Any, <: StructVector}, name::Symbol)
    name == :data && return getfield(x, :data)
    name == :spat && return getfield(x, :spat)
    SpArray(getproperty(getfield(x, :data), name), getfield(x, :spat))
end

# return zero if the index is not active
@inline function Base.getindex(x::SpArray, i::Int)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    index = spat.indices[i]
    @inbounds index !== -1 ? x.data[index] : zero_recursive(eltype(x))
end

# do nothing if the index is not active (don't throw error!!)
@inline function Base.setindex!(x::SpArray, v, i::Int)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    @inbounds begin
        index = spat.indices[i]
        index === -1 && return x
        x.data[index] = v
    end
    x
end

# faster than using `setindex!(dest, dest + getindex(src, i))` when using `SpArray`
# since the index is checked only once
@inline function add!(x::SpArray, v, i)
    @boundscheck checkbounds(x, i)
    spat = x.spat
    @inbounds begin
        index = spat.indices[i]
        index === -1 && return x
        x.data[index] += v
    end
    x
end
@inline function add!(x::AbstractArray, v, i)
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

@generated function extract_sparsity_patterns(args::Vararg{Any, N}) where {N}
    exps = []
    for i in 1:N
        if args[i] <: SpArray
            push!(exps, :(args[$i].spat))
        elseif (args[i] <: AbstractArray) && !(args[i] <: AbstractTensor)
            push!(exps, :nothing)
        end
    end
    quote
        tuple($(exps...))
    end
end
identical(x, ys...) = all(y -> y === x, ys)

_getspat(x::SpArray) = x.spat
_getspat(x::Any) = false
function Base.similar(bc::Broadcasted{ArrayStyle{SpArray}}, ::Type{ElType}) where {ElType}
    spat = broadcast(&, map(_getspat, bc.args)...)
    reinit!(SpArray(Vector{ElType}(undef, length(bc)), spat))
end

_getdata(x::SpArray) = x.data
_getdata(x::Any) = x
function Base.copyto!(dest::SpArray, bc::Broadcasted{ArrayStyle{SpArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.flatten(bc)
    if identical(extract_sparsity_patterns(dest, bc′.args...)...)
        broadcast!(bc′.f, _getdata(dest), map(_getdata, bc′.args)...)
    else
        copyto!(dest, convert(Broadcasted{Nothing}, bc′))
    end
    dest
end

function Base.copyto!(dest::SpArray, bc::Broadcasted{ThreadedStyle})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bc′ = Broadcast.flatten(bc.args[1])
    if identical(extract_sparsity_patterns(dest, bc′.args...)...)
        _copyto!(_getdata(dest), broadcasted(dot_threads, broadcasted(bc′.f, map(_getdata, bc′.args)...)))
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
