struct ScalarVector{T <: Real, dim} <: Real
    x::T
    ∇x::Vec{dim, T}
end

∇(a::ScalarVector) = a.∇x

Base.promote(a::ScalarVector, x) = promote(a.x, x)
Base.promote(x, a::ScalarVector) = promote(x, a.x)

Base.promote_type(::Type{ScalarVector{T, dim}}, ::Type{U}) where {T, dim, U} = promote_type(T, U)
Base.promote_type(::Type{U}, ::Type{ScalarVector{T, dim}}) where {T, dim, U} = promote_type(U, T)

Base.convert(::Type{T}, a::ScalarVector) where {T <: Real} = convert(T, a.x)
Base.convert(::Type{ScalarVector{T, dim}}, a::ScalarVector) where {T, dim} = ScalarVector{T, dim}(a.x, a.∇x)

Base.zero(::Type{ScalarVector{T, dim}}) where {T, dim} = ScalarVector(zero(T), zero(Vec{dim, T}))
Base.zero(::ScalarVector{T, dim}) where {T, dim} = zero(ScalarVector{T, dim})

# scalar vs scalar
for op in (:+, :-, :/, :*)
    @eval Base.$op(a::ScalarVector, b::ScalarVector) = $op(a.x, b.x)
end

Base.show(io::IO, a::ScalarVector) = show(io, a.x)


struct VectorTensor{dim, T, M} <: AbstractVector{T}
    x::Vec{dim, T}
    ∇x::Tensor{2, dim, T, M}
end

VectorTensor{dim, T}(x::Vec{dim, <: Any}, ∇x::Tensor{2, dim, <: Any, M}) where {dim, T, M} =
    VectorTensor{dim, T, M}(x, ∇x)

∇(v::VectorTensor) = v.∇x

Base.size(v::VectorTensor) = size(v.x)
Base.getindex(v::VectorTensor, i::Int) = (@_propagate_inbounds_meta; v.x[i])

Base.convert(::Type{T}, a::VectorTensor) where {T <: Vec} = convert(T, a.x)
Base.convert(::Type{VectorTensor{dim, T}}, a::VectorTensor) where {dim, T} = VectorTensor{dim, T}(a.x, a.∇x)

Base.zero(::Type{VectorTensor{dim, T}}) where {dim, T} = VectorTensor(zero(Vec{dim, T}), zero(Tensor{2, dim, T}))
Base.zero(::Type{VectorTensor{dim, T, M}}) where {dim, T, M} = zero(VectorTensor{dim, T})
Base.zero(::VectorTensor{dim, T}) where {dim, T} = zero(VectorTensor{dim, T})

# vector vs vector
# +, -, ⋅, ⊗, ×
for op in (:+, :-, :⋅, :⊗, :×)
    @eval begin
        Tensors.$op(a::VectorTensor, b::AbstractVector) = $op(a.x, b)
        Tensors.$op(a::AbstractVector, b::VectorTensor) = $op(a, b.x)
        Tensors.$op(a::VectorTensor, b::VectorTensor) = $op(a.x, b.x)
    end
end

# vector vs number
# *, /
for op in (:*, :/)
    @eval Tensors.$op(a::VectorTensor, b::Number) = $op(a.x, b)
    if op != :/
        @eval Tensors.$op(a::Number, b::VectorTensor) = $op(a, b.x)
    end
end

# vector vs matrix
# *, ⋅
for op in (:*, :⋅)
    @eval begin
        Tensors.$op(a::AbstractMatrix, b::VectorTensor) = $op(a, b.x)
        Tensors.$op(a::VectorTensor, b::AbstractMatrix) = $op(a.x, b)
    end
end

for op in (:gradient, :hessian, :divergence, :curl, :laplace)
    @eval Tensors.$op(f, v::VectorTensor, args...) = Tensors.$op(f, v.x, args...)
end

for op in (:norm, )
    @eval Tensors.$op(v::VectorTensor) = $op(v.x)
end


valgrad(x::Real, ∇x::Vec) = ScalarVector(x, ∇x)
valgrad(x::Vec, ∇x::Tensor{2}) = VectorTensor(x, ∇x)
