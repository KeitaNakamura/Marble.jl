struct PointToGridOperation{C <: AbstractCollection{2}}
    u_i::C
end

struct PointToGridMatrixOperation{C <: AbstractCollection{2}}
    K_ij::C
end

function ∑ₚ(c::AbstractCollection{2})
    ElType = eltype(c)
    if ElType <: AbstractCollection{0}
        return PointToGridOperation(c)
    elseif ElType <: AbstractCollection{-1}
        return PointToGridMatrixOperation(c)
    end
    throw(ArgumentError("wrong collection in ∑ₚ"))
end

for op in (:+, :-)
    @eval function Base.$op(x::PointToGridOperation, y::PointToGridOperation)
        PointToGridOperation($op(x.u_i, y.u_i))
    end
end

function set!(S::GridState, x::PointToGridOperation)
    nzval = nonzeros(zeros!(S))
    dofinds = S.dofindices
    @inbounds for p in eachindex(dofinds)
        u = view(nzval, dofinds[p])
        u .+= x.u_i[p]
    end
    S
end

for op in (:*, :/)
    @eval function Base.$op(x::PointToGridOperation, y::GridState)
        PointToGridOperation($op(x.u_i, GridCollection(y)))
    end
end


struct GridToPointOperation{C <: AbstractCollection{2}} <: AbstractCollection{2}
    u_p::C
end

function ∑ᵢ(c::AbstractCollection{2})
    GridToPointOperation(lazy(reduce, add, c))
end

Base.length(x::GridToPointOperation) = length(x.u_p)
Base.getindex(x::GridToPointOperation, i::Int) = (@_propagate_inbounds_meta; x.u_p[i])

function set!(ps::PointState, x::GridToPointOperation)
    @inbounds for p in 1:length(ps)
        ps[p] = x.u_p[p]
    end
    ps
end

add(a, b) = a + b
add(a::ScalVec, b::ScalVec) = ScalVec(a.x + b.x, a.∇x + b.∇x)
add(a::VecTensor, b::VecTensor) = VecTensor(a.x + b.x, a.∇x + b.∇x)