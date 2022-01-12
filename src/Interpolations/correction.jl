struct KernelCorrection{Weight <: Kernel} <: Interpolation
    weight::Weight
end

weight_function(c::KernelCorrection) = c.weight
support_length(c::KernelCorrection, args...) = support_length(weight_function(c), args...)

mutable struct KernelCorrectionValues{Weight, dim, T, nnodes} <: MPValues{dim, T}
    F::KernelCorrection{Weight}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    gridindices::MVector{nnodes, Index{dim}}
    x::Vec{dim, T}
    len::Int
end

function KernelCorrectionValues{Weight, dim, T, nnodes}() where {Weight, dim, T, nnodes}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    gridindices = MVector{nnodes, Index{dim}}(undef)
    x = zero(Vec{dim, T})
    KernelCorrectionValues(KernelCorrection(Weight()), N, ∇N, gridindices, x, 0)
end

function MPValues{dim, T}(c::KernelCorrection{Weight}) where {dim, T, Weight}
    L = nnodes(Weight(), Val(dim))
    KernelCorrectionValues{Weight, dim, T, L}()
end

function update!(mpvalues::KernelCorrectionValues{<: Any, dim, T}, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim, T}
    F = weight_function(mpvalues.F)
    mpvalues.N .= elzero(mpvalues.N)
    mpvalues.∇N .= elzero(mpvalues.∇N)
    mpvalues.x = x

    dx⁻¹ = gridsteps_inv(grid)
    iscompleted = update_gridindices!(mpvalues, neighboring_nodes(grid, x, support_length(F)), spat)
    if iscompleted
        wᵢ, ∇wᵢ = values_gradients(F, x .* dx⁻¹)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            mpvalues.N[i] = wᵢ[i]
            mpvalues.∇N[i] = ∇wᵢ[i] .* dx⁻¹
        end
    else
        A = zero(Mat{dim, dim, T})
        β = zero(Vec{dim, T})
        A′ = zero(Mat{dim, dim, T})
        β′ = zero(Vec{dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            ξ = (x - xᵢ) .* dx⁻¹
            ∇w, w = gradient(ξ -> value(F, ξ), ξ, :all)
            ∇w = ∇w .* dx⁻¹
            A += w * (xᵢ - x) ⊗ (xᵢ - x)
            β += w * (xᵢ - x)
            A′ += ∇w ⊗ (xᵢ - x)
            β′ += ∇w
            mpvalues.N[i] = w
            mpvalues.∇N[i] = ∇w
        end
        A⁻¹ = inv(A)
        β = -(A⁻¹ ⋅ β)
        A′⁻¹ = inv(A′)
        β′ = -(A′⁻¹ ⋅ β′)
        α = zero(T)
        α′ = zero(Mat{dim, dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            w = mpvalues.N[i]
            ∇w = mpvalues.∇N[i]
            α += w * (1 + β ⋅ (xᵢ - x))
            α′ += (xᵢ ⊗ ∇w) * (1 + β′ ⋅ (xᵢ - x))
        end
        α = inv(α)
        α′ = inv(α′)
        @inbounds @simd for i in 1:length(mpvalues)
            I = mpvalues.gridindices[i]
            xᵢ = grid[I]
            w = mpvalues.N[i]
            ∇w = mpvalues.∇N[i]
            mpvalues.N[i] = w * α * (1 + β ⋅ (xᵢ - x))
            mpvalues.∇N[i] = ∇w ⋅ α′ * (1 + β′ ⋅ (xᵢ - x))
        end
    end

    mpvalues
end

struct KernelCorrectionValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    I::Index{dim}
    x::Vec{dim, T}
end

@inline function Base.getindex(mpvalues::KernelCorrectionValues, i::Int)
    @_propagate_inbounds_meta
    KernelCorrectionValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.gridindices[i], mpvalues.x)
end
