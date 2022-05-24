struct KernelCorrection{K <: Kernel} <: Interpolation
end
@pure KernelCorrection(k::Kernel) = KernelCorrection{typeof(k)}()

@pure getkernelfunction(::KernelCorrection{K}) where {K} = K()
getsupportlength(c::KernelCorrection, args...) = getsupportlength(getkernelfunction(c), args...)


struct KernelCorrectionValue{dim, T} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    xp::Vec{dim, T}
end

mutable struct KernelCorrectionValues{K, dim, T, nnodes} <: MPValues{dim, T, KernelCorrectionValue{dim, T}}
    F::KernelCorrection{K}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    gridindices::MVector{nnodes, Index{dim}}
    xp::Vec{dim, T}
    len::Int
end

function KernelCorrectionValues{K, dim, T, nnodes}() where {K, dim, T, nnodes}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    gridindices = MVector{nnodes, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    KernelCorrectionValues(KernelCorrection(K()), N, ∇N, gridindices, xp, 0)
end

function MPValues{dim, T}(c::KernelCorrection{K}) where {dim, T, K}
    L = getnnodes(K(), Val(dim))
    KernelCorrectionValues{K, dim, T, L}()
end

getkernelfunction(c::KernelCorrectionValues) = getkernelfunction(c.F)

function _update!(mpvalues::KernelCorrectionValues{<: Any, dim, T}, grid::Grid{dim}, xp::Vec{dim}, spat::AbstractArray{Bool, dim}, inds, args...) where {dim, T}
    F = getkernelfunction(mpvalues)
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    mpvalues.xp = xp

    allactive = update_active_gridindices!(mpvalues, inds, spat)
    if allactive
        wᵢ, ∇wᵢ = values_gradients(F, grid, xp, args...)
        mpvalues.N .= wᵢ
        mpvalues.∇N .= ∇wᵢ
    else
        A = zero(Mat{dim, dim, T})
        β = zero(Vec{dim, T})
        A′ = zero(Mat{dim, dim, T})
        β′ = zero(Vec{dim, T})
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w, ∇w = value_gradient(F, grid, I, xp, args...)
            A += w * (xi - xp) ⊗ (xi - xp)
            β += w * (xi - xp)
            A′ += ∇w ⊗ (xi - xp)
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
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w = mpvalues.N[i]
            ∇w = mpvalues.∇N[i]
            α += w * (1 + β ⋅ (xi - xp))
            α′ += (xi ⊗ ∇w) * (1 + β′ ⋅ (xi - xp))
        end
        α = inv(α)
        α′ = inv(α′)
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w = mpvalues.N[i]
            ∇w = mpvalues.∇N[i]
            mpvalues.N[i] = w * α * (1 + β ⋅ (xi - xp))
            mpvalues.∇N[i] = ∇w ⋅ α′ * (1 + β′ ⋅ (xi - xp))
        end
    end

    mpvalues
end

function update!(mpvalues::KernelCorrectionValues, grid::Grid, xp::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    _update!(mpvalues, grid, xp, spat, neighboring_nodes(grid, xp, getsupportlength(F)))
end

function update!(mpvalues::KernelCorrectionValues{GIMP}, grid::Grid, xp::Vec, r::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    _update!(mpvalues, grid, xp, spat, neighboring_nodes(grid, xp, getsupportlength(F, r.*dx⁻¹)), r)
end

@inline function Base.getindex(mpvalues::KernelCorrectionValues, i::Int)
    @_propagate_inbounds_meta
    KernelCorrectionValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.xp)
end
