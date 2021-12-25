struct WLS{Basis <: AbstractBasis, Weight <: Kernel} <: Interpolation
    basis::Basis
    weight::Weight
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

WLS{Basis, Weight}() where {Basis, Weight} = WLS(Basis(), Weight())
WLS{Basis}(weight::Kernel) where {Basis} = WLS(Basis(), weight)

basis_function(wls::WLS) = wls.basis
weight_function(wls::WLS) = wls.weight

support_length(wls::WLS, args...) = support_length(weight_function(wls), args...)
active_length(::WLS, args...) = 1.0 # for sparsity pattern


struct WLSValues{Basis, Weight, dim, T, nnodes, L, L²} <: MPValues{dim, T}
    F::WLS{Basis, Weight}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    w::MVector{nnodes, T}
    M⁻¹::Base.RefValue{Mat{L, L, T, L²}}
    x::Base.RefValue{Vec{dim, T}}
    gridindices::MVector{nnodes, Index{dim}}
    len::Base.RefValue{Int}
end

basis_function(x::WLSValues) = basis_function(x.F)
weight_function(x::WLSValues) = weight_function(x.F)

function WLSValues{Basis, Weight, dim, T, nnodes, L, L²}() where {Basis, Weight, dim, T, nnodes, L, L²}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    w = MVector{nnodes, T}(undef)
    M⁻¹ = zero(Mat{L, L, T, L²})
    gridindices = MVector{nnodes, Index{dim}}(undef)
    x = Ref(zero(Vec{dim, T}))
    WLSValues(WLS{Basis, Weight}(), N, ∇N, w, Ref(M⁻¹), x, gridindices, Ref(0))
end

function MPValues{dim, T}(F::WLS{Basis, Weight}) where {Basis, Weight, dim, T}
    L = length(value(basis_function(F), zero(Vec{dim, T})))
    n = nnodes(weight_function(F), Val(dim))
    WLSValues{Basis, Weight, dim, T, n, L, L^2}()
end

function _update!(mpvalues::WLSValues{<: Any, <: Any, dim}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    mpvalues.N .= zero(mpvalues.N)
    mpvalues.∇N .= zero(mpvalues.∇N)
    mpvalues.w .= zero(mpvalues.w)
    P = basis_function(mpvalues)
    M = zero(mpvalues.M⁻¹[])
    mpvalues.x[] = x
    update_gridindices!(mpvalues, grid, x, spat)
    dx⁻¹ = 1 ./ gridsteps(grid)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) .* dx⁻¹
        w = F(ξ)
        p = value(P, xᵢ - x)
        M += w * p ⊗ p
        mpvalues.w[i] = w
    end
    mpvalues.M⁻¹[] = inv(M)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        q = mpvalues.M⁻¹[] ⋅ value(P, xᵢ - x)
        wq = mpvalues.w[i] * q
        mpvalues.N[i] = wq ⋅ value(P, x - x)
        mpvalues.∇N[i] = wq ⋅ gradient(P, x - x)
    end
    mpvalues
end

function _update!(mpvalues::WLSValues{PolynomialBasis{1}, <: Any, dim}, F, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    mpvalues.N .= zero(mpvalues.N)
    mpvalues.∇N .= zero(mpvalues.∇N)
    mpvalues.w .= zero(mpvalues.w)
    P = basis_function(mpvalues)
    M = zero(mpvalues.M⁻¹[])
    mpvalues.x[] = x
    update_gridindices!(mpvalues, grid, x, spat)
    dx⁻¹ = 1 ./ gridsteps(grid)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        ξ = (x - xᵢ) .* dx⁻¹
        w = F(ξ)
        p = value(P, xᵢ - x)
        M += w * p ⊗ p
        mpvalues.w[i] = w
    end
    mpvalues.M⁻¹[] = inv(M)
    @inbounds @simd for i in 1:length(mpvalues)
        I = mpvalues.gridindices[i]
        xᵢ = grid[I]
        q = mpvalues.M⁻¹[] ⋅ value(P, xᵢ - x)
        wq = mpvalues.w[i] * q
        mpvalues.N[i] = wq[1]
        mpvalues.∇N[i] = @Tensor wq[2:end]
    end
    mpvalues
end

function update!(mpvalues::WLSValues{<: Any, <: Any, dim}, grid::Grid{dim}, x::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = weight_function(mpvalues)
    _update!(mpvalues, ξ -> value(F, ξ), grid, x, spat)
end

function update!(mpvalues::WLSValues{<: Any, GIMP, dim}, grid::Grid{dim}, x::Vec{dim}, r::Vec{dim}, spat::AbstractArray{Bool, dim}) where {dim}
    F = weight_function(mpvalues)
    dx⁻¹ = 1 ./ gridsteps(grid)
    _update!(mpvalues, ξ -> value(F, ξ, r.*dx⁻¹), grid, x, spat)
end


struct WLSValue{dim, T, L, L²}
    N::T
    ∇N::Vec{dim, T}
    w::T
    M⁻¹::Mat{L, L, T, L²}
    x::Vec{dim, T}
    index::Index{dim}
end

@inline function Base.getindex(mpvalues::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.w[i], mpvalues.M⁻¹[], mpvalues.x[], mpvalues.gridindices[i])
end