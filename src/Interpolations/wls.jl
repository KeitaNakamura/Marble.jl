struct WLS{B <: AbstractBasis, K <: Kernel} <: Interpolation
end

const LinearWLS = WLS{PolynomialBasis{1}}
const BilinearWLS = WLS{BilinearBasis}

@pure WLS{B}(w::Kernel) where {B} = WLS{B, typeof(w)}()

@pure getbasisfunction(::WLS{B}) where {B} = B()
@pure getkernelfunction(::WLS{B, W}) where {B, W} = W()

getsupportlength(wls::WLS, args...) = getsupportlength(getkernelfunction(wls), args...)


struct WLSValue{dim, T, L, L²} <: MPValue
    N::T
    ∇N::Vec{dim, T}
    w::T
    Minv::Mat{L, L, T, L²}
    xp::Vec{dim, T}
end

mutable struct WLSValues{B, K, dim, T, nnodes, L, L²} <: MPValues{dim, T, WLSValue{dim, T, L, L²}}
    F::WLS{B, K}
    N::MVector{nnodes, T}
    ∇N::MVector{nnodes, Vec{dim, T}}
    w::MVector{nnodes, T}
    gridindices::MVector{nnodes, Index{dim}}
    Minv::Mat{L, L, T, L²}
    xp::Vec{dim, T}
    len::Int
end

getbasisfunction(x::WLSValues) = getbasisfunction(x.F)
getkernelfunction(x::WLSValues) = getkernelfunction(x.F)

function WLSValues{B, K, dim, T, nnodes, L, L²}() where {B, K, dim, T, nnodes, L, L²}
    N = MVector{nnodes, T}(undef)
    ∇N = MVector{nnodes, Vec{dim, T}}(undef)
    w = MVector{nnodes, T}(undef)
    Minv = zero(Mat{L, L, T, L²})
    gridindices = MVector{nnodes, Index{dim}}(undef)
    xp = zero(Vec{dim, T})
    WLSValues(WLS{B, K}(), N, ∇N, w, gridindices, Minv, xp, 0)
end

function MPValues{dim, T}(F::WLS{B, K}) where {B, K, dim, T}
    L = length(value(getbasisfunction(F), zero(Vec{dim, T})))
    n = getnnodes(getkernelfunction(F), Val(dim))
    WLSValues{B, K, dim, T, n, L, L^2}()
end

# general version
function _update!(mpvalues::WLSValues, grid::Grid, xp::Vec, spat::AbstractArray{Bool}, inds, args...)
    F = getkernelfunction(mpvalues)
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    fillzero!(mpvalues.w)
    P = getbasisfunction(mpvalues)
    M = zero(mpvalues.Minv)
    mpvalues.xp = xp
    update_active_gridindices!(mpvalues, inds, spat)
    @inbounds @simd for i in 1:length(mpvalues)
        I = gridindices(mpvalues, i)
        xi = grid[I]
        w = value(F, grid, I, xp, args...)
        p = value(P, xi - xp)
        M += w * p ⊗ p
        mpvalues.w[i] = w
    end
    mpvalues.Minv = inv(M)
    @inbounds @simd for i in 1:length(mpvalues)
        I = gridindices(mpvalues, i)
        xi = grid[I]
        q = mpvalues.Minv ⋅ value(P, xi - xp)
        wq = mpvalues.w[i] * q
        mpvalues.N[i] = wq ⋅ value(P, xp - xp)
        mpvalues.∇N[i] = wq ⋅ gradient(P, xp - xp)
    end
    mpvalues
end

# fast version for `LinearWLS(BSpline{order}())`
function _update!(mpvalues::WLSValues{PolynomialBasis{1}, <: BSpline, dim, T}, grid::Grid{dim}, xp::Vec{dim}, spat::AbstractArray{Bool, dim}, inds) where {dim, T}
    F = getkernelfunction(mpvalues)
    fillzero!(mpvalues.N)
    fillzero!(mpvalues.∇N)
    fillzero!(mpvalues.w)
    P = getbasisfunction(mpvalues)
    mpvalues.xp = xp

    allactive = update_active_gridindices!(mpvalues, inds, spat)
    if allactive
        # fast version
        D = zero(Vec{dim, T}) # diagonal entries
        wᵢ = values(getkernelfunction(mpvalues), grid, xp)
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w = wᵢ[i]
            D += w * (xi - xp) .* (xi - xp)
            mpvalues.w[i] = w
            mpvalues.∇N[i] = w * (xi - xp) # for fast computation
        end
        D⁻¹ = inv.(D)
        @inbounds @simd for i in 1:length(mpvalues)
            mpvalues.N[i] = wᵢ[i]
            mpvalues.∇N[i] = mpvalues.∇N[i] .* D⁻¹
        end
        Minv = diagm(vcat(1, D⁻¹))
    else
        M = zero(mpvalues.Minv)
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            w = value(F, grid, I, xp)
            p = value(P, xi - xp)
            M += w * p ⊗ p
            mpvalues.w[i] = w
        end
        Minv = inv(M)
        @inbounds @simd for i in 1:length(mpvalues)
            I = gridindices(mpvalues, i)
            xi = grid[I]
            q = Minv ⋅ value(P, xi - xp)
            wq = mpvalues.w[i] * q
            mpvalues.N[i] = wq[1]
            mpvalues.∇N[i] = @Tensor wq[2:end]
        end
    end
    mpvalues.Minv = Minv

    mpvalues
end

function update!(mpvalues::WLSValues, grid::Grid, xp::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    _update!(mpvalues, grid, xp, spat, neighboring_nodes(grid, xp, getsupportlength(F)))
end

function update!(mpvalues::WLSValues{<: Any, GIMP}, grid::Grid, xp::Vec, r::Vec, spat::AbstractArray{Bool})
    F = getkernelfunction(mpvalues)
    dx⁻¹ = gridsteps_inv(grid)
    _update!(mpvalues, grid, xp, spat, neighboring_nodes(grid, xp, getsupportlength(F, r.*dx⁻¹)), r)
end

@inline function Base.getindex(mpvalues::WLSValues, i::Int)
    @_propagate_inbounds_meta
    WLSValue(mpvalues.N[i], mpvalues.∇N[i], mpvalues.w[i], mpvalues.Minv, mpvalues.xp)
end
