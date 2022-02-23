struct NeoHookean{T} <: MaterialModel
    E::T
    K::T
    G::T
    λ::T
    ν::T
end

NeoHookean(; kwargs...) = NeoHookean{Float64}(; kwargs...)

function NeoHookean{T}(; kwargs...) where {T}
    lin = LinearElastic{T}(; kwargs...)
    NeoHookean{T}(lin.E, lin.K, lin.G, lin.λ, lin.ν)
end

function convert_type(::Type{T}, model::NeoHookean) where {T}
    NeoHookean{T}(
        convert(T, model.E),
        convert(T, model.K),
        convert(T, model.G),
        convert(T, model.λ),
        convert(T, model.ν),
    )
end

@inline function matcalc(::Val{:first_piola_kirchhoff}, model::NeoHookean, F::SecondOrderTensor{3}, J = det(F)) # J for pre-calculation
    G = model.G
    λ = model.λ
    F⁻ᵀ = inv(F)'
    G * (F - F⁻ᵀ) + λ * log(J) * F⁻ᵀ
end

@inline function matcalc(::Val{:stress}, model::NeoHookean, F::SecondOrderTensor{3})
    J = det(F)
    P = matcalc(Val(:first_piola_kirchhoff), model, F, J)
    σ = symmetric((P ⋅ F') / J, :U)
end
