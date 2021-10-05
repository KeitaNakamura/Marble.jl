struct SoilHypoelastic{T} <: MaterialModel
    κ::T
    ν::T
    e0::T
    D_p⁻¹::SymmetricFourthOrderTensor{3, T, 36}
end

function SoilHypoelastic(; κ::Real, ν::Real, e0::Real)
    K = (1 + e0) / κ
    G = 3K * (1-2ν) / 2(1+ν)
    λ = 3K*ν / (1+ν)
    T = promote_type(typeof.((κ, ν, e0, K, G, λ))...)
    δ = one(SymmetricSecondOrderTensor{3, T})
    I = one(SymmetricFourthOrderTensor{3, T})
    D = λ * δ ⊗ δ + 2G * I
    SoilHypoelastic(κ, ν, e0, D)
end

function update_stress(model::SoilHypoelastic, σ::SymmetricSecondOrderTensor{3}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ)
    @_inline_meta
    σ + compute_stiffness_tensor(model, σ) ⊡ dϵ
end

function compute_stiffness_tensor(model::SoilHypoelastic, σ::SymmetricSecondOrderTensor{3})
    model.D_p⁻¹ * abs(mean(σ))
end
