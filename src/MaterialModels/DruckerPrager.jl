struct DruckerPrager{T, Elastic <: Union{LinearElastic{T}, SoilElastic{T}}} <: MaterialModel
    elastic::Elastic
    A::T
    B::T
    b::T
end

function DruckerPrager(elastic; A::Real, B::Real, b::Real = B)
    DruckerPrager(elastic, A, B, b)
end

# for Mohr-Coulomb criterion
function DruckerPrager(elastic, mc_type::Symbol; c::Real, ϕ::Real, ψ::Real = ϕ)
    ϕ = deg2rad(ϕ)
    ψ = deg2rad(ψ)
    if mc_type == :circumscribed
        A = 6c*cos(ϕ) / (√3 * (3 - sin(ϕ)))
        B = 2sin(ϕ) / (√3 * (3 - sin(ϕ)))
        b = 2sin(ψ) / (√3 * (3 - sin(ψ)))
    elseif mc_type == :inscribed
        A = 6c*cos(ϕ) / (√3 * (3 + sin(ϕ)))
        B = 2sin(ϕ) / (√3 * (3 + sin(ϕ)))
        b = 2sin(ψ) / (√3 * (3 + sin(ψ)))
    elseif mc_type == :plane_strain
        A = 3c / sqrt(9 + 12tan(ϕ)^2)
        B = tan(ϕ) / sqrt(9 + 12tan(ϕ)^2)
        b = tan(ψ) / sqrt(9 + 12tan(ψ)^2)
    else
        throw(ArgumentError("Choose Mohr-Coulomb type from :circumscribed, :inscribed and :plane_strain"))
    end
    DruckerPrager(elastic; A, B, b)
end

function update_stress(model::DruckerPrager{<: Any, <: LinearElastic}, σ::SymmetricSecondOrderTensor{3}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ)
    # compute the stress at the elastic trial state
    De = model.elastic.D
    σ_trial = σ + De ⊡ dϵ
    # compute the yield function at the elastic trial state
    dfdσ, f_trial = gradient(σ_trial -> yield_function(model, σ_trial), σ_trial, :all)
    f_trial ≤ 0.0 && return σ_trial
    # compute the increment of the plastic multiplier
    dgdσ = plastic_flow(model, σ_trial)
    Δγ = f_trial / (dgdσ ⊡ De ⊡ dfdσ)
    # compute the stress
    σ_trial - Δγ * (De ⊡ dgdσ)
end

function update_stress(model::DruckerPrager{<: Any, <: SoilElastic}, σ::SymmetricSecondOrderTensor{3, T}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ) where {T}
    # compute the stress at the elastic trial state
    ϵᵉ = compute_elastic_strain(model.elastic, σ)
    ϵᵉ_trial = ϵᵉ + dϵ
    σ_trial = compute_stress(model.elastic, ϵᵉ_trial)
    yield_function(model, σ_trial) ≤ 0.0 && return σ_trial

    # prepare solution vector x
    ϵᵉ = ϵᵉ_trial
    Δγ = zero(T)
    for i in 1:40
        σ = ∇W(model.elastic, ϵᵉ)
        # dfdσ, f = gradient(σ -> yield_function(model, σ), σ, :all)
        f = yield_function(model, σ)
        dfdσ = gradient(σ -> yield_function(model, σ), σ)
        dgdσ = plastic_flow(model, σ)

        R = ϵᵉ - ϵᵉ_trial + Δγ*dgdσ
        norm(R) < sqrt(eps(T)) && abs(f) < sqrt(eps(T)) && break

        Dᵉ = ∇²W(model.elastic, ϵᵉ)
        dfdσ_Dᵉ = dfdσ ⊡ Dᵉ
        dΔγ = (f - dfdσ_Dᵉ ⊡ R) / (dfdσ_Dᵉ ⊡ dgdσ)

        Δγ += dΔγ
        dϵᵉ = -R - dΔγ * dgdσ
        ϵᵉ += dϵᵉ
    end
    σ
end

#=
function residual_vector(model::DruckerPrager, x, ϵᵉ_trial)
    @inbounds begin
        σ = frommandel(SymmetricSecondOrderTensor{3}, view(x, 1:6))
        Δγ = x[7]
    end
    ϵᵉ = compute_elastic_strain(model.elastic, σ)
    f = yield_function(model, σ)
    dgdσ = plastic_flow(model, σ)
    Vec{7}(Tuple(tomandel(ϵᵉ_trial - ϵᵉ - Δγ*dgdσ))..., f)
end

function update_stress(model::DruckerPrager{<: Any, <: SoilElastic}, σ::SymmetricSecondOrderTensor{3, T}, dϵ::SymmetricSecondOrderTensor{3})::typeof(dϵ) where {T}
    # compute the stress at the elastic trial state
    ϵᵉ = compute_elastic_strain(model.elastic, σ)
    ϵᵉ_trial = ϵᵉ + dϵ
    σ_trial = compute_stress(model.elastic, ϵᵉ_trial)
    yield_function(model, σ_trial) ≤ 0.0 && return σ_trial

    # prepare solution vector x
    x = Vec{7}(Tuple(tomandel(σ_trial))..., zero(T))

    # plastic corrector
    for i in 1:20 # should be converged in 20 iterations
        J, r = gradient(x -> residual_vector(model, x, ϵᵉ_trial), x, :all)
        norm(r) < sqrt(eps(T)) && break
        # update states
        x += -J \ r
    end
    @inbounds frommandel(SymmetricSecondOrderTensor{3}, view(x, 1:6))
end
=#

function yield_function(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})::eltype(σ)
    A = model.A
    B = model.B
    I₁ = tr(σ)
    s = dev(σ)
    J₂ = (s ⊡ s) / 2
    √J₂ - (A - B*I₁)
end

function plastic_flow(model::DruckerPrager, σ::SymmetricSecondOrderTensor{3})::typeof(σ)
    b = model.b
    s = dev(σ)
    J₂ = (s ⊡ s) / 2
    if J₂ < eps(typeof(J₂))
        dgdσ = b * one(σ)
    else
        dgdσ = s / (2*√J₂) + b * one(σ)
    end
    dgdσ
end
