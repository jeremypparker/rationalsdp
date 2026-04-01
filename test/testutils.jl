using Test
using JuMP
using DynamicPolynomials
using RationalSDP
using LinearAlgebra
using Dualization
import MathOptInterface as MOI

function rational_model(::Type{T}) where {T<:Real}
    model = GenericModel{T}(RationalSDP.Optimizer{T})
    set_silent(model)
    return model
end

function dual_model(::Type{T}) where {T<:Real}
    model = GenericModel{T}(dual_optimizer(RationalSDP.Optimizer{T}, coefficient_type=T, assume_min_if_feasibility=true))
    set_silent(model)
    return model
end
