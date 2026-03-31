using Test
using JuMP
using DynamicPolynomials
using RationalSDP
using LinearAlgebra
import MathOptInterface as MOI

function rational_model(::Type{T}) where {T<:Real}
    model = GenericModel{T}(RationalSDP.Optimizer{T})
    set_silent(model)
    return model
end
