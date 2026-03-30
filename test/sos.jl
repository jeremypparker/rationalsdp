using JuMP
using DynamicPolynomials
using RationalSDP
using LinearAlgebra

T = Rational{BigInt}

@polyvar x[1:3]

f = -x

model = GenericModel{T}(() -> RationalSDP.Optimizer{T}())


basis_V = monomials(x, 0:2)
@variable(model, coeffs_V[1:length(basis_V)])
V = dot(coeffs_V, basis_V)
LV = dot(f, differentiate(V, x))

basis_b1 = monomials(x, 0:1)
basis_b2 = monomials(x, 0:1)

@variable(model, Q1[1:length(basis_b1), 1:length(basis_b1)], PSD)
@constraint(model, -LV == basis_b1' * Q1 * basis_b1)
@variable(model, Q2[1:length(basis_b2), 1:length(basis_b2)], PSD)
@constraint(model, V - x^2 == basis_b2' * Q2 * basis_b2)

optimize!(model)