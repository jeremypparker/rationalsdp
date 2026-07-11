using Test
using JuMP
using DynamicPolynomials
using RationalSDP
using LinearAlgebra
using SumOfSquares
using Logging
import MathOptInterface as MOI

function rational_model(::Type{T}) where {T<:Real}
    model = GenericModel{T}(RationalSDP.Optimizer{T})
    set_silent(model)
    return model
end

is_psd_exact(matrix) = RationalSDP._positive_semidefinite_exact(Matrix(matrix))

function _bridge_gram_matrix(model::JuMP.GenericModel, cref::JuMP.ConstraintRef)
    bridge_optimizer = getfield(backend(model), :optimizer)
    root_bridge = MOI.Bridges.bridge(bridge_optimizer, JuMP.index(cref))

    kernel = nothing
    if hasfield(typeof(root_bridge), :slack_in_set)
        kernel = MOI.Bridges.bridge(bridge_optimizer, getfield(root_bridge, :slack_in_set))
    elseif hasfield(typeof(root_bridge), :scalar_constraints)
        for ci in getfield(root_bridge, :scalar_constraints)
            try
                child_bridge = MOI.Bridges.bridge(bridge_optimizer, ci)
                if hasfield(typeof(child_bridge), :slack_in_set)
                    kernel = MOI.Bridges.bridge(
                        bridge_optimizer,
                        getfield(child_bridge, :slack_in_set),
                    )
                    break
                end
            catch
            end
        end
    end
    kernel === nothing && error("Could not locate the KernelBridge for the SOS constraint.")

    gram_variables = getfield(kernel, :variables)[1]
    gram_basis = getfield(kernel, :set).gram_bases[1]
    raw_gram_values =
        MOI.get.(Ref(bridge_optimizer), Ref(MOI.VariablePrimal()), gram_variables)
    T = promote_type(map(typeof, raw_gram_values)...)
    gram_values = T[raw_gram_values...]
    gram_matrix = SumOfSquares.MultivariateMoments.SymMatrix(gram_values, length(gram_basis))
    return SumOfSquares.GramMatrix(gram_matrix, gram_basis)
end

function _unique_exact_gram_matrix(poly, basis, monos)
    n = length(monos)
    entry_data = Dict{typeof(first(monos) * first(monos)),Tuple{Int,Int,Int}}()
    for j in 1:n
        for i in 1:j
            mono = monos[i] * monos[j]
            if haskey(entry_data, mono)
                return nothing
            end
            entry_data[mono] = (i, j, i == j ? 1 : 2)
        end
    end

    Q = Matrix{Rational{BigInt}}(undef, n, n)
    fill!(Q, 0//1)
    for (mono, (i, j, scale)) in entry_data
        value = DynamicPolynomials.coefficient(poly, mono) / scale
        Q[i, j] = value
        Q[j, i] = value
    end
    return SumOfSquares.GramMatrix(Q, basis)
end

function exact_gram_matrix(model::JuMP.GenericModel, cref::JuMP.ConstraintRef, poly)
    gram = _bridge_gram_matrix(model, cref)
    exact_poly = value(poly)
    if eltype(Matrix(SumOfSquares.MultivariateMoments.value_matrix(gram))) <: Rational &&
        iszero(gram - exact_poly)
        return gram
    end

    fallback = _unique_exact_gram_matrix(
        exact_poly,
        certificate_basis(cref),
        certificate_monomials(cref),
    )
    fallback === nothing && return gram
    return fallback
end

function test_exact_sos_constraint(
    model::JuMP.GenericModel,
    cref::JuMP.ConstraintRef,
    poly,
)
    gram = exact_gram_matrix(model, cref, poly)
    @test iszero(value(cref) - value(poly))
    @test iszero(gram - value(poly))
    @test is_psd_exact(Matrix(SumOfSquares.MultivariateMoments.value_matrix(gram)))
    return gram
end

function test_exact_extracted_sdp(model::JuMP.GenericModel)
    bridge_optimizer = getfield(backend(model), :optimizer)
    opt = getfield(bridge_optimizer, :model)
    problem = RationalSDP._extract_problem(opt)
    variable_count = length(problem.original_variables)
    dimension = length(problem.objective_vector_raw)
    fixed_rows = zeros(Rational{BigInt}, variable_count, dimension)
    fixed_rhs = zeros(Rational{BigInt}, variable_count)
    for (position, variable) in enumerate(problem.original_variables)
        fixed_rows[position, position] = 1//1
        fixed_rhs[position] = RationalSDP._exact_rational(opt.variable_primal[variable])
    end
    fixed_affine = RationalSDP._solve_affine_system(
        vcat(problem.A, fixed_rows),
        vcat(problem.b, fixed_rhs),
    )
    @test fixed_affine !== nothing
    fixed_affine === nothing && return nothing
    point, nullspace = fixed_affine
    @test size(nullspace, 2) == 0
    @test problem.A * point == problem.b
    for position in problem.positive_scalars
        @test point[position] >= 0
    end
    for block in problem.blocks
        @test is_psd_exact(RationalSDP._vector_to_matrix(point, block))
    end
    @test RationalSDP._exact_primal_feasibility(problem, point).ok
    return point
end
