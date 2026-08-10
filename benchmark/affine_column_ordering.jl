import Pkg
using Printf
using LinearAlgebra

Pkg.activate(joinpath(@__DIR__, "..", "test"); io = devnull)
Pkg.instantiate(; io = devnull)

using RationalSDP
using DynamicPolynomials
using JuMP
using SumOfSquares

const ExactRational = Rational{BigInt}

function representative_sos_matrix()
    model = GenericModel{ExactRational}(RationalSDP.Optimizer{ExactRational})
    set_silent(model)
    @polyvar x[1:3]
    dynamics = [
        10 * (x[2] - x[1]);
        28 * x[1] - x[1] * x[3] - x[2];
        x[1] * x[2] - 8 // 3 * x[3];
    ]
    lyapunov_basis = monomials(x, 0:4)
    @variable(model, coefficients_V[1:length(lyapunov_basis)])
    V = dot(coefficients_V, lyapunov_basis)
    derivative = dot(dynamics, differentiate(V, x))
    gram_basis = monomials(x, 0:2)
    @variable(model, Q[1:length(gram_basis), 1:length(gram_basis)], PSD)
    @variable(model, bound)
    @constraint(
        model,
        coefficients(bound - x[3]^2 - derivative - gram_basis' * Q * gram_basis) .== 0,
    )
    @objective(model, Min, bound)

    RationalSDP.MOI.Utilities.attach_optimizer(backend(model))
    bridge_optimizer = getfield(backend(model), :optimizer)
    optimizer = getfield(bridge_optimizer, :model)
    return RationalSDP._extract_problem(optimizer).A
end

function coefficient_matching_fixture()
    row_count = 64
    dense_columns = 20
    singleton_columns = 128
    sparse_columns = 32
    zero_columns = 8
    column_count = dense_columns + singleton_columns + sparse_columns + zero_columns
    A = zeros(ExactRational, row_count, column_count)
    denominators = (2, 3, 5, 7, 11, 13)

    for column in 1:dense_columns, row in 1:row_count
        (row + 2column) % 5 == 0 && continue
        numerator_value = mod(3row + 5column, 17) - 8
        iszero(numerator_value) && (numerator_value = 1)
        denominator_index = mod1(row + column, length(denominators))
        A[row, column] = BigInt(numerator_value) // BigInt(denominators[denominator_index])
    end

    singleton_start = dense_columns + 1
    for offset in 0:(singleton_columns - 1)
        column = singleton_start + offset
        row = mod1(offset + 1, row_count)
        A[row, column] = BigInt(mod(offset, 7) + 1) // BigInt(mod(offset, 5) + 1)
    end

    sparse_start = singleton_start + singleton_columns
    for offset in 0:(sparse_columns - 1)
        column = sparse_start + offset
        for shift in (0, 11, 29)
            row = mod1(7offset + shift + 1, row_count)
            A[row, column] = BigInt(mod(offset + shift, 9) + 1) // BigInt(mod(shift, 5) + 1)
        end
    end

    point = [
        BigInt(mod(5column, 19) - 9) // BigInt(mod(column, 7) + 1) for
        column in 1:column_count
    ]
    return A, A * point
end

function structural_statistics(A)
    counts = RationalSDP._affine_column_nonzero_counts(A)
    return (
        nonzeros = sum(counts),
        zero_columns = count(iszero, counts),
        singleton_columns = count(==(1), counts),
    )
end

function validate_affine(A, b, affine)
    affine === nothing && error("The benchmark fixture unexpectedly became inconsistent.")
    particular, nullspace = affine
    A * particular == b || error("Particular point is infeasible.")
    A * nullspace == zeros(ExactRational, size(A, 1), size(nullspace, 2)) ||
        error("Nullspace invariant failed.")
    size(nullspace, 1) == size(A, 2) || error("Nullspace has the wrong row dimension.")
    return size(nullspace, 2)
end

function measured_solve(A, b, ordering)
    GC.gc()
    timed = @timed RationalSDP._solve_affine_system(A, b; column_order = ordering)
    return timed.value, timed.time, timed.bytes
end

function main()
    sos_A = representative_sos_matrix()
    sos_stats = structural_statistics(sos_A)
    println(
        @sprintf(
            "Representative Lorenz-quartic SOS: %d-by-%d coefficient matrix (%d-by-%d augmented), nnz=%d, zero columns=%d, singleton columns=%d",
            size(sos_A, 1),
            size(sos_A, 2),
            size(sos_A, 1),
            size(sos_A, 2) + 1,
            sos_stats.nonzeros,
            sos_stats.zero_columns,
            sos_stats.singleton_columns,
        ),
    )

    A, b = coefficient_matching_fixture()
    stats = structural_statistics(A)
    println(
        @sprintf(
            "RREF input: %d-by-%d coefficient matrix (%d-by-%d augmented), nnz=%d, zero columns=%d, singleton columns=%d",
            size(A, 1),
            size(A, 2),
            size(A, 1),
            size(A, 2) + 1,
            stats.nonzeros,
            stats.zero_columns,
            stats.singleton_columns,
        ),
    )

    # Compile both keyword paths before collecting diagnostic measurements.
    RationalSDP._solve_affine_system(A, b; column_order = :natural)
    RationalSDP._solve_affine_system(A, b; column_order = :sparsity)

    results = Dict{Symbol,Tuple{Any,Float64,Int}}()
    for ordering in (:natural, :sparsity)
        affine, elapsed, allocated = measured_solve(A, b, ordering)
        results[ordering] = (affine, elapsed, allocated)
        nullity = validate_affine(A, b, affine)
        println(
            @sprintf(
                "%-8s elapsed=%.6fs allocated=%.2f MiB nullity=%d",
                string(ordering),
                elapsed,
                allocated / 2.0^20,
                nullity,
            ),
        )
    end

    natural_affine = results[:natural][1]
    sparse_affine = results[:sparsity][1]
    size(natural_affine[2], 2) == size(sparse_affine[2], 2) ||
        error("Ordering strategies produced different affine dimensions.")
    A * (natural_affine[1] - sparse_affine[1]) == zeros(ExactRational, size(A, 1)) ||
        error("Particular points do not lie in the same affine set.")
    println("Affine feasibility and affine dimensions agree exactly.")
    return
end

main()
