import Pkg
using LinearAlgebra
using Printf
using Statistics

Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)

using DynamicPolynomials
using JuMP
using RationalSDP

const MOI = RationalSDP.MOI

function rational_model(::Type{T}) where {T<:Real}
    model = GenericModel{T}(RationalSDP.Optimizer{T})
    set_silent(model)
    return model
end

function optimizer_from_model(model)
    MOI.Utilities.attach_optimizer(backend(model))
    bridge_optimizer = getfield(backend(model), :optimizer)
    return getfield(bridge_optimizer, :model)
end

function build_lp_box_instance(n::Int)
    model = rational_model(Rational{BigInt})
    upper_bounds = vcat(fill(1 // 4, min(6, n)), fill(1 // 1, max(0, n - 6)))
    @variable(model, x[1:n])
    for i in 1:n
        @constraint(model, x[i] >= 0 // 1)
        @constraint(model, x[i] <= upper_bounds[i])
    end
    @constraint(model, sum(x) == 1 // 1)
    @objective(model, Min, sum((i // 1) * x[i] for i in 1:n))
    return model
end

function build_lorenz_quartic_instance()
    model = rational_model(Rational{BigInt})
    @polyvar x[1:3]

    f = [
        10 * (x[2] - x[1]);
        28 * x[1] - x[1] * x[3] - x[2];
        x[1] * x[2] - 8 // 3 * x[3];
    ]

    basis_V = monomials(x, 0:4)
    @variable(model, coeffs_V[1:length(basis_V)])
    V = dot(coeffs_V, basis_V)
    LV = dot(f, differentiate(V, x))

    basis_b = monomials(x, 0:2)
    @variable(model, Q[1:length(basis_b), 1:length(basis_b)], PSD)
    @variable(model, B)
    @constraint(model, coefficients(B - x[3]^2 - LV - basis_b' * Q * basis_b) .== 0)
    @objective(model, Min, B)
    return model
end

function build_lorenz_sextic_instance()
    model = rational_model(Rational{BigInt})
    @polyvar x[1:3]

    f = [
        10 * (x[2] - x[1]);
        28 * x[1] - x[1] * x[3] - x[2];
        x[1] * x[2] - 8 // 3 * x[3];
    ]

    basis_V = monomials(x, 0:6)
    @variable(model, coeffs_V[1:length(basis_V)])
    V = dot(coeffs_V, basis_V)
    LV = dot(f, differentiate(V, x))

    basis_b = monomials(x, 0:3)
    @variable(model, Q[1:length(basis_b), 1:length(basis_b)], PSD)
    @variable(model, B)
    @constraint(model, coefficients(B - x[3]^2 - LV - basis_b' * Q * basis_b) .== 0)
    @objective(model, Min, B)
    return model
end

function extraction_dimensions(opt)
    problem = RationalSDP._extract_problem(opt)
    nullity = problem.affine === nothing ? -1 : size(problem.affine[2], 2)
    return (
        variables = length(problem.objective_vector_raw),
        equations = size(problem.A, 1),
        blocks = length(problem.blocks),
        scalar_slacks = length(problem.positive_scalars),
        nullity = nullity,
    )
end

function benchmark_extraction_case(name::AbstractString, builder; repetitions::Int)
    warm_model = builder()
    warm_opt = optimizer_from_model(warm_model)
    dims = extraction_dimensions(warm_opt)
    println(
        @sprintf(
            "%-18s variables=%d equations=%d blocks=%d scalar_slacks=%d nullity=%d",
            name,
            dims.variables,
            dims.equations,
            dims.blocks,
            dims.scalar_slacks,
            dims.nullity,
        ),
    )

    times = Float64[]
    allocations = Int[]
    for repetition in 1:repetitions
        model = builder()
        opt = optimizer_from_model(model)
        timed = @timed extraction_dimensions(opt)
        elapsed = timed.time
        allocated = timed.bytes
        push!(times, elapsed)
        push!(allocations, allocated)
        println(
            @sprintf(
                "%-18s rep %d/%d: %.3fs, allocated=%.2f MiB",
                name,
                repetition,
                repetitions,
                elapsed,
                allocated / 2.0^20,
            ),
        )
    end

    println(
        @sprintf(
            "%-18s avg=%.3fs min=%.3fs max=%.3fs avg_alloc=%.2f MiB",
            name,
            mean(times),
            minimum(times),
            maximum(times),
            mean(allocations) / 2.0^20,
        ),
    )
    println()
    return
end

function main()
    repetitions = something(tryparse(Int, get(ENV, "RATIONALSDP_BENCH_REPS", "3")), 3)
    println("RationalSDP extraction benchmark suite")
    println("Repetitions: " * string(repetitions))
    println()

    benchmark_extraction_case("lp_box_24", () -> build_lp_box_instance(24); repetitions)
    benchmark_extraction_case("lp_box_96", () -> build_lp_box_instance(96); repetitions)
    benchmark_extraction_case("lorenz_quartic", build_lorenz_quartic_instance; repetitions)
    benchmark_extraction_case("lorenz_sextic", build_lorenz_sextic_instance; repetitions)
end

main()
