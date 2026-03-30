import Pkg
using Printf
using Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

Pkg.activate(mktempdir(); io = devnull)
Pkg.add(["JuMP", "DynamicPolynomials"]; io = devnull)
Pkg.develop(path = REPO_ROOT; io = devnull)

using JuMP
using DynamicPolynomials
using RationalSDP

function format_metric(x)
    return @sprintf("%.6f", Float64(BigFloat(x)))
end

function rational_model(::Type{T}) where {T<:Real}
    model = GenericModel{T}(() -> RationalSDP.Optimizer{T}(verbose = false))
    set_silent(model)
    return model
end

function build_lp_box_instance(n::Int)
    model = rational_model(Rational{BigInt})
    upper_bounds = vcat(fill(1 // 4, 6), fill(1 // 1, n - 6))
    @variable(model, x[1:n])
    for i in 1:n
        @constraint(model, x[i] >= 0 // 1)
        @constraint(model, x[i] <= upper_bounds[i])
    end
    @constraint(model, sum(x) == 1 // 1)
    @objective(model, Min, sum((i // 1) * x[i] for i in 1:n))
    return model
end

function build_large_mixed_instance()
    model = rational_model(Rational{BigInt})
    @variable(model, X[1:3, 1:3], PSD)
    @variable(model, Y[1:2, 1:2], PSD)
    @variable(model, u[1:20])
    for i in eachindex(u)
        @constraint(model, u[i] >= 0 // 1)
        @constraint(model, u[i] <= 1 // 1)
    end
    @constraint(model, sum(u) == 6 // 1)
    @constraint(model, X[1, 1] == 4 // 1)
    @constraint(model, X[2, 2] == 3 // 1)
    @constraint(model, X[3, 3] == 2 // 1)
    @constraint(model, X[1, 2] == u[1] - 1 // 5)
    @constraint(model, X[1, 3] == u[2] - 1 // 4)
    @constraint(model, X[2, 3] == u[3] - 1 // 6)
    @constraint(model, Y[1, 1] == 5 // 2)
    @constraint(model, Y[2, 2] == 7 // 3)
    @constraint(model, Y[1, 2] == u[4] - u[5])
    @constraint(model, u[6] + u[7] >= 3 // 5)
    @constraint(model, u[8] + u[9] <= 7 // 5)
    @constraint(model, u[10] - u[11] >= -1 // 2)
    @constraint(model, u[12] + u[13] + u[14] >= 1 // 1)
    @constraint(model, u[15] + u[16] <= 3 // 2)
    @constraint(model, u[17] + u[18] >= 4 // 5)
    @constraint(model, u[19] + u[20] <= 6 // 5)
    @objective(model, Min, sum((i // 1) * u[i] for i in eachindex(u)) - Y[1, 2])
    return model
end

function build_sos_instance()
    model = rational_model(Rational{BigInt})
    @polyvar z
    basis = monomials([z], 0:2)
    @variable(model, Q[1:3, 1:3], PSD)
    @variable(model, t)

    coeffs = Dict{Int,Any}(k => 0 // 1 for k in 0:4)
    for i in eachindex(basis)
        for j in i:length(basis)
            degree_ij = degree(basis[i] * basis[j], z)
            contribution = i == j ? Q[i, j] : 2 // 1 * Q[i, j]
            coeffs[degree_ij] = coeffs[degree_ij] + contribution
        end
    end

    @constraint(model, coeffs[0] == t)
    @constraint(model, coeffs[1] == 0 // 1)
    @constraint(model, coeffs[2] == -1 // 1)
    @constraint(model, coeffs[3] == 0 // 1)
    @constraint(model, coeffs[4] == 1 // 1)
    @objective(model, Min, t)
    return model
end

function benchmark_case(name::AbstractString, builder::Function; repetitions::Int)
    warmup = builder()
    optimize!(warmup)
    warmup_status = termination_status(warmup)
    println(@sprintf("%-18s warmup status=%s", name, string(warmup_status)))

    times = Float64[]
    for repetition in 1:repetitions
        model = builder()
        elapsed = @elapsed optimize!(model)
        push!(times, elapsed)
        println(
            @sprintf(
                "%-18s rep %d/%d: %.3fs, status=%s, objective=%s",
                name,
                repetition,
                repetitions,
                elapsed,
                string(termination_status(model)),
                format_metric(objective_value(model)),
            ),
        )
    end

    println(
        @sprintf(
            "%-18s avg=%.3fs min=%.3fs max=%.3fs",
            name,
            mean(times),
            minimum(times),
            maximum(times),
        ),
    )
    println()
    return
end

function main()
    repetitions = something(tryparse(Int, get(ENV, "RATIONALSDP_BENCH_REPS", "3")), 3)
    println("RationalSDP benchmark suite")
    println("Repository: " * REPO_ROOT)
    println("Repetitions: " * string(repetitions))
    println()

    benchmark_case("lp_box_24", () -> build_lp_box_instance(24); repetitions = repetitions)
    benchmark_case("mixed_cone_20", build_large_mixed_instance; repetitions = repetitions)
    benchmark_case("sos_quartic", build_sos_instance; repetitions = repetitions)
end

main()
