@testset "Quasiconvex parameter optimization via JuMP" begin
    model = rational_model(Rational{BigInt})
    set_optimizer_attribute(model, "phase1_backend", :native)
    set_optimizer_attribute(model, "working_float_type", Float64)
    set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 3)
    @test get_optimizer_attribute(
        model,
        "quasiconvex_skip_facial_reduction_after_clean_endpoint",
    )

    @variable(model, 0//1 <= gamma <= 2//1)
    @variable(model, 0//1 <= x <= 1//1)
    @constraint(model, Symmetric([gamma * x 1//1; 1//1 x]) in PSDCone())
    @objective(model, Min, gamma)

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test objective_value(model) == value(gamma)
    @test 1//1 <= value(gamma) <= 5//4
    @test 0//1 <= value(x) <= 1//1
    @test is_psd_exact([value(gamma) * value(x) 1//1; 1//1 value(x)])
    @test MOI.get(backend(model), MOI.RawStatusString()) ==
          "Solved by quasi-convex parameter search"
end

@testset "Quasiconvex fixed probes match substituted feasibility extraction" begin
    BR = Rational{BigInt}

    block_signature(problem) = [
        (
            block.size,
            block.global_positions,
            block.local_positions,
            block.diagonal_positions,
        ) for block in problem.blocks
    ]

    affine_row_signature(problem) = sort([
        Tuple(vcat(collect(problem.A[row, :]), problem.b[row])) for row in axes(problem.A, 1)
    ])

    function concrete_optimizer(model)
        MOI.Utilities.attach_optimizer(backend(model))
        opt = backend(model)
        while !(opt isa RationalSDP.Optimizer)
            if hasfield(typeof(opt), :optimizer)
                opt = getfield(opt, :optimizer)
            elseif hasfield(typeof(opt), :model)
                opt = getfield(opt, :model)
            else
                error("Could not unwrap RationalSDP optimizer from JuMP backend.")
            end
        end
        return opt
    end

    function fixed_child_problem(model, fixed_value)
        opt = concrete_optimizer(model)
        data = RationalSDP._detect_quasiconvex_parameter(opt)
        @test data !== nothing
        template = RationalSDP._fixed_parameter_template(opt, data)
        return RationalSDP._instantiate_fixed_parameter_problem(
            opt,
            template,
            convert(typeof(data.lower), fixed_value),
        )
    end

    function assert_same_numeric_problem(actual, expected)
        @test affine_row_signature(actual) == affine_row_signature(expected)
        @test actual.positive_scalars == expected.positive_scalars
        @test block_signature(actual) == block_signature(expected)
        @test actual.objective_vector_raw == expected.objective_vector_raw
        @test actual.objective_vector_min == expected.objective_vector_min
        @test actual.objective_constant_raw == expected.objective_constant_raw
        @test actual.affine == expected.affine
    end

    qc_scalar = rational_model(BR)
    @variable(qc_scalar, 0//1 <= gamma <= 2//1)
    @variable(qc_scalar, 0//1 <= x <= 1//1)
    @constraint(qc_scalar, gamma * x + gamma == 3//1)
    @objective(qc_scalar, Min, gamma)

    fixed_scalar = rational_model(BR)
    @variable(fixed_scalar, 0//1 <= x_fixed <= 1//1)
    @constraint(fixed_scalar, 2//1 * x_fixed + 2//1 == 3//1)
    @objective(fixed_scalar, Min, 0//1)

    assert_same_numeric_problem(
        fixed_child_problem(qc_scalar, 2//1),
        RationalSDP._extract_problem(concrete_optimizer(fixed_scalar)),
    )

    qc_psd = rational_model(BR)
    @variable(qc_psd, 0//1 <= gamma <= 2//1)
    @variable(qc_psd, 0//1 <= x <= 1//1)
    @constraint(qc_psd, Symmetric([gamma * x 1//1; 1//1 x]) in PSDCone())
    @objective(qc_psd, Min, gamma)

    fixed_psd = rational_model(BR)
    @variable(fixed_psd, 0//1 <= x_fixed <= 1//1)
    @constraint(fixed_psd, Symmetric([2//1 * x_fixed 1//1; 1//1 x_fixed]) in PSDCone())
    @objective(fixed_psd, Min, 0//1)

    assert_same_numeric_problem(
        fixed_child_problem(qc_psd, 2//1),
        RationalSDP._extract_problem(concrete_optimizer(fixed_psd)),
    )
end

@testset "Quasiconvex scalar quadratic equality via JuMP" begin
    model = rational_model(Rational{BigInt})
    set_optimizer_attribute(model, "phase1_backend", :native)
    set_optimizer_attribute(model, "working_float_type", Float64)
    set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 3)

    @variable(model, 0//1 <= gamma <= 2//1)
    @variable(model, 0//1 <= x <= 1//1)
    @constraint(model, gamma * x == 1//1)
    @objective(model, Min, gamma)

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test objective_value(model) == value(gamma)
    @test 1//1 <= value(gamma) <= 5//4
    @test 0//1 <= value(x) <= 1//1
    @test value(gamma) * value(x) == 1//1
    @test MOI.get(backend(model), MOI.RawStatusString()) ==
          "Solved by quasi-convex parameter search"
end

@testset "Quasiconvex endpoint handling" begin
    function endpoint_model(lower, upper)
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)
        set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 3)
        @variable(model, lower <= gamma <= upper)
        @variable(model, x)
        @constraint(model, x == 1//1)
        return model, gamma, x
    end

    min_lower_feasible, gamma, x = endpoint_model(0//1, 2//1)
    @constraint(min_lower_feasible, gamma * x >= 0//1)
    @objective(min_lower_feasible, Min, gamma)
    optimize!(min_lower_feasible)
    @test termination_status(min_lower_feasible) == MOI.OPTIMAL
    @test value(gamma) == 0//1

    min_upper_infeasible, gamma, x = endpoint_model(0//1, 1//2)
    @constraint(min_upper_infeasible, gamma * x >= 1//1)
    @objective(min_upper_infeasible, Min, gamma)
    optimize!(min_upper_infeasible)
    @test termination_status(min_upper_infeasible) == MOI.INFEASIBLE
    @test MOI.get(backend(min_upper_infeasible), MOI.RawStatusString()) ==
          "Quasi-convex parameter upper bound is infeasible"

    max_upper_feasible, gamma, x = endpoint_model(0//1, 2//1)
    @constraint(max_upper_feasible, gamma * x <= 2//1)
    @objective(max_upper_feasible, Max, gamma)
    optimize!(max_upper_feasible)
    @test termination_status(max_upper_feasible) == MOI.OPTIMAL
    @test value(gamma) == 2//1

    max_lower_infeasible, gamma, x = endpoint_model(0//1, 1//2)
    @constraint(max_lower_infeasible, gamma * x <= -1//1)
    @objective(max_lower_infeasible, Max, gamma)
    optimize!(max_lower_infeasible)
    @test termination_status(max_lower_infeasible) == MOI.INFEASIBLE
    @test MOI.get(backend(max_lower_infeasible), MOI.RawStatusString()) ==
          "Quasi-convex parameter lower bound is infeasible"
end

@testset "Quasiconvex max scalar quadratic via JuMP" begin
    model = rational_model(Rational{BigInt})
    set_optimizer_attribute(model, "phase1_backend", :native)
    set_optimizer_attribute(model, "working_float_type", Float64)
    set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 4)
    set_optimizer_attribute(
        model,
        "quasiconvex_skip_facial_reduction_after_clean_endpoint",
        false,
    )
    @test !get_optimizer_attribute(
        model,
        "quasiconvex_skip_facial_reduction_after_clean_endpoint",
    )

    @variable(model, 0//1 <= gamma <= 2//1)
    @variable(model, x)
    @constraint(model, x == 1//1)
    @constraint(model, gamma * x <= 1//1)
    @objective(model, Max, gamma)

    optimize!(model)

    @test termination_status(model) == MOI.OPTIMAL
    @test primal_status(model) == MOI.FEASIBLE_POINT
    @test objective_value(model) == value(gamma)
    @test 1//1 <= value(gamma) <= 9//8
    @test value(gamma) * value(x) <= 1//1
    @test MOI.get(backend(model), MOI.RawStatusString()) ==
          "Solved by quasi-convex parameter search"
end

@testset "Unsupported quadratic PSD constraints fail clearly" begin
    model = rational_model(Rational{BigInt})

    @variable(model, 0//1 <= gamma <= 2//1)
    @variable(model, 0//1 <= x <= 1//1)
    @variable(model, 0//1 <= y <= 1//1)
    @constraint(model, Symmetric([x * y 1//1; 1//1 x]) in PSDCone())
    @objective(model, Min, gamma)

    err = try
        optimize!(model)
        nothing
    catch caught
        caught
    end

    @test err isa MOI.UnsupportedConstraint
    @test occursin("one-parameter quasi-convex optimization", sprint(showerror, err))
    @test occursin("General quadratic SDP constraints are not supported", sprint(showerror, err))
end
