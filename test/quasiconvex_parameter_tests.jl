@testset "Quasiconvex parameter optimization via JuMP" begin
    model = rational_model(Rational{BigInt})
    set_optimizer_attribute(model, "phase1_backend", :native)
    set_optimizer_attribute(model, "working_float_type", Float64)
    set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 3)

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
