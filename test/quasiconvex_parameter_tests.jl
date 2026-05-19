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
