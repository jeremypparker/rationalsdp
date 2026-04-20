using SumOfSquares

@testset "SumOfSquares integration" begin
    @testset "Quartic lower bound via minimization" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar z
        @variable(model, t)
        @constraint(model, z^4 - z^2 + t >= 0, SumOfSquares.SOSCone())
        @objective(model, Min, t)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test value(t) > 249//1000
        @test value(t) < 251//1000
    end

    @testset "Quartic lower bound via maximization" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar z
        @variable(model, t)
        @constraint(model, z^4 - z^2 - t in SumOfSquares.SOSCone())
        @objective(model, Max, t)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test value(t) > -251//1000
        @test value(t) < -249//1000
    end

    @testset "Lyapunov feasibility" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar x[1:1]
        f = -x

        @variable(model, V, Poly(monomials(x, 0:2)))
        LV = dot(f, differentiate(V, x))

        @constraint(model, -LV >= 0, SumOfSquares.SOSCone())
        @constraint(model, V - dot(x, x) >= 0, SumOfSquares.SOSCone())
        @constraint(model, V(x[1] => 1//1) == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test objective_value(model) == 0//1
        @test value(V) == x[1]^2
    end

    @testset "Lorenz SOS mean upper bound" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar x[1:3]

        f = [
            10 * (x[2] - x[1]);
            28 * x[1] - x[1] * x[3] - x[2];
            x[1] * x[2] - 8//3 * x[3];
        ]

        @variable(model, V, Poly(monomials(x, 0:2)))
        LV = dot(f, differentiate(V, x))

        @variable(model, B)
        @constraint(model, B - x[3]^2 - LV >= 0, SumOfSquares.SOSCone())
        @objective(model, Min, B)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test value(B) > 729//1
        @test value(B) < 730//1
    end
end
