using SumOfSquares

function JuMP.add_variable(
    model::JuMP.GenericModel{T},
    v::SumOfSquares.PolyJuMP.Variable{<:SumOfSquares.PolyJuMP.Poly},
    name::String = "",
) where {T}
    function _newvar(i)
        vref = JuMP.GenericVariableRef{T}(model)
        if v.binary
            JuMP.set_binary(vref)
        end
        if v.integer
            JuMP.set_integer(vref)
        end
        return vref
    end
    return SumOfSquares.PolyJuMP.MB.algebra_element(_newvar, v.p.polynomial_basis)
end

@testset "SumOfSquares integration" begin
    @testset "Quartic lower bound via minimization" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar z
        @variable(model, t)
        poly = z^4 - z^2 + t
        cref = @constraint(model, poly >= 0, SumOfSquares.SOSCone())
        @objective(model, Min, t)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        test_exact_sos_constraint(model, cref, poly)
        @test value(t) > 249//1000
        @test value(t) < 251//1000
    end

    @testset "Quartic lower bound via maximization" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar z
        @variable(model, t)
        poly = z^4 - z^2 - t
        cref = @constraint(model, poly in SumOfSquares.SOSCone())
        @objective(model, Max, t)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        test_exact_sos_constraint(model, cref, poly)
        @test value(t) > -251//1000
        @test value(t) < -249//1000
    end

    @testset "Lyapunov feasibility" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)

        @polyvar x[1:2]
        f = -x
        radial = x[1]^2 + x[2]^2

        @variable(model, V, Poly(monomials(x, 0:2)))
        LV = dot(f, differentiate(V, x))

        poly1 = -LV - 1//2 * radial
        poly2 = V - 1//2 * radial
        cref1 = @constraint(model, poly1 >= 0, SumOfSquares.SOSCone())
        cref2 = @constraint(model, poly2 >= 0, SumOfSquares.SOSCone())
        @constraint(model, V(x[1] => 1//1, x[2] => 0//1) == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test objective_value(model) == 0//1
        test_exact_sos_constraint(model, cref1, poly1)
        test_exact_sos_constraint(model, cref2, poly2)
        @test value(V)(x[1] => 1//1, x[2] => 0//1) == 1//1
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
        poly = B - x[3]^2 - LV
        cref = @constraint(model, poly >= 0, SumOfSquares.SOSCone())
        @objective(model, Min, B)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        test_exact_sos_constraint(model, cref, poly)
        @test value(B) > 729//1
        @test value(B) < 730//1
    end
end
