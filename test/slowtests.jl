include("kse_timeaverage_helpers.jl")

@testset "RationalSDP slow regressions" begin
    @testset "Sextic Lorenz SOS mean upper bound" begin
        model = rational_model(Rational{BigInt})
        @polyvar x[1:3]

        f = [
            10 * (x[2] - x[1]);
            28 * x[1] - x[1] * x[3] - x[2];
            x[1] * x[2] - 8//3 * x[3];
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
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test value(B) > 729//1
        @test value(B) < 730//1
    end

    @testset "Dualized KSE time-average bound variable" begin
        result = solve_dualized_kse_timeaverage(; phase2_outer_iterations = 4)

        @test result.termination_status == MOI.OPTIMAL
        @test result.primal_status == MOI.FEASIBLE_POINT
        @test result.B_value > 20//1
        @test result.B_value < 21//1
        @test_broken all(
            iszero(value(coeff)) for
            expression in result.certificate.expressions for
            coeff in coefficients(expression)
        )
    end
end
