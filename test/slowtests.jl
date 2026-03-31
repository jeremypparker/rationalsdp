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
end
