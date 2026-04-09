function explicit_basis_even_no_s_singular(ws)
    return [
        one(ws.c),
        ws.v[1],
        ws.v[3],
        ws.c,
        ws.c * ws.v[1],
        ws.c * ws.v[3],
        ws.c^2,
        ws.c^2 * ws.v[1],
        ws.c^2 * ws.v[3],
        ws.c^3 * ws.v[1],
        ws.c * ws.v[1] * ws.v[3]
    ]
end

@testset "RationalSDP slow regressions" begin
    @testset "Working float type selection with BigFloat solve" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", BigFloat)
        @test get_optimizer_attribute(model, "working_float_type") == BigFloat
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == BigFloat
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == BigFloat

        set_optimizer_attribute(model, "phase1_hypatia_float_type", "Float64")
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        set_optimizer_attribute(model, "phase1_hypatia_float_type", "auto")
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == BigFloat

        set_optimizer_attribute(model, "facial_reduction_float_type", "Float64")
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == Float64
        set_optimizer_attribute(model, "facial_reduction_float_type", "auto")
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == BigFloat

        @variable(model, X[1:1, 1:1], PSD)
        @constraint(model, X[1, 1] == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test value(X[1, 1]) == 1//1
    end

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

    @testset "KSE time average bound with split even basis" begin
        model = rational_model(Rational{BigInt})
        instance = build_explicit_kse_model(
            3//4,
            model;
            basis_even_no_s_builder = explicit_basis_even_no_s_singular,
        )
        optimize!(instance.model)

        @test termination_status(instance.model) == MOI.OPTIMAL
        @test all(
            iszero(value(coeff)) for
            expr in instance.certificate.expressions for
            coeff in coefficients(expr)
        )
        @test is_psd_exact(value.(instance.certificate.Q_even))
        @test is_psd_exact(value.(instance.certificate.Q_odd))
        @test value(instance.B) > 280//100
        @test value(instance.B) < 281//100
    end
end
