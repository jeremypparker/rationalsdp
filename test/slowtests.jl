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

    @testset "Lorenz symmetric period bound with quasiconvex B" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 8)

        instance = build_lorenz_symmetric_period_model(model; da = 2, db = 3, dc = 6, lower_B = 700, upper_B=750)
        optimize!(instance.model)

        @test termination_status(instance.model) == MOI.OPTIMAL
        @test primal_status(instance.model) == MOI.FEASIBLE_POINT
        @test MOI.get(backend(instance.model), MOI.RawStatusString()) ==
              "Solved by quasi-convex parameter search"
        @test 704//1 <= value(instance.B) <= 705//1
        @test is_psd_exact(value.(instance.Q))
        @test is_psd_exact(value.(instance.Pe))
        @test is_psd_exact(value.(instance.Po))
        @test all(iszero(value(coeff)) for coeff in coefficients(instance.certificate))
    end

    @testset "SIRS log-domain SOS feasibility with facial reduction" begin
        model = rational_model(Rational{BigInt})

        @polyvar s i r G

        gamma = 1 // 10
        mu = 1 // 10
        beta = 1 // 1
        delta = 0 // 1
        alpha = 1 // 10
        Lambda = mu
        R0 = Lambda * beta / (mu * (mu + delta + gamma))
        S1 = Lambda / mu * ((1 // 1) / R0)
        I1 = Lambda * (alpha + mu) * (R0 - 1) /
             (R0 * ((gamma + delta + mu) * (alpha + mu) - alpha * gamma))
        R1 = I1 * gamma / (alpha + mu)

        S = s + S1
        I = i + I1
        R = r + R1

        dsdt = Lambda - beta * S * I - mu * S + alpha * R
        didt = beta * S * I - (delta + gamma + mu) * I
        drdt = gamma * I - (alpha + mu) * R
        dGdt = didt - I1 * (beta * S - (delta + gamma + mu))

        basisV = monomials([s, i, r, G], 0:2)
        @variable(model, coeffsV[1:length(basisV)])
        V = dot(basisV, coeffsV)

        dVdt =
            differentiate(V, s) * dsdt +
            differentiate(V, i) * didt +
            differentiate(V, r) * drdt +
            differentiate(V, G) * dGdt

        D = @set i + I1 >= 0 &&
                 s + S1 >= 0 &&
                 r + R1 >= 0 &&
                 G >= 0

        @constraint(model, V(s => 0, i => 0, r => 0, G => 0) == 0)
        @constraint(model, V >= s^2 + G + r^2, SOSCone(), domain = D)
        @constraint(model, -(s^2 + i^2 + r^2) >= dVdt, SOSCone(), domain = D)

        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
    end

    @testset "SIS log-domain full Volterra SOS feasibility with facial reduction" begin
        model = rational_model(Rational{BigInt})

        @polyvar i n l m

        gamma = 1 // 10
        mu = 1 // 10
        beta = 1 // 1
        delta = 0 // 1
        Lambda = mu

        R0 = Lambda * beta / (mu * (mu + delta + gamma))
        S1 = Lambda / (mu * R0)
        I1 = mu * (delta + gamma + mu) * (R0 - 1) / (beta * (delta + mu))

        s = -i - n
        S = s + S1
        I = i + I1

        dSdt = Lambda + gamma * I - beta * S * I - mu * S
        dIdt = beta * S * I - (delta + gamma + mu) * I
        dndt = -dSdt - dIdt

        basisV = monomials([i, n, l, m], 0:2)
        @variable(model, coeffsV[1:length(basisV)])
        V = dot(basisV, coeffsV)

        SIdVdt =
            S * I * differentiate(V, i) * dIdt +
            S * I * differentiate(V, n) * dndt +
            I * differentiate(V, l) * (s * dSdt) +
            S * differentiate(V, m) * (i * dIdt)

        D = @set S >= 0 &&
                 I >= 0 &&
                 n >= 0 &&
                 l >= 0 &&
                 m >= 0

        @constraint(model, V(i => 0, n => 0, l => 0, m => 0) == 0)
        @constraint(model, V >= l + m, SOSCone(), domain = D)
        @constraint(model, -S * I * (s^2 + i^2) - SIdVdt >= 0, SOSCone(), domain = D)

        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
    end
end
