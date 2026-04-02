include("testutils.jl")

@testset "RationalSDP JuMP integration" begin
    @testset "Optimizer metadata" begin
        model = rational_model(Rational{BigInt})
        @test JuMP.solver_name(model) == "RationalSDP"
    end

    @testset "Optimizer attributes" begin
        model = GenericModel{Rational{BigInt}}(RationalSDP.Optimizer{Rational{BigInt}})
        set_optimizer_attribute(model, "phase1_outer_iterations", 24)
        set_optimizer_attribute(model, "phase1_backend", "native")
        set_optimizer_attribute(model, "phase1_hypatia_float_type", "Float64")
        set_optimizer_attribute(model, "phase1_hypatia_margin_upper", "1e-4")
        set_optimizer_attribute(model, "phase1_hypatia_min_margin_upper", "1e-8")
        set_optimizer_attribute(model, "phase1_hypatia_margin_shrink", "0.2")
        set_optimizer_attribute(model, "phase1_hypatia_objective_bias", "1e-12")
        set_optimizer_attribute(model, "verbose", false)
        set_optimizer_attribute(model, "rational_tolerance", "1e-30")
        set_optimizer_attribute(model, "working_float_type", "BigFloat")
        @test get_optimizer_attribute(model, "phase1_outer_iterations") == 24
        @test get_optimizer_attribute(model, "phase1_backend") == :native
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        @test get_optimizer_attribute(model, "phase1_hypatia_margin_upper") == big"1e-4"
        @test get_optimizer_attribute(model, "phase1_hypatia_min_margin_upper") == big"1e-8"
        @test get_optimizer_attribute(model, "phase1_hypatia_margin_shrink") == big"0.2"
        @test get_optimizer_attribute(model, "phase1_hypatia_objective_bias") == big"1e-12"
        @test get_optimizer_attribute(model, "verbose") == false
        @test get_optimizer_attribute(model, "rational_tolerance") == big"1e-30"
        @test get_optimizer_attribute(model, "working_float_type") == BigFloat
    end

    @testset "Working float type selection" begin
        model = rational_model(Rational{BigInt})
        @test get_optimizer_attribute(model, "working_float_type") == RationalSDP.Double64
        @test get_optimizer_attribute(model, "phase1_backend") == :hypatia
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == RationalSDP.Double64
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == RationalSDP.Double64
        set_optimizer_attribute(model, "working_float_type", BigFloat)
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

    @testset "Reject approximate model coefficients" begin
        @test_throws ArgumentError RationalSDP.Optimizer{Float64}()
        @test_throws ArgumentError RationalSDP._exact_rational(0.1)
        @test_throws ArgumentError RationalSDP._exact_rational(pi)
    end

    @testset "Native Phase I backend override" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        @variable(model, X[1:1, 1:1], PSD)
        @constraint(model, X[1, 1] == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test value(X[1, 1]) == 1//1
    end

    @testset "Exact phase II segment refinement" begin
        problem = RationalSDP.ProblemData(
            [MOI.VariableIndex(1)],
            RationalSDP.BlockStructure[],
            [1],
            Rational{BigInt}[1//1],
            0//1,
            Rational{BigInt}[1//1],
            zeros(Rational{BigInt}, 0, 1),
            Rational{BigInt}[],
            nothing,
        )
        anchor = Rational{BigInt}[2//1]
        candidate = Rational{BigInt}[0//1]
        refined = RationalSDP._best_exact_interior_on_segment(
            anchor,
            candidate,
            problem;
            max_bisections = 8,
        )
        @test refined[1] > 0//1
        @test refined[1] < anchor[1]
        @test RationalSDP._exact_objective_value(problem, refined) <
              RationalSDP._exact_objective_value(problem, anchor)
    end

    @testset "PSD face pruning from forced nullspace directions" begin
        model = rational_model(Rational{BigInt})
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] == 0//1)
        @constraint(model, X[2, 2] == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        VX = value.(X)
        @test VX[1, 1] == 0//1
        @test VX[1, 2] == 0//1
        @test VX[2, 1] == 0//1
        @test VX[2, 2] == 1//1
        @test is_psd_exact(VX)
    end

    @testset "Facial reduction on a rational hidden PSD face" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] == X[1, 2])
        @constraint(model, X[2, 2] == X[1, 2])
        @constraint(model, X[1, 1] == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        VX = value.(X)
        @test VX == Rational{BigInt}[1//1 1//1; 1//1 1//1]
        @test is_psd_exact(VX)
    end

    @testset "Irrational exposed face detection" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        @variable(model, x)
        A = [
            2//1 - x 2//1 * x -1//1 - x;
            2//1 * x 2//1 + 2//1 * x 0//1;
            -1//1 - x 0//1 -1//1 - 2//1 * x
        ]
        B = [
            2//1 -1//1 - x x;
            -1//1 - x -2//1 * x 2//1;
            x 2//1 2//1 - 2//1 * x
        ]
        @constraint(model, Symmetric(A) in PSDCone())
        @constraint(model, Symmetric(B) in PSDCone())
        @objective(model, Min, 0//1)
        err = try
            optimize!(model)
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        @test occursin("could not be represented exactly", sprint(showerror, err))
    end

    @testset "Exact feasibility with Rational{BigInt}" begin
        model = rational_model(Rational{BigInt})
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] == 2//1)
        @constraint(model, X[2, 2] == 2//1)
        @constraint(model, X[1, 2] == 1//2)
        @objective(model, Min, 0//1)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test dual_status(model) == MOI.NO_SOLUTION
        V = value.(X)
        @test V[1, 1] == 2//1
        @test V[2, 2] == 2//1
        @test V[1, 2] == 1//2
        @test V[2, 1] == 1//2
        @test is_psd_exact(V)
    end

    @testset "Mixed PSD and scalar inequalities" begin
        model = rational_model(Rational{BigInt})
        @variable(model, X[1:2, 1:2], PSD)
        @variable(model, x)
        @variable(model, y)
        @constraint(model, X[1, 1] == 2//1)
        @constraint(model, X[2, 2] == 2//1)
        @constraint(model, X[1, 2] - x == 0//1)
        @constraint(model, x >= 1//2)
        @constraint(model, x <= 3//4)
        @constraint(model, y >= -1//3)
        @constraint(model, y <= 2//3)
        @constraint(model, x + y >= 1//3)
        @constraint(model, x - y <= 1//1)
        @objective(model, Min, y)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        vx = value(x)
        vy = value(y)
        VX = value.(X)
        @test VX[1, 2] == vx
        @test VX[2, 1] == vx
        @test vx >= 1//2
        @test vx <= 3//4
        @test vy >= -1//3
        @test vy <= 2//3
        @test vx + vy >= 1//3
        @test vx - vy <= 1//1
        @test vy < -(333//1000)
        @test is_psd_exact(VX)
    end

    @testset "LP scale with many interval constraints" begin
        model = rational_model(Rational{BigInt})
        n = 12
        upper_bounds = vcat(fill(1//4, 4), fill(1//1, n - 4))
        @variable(model, x[1:n])
        for i in 1:n
            @constraint(model, x[i] >= 0//1)
            @constraint(model, x[i] <= upper_bounds[i])
        end
        @constraint(model, sum(x) == 1//1)
        @objective(model, Min, sum((i // 1) * x[i] for i in 1:n))
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        values = value.(x)
        @test sum(values) == 1//1
        for i in 1:n
            @test values[i] >= 0//1
            @test values[i] <= upper_bounds[i]
        end
        @test objective_value(model) < 251//100
    end

    @testset "Larger mixed cone instance" begin
        model = rational_model(Rational{BigInt})
        @variable(model, X[1:3, 1:3], PSD)
        @variable(model, Y[1:2, 1:2], PSD)
        @variable(model, u[1:16])
        for i in eachindex(u)
            @constraint(model, u[i] >= 0//1)
            @constraint(model, u[i] <= 1//1)
        end
        @constraint(model, sum(u) == 5//1)
        @constraint(model, X[1, 1] == 4//1)
        @constraint(model, X[2, 2] == 3//1)
        @constraint(model, X[3, 3] == 2//1)
        @constraint(model, X[1, 2] == u[1])
        @constraint(model, X[1, 3] == u[2] - 1//4)
        @constraint(model, X[2, 3] == u[3] - 1//5)
        @constraint(model, Y[1, 1] == 5//2)
        @constraint(model, Y[2, 2] == 7//3)
        @constraint(model, Y[1, 2] == u[4] - u[5])
        @constraint(model, u[6] + u[7] >= 3//5)
        @constraint(model, u[8] + u[9] <= 7//5)
        @constraint(model, u[10] - u[11] >= -1//2)
        @constraint(model, u[12] + u[13] + u[14] >= 1//1)
        @constraint(model, u[15] + u[16] <= 3//2)
        @objective(model, Min, sum((i // 1) * u[i] for i in eachindex(u)) - Y[1, 2])
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        UX = value.(u)
        VX = value.(X)
        VY = value.(Y)
        @test sum(UX) == 5//1
        @test VX[1, 2] == UX[1]
        @test VX[1, 3] == UX[2] - 1//4
        @test VX[2, 3] == UX[3] - 1//5
        @test VY[1, 2] == UX[4] - UX[5]
        @test VY[2, 1] == UX[4] - UX[5]
        @test UX[6] + UX[7] >= 3//5
        @test UX[8] + UX[9] <= 7//5
        @test UX[10] - UX[11] >= -1//2
        @test UX[12] + UX[13] + UX[14] >= 1//1
        @test UX[15] + UX[16] <= 3//2
        @test is_psd_exact(VX)
        @test is_psd_exact(VY)
    end

    @testset "SOS-style polynomial lower bound via DynamicPolynomials" begin
        model = rational_model(Rational{BigInt})
        @polyvar z
        basis = monomials([z], 0:2)
        @variable(model, Q[1:3, 1:3], PSD)
        @variable(model, t)

        coeffs = Dict{Int,Any}(k => 0//1 for k in 0:4)
        for i in eachindex(basis)
            for j in i:length(basis)
                degree_ij = degree(basis[i] * basis[j], z)
                contribution = i == j ? Q[i, j] : 2//1 * Q[i, j]
                coeffs[degree_ij] = coeffs[degree_ij] + contribution
            end
        end

        @constraint(model, coeffs[0] == t)
        @constraint(model, coeffs[1] == 0//1)
        @constraint(model, coeffs[2] == -1//1)
        @constraint(model, coeffs[3] == 0//1)
        @constraint(model, coeffs[4] == 1//1)
        @objective(model, Min, t)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test value(Q[3, 3]) == 1//1
        @test value(Q[2, 3]) == 0//1
        @test value(t) < 251//1000
        @test is_psd_exact(value.(Q))
    end

    @testset "SOS Lyapunov feasibility with face pruning" begin
        model = rational_model(Rational{BigInt})
        @polyvar x[1:1]
        f = -x

        basis_V = monomials(x, 0:2)
        @variable(model, coeffs_V[1:length(basis_V)])
        V = dot(coeffs_V, basis_V)
        LV = dot(f, differentiate(V, x))

        basis_b1 = monomials(x, 0:1)
        basis_b2 = monomials(x, 0:1)

        @variable(model, Q1[1:length(basis_b1), 1:length(basis_b1)], PSD)
        p1 = -LV - basis_b1' * Q1 * basis_b1
        @constraint(model, coefficients(p1) .== 0)

        @variable(model, Q2[1:length(basis_b2), 1:length(basis_b2)], PSD)
        @constraint(model, coefficients(V - dot(x, x) - basis_b2' * Q2 * basis_b2) .== 0)

        @objective(model, Min, 0//1)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        VQ1 = value.(Q1)
        VQ2 = value.(Q2)
        @test VQ1[1, 1] == 0//1
        @test VQ1[1, 2] == 0//1
        @test VQ1[2, 1] == 0//1
        @test VQ1[2, 2] > 0//1
        @test VQ2[1, 1] > 0//1
        @test VQ2[2, 2] > 0//1
        @test is_psd_exact(VQ1)
        @test is_psd_exact(VQ2)
    end

    @testset "Lorenz SOS mean upper bound" begin
        model = rational_model(Rational{BigInt})
        @polyvar x[1:3]

        f = [
            10 * (x[2] - x[1]);
            28 * x[1] - x[1] * x[3] - x[2];
            x[1] * x[2] - 8//3 * x[3];
        ]

        basis_V = monomials(x, 0:2)
        @variable(model, coeffs_V[1:length(basis_V)])
        V = dot(coeffs_V, basis_V)
        LV = dot(f, differentiate(V, x))

        basis_b = monomials(x, 0:1)
        @variable(model, Q[1:length(basis_b), 1:length(basis_b)], PSD)
        @variable(model, B)
        @constraint(model, coefficients(B - x[3]^2 - LV - basis_b' * Q * basis_b) .== 0)
        @objective(model, Min, B)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test value(B) > 729//1
        @test value(B) < 730//1
        @test is_psd_exact(value.(Q))
    end

    @testset "Quartic Lorenz SOS mean upper bound" begin
        model = rational_model(Rational{BigInt})
        @polyvar x[1:3]

        f = [
            10 * (x[2] - x[1]);
            28 * x[1] - x[1] * x[3] - x[2];
            x[1] * x[2] - 8//3 * x[3];
        ]

        basis_V = monomials(x, 0:4)
        @variable(model, coeffs_V[1:length(basis_V)])
        V = dot(coeffs_V, basis_V)
        LV = dot(f, differentiate(V, x))

        basis_b = monomials(x, 0:2)
        @variable(model, Q[1:length(basis_b), 1:length(basis_b)], PSD)
        @variable(model, B)
        @constraint(model, coefficients(B - x[3]^2 - LV - basis_b' * Q * basis_b) .== 0)
        @objective(model, Min, B)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        @test value(B) > 728//1
        @test value(B) < 730//1
        @test is_psd_exact(value.(Q))
    end

    @testset "Alternative rational output type" begin
        model = rational_model(Rational{Int})
        @variable(model, X[1:1, 1:1], PSD)
        @variable(model, y)
        @constraint(model, X[1, 1] == 3//1)
        @constraint(model, y == 2//3)
        @objective(model, Min, 0//1)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test value(X[1, 1]) == 3//1
        @test value(y) == 2//3
    end
end

if "slow" in ARGS
    include("slowtests.jl")
end
