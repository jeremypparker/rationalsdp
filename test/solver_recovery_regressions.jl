using Test, JuMP, LinearAlgebra, RationalSDP

@testset "Solver recovery regressions" begin
    R = RationalSDP
    Rat = Rational{BigInt}
    MOI = JuMP.MOI

    @testset "An auxiliary margin does not override a strict interior point" begin
        model = GenericModel{Rat}(R.Optimizer{Rat})
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] == 1)
        @constraint(model, X[2, 2] == 1)
        MOI.Utilities.attach_optimizer(backend(model))
        opt = unsafe_backend(model)
        opt.settings.verbose = false
        problem = R._extract_problem(opt)
        candidate = Float64.(problem.affine[1])
        for (margin, status) in ((-4.44e-9, R.Hypatia.Solvers.SlowProgress),
                                 (1e-12, R.Hypatia.Solvers.Optimal))
            @test R._phase1_hypatia_margin_is_boundary(margin, status, 1e-8, 0.01)
            @test !R._phase1_hypatia_candidate_is_boundary(
                problem, candidate, margin, status, 1e-8, 0.01,
            )
        end
        anchor = R._phase1_exact_anchor_fallback(opt, candidate, problem, opt.settings, Float64)
        @test anchor !== nothing
        @test R._exact_primal_feasibility(problem, anchor).ok
        @test all(R._positive_definite_exact(R._vector_to_matrix(anchor, b)) for b in problem.blocks)
        boundary = zeros(length(candidate))
        @test R._phase1_hypatia_candidate_is_boundary(
            problem, boundary, 0.0, R.Hypatia.Solvers.Optimal, 1e-8, 0.01,
        )
    end

    @testset "Expected face recovery failures are solver statuses" begin
        opt = R.Optimizer{Rat}(verbose = false)
        for reason in (:inexact_face, :inconsistent_reduction)
            # Populate a previous result to ensure failure cannot leave stale values.
            opt.result_count = 1
            opt.primal_status = MOI.FEASIBLE_POINT
            opt.objective_value = Rat(7)
            opt.variable_primal[MOI.VariableIndex(1)] = Rat(7)
            err = R.FacialReductionRecoveryError(reason, "test face recovery failed")
            R._with_solver_failure_status(opt, time_ns()) do
                throw(err)
            end
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.NUMERICAL_ERROR
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION
            @test MOI.get(opt, MOI.ResultCount()) == 0
            @test opt.objective_value === nothing
            @test isempty(opt.variable_primal)
            @test occursin("test face recovery failed", MOI.get(opt, MOI.RawStatusString()))
            @test MOI.get(opt, MOI.SolveTimeSec()) >= 0
            @test R._is_inexact_facial_reduction_error(err) == (reason == :inexact_face)
        end
        @test_throws ArgumentError R._with_solver_failure_status(opt, time_ns()) do
            throw(ArgumentError("invalid input"))
        end
        @test_throws ErrorException R._with_solver_failure_status(opt, time_ns()) do
            error("unexpected programming error")
        end
        @test_throws InterruptException R._with_solver_failure_status(opt, time_ns()) do
            throw(InterruptException())
        end
    end
    @testset "Primal-dual recovery when the primal barrier has no center" begin
        # min x, x >= 0, y >= 0. The original optimum is 0, but
        # t*x - log(x) - log(y) is unbounded below as y grows.
        problem = R.ProblemData(
            MOI.VariableIndex[MOI.VariableIndex(1), MOI.VariableIndex(2)],
            R.BlockStructure[], [1, 2], Rat[1, 0], Rat(0), Rat[1, 0],
            zeros(Rat, 0, 2), Rat[], (zeros(Rat, 2), Matrix{Rat}(I, 2, 2)),
        )
        anchor = Rat[1, 1]
        opt = R.Optimizer{Rat}(verbose = false, max_iterations = 24,
                              phase2_outer_iterations = 10, phase2_hypatia_fallback = false)
        native = R._phase2_exact_solution(opt, problem, anchor, 2, R.Float64x2; subtitle = "test")
        @test native.termination_reason == :newton_iteration_limit
        @test isinf(native.gap_bound)
        @test native.x_exact[1] == 1
        opt.settings.phase2_hypatia_fallback = true
        recovered = R._phase2_exact_solution(opt, problem, anchor, 2, R.Float64x2; subtitle = "test")
        @test R._exact_primal_feasibility(problem, recovered.x_exact).ok
        @test 0 <= recovered.x_exact[1] < 1//10^10
        @test recovered.x_exact[1] < native.x_exact[1]
    end

    @testset "Fallback includes off-diagonal scaling, scalar cones, and affine offsets" begin
        model = GenericModel{Rat}(R.Optimizer{Rat})
        set_silent(model)
        @variable(model, X[1:2, 1:2], PSD)
        @variable(model, s >= 0)
        @constraint(model, X[1, 2] == 1)
        @constraint(model, s == 2)
        @objective(model, Min, X[1, 1] + X[2, 2] + s + 7)
        MOI.Utilities.attach_optimizer(backend(model))
        opt = unsafe_backend(model)
        problem = R._extract_problem(opt)
        a = R._phase1_anchor_attempt(opt, problem, R.Float64x2)
        @test a.anchor !== nothing
        basis = R._phase2_nullspace(problem, R.Float64x2)
        affine = R._numeric_affine_data(a.anchor, basis, R.Float64x2)
        N = R._numeric_nullspace!(affine)
        c = R._to_working_array(R.Float64x2, problem.objective_vector_min)
        result = R._phase2_hypatia_fallback(opt, problem, R._to_working_array(R.Float64x2, a.anchor), N, c)
        @test result !== nothing
        @test abs(Float64(dot(c, result.candidate)) - 4) < 1e-10
        exact = R._phase2_exact_refinement(result.candidate, a.anchor, problem, opt.settings, a.anchor, basis, affine)
        @test R._exact_primal_feasibility(problem, exact).ok
        @test 11 <= R._exact_objective_value(problem, exact) < 11 + 1//10^10
    end

    @testset "Exact refinement reuses its improved interior" begin
        problem = R.ProblemData(
            [MOI.VariableIndex(1)], R.BlockStructure[], [1], Rat[1], Rat(0), Rat[1],
            zeros(Rat, 0, 1), Rat[], (Rat[0], ones(Rat, 1, 1)),
        )
        anchor = Rat[2]
        opt = R.Optimizer{Rat}(verbose = false, exact_refinement_bisections = 8,
                              rational_tolerance = big"1e-12")
        affine = R._numeric_affine_data(Rat[0], ones(Rat, 1, 1), R.Float64x2)
        recovered = R._phase2_exact_refinement(
            R.Float64x2[0], anchor, problem, opt.settings, Rat[0], ones(Rat, 1, 1), affine,
        )
        @test R._exact_primal_feasibility(problem, recovered).ok
        @test 0 < recovered[1] < 1//10^10
        @test get_optimizer_attribute(GenericModel{Rat}(R.Optimizer{Rat}), "phase2_hypatia_fallback")
        # Coarse rounding of a tiny positive coordinate gives zero. Tighten
        # rounding before attempting an expensive segment interpolation.
        interior = R._phase2_exact_refinement(
            R.Float64x2[1e-10], anchor, problem, opt.settings, Rat[0], ones(Rat, 1, 1), affine,
        )
        @test interior[1] > 0
        @test abs(Float64(interior[1]) - 1e-10) <= 1e-12
        @test R._exact_primal_feasibility(problem, interior).ok
    end

    @testset "Inconsistent face returns a status through optimize!" begin
        # The exposing slack I has zero trace against every affine point,
        # but the resulting face X=0 contradicts X[1,1]=1. This reaches
        # the same face-application failure as the reported KSE exception.
        model = GenericModel{Rat}(R.Optimizer{Rat})
        set_silent(model)
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] == 1)
        @constraint(model, X[2, 2] == -1)
        optimize!(model)
        @test termination_status(model) == MOI.NUMERICAL_ERROR
        @test primal_status(model) == MOI.NO_SOLUTION
        @test result_count(model) == 0
        @test occursin("inconsistent affine restriction", raw_status(model))
        @test !occursin("could not be represented exactly", raw_status(model))
    end

    @testset "Phase II rounding keeps a common denominator and its tolerance" begin
        for F in (Float64, R.Float64x2, BigFloat)
            values = F[0, 1, -1.23456789, 1e-20, 1e12, sqrt(F(2))]
            tolerance = F(1e-12)
            rounded = R._phase2_rational_coefficients(values, tolerance)
            @test all(abs(BigFloat(values[i]) - BigFloat(rounded[i])) <= BigFloat(tolerance) for i in eachindex(values))
            common = foldl(lcm, denominator.(rounded); init = BigInt(1))
            @test ispow2(common)
            @test common <= BigInt(1) << 40
            @test rounded[1:2] == Rat[0, 1]
        end
    end

end
