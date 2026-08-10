include("testutils.jl")
include("slowtest_helpers.jl")
include("affine_column_ordering_tests.jl")

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
        set_optimizer_attribute(model, "phase1_hypatia_syssolver", "qrchol_dense")
        set_optimizer_attribute(model, "facial_reduction_oracle_syssolver", "symindef_dense")
        set_optimizer_attribute(model, "phase1_hypatia_target_margin", "0.02")
        set_optimizer_attribute(model, "phase1_hypatia_margin_upper", "1e-4")
        set_optimizer_attribute(model, "phase1_hypatia_min_margin_upper", "1e-8")
        set_optimizer_attribute(model, "phase1_hypatia_margin_shrink", "0.2")
        set_optimizer_attribute(model, "phase1_hypatia_boundary_margin_fraction", "0.005")
        set_optimizer_attribute(model, "phase1_hypatia_tol_rel_opt", "1e-8")
        set_optimizer_attribute(model, "phase1_hypatia_tol_abs_opt", "1e-9")
        set_optimizer_attribute(model, "phase1_hypatia_tol_feas", "1e-10")
        set_optimizer_attribute(model, "phase1_hypatia_default_tol_power", "0.75")
        set_optimizer_attribute(model, "phase1_hypatia_default_tol_relax", "0.9")
        set_optimizer_attribute(model, "phase1_hypatia_tol_slow", "1e-3")
        set_optimizer_attribute(model, "phase1_candidate_diagnostics", true)
        set_optimizer_attribute(model, "phase1_exact_recovery_diagnostics", true)
        set_optimizer_attribute(model, "phase1_exact_recovery_pivot_log_frequency", 4)
        set_optimizer_attribute(model, "verbose", false)
        set_optimizer_attribute(model, "rational_tolerance", "1e-30")
        set_optimizer_attribute(model, "recovery_tolerance_shrink", "0.01")
        set_optimizer_attribute(model, "working_float_type", "BigFloat")
        set_optimizer_attribute(model, "facial_reduction_save_file", "fr-cache.bin")
        set_optimizer_attribute(model, "facial_reduction_load_file", "fr-cache.bin")
        set_optimizer_attribute(model, "facial_reduction_row_space_max_entries", "1234")
        set_optimizer_attribute(
            model,
            "facial_reduction_weighted_subspace_max_affine_products",
            5678,
        )
        set_optimizer_attribute(model, "facial_reduction_weighted_max_candidate_sets", 6)
        set_optimizer_attribute(
            model,
            "facial_reduction_weighted_max_total_affine_products",
            123_456,
        )
        set_optimizer_attribute(
            model,
            "facial_reduction_weighted_exact_without_scout_limit",
            1,
        )
        set_optimizer_attribute(
            model,
            "facial_reduction_sparse_affine_validation_max_products",
            9012,
        )
        set_optimizer_attribute(model, "facial_reduction_sieve_transform_max_entries", "3456")
        set_optimizer_attribute(model, "facial_reduction_affine_compaction_factor", 7)
        set_optimizer_attribute(model, "facial_reduction_affine_compaction_max_entries", 8901)
        set_optimizer_attribute(model, "facial_reduction_subspace_max_charts", 5)
        set_optimizer_attribute(model, "facial_reduction_projector_recovery", false)
        set_optimizer_attribute(
            model,
            "facial_reduction_oracle_precision_escalation_max_retries",
            3,
        )
        set_optimizer_attribute(model, "facial_reduction_precision_escalation_max_retries", 2)
        set_optimizer_attribute(model, "facial_reduction_tentative_max_directions", 3)
        set_optimizer_attribute(model, "facial_reduction_affine_lift_chunk_columns", 7)
        @test get_optimizer_attribute(model, "phase1_outer_iterations") == 24
        @test get_optimizer_attribute(model, "phase1_backend") == :native
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        @test get_optimizer_attribute(model, "phase1_hypatia_syssolver") == :qrchol_dense
        @test get_optimizer_attribute(model, "facial_reduction_oracle_syssolver") ==
              :symindef_dense
        @test get_optimizer_attribute(model, "phase1_hypatia_target_margin") == big"0.02"
        @test get_optimizer_attribute(model, "phase1_hypatia_margin_upper") == big"1e-4"
        @test get_optimizer_attribute(model, "phase1_hypatia_min_margin_upper") == big"1e-8"
        @test get_optimizer_attribute(model, "phase1_hypatia_margin_shrink") == big"0.2"
        @test get_optimizer_attribute(model, "phase1_hypatia_boundary_margin_fraction") ==
              big"0.005"
        @test get_optimizer_attribute(model, "phase1_hypatia_tol_rel_opt") == big"1e-8"
        @test get_optimizer_attribute(model, "phase1_hypatia_tol_abs_opt") == big"1e-9"
        @test get_optimizer_attribute(model, "phase1_hypatia_tol_feas") == big"1e-10"
        @test get_optimizer_attribute(model, "phase1_hypatia_default_tol_power") == big"0.75"
        @test get_optimizer_attribute(model, "phase1_hypatia_default_tol_relax") == big"0.9"
        @test get_optimizer_attribute(model, "phase1_hypatia_tol_slow") == big"1e-3"
        @test get_optimizer_attribute(model, "phase1_candidate_diagnostics")
        @test get_optimizer_attribute(model, "phase1_exact_recovery_diagnostics")
        @test get_optimizer_attribute(model, "phase1_exact_recovery_pivot_log_frequency") == 4
        @test get_optimizer_attribute(model, "verbose") == false
        @test get_optimizer_attribute(model, "rational_tolerance") == big"1e-30"
        @test get_optimizer_attribute(model, "recovery_tolerance_shrink") == big"0.01"
        @test get_optimizer_attribute(model, "working_float_type") == BigFloat
        @test get_optimizer_attribute(model, "facial_reduction_save_file") == "fr-cache.bin"
        @test get_optimizer_attribute(model, "facial_reduction_load_file") == "fr-cache.bin"
        @test get_optimizer_attribute(model, "facial_reduction_rank_expansion_rounds") == 0
        @test get_optimizer_attribute(model, "facial_reduction_row_space_max_entries") == 1234
        @test get_optimizer_attribute(
            model,
            "facial_reduction_weighted_subspace_max_affine_products",
        ) == 5678
        @test get_optimizer_attribute(
            model,
            "facial_reduction_weighted_max_candidate_sets",
        ) == 6
        @test get_optimizer_attribute(
            model,
            "facial_reduction_weighted_max_total_affine_products",
        ) == 123_456
        @test get_optimizer_attribute(
            model,
            "facial_reduction_weighted_exact_without_scout_limit",
        ) == 1
        @test get_optimizer_attribute(
            model,
            "facial_reduction_sparse_affine_validation_max_products",
        ) == 9012
        @test get_optimizer_attribute(model, "facial_reduction_sieve_transform_max_entries") ==
              3456
        @test get_optimizer_attribute(model, "facial_reduction_affine_compaction_factor") == 7
        @test get_optimizer_attribute(model, "facial_reduction_affine_compaction_max_entries") ==
              8901
        @test get_optimizer_attribute(model, "facial_reduction_subspace_max_charts") == 5
        @test !get_optimizer_attribute(model, "facial_reduction_projector_recovery")
        @test get_optimizer_attribute(
            model,
            "facial_reduction_oracle_precision_escalation_max_retries",
        ) == 3
        @test get_optimizer_attribute(
            model,
            "facial_reduction_precision_escalation_max_retries",
        ) == 2
        @test get_optimizer_attribute(model, "facial_reduction_tentative_max_directions") == 3
        @test get_optimizer_attribute(model, "facial_reduction_affine_lift_chunk_columns") == 7
    end

    @testset "Working float type selection" begin
        model = rational_model(Rational{BigInt})
        @test get_optimizer_attribute(model, "working_float_type") == RationalSDP.Float64x2
        @test get_optimizer_attribute(model, "phase1_backend") == :hypatia
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == RationalSDP.Float64x2
        @test get_optimizer_attribute(model, "phase1_hypatia_syssolver") == :auto
        @test get_optimizer_attribute(model, "facial_reduction_oracle_syssolver") == :auto
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == RationalSDP.Float64x2

        set_optimizer_attribute(model, "working_float_type", Float64)
        @test get_optimizer_attribute(model, "working_float_type") == Float64
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == Float64

        set_optimizer_attribute(model, "phase1_hypatia_float_type", "Float64")
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        set_optimizer_attribute(model, "phase1_hypatia_float_type", "auto")
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        set_optimizer_attribute(model, "facial_reduction_float_type", "Float64")
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == Float64
        set_optimizer_attribute(model, "facial_reduction_float_type", "auto")
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == Float64

        set_optimizer_attribute(model, "working_float_type", RationalSDP.Float64x2)
        @test get_optimizer_attribute(model, "working_float_type") == RationalSDP.Float64x2
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == RationalSDP.Float64x2
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == RationalSDP.Float64x2

        set_optimizer_attribute(model, "working_float_type", "Float64x2")
        @test get_optimizer_attribute(model, "working_float_type") == RationalSDP.Float64x2
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == RationalSDP.Float64x2
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == RationalSDP.Float64x2

        set_optimizer_attribute(model, "phase1_hypatia_float_type", "MultiFloats.Float64x3")
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") ==
              RationalSDP.Float64x3
        set_optimizer_attribute(model, "facial_reduction_float_type", RationalSDP.Float64x4)
        @test get_optimizer_attribute(model, "facial_reduction_float_type") ==
              RationalSDP.Float64x4
        @test RationalSDP._to_working_float(RationalSDP.Float64x2, 1//3) isa
              RationalSDP.Float64x2
        huge_scale = big(10)^1000
        moderate_huge_rational = (3 * huge_scale + 1) // huge_scale
        converted_huge_rational = RationalSDP._to_working_float(
            RationalSDP.Float64x2,
            moderate_huge_rational,
        )
        @test isfinite(converted_huge_rational)
        @test isapprox(converted_huge_rational, RationalSDP.Float64x2(3))
        @test RationalSDP._rationalize_float(
            RationalSDP.Float64x2(1) / RationalSDP.Float64x2(3),
            RationalSDP.Float64x2(1e-8),
        ) == 1 // 3

        large_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            RationalSDP.BlockStructure[],
            collect(1:512),
            zeros(Rational{BigInt}, 512),
            0 // 1,
            zeros(Rational{BigInt}, 512),
            zeros(Rational{BigInt}, 0, 512),
            Rational{BigInt}[],
            nothing,
        )
        default_type_opt =
            RationalSDP.Optimizer{Rational{BigInt}}(verbose = false)
        @test RationalSDP._phase1_hypatia_effective_float_type(
            default_type_opt,
            large_problem,
        ) == RationalSDP.Float64x2
        @test RationalSDP._facial_reduction_oracle_float_type(
            default_type_opt,
            large_problem,
        ) == RationalSDP.Float64x2

        big_type_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            working_float_type = BigFloat,
        )
        @test RationalSDP._phase1_hypatia_effective_float_type(
            big_type_opt,
            large_problem,
        ) == BigFloat
        @test RationalSDP._facial_reduction_oracle_float_type(
            big_type_opt,
            large_problem,
        ) == BigFloat
    end

    @testset "Hypatia Phase I system solver selection" begin
        @test RationalSDP._hypatia_solution_is_usable(Float64[0.0, 1.0], 2)
        @test !RationalSDP._hypatia_solution_is_usable(Float64[NaN, 1.0], 2)
        @test !RationalSDP._hypatia_solution_is_usable(Float64[0.0, Inf], 2)
        @test !RationalSDP._hypatia_solution_is_usable(Float64[0.0], 2)
        @test RationalSDP._phase1_status_is_infeasible(
            string(RationalSDP.Hypatia.Solvers.PrimalInfeasible),
        )
        @test RationalSDP._phase1_status_is_infeasible(
            string(RationalSDP.Hypatia.Solvers.NearPrimalInfeasible),
        )
        @test !RationalSDP._phase1_status_is_infeasible(nothing)

        margin_goal = 1.0e-8
        boundary_fraction = 0.01
        @test RationalSDP._phase1_hypatia_margin_is_boundary(
            -1.0e-14,
            RationalSDP.Hypatia.Solvers.SlowProgress,
            margin_goal,
            boundary_fraction,
        )
        @test RationalSDP._phase1_hypatia_margin_is_boundary(
            5.358e-13,
            RationalSDP.Hypatia.Solvers.Optimal,
            margin_goal,
            boundary_fraction,
        )
        @test !RationalSDP._phase1_hypatia_margin_is_boundary(
            2.0e-10,
            RationalSDP.Hypatia.Solvers.Optimal,
            margin_goal,
            boundary_fraction,
        )
        @test !RationalSDP._phase1_hypatia_margin_is_boundary(
            5.358e-13,
            RationalSDP.Hypatia.Solvers.SlowProgress,
            margin_goal,
            boundary_fraction,
        )
        @test !RationalSDP._phase1_hypatia_margin_is_boundary(
            5.358e-13,
            RationalSDP.Hypatia.Solvers.Optimal,
            margin_goal,
            0.0,
        )
        @test RationalSDP._phase1_hypatia_boundary_margin_fraction(
            RationalSDP.Settings(phase1_hypatia_boundary_margin_fraction = big"0.5"),
        ) == big"0.5"
        @test_throws ErrorException RationalSDP._phase1_hypatia_boundary_margin_fraction(
            RationalSDP.Settings(phase1_hypatia_boundary_margin_fraction = big"-0.1"),
        )
        @test_throws ErrorException RationalSDP._phase1_hypatia_boundary_margin_fraction(
            RationalSDP.Settings(phase1_hypatia_boundary_margin_fraction = big"1.1"),
        )

        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(RationalSDP.Settings(), Float64)
        @test syssolver isa RationalSDP.Hypatia.Solvers.SymIndefSparseSystemSolver{Float64}
        @test !use_dense_model
        @test !preprocess

        settings = RationalSDP.Settings(phase1_hypatia_syssolver = :qrchol_dense)
        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(settings, RationalSDP.Float64x2)
        @test syssolver isa RationalSDP.Hypatia.Solvers.QRCholDenseSystemSolver{RationalSDP.Float64x2}
        @test use_dense_model
        @test preprocess

        settings = RationalSDP.Settings(phase1_hypatia_syssolver = :symindef_indirect)
        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(settings, RationalSDP.Float64x2)
        @test syssolver isa RationalSDP.Hypatia.Solvers.SymIndefIndirectSystemSolver{RationalSDP.Float64x2}
        @test !use_dense_model
        @test !preprocess

        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(RationalSDP.Settings(), RationalSDP.Float64x2)
        @test syssolver isa
              RationalSDP.Hypatia.Solvers.SymIndefDenseSystemSolver{RationalSDP.Float64x2}
        @test use_dense_model
        @test !preprocess

        settings = RationalSDP.Settings(phase1_hypatia_syssolver = :symindef_sparse)
        @test_throws ErrorException RationalSDP._hypatia_phase1_syssolver(
            settings,
            RationalSDP.Float64x2,
        )

        settings = RationalSDP.Settings(
            phase1_hypatia_tol_rel_opt = big"1e-8",
            phase1_hypatia_tol_abs_opt = big"1e-9",
            phase1_hypatia_tol_feas = big"1e-10",
            phase1_hypatia_default_tol_power = big"0.75",
            phase1_hypatia_default_tol_relax = big"0.9",
            phase1_hypatia_tol_slow = big"1e-3",
        )
        tolerance_kwargs = RationalSDP._phase1_hypatia_tolerance_kwargs(settings, Float64)
        @test (:tol_rel_opt => 1.0e-8) in tolerance_kwargs
        @test (:tol_abs_opt => 1.0e-9) in tolerance_kwargs
        @test (:tol_feas => 1.0e-10) in tolerance_kwargs
        @test (:default_tol_power => 0.75) in tolerance_kwargs
        @test (:default_tol_relax => 0.9) in tolerance_kwargs
        @test (:tol_slow => 1.0e-3) in tolerance_kwargs
        solver = RationalSDP.Hypatia.Solvers.Solver{Float64}(
            ;
            verbose = false,
            iter_limit = 1,
            tolerance_kwargs...,
        )
        @test solver isa RationalSDP.Hypatia.Solvers.Solver{Float64}

        settings = RationalSDP.Settings(phase1_hypatia_tol_slow = big"-1")
        @test isempty(RationalSDP._phase1_hypatia_tolerance_kwargs(settings, Float64))

        @test RationalSDP._facial_reduction_oracle_allows_candidate_status(
            RationalSDP.Hypatia.Solvers.Optimal,
        )
        @test RationalSDP._facial_reduction_oracle_allows_candidate_status(
            RationalSDP.Hypatia.Solvers.NearOptimal,
        )
        @test RationalSDP._facial_reduction_oracle_allows_candidate_status(
            RationalSDP.Hypatia.Solvers.SlowProgress,
        )
        @test !RationalSDP._facial_reduction_oracle_allows_candidate_status(
            RationalSDP.Hypatia.Solvers.PrimalInfeasible,
        )
        @test RationalSDP._facial_reduction_oracle_recommends_precision_retry(
            RationalSDP.Hypatia.Solvers.NearPrimalInfeasible,
        )
        @test !RationalSDP._facial_reduction_oracle_recommends_precision_retry(
            RationalSDP.Hypatia.Solvers.PrimalInfeasible,
        )
    end

    @testset "MultiFloat dense matmul" begin
        T = RationalSDP.Float64x2
        A = T[1 2 3; 4 5 6]
        B = T[2 1; 0 3; 4 5]
        C = fill(T(7), 2, 2)
        C0 = copy(C)

        LinearAlgebra.mul!(C, A, B, T(2), T(-1))
        expected = Matrix{T}(undef, 2, 2)
        for j in axes(expected, 2), i in axes(expected, 1)
            value = zero(T)
            for k in axes(A, 2)
                value += A[i, k] * B[k, j]
            end
            expected[i, j] = T(2) * value - C0[i, j]
        end
        @test C == expected

        Cview_parent = fill(T(-1), 4, 4)
        Cview = @view Cview_parent[1:3, 1:3]
        LinearAlgebra.mul!(Cview, A', A, true, false)
        expected_gram = Matrix{T}(undef, 3, 3)
        for j in axes(expected_gram, 2), i in axes(expected_gram, 1)
            value = zero(T)
            for k in axes(A, 1)
                value += A[k, i] * A[k, j]
            end
            expected_gram[i, j] = value
        end
        @test Cview == expected_gram

        n = 128
        A_large = fill(T(1.25), n, n)
        B_large = fill(T(-0.75), n, n)
        C_large = fill(T(3), n, n)
        C_large_original = copy(C_large)
        LinearAlgebra.mul!(C_large, A_large, B_large, T(2), T(-1))
        expected_large = fill(T(2) * T(n) * T(1.25) * T(-0.75), n, n)
        expected_large .-= C_large_original
        error_large = maximum(abs, C_large - expected_large)
        scale_large = max(one(T), maximum(abs, expected_large))
        @test error_large <= T(100) * eps(T) * scale_large

        if Threads.nthreads() > 1
            # The internal threaded loops must remain usable when mul! itself
            # is called from an already-threaded region.
            n_nested = 32
            A_nested = fill(T(1), n_nested, n_nested)
            B_nested = fill(T(2), n_nested, n_nested)
            results = Vector{Tuple{T,T}}(undef, min(Threads.nthreads(), 4))
            Threads.@threads :dynamic for index in eachindex(results)
                C_nested = zeros(T, n_nested, n_nested)
                LinearAlgebra.mul!(C_nested, A_nested, B_nested, true, false)
                C_scaled = fill(T(3), n_nested, n_nested)
                LinearAlgebra.mul!(C_scaled, A_nested, B_nested, false, T(2))
                results[index] = (C_nested[1, 1], C_scaled[1, 1])
            end
            @test all(result == (T(2 * n_nested), T(6)) for result in results)
        end
    end

    @testset "Hypatia centering warning filter" begin
        output = IOBuffer()
        Logging.with_logger(Logging.ConsoleLogger(output, Logging.Warn)) do
            RationalSDP._with_filtered_hypatia_logger() do
                Logging.handle_message(
                    Logging.current_logger(),
                    Logging.Warn,
                    "cannot step in centering direction",
                    RationalSDP.Hypatia.Solvers,
                    :test,
                    :hypatia_centering_warning,
                    "combined.jl",
                    111,
                )
                Logging.handle_message(
                    Logging.current_logger(),
                    Logging.Warn,
                    "different Hypatia warning",
                    RationalSDP.Hypatia.Solvers,
                    :test,
                    :hypatia_other_warning,
                    "combined.jl",
                    112,
                )
                Logging.handle_message(
                    Logging.current_logger(),
                    Logging.Warn,
                    "cannot step in centering direction",
                    RationalSDP,
                    :test,
                    :rationalsdp_same_text_warning,
                    "core.jl",
                    1,
                )
            end
        end
        text = String(take!(output))
        @test length(collect(eachmatch(r"cannot step in centering direction", text))) == 1
        @test occursin("different Hypatia warning", text)
        @test occursin("RationalSDP", text)
    end

    @testset "Reject approximate model coefficients" begin
        @test_throws ArgumentError RationalSDP.Optimizer{Float64}()
        @test_throws ArgumentError RationalSDP._exact_rational(0.1)
        @test_throws ArgumentError RationalSDP._exact_rational(pi)
    end

    @testset "Invalid numeric settings fail before solving" begin
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(inner_log_frequency = 0),
        )
        @test_throws ErrorException RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_oracle_syssolver = :invalid),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(line_search_shrink = big"1.0"),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(phase2_outer_iterations = 0),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_row_space_max_entries = -1),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_affine_compaction_factor = -1),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_affine_compaction_max_entries = -1),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_subspace_max_charts = 0),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_precision_escalation_max_retries = -1),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(
                facial_reduction_oracle_precision_escalation_max_retries = -1,
            ),
        )
        @test_throws ArgumentError RationalSDP._validate_settings(
            RationalSDP.Settings(facial_reduction_affine_lift_chunk_columns = 0),
        )
        @test RationalSDP._validate_settings(RationalSDP.Settings()) === nothing
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

    @testset "Scalar max objective over an interval" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase1_backend", :native)
        set_optimizer_attribute(model, "working_float_type", Float64)
        @variable(model, x)
        @constraint(model, x >= 0//1)
        @constraint(model, x <= 1//1)
        @objective(model, Max, x)
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test value(x) <= 1//1
        @test value(x) > 999//1000
        @test objective_value(model) == value(x)
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

    @testset "Phase II iteration limit is not reported as optimal" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "phase2_outer_iterations", 1)
        set_optimizer_attribute(model, "optimality_gap_tolerance", "1e-30")
        @variable(model, x >= 0//1)
        @objective(model, Min, x)

        optimize!(model)

        @test termination_status(model) == MOI.ITERATION_LIMIT
        test_facial_reduction_statistics(model)
        @test primal_status(model) == MOI.FEASIBLE_POINT
        @test result_count(model) == 1
        @test value(x) > 0//1
        @test MOI.get(backend(model), MOI.RawStatusString()) ==
              "Phase II outer iteration limit reached"
    end

    @testset "Phase I exact recovery fallback from candidate point" begin
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[MOI.VariableIndex(1), MOI.VariableIndex(2)],
            RationalSDP.BlockStructure[],
            [1],
            Rational{BigInt}[0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1],
            zeros(Rational{BigInt}, 0, 2),
            Rational{BigInt}[],
            (
                Rational{BigInt}[0//1, 0//1],
                Rational{BigInt}[1//1 0//1; 0//1 1//1],
            ),
        )
        settings = RationalSDP.Settings(rational_tolerance = big"1e-12")
        phase1_particular = Rational{BigInt}[0//1, 0//1]
        phase1_nullspace = Rational{BigInt}[1000000000000//1; 0//1]
        coordinates = Float64[5.0e-13]
        candidate = Float64[0.5, 0.0]

        direct = RationalSDP._phase1_exact_feasible_point_from_coordinates(
            RationalSDP.Optimizer{Rational{BigInt}}(verbose = false),
            coordinates,
            phase1_particular,
            reshape(phase1_nullspace, :, 1),
            problem,
            settings,
        )
        @test direct === nothing

        recovered = RationalSDP._phase1_exact_anchor_fallback(
            RationalSDP.Optimizer{Rational{BigInt}}(verbose = false),
            candidate,
            problem,
            settings,
            Float64,
        )
        @test recovered !== nothing
        @test recovered[1] == 1//2
        @test recovered[2] == 0//1
    end

    @testset "Nemo-backed exact affine elimination" begin
        A = Rational{BigInt}[
            0//1 0//1 1//1
            1//1 0//1 1//1
        ]
        b = Rational{BigInt}[3//1, 5//1]
        solver_checkpoints = String[]
        affine = RationalSDP._solve_affine_system(
            A,
            b;
            checkpoint = stage -> push!(solver_checkpoints, stage),
        )
        @test affine !== nothing
        @test any(stage -> occursin("building dense", stage), solver_checkpoints)
        @test any(stage -> occursin("computing exact RREF", stage), solver_checkpoints)
        @test any(stage -> occursin("interpreting exact RREF", stage), solver_checkpoints)
        particular, nullspace = affine
        @test A * particular == b
        @test A * nullspace == zeros(Rational{BigInt}, size(A, 1), size(nullspace, 2))
        @test size(nullspace, 2) == 1

        inconsistent_A = Rational{BigInt}[
            1//1 1//1
            2//1 2//1
        ]
        inconsistent_b = Rational{BigInt}[1//1, 3//1]
        @test RationalSDP._solve_affine_system(inconsistent_A, inconsistent_b) === nothing

        redundant_A = Rational{BigInt}[
            1//1 0//1
            2//1 0//1
            0//1 1//1
            0//1 0//1
        ]
        redundant_b = Rational{BigInt}[1//1, 2//1, 3//1, 0//1]
        independent = RationalSDP._independent_affine_equalities(redundant_A, redundant_b)
        @test independent !== nothing
        independent_A, independent_b = independent
        @test independent_A == Rational{BigInt}[1//1 0//1; 0//1 1//1]
        @test independent_b == Rational{BigInt}[1//1, 3//1]
        @test RationalSDP._independent_affine_equalities(inconsistent_A, inconsistent_b) === nothing

        backend_error = RationalSDP.Nemo.FlintException(
            RationalSDP.Nemo.FLINT_ERROR,
            "unable to allocate exact matrix",
        )
        wrapped_error = RationalSDP._exact_linear_algebra_error("test RREF", backend_error)
        @test RationalSDP._is_nemo_flint_exception(backend_error)
        @test !RationalSDP._is_nemo_flint_exception(ErrorException("not FLINT"))
        caught_backend_error = try
            RationalSDP._with_nemo_error("test wrapped operation") do
                throw(backend_error)
            end
            nothing
        catch err
            err
        end
        @test caught_backend_error isa RationalSDP.ExactLinearAlgebraError
        @test occursin("test wrapped operation", sprint(showerror, caught_backend_error))
        wrapped_message = sprint(showerror, wrapped_error)
        @test occursin("test RREF", wrapped_message)
        @test occursin("unable to allocate exact matrix", wrapped_message)
        @test !occursin("namemap", wrapped_message)

        augmented = Rational{BigInt}[
            1//2 1//3 5//6
            1//1 2//3 5//3
        ]
        reduced, pivots = RationalSDP._rref(augmented)
        @test reduced == Rational{BigInt}[1//1 2//3 5//3; 0//1 0//1 0//1]
        @test pivots == [1]

        nullspace_matrix = Rational{BigInt}[
            1//2 1//3 1//4
            0//1 2//3 1//5
        ]
        exact_nullspace = RationalSDP._nullspace_basis_exact(nullspace_matrix)
        @test size(exact_nullspace) == (3, 1)
        @test nullspace_matrix * exact_nullspace == zeros(Rational{BigInt}, 2, 1)

        empty_A = zeros(Rational{BigInt}, 0, 3)
        empty_affine = RationalSDP._solve_affine_system(empty_A, Rational{BigInt}[])
        @test empty_affine !== nothing
        @test empty_affine[1] == zeros(Rational{BigInt}, 3)
        @test empty_affine[2] == Matrix{Rational{BigInt}}(I, 3, 3)
        @test RationalSDP._nullspace_basis_exact(empty_A) ==
              Matrix{Rational{BigInt}}(I, 3, 3)

        no_variables = zeros(Rational{BigInt}, 1, 0)
        @test RationalSDP._solve_affine_system(no_variables, Rational{BigInt}[0//1]) !== nothing
        @test RationalSDP._solve_affine_system(no_variables, Rational{BigInt}[1//1]) === nothing
    end

    @testset "Exact recovery tolerance controls" begin
        settings = RationalSDP.Settings(
            rational_tolerance = big"1e-12",
            recovery_tolerance_shrink = big"0.01",
        )
        tolerances = RationalSDP._recovery_tolerances(settings, Float64)
        @test isapprox(tolerances[1], 1.0e-6)
        @test isapprox(tolerances[2], 1.0e-8)
        @test isapprox(tolerances[end], 1.0e-12)
        @test_throws ErrorException RationalSDP._recovery_tolerances(
            RationalSDP.Settings(recovery_tolerance_shrink = big"1.0"),
            Float64,
        )
    end

    @testset "Exact positive-definite checks reject bad diagonals" begin
        @test RationalSDP._positive_definite_exact(Rational{BigInt}[2//1 1//1; 1//1 2//1])
        @test !RationalSDP._positive_definite_exact(Rational{BigInt}[0//1 0//1; 0//1 1//1])
        @test !RationalSDP._positive_definite_exact(Rational{BigInt}[1//1 2//1; 2//1 1//1])
    end

    @testset "Early coordinate PSD face reduction" begin
        BR = Rational{BigInt}

        triangle_index(i, j) = i >= j ? div(i * (i - 1), 2) + j : div(j * (j - 1), 2) + i

        function direct_psd_blocks_optimizer(dims::Vector{Int})
            opt = RationalSDP.Optimizer{BR}(verbose = false)
            block_variables = Vector{MOI.VariableIndex}[]
            for dim in dims
                variables = [MOI.add_variable(opt) for _ in 1:div(dim * (dim + 1), 2)]
                MOI.add_constraint(
                    opt,
                    MOI.VectorOfVariables(variables),
                    MOI.PositiveSemidefiniteConeTriangle(dim),
                )
                push!(block_variables, variables)
            end
            MOI.set(opt, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
            return opt, block_variables
        end

        function direct_psd_optimizer(dim::Int)
            opt, block_variables = direct_psd_blocks_optimizer([dim])
            variables = only(block_variables)
            return opt, variables
        end

        function add_zero_variable_equality!(opt, variable)
            MOI.add_constraint(opt, variable, MOI.EqualTo{BR}(zero(BR)))
            return
        end

        function add_affine_equality!(opt, pairs, rhs)
            terms = MOI.ScalarAffineTerm{BR}[
                MOI.ScalarAffineTerm{BR}(coefficient, variable) for
                (coefficient, variable) in pairs
            ]
            MOI.add_constraint(
                opt,
                MOI.ScalarAffineFunction{BR}(terms, zero(BR)),
                MOI.EqualTo{BR}(rhs),
            )
            return
        end

        function has_coordinate_zero_row(A, b, index)
            for row in axes(A, 1)
                iszero(b[row]) || continue
                A[row, index] == one(BR) || continue
                if all(column -> column == index || iszero(A[row, column]), axes(A, 2))
                    return true
                end
            end
            return false
        end

        function fixed_zero(problem, index)
            problem.affine === nothing && return false
            particular, nullspace = problem.affine
            return RationalSDP._variable_fixed_zero(particular, nullspace, index)
        end

        @testset "affine coordinate-zero closure cascades" begin
            A = BR[
                1//1 0//1 0//1
                1//1 2//1 0//1
                0//1 1//1 -3//1
            ]
            b = zeros(BR, 3)

            blocks, A_reduced, b_reduced, pruned =
                RationalSDP._early_prune_psd_coordinate_faces(RationalSDP.BlockStructure[], A, b)

            @test isempty(blocks)
            @test pruned == 0
            @test all(index -> has_coordinate_zero_row(A_reduced, b_reduced, index), 1:3)
        end

        @testset "two-survivor zero row is not inferred" begin
            A = BR[1//1 1//1]
            b = BR[0//1]

            blocks, A_reduced, b_reduced, pruned =
                RationalSDP._early_prune_psd_coordinate_faces(RationalSDP.BlockStructure[], A, b)

            @test isempty(blocks)
            @test pruned == 0
            @test A_reduced == A
            @test b_reduced == b
        end

        @testset "helper restricts blocks before affine solve" begin
            block = RationalSDP.BlockStructure(
                3,
                Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
                collect(1:6),
                RationalSDP._triangle_positions(3),
                [1, 3, 6],
            )
            A = zeros(BR, 1, 6)
            A[1, 3] = one(BR)
            b = BR[zero(BR)]

            blocks, A_reduced, b_reduced, pruned =
                RationalSDP._early_prune_psd_coordinate_faces([block], A, b)

            @test pruned == 1
            @test length(blocks) == 1
            @test blocks[1].size == 2
            @test blocks[1].global_positions == [1, 4, 6]
            @test all(index -> has_coordinate_zero_row(A_reduced, b_reduced, index), [2, 3, 5])
        end

        @testset "PSD row-column zeros trigger diagonal cascade" begin
            block = RationalSDP.BlockStructure(
                3,
                Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
                collect(1:6),
                RationalSDP._triangle_positions(3),
                [1, 3, 6],
            )
            A = zeros(BR, 2, 6)
            A[1, 1] = one(BR)
            A[2, 2] = one(BR)
            A[2, 3] = 2 // 1
            b = zeros(BR, 2)

            blocks, A_reduced, b_reduced, pruned =
                RationalSDP._early_prune_psd_coordinate_faces([block], A, b)

            @test pruned == 2
            @test length(blocks) == 1
            @test blocks[1].size == 1
            @test blocks[1].global_positions == [6]
            @test all(index -> has_coordinate_zero_row(A_reduced, b_reduced, index), 1:5)
        end

        @testset "singleton diagonal removes its PSD row and column" begin
            opt, variables = direct_psd_optimizer(3)
            add_zero_variable_equality!(opt, variables[triangle_index(2, 2)])

            problem = RationalSDP._extract_problem(opt)

            @test length(problem.blocks) == 1
            @test problem.blocks[1].size == 2
            @test problem.blocks[1].global_positions == [1, 4, 6]
            @test all(index -> fixed_zero(problem, index), [2, 3, 5])
            @test all(index -> has_coordinate_zero_row(problem.A, problem.b, index), [2, 3, 5])
        end

        @testset "two singleton diagonals reduce a larger PSD block" begin
            opt, variables = direct_psd_optimizer(4)
            add_zero_variable_equality!(opt, variables[triangle_index(2, 2)])
            add_zero_variable_equality!(opt, variables[triangle_index(4, 4)])

            problem = RationalSDP._extract_problem(opt)

            @test length(problem.blocks) == 1
            @test problem.blocks[1].size == 2
            @test problem.blocks[1].global_positions == [1, 4, 6]
            @test all(index -> fixed_zero(problem, index), [2, 3, 5, 7, 8, 9, 10])
            particular, nullspace = problem.affine
            @test problem.A * particular == problem.b
            @test problem.A * nullspace ==
                  zeros(BR, size(problem.A, 1), size(nullspace, 2))
        end

        @testset "multi-block coordinate cascade regression" begin
            opt, block_variables = direct_psd_blocks_optimizer([4, 4])
            p = block_variables[1]
            q = block_variables[2]

            add_zero_variable_equality!(opt, p[triangle_index(1, 1)])
            add_affine_equality!(
                opt,
                [(one(BR), p[triangle_index(2, 1)]), (2 // 1, p[triangle_index(2, 2)])],
                zero(BR),
            )
            add_affine_equality!(
                opt,
                [(one(BR), p[triangle_index(3, 2)]), (-3 // 1, p[triangle_index(3, 3)])],
                zero(BR),
            )

            add_zero_variable_equality!(opt, q[triangle_index(4, 4)])
            add_affine_equality!(
                opt,
                [(one(BR), q[triangle_index(4, 3)]), (-3 // 1, q[triangle_index(3, 3)])],
                zero(BR),
            )
            add_affine_equality!(
                opt,
                [(one(BR), q[triangle_index(3, 2)]), (2 // 1, q[triangle_index(2, 2)])],
                zero(BR),
            )

            problem = RationalSDP._extract_problem(opt)

            @test [block.size for block in problem.blocks] == [1, 1]
            @test problem.blocks[1].global_positions == [triangle_index(4, 4)]
            @test problem.blocks[2].global_positions == [length(p) + triangle_index(1, 1)]
            @test all(index -> fixed_zero(problem, index), setdiff(1:20, [10, 11]))
            particular, nullspace = problem.affine
            @test problem.A * particular == problem.b
            @test problem.A * nullspace ==
                  zeros(BR, size(problem.A, 1), size(nullspace, 2))
        end

        @testset "non-singleton diagonal equality is left alone" begin
            opt, variables = direct_psd_optimizer(3)
            y = MOI.add_variable(opt)
            add_affine_equality!(
                opt,
                [
                    (one(BR), variables[triangle_index(1, 1)]),
                    (one(BR), y),
                ],
                zero(BR),
            )

            problem = RationalSDP._extract_problem(opt)

            @test length(problem.blocks) == 1
            @test problem.blocks[1].size == 3
            @test !fixed_zero(problem, triangle_index(1, 1))
        end

        @testset "singleton high-degree Gram coefficient removes a PSD direction" begin
            opt, q = direct_psd_optimizer(3)
            add_affine_equality!(opt, [(one(BR), q[1])], one(BR))
            add_affine_equality!(opt, [(2 * one(BR), q[2])], zero(BR))
            add_affine_equality!(opt, [(one(BR), q[3]), (2 * one(BR), q[4])], one(BR))
            add_affine_equality!(opt, [(2 * one(BR), q[5])], zero(BR))
            add_affine_equality!(opt, [(one(BR), q[6])], zero(BR))

            problem = RationalSDP._extract_problem(opt)

            @test length(problem.blocks) == 1
            @test problem.blocks[1].size == 2
            @test problem.blocks[1].global_positions == [1, 2, 3]
            @test all(index -> fixed_zero(problem, index), [4, 5, 6])
            particular, nullspace = problem.affine
            @test problem.A * particular == problem.b
            @test problem.A * nullspace ==
                  zeros(BR, size(problem.A, 1), size(nullspace, 2))
        end
    end

    @testset "Phase II nullspace ignores unused affine directions" begin
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[MOI.VariableIndex(i) for i in 1:4],
            RationalSDP.BlockStructure[],
            [1],
            Rational{BigInt}[0//1, 1//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 1//1, 0//1, 0//1],
            zeros(Rational{BigInt}, 0, 4),
            Rational{BigInt}[],
            (
                zeros(Rational{BigInt}, 4),
                Matrix{Rational{BigInt}}(I, 4, 4),
            ),
        )

        phase2_nullspace = RationalSDP._phase2_nullspace(problem, RationalSDP.Float64x2)

        @test RationalSDP._phase2_relevant_positions(problem) == [1, 2]
        @test size(phase2_nullspace) == (4, 2)
        @test phase2_nullspace[1:2, :] == Matrix{Rational{BigInt}}(I, 2, 2)
        @test all(iszero, phase2_nullspace[3:4, :])
    end

    @testset "Reduced Phase II initialization preserves visible coordinates" begin
        inverse_sqrt_two = inv(sqrt(2.0))
        reduced_basis = reshape(
            Float64[inverse_sqrt_two, 0.0, inverse_sqrt_two],
            3,
            1,
        )
        x0 = zeros(Float64, 3)
        phase1_point = Float64[1.0, 10.0, 0.0]

        old_projection = transpose(reduced_basis) * (phase1_point - x0)
        @test !isapprox(
            (x0 + reduced_basis * old_projection)[1],
            phase1_point[1],
        )

        coordinates = RationalSDP._phase2_initial_coordinates(
            phase1_point,
            x0,
            reduced_basis,
            [1];
            match_relevant = true,
        )
        @test coordinates !== nothing
        reconstructed = x0 + reduced_basis * coordinates
        @test isapprox(reconstructed[1], phase1_point[1])
    end

    @testset "Solver failure messages expose nested exceptions" begin
        nested = CompositeException(Any[ErrorException("inner phase failure")])
        message = RationalSDP._solver_failure_message("Phase II", nested)
        @test occursin("Phase II failed", message)
        @test occursin("CompositeException", message)
        @test occursin("inner phase failure", message)
    end

    @testset "Facial reduction helper regressions" begin
        rational_direction = Float64[1.0, 2.0, 5.0]
        candidate_direction = sqrt(2.0) .* rational_direction
        block_matrix = Matrix{Float64}(I, 3, 3) -
                        (rational_direction * transpose(rational_direction)) /
                        dot(rational_direction, rational_direction)
        heuristic = RationalSDP._heuristic_kernel_direction(
            block_matrix,
            candidate_direction,
            RationalSDP.Settings(),
            Float64,
        )
        @test heuristic !== nothing
        @test heuristic.direction == Rational{BigInt}[1//1, 2//1, 5//1]

        @test RationalSDP._precision_escalation_types(Float64, 3) ==
              DataType[RationalSDP.Float64x2, RationalSDP.Float64x4, BigFloat]
        @test RationalSDP._precision_escalation_types(RationalSDP.Float64x3, 2) ==
              DataType[RationalSDP.Float64x4, BigFloat]
        @test isempty(RationalSDP._precision_escalation_types(BigFloat, 3))

        subspace_basis = Float64[
            1 0
            0 1
            1 2
            2 -1
        ]
        irrational_rotation = Float64[sqrt(2.0) 1.0; -1.0 sqrt(2.0)]
        subspace_noise = 2.0e-5 .* Float64[
            1 2
            -2 1
            3 -1
            -1 2
        ]
        recovered_subspace = RationalSDP._pivoted_rational_subspace_directions(
            subspace_basis * irrational_rotation + subspace_noise,
            RationalSDP.Settings(),
            Float64,
        )
        @test length(recovered_subspace) == 2
        recovered_matrix = Float64.(hcat(recovered_subspace...))
        subspace_projector = subspace_basis * pinv(subspace_basis)
        @test norm((I - subspace_projector) * recovered_matrix) < 1.0e-10

        chart_data = RationalSDP._rational_subspace_pivot_charts(
            subspace_basis * irrational_rotation,
            RationalSDP.Settings(facial_reduction_subspace_max_charts = 8),
            Float64,
        )
        @test chart_data.rank == 2
        @test 2 <= length(chart_data.charts) <= 8

        high_dynamic_range_basis = Float64[
            1.0e6 0
            0 1.0e6
            1 0
            0 1
        ]
        high_dynamic_range_subspace = high_dynamic_range_basis * irrational_rotation
        coarse_first_chart = RationalSDP._pivoted_rational_subspace_directions(
            high_dynamic_range_subspace,
            RationalSDP.Settings(facial_reduction_subspace_max_charts = 8),
            Float64;
            relation_tolerance = 1.0e-4,
        )
        high_dynamic_projector = high_dynamic_range_basis * pinv(high_dynamic_range_basis)
        coarse_first_matrix = Float64.(hcat(coarse_first_chart...))
        @test norm(
            coarse_first_matrix * pinv(coarse_first_matrix) - high_dynamic_projector,
        ) > 1.0e-8
        chart_candidates = RationalSDP._rational_subspace_candidate_sets(
            high_dynamic_range_subspace,
            RationalSDP.Settings(
                facial_reduction_subspace_max_charts = 8,
                facial_reduction_projector_recovery = false,
            ),
            Float64,
            1.0e-4,
        )
        @test any(
            begin
                candidate_matrix = Float64.(hcat(candidate.directions...))
                norm(candidate_matrix * pinv(candidate_matrix) - high_dynamic_projector) < 1.0e-10
            end for candidate in chart_candidates
        )
        @test allunique(
            RationalSDP._canonical_rational_subspace_key(candidate.directions) for
            candidate in chart_candidates
        )
        multifloat_subspace = RationalSDP.Float64x2.(subspace_basis)
        multifloat_candidates = RationalSDP._rational_subspace_candidate_sets(
            multifloat_subspace,
            RationalSDP.Settings(
                facial_reduction_subspace_max_charts = 2,
                facial_reduction_projector_recovery = false,
            ),
            RationalSDP.Float64x2,
            RationalSDP.Float64x2(1.0e-10),
        )
        @test !isempty(multifloat_candidates)
        @test all(
            candidate.reconstruction_error isa RationalSDP.Float64x2 for
            candidate in multifloat_candidates
        )
        @test all(
            eltype(candidate.projector) === RationalSDP.Float64x2 for
            candidate in multifloat_candidates
        )
        @test all(length(candidate.fingerprint) == 8 for candidate in multifloat_candidates)

        extreme_integer = big(10)^10_000
        extreme_direction =
            Rational{BigInt}[extreme_integer // big(1), 0 // big(1)]
        extreme_directions = [extreme_direction]
        extreme_key =
            RationalSDP._canonical_rational_subspace_key(extreme_directions)
        extreme_metrics = RationalSDP._rational_subspace_candidate_metrics(
            reshape(
                RationalSDP.Float64x2[
                    RationalSDP.Float64x2(1),
                    RationalSDP.Float64x2(0),
                ],
                2,
                1,
            ),
            extreme_directions,
            extreme_key,
            RationalSDP.Float64x2,
        )
        @test extreme_metrics.numeric_usable
        @test isfinite(extreme_metrics.reconstruction_error)
        @test all(isfinite, extreme_metrics.projector)

        invalid_metrics = RationalSDP._rational_subspace_candidate_metrics(
            reshape(
                RationalSDP.Float64x2[
                    RationalSDP.Float64x2(NaN),
                    RationalSDP.Float64x2(0),
                ],
                2,
                1,
            ),
            extreme_directions,
            extreme_key,
            RationalSDP.Float64x2,
        )
        @test !invalid_metrics.numeric_usable
        @test occursin("nonfinite", invalid_metrics.numeric_issue)
        schedule = RationalSDP._weighted_subspace_candidate_schedule(
            Any[
                (
                    directions = extreme_directions,
                    method = "extreme valid",
                    key = extreme_key,
                    tolerance = RationalSDP.Float64x2(1.0e-10),
                    extreme_metrics...,
                ),
                (
                    directions = extreme_directions,
                    method = "nonfinite source",
                    key = extreme_key,
                    tolerance = RationalSDP.Float64x2(1.0e-11),
                    invalid_metrics...,
                ),
            ],
            RationalSDP.Settings(),
            RationalSDP.Float64x2,
        )
        @test length(schedule.candidates) == 1
        @test schedule.rejected_count == 1
        @test schedule.candidates[1].method == "extreme valid"
        simple_candidate = (
            directions = extreme_directions,
            method = "simple",
            key = (:simple,),
            tolerance = RationalSDP.Float64x2(1.0e-4),
            projector = RationalSDP.Float64x2[1 0; 0 0],
            reconstruction_error = RationalSDP.Float64x2(1.0e-4),
            coefficient_bits = 7,
            fingerprint = "simple",
            numeric_usable = true,
            numeric_issue = "",
        )
        overfit_candidate = merge(
            simple_candidate,
            (
            method = "overfit",
            key = (:overfit,),
            tolerance = RationalSDP.Float64x2(1.0e-30),
            reconstruction_error = RationalSDP.Float64x2(1.0e-30),
            coefficient_bits = 900,
            fingerprint = "overfit",
            ),
        )
        moderate_candidate = merge(
            simple_candidate,
            (
                method = "moderate",
                key = (:moderate,),
                tolerance = RationalSDP.Float64x2(1.0e-3),
                reconstruction_error = RationalSDP.Float64x2(1.0e-3),
                coefficient_bits = 12,
                fingerprint = "moderate",
            ),
        )
        complexity_schedule = RationalSDP._weighted_subspace_candidate_schedule(
            Any[simple_candidate, overfit_candidate],
            RationalSDP.Settings(),
            RationalSDP.Float64x2,
        )
        @test [candidate.method for candidate in complexity_schedule.candidates] ==
              ["overfit", "simple"]
        accuracy_candidates = Any[
            merge(
                overfit_candidate,
                (
                    method = "accuracy $(index)",
                    key = (:accuracy, index),
                    reconstruction_error =
                        RationalSDP.Float64x2(index) *
                        RationalSDP.Float64x2(1.0e-30),
                    coefficient_bits = 900 + index,
                    fingerprint = "accuracy$(index)",
                ),
            ) for index in 1:8
        ]
        reserved_schedule = RationalSDP._weighted_subspace_candidate_schedule(
            Any[accuracy_candidates; simple_candidate; moderate_candidate],
            RationalSDP.Settings(),
            RationalSDP.Float64x2,
        )
        @test [candidate.method for candidate in reserved_schedule.candidates[1:7]] ==
              ["accuracy $(index)" for index in 1:7]
        @test [candidate.method for candidate in reserved_schedule.candidates[8:9]] ==
              ["simple", "moderate"]
        @test reserved_schedule.candidates[10].method == "accuracy 8"
        @test reserved_schedule.reserved_candidate_keys ==
              Any[simple_candidate.key, moderate_candidate.key]
        schedule_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            RationalSDP.BlockStructure[],
            Int[],
            Rational{BigInt}[],
            0//1,
            Rational{BigInt}[],
            zeros(Rational{BigInt}, 0, 0),
            Rational{BigInt}[],
            (Rational{BigInt}[], zeros(Rational{BigInt}, 0, 0)),
        )
        schedule_cache = RationalSDP._FacialReductionExactCache(schedule_problem)
        work_candidates = [
            merge(
                candidate,
                (
                    weighted_work = (
                        form_entries = BigInt(1),
                        affine_products = BigInt(20),
                    ),
                    individual_work = (affine_products = BigInt(10),),
                    attempt_key = candidate.key,
                    cheap_weighted_attempted = false,
                ),
            ) for candidate in reserved_schedule.candidates
        ]
        work_reserved_settings = RationalSDP.Settings(
            facial_reduction_weighted_max_candidate_sets = 9,
            facial_reduction_weighted_max_total_affine_products = 160,
            facial_reduction_individual_max_affine_products = 100,
            facial_reduction_numeric_weighted_subspace_max_form_entries = 100,
            facial_reduction_numeric_weighted_subspace_max_affine_products = 100,
            facial_reduction_weighted_subspace_max_form_entries = 100,
            facial_reduction_weighted_subspace_max_affine_products = 100,
        )
        work_reserved_schedule =
            RationalSDP._work_reserved_weighted_subspace_candidate_schedule(
                work_candidates,
                reserved_schedule.reserved_candidate_keys,
                schedule_cache,
                work_reserved_settings,
                1,
        )
        @test work_reserved_schedule.promoted_from == [8, 9]
        @test work_reserved_schedule.promoted_to == 1
        @test work_reserved_schedule.reserved_affine_products == 100
        @test work_reserved_schedule.reserved_candidate_keys ==
              Any[simple_candidate.key, moderate_candidate.key]
        @test [candidate.method for candidate in work_reserved_schedule.candidates[1:2]] ==
              ["simple", "moderate"]
        @test work_reserved_schedule.candidates[3].method == "accuracy 1"
        per_block_cache = RationalSDP._FacialReductionExactCache(schedule_problem)
        per_block_settings = RationalSDP.Settings(
            facial_reduction_weighted_max_candidate_sets = 1,
            facial_reduction_weighted_max_total_affine_products = 100,
        )
        @test RationalSDP._reserve_weighted_subspace_work!(
            per_block_cache,
            per_block_settings,
            1,
            10;
            new_candidate = true,
        ) == :reserved
        @test RationalSDP._reserve_weighted_subspace_work!(
            per_block_cache,
            per_block_settings,
            1,
            10;
            new_candidate = true,
        ) == :candidate_limit
        @test RationalSDP._reserve_weighted_subspace_work!(
            per_block_cache,
            per_block_settings,
            2,
            10;
            new_candidate = true,
        ) == :reserved
        @test per_block_cache.weighted_search.candidate_sets_attempted ==
              Dict(1 => 1, 2 => 1)
        @test per_block_cache.weighted_search.total_affine_products == 20
        exact_reservation_settings = RationalSDP.Settings(
            facial_reduction_weighted_exact_without_scout_limit = 2,
        )
        @test RationalSDP._weighted_exact_without_scout_available(
            schedule_cache,
            exact_reservation_settings;
            reserve_count = 1,
        )
        @test !RationalSDP._weighted_exact_without_scout_available(
            schedule_cache,
            exact_reservation_settings;
            reserve_count = 2,
        )
        schedule_cache.weighted_search.exact_without_scout_attempts = 1
        @test !RationalSDP._weighted_exact_without_scout_available(
            schedule_cache,
            exact_reservation_settings;
            reserve_count = 1,
        )
        @test RationalSDP._weighted_exact_without_scout_available(
            schedule_cache,
            exact_reservation_settings,
        )
        schedule_cache.weighted_search.exact_without_scout_attempts = 0
        plan = RationalSDP._weighted_subspace_candidate_plan(
            Any[
                (
                    directions = extreme_directions,
                    method = "extreme valid",
                    key = extreme_key,
                    tolerance = RationalSDP.Float64x2(1.0e-10),
                    extreme_metrics...,
                ),
                (
                    directions = extreme_directions,
                    method = "nonfinite source",
                    key = extreme_key,
                    tolerance = RationalSDP.Float64x2(1.0e-11),
                    invalid_metrics...,
                ),
            ],
            RationalSDP.Settings(),
            RationalSDP.Float64x2,
        )
        @test length(plan.scheduled) == 1
        @test length(plan.cheap_exact_only) == 1
        @test length(plan.exact_checks) == 2
        @test plan.cheap_exact_only[1].method == "nonfinite source"
        @test plan.exact_checks[end].method == "nonfinite source"

        projector_directions = RationalSDP._rational_projector_subspace_directions(
            subspace_basis * irrational_rotation,
            2,
            1.0e-10,
        )
        @test length(projector_directions) == 2
        @test norm(
            (I - subspace_projector) * Float64.(hcat(projector_directions...)),
        ) < 1.0e-10

        block = RationalSDP.BlockStructure(
            4,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:10],
            collect(1:10),
            RationalSDP._triangle_positions(4),
            [1, 3, 6, 10],
        )
        retry_subspace_basis = Float64[
            37 0
            0 41
            13 17
            11 -19
        ]
        retry_subspace = retry_subspace_basis * irrational_rotation + 0.5 .* subspace_noise
        retry_projector = retry_subspace_basis * pinv(retry_subspace_basis)
        coarse_subspace = RationalSDP._pivoted_rational_subspace_directions(
            retry_subspace,
            RationalSDP.Settings(),
            Float64;
            relation_tolerance = 1.0e-2,
        )
        @test norm(
            (I - retry_projector) * Float64.(hcat(coarse_subspace...)),
        ) > 1.0e-3

        exact_subspace_basis = Rational{BigInt}[37 0; 0 41; 13 17; 11 -19]
        gram_form = exact_subspace_basis * transpose(exact_subspace_basis)
        trace_row = Rational{BigInt}[
            i == j ? gram_form[i, j] : 2 * gram_form[i, j]
            for (i, j) in block.local_positions
        ]
        trace_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 10),
            0//1,
            zeros(Rational{BigInt}, 10),
            reshape(trace_row, 1, :),
            Rational{BigInt}[0//1],
            RationalSDP._solve_affine_system(
                reshape(trace_row, 1, :),
                Rational{BigInt}[0//1],
            ),
        )
        reduction_opt = RationalSDP.Optimizer{Rational{BigInt}}(verbose = false)
        certified_subspace = RationalSDP._certified_pivoted_subspace_directions(
            reduction_opt,
            trace_problem,
            block,
            1,
            retry_subspace,
            Float64,
            "regression subspace",
        )
        @test length(certified_subspace) == 2
        @test norm(
            (I - retry_projector) * Float64.(hcat(certified_subspace...)),
        ) < 1.0e-10

        exposing_range = Rational{BigInt}[
            1 0
            0 1
            1 1
            2 -1
        ]
        exposing_weight = Rational{BigInt}[2 1; 1 3]
        exposing_matrix =
            exposing_range * exposing_weight * transpose(exposing_range)
        exposing_row = Rational{BigInt}[
            i == j ? exposing_matrix[i, j] : 2 * exposing_matrix[i, j]
            for (i, j) in block.local_positions
        ]
        # A trace-zero row-space direction makes the normalized exposing-slack
        # affine family nontrivial.  It is chosen orthogonal (in packed dual
        # coordinates) to the projector-fit residual, so the joint fit must
        # recover the positive-semidefinite exposing row rather than merely
        # rationalizing the projector itself.
        trace_zero_row = Rational{BigInt}[
            204, -270, -141, 44, 278, -100, -226, -4, 78, 37
        ]
        exposing_A = Matrix(transpose(hcat(exposing_row, trace_zero_row)))
        exposing_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 10),
            0//1,
            zeros(Rational{BigInt}, 10),
            exposing_A,
            Rational{BigInt}[0//1, 0//1],
            RationalSDP._solve_affine_system(
                exposing_A,
                Rational{BigInt}[0//1, 0//1],
            ),
        )
        exact_projector =
            exposing_range *
            inv(transpose(exposing_range) * exposing_range) *
            transpose(exposing_range)
        normalized_projector_slack = Rational{BigInt}[
            (i == j ? exact_projector[i, j] : 2 * exact_projector[i, j]) /
            tr(exact_projector) for (i, j) in block.local_positions
        ]
        @test !RationalSDP._dual_slack_has_exact_certificate(
            exposing_problem,
            normalized_projector_slack,
        )

        weighted_subspace =
            Float64.(exposing_range) * irrational_rotation + 0.25 .* subspace_noise
        weighted_certified = RationalSDP._certified_pivoted_subspace_directions(
            reduction_opt,
            exposing_problem,
            block,
            1,
            weighted_subspace,
            Float64,
            "weighted regression subspace",
        )
        @test length(weighted_certified) == 2
        @test norm(
            (I - Float64.(exact_projector)) * Float64.(hcat(weighted_certified...)),
        ) < 1.0e-10
        exact_weight_directions = [
            collect(view(exposing_range, :, column)) for column in axes(exposing_range, 2)
        ]
        numeric_weighted = RationalSDP._numeric_weighted_subspace_exposure(
            exposing_problem,
            block,
            exact_weight_directions,
            RationalSDP.Settings(),
            Float64,
        )
        @test numeric_weighted !== nothing
        @test length(numeric_weighted.directions) == 2

        # A rank-deficient PSD weight certifies only its exact range inside the
        # proposed numerical kernel. Returning the whole candidate here would
        # over-reduce the block.
        partial_A = zeros(Rational{BigInt}, 1, 10)
        partial_A[1, 1] = 1//1
        partial_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 10),
            0//1,
            zeros(Rational{BigInt}, 10),
            partial_A,
            Rational{BigInt}[0//1],
            RationalSDP._solve_affine_system(
                partial_A,
                Rational{BigInt}[0//1],
            ),
        )
        partial_directions = [
            Rational{BigInt}[1, 0, 0, 0],
            Rational{BigInt}[0, 1, 0, 0],
        ]
        partial_exposure = RationalSDP._block_weighted_subspace_exposure(
            partial_problem,
            block,
            partial_directions,
            RationalSDP.Settings(),
            Float64,
        )
        @test partial_exposure !== nothing
        @test RationalSDP._positive_semidefinite_exact(partial_exposure.weight)
        @test !RationalSDP._positive_definite_exact(partial_exposure.weight)
        @test length(partial_exposure.directions) == 1
        @test partial_exposure.directions[1][2:4] == zeros(Rational{BigInt}, 3)
        partial_numeric_exposure =
            RationalSDP._numeric_weighted_subspace_exposure(
                partial_problem,
                block,
                partial_directions,
                RationalSDP.Settings(),
                Float64,
            )
        @test partial_numeric_exposure !== nothing
        @test length(partial_numeric_exposure.directions) == 1
        partial_certified = RationalSDP._certified_pivoted_subspace_directions(
            reduction_opt,
            partial_problem,
            block,
            1,
            Float64.(hcat(partial_directions...)),
            Float64,
            "partial weighted regression subspace",
        )
        @test length(partial_certified) == 1
        @test partial_certified[1][2:4] == zeros(Rational{BigInt}, 3)
        invalid_partial_keep_basis = RationalSDP._orthogonal_complement_basis(
            partial_directions,
            block.size,
        )
        invalid_partial_reduction = RationalSDP._CertifiedFacialReduction(
            "invalid partial weighted regression",
            Int[],
            Dict(1 => invalid_partial_keep_basis),
        )
        @test RationalSDP._cached_facial_reduction_violation(
            partial_problem,
            invalid_partial_reduction,
        ) !== nothing
        weighted_keep_basis = RationalSDP._orthogonal_complement_basis(
            weighted_certified,
            block.size,
        )
        weighted_reduction = RationalSDP._CertifiedFacialReduction(
            "weighted regression",
            Int[],
            Dict(1 => weighted_keep_basis),
        )
        @test RationalSDP._cached_facial_reduction_violation(
            exposing_problem,
            weighted_reduction,
        ) === nothing

        # The weighted joint certificate is useful for small subspaces, but
        # must not materialize an enormous exact affine product merely to
        # decide whether such a certificate exists.
        oversized_weight_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 10),
            0//1,
            zeros(Rational{BigInt}, 10),
            zeros(Rational{BigInt}, 0, 10),
            Rational{BigInt}[],
            (
                zeros(Rational{BigInt}, 10),
                zeros(Rational{BigInt}, 10, 70_000),
            ),
        )
        oversized_weight_directions = [
            Rational{BigInt}[i == j ? 1//1 : 0//1 for i in 1:block.size] for
            j in 1:block.size
        ]
        oversized_weight_work = RationalSDP._weighted_subspace_exposure_work(
            oversized_weight_problem,
            block,
            length(oversized_weight_directions),
        )
        default_settings = RationalSDP.Settings()
        @test oversized_weight_work.affine_products >
              default_settings.facial_reduction_weighted_subspace_max_affine_products
        @test !RationalSDP._weighted_subspace_exposure_is_small(
            oversized_weight_work,
            default_settings,
        )
        oversized_individual_work = RationalSDP._individual_subspace_certificate_work(
            oversized_weight_problem,
            block,
            length(oversized_weight_directions),
        )
        @test oversized_individual_work.affine_products >
              default_settings.facial_reduction_individual_max_affine_products
        @test !RationalSDP._individual_subspace_certificate_is_small(
            oversized_individual_work,
            default_settings,
        )
        permissive_settings = RationalSDP.Settings(
            facial_reduction_weighted_subspace_max_form_entries = typemax(Int),
            facial_reduction_weighted_subspace_max_affine_products = typemax(Int),
            facial_reduction_individual_max_affine_products = typemax(Int),
        )
        @test RationalSDP._weighted_subspace_exposure_is_small(
            oversized_weight_work,
            permissive_settings,
        )
        @test RationalSDP._individual_subspace_certificate_is_small(
            oversized_individual_work,
            permissive_settings,
        )
        @test RationalSDP._block_weighted_subspace_exposure(
            oversized_weight_problem,
            block,
            oversized_weight_directions,
            RationalSDP.Settings(),
            Float64,
        ) === nothing

        fixed_block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:3],
            collect(1:3),
            RationalSDP._triangle_positions(2),
            [1, 3],
        )
        fixed_A = Matrix{Rational{BigInt}}(I, 3, 3)
        fixed_b = Rational{BigInt}[1, 0, 1]
        fixed_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [fixed_block],
            Int[],
            zeros(Rational{BigInt}, 3),
            0//1,
            zeros(Rational{BigInt}, 3),
            fixed_A,
            fixed_b,
            RationalSDP._solve_affine_system(fixed_A, fixed_b),
        )
        extreme_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [fixed_block],
            Int[],
            zeros(Rational{BigInt}, 3),
            0//1,
            zeros(Rational{BigInt}, 3),
            fixed_A,
            Rational{BigInt}[extreme_integer, 0, extreme_integer],
            (
                Rational{BigInt}[extreme_integer, 0, extreme_integer],
                zeros(Rational{BigInt}, 3, 0),
            ),
        )
        extreme_scout = RationalSDP._numeric_weighted_subspace_exposure_attempt(
            extreme_problem,
            fixed_block,
            [Rational{BigInt}[extreme_integer, 0]],
            RationalSDP.Settings(),
            RationalSDP.Float64x2,
        )
        @test extreme_scout.exposure === nothing
        @test extreme_scout.status == :unpromising
        @test !occursin("ArgumentError", extreme_scout.reason)
        @test !occursin("nonfinite", extreme_scout.reason)

        malformed_block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:3],
            collect(1:3),
            [(0, 0), (1, 2), (2, 2)],
            [1, 3],
        )
        unavailable_scout =
            RationalSDP._numeric_weighted_subspace_exposure_attempt(
                fixed_problem,
                malformed_block,
                [Rational{BigInt}[1, 0]],
                RationalSDP.Settings(),
                RationalSDP.Float64x2,
            )
        @test unavailable_scout.status == :unavailable
        @test !unavailable_scout.promising
        @test occursin("forming the weighted affine system", unavailable_scout.reason)
        @test occursin("BoundsError", unavailable_scout.reason)
        @test occursin(
            "precise scout diagnostic",
            RationalSDP._numerical_scout_exception_summary(
                ArgumentError("precise scout diagnostic"),
            ),
        )
        staged_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = true,
            facial_reduction_subspace_max_charts = 1,
            facial_reduction_projector_recovery = false,
            facial_reduction_cheap_weighted_subspace_max_weight_dimension = 0,
            facial_reduction_cheap_weighted_subspace_max_form_entries = 0,
            facial_reduction_cheap_weighted_subspace_max_affine_products = 0,
            facial_reduction_individual_max_affine_products = 0,
            facial_reduction_numeric_weighted_subspace_max_form_entries = 100,
            facial_reduction_numeric_weighted_subspace_max_affine_products = 100,
            facial_reduction_weighted_subspace_max_form_entries = 100,
            facial_reduction_weighted_subspace_max_affine_products = 100,
            facial_reduction_weighted_max_candidate_sets = 4,
            facial_reduction_weighted_max_total_affine_products = 100,
            facial_reduction_weighted_exact_without_scout_limit = 1,
        )
        staged_cache = RationalSDP._FacialReductionExactCache(fixed_problem)
        staged_result, staged_log_text = mktemp() do _, io
            result = redirect_stdout(io) do
                RationalSDP._certified_pivoted_subspace_directions(
                    staged_opt,
                    fixed_problem,
                    fixed_block,
                    1,
                    reshape(Float64[1, 0], 2, 1),
                    Float64,
                    "staged regression";
                    cache = staged_cache,
                )
            end
            flush(io)
            seekstart(io)
            return result, read(io, String)
        end
        @test isempty(staged_result)
        @test staged_cache.weighted_search.candidate_sets_attempted == Dict(1 => 1)
        @test staged_cache.weighted_search.total_affine_products == 6
        @test staged_cache.weighted_search.exact_without_scout_attempts == 1
        @test length(staged_cache.numeric_weighted_failures) == 1
        @test only(values(staged_cache.numeric_weighted_failures)).status ==
              :unpromising
        @test length(staged_cache.exact_weighted_failures) == 1
        @test occursin("relation_tol=", staged_log_text)
        @test occursin("method=pivot chart", staged_log_text)
        @test occursin("fingerprint=", staged_log_text)
        @test occursin("cumulative_products=3/100", staged_log_text)

        cached_state = deepcopy(staged_cache.weighted_search)
        cached_result, cached_log_text = mktemp() do _, io
            result = redirect_stdout(io) do
                RationalSDP._certified_pivoted_subspace_directions(
                    staged_opt,
                    fixed_problem,
                    fixed_block,
                    1,
                    reshape(Float64[1, 0], 2, 1),
                    Float64,
                    "staged regression";
                    cache = staged_cache,
                )
            end
            flush(io)
            seekstart(io)
            return result, read(io, String)
        end
        @test isempty(cached_result)
        @test staged_cache.weighted_search.candidate_sets_attempted ==
              cached_state.candidate_sets_attempted
        @test staged_cache.weighted_search.total_affine_products ==
              cached_state.total_affine_products
        @test occursin("cached_failures=2", cached_log_text)

        limited_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_subspace_max_charts = 1,
            facial_reduction_projector_recovery = false,
            facial_reduction_cheap_weighted_subspace_max_weight_dimension = 0,
            facial_reduction_cheap_weighted_subspace_max_form_entries = 0,
            facial_reduction_cheap_weighted_subspace_max_affine_products = 0,
            facial_reduction_individual_max_affine_products = 0,
            facial_reduction_numeric_weighted_subspace_max_form_entries = 100,
            facial_reduction_numeric_weighted_subspace_max_affine_products = 100,
            facial_reduction_weighted_subspace_max_form_entries = 100,
            facial_reduction_weighted_subspace_max_affine_products = 100,
            facial_reduction_weighted_max_candidate_sets = 4,
            facial_reduction_weighted_max_total_affine_products = 3,
            facial_reduction_weighted_exact_without_scout_limit = 1,
        )
        limited_cache = RationalSDP._FacialReductionExactCache(fixed_problem)
        @test isempty(RationalSDP._certified_pivoted_subspace_directions(
            limited_opt,
            fixed_problem,
            fixed_block,
            1,
            reshape(Float64[1, 0], 2, 1),
            Float64,
            "budget regression";
            cache = limited_cache,
        ))
        @test limited_cache.weighted_search.total_affine_products == 3
        @test isempty(limited_cache.exact_weighted_failures)

        directions = [
            Rational{BigInt}[1//1, 0//1],
            Rational{BigInt}[2//1, 0//1],
            Rational{BigInt}[0//1, 1//1],
        ]
        independent = RationalSDP._linearly_independent_directions(directions)
        @test length(independent) == 2
        @test any(direction == Rational{BigInt}[0//1, 1//1] for direction in independent)
        @test any(
            direction == Rational{BigInt}[1//1, 0//1] ||
            direction == Rational{BigInt}[2//1, 0//1] for
            direction in independent
        )

        D = Float64[
            1.0 0.0
            -1.0 1.0
            0.0 -1.0
        ]

        function triangle_entries(matrix::AbstractMatrix)
            return Rational{BigInt}[
                matrix[i, j] for (i, j) in RationalSDP._triangle_positions(size(matrix, 1))
            ]
        end

        function column_only_exposing_directions(
            opt::RationalSDP.Optimizer,
            problem::RationalSDP.ProblemData,
            block_index::Int,
            block_matrix::Matrix{F},
            ::Type{F},
        ) where {F<:AbstractFloat}
            block = problem.blocks[block_index]
            symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
            eigenvalues = eigvals(symmetric_matrix)
            isempty(eigenvalues) && return Vector{RationalSDP.ExactRational}[]
            exposure_tolerance = max(
                RationalSDP._to_working_float(
                    F,
                    opt.settings.facial_reduction_exposure_tolerance,
                ),
                F(100) * eps(F),
            )
            maximum(eigenvalues) <= exposure_tolerance &&
                return Vector{RationalSDP.ExactRational}[]
            singular_values = svdvals(Matrix(symmetric_matrix))
            rank_tolerance = max(
                RationalSDP._to_working_float(
                    F,
                    opt.settings.facial_reduction_rank_tolerance,
                ),
                F(100) * eps(F),
            )
            numeric_rank = count(value -> value > rank_tolerance, singular_values)
            numeric_rank == 0 && return Vector{RationalSDP.ExactRational}[]

            qr_factor = qr(Matrix(symmetric_matrix), ColumnNorm())
            candidate_columns = unique(qr_factor.p[1:numeric_rank])
            exact_directions = Vector{Vector{RationalSDP.ExactRational}}()
            for column_index in candidate_columns
                direction = RationalSDP._exact_face_direction(
                    problem,
                    block,
                    collect(view(block_matrix, :, column_index)),
                    opt.settings,
                    F,
                )
                direction === nothing && continue
                push!(exact_directions, direction)
            end
            return RationalSDP._linearly_independent_directions(exact_directions)
        end

        block3 = RationalSDP.BlockStructure(
            3,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
            collect(1:6),
            RationalSDP._triangle_positions(3),
            [1, 3, 6],
        )
        rank_one_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block3],
            Int[],
            zeros(Rational{BigInt}, 6),
            0//1,
            zeros(Rational{BigInt}, 6),
            Matrix{Rational{BigInt}}(I, 6, 6),
            triangle_entries(fill(1//1, 3, 3)),
            RationalSDP._solve_affine_system(
                Matrix{Rational{BigInt}}(I, 6, 6),
                triangle_entries(fill(1//1, 3, 3)),
            ),
        )
        M = Float64[
            sqrt(2.0) 0.2 * pi
            0.2 * pi sqrt(3.0)
        ]
        exposing_slack = D * M * transpose(D)
        pivot_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_irrational_behavior = :warn,
            rational_tolerance = big"1e-12",
            recovery_tolerance_shrink = big"0.01",
        )
        column_only_directions = column_only_exposing_directions(
            pivot_opt,
            rank_one_problem,
            1,
            exposing_slack,
            Float64,
        )
        @test isempty(column_only_directions)

        exposing_directions = RationalSDP._facial_reduction_block_directions(
            pivot_opt,
            rank_one_problem,
            1,
            exposing_slack,
            Float64,
        )
        @test length(exposing_directions) == 2
        @test all(direction -> sum(direction) == 0//1, exposing_directions)
        @test length(RationalSDP._linearly_independent_directions(exposing_directions)) == 2
        keep_basis = RationalSDP._orthogonal_complement_basis(exposing_directions, 3)
        @test size(keep_basis, 2) == 1

        reduced_rank_one_problem =
            RationalSDP._apply_facial_reduction(rank_one_problem, Int[], Dict(1 => keep_basis))
        @test [block.size for block in reduced_rank_one_problem.blocks] == [1]
        @test reduced_rank_one_problem.affine !== nothing
        reduced_particular, reduced_nullspace = reduced_rank_one_problem.affine
        @test reduced_rank_one_problem.A * reduced_particular == reduced_rank_one_problem.b
        @test reduced_rank_one_problem.A * reduced_nullspace ==
              zeros(
                  Rational{BigInt},
                  size(reduced_rank_one_problem.A, 1),
                  size(reduced_nullspace, 2),
              )
        compaction_checkpoints = String[]
        compacted_rank_one_problem = RationalSDP._apply_facial_reduction(
            rank_one_problem,
            Int[],
            Dict(1 => keep_basis);
            checkpoint = stage -> push!(compaction_checkpoints, stage),
            settings = RationalSDP.Settings(facial_reduction_affine_compaction_factor = 0),
        )
        @test size(compacted_rank_one_problem.A, 1) <= size(reduced_rank_one_problem.A, 1)
        @test any(
            stage -> occursin("eliminating superseded PSD coordinates", stage),
            compaction_checkpoints,
        )
        @test any(stage -> occursin("compacting redundant", stage), compaction_checkpoints)

        skipped_compaction_checkpoints = String[]
        uncompacted_rank_one_problem = RationalSDP._apply_facial_reduction(
            rank_one_problem,
            Int[],
            Dict(1 => keep_basis);
            checkpoint = stage -> push!(skipped_compaction_checkpoints, stage),
            settings = RationalSDP.Settings(
                facial_reduction_affine_compaction_factor = 0,
                facial_reduction_affine_compaction_max_entries = 1,
            ),
        )
        @test size(uncompacted_rank_one_problem.A, 1) ==
              size(reduced_rank_one_problem.A, 1)
        @test any(
            stage -> occursin("eliminating superseded PSD coordinates", stage),
            skipped_compaction_checkpoints,
        )
        @test any(
            stage -> occursin("skipping optional affine compaction", stage),
            skipped_compaction_checkpoints,
        )

        interior_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block3],
            Int[],
            zeros(Rational{BigInt}, 6),
            0//1,
            zeros(Rational{BigInt}, 6),
            zeros(Rational{BigInt}, 0, 6),
            Rational{BigInt}[],
            (
                triangle_entries(Matrix{Rational{BigInt}}(I, 3, 3)),
                zeros(Rational{BigInt}, 6, 0),
            ),
        )
        @test all(
            direction ->
                RationalSDP._block_annihilation_violation(interior_problem, block3, direction) !==
                nothing,
            exposing_directions,
        )
        @test isempty(
            RationalSDP._facial_reduction_block_directions(
                pivot_opt,
                interior_problem,
                1,
                exposing_slack,
                Float64,
            ),
        )

        block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing, nothing, nothing],
            [1, 2, 3],
            [(1, 1), (2, 1), (2, 2)],
            [1, 3],
        )
        helper_opt = RationalSDP.Optimizer{Rational{BigInt}}(verbose = false)
        exact_boundary_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            Rational{BigInt}[0//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1, 0//1],
            Matrix{Rational{BigInt}}(I, 3, 3),
            Rational{BigInt}[1//1, -1//1, 1//1],
            RationalSDP._solve_affine_system(
                Matrix{Rational{BigInt}}(I, 3, 3),
                Rational{BigInt}[1//1, -1//1, 1//1],
            ),
        )
        directions =
            RationalSDP._candidate_kernel_directions(
                helper_opt,
                exact_boundary_problem,
                1,
                Float64[1.0 -1.0; -1.0 1.0],
                Float64,
            )
        @test directions == [Rational{BigInt}[1//1, 1//1]]

        exact_cache = RationalSDP._FacialReductionExactCache(exact_boundary_problem)
        @test exact_cache.row_space === nothing
        @test exact_cache.block_exact_directions[1] === nothing
        cached_directions = RationalSDP._exact_block_nullspace_directions(
            exact_boundary_problem,
            block;
            cache = exact_cache,
            block_index = 1,
        )
        @test cached_directions == [Rational{BigInt}[1//1, 1//1]]
        @test exact_cache.block_exact_directions[1] === cached_directions
        @test RationalSDP._exact_block_nullspace_directions(
            exact_boundary_problem,
            block;
            cache = exact_cache,
            block_index = 1,
        ) === cached_directions

        exact_slack, _, _ = RationalSDP._facial_reduction_slack(
            exact_boundary_problem,
            Rational{BigInt}[];
            cache = exact_cache,
        )
        @test exact_slack == zeros(Rational{BigInt}, 3)
        RationalSDP._facial_reduction_row_space!(exact_cache, exact_boundary_problem)
        @test exact_cache.row_space !== nothing

        psd_certified_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            Rational{BigInt}[0//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1, 0//1],
            zeros(Rational{BigInt}, 0, 3),
            Rational{BigInt}[],
            (
                Rational{BigInt}[0//1, 0//1, 1//1],
                reshape(Rational{BigInt}[0//1, 1//1, 0//1], 3, 1),
            ),
        )
        direction = Rational{BigInt}[1//1, 0//1]
        @test RationalSDP._block_annihilation_violation(
            psd_certified_problem,
            block,
            direction,
        ) !== nothing
        @test RationalSDP._block_quadratic_vanish_violation(
            psd_certified_problem,
            block,
            direction,
        ) === nothing
        directions =
            RationalSDP._candidate_kernel_directions(
                helper_opt,
                psd_certified_problem,
                1,
                Float64[0.0 0.0; 0.0 1.0],
                Float64,
            )
        @test directions == [direction]

        trace_certified_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            Rational{BigInt}[0//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1, 0//1],
            zeros(Rational{BigInt}, 0, 3),
            Rational{BigInt}[],
            (
                Rational{BigInt}[0//1, 0//1, 0//1],
                reshape(Rational{BigInt}[1//1, 0//1, -1//1], 3, 1),
            ),
        )
        directions =
            RationalSDP._candidate_kernel_directions(
                helper_opt,
                trace_certified_problem,
                1,
                zeros(Float64, 2, 2),
                Float64,
            )
        @test directions == [
            Rational{BigInt}[1//1, 0//1],
            Rational{BigInt}[0//1, 1//1],
        ]

        # Exact affine vanishing is already a complete certificate that an
        # exposing form belongs to the affine row space.  On large instances,
        # avoid constructing the dense exact row-space RREF just to recover an
        # optional multiplier.
        large_dimension = 600
        large_block = RationalSDP.BlockStructure(
            1,
            Union{Nothing,MOI.VariableIndex}[nothing],
            [1],
            [(1, 1)],
            [1],
        )
        large_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [large_block],
            Int[],
            zeros(Rational{BigInt}, large_dimension),
            0//1,
            zeros(Rational{BigInt}, large_dimension),
            Matrix{Rational{BigInt}}(I, large_dimension, large_dimension),
            zeros(Rational{BigInt}, large_dimension),
            (
                zeros(Rational{BigInt}, large_dimension),
                zeros(Rational{BigInt}, large_dimension, 0),
            ),
        )
        large_cache = RationalSDP._FacialReductionExactCache(large_problem)
        @test !RationalSDP._facial_reduction_row_space_is_small(large_problem)
        @test RationalSDP._facial_reduction_row_space_is_small(
            large_problem,
            RationalSDP.Settings(facial_reduction_row_space_max_entries = 400_000),
        )
        @test RationalSDP._block_trace_vanish_violation(
            large_problem,
            large_block,
            [Rational{BigInt}[1//1]];
            cache = large_cache,
            block_index = 1,
        ) === nothing
        @test large_cache.row_space === nothing
        @test RationalSDP._row_space_multiplier(
            large_problem,
            [1],
            Rational{BigInt}[1//1];
            cache = large_cache,
        ) === nothing

        uncertified_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            Rational{BigInt}[0//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1, 0//1],
            zeros(Rational{BigInt}, 0, 3),
            Rational{BigInt}[],
            (
                Rational{BigInt}[1//1, 0//1, 1//1],
                zeros(Rational{BigInt}, 3, 0),
            ),
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_irrational_behavior = :warn,
        )
        @test isempty(
            RationalSDP._candidate_kernel_directions(
                opt,
                uncertified_problem,
                1,
                Float64[1.0 -1.0; -1.0 1.0],
                Float64,
            ),
        )

        # A single boundary point does not identify a face of the whole affine
        # slice.  Here X[1, 1] == 1 admits I as a positive-definite point, so the
        # singular candidate diag(1, 0) must not remove the second PSD direction.
        full_face_block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing, nothing, nothing],
            [1, 2, 3],
            [(1, 1), (2, 1), (2, 2)],
            [1, 3],
        )
        full_face_A = Rational{BigInt}[1//1 0//1 0//1]
        full_face_b = Rational{BigInt}[1//1]
        full_face_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [full_face_block],
            Int[],
            zeros(Rational{BigInt}, 3),
            0//1,
            zeros(Rational{BigInt}, 3),
            full_face_A,
            full_face_b,
            RationalSDP._solve_affine_system(full_face_A, full_face_b),
        )
        full_face_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            working_float_type = Float64,
            facial_reduction_float_type = Float64,
            facial_reduction_irrational_behavior = :warn,
        )
        unreduced = RationalSDP._facially_reduce_problem(
            full_face_opt,
            full_face_problem,
            Float64[1.0, 0.0, 0.0],
            Float64,
        )
        @test [candidate_block.size for candidate_block in unreduced.blocks] == [2]
        @test size(unreduced.A) == size(full_face_problem.A)

        @test RationalSDP._exact_primal_feasibility(
            full_face_problem,
            Rational{BigInt}[1//1, 0//1, 0//1],
        ).ok
        @test RationalSDP._exact_primal_feasibility(
            full_face_problem,
            Rational{BigInt}[1//1, 0//1, 1//1],
        ).ok
        @test !RationalSDP._exact_primal_feasibility(
            full_face_problem,
            Rational{BigInt}[2//1, 0//1, 1//1],
        ).ok
        @test !RationalSDP._exact_primal_feasibility(
            full_face_problem,
            Rational{BigInt}[1//1, 2//1, 1//1],
        ).ok

        scalar_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            RationalSDP.BlockStructure[],
            [1],
            Rational{BigInt}[0//1],
            0//1,
            Rational{BigInt}[0//1],
            zeros(Rational{BigInt}, 0, 1),
            Rational{BigInt}[],
            (Rational{BigInt}[0//1], reshape(Rational{BigInt}[1//1], 1, 1)),
        )
        @test RationalSDP._exact_primal_feasibility(
            scalar_problem,
            Rational{BigInt}[0//1],
        ).ok
        @test !RationalSDP._exact_primal_feasibility(
            scalar_problem,
            Rational{BigInt}[-1//1],
        ).ok
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

    @testset "Boundary recovery precision escalation is opt in" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        set_optimizer_attribute(model, "phase1_hypatia_float_type", Float64)
        set_optimizer_attribute(model, "facial_reduction_float_type", Float64)
        set_optimizer_attribute(model, "facial_reduction_precision_escalation_max_retries", 1)
        @variable(model, X[1:3, 1:3], PSD)
        for j in 1:3, i in j:3
            (i == 1 && j == 1) && continue
            @constraint(model, X[i, j] == X[1, 1])
        end
        @objective(model, Min, 0//1)
        optimize!(model)

        @test termination_status(model) == MOI.OPTIMAL
        bridge_optimizer = getfield(backend(model), :optimizer)
        opt = getfield(bridge_optimizer, :model)
        stats = RationalSDP.facial_reduction_statistics(opt)
        @test stats.precision_escalations_attempted >= 1
        VX = value.(X)
        @test is_psd_exact(VX)
        @test all(value == VX[1, 1] for value in VX)
    end

    @testset "Facial reduction affine lifting stays consistent" begin
        block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing, nothing, nothing],
            [1, 2, 3],
            [(1, 1), (2, 1), (2, 2)],
            [1, 3],
        )
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            Rational{BigInt}[0//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1, 0//1],
            zeros(Rational{BigInt}, 0, 3),
            Rational{BigInt}[],
            (
                zeros(Rational{BigInt}, 3),
                Matrix{Rational{BigInt}}(I, 3, 3),
            ),
        )
        keep_basis = reshape(Rational{BigInt}[1//1, 1//1], 2, 1)
        reduced_problem = RationalSDP._apply_facial_reduction(problem, Int[], Dict(1 => keep_basis))

        @test [block.size for block in reduced_problem.blocks] == [1]
        @test reduced_problem.affine !== nothing
        particular, nullspace = reduced_problem.affine
        @test size(reduced_problem.A, 2) == length(particular)
        @test reduced_problem.A * particular == reduced_problem.b
        @test reduced_problem.A * nullspace == zeros(Rational{BigInt}, size(reduced_problem.A, 1), size(nullspace, 2))
        @test RationalSDP._vector_to_matrix(particular, reduced_problem.blocks[1]) == Rational{BigInt}[0//1;;]
        @test length(reduced_problem.objective_vector_raw) == 1
        @test size(reduced_problem.A) == (0, 1)
        @test RationalSDP._lift_original_solution(
            reduced_problem,
            Rational{BigInt}[2//1],
        ) == Rational{BigInt}[2//1, 2//1, 2//1]
    end

    @testset "Repeated facial reductions compact coordinates and preserve cache replay" begin
        triangle3 = [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]
        block = RationalSDP.BlockStructure(
            3,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in triangle3],
            collect(1:6),
            triangle3,
            [1, 3, 6],
        )
        A = Rational{BigInt}[
            -1 1 0 0 0 0
            -1 0 1 0 0 0
            0 0 0 1 0 0
            0 0 0 0 1 0
            0 0 0 0 0 1
        ]
        b = zeros(Rational{BigInt}, 5)
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 6),
            0//1,
            zeros(Rational{BigInt}, 6),
            A,
            b,
            RationalSDP._solve_affine_system(A, b),
        )
        keep_two = Rational{BigInt}[
            1 0
            0 1
            0 0
        ]
        keep_one = reshape(Rational{BigInt}[1, 1], 2, 1)

        reduced_once = RationalSDP._apply_facial_reduction(
            problem,
            Int[],
            Dict(1 => keep_two),
        )
        reduced_twice = RationalSDP._apply_facial_reduction(
            reduced_once,
            Int[],
            Dict(1 => keep_one),
        )
        @test length(reduced_once.objective_vector_raw) == 3
        @test length(reduced_twice.objective_vector_raw) == 1
        @test size(reduced_once.A) == (5, 3)
        @test size(reduced_twice.A) == (5, 1)
        @test [item.size for item in reduced_twice.blocks] == [1]
        @test RationalSDP._lift_original_solution(
            reduced_twice,
            Rational{BigInt}[7//1],
        ) == Rational{BigInt}[7//1, 7//1, 7//1, 0//1, 0//1, 0//1]
        first_reduction = RationalSDP._CertifiedFacialReduction(
            "first compact face",
            Int[],
            Dict(1 => keep_two),
        )
        second_reduction = RationalSDP._CertifiedFacialReduction(
            "second compact face",
            Int[],
            Dict(1 => keep_one),
        )
        compact_records = Any[
            RationalSDP._facial_reduction_record(problem, first_reduction),
            RationalSDP._facial_reduction_record(reduced_once, second_reduction),
        ]
        @test [record.signature.dimension for record in compact_records] == [6, 3]
        @test [record.signature.equation_count for record in compact_records] == [5, 5]
        load_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction = false,
        )
        load_opt.facial_reduction_loaded_records = compact_records
        replay = RationalSDP._apply_loaded_facial_reductions(
            load_opt,
            problem;
            return_details = true,
        )
        @test replay.applied == 2
        @test length(replay.problem.objective_vector_raw) == 1
        @test [item.size for item in replay.problem.blocks] == [1]
        @test RationalSDP._lift_original_solution(
            replay.problem,
            Rational{BigInt}[11//1],
        ) == Rational{BigInt}[11//1, 11//1, 11//1, 0//1, 0//1, 0//1]

        rank_expansion_load_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction = true,
            facial_reduction_float_type = Float64,
            facial_reduction_rank_expansion_rounds = 1,
        )
        rank_expansion_load_opt.facial_reduction_loaded_records = compact_records
        deferred_replay = RationalSDP._apply_loaded_facial_reductions(
            rank_expansion_load_opt,
            problem;
            return_details = true,
        )
        # Enabling rank expansion must not strengthen the first cached face
        # before the second record's intermediate-problem signature is matched.
        @test deferred_replay.applied == 2
        @test length(deferred_replay.problem.objective_vector_raw) == 1

        mktempdir() do directory
            input_file = joinpath(directory, "compact-input.rsdpcache")
            output_file = joinpath(directory, "compact-output.rsdpcache")
            RationalSDP._write_facial_reduction_cache(input_file, compact_records)
            source_bytes = read(input_file)
            replay_save_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = false,
                facial_reduction = false,
                facial_reduction_load_file = input_file,
                facial_reduction_save_file = output_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(replay_save_opt)
            distinct_file_replay = RationalSDP._apply_loaded_facial_reductions(
                replay_save_opt,
                problem;
                return_details = true,
            )
            @test distinct_file_replay.applied == 2
            @test length(distinct_file_replay.problem.objective_vector_raw) == 1
            @test read(input_file) == source_bytes
            replayed_records =
                RationalSDP._read_facial_reduction_cache(output_file)
            @test length(replayed_records) == 2
            @test [
                record.signature.dimension for record in replayed_records
            ] == [6, 3]
            @test [
                record.signature.equation_count for record in replayed_records
            ] == [5, 5]
            @test all(
                RationalSDP._facial_reduction_signature_matches_signature(
                    compact.signature,
                    replayed.signature,
                ) for (compact, replayed) in zip(compact_records, replayed_records)
            )

            output_load_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = false,
                facial_reduction = false,
                facial_reduction_load_file = output_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(output_load_opt)
            output_replay = RationalSDP._apply_loaded_facial_reductions(
                output_load_opt,
                problem;
                return_details = true,
            )
            @test output_replay.applied == 2
            @test length(output_replay.problem.objective_vector_raw) == 1
        end
    end

    @testset "Exactly fixed PSD directions are eliminated without new equations" begin
        triangle2 = [(1, 1), (2, 1), (2, 2)]
        blocks = [
            RationalSDP.BlockStructure(
                2,
                Union{Nothing,MOI.VariableIndex}[nothing for _ in triangle2],
                collect(offset .+ (1:3)),
                triangle2,
                [offset + 1, offset + 3],
            ) for offset in (0, 3)
        ]
        A = reshape(Rational{BigInt}[0, 0, 0, 1, 0, 0], 1, 6)
        b = Rational{BigInt}[0]
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            blocks,
            Int[],
            zeros(Rational{BigInt}, 6),
            0 // 1,
            zeros(Rational{BigInt}, 6),
            A,
            b,
            RationalSDP._solve_affine_system(A, b),
        )
        checkpoints = String[]
        reduced = RationalSDP._apply_facial_reduction(
            problem,
            Int[],
            Dict(1 => reshape(Rational{BigInt}[1, 0], 2, 1));
            checkpoint = stage -> push!(checkpoints, stage),
        )
        @test [block.size for block in reduced.blocks] == [1, 1]
        @test length(reduced.objective_vector_raw) == 2
        @test size(reduced.A) == (1, 2)
        @test any(
            stage -> occursin(
                "eliminating exactly fixed zero cone coordinates",
                stage,
            ),
            checkpoints,
        )
        @test RationalSDP._lift_original_solution(
            reduced,
            Rational{BigInt}[5, 7],
        ) == Rational{BigInt}[5, 0, 0, 0, 0, 7]
    end

    @testset "Nullspace rank selection respects the selected float type" begin
        huge = big(10)^400
        exact_basis = Rational{BigInt}[
            huge//1 0//1
            0//1 1//1
        ]
        selected = RationalSDP._independent_nullspace_columns(
            exact_basis,
            [1, 2],
            BigFloat,
        )
        @test selected == exact_basis
    end

    @testset "Facial reduction cache validates matching faces" begin
        block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing, nothing, nothing],
            [1, 2, 3],
            [(1, 1), (2, 1), (2, 2)],
            [1, 3],
        )

        function cached_face_problem(objective::Vector{Rational{BigInt}}, rhs)
            A = Rational{BigInt}[
                0//1 1//1 0//1
                0//1 0//1 1//1
            ]
            b = Rational{BigInt}[RationalSDP._exact_rational(value) for value in rhs]
            return RationalSDP.ProblemData(
                MOI.VariableIndex[],
                [block],
                Int[],
                objective,
                0//1,
                objective,
                A,
                b,
                RationalSDP._solve_affine_system(A, b),
            )
        end

        cache_file = tempname()
        try
            problem = cached_face_problem(Rational{BigInt}[0//1, 0//1, 0//1], [0//1, 0//1])
            keep_basis = reshape(Rational{BigInt}[1//1, 0//1], 2, 1)
            reduction = RationalSDP._CertifiedFacialReduction(
                "unit test",
                Int[],
                Dict(1 => keep_basis),
            )
            save_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = false,
                facial_reduction_save_file = cache_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(save_opt)
            RationalSDP._record_successful_facial_reduction!(save_opt, problem, reduction)
            @test isfile(cache_file)
            saved_records = RationalSDP._read_facial_reduction_cache(cache_file)
            @test length(saved_records) == 1
            saved_reduction =
                RationalSDP._cached_facial_reduction(only(saved_records))
            @test saved_reduction !== nothing
            @test saved_reduction.exposing_slack !== nothing

            matching_problem =
                cached_face_problem(Rational{BigInt}[7//1, 0//1, 0//1], [0//1, 0//1])
            load_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = false,
                facial_reduction_load_file = cache_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(load_opt)
            loaded_details = RationalSDP._apply_loaded_facial_reductions(
                load_opt,
                matching_problem;
                return_details = true,
            )
            reduced_problem = loaded_details.problem
            @test loaded_details.applied == 1
            @test [block.size for block in reduced_problem.blocks] == [1]
            @test reduced_problem.objective_vector_raw == Rational{BigInt}[7//1]
            @test RationalSDP._lift_original_solution(
                reduced_problem,
                Rational{BigInt}[3//1],
            ) == Rational{BigInt}[3//1, 0//1, 0//1]

            invalid_problem =
                cached_face_problem(Rational{BigInt}[7//1, 0//1, 0//1], [0//1, 1//1])
            invalid_load_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = false,
                facial_reduction_load_file = cache_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(invalid_load_opt)
            invalid_details = RationalSDP._apply_loaded_facial_reductions(
                invalid_load_opt,
                invalid_problem;
                return_details = true,
            )
            unreduced_problem = invalid_details.problem
            @test invalid_details.applied == 0
            @test [block.size for block in unreduced_problem.blocks] == [2]
        finally
            rm(cache_file; force = true)
        end
    end

    @testset "Facial reduction cache validates joint multiblock faces" begin
        blocks = [
            RationalSDP.BlockStructure(
                1,
                Union{Nothing,MOI.VariableIndex}[nothing],
                [position],
                [(1, 1)],
                [position],
            ) for position in 1:2
        ]
        A = reshape(Rational{BigInt}[1//1, 1//1], 1, 2)
        b = Rational{BigInt}[0//1]
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            blocks,
            Int[],
            zeros(Rational{BigInt}, 2),
            0//1,
            zeros(Rational{BigInt}, 2),
            A,
            b,
            RationalSDP._solve_affine_system(A, b),
        )
        empty_keep_basis = zeros(Rational{BigInt}, 1, 0)
        reduction = RationalSDP._CertifiedFacialReduction(
            "joint cache regression",
            Int[],
            Dict(1 => empty_keep_basis, 2 => empty_keep_basis),
        )

        @test RationalSDP._cached_keep_basis_violation(
            problem,
            1,
            empty_keep_basis,
        ) !== nothing
        @test RationalSDP._cached_keep_basis_violation(
            problem,
            2,
            empty_keep_basis,
        ) !== nothing
        @test RationalSDP._cached_facial_reduction_violation(problem, reduction) === nothing
    end

    @testset "Facial reduction oracle fallback exposes scalar faces" begin
        block = RationalSDP.BlockStructure(
            1,
            Union{Nothing,MOI.VariableIndex}[nothing],
            [2],
            [(1, 1)],
            [2],
        )
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            [1],
            Rational{BigInt}[0//1, 0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1, 0//1],
            Rational{BigInt}[1//1 0//1 0//1; 0//1 1//1 0//1],
            Rational{BigInt}[0//1, 1//1],
            (
                Rational{BigInt}[0//1, 1//1, 0//1],
                reshape(Rational{BigInt}[0//1, 0//1, 1//1], 3, 1),
            ),
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            working_float_type = Float64,
            facial_reduction_float_type = Float64,
            phase1_hypatia_tol_rel_opt = big"1e-8",
        )
        @test !isempty(RationalSDP._phase1_hypatia_tolerance_kwargs(opt.settings, Float64))
        candidate = Float64[0.0, 1.0, 0.0]
        oracle_equalities = RationalSDP._facial_reduction_oracle_equalities(problem)
        @test oracle_equalities !== nothing
        equality_matrix, equality_rhs = oracle_equalities
        @test size(equality_matrix, 1) == 2
        @test equality_rhs == Rational{BigInt}[1//1, 0//1]

        @test isempty(
            RationalSDP._candidate_kernel_directions(
                opt,
                problem,
                1,
                RationalSDP._vector_to_matrix(candidate, block),
                Float64,
            ),
        )

        reduction = RationalSDP._facial_reduction_round(opt, problem, candidate, Float64)
        @test reduction !== nothing
        exposed_scalars = reduction.exposed_scalars
        keep_bases = reduction.keep_bases
        @test exposed_scalars == [1]
        @test isempty(keep_bases)

        reduced_problem = RationalSDP._apply_facial_reduction(problem, exposed_scalars, keep_bases)
        @test isempty(reduced_problem.positive_scalars)
        @test [reduced_block.size for reduced_block in reduced_problem.blocks] == [1]

        reduced_via_driver = RationalSDP._facially_reduce_problem(
            opt,
            problem,
            candidate,
            Float64,
        )
        @test isempty(reduced_via_driver.positive_scalars)
        @test [reduced_block.size for reduced_block in reduced_via_driver.blocks] == [1]
    end

    @testset "Phase I dual slack evidence certifies scalar faces first" begin
        block = RationalSDP.BlockStructure(
            1,
            Union{Nothing,MOI.VariableIndex}[nothing],
            [2],
            [(1, 1)],
            [2],
        )
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            [1],
            Rational{BigInt}[0//1, 0//1],
            0//1,
            Rational{BigInt}[0//1, 0//1],
            Rational{BigInt}[1//1 0//1; 0//1 1//1],
            Rational{BigInt}[0//1, 1//1],
            (
                Rational{BigInt}[0//1, 1//1],
                zeros(Rational{BigInt}, 2, 0),
            ),
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            working_float_type = Float64,
            facial_reduction_float_type = Float64,
        )
        candidate = Float64[0.0, 1.0]
        dual_slack = Float64[1.0, 0.0]

        evidence = RationalSDP._cheap_facial_reduction_evidence(candidate, dual_slack, Float64)
        @test [item.kind for item in evidence] == [:dual_slack, :boundary_primal]

        certified = RationalSDP._first_certified_facial_reduction(
            opt,
            problem,
            evidence,
            Float64,
        )
        @test certified !== nothing
        @test certified.source == "Phase I cone dual"
        @test certified.exposed_scalars == [1]
        @test isempty(certified.keep_bases)

        deferred_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            working_float_type = Float64,
            facial_reduction_float_type = Float64,
            facial_reduction_row_space_max_entries = 0,
        )
        @test RationalSDP._certify_facial_reduction_evidence(
            deferred_opt,
            problem,
            first(evidence),
            Float64,
        ) === nothing

        reduction = RationalSDP._certified_facial_reduction_from_initial_evidence(
            opt,
            problem,
            candidate,
            dual_slack,
            Float64,
        )
        @test reduction !== nothing
        exposed_scalars = reduction.exposed_scalars
        keep_bases = reduction.keep_bases
        @test exposed_scalars == [1]
        @test isempty(keep_bases)
    end

    @testset "Facial reduction merges certified complementary faces" begin
        block = RationalSDP.BlockStructure(
            3,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
            collect(1:6),
            [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)],
            [1, 3, 6],
        )
        A = Rational{BigInt}[
            1//1 0//1 0//1 0//1 0//1 0//1
            0//1 0//1 1//1 0//1 0//1 0//1
            0//1 0//1 0//1 0//1 0//1 1//1
        ]
        b = Rational{BigInt}[0//1, 0//1, 1//1]
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 6),
            0//1,
            zeros(Rational{BigInt}, 6),
            A,
            b,
            RationalSDP._solve_affine_system(A, b),
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_weighted_subspace_max_form_entries = 0,
            facial_reduction_weighted_subspace_max_affine_products = 0,
            facial_reduction_individual_max_affine_products = 0,
        )
        keep_without_first = Rational{BigInt}[0 0; 1 0; 0 1]
        keep_without_second = Rational{BigInt}[1 0; 0 0; 0 1]
        first = RationalSDP._CertifiedFacialReduction(
            "first",
            Int[],
            Dict(1 => keep_without_first),
            Rational{BigInt}[1, 0, 0, 0, 0, 0],
        )
        second = RationalSDP._CertifiedFacialReduction(
            "second",
            Int[],
            Dict(1 => keep_without_second),
            Rational{BigInt}[0, 0, 1, 0, 0, 0],
        )
        merged = RationalSDP._merge_certified_facial_reductions(opt, problem, [first, second])
        @test merged !== nothing
        @test occursin("first", merged.source)
        @test occursin("second", merged.source)
        @test size(merged.keep_bases[1], 2) == 1
        @test merged.exposing_slack == Rational{BigInt}[1, 0, 1, 0, 0, 0]
        reduced = RationalSDP._apply_facial_reduction(
            problem,
            merged.exposed_scalars,
            merged.keep_bases,
        )
        @test [reduced_block.size for reduced_block in reduced.blocks] == [1]
        @test reduced.affine !== nothing
    end

    @testset "Facial reduction rank expansion targets the residual face" begin
        block = RationalSDP.BlockStructure(
            3,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
            collect(1:6),
            [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)],
            [1, 3, 6],
        )
        A = Rational{BigInt}[
            1//1 0//1 0//1 0//1 0//1 0//1
            0//1 0//1 1//1 0//1 0//1 0//1
            0//1 0//1 0//1 0//1 0//1 1//1
        ]
        b = Rational{BigInt}[0//1, 0//1, 1//1]
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 6),
            0//1,
            zeros(Rational{BigInt}, 6),
            A,
            b,
            RationalSDP._solve_affine_system(A, b),
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_float_type = Float64,
            facial_reduction_rank_expansion_rounds = 1,
        )
        first = RationalSDP._CertifiedFacialReduction(
            "initial",
            Int[],
            Dict(1 => Rational{BigInt}[0 0; 1 0; 0 1]),
        )
        expanded = RationalSDP._facial_reduction_round_with_rank_expansion(
            opt,
            problem,
            first,
            Float64,
        )
        @test expanded !== nothing
        @test size(expanded.keep_bases[1], 2) == 1
        @test RationalSDP._cached_facial_reduction_violation(problem, expanded) === nothing

        cache_file = tempname()
        checkpoint_log_file = tempname()
        try
            save_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = true,
                facial_reduction_save_file = cache_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(save_opt)
            open(checkpoint_log_file, "w") do io
                redirect_stdout(io) do
                    RationalSDP._record_successful_facial_reduction!(
                        save_opt,
                        problem,
                        first,
                    )
                    RationalSDP._record_successful_facial_reduction!(
                        save_opt,
                        problem,
                        first,
                    )
                end
            end
            checkpoint_log = read(checkpoint_log_file, String)
            @test length(findall("checkpointed 1 reduction record(s)", checkpoint_log)) == 1

            load_opt = RationalSDP.Optimizer{Rational{BigInt}}(
                verbose = false,
                facial_reduction_float_type = Float64,
                facial_reduction_rank_expansion_rounds = 1,
                facial_reduction_save_file = cache_file,
                facial_reduction_load_file = cache_file,
            )
            RationalSDP._prepare_facial_reduction_cache!(load_opt)
            loaded = RationalSDP._apply_loaded_facial_reductions(
                load_opt,
                problem;
                return_details = true,
            )
            @test loaded.applied == 1
            @test [reduced_block.size for reduced_block in loaded.problem.blocks] == [1]

            checkpointed_records = RationalSDP._read_facial_reduction_cache(cache_file)
            @test length(checkpointed_records) == 1
            checkpointed = RationalSDP._cached_facial_reduction(only(checkpointed_records))
            @test checkpointed !== nothing
            @test size(checkpointed.keep_bases[1], 2) == 2
        finally
            rm(cache_file; force = true)
            rm(checkpoint_log_file; force = true)
        end
    end

    @testset "Sieve exact affine-row certificates" begin
        block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:3],
            collect(1:3),
            [(1, 1), (2, 1), (2, 2)],
            [1, 3],
        )
        make_problem(A, b; positive_scalars = Int[], dimension = size(A, 2)) =
            RationalSDP.ProblemData(
                MOI.VariableIndex[],
                [block],
                positive_scalars,
                zeros(Rational{BigInt}, dimension),
                0 // 1,
                zeros(Rational{BigInt}, dimension),
                A,
                b,
                RationalSDP._solve_affine_system(A, b),
            )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(verbose = false)

        problem = make_problem(Rational{BigInt}[1 2 1], Rational{BigInt}[0])
        positive = RationalSDP._sieve_row_reduction(
            opt,
            problem,
            Rational{BigInt}[1],
            "positive row",
        )
        negative = RationalSDP._sieve_row_reduction(
            opt,
            problem,
            Rational{BigInt}[-1],
            "negative row",
        )
        @test positive !== nothing
        @test size(positive.reduction.keep_bases[1], 2) == 1
        @test negative === nothing

        free_problem = make_problem(
            Rational{BigInt}[1 1 2 1],
            Rational{BigInt}[0];
            dimension = 4,
        )
        @test RationalSDP._sieve_row_reduction(
            opt,
            free_problem,
            Rational{BigInt}[1],
            "free row",
        ) === nothing
        rhs_problem = make_problem(Rational{BigInt}[1 2 1], Rational{BigInt}[1])
        @test RationalSDP._sieve_row_reduction(
            opt,
            rhs_problem,
            Rational{BigInt}[1],
            "nonzero rhs",
        ) === nothing
        indefinite_problem = make_problem(Rational{BigInt}[1 0 -1], Rational{BigInt}[0])
        @test RationalSDP._sieve_row_reduction(
            opt,
            indefinite_problem,
            Rational{BigInt}[1],
            "indefinite row",
        ) === nothing

        transformed_problem = make_problem(
            Rational{BigInt}[
                1 1 1
                1 1 0
            ],
            Rational{BigInt}[0, 0],
        )
        certificates = RationalSDP._sieve_facial_reduction_certificates(
            opt,
            transformed_problem,
        )
        transformed = filter(
            certificate -> occursin("transformed", certificate.source),
            certificates,
        )
        @test !isempty(transformed)
        @test any(
            certificate -> certificate.row[1:3] == Rational{BigInt}[0, 0, 1] &&
                           certificate.multiplier[1:2] == Rational{BigInt}[1, -1],
            transformed,
        )
        sieve_limited_opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_sieve_transform_max_entries = 0,
        )
        limited_certificates = RationalSDP._sieve_facial_reduction_certificates(
            sieve_limited_opt,
            transformed_problem,
        )
        @test all(
            certificate -> !occursin("transformed", certificate.source),
            limited_certificates,
        )

        mixed_block = RationalSDP.BlockStructure(
            2,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:3],
            collect(2:4),
            [(1, 1), (2, 1), (2, 2)],
            [2, 4],
        )
        mixed_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [mixed_block],
            [1],
            zeros(Rational{BigInt}, 4),
            0 // 1,
            zeros(Rational{BigInt}, 4),
            Rational{BigInt}[1 1 2 1],
            Rational{BigInt}[0],
            RationalSDP._solve_affine_system(
                Rational{BigInt}[1 1 2 1],
                Rational{BigInt}[0],
            ),
        )
        mixed = RationalSDP._sieve_row_reduction(
            opt,
            mixed_problem,
            Rational{BigInt}[1],
            "mixed row",
        )
        @test mixed !== nothing
        @test mixed.reduction.exposed_scalars == [1]
        @test size(mixed.reduction.keep_bases[1], 2) == 1

        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] + 2 * X[1, 2] + X[2, 2] == 0)
        @constraint(model, X[2, 2] == 1)
        @objective(model, Min, 0 // 1)
        optimize!(model)
        bridge_optimizer = getfield(backend(model), :optimizer)
        solved_opt = getfield(bridge_optimizer, :model)
        @test termination_status(model) == MOI.OPTIMAL
        @test RationalSDP.facial_reduction_statistics(solved_opt).oracle_attempts == 0

        cascade_block = RationalSDP.BlockStructure(
            3,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
            collect(1:6),
            [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)],
            [1, 3, 6],
        )
        cascade_A = Rational{BigInt}[
            1 0 0 0 0 0
            -1 0 1 0 0 0
        ]
        cascade_b = Rational{BigInt}[0, 0]
        cascade_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [cascade_block],
            Int[],
            zeros(Rational{BigInt}, 6),
            0 // 1,
            zeros(Rational{BigInt}, 6),
            cascade_A,
            cascade_b,
            RationalSDP._solve_affine_system(cascade_A, cascade_b),
        )
        cascaded = RationalSDP._sieve_facial_reduction_problem(opt, cascade_problem)
        @test [reduced_block.size for reduced_block in cascaded.blocks] == [1]

        # Compare row-space membership with the affine particular/nullspace
        # definition on several small, rank-deficient rational systems.
        for case in 1:8
            p = 5
            m = 4
            A_small = Rational{BigInt}[
                ((i + 2j + case) % 5 - 2) // 1 for i in 1:m, j in 1:p
            ]
            x0 = Rational{BigInt}[((case + j) % 4 - 1) // 1 for j in 1:p]
            b_small = A_small * x0
            small_problem = RationalSDP.ProblemData(
                MOI.VariableIndex[],
                RationalSDP.BlockStructure[],
                Int[],
                zeros(Rational{BigInt}, p),
                0 // 1,
                zeros(Rational{BigInt}, p),
                A_small,
                b_small,
                RationalSDP._solve_affine_system(A_small, b_small),
            )
            cache = RationalSDP._FacialReductionExactCache(small_problem)
            for query in 1:6
                indices = [j for j in 1:p if (j + query + case) % 3 == 0]
                values = Rational{BigInt}[((j + 2query + case) % 5 - 2) // 1 for j in indices]
                ell = zeros(Rational{BigInt}, p)
                ell[indices] = values
                particular, nullspace = small_problem.affine
                old_ok = dot(ell, particular) == 0 // 1 &&
                         all(iszero, transpose(ell) * nullspace)
                multiplier = RationalSDP._row_space_multiplier(
                    small_problem,
                    indices,
                    values;
                    cache,
                )
                @test (multiplier !== nothing) == old_ok
                if multiplier !== nothing
                    @test transpose(A_small) * multiplier == ell
                    @test dot(b_small, multiplier) == 0 // 1
                end
            end
        end

        # The selected row-space basis need not be symmetric.  Its cached
        # inverse solves square * coefficients = rhs_selected.
        A_nonsymmetric = Rational{BigInt}[
            1 2
            0 1
        ]
        b_nonsymmetric = Rational{BigInt}[0, 0]
        nonsymmetric_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            RationalSDP.BlockStructure[],
            Int[],
            zeros(Rational{BigInt}, 2),
            0 // 1,
            zeros(Rational{BigInt}, 2),
            A_nonsymmetric,
            b_nonsymmetric,
            RationalSDP._solve_affine_system(A_nonsymmetric, b_nonsymmetric),
        )
        @test RationalSDP._row_space_multiplier(
            nonsymmetric_problem,
            [1, 2],
            Rational{BigInt}[1, 2],
        ) == Rational{BigInt}[1, 0]

        empty_problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            RationalSDP.BlockStructure[],
            Int[],
            zeros(Rational{BigInt}, 3),
            0 // 1,
            zeros(Rational{BigInt}, 3),
            zeros(Rational{BigInt}, 0, 3),
            Rational{BigInt}[],
            RationalSDP._solve_affine_system(zeros(Rational{BigInt}, 0, 3), Rational{BigInt}[]),
        )
        @test RationalSDP._row_space_multiplier(
            empty_problem,
            [1],
            Rational{BigInt}[1],
        ) === nothing
    end

    @testset "Incremental affine restriction matches full elimination" begin
        A0 = Rational{BigInt}[
            1 2 0 1
            0 1 1 -1
            1 3 1 0
        ]
        b0 = Rational{BigInt}[2, 1, 3]
        affine0 = RationalSDP._solve_affine_system(A0, b0)
        @test affine0 !== nothing
        restrictions = Rational{BigInt}[
            1 0 0 0 1 0
            0 0 1 0 0 1
            0 1 0 0 0 0
        ]
        restriction_rhs = Rational{BigInt}[0, 0, 1]
        restriction_checkpoints = String[]
        incremental = RationalSDP._extend_and_restrict_affine_system(
            affine0,
            2,
            restrictions,
            restriction_rhs,
            checkpoint = stage -> push!(restriction_checkpoints, stage),
        )
        @test incremental !== nothing
        @test any(stage -> occursin("forming dense", stage), restriction_checkpoints)
        @test any(
            stage -> occursin("validating restricted affine basis", stage),
            restriction_checkpoints,
        )
        p_inc, N_inc = incremental
        p_ext = vcat(affine0[1], zeros(Rational{BigInt}, 2))
        N_ext = zeros(Rational{BigInt}, 6, size(affine0[2], 2) + 2)
        N_ext[1:4, 1:size(affine0[2], 2)] = affine0[2]
        N_ext[5:6, size(affine0[2], 2) + 1:end] = Matrix{Rational{BigInt}}(I, 2, 2)
        sparse_indices = [
            [column for column in axes(restrictions, 2) if !iszero(restrictions[row, column])]
            for row in axes(restrictions, 1)
        ]
        sparse_values = [
            Rational{BigInt}[restrictions[row, column] for column in row_indices] for
            (row, row_indices) in enumerate(sparse_indices)
        ]
        sparse_restrictions = RationalSDP._SparseAffineRestrictions(
            sparse_indices,
            sparse_values,
        )
        sparse_coordinate_rows, sparse_coordinate_rhs =
            RationalSDP._sparse_coordinate_restriction_system(
                affine0,
                2,
                sparse_restrictions,
                restriction_rhs,
            )
        @test sparse_coordinate_rows == restrictions * N_ext
        @test sparse_coordinate_rhs == restriction_rhs - restrictions * p_ext
        sparse_checkpoints = String[]
        sparse_incremental = RationalSDP._extend_and_restrict_affine_system(
            affine0,
            2,
            sparse_restrictions,
            restriction_rhs,
            checkpoint = stage -> push!(sparse_checkpoints, stage),
        )
        @test sparse_incremental !== nothing
        @test any(stage -> occursin("block-sparse", stage), sparse_checkpoints)
        @test sparse_incremental[1] == p_inc
        @test sparse_incremental[2] == N_inc
        skipped_validation_checkpoints = String[]
        sparse_without_redundant_validation =
            RationalSDP._extend_and_restrict_affine_system(
                affine0,
                2,
                sparse_restrictions,
                restriction_rhs;
                checkpoint = stage -> push!(skipped_validation_checkpoints, stage),
                settings = RationalSDP.Settings(
                    facial_reduction_sparse_affine_validation_max_products = 0,
                ),
            )
        @test sparse_without_redundant_validation == sparse_incremental
        @test any(
            stage -> occursin("skipping redundant sparse validation", stage),
            skipped_validation_checkpoints,
        )
        full = RationalSDP._solve_affine_system(
            restrictions * N_ext,
            restriction_rhs - restrictions * p_ext,
        )
        @test full !== nothing
        p_full_coord, N_full_coord = full
        @test p_inc == p_ext + N_ext * p_full_coord
        @test N_inc == N_ext * N_full_coord
        @test restrictions * p_inc == restriction_rhs
        @test restrictions * N_inc == zeros(Rational{BigInt}, 3, size(N_inc, 2))

        chunk_left = Rational{BigInt}[1 2 3; 4 5 6]
        chunk_right = Rational{BigInt}[1 0 2 1; 0 1 3 2; 1 1 0 4]
        @test RationalSDP._nemo_matrix_product_chunked(chunk_left, chunk_right, 2) ==
              chunk_left * chunk_right
        lift_stats = RationalSDP.FacialReductionStatistics()
        bounded_lift_error = try
            RationalSDP._with_facial_reduction_statistics(lift_stats) do
                RationalSDP._extend_and_restrict_affine_system(
                    affine0,
                    2,
                    restrictions,
                    restriction_rhs;
                    settings = RationalSDP.Settings(
                        facial_reduction_affine_lift_max_output_entries = 1,
                    ),
                )
            end
            nothing
        catch err
            err
        end
        @test bounded_lift_error isa RationalSDP.ExactLinearAlgebraError
        @test occursin("output entries", sprint(showerror, bounded_lift_error))
        @test lift_stats.affine_lifts_skipped_by_budget == 1

        inconsistent = RationalSDP._extend_and_restrict_affine_system(
            affine0,
            0,
            Rational{BigInt}[0 0 0 0],
            Rational{BigInt}[999],
        )
        @test inconsistent === nothing
    end

    @testset "SIRS facial reduction handles uncertified boundary candidates" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        set_optimizer_attribute(model, "facial_reduction_float_type", Float64)

        @polyvar S I R L
        N = 1

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
        L1 = 0 // 1

        dSdt = Lambda - beta * S * I / N - mu * S + alpha * R
        dIdt = beta * S * I / N - (delta + gamma + mu) * I
        dRdt = gamma * I - (alpha + mu) * R
        dLdt = beta * S / N - (delta + gamma + mu)

        basisV = monomials([S, I, R, L], 0:2)
        @variable(model, coeffsV[1:length(basisV)])
        V = dot(basisV, coeffsV)

        dVdt =
            differentiate(V, S) * dSdt +
            differentiate(V, I) * dIdt +
            differentiate(V, R) * dRdt +
            differentiate(V, L) * dLdt

        D = @set I >= 0 &&
                 S >= 0 &&
                 R >= 0 &&
                 I - I1 - I1 * L >= 0

        @constraint(model, V(S => S1, I => I1, R => R1, L => L1) == 0)
        @constraint(model, V >= (S - S1)^2 + (I - I1 - I1 * L) + (R - R1)^2, SOSCone(), domain = D)
        @constraint(model, -((S - S1)^2 + (I - I1)^2 + (R - R1)^2) >= dVdt, SOSCone(), domain = D)

        MOI.Utilities.attach_optimizer(backend(model))
        bridge_optimizer = getfield(backend(model), :optimizer)
        opt = getfield(bridge_optimizer, :model)

        problem = RationalSDP._extract_problem(opt)
        result1 = RationalSDP._phase1_anchor_attempt(opt, problem, Float64)
        @test result1.phase1_candidate !== nothing

        reduction1 = RationalSDP._facial_reduction_round(
            opt,
            problem,
            result1.phase1_candidate,
            Float64,
        )
        if reduction1 === nothing
            @test reduction1 === nothing
        else
            exposed_scalars1 = reduction1.exposed_scalars
            keep_bases1 = reduction1.keep_bases
            @test !isempty(keep_bases1)

            reduced_problem = RationalSDP._apply_facial_reduction(problem, exposed_scalars1, keep_bases1)
            result2 = RationalSDP._phase1_anchor_attempt(opt, reduced_problem, Float64)
            @test result2.phase1_candidate !== nothing

            for block in reduced_problem.blocks
                X = RationalSDP._vector_to_matrix(result2.phase1_candidate, block)
                eigs = eigvals(Symmetric((X + transpose(X)) / 2))
                @test minimum(eigs) > -1.0e-6
            end
        end
    end

    @testset "Uncertified facial reduction directions are skipped" begin
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
        optimize!(model)
        @test termination_status(model) == MOI.NUMERICAL_ERROR
        @test primal_status(model) == MOI.NO_SOLUTION
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

        @test termination_status(model) == MOI.ITERATION_LIMIT
        @test value(B) > 729//1
        @test value(B) < 730//1
        @test is_psd_exact(value.(Q))
        test_facial_reduction_statistics(model)
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

        @test termination_status(model) == MOI.ITERATION_LIMIT
        @test value(B) > 728//1
        @test value(B) < 730//1
        @test is_psd_exact(value.(Q))
        test_facial_reduction_statistics(model)
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

    @testset "KSE time average bound" begin
        model = rational_model(Rational{BigInt})
        instance = build_explicit_kse_model(3//4, model)
        optimize!(instance.model)
        test_facial_reduction_statistics(instance.model)

        @test termination_status(instance.model) == MOI.OPTIMAL
        @test all(iszero(value(coeff)) for expr in instance.certificate.expressions for coeff in coefficients(expr))
        @test is_psd_exact(value.(instance.certificate.Q_even))
        @test is_psd_exact(value.(instance.certificate.Q_odd))
        @test value(instance.B) > 280//100
        @test value(instance.B) < 281//100
    end
end

include("facial_reduction_statistics_tests.jl")

include("quasiconvex_parameter_tests.jl")
include("sumofsquares_tests.jl")

if "slow" in ARGS
    include("slowtests.jl")
end
