include("testutils.jl")
include("slowtest_helpers.jl")

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
        set_optimizer_attribute(model, "phase1_hypatia_target_margin", "0.02")
        set_optimizer_attribute(model, "phase1_hypatia_margin_upper", "1e-4")
        set_optimizer_attribute(model, "phase1_hypatia_min_margin_upper", "1e-8")
        set_optimizer_attribute(model, "phase1_hypatia_margin_shrink", "0.2")
        set_optimizer_attribute(model, "phase1_hypatia_objective_bias", "1e-12")
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
        @test get_optimizer_attribute(model, "phase1_outer_iterations") == 24
        @test get_optimizer_attribute(model, "phase1_backend") == :native
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == Float64
        @test get_optimizer_attribute(model, "phase1_hypatia_syssolver") == :qrchol_dense
        @test get_optimizer_attribute(model, "phase1_hypatia_target_margin") == big"0.02"
        @test get_optimizer_attribute(model, "phase1_hypatia_margin_upper") == big"1e-4"
        @test get_optimizer_attribute(model, "phase1_hypatia_min_margin_upper") == big"1e-8"
        @test get_optimizer_attribute(model, "phase1_hypatia_margin_shrink") == big"0.2"
        @test get_optimizer_attribute(model, "phase1_hypatia_objective_bias") == big"1e-12"
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
    end

    @testset "Working float type selection" begin
        model = rational_model(Rational{BigInt})
        @test get_optimizer_attribute(model, "working_float_type") == RationalSDP.Double64
        @test get_optimizer_attribute(model, "phase1_backend") == :hypatia
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == RationalSDP.Double64
        @test get_optimizer_attribute(model, "phase1_hypatia_syssolver") == :auto
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == RationalSDP.Double64

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

        set_optimizer_attribute(model, "working_float_type", RationalSDP.Double64)
        @test get_optimizer_attribute(model, "working_float_type") == RationalSDP.Double64
        @test get_optimizer_attribute(model, "phase1_hypatia_float_type") == RationalSDP.Double64
        @test get_optimizer_attribute(model, "facial_reduction_float_type") == RationalSDP.Double64
    end

    @testset "Hypatia Phase I system solver selection" begin
        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(RationalSDP.Settings(), Float64)
        @test syssolver isa RationalSDP.Hypatia.Solvers.SymIndefSparseSystemSolver{Float64}
        @test !use_dense_model
        @test !preprocess

        settings = RationalSDP.Settings(phase1_hypatia_syssolver = :qrchol_dense)
        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(settings, RationalSDP.Double64)
        @test syssolver isa RationalSDP.Hypatia.Solvers.QRCholDenseSystemSolver{RationalSDP.Double64}
        @test use_dense_model
        @test preprocess

        settings = RationalSDP.Settings(phase1_hypatia_syssolver = :symindef_indirect)
        syssolver, use_dense_model, preprocess =
            RationalSDP._hypatia_phase1_syssolver(settings, RationalSDP.Double64)
        @test syssolver isa RationalSDP.Hypatia.Solvers.SymIndefIndirectSystemSolver{RationalSDP.Double64}
        @test !use_dense_model
        @test !preprocess

        settings = RationalSDP.Settings(phase1_hypatia_syssolver = :symindef_sparse)
        @test_throws ErrorException RationalSDP._hypatia_phase1_syssolver(
            settings,
            RationalSDP.Double64,
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
        affine = RationalSDP._solve_affine_system(A, b)
        @test affine !== nothing
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

        phase2_nullspace = RationalSDP._phase2_nullspace(problem)

        @test RationalSDP._phase2_relevant_positions(problem) == [1, 2]
        @test size(phase2_nullspace) == (4, 2)
        @test phase2_nullspace[1:2, :] == Matrix{Rational{BigInt}}(I, 2, 2)
        @test all(iszero, phase2_nullspace[3:4, :])
    end

    @testset "Facial reduction helper regressions" begin
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

        function legacy_exposing_column_directions(
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
            zeros(Rational{BigInt}, 0, 6),
            Rational{BigInt}[],
            (
                triangle_entries(fill(1//1, 3, 3)),
                zeros(Rational{BigInt}, 6, 0),
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
        legacy_directions = legacy_exposing_column_directions(
            pivot_opt,
            rank_one_problem,
            1,
            exposing_slack,
            Float64,
        )
        @test isempty(legacy_directions)

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
            zeros(Rational{BigInt}, 0, 3),
            Rational{BigInt}[],
            (
                Rational{BigInt}[1//1, -1//1, 1//1],
                zeros(Rational{BigInt}, 3, 0),
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
        @test exact_cache.block_affine_slices[1] === nothing
        @test exact_cache.block_exact_directions[1] === nothing
        cached_directions = RationalSDP._exact_block_nullspace_directions(
            exact_boundary_problem,
            block;
            cache = exact_cache,
            block_index = 1,
        )
        @test cached_directions == [Rational{BigInt}[1//1, 1//1]]
        @test exact_cache.block_affine_slices[1] !== nothing
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
        @test exact_cache.A_transpose !== nothing

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
        exposed_scalars, keep_bases = reduction
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

        reduction = RationalSDP._certified_facial_reduction_from_initial_evidence(
            opt,
            problem,
            candidate,
            dual_slack,
            Float64,
        )
        @test reduction !== nothing
        exposed_scalars, keep_bases = reduction
        @test exposed_scalars == [1]
        @test isempty(keep_bases)
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
            exposed_scalars1, keep_bases1 = reduction1
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

    @testset "KSE time average bound" begin
        model = rational_model(Rational{BigInt})
        instance = build_explicit_kse_model(3//4, model)
        optimize!(instance.model)

        @test termination_status(instance.model) == MOI.OPTIMAL
        @test all(iszero(value(coeff)) for expr in instance.certificate.expressions for coeff in coefficients(expr))
        @test is_psd_exact(value.(instance.certificate.Q_even))
        @test is_psd_exact(value.(instance.certificate.Q_odd))
        @test value(instance.B) > 280//100
        @test value(instance.B) < 281//100
    end
end

include("quasiconvex_parameter_tests.jl")
include("sumofsquares_tests.jl")

if "slow" in ARGS
    include("slowtests.jl")
end
