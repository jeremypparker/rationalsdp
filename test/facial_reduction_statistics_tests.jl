@testset "Facial-reduction statistics and synthetic corpus" begin
    function synthetic_face_problem(block_sizes::Vector{Int})
        blocks = RationalSDP.BlockStructure[]
        positions = Int[]
        next_position = 1
        particular = Rational{BigInt}[]
        for (block_index, block_size) in enumerate(block_sizes)
            local_positions = RationalSDP._triangle_positions(block_size)
            global_positions = collect(next_position:(next_position + length(local_positions) - 1))
            diagonal_positions = [
                global_positions[index] for
                (index, (i, j)) in enumerate(local_positions) if i == j
            ]
            push!(blocks, RationalSDP.BlockStructure(
                block_size,
                Union{Nothing,MOI.VariableIndex}[nothing for _ in local_positions],
                global_positions,
                local_positions,
                diagonal_positions,
            ))
            append!(positions, global_positions)
            for (i, j) in local_positions
                push!(particular, i == 1 && j == 1 ? 1//1 : 0//1)
            end
            next_position += length(local_positions)
        end
        dimension = length(particular)
        return RationalSDP.ProblemData(
            MOI.VariableIndex[],
            blocks,
            Int[],
            zeros(Rational{BigInt}, dimension),
            0//1,
            zeros(Rational{BigInt}, dimension),
            zeros(Rational{BigInt}, 0, dimension),
            Rational{BigInt}[],
            (particular, zeros(Rational{BigInt}, dimension, 0)),
        )
    end

    @testset "statistics are exposed for a solve" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        set_optimizer_attribute(model, "phase1_backend", :native)
        @variable(model, X[1:2, 1:2], PSD)
        @objective(model, Min, X[1, 1])
        optimize!(model)

        bridge_optimizer = getfield(backend(model), :optimizer)
        opt = getfield(bridge_optimizer, :model)
        stats = RationalSDP.facial_reduction_statistics(opt)
        @test stats.phase1_attempts >= 1
        @test stats.phase1_time_sec >= 0
        @test stats.oracle_attempts >= 0
        @test stats.oracle_iterations >= 0
        @test stats.exact_rref_calls == length(stats.exact_rref_dimensions)
        @test all(rows >= 0 && columns >= 0 for (rows, columns) in stats.exact_rref_dimensions)
        @test stats.exact_rref_time_sec >= 0
        @test stats.exact_row_space_checks >= 0
        @test stats.exact_certificate_checks >= 0
        @test stats.exact_row_space_check_time_sec >= 0
        @test stats.exact_certificate_time_sec >= 0
        @test all(size >= 0 && count >= 0 for (size, count) in stats.psd_eigendecompositions_by_block_size)
        @test stats.affine_cache_peak_bytes > 0
        @test stats.facial_reduction_cache_peak_bytes >= 0
        @test stats.certified_directions_proposed >=
              stats.certified_directions_accepted
        @test stats.tentative_directions_proposed >=
              stats.tentative_directions_accepted
        @test stats.rational_subspace_charts_attempted >= 0
        @test stats.rational_projectors_attempted >= 0
        @test stats.precision_escalations_attempted >= 0
        @test stats.tentative_batches_skipped_by_budget >= 0
        @test stats.affine_lifts_skipped_by_budget >= 0
        @test all(removed > 0 for removed in stats.cone_dimension_removed_per_round)
        @test termination_status(model) == MOI.OPTIMAL
    end

    @testset "many independent PSD faces" begin
        problem = synthetic_face_problem([2, 2, 2])
        current = problem
        for block_index in 1:3
            keep_basis = reshape(Rational{BigInt}[1//1, 0//1], 2, 1)
            current = RationalSDP._apply_facial_reduction(
                current,
                Int[],
                Dict(block_index => keep_basis),
            )
        end
        @test RationalSDP._barrier_dimension(current) == 3
        particular, nullspace = current.affine
        @test current.A * particular == current.b
        @test current.A * nullspace == zeros(Rational{BigInt}, size(current.A, 1), size(nullspace, 2))
    end

    @testset "high-dimensional faces exposed one direction at a time" begin
        current = synthetic_face_problem([4])
        for removed_dimension in 4:-1:2
            keep_basis = zeros(Rational{BigInt}, removed_dimension, removed_dimension - 1)
            for index in 1:(removed_dimension - 1)
                keep_basis[index, index] = 1//1
            end
            current = RationalSDP._apply_facial_reduction(
                current,
                Int[],
                Dict(1 => keep_basis),
            )
        end
        @test RationalSDP._barrier_dimension(current) == 1
    end

    @testset "compatible and incompatible directions are distinguished" begin
        problem = synthetic_face_problem([3])
        block = problem.blocks[1]
        cache = RationalSDP._FacialReductionExactCache(problem)
        compatible = RationalSDP._block_face_direction_certificate(
            problem,
            block,
            Rational{BigInt}[0//1, 1//1, 0//1];
            cache,
        )
        incompatible = RationalSDP._block_face_direction_certificate(
            problem,
            block,
            Rational{BigInt}[1//1, 1//1, 0//1];
            cache,
        )
        @test compatible.kind != :none
        @test incompatible.kind == :none

        opt = RationalSDP.Optimizer{Rational{BigInt}}(verbose = false)
        stats = RationalSDP.FacialReductionStatistics()
        tentative_problem = RationalSDP._with_facial_reduction_statistics(stats) do
            RationalSDP._tentative_feasibility_search_problem(
                opt,
                problem,
                Float64[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                Float64,
            )
        end
        @test tentative_problem !== nothing
        @test isempty(opt.facial_reduction_save_records)
        @test stats.tentative_directions_proposed >= 1
        @test stats.tentative_directions_accepted >= 2
        @test stats.tentative_restrictions_applied == 0
    end

    @testset "tentative batches span blocks and deduplicate directions" begin
        problem = synthetic_face_problem([3, 3])
        directions = [
            RationalSDP._TentativeFaceDirection{Float64}(1, Rational{BigInt}[0//1, 1//1, 0//1], 0.0, 0.0, 0.0),
            RationalSDP._TentativeFaceDirection{Float64}(1, Rational{BigInt}[0//1, 2//1, 0//1], 0.0, 0.0, 0.1),
            RationalSDP._TentativeFaceDirection{Float64}(2, Rational{BigInt}[0//1, 1//1, 0//1], 0.0, 0.0, 0.0),
        ]
        keep_bases = RationalSDP._tentative_candidate_keep_bases(problem, directions)
        @test size(keep_bases[1], 2) == 2
        @test size(keep_bases[2], 2) == 2
        reduced = RationalSDP._tentative_batch_problem(problem, directions)
        @test reduced.affine !== nothing
        @test RationalSDP._barrier_dimension(reduced) == 4
    end

    @testset "oversized tentative batches admit one direction incrementally" begin
        problem = synthetic_face_problem([3, 3])
        settings = RationalSDP.Settings(
            facial_reduction_tentative_max_directions = 1,
            facial_reduction_tentative_max_coordinate_entries = typemax(Int),
            facial_reduction_tentative_max_lift_products = typemax(Int),
            facial_reduction_tentative_max_estimated_bytes = typemax(Int),
            facial_reduction_affine_lift_max_output_entries = typemax(Int),
            facial_reduction_affine_lift_max_estimated_bytes = typemax(Int),
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(
            verbose = false,
            facial_reduction_tentative_max_directions = 1,
            facial_reduction_tentative_max_coordinate_entries = typemax(Int),
            facial_reduction_tentative_max_lift_products = typemax(Int),
            facial_reduction_tentative_max_estimated_bytes = typemax(Int),
            facial_reduction_affine_lift_max_output_entries = typemax(Int),
            facial_reduction_affine_lift_max_estimated_bytes = typemax(Int),
        )
        directions = [
            RationalSDP._TentativeFaceDirection{Float64}(1, Rational{BigInt}[0, 1, 0], 0.0, 0.0, 0.0),
            RationalSDP._TentativeFaceDirection{Float64}(2, Rational{BigInt}[0, 1, 0], 0.0, 0.0, 0.0),
        ]
        work = RationalSDP._tentative_batch_work(problem, directions)
        @test !RationalSDP._tentative_batch_within_budget(work, settings)

        stats = RationalSDP.FacialReductionStatistics()
        reduced = RationalSDP._with_facial_reduction_statistics(stats) do
            RationalSDP._tentative_feasibility_search_problem(
                opt,
                problem,
                vcat(Float64[1, 0, 0, 0, 0, 0], Float64[1, 0, 0, 0, 0, 0]),
                Float64,
            )
        end
        @test reduced !== nothing
        @test RationalSDP._barrier_dimension(reduced) == 5
        @test stats.tentative_batches_skipped_by_budget == 1
        @test stats.tentative_directions_accepted == 1
    end

    @testset "jointly inconsistent tentative restrictions roll back greedily" begin
        block = RationalSDP.BlockStructure(
            3,
            Union{Nothing,MOI.VariableIndex}[nothing for _ in 1:6],
            collect(1:6),
            RationalSDP._triangle_positions(3),
            [1, 3, 6],
        )
        A = Rational{BigInt}[
            1//1 0//1 1//1 0//1 0//1 0//1
            0//1 0//1 0//1 0//1 0//1 1//1
        ]
        b = Rational{BigInt}[1//1, 0//1]
        affine = RationalSDP._solve_affine_system(A, b)
        problem = RationalSDP.ProblemData(
            MOI.VariableIndex[],
            [block],
            Int[],
            zeros(Rational{BigInt}, 6),
            0//1,
            zeros(Rational{BigInt}, 6),
            A,
            b,
            affine,
        )
        opt = RationalSDP.Optimizer{Rational{BigInt}}(verbose = false)
        stats = RationalSDP.FacialReductionStatistics()
        reduced = RationalSDP._with_facial_reduction_statistics(stats) do
            RationalSDP._tentative_feasibility_search_problem(
                opt,
                problem,
                Float64[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                Float64,
            )
        end
        @test reduced !== nothing
        @test reduced.affine !== nothing
        @test stats.tentative_directions_proposed == 2
        @test stats.tentative_directions_accepted == 1
        @test stats.tentative_directions_rejected == 1
        @test RationalSDP._barrier_dimension(reduced) == 1
        reduced_particular, _ = reduced.affine
        original_point = reduced_particular[1:length(problem.objective_vector_raw)]
        @test RationalSDP._exact_primal_feasibility(problem, original_point).ok
        @test RationalSDP._positive_semidefinite_exact(
            RationalSDP._vector_to_matrix(original_point, problem.blocks[1]),
        )
    end

    @testset "many scalar cones and large affine nullity" begin
        model = rational_model(Rational{BigInt})
        set_optimizer_attribute(model, "working_float_type", Float64)
        set_optimizer_attribute(model, "phase1_backend", :native)
        @variable(model, x[1:16] >= 0)
        @variable(model, X[1:2, 1:2], PSD)
        @constraint(model, X[1, 1] == 1//1)
        @constraint(model, X[2, 2] == 1//1)
        @constraint(model, sum(x) == 1//1)
        @objective(model, Min, 0//1)
        optimize!(model)
        bridge_optimizer = getfield(backend(model), :optimizer)
        opt = getfield(bridge_optimizer, :model)
        stats = RationalSDP.facial_reduction_statistics(opt)
        @test stats.exact_rref_calls >= 0
        @test stats.exact_rref_calls == length(stats.exact_rref_dimensions)
        @test termination_status(model) == MOI.OPTIMAL
    end
end
