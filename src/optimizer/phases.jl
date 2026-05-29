# Phase I and Phase II routines, including exact recovery and Hypatia integration.

function _recovery_tolerances(settings::Settings, ::Type{F}) where {F<:AbstractFloat}
    tolerances = F[]
    tolerance = max(F(1.0e-8), sqrt(_to_working_float(F, settings.rational_tolerance)))
    final_tolerance = _to_working_float(F, settings.rational_tolerance)
    shrink = _to_working_float(F, settings.recovery_tolerance_shrink)
    zero(F) < shrink < one(F) || error("recovery_tolerance_shrink must lie strictly between 0 and 1.")
    while tolerance > final_tolerance
        push!(tolerances, tolerance)
        tolerance *= shrink
    end
    if isempty(tolerances) || tolerances[end] != final_tolerance
        push!(tolerances, final_tolerance)
    end
    return tolerances
end

function _recovery_tolerances(settings::Settings)
    return _recovery_tolerances(settings, _working_float_type(settings))
end

function _newton_phase1!(
    opt::Optimizer,
    x::Vector{F},
    problem::ProblemData,
    numeric_blocks::Vector{NumericBlock},
    A_big::AbstractMatrix{F},
    At_big::AbstractMatrix{F},
    AtA::Union{Nothing,Matrix{F}},
    b_big::Vector{F},
    penalty::F,
    center::Vector{F},
    numeric_settings,
) where {F<:AbstractFloat}
    settings = numeric_settings
    center_weight = settings.phase1_center_weight
    use_iterative_linear_solver =
        settings.iterative_linear_solver && length(x) >= settings.iterative_solver_min_dimension
    for iteration in 1:settings.max_iterations
        barrier_value = zero(F)
        barrier_grad = zeros(F, length(x))
        barrier_hess = nothing
        barrier_caches = PSDBarrierCache{F}[]
        if use_iterative_linear_solver
            barrier_value, barrier_grad, _, barrier_caches = _barrier_value_grad_diag(
                x,
                numeric_blocks,
                problem.positive_scalars,
            )
        else
            barrier_value, barrier_grad, barrier_hess = _barrier_value_grad_hess(
                x,
                numeric_blocks,
                problem.positive_scalars,
                settings,
            )
        end
        residual = A_big * x - b_big
        deviation = x - center
        residual_norm = _max_abs(residual)
        value = barrier_value + penalty * dot(residual, residual) / 2 + center_weight * dot(deviation, deviation) / 2
        grad = barrier_grad + penalty * (At_big * residual) + center_weight * deviation
        grad_norm = _max_abs(grad)
        if iteration == 1 || iteration % settings.inner_log_frequency == 0
            _log_newton(
                opt,
                "phase I newton $(iteration): residual=$(_format_metric(residual_norm)), grad=$(_format_metric(grad_norm))",
            )
        end
        if grad_norm <= settings.gradient_tolerance
            return x
        end
        if use_iterative_linear_solver
            direction = _schur_phase1_direction(
                opt,
                grad,
                A_big,
                At_big,
                penalty,
                barrier_caches,
                problem.positive_scalars,
                x,
                center_weight,
            )
        else
            hess = barrier_hess + something(AtA, At_big * A_big) * penalty
            for diagonal in 1:length(x)
                hess[diagonal, diagonal] += center_weight
            end
            direction = -_solve_spd_system(hess, grad)
        end
        directional_derivative = dot(grad, direction)
        step = min(
            one(F),
            _max_step_to_boundary(
                x,
                direction,
                numeric_blocks,
                problem.positive_scalars,
                settings.boundary_fraction,
            ),
        )
        accepted = false
        while step >= settings.min_step
            trial = x + step * direction
            if _strictly_interior_numeric(trial, numeric_blocks, problem.positive_scalars)
                trial_barrier = _barrier_value_only(
                    trial,
                    numeric_blocks,
                    problem.positive_scalars,
                )
                trial_residual = A_big * trial - b_big
                trial_deviation = trial - center
                trial_value = trial_barrier + penalty * dot(trial_residual, trial_residual) / 2 + center_weight * dot(trial_deviation, trial_deviation) / 2
                if trial_value <= value + settings.armijo_fraction * step * directional_derivative
                    x .= trial
                    accepted = true
                    break
                end
            end
            step *= settings.line_search_shrink
        end
        accepted || return x
    end
    return x
end

function _project_exact_solution(
    coefficients::Vector{F},
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational};
    tolerance::F,
) where {F<:AbstractFloat}
    if isempty(coefficients)
        return particular
    end
    rational_coefficients = [
        rationalize(BigInt, value; tol = tolerance) for value in coefficients
    ]
    return particular + nullspace * rational_coefficients
end

function _affine_coordinates(
    x_approx::Vector{F},
    numeric_affine::NumericAffineData{F},
) where {F<:AbstractFloat}
    if size(numeric_affine.exact_nullspace, 2) == 0
        return F[]
    end
    return _nullspace_factor!(numeric_affine) \ (x_approx .- numeric_affine.particular)
end

function _best_exact_interior_on_segment(
    anchor::Vector{ExactRational},
    candidate::Vector{ExactRational},
    problem::ProblemData;
    max_bisections::Int,
)
    candidate_objective = _exact_phase2_objective_value(problem, candidate)
    anchor_objective = _exact_phase2_objective_value(problem, anchor)
    candidate_objective < anchor_objective || return anchor

    if _strictly_interior_exact(candidate, problem.blocks, problem.positive_scalars)
        return candidate
    end

    delta = candidate - anchor
    any(!iszero, delta) || return anchor

    lower = 0 // 1
    upper = 1 // 1
    best = anchor
    for _ in 1:max_bisections
        weight = (lower + upper) / 2
        trial = anchor + weight * delta
        if _strictly_interior_exact(trial, problem.blocks, problem.positive_scalars)
            lower = weight
            best = trial
        else
            upper = weight
        end
    end

    return best
end

function _phase2_exact_refinement(
    x_approx::Vector{F},
    anchor::Vector{ExactRational},
    problem::ProblemData,
    settings::Settings,
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    numeric_affine::NumericAffineData{F},
) where {F<:AbstractFloat}
    coefficients = _affine_coordinates(x_approx, numeric_affine)
    best = anchor

    for tolerance in _recovery_tolerances(settings, F)
        candidate = _project_exact_solution(
            coefficients,
            particular,
            nullspace;
            tolerance = tolerance,
        )
        refined = _best_exact_interior_on_segment(
            anchor,
            candidate,
            problem;
            max_bisections = settings.exact_refinement_bisections,
        )
        if _exact_phase2_objective_value(problem, refined) <
           _exact_phase2_objective_value(problem, best)
            best = refined
        end
    end

    return best
end

function _phase1_exact_recovery_log(opt::Optimizer, message::AbstractString)
    opt.settings.phase1_exact_recovery_diagnostics || return
    _log(opt, "Phase I exact recovery: " * message)
    return
end

function _phase1_exact_recovery_elapsed(start_time)
    return @sprintf("%.2f", (time_ns() - start_time) / 1.0e9)
end

function _phase1_exact_recovery_rationalize_coordinates(
    opt::Optimizer,
    coordinates::AbstractVector{F},
    tolerance::F,
    context::AbstractString,
) where {F<:AbstractFloat}
    if !opt.settings.phase1_exact_recovery_diagnostics
        return [rationalize(BigInt, value; tol = tolerance) for value in coordinates]
    end

    start_time = time_ns()
    rational_coordinates = Vector{ExactRational}(undef, length(coordinates))
    log_frequency = max(1, div(length(coordinates), 8))
    for index in eachindex(coordinates)
        rational_coordinates[index] = rationalize(BigInt, coordinates[index]; tol = tolerance)
        if index == 1 || index == length(coordinates) || index % log_frequency == 0
            _phase1_exact_recovery_log(
                opt,
                "$(context): rationalized coordinate $(index)/$(length(coordinates)) in $(_phase1_exact_recovery_elapsed(start_time))s",
            )
        end
    end
    return rational_coordinates
end

function _project_exact_solution_from_rational_coordinates_recovery(
    opt::Optimizer,
    rational_coordinates::Vector{ExactRational},
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    context::AbstractString,
)
    if !opt.settings.phase1_exact_recovery_diagnostics
        return particular + nullspace * rational_coordinates
    end

    row_count, column_count = size(nullspace)
    start_time = time_ns()
    result = copy(particular)
    nonzero_coordinates = count(!iszero, rational_coordinates)
    _phase1_exact_recovery_log(
        opt,
        "$(context): exact projection rows=$(row_count), cols=$(column_count), nonzero_coordinates=$(nonzero_coordinates)",
    )
    log_frequency = max(1, div(row_count, 8))
    for row in 1:row_count
        accumulator = result[row]
        row_terms = 0
        @inbounds for column in 1:column_count
            coefficient = rational_coordinates[column]
            iszero(coefficient) && continue
            entry = nullspace[row, column]
            iszero(entry) && continue
            accumulator += entry * coefficient
            row_terms += 1
        end
        result[row] = accumulator
        if row == 1 || row == row_count || row % log_frequency == 0
            _phase1_exact_recovery_log(
                opt,
                "$(context): projected row $(row)/$(row_count), row_terms=$(row_terms), elapsed=$(_phase1_exact_recovery_elapsed(start_time))s",
            )
        end
    end
    return result
end

function _positive_definite_exact_recovery_check(
    opt::Optimizer,
    matrix::Matrix{ExactRational},
    block_index::Int,
    context::AbstractString,
)
    size(matrix, 1) == size(matrix, 2) ||
        return (ok = false, reason = "block $(block_index) is not square")
    n = size(matrix, 1)
    n == 0 && return (ok = true, reason = "empty block")
    any(matrix .!= transpose(matrix)) &&
        return (ok = false, reason = "block $(block_index) is not symmetric")
    for diagonal in 1:n
        value = matrix[diagonal, diagonal]
        if !(value > 0)
            return (
                ok = false,
                reason = "block $(block_index) diagonal $(diagonal) is not positive: $(_format_exact_rational_compact(value))",
            )
        end
    end

    start_time = time_ns()
    remainder = copy(matrix)
    active = collect(1:n)
    pivot_count = 0
    log_frequency = max(0, opt.settings.phase1_exact_recovery_pivot_log_frequency)
    while !isempty(active)
        for diagonal in eachindex(active)
            value = remainder[diagonal, diagonal]
            if !(value > 0)
                return (
                    ok = false,
                    reason = "block $(block_index) Schur diagonal $(active[diagonal]) is not positive after $(pivot_count) pivot(s): $(_format_exact_rational_compact(value))",
                )
            end
        end
        pivot_position = findfirst(index -> remainder[index, index] > 0, eachindex(active))
        pivot_position === nothing &&
            return (ok = false, reason = "block $(block_index) has no positive exact pivot")
        if pivot_position != 1
            permutation = [pivot_position; setdiff(collect(1:length(active)), pivot_position)]
            remainder = remainder[permutation, permutation]
            active = active[permutation]
        end

        pivot = remainder[1, 1]
        pivot > 0 ||
            return (ok = false, reason = "block $(block_index) pivot is not positive: $(_format_exact_rational_compact(pivot))")
        pivot_count += 1
        if opt.settings.phase1_exact_recovery_diagnostics &&
           (pivot_count == 1 ||
            length(active) == 1 ||
            (log_frequency > 0 && pivot_count % log_frequency == 0))
            _log(
                opt,
                "Phase I exact recovery: $(context), block $(block_index) pivot $(pivot_count)/$(n), active=$(length(active)), $(_format_exact_rational_size(pivot)), elapsed=$(_phase1_exact_recovery_elapsed(start_time))s",
            )
        end
        length(active) == 1 && return (ok = true, reason = "positive definite")
        trailing = Matrix{ExactRational}(undef, length(active) - 1, length(active) - 1)
        for row in 2:length(active), column in 2:length(active)
            trailing[row - 1, column - 1] =
                remainder[row, column] - remainder[row, 1] * remainder[1, column] / pivot
        end
        remainder = trailing
        active = active[2:end]
    end

    return (ok = true, reason = "positive definite")
end

function _strictly_interior_exact_recovery_check(
    opt::Optimizer,
    x::Vector{ExactRational},
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    context::AbstractString,
)
    if !opt.settings.phase1_exact_recovery_diagnostics
        return _strictly_interior_exact(x, blocks, positive_scalars)
    end

    for index in positive_scalars
        if !(x[index] > 0)
            _phase1_exact_recovery_log(
                opt,
                "$(context) failed: scalar slack at position $(index) is not positive: $(_format_exact_rational_compact(x[index]))",
            )
            return false
        end
    end

    for (block_index, block) in enumerate(blocks)
        block_start = time_ns()
        _phase1_exact_recovery_log(
            opt,
            "$(context): checking exact positive definiteness of block $(block_index) (size=$(block.size))",
        )
        status = _positive_definite_exact_recovery_check(
            opt,
            _vector_to_matrix(x, block),
            block_index,
            context,
        )
        if !status.ok
            _phase1_exact_recovery_log(
                opt,
                "$(context) failed after $(_phase1_exact_recovery_elapsed(block_start))s: $(status.reason)",
            )
            return false
        end
        _phase1_exact_recovery_log(
            opt,
            "$(context): block $(block_index) passed in $(_phase1_exact_recovery_elapsed(block_start))s",
        )
    end
    return true
end

function _phase1_exact_feasible_point(
    opt::Optimizer,
    x_phase1::Vector{F},
    problem::ProblemData,
    settings::Settings,
    numeric_affine::NumericAffineData{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    particular, nullspace = problem.affine
    coefficients = _affine_coordinates(x_phase1, numeric_affine)
    tolerances = _recovery_tolerances(settings, F)
    _phase1_exact_recovery_log(
        opt,
        "fallback from full candidate with $(length(coefficients)) affine coordinate(s) and $(length(tolerances)) tolerance attempt(s)",
    )
    for (attempt_index, tolerance) in enumerate(tolerances)
        context = "fallback attempt $(attempt_index)/$(length(tolerances)), tol=$(_format_metric(tolerance))"
        attempt_start = time_ns()
        _phase1_exact_recovery_log(opt, "$(context): rationalizing coordinates")
        rational_coefficients = _phase1_exact_recovery_rationalize_coordinates(
            opt,
            coefficients,
            tolerance,
            context,
        )
        _phase1_exact_recovery_log(
            opt,
            "$(context): rationalized coordinates in $(_phase1_exact_recovery_elapsed(attempt_start))s; projecting exact candidate",
        )
        candidate = _project_exact_solution_from_rational_coordinates_recovery(
            opt,
            rational_coefficients,
            particular,
            nullspace,
            context,
        )
        _phase1_exact_recovery_log(
            opt,
            "$(context): projection finished in $(_phase1_exact_recovery_elapsed(attempt_start))s",
        )
        if _strictly_interior_exact_recovery_check(
            opt,
            candidate,
            problem.blocks,
            problem.positive_scalars,
            context,
        )
            _phase1_exact_recovery_log(opt, "$(context): recovered exact strict interior point")
            return candidate
        end
        _phase1_exact_recovery_log(
            opt,
            "$(context): no exact strict interior point after $(_phase1_exact_recovery_elapsed(attempt_start))s",
        )
    end
    _phase1_exact_recovery_log(opt, "fallback failed")
    return nothing
end

function _phase1_exact_feasible_point_from_coordinates(
    opt::Optimizer,
    coordinates::AbstractVector{F},
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    problem::ProblemData,
    settings::Settings,
) where {F<:AbstractFloat}
    active_positions = _phase1_active_positions(problem)
    position_map = Dict(position => local_index for (local_index, position) in enumerate(active_positions))
    active_blocks = BlockStructure[
        BlockStructure(
            block.size,
            block.variables,
            [position_map[position] for position in block.global_positions],
            block.local_positions,
            [position_map[position] for position in block.diagonal_positions],
        ) for block in problem.blocks
    ]
    active_positive_scalars = [position_map[position] for position in problem.positive_scalars]
    particular_active = particular[active_positions]
    nullspace_active = nullspace[active_positions, :]
    tolerances = _recovery_tolerances(settings, F)
    _phase1_exact_recovery_log(
        opt,
        "coordinate recovery with $(length(coordinates)) coordinate(s), $(length(active_positions)) active cone position(s), and $(length(tolerances)) tolerance attempt(s)",
    )

    for (attempt_index, tolerance) in enumerate(tolerances)
        context = "coordinate attempt $(attempt_index)/$(length(tolerances)), tol=$(_format_metric(tolerance))"
        attempt_start = time_ns()
        _phase1_exact_recovery_log(opt, "$(context): rationalizing coordinates")
        rational_coordinates = _phase1_exact_recovery_rationalize_coordinates(
            opt,
            coordinates,
            tolerance,
            context,
        )
        _phase1_exact_recovery_log(
            opt,
            "$(context): rationalized coordinates in $(_phase1_exact_recovery_elapsed(attempt_start))s; projecting active cone positions",
        )
        candidate_active = _project_exact_solution_from_rational_coordinates_recovery(
            opt,
            rational_coordinates,
            particular_active,
            nullspace_active,
            context,
        )
        _phase1_exact_recovery_log(
            opt,
            "$(context): active projection finished in $(_phase1_exact_recovery_elapsed(attempt_start))s",
        )
        if _strictly_interior_exact_recovery_check(
            opt,
            candidate_active,
            active_blocks,
            active_positive_scalars,
            context,
        )
            _phase1_exact_recovery_log(
                opt,
                "$(context): active point is exact strict interior; reconstructing full vector",
            )
            return _project_exact_solution_from_rational_coordinates_recovery(
                opt,
                rational_coordinates,
                particular,
                nullspace,
                context * " full reconstruction",
            )
        end
        _phase1_exact_recovery_log(
            opt,
            "$(context): no exact strict interior point after $(_phase1_exact_recovery_elapsed(attempt_start))s",
        )
    end
    _phase1_exact_recovery_log(opt, "coordinate recovery failed")
    return nothing
end

function _phase1_exact_anchor_fallback(
    opt::Optimizer,
    candidate::Vector{F},
    problem::ProblemData,
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    numeric_affine = _numeric_affine_data(problem, F)
    return _phase1_exact_feasible_point(
        opt,
        candidate,
        problem,
        settings,
        numeric_affine,
    )
end

function _should_attempt_phase1_recovery(
    residual::F,
    iteration::Int,
    last_probe_residual::Union{Nothing,F},
    settings,
) where {F<:AbstractFloat}
    residual <= settings.feasibility_tolerance && return true
    residual <= F(1.0e-6) || return false
    last_probe_residual === nothing && return iteration >= 2
    return residual * 32 <= last_probe_residual
end

function _hypatia_phase1_syssolver(settings::Settings, ::Type{F}) where {F<:AbstractFloat}
    choice = _phase1_hypatia_syssolver(settings)
    if choice == :auto
        if F == Float64
            return Hypatia.Solvers.SymIndefSparseSystemSolver{F}(), false, false
        end
        return Hypatia.Solvers.SymIndefDenseSystemSolver{F}(), true, false
    elseif choice == :symindef_sparse
        F == Float64 || error("Hypatia sparse symmetric-indefinite solver is only supported for Float64.")
        return Hypatia.Solvers.SymIndefSparseSystemSolver{F}(), false, false
    elseif choice == :symindef_dense
        return Hypatia.Solvers.SymIndefDenseSystemSolver{F}(), true, false
    elseif choice == :symindef_indirect
        return Hypatia.Solvers.SymIndefIndirectSystemSolver{F}(), false, false
    elseif choice == :qrchol_dense
        return Hypatia.Solvers.QRCholDenseSystemSolver{F}(), true, true
    elseif choice == :naive_dense
        return Hypatia.Solvers.NaiveDenseSystemSolver{F}(), true, false
    elseif choice == :naiveelim_dense
        return Hypatia.Solvers.NaiveElimDenseSystemSolver{F}(), true, false
    end
    error("Unhandled phase1_hypatia_syssolver choice $(choice).")
end

function _append_phase1_hypatia_tolerance!(
    kwargs::Vector{Pair{Symbol,Any}},
    name::Symbol,
    settings_value::BigFloat,
    ::Type{F},
) where {F<:AbstractFloat}
    settings_value > 0 || return kwargs
    push!(kwargs, name => _to_working_float(F, settings_value))
    return kwargs
end

function _phase1_hypatia_tolerance_kwargs(settings::Settings, ::Type{F}) where {F<:AbstractFloat}
    kwargs = Pair{Symbol,Any}[]
    _append_phase1_hypatia_tolerance!(kwargs, :tol_rel_opt, settings.phase1_hypatia_tol_rel_opt, F)
    _append_phase1_hypatia_tolerance!(kwargs, :tol_abs_opt, settings.phase1_hypatia_tol_abs_opt, F)
    _append_phase1_hypatia_tolerance!(kwargs, :tol_feas, settings.phase1_hypatia_tol_feas, F)
    _append_phase1_hypatia_tolerance!(
        kwargs,
        :default_tol_power,
        settings.phase1_hypatia_default_tol_power,
        F,
    )
    _append_phase1_hypatia_tolerance!(
        kwargs,
        :default_tol_relax,
        settings.phase1_hypatia_default_tol_relax,
        F,
    )
    _append_phase1_hypatia_tolerance!(kwargs, :tol_slow, settings.phase1_hypatia_tol_slow, F)
    return kwargs
end

function _phase1_hypatia_prefers_sparse_float64(problem::ProblemData)
    cone_rows =
        length(problem.positive_scalars) +
        sum((length(block.local_positions) for block in problem.blocks); init = 0)
    nullspace_columns = problem.affine === nothing ? 0 : size(problem.affine[2], 2)
    return cone_rows >= 512 || nullspace_columns >= 512
end

function _phase1_hypatia_effective_float_type(opt::Optimizer, problem::ProblemData)
    configured_type = _phase1_hypatia_float_type(opt.settings)
    if _phase1_hypatia_float_type_is_auto(opt.settings) &&
       configured_type != Float64 &&
       _phase1_hypatia_prefers_sparse_float64(problem)
        _log(
            opt,
            "Hypatia Phase I: using Float64 sparse linear algebra for this large sparse model",
        )
        return Float64
    end
    return configured_type
end

function _build_hypatia_phase1_model(
    problem::ProblemData,
    settings::Settings,
    phase1_nullspace::Matrix{ExactRational},
    ::Type{F},
    margin_upper::F,
) where {F<:AbstractFloat}
    problem.affine === nothing && error("Hypatia Phase I requires a consistent affine reduction.")
    particular, _ = problem.affine
    reduced_dimension = size(phase1_nullspace, 2)
    total_dimension = reduced_dimension + 1
    margin_index = total_dimension
    scalar_margin_rows = length(problem.positive_scalars) + 2
    psd_rows = sum((length(block.local_positions) for block in problem.blocks); init = 0)
    total_cone_dimension = scalar_margin_rows + psd_rows

    c = zeros(F, total_dimension)
    c[margin_index] = -one(F)

    A_reduced = spzeros(F, 0, total_dimension)
    b_reduced = F[]
    particular_numeric = _to_working_array(F, particular)
    nullspace_numeric = _to_working_array(F, phase1_nullspace)
    objective_bias = _to_working_float(F, settings.phase1_hypatia_objective_bias)
    if objective_bias > zero(F)
        reduced_objective = transpose(nullspace_numeric) * _to_working_array(F, problem.objective_vector_min)
        c[1:reduced_dimension] .+= objective_bias .* reduced_objective
    end

    row_indices = Int[]
    column_indices = Int[]
    values = F[]
    h = zeros(F, total_cone_dimension)
    cones = Hypatia.Cones.Cone{F}[]

    margin_upper > zero(F) || error("phase1_hypatia_margin_upper must be positive.")

    row = 1
    for index in problem.positive_scalars
        h[row] = particular_numeric[index]
        for column in 1:reduced_dimension
            coefficient = nullspace_numeric[index, column]
            iszero(coefficient) && continue
            push!(row_indices, row)
            push!(column_indices, column)
            push!(values, -coefficient)
        end
        push!(row_indices, row)
        push!(column_indices, margin_index)
        push!(values, one(F))
        row += 1
    end

    push!(row_indices, row)
    push!(column_indices, margin_index)
    push!(values, -one(F))
    row += 1

    h[row] = margin_upper
    push!(row_indices, row)
    push!(column_indices, margin_index)
    push!(values, one(F))
    row += 1

    push!(cones, Hypatia.Cones.Nonnegative{F}(scalar_margin_rows))

    rt2 = sqrt(F(2))
    for block in problem.blocks
        block_dimension = length(block.local_positions)
        push!(cones, Hypatia.Cones.PosSemidefTri{F,F}(block_dimension))
        for (local_index, (i, j)) in enumerate(block.local_positions)
            row_index = row + local_index - 1
            global_index = block.global_positions[local_index]
            scale = i == j ? one(F) : rt2
            h[row_index] = scale * particular_numeric[global_index]
            for column in 1:reduced_dimension
                coefficient = scale * nullspace_numeric[global_index, column]
                iszero(coefficient) && continue
                push!(row_indices, row_index)
                push!(column_indices, column)
                push!(values, -coefficient)
            end
            if i == j
                push!(row_indices, row_index)
                push!(column_indices, margin_index)
                push!(values, one(F))
            end
        end
        row += block_dimension
    end

    @assert row == total_cone_dimension + 1
    G = sparse(row_indices, column_indices, values, total_cone_dimension, total_dimension)
    return Hypatia.Models.Model{F}(c, A_reduced, b_reduced, G, h, cones)
end

function _phase1_hypatia_margin_caps(problem::ProblemData, settings::Settings)
    max_cap = settings.phase1_hypatia_margin_upper
    min_cap = settings.phase1_hypatia_min_margin_upper
    shrink = settings.phase1_hypatia_margin_shrink

    max_cap > 0 || error("phase1_hypatia_margin_upper must be positive.")
    min_cap > 0 || error("phase1_hypatia_min_margin_upper must be positive.")
    zero(shrink) < shrink < one(shrink) || error("phase1_hypatia_margin_shrink must lie strictly between 0 and 1.")

    if !any(!iszero, problem.objective_vector_min)
        return BigFloat[max_cap]
    end

    caps = BigFloat[]
    cap = min(max_cap, min_cap)
    push!(caps, cap)
    while cap < max_cap
        next_cap = min(max_cap, cap / shrink)
        next_cap == cap && break
        push!(caps, next_cap)
        cap = next_cap
    end
    return caps
end

function _hypatia_phase1_point(
    particular::Vector{ExactRational},
    phase1_nullspace::Matrix{ExactRational},
    coordinates::AbstractVector{F},
) where {F<:AbstractFloat}
    point = _to_working_array(F, particular)
    if !isempty(coordinates)
        point .+= _to_working_array(F, phase1_nullspace) * coordinates
    end
    return point
end

function _phase1_hypatia_reason(attempt::Phase1HypatiaAttempt)
    if attempt.reason == :exact_anchor
        return "exact anchor recovered"
    elseif attempt.reason == :no_usable_point
        return "no usable primal point"
    elseif attempt.reason == :nonfinite_point
        return "non-finite primal point"
    elseif attempt.reason == :large_residual
        return "affine residual too large for exact recovery"
    elseif attempt.reason == :exact_recovery_failed
        if attempt.margin !== nothing && attempt.margin <= zero(attempt.margin)
            return "numerical point is on or near the cone boundary"
        end
        return "numerical point could not be recovered as an exact strict interior point"
    end
    return "no exact anchor"
end

function _log_phase1_hypatia_attempt(
    opt::Optimizer,
    attempt::Phase1HypatiaAttempt,
)
    elapsed = @sprintf("%.2f", attempt.elapsed_sec)
    details = String[
        "status=$(attempt.status)",
        "iter=$(attempt.iterations)",
    ]
    attempt.margin === nothing || push!(details, "margin=$(_format_metric(attempt.margin))")
    attempt.residual === nothing || push!(details, "residual=$(_format_metric(attempt.residual))")
    push!(details, "time=$(elapsed)s")
    summary = "Hypatia Phase I: " * join(details, ", ")
    if attempt.reason == :exact_anchor
        _log(opt, summary)
    else
        _log(opt, summary * "; " * _phase1_hypatia_reason(attempt))
    end
    return
end

function _phase1_hypatia_anchor_once(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{HF},
    margin_upper::BigFloat,
) where {HF<:AbstractFloat}
    return _with_float_precision(HF, opt.settings.working_precision, function (::Type{HF})
        problem.affine === nothing && error("Hypatia Phase I requires affine data.")
        particular, _ = problem.affine
        phase1_nullspace = _phase1_nullspace(problem)
        numeric_margin_upper = _to_working_float(HF, margin_upper)
        model = _build_hypatia_phase1_model(
            problem,
            opt.settings,
            phase1_nullspace,
            HF,
            numeric_margin_upper,
        )
        problem.phase1_nullspace = nothing
        phase1_nullspace = nothing
        _gc_checkpoint!(opt, "before Hypatia load")
        syssolver, use_dense_model, preprocess = _hypatia_phase1_syssolver(opt.settings, HF)
        tolerance_kwargs = _phase1_hypatia_tolerance_kwargs(opt.settings, HF)
        solver = Hypatia.Solvers.Solver{HF}(
            ;
            verbose = !opt.silent && opt.settings.verbose,
            iter_limit = opt.settings.phase1_hypatia_iter_limit,
            tolerance_kwargs...,
            preprocess = preprocess,
            reduce = false,
            syssolver = syssolver,
            use_dense_model = use_dense_model,
        )
        start_time = time_ns()
        Hypatia.Solvers.load(solver, model)
        Hypatia.Solvers.solve(solver)

        status = Hypatia.Solvers.get_status(solver)
        iterations = Hypatia.Solvers.get_num_iters(solver)
        total_time_sec = (time_ns() - start_time) / 1.0e9
        solution_length = size(model.G, 2)
        raw_solution = try
            copy(Hypatia.Solvers.get_x(solver))
        catch
            nothing
        end

        if raw_solution === nothing || length(raw_solution) != solution_length
            solver = nothing
            model = nothing
            _gc_checkpoint!(opt, "after Hypatia solve (no point)")
            attempt = Phase1HypatiaAttempt{HF}(
                nothing,
                nothing,
                string(status),
                iterations,
                nothing,
                nothing,
                total_time_sec,
                :no_usable_point,
            )
            _log_phase1_hypatia_attempt(opt, attempt)
            return attempt
        end

        coordinates = raw_solution[1:(end - 1)]
        margin = raw_solution[end]
        raw_solution_finite = all(isfinite, raw_solution)
        raw_solution = nothing
        solver = nothing
        model = nothing
        _gc_checkpoint!(opt, "after Hypatia solve")

        phase1_nullspace = _phase1_nullspace(problem)
        candidate = _hypatia_phase1_point(particular, phase1_nullspace, coordinates)
        A_numeric = _to_working_sparse_matrix(HF, problem.A)
        b_numeric = _to_working_array(HF, problem.b)
        residual = size(A_numeric, 1) == 0 ? zero(HF) : _max_abs(A_numeric * candidate - b_numeric)
        _log_phase1_candidate_diagnostics(
            opt,
            problem,
            candidate,
            margin,
            residual,
            HF,
        )
        if !raw_solution_finite
            attempt = Phase1HypatiaAttempt{HF}(
                nothing,
                candidate,
                string(status),
                iterations,
                margin,
                residual,
                total_time_sec,
                :nonfinite_point,
            )
            _log_phase1_hypatia_attempt(opt, attempt)
            return attempt
        end
        if opt.settings.phase1_stop_after_candidate_diagnostics
            _log(
                opt,
                "Phase I candidate diagnostics: stopping before exact recovery by phase1_stop_after_candidate_diagnostics",
            )
            attempt = Phase1HypatiaAttempt{HF}(
                nothing,
                candidate,
                string(status),
                iterations,
                margin,
                residual,
                total_time_sec,
                :exact_recovery_failed,
            )
            _log_phase1_hypatia_attempt(opt, attempt)
            return attempt
        end
        recovery_threshold = max(HF(1.0e-6), sqrt(eps(HF)))
        if residual > recovery_threshold
            anchor = nothing
            if margin > zero(HF)
                anchor = _phase1_exact_feasible_point_from_coordinates(
                    opt,
                    coordinates,
                    particular,
                    phase1_nullspace,
                    problem,
                    opt.settings,
                )
            end
            attempt = Phase1HypatiaAttempt{HF}(
                anchor,
                candidate,
                string(status),
                iterations,
                margin,
                residual,
                total_time_sec,
                anchor === nothing ? :large_residual : :exact_anchor,
            )
            _log_phase1_hypatia_attempt(opt, attempt)
            return attempt
        end
        if margin <= zero(HF)
            attempt = Phase1HypatiaAttempt{HF}(
                nothing,
                candidate,
                string(status),
                iterations,
                margin,
                residual,
                total_time_sec,
                :exact_recovery_failed,
            )
            _log_phase1_hypatia_attempt(opt, attempt)
            return attempt
        end

        anchor = _phase1_exact_feasible_point_from_coordinates(
            opt,
            coordinates,
            particular,
            phase1_nullspace,
            problem,
            opt.settings,
        )
        if anchor === nothing
            phase1_nullspace = nothing
            _gc_checkpoint!(opt, "before phase1 exact recovery")
            anchor = _phase1_exact_anchor_fallback(
                opt,
                candidate,
                problem,
                opt.settings,
                HF,
            )
        end
        attempt = Phase1HypatiaAttempt{HF}(
            anchor,
            candidate,
            string(status),
            iterations,
            margin,
            residual,
            total_time_sec,
            anchor === nothing ? :exact_recovery_failed : :exact_anchor,
        )
        _log_phase1_hypatia_attempt(opt, attempt)
        return attempt
    end)
end

function _phase1_hypatia_anchor(
    opt::Optimizer,
    problem::ProblemData,
)
    primary_float_type = _phase1_hypatia_effective_float_type(opt, problem)
    last_attempt = nothing
    for margin_cap in _phase1_hypatia_margin_caps(problem, opt.settings)
        attempt = _phase1_hypatia_anchor_once(opt, problem, primary_float_type, margin_cap)
        attempt.anchor !== nothing && return attempt
        if attempt.candidate !== nothing &&
           attempt.reason == :exact_recovery_failed &&
           attempt.margin !== nothing &&
           attempt.margin <= zero(attempt.margin)
            _log(opt, "Hypatia Phase I: boundary candidate detected; trying facial reduction")
            return attempt
        end
        last_attempt = attempt
    end
    return last_attempt
end

function _phase1_seed_point(
    candidate::Union{Nothing,AbstractVector{F}},
    problem::ProblemData,
    settings,
    numeric_blocks::Vector{NumericBlock},
    numeric_affine::Union{Nothing,NumericAffineData{F}},
) where {F<:AbstractFloat}
    candidate === nothing &&
        return _build_phase1_initial_point(problem, settings, numeric_blocks, numeric_affine)

    x = copy(candidate)
    safety_margin = max(F(1.0e-10), sqrt(eps(F)))
    for _ in 1:12
        for index in problem.positive_scalars
            if !(x[index] > safety_margin)
                x[index] = safety_margin
            end
        end
        for numeric_block in numeric_blocks
            block = numeric_block.structure
            X = _vector_to_matrix(x, block)
            min_eigenvalue = try
                eigmin(Hermitian(X))
            catch
                -safety_margin
            end
            if !(min_eigenvalue > safety_margin)
                shift = max(zero(F), safety_margin - min_eigenvalue)
                for diagonal_position in block.diagonal_positions
                    x[diagonal_position] += shift
                end
            end
        end
        _strictly_interior_numeric(x, numeric_blocks, problem.positive_scalars) && return x
        safety_margin *= F(2)
    end

    return _build_phase1_initial_point(problem, settings, numeric_blocks, numeric_affine)
end

function _phase1_anchor_attempt(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{F},
) where {F<:AbstractFloat}
    anchor = nothing
    phase2_initial_point = nothing
    phase1_candidate = nothing
    phase1_margin = nothing
    phase1_residual = nothing
    phase1_status = nothing

    if _phase1_backend(opt.settings) == :hypatia
        _log(opt, "Phase I via Hypatia")
        attempt = try
            _phase1_hypatia_anchor(opt, problem)
        catch err
            _log(opt, "Hypatia Phase I failed: $(typeof(err))")
            nothing
        end
        if attempt !== nothing
            anchor = attempt.anchor
            phase1_margin = attempt.margin
            phase1_residual = attempt.residual
            phase1_status = attempt.status
            if attempt.candidate !== nothing
                phase1_candidate = _to_working_array(F, attempt.candidate)
                phase2_initial_point = _to_working_array(F, attempt.candidate)
            end
        end
    end

    if anchor === nothing && _phase1_backend(opt.settings) == :native
        numeric_settings = _numeric_settings(opt.settings, F)
        numeric_blocks = _numeric_blocks(problem.blocks)
        numeric_affine = _numeric_affine_data(problem, F)
        use_iterative_phase1 =
            numeric_settings.iterative_linear_solver &&
            length(problem.objective_vector_raw) >= numeric_settings.iterative_solver_min_dimension
        A_big = use_iterative_phase1 ?
            _to_working_sparse_matrix(F, problem.A) :
            _to_working_array(F, problem.A)
        At_big = transpose(A_big)
        b_big = _to_working_array(F, problem.b)
        AtA_big = use_iterative_phase1 ? nothing : At_big * A_big
        x = _phase1_seed_point(
            nothing,
            problem,
            numeric_settings,
            numeric_blocks,
            numeric_affine,
        )
        phase1_center = copy(x)
        penalty = numeric_settings.initial_penalty

        phase1_columns = ["Iter", "Penalty", "Residual", "Barrier", "Time (s)"]
        phase1_alignments = vcat([:left], fill(:right, length(phase1_columns) - 1))
        phase1_widths = _phase_table_widths(phase1_columns, opt.settings.phase1_outer_iterations)
        _log_table_header(
            opt,
            "Phase I",
            phase1_columns,
            phase1_widths;
            subtitle = "Feasibility search",
        )
        phase1_start_time = time_ns()
        last_phase1_recovery_probe = nothing
        last_phase1_residual = typemax(F)
        for outer_iteration in 1:opt.settings.phase1_outer_iterations
            x = _newton_phase1!(
                opt,
                x,
                problem,
                numeric_blocks,
                A_big,
                At_big,
                AtA_big,
                b_big,
                penalty,
                phase1_center,
                numeric_settings,
            )
            residual = _max_abs(A_big * x - b_big)
            last_phase1_residual = residual
            barrier_value, _, _ = _barrier_value_grad_hess(
                x,
                numeric_blocks,
                problem.positive_scalars,
                numeric_settings,
            )
            phase1_row = [
                string(outer_iteration),
                _format_metric(penalty),
                _format_metric(residual),
                _format_metric(barrier_value),
                @sprintf("%.2f", (time_ns() - phase1_start_time) / 1.0e9),
            ]
            _log_table_row(opt, phase1_row, phase1_widths, phase1_alignments)
            if _should_attempt_phase1_recovery(
                residual,
                outer_iteration,
                last_phase1_recovery_probe,
                numeric_settings,
            )
                anchor = _phase1_exact_feasible_point(opt, x, problem, opt.settings, numeric_affine)
                last_phase1_recovery_probe = residual
            end
            if residual <= numeric_settings.feasibility_tolerance || anchor !== nothing
                break
            end
            penalty *= numeric_settings.penalty_growth
        end

        if anchor === nothing && last_phase1_residual <= F(1.0e-6)
            anchor = _phase1_exact_feasible_point(opt, x, problem, opt.settings, numeric_affine)
        end
    end

    return (
        anchor = anchor,
        phase2_initial_point = phase2_initial_point,
        phase1_candidate = phase1_candidate,
        phase1_margin = phase1_margin,
        phase1_residual = phase1_residual,
        phase1_status = phase1_status,
    )
end

function _newton_phase2!(
    opt::Optimizer,
    z::Vector{F},
    x0::Vector{F},
    N_big::Matrix{F},
    c_big::Vector{F},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
    barrier_parameter::F,
    numeric_settings,
) where {F<:AbstractFloat}
    settings = numeric_settings
    Nt_big = transpose(N_big)
    for iteration in 1:settings.max_iterations
        x = x0 + N_big * z
        barrier_value, barrier_grad_x, barrier_hess_x = _barrier_value_grad_hess(
            x,
            numeric_blocks,
            positive_scalars,
            settings,
        )
        value = barrier_parameter * dot(c_big, x) + barrier_value
        grad_x = barrier_parameter * c_big + barrier_grad_x
        grad_z = Nt_big * grad_x
        grad_norm = _max_abs(grad_z)
        if iteration == 1 || iteration % settings.inner_log_frequency == 0
            _log_newton(
                opt,
                "phase II newton $(iteration): grad=$(_format_metric(grad_norm))",
            )
        end
        grad_norm <= settings.gradient_tolerance && return z
        hess_z = Nt_big * barrier_hess_x * N_big
        direction = -_solve_spd_system(hess_z, grad_z)
        direction_x = N_big * direction
        directional_derivative = dot(grad_z, direction)
        step = min(
            one(F),
            _max_step_to_boundary(
                x,
                direction_x,
                numeric_blocks,
                positive_scalars,
                settings.boundary_fraction,
            ),
        )
        accepted = false
        while step >= settings.min_step
            trial_z = z + step * direction
            trial_x = x + step * direction_x
            if _strictly_interior_numeric(trial_x, numeric_blocks, positive_scalars)
                trial_barrier, _, _ = _barrier_value_grad_hess(
                    trial_x,
                    numeric_blocks,
                    positive_scalars,
                    settings,
                )
                trial_value = barrier_parameter * dot(c_big, trial_x) + trial_barrier
                if trial_value <= value + settings.armijo_fraction * step * directional_derivative
                    z .= trial_z
                    accepted = true
                    break
                end
            end
            step *= settings.line_search_shrink
        end
        accepted || return z
    end
    return z
end

function _exact_objective_value(problem::ProblemData, x::Vector{ExactRational})
    return dot(problem.objective_vector_raw, x) + problem.objective_constant_raw
end

function _exact_phase2_objective_value(problem::ProblemData, x::Vector{ExactRational})
    return dot(problem.objective_vector_min, x)
end

function _barrier_dimension(problem::ProblemData)
    return length(problem.positive_scalars) + sum((block.size for block in problem.blocks); init = 0)
end

function _objective_nullspace_direction(
    c::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
)
    if size(nullspace, 2) == 0
        return zeros(ExactRational, 0)
    end
    return transpose(nullspace) * c
end

function _phase2_exact_solution(
    opt::Optimizer,
    problem::ProblemData,
    anchor::Vector{ExactRational},
    barrier_dim::Int,
    ::Type{F};
    initial_point::Union{Nothing,Vector{F}} = nothing,
    subtitle::String,
) where {F<:AbstractFloat}
    problem.affine === nothing &&
        return (
            x_exact = anchor,
            x_numeric = _to_working_array(F, anchor),
            barrier_parameter = one(F),
        )
    particular, nullspace = problem.affine
    phase2_nullspace = _phase2_nullspace(problem)
    if size(phase2_nullspace, 2) == 0
        if size(nullspace, 2) > 0
            _log(opt, "Phase II skipped: no objective/barrier-visible affine directions")
        end
        return (
            x_exact = anchor,
            x_numeric = _to_working_array(F, anchor),
            barrier_parameter = one(F),
        )
    elseif size(phase2_nullspace, 2) < size(nullspace, 2)
        _log(
            opt,
            "Phase II: reduced affine directions from $(size(nullspace, 2)) to $(size(phase2_nullspace, 2))",
        )
    end
    numeric_affine = _numeric_affine_data(anchor, phase2_nullspace, F)

    numeric_settings = _numeric_settings(opt.settings, F)
    numeric_blocks = _numeric_blocks(problem.blocks)
    x0 = _to_working_array(F, anchor)
    N_big = _numeric_nullspace!(numeric_affine)
    c_big = _to_working_array(F, problem.objective_vector_min)
    z = zeros(F, size(phase2_nullspace, 2))
    if initial_point !== nothing && !isempty(z)
        z .= transpose(N_big) * (initial_point - x0)
    end
    barrier_parameter = one(F)

    phase2_columns = ["Iter", "Mu", "Objective", "Gap", "Time (s)"]
    phase2_alignments = vcat([:left], fill(:right, length(phase2_columns) - 1))
    phase2_widths = _phase_table_widths(phase2_columns, opt.settings.phase2_outer_iterations)
    _log_table_header(
        opt,
        "Phase II",
        phase2_columns,
        phase2_widths;
        subtitle = subtitle,
    )
    phase2_start_time = time_ns()
    previous_objective = nothing
    previous_z = copy(z)
    stagnation_tolerance = max(F(1.0e-8), sqrt(eps(F)))
    stagnation_count = 0
    last_barrier_parameter = barrier_parameter
    for outer_iteration in 1:opt.settings.phase2_outer_iterations
        last_barrier_parameter = barrier_parameter
        previous_z .= z
        z = _newton_phase2!(
            opt,
            z,
            x0,
            N_big,
            c_big,
            numeric_blocks,
            problem.positive_scalars,
            barrier_parameter,
            numeric_settings,
        )
        x_trial = x0 + N_big * z
        approximate_objective = dot(c_big, x_trial) + _to_working_float(F, problem.objective_constant_raw)
        gap_bound = _to_working_float(F, barrier_dim) / barrier_parameter
        phase2_row = [
            string(outer_iteration),
            _format_metric(barrier_parameter),
            _format_metric(approximate_objective),
            _format_metric(gap_bound),
            @sprintf("%.2f", (time_ns() - phase2_start_time) / 1.0e9),
        ]
        _log_table_row(opt, phase2_row, phase2_widths, phase2_alignments)
        if gap_bound <= numeric_settings.optimality_gap_tolerance
            break
        end
        if previous_objective !== nothing
            objective_scale = max(one(F), abs(previous_objective), abs(approximate_objective))
            objective_change = abs(approximate_objective - previous_objective) / objective_scale
            z_scale = max(one(F), _max_abs(previous_z), _max_abs(z))
            z_change = _max_abs(z - previous_z) / z_scale
            if objective_change <= stagnation_tolerance && z_change <= stagnation_tolerance
                stagnation_count += 1
                if stagnation_count >= 2
                    _log(opt, "Phase II stalled; stopping early")
                    break
                end
            else
                stagnation_count = 0
            end
        end
        previous_objective = approximate_objective
        barrier_parameter *= numeric_settings.path_parameter_growth
    end

    x_numeric = x0 + N_big * z
    return (
        x_exact = _phase2_exact_refinement(
            x_numeric,
            anchor,
            problem,
            opt.settings,
            anchor,
            phase2_nullspace,
            numeric_affine,
        ),
        x_numeric = x_numeric,
        barrier_parameter = last_barrier_parameter,
    )
end
