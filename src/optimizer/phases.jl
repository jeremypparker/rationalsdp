# Phase I and Phase II routines, including exact recovery and Hypatia integration.

function _recovery_tolerances(settings::Settings, ::Type{F}) where {F<:AbstractFloat}
    tolerances = F[]
    tolerance = max(F(1.0e-8), sqrt(_to_working_float(F, settings.rational_tolerance)))
    final_tolerance = _to_working_float(F, settings.rational_tolerance)
    while tolerance > final_tolerance
        push!(tolerances, tolerance)
        tolerance /= F(1.0e4)
    end
    push!(tolerances, final_tolerance)
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
    candidate_objective = _exact_objective_value(problem, candidate)
    anchor_objective = _exact_objective_value(problem, anchor)
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
        if _exact_objective_value(problem, refined) < _exact_objective_value(problem, best)
            best = refined
        end
    end

    return best
end

function _phase1_exact_feasible_point(
    x_phase1::Vector{F},
    problem::ProblemData,
    settings::Settings,
    numeric_affine::NumericAffineData{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    particular, nullspace = problem.affine
    coefficients = _affine_coordinates(x_phase1, numeric_affine)
    for tolerance in _recovery_tolerances(settings, F)
        candidate = _project_exact_solution(
            coefficients,
            particular,
            nullspace;
            tolerance = tolerance,
        )
        if _strictly_interior_exact(candidate, problem.blocks, problem.positive_scalars)
            return candidate
        end
    end
    return nothing
end

function _phase1_exact_feasible_point_from_coordinates(
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

    for tolerance in _recovery_tolerances(settings, F)
        rational_coordinates = [
            rationalize(BigInt, value; tol = tolerance) for value in coordinates
        ]
        candidate_active = particular_active + nullspace_active * rational_coordinates
        if _strictly_interior_exact(candidate_active, active_blocks, active_positive_scalars)
            return particular + nullspace * rational_coordinates
        end
    end
    return nothing
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

function _hypatia_phase1_syssolver(::Type{F}) where {F<:AbstractFloat}
    if F == Float64
        return Hypatia.Solvers.SymIndefSparseSystemSolver{F}(), false
    end
    return Hypatia.Solvers.SymIndefDenseSystemSolver{F}(), true
end

function _build_hypatia_phase1_model(
    problem::ProblemData,
    settings::Settings,
    ::Type{F},
    margin_upper::F,
) where {F<:AbstractFloat}
    problem.affine === nothing && error("Hypatia Phase I requires a consistent affine reduction.")
    particular, _ = problem.affine
    nullspace = _phase1_nullspace(problem)
    reduced_dimension = size(nullspace, 2)
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
    nullspace_numeric = _to_working_array(F, nullspace)
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
    problem::ProblemData,
    coordinates::AbstractVector{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && error("Hypatia Phase I requires a consistent affine reduction.")
    particular, _ = problem.affine
    nullspace = _phase1_nullspace(problem)
    point = _to_working_array(F, particular)
    if !isempty(coordinates)
        point .+= _to_working_array(F, nullspace) * coordinates
    end
    return point
end

function _affine_point(
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    coordinates::AbstractVector{F},
) where {F<:AbstractFloat}
    point = _to_working_array(F, particular)
    if !isempty(coordinates)
        point .+= _to_working_array(F, nullspace) * coordinates
    end
    return point
end

function _build_hypatia_phase2_model(
    problem::ProblemData,
    phase2_nullspace::Matrix{ExactRational},
    ::Type{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && error("Hypatia Phase II requires affine data.")
    particular, _ = problem.affine
    reduced_dimension = size(phase2_nullspace, 2)
    particular_numeric = _to_working_array(F, particular)
    nullspace_numeric = _to_working_array(F, phase2_nullspace)

    c = transpose(nullspace_numeric) * _to_working_array(F, problem.objective_vector_min)
    A_reduced = spzeros(F, 0, reduced_dimension)
    b_reduced = F[]

    row_indices = Int[]
    column_indices = Int[]
    values = F[]
    scalar_rows = length(problem.positive_scalars)
    psd_rows = sum((length(block.local_positions) for block in problem.blocks); init = 0)
    total_cone_dimension = scalar_rows + psd_rows
    h = zeros(F, total_cone_dimension)
    cones = Hypatia.Cones.Cone{F}[]

    row = 1
    if scalar_rows > 0
        push!(cones, Hypatia.Cones.Nonnegative{F}(scalar_rows))
        for index in problem.positive_scalars
            h[row] = particular_numeric[index]
            for column in 1:reduced_dimension
                coefficient = nullspace_numeric[index, column]
                iszero(coefficient) && continue
                push!(row_indices, row)
                push!(column_indices, column)
                push!(values, -coefficient)
            end
            row += 1
        end
    end

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
        end
        row += block_dimension
    end

    @assert row == total_cone_dimension + 1
    G = sparse(row_indices, column_indices, values, total_cone_dimension, reduced_dimension)
    return Hypatia.Models.Model{F}(c, A_reduced, b_reduced, G, h, cones)
end

function _phase2_hypatia_candidate(
    opt::Optimizer,
    problem::ProblemData,
    particular::Vector{ExactRational},
    phase2_nullspace::Matrix{ExactRational},
    ::Type{F},
) where {F<:AbstractFloat}
    return _with_float_precision(F, opt.settings.working_precision, function (::Type{F})
        model = _build_hypatia_phase2_model(problem, phase2_nullspace, F)
        syssolver, use_dense_model = _hypatia_phase1_syssolver(F)
        solver = Hypatia.Solvers.Solver{F}(
            verbose = !opt.silent && opt.settings.verbose,
            iter_limit = opt.settings.phase1_hypatia_iter_limit,
            preprocess = true,
            reduce = true,
            syssolver = syssolver,
            use_dense_model = use_dense_model,
        )
        start_time = time_ns()
        Hypatia.Solvers.load(solver, model)
        Hypatia.Solvers.solve(solver)
        elapsed_sec = (time_ns() - start_time) / 1.0e9

        status = string(Hypatia.Solvers.get_status(solver))
        iterations = Hypatia.Solvers.get_num_iters(solver)
        raw_solution = try
            vec(collect(Hypatia.Solvers.get_x(solver)))
        catch
            nothing
        end
        if raw_solution === nothing || length(raw_solution) != size(model.G, 2) || !all(isfinite, raw_solution)
            _log(opt, "Hypatia Phase II: status=$(status), iter=$(iterations), time=$(@sprintf("%.2f", elapsed_sec))s; no usable point")
            return nothing
        end

        candidate = _affine_point(particular, phase2_nullspace, raw_solution)
        approximate_objective =
            dot(_to_working_array(F, problem.objective_vector_min), candidate) +
            _to_working_float(F, problem.objective_constant_raw)
        _log(
            opt,
            "Hypatia Phase II: status=$(status), iter=$(iterations), objective=$(_format_metric(approximate_objective)), time=$(@sprintf("%.2f", elapsed_sec))s",
        )
        return candidate
    end)
end

function _phase2_hypatia_refinement(
    opt::Optimizer,
    problem::ProblemData,
    anchor::Vector{ExactRational},
    particular::Vector{ExactRational},
    phase2_nullspace::Matrix{ExactRational},
    ::Type{F},
) where {F<:AbstractFloat}
    candidate = _phase2_hypatia_candidate(opt, problem, particular, phase2_nullspace, F)
    candidate === nothing && return nothing
    numeric_affine = _numeric_affine_data(particular, phase2_nullspace, F)
    refined = _phase2_exact_refinement(
        candidate,
        anchor,
        problem,
        opt.settings,
        particular,
        phase2_nullspace,
        numeric_affine,
    )
    if _exact_objective_value(problem, refined) < _exact_objective_value(problem, anchor)
        return refined
    end
    _log(opt, "Hypatia Phase II: no exact improvement")
    return nothing
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
        model = _build_hypatia_phase1_model(problem, opt.settings, HF, numeric_margin_upper)
        syssolver, use_dense_model = _hypatia_phase1_syssolver(HF)
        solver = Hypatia.Solvers.Solver{HF}(
            verbose = !opt.silent && opt.settings.verbose,
            iter_limit = opt.settings.phase1_hypatia_iter_limit,
            preprocess = false,
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
        raw_solution = try
            Hypatia.Solvers.get_x(solver)
        catch
            nothing
        end

        if raw_solution === nothing || length(raw_solution) != size(model.G, 2)
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

        candidate = _hypatia_phase1_point(problem, raw_solution[1:(end - 1)])
        margin = raw_solution[end]
        A_numeric = _to_working_sparse_matrix(HF, problem.A)
        b_numeric = _to_working_array(HF, problem.b)
        residual = size(A_numeric, 1) == 0 ? zero(HF) : _max_abs(A_numeric * candidate - b_numeric)
        if !all(isfinite, raw_solution)
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
        recovery_threshold = max(HF(1.0e-6), sqrt(eps(HF)))
        if residual > recovery_threshold
            attempt = Phase1HypatiaAttempt{HF}(
                nothing,
                candidate,
                string(status),
                iterations,
                margin,
                residual,
                total_time_sec,
                :large_residual,
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
            raw_solution[1:(end - 1)],
            particular,
            phase1_nullspace,
            problem,
            opt.settings,
        )
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
    primary_float_type = _phase1_hypatia_float_type(opt.settings)
    last_attempt = nothing
    for margin_cap in _phase1_hypatia_margin_caps(problem, opt.settings)
        attempt = _phase1_hypatia_anchor_once(opt, problem, primary_float_type, margin_cap)
        attempt.anchor !== nothing && return attempt
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
    problem.affine === nothing && return anchor
    particular, nullspace = problem.affine
    numeric_affine = _numeric_affine_data(problem, F)

    numeric_settings = _numeric_settings(opt.settings, F)
    numeric_blocks = _numeric_blocks(problem.blocks)
    x0 = _to_working_array(F, anchor)
    N_big = _numeric_nullspace!(numeric_affine)
    c_big = _to_working_array(F, problem.objective_vector_min)
    z = zeros(F, size(nullspace, 2))
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
    for outer_iteration in 1:opt.settings.phase2_outer_iterations
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

    return _phase2_exact_refinement(
        x0 + N_big * z,
        anchor,
        problem,
        opt.settings,
        particular,
        nullspace,
        numeric_affine,
    )
end
