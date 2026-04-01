# Top-level solve orchestration and result reporting.

function _dual_output_tolerance(settings::Settings)
    return max(big"1e-6", sqrt(settings.rational_tolerance))
end

function _convert_dual_output(::Type{T}, value, tolerance::BigFloat) where {T<:Real}
    if value isa AbstractVector
        return [_convert_dual_output(T, entry, tolerance) for entry in value]
    elseif value isa Integer
        return _to_output_type(T, _exact_rational(value))
    end
    return _to_output_type(T, rationalize(BigInt, BigFloat(value); tol = tolerance))
end

function _solve_dual_rows_least_squares(
    M::AbstractMatrix{F},
    rhs::AbstractVector{F},
) where {F<:AbstractFloat}
    solution = try
        M \ rhs
    catch
        nothing
    end
    if solution !== nothing && all(isfinite, solution)
        return solution
    end

    # Dualized models can make `A'` rank-deficient even when the KKT system is
    # numerically well behaved. Use a pseudoinverse solve in that case so we can
    # still recover consistent approximate duals for MOI / Dualization output.
    svd_factor = svd(Matrix(M))
    isempty(svd_factor.S) && return nothing
    tolerance =
        max(size(M)...) * eps(F) * maximum(svd_factor.S)
    inverted_singular_values = [
        sigma > tolerance ? inv(sigma) : zero(F) for sigma in svd_factor.S
    ]
    solution =
        svd_factor.V *
        Diagonal(inverted_singular_values) *
        transpose(svd_factor.U) * rhs
    all(isfinite, solution) || return nothing
    return solution
end

function _convert_moi_function(
    ::Type{F},
    func::MOI.VariableIndex,
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
) where {F<:AbstractFloat}
    return variable_map[func]
end

function _convert_moi_function(
    ::Type{F},
    func::MOI.VectorOfVariables,
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
) where {F<:AbstractFloat}
    return MOI.VectorOfVariables([variable_map[vi] for vi in func.variables])
end

function _convert_moi_function(
    ::Type{F},
    func::MOI.ScalarAffineFunction{T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
) where {F<:AbstractFloat,T}
    terms = MOI.ScalarAffineTerm{F}[
        MOI.ScalarAffineTerm(
            _to_working_float(F, term.coefficient),
            variable_map[term.variable],
        ) for term in func.terms
    ]
    return MOI.ScalarAffineFunction(terms, _to_working_float(F, func.constant))
end

function _convert_moi_function(
    ::Type{F},
    func::MOI.VectorAffineFunction{T},
    variable_map::Dict{MOI.VariableIndex,MOI.VariableIndex},
) where {F<:AbstractFloat,T}
    terms = MOI.VectorAffineTerm{F}[
        MOI.VectorAffineTerm(
            term.output_index,
            MOI.ScalarAffineTerm(
                _to_working_float(F, term.scalar_term.coefficient),
                variable_map[term.scalar_term.variable],
            ),
        ) for term in func.terms
    ]
    constants = _to_working_array(F, func.constants)
    return MOI.VectorAffineFunction(terms, constants)
end

_convert_moi_set(::Type{F}, set::MOI.EqualTo{T}) where {F<:AbstractFloat,T} =
    MOI.EqualTo(_to_working_float(F, set.value))
_convert_moi_set(::Type{F}, set::MOI.GreaterThan{T}) where {F<:AbstractFloat,T} =
    MOI.GreaterThan(_to_working_float(F, set.lower))
_convert_moi_set(::Type{F}, set::MOI.LessThan{T}) where {F<:AbstractFloat,T} =
    MOI.LessThan(_to_working_float(F, set.upper))
_convert_moi_set(::Type{F}, set::MOI.Interval{T}) where {F<:AbstractFloat,T} =
    MOI.Interval(_to_working_float(F, set.lower), _to_working_float(F, set.upper))
_convert_moi_set(::Type{F}, set::MOI.PositiveSemidefiniteConeTriangle) where {F<:AbstractFloat} = set

function _copy_storage_to_hypatia_optimizer(
    storage,
    target,
    ::Type{T},
    ::Type{F},
) where {T<:Real,F<:AbstractFloat}
    variables = MOI.get(storage, MOI.ListOfVariableIndices())
    mapped_variables = MOI.add_variables(target, length(variables))
    variable_map = Dict{MOI.VariableIndex,MOI.VariableIndex}(
        variable => mapped_variables[index] for (index, variable) in enumerate(variables)
    )
    constraint_map = Dict{Any,Any}()

    sense = MOI.get(storage, MOI.ObjectiveSense())
    MOI.set(target, MOI.ObjectiveSense(), sense)
    if sense != MOI.FEASIBILITY_SENSE
        objective_type = MOI.get(storage, MOI.ObjectiveFunctionType())
        if objective_type == MOI.VariableIndex
            func = MOI.get(storage, MOI.ObjectiveFunction{MOI.VariableIndex}())
            MOI.set(
                target,
                MOI.ObjectiveFunction{MOI.VariableIndex}(),
                _convert_moi_function(F, func, variable_map),
            )
        elseif objective_type <: MOI.ScalarAffineFunction
            func = MOI.get(storage, MOI.ObjectiveFunction{objective_type}())
            MOI.set(
                target,
                MOI.ObjectiveFunction{MOI.ScalarAffineFunction{F}}(),
                _convert_moi_function(F, func, variable_map),
            )
        else
            error("Unsupported objective function type for Hypatia dual postsolve: $objective_type")
        end
    end

    scalar_constraint_types = (
        (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.Interval{T}),
        (MOI.VariableIndex, MOI.EqualTo{T}),
        (MOI.VariableIndex, MOI.GreaterThan{T}),
        (MOI.VariableIndex, MOI.LessThan{T}),
        (MOI.VariableIndex, MOI.Interval{T}),
    )

    for (FuncType, SetType) in scalar_constraint_types
        for ci in MOI.get(storage, MOI.ListOfConstraintIndices{FuncType,SetType}())
            func = MOI.get(storage, MOI.ConstraintFunction(), ci)
            set = MOI.get(storage, MOI.ConstraintSet(), ci)
            constraint_map[ci] = MOI.add_constraint(
                target,
                _convert_moi_function(F, func, variable_map),
                _convert_moi_set(F, set),
            )
        end
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        constraint_map[ci] = MOI.add_constraint(
            target,
            _convert_moi_function(F, func, variable_map),
            _convert_moi_set(F, set),
        )
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        constraint_map[ci] = MOI.add_constraint(
            target,
            _convert_moi_function(F, func, variable_map),
            _convert_moi_set(F, set),
        )
    end

    return constraint_map
end

function _cone_positions(problem::ProblemData)
    positions = Int[]
    append!(positions, problem.positive_scalars)
    for block in problem.blocks
        append!(positions, block.global_positions)
    end
    return unique(sort(positions))
end

function _dual_free_positions(problem::ProblemData)
    return setdiff(collect(eachindex(problem.objective_vector_min)), _cone_positions(problem))
end

function _inverse_psd_affine_dual_vector(
    raw::AbstractVector{F},
    block::BlockStructure,
) where {F<:AbstractFloat}
    dual_vector = similar(raw)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        value = raw[local_index]
        if i != j
            value *= 2
        end
        dual_vector[local_index] = value
    end
    return dual_vector
end

function _hypatia_dual_targets(
    target,
    constraint_map::Dict{Any,Any},
    storage,
    problem::ProblemData,
    ::Type{T},
    ::Type{F},
) where {T<:Real,F<:AbstractFloat}
    row_count = size(problem.A, 1)
    y_target = zeros(F, row_count)

    scalar_constraint_types = (
        (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.Interval{T}),
        (MOI.VariableIndex, MOI.EqualTo{T}),
        (MOI.VariableIndex, MOI.GreaterThan{T}),
        (MOI.VariableIndex, MOI.LessThan{T}),
        (MOI.VariableIndex, MOI.Interval{T}),
    )

    for (FuncType, SetType) in scalar_constraint_types
        for ci in MOI.get(storage, MOI.ListOfConstraintIndices{FuncType,SetType}())
            rows = get(problem.scalar_constraint_rows, ci, Int[])
            length(rows) == 1 || continue
            mapped_ci = get(constraint_map, ci, nothing)
            mapped_ci === nothing && continue
            y_target[only(rows)] = MOI.get(target, MOI.ConstraintDual(), mapped_ci)
        end
    end

    A_numeric = _to_working_array(F, problem.A)
    s_target = _to_working_array(F, problem.objective_vector_min) - transpose(A_numeric) * y_target

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        mapped_ci = get(constraint_map, ci, nothing)
        mapped_ci === nothing && continue
        block = problem.blocks[problem.psd_constraint_blocks[ci]]
        s_target[block.global_positions] .= MOI.get(target, MOI.ConstraintDual(), mapped_ci)
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        mapped_ci = get(constraint_map, ci, nothing)
        mapped_ci === nothing && continue
        block = problem.blocks[problem.psd_constraint_blocks[ci]]
        raw_dual = MOI.get(target, MOI.ConstraintDual(), mapped_ci)
        s_target[block.global_positions] .= _inverse_psd_affine_dual_vector(raw_dual, block)
    end

    return y_target, s_target
end

function _dual_affine_exact_data(problem::ProblemData)
    row_count = size(problem.A, 1)
    free_positions = _dual_free_positions(problem)
    if isempty(free_positions)
        return (
            zeros(ExactRational, row_count),
            Matrix{ExactRational}(I, row_count, row_count),
            _cone_positions(problem),
        )
    end

    affine = _solve_affine_system(
        Matrix(transpose(problem.A[:, free_positions])),
        problem.objective_vector_min[free_positions],
    )
    affine === nothing && return nothing
    y_particular, y_nullspace = affine
    return y_particular, y_nullspace, _cone_positions(problem)
end

function _dual_candidate_margins(
    problem::ProblemData,
    s_exact::Vector{ExactRational},
    ::Type{F},
) where {F<:AbstractFloat}
    min_scalar = typemax(F)
    if !isempty(problem.positive_scalars)
        scalar_values = try
            [_to_working_float(F, s_exact[index]) for index in problem.positive_scalars]
        catch
            return -typemax(F), -typemax(F)
        end
        all(isfinite, scalar_values) || return -typemax(F), -typemax(F)
        min_scalar = minimum(scalar_values)
    end
    min_psd = typemax(F)
    for block in problem.blocks
        dual_matrix = try
            _to_working_array(F, _dual_vector_to_matrix(s_exact, block))
        catch
            return -typemax(F), -typemax(F)
        end
        all(isfinite, dual_matrix) || return -typemax(F), -typemax(F)
        eigen_min = try
            minimum(eigvals(Symmetric(dual_matrix)))
        catch
            return -typemax(F), -typemax(F)
        end
        isfinite(eigen_min) || return -typemax(F), -typemax(F)
        min_psd = min(min_psd, eigen_min)
    end
    return min_scalar, min_psd
end

function _select_exact_dual_candidate(
    opt::Optimizer,
    problem::ProblemData,
    y_target::Vector{F},
    s_target::Vector{F},
    ::Type{F},
) where {F<:AbstractFloat}
    exact_affine = _dual_affine_exact_data(problem)
    exact_affine === nothing && return nothing
    y_particular, y_nullspace, cone_positions = exact_affine

    Y_numeric = _to_working_array(F, y_nullspace)
    y_rhs = y_target - _to_working_array(F, y_particular)

    cone_matrix = problem.A[:, cone_positions]
    s_particular =
        problem.objective_vector_min[cone_positions] - transpose(cone_matrix) * y_particular
    S_numeric = -(_to_working_array(F, transpose(cone_matrix)) * Y_numeric)
    s_rhs = s_target[cone_positions] - _to_working_array(F, s_particular)

    weights = (F(1), F(10), F(100))
    best = nothing
    best_margin = -typemax(F)
    best_violation = typemax(F)

    for weight in weights
        coordinates = if size(Y_numeric, 2) == 0
            F[]
        else
            fitting_matrix =
                isempty(cone_positions) ?
                Y_numeric :
                [Y_numeric; sqrt(weight) * S_numeric]
            fitting_rhs =
                isempty(cone_positions) ?
                y_rhs :
                [y_rhs; sqrt(weight) * s_rhs]
            solved = try
                fitting_matrix \ fitting_rhs
            catch
                nothing
            end
            solved === nothing && continue
            all(isfinite, solved) || continue
            solved
        end

        for tolerance in _recovery_tolerances(opt.settings, F)
            y_exact = _project_exact_solution(
                coordinates,
                y_particular,
                y_nullspace;
                tolerance = tolerance,
            )
            s_exact = problem.objective_vector_min - transpose(problem.A) * y_exact
            min_scalar, min_psd = _dual_candidate_margins(problem, s_exact, F)
            margin = min(min_scalar, min_psd)
            violation = max(zero(F), -min_scalar) + max(zero(F), -min_psd)
            if margin > best_margin + sqrt(eps(F)) ||
               (abs(margin - best_margin) <= sqrt(eps(F)) && violation < best_violation)
                best = (y_exact = y_exact, s_exact = s_exact, min_scalar = min_scalar, min_psd = min_psd)
                best_margin = margin
                best_violation = violation
            end
        end
    end

    return best
end

function _set_exact_constraint_duals!(
    opt::Optimizer{T},
    problem::ProblemData,
    y_exact::Vector{ExactRational},
    s_exact::Vector{ExactRational},
) where {T<:Real}
    scalar_constraint_types = (
        (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.Interval{T}),
        (MOI.VariableIndex, MOI.EqualTo{T}),
        (MOI.VariableIndex, MOI.GreaterThan{T}),
        (MOI.VariableIndex, MOI.LessThan{T}),
        (MOI.VariableIndex, MOI.Interval{T}),
    )

    for (FuncType, SetType) in scalar_constraint_types
        for ci in MOI.get(opt.storage, MOI.ListOfConstraintIndices{FuncType,SetType}())
            rows = get(problem.scalar_constraint_rows, ci, Int[])
            length(rows) == 1 || continue
            opt.constraint_dual[ci] = _to_output_type(T, y_exact[only(rows)])
        end
    end

    for ci in MOI.get(
        opt.storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        block = problem.blocks[problem.psd_constraint_blocks[ci]]
        opt.constraint_dual[ci] = _to_output_type.(T, s_exact[block.global_positions])
    end

    for ci in MOI.get(
        opt.storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        block = problem.blocks[problem.psd_constraint_blocks[ci]]
        dual_vector = _psd_affine_dual_vector(s_exact[block.global_positions], block)
        opt.constraint_dual[ci] = _to_output_type.(T, dual_vector)
    end
    return
end

function _hypatia_storage_dual_postsolve!(
    opt::Optimizer{T},
    problem::ProblemData,
) where {T<:Real}
    HF = _dual_postsolve_float_type(opt.settings)
    return _with_float_precision(HF, opt.settings.working_precision, function (::Type{F}) where {F<:AbstractFloat}
        tolerance = _dual_output_tolerance(opt.settings)
        target = MOI.instantiate(
            () -> Hypatia.Optimizer{F}(verbose = false);
            with_bridge_type = F,
        )
        MOI.set(target, MOI.Silent(), opt.silent || !opt.settings.verbose)
        MOI.set(target, MOI.RawOptimizerAttribute("iter_limit"), opt.settings.phase1_hypatia_iter_limit)
        constraint_map = _copy_storage_to_hypatia_optimizer(opt.storage, target, T, F)
        MOI.optimize!(target)

        termination = MOI.get(target, MOI.TerminationStatus())
        dual_status = MOI.get(target, MOI.DualStatus())
        if termination != MOI.OPTIMAL && termination != MOI.ALMOST_OPTIMAL
            _log(
                opt,
                "Hypatia dual postsolve unavailable: term=$(termination), dual=$(dual_status)",
            )
            return false
        end
        if dual_status != MOI.FEASIBLE_POINT && dual_status != MOI.NEARLY_FEASIBLE_POINT
            _log(
                opt,
                "Hypatia dual postsolve unavailable: term=$(termination), dual=$(dual_status)",
            )
            return false
        end

        y_target = nothing
        s_target = nothing
        try
            y_target, s_target = _hypatia_dual_targets(target, constraint_map, opt.storage, problem, T, F)
        catch err
            _log(opt, "Hypatia dual postsolve fit unavailable: $(typeof(err))")
        end
        exact_dual =
            y_target === nothing || s_target === nothing ?
            nothing :
            _select_exact_dual_candidate(opt, problem, y_target, s_target, F)
        if exact_dual !== nothing
            _set_exact_constraint_duals!(
                opt,
                problem,
                exact_dual.y_exact,
                exact_dual.s_exact,
            )
            _log(
                opt,
                "Hypatia dual postsolve: exact fit with scalar=$(_format_metric(exact_dual.min_scalar)), psd=$(_format_metric(exact_dual.min_psd))",
            )
            return true
        end

        for (original_ci, mapped_ci) in constraint_map
            dual_value = try
                MOI.get(target, MOI.ConstraintDual(), mapped_ci)
            catch
                continue
            end
            opt.constraint_dual[original_ci] = _convert_dual_output(T, dual_value, tolerance)
        end
        return true
    end)
end

function _recover_dual_kkt(
    problem::ProblemData,
    x_numeric::Vector{F},
    barrier_parameter::F,
    numeric_blocks::Vector{NumericBlock},
    numeric_settings,
) where {F<:AbstractFloat}
    barrier_parameter > zero(F) || return nothing
    _, barrier_grad, _ = _barrier_value_grad_hess(
        x_numeric,
        numeric_blocks,
        problem.positive_scalars,
        numeric_settings,
    )
    dual_slack = -barrier_grad / barrier_parameter

    if size(problem.A, 1) == 0
        return zeros(F, 0), dual_slack
    end

    A_big = _to_working_array(F, problem.A)
    rhs = _to_working_array(F, problem.objective_vector_min) - dual_slack
    dual_rows = _solve_dual_rows_least_squares(transpose(A_big), rhs)
    dual_rows === nothing && return nothing
    all(isfinite, dual_rows) || return nothing
    return dual_rows, dual_slack
end

function _constraint_primal_value(
    func::MOI.ScalarAffineFunction{T},
    variable_primal::Dict{MOI.VariableIndex,T},
) where {T}
    value = func.constant
    for term in func.terms
        value += term.coefficient * variable_primal[term.variable]
    end
    return value
end

function _constraint_primal_value(
    func::MOI.VectorAffineFunction{T},
    variable_primal::Dict{MOI.VariableIndex,T},
) where {T}
    values = copy(func.constants)
    for term in func.terms
        values[term.output_index] +=
            term.scalar_term.coefficient * variable_primal[term.scalar_term.variable]
    end
    return values
end

function _psd_affine_dual_vector(
    dual_entries::AbstractVector,
    block::BlockStructure,
)
    matrix = zeros(eltype(dual_entries), block.size, block.size)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        value = dual_entries[local_index]
        if i != j
            value /= 2
        end
        matrix[i, j] = value
        matrix[j, i] = value
    end
    vector = similar(dual_entries)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        vector[local_index] = matrix[i, j]
    end
    return vector
end

function _populate_constraint_results!(
    opt::Optimizer{T},
    problem::ProblemData,
    x_exact::Vector{ExactRational};
    dual_rows = nothing,
    dual_slack = nothing,
) where {T<:Real}
    tolerance = _dual_output_tolerance(opt.settings)
    scalar_constraint_types = (
        (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.Interval{T}),
        (MOI.VariableIndex, MOI.EqualTo{T}),
        (MOI.VariableIndex, MOI.GreaterThan{T}),
        (MOI.VariableIndex, MOI.LessThan{T}),
        (MOI.VariableIndex, MOI.Interval{T}),
    )

    for (F, S) in scalar_constraint_types
        for ci in MOI.get(opt.storage, MOI.ListOfConstraintIndices{F,S}())
            func = MOI.get(opt.storage, MOI.ConstraintFunction(), ci)
            primal_value = func isa MOI.VariableIndex ? opt.variable_primal[func] : _constraint_primal_value(func, opt.variable_primal)
            opt.constraint_primal[ci] = primal_value
            if dual_rows !== nothing && haskey(problem.scalar_constraint_rows, ci)
                rows = problem.scalar_constraint_rows[ci]
                length(rows) == 1 || continue
                opt.constraint_dual[ci] = _convert_dual_output(T, dual_rows[only(rows)], tolerance)
            end
        end
    end

    for ci in MOI.get(
        opt.storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        func = MOI.get(opt.storage, MOI.ConstraintFunction(), ci)
        opt.constraint_primal[ci] = [opt.variable_primal[variable] for variable in func.variables]
        if dual_slack !== nothing && haskey(problem.psd_constraint_blocks, ci)
            block = problem.blocks[problem.psd_constraint_blocks[ci]]
            opt.constraint_dual[ci] = _convert_dual_output(T, dual_slack[block.global_positions], tolerance)
        end
    end
    for ci in MOI.get(
        opt.storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        func = MOI.get(opt.storage, MOI.ConstraintFunction(), ci)
        opt.constraint_primal[ci] = _constraint_primal_value(func, opt.variable_primal)
        if dual_slack !== nothing && haskey(problem.psd_constraint_blocks, ci)
            block = problem.blocks[problem.psd_constraint_blocks[ci]]
            opt.constraint_dual[ci] = _convert_dual_output(
                T,
                _psd_affine_dual_vector(dual_slack[block.global_positions], block),
                tolerance,
            )
        end
    end
    return
end

function MOI.optimize!(opt::Optimizer{T}) where {T}
    start_time = time_ns()
    _reset_results!(opt)
    _with_working_precision(opt.settings, function (F)
        numeric_settings = _numeric_settings(opt.settings, F)
        problem = _extract_problem(opt)
        _log_banner(opt, problem)
        if problem.affine === nothing
            opt.termination_status = MOI.INFEASIBLE
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Inconsistent affine system"
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            return
        end

        particular, nullspace = problem.affine
        barrier_dim = _barrier_dimension(problem)
        if barrier_dim == 0 && size(nullspace, 2) == 0
            objective_value = _exact_objective_value(problem, particular)
            for (index, variable) in enumerate(problem.original_variables)
                opt.variable_primal[variable] = _to_output_type(T, particular[index])
            end
            opt.objective_value = _to_output_type(T, objective_value)
            opt.termination_status = MOI.OPTIMAL
            opt.primal_status = MOI.FEASIBLE_POINT
            opt.dual_status = MOI.NO_SOLUTION
            opt.raw_status = "Solved by affine elimination"
            opt.result_count = 1
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "done by affine elimination")
            return
        elseif barrier_dim == 0
            objective_direction = _objective_nullspace_direction(problem.objective_vector_min, nullspace)
            if any(!iszero, objective_direction)
                opt.termination_status = MOI.DUAL_INFEASIBLE
                opt.primal_status = MOI.NO_SOLUTION
                opt.raw_status = "Unbounded on affine nullspace"
                opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
                _log(opt, "unbounded on affine nullspace")
                return
            end
            objective_value = _exact_objective_value(problem, particular)
            for (index, variable) in enumerate(problem.original_variables)
                opt.variable_primal[variable] = _to_output_type(T, particular[index])
            end
            opt.objective_value = _to_output_type(T, objective_value)
            opt.termination_status = MOI.OPTIMAL
            opt.primal_status = MOI.FEASIBLE_POINT
            opt.dual_status = MOI.NO_SOLUTION
            opt.raw_status = "Constant on affine feasible set"
            opt.result_count = 1
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "done on affine feasible set")
            return
        end

        numeric_blocks = _numeric_blocks(problem.blocks)
        numeric_affine = _numeric_affine_data(problem, F)
        anchor = nothing
        phase2_initial_point = nothing
        dual_rows = nothing
        dual_slack = nothing
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
                if attempt.candidate !== nothing
                    phase2_initial_point = _to_working_array(F, attempt.candidate)
                end
            end
        end

        if anchor === nothing && _phase1_backend(opt.settings) == :native
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
                    anchor = _phase1_exact_feasible_point(x, problem, opt.settings, numeric_affine)
                    last_phase1_recovery_probe = residual
                end
                if residual <= numeric_settings.feasibility_tolerance || anchor !== nothing
                    break
                end
                penalty *= numeric_settings.penalty_growth
            end

            if anchor === nothing && last_phase1_residual <= F(1.0e-6)
                anchor = _phase1_exact_feasible_point(x, problem, opt.settings, numeric_affine)
            end
        end
        if anchor === nothing
            opt.termination_status = MOI.NUMERICAL_ERROR
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Exact interior recovery failed"
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "Phase I exact recovery failed")
            return
        end

        x_exact = anchor
        if size(nullspace, 2) > 0 && any(!iszero, problem.objective_vector_min)
            try
                phase2_result = _phase2_exact_solution(
                    opt,
                    problem,
                    anchor,
                    barrier_dim,
                    F;
                    initial_point = phase2_initial_point,
                    subtitle = "Objective path-following",
                )
                x_exact = phase2_result.x_exact
                dual_recovery = _recover_dual_kkt(
                    problem,
                    phase2_result.x_numeric,
                    phase2_result.barrier_parameter,
                    numeric_blocks,
                    numeric_settings,
                )
                if dual_recovery !== nothing
                    dual_rows, dual_slack = dual_recovery
                end
            catch err
                opt.termination_status = MOI.NUMERICAL_ERROR
                opt.primal_status = MOI.NO_SOLUTION
                opt.raw_status = "Phase II failed"
                opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
                _log(opt, "Phase II failed: $(typeof(err))")
                return
            end
        end

        objective_value = _exact_objective_value(problem, x_exact)
        for (index, variable) in enumerate(problem.original_variables)
            opt.variable_primal[variable] = _to_output_type(T, x_exact[index])
        end
        _populate_constraint_results!(
            opt,
            problem,
            x_exact;
            dual_rows = dual_rows,
            dual_slack = dual_slack,
        )
        if _dual_postsolve_backend(opt.settings) == :hypatia
            if _hypatia_storage_dual_postsolve!(opt, problem)
                dual_rows = zeros(F, 0)
            end
        end
        opt.objective_value = _to_output_type(T, objective_value)
        opt.termination_status = MOI.OPTIMAL
        opt.primal_status = MOI.FEASIBLE_POINT
        opt.dual_status = dual_rows === nothing ? MOI.NO_SOLUTION : MOI.FEASIBLE_POINT
        opt.raw_status = "Solved"
        opt.result_count = 1
        opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
        _log_raw(opt)
        _log(opt, "done in " * @sprintf("%.3f", opt.solve_time_sec) * "s, objective=$(objective_value)")
    end)
    return
end
