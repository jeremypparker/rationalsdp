# Top-level solve orchestration and primal result reporting.

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
    func::MOI.ScalarQuadraticFunction{T},
    variable_primal::Dict{MOI.VariableIndex,T},
) where {T}
    value = func.constant
    for term in func.affine_terms
        value += term.coefficient * variable_primal[term.variable]
    end
    for term in func.quadratic_terms
        value +=
            term.coefficient *
            variable_primal[term.variable_1] *
            variable_primal[term.variable_2]
    end
    return value
end

function _constraint_primal_value(
    func::MOI.VectorQuadraticFunction{T},
    variable_primal::Dict{MOI.VariableIndex,T},
) where {T}
    values = copy(func.constants)
    for term in func.affine_terms
        values[term.output_index] +=
            term.scalar_term.coefficient * variable_primal[term.scalar_term.variable]
    end
    for term in func.quadratic_terms
        scalar_term = term.scalar_term
        values[term.output_index] +=
            scalar_term.coefficient *
            variable_primal[scalar_term.variable_1] *
            variable_primal[scalar_term.variable_2]
    end
    return values
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

function _populate_constraint_results!(
    opt::Optimizer{T},
    problem::ProblemData,
    x_exact::Vector{ExactRational},
) where {T<:Real}
    scalar_constraint_types = (
        (MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarAffineFunction{T}, MOI.Interval{T}),
        (MOI.ScalarQuadraticFunction{T}, MOI.EqualTo{T}),
        (MOI.ScalarQuadraticFunction{T}, MOI.GreaterThan{T}),
        (MOI.ScalarQuadraticFunction{T}, MOI.LessThan{T}),
        (MOI.ScalarQuadraticFunction{T}, MOI.Interval{T}),
        (MOI.VariableIndex, MOI.EqualTo{T}),
        (MOI.VariableIndex, MOI.GreaterThan{T}),
        (MOI.VariableIndex, MOI.LessThan{T}),
        (MOI.VariableIndex, MOI.Interval{T}),
    )

    for (F, S) in scalar_constraint_types
        for ci in MOI.get(opt.storage, MOI.ListOfConstraintIndices{F,S}())
            func = MOI.get(opt.storage, MOI.ConstraintFunction(), ci)
            primal_value =
                func isa MOI.VariableIndex ?
                opt.variable_primal[func] :
                _constraint_primal_value(func, opt.variable_primal)
            opt.constraint_primal[ci] = primal_value
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
    end

    for (ci, func) in opt.scalar_quadratic_functions
        opt.constraint_primal[ci] = _constraint_primal_value(func, opt.variable_primal)
    end
    for (ci, func) in opt.quadratic_psd_functions
        opt.constraint_primal[ci] = _constraint_primal_value(func, opt.variable_primal)
    end
    return
end

function MOI.optimize!(opt::Optimizer{T}) where {T}
    stats = FacialReductionStatistics()
    opt.facial_reduction_statistics = stats
    return _with_facial_reduction_statistics(stats) do
        _optimize_impl!(opt)
    end
end

function _optimize_impl!(opt::Optimizer{T}) where {T}
    start_time = time_ns()
    _reset_results!(opt)
    _validate_settings(opt.settings)
    _prepare_facial_reduction_cache!(opt)
    if _try_quasiconvex_parameter_solve!(opt)
        opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
        return
    end
    if !isempty(opt.quadratic_psd_functions) || !isempty(opt.scalar_quadratic_functions)
        throw(_unsupported_quadratic_error(opt))
    end
    _with_working_precision(opt.settings, function (F)
        numeric_settings = _numeric_settings(opt.settings, F)
        _log(opt, "Extracting problem")
        _gc_checkpoint!(opt, "before extraction")
        problem = _extract_problem(opt)
        original_problem = problem
        _gc_checkpoint!(opt, "after extraction")
        _log(opt, "Problem extracted")
        _log_banner(opt, problem)
        if problem.affine === nothing
            opt.termination_status = MOI.INFEASIBLE
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Inconsistent affine system"
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            return
        end
        cached_problem = _apply_loaded_facial_reductions(opt, problem)
        cached_problem_changed =
            length(cached_problem.objective_vector_raw) != length(problem.objective_vector_raw) ||
            size(cached_problem.A) != size(problem.A) ||
            _barrier_dimension(cached_problem) != _barrier_dimension(problem)
        problem = cached_problem
        cached_problem_changed && _log_banner(opt, problem)

        particular, nullspace = problem.affine
        barrier_dim = _barrier_dimension(problem)
        if barrier_dim == 0 && size(nullspace, 2) == 0
            feasibility = _exact_primal_feasibility(original_problem, particular)
            feasibility.ok || begin
                opt.termination_status = MOI.NUMERICAL_ERROR
                opt.primal_status = MOI.NO_SOLUTION
                opt.raw_status = "Exact validation against the original SDP failed: $(feasibility.reason)"
                opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
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
            feasibility = _exact_primal_feasibility(original_problem, particular)
            feasibility.ok || begin
                opt.termination_status = MOI.NUMERICAL_ERROR
                opt.primal_status = MOI.NO_SOLUTION
                opt.raw_status = "Exact validation against the original SDP failed: $(feasibility.reason)"
                opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
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
        phase1_result = _phase1_anchor_attempt(opt, problem, F)
        anchor = phase1_result.anchor
        phase2_initial_point = phase1_result.phase2_initial_point
        phase1_candidate = phase1_result.phase1_candidate
        phase1_dual_slack = phase1_result.phase1_dual_slack
        tentative_face_search_used = false
        feasibility_objective = all(iszero, original_problem.objective_vector_min)
        tentative_fallback_problem = nothing

        facial_reduction_round = 0
        while anchor === nothing &&
              opt.settings.facial_reduction &&
              phase1_candidate !== nothing &&
              facial_reduction_round < opt.settings.facial_reduction_max_rounds
            _log(opt, "Attempting facial reduction")
            reduction_result = _facially_reduce_search_problem(
                opt,
                problem,
                phase1_candidate,
                phase1_dual_slack,
                F,
            )
            reduced_problem = reduction_result.problem
            tentative_face_search_used |= reduction_result.tentative
            tentative_fallback_problem = reduction_result.fallback_problem
            problem_changed =
                length(reduced_problem.objective_vector_raw) != length(problem.objective_vector_raw) ||
                size(reduced_problem.A) != size(problem.A) ||
                _barrier_dimension(reduced_problem) != _barrier_dimension(problem)
            problem_changed || break
            facial_reduction_round += 1
            problem = reduced_problem
            numeric_blocks = _numeric_blocks(problem.blocks)
            _log_banner(opt, problem)
            phase1_result = _phase1_anchor_attempt(opt, problem, F)
            anchor = phase1_result.anchor
            phase2_initial_point = phase1_result.phase2_initial_point
            phase1_candidate = phase1_result.phase1_candidate
            phase1_dual_slack = phase1_result.phase1_dual_slack

            if anchor === nothing && tentative_fallback_problem !== nothing
                fallback_problem = tentative_fallback_problem
                tentative_fallback_problem = nothing
                facial_reduction_round += 1
                problem = fallback_problem
                numeric_blocks = _numeric_blocks(problem.blocks)
                _log(
                    opt,
                    "Tentative batch did not recover an exact interior; retrying its deterministic greedy fallback",
                )
                phase1_result = _phase1_anchor_attempt(opt, problem, F)
                anchor = phase1_result.anchor
                phase2_initial_point = phase1_result.phase2_initial_point
                phase1_candidate = phase1_result.phase1_candidate
                phase1_dual_slack = phase1_result.phase1_dual_slack
            end
        end

        if problem.affine === nothing
            opt.termination_status = MOI.NUMERICAL_ERROR
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Inconsistent affine reduction"
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "Phase I exact recovery failed")
            return
        end
        particular, nullspace = problem.affine
        barrier_dim = _barrier_dimension(problem)

        if anchor === nothing
            opt.termination_status = MOI.NUMERICAL_ERROR
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Exact interior recovery failed"
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "Phase I exact recovery failed")
            return
        end

        x_exact = anchor
        phase2_termination_reason = :optimal
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
                phase2_termination_reason = phase2_result.termination_reason
            catch err
                opt.termination_status = MOI.NUMERICAL_ERROR
                opt.primal_status = MOI.NO_SOLUTION
                opt.raw_status = "Phase II failed"
                opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
                _log(opt, "Phase II failed: $(typeof(err))")
                return
            end
        end

        feasibility = _exact_primal_feasibility(original_problem, x_exact)
        if !feasibility.ok
            opt.termination_status = MOI.NUMERICAL_ERROR
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Exact validation against the original SDP failed: $(feasibility.reason)"
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, opt.raw_status)
            return
        end

        objective_value = _exact_objective_value(problem, x_exact)
        for (index, variable) in enumerate(problem.original_variables)
            opt.variable_primal[variable] = _to_output_type(T, x_exact[index])
        end
        _populate_constraint_results!(opt, problem, x_exact)
        opt.objective_value = _to_output_type(T, objective_value)
        opt.primal_status = MOI.FEASIBLE_POINT
        opt.dual_status = MOI.NO_SOLUTION
        if tentative_face_search_used && !feasibility_objective
            opt.termination_status = MOI.OTHER_LIMIT
            opt.raw_status =
                "Exact feasible point found using a tentative face restriction; optimality for the original SDP is not established"
        elseif phase2_termination_reason == :optimal
            opt.termination_status = MOI.OPTIMAL
            opt.raw_status = tentative_face_search_used ?
                             "Feasible point found by tentative face search and validated exactly against the original SDP" :
                             "Solved"
        elseif phase2_termination_reason == :outer_iteration_limit
            opt.termination_status = MOI.ITERATION_LIMIT
            opt.raw_status = "Phase II outer iteration limit reached"
        elseif phase2_termination_reason == :newton_iteration_limit
            opt.termination_status = MOI.ITERATION_LIMIT
            opt.raw_status = "Phase II Newton iteration limit reached"
        elseif phase2_termination_reason == :line_search_failed
            opt.termination_status = MOI.SLOW_PROGRESS
            opt.raw_status = "Phase II line search failed"
        else
            opt.termination_status = MOI.OTHER_ERROR
            opt.raw_status = "Phase II stopped for an unknown reason"
        end
        opt.result_count = 1
        opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
        _log_raw(opt)
        _log(opt, "done in " * @sprintf("%.3f", opt.solve_time_sec) * "s, objective=$(objective_value)")
    end)
    return
end
