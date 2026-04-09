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
    return
end

function MOI.optimize!(opt::Optimizer{T}) where {T}
    start_time = time_ns()
    _reset_results!(opt)
    _with_working_precision(opt.settings, function (F)
        numeric_settings = _numeric_settings(opt.settings, F)
        _log(opt, "Extracting problem")
        problem = _extract_problem(opt)
        _log(opt, "Problem extracted")
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
        phase1_result = _phase1_anchor_attempt(opt, problem, F)
        anchor = phase1_result.anchor
        phase2_initial_point = phase1_result.phase2_initial_point
        phase1_candidate = phase1_result.phase1_candidate

        facial_reduction_round = 0
        while anchor === nothing &&
              opt.settings.facial_reduction &&
              phase1_candidate !== nothing &&
              facial_reduction_round < opt.settings.facial_reduction_max_rounds
            _log(opt, "Attempting facial reduction")
            reduced_problem = _facially_reduce_problem(opt, problem, phase1_candidate, F)
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
        _populate_constraint_results!(opt, problem, x_exact)
        opt.objective_value = _to_output_type(T, objective_value)
        opt.termination_status = MOI.OPTIMAL
        opt.primal_status = MOI.FEASIBLE_POINT
        opt.dual_status = MOI.NO_SOLUTION
        opt.raw_status = "Solved"
        opt.result_count = 1
        opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
        _log_raw(opt)
        _log(opt, "done in " * @sprintf("%.3f", opt.solve_time_sec) * "s, objective=$(objective_value)")
    end)
    return
end
