# Top-level solve orchestration and result reporting.

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
                x_exact = _phase2_exact_solution(
                    opt,
                    problem,
                    anchor,
                    barrier_dim,
                    F;
                    initial_point = phase2_initial_point,
                    subtitle = "Objective path-following",
                )
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
