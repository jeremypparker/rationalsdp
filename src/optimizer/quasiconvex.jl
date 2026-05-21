# Quasi-convex one-parameter optimization.
#
# This path handles models where the JuMP objective variable multiplies other
# variables in PSD constraints. For a fixed objective value, the model is an
# ordinary SDP, so RationalSDP can solve the explicit optimization by bisection
# over that objective variable.

struct QuasiconvexParameterData{T<:Real}
    parameter::MOI.VariableIndex
    sense::MOI.OptimizationSense
    lower::T
    upper::T
    quadratic_psd_constraints::Vector{
        MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.PositiveSemidefiniteConeTriangle}
    }
    scalar_quadratic_constraints::Vector{Any}
end

struct QuasiconvexCoefficientPatch
    row::Int
    column::Int
    coefficient::ExactRational
end

struct QuasiconvexRhsQuadraticPatch
    row::Int
    coefficient::ExactRational
end

struct QuasiconvexProblemTemplate
    original_variables::Vector{MOI.VariableIndex}
    blocks::Vector{BlockStructure}
    positive_scalars::Vector{Int}
    objective::Vector{ExactRational}
    A_base::Matrix{ExactRational}
    b_base::Vector{ExactRational}
    scalar_constraint_rows::Dict{Any,Vector{Int}}
    psd_constraint_blocks::Dict{Any,Int}
    fixed_parameter_row::Int
    coefficient_patches::Vector{QuasiconvexCoefficientPatch}
    rhs_quadratic_patches::Vector{QuasiconvexRhsQuadraticPatch}
end

function _objective_parameter(storage, ::Type{T}) where {T<:Real}
    sense = MOI.get(storage, MOI.ObjectiveSense())
    sense == MOI.MIN_SENSE || sense == MOI.MAX_SENSE || return nothing
    objective_type = MOI.get(storage, MOI.ObjectiveFunctionType())
    if objective_type == MOI.VariableIndex
        return MOI.get(storage, MOI.ObjectiveFunction{MOI.VariableIndex}())
    elseif objective_type <: MOI.ScalarAffineFunction
        func = MOI.get(storage, MOI.ObjectiveFunction{objective_type}())
        iszero(func.constant) || return nothing
        length(func.terms) == 1 || return nothing
        term = only(func.terms)
        term.coefficient == one(T) || return nothing
        return term.variable
    end
    return nothing
end

function _parameter_bounds(storage, parameter::MOI.VariableIndex, ::Type{T}) where {T<:Real}
    lower = nothing
    upper = nothing
    for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{T}}())
        MOI.get(storage, MOI.ConstraintFunction(), ci) == parameter || continue
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        lower = lower === nothing ? set.lower : max(lower, set.lower)
    end
    for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{T}}())
        MOI.get(storage, MOI.ConstraintFunction(), ci) == parameter || continue
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        upper = upper === nothing ? set.upper : min(upper, set.upper)
    end
    for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Interval{T}}())
        MOI.get(storage, MOI.ConstraintFunction(), ci) == parameter || continue
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        lower = lower === nothing ? set.lower : max(lower, set.lower)
        upper = upper === nothing ? set.upper : min(upper, set.upper)
    end
    lower === nothing && return nothing
    upper === nothing && return nothing
    lower <= upper || return nothing
    return lower, upper
end

function _is_parameter_quadratic_function(
    func::MOI.VectorQuadraticFunction{T},
    parameter::MOI.VariableIndex,
) where {T<:Real}
    for term in func.quadratic_terms
        scalar = term.scalar_term
        if scalar.variable_1 != parameter && scalar.variable_2 != parameter
            return false
        end
    end
    return true
end

function _is_parameter_quadratic_function(
    func::MOI.ScalarQuadraticFunction{T},
    parameter::MOI.VariableIndex,
) where {T<:Real}
    for term in func.quadratic_terms
        if term.variable_1 != parameter && term.variable_2 != parameter
            return false
        end
    end
    return true
end

function _scalar_quadratic_constraint_indices(opt::Optimizer{T}) where {T<:Real}
    scalar_sets = (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval)
    constraints = Any[]
    for S in scalar_sets
        append!(
            constraints,
            MOI.get(opt, MOI.ListOfConstraintIndices{MOI.ScalarQuadraticFunction{T},S{T}}()),
        )
    end
    return constraints
end

function _detect_quasiconvex_parameter(opt::Optimizer{T}) where {T<:Real}
    storage = opt.storage
    quadratic_psd_constraints = collect(keys(opt.quadratic_psd_functions))
    scalar_quadratic_constraints = _scalar_quadratic_constraint_indices(opt)
    isempty(quadratic_psd_constraints) && isempty(scalar_quadratic_constraints) && return nothing

    parameter = _objective_parameter(storage, T)
    parameter === nothing && return nothing
    bounds = _parameter_bounds(storage, parameter, T)
    bounds === nothing && return nothing
    lower, upper = bounds

    for ci in quadratic_psd_constraints
        func = opt.quadratic_psd_functions[ci]
        _is_parameter_quadratic_function(func, parameter) || return nothing
    end
    for ci in scalar_quadratic_constraints
        func = opt.scalar_quadratic_functions[ci]
        _is_parameter_quadratic_function(func, parameter) || return nothing
    end

    return QuasiconvexParameterData{T}(
        parameter,
        MOI.get(storage, MOI.ObjectiveSense()),
        lower,
        upper,
        quadratic_psd_constraints,
        scalar_quadratic_constraints,
    )
end

function _fixed_parameter_feasible(
    opt::Optimizer{T},
    template::QuasiconvexProblemTemplate,
    fixed_value::T,
) where {T<:Real}
    problem = _instantiate_fixed_parameter_problem(opt, template, fixed_value)
    return _with_working_precision(opt.settings, function (F)
        x_exact = _quasiconvex_feasible_point(opt, problem, F)
        return x_exact !== nothing, problem, x_exact
    end)
end

function _populate_from_quasiconvex_child!(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    problem::ProblemData,
    x_exact::Vector{ExactRational},
    objective_value::T,
) where {T<:Real}
    for (index, variable) in enumerate(problem.original_variables)
        opt.variable_primal[variable] = _to_output_type(T, x_exact[index])
    end
    opt.variable_primal[data.parameter] = objective_value
    opt.objective_value = objective_value
    opt.termination_status = MOI.OPTIMAL
    opt.primal_status = MOI.FEASIBLE_POINT
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status = "Solved by quasi-convex parameter search"
    opt.result_count = 1
    return
end

function _append_exact_row(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    row::Vector{ExactRational},
    rhs,
)
    return vcat(A, reshape(row, 1, :)), vcat(b, _exact_rational(rhs))
end

function _append_zero_column(A::Matrix{ExactRational})
    return hcat(A, zeros(ExactRational, size(A, 1)))
end

function _append_parameterized_scalar_quadratic_row(
    opt::Optimizer,
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    positive_scalars::Vector{Int},
    coefficient_patches::Vector{QuasiconvexCoefficientPatch},
    rhs_quadratic_patches::Vector{QuasiconvexRhsQuadraticPatch},
    func::MOI.ScalarQuadraticFunction,
    rhs::ExactRational,
    slack_sign::Int,
    parameter::MOI.VariableIndex,
    var_to_pos::Dict{MOI.VariableIndex,Int},
)
    if slack_sign != 0
        A = _append_zero_column(A)
    end
    row = zeros(ExactRational, size(A, 2))
    for term in func.affine_terms
        row[var_to_pos[term.variable]] += _exact_rational(term.coefficient)
    end
    if slack_sign != 0
        slack_position = size(A, 2)
        row[slack_position] = slack_sign
        push!(positive_scalars, slack_position)
    end

    row_index = size(A, 1) + 1
    for term in func.quadratic_terms
        coefficient = _exact_rational(term.coefficient)
        if term.variable_1 == parameter && term.variable_2 == parameter
            push!(rhs_quadratic_patches, QuasiconvexRhsQuadraticPatch(row_index, coefficient))
        elseif term.variable_1 == parameter
            push!(
                coefficient_patches,
                QuasiconvexCoefficientPatch(row_index, var_to_pos[term.variable_2], coefficient),
            )
        elseif term.variable_2 == parameter
            push!(
                coefficient_patches,
                QuasiconvexCoefficientPatch(row_index, var_to_pos[term.variable_1], coefficient),
            )
        else
            throw(_unsupported_quadratic_error(opt))
        end
    end

    return _append_exact_row(A, b, row, rhs - _exact_rational(func.constant))
end

function _add_parameterized_scalar_quadratic_constraints(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    positive_scalars::Vector{Int},
    scalar_constraint_rows::Dict{Any,Vector{Int}},
    coefficient_patches::Vector{QuasiconvexCoefficientPatch},
    rhs_quadratic_patches::Vector{QuasiconvexRhsQuadraticPatch},
    var_to_pos::Dict{MOI.VariableIndex,Int},
) where {T<:Real}
    for ci in data.scalar_quadratic_constraints
        func = opt.scalar_quadratic_functions[ci]
        set = opt.scalar_quadratic_sets[ci]
        first_row = size(A, 1) + 1
        if set isa MOI.EqualTo
            A, b = _append_parameterized_scalar_quadratic_row(
                opt,
                A,
                b,
                positive_scalars,
                coefficient_patches,
                rhs_quadratic_patches,
                func,
                _exact_rational(set.value),
                0,
                data.parameter,
                var_to_pos,
            )
            scalar_constraint_rows[ci] = [first_row]
        elseif set isa MOI.GreaterThan
            A, b = _append_parameterized_scalar_quadratic_row(
                opt,
                A,
                b,
                positive_scalars,
                coefficient_patches,
                rhs_quadratic_patches,
                func,
                _exact_rational(set.lower),
                -1,
                data.parameter,
                var_to_pos,
            )
            scalar_constraint_rows[ci] = [first_row]
        elseif set isa MOI.LessThan
            A, b = _append_parameterized_scalar_quadratic_row(
                opt,
                A,
                b,
                positive_scalars,
                coefficient_patches,
                rhs_quadratic_patches,
                func,
                _exact_rational(set.upper),
                1,
                data.parameter,
                var_to_pos,
            )
            scalar_constraint_rows[ci] = [first_row]
        elseif set isa MOI.Interval
            A, b = _append_parameterized_scalar_quadratic_row(
                opt,
                A,
                b,
                positive_scalars,
                coefficient_patches,
                rhs_quadratic_patches,
                func,
                _exact_rational(set.lower),
                -1,
                data.parameter,
                var_to_pos,
            )
            A, b = _append_parameterized_scalar_quadratic_row(
                opt,
                A,
                b,
                positive_scalars,
                coefficient_patches,
                rhs_quadratic_patches,
                func,
                _exact_rational(set.upper),
                1,
                data.parameter,
                var_to_pos,
            )
            scalar_constraint_rows[ci] = [first_row, first_row + 1]
        else
            throw(_unsupported_quadratic_error(opt))
        end
    end
    return A, b, positive_scalars, scalar_constraint_rows, coefficient_patches, rhs_quadratic_patches
end

function _add_parameterized_psd_blocks(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    blocks::Vector{BlockStructure},
    psd_constraint_blocks::Dict{Any,Int},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    original_dimension::Int,
    affine_psd_auxiliary_count::Int,
) where {T<:Real}
    storage = opt.storage
    original_variables = MOI.get(storage, MOI.ListOfVariableIndices())
    var_to_pos = Dict{MOI.VariableIndex,Int}(vi => i for (i, vi) in enumerate(original_variables))
    next_auxiliary_position = original_dimension + affine_psd_auxiliary_count + 1
    coefficient_patches = QuasiconvexCoefficientPatch[]
    rhs_quadratic_patches = QuasiconvexRhsQuadraticPatch[]

    for ci in data.quadratic_psd_constraints
        func = opt.quadratic_psd_functions[ci]
        set = opt.quadratic_psd_sets[ci]
        positions = _triangle_positions(set.side_dimension)
        length(positions) == MOI.output_dimension(func) || error("Malformed quadratic PSD block.")
        global_positions = collect(next_auxiliary_position:(next_auxiliary_position + length(positions) - 1))
        diagonal_positions = Int[]
        for (local_index, (i, j)) in enumerate(positions)
            i == j && push!(diagonal_positions, global_positions[local_index])
        end
        push!(
            blocks,
            BlockStructure(
                set.side_dimension,
                fill(nothing, length(positions)),
                global_positions,
                positions,
                diagonal_positions,
            ),
        )
        psd_constraint_blocks[ci] = length(blocks)

        row_indices = [Int[] for _ in positions]
        row_values = [ExactRational[] for _ in positions]
        offsets = ExactRational[_exact_rational(value) for value in func.constants]

        for term in func.affine_terms
            push!(row_indices[term.output_index], var_to_pos[term.scalar_term.variable])
            push!(row_values[term.output_index], _exact_rational(term.scalar_term.coefficient))
        end

        first_row = size(A, 1) + 1
        for term in func.quadratic_terms
            scalar = term.scalar_term
            coefficient = _exact_rational(scalar.coefficient)
            row = first_row + term.output_index - 1
            if scalar.variable_1 == data.parameter && scalar.variable_2 == data.parameter
                push!(rhs_quadratic_patches, QuasiconvexRhsQuadraticPatch(row, coefficient))
            elseif scalar.variable_1 == data.parameter
                push!(
                    coefficient_patches,
                    QuasiconvexCoefficientPatch(row, var_to_pos[scalar.variable_2], coefficient),
                )
            elseif scalar.variable_2 == data.parameter
                push!(
                    coefficient_patches,
                    QuasiconvexCoefficientPatch(row, var_to_pos[scalar.variable_1], coefficient),
                )
            else
                throw(_unsupported_quadratic_psd_error(opt))
            end
        end

        for local_index in eachindex(positions)
            row = zeros(ExactRational, size(A, 2))
            for (index, value) in zip(row_indices[local_index], row_values[local_index])
                row[index] += value
            end
            row[global_positions[local_index]] -= 1 // 1
            A, b = _append_exact_row(A, b, row, -offsets[local_index])
        end

        next_auxiliary_position += length(positions)
    end

    return blocks, psd_constraint_blocks, A, b, coefficient_patches, rhs_quadratic_patches
end

function _fixed_parameter_template(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
) where {T<:Real}
    storage = opt.storage
    original_variables = MOI.get(storage, MOI.ListOfVariableIndices())
    original_dimension = length(original_variables)
    var_to_pos = Dict{MOI.VariableIndex,Int}(vi => i for (i, vi) in enumerate(original_variables))

    psd_variable_indices = MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
    psd_affine_indices = MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
    affine_psd_auxiliary_count = sum(
        (MOI.output_dimension(MOI.get(storage, MOI.ConstraintFunction(), ci)) for ci in psd_affine_indices);
        init = 0,
    )
    quadratic_psd_auxiliary_count = sum(
        (MOI.output_dimension(opt.quadratic_psd_functions[ci]) for ci in data.quadratic_psd_constraints);
        init = 0,
    )
    base_dimension = original_dimension + affine_psd_auxiliary_count + quadratic_psd_auxiliary_count

    blocks = BlockStructure[]
    psd_constraint_blocks = Dict{Any,Int}()
    seen_psd_variables = Dict{MOI.VariableIndex,Int}()
    for ci in psd_variable_indices
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        positions = _triangle_positions(set.side_dimension)
        length(func.variables) == length(positions) || error("Malformed PSD block.")
        global_positions = Int[]
        diagonal_positions = Int[]
        for (local_index, variable) in enumerate(func.variables)
            haskey(var_to_pos, variable) || error("Unknown variable in PSD block.")
            global_index = var_to_pos[variable]
            push!(global_positions, global_index)
            haskey(seen_psd_variables, variable) &&
                error("Each variable may belong to at most one PSD block.")
            seen_psd_variables[variable] = length(blocks) + 1
            positions[local_index][1] == positions[local_index][2] &&
                push!(diagonal_positions, global_index)
        end
        push!(
            blocks,
            BlockStructure(
                set.side_dimension,
                Union{Nothing,MOI.VariableIndex}[func.variables...],
                global_positions,
                positions,
                diagonal_positions,
            ),
        )
        psd_constraint_blocks[ci] = length(blocks)
    end

    A, b, positive_scalars, _, _, _, scalar_constraint_rows =
        _extract_constraint_system(
            opt,
            storage,
            original_variables,
            var_to_pos,
            base_dimension,
            psd_affine_indices,
            blocks,
            psd_constraint_blocks,
        )

    blocks,
    psd_constraint_blocks,
    A,
    b,
    coefficient_patches,
    rhs_quadratic_patches = _add_parameterized_psd_blocks(
        opt,
        data,
        blocks,
        psd_constraint_blocks,
        A,
        b,
        original_dimension,
        affine_psd_auxiliary_count,
    )

    A,
    b,
    positive_scalars,
    scalar_constraint_rows,
    coefficient_patches,
    rhs_quadratic_patches = _add_parameterized_scalar_quadratic_constraints(
        opt,
        data,
        A,
        b,
        positive_scalars,
        scalar_constraint_rows,
        coefficient_patches,
        rhs_quadratic_patches,
        var_to_pos,
    )

    fixed_row = zeros(ExactRational, size(A, 2))
    fixed_row[var_to_pos[data.parameter]] = 1 // 1
    A, b = _append_exact_row(A, b, fixed_row, 0 // 1)
    fixed_parameter_row = length(b)
    objective = zeros(ExactRational, size(A, 2))

    return QuasiconvexProblemTemplate(
        original_variables,
        blocks,
        positive_scalars,
        objective,
        A,
        b,
        scalar_constraint_rows,
        psd_constraint_blocks,
        fixed_parameter_row,
        coefficient_patches,
        rhs_quadratic_patches,
    )
end

function _instantiate_fixed_parameter_problem(
    opt::Optimizer{T},
    template::QuasiconvexProblemTemplate,
    fixed_value::T,
) where {T<:Real}
    fixed_exact = _exact_rational(fixed_value)
    A = copy(template.A_base)
    b = copy(template.b_base)
    for patch in template.coefficient_patches
        A[patch.row, patch.column] += patch.coefficient * fixed_exact
    end
    for patch in template.rhs_quadratic_patches
        b[patch.row] -= patch.coefficient * fixed_exact * fixed_exact
    end
    b[template.fixed_parameter_row] = fixed_exact

    affine = _solve_affine_system(A, b; checkpoint = label -> _gc_checkpoint!(opt, label))
    positive_scalars, pruned_scalar_faces =
        _prune_positive_scalar_faces(template.positive_scalars, affine)
    blocks, A, b, affine, pruned_directions =
        _prune_psd_faces(copy(template.blocks), A, b, affine)
    if pruned_scalar_faces > 0 || pruned_directions > 0
        _log(
            opt,
            "quasi-convex fixed-parameter problem pruned $(pruned_scalar_faces) scalar and $(pruned_directions) PSD direction(s)",
        )
    end

    return ProblemData(
        template.original_variables,
        blocks,
        positive_scalars,
        copy(template.objective),
        0 // 1,
        copy(template.objective),
        A,
        b,
        affine,
        nothing,
        template.scalar_constraint_rows,
        template.psd_constraint_blocks,
    )
end

function _add_fixed_parameter_psd_blocks(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    fixed_value::T,
    blocks::Vector{BlockStructure},
    psd_constraint_blocks::Dict{Any,Int},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    original_dimension::Int,
    affine_psd_auxiliary_count::Int,
) where {T<:Real}
    storage = opt.storage
    original_variables = MOI.get(storage, MOI.ListOfVariableIndices())
    var_to_pos = Dict{MOI.VariableIndex,Int}(vi => i for (i, vi) in enumerate(original_variables))
    next_auxiliary_position = original_dimension + affine_psd_auxiliary_count + 1
    fixed_exact = _exact_rational(fixed_value)

    for ci in data.quadratic_psd_constraints
        func = opt.quadratic_psd_functions[ci]
        set = opt.quadratic_psd_sets[ci]
        positions = _triangle_positions(set.side_dimension)
        length(positions) == MOI.output_dimension(func) || error("Malformed quadratic PSD block.")
        global_positions = collect(next_auxiliary_position:(next_auxiliary_position + length(positions) - 1))
        diagonal_positions = Int[]
        for (local_index, (i, j)) in enumerate(positions)
            i == j && push!(diagonal_positions, global_positions[local_index])
        end
        push!(
            blocks,
            BlockStructure(
                set.side_dimension,
                fill(nothing, length(positions)),
                global_positions,
                positions,
                diagonal_positions,
            ),
        )
        psd_constraint_blocks[ci] = length(blocks)

        row_indices = [Int[] for _ in positions]
        row_values = [ExactRational[] for _ in positions]
        offsets = ExactRational[_exact_rational(value) for value in func.constants]

        for term in func.affine_terms
            push!(row_indices[term.output_index], var_to_pos[term.scalar_term.variable])
            push!(row_values[term.output_index], _exact_rational(term.scalar_term.coefficient))
        end

        for term in func.quadratic_terms
            scalar = term.scalar_term
            coefficient = _exact_rational(scalar.coefficient)
            if scalar.variable_1 == data.parameter && scalar.variable_2 == data.parameter
                offsets[term.output_index] += coefficient * fixed_exact * fixed_exact
            elseif scalar.variable_1 == data.parameter
                push!(row_indices[term.output_index], var_to_pos[scalar.variable_2])
                push!(row_values[term.output_index], coefficient * fixed_exact)
            elseif scalar.variable_2 == data.parameter
                push!(row_indices[term.output_index], var_to_pos[scalar.variable_1])
                push!(row_values[term.output_index], coefficient * fixed_exact)
            else
                throw(_unsupported_quadratic_psd_error(opt))
            end
        end

        for local_index in eachindex(positions)
            row = zeros(ExactRational, size(A, 2))
            for (index, value) in zip(row_indices[local_index], row_values[local_index])
                row[index] += value
            end
            row[global_positions[local_index]] -= 1 // 1
            A, b = _append_exact_row(A, b, row, -offsets[local_index])
        end

        next_auxiliary_position += length(positions)
    end

    return blocks, psd_constraint_blocks, A, b
end

function _fixed_parameter_problem(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    fixed_value::T,
) where {T<:Real}
    storage = opt.storage
    original_variables = MOI.get(storage, MOI.ListOfVariableIndices())
    original_dimension = length(original_variables)
    var_to_pos = Dict{MOI.VariableIndex,Int}(vi => i for (i, vi) in enumerate(original_variables))

    psd_variable_indices = MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
    psd_affine_indices = MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
    affine_psd_auxiliary_count = sum(
        (MOI.output_dimension(MOI.get(storage, MOI.ConstraintFunction(), ci)) for ci in psd_affine_indices);
        init = 0,
    )
    quadratic_psd_auxiliary_count = sum(
        (MOI.output_dimension(opt.quadratic_psd_functions[ci]) for ci in data.quadratic_psd_constraints);
        init = 0,
    )
    base_dimension = original_dimension + affine_psd_auxiliary_count + quadratic_psd_auxiliary_count

    blocks = BlockStructure[]
    psd_constraint_blocks = Dict{Any,Int}()
    seen_psd_variables = Dict{MOI.VariableIndex,Int}()
    for ci in psd_variable_indices
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        positions = _triangle_positions(set.side_dimension)
        length(func.variables) == length(positions) || error("Malformed PSD block.")
        global_positions = Int[]
        diagonal_positions = Int[]
        for (local_index, variable) in enumerate(func.variables)
            haskey(var_to_pos, variable) || error("Unknown variable in PSD block.")
            global_index = var_to_pos[variable]
            push!(global_positions, global_index)
            haskey(seen_psd_variables, variable) &&
                error("Each variable may belong to at most one PSD block.")
            seen_psd_variables[variable] = length(blocks) + 1
            positions[local_index][1] == positions[local_index][2] &&
                push!(diagonal_positions, global_index)
        end
        push!(
            blocks,
            BlockStructure(
                set.side_dimension,
                Union{Nothing,MOI.VariableIndex}[func.variables...],
                global_positions,
                positions,
                diagonal_positions,
            ),
        )
        psd_constraint_blocks[ci] = length(blocks)
    end

    A, b, positive_scalars, _, _, _, scalar_constraint_rows =
        _extract_constraint_system(
            opt,
            storage,
            original_variables,
            var_to_pos,
            base_dimension,
            psd_affine_indices,
            blocks,
            psd_constraint_blocks,
        )

    blocks, psd_constraint_blocks, A, b = _add_fixed_parameter_psd_blocks(
        opt,
        data,
        fixed_value,
        blocks,
        psd_constraint_blocks,
        A,
        b,
        original_dimension,
        affine_psd_auxiliary_count,
    )

    fixed_row = zeros(ExactRational, size(A, 2))
    fixed_row[var_to_pos[data.parameter]] = 1 // 1
    A, b = _append_exact_row(A, b, fixed_row, _exact_rational(fixed_value))

    dimension = size(A, 2)
    objective = zeros(ExactRational, dimension)
    affine = _solve_affine_system(A, b; checkpoint = label -> _gc_checkpoint!(opt, label))
    positive_scalars, pruned_scalar_faces = _prune_positive_scalar_faces(positive_scalars, affine)
    blocks, A, b, affine, pruned_directions = _prune_psd_faces(blocks, A, b, affine)
    if pruned_scalar_faces > 0 || pruned_directions > 0
        _log(
            opt,
            "quasi-convex fixed-parameter problem pruned $(pruned_scalar_faces) scalar and $(pruned_directions) PSD direction(s)",
        )
    end

    return ProblemData(
        original_variables,
        blocks,
        positive_scalars,
        objective,
        0 // 1,
        copy(objective),
        A,
        b,
        affine,
        nothing,
        scalar_constraint_rows,
        psd_constraint_blocks,
    )
end

function _quasiconvex_feasible_point(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    particular, nullspace = problem.affine
    barrier_dim = _barrier_dimension(problem)
    barrier_dim == 0 && return particular

    phase1_result = _phase1_anchor_attempt(opt, problem, F)
    anchor = phase1_result.anchor
    phase1_candidate = phase1_result.phase1_candidate

    facial_reduction_round = 0
    while anchor === nothing &&
          opt.settings.facial_reduction &&
          phase1_candidate !== nothing &&
          facial_reduction_round < opt.settings.facial_reduction_max_rounds
        reduced_problem = _facially_reduce_problem(opt, problem, phase1_candidate, F)
        problem_changed =
            length(reduced_problem.objective_vector_raw) != length(problem.objective_vector_raw) ||
            size(reduced_problem.A) != size(problem.A) ||
            _barrier_dimension(reduced_problem) != _barrier_dimension(problem)
        problem_changed || break
        facial_reduction_round += 1
        problem = reduced_problem
        phase1_result = _phase1_anchor_attempt(opt, problem, F)
        anchor = phase1_result.anchor
        phase1_candidate = phase1_result.phase1_candidate
    end

    return anchor
end

function _set_quasiconvex_endpoint_infeasible!(opt::Optimizer, endpoint_name::String)
    opt.termination_status = MOI.INFEASIBLE
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status = "Quasi-convex parameter $(endpoint_name) bound is infeasible"
    return true
end

function _quasiconvex_parameter_search!(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    template::QuasiconvexProblemTemplate,
) where {T<:Real}
    feasible_side_is_upper = data.sense == MOI.MIN_SENSE
    best_endpoint = feasible_side_is_upper ? data.lower : data.upper
    fallback_endpoint = feasible_side_is_upper ? data.upper : data.lower
    fallback_name = feasible_side_is_upper ? "upper" : "lower"

    best_feasible, best_problem, best_point =
        _fixed_parameter_feasible(opt, template, best_endpoint)
    if best_feasible
        _populate_from_quasiconvex_child!(opt, data, best_problem, best_point, best_endpoint)
        return true
    end

    fallback_feasible, fallback_problem, fallback_point =
        _fixed_parameter_feasible(opt, template, fallback_endpoint)
    fallback_feasible || return _set_quasiconvex_endpoint_infeasible!(opt, fallback_name)

    lower = data.lower
    upper = data.upper
    best_value = fallback_endpoint
    best_problem = fallback_problem
    best_point = fallback_point

    for _ in 1:opt.settings.quasiconvex_bisection_iterations
        midpoint = (lower + upper) / 2
        feasible, problem, point = _fixed_parameter_feasible(opt, template, midpoint)
        if feasible
            best_value = midpoint
            best_problem = problem
            best_point = point
            if feasible_side_is_upper
                upper = midpoint
            else
                lower = midpoint
            end
        elseif feasible_side_is_upper
            lower = midpoint
        else
            upper = midpoint
        end
    end

    _populate_from_quasiconvex_child!(opt, data, best_problem, best_point, best_value)
    return true
end

function _try_quasiconvex_parameter_solve!(opt::Optimizer{T}) where {T<:Real}
    data = _detect_quasiconvex_parameter(opt)
    data === nothing && return false
    template = _fixed_parameter_template(opt, data)
    return _quasiconvex_parameter_search!(opt, data, template)
end

function _unsupported_quadratic_message()
    return (
        "RationalSDP only supports quadratic constraints for " *
        "one-parameter quasi-convex optimization: the model must minimize a " *
        "single scalar objective variable with finite lower and upper bounds, " *
        "and every quadratic term in each quadratic constraint must contain that " *
        "objective variable. General quadratic SDP constraints are not supported."
    )
end

function _unsupported_quadratic_psd_error(::Optimizer{T}) where {T<:Real}
    return MOI.UnsupportedConstraint{
        MOI.VectorQuadraticFunction{T},
        MOI.PositiveSemidefiniteConeTriangle,
    }(_unsupported_quadratic_message())
end

function _unsupported_quadratic_error(::Optimizer{T}) where {T<:Real}
    return MOI.UnsupportedConstraint{MOI.ScalarQuadraticFunction{T},MOI.EqualTo{T}}(
        _unsupported_quadratic_message(),
    )
end
