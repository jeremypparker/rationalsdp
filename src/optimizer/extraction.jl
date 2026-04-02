# JuMP/MOI model extraction into the solver's internal SDP form.

function _extract_affine_row(
    func::MOI.ScalarAffineFunction,
    var_to_pos::Dict{MOI.VariableIndex,Int},
    p::Int,
)
    row = zeros(ExactRational, p)
    for term in func.terms
        row[var_to_pos[term.variable]] += _exact_rational(term.coefficient)
    end
    return row, _exact_rational(func.constant)
end

function _single_variable_row(
    variable::MOI.VariableIndex,
    var_to_pos::Dict{MOI.VariableIndex,Int},
    p::Int,
)
    row = zeros(ExactRational, p)
    row[var_to_pos[variable]] = 1 // 1
    return row
end

function _extract_vector_affine_rows(
    func::MOI.VectorAffineFunction,
    var_to_pos::Dict{MOI.VariableIndex,Int},
    p::Int,
)
    dimension = MOI.output_dimension(func)
    rows = [zeros(ExactRational, p) for _ in 1:dimension]
    offsets = ExactRational[_exact_rational(value) for value in func.constants]
    for term in func.terms
        rows[term.output_index][var_to_pos[term.scalar_term.variable]] +=
            _exact_rational(term.scalar_term.coefficient)
    end
    return rows, offsets
end

function _objective_data(storage, vars, var_to_pos)
    p = length(vars)
    c_raw = zeros(ExactRational, p)
    constant = zero(ExactRational)
    sense = MOI.get(storage, MOI.ObjectiveSense())
    if sense == MOI.FEASIBILITY_SENSE
        return c_raw, constant, c_raw
    end
    objective_type = MOI.get(storage, MOI.ObjectiveFunctionType())
    if objective_type == MOI.VariableIndex
        vi = MOI.get(storage, MOI.ObjectiveFunction{MOI.VariableIndex}())
        c_raw[var_to_pos[vi]] = 1 // 1
    elseif objective_type <: MOI.ScalarAffineFunction
        func = MOI.get(storage, MOI.ObjectiveFunction{objective_type}())
        for term in func.terms
            c_raw[var_to_pos[term.variable]] += _exact_rational(term.coefficient)
        end
        constant = _exact_rational(func.constant)
    else
        error("Unsupported objective function type: $objective_type")
    end
    c_min = sense == MOI.MAX_SENSE ? -c_raw : c_raw
    return c_raw, constant, c_min
end

function _push_equality!(
    templates::Vector{EquationTemplate},
    coefficients::Vector{ExactRational},
    rhs::ExactRational,
)
    push!(templates, EquationTemplate(coefficients, rhs, 0))
    return
end

function _push_greater_than!(
    templates::Vector{EquationTemplate},
    coefficients::Vector{ExactRational},
    rhs::ExactRational,
)
    push!(templates, EquationTemplate(coefficients, rhs, -1))
    return
end

function _push_less_than!(
    templates::Vector{EquationTemplate},
    coefficients::Vector{ExactRational},
    rhs::ExactRational,
)
    push!(templates, EquationTemplate(coefficients, rhs, 1))
    return
end

function _assemble_system(
    templates::Vector{EquationTemplate},
    original_dimension::Int,
)
    slack_count = count(template -> template.slack_sign != 0, templates)
    total_dimension = original_dimension + slack_count
    A = zeros(ExactRational, length(templates), total_dimension)
    b = Vector{ExactRational}(undef, length(templates))
    positive_scalars = Int[]
    next_slack = original_dimension + 1
    for (row_index, template) in enumerate(templates)
        if original_dimension > 0
            A[row_index, 1:original_dimension] = template.coefficients
        end
        b[row_index] = template.rhs
        if template.slack_sign != 0
            A[row_index, next_slack] = template.slack_sign
            push!(positive_scalars, next_slack)
            next_slack += 1
        end
    end
    return A, b, positive_scalars
end

function _extract_problem(opt::Optimizer{T}) where {T}
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
    psd_auxiliary_count = sum(
        (MOI.output_dimension(MOI.get(storage, MOI.ConstraintFunction(), ci)) for ci in psd_affine_indices);
        init = 0,
    )
    base_dimension = original_dimension + psd_auxiliary_count

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
            if haskey(seen_psd_variables, variable)
                error("Each variable may belong to at most one PSD block.")
            end
            seen_psd_variables[variable] = length(blocks) + 1
            if positions[local_index][1] == positions[local_index][2]
                push!(diagonal_positions, global_index)
            end
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

    templates = EquationTemplate[]
    next_auxiliary_position = original_dimension + 1
    for ci in psd_affine_indices
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        positions = _triangle_positions(set.side_dimension)
        length(positions) == MOI.output_dimension(func) || error("Malformed affine PSD block.")
        rows, offsets = _extract_vector_affine_rows(func, var_to_pos, base_dimension)
        global_positions = collect(next_auxiliary_position:(next_auxiliary_position + length(positions) - 1))
        diagonal_positions = Int[]
        for local_index in eachindex(positions)
            row = rows[local_index]
            row[global_positions[local_index]] -= 1 // 1
            _push_equality!(templates, row, -offsets[local_index])
            if positions[local_index][1] == positions[local_index][2]
                push!(diagonal_positions, global_positions[local_index])
            end
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
        next_auxiliary_position += length(positions)
    end

    c_original_raw, constant_raw, c_original_min = _objective_data(storage, original_variables, var_to_pos)
    scalar_constraint_rows = Dict{Any,Vector{Int}}()

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.EqualTo{T},
        }(),
    )
        row_index = length(templates) + 1
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, base_dimension)
        _push_equality!(templates, row, _exact_rational(set.value) - offset)
        scalar_constraint_rows[ci] = [row_index]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.GreaterThan{T},
        }(),
    )
        row_index = length(templates) + 1
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, base_dimension)
        _push_greater_than!(templates, row, _exact_rational(set.lower) - offset)
        scalar_constraint_rows[ci] = [row_index]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.LessThan{T},
        }(),
    )
        row_index = length(templates) + 1
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, base_dimension)
        _push_less_than!(templates, row, _exact_rational(set.upper) - offset)
        scalar_constraint_rows[ci] = [row_index]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.Interval{T},
        }(),
    )
        row_index = length(templates) + 1
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, base_dimension)
        _push_greater_than!(templates, copy(row), _exact_rational(set.lower) - offset)
        _push_less_than!(templates, row, _exact_rational(set.upper) - offset)
        scalar_constraint_rows[ci] = [row_index, row_index + 1]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.EqualTo{T}}(),
    )
        row_index = length(templates) + 1
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        _push_equality!(
            templates,
            _single_variable_row(variable, var_to_pos, base_dimension),
            _exact_rational(set.value),
        )
        scalar_constraint_rows[ci] = [row_index]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{T}}(),
    )
        row_index = length(templates) + 1
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        _push_greater_than!(
            templates,
            _single_variable_row(variable, var_to_pos, base_dimension),
            _exact_rational(set.lower),
        )
        scalar_constraint_rows[ci] = [row_index]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{T}}(),
    )
        row_index = length(templates) + 1
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        _push_less_than!(
            templates,
            _single_variable_row(variable, var_to_pos, base_dimension),
            _exact_rational(set.upper),
        )
        scalar_constraint_rows[ci] = [row_index]
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Interval{T}}(),
    )
        row_index = length(templates) + 1
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row = _single_variable_row(variable, var_to_pos, base_dimension)
        _push_greater_than!(templates, copy(row), _exact_rational(set.lower))
        _push_less_than!(templates, row, _exact_rational(set.upper))
        scalar_constraint_rows[ci] = [row_index, row_index + 1]
    end

    A, b, positive_scalars = _assemble_system(templates, base_dimension)
    slack_count = length(positive_scalars)
    c_raw = vcat(c_original_raw, zeros(ExactRational, psd_auxiliary_count + slack_count))
    c_min = vcat(c_original_min, zeros(ExactRational, psd_auxiliary_count + slack_count))
    affine = _solve_affine_system(A, b)
    positive_scalars, pruned_scalar_faces = _prune_positive_scalar_faces(positive_scalars, affine)
    blocks, A, b, affine, pruned_directions = _prune_psd_faces(blocks, A, b, affine)
    if pruned_scalar_faces > 0 || pruned_directions > 0
        _log(
            opt,
            "pruned $(pruned_scalar_faces) scalar and $(pruned_directions) PSD direction(s) fixed to the cone boundary before barrier solve",
        )
    end

    return ProblemData(
        original_variables,
        blocks,
        positive_scalars,
        c_raw,
        constant_raw,
        c_min,
        A,
        b,
        affine,
        nothing,
        scalar_constraint_rows,
        psd_constraint_blocks,
    )
end
