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

    psd_indices = MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )

    blocks = BlockStructure[]
    seen_psd_variables = Dict{MOI.VariableIndex,Int}()
    for ci in psd_indices
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
                collect(func.variables),
                global_positions,
                positions,
                diagonal_positions,
            ),
        )
    end

    c_original_raw, constant_raw, c_original_min = _objective_data(storage, original_variables, var_to_pos)
    templates = EquationTemplate[]

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.EqualTo{T},
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, original_dimension)
        _push_equality!(templates, row, _exact_rational(set.value) - offset)
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.GreaterThan{T},
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, original_dimension)
        _push_greater_than!(templates, row, _exact_rational(set.lower) - offset)
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.LessThan{T},
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, original_dimension)
        _push_less_than!(templates, row, _exact_rational(set.upper) - offset)
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.ScalarAffineFunction{T},
            MOI.Interval{T},
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row, offset = _extract_affine_row(func, var_to_pos, original_dimension)
        _push_greater_than!(templates, copy(row), _exact_rational(set.lower) - offset)
        _push_less_than!(templates, row, _exact_rational(set.upper) - offset)
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.EqualTo{T}}(),
    )
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        _push_equality!(
            templates,
            _single_variable_row(variable, var_to_pos, original_dimension),
            _exact_rational(set.value),
        )
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{T}}(),
    )
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        _push_greater_than!(
            templates,
            _single_variable_row(variable, var_to_pos, original_dimension),
            _exact_rational(set.lower),
        )
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{T}}(),
    )
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        _push_less_than!(
            templates,
            _single_variable_row(variable, var_to_pos, original_dimension),
            _exact_rational(set.upper),
        )
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Interval{T}}(),
    )
        variable = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        row = _single_variable_row(variable, var_to_pos, original_dimension)
        _push_greater_than!(templates, copy(row), _exact_rational(set.lower))
        _push_less_than!(templates, row, _exact_rational(set.upper))
    end

    A, b, positive_scalars = _assemble_system(templates, original_dimension)
    slack_count = length(positive_scalars)
    c_raw = vcat(c_original_raw, zeros(ExactRational, slack_count))
    c_min = vcat(c_original_min, zeros(ExactRational, slack_count))
    affine = _solve_affine_system(A, b)
    blocks, A, b, affine, pruned_directions = _prune_psd_faces(blocks, A, b, affine)
    if pruned_directions > 0
        _log(
            opt,
            "pruned $(pruned_directions) PSD direction(s) fixed to the cone boundary before barrier solve",
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
    )
end
