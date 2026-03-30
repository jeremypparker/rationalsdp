const MOIU = MOI.Utilities
const ExactRational = Rational{BigInt}

MOIU.@model(
    StorageModel,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (MOI.PositiveSemidefiniteConeTriangle,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (),
    false,
)

Base.@kwdef mutable struct Settings
    max_iterations::Int = 80
    phase1_outer_iterations::Int = 14
    phase2_outer_iterations::Int = 12
    feasibility_tolerance::BigFloat = big"1e-22"
    optimality_gap_tolerance::BigFloat = big"1e-16"
    gradient_tolerance::BigFloat = big"1e-24"
    line_search_shrink::BigFloat = big"0.5"
    armijo_fraction::BigFloat = big"1e-4"
    min_step::BigFloat = big"1e-28"
    initial_scale::BigFloat = big"3.0"
    initial_penalty::BigFloat = big"1.0"
    penalty_growth::BigFloat = big"8.0"
    path_parameter_growth::BigFloat = big"8.0"
    phase1_center_weight::BigFloat = big"1e-2"
    boundary_fraction::BigFloat = big"0.99"
    working_precision::Int = 448
    rational_tolerance::BigFloat = big"1e-40"
    verbose::Bool = true
    inner_log_frequency::Int = 10
end

struct BlockStructure
    size::Int
    variables::Vector{MOI.VariableIndex}
    global_positions::Vector{Int}
    local_positions::Vector{Tuple{Int,Int}}
    diagonal_positions::Vector{Int}
end

struct EquationTemplate
    coefficients::Vector{ExactRational}
    rhs::ExactRational
    slack_sign::Int
end

struct ProblemData
    original_variables::Vector{MOI.VariableIndex}
    blocks::Vector{BlockStructure}
    positive_scalars::Vector{Int}
    objective_vector_raw::Vector{ExactRational}
    objective_constant_raw::ExactRational
    objective_vector_min::Vector{ExactRational}
    A::Matrix{ExactRational}
    b::Vector{ExactRational}
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}}
end

struct NumericBlock
    structure::BlockStructure
    bases::Vector{Matrix{BigFloat}}
end

mutable struct Optimizer{T<:Real} <: MOI.AbstractOptimizer
    settings::Settings
    storage::StorageModel{T}
    silent::Bool
    termination_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
    dual_status::MOI.ResultStatusCode
    raw_status::String
    solve_time_sec::Float64
    result_count::Int
    variable_primal::Dict{MOI.VariableIndex,T}
    objective_value::Union{Nothing,T}
end

function Optimizer{T}(; kwargs...) where {T<:Real}
    return Optimizer{T}(
        Settings(; kwargs...),
        StorageModel{T}(),
        false,
        MOI.OPTIMIZE_NOT_CALLED,
        MOI.NO_SOLUTION,
        MOI.NO_SOLUTION,
        "Optimizer not called",
        0.0,
        0,
        Dict{MOI.VariableIndex,T}(),
        nothing,
    )
end

Optimizer(; kwargs...) = Optimizer{Float64}(; kwargs...)

function _reset_results!(opt::Optimizer)
    opt.termination_status = MOI.OPTIMIZE_NOT_CALLED
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status = "Optimizer not called"
    opt.solve_time_sec = 0.0
    opt.result_count = 0
    empty!(opt.variable_primal)
    opt.objective_value = nothing
    return
end

function _format_metric(x)
    value = try
        Float64(x)
    catch
        NaN
    end
    if isfinite(value)
        return @sprintf("%.3e", value)
    end
    return string(x)
end

function _log(opt::Optimizer, message::AbstractString)
    if !opt.silent && opt.settings.verbose
        println("[RationalSDP] ", message)
    end
    return
end

MOI.supports_incremental_interface(::Optimizer) = true
MOI.copy_to(dest::Optimizer, src::MOI.ModelLike) = MOIU.default_copy_to(dest, src)
MOI.is_empty(opt::Optimizer) = MOI.is_empty(opt.storage)

function MOI.empty!(opt::Optimizer)
    MOI.empty!(opt.storage)
    _reset_results!(opt)
    return
end

MOI.supports_constraint(
    opt::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet} = MOI.supports_constraint(opt.storage, F, S)

MOI.is_valid(opt::Optimizer, vi::MOI.VariableIndex) = MOI.is_valid(opt.storage, vi)

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    return MOI.is_valid(opt.storage, ci)
end

MOI.add_variable(opt::Optimizer) = MOI.add_variable(opt.storage)

function MOI.add_constraint(
    opt::Optimizer,
    func::F,
    set::S,
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet}
    return MOI.add_constraint(opt.storage, func, set)
end

function MOI.supports(
    ::Optimizer,
    attr::MOI.AbstractOptimizerAttribute,
)
    return attr isa MOI.Silent
end

MOI.supports(opt::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.supports(opt.storage, attr)

function MOI.supports(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    ::Type{MOI.VariableIndex},
)
    return MOI.supports(opt.storage, attr, MOI.VariableIndex)
end

function MOI.supports(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F,S}
    return MOI.supports(opt.storage, attr, MOI.ConstraintIndex{F,S})
end

function MOI.set(
    opt::Optimizer,
    ::MOI.Silent,
    value::Bool,
)
    opt.silent = value
    return
end

MOI.get(opt::Optimizer, ::MOI.Silent) = opt.silent

function MOI.set(
    opt::Optimizer,
    attr::MOI.AbstractModelAttribute,
    value,
)
    return MOI.set(opt.storage, attr, value)
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
    value,
)
    return MOI.set(opt.storage, attr, vi, value)
end

function MOI.set(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex{F,S},
    value,
) where {F,S}
    return MOI.set(opt.storage, attr, ci, value)
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = opt.result_count
MOI.get(opt::Optimizer, ::MOI.TerminationStatus) = opt.termination_status
MOI.get(opt::Optimizer, ::MOI.PrimalStatus) = opt.primal_status
MOI.get(opt::Optimizer, ::MOI.DualStatus) = opt.dual_status
MOI.get(opt::Optimizer, ::MOI.RawStatusString) = opt.raw_status
MOI.get(opt::Optimizer, ::MOI.SolveTimeSec) = opt.solve_time_sec
MOI.get(::Optimizer, ::MOI.ListOfOptimizerAttributesSet) = MOI.AbstractOptimizerAttribute[]

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return something(opt.objective_value)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(opt, attr)
    return opt.variable_primal[vi]
end

MOI.get(opt::Optimizer, attr::MOI.AbstractModelAttribute) = MOI.get(opt.storage, attr)

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractVariableAttribute,
    vi::MOI.VariableIndex,
)
    return MOI.get(opt.storage, attr, vi)
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    return MOI.get(opt.storage, attr, ci)
end
_exact_rational(x::ExactRational) = x

function _exact_rational(x::Rational{S}) where {S<:Integer}
    return ExactRational(BigInt(numerator(x)), BigInt(denominator(x)))
end

_exact_rational(x::Integer) = ExactRational(BigInt(x), BigInt(1))

function _exact_rational(x::AbstractFloat)
    return rationalize(BigInt, BigFloat(x))
end

function _to_output_type(::Type{T}, x::ExactRational) where {T<:Real}
    return convert(T, x)
end

function _to_output_type(::Type{Rational{S}}, x::ExactRational) where {S<:Integer}
    S == BigInt && return x
    numerator(x) in typemin(S):typemax(S) || error(
        "Exact rational numerator does not fit in $(S); use Rational{BigInt} for guaranteed exact output.",
    )
    denominator(x) in typemin(S):typemax(S) || error(
        "Exact rational denominator does not fit in $(S); use Rational{BigInt} for guaranteed exact output.",
    )
    return Rational{S}(convert(S, numerator(x)), convert(S, denominator(x)))
end

function _triangle_positions(dim::Int)
    positions = Tuple{Int,Int}[]
    for i in 1:dim
        for j in 1:i
            push!(positions, (i, j))
        end
    end
    return positions
end

function _vector_to_matrix(
    x::AbstractVector{S},
    block::BlockStructure,
) where {S}
    X = zeros(S, block.size, block.size)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        value = x[block.global_positions[local_index]]
        X[i, j] = value
        X[j, i] = value
    end
    return X
end

function _strictly_pd(matrix::Matrix{BigFloat})
    try
        cholesky(Hermitian(matrix))
        return true
    catch
        return false
    end
end

function _logdet_spd(matrix::Matrix{BigFloat})
    factor = cholesky(Hermitian(matrix))
    value = zero(BigFloat)
    for d in diag(factor.L)
        value += log(d)
    end
    return 2 * value
end

function _trace(matrix::AbstractMatrix)
    value = zero(eltype(matrix))
    for i in axes(matrix, 1)
        value += matrix[i, i]
    end
    return value
end

function _rref(aug::Matrix{ExactRational})
    rows, cols = size(aug)
    pivot_columns = Int[]
    pivot_row = 1
    for col in 1:(cols - 1)
        pivot_index = 0
        for row in pivot_row:rows
            if !iszero(aug[row, col])
                pivot_index = row
                break
            end
        end
        pivot_index == 0 && continue
        if pivot_index != pivot_row
            aug[pivot_row, :], aug[pivot_index, :] = aug[pivot_index, :], aug[pivot_row, :]
        end
        pivot_value = aug[pivot_row, col]
        aug[pivot_row, :] ./= pivot_value
        for row in 1:rows
            row == pivot_row && continue
            factor = aug[row, col]
            iszero(factor) && continue
            aug[row, :] .-= factor .* aug[pivot_row, :]
        end
        push!(pivot_columns, col)
        pivot_row += 1
        pivot_row > rows && break
    end
    return aug, pivot_columns
end

function _solve_affine_system(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
)
    p = size(A, 2)
    if size(A, 1) == 0
        return zeros(ExactRational, p), Matrix{ExactRational}(I, p, p)
    end
    aug = hcat(copy(A), copy(b))
    reduced, pivots = _rref(aug)
    for row in 1:size(reduced, 1)
        if all(iszero, reduced[row, 1:p]) && !iszero(reduced[row, p + 1])
            return nothing
        end
    end
    particular = zeros(ExactRational, p)
    for (row, pivot_col) in enumerate(pivots)
        particular[pivot_col] = reduced[row, p + 1]
    end
    free_columns = setdiff(collect(1:p), pivots)
    nullspace = zeros(ExactRational, p, length(free_columns))
    for (basis_index, free_col) in enumerate(free_columns)
        nullspace[free_col, basis_index] = 1 // 1
        for (row, pivot_col) in enumerate(pivots)
            nullspace[pivot_col, basis_index] = -reduced[row, free_col]
        end
    end
    return particular, nullspace
end

function _leading_principal_determinants_positive(matrix::Matrix{ExactRational})
    n = size(matrix, 1)
    for k in 1:n
        if det(matrix[1:k, 1:k]) <= 0
            return false
        end
    end
    return true
end

function _strictly_positive_exact(x::Vector{ExactRational}, positive_scalars::Vector{Int})
    for index in positive_scalars
        if !(x[index] > 0)
            return false
        end
    end
    return true
end

function _strictly_interior_exact(
    x::Vector{ExactRational},
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
)
    _strictly_positive_exact(x, positive_scalars) || return false
    for block in blocks
        if !_leading_principal_determinants_positive(_vector_to_matrix(x, block))
            return false
        end
    end
    return true
end

function _numeric_blocks(blocks::Vector{BlockStructure})
    numeric_blocks = Vector{NumericBlock}(undef, length(blocks))
    for (index, block) in enumerate(blocks)
        bases = Vector{Matrix{BigFloat}}(undef, length(block.local_positions))
        for (basis_index, (i, j)) in enumerate(block.local_positions)
            basis = zeros(BigFloat, block.size, block.size)
            basis[i, j] = 1
            basis[j, i] = 1
            if i == j
                basis[i, j] = 1
            end
            bases[basis_index] = basis
        end
        numeric_blocks[index] = NumericBlock(block, bases)
    end
    return numeric_blocks
end

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
    )
end
function _barrier_value_grad_hess(
    x::Vector{BigFloat},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
)
    p = length(x)
    value = zero(BigFloat)
    grad = zeros(BigFloat, p)
    hess = zeros(BigFloat, p, p)

    for numeric_block in numeric_blocks
        block = numeric_block.structure
        X = _vector_to_matrix(x, block)
        value -= _logdet_spd(X)
        invX = inv(X)
        transformed = [invX * basis for basis in numeric_block.bases]
        global_positions = block.global_positions
        for a in eachindex(transformed)
            ga = global_positions[a]
            grad[ga] -= _trace(transformed[a])
            for b in a:length(transformed)
                gb = global_positions[b]
                contribution = _trace(transformed[a] * transformed[b])
                hess[ga, gb] += contribution
                if ga != gb
                    hess[gb, ga] += contribution
                end
            end
        end
    end

    for index in positive_scalars
        slack = x[index]
        inverse_slack = inv(slack)
        value -= log(slack)
        grad[index] -= inverse_slack
        hess[index, index] += inverse_slack^2
    end

    return value, grad, hess
end

function _strictly_interior_numeric(
    x::Vector{BigFloat},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
)
    for index in positive_scalars
        if !(x[index] > 0)
            return false
        end
    end
    for block in numeric_blocks
        if !_strictly_pd(_vector_to_matrix(x, block.structure))
            return false
        end
    end
    return true
end

function _build_phase1_initial_point(
    problem::ProblemData,
    settings::Settings,
    numeric_blocks::Vector{NumericBlock},
)
    total_dimension = length(problem.objective_vector_raw)
    x = if problem.affine === nothing
        zeros(BigFloat, total_dimension)
    else
        BigFloat.(problem.affine[1])
    end

    scale = settings.initial_scale
    for _ in 1:10
        for index in problem.positive_scalars
            x[index] = max(x[index], scale)
        end
        for block in problem.blocks
            if !_strictly_pd(_vector_to_matrix(x, block))
                for diagonal_position in block.diagonal_positions
                    x[diagonal_position] += scale
                end
            end
        end
        _strictly_interior_numeric(x, numeric_blocks, problem.positive_scalars) && return x
        scale *= 2
    end
    return x
end

function _max_abs(vector::AbstractVector)
    isempty(vector) && return zero(BigFloat)
    result = zero(BigFloat)
    for value in vector
        result = max(result, abs(value))
    end
    return result
end

function _recovery_tolerances(settings::Settings)
    tolerances = BigFloat[]
    tolerance = max(big"1e-8", sqrt(settings.rational_tolerance))
    while tolerance > settings.rational_tolerance
        push!(tolerances, tolerance)
        tolerance /= big"1e4"
    end
    push!(tolerances, settings.rational_tolerance)
    return tolerances
end

function _max_step_to_boundary(
    x::Vector{BigFloat},
    direction::Vector{BigFloat},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
    fraction::BigFloat,
)
    max_step = one(BigFloat)
    for index in positive_scalars
        if direction[index] < 0
            max_step = min(max_step, fraction * x[index] / (-direction[index]))
        end
    end
    for numeric_block in numeric_blocks
        X = _vector_to_matrix(x, numeric_block.structure)
        D = _vector_to_matrix(direction, numeric_block.structure)
        factor = cholesky(Hermitian(X))
        scaled_direction = factor.L \ D / transpose(factor.L)
        frobenius_bound = sqrt(sum(abs2, scaled_direction))
        if frobenius_bound > 0
            max_step = min(max_step, fraction / frobenius_bound)
        end
    end
    return max(zero(BigFloat), max_step)
end

function _solve_spd_system(matrix::Matrix{BigFloat}, rhs::Vector{BigFloat})
    regularization = big"0"
    identity_matrix = Matrix{BigFloat}(I, size(matrix, 1), size(matrix, 2))
    for _ in 1:10
        trial = matrix + regularization * identity_matrix
        try
            factor = cholesky(Hermitian(trial))
            return factor \ rhs
        catch
            regularization = iszero(regularization) ? big"1e-30" : 10 * regularization
        end
    end
    error("Failed to factor Newton system.")
end

function _newton_phase1!(
    opt::Optimizer,
    x::Vector{BigFloat},
    problem::ProblemData,
    numeric_blocks::Vector{NumericBlock},
    A_big::Matrix{BigFloat},
    b_big::Vector{BigFloat},
    penalty::BigFloat,
    center::Vector{BigFloat},
)
    settings = opt.settings
    At = transpose(A_big)
    AtA = At * A_big
    center_weight = settings.phase1_center_weight
    for iteration in 1:settings.max_iterations
        barrier_value, barrier_grad, barrier_hess = _barrier_value_grad_hess(
            x,
            numeric_blocks,
            problem.positive_scalars,
        )
        residual = A_big * x - b_big
        deviation = x - center
        residual_norm = _max_abs(residual)
        value = barrier_value + penalty * dot(residual, residual) / 2 + center_weight * dot(deviation, deviation) / 2
        grad = barrier_grad + penalty * (At * residual) + center_weight * deviation
        grad_norm = _max_abs(grad)
        if iteration == 1 || iteration % settings.inner_log_frequency == 0
            _log(
                opt,
                "phase I newton $(iteration): residual=$(_format_metric(residual_norm)), grad=$(_format_metric(grad_norm))",
            )
        end
        if grad_norm <= settings.gradient_tolerance
            return x
        end
        hess = barrier_hess + penalty * AtA
        for diagonal in 1:length(x)
            hess[diagonal, diagonal] += center_weight
        end
        direction = -_solve_spd_system(hess, grad)
        directional_derivative = dot(grad, direction)
        step = min(
            one(BigFloat),
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
                trial_barrier, _, _ = _barrier_value_grad_hess(
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
    x_approx::Vector{BigFloat},
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational};
    tolerance::BigFloat,
)
    if size(nullspace, 2) == 0
        return particular
    end
    delta = x_approx .- BigFloat.(particular)
    N_big = BigFloat.(nullspace)
    coefficients = N_big \ delta
    rational_coefficients = [
        rationalize(BigInt, value; tol = tolerance) for value in coefficients
    ]
    return particular + nullspace * rational_coefficients
end

function _blend_to_interior(
    anchor::Vector{ExactRational},
    candidate::Vector{ExactRational},
    problem::ProblemData,
)
    if _strictly_interior_exact(candidate, problem.blocks, problem.positive_scalars)
        return candidate
    end
    delta = candidate - anchor
    weight = 1 // 1
    for _ in 1:128
        trial = anchor + weight * delta
        if _strictly_interior_exact(trial, problem.blocks, problem.positive_scalars)
            return trial
        end
        weight //= 2
    end
    return anchor
end

function _phase1_exact_feasible_point(
    x_phase1::Vector{BigFloat},
    problem::ProblemData,
    settings::Settings,
)
    problem.affine === nothing && return nothing
    particular, nullspace = problem.affine
    for tolerance in _recovery_tolerances(settings)
        candidate = _project_exact_solution(
            x_phase1,
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

function _newton_phase2!(
    opt::Optimizer,
    z::Vector{BigFloat},
    x0::Vector{BigFloat},
    N_big::Matrix{BigFloat},
    c_big::Vector{BigFloat},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
    barrier_parameter::BigFloat,
)
    settings = opt.settings
    Nt_big = transpose(N_big)
    for iteration in 1:settings.max_iterations
        x = x0 + N_big * z
        barrier_value, barrier_grad_x, barrier_hess_x = _barrier_value_grad_hess(
            x,
            numeric_blocks,
            positive_scalars,
        )
        value = barrier_parameter * dot(c_big, x) + barrier_value
        grad_x = barrier_parameter * c_big + barrier_grad_x
        grad_z = Nt_big * grad_x
        grad_norm = _max_abs(grad_z)
        if iteration == 1 || iteration % settings.inner_log_frequency == 0
            _log(
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
            one(BigFloat),
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

function MOI.optimize!(opt::Optimizer{T}) where {T}
    start_time = time_ns()
    _reset_results!(opt)
    setprecision(opt.settings.working_precision) do
        problem = _extract_problem(opt)
        _log(
            opt,
            "starting solve with $(length(problem.original_variables)) original vars, $(length(problem.blocks)) PSD blocks, $(length(problem.positive_scalars)) scalar slacks, and $(size(problem.A, 1)) affine equations",
        )
        if problem.affine === nothing
            opt.termination_status = MOI.INFEASIBLE
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Affine equalities are inconsistent."
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
            opt.raw_status = "Solved by exact affine elimination."
            opt.result_count = 1
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "finished by exact elimination without barrier work")
            return
        elseif barrier_dim == 0
            objective_direction = _objective_nullspace_direction(problem.objective_vector_min, nullspace)
            if any(!iszero, objective_direction)
                opt.termination_status = MOI.DUAL_INFEASIBLE
                opt.primal_status = MOI.NO_SOLUTION
                opt.raw_status = "Objective is unbounded along the affine nullspace."
                opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
                _log(opt, "detected unbounded affine objective")
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
            opt.raw_status = "Objective is constant over the affine feasible region."
            opt.result_count = 1
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "finished on a barrier-free affine problem")
            return
        end

        numeric_blocks = _numeric_blocks(problem.blocks)
        A_big = BigFloat.(problem.A)
        b_big = BigFloat.(problem.b)
        x = _build_phase1_initial_point(problem, opt.settings, numeric_blocks)
        phase1_center = copy(x)
        penalty = opt.settings.initial_penalty

        _log(opt, "phase I: searching for a strictly feasible interior point")
        for outer_iteration in 1:opt.settings.phase1_outer_iterations
            x = _newton_phase1!(
                opt,
                x,
                problem,
                numeric_blocks,
                A_big,
                b_big,
                penalty,
                phase1_center,
            )
            residual = _max_abs(A_big * x - b_big)
            barrier_value, _, _ = _barrier_value_grad_hess(x, numeric_blocks, problem.positive_scalars)
            _log(
                opt,
                "phase I outer $(outer_iteration)/$(opt.settings.phase1_outer_iterations): penalty=$(_format_metric(penalty)), residual=$(_format_metric(residual)), barrier=$(_format_metric(barrier_value))",
            )
            residual <= opt.settings.feasibility_tolerance && break
            penalty *= opt.settings.penalty_growth
        end

        anchor = _phase1_exact_feasible_point(x, problem, opt.settings)
        if anchor === nothing
            opt.termination_status = MOI.NUMERICAL_ERROR
            opt.primal_status = MOI.NO_SOLUTION
            opt.raw_status = "Failed to recover an exact strictly feasible rational point."
            opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
            _log(opt, "exact rational interior recovery failed after phase I")
            return
        end

        x_exact = anchor
        if size(nullspace, 2) > 0 && any(!iszero, problem.objective_vector_min)
            x0 = BigFloat.(x_exact)
            N_big = BigFloat.(nullspace)
            c_big = BigFloat.(problem.objective_vector_min)
            z = zeros(BigFloat, size(nullspace, 2))
            barrier_parameter = one(BigFloat)

            _log(opt, "phase II: following the barrier path toward the objective minimum")
            for outer_iteration in 1:opt.settings.phase2_outer_iterations
                z = _newton_phase2!(
                    opt,
                    z,
                    x0,
                    N_big,
                    c_big,
                    numeric_blocks,
                    problem.positive_scalars,
                    barrier_parameter,
                )
                x_trial = x0 + N_big * z
                approximate_objective = dot(c_big, x_trial) + BigFloat(problem.objective_constant_raw)
                gap_bound = BigFloat(barrier_dim) / barrier_parameter
                _log(
                    opt,
                    "phase II outer $(outer_iteration)/$(opt.settings.phase2_outer_iterations): mu=$(_format_metric(barrier_parameter)), objective≈$(_format_metric(approximate_objective)), gap<=$(_format_metric(gap_bound))",
                )
                if gap_bound <= opt.settings.optimality_gap_tolerance
                    break
                end
                barrier_parameter *= opt.settings.path_parameter_growth
            end

            x_candidate = _project_exact_solution(
                x0 + N_big * z,
                particular,
                nullspace;
                tolerance = opt.settings.rational_tolerance,
            )
            x_exact = _blend_to_interior(anchor, x_candidate, problem)
        end

        objective_value = _exact_objective_value(problem, x_exact)
        for (index, variable) in enumerate(problem.original_variables)
            opt.variable_primal[variable] = _to_output_type(T, x_exact[index])
        end
        opt.objective_value = _to_output_type(T, objective_value)
        opt.termination_status = MOI.OPTIMAL
        opt.primal_status = MOI.FEASIBLE_POINT
        opt.dual_status = MOI.NO_SOLUTION
        opt.raw_status = "Solved with a mixed cone barrier method and exact rational recovery."
        opt.result_count = 1
        opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
        _log(opt, "finished in " * @sprintf("%.3f", opt.solve_time_sec) * "s with objective $(objective_value)")
    end
    return
end
