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
    phase1_outer_iterations::Int = 100
    phase2_outer_iterations::Int = 12
    working_float_type::DataType = Double64
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
    verbose_newton::Bool = false
    live_progress::Bool = true
    inner_log_frequency::Int = 10
    threaded::Bool = true
    threading_min_block_size::Int = 48
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
end

struct NumericAffineData{F,S}
    particular::Vector{F}
    nullspace::Matrix{F}
    nullspace_factor::S
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

function _log_raw(opt::Optimizer, message::AbstractString = "")
    if !opt.silent && opt.settings.verbose
        println(message)
    end
    return
end

function _log_newton(opt::Optimizer, message::AbstractString)
    if !opt.silent && opt.settings.verbose && opt.settings.verbose_newton
        println("[RationalSDP] ", message)
    end
    return
end

function _completed_rows(rows::Vector{Vector{String}})
    return [row for row in rows if any(!isempty, row[2:end])]
end

function _column_widths(columns::Vector{String}, rows::Vector{Vector{String}})
    widths = [textwidth(column) for column in columns]
    for row in rows
        for index in eachindex(columns)
            widths[index] = max(widths[index], textwidth(row[index]))
        end
    end
    return widths
end

function _format_table_row(
    row::Vector{String},
    widths::Vector{Int},
    alignments::Vector{Symbol},
)
    cells = String[]
    for index in eachindex(row)
        padding = max(0, widths[index] - textwidth(row[index]))
        if alignments[index] == :right
            push!(cells, repeat(" ", padding) * row[index])
        else
            push!(cells, row[index] * repeat(" ", padding))
        end
    end
    return "  " * join(cells, "  ")
end

function _table_separator(widths::Vector{Int})
    return "  " * join((repeat("-", width) for width in widths), "  ")
end

function _log_table(
    opt::Optimizer,
    title::AbstractString,
    columns::Vector{String},
    rows::Vector{Vector{String}};
    subtitle::AbstractString = "",
    alignments::Vector{Symbol} = vcat([:left], fill(:right, length(columns) - 1)),
)
    visible_rows = isempty(rows) ? rows : _completed_rows(rows)
    widths = _column_widths(columns, visible_rows)
    _log_raw(opt)
    _log_raw(opt, title)
    if !isempty(subtitle)
        _log_raw(opt, subtitle)
    end
    _log_raw(opt, _format_table_row(columns, widths, fill(:left, length(columns))))
    _log_raw(opt, _table_separator(widths))
    for row in visible_rows
        _log_raw(opt, _format_table_row(row, widths, alignments))
    end
    return
end

function _phase_table_widths(columns::Vector{String}, total_iterations::Int)
    sample_rows = [[
        string(total_iterations),
        "1.000e+00",
        "1.000e+00",
        "-1.000e+00",
        "9999.99",
    ]]
    return _column_widths(columns, sample_rows)
end

function _log_table_header(
    opt::Optimizer,
    title::AbstractString,
    columns::Vector{String},
    widths::Vector{Int};
    subtitle::AbstractString = "",
)
    _log_raw(opt)
    _log_raw(opt, title)
    if !isempty(subtitle)
        _log_raw(opt, subtitle)
    end
    _log_raw(opt, _format_table_row(columns, widths, fill(:left, length(columns))))
    _log_raw(opt, _table_separator(widths))
    return
end

function _log_table_row(
    opt::Optimizer,
    row::Vector{String},
    widths::Vector{Int},
    alignments::Vector{Symbol},
)
    _log_raw(opt, _format_table_row(row, widths, alignments))
    return
end

function _log_banner(opt::Optimizer, problem::ProblemData)
    thread_count = opt.settings.threaded ? nthreads() : 1
    _log_table(
        opt,
        "RationalSDP",
        ["Item", "Value"],
        [
            ["Method", "Primal-only interior-point method with exact rational recovery"],
            ["Variables", string(length(problem.original_variables))],
            ["PSD blocks", string(length(problem.blocks))],
            ["Scalar slacks", string(length(problem.positive_scalars))],
            ["Affine equations", string(size(problem.A, 1))],
            ["Threads", string(thread_count)],
        ];
        subtitle = "Solve summary",
        alignments = [:left, :left],
    )
    return
end

const _SETTINGS_DEFAULTS = Settings()
const _SETTING_FIELDNAMES = fieldnames(Settings)
const _SETTING_NAME_SET = Set(String(name) for name in _SETTING_FIELDNAMES)

function _setting_symbol(name::AbstractString)
    symbol = Symbol(name)
    symbol in _SETTING_FIELDNAMES || throw(MOI.UnsupportedAttribute(MOI.RawOptimizerAttribute(name)))
    return symbol
end

function _convert_setting_value(::Type{BigFloat}, value)
    if value isa AbstractString
        return parse(BigFloat, value)
    end
    return BigFloat(value)
end

function _convert_setting_value(::Type{DataType}, value)
    parsed = if value isa AbstractString
        symbol = Symbol(value)
        if isdefined(@__MODULE__, symbol)
            getfield(@__MODULE__, symbol)
        elseif isdefined(Base, symbol)
            getfield(Base, symbol)
        else
            error("Unknown working float type: $(value)")
        end
    else
        value
    end
    parsed isa Type || error("working_float_type must be a floating-point type.")
    parsed <: AbstractFloat || error("working_float_type must be a subtype of AbstractFloat.")
    return parsed
end

function _convert_setting_value(::Type{Bool}, value)
    if value isa AbstractString
        lowercase_value = lowercase(value)
        lowercase_value == "true" && return true
        lowercase_value == "false" && return false
    end
    return convert(Bool, value)
end

function _convert_setting_value(::Type{Int}, value)
    if value isa AbstractString
        return parse(Int, value)
    end
    return convert(Int, value)
end

_convert_setting_value(::Type{T}, value) where {T} = convert(T, value)

function _working_float_type(settings::Settings)
    F = settings.working_float_type
    F <: AbstractFloat || error("working_float_type must be a subtype of AbstractFloat.")
    return F
end

_to_working_float(::Type{F}, x::ExactRational) where {F<:AbstractFloat} = F(numerator(x)) / F(denominator(x))
_to_working_float(::Type{F}, x::Rational{S}) where {F<:AbstractFloat,S<:Integer} = F(numerator(x)) / F(denominator(x))
_to_working_float(::Type{F}, x::Integer) where {F<:AbstractFloat} = F(x)
_to_working_float(::Type{F}, x::AbstractFloat) where {F<:AbstractFloat} = F(x)

function _to_working_array(::Type{F}, values::AbstractVector) where {F<:AbstractFloat}
    converted = Vector{F}(undef, length(values))
    for index in eachindex(values)
        converted[index] = _to_working_float(F, values[index])
    end
    return converted
end

function _to_working_array(::Type{F}, values::AbstractMatrix) where {F<:AbstractFloat}
    converted = Matrix{F}(undef, size(values)...)
    for index in eachindex(values)
        converted[index] = _to_working_float(F, values[index])
    end
    return converted
end

function _with_working_precision(settings::Settings, f::Function)
    F = _working_float_type(settings)
    if F === BigFloat
        return setprecision(BigFloat, settings.working_precision) do
            f(F)
        end
    end
    return f(F)
end

function _numeric_settings(settings::Settings, ::Type{F}) where {F<:AbstractFloat}
    return (
        max_iterations = settings.max_iterations,
        phase1_outer_iterations = settings.phase1_outer_iterations,
        phase2_outer_iterations = settings.phase2_outer_iterations,
        working_float_type = F,
        feasibility_tolerance = _to_working_float(F, settings.feasibility_tolerance),
        optimality_gap_tolerance = _to_working_float(F, settings.optimality_gap_tolerance),
        gradient_tolerance = _to_working_float(F, settings.gradient_tolerance),
        line_search_shrink = _to_working_float(F, settings.line_search_shrink),
        armijo_fraction = _to_working_float(F, settings.armijo_fraction),
        min_step = _to_working_float(F, settings.min_step),
        initial_scale = _to_working_float(F, settings.initial_scale),
        initial_penalty = _to_working_float(F, settings.initial_penalty),
        penalty_growth = _to_working_float(F, settings.penalty_growth),
        path_parameter_growth = _to_working_float(F, settings.path_parameter_growth),
        phase1_center_weight = _to_working_float(F, settings.phase1_center_weight),
        boundary_fraction = _to_working_float(F, settings.boundary_fraction),
        working_precision = settings.working_precision,
        rational_tolerance = _to_working_float(F, settings.rational_tolerance),
        verbose = settings.verbose,
        verbose_newton = settings.verbose_newton,
        live_progress = settings.live_progress,
        inner_log_frequency = settings.inner_log_frequency,
        threaded = settings.threaded,
        threading_min_block_size = settings.threading_min_block_size,
    )
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
    return (
        attr isa MOI.Silent ||
        attr isa MOI.SolverName ||
        (attr isa MOI.RawOptimizerAttribute && attr.name in _SETTING_NAME_SET)
    )
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
    attr::MOI.RawOptimizerAttribute,
    value,
)
    symbol = _setting_symbol(attr.name)
    field_type = fieldtype(Settings, symbol)
    setfield!(opt.settings, symbol, _convert_setting_value(field_type, value))
    return
end

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
MOI.get(::Optimizer, ::MOI.SolverName) = "RationalSDP"

function MOI.get(opt::Optimizer, ::MOI.ListOfOptimizerAttributesSet)
    attrs = MOI.AbstractOptimizerAttribute[]
    opt.silent && push!(attrs, MOI.Silent())
    for name in _SETTING_FIELDNAMES
        current = getfield(opt.settings, name)
        default = getfield(_SETTINGS_DEFAULTS, name)
        current == default || push!(attrs, MOI.RawOptimizerAttribute(String(name)))
    end
    return attrs
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.RawOptimizerAttribute,
)
    symbol = _setting_symbol(attr.name)
    return getfield(opt.settings, symbol)
end

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

function _strictly_pd(matrix::AbstractMatrix{F}) where {F<:AbstractFloat}
    try
        cholesky(Hermitian(matrix))
        return true
    catch
        return false
    end
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

function _variable_fixed_zero(
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    index::Int,
)
    iszero(particular[index]) || return false
    for column in axes(nullspace, 2)
        iszero(nullspace[index, column]) || return false
    end
    return true
end

function _block_direction_entry_indices(
    block::BlockStructure,
    local_direction::Int,
)
    indices = Int[]
    for (local_index, (i, j)) in enumerate(block.local_positions)
        if i == local_direction || j == local_direction
            push!(indices, block.global_positions[local_index])
        end
    end
    return indices
end

function _restrict_block(
    block::BlockStructure,
    keep_directions::Vector{Int},
)
    keep_lookup = Dict(direction => new_index for (new_index, direction) in enumerate(keep_directions))
    variables = MOI.VariableIndex[]
    global_positions = Int[]
    local_positions = Tuple{Int,Int}[]
    diagonal_positions = Int[]
    for (local_index, (i, j)) in enumerate(block.local_positions)
        haskey(keep_lookup, i) || continue
        haskey(keep_lookup, j) || continue
        new_position = (keep_lookup[i], keep_lookup[j])
        push!(variables, block.variables[local_index])
        push!(global_positions, block.global_positions[local_index])
        push!(local_positions, new_position)
        if new_position[1] == new_position[2]
            push!(diagonal_positions, block.global_positions[local_index])
        end
    end
    return BlockStructure(
        length(keep_directions),
        variables,
        global_positions,
        local_positions,
        diagonal_positions,
    )
end

function _append_zero_equalities(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    indices::Vector{Int},
)
    isempty(indices) && return A, b
    unique_indices = unique(sort(indices))
    rows, cols = size(A)
    A_augmented = zeros(ExactRational, rows + length(unique_indices), cols)
    b_augmented = zeros(ExactRational, rows + length(unique_indices))
    if rows > 0
        A_augmented[1:rows, :] = A
        b_augmented[1:rows] = b
    end
    for (offset, index) in enumerate(unique_indices)
        row = rows + offset
        A_augmented[row, index] = 1 // 1
    end
    return A_augmented, b_augmented
end

function _prune_psd_faces(
    blocks::Vector{BlockStructure},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
)
    affine === nothing && return blocks, A, b, affine, 0
    total_pruned = 0
    while true
        particular, nullspace = affine
        zero_indices = Int[]
        new_blocks = BlockStructure[]
        changed = false
        for block in blocks
            remove_directions = Int[]
            for local_direction in 1:block.size
                if _variable_fixed_zero(particular, nullspace, block.diagonal_positions[local_direction])
                    push!(remove_directions, local_direction)
                end
            end
            if isempty(remove_directions)
                push!(new_blocks, block)
                continue
            end
            changed = true
            total_pruned += length(remove_directions)
            for local_direction in remove_directions
                append!(zero_indices, _block_direction_entry_indices(block, local_direction))
            end
            keep_directions = setdiff(collect(1:block.size), remove_directions)
            isempty(keep_directions) || push!(new_blocks, _restrict_block(block, keep_directions))
        end
        changed || return blocks, A, b, affine, total_pruned
        A, b = _append_zero_equalities(A, b, zero_indices)
        blocks = new_blocks
        affine = _solve_affine_system(A, b)
        affine === nothing && return blocks, A, b, affine, total_pruned
    end
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
    return [NumericBlock(block) for block in blocks]
end

function _numeric_affine_data(problem::ProblemData, ::Type{F}) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    particular, nullspace = problem.affine
    particular_numeric = _to_working_array(F, particular)
    nullspace_numeric = _to_working_array(F, nullspace)
    nullspace_factor = size(nullspace, 2) == 0 ? nothing : qr(nullspace_numeric)
    return NumericAffineData(particular_numeric, nullspace_numeric, nullspace_factor)
end

@inline function _barrier_gradient_entry(
    inverse_matrix::AbstractMatrix{F},
    i::Int,
    j::Int,
) where {F<:AbstractFloat}
    return i == j ? -inverse_matrix[i, i] : -2 * inverse_matrix[i, j]
end

@inline function _barrier_hessian_entry(
    inverse_matrix::AbstractMatrix{F},
    i::Int,
    j::Int,
    k::Int,
    l::Int,
) where {F<:AbstractFloat}
    if i == j
        return k == l ? inverse_matrix[i, k]^2 : 2 * inverse_matrix[i, k] * inverse_matrix[i, l]
    elseif k == l
        return 2 * inverse_matrix[k, i] * inverse_matrix[k, j]
    end
    return 2 * (
        inverse_matrix[i, k] * inverse_matrix[j, l] +
        inverse_matrix[i, l] * inverse_matrix[j, k]
    )
end

function _block_barrier_value_grad_hess!(
    grad::Vector{F},
    hess::Matrix{F},
    x::Vector{F},
    numeric_block::NumericBlock,
    settings,
    allow_threads::Bool,
) where {F<:AbstractFloat}
    block = numeric_block.structure
    X = _vector_to_matrix(x, block)
    factor = cholesky(Hermitian(X))
    inverse_matrix = Matrix{F}(I, block.size, block.size)
    ldiv!(factor, inverse_matrix)

    logdet = zero(F)
    for diagonal_entry in diag(factor.L)
        logdet += log(diagonal_entry)
    end

    local_positions = block.local_positions
    global_positions = block.global_positions
    local_dimension = length(local_positions)
    use_threads = allow_threads && settings.threaded && nthreads() > 1 && local_dimension >= settings.threading_min_block_size

    if use_threads
        @threads for a in 1:local_dimension
            i, j = local_positions[a]
            ga = global_positions[a]
            grad[ga] = _barrier_gradient_entry(inverse_matrix, i, j)
            for b in a:local_dimension
                k, l = local_positions[b]
                gb = global_positions[b]
                hess[ga, gb] = _barrier_hessian_entry(inverse_matrix, i, j, k, l)
            end
        end
    else
        for a in 1:local_dimension
            i, j = local_positions[a]
            ga = global_positions[a]
            grad[ga] = _barrier_gradient_entry(inverse_matrix, i, j)
            for b in a:local_dimension
                k, l = local_positions[b]
                gb = global_positions[b]
                hess[ga, gb] = _barrier_hessian_entry(inverse_matrix, i, j, k, l)
            end
        end
    end

    for a in 1:local_dimension
        ga = global_positions[a]
        for b in (a + 1):local_dimension
            gb = global_positions[b]
            hess[gb, ga] = hess[ga, gb]
        end
    end

    return -2 * logdet
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
    )
end
function _barrier_value_grad_hess(
    x::Vector{F},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
    settings,
) where {F<:AbstractFloat}
    p = length(x)
    value = zero(F)
    grad = zeros(F, p)
    hess = zeros(F, p, p)

    if settings.threaded && nthreads() > 1 && length(numeric_blocks) > 1
        values = zeros(F, Base.Threads.maxthreadid())
        @threads for block_index in eachindex(numeric_blocks)
            tid = threadid()
            values[tid] += _block_barrier_value_grad_hess!(
                grad,
                hess,
                x,
                numeric_blocks[block_index],
                settings,
                false,
            )
        end
        value += sum(values)
    else
        for numeric_block in numeric_blocks
            value += _block_barrier_value_grad_hess!(
                grad,
                hess,
                x,
                numeric_block,
                settings,
                true,
            )
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
    x::Vector{F},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
) where {F<:AbstractFloat}
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
    settings,
    numeric_blocks::Vector{NumericBlock},
    numeric_affine::Union{Nothing,NumericAffineData},
)
    F = settings.working_float_type
    total_dimension = length(problem.objective_vector_raw)
    x = if numeric_affine === nothing
        zeros(F, total_dimension)
    else
        copy(numeric_affine.particular)
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
    isempty(vector) && return zero(eltype(vector))
    result = zero(eltype(vector))
    for value in vector
        result = max(result, abs(value))
    end
    return result
end

function _recovery_tolerances(settings::Settings)
    F = _working_float_type(settings)
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

function _max_step_to_boundary(
    x::Vector{F},
    direction::Vector{F},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
    fraction::F,
) where {F<:AbstractFloat}
    max_step = one(F)
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
    return max(zero(F), max_step)
end

function _solve_spd_system(matrix::Matrix{F}, rhs::Vector{F}) where {F<:AbstractFloat}
    regularization = zero(F)
    identity_matrix = Matrix{F}(I, size(matrix, 1), size(matrix, 2))
    seed = _to_working_float(F, big"1e-30")
    for _ in 1:10
        trial = matrix + regularization * identity_matrix
        try
            factor = cholesky(Hermitian(trial))
            return factor \ rhs
        catch
            regularization = iszero(regularization) ? seed : F(10) * regularization
        end
    end
    error("Failed to factor Newton system.")
end

function _newton_phase1!(
    opt::Optimizer,
    x::Vector{F},
    problem::ProblemData,
    numeric_blocks::Vector{NumericBlock},
    A_big::Matrix{F},
    b_big::Vector{F},
    penalty::F,
    center::Vector{F},
    numeric_settings,
) where {F<:AbstractFloat}
    settings = numeric_settings
    At = transpose(A_big)
    AtA = At * A_big
    center_weight = settings.phase1_center_weight
    for iteration in 1:settings.max_iterations
        barrier_value, barrier_grad, barrier_hess = _barrier_value_grad_hess(
            x,
            numeric_blocks,
            problem.positive_scalars,
            settings,
        )
        residual = A_big * x - b_big
        deviation = x - center
        residual_norm = _max_abs(residual)
        value = barrier_value + penalty * dot(residual, residual) / 2 + center_weight * dot(deviation, deviation) / 2
        grad = barrier_grad + penalty * (At * residual) + center_weight * deviation
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
        hess = barrier_hess + penalty * AtA
        for diagonal in 1:length(x)
            hess[diagonal, diagonal] += center_weight
        end
        direction = -_solve_spd_system(hess, grad)
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
                trial_barrier, _, _ = _barrier_value_grad_hess(
                    trial,
                    numeric_blocks,
                    problem.positive_scalars,
                    settings,
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
    if size(numeric_affine.nullspace, 2) == 0
        return F[]
    end
    return numeric_affine.nullspace_factor \ (x_approx .- numeric_affine.particular)
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
    x_phase1::Vector{F},
    problem::ProblemData,
    settings::Settings,
    numeric_affine::NumericAffineData{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    particular, nullspace = problem.affine
    coefficients = _affine_coordinates(x_phase1, numeric_affine)
    for tolerance in _recovery_tolerances(settings)
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

function _should_attempt_phase1_recovery(
    residual::F,
    iteration::Int,
    last_probe_residual::Union{Nothing,F},
    settings,
) where {F<:AbstractFloat}
    iteration == 1 && return true
    residual <= settings.feasibility_tolerance && return true
    residual <= F(1.0e-6) && return true
    last_probe_residual === nothing && return true
    return residual * 64 <= last_probe_residual
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
        numeric_affine = _numeric_affine_data(problem, F)
        A_big = _to_working_array(F, problem.A)
        b_big = _to_working_array(F, problem.b)
        x = _build_phase1_initial_point(problem, numeric_settings, numeric_blocks, numeric_affine)
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
        anchor = nothing
        last_phase1_recovery_probe = nothing
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
                numeric_settings,
            )
            residual = _max_abs(A_big * x - b_big)
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

        if anchor === nothing
            anchor = _phase1_exact_feasible_point(x, problem, opt.settings, numeric_affine)
        end
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
            x0 = _to_working_array(F, x_exact)
            N_big = numeric_affine.nullspace
            c_big = _to_working_array(F, problem.objective_vector_min)
            z = zeros(F, size(nullspace, 2))
            barrier_parameter = one(F)

            phase2_columns = ["Iter", "Mu", "Objective", "Gap", "Time (s)"]
            phase2_alignments = vcat([:left], fill(:right, length(phase2_columns) - 1))
            phase2_widths = _phase_table_widths(phase2_columns, opt.settings.phase2_outer_iterations)
            _log_table_header(
                opt,
                "Phase II",
                phase2_columns,
                phase2_widths;
                subtitle = "Objective path-following",
            )
            phase2_start_time = time_ns()
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
                barrier_parameter *= numeric_settings.path_parameter_growth
            end

            x_candidate = _project_exact_solution(
                _affine_coordinates(x0 + N_big * z, numeric_affine),
                particular,
                nullspace;
                tolerance = numeric_settings.rational_tolerance,
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
        opt.raw_status = "Solved with a primal-only mixed cone barrier method and exact rational recovery."
        opt.result_count = 1
        opt.solve_time_sec = (time_ns() - start_time) / 1.0e9
        _log_raw(opt)
        _log(opt, "finished in " * @sprintf("%.3f", opt.solve_time_sec) * "s with objective $(objective_value)")
    end)
    return
end
