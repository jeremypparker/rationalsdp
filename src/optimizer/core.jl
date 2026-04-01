# Core solver types, logging helpers, and MOI attribute plumbing.

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
    (MOI.VectorAffineFunction,),
    false,
)

Base.@kwdef mutable struct Settings
    max_iterations::Int = 80
    phase1_outer_iterations::Int = 100
    phase2_outer_iterations::Int = 12
    phase1_backend::Symbol = :hypatia
    phase1_hypatia_float_type::DataType = AbstractFloat
    phase1_hypatia_iter_limit::Int = 400
    phase1_hypatia_margin_upper::BigFloat = big"1.0"
    phase1_hypatia_min_margin_upper::BigFloat = big"1e-8"
    phase1_hypatia_margin_shrink::BigFloat = big"0.1"
    phase1_hypatia_objective_bias::BigFloat = big"1e-12"
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
    exact_refinement_bisections::Int = 48
    verbose::Bool = true
    verbose_newton::Bool = false
    live_progress::Bool = true
    inner_log_frequency::Int = 10
    threaded::Bool = true
    threading_min_block_size::Int = 48
    iterative_linear_solver::Bool = true
    iterative_solver_min_dimension::Int = 384
end

struct BlockStructure
    size::Int
    variables::Vector{Union{Nothing,MOI.VariableIndex}}
    global_positions::Vector{Int}
    local_positions::Vector{Tuple{Int,Int}}
    diagonal_positions::Vector{Int}
end

struct EquationTemplate
    coefficients::Vector{ExactRational}
    rhs::ExactRational
    slack_sign::Int
end

mutable struct ProblemData
    original_variables::Vector{MOI.VariableIndex}
    blocks::Vector{BlockStructure}
    positive_scalars::Vector{Int}
    objective_vector_raw::Vector{ExactRational}
    objective_constant_raw::ExactRational
    objective_vector_min::Vector{ExactRational}
    A::Matrix{ExactRational}
    b::Vector{ExactRational}
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}}
    phase1_nullspace::Union{Nothing,Matrix{ExactRational}}
    scalar_constraint_rows::Dict{Any,Vector{Int}}
    psd_constraint_blocks::Dict{Any,Int}
end

struct NumericBlock
    structure::BlockStructure
end

struct PSDBarrierCache{F}
    numeric_block::NumericBlock
    primal_matrix::Matrix{F}
    inverse_matrix::Matrix{F}
    eigenvectors::Matrix{F}
    eigenvalues::Vector{F}
end

mutable struct NumericAffineData{F}
    particular::Vector{F}
    exact_nullspace::Matrix{ExactRational}
    numeric_exact_nullspace::Union{Nothing,Matrix{F}}
    numeric_phase2_basis::Union{Nothing,Matrix{F}}
    nullspace_factor::Any
end

struct Phase1HypatiaAttempt{F<:AbstractFloat}
    anchor::Union{Nothing,Vector{ExactRational}}
    candidate::Union{Nothing,Vector{F}}
    status::String
    iterations::Int
    margin::Union{Nothing,F}
    residual::Union{Nothing,F}
    elapsed_sec::Float64
    reason::Symbol
end

struct SPDSystemFactorizationError <: Exception
    dimension::Int
end

function Base.showerror(io::IO, err::SPDSystemFactorizationError)
    print(io, "Failed to factor Newton system (dimension=", err.dimension, ").")
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
    constraint_primal::Dict{Any,Any}
    constraint_dual::Dict{Any,Any}
end

function Optimizer{T}(; kwargs...) where {T<:Rational}
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
        Dict{Any,Any}(),
        Dict{Any,Any}(),
    )
end

function Optimizer{T}(; kwargs...) where {T<:Real}
    throw(
        ArgumentError(
            "RationalSDP.Optimizer must be parameterized by a rational type, " *
            "for example `RationalSDP.Optimizer{Rational{BigInt}}`.",
        ),
    )
end

Optimizer(; kwargs...) = Optimizer{Rational{BigInt}}(; kwargs...)

function _reset_results!(opt::Optimizer)
    opt.termination_status = MOI.OPTIMIZE_NOT_CALLED
    opt.primal_status = MOI.NO_SOLUTION
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status = "Optimizer not called"
    opt.solve_time_sec = 0.0
    opt.result_count = 0
    empty!(opt.variable_primal)
    opt.objective_value = nothing
    empty!(opt.constraint_primal)
    empty!(opt.constraint_dual)
    return
end

# Logging

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
        flush(stdout)
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
            ["Method", "Primal barrier + exact recovery"],
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

# Settings and numeric-type helpers

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
        lowercase(value) == "auto" && return AbstractFloat
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
    parsed isa Type || error("Float-type settings must be assigned a floating-point type.")
    parsed <: AbstractFloat || error("Float-type settings must be subtypes of AbstractFloat.")
    return parsed
end

function _convert_setting_value(::Type{Symbol}, value)
    return value isa AbstractString ? Symbol(value) : convert(Symbol, value)
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

function _phase1_backend(settings::Settings)
    backend = settings.phase1_backend
    backend in (:hypatia, :native) || error("phase1_backend must be :hypatia or :native.")
    return backend
end

function _phase1_hypatia_float_type(settings::Settings)
    F = settings.phase1_hypatia_float_type
    if F === AbstractFloat
        return _working_float_type(settings)
    end
    F <: AbstractFloat || error("phase1_hypatia_float_type must be a subtype of AbstractFloat or `auto`.")
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

function _to_working_sparse_matrix(::Type{F}, values::AbstractMatrix) where {F<:AbstractFloat}
    row_indices = Int[]
    column_indices = Int[]
    entries = F[]
    for column in 1:size(values, 2)
        for row in 1:size(values, 1)
            value = values[row, column]
            if !iszero(value)
                push!(row_indices, row)
                push!(column_indices, column)
                push!(entries, _to_working_float(F, value))
            end
        end
    end
    return sparse(row_indices, column_indices, entries, size(values)...)
end

function _with_working_precision(settings::Settings, f::Function)
    F = _working_float_type(settings)
    return _with_float_precision(F, settings.working_precision, f)
end

function _with_float_precision(::Type{F}, precision::Int, f::Function) where {F<:AbstractFloat}
    if F === BigFloat
        return setprecision(BigFloat, precision) do
            f(F)
        end
    end
    return f(F)
end

function _numeric_settings(settings::Settings, ::Type{F}) where {F<:AbstractFloat}
    effective_gradient_tolerance = max(
        _to_working_float(F, settings.gradient_tolerance),
        sqrt(eps(F)),
    )
    return (
        max_iterations = settings.max_iterations,
        phase1_outer_iterations = settings.phase1_outer_iterations,
        phase2_outer_iterations = settings.phase2_outer_iterations,
        working_float_type = F,
        feasibility_tolerance = _to_working_float(F, settings.feasibility_tolerance),
        optimality_gap_tolerance = _to_working_float(F, settings.optimality_gap_tolerance),
        gradient_tolerance = effective_gradient_tolerance,
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
        iterative_linear_solver = settings.iterative_linear_solver,
        iterative_solver_min_dimension = settings.iterative_solver_min_dimension,
    )
end

# MOI attribute plumbing

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
    if attr isa MOI.ConstraintDual || attr isa MOI.ConstraintPrimal
        return true
    end
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
    symbol === :phase1_hypatia_float_type && return _phase1_hypatia_float_type(opt.settings)
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
    if attr isa MOI.ConstraintDual && haskey(opt.constraint_dual, ci)
        MOI.check_result_index_bounds(opt, attr)
        return opt.constraint_dual[ci]
    elseif attr isa MOI.ConstraintPrimal && haskey(opt.constraint_primal, ci)
        MOI.check_result_index_bounds(opt, attr)
        return opt.constraint_primal[ci]
    end
    return MOI.get(opt.storage, attr, ci)
end
