# Core solver types, logging helpers, and MOI attribute plumbing.

const MOIU = MOI.Utilities
const ExactRational = Rational{BigInt}

"""Mutable counters collected during one `optimize!` call."""
mutable struct FacialReductionStatistics
    phase1_attempts::Int
    phase1_time_sec::Float64
    oracle_attempts::Int
    oracle_iterations::Int
    oracle_time_sec::Float64
    exact_rref_calls::Int
    exact_rref_dimensions::Vector{Tuple{Int,Int}}
    exact_rref_time_sec::Float64
    exact_row_space_checks::Int
    exact_row_space_check_time_sec::Float64
    exact_certificate_checks::Int
    exact_certificate_time_sec::Float64
    psd_eigendecompositions_by_block_size::Dict{Int,Int}
    psd_eigendecomposition_time_sec::Float64
    certified_directions_proposed::Int
    certified_directions_accepted::Int
    certified_directions_rejected::Int
    tentative_directions_proposed::Int
    tentative_directions_accepted::Int
    tentative_directions_rejected::Int
    cone_dimension_removed_per_round::Vector{Int}
    affine_cache_peak_bytes::Int
    facial_reduction_cache_peak_bytes::Int
    certified_reductions_applied::Int
    tentative_restrictions_applied::Int
    rational_subspace_charts_attempted::Int
    rational_projectors_attempted::Int
    precision_escalations_attempted::Int
    tentative_batches_skipped_by_budget::Int
    affine_lifts_skipped_by_budget::Int
    recovery_path_attempts::Dict{Symbol,Int}
    recovery_path_successes::Dict{Symbol,Int}
end

FacialReductionStatistics() = FacialReductionStatistics(
    0, 0.0, 0, 0, 0.0, 0, Tuple{Int,Int}[], 0.0, 0, 0.0, 0, 0.0,
    Dict{Int,Int}(), 0.0, 0, 0, 0, 0, 0, 0, Int[], 0, 0, 0, 0, 0, 0, 0, 0, 0,
    Dict{Symbol,Int}(), Dict{Symbol,Int}(),
)

function _facial_reduction_statistics_snapshot(stats::FacialReductionStatistics)
    return (
        phase1_attempts = stats.phase1_attempts,
        phase1_time_sec = stats.phase1_time_sec,
        oracle_attempts = stats.oracle_attempts,
        oracle_iterations = stats.oracle_iterations,
        oracle_time_sec = stats.oracle_time_sec,
        exact_rref_calls = stats.exact_rref_calls,
        exact_rref_dimensions = copy(stats.exact_rref_dimensions),
        exact_rref_time_sec = stats.exact_rref_time_sec,
        exact_row_space_checks = stats.exact_row_space_checks,
        exact_row_space_check_time_sec = stats.exact_row_space_check_time_sec,
        exact_certificate_checks = stats.exact_certificate_checks,
        exact_certificate_time_sec = stats.exact_certificate_time_sec,
        psd_eigendecompositions_by_block_size = copy(stats.psd_eigendecompositions_by_block_size),
        psd_eigendecomposition_time_sec = stats.psd_eigendecomposition_time_sec,
        certified_directions_proposed = stats.certified_directions_proposed,
        certified_directions_accepted = stats.certified_directions_accepted,
        certified_directions_rejected = stats.certified_directions_rejected,
        tentative_directions_proposed = stats.tentative_directions_proposed,
        tentative_directions_accepted = stats.tentative_directions_accepted,
        tentative_directions_rejected = stats.tentative_directions_rejected,
        cone_dimension_removed_per_round = copy(stats.cone_dimension_removed_per_round),
        affine_cache_peak_bytes = stats.affine_cache_peak_bytes,
        facial_reduction_cache_peak_bytes = stats.facial_reduction_cache_peak_bytes,
        certified_reductions_applied = stats.certified_reductions_applied,
        tentative_restrictions_applied = stats.tentative_restrictions_applied,
        rational_subspace_charts_attempted = stats.rational_subspace_charts_attempted,
        rational_projectors_attempted = stats.rational_projectors_attempted,
        precision_escalations_attempted = stats.precision_escalations_attempted,
        tentative_batches_skipped_by_budget = stats.tentative_batches_skipped_by_budget,
        affine_lifts_skipped_by_budget = stats.affine_lifts_skipped_by_budget,
        recovery_path_attempts = copy(stats.recovery_path_attempts),
        recovery_path_successes = copy(stats.recovery_path_successes),
    )
end

function _record_facial_reduction_path!(path::Symbol; success::Bool = false)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    counts = success ? stats.recovery_path_successes : stats.recovery_path_attempts
    counts[path] = get(counts, path, 0) + 1
    return
end

function _record_facial_reduction_event!(field::Symbol, count::Int = 1)
    count >= 0 || error("Facial-reduction event count cannot be negative.")
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    field in (
        :rational_subspace_charts_attempted,
        :rational_projectors_attempted,
        :precision_escalations_attempted,
        :tentative_batches_skipped_by_budget,
        :affine_lifts_skipped_by_budget,
    ) || error("Unknown facial-reduction statistics field $(field).")
    setfield!(stats, field, getfield(stats, field) + count)
    return
end

function _current_facial_reduction_statistics()
    return try
        Base.task_local_storage(:rational_sdp_facial_reduction_statistics)
    catch
        nothing
    end
end

function _with_facial_reduction_statistics(f::Function, stats::FacialReductionStatistics)
    previous = try
        Base.task_local_storage(:rational_sdp_facial_reduction_statistics)
    catch
        nothing
    end
    Base.task_local_storage(:rational_sdp_facial_reduction_statistics, stats)
    try
        return f()
    finally
        Base.task_local_storage(:rational_sdp_facial_reduction_statistics, previous)
    end
end

function _record_approximate_cache_memory!(kind::Symbol, value)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    bytes = try
        Base.summarysize(value)
    catch
        0
    end
    if kind === :affine
        stats.affine_cache_peak_bytes = max(stats.affine_cache_peak_bytes, bytes)
    elseif kind === :facial_reduction
        stats.facial_reduction_cache_peak_bytes = max(
            stats.facial_reduction_cache_peak_bytes,
            bytes,
        )
    end
    return
end

function _record_phase1_attempt!(elapsed_sec::Real)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    stats.phase1_attempts += 1
    stats.phase1_time_sec += Float64(elapsed_sec)
    return
end

function _record_oracle_attempt!(iterations::Integer, elapsed_sec::Real)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    stats.oracle_attempts += 1
    stats.oracle_iterations += Int(iterations)
    stats.oracle_time_sec += Float64(elapsed_sec)
    return
end

function _record_row_space_check!(elapsed_sec::Real)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    stats.exact_row_space_checks += 1
    stats.exact_row_space_check_time_sec += Float64(elapsed_sec)
    return
end

function _record_certificate_check!(elapsed_sec::Real)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    stats.exact_certificate_checks += 1
    stats.exact_certificate_time_sec += Float64(elapsed_sec)
    return
end

function _record_psd_eigendecomposition!(block_size::Int, elapsed_sec::Real)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    stats.psd_eigendecompositions_by_block_size[block_size] =
        get(stats.psd_eigendecompositions_by_block_size, block_size, 0) + 1
    stats.psd_eigendecomposition_time_sec += Float64(elapsed_sec)
    return
end

function _record_directions!(kind::Symbol, proposed::Int, accepted::Int, rejected::Int)
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    proposed >= 0 && accepted >= 0 && rejected >= 0 ||
        error("Direction statistics cannot be negative.")
    if kind === :certified
        stats.certified_directions_proposed += proposed
        stats.certified_directions_accepted += accepted
        stats.certified_directions_rejected += rejected
    elseif kind === :tentative
        stats.tentative_directions_proposed += proposed
        stats.tentative_directions_accepted += accepted
        stats.tentative_directions_rejected += rejected
    else
        error("Unknown facial-reduction direction kind $(kind).")
    end
    return
end

function _record_reduction_round!(old_barrier_dimension::Int, new_barrier_dimension::Int; tentative::Bool)
    new_barrier_dimension < old_barrier_dimension ||
        error("A reported facial reduction did not decrease barrier dimension.")
    stats = _current_facial_reduction_statistics()
    stats isa FacialReductionStatistics || return
    push!(
        stats.cone_dimension_removed_per_round,
        old_barrier_dimension - new_barrier_dimension,
    )
    if tentative
        stats.tentative_restrictions_applied += 1
    else
        stats.certified_reductions_applied += 1
    end
    return
end

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
    phase2_outer_iterations::Int = 24
    phase1_backend::Symbol = :hypatia
    phase1_hypatia_float_type::DataType = AbstractFloat
    phase1_hypatia_syssolver::Symbol = :auto
    phase1_hypatia_iter_limit::Int = 400
    phase1_hypatia_target_margin::BigFloat = big"1e-8"
    phase1_hypatia_margin_upper::BigFloat = big"1.0"
    phase1_hypatia_min_margin_upper::BigFloat = big"1e-8"
    phase1_hypatia_margin_shrink::BigFloat = big"0.1"
    phase1_hypatia_boundary_margin_fraction::BigFloat = big"0.01"
    phase1_hypatia_tol_rel_opt::BigFloat = big"-1"
    phase1_hypatia_tol_abs_opt::BigFloat = big"-1"
    phase1_hypatia_tol_feas::BigFloat = big"-1"
    phase1_hypatia_default_tol_power::BigFloat = big"-1"
    phase1_hypatia_default_tol_relax::BigFloat = big"-1"
    phase1_hypatia_tol_slow::BigFloat = big"-1"
    phase1_candidate_diagnostics::Bool = false
    phase1_stop_after_candidate_diagnostics::Bool = false
    phase1_exact_recovery_diagnostics::Bool = false
    phase1_exact_recovery_pivot_log_frequency::Int = 10
    working_float_type::DataType = Float64x2
    facial_reduction::Bool = true
    facial_reduction_max_rounds::Int = 24
    # Rank expansion launches another exposing-vector solve after an exact
    # face has already been certified.  Keep it opt-in: the common case is
    # that the first certified face is already sufficient.
    facial_reduction_rank_expansion_rounds::Int = 0
    facial_reduction_float_type::DataType = AbstractFloat
    facial_reduction_exposure_tolerance::BigFloat = big"1e-8"
    facial_reduction_rank_tolerance::BigFloat = big"1e-8"
    facial_reduction_subspace_max_charts::Int = 8
    facial_reduction_projector_recovery::Bool = true
    # Coupled multi-block and stable partial rational reconstruction are
    # default-on fallbacks, but are attempted only after the established
    # initial-evidence and exposing-vector oracle routes have failed.
    facial_reduction_multiblock_recovery::Bool = true
    facial_reduction_stable_projective_recovery::Bool = true
    # `:auto` tries the Phase-I system solver first and then a stable oracle
    # fallback.  An explicit choice is tried first without changing Phase I.
    facial_reduction_oracle_syssolver::Symbol = :auto
    # Oracle recovery is independent of Phase-I precision escalation.  The
    # configured precision and system solver are still tried first; this only
    # controls higher-precision retries after those attempts fail.
    facial_reduction_oracle_precision_escalation_max_retries::Int = 1
    facial_reduction_precision_escalation_max_retries::Int = 0
    facial_reduction_irrational_behavior::Symbol = :error
    facial_reduction_save_file::String = ""
    facial_reduction_load_file::String = ""
    # Work limits for optional facial-reduction certificates, numerical
    # scouts, and redundant validation. Zero disables the max-work routes
    # without disabling facial reduction itself; a zero compaction factor
    # compacts whenever extra affine equations are present.
    facial_reduction_row_space_max_entries::Int = 250_000
    facial_reduction_weighted_subspace_max_form_entries::Int = 250_000
    facial_reduction_weighted_subspace_max_affine_products::Int = 5_000_000
    facial_reduction_cheap_weighted_subspace_max_weight_dimension::Int = 6
    facial_reduction_cheap_weighted_subspace_max_form_entries::Int = 25_000
    facial_reduction_cheap_weighted_subspace_max_affine_products::Int = 500_000
    facial_reduction_numeric_weighted_subspace_max_form_entries::Int = 1_000_000
    facial_reduction_numeric_weighted_subspace_max_affine_products::Int = 25_000_000
    facial_reduction_individual_max_affine_products::Int = 5_000_000
    facial_reduction_sparse_affine_validation_max_products::Int = 5_000_000
    facial_reduction_sieve_transform_max_entries::Int = 250_000
    facial_reduction_affine_compaction_factor::Int = 4
    facial_reduction_affine_compaction_max_entries::Int = 5_000_000
    facial_reduction_tentative_max_directions::Int = 8
    facial_reduction_tentative_max_coordinate_entries::Int = 5_000_000
    facial_reduction_tentative_max_lift_products::Int = 250_000_000
    facial_reduction_tentative_max_estimated_bytes::Int = 1_000_000_000
    facial_reduction_affine_lift_max_output_entries::Int = 5_000_000
    facial_reduction_affine_lift_max_estimated_bytes::Int = 1_000_000_000
    facial_reduction_affine_lift_chunk_columns::Int = 32
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
    recovery_tolerance_shrink::BigFloat = big"0.1"
    exact_refinement_bisections::Int = 48
    verbose::Bool = true
    verbose_newton::Bool = false
    live_progress::Bool = true
    inner_log_frequency::Int = 10
    threaded::Bool = true
    threading_min_block_size::Int = 48
    iterative_linear_solver::Bool = true
    iterative_solver_min_dimension::Int = 384
    gc_collect_extraction::Bool = false
    gc_collect_full::Bool = true
    gc_log::Bool = false
    quasiconvex_bisection_iterations::Int = 24
    quasiconvex_skip_facial_reduction_after_clean_endpoint::Bool = true
end

struct BlockStructure
    size::Int
    variables::Vector{Union{Nothing,MOI.VariableIndex}}
    global_positions::Vector{Int}
    local_positions::Vector{Tuple{Int,Int}}
    diagonal_positions::Vector{Int}
end

struct EquationTemplate
    indices::Vector{Int}
    values::Vector{ExactRational}
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
    solution_lift::SparseMatrixCSC{ExactRational,Int}
    legacy_facial_reduction_signature::Any
    legacy_position_map::Vector{Int}
    legacy_coordinate_lift::SparseMatrixCSC{ExactRational,Int}
    phase1_nullspace_float_type::Union{Nothing,DataType}
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
    dual_slack::Union{Nothing,Vector{F}}
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
    quadratic_psd_functions::Dict{
        MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.PositiveSemidefiniteConeTriangle},
        MOI.VectorQuadraticFunction{T},
    }
    quadratic_psd_sets::Dict{
        MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.PositiveSemidefiniteConeTriangle},
        MOI.PositiveSemidefiniteConeTriangle,
    }
    next_quadratic_psd_index::Int
    scalar_quadratic_functions::Dict{Any,MOI.ScalarQuadraticFunction{T}}
    scalar_quadratic_sets::Dict{Any,Any}
    next_scalar_quadratic_index::Int
    facial_reduction_save_records::Vector{Any}
    facial_reduction_loaded_records::Union{Nothing,Vector{Any}}
    facial_reduction_statistics::FacialReductionStatistics
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
        Dict{
            MOI.ConstraintIndex{
                MOI.VectorQuadraticFunction{T},
                MOI.PositiveSemidefiniteConeTriangle,
            },
            MOI.VectorQuadraticFunction{T},
        }(),
        Dict{
            MOI.ConstraintIndex{
                MOI.VectorQuadraticFunction{T},
                MOI.PositiveSemidefiniteConeTriangle,
            },
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
        1,
        Dict{Any,MOI.ScalarQuadraticFunction{T}}(),
        Dict{Any,Any}(),
        1,
        Any[],
        nothing,
        FacialReductionStatistics(),
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

function facial_reduction_statistics(opt::Optimizer)
    return _facial_reduction_statistics_snapshot(opt.facial_reduction_statistics)
end

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
    return
end

# Logging

struct _HypatiaCenteringWarningFilter <: Logging.AbstractLogger
    logger::Logging.AbstractLogger
end

function _is_filtered_hypatia_warning(level, message, _module)
    level == Logging.Warn || return false
    startswith(string(_module), "Hypatia") || return false
    message_text = string(message)
    return occursin("cannot step in centering direction", message_text) ||
           occursin("some dual equalities appear to be dependent", message_text)
end

Logging.min_enabled_level(logger::_HypatiaCenteringWarningFilter) =
    Logging.min_enabled_level(logger.logger)

Logging.shouldlog(
    logger::_HypatiaCenteringWarningFilter,
    level,
    _module,
    group,
    id,
) = Logging.shouldlog(logger.logger, level, _module, group, id)

Logging.catch_exceptions(logger::_HypatiaCenteringWarningFilter) =
    Logging.catch_exceptions(logger.logger)

function Logging.handle_message(
    logger::_HypatiaCenteringWarningFilter,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    if _is_filtered_hypatia_warning(level, message, _module)
        return
    end
    return Logging.handle_message(
        logger.logger,
        level,
        message,
        _module,
        group,
        id,
        file,
        line;
        kwargs...,
    )
end

function _with_filtered_hypatia_logger(f::Function)
    logger = _HypatiaCenteringWarningFilter(Logging.current_logger())
    return Logging.with_logger(logger) do
        f()
    end
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

function _format_exact_rational_compact(value::ExactRational; max_chars::Int = 96)
    text = string(value)
    ncodeunits(text) <= max_chars && return text
    return first(text, max_chars) * "..."
end

function _bigint_bits(value::BigInt)
    iszero(value) && return 0
    return ndigits(abs(value); base = 2)
end

function _format_exact_rational_size(value::ExactRational)
    return "num_bits=$(_bigint_bits(numerator(value))), den_bits=$(_bigint_bits(denominator(value)))"
end

function _log(opt::Optimizer, message::AbstractString)
    if !opt.silent && opt.settings.verbose
        println("[RationalSDP] ", message)
        flush(stdout)
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

function _format_bytes(bytes::Integer)
    value = float(bytes)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    unit_index = 1
    while unit_index < length(units) && value >= 1024
        value /= 1024
        unit_index += 1
    end
    if unit_index == 1
        return string(bytes, " ", units[unit_index])
    end
    return @sprintf("%.2f %s", value, units[unit_index])
end

function _gc_snapshot()
    stats = Base.gc_num()
    live_bytes = try
        Int(Base.gc_live_bytes())
    catch
        -1
    end
    return (
        live_bytes = live_bytes,
        full_sweeps = Int(getfield(stats, :full_sweep)),
        allocd = Int(getfield(stats, :allocd)),
    )
end

function _log_gc_state(opt::Optimizer, label::AbstractString)
    opt.settings.gc_log || return
    snapshot = _gc_snapshot()
    live_text = snapshot.live_bytes >= 0 ? _format_bytes(snapshot.live_bytes) : "unavailable"
    _log(
        opt,
        "GC $(label): live=$(live_text), full_sweeps=$(snapshot.full_sweeps), " *
        "allocd_since_gc=$(_format_bytes(snapshot.allocd))",
    )
    return
end

function _gc_checkpoint!(opt::Optimizer, label::AbstractString)
    opt.settings.gc_log && _log_gc_state(opt, label * " (before)")
    if opt.settings.gc_collect_extraction
        GC.gc(false)
        if opt.settings.gc_collect_full
            GC.gc(true)
        end
    end
    opt.settings.gc_log && _log_gc_state(opt, label * " (after)")
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
        parts = split(value, '.')
        if length(parts) == 2 && parts[1] == "MultiFloats"
            symbol = Symbol(parts[2])
            isdefined(MultiFloats, symbol) || error("Unknown working float type: $(value)")
            getfield(MultiFloats, symbol)
        else
            symbol = Symbol(value)
            if isdefined(@__MODULE__, symbol)
                getfield(@__MODULE__, symbol)
            elseif isdefined(Base, symbol)
                getfield(Base, symbol)
            else
                error("Unknown working float type: $(value)")
            end
        end
    else
        value
    end
    parsed isa DataType || error("Float-type settings must be assigned a floating-point data type.")
    parsed <: AbstractFloat || error("Float-type settings must be subtypes of AbstractFloat.")
    return parsed
end

function _convert_setting_value(::Type{Symbol}, value)
    return value isa AbstractString ? Symbol(value) : convert(Symbol, value)
end

function _convert_setting_value(::Type{String}, value)
    value === nothing && return ""
    return String(value)
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

function _phase1_hypatia_float_type_is_auto(settings::Settings)
    return settings.phase1_hypatia_float_type === AbstractFloat
end

function _phase1_hypatia_syssolver(settings::Settings)
    syssolver = settings.phase1_hypatia_syssolver
    syssolver in (
        :auto,
        :symindef_sparse,
        :symindef_dense,
        :symindef_indirect,
        :qrchol_dense,
        :naive_dense,
        :naiveelim_dense,
    ) || error(
        "phase1_hypatia_syssolver must be one of " *
        ":auto, :symindef_sparse, :symindef_dense, :symindef_indirect, " *
        ":qrchol_dense, :naive_dense, :naiveelim_dense.",
    )
    return syssolver
end

function _facial_reduction_oracle_syssolver(settings::Settings)
    syssolver = settings.facial_reduction_oracle_syssolver
    syssolver in (
        :auto,
        :symindef_sparse,
        :symindef_dense,
        :symindef_indirect,
        :qrchol_dense,
        :naive_dense,
        :naiveelim_dense,
    ) || error(
        "facial_reduction_oracle_syssolver must be one of " *
        ":auto, :symindef_sparse, :symindef_dense, :symindef_indirect, " *
        ":qrchol_dense, :naive_dense, :naiveelim_dense.",
    )
    return syssolver
end

function _phase1_hypatia_target_margin(settings::Settings)
    target = settings.phase1_hypatia_target_margin
    target >= 0 || error("phase1_hypatia_target_margin must be nonnegative.")
    return target
end

function _phase1_hypatia_boundary_margin_fraction(settings::Settings)
    fraction = settings.phase1_hypatia_boundary_margin_fraction
    zero(fraction) <= fraction <= one(fraction) ||
        error("phase1_hypatia_boundary_margin_fraction must lie between 0 and 1.")
    return fraction
end

function _facial_reduction_float_type(settings::Settings)
    F = settings.facial_reduction_float_type
    if F === AbstractFloat
        return _working_float_type(settings)
    end
    F <: AbstractFloat ||
        error("facial_reduction_float_type must be a subtype of AbstractFloat or `auto`.")
    return F
end

function _facial_reduction_irrational_behavior(settings::Settings)
    behavior = settings.facial_reduction_irrational_behavior
    behavior in (:error, :warn) ||
        error("facial_reduction_irrational_behavior must be :error or :warn.")
    return behavior
end

function _precision_escalation_types(::Type{F}, max_retries::Int) where {F<:AbstractFloat}
    max_retries >= 0 || error("Precision escalation retries must be nonnegative.")
    ladder = DataType[Float64, Float64x2, Float64x3, Float64x4, BigFloat]
    start = findfirst(==(F), ladder)
    candidates = if start === nothing
        F === BigFloat ? DataType[] : DataType[BigFloat]
    else
        ladder[(start + 1):end]
    end
    # Float64x3 rarely provides enough separation from Float64x2 to justify
    # another conic solve. Keep it available when explicitly configured, but
    # skip it in automatic escalation from lower precision.
    F in (Float64, Float64x2) && filter!(!=(Float64x3), candidates)
    return candidates[1:min(max_retries, length(candidates))]
end

function _validate_settings(settings::Settings)
    positive_integer_fields = (
        :max_iterations,
        :phase1_outer_iterations,
        :phase2_outer_iterations,
        :phase1_hypatia_iter_limit,
        :working_precision,
        :inner_log_frequency,
        :threading_min_block_size,
        :iterative_solver_min_dimension,
        :quasiconvex_bisection_iterations,
    )
    for name in positive_integer_fields
        getfield(settings, name) > 0 || throw(ArgumentError("$(name) must be positive."))
    end
    settings.facial_reduction_max_rounds >= 0 ||
        throw(ArgumentError("facial_reduction_max_rounds must be nonnegative."))
    settings.facial_reduction_rank_expansion_rounds >= 0 ||
        throw(ArgumentError("facial_reduction_rank_expansion_rounds must be nonnegative."))
    facial_reduction_work_limits = (
        :facial_reduction_row_space_max_entries,
        :facial_reduction_weighted_subspace_max_form_entries,
        :facial_reduction_weighted_subspace_max_affine_products,
        :facial_reduction_cheap_weighted_subspace_max_weight_dimension,
        :facial_reduction_cheap_weighted_subspace_max_form_entries,
        :facial_reduction_cheap_weighted_subspace_max_affine_products,
        :facial_reduction_numeric_weighted_subspace_max_form_entries,
        :facial_reduction_numeric_weighted_subspace_max_affine_products,
        :facial_reduction_individual_max_affine_products,
        :facial_reduction_sparse_affine_validation_max_products,
        :facial_reduction_sieve_transform_max_entries,
        :facial_reduction_affine_compaction_factor,
        :facial_reduction_affine_compaction_max_entries,
        :facial_reduction_tentative_max_directions,
        :facial_reduction_tentative_max_coordinate_entries,
        :facial_reduction_tentative_max_lift_products,
        :facial_reduction_tentative_max_estimated_bytes,
        :facial_reduction_affine_lift_max_output_entries,
        :facial_reduction_affine_lift_max_estimated_bytes,
    )
    for name in facial_reduction_work_limits
        getfield(settings, name) >= 0 ||
            throw(ArgumentError("$(name) must be nonnegative."))
    end
    settings.phase1_exact_recovery_pivot_log_frequency >= 0 ||
        throw(ArgumentError("phase1_exact_recovery_pivot_log_frequency must be nonnegative."))
    settings.facial_reduction_subspace_max_charts > 0 ||
        throw(ArgumentError("facial_reduction_subspace_max_charts must be positive."))
    settings.facial_reduction_precision_escalation_max_retries >= 0 ||
        throw(ArgumentError("facial_reduction_precision_escalation_max_retries must be nonnegative."))
    settings.facial_reduction_oracle_precision_escalation_max_retries >= 0 ||
        throw(ArgumentError("facial_reduction_oracle_precision_escalation_max_retries must be nonnegative."))
    settings.facial_reduction_affine_lift_chunk_columns > 0 ||
        throw(ArgumentError("facial_reduction_affine_lift_chunk_columns must be positive."))
    settings.exact_refinement_bisections >= 0 ||
        throw(ArgumentError("exact_refinement_bisections must be nonnegative."))

    _working_float_type(settings)
    _phase1_backend(settings)
    _phase1_hypatia_float_type(settings)
    _phase1_hypatia_syssolver(settings)
    _facial_reduction_oracle_syssolver(settings)
    _phase1_hypatia_target_margin(settings)
    _phase1_hypatia_boundary_margin_fraction(settings)
    _facial_reduction_float_type(settings)
    _facial_reduction_irrational_behavior(settings)

    settings.phase1_hypatia_margin_upper > 0 ||
        throw(ArgumentError("phase1_hypatia_margin_upper must be positive."))
    settings.phase1_hypatia_min_margin_upper > 0 ||
        throw(ArgumentError("phase1_hypatia_min_margin_upper must be positive."))
    0 < settings.phase1_hypatia_margin_shrink < 1 ||
        throw(ArgumentError("phase1_hypatia_margin_shrink must lie strictly between 0 and 1."))
    settings.facial_reduction_exposure_tolerance >= 0 ||
        throw(ArgumentError("facial_reduction_exposure_tolerance must be nonnegative."))
    settings.facial_reduction_rank_tolerance >= 0 ||
        throw(ArgumentError("facial_reduction_rank_tolerance must be nonnegative."))

    positive_real_fields = (
        :feasibility_tolerance,
        :optimality_gap_tolerance,
        :gradient_tolerance,
        :min_step,
        :initial_scale,
        :initial_penalty,
        :phase1_center_weight,
        :rational_tolerance,
    )
    for name in positive_real_fields
        getfield(settings, name) > 0 || throw(ArgumentError("$(name) must be positive."))
    end
    0 < settings.line_search_shrink < 1 ||
        throw(ArgumentError("line_search_shrink must lie strictly between 0 and 1."))
    0 < settings.armijo_fraction < 1 ||
        throw(ArgumentError("armijo_fraction must lie strictly between 0 and 1."))
    0 < settings.boundary_fraction < 1 ||
        throw(ArgumentError("boundary_fraction must lie strictly between 0 and 1."))
    0 < settings.recovery_tolerance_shrink < 1 ||
        throw(ArgumentError("recovery_tolerance_shrink must lie strictly between 0 and 1."))
    settings.penalty_growth > 1 || throw(ArgumentError("penalty_growth must be greater than 1."))
    settings.path_parameter_growth > 1 ||
        throw(ArgumentError("path_parameter_growth must be greater than 1."))
    return nothing
end

_to_working_float(::Type{F}, x::ExactRational) where {F<:AbstractFloat} = F(numerator(x)) / F(denominator(x))
_to_working_float(::Type{F}, x::Rational{S}) where {F<:AbstractFloat,S<:Integer} = F(numerator(x)) / F(denominator(x))
_to_working_float(::Type{F}, x::Integer) where {F<:AbstractFloat} = F(x)
_to_working_float(::Type{F}, x::AbstractFloat) where {F<:AbstractFloat} = F(x)

function _rationalize_float(value::F, tolerance::F) where {F<:AbstractFloat}
    return rationalize(BigInt, value; tol = tolerance)
end

function _rationalize_multifloat(value::MultiFloat, tolerance::AbstractFloat)
    precision_bits = max(precision(typeof(value)), precision(typeof(tolerance)))
    return setprecision(BigFloat, precision_bits) do
        rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance))
    end
end

_rationalize_float(value::MultiFloat, tolerance::AbstractFloat) =
    _rationalize_multifloat(value, tolerance)

_rationalize_float(value::F, tolerance::F) where {F<:MultiFloat} =
    _rationalize_multifloat(value, tolerance)

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
    effective_optimality_gap_tolerance = max(
        _to_working_float(F, settings.optimality_gap_tolerance),
        F(10) * sqrt(eps(F)),
    )
    effective_phase2_gradient_tolerance = max(
        effective_gradient_tolerance,
        F(100) * sqrt(eps(F)),
    )
    return (
        max_iterations = settings.max_iterations,
        phase1_outer_iterations = settings.phase1_outer_iterations,
        phase2_outer_iterations = settings.phase2_outer_iterations,
        working_float_type = F,
        feasibility_tolerance = _to_working_float(F, settings.feasibility_tolerance),
        optimality_gap_tolerance = effective_optimality_gap_tolerance,
        gradient_tolerance = effective_gradient_tolerance,
        phase2_gradient_tolerance = effective_phase2_gradient_tolerance,
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
MOI.is_empty(opt::Optimizer) =
    MOI.is_empty(opt.storage) &&
    isempty(opt.quadratic_psd_functions) &&
    isempty(opt.scalar_quadratic_functions)

function MOI.empty!(opt::Optimizer)
    MOI.empty!(opt.storage)
    empty!(opt.quadratic_psd_functions)
    empty!(opt.quadratic_psd_sets)
    opt.next_quadratic_psd_index = 1
    empty!(opt.scalar_quadratic_functions)
    empty!(opt.scalar_quadratic_sets)
    opt.next_scalar_quadratic_index = 1
    _reset_results!(opt)
    return
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.VectorQuadraticFunction{T}},
    ::Type{MOI.PositiveSemidefiniteConeTriangle},
) where {T<:Real}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{MOI.ScalarQuadraticFunction{T}},
    ::Type{S},
) where {T<:Real,S<:Union{MOI.EqualTo{T},MOI.GreaterThan{T},MOI.LessThan{T},MOI.Interval{T}}}
    return true
end

MOI.supports_constraint(
    opt::Optimizer,
    ::Type{F},
    ::Type{S},
) where {F<:MOI.AbstractFunction,S<:MOI.AbstractSet} = MOI.supports_constraint(opt.storage, F, S)

MOI.is_valid(opt::Optimizer, vi::MOI.VariableIndex) = MOI.is_valid(opt.storage, vi)

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.PositiveSemidefiniteConeTriangle},
) where {T<:Real}
    return haskey(opt.quadratic_psd_functions, ci)
end

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
) where {T<:Real,S}
    return haskey(opt.scalar_quadratic_functions, ci)
end

function MOI.is_valid(
    opt::Optimizer,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    return MOI.is_valid(opt.storage, ci)
end

MOI.add_variable(opt::Optimizer) = MOI.add_variable(opt.storage)

function MOI.add_constraint(
    opt::Optimizer{T},
    func::MOI.VectorQuadraticFunction{T},
    set::MOI.PositiveSemidefiniteConeTriangle,
) where {T<:Real}
    ci = MOI.ConstraintIndex{
        MOI.VectorQuadraticFunction{T},
        MOI.PositiveSemidefiniteConeTriangle,
    }(opt.next_quadratic_psd_index)
    opt.next_quadratic_psd_index += 1
    opt.quadratic_psd_functions[ci] = func
    opt.quadratic_psd_sets[ci] = set
    return ci
end

function MOI.add_constraint(
    opt::Optimizer{T},
    func::MOI.ScalarQuadraticFunction{T},
    set::S,
) where {T<:Real,S<:Union{MOI.EqualTo{T},MOI.GreaterThan{T},MOI.LessThan{T},MOI.Interval{T}}}
    ci = MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}(
        opt.next_scalar_quadratic_index,
    )
    opt.next_scalar_quadratic_index += 1
    opt.scalar_quadratic_functions[ci] = func
    opt.scalar_quadratic_sets[ci] = set
    return ci
end

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
    if attr isa MOI.ConstraintPrimal
        return true
    end
    if F <: MOI.VectorQuadraticFunction && S <: MOI.PositiveSemidefiniteConeTriangle
        return false
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
    symbol === :facial_reduction_float_type && return _facial_reduction_float_type(opt.settings)
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
    opt::Optimizer{T},
    ::MOI.ListOfConstraintIndices{
        MOI.VectorQuadraticFunction{T},
        MOI.PositiveSemidefiniteConeTriangle,
    },
) where {T<:Real}
    return collect(keys(opt.quadratic_psd_functions))
end

function MOI.get(
    opt::Optimizer{T},
    ::MOI.ListOfConstraintIndices{MOI.ScalarQuadraticFunction{T},S},
) where {T<:Real,S}
    return MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}[
        ci for ci in keys(opt.scalar_quadratic_functions) if
        ci isa MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S}
    ]
end

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
    ci::MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.PositiveSemidefiniteConeTriangle},
) where {T<:Real}
    if attr isa MOI.ConstraintPrimal && haskey(opt.constraint_primal, ci)
        MOI.check_result_index_bounds(opt, attr)
        return opt.constraint_primal[ci]
    elseif attr isa MOI.ConstraintFunction
        return opt.quadratic_psd_functions[ci]
    elseif attr isa MOI.ConstraintSet
        return opt.quadratic_psd_sets[ci]
    end
    throw(MOI.UnsupportedAttribute(attr))
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex{MOI.ScalarQuadraticFunction{T},S},
) where {T<:Real,S}
    if attr isa MOI.ConstraintPrimal && haskey(opt.constraint_primal, ci)
        MOI.check_result_index_bounds(opt, attr)
        return opt.constraint_primal[ci]
    elseif attr isa MOI.ConstraintFunction
        return opt.scalar_quadratic_functions[ci]
    elseif attr isa MOI.ConstraintSet
        return opt.scalar_quadratic_sets[ci]
    end
    throw(MOI.UnsupportedAttribute(attr))
end

function MOI.get(
    opt::Optimizer,
    attr::MOI.AbstractConstraintAttribute,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    if attr isa MOI.ConstraintPrimal && haskey(opt.constraint_primal, ci)
        MOI.check_result_index_bounds(opt, attr)
        return opt.constraint_primal[ci]
    end
    return MOI.get(opt.storage, attr, ci)
end
