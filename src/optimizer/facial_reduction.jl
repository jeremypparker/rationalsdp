# Internal facial reduction on the extracted primal conic form.
#
# We use Hypatia as a floating-point oracle to find exposing vectors for the
# current affine slice, then convert the exposed nullspace directions back into
# exact rational linear constraints on the original primal variables.

mutable struct _FacialReductionRowSpaceCache
    pivot_columns::Vector{Int}
    selected_rows::Vector{Int}
    inverse::Matrix{ExactRational}
end

mutable struct _WeightedSubspaceSearchState
    # Candidate allowances are per PSD block so an earlier block cannot starve
    # later blocks. The product and exact-without-scout caps remain cumulative
    # across the exact-cache lifetime because they bound total exact work.
    candidate_sets_attempted::Dict{Int,Int}
    total_affine_products::BigInt
    exact_without_scout_attempts::Int
end

mutable struct _FacialReductionExactCache
    problem::ProblemData
    row_space::Union{Nothing,_FacialReductionRowSpaceCache}
    block_exact_directions::Vector{Union{Nothing,Vector{Vector{ExactRational}}}}
    A_transpose::Union{Nothing,Nemo.QQMatrix}
    weighted_search::_WeightedSubspaceSearchState
    individual_subspace_failures::Set{Any}
    numeric_weighted_failures::Dict{Any,Any}
    exact_weighted_failures::Set{Any}
end

struct _TentativeFaceDirection{F<:AbstractFloat}
    block_index::Int
    direction::Vector{ExactRational}
    eigenvalue::F
    residual::F
    score::F
end

function _FacialReductionExactCache(problem::ProblemData)
    cache = _FacialReductionExactCache(
        problem,
        nothing,
        Union{Nothing,Vector{Vector{ExactRational}}}[nothing for _ in problem.blocks],
        nothing,
        _WeightedSubspaceSearchState(Dict{Int,Int}(), BigInt(0), 0),
        Set{Any}(),
        Dict{Any,Any}(),
        Set{Any}(),
    )
    _record_approximate_cache_memory!(:facial_reduction, cache)
    return cache
end

function _facial_reduction_eigen(opt::Optimizer, matrix::AbstractMatrix{F}) where {F<:AbstractFloat}
    start_time = time_ns()
    result = eigen(Symmetric((matrix + transpose(matrix)) / 2))
    _record_psd_eigendecomposition!(size(matrix, 1), (time_ns() - start_time) / 1.0e9)
    return result
end

function _facial_reduction_eigvals(opt::Optimizer, matrix::AbstractMatrix{F}) where {F<:AbstractFloat}
    start_time = time_ns()
    result = eigvals(Symmetric((matrix + transpose(matrix)) / 2))
    _record_psd_eigendecomposition!(size(matrix, 1), (time_ns() - start_time) / 1.0e9)
    return result
end

function _facial_reduction_A_transpose!(
    cache::_FacialReductionExactCache,
    problem::ProblemData,
)
    cache.problem === problem || error("Facial-reduction exact cache belongs to another problem.")
    if cache.A_transpose === nothing
        cache.A_transpose = transpose(_to_nemo_matrix(problem.A))
        _record_approximate_cache_memory!(:facial_reduction, cache)
    end
    return cache.A_transpose
end

_nemo_direction_column(direction::Vector{ExactRational}) =
    _to_nemo_matrix(reshape(direction, :, 1))

function _nemo_product_nonzero(product)
    for row in axes(product, 1)
        value = product[row, 1]
        iszero(value) || return row, value
    end
    return nothing
end

function _nemo_quadratic_value(matrix, direction_column)
    return _with_nemo_error("exact rational quadratic form") do
        (transpose(direction_column) * matrix * direction_column)[1, 1]
    end
end

function _normalize_rational_direction(direction::Vector{ExactRational})
    nonzero_entries = [entry for entry in direction if !iszero(entry)]
    isempty(nonzero_entries) && return direction

    common_denominator = foldl(lcm, (denominator(entry) for entry in nonzero_entries); init = BigInt(1))
    integer_entries = BigInt[
        numerator(entry) * (common_denominator ÷ denominator(entry)) for entry in direction
    ]
    common_divisor = foldl(gcd, (abs(entry) for entry in integer_entries if !iszero(entry)); init = BigInt(0))
    common_divisor == 0 && return direction
    integer_entries ./= common_divisor
    first_nonzero = findfirst(!iszero, integer_entries)
    if first_nonzero !== nothing && integer_entries[first_nonzero] < 0
        integer_entries .*= -1
    end
    return ExactRational[entry // 1 for entry in integer_entries]
end

function _projective_rational_direction(
    candidate::Vector{F},
    tolerance::F,
) where {F<:AbstractFloat}
    isempty(candidate) && return ExactRational[]
    all(isfinite, candidate) || return ExactRational[]

    # Eigenvectors are arbitrarily scaled, and their unit normalization is
    # usually irrational even when the exposed line is rational.  Recover the
    # line from coordinate ratios instead of rationalizing that common scale.
    pivot = argmax(index -> abs(candidate[index]), eachindex(candidate))
    pivot_value = candidate[pivot]
    iszero(pivot_value) && return ExactRational[]

    direction = zeros(ExactRational, length(candidate))
    direction[pivot] = 1 // 1
    for index in eachindex(candidate)
        index == pivot && continue
        direction[index] = rationalize(
            BigInt,
            BigFloat(candidate[index] / pivot_value);
            tol = BigFloat(tolerance),
        )
    end
    return direction
end

function _format_exact_direction(
    direction::Vector{ExactRational};
    max_entries::Int = 64,
)
    support = findall(!iszero, direction)
    isempty(support) && return "support=0/$(length(direction)), entries=[]"

    display_count = min(length(support), max_entries)
    entries = String[]
    for index in support[1:display_count]
        push!(entries, "$(index)=>$(_format_exact_rational_compact(direction[index]))")
    end
    suffix = length(support) > display_count ? ", ..." : ""
    return "support=$(length(support))/$(length(direction)), entries=[" *
           join(entries, ", ") *
           suffix *
           "]"
end

function _facial_reduction_row_space_is_small(
    problem::ProblemData,
    settings::Settings = Settings(),
)
    row_count, variable_count = size(problem.A)
    return BigInt(row_count) * (variable_count + 1) <=
           settings.facial_reduction_row_space_max_entries
end

function _weighted_subspace_exposure_work(
    problem::ProblemData,
    block::BlockStructure,
    rank::Int,
)
    rank > 0 || return (
        affine_dimension = 0,
        block_entries = 0,
        weight_dimension = 0,
        form_entries = BigInt(0),
        affine_products = BigInt(0),
    )
    affine_dimension = problem.affine === nothing ? 0 : 1 + size(problem.affine[2], 2)
    block_entries = length(block.local_positions)
    weight_dimension = div(rank * (rank + 1), 2)
    form_entries = BigInt(block_entries) * weight_dimension
    affine_products = BigInt(affine_dimension) * form_entries
    return (
        affine_dimension = affine_dimension,
        block_entries = block_entries,
        weight_dimension = weight_dimension,
        form_entries = form_entries,
        affine_products = affine_products,
    )
end

function _weighted_subspace_exposure_is_small(work, settings::Settings = Settings())
    return work.form_entries <=
           settings.facial_reduction_weighted_subspace_max_form_entries &&
           work.affine_products <=
           settings.facial_reduction_weighted_subspace_max_affine_products
end

function _weighted_subspace_exposure_is_cheap(work, settings::Settings = Settings())
    return work.weight_dimension <=
           settings.facial_reduction_cheap_weighted_subspace_max_weight_dimension &&
           work.form_entries <=
           settings.facial_reduction_cheap_weighted_subspace_max_form_entries &&
           work.affine_products <=
           settings.facial_reduction_cheap_weighted_subspace_max_affine_products
end

function _numeric_weighted_subspace_exposure_is_small(work, settings::Settings = Settings())
    return work.form_entries <=
           settings.facial_reduction_numeric_weighted_subspace_max_form_entries &&
           work.affine_products <=
           settings.facial_reduction_numeric_weighted_subspace_max_affine_products
end

function _individual_subspace_certificate_work(
    problem::ProblemData,
    block::BlockStructure,
    direction_count::Int,
)
    affine_dimension = problem.affine === nothing ? 0 : 1 + size(problem.affine[2], 2)
    # Across all annihilation rows for one direction, each packed block entry
    # is inspected at most twice.  This is a conservative estimate of the
    # exact affine products performed by the individual-certification loop.
    affine_products =
        BigInt(2) * direction_count * affine_dimension * length(block.local_positions)
    return (affine_dimension = affine_dimension, affine_products = affine_products)
end

_individual_subspace_certificate_is_small(work, settings::Settings = Settings()) =
    work.affine_products <= settings.facial_reduction_individual_max_affine_products

function _facial_reduction_row_space!(
    cache::_FacialReductionExactCache,
    problem::ProblemData,
)
    cache.problem === problem || error("Facial-reduction exact cache belongs to another problem.")
    cached = cache.row_space
    cached === nothing || return cached

    row_count, variable_count = size(problem.A)
    C = zeros(ExactRational, variable_count + 1, row_count)
    row_count > 0 && (C[1:variable_count, :] = transpose(problem.A))
    row_count > 0 && (C[variable_count + 1, :] = transpose(problem.b))
    if row_count == 0
        result = _FacialReductionRowSpaceCache(Int[], Int[], zeros(ExactRational, 0, 0))
        cache.row_space = result
        return result
    end

    reduced = _from_nemo_matrix(_exact_rref(C))
    pivot_columns = Int[]
    for row in axes(reduced, 1)
        pivot = findfirst(column -> !iszero(reduced[row, column]), axes(reduced, 2))
        pivot === nothing || push!(pivot_columns, pivot)
    end
    rank = length(pivot_columns)
    if rank == 0
        result = _FacialReductionRowSpaceCache(Int[], Int[], zeros(ExactRational, 0, 0))
        cache.row_space = result
        return result
    end

    basis = C[:, pivot_columns]
    _, selected_rows = _rref_row_pivots(transpose(basis))
    selected_rows = selected_rows[1:rank]
    square = basis[selected_rows, :]
    inverse = _from_nemo_matrix(
        _exact_rref(hcat(square, Matrix{ExactRational}(I, rank, rank)))[:, rank + 1:(2 * rank)],
    )
    result = _FacialReductionRowSpaceCache(pivot_columns, selected_rows, inverse)
    cache.row_space = result
    _record_approximate_cache_memory!(:facial_reduction, result)
    return result
end

function _row_space_multiplier(
    problem::ProblemData,
    indices::Vector{Int},
    values::Vector{ExactRational};
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    settings::Settings = Settings(),
)
    length(indices) == length(values) || error("Sparse row indices and values must have equal lengths.")
    # Exact affine vanishing is sufficient to certify the forms used in
    # facial reduction.  Avoid materializing a dense rational row-space RREF
    # merely to obtain an optional multiplier on large systems.
    _facial_reduction_row_space_is_small(problem, settings) || return nothing
    row_space = _facial_reduction_row_space!(cache, problem)
    rank = length(row_space.pivot_columns)
    rank == 0 && return isempty(indices) ? zeros(ExactRational, size(problem.A, 1)) : nothing
    variable_count = size(problem.A, 2)
    rhs_selected = zeros(ExactRational, rank)
    for (index, value) in zip(indices, values)
        1 <= index <= variable_count || error("Sparse row index is out of bounds.")
        for (selected_index, row_index) in enumerate(row_space.selected_rows)
            row_index == index && (rhs_selected[selected_index] += value)
        end
    end
    coefficients = row_space.inverse * rhs_selected
    multiplier = zeros(ExactRational, size(problem.A, 1))
    multiplier[row_space.pivot_columns] = coefficients

    lhs = zeros(ExactRational, variable_count)
    for (equation_index, coefficient) in zip(row_space.pivot_columns, coefficients)
        iszero(coefficient) || (lhs .+= coefficient .* vec(problem.A[equation_index, :]))
    end
    all(iszero, lhs[setdiff(collect(1:variable_count), indices)]) || return nothing
    for (index, value) in zip(indices, values)
        lhs[index] == value || return nothing
    end
    iszero(dot(problem.b, multiplier)) || return nothing
    return multiplier
end

function _block_annihilation_forms(block::BlockStructure, direction::Vector{ExactRational})
    length(direction) == block.size || error("PSD kernel direction has the wrong dimension.")
    forms = Tuple{Vector{Int},Vector{ExactRational}}[]
    for row_index in 1:block.size
        coefficients = Dict{Int,ExactRational}()
        for (local_index, (i, j)) in enumerate(block.local_positions)
            coefficient = if i == j == row_index
                direction[j]
            elseif i != j && i == row_index
                direction[j]
            elseif i != j && j == row_index
                direction[i]
            else
                zero(ExactRational)
            end
            iszero(coefficient) || (coefficients[block.global_positions[local_index]] = coefficient)
        end
        indices = sort(collect(keys(coefficients)))
        push!(forms, (indices, [coefficients[index] for index in indices]))
    end
    return forms
end

function _block_quadratic_form(block::BlockStructure, direction::Vector{ExactRational})
    length(direction) == block.size || error("PSD kernel direction has the wrong dimension.")
    coefficients = Dict{Int,ExactRational}()
    for (local_index, (i, j)) in enumerate(block.local_positions)
        coefficient = i == j ? direction[i]^2 : 2 * direction[i] * direction[j]
        iszero(coefficient) || (coefficients[block.global_positions[local_index]] = coefficient)
    end
    indices = sort(collect(keys(coefficients)))
    return indices, [coefficients[index] for index in indices]
end

function _affine_form_violation(
    problem::ProblemData,
    indices::Vector{Int},
    values::Vector{ExactRational},
)
    problem.affine === nothing && return "no exact affine parametrization is available"
    particular, nullspace = problem.affine
    value = sum((particular[index] * coefficient for (index, coefficient) in zip(indices, values)); init = zero(ExactRational))
    iszero(value) || return "particular value=$(_format_exact_rational_compact(value))"
    for column in axes(nullspace, 2)
        value = sum(
            (nullspace[index, column] * coefficient for (index, coefficient) in zip(indices, values));
            init = zero(ExactRational),
        )
        iszero(value) || return "affine_basis=$(column), value=$(_format_exact_rational_compact(value))"
    end
    return nothing
end

function _block_annihilation_violation(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    block_index::Int = something(findfirst(==(block), problem.blocks)),
)
    start_time = time_ns()
    try
        problem.affine === nothing && return "no exact affine parametrization is available"
        for (row_index, (indices, values)) in enumerate(_block_annihilation_forms(block, direction))
            violation = _affine_form_violation(problem, indices, values)
            violation === nothing && continue
            return "row=$(row_index), affine form does not vanish ($(violation))"
        end
        return nothing
    finally
        _record_row_space_check!((time_ns() - start_time) / 1.0e9)
    end
end

function _block_quadratic_vanish_violation(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    block_index::Int = something(findfirst(==(block), problem.blocks)),
)
    start_time = time_ns()
    try
        problem.affine === nothing && return "no exact affine parametrization is available"
        indices, values = _block_quadratic_form(block, direction)
        violation = _affine_form_violation(problem, indices, values)
        violation === nothing && return nothing
        return "quadratic affine form does not vanish ($(violation))"
    finally
        _record_certificate_check!((time_ns() - start_time) / 1.0e9)
    end
end

function _block_trace_vanish_violation(
    problem::ProblemData,
    block::BlockStructure,
    directions::Vector{Vector{ExactRational}},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    block_index::Int = something(findfirst(==(block), problem.blocks)),
)
    start_time = time_ns()
    try
        isempty(directions) && return "no directions"
        problem.affine === nothing && return "no exact affine parametrization is available"
        coefficients = Dict{Int,ExactRational}()
        for direction in directions
            indices, values = _block_quadratic_form(block, direction)
            for (index, value) in zip(indices, values)
                coefficients[index] = get(coefficients, index, zero(ExactRational)) + value
            end
        end
        filter!(pair -> !iszero(pair.second), coefficients)
        indices = sort(collect(keys(coefficients)))
        values = [coefficients[index] for index in indices]
        violation = _affine_form_violation(problem, indices, values)
        violation === nothing && return nothing
        return "trace affine form does not vanish ($(violation))"
    finally
        _record_certificate_check!((time_ns() - start_time) / 1.0e9)
    end
end

function _block_weighted_subspace_form(
    block::BlockStructure,
    direction_matrix::AbstractMatrix{ExactRational},
    weight_matrix::AbstractMatrix{ExactRational},
)
    exposed_matrix = direction_matrix * weight_matrix * transpose(direction_matrix)
    exposed_indices = Int[]
    exposed_values = ExactRational[]
    for (local_index, (i, j)) in enumerate(block.local_positions)
        value = i == j ? exposed_matrix[i, j] : 2 * exposed_matrix[i, j]
        iszero(value) && continue
        push!(exposed_indices, block.global_positions[local_index])
        push!(exposed_values, value)
    end
    return exposed_indices, exposed_values
end

function _weighted_exposure_directions(
    direction_matrix::AbstractMatrix{ExactRational},
    weight_matrix::Matrix{ExactRational},
)
    size(weight_matrix, 1) == size(weight_matrix, 2) || return Vector{Vector{ExactRational}}()
    size(direction_matrix, 2) == size(weight_matrix, 1) ||
        return Vector{Vector{ExactRational}}()
    augmented = hcat(
        copy(weight_matrix),
        zeros(ExactRational, size(weight_matrix, 1)),
    )
    _, pivot_columns = _rref(augmented)
    return [
        collect(direction_matrix * view(weight_matrix, :, column)) for
        column in pivot_columns
    ]
end

function _finite_rational_directions(
    ::Type{F},
    directions::Vector{Vector{ExactRational}},
) where {F<:AbstractFloat}
    return [
        begin
            numeric_direction = try
                _to_working_array(F, direction)
            catch
                nothing
            end
            if numeric_direction !== nothing && all(isfinite, numeric_direction)
                copy(direction)
            else
                scale = maximum(abs, direction; init = zero(ExactRational))
                iszero(scale) ?
                copy(direction) :
                ExactRational[value / scale for value in direction]
            end
        end for direction in directions
    ]
end

function _finite_exact_columns_to_working_array(
    ::Type{F},
    values::AbstractMatrix{ExactRational},
) where {F<:AbstractFloat}
    converted = try
        _to_working_array(F, values)
    catch
        nothing
    end
    fast_conversion_succeeded = converted !== nothing
    converted === nothing && (converted = Matrix{F}(undef, size(values)...))
    for column in axes(values, 2)
        converted_column = view(converted, :, column)
        if fast_conversion_succeeded && all(isfinite, converted_column)
            continue
        else
            exact_scale =
                maximum(abs, view(values, :, column); init = zero(ExactRational))
            if iszero(exact_scale)
                fill!(converted_column, zero(F))
                continue
            end
            setprecision(BigFloat, max(precision(F), 64)) do
                numeric_scale = BigFloat(exact_scale)
                for row in axes(values, 1)
                    converted[row, column] =
                        F(BigFloat(values[row, column]) / numeric_scale)
                end
            end
        end
    end
    return converted
end

function _numerical_scout_exception_summary(err)
    detail = replace(sprint(showerror, err), r"\s+" => " ")
    isempty(detail) && (detail = string(nameof(typeof(err))))
    return first(detail, min(length(detail), 500))
end

function _numeric_weighted_subspace_exposure_attempt(
    problem::ProblemData,
    block::BlockStructure,
    directions::Vector{Vector{ExactRational}},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    failure(reason; promising::Bool = false, status::Symbol = :unpromising) = (
        exposure = nothing,
        promising = promising,
        status = status,
        reason = reason,
    )
    isempty(directions) &&
        return failure("the candidate subspace is empty"; status = :unavailable)
    problem.affine === nothing &&
        return failure("the affine system is inconsistent"; status = :unavailable)

    scaled_directions = _finite_rational_directions(F, directions)
    direction_matrix = hcat(scaled_directions...)
    size(direction_matrix, 1) == block.size ||
        return failure(
            "the candidate dimension does not match the PSD block";
            status = :unavailable,
        )
    rank = size(direction_matrix, 2)
    work = _weighted_subspace_exposure_work(problem, block, rank)
    _numeric_weighted_subspace_exposure_is_small(work, settings) ||
        return failure(
            "the numerical scout exceeds its per-candidate work limit";
            status = :skipped,
        )

    weight_positions = _triangle_positions(rank)
    stage = "converting and scaling the candidate data"
    try
        numeric_directions = _to_working_array(F, direction_matrix)
        particular, nullspace = problem.affine
        affine_basis = hcat(particular, nullspace)
        numeric_affine_basis = _finite_exact_columns_to_working_array(
            F,
            affine_basis[block.global_positions, :],
        )
        all(isfinite, numeric_directions) && all(isfinite, numeric_affine_basis) ||
            return failure(
                "the scaled numerical input contains nonfinite values";
                status = :unavailable,
            )

        stage = "forming the weighted affine system"
        form_columns = zeros(F, length(block.local_positions), length(weight_positions))
        for (weight_index, (a, b)) in enumerate(weight_positions)
            weight_basis = zeros(F, rank, rank)
            weight_basis[a, b] = one(F)
            weight_basis[b, a] = one(F)
            exposed_matrix = numeric_directions * weight_basis * transpose(numeric_directions)
            for (local_index, (i, j)) in enumerate(block.local_positions)
                form_columns[local_index, weight_index] =
                    i == j ? exposed_matrix[i, j] : 2 * exposed_matrix[i, j]
            end
        end
        all(isfinite, form_columns) ||
            return failure(
                "the weighted form matrix contains nonfinite values";
                status = :unavailable,
            )

        vanish_constraints = transpose(numeric_affine_basis) * form_columns
        all(isfinite, vanish_constraints) ||
            return failure(
                "the weighted affine product contains nonfinite values";
                status = :unavailable,
            )

        stage = "factorizing the weighted affine system"
        singular_factor = svd(vanish_constraints)
        singular_values = singular_factor.S
        isempty(singular_values) &&
            return failure(
                "the weighted affine system has no singular values";
                status = :unavailable,
            )
        all(isfinite, singular_values) ||
            return failure(
                "the weighted affine singular values are nonfinite";
                status = :unavailable,
            )
        scale = max(one(F), maximum(abs, singular_values))
        rank_tolerance = max(sqrt(eps(F)), F(100) * eps(F)) * scale
        constraint_rank = count(value -> value > rank_tolerance, singular_values)
        constraint_rank < length(weight_positions) ||
            return failure("the numerical weighted affine system has no nullspace")
        numeric_weight_subspace =
            transpose(singular_factor.Vt)[:, (constraint_rank + 1):end]
        all(isfinite, numeric_weight_subspace) ||
            return failure(
                "the numerical weight subspace contains nonfinite values";
                status = :unavailable,
            )

        stage = "fitting a positive-semidefinite numerical weight"
        identity_target = F[a == b ? one(F) : zero(F) for (a, b) in weight_positions]
        numeric_coordinates = numeric_weight_subspace \ identity_target
        all(isfinite, numeric_coordinates) ||
            return failure(
                "the numerical weight coordinates are nonfinite";
                status = :unavailable,
            )
        numeric_weight_vector = numeric_weight_subspace * numeric_coordinates
        all(isfinite, numeric_weight_vector) ||
            return failure("the numerical weight is nonfinite"; status = :unavailable)

        numeric_weight_matrix = zeros(F, rank, rank)
        for (weight_index, (a, b)) in enumerate(weight_positions)
            numeric_weight_matrix[a, b] = numeric_weight_vector[weight_index]
            numeric_weight_matrix[b, a] = numeric_weight_vector[weight_index]
        end
        all(isfinite, numeric_weight_matrix) ||
            return failure(
                "the fitted numerical weight matrix is nonfinite";
                status = :unavailable,
            )
        stage = "checking the numerical weight eigenvalues"
        weight_eigenvalues = eigvals(Symmetric(numeric_weight_matrix))
        all(isfinite, weight_eigenvalues) ||
            return failure(
                "the fitted numerical weight eigenvalues are nonfinite";
                status = :unavailable,
            )
        minimum_weight_eigenvalue = minimum(weight_eigenvalues)
        maximum_weight_eigenvalue = maximum(weight_eigenvalues)
        promising =
            minimum_weight_eigenvalue >= -rank_tolerance &&
            maximum_weight_eigenvalue > rank_tolerance

        stage = "rationalizing the numerical weight"
        for tolerance in _facial_reduction_subspace_tolerances(settings, F)
            weight_matrix = zeros(ExactRational, rank, rank)
            for (weight_index, (a, b)) in enumerate(weight_positions)
                value = _rationalize_float(numeric_weight_vector[weight_index], tolerance)
                weight_matrix[a, b] = value
                weight_matrix[b, a] = value
            end
            _positive_semidefinite_exact(weight_matrix) || continue
            exposure_directions =
                _weighted_exposure_directions(direction_matrix, weight_matrix)
            isempty(exposure_directions) && continue
            exposed_indices, exposed_values =
                _block_weighted_subspace_form(block, direction_matrix, weight_matrix)
            _affine_form_violation(problem, exposed_indices, exposed_values) === nothing ||
                continue
            return (
                exposure = (
                    weight = weight_matrix,
                    directions = exposure_directions,
                    tolerance = tolerance,
                ),
                promising = true,
                status = :certified,
                reason = "certified",
            )
        end
        return failure(
            promising ? "a nonzero positive-semidefinite numerical weight did not rationalize to an exact certificate" :
            "the fitted numerical weight was not nonzero positive semidefinite";
            promising,
            status = promising ? :promising : :unpromising,
        )
    catch err
        detail = _numerical_scout_exception_summary(err)
        return failure(
            "the numerical scout was unavailable during $(stage): $(detail)";
            status = :unavailable,
        )
    end
end

function _numeric_weighted_subspace_exposure(
    problem::ProblemData,
    block::BlockStructure,
    directions::Vector{Vector{ExactRational}},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    return _numeric_weighted_subspace_exposure_attempt(
        problem,
        block,
        directions,
        settings,
        F,
    ).exposure
end

function _block_weighted_subspace_exposure(
    problem::ProblemData,
    block::BlockStructure,
    directions::Vector{Vector{ExactRational}},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    isempty(directions) && return nothing
    problem.affine === nothing && return nothing

    direction_matrix = hcat(directions...)
    size(direction_matrix, 1) == block.size || return nothing
    rank = size(direction_matrix, 2)
    work = _weighted_subspace_exposure_work(problem, block, rank)
    _weighted_subspace_exposure_is_small(work, settings) || return nothing
    weight_positions = _triangle_positions(rank)
    weight_dimension = length(weight_positions)
    form_columns = zeros(ExactRational, length(block.local_positions), weight_dimension)

    for (weight_index, (a, b)) in enumerate(weight_positions)
        weight_basis = zeros(ExactRational, rank, rank)
        weight_basis[a, b] = 1 // 1
        weight_basis[b, a] = 1 // 1
        exposed_matrix = direction_matrix * weight_basis * transpose(direction_matrix)
        for (local_index, (i, j)) in enumerate(block.local_positions)
            form_columns[local_index, weight_index] =
                i == j ? exposed_matrix[i, j] : 2 * exposed_matrix[i, j]
        end
    end

    particular, nullspace = problem.affine
    affine_basis = hcat(particular, nullspace)
    block_affine_basis = affine_basis[block.global_positions, :]
    vanish_constraints = transpose(block_affine_basis) * form_columns
    weight_subspace = _nullspace_basis_exact(vanish_constraints)
    size(weight_subspace, 2) == 0 && return nothing

    identity_target = ExactRational[
        a == b ? 1 // 1 : 0 // 1 for (a, b) in weight_positions
    ]
    numeric_weight_subspace = _to_working_array(F, weight_subspace)
    numeric_coordinates = try
        numeric_weight_subspace \ _to_working_array(F, identity_target)
    catch
        return nothing
    end
    all(isfinite, numeric_coordinates) || return nothing

    for tolerance in _facial_reduction_subspace_tolerances(settings, F)
        rational_coordinates = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for
            value in numeric_coordinates
        ]
        weight_vector = weight_subspace * rational_coordinates
        weight_matrix = zeros(ExactRational, rank, rank)
        for (weight_index, (a, b)) in enumerate(weight_positions)
            weight_matrix[a, b] = weight_vector[weight_index]
            weight_matrix[b, a] = weight_vector[weight_index]
        end
        _positive_semidefinite_exact(weight_matrix) || continue
        exposure_directions =
            _weighted_exposure_directions(direction_matrix, weight_matrix)
        isempty(exposure_directions) && continue

        exposed_indices, exposed_values =
            _block_weighted_subspace_form(block, direction_matrix, weight_matrix)
        _affine_form_violation(problem, exposed_indices, exposed_values) === nothing ||
            continue
        return (
            weight = weight_matrix,
            directions = exposure_directions,
            tolerance = tolerance,
        )
    end

    return nothing
end

function _multiblock_weighted_subspace_exposure(
    problem::ProblemData,
    directions_by_block::Dict{Int,Vector{Vector{ExactRational}}},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    isempty(directions_by_block) && return nothing
    problem.affine === nothing && return nothing

    block_indices = sort(collect(keys(directions_by_block)))
    direction_matrices = Dict{Int,Matrix{ExactRational}}()
    weight_positions_by_block = Dict{Int,Vector{Tuple{Int,Int}}}()
    total_weight_dimension = 0
    total_form_entries = BigInt(0)
    for block_index in block_indices
        1 <= block_index <= length(problem.blocks) || return nothing
        directions = _linearly_independent_directions(directions_by_block[block_index])
        isempty(directions) && return nothing
        direction_matrix = hcat(directions...)
        block = problem.blocks[block_index]
        size(direction_matrix, 1) == block.size || return nothing
        direction_matrices[block_index] = direction_matrix
        weight_positions = _triangle_positions(size(direction_matrix, 2))
        weight_positions_by_block[block_index] = weight_positions
        total_weight_dimension += length(weight_positions)
        total_form_entries += BigInt(length(block.local_positions)) * length(weight_positions)
    end

    affine_dimension = 1 + size(problem.affine[2], 2)
    total_form_entries <= settings.facial_reduction_weighted_subspace_max_form_entries ||
        return nothing
    BigInt(affine_dimension) * total_form_entries <=
        settings.facial_reduction_weighted_subspace_max_affine_products || return nothing

    form_columns = zeros(
        ExactRational,
        length(problem.objective_vector_raw),
        total_weight_dimension,
    )
    identity_target = zeros(ExactRational, total_weight_dimension)
    weight_ranges = Dict{Int,UnitRange{Int}}()
    next_weight = 1
    for block_index in block_indices
        block = problem.blocks[block_index]
        direction_matrix = direction_matrices[block_index]
        weight_positions = weight_positions_by_block[block_index]
        weight_range = next_weight:(next_weight + length(weight_positions) - 1)
        weight_ranges[block_index] = weight_range
        for (local_weight_index, (a, b)) in enumerate(weight_positions)
            global_weight_index = weight_range[local_weight_index]
            weight_basis = zeros(ExactRational, size(direction_matrix, 2), size(direction_matrix, 2))
            weight_basis[a, b] = 1 // 1
            weight_basis[b, a] = 1 // 1
            indices, values =
                _block_weighted_subspace_form(block, direction_matrix, weight_basis)
            form_columns[indices, global_weight_index] = values
            a == b && (identity_target[global_weight_index] = 1 // 1)
        end
        next_weight = last(weight_range) + 1
    end

    particular, nullspace = problem.affine
    affine_basis = hcat(particular, nullspace)
    vanish_constraints = transpose(affine_basis) * form_columns
    weight_subspace = _nullspace_basis_exact(vanish_constraints)
    size(weight_subspace, 2) == 0 && return nothing

    numeric_weight_subspace = _to_working_array(F, weight_subspace)
    numeric_coordinates = try
        numeric_weight_subspace \ _to_working_array(F, identity_target)
    catch
        return nothing
    end
    all(isfinite, numeric_coordinates) || return nothing

    for tolerance in _facial_reduction_subspace_tolerances(settings, F)
        rational_coordinates = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for
            value in numeric_coordinates
        ]
        weight_vector = weight_subspace * rational_coordinates
        weights = Dict{Int,Matrix{ExactRational}}()
        all_positive_definite = true
        for block_index in block_indices
            weight_positions = weight_positions_by_block[block_index]
            weight_range = weight_ranges[block_index]
            rank = size(direction_matrices[block_index], 2)
            weight_matrix = zeros(ExactRational, rank, rank)
            for (local_weight_index, (a, b)) in enumerate(weight_positions)
                value = weight_vector[weight_range[local_weight_index]]
                weight_matrix[a, b] = value
                weight_matrix[b, a] = value
            end
            if !_positive_definite_exact(weight_matrix)
                all_positive_definite = false
                break
            end
            weights[block_index] = weight_matrix
        end
        all_positive_definite || continue
        all(iszero, transpose(affine_basis) * (form_columns * weight_vector)) || continue
        return (weights = weights, tolerance = tolerance)
    end

    return nothing
end

function _block_face_direction_certificate(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    block_index::Int = something(findfirst(==(block), problem.blocks)),
)
    start_time = time_ns()
    try
        row_violation = _block_annihilation_violation(
            problem,
            block,
            direction;
            cache,
            block_index,
        )
        row_violation === nothing && return (kind = :affine_rows, violation = nothing)

        diagonal_violation = _block_quadratic_vanish_violation(
            problem,
            block,
            direction;
            cache,
            block_index,
        )
        diagonal_violation === nothing && return (kind = :psd_diagonal, violation = nothing)

        return (
            kind = :none,
            violation =
                "row certificate failed ($(row_violation)); PSD diagonal certificate failed ($(diagonal_violation))",
        )
    finally
        _record_certificate_check!((time_ns() - start_time) / 1.0e9)
    end
end

function _exact_face_direction(
    problem::ProblemData,
    block::BlockStructure,
    candidate::Vector{F},
    settings::Settings,
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    block_index::Int = something(findfirst(==(block), problem.blocks)),
) where {F<:AbstractFloat}
    for tolerance in _recovery_tolerances(settings, F)
        direction = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for value in candidate
        ]
        direction = _normalize_rational_direction(direction)
        any(!iszero, direction) || continue
        if _block_face_direction_certificate(
            problem,
            block,
            direction;
            cache,
            block_index,
        ).kind != :none
            return direction
        end
    end
    return nothing
end

function _heuristic_kernel_direction(
    block_matrix::Matrix{F},
    candidate::Vector{F},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    matrix_scale = max(one(F), maximum(abs, block_matrix))
    residual_tolerance = max(
        F(1.0e-10),
        sqrt(eps(F)),
        F(100) * _to_working_float(F, settings.facial_reduction_exposure_tolerance),
    ) * matrix_scale
    tolerances = _recovery_tolerances(settings, F)
    coarse_tolerance = max(
        F(1.0e-4),
        _to_working_float(F, settings.facial_reduction_exposure_tolerance),
    )
    if isempty(tolerances) || coarse_tolerance > first(tolerances)
        pushfirst!(tolerances, coarse_tolerance)
    end
    for tolerance in tolerances
        raw_direction = _projective_rational_direction(candidate, tolerance)
        any(!iszero, raw_direction) || continue
        numeric_direction = _to_working_array(F, raw_direction)
        direction_scale = max(one(F), _max_abs(numeric_direction))
        residual = _max_abs(block_matrix * numeric_direction) / direction_scale
        if residual <= residual_tolerance
            return (
                direction = _normalize_rational_direction(raw_direction),
                residual = residual,
                tolerance = tolerance,
                residual_tolerance = residual_tolerance,
            )
        end
    end
    return nothing
end

function _rational_subspace_pivot_charts(
    subspace::AbstractMatrix{F},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    dimension, column_count = size(subspace)
    (dimension == 0 || column_count == 0) && return (rank = 0, charts = Vector{Vector{Int}}())
    all(isfinite, subspace) || return (rank = 0, charts = Vector{Vector{Int}}())

    subspace_matrix = Matrix{F}(subspace)
    row_space_matrix = Matrix(transpose(subspace_matrix))
    qr_factor = qr(row_space_matrix, ColumnNorm())
    diagonal = abs.(diag(qr_factor.R))
    isempty(diagonal) && return (rank = 0, charts = Vector{Vector{Int}}())

    scale = max(one(F), maximum(abs, subspace_matrix), maximum(diagonal))
    rank_tolerance = max(
        _to_working_float(F, settings.facial_reduction_rank_tolerance),
        F(max(size(row_space_matrix)...)) * eps(F) * scale,
        F(100) * eps(F),
    )
    rank = count(value -> value > rank_tolerance, diagonal)
    rank == 0 && return (rank = 0, charts = Vector{Vector{Int}}())

    pivot_order = collect(qr_factor.p)
    pivot_indices = pivot_order[1:rank]
    charts = Vector{Vector{Int}}([copy(pivot_indices)])
    chart_keys = Set{Tuple{Vararg{Int}}}([Tuple(sort(pivot_indices))])
    column_norms = [norm(view(row_space_matrix, :, column)) for column in axes(row_space_matrix, 2)]
    normalized_reversed = copy(row_space_matrix[:, end:-1:1])
    for column in axes(normalized_reversed, 2)
        original_column = dimension - column + 1
        column_norms[original_column] > zero(F) &&
            (normalized_reversed[:, column] ./= column_norms[original_column])
    end
    normalized_qr = qr(normalized_reversed, ColumnNorm())
    normalized_chart = [dimension - index + 1 for index in normalized_qr.p[1:rank]]
    normalized_key = Tuple(sort(normalized_chart))
    normalized_singular_values = svdvals(row_space_matrix[:, normalized_chart])
    if length(charts) < settings.facial_reduction_subspace_max_charts &&
       !(normalized_key in chart_keys) &&
       length(normalized_singular_values) == rank &&
       minimum(normalized_singular_values) > rank_tolerance
        push!(charts, normalized_chart)
        push!(chart_keys, normalized_key)
    end
    alternatives = Tuple{F,Vector{Int}}[]
    for pivot_offset in 1:rank
        for replacement in pivot_order[(rank + 1):end]
            trial = copy(pivot_indices)
            trial[pivot_offset] = replacement
            key = Tuple(sort(trial))
            key in chart_keys && continue
            chart_matrix = row_space_matrix[:, trial]
            singular_values = svdvals(chart_matrix)
            length(singular_values) == rank || continue
            minimum(singular_values) > rank_tolerance || continue
            condition_number = maximum(singular_values) / minimum(singular_values)
            push!(alternatives, (condition_number, trial))
            push!(chart_keys, key)
        end
    end
    sort!(alternatives; by = item -> (item[1], Tuple(item[2])))
    for (_, chart) in alternatives
        length(charts) >= settings.facial_reduction_subspace_max_charts && break
        push!(charts, chart)
    end
    return (rank = rank, charts = charts)
end

function _pivoted_rational_subspace_directions(
    subspace::AbstractMatrix{F},
    settings::Settings,
    ::Type{F};
    relation_tolerance = nothing,
    pivot_indices::Union{Nothing,Vector{Int}} = nothing,
) where {F<:AbstractFloat}
    dimension, column_count = size(subspace)
    (dimension == 0 || column_count == 0) && return Vector{ExactRational}[]
    all(isfinite, subspace) || return Vector{ExactRational}[]

    chart_data = _rational_subspace_pivot_charts(subspace, settings, F)
    rank = chart_data.rank
    rank == 0 && return Vector{ExactRational}[]
    chosen_pivots = pivot_indices === nothing ? first(chart_data.charts) : pivot_indices
    length(chosen_pivots) == rank || return Vector{ExactRational}[]

    subspace_matrix = Matrix{F}(subspace)
    row_space_matrix = Matrix(transpose(subspace_matrix))
    pivot_set = Set(chosen_pivots)
    remaining_indices = [index for index in 1:dimension if !(index in pivot_set)]

    relations = if isempty(remaining_indices)
        zeros(F, rank, 0)
    else
        row_space_matrix[:, chosen_pivots] \ row_space_matrix[:, remaining_indices]
    end

    tolerances = _facial_reduction_subspace_tolerances(
        settings,
        F;
        relation_tolerance,
    )

    for tolerance in tolerances
        rational_relations = Matrix{ExactRational}(undef, size(relations)...)
        for index in eachindex(relations)
            rational_relations[index] =
                rationalize(BigInt, BigFloat(relations[index]); tol = BigFloat(tolerance))
        end

        directions = Vector{Vector{ExactRational}}()
        for pivot_offset in 1:rank
            direction = zeros(ExactRational, dimension)
            direction[chosen_pivots[pivot_offset]] = 1 // 1
            for (remaining_offset, remaining_index) in enumerate(remaining_indices)
                direction[remaining_index] = rational_relations[pivot_offset, remaining_offset]
            end
            direction = _normalize_rational_direction(direction)
            any(!iszero, direction) || continue
            push!(directions, direction)
        end

        directions = _linearly_independent_directions(directions)
        isempty(directions) || return directions
    end

    return Vector{ExactRational}[]
end

function _canonical_rational_subspace_key(directions::Vector{Vector{ExactRational}})
    isempty(directions) && return ()
    row_basis = Matrix(transpose(hcat(directions...)))
    reduced, _ = _rref(hcat(row_basis, zeros(ExactRational, size(row_basis, 1))))
    return Tuple(vec(reduced[:, 1:size(row_basis, 2)]))
end

function _rational_subspace_coefficient_bits(key)
    maximum_bits = 0
    for value in key
        value isa ExactRational || continue
        maximum_bits = max(
            maximum_bits,
            ndigits(abs(numerator(value)); base = 2),
            ndigits(denominator(value); base = 2),
        )
    end
    return maximum_bits
end

function _stable_rational_subspace_fingerprint(key)
    state = UInt64(0xcbf29ce484222325)
    for byte in codeunits(string(key))
        state = xor(state, UInt64(byte)) * UInt64(0x100000001b3)
    end
    return first(lpad(string(state; base = 16), 16, '0'), 8)
end

function _numeric_subspace_projector(
    subspace::AbstractMatrix{F},
    rank::Int,
) where {F<:AbstractFloat}
    rank > 0 || return zeros(F, size(subspace, 1), size(subspace, 1))
    orthogonal_basis = Matrix(qr(Matrix{F}(subspace)).Q[:, 1:rank])
    return orthogonal_basis * transpose(orthogonal_basis)
end

function _rational_subspace_candidate_metrics(
    subspace::AbstractMatrix{F},
    directions::Vector{Vector{ExactRational}},
    key,
    ::Type{F},
) where {F<:AbstractFloat}
    coefficient_bits = _rational_subspace_coefficient_bits(key)
    fingerprint = _stable_rational_subspace_fingerprint(key)
    try
        all(isfinite, subspace) || return (
            projector = nothing,
            reconstruction_error = nothing,
            coefficient_bits = coefficient_bits,
            fingerprint = fingerprint,
            numeric_usable = false,
            numeric_issue = "the source subspace contains nonfinite values",
        )
        scaled_directions = _finite_rational_directions(F, directions)
        direction_matrix = _to_working_array(F, hcat(scaled_directions...))
        all(isfinite, direction_matrix) || return (
            projector = nothing,
            reconstruction_error = nothing,
            coefficient_bits = coefficient_bits,
            fingerprint = fingerprint,
            numeric_usable = false,
            numeric_issue = "the scaled rational directions contain nonfinite values",
        )
        rank = size(direction_matrix, 2)
        projector = _numeric_subspace_projector(direction_matrix, rank)
        target_projector = _numeric_subspace_projector(subspace, rank)
        reconstruction_error =
            maximum(abs, projector - target_projector; init = zero(F))
        all(isfinite, projector) &&
            all(isfinite, target_projector) &&
            isfinite(reconstruction_error) || return (
            projector = nothing,
            reconstruction_error = nothing,
            coefficient_bits = coefficient_bits,
            fingerprint = fingerprint,
            numeric_usable = false,
            numeric_issue = "the reconstructed projector metric is nonfinite",
        )
        return (
            projector = projector,
            reconstruction_error = reconstruction_error,
            coefficient_bits = coefficient_bits,
            fingerprint = fingerprint,
            numeric_usable = true,
            numeric_issue = "",
        )
    catch err
        return (
            projector = nothing,
            reconstruction_error = nothing,
            coefficient_bits = coefficient_bits,
            fingerprint = fingerprint,
            numeric_usable = false,
            numeric_issue = "projector metric failed: $(_numerical_scout_exception_summary(err))",
        )
    end
end

function _rational_projector_subspace_directions(
    subspace::AbstractMatrix{F},
    rank::Int,
    tolerance::F,
) where {F<:AbstractFloat}
    rank > 0 || return Vector{ExactRational}[]
    size(subspace, 1) >= rank || return Vector{ExactRational}[]
    orthogonal_basis = try
        Matrix(qr(Matrix{F}(subspace)).Q[:, 1:rank])
    catch
        return Vector{ExactRational}[]
    end
    numeric_projector = orthogonal_basis * transpose(orthogonal_basis)
    dimension = size(numeric_projector, 1)
    projector = zeros(ExactRational, dimension, dimension)
    for j in 1:dimension
        for i in 1:j
            numeric_value = (numeric_projector[i, j] + numeric_projector[j, i]) / 2
            value = rationalize(BigInt, BigFloat(numeric_value); tol = BigFloat(tolerance))
            projector[i, j] = value
            projector[j, i] = value
        end
    end
    projector * projector == projector || return Vector{ExactRational}[]
    directions = [
        _normalize_rational_direction(collect(view(projector, :, column))) for
        column in axes(projector, 2) if any(!iszero, view(projector, :, column))
    ]
    directions = _linearly_independent_directions(directions)
    length(directions) == rank || return Vector{ExactRational}[]
    return directions
end

function _rational_subspace_candidate_sets(
    subspace::AbstractMatrix{F},
    settings::Settings,
    ::Type{F},
    tolerance::F,
) where {F<:AbstractFloat}
    chart_data = _rational_subspace_pivot_charts(subspace, settings, F)
    chart_data.rank == 0 && return NamedTuple[]
    candidate_sets = NamedTuple[]
    seen = Set{Any}()
    for (chart_index, pivots) in enumerate(chart_data.charts)
        _record_facial_reduction_event!(:rational_subspace_charts_attempted)
        directions = _pivoted_rational_subspace_directions(
            subspace,
            settings,
            F;
            relation_tolerance = tolerance,
            pivot_indices = pivots,
        )
        isempty(directions) && continue
        key = _canonical_rational_subspace_key(directions)
        key in seen && continue
        push!(seen, key)
        metrics = _rational_subspace_candidate_metrics(subspace, directions, key, F)
        push!(
            candidate_sets,
            (
                directions = directions,
                method = "pivot chart $(chart_index)",
                key = key,
                metrics...,
            ),
        )
    end
    if settings.facial_reduction_projector_recovery
        _record_facial_reduction_event!(:rational_projectors_attempted)
        directions = _rational_projector_subspace_directions(
            subspace,
            chart_data.rank,
            tolerance,
        )
        if !isempty(directions)
            key = _canonical_rational_subspace_key(directions)
            if !(key in seen)
                metrics = _rational_subspace_candidate_metrics(subspace, directions, key, F)
                push!(
                    candidate_sets,
                    (
                        directions = directions,
                        method = "rational projector",
                        key = key,
                        metrics...,
                    ),
                )
            end
        end
    end
    return candidate_sets
end

function _weighted_subspace_candidate_schedule(
    candidates::Vector,
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    isempty(candidates) &&
        return (
            candidates = candidates,
            cluster_count = 0,
            rejected_count = 0,
            reserved_candidate_keys = Any[],
        )
    usable_candidates = filter(candidates) do candidate
        hasproperty(candidate, :numeric_usable) &&
            candidate.numeric_usable &&
            candidate.projector !== nothing &&
            candidate.reconstruction_error !== nothing &&
            all(isfinite, candidate.projector) &&
            isfinite(candidate.reconstruction_error)
    end
    rejected_count = length(candidates) - length(usable_candidates)
    isempty(usable_candidates) &&
        return (
            candidates = usable_candidates,
            cluster_count = 0,
            rejected_count,
            reserved_candidate_keys = Any[],
        )
    threshold = max(
        _to_working_float(F, settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    function clustered_schedule(ordered)
        clusters = Vector{Vector{Any}}()
        centers = Matrix{F}[]
        for candidate in ordered
            cluster_index = findfirst(
                center ->
                    maximum(
                        abs,
                        candidate.projector - center;
                        init = zero(F),
                    ) <= threshold,
                centers,
            )
            if cluster_index === nothing
                push!(centers, candidate.projector)
                push!(clusters, Any[candidate])
            else
                push!(clusters[cluster_index], candidate)
            end
        end
        scheduled = Any[]
        for depth in 1:maximum(length, clusters)
            for cluster in clusters
                depth <= length(cluster) && push!(scheduled, cluster[depth])
            end
        end
        return scheduled, length(clusters)
    end

    accuracy_order = sort(
        collect(usable_candidates);
        by = candidate -> (
            candidate.reconstruction_error,
            candidate.coefficient_bits,
            -candidate.tolerance,
            candidate.method,
            candidate.fingerprint,
        ),
    )
    simplicity_order = sort(
        collect(usable_candidates);
        by = candidate -> (
            candidate.coefficient_bits,
            candidate.reconstruction_error,
            -candidate.tolerance,
            candidate.method,
            candidate.fingerprint,
        ),
    )
    accuracy_schedule, accuracy_cluster_count =
        clustered_schedule(accuracy_order)
    simplicity_schedule, simplicity_cluster_count =
        clustered_schedule(simplicity_order)

    # Preserve an accuracy-first prefix, then reserve a ladder of distinct
    # low-complexity candidates. An unpromising numerical scout is not enough
    # to reject an exact face, so the ladder is sized to the number of exact
    # fallbacks that can actually be attempted.
    scheduled = Any[]
    seen = Set{Any}()
    reserved_candidate_keys = Any[]
    reserved_count = min(
        length(simplicity_schedule),
        max(
            1,
            min(
                settings.facial_reduction_weighted_exact_without_scout_limit,
                settings.facial_reduction_weighted_max_candidate_sets,
            ),
        ),
    )
    accuracy_prefix = min(
        length(accuracy_schedule),
        max(
            0,
            settings.facial_reduction_weighted_max_candidate_sets -
            reserved_count,
        ),
    )
    for candidate in Iterators.take(accuracy_schedule, accuracy_prefix)
        push!(seen, candidate.key)
        push!(scheduled, candidate)
    end
    for candidate in simplicity_schedule
        length(reserved_candidate_keys) >= reserved_count && break
        candidate.key in seen && continue
        push!(reserved_candidate_keys, candidate.key)
        push!(seen, candidate.key)
        push!(scheduled, candidate)
    end
    for index in eachindex(accuracy_schedule)
        for candidate in (accuracy_schedule[index], simplicity_schedule[index])
            candidate.key in seen && continue
            push!(seen, candidate.key)
            push!(scheduled, candidate)
        end
    end
    return (
        candidates = scheduled,
        cluster_count = max(accuracy_cluster_count, simplicity_cluster_count),
        rejected_count = rejected_count,
        reserved_candidate_keys,
    )
end

function _weighted_subspace_candidate_plan(
    candidates::Vector,
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    schedule = _weighted_subspace_candidate_schedule(candidates, settings, F)
    cheap_exact_only = filter(candidate -> !candidate.numeric_usable, candidates)
    return (
        scheduled = schedule.candidates,
        cheap_exact_only = cheap_exact_only,
        exact_checks = vcat(schedule.candidates, cheap_exact_only),
        cluster_count = schedule.cluster_count,
        rejected_count = schedule.rejected_count,
    )
end

function _facial_reduction_subspace_tolerances(
    settings::Settings,
    ::Type{F};
    relation_tolerance = nothing,
) where {F<:AbstractFloat}
    if relation_tolerance !== nothing
        return F[_to_working_float(F, relation_tolerance)]
    end

    tolerances = _recovery_tolerances(settings, F)
    append!(
        tolerances,
        F[
            F(1.0e-2),
            F(1.0e-3),
            F(1.0e-4),
            F(1.0e-5),
            F(1.0e-6),
            F(1.0e-7),
            _to_working_float(F, settings.facial_reduction_exposure_tolerance),
        ],
    )
    filter!(tolerance -> tolerance > zero(F), tolerances)
    return sort!(unique(tolerances); rev = true)
end

function _weighted_subspace_attempt_key(
    block_index::Int,
    candidate,
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    return (
        block_index,
        candidate.key,
        F,
        F === BigFloat ? precision(BigFloat) : 0,
        settings.facial_reduction_exposure_tolerance,
        settings.facial_reduction_rank_tolerance,
        settings.rational_tolerance,
        settings.recovery_tolerance_shrink,
    )
end

function _reserve_weighted_subspace_work!(
    cache::_FacialReductionExactCache,
    settings::Settings,
    block_index::Int,
    affine_products::Integer;
    new_candidate::Bool,
)
    state = cache.weighted_search
    block_candidate_sets = get(state.candidate_sets_attempted, block_index, 0)
    if new_candidate &&
       block_candidate_sets >= settings.facial_reduction_weighted_max_candidate_sets
        return :candidate_limit
    end
    proposed_total = state.total_affine_products + BigInt(affine_products)
    if proposed_total >
       BigInt(settings.facial_reduction_weighted_max_total_affine_products)
        return :product_limit
    end
    new_candidate &&
        (state.candidate_sets_attempted[block_index] = block_candidate_sets + 1)
    state.total_affine_products = proposed_total
    return :reserved
end

function _weighted_subspace_budget_summary(
    cache::_FacialReductionExactCache,
    settings::Settings,
    block_index::Int,
)
    state = cache.weighted_search
    block_candidate_sets = get(state.candidate_sets_attempted, block_index, 0)
    return "candidate_sets(block $(block_index))=$(block_candidate_sets)/$(settings.facial_reduction_weighted_max_candidate_sets), " *
           "affine_products=$(state.total_affine_products)/$(settings.facial_reduction_weighted_max_total_affine_products), " *
           "exact_without_scout=$(state.exact_without_scout_attempts)/$(settings.facial_reduction_weighted_exact_without_scout_limit)"
end

function _weighted_exact_without_scout_available(
    cache::_FacialReductionExactCache,
    settings::Settings;
    reserve_count::Int = 0,
)
    limit = settings.facial_reduction_weighted_exact_without_scout_limit
    effective_limit = max(0, limit - max(0, reserve_count))
    return cache.weighted_search.exact_without_scout_attempts < effective_limit
end

function _weighted_subspace_candidate_remaining_work(
    candidate,
    cache::_FacialReductionExactCache,
    settings::Settings,
)
    products = BigInt(0)
    needs_candidate_slot = false
    if _individual_subspace_certificate_is_small(candidate.individual_work, settings) &&
       !(candidate.attempt_key in cache.individual_subspace_failures)
        products += BigInt(candidate.individual_work.affine_products)
        needs_candidate_slot = true
    end
    if _numeric_weighted_subspace_exposure_is_small(candidate.weighted_work, settings) &&
       !haskey(cache.numeric_weighted_failures, candidate.attempt_key)
        products += BigInt(candidate.weighted_work.affine_products)
        needs_candidate_slot = true
    end
    if !candidate.cheap_weighted_attempted &&
       _weighted_subspace_exposure_is_small(candidate.weighted_work, settings) &&
       !(candidate.attempt_key in cache.exact_weighted_failures)
        # A promising scout may still require the exact weighted fallback. Reserve
        # enough work for that path instead of assuming that an unpromising scout
        # will make it unnecessary.
        products += BigInt(candidate.weighted_work.affine_products)
        needs_candidate_slot = true
    end
    return (affine_products = products, candidate_slots = needs_candidate_slot ? 1 : 0)
end

function _work_reserved_weighted_subspace_candidate_schedule(
    candidates::Vector,
    reserved_candidate_keys::Vector,
    cache::_FacialReductionExactCache,
    settings::Settings,
    block_index::Int,
)
    isempty(reserved_candidate_keys) && return (
        candidates,
        reserved_candidate_keys = Any[],
        promoted_from = Int[],
        promoted_to = nothing,
        reserved_affine_products = BigInt(0),
    )
    keyed_candidates = Dict(candidate.key => candidate for candidate in candidates)
    available_keys =
        filter(key -> haskey(keyed_candidates, key), reserved_candidate_keys)
    isempty(available_keys) && return (
        candidates,
        reserved_candidate_keys = Any[],
        promoted_from = Int[],
        promoted_to = nothing,
        reserved_affine_products = BigInt(0),
    )

    available_products = max(
        BigInt(0),
        BigInt(settings.facial_reduction_weighted_max_total_affine_products) -
        cache.weighted_search.total_affine_products,
    )
    available_candidate_slots = max(
        0,
        settings.facial_reduction_weighted_max_candidate_sets -
        get(cache.weighted_search.candidate_sets_attempted, block_index, 0),
    )

    retained_keys = Any[]
    reserved_products = BigInt(0)
    reserved_candidate_slots = 0
    for key in available_keys
        work = _weighted_subspace_candidate_remaining_work(
            keyed_candidates[key],
            cache,
            settings,
        )
        reserved_products + work.affine_products > available_products && break
        reserved_candidate_slots + work.candidate_slots >
        available_candidate_slots && break
        push!(retained_keys, key)
        reserved_products += work.affine_products
        reserved_candidate_slots += work.candidate_slots
    end
    isempty(retained_keys) && return (
        candidates,
        reserved_candidate_keys = Any[],
        promoted_from = Int[],
        promoted_to = nothing,
        reserved_affine_products = BigInt(0),
    )

    reserved_key_set = Set(retained_keys)
    original_positions = [
        findfirst(candidate -> candidate.key == key, candidates) for
        key in retained_keys
    ]
    reserved_candidates = [keyed_candidates[key] for key in retained_keys]
    remaining_candidates =
        filter(candidate -> !(candidate.key in reserved_key_set), candidates)
    # Exact structural faces commonly have very small rational coefficients,
    # whereas a slightly better numerical fit can carry thousands of bits.
    # Try the reserved low-complexity ladder first. Merely reserving enough
    # work for it still lets expensive overfit candidates consume many
    # minutes before the useful candidate is reached.
    reordered = vcat(reserved_candidates, remaining_candidates)
    promoted_to = 1
    already_in_place =
        original_positions ==
        collect(promoted_to:(promoted_to + length(retained_keys) - 1))
    return (
        candidates = reordered,
        reserved_candidate_keys = retained_keys,
        promoted_from = already_in_place ? Int[] : original_positions,
        promoted_to = already_in_place ? nothing : promoted_to,
        reserved_affine_products = reserved_products,
    )
end

function _linearly_independent_directions(directions::Vector{Vector{ExactRational}})
    isempty(directions) && return directions
    matrix = hcat(directions...)
    augmented = hcat(copy(matrix), zeros(ExactRational, size(matrix, 1)))
    _, pivots = _rref(augmented)
    return [directions[index] for index in pivots]
end

function _certified_pivoted_subspace_directions(
    opt::Optimizer,
    problem::ProblemData,
    block::BlockStructure,
    block_index::Int,
    subspace::AbstractMatrix{F},
    ::Type{F},
    description::AbstractString,
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    tolerances = nothing,
    candidate_limit::Union{Nothing,Int} = nothing,
    probe_coarsest_candidate::Bool = true,
) where {F<:AbstractFloat}
    selected_tolerances =
        tolerances === nothing ?
        _facial_reduction_subspace_tolerances(opt.settings, F) :
        collect(tolerances)
    if probe_coarsest_candidate && length(selected_tolerances) > 1
        coarse_directions = _certified_pivoted_subspace_directions(
            opt,
            problem,
            block,
            block_index,
            subspace,
            F,
            description;
            cache,
            tolerances = selected_tolerances[1:1],
            candidate_limit = 1,
            probe_coarsest_candidate = false,
        )
        isempty(coarse_directions) || return coarse_directions
        return _certified_pivoted_subspace_directions(
            opt,
            problem,
            block,
            block_index,
            subspace,
            F,
            description;
            cache,
            tolerances = selected_tolerances,
            probe_coarsest_candidate = false,
        )
    end

    # Generate and exactly deduplicate every tolerance/chart reconstruction
    # before launching work whose cost scales with the affine nullspace.
    attempted_tolerances = 0
    generated_candidate_sets = 0
    seen_subspaces = Set{Any}()
    candidates = Any[]
    for tolerance in selected_tolerances
        candidate_sets =
            _rational_subspace_candidate_sets(subspace, opt.settings, F, tolerance)
        isempty(candidate_sets) && continue
        attempted_tolerances += 1
        for candidate_set in candidate_sets
            generated_candidate_sets += 1
            candidate_set.key in seen_subspaces && continue
            push!(seen_subspaces, candidate_set.key)
            push!(candidates, (candidate_set..., tolerance = tolerance))
            candidate_limit !== nothing &&
                length(candidates) >= candidate_limit &&
                break
        end
        candidate_limit !== nothing &&
            length(candidates) >= candidate_limit &&
            break
    end
    isempty(candidates) && return Vector{ExactRational}[]

    plan = _weighted_subspace_candidate_plan(candidates, opt.settings, F)
    unusable_candidates = plan.cheap_exact_only
    scheduled_candidates = plan.scheduled
    exact_check_candidates = plan.exact_checks
    if plan.rejected_count > 0
        examples = join(
            [
                "fingerprint=$(candidate.fingerprint), relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), reason=$(candidate.numeric_issue)" for
                candidate in Iterators.take(unusable_candidates, 3)
            ],
            "; ",
        )
        suffix =
            plan.rejected_count > 3 ? "; $(plan.rejected_count - 3) more omitted" : ""
        _log(
            opt,
            "Facial reduction: retained $(plan.rejected_count) numerically unusable pivoted $(description) candidate subspace(s) for cheap exact checks but excluded them from weighted scheduling ($(examples)$(suffix))",
        )
    end
    _log(
        opt,
        "Facial reduction: collected $(length(exact_check_candidates)) unique pivoted $(description) candidate subspace(s) from $(generated_candidate_sets) reconstruction(s) across $(attempted_tolerances) relation tolerance(s); scheduled $(length(scheduled_candidates)) numerically usable candidate(s) in $(plan.cluster_count) approximate projector cluster(s)",
    )
    if !_facial_reduction_row_space_is_small(problem, opt.settings)
        row_count, variable_count = size(problem.A)
        _log(
            opt,
            "Facial reduction: certifying pivoted $(description) candidates with exact affine tests; skipping dense row-space provenance for large affine system ($(row_count)x$(variable_count))",
        )
    end

    proposed_directions = 0
    last_violation = nothing
    weighted_candidates = Any[]
    # Basis-invariant trace checks and explicitly cheap weighted checks remain
    # available to every candidate. They can certify a face before any global
    # expensive-work budget is consumed.
    for candidate in exact_check_candidates
        directions = candidate.directions
        proposed_directions += length(directions)
        _record_directions!(:certified, length(directions), 0, 0)
        joint_violation = _block_trace_vanish_violation(
            problem,
            block,
            directions;
            cache,
            block_index,
        )
        if joint_violation === nothing
            _record_directions!(:certified, 0, length(directions), 0)
            _log(
                opt,
                "Facial reduction: using certified pivoted $(description) subspace for PSD block $(block_index) (joint PSD trace certificate; $(length(directions)) direction(s); relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint))",
            )
            return directions
        end
        last_violation = joint_violation

        work = _weighted_subspace_exposure_work(problem, block, length(directions))
        attempt_key =
            _weighted_subspace_attempt_key(block_index, candidate, opt.settings, F)
        cheap_weighted_attempted =
            _weighted_subspace_exposure_is_cheap(work, opt.settings) &&
            _weighted_subspace_exposure_is_small(work, opt.settings)
        if cheap_weighted_attempted
            _log(
                opt,
                "Facial reduction: cheap exact weighted candidate for PSD block $(block_index) (relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), affine_products=$(work.affine_products))",
            )
            weighted_exposure = if attempt_key in cache.exact_weighted_failures
                nothing
            else
                result = _block_weighted_subspace_exposure(
                    problem,
                    block,
                    directions,
                    opt.settings,
                    F,
                )
                result === nothing &&
                    push!(cache.exact_weighted_failures, attempt_key)
                result
            end
            if weighted_exposure !== nothing
                exposure_directions = weighted_exposure.directions
                _record_directions!(
                    :certified,
                    0,
                    length(exposure_directions),
                    0,
                )
                _log(
                    opt,
                    "Facial reduction: using certified pivoted $(description) subspace for PSD block $(block_index) (weighted joint PSD certificate; $(length(exposure_directions))/$(length(directions)) direction(s); relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), weight_tol=$(_format_metric(weighted_exposure.tolerance)))",
                )
                return exposure_directions
            end
        end
        candidate.numeric_usable || continue

        push!(
            weighted_candidates,
            (
                candidate...,
                weighted_work = work,
                individual_work = _individual_subspace_certificate_work(
                    problem,
                    block,
                    length(directions),
                ),
                attempt_key = attempt_key,
                cheap_weighted_attempted = cheap_weighted_attempted,
                joint_violation = joint_violation,
            ),
        )
    end

    weighted_schedule =
        _weighted_subspace_candidate_schedule(weighted_candidates, opt.settings, F)
    work_reserved_schedule = _work_reserved_weighted_subspace_candidate_schedule(
        weighted_schedule.candidates,
        weighted_schedule.reserved_candidate_keys,
        cache,
        opt.settings,
        block_index,
    )
    weighted_candidates = work_reserved_schedule.candidates
    reserved_candidate_key_set =
        Set(work_reserved_schedule.reserved_candidate_keys)
    function reserved_candidate_needs_exact(candidate)
        return candidate.key in reserved_candidate_key_set &&
            opt.settings.facial_reduction_weighted_exact_without_scout_limit > 0 &&
            !candidate.cheap_weighted_attempted &&
            _weighted_subspace_exposure_is_small(
                candidate.weighted_work,
                opt.settings,
            ) &&
            !(candidate.attempt_key in cache.exact_weighted_failures)
    end
    reserved_candidate_indices = [
        index for (index, candidate) in enumerate(weighted_candidates) if
        reserved_candidate_needs_exact(candidate)
    ]
    if work_reserved_schedule.promoted_to !== nothing
        promoted_from = join(work_reserved_schedule.promoted_from, ", ")
        promoted_range = work_reserved_schedule.promoted_to:(
            work_reserved_schedule.promoted_to +
            length(work_reserved_schedule.reserved_candidate_keys) - 1
        )
        promoted_candidates = weighted_candidates[promoted_range]
        candidate_details = join(
            [
                "relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), coefficient_bits=$(candidate.coefficient_bits)" for
                candidate in promoted_candidates
            ],
            "; ",
        )
        _log(
            opt,
            "Facial reduction: promoted reserved low-complexity $(description) candidate ladder for PSD block $(block_index) from scheduled position(s) $(promoted_from) to $(first(promoted_range))-$(last(promoted_range)) to preserve $(work_reserved_schedule.reserved_affine_products) affine products of remaining work ($(candidate_details))",
        )
    end
    start_candidates = get(cache.weighted_search.candidate_sets_attempted, block_index, 0)
    start_products = cache.weighted_search.total_affine_products
    cached_failures = 0
    budget_truncated = false
    # Approximate projector clusters diversify this order. The cluster test is
    # only a scheduling heuristic; all accepted faces still pass exact checks.
    for (scheduled_index, candidate) in enumerate(weighted_candidates)
        work = candidate.weighted_work
        candidate_started = false
        scout_promising = false
        scout_status = :skipped
        scout_reason = "disabled by its per-candidate work limit"
        last_violation = candidate.joint_violation

        if _individual_subspace_certificate_is_small(
            candidate.individual_work,
            opt.settings,
        )
            if candidate.attempt_key in cache.individual_subspace_failures
                cached_failures += 1
            else
                reservation = _reserve_weighted_subspace_work!(
                    cache,
                    opt.settings,
                    block_index,
                    candidate.individual_work.affine_products;
                    new_candidate = true,
                )
                if reservation !== :reserved
                    _log(
                        opt,
                        "Facial reduction: subspace candidate search reached the $(reservation === :candidate_limit ? "candidate-set" : "cumulative-product") limit before individual checks for scheduled candidate $(scheduled_index)/$(length(weighted_candidates)); $(_weighted_subspace_budget_summary(cache, opt.settings, block_index))",
                    )
                    budget_truncated = true
                    break
                end
                candidate_started = true
                _log(
                    opt,
                    "Facial reduction: weighted candidate $(get(cache.weighted_search.candidate_sets_attempted, block_index, 0))/$(opt.settings.facial_reduction_weighted_max_candidate_sets) for PSD block $(block_index): individual exact direction checks (scheduled=$(scheduled_index)/$(length(weighted_candidates)), relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), affine_products=$(candidate.individual_work.affine_products), cumulative_products=$(cache.weighted_search.total_affine_products)/$(opt.settings.facial_reduction_weighted_max_total_affine_products))",
                )
                accepted = Vector{Vector{ExactRational}}()
                rejected = 0
                for direction in candidate.directions
                    violation = _block_annihilation_violation(
                        problem,
                        block,
                        direction;
                        cache,
                        block_index,
                    )
                    if violation === nothing
                        push!(accepted, direction)
                    else
                        rejected += 1
                        last_violation = violation
                    end
                end
                _record_directions!(:certified, 0, length(accepted), rejected)
                accepted = _linearly_independent_directions(accepted)
                if !isempty(accepted)
                    _log(
                        opt,
                        "Facial reduction: using certified pivoted $(description) subspace for PSD block $(block_index) ($(length(accepted)) direction(s); relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint))",
                    )
                    return accepted
                end
                push!(cache.individual_subspace_failures, candidate.attempt_key)
            end
        end

        if _numeric_weighted_subspace_exposure_is_small(work, opt.settings)
            if haskey(cache.numeric_weighted_failures, candidate.attempt_key)
                cached_failures += 1
                cached_scout =
                    cache.numeric_weighted_failures[candidate.attempt_key]
                scout_promising = cached_scout.promising
                scout_status = cached_scout.status
                scout_reason = "cached: $(cached_scout.reason)"
            else
                reservation = _reserve_weighted_subspace_work!(
                    cache,
                    opt.settings,
                    block_index,
                    work.affine_products;
                    new_candidate = !candidate_started,
                )
                if reservation !== :reserved
                    _log(
                        opt,
                        "Facial reduction: weighted candidate search reached the $(reservation === :candidate_limit ? "candidate-set" : "cumulative-product") limit before scheduled candidate $(scheduled_index)/$(length(weighted_candidates)); $(_weighted_subspace_budget_summary(cache, opt.settings, block_index))",
                    )
                    budget_truncated = true
                    break
                end
                candidate_started = true
                _log(
                    opt,
                    "Facial reduction: weighted candidate $(get(cache.weighted_search.candidate_sets_attempted, block_index, 0))/$(opt.settings.facial_reduction_weighted_max_candidate_sets) for PSD block $(block_index): numerical scout (scheduled=$(scheduled_index)/$(length(weighted_candidates)), relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), reconstruction_error=$(_format_metric(candidate.reconstruction_error)), coefficient_bits=$(candidate.coefficient_bits), affine_products=$(work.affine_products), cumulative_products=$(cache.weighted_search.total_affine_products)/$(opt.settings.facial_reduction_weighted_max_total_affine_products))",
                )
                scout = _numeric_weighted_subspace_exposure_attempt(
                    problem,
                    block,
                    candidate.directions,
                    opt.settings,
                    F,
                )
                if scout.exposure !== nothing
                    exposure_directions = scout.exposure.directions
                    _record_directions!(
                        :certified,
                        0,
                        length(exposure_directions),
                        0,
                    )
                    _log(
                        opt,
                        "Facial reduction: using certified pivoted $(description) subspace for PSD block $(block_index) (numerical-to-exact weighted PSD certificate; $(length(exposure_directions))/$(length(candidate.directions)) direction(s); relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), weight_tol=$(_format_metric(scout.exposure.tolerance)))",
                    )
                    return exposure_directions
                end
                scout_promising = scout.promising
                scout_status = scout.status
                scout_reason = scout.reason
                cache.numeric_weighted_failures[candidate.attempt_key] = (
                    promising = scout.promising,
                    status = scout.status,
                    reason = scout.reason,
                )
                if scout.status === :unavailable
                    _log(
                        opt,
                        "Facial reduction: weighted candidate fingerprint=$(candidate.fingerprint) numerical scout unavailable ($(scout_reason))",
                    )
                else
                    _log(
                        opt,
                        "Facial reduction: weighted candidate fingerprint=$(candidate.fingerprint) numerical scout did not certify a face (status=$(scout.status), $(scout_reason))",
                    )
                end
            end
        end

        candidate.cheap_weighted_attempted && continue
        _weighted_subspace_exposure_is_small(work, opt.settings) || continue
        if candidate.attempt_key in cache.exact_weighted_failures
            cached_failures += 1
            continue
        end
        future_reserved_indices =
            filter(index -> index > scheduled_index, reserved_candidate_indices)
        reserved_exact_count = length(future_reserved_indices)
        exact_without_scout =
            !scout_promising &&
            _weighted_exact_without_scout_available(
                cache,
                opt.settings;
                reserve_count = reserved_exact_count,
            )
        if !scout_promising && !exact_without_scout
            if reserved_exact_count > 0 &&
               !(candidate.key in reserved_candidate_key_set) &&
               _weighted_exact_without_scout_available(cache, opt.settings)
                reserved_positions = join(future_reserved_indices, ", ")
                _log(
                    opt,
                    "Facial reduction: weighted candidate fingerprint=$(candidate.fingerprint) exact certificate skipped after an unpromising numerical scout; preserving $(reserved_exact_count) exact-without-scout attempt(s) for the reserved low-complexity candidates at scheduled position(s) $(reserved_positions)",
                )
                continue
            end
            scout_description =
                scout_status === :unavailable ? "an unavailable numerical scout" :
                scout_status === :unpromising ? "an unpromising numerical scout" :
                "no promising numerical scout"
            _log(
                opt,
                "Facial reduction: weighted candidate fingerprint=$(candidate.fingerprint) exact certificate skipped after $(scout_description); exact-without-scout allowance exhausted",
            )
            continue
        end

        reservation = _reserve_weighted_subspace_work!(
            cache,
            opt.settings,
            block_index,
            work.affine_products;
            new_candidate = !candidate_started,
        )
        if reservation !== :reserved
            _log(
                opt,
                "Facial reduction: exact weighted certificate for fingerprint=$(candidate.fingerprint) skipped by the $(reservation === :candidate_limit ? "candidate-set" : "cumulative-product") limit; $(_weighted_subspace_budget_summary(cache, opt.settings, block_index))",
            )
            budget_truncated = true
            break
        end
        exact_without_scout &&
            (cache.weighted_search.exact_without_scout_attempts += 1)
        _log(
            opt,
            "Facial reduction: weighted candidate $(get(cache.weighted_search.candidate_sets_attempted, block_index, 0))/$(opt.settings.facial_reduction_weighted_max_candidate_sets) for PSD block $(block_index): exact joint certificate (relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), scout_status=$(scout_status), scout=$(scout_reason), affine_products=$(work.affine_products), cumulative_products=$(cache.weighted_search.total_affine_products)/$(opt.settings.facial_reduction_weighted_max_total_affine_products))",
        )
        weighted_exposure = _block_weighted_subspace_exposure(
            problem,
            block,
            candidate.directions,
            opt.settings,
            F,
        )
        if weighted_exposure !== nothing
            exposure_directions = weighted_exposure.directions
            _record_directions!(
                :certified,
                0,
                length(exposure_directions),
                0,
            )
            _log(
                opt,
                "Facial reduction: using certified pivoted $(description) subspace for PSD block $(block_index) (weighted joint PSD certificate; $(length(exposure_directions))/$(length(candidate.directions)) direction(s); relation_tol=$(_format_metric(candidate.tolerance)), method=$(candidate.method), fingerprint=$(candidate.fingerprint), weight_tol=$(_format_metric(weighted_exposure.tolerance)))",
            )
            return exposure_directions
        end
        push!(cache.exact_weighted_failures, candidate.attempt_key)
    end

    delta_candidates =
        get(cache.weighted_search.candidate_sets_attempted, block_index, 0) -
        start_candidates
    delta_products = cache.weighted_search.total_affine_products - start_products
    _log(
        opt,
        "Facial reduction: weighted $(description) search summary for PSD block $(block_index): scheduled=$(length(weighted_candidates)), attempted=$(delta_candidates), affine_products=$(delta_products), cached_failures=$(cached_failures), truncated_by_budget=$(budget_truncated); $(_weighted_subspace_budget_summary(cache, opt.settings, block_index))",
    )
    proposed_directions == 0 && return Vector{ExactRational}[]
    _log(
        opt,
        "Facial reduction: no exact subspace certificate recovered from $(length(exact_check_candidates)) unique joint $(description) candidate(s) for PSD block $(block_index) across $(attempted_tolerances) relation tolerance(s); $(proposed_directions) direction proposal(s) checked$(budget_truncated ? " before the weighted search budget was exhausted" : "") ($(last_violation))",
    )
    return Vector{ExactRational}[]
end

function _orthogonal_complement_basis(directions::Vector{Vector{ExactRational}}, dimension::Int)
    if isempty(directions)
        return Matrix{ExactRational}(I, dimension, dimension)
    end
    matrix = Matrix(transpose(hcat(directions...)))
    return _nullspace_basis_exact(matrix)
end

function _exact_block_nullspace_directions(
    problem::ProblemData,
    block::BlockStructure,
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    block_index::Int = something(findfirst(==(block), problem.blocks)),
)
    problem.affine === nothing && return Vector{ExactRational}[]
    cached = cache.block_exact_directions[block_index]
    cached === nothing || return cached

    particular, nullspace = problem.affine
    if (1 + size(nullspace, 2)) * block.size^2 > 20_000
        cache.block_exact_directions[block_index] = Vector{Vector{ExactRational}}()
        return cache.block_exact_directions[block_index]
    end
    basis = _with_nemo_error("exact block-nullspace initialization") do
        Nemo.identity_matrix(Nemo.QQ, block.size)
    end
    matrices = Nemo.QQMatrix[
        _to_nemo_matrix(_vector_to_matrix(particular, block)),
    ]
    for column in axes(nullspace, 2)
        push!(matrices, _to_nemo_matrix(_vector_to_matrix(view(nullspace, :, column), block)))
    end
    for matrix in matrices
        image = _with_nemo_error("exact block-nullspace matrix multiplication") do
            matrix * basis
        end
        _, kernel = _with_nemo_error("exact block-nullspace computation") do
            Nemo.nullspace(image)
        end
        basis = _with_nemo_error("exact block-nullspace basis update") do
            basis * kernel
        end
        size(basis, 2) == 0 && break
    end

    basis_exact = _from_nemo_matrix(basis)
    directions = [
        _normalize_rational_direction(collect(view(basis_exact, :, column))) for
        column in axes(basis_exact, 2)
    ]
    filter!(direction -> any(!iszero, direction), directions)
    directions = _linearly_independent_directions(directions)
    cache.block_exact_directions[block_index] = directions
    _record_approximate_cache_memory!(:facial_reduction, cache)
    return directions
end

function _candidate_kernel_directions(
    opt::Optimizer,
    problem::ProblemData,
    block_index::Int,
    block_matrix::Matrix{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    block = problem.blocks[block_index]
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = _facial_reduction_eigen(opt, symmetric_matrix)
    exposure_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    kernel_indices = [
        index for (index, value) in enumerate(eigen_factor.values) if abs(value) <= exposure_tolerance
    ]
    isempty(kernel_indices) && return Vector{ExactRational}[]

    kernel_subspace = Matrix(eigen_factor.vectors[:, kernel_indices])
    pivoted_directions = _certified_pivoted_subspace_directions(
        opt,
        problem,
        block,
        block_index,
        kernel_subspace,
        F,
        "boundary kernel",
        ;
        cache,
    )
    isempty(pivoted_directions) || return pivoted_directions

    exact_directions = _exact_block_nullspace_directions(
        problem,
        block;
        cache,
        block_index,
    )

    if isempty(exact_directions)
        heuristic_directions = Vector{Vector{ExactRational}}()
        individually_certified_directions = Vector{Vector{ExactRational}}()
        rejected_heuristics = 0
        for kernel_index in sort(kernel_indices; by = index -> abs(eigen_factor.values[index]))
            heuristic = _heuristic_kernel_direction(
                block_matrix,
                collect(view(eigen_factor.vectors, :, kernel_index)),
                opt.settings,
                F,
            )
            heuristic === nothing && continue

            direction = heuristic.direction
            _record_directions!(:certified, 1, 0, 0)
            certificate = _block_face_direction_certificate(
                problem,
                block,
                direction;
                cache,
                block_index,
            )
            direction_summary = _format_exact_direction(direction)
            if certificate.kind != :none
                certificate_label =
                    certificate.kind == :affine_rows ? "affine row" : "PSD diagonal"
                _log(
                    opt,
                    "Facial reduction: certified rationalized boundary kernel direction for PSD block $(block_index) ($(certificate_label) certificate; eig=$(_format_metric(eigen_factor.values[kernel_index])), residual=$(_format_metric(heuristic.residual)), rationalize_tol=$(_format_metric(heuristic.tolerance))): $(direction_summary)",
                )
                push!(heuristic_directions, direction)
                push!(individually_certified_directions, direction)
                _record_directions!(:certified, 0, 1, 0)
            else
                rejected_heuristics += 1
                _log(
                    opt,
                    "Facial reduction: rejected rationalized boundary kernel direction for PSD block $(block_index) (eig=$(_format_metric(eigen_factor.values[kernel_index])), residual=$(_format_metric(heuristic.residual)), rationalize_tol=$(_format_metric(heuristic.tolerance))); no exact face certificate ($(certificate.violation)): $(direction_summary)",
                )
                push!(heuristic_directions, direction)
                _record_directions!(:certified, 0, 0, 1)
            end
        end
        heuristic_directions = _linearly_independent_directions(heuristic_directions)
        if length(heuristic_directions) > 1
            violation = _block_trace_vanish_violation(
                problem,
                block,
                heuristic_directions;
                cache,
                block_index,
            )
            if violation === nothing
                _log(
                    opt,
                    "Facial reduction: certified $(length(heuristic_directions))-dimensional rationalized boundary kernel subspace for PSD block $(block_index) (PSD trace certificate)",
                )
                return heuristic_directions
            end
            _log(
                opt,
                "Facial reduction: rejected rationalized boundary kernel subspace for PSD block $(block_index); no exact PSD trace certificate ($(violation))",
            )
        end

        individually_certified_directions =
            _linearly_independent_directions(individually_certified_directions)
        if !isempty(individually_certified_directions)
            _log(
                opt,
                "Facial reduction: using certified rationalized boundary kernel directions for PSD block $(block_index)",
            )
            return individually_certified_directions
        end

        if rejected_heuristics > 0
            _log(
                opt,
                "Facial reduction: rejected $(rejected_heuristics) uncertified rationalized boundary kernel direction(s) for PSD block $(block_index)",
            )
            return Vector{ExactRational}[]
        end

        message =
            "Facial reduction found a PSD block on the cone boundary, " *
            "but the exposed nullspace directions could not be represented exactly over the rational coefficient field."
        if _facial_reduction_irrational_behavior(opt.settings) == :warn
            _log(opt, message)
            return Vector{ExactRational}[]
        end
        throw(ErrorException(message))
    end

    return exact_directions
end

function _heuristic_kernel_direction_candidates(
    opt::Optimizer,
    problem::ProblemData,
    block_index::Int,
    block_matrix::Matrix{F},
    ::Type{F},
) where {F<:AbstractFloat}
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = _facial_reduction_eigen(opt, symmetric_matrix)
    exposure_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    kernel_indices = [
        index for (index, value) in enumerate(eigen_factor.values) if
        abs(value) <= exposure_tolerance
    ]
    candidates = _TentativeFaceDirection{F}[]
    for kernel_index in sort(kernel_indices; by = index -> abs(eigen_factor.values[index]))
        heuristic = _heuristic_kernel_direction(
            block_matrix,
            collect(view(eigen_factor.vectors, :, kernel_index)),
            opt.settings,
            F,
        )
        heuristic === nothing && continue
        eigenvalue = eigen_factor.values[kernel_index]
        score = abs(heuristic.residual) + abs(eigenvalue)
        push!(candidates, _TentativeFaceDirection{F}(
            block_index,
            _normalize_rational_direction(heuristic.direction),
            eigenvalue,
            heuristic.residual,
            score,
        ))
    end
    return candidates
end

function _tentative_candidate_sort_key(candidate::_TentativeFaceDirection)
    return (
        candidate.score,
        abs(candidate.eigenvalue),
        candidate.residual,
        candidate.block_index,
        string(candidate.direction),
    )
end

function _tentative_candidate_keep_bases(
    problem::ProblemData,
    candidates::Vector{<:_TentativeFaceDirection},
)
    directions_by_block = Dict{Int,Vector{Vector{ExactRational}}}()
    for candidate in candidates
        push!(
            get!(directions_by_block, candidate.block_index, Vector{Vector{ExactRational}}()),
            candidate.direction,
        )
    end

    keep_bases = Dict{Int,Matrix{ExactRational}}()
    for block_index in sort(collect(keys(directions_by_block)))
        directions = _linearly_independent_directions(directions_by_block[block_index])
        isempty(directions) && continue
        keep_basis = _orthogonal_complement_basis(
            directions,
            problem.blocks[block_index].size,
        )
        size(keep_basis, 2) == problem.blocks[block_index].size && continue
        keep_bases[block_index] = keep_basis
    end
    return keep_bases
end

function _tentative_batch_work(
    problem::ProblemData,
    candidates::Vector{<:_TentativeFaceDirection},
)
    keep_bases = _tentative_candidate_keep_bases(problem, candidates)
    added_dimension = 0
    restriction_rows = 0
    for (block_index, keep_basis) in keep_bases
        reduced_dimension = size(keep_basis, 2)
        added_dimension += reduced_dimension * (reduced_dimension + 1) ÷ 2
        restriction_rows += length(problem.blocks[block_index].local_positions)
    end
    affine_dimension = problem.affine === nothing ? 0 : size(problem.affine[2], 2)
    coordinate_dimension = affine_dimension + added_dimension
    estimated_free_dimension = max(0, coordinate_dimension - restriction_rows)
    coordinate_entries = BigInt(restriction_rows) * (coordinate_dimension + 1)
    lift_products = BigInt(length(problem.objective_vector_raw)) * affine_dimension *
                    (estimated_free_dimension + 1)
    output_entries = BigInt(length(problem.objective_vector_raw) + added_dimension) *
                     (estimated_free_dimension + 1)
    affine_entry_count = if problem.affine === nothing
        0
    else
        length(problem.affine[1]) + length(problem.affine[2])
    end
    average_entry_bytes = if affine_entry_count == 0
        32
    else
        max(32, cld(Base.summarysize(problem.affine), affine_entry_count))
    end
    estimated_bytes = output_entries * average_entry_bytes
    return (
        keep_bases = keep_bases,
        directions = length(candidates),
        restriction_rows = restriction_rows,
        coordinate_dimension = coordinate_dimension,
        coordinate_entries = coordinate_entries,
        lift_products = lift_products,
        output_entries = output_entries,
        estimated_bytes = estimated_bytes,
    )
end

function _tentative_batch_within_budget(work, settings::Settings)
    return work.directions <= settings.facial_reduction_tentative_max_directions &&
           work.coordinate_entries <=
           settings.facial_reduction_tentative_max_coordinate_entries &&
           work.lift_products <= settings.facial_reduction_tentative_max_lift_products &&
           work.estimated_bytes <= settings.facial_reduction_tentative_max_estimated_bytes &&
           work.output_entries <= settings.facial_reduction_affine_lift_max_output_entries &&
           work.estimated_bytes <= settings.facial_reduction_affine_lift_max_estimated_bytes
end

function _tentative_batch_work_summary(work, settings::Settings)
    return "directions=$(work.directions)/$(settings.facial_reduction_tentative_max_directions), " *
           "coordinate_entries=$(work.coordinate_entries)/$(settings.facial_reduction_tentative_max_coordinate_entries), " *
           "lift_products=$(work.lift_products)/$(settings.facial_reduction_tentative_max_lift_products), " *
           "estimated_bytes=$(work.estimated_bytes)/$(settings.facial_reduction_tentative_max_estimated_bytes), " *
           "output_entries=$(work.output_entries)/$(settings.facial_reduction_affine_lift_max_output_entries), " *
           "affine_lift_bytes=$(work.estimated_bytes)/$(settings.facial_reduction_affine_lift_max_estimated_bytes)"
end

function _tentative_batch_problem(
    problem::ProblemData,
    candidates::Vector{<:_TentativeFaceDirection},
    ;
    checkpoint::Union{Nothing,Function} = nothing,
    settings::Settings = Settings(),
)
    checkpoint !== nothing && checkpoint("constructing per-block keep bases")
    keep_bases = _tentative_candidate_keep_bases(problem, candidates)
    isempty(keep_bases) && return problem
    removed_dimensions = sum(
        problem.blocks[block_index].size - size(keep_basis, 2) for
        (block_index, keep_basis) in keep_bases
    )
    checkpoint !== nothing && checkpoint(
        "applying tentative face across $(length(keep_bases)) PSD block(s), removing $(removed_dimensions) PSD dimension(s)",
    )
    return _apply_facial_reduction(
        problem,
        Int[],
        keep_bases;
        certified = false,
        checkpoint,
        settings,
    )
end

function _tentative_greedy_admission(
    problem::ProblemData,
    candidates::Vector{_TentativeFaceDirection{F}},
    settings::Settings = Settings(),
) where {F<:AbstractFloat}
    accepted_candidates = _TentativeFaceDirection{F}[]
    consistent_problem = nothing
    for candidate_item in candidates
        trial_candidates = vcat(accepted_candidates, [candidate_item])
        _tentative_batch_within_budget(
            _tentative_batch_work(problem, trial_candidates),
            settings,
        ) || continue
        trial_problem = _tentative_batch_problem(problem, trial_candidates; settings)
        if trial_problem.affine === nothing
            continue
        end
        push!(accepted_candidates, candidate_item)
        consistent_problem = trial_problem
    end
    return consistent_problem, accepted_candidates
end

function _tentative_search_result(
    problem,
    fallback_problem,
    return_details::Bool,
)
    return return_details ?
           (problem = problem, fallback_problem = fallback_problem) :
           problem
end

"""
Construct a heuristic face used only to search for an exact feasible point. This is not
a facial-reduction certificate: callers must validate any recovered point
exactly against the unreduced problem and must not infer infeasibility or an
objective bound from this restriction.
"""
function _tentative_feasibility_search_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
    ;
    return_details::Bool = false,
) where {F<:AbstractFloat}
    candidates = _TentativeFaceDirection{F}[]
    for (block_index, block) in enumerate(problem.blocks)
        append!(
            candidates,
            _heuristic_kernel_direction_candidates(
                opt,
                problem,
                block_index,
                _vector_to_matrix(candidate, block),
                F,
            ),
        )
    end
    isempty(candidates) && return _tentative_search_result(nothing, nothing, return_details)
    sort!(candidates; by = _tentative_candidate_sort_key)

    # Keep only one exact representative and an exact independent set per block.
    unique_candidates = _TentativeFaceDirection{F}[]
    directions_by_block = Dict{Int,Vector{Vector{ExactRational}}}()
    seen_by_block = Dict{Int,Set{Any}}()
    for candidate_item in candidates
        seen = get!(seen_by_block, candidate_item.block_index, Set{Any}())
        key = Tuple(candidate_item.direction)
        if key in seen
            continue
        end
        push!(seen, key)
        directions = get!(
            directions_by_block,
            candidate_item.block_index,
            Vector{Vector{ExactRational}}(),
        )
        independent = _linearly_independent_directions(vcat(directions, [candidate_item.direction]))
        if length(independent) == length(directions)
            continue
        end
        push!(directions, candidate_item.direction)
        push!(unique_candidates, candidate_item)
    end

    isempty(unique_candidates) && begin
        _record_directions!(:tentative, length(candidates), 0, length(candidates))
        return _tentative_search_result(nothing, nothing, return_details)
    end

    # Prefer the complete batch. It is formed from the original problem so a
    # failed batch never mutates the problem that will be retried.
    block_count = length(Set(candidate_item.block_index for candidate_item in unique_candidates))
    _log(
        opt,
        "Feasibility search: preparing tentative batch of $(length(unique_candidates)) PSD direction(s) across $(block_count) block(s)",
    )
    batch_work = _tentative_batch_work(problem, unique_candidates)
    if !_tentative_batch_within_budget(batch_work, opt.settings)
        _record_facial_reduction_event!(:tentative_batches_skipped_by_budget)
        _log(
            opt,
            "Feasibility search: complete tentative batch exceeds work budget ($(_tentative_batch_work_summary(batch_work, opt.settings))); trying one direction before recomputing the numerical face",
        )
        for candidate_item in unique_candidates
            incremental_candidates = [candidate_item]
            incremental_work = _tentative_batch_work(problem, incremental_candidates)
            _tentative_batch_within_budget(incremental_work, opt.settings) || continue
            checkpoint = stage -> _log(opt, "Feasibility search: incremental tentative face: $(stage)")
            incremental_problem = _tentative_batch_problem(
                problem,
                incremental_candidates;
                checkpoint,
                settings = opt.settings,
            )
            incremental_problem.affine === nothing && continue
            _record_directions!(
                :tentative,
                length(candidates),
                1,
                length(candidates) - 1,
            )
            _log(
                opt,
                "Feasibility search: admitted one budgeted tentative direction from PSD block $(candidate_item.block_index); remaining kernels will be recomputed on the restricted problem",
            )
            return _tentative_search_result(incremental_problem, nothing, return_details)
        end
        _record_directions!(:tentative, length(candidates), 0, length(candidates))
        _log(opt, "Feasibility search: no tentative direction fits the configured exact-work budget")
        return _tentative_search_result(nothing, nothing, return_details)
    end
    tentative_checkpoint = stage -> _log(opt, "Feasibility search: tentative batch: $(stage)")
    batch_problem = _tentative_batch_problem(
        problem,
        unique_candidates;
        checkpoint = tentative_checkpoint,
        settings = opt.settings,
    )
    if batch_problem.affine !== nothing
        fallback_problem = nothing
        if length(unique_candidates) > 1
            conservative_candidates = unique_candidates[1:1]
            conservative_work =
                _tentative_batch_work(problem, conservative_candidates)
            if _tentative_batch_within_budget(conservative_work, opt.settings)
                _log(
                    opt,
                    "Feasibility search: building optional conservative one-direction fallback",
                )
                conservative_checkpoint = stage -> _log(
                    opt,
                    "Feasibility search: conservative fallback: $(stage)",
                )
                conservative_problem = _tentative_batch_problem(
                    problem,
                    conservative_candidates;
                    checkpoint = conservative_checkpoint,
                    settings = opt.settings,
                )
                if conservative_problem.affine !== nothing &&
                   _barrier_dimension(conservative_problem) >
                   _barrier_dimension(batch_problem)
                    fallback_problem = conservative_problem
                    _log(
                        opt,
                        "Feasibility search: retained the conservative one-direction fallback",
                    )
                else
                    _log(
                        opt,
                        "Feasibility search: discarded the conservative one-direction fallback",
                    )
                end
            else
                _record_facial_reduction_event!(:tentative_batches_skipped_by_budget)
                _log(
                    opt,
                    "Feasibility search: skipped optional conservative one-direction fallback because it exceeds the work budget ($(_tentative_batch_work_summary(conservative_work, opt.settings)))",
                )
            end
        end
        _record_directions!(
            :tentative,
            length(candidates),
            length(unique_candidates),
            length(candidates) - length(unique_candidates),
        )
        _log(
            opt,
            "Feasibility search: tentatively batched $(length(unique_candidates)) PSD direction(s) across $(block_count) block(s); any recovered point will be checked exactly against the unreduced SDP",
        )
        return _tentative_search_result(batch_problem, fallback_problem, return_details)
    end

    # Deterministic rollback: admit candidates one at a time, always
    # recomputing the combined face from the original problem.
    _log(
        opt,
        "Feasibility search: tentative batch was inconsistent; trying greedy admission",
    )
    consistent_problem, accepted_candidates = _tentative_greedy_admission(
        problem,
        unique_candidates,
        opt.settings,
    )

    accepted_count = length(accepted_candidates)
    _record_directions!(
        :tentative,
        length(candidates),
        accepted_count,
        length(candidates) - accepted_count,
    )
    consistent_problem === nothing &&
        return _tentative_search_result(nothing, nothing, return_details)
    _log(
        opt,
        "Feasibility search: batched tentative admission rolled back to $(accepted_count) of $(length(unique_candidates)) PSD direction(s); any recovered point will be checked exactly against the unreduced SDP",
    )
    return _tentative_search_result(consistent_problem, nothing, return_details)
end

function _is_inexact_facial_reduction_error(err)
    err isa ErrorException || return false
    return occursin(
        "could not be represented exactly over the rational coefficient field",
        err.msg,
    )
end

const _PHASE1_DIAGNOSTIC_THRESHOLDS = BigFloat[
    big"1e-6",
    big"1e-8",
    big"1e-10",
    big"1e-12",
]

function _phase1_threshold_summary(eigenvalues, ::Type{F}) where {F<:AbstractFloat}
    parts = String[]
    for threshold in _PHASE1_DIAGNOSTIC_THRESHOLDS
        numeric_threshold = _to_working_float(F, threshold)
        push!(
            parts,
            "<=$(_format_metric(numeric_threshold)):$(count(value -> value <= numeric_threshold, eigenvalues))",
        )
    end
    return join(parts, ", ")
end

function _log_phase1_candidate_diagnostics(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Union{Nothing,Vector{F}},
    margin,
    residual,
    ::Type{F},
) where {F<:AbstractFloat}
    opt.settings.phase1_candidate_diagnostics || return
    candidate === nothing && return

    details = String[]
    margin === nothing || push!(details, "margin=$(_format_metric(margin))")
    residual === nothing || push!(details, "residual=$(_format_metric(residual))")
    isempty(details) || _log(opt, "Phase I candidate diagnostics: " * join(details, ", "))

    if !isempty(problem.positive_scalars)
        scalar_values = candidate[problem.positive_scalars]
        _log(
            opt,
            "Phase I candidate diagnostics: scalar slacks min=$(_format_metric(minimum(scalar_values))), <=0=$(count(value -> value <= zero(F), scalar_values))",
        )
    end

    for (block_index, block) in enumerate(problem.blocks)
        block_matrix = _vector_to_matrix(candidate, block)
        eigenvalues = _facial_reduction_eigvals(opt, block_matrix)
        isempty(eigenvalues) && continue
        _log(
            opt,
            "Phase I candidate diagnostics: block $(block_index) size=$(block.size), min_eig=$(_format_metric(minimum(eigenvalues))), max_eig=$(_format_metric(maximum(eigenvalues))), negative=$(count(value -> value < zero(F), eigenvalues)), $(_phase1_threshold_summary(eigenvalues, F))",
        )
    end
    return
end

function _facial_reduction_free_positions(problem::ProblemData)
    cone_positions = Set(_phase1_active_positions(problem))
    return [index for index in eachindex(problem.objective_vector_raw) if !(index in cone_positions)]
end

function _facial_reduction_trace_row(problem::ProblemData)
    row = zeros(ExactRational, size(problem.A, 1))
    for index in problem.positive_scalars
        row .+= problem.A[:, index]
    end
    for block in problem.blocks
        for diagonal in block.diagonal_positions
            row .+= problem.A[:, diagonal]
        end
    end
    return row
end

function _build_facial_reduction_oracle(
    problem::ProblemData,
    ::Type{F},
    ;
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
) where {F<:AbstractFloat}
    row_count = size(problem.A, 1)
    cone_positions = _phase1_active_positions(problem)
    oracle_equalities = _facial_reduction_oracle_equalities(
        problem;
        normalization_row,
    )
    oracle_equalities === nothing && return nothing
    equality_matrix, equality_rhs = oracle_equalities
    A_eq = _to_working_sparse_matrix(F, equality_matrix)
    b_eq = _to_working_array(F, equality_rhs)

    scalar_rows = length(problem.positive_scalars)
    psd_rows = sum(length(block.local_positions) for block in problem.blocks)
    total_cone_dimension = scalar_rows + psd_rows
    row_indices = Int[]
    column_indices = Int[]
    values = F[]
    h = zeros(F, total_cone_dimension)
    cones = Hypatia.Cones.Cone{F}[]

    row = 1
    if scalar_rows > 0
        push!(cones, Hypatia.Cones.Nonnegative{F}(scalar_rows))
        for index in problem.positive_scalars
            for equality_index in 1:row_count
                coefficient = problem.A[equality_index, index]
                iszero(coefficient) && continue
                push!(row_indices, row)
                push!(column_indices, equality_index)
                push!(values, -_to_working_float(F, coefficient))
            end
            row += 1
        end
    end

    rt2 = sqrt(F(2))
    for block in problem.blocks
        push!(cones, Hypatia.Cones.PosSemidefTri{F,F}(length(block.local_positions)))
        for (local_index, (i, j)) in enumerate(block.local_positions)
            row_index = row + local_index - 1
            scale = i == j ? one(F) : rt2
            position = block.global_positions[local_index]
            for equality_index in 1:row_count
                coefficient = problem.A[equality_index, position]
                iszero(coefficient) && continue
                push!(row_indices, row_index)
                push!(column_indices, equality_index)
                push!(values, -scale * _to_working_float(F, coefficient))
            end
        end
        row += length(block.local_positions)
    end

    @assert row == total_cone_dimension + 1
    G = sparse(row_indices, column_indices, values, total_cone_dimension, row_count)
    c = zeros(F, row_count)
    return Hypatia.Models.Model{F}(c, A_eq, b_eq, G, h, cones)
end

function _facial_reduction_oracle_allows_candidate_status(status)
    return status in (
        Hypatia.Solvers.Optimal,
        Hypatia.Solvers.NearOptimal,
        Hypatia.Solvers.SlowProgress,
    )
end

function _facial_reduction_oracle_recommends_precision_retry(status)
    # A near infeasibility certificate is deliberately tolerance-based. Retry
    # it at the next configured precision before concluding that no normalized
    # exposing vector exists.
    return status == Hypatia.Solvers.NearPrimalInfeasible
end

function _facial_reduction_oracle_attempt(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{HF},
    ;
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
    syssolver_override::Union{Nothing,Symbol} = nothing,
    return_details::Bool = false,
) where {HF<:AbstractFloat}
    return _with_float_precision(HF, opt.settings.working_precision, function (::Type{HF})
        attempt_result(candidate, retry_recommended) = return_details ?
            (candidate = candidate, retry_recommended = retry_recommended) : candidate
        oracle_start_time = time_ns()
        oracle_recorded = false
        record_oracle(iterations::Integer = 0) = begin
            oracle_recorded && return
            _record_oracle_attempt!(
                iterations,
                (time_ns() - oracle_start_time) / 1.0e9,
            )
            oracle_recorded = true
            return
        end
        model = _build_facial_reduction_oracle(
            problem,
            HF;
            normalization_row,
        )
        if model === nothing
            _log(opt, "Facial reduction oracle unavailable: exact normalization equalities are inconsistent")
            record_oracle()
            return attempt_result(nothing, false)
        end
        syssolver, use_dense_model, preprocess = _hypatia_phase1_syssolver(
            opt.settings,
            HF;
            choice_override = syssolver_override,
        )
        tolerance_kwargs = _phase1_hypatia_tolerance_kwargs(opt.settings, HF)
        solver = Hypatia.Solvers.Solver{HF}(
            ;
            verbose = false,
            iter_limit = opt.settings.phase1_hypatia_iter_limit,
            tolerance_kwargs...,
            preprocess = preprocess,
            reduce = false,
            syssolver = syssolver,
            use_dense_model = use_dense_model,
        )
        start_time = time_ns()
        try
            _with_filtered_hypatia_logger() do
                Hypatia.Solvers.load(solver, model)
                Hypatia.Solvers.solve(solver)
            end
        catch err
            failure = if err isa SingularException
                "singular numerical system"
            else
                summary = _exception_message(err)
                isempty(summary) ? _exception_type_name(err) : summary
            end
            _log(
                opt,
                "Facial reduction oracle unavailable: $(failure)",
            )
            record_oracle()
            return attempt_result(nothing, true)
        end
        elapsed_sec = (time_ns() - start_time) / 1.0e9
        status = Hypatia.Solvers.get_status(solver)
        if !_facial_reduction_oracle_allows_candidate_status(status)
            _log(
                opt,
                "Facial reduction oracle: status=$(status), time=$(@sprintf("%.2f", elapsed_sec))s",
            )
            record_oracle(Hypatia.Solvers.get_num_iters(solver))
            return attempt_result(
                nothing,
                _facial_reduction_oracle_recommends_precision_retry(status),
            )
        end
        candidate = try
            vec(collect(Hypatia.Solvers.get_x(solver)))
        catch
            nothing
        end
        if candidate === nothing
            record_oracle(Hypatia.Solvers.get_num_iters(solver))
            return attempt_result(nothing, true)
        end
        if !all(isfinite, candidate)
            record_oracle(Hypatia.Solvers.get_num_iters(solver))
            return attempt_result(nothing, true)
        end
        slow_progress_note =
            status == Hypatia.Solvers.SlowProgress ? "; trying current iterate" : ""
        _log(
            opt,
            "Facial reduction oracle: status=$(status), iter=$(Hypatia.Solvers.get_num_iters(solver)), time=$(@sprintf("%.2f", elapsed_sec))s$(slow_progress_note)",
        )
        record_oracle(Hypatia.Solvers.get_num_iters(solver))
        return attempt_result(candidate, false)
    end)
end

function _facial_reduction_slack(problem::ProblemData, y::Vector{F}) where {F<:AbstractFloat}
    s = transpose(_to_working_array(F, problem.A)) * y
    scalar_slack = Dict{Int,F}()
    for index in problem.positive_scalars
        scalar_slack[index] = s[index]
    end
    block_slack = Dict{Int,Matrix{F}}()
    for (block_index, block) in enumerate(problem.blocks)
        block_slack[block_index] = _dual_vector_to_matrix(s, block)
    end
    return scalar_slack, block_slack
end

function _facial_reduction_slack(
    problem::ProblemData,
    y::Vector{ExactRational};
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
)
    y_nemo = _to_nemo_matrix(reshape(y, :, 1))
    s = if isempty(y)
        zeros(ExactRational, size(problem.A, 2))
    else
        product = _with_nemo_error("exact facial-reduction slack multiplication") do
            _facial_reduction_A_transpose!(cache, problem) * y_nemo
        end
        vec(_from_nemo_matrix(product))
    end
    scalar_slack = Dict{Int,ExactRational}()
    for index in problem.positive_scalars
        scalar_slack[index] = s[index]
    end
    block_slack = Dict{Int,Matrix{ExactRational}}()
    for (block_index, block) in enumerate(problem.blocks)
        block_slack[block_index] = _dual_vector_to_matrix(s, block)
    end
    return s, scalar_slack, block_slack
end

function _facial_reduction_oracle_tolerances(
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    tolerances = _recovery_tolerances(settings, F)
    coarse = max(F(1.0e-4), _to_working_float(F, settings.facial_reduction_exposure_tolerance))
    if isempty(tolerances) || coarse > first(tolerances)
        pushfirst!(tolerances, coarse)
    end
    append!(
        tolerances,
        F[F(1.0e-3), F(1.0e-4), F(1.0e-5), F(1.0e-6), F(1.0e-7)],
    )
    filter!(tolerance -> tolerance > zero(F), tolerances)
    return sort!(unique(tolerances); rev = true)
end

function _facial_reduction_oracle_equalities(
    problem::ProblemData;
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
)
    equality_rows = Vector{Vector{ExactRational}}()
    equality_rhs = ExactRational[]
    for position in _facial_reduction_free_positions(problem)
        push!(equality_rows, collect(problem.A[:, position]))
        push!(equality_rhs, 0 // 1)
    end
    push!(equality_rows, copy(problem.b))
    push!(equality_rhs, 0 // 1)
    length(normalization_row) == size(problem.A, 1) ||
        error("Facial-reduction oracle normalization row has the wrong length.")
    push!(equality_rows, copy(normalization_row))
    push!(equality_rhs, 1 // 1)

    equality_matrix = Matrix(transpose(hcat(equality_rows...)))
    return _independent_affine_equalities(equality_matrix, equality_rhs)
end

function _slack_has_exposure(
    scalar_slack::Dict{Int,ExactRational},
    block_slack::Dict{Int,Matrix{ExactRational}},
)
    any(value -> value > 0 // 1, values(scalar_slack)) && return true
    return any(matrix -> any(!iszero, matrix), values(block_slack))
end

function _dual_slack_has_exact_certificate(
    problem::ProblemData,
    slack::Vector{ExactRational},
)
    certificate_matrix = zeros(
        ExactRational,
        length(slack) + 1,
        size(problem.A, 1),
    )
    certificate_matrix[1:length(slack), :] = transpose(problem.A)
    certificate_matrix[end, :] = transpose(problem.b)
    certificate_rhs = vcat(slack, 0 // 1)
    return _solve_affine_system(certificate_matrix, certificate_rhs) !== nothing
end

function _exact_exposing_slack_from_stable_projective_entries(
    opt::Optimizer,
    problem::ProblemData,
    numeric_slack::AbstractVector{F},
    ::Type{F},
    source::AbstractString,
) where {F<:AbstractFloat}
    pivot = argmax(index -> abs(numeric_slack[index]), eachindex(numeric_slack))
    pivot_value = numeric_slack[pivot]
    iszero(pivot_value) && return nothing
    projective_slack = numeric_slack ./ pivot_value
    tolerances = _facial_reduction_oracle_tolerances(opt.settings, F)
    length(tolerances) >= 2 || return nothing
    rounded = [
        ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for
            value in projective_slack
        ] for tolerance in tolerances
    ]

    numeric_A_transpose = transpose(_to_working_array(F, problem.A))
    numeric_y = try
        numeric_A_transpose \ projective_slack
    catch
        return nothing
    end
    all(isfinite, numeric_y) || return nothing

    for tolerance_index in 1:(length(tolerances) - 1)
        coarse = rounded[tolerance_index]
        fine = rounded[tolerance_index + 1]
        stable_positions = [
            index for index in eachindex(coarse) if coarse[index] == fine[index]
        ]
        pivot in stable_positions || push!(stable_positions, pivot)
        length(stable_positions) >= 2 || continue

        equality_rows = Vector{Vector{ExactRational}}([copy(problem.b)])
        equality_rhs = ExactRational[0 // 1]
        for position in stable_positions
            push!(equality_rows, collect(problem.A[:, position]))
            push!(equality_rhs, position == pivot ? 1 // 1 : coarse[position])
        end
        equalities = _independent_affine_equalities(
            Matrix(transpose(hcat(equality_rows...))),
            equality_rhs,
        )
        equalities === nothing && continue
        equality_matrix, independent_rhs = equalities
        affine = _solve_affine_system(Matrix(equality_matrix), independent_rhs)
        affine === nothing && continue
        particular, nullspace = affine
        numeric_coordinates = if size(nullspace, 2) == 0
            F[]
        else
            _to_working_array(F, nullspace) \
                (numeric_y - _to_working_array(F, particular))
        end
        all(isfinite, numeric_coordinates) || continue

        for coordinate_tolerance in tolerances
            y = if isempty(numeric_coordinates)
                particular
            else
                rational_coordinates = ExactRational[
                    rationalize(
                        BigInt,
                        BigFloat(value);
                        tol = BigFloat(coordinate_tolerance),
                    ) for value in numeric_coordinates
                ]
                particular + nullspace * rational_coordinates
            end
            s, scalar_slack, block_slack = _facial_reduction_slack(problem, y)
            s[pivot] == 1 // 1 || continue
            iszero(dot(problem.b, y)) || continue
            all(index -> iszero(s[index]), _facial_reduction_free_positions(problem)) ||
                continue
            all(value -> value >= 0 // 1, values(scalar_slack)) || continue
            all(matrix -> _positive_semidefinite_exact(matrix), values(block_slack)) || continue
            _slack_has_exposure(scalar_slack, block_slack) || continue
            _log(
                opt,
                "Facial reduction: recovered exact exposing slack from $(source) " *
                "using $(length(stable_positions)) stable projective entries " *
                "(entry_tol=$(_format_metric(tolerances[tolerance_index])), " *
                "coordinate_tol=$(_format_metric(coordinate_tolerance)))",
            )
            return scalar_slack, block_slack
        end
    end
    return nothing
end

function _exact_exposing_slack_from_numeric_slack(
    opt::Optimizer,
    problem::ProblemData,
    numeric_slack::AbstractVector{F},
    ::Type{F},
    source::AbstractString,
) where {F<:AbstractFloat}
    length(numeric_slack) == length(problem.objective_vector_raw) || return nothing
    all(isfinite, numeric_slack) || return nothing

    free_positions = _facial_reduction_free_positions(problem)
    pivot = argmax(index -> abs(numeric_slack[index]), eachindex(numeric_slack))
    pivot_value = numeric_slack[pivot]
    for tolerance in _facial_reduction_oracle_tolerances(opt.settings, F)
        candidates = Tuple{String,Vector{ExactRational}}[]
        if !iszero(pivot_value)
            projective_slack = ExactRational[
                rationalize(
                    BigInt,
                    BigFloat(value / pivot_value);
                    tol = BigFloat(tolerance),
                ) for value in numeric_slack
            ]
            projective_slack[pivot] = 1 // 1
            push!(candidates, ("projective", projective_slack))
        end
        direct_slack = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for
            value in numeric_slack
        ]
        push!(candidates, ("direct", direct_slack))

        for (recovery_mode, slack) in candidates
        all(index -> iszero(slack[index]), free_positions) || continue

        scalar_slack = Dict{Int,ExactRational}()
        for index in problem.positive_scalars
            scalar_slack[index] = slack[index]
        end
        all(value -> value >= 0 // 1, values(scalar_slack)) || continue

        block_slack = Dict{Int,Matrix{ExactRational}}()
        for (block_index, block) in enumerate(problem.blocks)
            block_slack[block_index] = _dual_vector_to_matrix(slack, block)
        end
        all(matrix -> _positive_semidefinite_exact(matrix), values(block_slack)) || continue
        _slack_has_exposure(scalar_slack, block_slack) || continue
        _dual_slack_has_exact_certificate(problem, slack) || continue

        _log(
            opt,
            "Facial reduction: recovered exact exposing slack from $(source) " *
            "($(recovery_mode), tol=$(_format_metric(tolerance)))",
        )
        return scalar_slack, block_slack
        end
    end

    return nothing
end

function _exact_facial_reduction_oracle_slack_from_rounded_slack(
    opt::Optimizer,
    problem::ProblemData,
    oracle_point::Vector{F},
    ::Type{F},
) where {F<:AbstractFloat}
    numeric_slack = transpose(_to_working_array(F, problem.A)) * oracle_point
    return _exact_exposing_slack_from_numeric_slack(
        opt,
        problem,
        numeric_slack,
        F,
        "oracle dual slack",
    )
end

function _exact_facial_reduction_oracle_slack(
    opt::Optimizer,
    problem::ProblemData,
    oracle_point::Vector{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
) where {F<:AbstractFloat}
    free_positions = _facial_reduction_free_positions(problem)
    trace_row = normalization_row
    oracle_equalities = _facial_reduction_oracle_equalities(
        problem;
        normalization_row,
    )
    oracle_equalities === nothing && return nothing
    equality_matrix, equality_rhs = oracle_equalities
    oracle_affine = _solve_affine_system(Matrix(equality_matrix), equality_rhs)
    oracle_affine === nothing && return nothing
    particular, nullspace = oracle_affine
    coordinates = if size(nullspace, 2) == 0
        F[]
    else
        _to_working_array(F, nullspace) \
            (oracle_point - _to_working_array(F, particular))
    end

    for tolerance in _facial_reduction_oracle_tolerances(opt.settings, F)
        y = if isempty(coordinates)
            particular
        else
            rational_coordinates = ExactRational[
                rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for
                value in coordinates
            ]
            particular + nullspace * rational_coordinates
        end
        s, scalar_slack, block_slack = _facial_reduction_slack(problem, y; cache)

        all(index -> iszero(s[index]), free_positions) || continue
        iszero(dot(problem.b, y)) || continue
        dot(trace_row, y) == 1 // 1 || continue
        all(value -> value >= 0 // 1, values(scalar_slack)) || continue
        all(matrix -> _positive_semidefinite_exact(matrix), values(block_slack)) || continue

        _log(
            opt,
            "Facial reduction oracle: recovered exact exposing-vector certificate (tol=$(_format_metric(tolerance)))",
        )
        return scalar_slack, block_slack
    end

    rounded_slack = _exact_facial_reduction_oracle_slack_from_rounded_slack(
        opt,
        problem,
        oracle_point,
        F,
    )
    rounded_slack === nothing || return rounded_slack

    _log(opt, "Facial reduction oracle: no exact exposing-vector certificate recovered")
    return nothing
end

struct _FacialReductionEvidence{F<:AbstractFloat}
    kind::Symbol
    source::String
    vector::Vector{F}
end

struct _CertifiedFacialReduction
    source::String
    exposed_scalars::Vector{Int}
    keep_bases::Dict{Int,Matrix{ExactRational}}
    exposing_slack::Union{Nothing,Vector{ExactRational}}
end

_CertifiedFacialReduction(
    source::AbstractString,
    exposed_scalars::Vector{Int},
    keep_bases::Dict{Int,Matrix{ExactRational}},
) = _CertifiedFacialReduction(String(source), exposed_scalars, keep_bases, nothing)

struct _SieveRowCertificate
    source::String
    multiplier::Vector{ExactRational}
    row::Vector{ExactRational}
    reduction::_CertifiedFacialReduction
end

const _FACIAL_REDUCTION_CACHE_MAGIC = "RationalSDP facial reduction cache"
const _FACIAL_REDUCTION_CACHE_VERSION = 2

function _facial_reduction_cache_path(path::AbstractString)
    stripped = strip(path)
    return isempty(stripped) ? nothing : stripped
end

function _facial_reduction_cache_records(payload)
    payload isa NamedTuple ||
        throw(ArgumentError("Facial reduction cache is not a RationalSDP cache payload."))
    (:magic in keys(payload) && payload.magic == _FACIAL_REDUCTION_CACHE_MAGIC) ||
        throw(ArgumentError("Facial reduction cache has an unrecognized file header."))
    (:version in keys(payload) && payload.version == _FACIAL_REDUCTION_CACHE_VERSION) ||
        throw(ArgumentError("Unsupported facial reduction cache version."))
    (:records in keys(payload) && payload.records isa AbstractVector) ||
        throw(ArgumentError("Facial reduction cache is missing its record list."))
    return Any[record for record in payload.records]
end

function _read_facial_reduction_cache(path::AbstractString; missing_ok::Bool = false)
    full_path = abspath(path)
    if !isfile(full_path)
        missing_ok && return Any[]
        throw(ArgumentError("Facial reduction cache file does not exist: $(full_path)"))
    end
    payload = try
        open(full_path, "r") do io
            Serialization.deserialize(io)
        end
    catch err
        throw(ArgumentError("Could not read facial reduction cache $(full_path): $(err)"))
    end
    return _facial_reduction_cache_records(payload)
end

function _write_facial_reduction_cache(
    path::AbstractString,
    records::Vector{Any};
    overwrite::Bool = true,
)
    full_path = abspath(path)
    mkpath(dirname(full_path))
    payload = (
        magic = _FACIAL_REDUCTION_CACHE_MAGIC,
        version = _FACIAL_REDUCTION_CACHE_VERSION,
        records = copy(records),
    )
    temporary_path, io = mktemp(dirname(full_path); cleanup = false)
    try
        Serialization.serialize(io, payload)
        close(io)
        mv(temporary_path, full_path; force = overwrite)
    catch
        isopen(io) && close(io)
        rm(temporary_path; force = true)
        rethrow()
    end
    return full_path
end

function _prepare_facial_reduction_cache!(opt::Optimizer)
    empty!(opt.facial_reduction_save_records)
    opt.facial_reduction_loaded_records = nothing

    save_path = _facial_reduction_cache_path(opt.settings.facial_reduction_save_file)
    load_path = _facial_reduction_cache_path(opt.settings.facial_reduction_load_file)
    if save_path !== nothing &&
       load_path !== nothing &&
       abspath(save_path) == abspath(load_path)
        records = _read_facial_reduction_cache(load_path; missing_ok = true)
        opt.facial_reduction_loaded_records = records
        append!(opt.facial_reduction_save_records, records)
        _record_approximate_cache_memory!(:facial_reduction, opt.facial_reduction_loaded_records)
        _record_approximate_cache_memory!(:facial_reduction, opt.facial_reduction_save_records)
    end
    return
end

function _loaded_facial_reduction_records!(opt::Optimizer)
    opt.facial_reduction_loaded_records !== nothing &&
        return opt.facial_reduction_loaded_records

    load_path = _facial_reduction_cache_path(opt.settings.facial_reduction_load_file)
    if load_path === nothing
        opt.facial_reduction_loaded_records = Any[]
    else
        records = _read_facial_reduction_cache(load_path)
        opt.facial_reduction_loaded_records = records
        _record_approximate_cache_memory!(:facial_reduction, records)
        _log(
            opt,
            "Facial reduction: loaded $(length(records)) cached reduction record(s) from $(abspath(load_path))",
        )
    end
    return opt.facial_reduction_loaded_records
end

function _checkpoint_loaded_facial_reductions_to_distinct_file(opt::Optimizer)
    save_path = _facial_reduction_cache_path(opt.settings.facial_reduction_save_file)
    load_path = _facial_reduction_cache_path(opt.settings.facial_reduction_load_file)
    return save_path !== nothing &&
           load_path !== nothing &&
           abspath(save_path) != abspath(load_path)
end

function _facial_reduction_block_signature(block::BlockStructure)
    return (
        size = block.size,
        global_positions = copy(block.global_positions),
        local_positions = copy(block.local_positions),
        diagonal_positions = copy(block.diagonal_positions),
    )
end

function _facial_reduction_problem_signature(problem::ProblemData)
    return (
        dimension = length(problem.objective_vector_raw),
        equation_count = size(problem.A, 1),
        positive_scalars = copy(problem.positive_scalars),
        blocks = [_facial_reduction_block_signature(block) for block in problem.blocks],
    )
end

function _facial_reduction_signature_matches_signature(candidate, signature)
    signature isa NamedTuple || return false
    required = (:dimension, :equation_count, :positive_scalars, :blocks)
    all(name -> name in keys(signature), required) || return false
    signature.dimension == candidate.dimension || return false
    signature.equation_count == candidate.equation_count || return false
    collect(signature.positive_scalars) == collect(candidate.positive_scalars) || return false
    length(signature.blocks) == length(candidate.blocks) || return false
    for (candidate_block, block_signature) in zip(candidate.blocks, signature.blocks)
        block_signature isa NamedTuple || return false
        block_required = (:size, :global_positions, :local_positions, :diagonal_positions)
        all(name -> name in keys(block_signature), block_required) || return false
        block_signature.size == candidate_block.size || return false
        collect(block_signature.global_positions) ==
            collect(candidate_block.global_positions) || return false
        collect(block_signature.local_positions) ==
            collect(candidate_block.local_positions) || return false
        collect(block_signature.diagonal_positions) ==
            collect(candidate_block.diagonal_positions) || return false
    end
    return true
end

function _facial_reduction_signature_matches(problem::ProblemData, signature)
    return _facial_reduction_signature_matches_signature(
        _facial_reduction_problem_signature(problem),
        signature,
    )
end

function _facial_reduction_record(
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
)
    keep_bases = [
        (block_index = block_index, basis = copy(reduction.keep_bases[block_index])) for
        block_index in sort(collect(keys(reduction.keep_bases)))
    ]
    return (
        signature = _facial_reduction_problem_signature(problem),
        source = reduction.source,
        exposed_scalars = copy(reduction.exposed_scalars),
        keep_bases = keep_bases,
        exposing_slack =
            reduction.exposing_slack === nothing ? nothing : copy(reduction.exposing_slack),
    )
end

function _facial_reductions_same_face(
    problem::ProblemData,
    left::_CertifiedFacialReduction,
    right::_CertifiedFacialReduction,
)
    Set(left.exposed_scalars) == Set(right.exposed_scalars) || return false
    for (block_index, block) in enumerate(problem.blocks)
        left_basis = get(
            left.keep_bases,
            block_index,
            Matrix{ExactRational}(I, block.size, block.size),
        )
        right_basis = get(
            right.keep_bases,
            block_index,
            Matrix{ExactRational}(I, block.size, block.size),
        )
        size(left_basis) == size(right_basis) || return false
        _exact_column_rank(hcat(left_basis, right_basis)) == size(left_basis, 2) ||
            return false
    end
    return true
end

function _record_successful_facial_reduction!(
    opt::Optimizer,
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
    ;
    supersedes::Union{Nothing,_CertifiedFacialReduction} = nothing,
)
    save_path = _facial_reduction_cache_path(opt.settings.facial_reduction_save_file)
    save_path === nothing && return
    attached_exposing_slack = false
    if reduction.exposing_slack === nothing
        upgraded = _with_exact_exposing_slack(problem, reduction, opt.settings)
        if upgraded !== nothing
            reduction = upgraded
            attached_exposing_slack = true
        end
    end
    target = supersedes === nothing ? reduction : supersedes
    replace_indices = Int[]
    for (record_index, record) in enumerate(opt.facial_reduction_save_records)
        record isa NamedTuple || continue
        (:signature in keys(record)) || continue
        _facial_reduction_signature_matches(problem, record.signature) || continue
        cached = try
            _cached_facial_reduction(record)
        catch
            nothing
        end
        cached === nothing && continue
        _facial_reductions_same_face(problem, cached, target) || continue
        push!(replace_indices, record_index)
    end
    new_record = _facial_reduction_record(problem, reduction)
    if length(replace_indices) == 1 &&
       isequal(opt.facial_reduction_save_records[only(replace_indices)], new_record)
        return
    end
    if attached_exposing_slack
        _log(
            opt,
            "Facial reduction: attached a composable exact exposing slack before checkpointing",
        )
    end
    if isempty(replace_indices)
        push!(opt.facial_reduction_save_records, new_record)
    else
        opt.facial_reduction_save_records[first(replace_indices)] = new_record
        for record_index in Iterators.reverse(replace_indices[2:end])
            deleteat!(opt.facial_reduction_save_records, record_index)
        end
    end
    _record_approximate_cache_memory!(:facial_reduction, opt.facial_reduction_save_records)
    full_path = _write_facial_reduction_cache(
        save_path,
        opt.facial_reduction_save_records,
    )
    _log(
        opt,
        "Facial reduction: checkpointed $(length(opt.facial_reduction_save_records)) reduction record(s) to $(full_path)",
    )
    return
end

function _exact_matrix_from_cache(value)
    value isa AbstractMatrix ||
        throw(ArgumentError("Cached facial reduction basis is not a matrix."))
    return ExactRational[
        _exact_rational(value[row, column]) for
        row in axes(value, 1), column in axes(value, 2)
    ]
end

function _cached_facial_reduction(record)
    record isa NamedTuple || return nothing
    (:source in keys(record)) || return nothing
    (:exposed_scalars in keys(record)) || return nothing
    (:keep_bases in keys(record)) || return nothing

    exposed_scalars = unique(sort(Int[Int(index) for index in record.exposed_scalars]))
    keep_bases = Dict{Int,Matrix{ExactRational}}()
    for item in record.keep_bases
        item isa NamedTuple || return nothing
        (:block_index in keys(item) && :basis in keys(item)) || return nothing
        block_index = Int(item.block_index)
        haskey(keep_bases, block_index) && return nothing
        keep_bases[block_index] = _exact_matrix_from_cache(item.basis)
    end
    exposing_slack = if :exposing_slack in keys(record) && record.exposing_slack !== nothing
        ExactRational[_exact_rational(value) for value in record.exposing_slack]
    else
        nothing
    end
    return _CertifiedFacialReduction(
        String(record.source),
        exposed_scalars,
        keep_bases,
        exposing_slack,
    )
end

function _exact_column_rank(matrix::Matrix{ExactRational})
    size(matrix, 2) == 0 && return 0
    _, pivots = _rref(hcat(matrix, zeros(ExactRational, size(matrix, 1))))
    return length(pivots)
end

function _add_block_exposing_matrix!(
    slack::Vector{ExactRational},
    block::BlockStructure,
    matrix::Matrix{ExactRational},
)
    size(matrix) == (block.size, block.size) ||
        error("Exact exposing slack matrix has the wrong PSD block size.")
    for (local_index, (i, j)) in enumerate(block.local_positions)
        slack[block.global_positions[local_index]] +=
            i == j ? matrix[i, j] : 2 * matrix[i, j]
    end
    return slack
end

function _exact_exposing_slack_violation(
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
)
    slack = reduction.exposing_slack
    slack === nothing && return "no exact exposing slack is attached"
    length(slack) == length(problem.objective_vector_raw) ||
        return "exact exposing slack has length $(length(slack)), expected $(length(problem.objective_vector_raw))"
    all(index -> iszero(slack[index]), _facial_reduction_free_positions(problem)) ||
        return "exact exposing slack is nonzero in a free position"

    exposed_scalars = Set(reduction.exposed_scalars)
    for position in problem.positive_scalars
        value = slack[position]
        value >= 0 // 1 ||
            return "exact exposing slack is negative at scalar position $(position)"
        (position in exposed_scalars) == (value > 0 // 1) ||
            return "exact exposing slack does not expose the recorded scalar face at position $(position)"
    end

    for (block_index, block) in enumerate(problem.blocks)
        block_slack = _dual_vector_to_matrix(slack, block)
        _positive_semidefinite_exact(block_slack) ||
            return "exact exposing slack is not positive semidefinite in PSD block $(block_index)"
        recorded_keep = get(reduction.keep_bases, block_index, nothing)
        if recorded_keep === nothing
            all(iszero, block_slack) ||
                return "exact exposing slack exposes an unrecorded face in PSD block $(block_index)"
            continue
        end
        structure_violation =
            _cached_keep_basis_structure_violation(problem, block_index, recorded_keep)
        structure_violation === nothing || return structure_violation
        slack_keep = _nullspace_basis_exact(block_slack)
        size(slack_keep, 2) == size(recorded_keep, 2) ||
            return "exact exposing slack has a different kernel dimension in PSD block $(block_index)"
        _exact_column_rank(hcat(slack_keep, recorded_keep)) == size(recorded_keep, 2) ||
            return "exact exposing slack has a different kernel in PSD block $(block_index)"
    end

    any(!iszero, slack) || return "exact exposing slack is zero"
    indices = findall(!iszero, slack)
    affine_violation = _affine_form_violation(problem, indices, slack[indices])
    affine_violation === nothing ||
        return "exact exposing slack does not vanish on the affine slice ($(affine_violation))"
    return nothing
end

function _reconstruct_exact_exposing_slack(
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
    settings::Settings,
)
    slack = zeros(ExactRational, length(problem.objective_vector_raw))
    for position in reduction.exposed_scalars
        1 <= position <= length(slack) || return nothing
        slack[position] += 1 // 1
    end

    directions_by_block = Dict{Int,Vector{Vector{ExactRational}}}()
    for block_index in sort(collect(keys(reduction.keep_bases)))
        keep_basis = reduction.keep_bases[block_index]
        _cached_keep_basis_structure_violation(problem, block_index, keep_basis) === nothing ||
            return nothing
        removed = _nullspace_basis_exact(Matrix(transpose(keep_basis)))
        directions_by_block[block_index] = [
            collect(view(removed, :, column)) for column in axes(removed, 2)
        ]
    end

    unresolved_blocks = sort(collect(keys(directions_by_block)))
    if length(unresolved_blocks) > 1
        joint = _multiblock_weighted_subspace_exposure(
            problem,
            directions_by_block,
            settings,
            _facial_reduction_float_type(settings),
        )
        if joint !== nothing
            for block_index in unresolved_blocks
                directions = hcat(directions_by_block[block_index]...)
                block_slack = directions * joint.weights[block_index] * transpose(directions)
                _add_block_exposing_matrix!(
                    slack,
                    problem.blocks[block_index],
                    block_slack,
                )
            end
            empty!(unresolved_blocks)
        end
    end

    cache = _FacialReductionExactCache(problem)
    for block_index in unresolved_blocks
        block = problem.blocks[block_index]
        directions = directions_by_block[block_index]
        exposure = _block_weighted_subspace_exposure(
            problem,
            block,
            directions,
            settings,
            _facial_reduction_float_type(settings),
        )
        block_slack = if exposure === nothing ||
                         length(exposure.directions) != length(directions)
            matrix = zeros(ExactRational, block.size, block.size)
            for direction in directions
                certificate = _block_face_direction_certificate(
                    problem,
                    block,
                    direction;
                    cache,
                    block_index,
                )
                certificate.kind == :none && return nothing
                matrix .+= direction * transpose(direction)
            end
            matrix
        else
            direction_matrix = hcat(directions...)
            direction_matrix * exposure.weight * transpose(direction_matrix)
        end
        _add_block_exposing_matrix!(slack, block, block_slack)
    end

    candidate = _CertifiedFacialReduction(
        reduction.source,
        copy(reduction.exposed_scalars),
        copy(reduction.keep_bases),
        slack,
    )
    _exact_exposing_slack_violation(problem, candidate) === nothing || return nothing
    return slack
end

function _with_exact_exposing_slack(
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
    settings::Settings,
)
    if reduction.exposing_slack !== nothing
        return _exact_exposing_slack_violation(problem, reduction) === nothing ? reduction :
               nothing
    end
    slack = _reconstruct_exact_exposing_slack(problem, reduction, settings)
    slack === nothing && return nothing
    return _CertifiedFacialReduction(
        reduction.source,
        copy(reduction.exposed_scalars),
        copy(reduction.keep_bases),
        slack,
    )
end

function _cached_scalar_face_violation(problem::ProblemData, position::Int)
    position in problem.positive_scalars ||
        return "scalar position $(position) is not an active positive scalar"
    problem.affine === nothing && return "no exact affine parametrization is available"
    particular, nullspace = problem.affine
    1 <= position <= length(particular) ||
        return "scalar position $(position) is outside the problem dimension"
    iszero(particular[position]) ||
        return "scalar position $(position) has affine particular value $(particular[position])"
    if any(!iszero, view(nullspace, position, :))
        return "scalar position $(position) is not fixed by the affine nullspace"
    end
    return nothing
end

function _cached_keep_basis_structure_violation(
    problem::ProblemData,
    block_index::Int,
    keep_basis::Matrix{ExactRational},
)
    1 <= block_index <= length(problem.blocks) ||
        return "PSD block $(block_index) does not exist"
    block = problem.blocks[block_index]
    size(keep_basis, 1) == block.size ||
        return "PSD block $(block_index) cache basis has $(size(keep_basis, 1)) row(s), expected $(block.size)"
    0 <= size(keep_basis, 2) < block.size ||
        return "PSD block $(block_index) cache basis does not reduce the block"
    _exact_column_rank(keep_basis) == size(keep_basis, 2) ||
        return "PSD block $(block_index) cache basis columns are linearly dependent"
    return nothing
end

function _cached_keep_basis_violation(
    problem::ProblemData,
    block_index::Int,
    keep_basis::Matrix{ExactRational},
    settings::Settings = Settings(),
)
    structure_violation =
        _cached_keep_basis_structure_violation(problem, block_index, keep_basis)
    structure_violation === nothing || return structure_violation

    block = problem.blocks[block_index]
    removed_directions = _nullspace_basis_exact(Matrix(transpose(keep_basis)))
    directions = [
        collect(view(removed_directions, :, column)) for
        column in axes(removed_directions, 2)
    ]
    weighted_exposure = _block_weighted_subspace_exposure(
        problem,
        block,
        directions,
        settings,
        _facial_reduction_float_type(settings),
    )
    if weighted_exposure !== nothing &&
       length(weighted_exposure.directions) == length(directions)
        return nothing
    end

    cache = _FacialReductionExactCache(problem)
    last_violation = nothing
    for direction in directions
        certificate = _block_face_direction_certificate(
            problem,
            block,
            direction;
            cache,
            block_index,
        )
        certificate.kind == :none || continue
        last_violation = certificate.violation
        break
    end
    last_violation === nothing && return nothing

    trace_violation = _block_trace_vanish_violation(
        problem,
        block,
        directions;
        cache,
        block_index,
    )
    trace_violation === nothing && return nothing
    return "PSD block $(block_index) cache face is not valid for the current affine slice ($(last_violation); $(trace_violation))"
end

function _cached_facial_reduction_violation(
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
    settings::Settings = Settings(),
)
    if isempty(reduction.exposed_scalars) && isempty(reduction.keep_bases)
        return "cached reduction has no exposed scalar or PSD face"
    end
    if reduction.exposing_slack !== nothing
        return _exact_exposing_slack_violation(problem, reduction)
    end
    for position in reduction.exposed_scalars
        violation = _cached_scalar_face_violation(problem, position)
        violation === nothing || return violation
    end

    directions_by_block = Dict{Int,Vector{Vector{ExactRational}}}()
    for block_index in sort(collect(keys(reduction.keep_bases)))
        keep_basis = reduction.keep_bases[block_index]
        violation =
            _cached_keep_basis_structure_violation(problem, block_index, keep_basis)
        violation === nothing || return violation
        removed_directions = _nullspace_basis_exact(Matrix(transpose(keep_basis)))
        directions_by_block[block_index] = [
            collect(view(removed_directions, :, column)) for
            column in axes(removed_directions, 2)
        ]
    end

    if length(directions_by_block) > 1
        joint_exposure = _multiblock_weighted_subspace_exposure(
            problem,
            directions_by_block,
            settings,
            _facial_reduction_float_type(settings),
        )
        joint_exposure === nothing || return nothing
    end

    for block_index in sort(collect(keys(reduction.keep_bases)))
        violation = _cached_keep_basis_violation(
            problem,
            block_index,
            reduction.keep_bases[block_index],
            settings,
        )
        violation === nothing || return violation
    end
    return nothing
end

function _apply_loaded_facial_reductions(
    opt::Optimizer,
    problem::ProblemData;
    return_details::Bool = false,
)
    records = _loaded_facial_reduction_records!(opt)
    isempty(records) && return return_details ?
        (problem = problem, applied = 0, matched = 0) : problem

    current = problem
    matched = 0
    applied = 0
    for record in records
        record isa NamedTuple || continue
        (:signature in keys(record)) || continue
        _facial_reduction_signature_matches(current, record.signature) || continue
        matched += 1

        reduction = try
            _cached_facial_reduction(record)
        catch err
            _log(opt, "Facial reduction: skipped malformed cache record ($(err))")
            continue
        end
        if reduction === nothing
            _log(opt, "Facial reduction: skipped malformed cache record")
            continue
        end
        violation = _cached_facial_reduction_violation(current, reduction, opt.settings)
        if violation !== nothing
            _log(opt, "Facial reduction: cached face did not validate ($(violation))")
            continue
        end

        reduction_with_slack =
            _with_exact_exposing_slack(current, reduction, opt.settings)
        if reduction_with_slack === nothing
            _log(
                opt,
                "Facial reduction: cached face validated, but an exact exposing slack could not be reconstructed",
            )
        else
            if reduction.exposing_slack === nothing
                _log(
                    opt,
                    "Facial reduction: upgraded cached face to a composable exact exposing-slack certificate",
                )
            end
            reduction = reduction_with_slack
        end

        # Cached records form a sequential chain of problem signatures.  Do
        # not rank-expand an intermediate record: strengthening it can skip
        # the problem on which the next record is keyed.  After the complete
        # chain is applied, the normal Phase-I facial-reduction path performs
        # rank expansion on any newly certified face.
        reduced_problem = _apply_facial_reduction(
            current,
            reduction.exposed_scalars,
            reduction.keep_bases,
            ;
            # The exact cache validation immediately above is the certificate
            # for this application.  Re-validating inside _apply_facial_reduction
            # duplicates the expensive exact row-space/PSD checks.
            certified = false,
            settings = opt.settings,
        )
        if reduced_problem.affine === nothing
            _log(
                opt,
                "Facial reduction: cached face produced an inconsistent affine system; ignoring it",
            )
            continue
        end
        if _checkpoint_loaded_facial_reductions_to_distinct_file(opt)
            # The save cache is a compact-coordinate replay chain, not merely
            # the reductions discovered after loading. Re-record each validated
            # loaded face against the current compact problem so a distinct
            # output file remains independently replayable.
            _record_successful_facial_reduction!(opt, current, reduction)
        end
        applied += 1
        old_barrier_dimension = _barrier_dimension(current)
        removed_psd_directions = sum(
            (current.blocks[index].size - size(reduction.keep_bases[index], 2) for
             index in keys(reduction.keep_bases));
            init = 0,
        )
        new_barrier_dimension = _barrier_dimension(reduced_problem)
        _record_reduction_round!(old_barrier_dimension, new_barrier_dimension; tentative = false)
        _log(
            opt,
            "Facial reduction: applied cached face from $(reduction.source), fixed $(length(reduction.exposed_scalars)) scalar cone direction(s) and removed $(removed_psd_directions) PSD direction(s)",
        )
        current = reduced_problem
    end

    if applied == 0
        if matched == 0
            _log(opt, "Facial reduction: no cached reduction record matched the current problem")
        else
            _log(opt, "Facial reduction: no cached reduction record validated for the current problem")
        end
    end
    return return_details ?
        (problem = current, applied = applied, matched = matched) : current
end

function _exact_slack_keep_bases(
    opt::Optimizer,
    problem::ProblemData,
    block_slack::Dict{Int,Matrix{ExactRational}},
    source::AbstractString,
)
    keep_bases = Dict{Int,Matrix{ExactRational}}()
    for (block_index, block) in enumerate(problem.blocks)
        slack = block_slack[block_index]
        any(!iszero, slack) || continue
        keep_basis = _nullspace_basis_exact(slack)
        if size(keep_basis, 2) == block.size
            continue
        end
        keep_bases[block_index] = keep_basis
        _log(
            opt,
            "Facial reduction: $(source) certified exact exposed face for PSD block $(block_index)",
        )
    end
    return keep_bases
end

function _exact_exposing_slack_vector(
    problem::ProblemData,
    scalar_slack::Dict{Int,ExactRational},
    block_slack::Dict{Int,Matrix{ExactRational}},
)
    slack = zeros(ExactRational, length(problem.objective_vector_raw))
    for (position, value) in scalar_slack
        slack[position] = value
    end
    for (block_index, matrix) in block_slack
        block = problem.blocks[block_index]
        size(matrix) == (block.size, block.size) ||
            error("Exact exposing slack matrix has the wrong size for PSD block $(block_index).")
        for (local_index, (i, j)) in enumerate(block.local_positions)
            slack[block.global_positions[local_index]] =
                i == j ? matrix[i, j] : 2 * matrix[i, j]
        end
    end
    return slack
end

function _certified_reduction_from_exact_slack(
    opt::Optimizer,
    problem::ProblemData,
    scalar_slack_exact::Dict{Int,ExactRational},
    block_slack_exact::Dict{Int,Matrix{ExactRational}},
    source::AbstractString,
)
    exposed_scalars = sort([
        index for index in problem.positive_scalars if
        get(scalar_slack_exact, index, zero(ExactRational)) > 0 // 1
    ])
    keep_bases = _exact_slack_keep_bases(opt, problem, block_slack_exact, source)
    if !isempty(exposed_scalars) || !isempty(keep_bases)
        _log(
            opt,
            "Facial reduction: $(source) exposed $(length(exposed_scalars)) scalar cone direction(s) and $(length(keep_bases)) PSD block face(s)",
        )
        exposing_slack =
            _exact_exposing_slack_vector(problem, scalar_slack_exact, block_slack_exact)
        return _CertifiedFacialReduction(
            String(source),
            exposed_scalars,
            keep_bases,
            exposing_slack,
        )
    end
    return nothing
end

function _sieve_row_reduction(
    opt::Optimizer,
    problem::ProblemData,
    multiplier::Vector{ExactRational},
    source::AbstractString,
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    row_override::Union{Nothing,Vector{ExactRational}} = nothing,
    rhs_override::Union{Nothing,ExactRational} = nothing,
)
    length(multiplier) == size(problem.A, 1) || return nothing
    all(iszero, multiplier) && return nothing

    row = row_override === nothing ? vec(transpose(multiplier) * problem.A) : row_override
    rhs = rhs_override === nothing ? dot(multiplier, problem.b) : rhs_override
    iszero(rhs) || return nothing

    free_positions = _facial_reduction_free_positions(problem)
    all(iszero, row[free_positions]) || return nothing

    scalar_slack = Dict{Int,ExactRational}(
        index => row[index] for index in problem.positive_scalars
    )
    block_slack = Dict{Int,Matrix{ExactRational}}(
        block_index => _dual_vector_to_matrix(row, block) for
        (block_index, block) in enumerate(problem.blocks)
    )
    all(value -> value >= 0 // 1, values(scalar_slack)) || return nothing

    # Most affine rows are not exposing PSD slacks.  Reject matrices that are
    # numerically and decisively indefinite before invoking the much more
    # expensive exact PSD test.  The screen is conservative: conversion or
    # eigensolver failure keeps the exact path.
    for block_index in keys(block_slack)
        matrix = block_slack[block_index]
        F = _facial_reduction_float_type(opt.settings)
        numeric_matrix = try
            F.(matrix)
        catch
            nothing
        end
        numeric_matrix === nothing && continue
        all(isfinite, numeric_matrix) || continue
        scale = max(one(F), opnorm(numeric_matrix, 1))
        tolerance = F(100) * eps(F) * scale * F(max(1, size(matrix, 1)))
        minimum_eigenvalue = try
            eigmin(Symmetric(numeric_matrix))
        catch
            nothing
        end
        minimum_eigenvalue === nothing && continue
        minimum_eigenvalue < -tolerance && return nothing
    end

    all(_positive_semidefinite_exact(block_slack[index]) for index in keys(block_slack)) ||
        return nothing
    any(!iszero, row) || return nothing

    reduction = _certified_reduction_from_exact_slack(
        opt,
        problem,
        scalar_slack,
        block_slack,
        source,
    )
    reduction === nothing && return nothing
    return _SieveRowCertificate(
        String(source),
        copy(multiplier),
        vcat(row, rhs),
        reduction,
    )
end

function _rref_row_pivots(matrix::AbstractMatrix{ExactRational})
    reduced = _from_nemo_matrix(_exact_rref(Matrix(matrix)))
    pivots = Int[]
    for row in axes(reduced, 1)
        pivot = findfirst(column -> !iszero(reduced[row, column]), axes(reduced, 2))
        pivot === nothing || push!(pivots, pivot)
    end
    return reduced, pivots
end

function _exact_row_reduction_with_multipliers(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
)
    size(A, 1) == length(b) || error("Affine equality matrix and rhs dimensions must match.")
    row_count = size(A, 1)
    augmented = hcat(A, b)
    reduced, pivot_columns = _rref(augmented)
    nonzero_rows = [
        row for row in axes(reduced, 1) if any(!iszero, reduced[row, 1:size(A, 2)])
    ]
    isempty(nonzero_rows) && return reduced, Vector{Vector{ExactRational}}()

    # Pick a square nonsingular set of original rows.  The selected rows span
    # the same row space as the RREF rows, so their coefficients give exact
    # provenance without forming an augmented matrix with a full identity
    # block.
    _, independent_rows = _rref_row_pivots(transpose(augmented))
    rank = length(pivot_columns)
    rank > 0 || return reduced, Vector{Vector{ExactRational}}()
    length(independent_rows) >= rank || return reduced, Vector{Vector{ExactRational}}()
    independent_rows = independent_rows[1:rank]
    pivot_columns = pivot_columns[1:rank]
    square = augmented[independent_rows, pivot_columns]
    inverse = _from_nemo_matrix(
        _exact_rref(hcat(transpose(square), Matrix{ExactRational}(I, rank, rank)))[:, rank + 1:(2 * rank)],
    )

    multipliers = Vector{Vector{ExactRational}}()
    for row_index in nonzero_rows
        coefficients = inverse * vec(reduced[row_index, pivot_columns])
        multiplier = zeros(ExactRational, row_count)
        multiplier[independent_rows] = coefficients
        vec(transpose(multiplier) * augmented) == vec(reduced[row_index, :]) ||
            return reduced, Vector{Vector{ExactRational}}()
        push!(multipliers, multiplier)
    end
    return reduced, multipliers
end

function _sieve_facial_reduction_certificates(
    opt::Optimizer,
    problem::ProblemData,
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
)
    problem.affine === nothing && return _SieveRowCertificate[]
    row_count = size(problem.A, 1)
    certificates = _SieveRowCertificate[]

    for row_index in 1:row_count
        unit = zeros(ExactRational, row_count)
        unit[row_index] = 1 // 1
        for sign in (1 // 1, -1 // 1)
            multiplier = sign .* unit
            source = "Sieve affine row $(row_index) ($(sign > 0 ? "+" : "-"))"
            certificate = _sieve_row_reduction(
                opt,
                problem,
                multiplier,
                source;
                cache,
                row_override = sign .* vec(problem.A[row_index, :]),
                rhs_override = sign * problem.b[row_index],
            )
            certificate === nothing || push!(certificates, certificate)
        end
    end

    if BigInt(row_count) * (size(problem.A, 2) + 1) >
       opt.settings.facial_reduction_sieve_transform_max_entries
        _log(
            opt,
            "Facial reduction Sieve: skipping transformed-row provenance for " *
            "a large affine system ($(row_count)×$(size(problem.A, 2))); " *
            "individual exact rows were still inspected",
        )
        return certificates
    end

    _, multipliers = _exact_row_reduction_with_multipliers(problem.A, problem.b)
    for (row_index, multiplier) in enumerate(multipliers)
        for sign in (1 // 1, -1 // 1)
            signed_multiplier = sign .* multiplier
            source = "Sieve transformed row $(row_index) ($(sign > 0 ? "+" : "-"))"
            certificate = _sieve_row_reduction(
                opt,
                problem,
                signed_multiplier,
                source;
                cache,
            )
            certificate === nothing || push!(certificates, certificate)
        end
    end
    return certificates
end

function _certify_dual_slack_evidence(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    if !_facial_reduction_row_space_is_small(problem, opt.settings)
        row_count, variable_count = size(problem.A)
        _log(
            opt,
            "Facial reduction: skipping exact Phase I cone-dual reconstruction for large affine system ($(row_count)x$(variable_count)); trying boundary subspaces first",
        )
        return nothing
    end
    exact_slack = _exact_exposing_slack_from_numeric_slack(
        opt,
        problem,
        evidence.vector,
        F,
        evidence.source,
    )
    if exact_slack === nothing
        _log(opt, "Facial reduction: no exact exposing slack recovered from $(evidence.source)")
        return nothing
    end
    scalar_slack_exact, block_slack_exact = exact_slack
    return _certified_reduction_from_exact_slack(
        opt,
        problem,
        scalar_slack_exact,
        block_slack_exact,
        evidence.source,
    )
end

function _facial_reduction_oracle_float_type(opt::Optimizer, problem::ProblemData)
    if opt.settings.facial_reduction_float_type !== AbstractFloat
        return _facial_reduction_float_type(opt.settings)
    end
    return _phase1_hypatia_float_type(opt.settings)
end

function _rational_boundary_kernel_candidate_sets(
    opt::Optimizer,
    block_matrix::Matrix{F},
    ::Type{F},
) where {F<:AbstractFloat}
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = _facial_reduction_eigen(opt, symmetric_matrix)
    exposure_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    kernel_indices = [
        index for (index, value) in enumerate(eigen_factor.values) if
        abs(value) <= exposure_tolerance
    ]
    isempty(kernel_indices) && return NamedTuple[]

    kernel_subspace = Matrix(eigen_factor.vectors[:, kernel_indices])
    candidate_sets = NamedTuple[]
    seen = Set{Any}()
    for tolerance in _facial_reduction_subspace_tolerances(opt.settings, F)
        for candidate_set in _rational_subspace_candidate_sets(
            kernel_subspace,
            opt.settings,
            F,
            tolerance,
        )
            directions = _linearly_independent_directions(candidate_set.directions)
            length(directions) == length(kernel_indices) || continue
            key = _canonical_rational_subspace_key(directions)
            key in seen && continue
            push!(seen, key)
            push!(candidate_sets, (
                directions = directions,
                method = candidate_set.method,
                tolerance = tolerance,
                key = key,
                projector = candidate_set.projector,
                reconstruction_error = candidate_set.reconstruction_error,
                coefficient_bits = candidate_set.coefficient_bits,
                fingerprint = candidate_set.fingerprint,
            ))
        end
    end
    return candidate_sets
end

function _numeric_joint_boundary_exposing_slack(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    exposure_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    block_indices = Int[]
    direction_matrices = Matrix{F}[]
    weight_positions_by_block = Vector{Vector{Tuple{Int,Int}}}()
    total_weight_dimension = 0
    for (block_index, block) in enumerate(problem.blocks)
        block_matrix = _vector_to_matrix(evidence.vector, block)
        eigen_factor = _facial_reduction_eigen(
            opt,
            Symmetric((block_matrix + transpose(block_matrix)) / 2),
        )
        kernel_indices = [
            index for (index, value) in enumerate(eigen_factor.values) if
            abs(value) <= exposure_tolerance
        ]
        isempty(kernel_indices) && continue
        directions = Matrix(eigen_factor.vectors[:, kernel_indices])
        weight_positions = _triangle_positions(length(kernel_indices))
        push!(block_indices, block_index)
        push!(direction_matrices, directions)
        push!(weight_positions_by_block, weight_positions)
        total_weight_dimension += length(weight_positions)
    end
    length(block_indices) >= 2 || return nothing

    variable_count = length(problem.objective_vector_raw)
    form_columns = zeros(F, variable_count, total_weight_dimension)
    identity_target = zeros(F, total_weight_dimension)
    weight_ranges = UnitRange{Int}[]
    next_weight = 1
    for offset in eachindex(block_indices)
        block = problem.blocks[block_indices[offset]]
        directions = direction_matrices[offset]
        weight_positions = weight_positions_by_block[offset]
        weight_range = next_weight:(next_weight + length(weight_positions) - 1)
        push!(weight_ranges, weight_range)
        for (local_weight_index, (a, b)) in enumerate(weight_positions)
            global_weight_index = weight_range[local_weight_index]
            weight_basis = zeros(F, size(directions, 2), size(directions, 2))
            weight_basis[a, b] = one(F)
            weight_basis[b, a] = one(F)
            exposed_matrix = directions * weight_basis * transpose(directions)
            for (local_index, (i, j)) in enumerate(block.local_positions)
                form_columns[block.global_positions[local_index], global_weight_index] =
                    i == j ? exposed_matrix[i, j] : 2 * exposed_matrix[i, j]
            end
            a == b && (identity_target[global_weight_index] = one(F))
        end
        next_weight = last(weight_range) + 1
    end

    particular, nullspace = problem.affine
    affine_basis = _to_working_array(F, hcat(particular, nullspace))
    vanish_constraints = transpose(affine_basis) * form_columns
    singular_factor = try
        svd(vanish_constraints)
    catch
        return nothing
    end
    singular_values = singular_factor.S
    scale = max(one(F), isempty(singular_values) ? zero(F) : maximum(abs, singular_values))
    rank_tolerance = max(
        sqrt(eps(F)),
        F(1000) * exposure_tolerance,
        F(1.0e-8),
    ) * scale
    constraint_rank = count(value -> value > rank_tolerance, singular_values)
    if constraint_rank >= total_weight_dimension
        _log(
            opt,
            "Facial reduction: joint numerical scout found no approximate weighted " *
            "nullspace ($(total_weight_dimension) weight variable(s), tol=$(_format_metric(rank_tolerance)))",
        )
        return nothing
    end
    weight_subspace = transpose(singular_factor.Vt)[:, (constraint_rank + 1):end]
    coordinates = try
        weight_subspace \ identity_target
    catch
        return nothing
    end
    all(isfinite, coordinates) || return nothing
    weight_vector = weight_subspace * coordinates

    for offset in eachindex(block_indices)
        weight_positions = weight_positions_by_block[offset]
        weight_range = weight_ranges[offset]
        rank = size(direction_matrices[offset], 2)
        weight_matrix = zeros(F, rank, rank)
        for (local_weight_index, (a, b)) in enumerate(weight_positions)
            value = weight_vector[weight_range[local_weight_index]]
            weight_matrix[a, b] = value
            weight_matrix[b, a] = value
        end
        eigenvalues = try
            eigvals(Symmetric(weight_matrix))
        catch
            return nothing
        end
        if minimum(eigenvalues) <= rank_tolerance
            _log(
                opt,
                "Facial reduction: joint numerical scout weight for PSD block " *
                "$(block_indices[offset]) was not positive definite " *
                "(min_eig=$(_format_metric(minimum(eigenvalues))))",
            )
            return nothing
        end
    end

    numeric_slack = form_columns * weight_vector
    numeric_A_transpose = transpose(_to_working_array(F, problem.A))
    y = try
        numeric_A_transpose \ numeric_slack
    catch
        return nothing
    end
    all(isfinite, y) || return nothing
    trace_row = _facial_reduction_trace_row(problem)
    normalization = dot(_to_working_array(F, trace_row), y)
    abs(normalization) > rank_tolerance || return nothing
    y ./= normalization

    exact_slack = _exact_facial_reduction_oracle_slack(
        opt,
        problem,
        collect(y),
        F;
        cache,
        normalization_row = trace_row,
    )
    if exact_slack === nothing
        _log(
            opt,
            "Facial reduction: joint numerical scout found a coupled PSD slack but " *
            "could not recover it exactly",
        )
        return nothing
    end
    _log(
        opt,
        "Facial reduction: recovered an exact joint exposing slack by numerically " *
        "coupling $(length(block_indices)) boundary PSD subspaces",
    )
    return exact_slack
end

function _certify_joint_boundary_primal_evidence(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    numeric_exact_slack = _numeric_joint_boundary_exposing_slack(
        opt,
        problem,
        evidence,
        F;
        cache,
    )
    if numeric_exact_slack !== nothing
        scalar_slack_exact, block_slack_exact = numeric_exact_slack
        reduction = _certified_reduction_from_exact_slack(
            opt,
            problem,
            scalar_slack_exact,
            block_slack_exact,
            "$(evidence.source) joint numerical scout",
        )
        reduction === nothing || return reduction
    end

    candidate_blocks = Int[]
    candidates_by_block = Vector{Vector{NamedTuple}}()
    for (block_index, block) in enumerate(problem.blocks)
        block_matrix = _vector_to_matrix(evidence.vector, block)
        candidate_sets = _rational_boundary_kernel_candidate_sets(opt, block_matrix, F)
        isempty(candidate_sets) && continue
        push!(candidate_blocks, block_index)
        push!(candidates_by_block, candidate_sets)
    end
    length(candidate_blocks) >= 2 || return nothing

    max_combinations = max(1, opt.settings.facial_reduction_subspace_max_charts^2)
    attempted = 0
    aligned_combinations = Tuple[]
    common_tolerances = reduce(
        intersect,
        [Set(candidate.tolerance for candidate in block_candidates) for
         block_candidates in candidates_by_block],
    )
    for tolerance in sort!(collect(common_tolerances); rev = true)
        aligned = Tuple(
            first(candidate for candidate in block_candidates if candidate.tolerance == tolerance) for
            block_candidates in candidates_by_block
        )
        push!(aligned_combinations, aligned)
    end
    combinations = Iterators.flatten((
        aligned_combinations,
        Iterators.product(candidates_by_block...),
    ))
    seen_combinations = Set{Any}()
    for combination in combinations
        combination_key = Tuple(
            _canonical_rational_subspace_key(candidate.directions) for candidate in combination
        )
        combination_key in seen_combinations && continue
        push!(seen_combinations, combination_key)
        attempted += 1
        attempted > max_combinations && break
        directions_by_block = Dict{Int,Vector{Vector{ExactRational}}}(
            block_index => combination[offset].directions for
            (offset, block_index) in enumerate(candidate_blocks)
        )
        exposure = _multiblock_weighted_subspace_exposure(
            problem,
            directions_by_block,
            opt.settings,
            F,
        )
        exposure === nothing && continue

        keep_bases = Dict{Int,Matrix{ExactRational}}()
        for block_index in candidate_blocks
            directions = directions_by_block[block_index]
            keep_basis = _orthogonal_complement_basis(
                directions,
                problem.blocks[block_index].size,
            )
            size(keep_basis, 2) == problem.blocks[block_index].size && continue
            keep_bases[block_index] = keep_basis
        end
        isempty(keep_bases) && continue
        _record_directions!(
            :certified,
            sum(length, values(directions_by_block)),
            sum(length, values(directions_by_block)),
            0,
        )
        _log(
            opt,
            "Facial reduction: recovered a joint exact exposing certificate from " *
            "$(evidence.source) across $(length(keep_bases)) PSD blocks " *
            "(weight_tol=$(_format_metric(exposure.tolerance)))",
        )
        return _CertifiedFacialReduction(
            "$(evidence.source) joint PSD certificate",
            Int[],
            keep_bases,
        )
    end
    attempted > 0 && _log(
        opt,
        "Facial reduction: no joint exact exposing certificate recovered from " *
        "$(evidence.source) across $(length(candidate_blocks)) candidate PSD blocks " *
        "after $(min(attempted, max_combinations)) candidate combination(s)",
    )
    return nothing
end

function _certify_boundary_primal_evidence(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    keep_bases = Dict{Int,Matrix{ExactRational}}()

    for (block_index, block) in enumerate(problem.blocks)
        matrix = _vector_to_matrix(evidence.vector, block)
        directions = _candidate_kernel_directions(
            opt,
            problem,
            block_index,
            matrix,
            F;
            cache,
        )
        isempty(directions) && continue
        keep_basis = _orthogonal_complement_basis(directions, block.size)
        if size(keep_basis, 2) == problem.blocks[block_index].size
            continue
        end
        keep_bases[block_index] = keep_basis
    end

    isempty(keep_bases) && return nothing
    return _CertifiedFacialReduction(evidence.source, Int[], keep_bases)
end

function _facial_reduction_block_directions(
    opt::Optimizer,
    problem::ProblemData,
    block_index::Int,
    block_matrix::Matrix{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    block = problem.blocks[block_index]
    exact_directions = _exact_block_nullspace_directions(
        problem,
        block;
        cache,
        block_index,
    )
    isempty(exact_directions) || return exact_directions

    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = _facial_reduction_eigen(opt, symmetric_matrix)
    eigenvalues = eigen_factor.values
    isempty(eigenvalues) && return Vector{ExactRational}[]

    exposure_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    maximum(eigenvalues) <= exposure_tolerance && return Vector{ExactRational}[]

    singular_values = svdvals(Matrix(symmetric_matrix))
    rank_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_rank_tolerance),
        F(100) * eps(F),
    )
    numeric_rank = count(value -> value > rank_tolerance, singular_values)
    numeric_rank == 0 && return Vector{ExactRational}[]

    range_indices = [
        index for (index, value) in enumerate(eigen_factor.values) if value > rank_tolerance
    ]
    if !isempty(range_indices)
        range_subspace = Matrix(eigen_factor.vectors[:, range_indices])
        pivoted_directions = _certified_pivoted_subspace_directions(
            opt,
            problem,
            block,
            block_index,
            range_subspace,
            F,
            "exposing-vector",
            ;
            cache,
        )
        isempty(pivoted_directions) || return pivoted_directions
    end

    qr_factor = qr(Matrix(symmetric_matrix), ColumnNorm())
    candidate_columns = unique(qr_factor.p[1:numeric_rank])
    _record_directions!(:certified, length(candidate_columns), 0, 0)
    exact_directions = Vector{Vector{ExactRational}}()
    for column_index in candidate_columns
        direction = _exact_face_direction(
            problem,
            block,
            collect(view(block_matrix, :, column_index)),
            opt.settings,
            F,
            ;
            cache,
            block_index,
        )
        direction === nothing && continue
        push!(exact_directions, direction)
    end
    _record_directions!(
        :certified,
        0,
        length(exact_directions),
        length(candidate_columns) - length(exact_directions),
    )
    exact_directions = _linearly_independent_directions(exact_directions)

    if isempty(exact_directions) && maximum(eigenvalues) > exposure_tolerance
        message =
            "Facial reduction found a non-coordinate exposed face for a PSD block, " *
            "but its nullspace could not be represented exactly over the rational coefficient field."
        if _facial_reduction_irrational_behavior(opt.settings) == :warn
            _log(opt, message)
            return Vector{ExactRational}[]
        end
        throw(ErrorException(message))
    end

    return exact_directions
end

function _certify_oracle_point_evidence(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
    attempt_exact_slack::Bool = true,
    allow_subspace_recovery::Bool = true,
) where {F<:AbstractFloat}
    if attempt_exact_slack
        exact_oracle_slack = _exact_facial_reduction_oracle_slack(
            opt,
            problem,
            evidence.vector,
            F,
            ;
            cache,
            normalization_row,
        )
        if exact_oracle_slack !== nothing
            scalar_slack_exact, block_slack_exact = exact_oracle_slack
            reduction = _certified_reduction_from_exact_slack(
                opt,
                problem,
                scalar_slack_exact,
                block_slack_exact,
                evidence.source,
            )
            reduction === nothing || return reduction
        end
    end

    allow_subspace_recovery || return nothing
    _, block_slack = _facial_reduction_slack(problem, evidence.vector)
    keep_bases = Dict{Int,Matrix{ExactRational}}()
    for block_index in eachindex(problem.blocks)
        directions = _facial_reduction_block_directions(
            opt,
            problem,
            block_index,
            block_slack[block_index],
            F,
            ;
            cache,
        )
        isempty(directions) && continue
        keep_basis = _orthogonal_complement_basis(directions, problem.blocks[block_index].size)
        if size(keep_basis, 2) == problem.blocks[block_index].size
            continue
        end
        keep_bases[block_index] = keep_basis
    end

    isempty(keep_bases) && return nothing
    _log(
        opt,
        "Facial reduction: $(evidence.source) exposed 0 scalar cone direction(s) and $(length(keep_bases)) PSD block face(s)",
    )
    return _CertifiedFacialReduction(evidence.source, Int[], keep_bases)
end

function _certify_facial_reduction_evidence(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    path = if evidence.kind == :dual_slack
        :phase1_dual_slack
    elseif evidence.kind == :boundary_primal
        :phase1_boundary_primal
    elseif evidence.kind == :oracle_point
        :oracle
    else
        error("Unhandled facial reduction evidence kind $(evidence.kind).")
    end
    _record_facial_reduction_path!(path)
    reduction = if evidence.kind == :dual_slack
        _certify_dual_slack_evidence(opt, problem, evidence, F; cache)
    elseif evidence.kind == :boundary_primal
        _certify_boundary_primal_evidence(opt, problem, evidence, F; cache)
    else
        _certify_oracle_point_evidence(opt, problem, evidence, F; cache)
    end
    reduction === nothing || _record_facial_reduction_path!(path; success = true)
    return reduction
end

function _first_certified_facial_reduction(
    opt::Optimizer,
    problem::ProblemData,
    evidence_list::Vector{_FacialReductionEvidence{F}},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    for evidence in evidence_list
        reduction = _certify_facial_reduction_evidence(
            opt,
            problem,
            evidence,
            F;
            cache,
        )
        reduction === nothing && continue
        return reduction
    end
    return nothing
end

function _merge_certified_facial_reductions(
    opt::Optimizer,
    problem::ProblemData,
    reductions::Vector{_CertifiedFacialReduction},
)
    isempty(reductions) && return nothing

    exposed_scalars = sort(unique(vcat((reduction.exposed_scalars for reduction in reductions)...)))
    directions_by_block = Dict{Int,Vector{Vector{ExactRational}}}()
    for reduction in reductions
        for (block_index, keep_basis) in reduction.keep_bases
            removed_directions = _nullspace_basis_exact(Matrix(transpose(keep_basis)))
            directions = get!(directions_by_block, block_index, Vector{Vector{ExactRational}}())
            append!(
                directions,
                [
                    _normalize_rational_direction(collect(view(removed_directions, :, column))) for
                    column in axes(removed_directions, 2)
                ],
            )
        end
    end

    keep_bases = Dict{Int,Matrix{ExactRational}}()
    for (block_index, directions) in directions_by_block
        directions = _linearly_independent_directions(directions)
        isempty(directions) && continue
        keep_basis = _orthogonal_complement_basis(
            directions,
            problem.blocks[block_index].size,
        )
        size(keep_basis, 2) == problem.blocks[block_index].size && continue
        keep_bases[block_index] = keep_basis
    end

    source = join(unique(reduction.source for reduction in reductions), " + ")
    reductions_with_slack = [
        _with_exact_exposing_slack(problem, reduction, opt.settings) for
        reduction in reductions
    ]
    if all(reduction -> reduction !== nothing, reductions_with_slack)
        exposing_slack = sum(
            reduction.exposing_slack for reduction in reductions_with_slack;
            init = zeros(ExactRational, length(problem.objective_vector_raw)),
        )
        merged = _CertifiedFacialReduction(
            source,
            exposed_scalars,
            keep_bases,
            exposing_slack,
        )
        violation = _exact_exposing_slack_violation(problem, merged)
        if violation !== nothing
            _log(
                opt,
                "Facial reduction: rejected compositional merge of exact exposing slacks ($(violation))",
            )
            return nothing
        end
        _log(
            opt,
            "Facial reduction: merged $(length(reductions)) certified face(s) by composing their exact exposing slacks",
        )
        return merged
    end

    _log(
        opt,
        "Facial reduction: exact exposing-slack provenance was unavailable for a merge input; falling back to face reconstruction",
    )
    merged = _CertifiedFacialReduction(source, exposed_scalars, keep_bases)
    violation = _cached_facial_reduction_violation(problem, merged, opt.settings)
    if violation !== nothing
        _log(opt, "Facial reduction: reconstructed face merge was rejected ($(violation))")
        return nothing
    end
    return merged
end

function _sieve_facial_reduction_pass(
    opt::Optimizer,
    problem::ProblemData,
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
)
    certificates = _sieve_facial_reduction_certificates(opt, problem; cache)
    isempty(certificates) && return nothing
    reductions = [certificate.reduction for certificate in certificates]
    merged = _merge_certified_facial_reductions(opt, problem, reductions)
    merged === nothing && return nothing
    return (reduction = merged, certificates = certificates)
end

function _sieve_facial_reduction_problem(
    opt::Optimizer,
    problem::ProblemData,
)
    problem.affine === nothing && return problem
    current = problem
    pass_limit = max(1, _barrier_dimension(problem) + 1)
    for pass_index in 1:pass_limit
        _record_facial_reduction_path!(:sieve)
        cache = _FacialReductionExactCache(current)
        result = _sieve_facial_reduction_pass(opt, current; cache)
        result === nothing && break
        reduction = result.reduction
        reduced = _apply_facial_reduction(
            current,
            reduction.exposed_scalars,
            reduction.keep_bases;
            # _merge_certified_facial_reductions already validated the merged
            # certificate against the current affine slice.
            certified = false,
            settings = opt.settings,
        )
        reduced.affine === nothing && break
        old_dimension = _barrier_dimension(current)
        new_dimension = _barrier_dimension(reduced)
        new_dimension < old_dimension || break
        _record_facial_reduction_path!(:sieve; success = true)
        _record_reduction_round!(old_dimension, new_dimension; tentative = false)
        _record_successful_facial_reduction!(opt, current, reduction)
        _log(
            opt,
            "Facial reduction Sieve pass $(pass_index): removed " *
            "$(old_dimension - new_dimension) barrier direction(s) from " *
            "$(length(result.certificates)) exact row certificate(s)",
        )
        current = reduced
    end
    return current
end

function _face_reduction_rows(
    block::BlockStructure,
    keep_basis::Matrix{ExactRational},
    reduced_block::Union{Nothing,BlockStructure},
)
    indices = Vector{Vector{Int}}()
    values = Vector{Vector{ExactRational}}()
    rhs = ExactRational[]
    if reduced_block === nothing
        for position in block.global_positions
            push!(indices, [position])
            push!(values, ExactRational[1 // 1])
            push!(rhs, 0 // 1)
        end
        return _SparseAffineRestrictions(indices, values), rhs
    end

    for (old_local_index, (i, j)) in enumerate(block.local_positions)
        row_indices = Int[block.global_positions[old_local_index]]
        row_values = ExactRational[1 // 1]
        for (new_local_index, (a, b)) in enumerate(reduced_block.local_positions)
            coefficient = if a == b
                keep_basis[i, a] * keep_basis[j, a]
            else
                keep_basis[i, a] * keep_basis[j, b] + keep_basis[i, b] * keep_basis[j, a]
            end
            iszero(coefficient) && continue
            push!(row_indices, reduced_block.global_positions[new_local_index])
            push!(row_values, -coefficient)
        end
        push!(indices, row_indices)
        push!(values, row_values)
        push!(rhs, 0 // 1)
    end

    return _SparseAffineRestrictions(indices, values), rhs
end

function _facial_reduction_oracle_solver_overrides(
    settings::Settings,
    problem::ProblemData,
    ::Type{F},
) where {F<:AbstractFloat}
    phase1_solver = _phase1_hypatia_syssolver(settings)
    oracle_solver = _facial_reduction_oracle_syssolver(settings)
    stable_solver = if F === Float64 && _phase1_hypatia_prefers_sparse_float64(problem)
        :symindef_indirect
    else
        :qrchol_dense
    end
    if oracle_solver != :auto
        # An explicit oracle solver is a complete policy, not merely a
        # preference. In particular, do not fall back to the Phase-I solver:
        # on large degenerate systems that fallback can spend hours making
        # essentially no progress.
        return Union{Nothing,Symbol}[oracle_solver]
    end
    primary_override = oracle_solver == :auto ? nothing : oracle_solver
    primary_solver = oracle_solver == :auto ? phase1_solver : oracle_solver
    fallback_override = oracle_solver == :auto ? stable_solver : nothing
    fallback_solver = oracle_solver == :auto ? stable_solver : phase1_solver
    solver_overrides = Union{Nothing,Symbol}[primary_override]
    primary_solver == fallback_solver || push!(solver_overrides, fallback_override)
    return solver_overrides
end

function _defer_oracle_subspace_recovery(problem::ProblemData, settings::Settings)
    affine_products = BigInt(size(problem.A, 1)) * size(problem.A, 2)
    return affine_products > settings.facial_reduction_individual_max_affine_products
end

function _facial_reduction_oracle_round(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{HF},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
    source::AbstractString = "oracle",
) where {HF<:AbstractFloat}
    _record_facial_reduction_path!(:oracle)
    defer_subspace_recovery = _defer_oracle_subspace_recovery(problem, opt.settings)
    deferred_candidates = Any[]
    float_types = DataType[HF]
    oracle_precision_retries = max(
        opt.settings.facial_reduction_oracle_precision_escalation_max_retries,
        opt.settings.facial_reduction_precision_escalation_max_retries,
    )
    append!(
        float_types,
        _precision_escalation_types(
            HF,
            oracle_precision_retries,
        ),
    )
    for (precision_index, oracle_float_type) in enumerate(float_types)
        precision_index > 1 && begin
            _record_facial_reduction_event!(:precision_escalations_attempted)
            _log(opt, "Facial reduction oracle: retrying at $(oracle_float_type)")
        end
        solver_overrides = _facial_reduction_oracle_solver_overrides(
            opt.settings,
            problem,
            oracle_float_type,
        )
        precision_retry_recommended = false
        for (solver_index, solver_override) in enumerate(solver_overrides)
            solver_label = solver_override === nothing ?
                           _phase1_hypatia_syssolver(opt.settings) : solver_override
            if solver_index == 1 && solver_override !== nothing
                _log(
                    opt,
                    "Facial reduction oracle: using preferred $(solver_label) system solver",
                )
            elseif solver_index > 1
                _log(
                    opt,
                    "Facial reduction oracle: retrying $(oracle_float_type) with $(solver_label) system solver",
                )
            end
            oracle_attempt = _facial_reduction_oracle_attempt(
                opt,
                problem,
                oracle_float_type;
                normalization_row,
                syssolver_override = solver_override,
                return_details = true,
            )
            oracle_point = oracle_attempt.candidate
            precision_retry_recommended |= oracle_attempt.retry_recommended
            oracle_point === nothing && continue
            evidence_source = solver_override === nothing ? String(source) :
                              "$(source) ($(solver_override))"
            evidence = _FacialReductionEvidence(
                :oracle_point,
                evidence_source,
                oracle_point,
            )
            reduction = try
                _certify_oracle_point_evidence(
                    opt,
                    problem,
                    evidence,
                    oracle_float_type;
                    cache,
                    normalization_row,
                    allow_subspace_recovery = !defer_subspace_recovery,
                )
            catch err
                if _is_inexact_facial_reduction_error(err)
                    _log(
                        opt,
                        "Facial reduction oracle: $(oracle_float_type) candidate could not be certified exactly; continuing with other numerical attempts",
                    )
                    precision_retry_recommended = true
                    nothing
                else
                    rethrow()
                end
            end
            if reduction !== nothing
                _record_facial_reduction_path!(:oracle; success = true)
                return reduction
            end
            if defer_subspace_recovery
                push!(
                    deferred_candidates,
                    (
                        evidence = evidence,
                        float_type = oracle_float_type,
                        normalization_row = normalization_row,
                    ),
                )
                _log(
                    opt,
                    "Facial reduction oracle: deferring expensive rational subspace recovery until alternate solvers and precisions have been tried",
                )
            end
            precision_retry_recommended = true
        end
        if precision_index < length(float_types) && !precision_retry_recommended
            _log(
                opt,
                "Facial reduction oracle: higher precision is unnecessary after a completed numerical oracle solve",
            )
            break
        end
    end

    if !isempty(deferred_candidates)
        _log(
            opt,
            "Facial reduction oracle: numerical alternatives exhausted; attempting deferred rational subspace recovery on $(length(deferred_candidates)) candidate(s)",
        )
        for candidate in deferred_candidates
            reduction = try
                _certify_oracle_point_evidence(
                    opt,
                    problem,
                    candidate.evidence,
                    candidate.float_type;
                    cache,
                    normalization_row = candidate.normalization_row,
                    attempt_exact_slack = false,
                    allow_subspace_recovery = true,
                )
            catch err
                if _is_inexact_facial_reduction_error(err)
                    _log(
                        opt,
                        "Facial reduction oracle: deferred $(candidate.float_type) subspace candidate could not be certified exactly; continuing",
                    )
                    nothing
                else
                    rethrow()
                end
            end
            if reduction !== nothing
                _record_facial_reduction_path!(:oracle; success = true)
                return reduction
            end
        end
    end
    return nothing
end

function _cheap_facial_reduction_evidence(
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
) where {F<:AbstractFloat}
    evidence = _FacialReductionEvidence{F}[]
    if phase1_dual_slack !== nothing
        push!(
            evidence,
            _FacialReductionEvidence(:dual_slack, "Phase I cone dual", phase1_dual_slack),
        )
    end
    push!(evidence, _FacialReductionEvidence(:boundary_primal, "Phase I boundary point", candidate))
    return evidence
end

function _certified_facial_reduction_from_initial_evidence(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    merge_evidence::Bool = true,
) where {F<:AbstractFloat}
    merge_evidence || return _first_certified_facial_reduction(
        opt,
        problem,
        _cheap_facial_reduction_evidence(candidate, phase1_dual_slack, F),
        F;
        cache,
    )
    reductions = _CertifiedFacialReduction[]
    for evidence in _cheap_facial_reduction_evidence(candidate, phase1_dual_slack, F)
        reduction = _certify_facial_reduction_evidence(
            opt,
            problem,
            evidence,
            F;
            cache,
        )
        reduction === nothing || push!(reductions, reduction)
    end
    length(reductions) == 1 && return first(reductions)
    return _merge_certified_facial_reductions(opt, problem, reductions)
end

function _advanced_facial_reduction_from_initial_evidence(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    if opt.settings.facial_reduction_stable_projective_recovery &&
       phase1_dual_slack !== nothing
        _record_facial_reduction_path!(:stable_projective)
        exact_slack = _exact_exposing_slack_from_stable_projective_entries(
            opt,
            problem,
            phase1_dual_slack,
            F,
            "Phase I cone dual",
        )
        if exact_slack !== nothing
            scalar_slack, block_slack = exact_slack
            reduction = _certified_reduction_from_exact_slack(
                opt,
                problem,
                scalar_slack,
                block_slack,
                "Phase I cone dual stable projective fallback",
            )
            if reduction !== nothing
                _record_facial_reduction_path!(:stable_projective; success = true)
                return reduction
            end
        end
    end

    if opt.settings.facial_reduction_multiblock_recovery
        _record_facial_reduction_path!(:joint_multiblock)
        evidence = _FacialReductionEvidence(
            :boundary_primal,
            "Phase I boundary point",
            candidate,
        )
        reduction = _certify_joint_boundary_primal_evidence(
            opt,
            problem,
            evidence,
            F;
            cache,
        )
        if reduction !== nothing
            _record_facial_reduction_path!(:joint_multiblock; success = true)
            return reduction
        end
    end
    return nothing
end

function _facial_reduction_target_trace_row(
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
)
    row = zeros(ExactRational, size(problem.A, 1))
    exposed_scalars = Set(reduction.exposed_scalars)
    for index in problem.positive_scalars
        index in exposed_scalars && continue
        row .+= problem.A[:, index]
    end
    for (block_index, block) in enumerate(problem.blocks)
        keep_basis = get(
            reduction.keep_bases,
            block_index,
            Matrix{ExactRational}(I, block.size, block.size),
        )
        isempty(keep_basis) && continue
        face_matrix = keep_basis * transpose(keep_basis)
        for (local_index, (i, j)) in enumerate(block.local_positions)
            coefficient = i == j ? face_matrix[i, j] : 2 * face_matrix[i, j]
            iszero(coefficient) && continue
            row .+= coefficient .* problem.A[:, block.global_positions[local_index]]
        end
    end
    return row
end

function _facial_reduction_round_with_rank_expansion(
    opt::Optimizer,
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
) where {F<:AbstractFloat}
    current = reduction
    _record_successful_facial_reduction!(opt, problem, current)
    for expansion_round in 1:opt.settings.facial_reduction_rank_expansion_rounds
        normalization_row = _facial_reduction_target_trace_row(problem, current)
        any(!iszero, normalization_row) || break
        _log(
            opt,
            "Facial reduction: rank-expansion oracle round $(expansion_round) targeting the current residual face",
        )
        next = _facial_reduction_oracle_round(
            opt,
            problem,
            _facial_reduction_oracle_float_type(opt, problem);
            cache,
            normalization_row,
            source = "rank-expansion oracle $(expansion_round)",
        )
        if next === nothing
            _log(
                opt,
                "Facial reduction: rank expansion stopped after round $(expansion_round); no additional exact face was certified",
            )
            break
        end
        merged = _merge_certified_facial_reductions(opt, problem, [current, next])
        if merged === nothing
            _log(
                opt,
                "Facial reduction: rank expansion stopped after round $(expansion_round); the new exact face could not be merged",
            )
            break
        end
        old_removed = sum(
            problem.blocks[index].size - size(current.keep_bases[index], 2) for
            index in keys(current.keep_bases);
            init = 0,
        ) + length(current.exposed_scalars)
        new_removed = sum(
            problem.blocks[index].size - size(merged.keep_bases[index], 2) for
            index in keys(merged.keep_bases);
            init = 0,
        ) + length(merged.exposed_scalars)
        if new_removed <= old_removed
            _log(
                opt,
                "Facial reduction: rank expansion stopped after round $(expansion_round); the merged certificate exposed no new cone direction",
            )
            break
        end
        previous = current
        current = merged
        _record_successful_facial_reduction!(
            opt,
            problem,
            current;
            supersedes = previous,
        )
    end
    return current
end

function _remap_block_structure(
    block::BlockStructure,
    position_map::AbstractVector{Int},
)
    global_positions = Int[position_map[position] for position in block.global_positions]
    all(>(0), global_positions) ||
        error("Compact facial-reduction map removed an active PSD coordinate.")
    diagonal_positions = Int[position_map[position] for position in block.diagonal_positions]
    return BlockStructure(
        block.size,
        copy(block.variables),
        global_positions,
        copy(block.local_positions),
        diagonal_positions,
    )
end

function _face_basis_coefficient(
    keep_basis::Matrix{ExactRational},
    i::Int,
    j::Int,
    a::Int,
    b::Int,
)
    return if a == b
        keep_basis[i, a] * keep_basis[j, a]
    else
        keep_basis[i, a] * keep_basis[j, b] +
        keep_basis[i, b] * keep_basis[j, a]
    end
end

function _compact_face_coordinate_data(
    problem::ProblemData,
    exposed_scalars::Vector{Int},
    keep_bases::Dict{Int,Matrix{ExactRational}},
    block_replacements::Dict{Int,Union{Nothing,BlockStructure}},
    extended_dimension::Int,
)
    old_dimension = length(problem.objective_vector_raw)
    removed_old_positions = Set{Int}(exposed_scalars)
    for block_index in keys(keep_bases)
        union!(removed_old_positions, problem.blocks[block_index].global_positions)
    end
    kept_old_positions = [
        position for position in 1:old_dimension if !(position in removed_old_positions)
    ]
    appended_positions = collect((old_dimension + 1):extended_dimension)
    source_positions = vcat(kept_old_positions, appended_positions)
    extended_to_compact = zeros(Int, extended_dimension)
    for (compact_position, source_position) in enumerate(source_positions)
        extended_to_compact[source_position] = compact_position
    end

    old_to_compact = view(extended_to_compact, 1:old_dimension)
    compact_dimension = length(source_positions)
    old_from_compact = spzeros(ExactRational, old_dimension, compact_dimension)
    for old_position in kept_old_positions
        old_from_compact[old_position, old_to_compact[old_position]] = one(ExactRational)
    end

    compact_blocks = BlockStructure[]
    for (block_index, block) in enumerate(problem.blocks)
        keep_basis = get(keep_bases, block_index, nothing)
        if keep_basis === nothing
            push!(compact_blocks, _remap_block_structure(block, extended_to_compact))
            continue
        end
        replacement = get(block_replacements, block_index, nothing)
        replacement === nothing && continue
        compact_replacement = _remap_block_structure(replacement, extended_to_compact)
        push!(compact_blocks, compact_replacement)
        for (old_local_index, (i, j)) in enumerate(block.local_positions)
            old_position = block.global_positions[old_local_index]
            for (new_local_index, (a, b)) in enumerate(replacement.local_positions)
                coefficient = _face_basis_coefficient(keep_basis, i, j, a, b)
                iszero(coefficient) && continue
                old_from_compact[
                    old_position,
                    compact_replacement.global_positions[new_local_index],
                ] += coefficient
            end
        end
    end

    compact_positive_scalars = Int[]
    for position in problem.positive_scalars
        position in removed_old_positions && continue
        compact_position = old_to_compact[position]
        compact_position > 0 ||
            error("Compact facial-reduction map removed an active scalar coordinate.")
        push!(compact_positive_scalars, compact_position)
    end
    return (
        source_positions = source_positions,
        old_from_compact = old_from_compact,
        blocks = compact_blocks,
        positive_scalars = compact_positive_scalars,
    )
end

function _fixed_zero_cone_face(problem::ProblemData)
    problem.affine === nothing && return Int[], Dict{Int,Matrix{ExactRational}}()
    particular, nullspace = problem.affine
    exposed_scalars = Int[
        position for position in problem.positive_scalars if
        _variable_fixed_zero(particular, nullspace, position)
    ]
    keep_bases = Dict{Int,Matrix{ExactRational}}()
    for (block_index, block) in enumerate(problem.blocks)
        keep_directions = Int[
            direction for direction in 1:block.size if !_variable_fixed_zero(
                particular,
                nullspace,
                block.diagonal_positions[direction],
            )
        ]
        length(keep_directions) == block.size && continue
        keep_basis = zeros(ExactRational, block.size, length(keep_directions))
        for (column, direction) in enumerate(keep_directions)
            keep_basis[direction, column] = one(ExactRational)
        end
        keep_bases[block_index] = keep_basis
    end
    return exposed_scalars, keep_bases
end

function _orthogonal_face_keep_basis(U::Matrix{ExactRational})
    V = copy(U)
    squared_norms = ExactRational[]
    for j in axes(V, 2)
        for i in 1:(j - 1)
            coefficient = dot(view(V, :, i), view(V, :, j)) / squared_norms[i]
            iszero(coefficient) && continue
            V[:, j] -= coefficient * V[:, i]
        end
        norm_squared = dot(view(V, :, j), view(V, :, j))
        norm_squared > 0 || error("Facial-reduction keep basis has dependent columns.")
        exponent = setprecision(BigFloat, 256) do
            round(Int, log2(BigFloat(norm_squared)) / 2)
        end
        scale = exponent >= 0 ?
                (big(1) << exponent) // big(1) : big(1) // (big(1) << -exponent)
        V[:, j] /= scale
        push!(squared_norms, norm_squared / scale^2)
    end
    # Each step is an invertible rational column operation, so range(V) =
    # range(U). No proposed kernel direction is certified by conditioning.
    return V
end

function _apply_facial_reduction(
    problem::ProblemData,
    exposed_scalars::Vector{Int},
    keep_bases::Dict{Int,Matrix{ExactRational}},
    ;
    certified::Bool = false,
    checkpoint::Union{Nothing,Function} = nothing,
    settings::Settings = Settings(),
)
    if certified
        violation = _cached_facial_reduction_violation(
            problem,
            _CertifiedFacialReduction("in-memory", exposed_scalars, keep_bases),
            settings,
        )
        violation === nothing ||
            error("Certified facial reduction failed exact preservation checks: $(violation)")
    end
    if settings.facial_reduction_orthogonalize
        checkpoint !== nothing && checkpoint("face application: orthogonalizing exact keep bases")
        keep_bases = Dict(index => _orthogonal_face_keep_basis(basis) for (index, basis) in keep_bases)
    end
    old_dimension = length(problem.objective_vector_raw)
    old_barrier_dimension = _barrier_dimension(problem)
    blocks = BlockStructure[]
    block_replacements = Dict{Int,Union{Nothing,BlockStructure}}()
    next_position = old_dimension + 1

    for (block_index, block) in enumerate(problem.blocks)
        keep_basis = get(keep_bases, block_index, nothing)
        if keep_basis === nothing
            push!(blocks, block)
            continue
        end
        reduced_dimension = size(keep_basis, 2)
        if reduced_dimension == 0
            block_replacements[block_index] = nothing
            continue
        end
        local_positions = _triangle_positions(reduced_dimension)
        global_positions =
            collect(next_position:(next_position + length(local_positions) - 1))
        diagonal_positions = [
            global_positions[index] for
            (index, (i, j)) in enumerate(local_positions) if i == j
        ]
        reduced_block = BlockStructure(
            reduced_dimension,
            fill(nothing, length(local_positions)),
            global_positions,
            local_positions,
            diagonal_positions,
        )
        block_replacements[block_index] = reduced_block
        push!(blocks, reduced_block)
        next_position += length(local_positions)
    end

    total_dimension = next_position - 1
    checkpoint !== nothing && checkpoint(
        "face application: rebuilding $(length(problem.blocks)) PSD block(s) in $(total_dimension) variables",
    )
    extra_row_indices = Vector{Vector{Int}}()
    extra_row_values = Vector{Vector{ExactRational}}()
    extra_rhs = ExactRational[]

    for position in unique(sort(exposed_scalars))
        push!(extra_row_indices, [position])
        push!(extra_row_values, ExactRational[1 // 1])
        push!(extra_rhs, 0 // 1)
    end

    for (block_index, block) in enumerate(problem.blocks)
        keep_basis = get(keep_bases, block_index, nothing)
        keep_basis === nothing && continue
        restrictions, rhs = _face_reduction_rows(
            block,
            keep_basis,
            get(block_replacements, block_index, nothing),
        )
        append!(extra_row_indices, restrictions.indices)
        append!(extra_row_values, restrictions.values)
        append!(extra_rhs, rhs)
    end
    extra_restrictions = _SparseAffineRestrictions(extra_row_indices, extra_row_values)

    checkpoint !== nothing && checkpoint(
        "face application: restricting affine parametrization with $(length(extra_rhs)) exact face equation(s)",
    )
    extended_affine = _extend_and_restrict_affine_system(
        problem.affine,
        total_dimension - old_dimension,
        extra_restrictions,
        extra_rhs,
        checkpoint = checkpoint,
        settings = settings,
    )
    checkpoint !== nothing && checkpoint(
        "face application: eliminating superseded PSD coordinates",
    )
    compact_data = _compact_face_coordinate_data(
        problem,
        exposed_scalars,
        keep_bases,
        block_replacements,
        total_dimension,
    )
    old_from_compact = compact_data.old_from_compact
    A = Matrix{ExactRational}(problem.A * old_from_compact)
    b = copy(problem.b)
    objective_vector_raw =
        Vector{ExactRational}(transpose(old_from_compact) * problem.objective_vector_raw)
    objective_vector_min =
        Vector{ExactRational}(transpose(old_from_compact) * problem.objective_vector_min)
    affine = if extended_affine === nothing
        nothing
    else
        extended_particular, extended_nullspace = extended_affine
        (
            extended_particular[compact_data.source_positions],
            extended_nullspace[compact_data.source_positions, :],
        )
    end
    old_particular, old_nullspace = problem.affine === nothing ?
        (ExactRational[], zeros(ExactRational, 0, 0)) : problem.affine
    affine_representation_complete = size(problem.A, 1) > 0 ||
                                     size(old_nullspace, 2) == length(old_particular)
    # _extend_and_restrict_affine_system solves and validates the new
    # restriction in coordinates of the old exact affine system.  Rechecking
    # A*p and A*N here repeated several dense BigInt-rational products; when
    # the old affine representation is complete, those equalities follow
    # algebraically from the old invariant and the coordinate solve.
    incremental_valid = affine_representation_complete && affine !== nothing
    if !incremental_valid
        @debug "Facial reduction incremental affine restriction failed; falling back to full exact elimination"
        checkpoint !== nothing && checkpoint("face application: incremental restriction unavailable; solving full exact affine system")
        affine = _solve_affine_system(A, b; checkpoint = checkpoint)
    end
    solution_lift = problem.solution_lift * old_from_compact
    blocks = compact_data.blocks
    positive_scalars = compact_data.positive_scalars
    compaction_requested = size(A, 1) >
                           BigInt(settings.facial_reduction_affine_compaction_factor) *
                           max(1, size(problem.A, 1))
    compaction_entries = BigInt(size(A, 1)) * (size(A, 2) + 1)
    compaction_allowed = settings.facial_reduction_affine_compaction_max_entries > 0 &&
                         compaction_entries <=
                         settings.facial_reduction_affine_compaction_max_entries
    if compaction_requested && compaction_allowed
        checkpoint !== nothing &&
            checkpoint("face application: compacting redundant affine equations")
        compacted = try
            _independent_affine_equalities(A, b; checkpoint = checkpoint)
        catch err
            err isa ExactLinearAlgebraError || rethrow()
            checkpoint !== nothing && checkpoint(
                "face application: affine compaction unavailable ($(_exception_message(err))); retaining un-compacted exact system",
            )
            nothing
        end
        if compacted !== nothing
            A, b = compacted
            # Compaction is an exact row operation on [A b], so the affine
            # parametrization computed above remains complete.
            checkpoint !== nothing && checkpoint(
                "face application: retaining existing exact affine parametrization after compaction",
            )
        end
    elseif compaction_requested
        checkpoint !== nothing && checkpoint(
            "face application: skipping optional affine compaction ($(compaction_entries) entries; limit $(settings.facial_reduction_affine_compaction_max_entries))",
        )
    end
    reduced_problem = ProblemData(
        problem.original_variables,
        blocks,
        positive_scalars,
        objective_vector_raw,
        problem.objective_constant_raw,
        objective_vector_min,
        A,
        b,
        affine,
        nothing,
        problem.scalar_constraint_rows,
        problem.psd_constraint_blocks,
        solution_lift,
        nothing,
    )
    new_barrier_dimension = _barrier_dimension(reduced_problem)
    new_barrier_dimension < old_barrier_dimension ||
        error("Facial reduction was applied without decreasing barrier dimension.")
    auto_exposed_scalars, auto_keep_bases =
        _fixed_zero_cone_face(reduced_problem)
    if !isempty(auto_exposed_scalars) || !isempty(auto_keep_bases)
        checkpoint !== nothing && checkpoint(
            "face application: eliminating exactly fixed zero cone coordinates",
        )
        return _apply_facial_reduction(
            reduced_problem,
            auto_exposed_scalars,
            auto_keep_bases;
            certified = false,
            checkpoint = checkpoint,
            settings = settings,
        )
    end
    checkpoint !== nothing && checkpoint("face application: completed")
    return reduced_problem
end

function _facial_reduction_round(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    rank_expansion::Bool = true,
    merge_evidence::Bool = true,
) where {F<:AbstractFloat}
    return _facial_reduction_round(
        opt,
        problem,
        candidate,
        nothing,
        F;
        cache,
        rank_expansion,
        merge_evidence,
    )
end

function _facial_reduction_round(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
    ;
    cache::_FacialReductionExactCache = _FacialReductionExactCache(problem),
    rank_expansion::Bool = true,
    merge_evidence::Bool = true,
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    reduction = _certified_facial_reduction_from_initial_evidence(
        opt,
        problem,
        candidate,
        phase1_dual_slack,
        F,
        ;
        cache,
        merge_evidence,
    )
    if reduction === nothing
        _log(
            opt,
            "Facial reduction: initial evidence found no exact reducing face; trying exposing-vector oracle",
        )
        reduction = _facial_reduction_oracle_round(
            opt,
            problem,
            _facial_reduction_oracle_float_type(opt, problem);
            cache,
        )
    end
    if reduction === nothing
        _log(
            opt,
            "Facial reduction: established recovery and oracle found no exact face; " *
            "trying stronger exact-recovery fallbacks",
        )
        reduction = _advanced_facial_reduction_from_initial_evidence(
            opt,
            problem,
            candidate,
            phase1_dual_slack,
            F;
            cache,
        )
    end
    reduction === nothing && return nothing
    rank_expansion || return reduction
    return _facial_reduction_round_with_rank_expansion(
        opt,
        problem,
        reduction,
        _facial_reduction_oracle_float_type(opt, problem);
        cache,
    )
end

function _facially_reduce_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
    ;
    rank_expansion::Bool = true,
    merge_evidence::Bool = true,
) where {F<:AbstractFloat}
    return _facially_reduce_problem(
        opt,
        problem,
        candidate,
        nothing,
        F;
        rank_expansion,
        merge_evidence,
    )
end

function _facially_reduce_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
    ;
    rank_expansion::Bool = true,
    merge_evidence::Bool = true,
) where {F<:AbstractFloat}
    opt.settings.facial_reduction || return problem
    reduction = _facial_reduction_round(
        opt,
        problem,
        candidate,
        phase1_dual_slack,
        F;
        rank_expansion,
        merge_evidence,
    )
    reduction === nothing && return problem
    exposed_scalars = reduction.exposed_scalars
    keep_bases = reduction.keep_bases
    removed_psd_directions = sum(
        (problem.blocks[index].size - size(keep_bases[index], 2) for index in keys(keep_bases));
        init = 0,
    )
    reduced_problem = _apply_facial_reduction(
        problem,
        exposed_scalars,
        keep_bases;
        certified = false,
        settings = opt.settings,
    )
    if reduced_problem.affine === nothing
        message =
            "Facial reduction found a PSD block on the cone boundary, " *
            "but the exposed nullspace directions could not be represented exactly over the rational coefficient field."
        if _facial_reduction_irrational_behavior(opt.settings) == :warn
            _log(opt, message)
            return problem
        end
        throw(ErrorException(message))
    end
    _record_reduction_round!(
        _barrier_dimension(problem),
        _barrier_dimension(reduced_problem);
        tentative = false,
    )
    _record_successful_facial_reduction!(opt, problem, reduction)
    _log(
        opt,
        "facial reduction: fixed $(length(exposed_scalars)) scalar cone direction(s) and removed $(removed_psd_directions) PSD direction(s)",
    )
    return reduced_problem
end

function _facially_reduce_search_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
) where {F<:AbstractFloat}
    reduction = try
        _facial_reduction_round(opt, problem, candidate, phase1_dual_slack, F)
    catch err
        if _is_inexact_facial_reduction_error(err)
            _log(opt, "Facial reduction: exact recovery of the candidate face was impossible")
            nothing
        elseif err isa ExactLinearAlgebraError
            _log(
                opt,
                "Facial reduction: optional exact preprocessing unavailable ($(_exception_message(err))); continuing with the original SDP",
            )
            return (
                problem = problem,
                tentative = false,
                fallback_problem = nothing,
            )
        else
            rethrow()
        end
    end
    if reduction !== nothing
        reduced_problem = try
            _apply_facial_reduction(
                problem,
                reduction.exposed_scalars,
                reduction.keep_bases,
                ;
                certified = false,
                settings = opt.settings,
            )
        catch err
            err isa ExactLinearAlgebraError || rethrow()
            _log(
                opt,
                "Facial reduction: exact face application unavailable ($(_exception_message(err))); continuing with the original SDP",
            )
            return (
                problem = problem,
                tentative = false,
                fallback_problem = nothing,
            )
        end
        if reduced_problem.affine === nothing
            message =
                "Facial reduction found a PSD block on the cone boundary, " *
                "but the exposed nullspace directions could not be represented exactly over the rational coefficient field."
            if _facial_reduction_irrational_behavior(opt.settings) == :warn
                _log(opt, message)
                return (
                    problem = problem,
                    tentative = false,
                    fallback_problem = nothing,
                )
            end
            throw(ErrorException(message))
        end
        _record_reduction_round!(
            _barrier_dimension(problem),
            _barrier_dimension(reduced_problem);
            tentative = false,
        )
        _record_successful_facial_reduction!(opt, problem, reduction)
        return (
            problem = reduced_problem,
            tentative = false,
            fallback_problem = nothing,
        )
    end
    _record_facial_reduction_path!(:tentative_restriction)
    tentative_result = try
        _tentative_feasibility_search_problem(
            opt,
            problem,
            candidate,
            F;
            return_details = true,
        )
    catch err
        err isa ExactLinearAlgebraError || rethrow()
        _log(
            opt,
            "Feasibility search: optional tentative face preprocessing unavailable ($(_exception_message(err))); continuing with the original SDP",
        )
        return (
            problem = problem,
            tentative = false,
            fallback_problem = nothing,
        )
    end
    tentative_problem = tentative_result.problem
    if tentative_problem !== nothing
        _record_facial_reduction_path!(:tentative_restriction; success = true)
        _record_reduction_round!(
            _barrier_dimension(problem),
            _barrier_dimension(tentative_problem);
            tentative = true,
        )
    end
    return (
        problem = tentative_problem === nothing ? problem : tentative_problem,
        tentative = tentative_problem !== nothing,
        fallback_problem = tentative_result.fallback_problem,
    )
end
