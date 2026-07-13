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

mutable struct _FacialReductionExactCache
    problem::ProblemData
    row_space::Union{Nothing,_FacialReductionRowSpaceCache}
    block_exact_directions::Vector{Union{Nothing,Vector{Vector{ExactRational}}}}
    A_transpose::Union{Nothing,Nemo.QQMatrix}
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
    return (transpose(direction_column) * matrix * direction_column)[1, 1]
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
        _exact_rref(hcat(transpose(square), Matrix{ExactRational}(I, rank, rank)))[:, rank + 1:(2 * rank)],
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
)
    length(indices) == length(values) || error("Sparse row indices and values must have equal lengths.")
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
            _row_space_multiplier(problem, indices, values; cache) === nothing || continue
            _affine_form_violation(problem, indices, values) === nothing && continue
            return "row=$(row_index), row-space membership certificate unavailable"
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
        _row_space_multiplier(problem, indices, values; cache) === nothing || return nothing
        _affine_form_violation(problem, indices, values) === nothing && return nothing
        return "quadratic row-space membership certificate unavailable"
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
        _row_space_multiplier(problem, indices, values; cache) === nothing || return nothing
        _affine_form_violation(problem, indices, values) === nothing && return nothing
        return "trace row-space membership certificate unavailable"
    finally
        _record_certificate_check!((time_ns() - start_time) / 1.0e9)
    end
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
        raw_direction = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for value in candidate
        ]
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

function _pivoted_rational_subspace_directions(
    subspace::AbstractMatrix{F},
    settings::Settings,
    ::Type{F};
    relation_tolerance = nothing,
) where {F<:AbstractFloat}
    dimension, column_count = size(subspace)
    (dimension == 0 || column_count == 0) && return Vector{ExactRational}[]
    all(isfinite, subspace) || return Vector{ExactRational}[]

    subspace_matrix = Matrix{F}(subspace)
    row_space_matrix = Matrix(transpose(subspace_matrix))
    qr_factor = qr(row_space_matrix, ColumnNorm())
    diagonal = abs.(diag(qr_factor.R))
    isempty(diagonal) && return Vector{ExactRational}[]

    scale = max(one(F), maximum(abs, subspace_matrix), maximum(diagonal))
    rank_tolerance = max(
        _to_working_float(F, settings.facial_reduction_rank_tolerance),
        F(max(size(row_space_matrix)...)) * eps(F) * scale,
        F(100) * eps(F),
    )
    rank = count(value -> value > rank_tolerance, diagonal)
    rank == 0 && return Vector{ExactRational}[]

    pivot_indices = collect(qr_factor.p[1:rank])
    pivot_set = Set(pivot_indices)
    remaining_indices = [index for index in 1:dimension if !(index in pivot_set)]

    relations = if isempty(remaining_indices)
        zeros(F, rank, 0)
    else
        row_space_matrix[:, pivot_indices] \ row_space_matrix[:, remaining_indices]
    end

    tolerances = if relation_tolerance === nothing
        _recovery_tolerances(settings, F)
    else
        F[_to_working_float(F, relation_tolerance)]
    end

    for tolerance in tolerances
        rational_relations = Matrix{ExactRational}(undef, size(relations)...)
        for index in eachindex(relations)
            rational_relations[index] =
                rationalize(BigInt, BigFloat(relations[index]); tol = BigFloat(tolerance))
        end

        directions = Vector{Vector{ExactRational}}()
        for pivot_offset in 1:rank
            direction = zeros(ExactRational, dimension)
            direction[pivot_indices[pivot_offset]] = 1 // 1
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
) where {F<:AbstractFloat}
    candidates = _pivoted_rational_subspace_directions(subspace, opt.settings, F)
    isempty(candidates) && return Vector{ExactRational}[]
    _record_directions!(:certified, length(candidates), 0, 0)

    accepted = Vector{Vector{ExactRational}}()
    rejected = 0
    last_violation = nothing
    for direction in candidates
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
            "Facial reduction: using certified pivoted $(description) subspace for PSD block $(block_index) ($(length(accepted)) direction(s))",
        )
        return accepted
    end

    _log(
        opt,
        "Facial reduction: rejected pivoted $(description) candidate for PSD block $(block_index); no exact affine row certificate for $(rejected) direction(s) ($(last_violation))",
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
    basis = Nemo.identity_matrix(Nemo.QQ, block.size)
    matrices = Nemo.QQMatrix[
        _to_nemo_matrix(_vector_to_matrix(particular, block)),
    ]
    for column in axes(nullspace, 2)
        push!(matrices, _to_nemo_matrix(_vector_to_matrix(view(nullspace, :, column), block)))
    end
    for matrix in matrices
        _, kernel = Nemo.nullspace(matrix * basis)
        basis = basis * kernel
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

function _tentative_batch_problem(
    problem::ProblemData,
    candidates::Vector{<:_TentativeFaceDirection},
)
    keep_bases = _tentative_candidate_keep_bases(problem, candidates)
    isempty(keep_bases) && return problem
    return _apply_facial_reduction(
        problem,
        Int[],
        keep_bases;
        certified = false,
    )
end

function _tentative_greedy_admission(
    problem::ProblemData,
    candidates::Vector{_TentativeFaceDirection{F}},
) where {F<:AbstractFloat}
    accepted_candidates = _TentativeFaceDirection{F}[]
    consistent_problem = nothing
    for candidate_item in candidates
        trial_candidates = vcat(accepted_candidates, [candidate_item])
        trial_problem = _tentative_batch_problem(problem, trial_candidates)
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
    batch_problem = _tentative_batch_problem(problem, unique_candidates)
    if batch_problem.affine !== nothing
        fallback_problem = nothing
        if length(unique_candidates) > 1
            conservative_problem = _tentative_batch_problem(
                problem,
                unique_candidates[1:1],
            )
            if conservative_problem.affine !== nothing &&
               _barrier_dimension(conservative_problem) >
               _barrier_dimension(batch_problem)
                fallback_problem = conservative_problem
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
            "Feasibility search: tentatively batched $(length(unique_candidates)) PSD direction(s) across $(length(Set(candidate_item.block_index for candidate_item in unique_candidates))) block(s); any recovered point will be checked exactly against the unreduced SDP",
        )
        return _tentative_search_result(batch_problem, fallback_problem, return_details)
    end

    # Deterministic rollback: admit candidates one at a time, always
    # recomputing the combined face from the original problem.
    consistent_problem, accepted_candidates = _tentative_greedy_admission(
        problem,
        unique_candidates,
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

function _diagnose_phase1_candidate_kernel!(
    opt::Optimizer,
    problem::ProblemData,
    block_index::Int,
    block_matrix::Matrix{F},
    ::Type{F},
) where {F<:AbstractFloat}
    old_tolerance = opt.settings.facial_reduction_exposure_tolerance
    try
        for threshold in _PHASE1_DIAGNOSTIC_THRESHOLDS
            opt.settings.facial_reduction_exposure_tolerance = threshold
            directions = try
                _candidate_kernel_directions(opt, problem, block_index, block_matrix, F)
            catch err
                _log(
                    opt,
                    "Phase I diagnostics: block $(block_index) loose kernel tol=$(_format_metric(_to_working_float(F, threshold))) raised $(typeof(err))",
                )
                continue
            end
            isempty(directions) && continue
            _log(
                opt,
                "Phase I diagnostics: block $(block_index) loose kernel tol=$(_format_metric(_to_working_float(F, threshold))) recovered $(length(directions)) rationalized candidate-kernel direction(s)",
            )
            break
        end
    finally
        opt.settings.facial_reduction_exposure_tolerance = old_tolerance
    end
    return
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
        symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
        eigenvalues = _facial_reduction_eigvals(opt, block_matrix)
        isempty(eigenvalues) && continue
        _log(
            opt,
            "Phase I candidate diagnostics: block $(block_index) size=$(block.size), min_eig=$(_format_metric(minimum(eigenvalues))), max_eig=$(_format_metric(maximum(eigenvalues))), negative=$(count(value -> value < zero(F), eigenvalues)), $(_phase1_threshold_summary(eigenvalues, F))",
        )
        _diagnose_phase1_candidate_kernel!(opt, problem, block_index, block_matrix, F)
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

function _facial_reduction_oracle_attempt(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{HF},
    ;
    normalization_row::Vector{ExactRational} = _facial_reduction_trace_row(problem),
) where {HF<:AbstractFloat}
    return _with_float_precision(HF, opt.settings.working_precision, function (::Type{HF})
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
            return nothing
        end
        syssolver, use_dense_model, preprocess = _hypatia_phase1_syssolver(opt.settings, HF)
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
            _log(
                opt,
                "Facial reduction oracle unavailable: $(typeof(err))",
            )
            record_oracle()
            return nothing
        end
        elapsed_sec = (time_ns() - start_time) / 1.0e9
        status = Hypatia.Solvers.get_status(solver)
        if !_facial_reduction_oracle_allows_candidate_status(status)
            _log(
                opt,
                "Facial reduction oracle: status=$(status), time=$(@sprintf("%.2f", elapsed_sec))s",
            )
            record_oracle(Hypatia.Solvers.get_num_iters(solver))
            return nothing
        end
        candidate = try
            vec(collect(Hypatia.Solvers.get_x(solver)))
        catch
            nothing
        end
        if candidate === nothing
            record_oracle(Hypatia.Solvers.get_num_iters(solver))
            return nothing
        end
        if !all(isfinite, candidate)
            record_oracle(Hypatia.Solvers.get_num_iters(solver))
            return nothing
        end
        slow_progress_note =
            status == Hypatia.Solvers.SlowProgress ? "; trying current iterate" : ""
        _log(
            opt,
            "Facial reduction oracle: status=$(status), iter=$(Hypatia.Solvers.get_num_iters(solver)), time=$(@sprintf("%.2f", elapsed_sec))s$(slow_progress_note)",
        )
        record_oracle(Hypatia.Solvers.get_num_iters(solver))
        return candidate
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
        vec(_from_nemo_matrix(_facial_reduction_A_transpose!(cache, problem) * y_nemo))
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
    return unique(tolerances)
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
    for tolerance in _facial_reduction_oracle_tolerances(opt.settings, F)
        slack = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for
            value in numeric_slack
        ]
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
            "Facial reduction: recovered exact exposing slack from $(source) (tol=$(_format_metric(tolerance)))",
        )
        return scalar_slack, block_slack
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
end

struct _SieveRowCertificate
    source::String
    multiplier::Vector{ExactRational}
    row::Vector{ExactRational}
    reduction::_CertifiedFacialReduction
end

const _SIEVE_TRANSFORM_MAX_ENTRIES = 250_000
const _FACIAL_REDUCTION_AFFINE_COMPACTION_FACTOR = 4

const _FACIAL_REDUCTION_CACHE_MAGIC = "RationalSDP facial reduction cache"
const _FACIAL_REDUCTION_CACHE_VERSION = 1

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

function _write_facial_reduction_cache(path::AbstractString, records::Vector{Any})
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
        mv(temporary_path, full_path; force = true)
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

function _facial_reduction_signature_matches(problem::ProblemData, signature)
    signature isa NamedTuple || return false
    required = (:dimension, :equation_count, :positive_scalars, :blocks)
    all(name -> name in keys(signature), required) || return false
    signature.dimension == length(problem.objective_vector_raw) || return false
    signature.equation_count == size(problem.A, 1) || return false
    collect(signature.positive_scalars) == problem.positive_scalars || return false
    length(signature.blocks) == length(problem.blocks) || return false
    for (block, block_signature) in zip(problem.blocks, signature.blocks)
        block_signature isa NamedTuple || return false
        block_required = (:size, :global_positions, :local_positions, :diagonal_positions)
        all(name -> name in keys(block_signature), block_required) || return false
        block_signature.size == block.size || return false
        collect(block_signature.global_positions) == block.global_positions || return false
        collect(block_signature.local_positions) == block.local_positions || return false
        collect(block_signature.diagonal_positions) == block.diagonal_positions || return false
    end
    return true
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
    )
end

function _record_successful_facial_reduction!(
    opt::Optimizer,
    problem::ProblemData,
    reduction::_CertifiedFacialReduction,
)
    save_path = _facial_reduction_cache_path(opt.settings.facial_reduction_save_file)
    save_path === nothing && return
    push!(opt.facial_reduction_save_records, _facial_reduction_record(problem, reduction))
    _record_approximate_cache_memory!(:facial_reduction, opt.facial_reduction_save_records)
    full_path = _write_facial_reduction_cache(
        save_path,
        opt.facial_reduction_save_records,
    )
    _log(
        opt,
        "Facial reduction: saved $(length(opt.facial_reduction_save_records)) reduction record(s) to $(full_path)",
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
    return _CertifiedFacialReduction(String(record.source), exposed_scalars, keep_bases)
end

function _exact_column_rank(matrix::Matrix{ExactRational})
    size(matrix, 2) == 0 && return 0
    _, pivots = _rref(hcat(matrix, zeros(ExactRational, size(matrix, 1))))
    return length(pivots)
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

function _cached_keep_basis_violation(
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

    removed_directions = _nullspace_basis_exact(Matrix(transpose(keep_basis)))
    directions = [
        collect(view(removed_directions, :, column)) for
        column in axes(removed_directions, 2)
    ]
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
)
    if isempty(reduction.exposed_scalars) && isempty(reduction.keep_bases)
        return "cached reduction has no exposed scalar or PSD face"
    end
    for position in reduction.exposed_scalars
        violation = _cached_scalar_face_violation(problem, position)
        violation === nothing || return violation
    end
    for block_index in sort(collect(keys(reduction.keep_bases)))
        violation = _cached_keep_basis_violation(
            problem,
            block_index,
            reduction.keep_bases[block_index],
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

        violation = _cached_facial_reduction_violation(current, reduction)
        if violation !== nothing
            _log(opt, "Facial reduction: cached face did not validate ($(violation))")
            continue
        end

        reduced_problem = _apply_facial_reduction(
            current,
            reduction.exposed_scalars,
            reduction.keep_bases,
            ;
            # The exact cache validation immediately above is the certificate
            # for this application.  Re-validating inside _apply_facial_reduction
            # duplicates the expensive exact row-space/PSD checks.
            certified = false,
        )
        if reduced_problem.affine === nothing
            _log(
                opt,
                "Facial reduction: cached face produced an inconsistent affine system; ignoring it",
            )
            continue
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
        return _CertifiedFacialReduction(source, exposed_scalars, keep_bases)
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
        numeric_matrix = try
            Float64.(matrix)
        catch
            nothing
        end
        numeric_matrix === nothing && continue
        all(isfinite, numeric_matrix) || continue
        scale = max(1.0, opnorm(numeric_matrix, 1))
        tolerance = 100 * eps(Float64) * scale * max(1, size(matrix, 1))
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

    if row_count * (size(problem.A, 2) + 1) > _SIEVE_TRANSFORM_MAX_ENTRIES
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

    configured_type = _phase1_hypatia_float_type(opt.settings)
    if _phase1_hypatia_float_type_is_auto(opt.settings) &&
       configured_type != Float64 &&
       _phase1_hypatia_prefers_sparse_float64(problem)
        _log(
            opt,
            "Facial reduction oracle: using Float64 sparse linear algebra for this large sparse model",
        )
        return Float64
    end
    return configured_type
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
) where {F<:AbstractFloat}
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
    evidence.kind == :dual_slack &&
        return _certify_dual_slack_evidence(opt, problem, evidence, F; cache)
    evidence.kind == :boundary_primal &&
        return _certify_boundary_primal_evidence(opt, problem, evidence, F; cache)
    evidence.kind == :oracle_point &&
        return _certify_oracle_point_evidence(opt, problem, evidence, F; cache)
    error("Unhandled facial reduction evidence kind $(evidence.kind).")
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

    merged = _CertifiedFacialReduction(
        join(unique(reduction.source for reduction in reductions), " + "),
        exposed_scalars,
        keep_bases,
    )
    _cached_facial_reduction_violation(problem, merged) === nothing || return nothing
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
        )
        reduced.affine === nothing && break
        old_dimension = _barrier_dimension(current)
        new_dimension = _barrier_dimension(reduced)
        new_dimension < old_dimension || break
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
    total_dimension::Int,
)
    rows = Vector{Vector{ExactRational}}()
    rhs = ExactRational[]
    if reduced_block === nothing
        for position in block.global_positions
            row = zeros(ExactRational, total_dimension)
            row[position] = 1 // 1
            push!(rows, row)
            push!(rhs, 0 // 1)
        end
        return rows, rhs
    end

    for (old_local_index, (i, j)) in enumerate(block.local_positions)
        row = zeros(ExactRational, total_dimension)
        row[block.global_positions[old_local_index]] = 1 // 1
        for (new_local_index, (a, b)) in enumerate(reduced_block.local_positions)
            coefficient = if a == b
                keep_basis[i, a] * keep_basis[j, a]
            else
                keep_basis[i, a] * keep_basis[j, b] + keep_basis[i, b] * keep_basis[j, a]
            end
            iszero(coefficient) && continue
            row[reduced_block.global_positions[new_local_index]] -= coefficient
        end
        push!(rows, row)
        push!(rhs, 0 // 1)
    end

    return rows, rhs
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
    oracle_point = _facial_reduction_oracle_attempt(
        opt,
        problem,
        HF;
        normalization_row,
    )
    oracle_point === nothing && return nothing

    evidence = _FacialReductionEvidence(:oracle_point, String(source), oracle_point)
    reduction = _certify_oracle_point_evidence(
        opt,
        problem,
        evidence,
        HF;
        cache,
        normalization_row,
    )
    reduction === nothing && return nothing
    return reduction
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
    return _merge_certified_facial_reductions(opt, problem, reductions)
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
        next === nothing && break
        merged = _merge_certified_facial_reductions(opt, problem, [current, next])
        merged === nothing && break
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
        new_removed > old_removed || break
        current = merged
    end
    return current
end

function _apply_facial_reduction(
    problem::ProblemData,
    exposed_scalars::Vector{Int},
    keep_bases::Dict{Int,Matrix{ExactRational}},
    ;
    certified::Bool = false,
)
    if certified
        violation = _cached_facial_reduction_violation(
            problem,
            _CertifiedFacialReduction("in-memory", exposed_scalars, keep_bases),
        )
        violation === nothing ||
            error("Certified facial reduction failed exact preservation checks: $(violation)")
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
    A = zeros(ExactRational, size(problem.A, 1), total_dimension)
    if !isempty(problem.A)
        A[:, 1:size(problem.A, 2)] = problem.A
    end
    b = copy(problem.b)
    extra_rows = Vector{Vector{ExactRational}}()
    extra_rhs = ExactRational[]

    for position in unique(sort(exposed_scalars))
        row = zeros(ExactRational, total_dimension)
        row[position] = 1 // 1
        push!(extra_rows, row)
        push!(extra_rhs, 0 // 1)

    end

    for (block_index, block) in enumerate(problem.blocks)
        keep_basis = get(keep_bases, block_index, nothing)
        keep_basis === nothing && continue
        rows, rhs = _face_reduction_rows(
            block,
            keep_basis,
            get(block_replacements, block_index, nothing),
            total_dimension,
        )
        append!(extra_rows, rows)
        append!(extra_rhs, rhs)
    end

    if !isempty(extra_rows)
        A_augmented = zeros(ExactRational, size(A, 1) + length(extra_rows), total_dimension)
        b_augmented = zeros(ExactRational, length(b) + length(extra_rhs))
        if size(A, 1) > 0
            A_augmented[1:size(A, 1), :] = A
            b_augmented[1:length(b)] = b
        end
        for (offset, row) in enumerate(extra_rows)
            A_augmented[size(A, 1) + offset, :] = row
            b_augmented[length(b) + offset] = extra_rhs[offset]
        end
        A = A_augmented
        b = b_augmented
    end

    positive_scalars = [index for index in problem.positive_scalars if !(index in exposed_scalars)]
    objective_extension = zeros(ExactRational, total_dimension - old_dimension)

    restriction_matrix = if isempty(extra_rows)
        zeros(ExactRational, 0, total_dimension)
    else
        reduce(vcat, (reshape(row, 1, :) for row in extra_rows))
    end
    affine = _extend_and_restrict_affine_system(
        problem.affine,
        total_dimension - old_dimension,
        restriction_matrix,
        extra_rhs,
    )
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
        affine = _solve_affine_system(A, b)
    end
    if size(A, 1) > _FACIAL_REDUCTION_AFFINE_COMPACTION_FACTOR * max(1, size(problem.A, 1))
        compacted = _independent_affine_equalities(A, b)
        if compacted !== nothing
            A, b = compacted
            affine = _solve_affine_system(A, b)
        end
    end
    positive_scalars, _ = _prune_positive_scalar_faces(positive_scalars, affine)
    blocks, A, b, affine, _ = _prune_psd_faces(blocks, A, b, affine)
    reduced_problem = ProblemData(
        problem.original_variables,
        blocks,
        positive_scalars,
        vcat(problem.objective_vector_raw, objective_extension),
        problem.objective_constant_raw,
        vcat(problem.objective_vector_min, objective_extension),
        A,
        b,
        affine,
        nothing,
        problem.scalar_constraint_rows,
        problem.psd_constraint_blocks,
    )
    new_barrier_dimension = _barrier_dimension(reduced_problem)
    new_barrier_dimension < old_barrier_dimension ||
        error("Facial reduction was applied without decreasing barrier dimension.")
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
        _is_inexact_facial_reduction_error(err) || rethrow()
        _log(opt, "Facial reduction: exact recovery of the candidate face was impossible")
        nothing
    end
    if reduction !== nothing
        reduced_problem = _apply_facial_reduction(
            problem,
            reduction.exposed_scalars,
            reduction.keep_bases,
            ;
            certified = false,
        )
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
    tentative_result = _tentative_feasibility_search_problem(
        opt,
        problem,
        candidate,
        F;
        return_details = true,
    )
    tentative_problem = tentative_result.problem
    if tentative_problem !== nothing
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
