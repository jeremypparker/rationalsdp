# Internal facial reduction on the extracted primal conic form.
#
# We use Hypatia as a floating-point oracle to find exposing vectors for the
# current affine slice, then convert the exposed nullspace directions back into
# exact rational linear constraints on the original primal variables.

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

function _block_row_linear_form(
    block::BlockStructure,
    row_index::Int,
    direction::Vector{ExactRational},
    dimension::Int,
)
    form = zeros(ExactRational, dimension)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        coefficient = zero(ExactRational)
        if i == row_index && j == row_index
            coefficient = direction[row_index]
        elseif i == row_index
            coefficient = direction[j]
        elseif j == row_index
            coefficient = direction[i]
        end
        iszero(coefficient) && continue
        form[block.global_positions[local_index]] += coefficient
    end
    return form
end

function _block_quadratic_linear_form(
    block::BlockStructure,
    direction::Vector{ExactRational},
    dimension::Int,
)
    form = zeros(ExactRational, dimension)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        coefficient = if i == j
            direction[i] * direction[i]
        else
            2 * direction[i] * direction[j]
        end
        iszero(coefficient) && continue
        form[block.global_positions[local_index]] += coefficient
    end
    return form
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

function _block_annihilation_violation(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
)
    problem.affine === nothing && return "no exact affine parametrization is available"
    particular, nullspace = problem.affine
    dimension = length(problem.objective_vector_raw)
    for row_index in 1:block.size
        form = _block_row_linear_form(block, row_index, direction, dimension)
        particular_value = dot(form, particular)
        if !iszero(particular_value)
            return "row=$(row_index), affine=particular, value=$(_format_exact_rational_compact(particular_value))"
        end
        for basis_index in axes(nullspace, 2)
            basis_value = dot(form, view(nullspace, :, basis_index))
            if !iszero(basis_value)
                return "row=$(row_index), affine_basis=$(basis_index), value=$(_format_exact_rational_compact(basis_value))"
            end
        end
    end
    return nothing
end

function _block_quadratic_vanish_violation(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
)
    problem.affine === nothing && return "no exact affine parametrization is available"
    particular, nullspace = problem.affine
    dimension = length(problem.objective_vector_raw)
    form = _block_quadratic_linear_form(block, direction, dimension)
    particular_value = dot(form, particular)
    if !iszero(particular_value)
        return "quadratic=particular, value=$(_format_exact_rational_compact(particular_value))"
    end
    for basis_index in axes(nullspace, 2)
        basis_value = dot(form, view(nullspace, :, basis_index))
        if !iszero(basis_value)
            return "quadratic=affine_basis=$(basis_index), value=$(_format_exact_rational_compact(basis_value))"
        end
    end
    return nothing
end

function _block_trace_vanish_violation(
    problem::ProblemData,
    block::BlockStructure,
    directions::Vector{Vector{ExactRational}},
)
    isempty(directions) && return "no directions"
    problem.affine === nothing && return "no exact affine parametrization is available"
    particular, nullspace = problem.affine
    dimension = length(problem.objective_vector_raw)
    form = zeros(ExactRational, dimension)
    for direction in directions
        form .+= _block_quadratic_linear_form(block, direction, dimension)
    end
    particular_value = dot(form, particular)
    if !iszero(particular_value)
        return "trace=particular, value=$(_format_exact_rational_compact(particular_value))"
    end
    for basis_index in axes(nullspace, 2)
        basis_value = dot(form, view(nullspace, :, basis_index))
        if !iszero(basis_value)
            return "trace=affine_basis=$(basis_index), value=$(_format_exact_rational_compact(basis_value))"
        end
    end
    return nothing
end

function _block_face_direction_certificate(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
)
    row_violation = _block_annihilation_violation(problem, block, direction)
    row_violation === nothing && return (kind = :affine_rows, violation = nothing)

    diagonal_violation = _block_quadratic_vanish_violation(problem, block, direction)
    diagonal_violation === nothing && return (kind = :psd_diagonal, violation = nothing)

    return (
        kind = :none,
        violation =
            "row certificate failed ($(row_violation)); PSD diagonal certificate failed ($(diagonal_violation))",
    )
end

function _exact_face_direction(
    problem::ProblemData,
    block::BlockStructure,
    candidate::Vector{F},
    settings::Settings,
    ::Type{F},
) where {F<:AbstractFloat}
    for tolerance in _recovery_tolerances(settings, F)
        direction = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for value in candidate
        ]
        direction = _normalize_rational_direction(direction)
        any(!iszero, direction) || continue
        if _block_face_direction_certificate(problem, block, direction).kind != :none
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
) where {F<:AbstractFloat}
    candidates = _pivoted_rational_subspace_directions(subspace, opt.settings, F)
    isempty(candidates) && return Vector{ExactRational}[]

    accepted = Vector{Vector{ExactRational}}()
    rejected = 0
    last_violation = nothing
    for direction in candidates
        violation = _block_annihilation_violation(problem, block, direction)
        if violation === nothing
            push!(accepted, direction)
        else
            rejected += 1
            last_violation = violation
        end
    end

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
)
    problem.affine === nothing && return Vector{ExactRational}[]
    particular, nullspace = problem.affine
    basis = Matrix{ExactRational}(I, block.size, block.size)

    basis = basis * _nullspace_basis_exact(_vector_to_matrix(particular, block) * basis)
    for column in axes(nullspace, 2)
        size(basis, 2) == 0 && break
        basis = basis * _nullspace_basis_exact(_vector_to_matrix(view(nullspace, :, column), block) * basis)
    end

    directions = [
        _normalize_rational_direction(collect(view(basis, :, column))) for column in axes(basis, 2)
    ]
    filter!(direction -> any(!iszero, direction), directions)
    return _linearly_independent_directions(directions)
end

function _candidate_kernel_directions(
    opt::Optimizer,
    problem::ProblemData,
    block_index::Int,
    block_matrix::Matrix{F},
    ::Type{F},
) where {F<:AbstractFloat}
    block = problem.blocks[block_index]
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = eigen(symmetric_matrix)
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
    )
    isempty(pivoted_directions) || return pivoted_directions

    exact_directions = _exact_block_nullspace_directions(problem, block)

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
            certificate = _block_face_direction_certificate(problem, block, direction)
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
            else
                rejected_heuristics += 1
                _log(
                    opt,
                    "Facial reduction: rejected rationalized boundary kernel direction for PSD block $(block_index) (eig=$(_format_metric(eigen_factor.values[kernel_index])), residual=$(_format_metric(heuristic.residual)), rationalize_tol=$(_format_metric(heuristic.tolerance))); no exact face certificate ($(certificate.violation)): $(direction_summary)",
                )
                push!(heuristic_directions, direction)
            end
        end
        heuristic_directions = _linearly_independent_directions(heuristic_directions)
        if length(heuristic_directions) > 1
            violation = _block_trace_vanish_violation(problem, block, heuristic_directions)
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
            individually_certified_directions = individually_certified_directions[1:1]
            _log(
                opt,
                "Facial reduction: using certified rationalized boundary kernel directions for PSD block $(block_index)",
            )
            return individually_certified_directions
        end

        if rejected_heuristics > 0
            _log(
                opt,
                "Facial reduction: rejected $(rejected_heuristics) rationalized boundary kernel direction(s) for PSD block $(block_index); deferring uncertified directions to tentative fallback",
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
    block = problem.blocks[block_index]
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = eigen(symmetric_matrix)
    exposure_tolerance = max(
        _to_working_float(F, opt.settings.facial_reduction_exposure_tolerance),
        F(100) * eps(F),
    )
    kernel_indices = [
        index for (index, value) in enumerate(eigen_factor.values) if abs(value) <= exposure_tolerance
    ]
    isempty(kernel_indices) && return NamedTuple[]

    candidates = NamedTuple[]
    for kernel_index in sort(kernel_indices; by = index -> abs(eigen_factor.values[index]))
        heuristic = _heuristic_kernel_direction(
            block_matrix,
            collect(view(eigen_factor.vectors, :, kernel_index)),
            opt.settings,
            F,
        )
        heuristic === nothing && continue
        push!(
            candidates,
            (
                residual = heuristic.residual,
                eigenvalue = eigen_factor.values[kernel_index],
                tolerance = heuristic.tolerance,
                block_index = block_index,
                direction = heuristic.direction,
            ),
        )
    end
    return candidates
end

function _tentative_facial_reduction_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
) where {F<:AbstractFloat}
    candidates = NamedTuple[]
    for (block_index, block) in enumerate(problem.blocks)
        matrix = _vector_to_matrix(candidate, block)
        append!(
            candidates,
            _heuristic_kernel_direction_candidates(opt, problem, block_index, matrix, F),
        )
    end
    sort!(candidates; by = item -> (abs(item.residual), abs(item.eigenvalue), item.block_index))

    for item in candidates
        keep_basis = _orthogonal_complement_basis(
            [item.direction],
            problem.blocks[item.block_index].size,
        )
        reduced_problem = _apply_facial_reduction(
            problem,
            Int[],
            Dict(item.block_index => keep_basis),
        )
        if reduced_problem.affine === nothing
            _log(
                opt,
                "Facial reduction: rejected tentative rationalized boundary kernel direction for PSD block $(item.block_index); exact reduced affine system is inconsistent",
            )
            continue
        end
        _log(
            opt,
            "Facial reduction: using tentative rationalized boundary kernel direction for PSD block $(item.block_index) (eig=$(_format_metric(item.eigenvalue)), residual=$(_format_metric(item.residual)), rationalize_tol=$(_format_metric(item.tolerance))); exact reduced affine system is consistent",
        )
        _log(opt, "facial reduction: fixed 0 scalar cone direction(s) and removed 1 PSD direction(s)")
        return reduced_problem
    end

    isempty(candidates) || _log(
        opt,
        "Facial reduction: rejected $(length(candidates)) tentative rationalized boundary kernel direction(s)",
    )
    return nothing
end

function _is_inexact_facial_reduction_error(err)
    err isa ErrorException || return false
    message = err.msg
    return occursin("could not be represented exactly over the rational coefficient field", message)
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
        eigenvalues = eigvals(symmetric_matrix)
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
) where {F<:AbstractFloat}
    row_count = size(problem.A, 1)
    cone_positions = _phase1_active_positions(problem)
    equality_matrix, equality_rhs = _facial_reduction_oracle_equalities(problem)
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

function _facial_reduction_oracle_attempt(
    opt::Optimizer,
    problem::ProblemData,
    ::Type{HF},
) where {HF<:AbstractFloat}
    return _with_float_precision(HF, opt.settings.working_precision, function (::Type{HF})
        model = _build_facial_reduction_oracle(problem, HF)
        syssolver, use_dense_model, preprocess = _hypatia_phase1_syssolver(opt.settings, HF)
        tolerance_kwargs = _phase1_hypatia_tolerance_kwargs(opt.settings, HF)
        solver = Hypatia.Solvers.Solver{HF}(
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
            return nothing
        end
        elapsed_sec = (time_ns() - start_time) / 1.0e9
        status = Hypatia.Solvers.get_status(solver)
        if status ∉ (Hypatia.Solvers.Optimal, Hypatia.Solvers.NearOptimal)
            _log(
                opt,
                "Facial reduction oracle: status=$(status), time=$(@sprintf("%.2f", elapsed_sec))s",
            )
            return nothing
        end
        candidate = try
            vec(collect(Hypatia.Solvers.get_x(solver)))
        catch
            nothing
        end
        candidate === nothing && return nothing
        all(isfinite, candidate) || return nothing
        _log(
            opt,
            "Facial reduction oracle: status=$(status), iter=$(Hypatia.Solvers.get_num_iters(solver)), time=$(@sprintf("%.2f", elapsed_sec))s",
        )
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

function _facial_reduction_slack(problem::ProblemData, y::Vector{ExactRational})
    s = transpose(problem.A) * y
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

function _facial_reduction_oracle_equalities(problem::ProblemData)
    equality_rows = Vector{Vector{ExactRational}}()
    equality_rhs = ExactRational[]
    for position in _facial_reduction_free_positions(problem)
        push!(equality_rows, collect(problem.A[:, position]))
        push!(equality_rhs, 0 // 1)
    end
    push!(equality_rows, copy(problem.b))
    push!(equality_rhs, 0 // 1)
    push!(equality_rows, _facial_reduction_trace_row(problem))
    push!(equality_rhs, 1 // 1)

    return transpose(hcat(equality_rows...)), equality_rhs
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
) where {F<:AbstractFloat}
    free_positions = _facial_reduction_free_positions(problem)
    trace_row = _facial_reduction_trace_row(problem)
    equality_matrix, equality_rhs = _facial_reduction_oracle_equalities(problem)
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
        s, scalar_slack, block_slack = _facial_reduction_slack(problem, y)

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

_facial_reduction_tuple(reduction::_CertifiedFacialReduction) =
    reduction.exposed_scalars, reduction.keep_bases

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

function _certify_dual_slack_evidence(
    opt::Optimizer,
    problem::ProblemData,
    evidence::_FacialReductionEvidence{F},
    ::Type{F},
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
) where {F<:AbstractFloat}
    keep_bases = Dict{Int,Matrix{ExactRational}}()

    for (block_index, block) in enumerate(problem.blocks)
        matrix = _vector_to_matrix(evidence.vector, block)
        directions = _candidate_kernel_directions(opt, problem, block_index, matrix, F)
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
) where {F<:AbstractFloat}
    block = problem.blocks[block_index]
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigen_factor = eigen(symmetric_matrix)
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
        )
        isempty(pivoted_directions) || return pivoted_directions
    end

    qr_factor = qr(Matrix(symmetric_matrix), ColumnNorm())
    candidate_columns = unique(qr_factor.p[1:numeric_rank])
    exact_directions = Vector{Vector{ExactRational}}()
    for column_index in candidate_columns
        direction = _exact_face_direction(
            problem,
            block,
            collect(view(block_matrix, :, column_index)),
            opt.settings,
            F,
        )
        direction === nothing && continue
        push!(exact_directions, direction)
    end
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
) where {F<:AbstractFloat}
    exact_oracle_slack = _exact_facial_reduction_oracle_slack(
        opt,
        problem,
        evidence.vector,
        F,
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
) where {F<:AbstractFloat}
    evidence.kind == :dual_slack &&
        return _certify_dual_slack_evidence(opt, problem, evidence, F)
    evidence.kind == :boundary_primal &&
        return _certify_boundary_primal_evidence(opt, problem, evidence, F)
    evidence.kind == :oracle_point &&
        return _certify_oracle_point_evidence(opt, problem, evidence, F)
    error("Unhandled facial reduction evidence kind $(evidence.kind).")
end

function _first_certified_facial_reduction(
    opt::Optimizer,
    problem::ProblemData,
    evidence_list::Vector{_FacialReductionEvidence{F}},
    ::Type{F},
) where {F<:AbstractFloat}
    for evidence in evidence_list
        reduction = _certify_facial_reduction_evidence(opt, problem, evidence, F)
        reduction === nothing && continue
        return reduction
    end
    return nothing
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
) where {HF<:AbstractFloat}
    oracle_point = _facial_reduction_oracle_attempt(opt, problem, HF)
    oracle_point === nothing && return nothing

    evidence = _FacialReductionEvidence(:oracle_point, "oracle", oracle_point)
    reduction = _certify_facial_reduction_evidence(opt, problem, evidence, HF)
    reduction === nothing && return nothing
    return _facial_reduction_tuple(reduction)
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
) where {F<:AbstractFloat}
    reduction = _first_certified_facial_reduction(
        opt,
        problem,
        _cheap_facial_reduction_evidence(candidate, phase1_dual_slack, F),
        F,
    )
    reduction === nothing && return nothing
    return _facial_reduction_tuple(reduction)
end

function _face_membership_rows(
    block::BlockStructure,
    keep_basis::Matrix{ExactRational},
    dimension::Int,
)
    removed_directions = _nullspace_basis_exact(Matrix(transpose(keep_basis)))
    rows = Vector{Vector{ExactRational}}()
    for column in axes(removed_directions, 2)
        direction = collect(view(removed_directions, :, column))
        for row_index in 1:block.size
            push!(rows, _block_row_linear_form(block, row_index, direction, dimension))
        end
    end
    return rows
end

function _apply_facial_reduction(
    problem::ProblemData,
    exposed_scalars::Vector{Int},
    keep_bases::Dict{Int,Matrix{ExactRational}},
)
    old_dimension = length(problem.objective_vector_raw)
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
    face_rows_old = Vector{Vector{ExactRational}}()

    for position in unique(sort(exposed_scalars))
        row = zeros(ExactRational, total_dimension)
        row[position] = 1 // 1
        push!(extra_rows, row)
        push!(extra_rhs, 0 // 1)

        face_row = zeros(ExactRational, old_dimension)
        face_row[position] = 1 // 1
        push!(face_rows_old, face_row)
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
        append!(face_rows_old, _face_membership_rows(block, keep_basis, old_dimension))
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

    # Recompute the affine representation from the reduced system directly.
    # Incrementally lifting the old affine basis can drift away from the exact
    # reduced equations after multiple PSD face reductions on SOS-style models.
    affine = _solve_affine_system(A, b)
    positive_scalars, _ = _prune_positive_scalar_faces(positive_scalars, affine)
    blocks, A, b, affine, _ = _prune_psd_faces(blocks, A, b, affine)
    return ProblemData(
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
end

function _facial_reduction_round(
    opt::Optimizer,
    problem::ProblemData,
) 
    return nothing
end

function _facial_reduction_round(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
) where {F<:AbstractFloat}
    return _facial_reduction_round(opt, problem, candidate, nothing, F)
end

function _facial_reduction_round(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
) where {F<:AbstractFloat}
    problem.affine === nothing && return nothing
    reduction = _certified_facial_reduction_from_initial_evidence(
        opt,
        problem,
        candidate,
        phase1_dual_slack,
        F,
    )
    reduction === nothing || return reduction
    _log(
        opt,
        "Facial reduction: initial evidence found no exact reducing face; trying exposing-vector oracle",
    )
    return _facial_reduction_oracle_round(
        opt,
        problem,
        _facial_reduction_oracle_float_type(opt, problem),
    )
end

function _facially_reduce_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
) where {F<:AbstractFloat}
    return _facially_reduce_problem(opt, problem, candidate, nothing, F)
end

function _facially_reduce_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    phase1_dual_slack::Union{Nothing,Vector{F}},
    ::Type{F},
) where {F<:AbstractFloat}
    opt.settings.facial_reduction || return problem
    reduction = try
        _facial_reduction_round(opt, problem, candidate, phase1_dual_slack, F)
    catch err
        if _is_inexact_facial_reduction_error(err)
            _log(
                opt,
                "Facial reduction: exact face recovery failed for a non-coordinate face; trying tentative rationalized kernel direction",
            )
            nothing
        else
            rethrow()
        end
    end
    if reduction === nothing
        tentative_problem = _tentative_facial_reduction_problem(opt, problem, candidate, F)
        tentative_problem === nothing && return problem
        return tentative_problem
    end
    exposed_scalars, keep_bases = reduction
    removed_psd_directions = sum(
        (problem.blocks[index].size - size(keep_bases[index], 2) for index in keys(keep_bases));
        init = 0,
    )
    reduced_problem = _apply_facial_reduction(problem, exposed_scalars, keep_bases)
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
    _log(
        opt,
        "facial reduction: fixed $(length(exposed_scalars)) scalar cone direction(s) and removed $(removed_psd_directions) PSD direction(s)",
    )
    return reduced_problem
end
