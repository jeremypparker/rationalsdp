# Exact arithmetic helpers and affine/PSD structure manipulations.

_exact_rational(x::ExactRational) = x

function _exact_rational(x::Rational{S}) where {S<:Integer}
    return ExactRational(BigInt(numerator(x)), BigInt(denominator(x)))
end

_exact_rational(x::Integer) = ExactRational(BigInt(x), BigInt(1))

function _unsupported_exact_input_error(x, kind::AbstractString)
    throw(
        ArgumentError(
            "RationalSDP requires exact integer/rational model coefficients. " *
            "Received $(kind) value $(repr(x))::$(typeof(x)). " *
            "Rewrite it using explicit rationals such as `1//10`.",
        ),
    )
end

function _exact_rational(x::AbstractFloat)
    _unsupported_exact_input_error(x, "floating-point")
end

function _exact_rational(x::AbstractIrrational)
    _unsupported_exact_input_error(x, "irrational")
end

function _exact_rational(x::Real)
    throw(
        ArgumentError(
            "RationalSDP only accepts integer and rational model coefficients; " *
            "received $(repr(x))::$(typeof(x)).",
        ),
    )
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

function _dual_vector_to_matrix(
    x::AbstractVector{S},
    block::BlockStructure,
) where {S}
    X = zeros(S, block.size, block.size)
    for (local_index, (i, j)) in enumerate(block.local_positions)
        value = x[block.global_positions[local_index]]
        if i != j
            value /= 2
        end
        X[i, j] = value
        X[j, i] = value
    end
    return X
end

function _matrix_to_vector!(
    destination::AbstractVector{S},
    X::AbstractMatrix{S},
    block::BlockStructure,
) where {S}
    for (local_index, (i, j)) in enumerate(block.local_positions)
        destination[block.global_positions[local_index]] = X[i, j]
    end
    return destination
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

function _restrict_affine_system(
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
    rows::Matrix{ExactRational},
    rhs::Vector{ExactRational},
)
    affine === nothing && return nothing
    size(rows, 1) == length(rhs) || error("Affine restriction rows and rhs must match.")
    isempty(rhs) && return affine
    particular, nullspace = affine
    coordinate_affine = _solve_affine_system(rows * nullspace, rhs - rows * particular)
    coordinate_affine === nothing && return nothing
    coordinate_particular, coordinate_nullspace = coordinate_affine
    return (
        particular + nullspace * coordinate_particular,
        nullspace * coordinate_nullspace,
    )
end

function _coordinate_equality_rows(dimension::Int, indices::Vector{Int})
    unique_indices = unique(sort(indices))
    rows = zeros(ExactRational, length(unique_indices), dimension)
    for (row_index, index) in enumerate(unique_indices)
        rows[row_index, index] = 1 // 1
    end
    return rows
end

function _phase1_active_positions(problem::ProblemData)
    positions = Int[]
    append!(positions, problem.positive_scalars)
    for block in problem.blocks
        append!(positions, block.global_positions)
    end
    return unique(sort(positions))
end

function _compute_phase1_nullspace(
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
)
    affine === nothing && return nothing
    _, nullspace = affine
    size(nullspace, 2) == 0 && return nullspace

    active_positions = Int[]
    append!(active_positions, positive_scalars)
    for block in blocks
        append!(active_positions, block.global_positions)
    end
    active_positions = unique(sort(active_positions))
    isempty(active_positions) && return zeros(ExactRational, size(nullspace, 1), 0)

    reduced = nullspace[active_positions, :]
    numeric_reduced = Matrix{Float64}(undef, size(reduced))
    for column in axes(reduced, 2), row in axes(reduced, 1)
        numeric_reduced[row, column] = Float64(reduced[row, column])
    end
    factorization = qr(numeric_reduced, ColumnNorm())
    diagonal = abs.(diag(factorization.R))
    rank_tolerance =
        isempty(diagonal) ? 0.0 : maximum(diagonal) * max(size(numeric_reduced)...) * eps(Float64)
    rank = count(value -> value > rank_tolerance, diagonal)
    pivots = isempty(diagonal) ? Int[] : sort(factorization.p[1:rank])
    isempty(pivots) && return zeros(ExactRational, size(nullspace, 1), 0)
    return nullspace[:, pivots]
end

function _phase1_nullspace(problem::ProblemData)
    problem.affine === nothing && error("Phase I nullspace requested without affine data.")
    if problem.phase1_nullspace === nothing
        problem.phase1_nullspace = _compute_phase1_nullspace(
            problem.blocks,
            problem.positive_scalars,
            problem.affine,
        )
    end
    return problem.phase1_nullspace
end

function ProblemData(
    original_variables::Vector{MOI.VariableIndex},
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    objective_vector_raw::Vector{ExactRational},
    objective_constant_raw::ExactRational,
    objective_vector_min::Vector{ExactRational},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
)
    return ProblemData(
        original_variables,
        blocks,
        positive_scalars,
        objective_vector_raw,
        objective_constant_raw,
        objective_vector_min,
        A,
        b,
        affine,
        nothing,
        Dict{Any,Vector{Int}}(),
        Dict{Any,Int}(),
    )
end

function ProblemData(
    original_variables::Vector{MOI.VariableIndex},
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    objective_vector_raw::AbstractVector,
    objective_constant_raw,
    objective_vector_min::AbstractVector,
    A::AbstractMatrix,
    b::AbstractVector,
    affine,
)
    objective_vector_raw_exact = ExactRational[_exact_rational(value) for value in objective_vector_raw]
    objective_vector_min_exact = ExactRational[_exact_rational(value) for value in objective_vector_min]
    A_exact = ExactRational[_exact_rational(A[row, column]) for row in axes(A, 1), column in axes(A, 2)]
    b_exact = ExactRational[_exact_rational(value) for value in b]
    affine_exact = if affine === nothing
        nothing
    else
        particular, nullspace = affine
        (
            ExactRational[_exact_rational(value) for value in particular],
            ExactRational[_exact_rational(nullspace[row, column]) for row in axes(nullspace, 1), column in axes(nullspace, 2)],
        )
    end
    return ProblemData(
        original_variables,
        blocks,
        positive_scalars,
        objective_vector_raw_exact,
        _exact_rational(objective_constant_raw),
        objective_vector_min_exact,
        A_exact,
        b_exact,
        affine_exact,
    )
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
    variables = Union{Nothing,MOI.VariableIndex}[]
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

function _prune_positive_scalar_faces(
    positive_scalars::Vector{Int},
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
)
    affine === nothing && return positive_scalars, 0
    particular, nullspace = affine
    pruned = Int[]
    kept = Int[]
    for index in positive_scalars
        if _variable_fixed_zero(particular, nullspace, index)
            push!(pruned, index)
        else
            push!(kept, index)
        end
    end
    return kept, length(pruned)
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
        affine = _restrict_affine_system(
            affine,
            _coordinate_equality_rows(size(A, 2), zero_indices),
            zeros(ExactRational, length(unique(sort(zero_indices)))),
        )
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

function _positive_semidefinite_exact(matrix::Matrix{ExactRational})
    size(matrix, 1) == size(matrix, 2) || return false
    n = size(matrix, 1)
    n == 0 && return true
    any(matrix .!= transpose(matrix)) && return false

    remainder = copy(matrix)
    active = collect(1:n)
    while !isempty(active)
        pivot_position = findfirst(index -> remainder[index, index] > 0, eachindex(active))
        if pivot_position === nothing
            for index in eachindex(active)
                if !iszero(remainder[index, index])
                    return false
                end
                for column in 1:length(active)
                    if !iszero(remainder[index, column]) || !iszero(remainder[column, index])
                        return false
                    end
                end
            end
            return true
        end

        if pivot_position != 1
            permutation = [pivot_position; setdiff(collect(1:length(active)), pivot_position)]
            remainder = remainder[permutation, permutation]
            active = active[permutation]
        end

        pivot = remainder[1, 1]
        pivot > 0 || return false
        if length(active) == 1
            return true
        end

        trailing = Matrix{ExactRational}(undef, length(active) - 1, length(active) - 1)
        for row in 2:length(active), column in 2:length(active)
            trailing[row - 1, column - 1] =
                remainder[row, column] - remainder[row, 1] * remainder[1, column] / pivot
        end
        remainder = trailing
        active = active[2:end]
    end

    return true
end

function _positive_definite_exact(matrix::Matrix{ExactRational})
    size(matrix, 1) == size(matrix, 2) || return false
    n = size(matrix, 1)
    n == 0 && return true
    any(matrix .!= transpose(matrix)) && return false

    remainder = copy(matrix)
    active = collect(1:n)
    while !isempty(active)
        pivot_position = findfirst(index -> remainder[index, index] > 0, eachindex(active))
        pivot_position === nothing && return false
        if pivot_position != 1
            permutation = [pivot_position; setdiff(collect(1:length(active)), pivot_position)]
            remainder = remainder[permutation, permutation]
            active = active[permutation]
        end

        pivot = remainder[1, 1]
        pivot > 0 || return false
        length(active) == 1 && return true
        trailing = Matrix{ExactRational}(undef, length(active) - 1, length(active) - 1)
        for row in 2:length(active), column in 2:length(active)
            trailing[row - 1, column - 1] =
                remainder[row, column] - remainder[row, 1] * remainder[1, column] / pivot
        end
        remainder = trailing
        active = active[2:end]
    end

    return true
end

function _nullspace_basis_exact(matrix::Matrix{ExactRational})
    rhs = zeros(ExactRational, size(matrix, 1))
    affine = _solve_affine_system(matrix, rhs)
    affine === nothing && return zeros(ExactRational, size(matrix, 2), 0)
    _, nullspace = affine
    return nullspace
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
        if !_positive_definite_exact(_vector_to_matrix(x, block))
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
    return _numeric_affine_data(particular, nullspace, F)
end

function _numeric_affine_data(
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    ::Type{F},
) where {F<:AbstractFloat}
    particular_numeric = _to_working_array(F, particular)
    return NumericAffineData{F}(particular_numeric, nullspace, nothing, nothing, nothing)
end

function _numeric_exact_nullspace!(numeric_affine::NumericAffineData{F}) where {F<:AbstractFloat}
    if size(numeric_affine.exact_nullspace, 2) == 0
        if numeric_affine.numeric_exact_nullspace === nothing
            numeric_affine.numeric_exact_nullspace = Matrix{F}(undef, size(numeric_affine.exact_nullspace)...)
        end
        return numeric_affine.numeric_exact_nullspace
    end
    if numeric_affine.numeric_exact_nullspace === nothing
        numeric_affine.numeric_exact_nullspace = _to_working_array(F, numeric_affine.exact_nullspace)
    end
    return numeric_affine.numeric_exact_nullspace
end

function _numeric_nullspace!(numeric_affine::NumericAffineData{F}) where {F<:AbstractFloat}
    if size(numeric_affine.exact_nullspace, 2) == 0
        if numeric_affine.numeric_phase2_basis === nothing
            numeric_affine.numeric_phase2_basis = Matrix{F}(undef, size(numeric_affine.exact_nullspace)...)
        end
        return numeric_affine.numeric_phase2_basis
    end
    if numeric_affine.numeric_phase2_basis === nothing
        exact_numeric_nullspace = _numeric_exact_nullspace!(numeric_affine)
        basis = Matrix{F}(I, size(exact_numeric_nullspace, 1), size(exact_numeric_nullspace, 2))
        factor = qr(exact_numeric_nullspace)
        lmul!(factor.Q, basis)
        numeric_affine.numeric_phase2_basis = basis
    end
    return numeric_affine.numeric_phase2_basis
end

function _nullspace_factor!(numeric_affine::NumericAffineData{F}) where {F<:AbstractFloat}
    size(numeric_affine.exact_nullspace, 2) == 0 && return nothing
    if numeric_affine.nullspace_factor === nothing
        numeric_affine.nullspace_factor = qr(_numeric_exact_nullspace!(numeric_affine))
    end
    return numeric_affine.nullspace_factor
end
