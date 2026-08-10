# Exact arithmetic helpers and affine/PSD structure manipulations.

struct ExactLinearAlgebraError <: Exception
    operation::String
    backend_type::String
    backend_message::String
end

function Base.showerror(io::IO, err::ExactLinearAlgebraError)
    print(io, "Exact linear algebra failed during ", err.operation)
    isempty(err.backend_type) || print(io, " (", err.backend_type, ")")
    isempty(err.backend_message) || print(io, ": ", err.backend_message)
end

function _exception_type_name(err)
    T = typeof(err)
    return string(parentmodule(T), ".", nameof(T))
end

function _exception_message(err)
    # Nemo's FlintException contains the useful C-library message in `msg`,
    # but rendering the exception can itself hit an Enum world-age error in
    # long-lived REPLs. Read the string field directly when it is available.
    if hasfield(typeof(err), :msg)
        message = getfield(err, :msg)
        message isa AbstractString && return strip(String(message))
    end
    try
        return strip(sprint(showerror, err))
    catch
        return ""
    end
end

function _exact_linear_algebra_error(operation::AbstractString, err)
    return ExactLinearAlgebraError(
        String(operation),
        _exception_type_name(err),
        _exception_message(err),
    )
end

function _is_nemo_flint_exception(err)
    T = typeof(err)
    return parentmodule(T) === Nemo && nameof(T) === :FlintException
end

function _with_nemo_error(f::Function, operation::AbstractString)
    try
        return f()
    catch err
        _is_nemo_flint_exception(err) || rethrow()
        throw(_exact_linear_algebra_error(operation, err))
    end
end

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

# Keep Rational{BigInt} at the MOI and solver-state boundaries, and use Nemo
# only for the exact matrix kernels where FLINT provides the performance gain.
function _to_nemo_matrix(values::AbstractMatrix{ExactRational})
    return _with_nemo_error("exact rational matrix conversion") do
        Nemo.matrix(Nemo.QQ, values)
    end
end

function _nemo_matrix_product(
    left::AbstractMatrix{ExactRational},
    right::AbstractMatrix{ExactRational},
)
    size(left, 2) == size(right, 1) || error("Exact matrix product dimensions must match.")
    product = _with_nemo_error("exact rational matrix multiplication") do
        _to_nemo_matrix(left) * _to_nemo_matrix(right)
    end
    return _from_nemo_matrix(product)
end

function _nemo_matrix_product_chunked(
    left::AbstractMatrix{ExactRational},
    right::AbstractMatrix{ExactRational},
    chunk_columns::Int,
)
    size(left, 2) == size(right, 1) || error("Exact matrix product dimensions must match.")
    chunk_columns > 0 || error("Exact matrix-product chunk size must be positive.")
    column_count = size(right, 2)
    result = Matrix{ExactRational}(undef, size(left, 1), column_count)
    column_count == 0 && return result
    left_nemo = _to_nemo_matrix(left)
    for first_column in 1:chunk_columns:column_count
        last_column = min(column_count, first_column + chunk_columns - 1)
        right_chunk = _to_nemo_matrix(view(right, :, first_column:last_column))
        product_chunk = _with_nemo_error("chunked exact rational matrix multiplication") do
            left_nemo * right_chunk
        end
        result[:, first_column:last_column] = _from_nemo_matrix(product_chunk)
    end
    return result
end

function _from_nemo_rational(value)
    return _with_nemo_error("exact rational scalar extraction") do
        ExactRational(BigInt(numerator(value)), BigInt(denominator(value)))
    end
end

function _from_nemo_matrix(values)
    return _with_nemo_error("exact rational matrix extraction") do
        converted = Matrix{ExactRational}(undef, size(values)...)
        for column in axes(converted, 2), row in axes(converted, 1)
            converted[row, column] = _from_nemo_rational(values[row, column])
        end
        converted
    end
end

function _rref_pivots(reduced, variable_count::Int)
    pivot_rows = Int[]
    pivot_columns = Int[]
    # `reduced` is in RREF, so variable pivot columns are strictly increasing.
    # Continuing from the previous pivot avoids repeatedly crossing the dense
    # Nemo matrix from column one for every row.
    first_possible_column = 1
    for row in axes(reduced, 1)
        pivot_column = nothing
        for column in first_possible_column:variable_count
            if !iszero(reduced[row, column])
                pivot_column = column
                break
            end
        end
        pivot_column === nothing && continue
        push!(pivot_rows, row)
        push!(pivot_columns, pivot_column)
        first_possible_column = pivot_column + 1
    end
    return pivot_rows, pivot_columns
end

function _exact_rref(aug::Matrix{ExactRational})
    start_time = time_ns()
    try
        _, reduced = _with_nemo_error("exact rational RREF") do
            Nemo.rref(_to_nemo_matrix(aug))
        end
        return reduced
    finally
        stats = _current_facial_reduction_statistics()
        if stats isa FacialReductionStatistics
            stats.exact_rref_calls += 1
            push!(stats.exact_rref_dimensions, size(aug))
            stats.exact_rref_time_sec += (time_ns() - start_time) / 1.0e9
        end
    end
end

function _rref(
    aug::Matrix{ExactRational};
    checkpoint::Union{Nothing,Function} = nothing,
)
    checkpoint !== nothing && checkpoint("affine elimination: converting to Nemo and computing exact RREF")
    reduced_nemo = _exact_rref(aug)
    reduced = _from_nemo_matrix(reduced_nemo)
    checkpoint !== nothing && checkpoint("affine elimination: interpreting exact RREF")
    _, pivot_columns = _rref_pivots(reduced, size(aug, 2) - 1)
    return reduced, pivot_columns
end

function _affine_column_nonzero_counts(A::AbstractMatrix)
    column_nonzeros = zeros(Int, size(A, 2))
    for column in axes(A, 2)
        count = 0
        for row in axes(A, 1)
            count += !iszero(A[row, column])
        end
        column_nonzeros[column] = count
    end
    return column_nonzeros
end

function _affine_column_permutation(
    A::AbstractMatrix,
    column_nonzeros::AbstractVector{<:Integer} = _affine_column_nonzero_counts(A),
)
    length(column_nonzeros) == size(A, 2) ||
        error("Affine column nonzero counts have the wrong dimension.")
    permutation = collect(1:size(A, 2))
    sort!(
        permutation;
        by = column -> begin
            nnz = column_nonzeros[column]
            iszero(nnz) ? (1, 0, column) : (0, nnz, column)
        end,
    )
    return permutation
end

function _solve_affine_system(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    ;
    checkpoint::Union{Nothing,Function} = nothing,
    column_order::Symbol = :sparsity,
)
    size(A, 1) == length(b) || error("Affine equality matrix and rhs dimensions must match.")
    column_order in (:sparsity, :natural) || throw(
        ArgumentError(
            "Unsupported affine column ordering $(repr(column_order)); expected :sparsity or :natural.",
        ),
    )

    row_count, p = size(A)
    column_nonzeros = _affine_column_nonzero_counts(A)
    total_nonzeros = sum(column_nonzeros)
    zero_columns = count(iszero, column_nonzeros)
    singleton_columns = count(==(1), column_nonzeros)
    permutation = column_order == :sparsity ?
                  _affine_column_permutation(A, column_nonzeros) : collect(1:p)

    checkpoint !== nothing && checkpoint(
        "affine elimination: building dense $(row_count)-by-$(p + 1) augmented system; " *
        "coefficient matrix $(row_count)-by-$(p), total structural nonzeros=$(total_nonzeros), " *
        "zero columns=$(zero_columns), singleton columns=$(singleton_columns), " *
        "column ordering=$(column_order)",
    )

    if row_count == 0
        return zeros(ExactRational, p), Matrix{ExactRational}(I, p, p)
    end

    rhs_column = p + 1
    augmented = Matrix{ExactRational}(undef, row_count, rhs_column)
    for permuted_column in 1:p
        original_column = permutation[permuted_column]
        for row in 1:row_count
            augmented[row, permuted_column] = A[row, original_column]
        end
    end
    for row in 1:row_count
        augmented[row, rhs_column] = b[row]
    end
    checkpoint !== nothing && checkpoint(
        "affine elimination: converting to Nemo and computing exact RREF",
    )
    reduced = _exact_rref(augmented)
    checkpoint !== nothing && checkpoint("affine elimination: interpreting exact RREF")
    pivot_rows, pivot_columns = _rref_pivots(reduced, p)
    pivot_row_set = BitSet(pivot_rows)
    for row in axes(reduced, 1)
        row in pivot_row_set && continue
        iszero(reduced[row, rhs_column]) || return nothing
    end

    particular = zeros(ExactRational, p)
    for (row, permuted_pivot_column) in zip(pivot_rows, pivot_columns)
        original_pivot_column = permutation[permuted_pivot_column]
        particular[original_pivot_column] = _from_nemo_rational(reduced[row, rhs_column])
    end

    pivot_set = Set(pivot_columns)
    permuted_free_columns = [column for column in 1:p if !(column in pivot_set)]
    nullspace = zeros(ExactRational, p, length(permuted_free_columns))
    for (basis_index, permuted_free_column) in enumerate(permuted_free_columns)
        original_free_column = permutation[permuted_free_column]
        nullspace[original_free_column, basis_index] = one(ExactRational)
        for (row, permuted_pivot_column) in zip(pivot_rows, pivot_columns)
            coefficient = reduced[row, permuted_free_column]
            iszero(coefficient) ||
                (nullspace[permutation[permuted_pivot_column], basis_index] =
                    -_from_nemo_rational(coefficient))
        end
    end
    # The basis is read directly from an exact Nemo RREF. Recomputing A*p and
    # A*N here performs a dense Rational{BigInt} matrix multiplication that
    # merely repeats the elimination and can dominate extraction time.
    return particular, nullspace
end

function _assert_affine_invariant(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    affine::Tuple{Vector{ExactRational},Matrix{ExactRational}},
)
    particular, nullspace = affine
    size(A, 1) == length(b) || error("Affine equality matrix and rhs dimensions must match.")
    size(A, 2) == length(particular) || error("Affine particular point has the wrong dimension.")
    A * particular == b || error("Exact affine invariant failed: A*p != b.")
    A * nullspace == zeros(ExactRational, size(A, 1), size(nullspace, 2)) ||
        error("Exact affine invariant failed: A*N != 0.")
    return affine
end

function _independent_affine_equalities(
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    ;
    checkpoint::Union{Nothing,Function} = nothing,
)
    size(A, 1) == length(b) || error("Affine equality matrix and rhs dimensions must match.")
    isempty(b) && return A, b

    checkpoint !== nothing && checkpoint(
        "affine compaction: building dense $(size(A, 1))-by-$(size(A, 2) + 1) augmented system",
    )
    reduced, _ = _rref(hcat(A, b); checkpoint = checkpoint)
    rhs_column = size(A, 2) + 1
    rows = Vector{Vector{ExactRational}}()
    rhs = ExactRational[]
    for row_index in axes(reduced, 1)
        row = collect(view(reduced, row_index, 1:size(A, 2)))
        if all(iszero, row)
            iszero(reduced[row_index, rhs_column]) || return nothing
            continue
        end
        push!(rows, row)
        push!(rhs, reduced[row_index, rhs_column])
    end

    if isempty(rows)
        return zeros(ExactRational, 0, size(A, 2)), ExactRational[]
    end
    reduced_A = zeros(ExactRational, length(rows), size(A, 2))
    for (row_index, row) in enumerate(rows)
        reduced_A[row_index, :] = row
    end
    return reduced_A, rhs
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
    result = (
        particular + nullspace * coordinate_particular,
        nullspace * coordinate_nullspace,
    )
    return _assert_affine_invariant(rows, rhs, result)
end

struct _SparseAffineRestrictions
    indices::Vector{Vector{Int}}
    values::Vector{Vector{ExactRational}}

    function _SparseAffineRestrictions(
        indices::Vector{Vector{Int}},
        values::Vector{Vector{ExactRational}},
    )
        length(indices) == length(values) ||
            error("Sparse affine restriction indices and values must have the same row count.")
        for (row_indices, row_values) in zip(indices, values)
            length(row_indices) == length(row_values) ||
                error("Sparse affine restriction row indices and values must have the same length.")
            allunique(row_indices) || error("Sparse affine restriction rows must not repeat an index.")
        end
        return new(indices, values)
    end
end

_sparse_restriction_entry_count(restrictions::_SparseAffineRestrictions) =
    sum(length, restrictions.indices; init = 0)

function _assert_sparse_affine_invariant(
    restrictions::_SparseAffineRestrictions,
    rhs::Vector{ExactRational},
    affine::Tuple{Vector{ExactRational},Matrix{ExactRational}},
)
    length(restrictions.indices) == length(rhs) ||
        error("Sparse affine restriction rows and rhs must match.")
    particular, nullspace = affine
    dimension = length(particular)
    size(nullspace, 1) == dimension || error("Affine nullspace has the wrong row count.")
    for row_index in eachindex(rhs)
        row_indices = restrictions.indices[row_index]
        row_values = restrictions.values[row_index]
        all(index -> 1 <= index <= dimension, row_indices) ||
            error("Sparse affine restriction index is out of bounds.")
        particular_residual = -rhs[row_index]
        nullspace_residual = zeros(ExactRational, size(nullspace, 2))
        for (index, value) in zip(row_indices, row_values)
            particular_residual += value * particular[index]
            for column in axes(nullspace, 2)
                nullspace_residual[column] += value * nullspace[index, column]
            end
        end
        iszero(particular_residual) ||
            error("Exact affine invariant failed: sparse restriction row $(row_index) is violated by p.")
        all(iszero, nullspace_residual) ||
            error("Exact affine invariant failed: sparse restriction row $(row_index) is not zero on N.")
    end
    return affine
end

function _sparse_coordinate_restriction_system(
    affine::Tuple{Vector{ExactRational},Matrix{ExactRational}},
    added_dimension::Int,
    restrictions::_SparseAffineRestrictions,
    rhs::Vector{ExactRational},
)
    length(restrictions.indices) == length(rhs) ||
        error("Sparse affine restriction rows and rhs must match.")
    particular, nullspace = affine
    old_dimension = length(particular)
    size(nullspace, 1) == old_dimension || error("Affine nullspace has the wrong row count.")
    added_dimension >= 0 || error("Added affine dimension must be nonnegative.")
    nullspace_dimension = size(nullspace, 2)
    coordinate_dimension = nullspace_dimension + added_dimension
    total_dimension = old_dimension + added_dimension
    coordinate_rows = zeros(
        ExactRational,
        length(rhs),
        coordinate_dimension,
    )
    coordinate_rhs = copy(rhs)
    for row_index in eachindex(rhs)
        row_indices = restrictions.indices[row_index]
        row_values = restrictions.values[row_index]
        all(index -> 1 <= index <= total_dimension, row_indices) ||
            error("Sparse affine restriction index is out of bounds.")
        for (index, value) in zip(row_indices, row_values)
            if index <= old_dimension
                coordinate_rhs[row_index] -= value * particular[index]
                for column in axes(nullspace, 2)
                    coordinate_rows[row_index, column] += value * nullspace[index, column]
                end
            else
                coordinate_rows[row_index, nullspace_dimension + index - old_dimension] += value
            end
        end
    end
    return coordinate_rows, coordinate_rhs
end

function _lift_affine_basis_with_nemo(
    particular::Vector{ExactRational},
    nullspace::Matrix{ExactRational},
    added_dimension::Int,
    coordinate_particular::Vector{ExactRational},
    coordinate_nullspace::Matrix{ExactRational},
    settings::Settings = Settings(),
)
    old_dimension = length(particular)
    coordinate_dimension = size(nullspace, 2) + added_dimension
    length(coordinate_particular) == coordinate_dimension ||
        error("Affine coordinate particular point has the wrong dimension.")
    size(coordinate_nullspace, 1) == coordinate_dimension ||
        error("Affine coordinate nullspace has the wrong row count.")

    coordinate_basis = hcat(
        reshape(coordinate_particular, :, 1),
        coordinate_nullspace,
    )
    output_entries = BigInt(old_dimension + added_dimension) * size(coordinate_basis, 2)
    output_limit = settings.facial_reduction_affine_lift_max_output_entries
    input_entry_count = length(particular) + length(nullspace) + length(coordinate_basis)
    average_entry_bytes = input_entry_count == 0 ? 32 : max(
        32,
        cld(
            Base.summarysize(particular) + Base.summarysize(nullspace) +
            Base.summarysize(coordinate_basis),
            input_entry_count,
        ),
    )
    estimated_bytes = output_entries * average_entry_bytes
    byte_limit = settings.facial_reduction_affine_lift_max_estimated_bytes
    if output_limit == 0 || output_entries > output_limit ||
       byte_limit == 0 || estimated_bytes > byte_limit
        _record_facial_reduction_event!(:affine_lifts_skipped_by_budget)
        throw(
            ExactLinearAlgebraError(
                "bounded affine basis lift",
                "RationalSDP work limit",
                "output entries $(output_entries)/$(output_limit), estimated bytes $(estimated_bytes)/$(byte_limit)",
            ),
        )
    end

    old_coordinate_dimension = size(nullspace, 2)
    old_coordinate_basis = view(coordinate_basis, 1:old_coordinate_dimension, :)
    lifted_old_basis = if old_coordinate_dimension == 0
        zeros(ExactRational, old_dimension, size(coordinate_basis, 2))
    else
        _nemo_matrix_product_chunked(
            nullspace,
            old_coordinate_basis,
            settings.facial_reduction_affine_lift_chunk_columns,
        )
    end
    added_basis = view(coordinate_basis, (old_coordinate_dimension + 1):coordinate_dimension, :)
    lifted_basis = added_dimension == 0 ? lifted_old_basis : vcat(lifted_old_basis, added_basis)
    return (
        vcat(particular, zeros(ExactRational, added_dimension)) + vec(lifted_basis[:, 1]),
        Matrix(lifted_basis[:, 2:end]),
    )
end

function _extend_and_restrict_affine_system(
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
    added_dimension::Int,
    restrictions::_SparseAffineRestrictions,
    restriction_rhs::Vector{ExactRational},
    ;
    checkpoint::Union{Nothing,Function} = nothing,
    settings::Settings = Settings(),
)
    affine === nothing && return nothing
    added_dimension >= 0 || error("Added affine dimension must be nonnegative.")
    length(restrictions.indices) == length(restriction_rhs) ||
        error("Sparse affine restriction rows and rhs must match.")
    particular, nullspace = affine
    old_dimension = length(particular)
    size(nullspace, 1) == old_dimension || error("Affine nullspace has the wrong row count.")
    coordinate_dimension = size(nullspace, 2) + added_dimension
    checkpoint !== nothing && checkpoint(
        "affine restriction: assembling $(length(restriction_rhs))-by-$(coordinate_dimension) coordinate system from $(length(restriction_rhs)) block-sparse face equation(s) with $(_sparse_restriction_entry_count(restrictions)) nonzero(s)",
    )
    coordinate_rows, coordinate_rhs = _sparse_coordinate_restriction_system(
        affine,
        added_dimension,
        restrictions,
        restriction_rhs,
    )
    checkpoint !== nothing && checkpoint("affine restriction: solving coordinate system exactly")
    coordinate_affine = _solve_affine_system(
        coordinate_rows,
        coordinate_rhs;
        checkpoint,
    )
    coordinate_affine === nothing && return nothing
    coordinate_particular, coordinate_nullspace = coordinate_affine
    checkpoint !== nothing && checkpoint(
        "affine restriction: lifting restricted affine basis with Nemo exact multiplication",
    )
    result = _lift_affine_basis_with_nemo(
        particular,
        nullspace,
        added_dimension,
        coordinate_particular,
        coordinate_nullspace,
        settings,
    )
    checkpoint !== nothing && checkpoint("affine restriction: validating restricted affine basis")
    validation_products =
        BigInt(_sparse_restriction_entry_count(restrictions)) *
        (1 + size(result[2], 2))
    validation_limit = settings.facial_reduction_sparse_affine_validation_max_products
    if validation_products <= validation_limit
        _assert_sparse_affine_invariant(restrictions, restriction_rhs, result)
    else
        checkpoint !== nothing && checkpoint(
            "affine restriction: skipping redundant sparse validation ($(validation_products) exact products; limit $(validation_limit)); exact RREF and Nemo lifting preserve the restriction algebraically",
        )
    end
    checkpoint !== nothing && checkpoint("affine restriction: completed")
    return result
end

function _extend_and_restrict_affine_system(
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
    added_dimension::Int,
    restriction_rows::Matrix{ExactRational},
    restriction_rhs::Vector{ExactRational},
    ;
    checkpoint::Union{Nothing,Function} = nothing,
    settings::Settings = Settings(),
)
    affine === nothing && return nothing
    added_dimension >= 0 || error("Added affine dimension must be nonnegative.")
    size(restriction_rows, 1) == length(restriction_rhs) ||
        error("Affine restriction rows and rhs must match.")
    old_dimension = length(affine[1])
    coordinate_dimension = size(affine[2], 2) + added_dimension
    checkpoint !== nothing && checkpoint(
        "affine restriction: extending $(old_dimension) variables by $(added_dimension) face coordinate(s)",
    )
    checkpoint !== nothing && checkpoint(
        "affine restriction: forming dense $(size(restriction_rows, 1))-by-$(coordinate_dimension) coordinate system from $(size(restriction_rows, 1)) face equation(s)",
    )
    indices = [
        [column for column in axes(restriction_rows, 2) if !iszero(restriction_rows[row, column])]
        for row in axes(restriction_rows, 1)
    ]
    values = [
        ExactRational[restriction_rows[row, column] for column in row_indices] for
        (row, row_indices) in enumerate(indices)
    ]
    return _extend_and_restrict_affine_system(
        affine,
        added_dimension,
        _SparseAffineRestrictions(indices, values),
        restriction_rhs;
        checkpoint,
        settings,
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

function _independent_nullspace_columns(
    nullspace::Matrix{ExactRational},
    active_positions::Vector{Int},
    ::Type{F},
) where {F<:AbstractFloat}
    size(nullspace, 2) == 0 && return nullspace
    isempty(active_positions) && return zeros(ExactRational, size(nullspace, 1), 0)

    reduced = nullspace[active_positions, :]
    numeric_reduced = Matrix{F}(undef, size(reduced))
    for column in axes(reduced, 2), row in axes(reduced, 1)
        numeric_reduced[row, column] = F(reduced[row, column])
    end
    for column in axes(numeric_reduced, 2)
        column_scale = norm(view(numeric_reduced, :, column))
        iszero(column_scale) && continue
        view(numeric_reduced, :, column) ./= column_scale
    end
    factorization = qr(numeric_reduced, ColumnNorm())
    diagonal = abs.(diag(factorization.R))
    rank_tolerance =
        isempty(diagonal) ? zero(F) :
        maximum(diagonal) * F(max(size(numeric_reduced)...)) * eps(F)
    rank = count(value -> value > rank_tolerance, diagonal)
    pivots = isempty(diagonal) ? Int[] : sort(factorization.p[1:rank])
    isempty(pivots) && return zeros(ExactRational, size(nullspace, 1), 0)
    return nullspace[:, pivots]
end

function _compute_phase1_nullspace(
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
    ::Type{F},
) where {F<:AbstractFloat}
    affine === nothing && return nothing
    _, nullspace = affine

    active_positions = Int[]
    append!(active_positions, positive_scalars)
    for block in blocks
        append!(active_positions, block.global_positions)
    end
    active_positions = unique(sort(active_positions))
    return _independent_nullspace_columns(nullspace, active_positions, F)
end

function _phase1_nullspace(problem::ProblemData, ::Type{F}) where {F<:AbstractFloat}
    problem.affine === nothing && error("Phase I nullspace requested without affine data.")
    if problem.phase1_nullspace === nothing || problem.phase1_nullspace_float_type !== F
        problem.phase1_nullspace = _compute_phase1_nullspace(
            problem.blocks,
            problem.positive_scalars,
            problem.affine,
            F,
        )
        problem.phase1_nullspace_float_type = F
    end
    return problem.phase1_nullspace
end

function _phase2_relevant_positions(problem::ProblemData)
    positions = _phase1_active_positions(problem)
    for index in eachindex(problem.objective_vector_min)
        iszero(problem.objective_vector_min[index]) || push!(positions, index)
    end
    return unique(sort(positions))
end

function _phase2_nullspace(problem::ProblemData, ::Type{F}) where {F<:AbstractFloat}
    problem.affine === nothing && error("Phase II nullspace requested without affine data.")
    _, nullspace = problem.affine
    return _independent_nullspace_columns(
        nullspace,
        _phase2_relevant_positions(problem),
        F,
    )
end

function ProblemData(
    original_variables::Vector{MOI.VariableIndex},
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    objective_vector_raw::Vector{ExactRational},
    objective_constant_raw,
    objective_vector_min::Vector{ExactRational},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
    affine::Union{Nothing,Tuple{Vector{ExactRational},Matrix{ExactRational}}},
    phase1_nullspace::Union{Nothing,Matrix{ExactRational}},
    scalar_constraint_rows::Dict{Any,Vector{Int}},
    psd_constraint_blocks::Dict{Any,Int},
)
    dimension = length(objective_vector_raw)
    identity_lift = sparse(
        1:dimension,
        1:dimension,
        fill(one(ExactRational), dimension),
        dimension,
        dimension,
    )
    return ProblemData(
        original_variables,
        blocks,
        positive_scalars,
        objective_vector_raw,
        _exact_rational(objective_constant_raw),
        objective_vector_min,
        A,
        b,
        affine,
        phase1_nullspace,
        scalar_constraint_rows,
        psd_constraint_blocks,
        identity_lift,
        nothing,
    )
end

function ProblemData(
    original_variables::Vector{MOI.VariableIndex},
    blocks::Vector{BlockStructure},
    positive_scalars::Vector{Int},
    objective_vector_raw::Vector{ExactRational},
    objective_constant_raw,
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
        _exact_rational(objective_constant_raw),
        objective_vector_min,
        A,
        b,
        affine,
        nothing,
        Dict{Any,Vector{Int}}(),
        Dict{Any,Int}(),
    )
end

function _lift_original_solution(
    problem::ProblemData,
    point::AbstractVector{ExactRational},
)
    length(point) == size(problem.solution_lift, 2) ||
        error("Compact solution vector has the wrong dimension.")
    return Vector{ExactRational}(problem.solution_lift * point)
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

function _early_prune_psd_coordinate_faces(
    blocks::Vector{BlockStructure},
    A::Matrix{ExactRational},
    b::Vector{ExactRational},
)
    known_zero = Set{Int}()
    removed_directions = [Set{Int}() for _ in blocks]

    changed = true
    while changed
        changed = false

        for row in axes(A, 1)
            iszero(b[row]) || continue
            survivor = 0
            survivor_count = 0
            for column in axes(A, 2)
                iszero(A[row, column]) && continue
                column in known_zero && continue
                survivor = column
                survivor_count += 1
                survivor_count > 1 && break
            end
            if survivor_count == 1 && !(survivor in known_zero)
                push!(known_zero, survivor)
                changed = true
            end
        end

        for (block_index, block) in enumerate(blocks)
            removed = removed_directions[block_index]
            for local_direction in 1:block.size
                local_direction in removed && continue
                block.diagonal_positions[local_direction] in known_zero || continue
                push!(removed, local_direction)
                changed = true
                for index in _block_direction_entry_indices(block, local_direction)
                    if !(index in known_zero)
                        push!(known_zero, index)
                        changed = true
                    end
                end
            end
        end
    end

    isempty(known_zero) && return blocks, A, b, 0

    new_blocks = BlockStructure[]
    pruned_directions = 0
    for (block_index, block) in enumerate(blocks)
        remove_directions = sort(collect(removed_directions[block_index]))
        if isempty(remove_directions)
            push!(new_blocks, block)
        else
            pruned_directions += length(remove_directions)
            keep_directions = setdiff(collect(1:block.size), remove_directions)
            isempty(keep_directions) || push!(new_blocks, _restrict_block(block, keep_directions))
        end
    end

    A, b = _append_zero_equalities(A, b, collect(known_zero))
    return new_blocks, A, b, pruned_directions
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

function _exact_primal_feasibility(problem::ProblemData, x::Vector{ExactRational})
    dimension = length(problem.objective_vector_raw)
    length(x) >= dimension || return (ok = false, reason = "solution vector is too short")
    original_x = x[1:dimension]

    problem.A * original_x == problem.b ||
        return (ok = false, reason = "an original affine equation is violated")
    for position in problem.positive_scalars
        original_x[position] >= 0 ||
            return (ok = false, reason = "an original scalar cone constraint is violated")
    end
    for (block_index, block) in enumerate(problem.blocks)
        _positive_semidefinite_exact(_vector_to_matrix(original_x, block)) || return (
            ok = false,
            reason = "original PSD block $(block_index) is not positive semidefinite",
        )
    end
    return (ok = true, reason = "exactly feasible for the original SDP")
end

function _positive_definite_exact(matrix::Matrix{ExactRational})
    size(matrix, 1) == size(matrix, 2) || return false
    n = size(matrix, 1)
    n == 0 && return true
    any(matrix .!= transpose(matrix)) && return false
    for diagonal in 1:n
        matrix[diagonal, diagonal] > 0 || return false
    end

    remainder = copy(matrix)
    active = collect(1:n)
    while !isempty(active)
        for diagonal in eachindex(active)
            remainder[diagonal, diagonal] > 0 || return false
        end
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
    column_count = size(matrix, 2)
    size(matrix, 1) == 0 &&
        return Matrix{ExactRational}(I, column_count, column_count)
    column_count == 0 && return zeros(ExactRational, 0, 0)
    _, nullspace = _with_nemo_error("exact rational nullspace") do
        Nemo.nullspace(_to_nemo_matrix(matrix))
    end
    return _from_nemo_matrix(nullspace)
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
    result = NumericAffineData{F}(particular_numeric, nullspace, nothing, nothing, nothing)
    _record_approximate_cache_memory!(:affine, result)
    return result
end

function _numeric_exact_nullspace!(numeric_affine::NumericAffineData{F}) where {F<:AbstractFloat}
    if size(numeric_affine.exact_nullspace, 2) == 0
        if numeric_affine.numeric_exact_nullspace === nothing
            numeric_affine.numeric_exact_nullspace = Matrix{F}(undef, size(numeric_affine.exact_nullspace)...)
            _record_approximate_cache_memory!(:affine, numeric_affine)
        end
        return numeric_affine.numeric_exact_nullspace
    end
    if numeric_affine.numeric_exact_nullspace === nothing
        numeric_affine.numeric_exact_nullspace = _to_working_array(F, numeric_affine.exact_nullspace)
        _record_approximate_cache_memory!(:affine, numeric_affine)
    end
    return numeric_affine.numeric_exact_nullspace
end

function _numeric_nullspace!(numeric_affine::NumericAffineData{F}) where {F<:AbstractFloat}
    if size(numeric_affine.exact_nullspace, 2) == 0
        if numeric_affine.numeric_phase2_basis === nothing
            numeric_affine.numeric_phase2_basis = Matrix{F}(undef, size(numeric_affine.exact_nullspace)...)
            _record_approximate_cache_memory!(:affine, numeric_affine)
        end
        return numeric_affine.numeric_phase2_basis
    end
    if numeric_affine.numeric_phase2_basis === nothing
        exact_numeric_nullspace = _numeric_exact_nullspace!(numeric_affine)
        basis = Matrix{F}(I, size(exact_numeric_nullspace, 1), size(exact_numeric_nullspace, 2))
        factor = qr(exact_numeric_nullspace)
        lmul!(factor.Q, basis)
        numeric_affine.numeric_phase2_basis = basis
        _record_approximate_cache_memory!(:affine, numeric_affine)
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
