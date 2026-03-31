# Barrier evaluations and Newton-system linear algebra.

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

function _block_barrier_value_grad_diag!(
    grad::Vector{F},
    diag_hess::Vector{F},
    x::Vector{F},
    numeric_block::NumericBlock,
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
    for a in eachindex(local_positions)
        i, j = local_positions[a]
        ga = global_positions[a]
        grad[ga] = _barrier_gradient_entry(inverse_matrix, i, j)
        diag_hess[ga] = _barrier_hessian_entry(inverse_matrix, i, j, i, j)
    end

    spectral = eigen(Hermitian(X))
    return -2 * logdet, PSDBarrierCache(
        numeric_block,
        X,
        inverse_matrix,
        spectral.vectors,
        spectral.values,
    )
end

function _barrier_value_only(
    x::Vector{F},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
) where {F<:AbstractFloat}
    value = zero(F)
    for numeric_block in numeric_blocks
        block = numeric_block.structure
        X = _vector_to_matrix(x, block)
        factor = cholesky(Hermitian(X))
        logdet = zero(F)
        for diagonal_entry in diag(factor.L)
            logdet += log(diagonal_entry)
        end
        value -= 2 * logdet
    end
    for index in positive_scalars
        value -= log(x[index])
    end
    return value
end

function _barrier_value_grad_diag(
    x::Vector{F},
    numeric_blocks::Vector{NumericBlock},
    positive_scalars::Vector{Int},
) where {F<:AbstractFloat}
    value = zero(F)
    grad = zeros(F, length(x))
    diag_hess = zeros(F, length(x))
    caches = Vector{PSDBarrierCache{F}}(undef, length(numeric_blocks))

    for block_index in eachindex(numeric_blocks)
        block_value, cache = _block_barrier_value_grad_diag!(
            grad,
            diag_hess,
            x,
            numeric_blocks[block_index],
        )
        value += block_value
        caches[block_index] = cache
    end

    for index in positive_scalars
        inverse_slack = inv(x[index])
        value -= log(x[index])
        grad[index] -= inverse_slack
        diag_hess[index] += inverse_slack^2
    end

    return value, grad, diag_hess, caches
end

function _barrier_hessian_mul!(
    destination::Vector{F},
    direction::Vector{F},
    caches::Vector{PSDBarrierCache{F}},
    positive_scalars::Vector{Int},
    x::Vector{F},
) where {F<:AbstractFloat}
    fill!(destination, zero(F))
    for cache in caches
        block = cache.numeric_block.structure
        D = _vector_to_matrix(direction, block)
        M = cache.inverse_matrix * D * cache.inverse_matrix
        for local_index in eachindex(block.local_positions)
            i, j = block.local_positions[local_index]
            global_index = block.global_positions[local_index]
            destination[global_index] += i == j ? M[i, i] : 2 * M[i, j]
        end
    end
    for index in positive_scalars
        destination[index] += direction[index] / (x[index]^2)
    end
    return destination
end

function _apply_block_regularized_inverse!(
    destination::AbstractVector{F},
    rhs::AbstractVector{F},
    cache::PSDBarrierCache{F},
    center_weight::F,
) where {F<:AbstractFloat}
    block = cache.numeric_block.structure
    rhs_matrix = _dual_vector_to_matrix(rhs, block)
    Q = cache.eigenvectors
    λ = cache.eigenvalues
    transformed_rhs = transpose(Q) * rhs_matrix * Q
    transformed_solution = similar(transformed_rhs)
    @inbounds for j in axes(transformed_rhs, 2)
        for i in axes(transformed_rhs, 1)
            transformed_solution[i, j] =
                transformed_rhs[i, j] / (inv(λ[i] * λ[j]) + center_weight)
        end
    end
    solution_matrix = Q * transformed_solution * transpose(Q)
    _matrix_to_vector!(destination, solution_matrix, block)
    return destination
end

function _apply_regularized_barrier_inverse!(
    destination::Vector{F},
    rhs::AbstractVector{F},
    caches::Vector{PSDBarrierCache{F}},
    positive_scalars::Vector{Int},
    x::Vector{F},
    center_weight::F,
) where {F<:AbstractFloat}
    destination .= rhs ./ center_weight
    for cache in caches
        _apply_block_regularized_inverse!(destination, rhs, cache, center_weight)
    end
    for index in positive_scalars
        destination[index] = rhs[index] / (inv(x[index]^2) + center_weight)
    end
    return destination
end

function _schur_phase1_direction(
    opt::Optimizer,
    grad::Vector{F},
    A_big::AbstractMatrix{F},
    At_big::AbstractMatrix{F},
    penalty::F,
    caches::Vector{PSDBarrierCache{F}},
    positive_scalars::Vector{Int},
    x::Vector{F},
    center_weight::F,
) where {F<:AbstractFloat}
    inverse_grad = similar(grad)
    _apply_regularized_barrier_inverse!(
        inverse_grad,
        grad,
        caches,
        positive_scalars,
        x,
        center_weight,
    )

    m = size(A_big, 1)
    n = size(A_big, 2)
    inverse_At = Matrix{F}(undef, n, m)
    rhs_column = Vector{F}(undef, n)
    solution_column = Vector{F}(undef, n)
    for row_index in 1:m
        rhs_column .= @view At_big[:, row_index]
        _apply_regularized_barrier_inverse!(
            solution_column,
            rhs_column,
            caches,
            positive_scalars,
            x,
            center_weight,
        )
        @views inverse_At[:, row_index] .= solution_column
    end

    schur = A_big * inverse_At
    diagonal_shift = inv(penalty)
    for row_index in 1:m
        schur[row_index, row_index] += diagonal_shift
    end
    rhs = A_big * inverse_grad
    y = _solve_spd_system(schur, rhs)
    return -inverse_grad + inverse_At * y
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
        scaled_direction = Hermitian(factor.L \ D / transpose(factor.L))
        minimum_eigenvalue = minimum(eigen(scaled_direction).values)
        if minimum_eigenvalue < 0
            max_step = min(max_step, fraction / (-minimum_eigenvalue))
        end
    end
    return max(zero(F), max_step)
end

function _solve_spd_system(matrix::Matrix{F}, rhs::Vector{F}) where {F<:AbstractFloat}
    isempty(rhs) && return similar(rhs, 0)
    regularization = zero(F)
    identity_matrix = Matrix{F}(I, size(matrix, 1), size(matrix, 2))
    matrix_scale = max(one(F), maximum(abs, matrix), maximum(abs, diag(matrix)))
    seed = max(_to_working_float(F, big"1e-30"), sqrt(eps(F)) * matrix_scale)
    for _ in 1:16
        trial = matrix + regularization * identity_matrix
        try
            factor = cholesky(Hermitian(trial))
            return factor \ rhs
        catch
            regularization = iszero(regularization) ? seed : F(10) * regularization
        end
    end
    try
        spectral = eigen(Hermitian(matrix))
        shift = max(seed, -minimum(spectral.values) + seed)
        coefficients = transpose(spectral.vectors) * rhs
        coefficients ./= (spectral.values .+ shift)
        return spectral.vectors * coefficients
    catch
        throw(SPDSystemFactorizationError(size(matrix, 1)))
    end
end

function _l2_norm(vector::AbstractVector{F}) where {F<:AbstractFloat}
    isempty(vector) && return zero(F)
    return sqrt(sum(abs2, vector))
end
