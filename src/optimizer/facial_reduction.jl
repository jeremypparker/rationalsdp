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

function _block_annihilates_direction_exact(
    problem::ProblemData,
    block::BlockStructure,
    direction::Vector{ExactRational},
)
    problem.affine === nothing && return false
    particular, nullspace = problem.affine
    dimension = length(problem.objective_vector_raw)
    for row_index in 1:block.size
        form = _block_row_linear_form(block, row_index, direction, dimension)
        iszero(dot(form, particular)) || return false
        for basis_index in axes(nullspace, 2)
            iszero(dot(form, view(nullspace, :, basis_index))) || return false
        end
    end
    return true
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
        if _block_annihilates_direction_exact(problem, block, direction)
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
    residual_tolerance = max(F(1.0e-10), sqrt(eps(F))) * matrix_scale
    for tolerance in _recovery_tolerances(settings, F)
        direction = ExactRational[
            rationalize(BigInt, BigFloat(value); tol = BigFloat(tolerance)) for value in candidate
        ]
        direction = _normalize_rational_direction(direction)
        any(!iszero, direction) || continue
        numeric_direction = _to_working_array(F, direction)
        residual = _max_abs(block_matrix * numeric_direction)
        if residual <= residual_tolerance
            return direction
        end
    end
    return nothing
end

function _linearly_independent_directions(directions::Vector{Vector{ExactRational}})
    isempty(directions) && return directions
    matrix = hcat(directions...)
    augmented = hcat(copy(matrix), zeros(ExactRational, size(matrix, 1)))
    _, pivots = _rref(augmented)
    return [directions[index] for index in pivots]
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

    exact_directions = _exact_block_nullspace_directions(problem, block)

    if isempty(exact_directions)
        heuristic_directions = Vector{Vector{ExactRational}}()
        for kernel_index in kernel_indices
            direction = _heuristic_kernel_direction(
                block_matrix,
                collect(view(eigen_factor.vectors, :, kernel_index)),
                opt.settings,
                F,
            )
            direction === nothing && continue
            push!(heuristic_directions, direction)
        end
        heuristic_directions = _linearly_independent_directions(heuristic_directions)
        if !isempty(heuristic_directions)
            _log(
                opt,
                "Facial reduction: using rationalized boundary kernel directions for PSD block $(block_index)",
            )
            return heuristic_directions
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
    free_positions = _facial_reduction_free_positions(problem)

    equality_rows = Vector{Vector{ExactRational}}()
    equality_rhs = ExactRational[]
    for position in free_positions
        push!(equality_rows, collect(problem.A[:, position]))
        push!(equality_rhs, 0 // 1)
    end
    push!(equality_rows, copy(problem.b))
    push!(equality_rhs, 0 // 1)
    push!(equality_rows, _facial_reduction_trace_row(problem))
    push!(equality_rhs, 1 // 1)

    A_eq = isempty(equality_rows) ? spzeros(F, 0, row_count) : _to_working_sparse_matrix(F, transpose(hcat(equality_rows...)))
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
        syssolver, use_dense_model = _hypatia_phase1_syssolver(HF)
        solver = Hypatia.Solvers.Solver{HF}(
            verbose = false,
            iter_limit = opt.settings.phase1_hypatia_iter_limit,
            preprocess = true,
            reduce = true,
            syssolver = syssolver,
            use_dense_model = use_dense_model,
        )
        start_time = time_ns()
        try
            Hypatia.Solvers.load(solver, model)
            Hypatia.Solvers.solve(solver)
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

function _facial_reduction_block_directions(
    opt::Optimizer,
    problem::ProblemData,
    block_index::Int,
    block_matrix::Matrix{F},
    ::Type{F},
) where {F<:AbstractFloat}
    block = problem.blocks[block_index]
    symmetric_matrix = Symmetric((block_matrix + transpose(block_matrix)) / 2)
    eigenvalues = eigvals(symmetric_matrix)
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
    problem.affine === nothing && return nothing
    keep_bases = Dict{Int,Matrix{ExactRational}}()

    for (block_index, block) in enumerate(problem.blocks)
        matrix = _vector_to_matrix(candidate, block)
        directions = _candidate_kernel_directions(opt, problem, block_index, matrix, F)
        isempty(directions) && continue
        keep_basis = _orthogonal_complement_basis(directions, block.size)
        if size(keep_basis, 2) == problem.blocks[block_index].size
            continue
        end
        keep_bases[block_index] = keep_basis
    end

    isempty(keep_bases) && return nothing
    return Int[], keep_bases
end

function _facially_reduce_problem(
    opt::Optimizer,
    problem::ProblemData,
    candidate::Vector{F},
    ::Type{F},
) where {F<:AbstractFloat}
    opt.settings.facial_reduction || return problem
    reduction = _facial_reduction_round(opt, problem, candidate, F)
    reduction === nothing && return problem
    exposed_scalars, keep_bases = reduction
    removed_psd_directions = sum(
        problem.blocks[index].size - size(keep_bases[index], 2) for index in keys(keep_bases)
    )
    reduced_problem = _apply_facial_reduction(problem, exposed_scalars, keep_bases)
    _log(
        opt,
        "facial reduction: fixed $(length(exposed_scalars)) scalar cone direction(s) and removed $(removed_psd_directions) PSD direction(s)",
    )
    return reduced_problem
end
