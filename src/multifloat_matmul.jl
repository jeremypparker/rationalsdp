"""
    _MULTIFLOAT_MATMUL_MIN_WORK

Minimum `m * k * n` work for which the MultiFloat matmul kernel uses Julia
threads. The method itself is also used below this threshold, but remains
single-threaded to avoid paying thread scheduling overhead on small blocks.
"""
const _MULTIFLOAT_MATMUL_MIN_WORK = 16_384
# Tullio's generated threaded kernel only wins once its scheduling overhead is
# amortized by a large product; below this crossover the simpler kernel wins.
const _MULTIFLOAT_TULLIO_MIN_WORK = 2_000_000

const _MultiFloatMatrixLike{T<:MultiFloat} = Union{
    StridedMatrix{T},
    Adjoint{T,<:StridedMatrix{T}},
    Transpose{T,<:StridedMatrix{T}},
    Symmetric{T,<:StridedMatrix{T}},
    Hermitian{T,<:StridedMatrix{T}},
}

@inline function _multifloat_scalar(::Type{T}, x::Bool) where {T<:MultiFloat}
    return x ? one(T) : zero(T)
end

@inline function _multifloat_scalar(::Type{T}, x) where {T<:MultiFloat}
    x isa T && return x
    return convert(T, x)
end

@inline function _multifloat_matmul_use_threads(C, A, B)
    work = size(A, 1) * size(A, 2) * size(B, 2)
    return nthreads() > 1 && work >= _MULTIFLOAT_MATMUL_MIN_WORK
end

@inline function _multifloat_matmul_use_tullio(C, A, B)
    work = size(A, 1) * size(A, 2) * size(B, 2)
    return nthreads() > 1 && work >= _MULTIFLOAT_TULLIO_MIN_WORK
end

@inline function _multifloat_matmul_entry!(
    C::StridedMatrix{T},
    A::_MultiFloatMatrixLike{T},
    B::_MultiFloatMatrixLike{T},
    i,
    j,
    α::T,
    β::T,
    ::Val{beta_mode},
) where {T<:MultiFloat,beta_mode}
    value = zero(T)
    @inbounds for k in axes(A, 2)
        value += A[i, k] * B[k, j]
    end
    value *= α

    @inbounds begin
        if beta_mode === :zero
            C[i, j] = value
        elseif beta_mode === :one
            C[i, j] = value + C[i, j]
        else
            C[i, j] = value + β * C[i, j]
        end
    end
    return nothing
end

function _multifloat_matmul_kernel!(
    C::StridedMatrix{T},
    A::_MultiFloatMatrixLike{T},
    B::_MultiFloatMatrixLike{T},
    α::T,
    β::T,
    beta_mode::Val,
    threaded::Bool,
) where {T<:MultiFloat}
    if threaded
        output_indices = CartesianIndices(C)
        @threads :static for linear_index in eachindex(C)
            @inbounds begin
                I = output_indices[linear_index]
                _multifloat_matmul_entry!(C, A, B, I[1], I[2], α, β, beta_mode)
            end
        end
    else
        @inbounds for j in axes(C, 2)
            for i in axes(C, 1)
                _multifloat_matmul_entry!(C, A, B, i, j, α, β, beta_mode)
            end
        end
    end
    return C
end

function _multifloat_scale!(C::StridedMatrix{T}, β::T, threaded::Bool) where {T<:MultiFloat}
    if iszero(β)
        fill!(C, zero(T))
    elseif β != one(T)
        if threaded
            @threads :static for linear_index in eachindex(C)
                C[linear_index] *= β
            end
        else
            @inbounds for i in eachindex(C)
                C[i] *= β
            end
        end
    end
    return C
end

function _multifloat_tullio_matmul!(
    C::StridedMatrix{T},
    A::_MultiFloatMatrixLike{T},
    B::_MultiFloatMatrixLike{T},
    α::T,
    β::T,
) where {T<:MultiFloat}
    if iszero(β)
        fill!(C, zero(T))
    elseif β != one(T)
        _multifloat_scale!(C, β, true)
    end

    @tullio threads=true avx=false fastmath=false C[i,j] += $α * A[i,k] * B[k,j]
    return C
end

"""
    mul!(C, A, B, α, β)

Threaded dense matmul for the real `MultiFloats.MultiFloat` types used by
Hypatia. This is deliberately a single, narrow extension of
`LinearAlgebra.mul!`; all other element types and matrix representations keep
their normal LinearAlgebra methods.
"""
function mul!(
    C::StridedMatrix{T},
    A::_MultiFloatMatrixLike{T},
    B::_MultiFloatMatrixLike{T},
    α::Number,
    β::Number,
) where {T<:MultiFloat}
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("A has size $(size(A)), B has size $(size(B))"))
    size(C, 1) == size(A, 1) && size(C, 2) == size(B, 2) ||
        throw(DimensionMismatch("C has size $(size(C)), but A*B has size ($(size(A, 1)), $(size(B, 2)))"))

    αT = _multifloat_scalar(T, α)
    βT = _multifloat_scalar(T, β)
    threaded = _multifloat_matmul_use_threads(C, A, B)

    if iszero(αT)
        return _multifloat_scale!(C, βT, threaded)
    elseif _multifloat_matmul_use_tullio(C, A, B)
        return _multifloat_tullio_matmul!(C, A, B, αT, βT)
    elseif iszero(βT)
        return _multifloat_matmul_kernel!(
            C,
            A,
            B,
            αT,
            βT,
            Val(:zero),
            threaded,
        )
    elseif βT == one(T)
        return _multifloat_matmul_kernel!(
            C,
            A,
            B,
            αT,
            βT,
            Val(:one),
            threaded,
        )
    else
        return _multifloat_matmul_kernel!(
            C,
            A,
            B,
            αT,
            βT,
            Val(:other),
            threaded,
        )
    end
end
