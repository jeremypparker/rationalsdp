using Test
using RationalSDP

const _AffineRational = Rational{BigInt}

function _affine_test_rank(A::Matrix{_AffineRational})
    isempty(A) && return 0
    augmented = hcat(A, zeros(_AffineRational, size(A, 1)))
    _, pivots = RationalSDP._rref(augmented)
    return length(pivots)
end

function _test_affine_invariants(A, b, affine)
    @test affine !== nothing
    affine === nothing && return
    particular, nullspace = affine
    rank_A = _affine_test_rank(A)
    @test A * particular == b
    @test A * nullspace == zeros(_AffineRational, size(A, 1), size(nullspace, 2))
    @test size(nullspace, 1) == size(A, 2)
    @test size(nullspace, 2) == size(A, 2) - rank_A
    return
end

function _test_same_affine_space(A, natural, sparsity)
    natural === nothing && return
    sparsity === nothing && return
    natural_particular, natural_nullspace = natural
    sparse_particular, sparse_nullspace = sparsity
    nullity = size(natural_nullspace, 2)
    @test size(sparse_nullspace, 2) == nullity
    @test A * (sparse_particular - natural_particular) ==
          zeros(_AffineRational, size(A, 1))
    @test _affine_test_rank(hcat(natural_nullspace, sparse_nullspace)) == nullity
    @test _affine_test_rank(
        hcat(natural_nullspace, reshape(sparse_particular - natural_particular, :, 1)),
    ) == nullity
    return
end

function _test_both_affine_orders(A, b)
    original_A = copy(A)
    original_b = copy(b)
    natural = RationalSDP._solve_affine_system(A, b; column_order = :natural)
    sparsity = RationalSDP._solve_affine_system(A, b; column_order = :sparsity)
    @test A == original_A
    @test b == original_b
    _test_affine_invariants(A, b, natural)
    _test_affine_invariants(A, b, sparsity)
    _test_same_affine_space(A, natural, sparsity)
    return natural, sparsity
end

@testset "Sparsity-aware exact affine elimination" begin
    @testset "column permutation" begin
        A = _AffineRational[
            1 0 0 1 0 0
            1 0 0 1 2 0
            1 0 3 0 0 0
        ]
        expected = [3, 5, 4, 1, 2, 6]
        @test RationalSDP._affine_column_permutation(A) == expected
        @test RationalSDP._affine_column_permutation(A) == expected
        @test all(RationalSDP._affine_column_permutation(A) == expected for _ in 1:5)

        dense_first = _AffineRational[
            1 1 2 0 0 0
            1 2 0 3 0 0
            1 3 0 0 4 0
            1 0 0 0 0 0
        ]
        permutation = RationalSDP._affine_column_permutation(dense_first)
        @test permutation == [3, 4, 5, 2, 1, 6]
        @test permutation[1:3] == [3, 4, 5]
    end

    @testset "rational affine-space equivalence" begin
        A = _AffineRational[
            1//2 1//3 0//1 1//7 0//1
            2//3 0//1 1//5 2//7 0//1
            5//6 1//3 1//5 3//7 0//1
        ]
        point = _AffineRational[2//5, -3//4, 7//6, 1//2, 11//9]
        b = A * point
        _test_both_affine_orders(A, b)

        redundant_A = _AffineRational[
            1//2 1//3 0//1 0//1
            1//1 2//3 0//1 0//1
            0//1 1//5 1//7 0//1
            0//1 2//5 2//7 0//1
        ]
        redundant_point = _AffineRational[3//2, -2//3, 5//4, 9//8]
        _test_both_affine_orders(redundant_A, redundant_A * redundant_point)
    end

    @testset "inconsistent and edge systems" begin
        inconsistent_A = _AffineRational[1 1; 2 2]
        inconsistent_b = _AffineRational[1, 3]
        @test RationalSDP._solve_affine_system(
            inconsistent_A,
            inconsistent_b;
            column_order = :natural,
        ) === nothing
        @test RationalSDP._solve_affine_system(
            inconsistent_A,
            inconsistent_b;
            column_order = :sparsity,
        ) === nothing

        zero_column_A = _AffineRational[1 0 0; 0 0 2]
        zero_column_b = _AffineRational[3, 4]
        _test_both_affine_orders(zero_column_A, zero_column_b)

        empty_A = zeros(_AffineRational, 0, 4)
        _test_both_affine_orders(empty_A, _AffineRational[])
        no_variables = zeros(_AffineRational, 2, 0)
        _test_both_affine_orders(no_variables, zeros(_AffineRational, 2))
        @test RationalSDP._solve_affine_system(
            no_variables,
            _AffineRational[0, 1];
            column_order = :natural,
        ) === nothing
        @test_throws ErrorException RationalSDP._solve_affine_system(
            zeros(_AffineRational, 0, 2),
            _AffineRational[1],
        )
        @test_throws ArgumentError RationalSDP._solve_affine_system(
            empty_A,
            _AffineRational[];
            column_order = :unknown,
        )
    end

    @testset "checkpoint structural statistics" begin
        A = _AffineRational[1 0 0; 2 3 0]
        checkpoints = String[]
        RationalSDP._solve_affine_system(
            A,
            _AffineRational[0, 0];
            checkpoint = message -> push!(checkpoints, message),
        )
        structural = only(
            filter(message -> occursin("total structural nonzeros", message), checkpoints),
        )
        @test occursin("coefficient matrix 2-by-3", structural)
        @test occursin("total structural nonzeros=3", structural)
        @test occursin("zero columns=1", structural)
        @test occursin("singleton columns=1", structural)
        @test occursin("column ordering=sparsity", structural)
    end

    @testset "fixed sparse rational systems" begin
        fixtures = [
            (
                _AffineRational[
                    1//2 0 0 -3//4 0 0 2//3
                    0 1//3 0 0 5//6 0 0
                    0 0 -2//5 0 0 7//8 0
                ],
                _AffineRational[1//2, -2//3, 3//4, 0, 5//6, -1//2, 2//5],
            ),
            (
                _AffineRational[
                    1//2 0 0 2//3 0 0 0 0 -1//4
                    0 -3//5 0 0 1//7 0 0 4//9 0
                    0 0 5//6 0 0 -2//3 0 0 1//8
                    3//4 0 0 0 0 0 7//10 0 0
                    0 0 0 -1//2 0 5//9 0 2//7 0
                ],
                _AffineRational[-1//2, 2//3, 1//4, -3//5, 0, 5//6, -2//3, 1//2, 3//7],
            ),
            (
                _AffineRational[
                    1//2 0 0 0 0 0 0 2//3 0 0 0 0
                    0 -3//4 0 0 5//6 0 0 0 0 0 0 1//7
                    0 0 2//5 0 0 0 -1//3 0 0 0 4//9 0
                    0 0 0 7//8 0 0 0 0 -2//7 0 0 0
                    3//5 0 0 0 0 1//6 0 0 0 0 0 0
                    0 0 0 0 -4//7 0 0 5//8 0 0 0 0
                    0 0 0 0 0 0 2//9 0 0 -3//10 0 1//2
                ],
                _AffineRational[1//3, -2//5, 3//4, 0, 5//6, -1//2, 2//7, -3//8, 4//9, 1//5, -2//3, 3//7],
            ),
        ]
        for (A, point) in fixtures
            _test_both_affine_orders(A, A * point)
        end
    end
end
