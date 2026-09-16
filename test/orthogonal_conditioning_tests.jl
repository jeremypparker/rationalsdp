using Test, JuMP, LinearAlgebra, RationalSDP

@testset "Conditioning in the full solver" begin
    R = RationalSDP
    Rat = Rational{BigInt}
    F = R.Float64x2

    @testset "Exact keep basis preserves the face" begin
        U = Rat[1 10^9; -1 -10^9; 0 1]
        V = R._orthogonal_face_keep_basis(U)
        @test V'V == Diagonal(diag(V'V))
        @test all(n -> 1//2 <= n <= 2, diag(V'V))
        @test Rat[1, 1, 0]' * V == zeros(Rat, 1, 2)
        @test size(V) == size(U)
        @test_throws ErrorException R._orthogonal_face_keep_basis(Rat[1 2; 0 0])
        @test size(R._orthogonal_face_keep_basis(zeros(Rat, 3, 0))) == (3, 0)
    end

    @testset "Numerical cone coordinates and inverse" begin
        tiny = F(10)^(-14)
        G = F[1 10^9 0; 0 F(10^9)*tiny 0; 0 0 -1; 0 0 1]
        model = R.Hypatia.Models.Model{F}(
            F[0, 0, -1], zeros(F, 0, 3), F[], copy(G), F[1, 1, 0, 1e-8],
            [R.Hypatia.Cones.Nonnegative{F}(4)],
        )
        transform = R._orthogonalize_hypatia_phase1!(model)
        Q = model.G[:, 1:2]
        @test maximum(abs, Q'Q - Matrix{F}(I, 2, 2)) < F(1e-28)
        original = F[0.5, -3e-10]
        z = transform.upper * (original .* transform.scales)[transform.pivots]
        recovered = R._phase1_original_coordinates(transform, z)
        @test maximum(abs, recovered - original) < F(1e-17)
        @test maximum(abs, G[:, 1:2]*original - Q*z) < F(1e-28)
        @test model.G[:, end] == G[:, end]
    end

    function conditioned_model()
        model = GenericModel{Rat}(R.Optimizer{Rat})
        set_silent(model)
        for (name, value) in (
            "phase1_hypatia_orthogonalize" => true,
            "facial_reduction_orthogonalize" => true,
            "phase1_hypatia_float_type" => F,
            "phase1_hypatia_syssolver" => :symindef_dense,
            "phase1_hypatia_tol_rel_opt" => big"1e-18",
            "phase1_hypatia_tol_abs_opt" => big"1e-18",
            "phase1_hypatia_tol_feas" => big"1e-18",
            "phase1_hypatia_iter_limit" => 60,
        )
            set_optimizer_attribute(model, name, value)
        end
        return model
    end

    @testset "Phase I handles cone-invisible affine directions" begin
        model = conditioned_model()
        @variable(model, X[1:2, 1:2], PSD)
        @variable(model, free_variable)
        @constraint(model, X[1, 1] == 1)
        @constraint(model, X[2, 2] == 1)
        optimize!(model)
        @test termination_status(model) == JuMP.MOI.OPTIMAL
        @test value(X[1, 1]) == value(X[2, 2]) == 1
        @test abs(value(X[1, 2])) <= 1
        @test value(free_variable) isa Rat
    end

    @testset "Full solve reduces a non-coordinate face and optimizes" begin
        # PSD and [1,1,0]'X[1,1,0] = 0 imply X[1,1] = X[2,2],
        # X[1,2] = -X[2,2]. Trace(X) = 1 gives min X[3,3] = 0.
        model = conditioned_model()
        @variable(model, X[1:3, 1:3], PSD)
        @constraint(model, X[1, 1] + 2X[1, 2] + X[2, 2] == 0)
        @constraint(model, sum(X[i, i] for i in 1:3) == 1)
        @objective(model, Min, X[3, 3])
        optimize!(model)
        @test termination_status(model) == JuMP.MOI.OPTIMAL
        if has_values(model)
            values = value.(X)
            @test values * Rat[1, 1, 0] == zeros(Rat, 3)
            @test tr(values) == 1
            @test 0 <= objective_value(model) < 1//10^8
            @test all(det(values[indices, indices]) >= 0 for indices in ([1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]))
        end
    end
end
