# Minimal KSE time-average helpers extracted for a self-contained dualized
# regression. This keeps the test local to RationalSDP instead of depending on
# the separate inertial repository.

_kse_reduce_s_degree(c, s, p::Number) = p

function _kse_reduce_s_degree(c, s, p)
    result = zero(c)
    for (monom, coeff) in zip(monomials(p), coefficients(p))
        s_degree = degree(monom, s)
        c_degree = degree(monom, c)

        term = coeff * (1 - c^2)^div(s_degree, 2) * s^(s_degree % 2) * c^c_degree
        for var in variables(p)
            if var != c && var != s
                term *= var^degree(monom, var)
            end
        end

        result += term
    end

    return result
end

_kse_max_degree(::Number, vars) = 0

function _kse_max_degree(p, vars)
    max_degree = 0
    unique_vars = unique(collect(vars))
    for monom in monomials(p)
        degree_in_vars = sum(degree(monom, var) for var in unique_vars)
        max_degree = max(max_degree, degree_in_vars)
    end
    return max_degree
end

function _kse_makebasis(ws, vars, state_degree::Int, space_degree::Int; odd::Bool = false, max_s_degree::Int = 1)
    vars_with_trig = [collect(vars); ws.s; ws.c]
    if state_degree < 0 || space_degree < 0
        return typeof(ws.c * ws.v[1])[]
    end

    basis = typeof(ws.c * ws.v[1])[]
    for monom in monomials(vars_with_trig, 0:(state_degree + space_degree))
        degree(monom, ws.s) <= max_s_degree || continue
        sum(degree(monom, Uvar) for Uvar in ws.U) + degree(monom, ws.s) + degree(monom, ws.c) <= space_degree || continue
        sum(degree(monom, var) for var in vars) <= state_degree || continue
        expected = odd ? -monom : monom
        subs(monom, ws.oddvars => -ws.oddvars) == expected || continue
        push!(basis, monom)
    end

    return basis
end

function _kse_makepoly_from_basis(model, basis)
    if isempty(basis)
        return 0, VariableRef[]
    end
    coeffs = @variable(model, [1:length(basis)])
    return dot(coeffs, basis), coeffs
end

function _kse_build_monom_index(poly_sets...)
    monom_index = Dict{Any,Int}()
    for polys in poly_sets
        for poly in polys
            for monom in monomials(poly)
                get!(monom_index, monom, length(monom_index) + 1)
            end
        end
    end
    return monom_index
end

function _kse_coefficient_matrix(polys, monom_index)
    matrix = zeros(Rational{BigInt}, length(monom_index), length(polys))
    for (column, poly) in enumerate(polys)
        for monom in monomials(poly)
            matrix[monom_index[monom], column] = Rational{BigInt}(DynamicPolynomials.coefficient(poly, monom))
        end
    end
    return matrix
end

function _kse_make_workspace(q::Int, k::Rational; model = nothing)
    model = isnothing(model) ? GenericModel{Rational{BigInt}}() : model

    @polyvar s c v[1:q+1]
    @polyvar U[1:q+1]

    oddvars = [v[2:2:end]; s; U[1:2:end]]
    u = v[1] * s

    ddx(::Number) = 0
    function ddx(p)
        @assert all(degree(monom, v[q + 1]) == 0 for monom in monomials(p))
        @assert all(degree(monom, U[q + 1]) == 0 for monom in monomials(p))

        return k * c * differentiate(p, s) -
               k * s * differentiate(p, c) +
               sum(v[order + 1] * differentiate(p, v[order]) for order in 1:q) +
               sum(U[order + 1] * differentiate(p, U[order]) for order in 1:q)
    end

    function ddx(p, n::Int)
        result = p
        for _ in 1:n
            result = ddx(result)
        end
        return result
    end

    dynamics = -u * ddx(u) - ddx(u, 2) - ddx(u, 4)
    u_derivatives = [ddx(u, order - 1) for order in 1:(q + 1)]
    u_derivatives = Vector{typeof(u_derivatives[end])}(u_derivatives)

    return (
        model = model,
        q = q,
        k = k,
        s = s,
        c = c,
        v = v,
        U = U,
        oddvars = oddvars,
        u = u,
        dynamics = dynamics,
        ddx = ddx,
        u_derivatives = u_derivatives,
    )
end

_kse_reduce_trig(ws, p) = _kse_reduce_s_degree(ws.c, ws.s, p)

function _kse_ddt_image(ws, w_monom)
    V_monom = _kse_reduce_trig(ws, subs(w_monom, ws.U => ws.u_derivatives))
    dwdt_monom = ws.dynamics * sum(
        (-1)^(order - 1) * ws.ddx(differentiate(w_monom, ws.U[order]), order - 1)
        for order in 1:ws.q
    )
    LV_monom = _kse_reduce_trig(ws, subs(dwdt_monom, ws.U => ws.u_derivatives))
    return V_monom, LV_monom
end

function _kse_reduce_ddt_basis(ws, gauge_basis, w_basis)
    isempty(w_basis) && return w_basis

    candidate_images = [begin
        _, LV_monom = _kse_ddt_image(ws, w_monom)
        LV_monom
    end for w_monom in w_basis]
    gauge_images = [_kse_reduce_trig(ws, ws.ddx(basis_monom)) for basis_monom in gauge_basis]

    monom_index = _kse_build_monom_index(gauge_images, candidate_images)
    isempty(monom_index) && return w_basis[1:0]

    gauge_matrix = _kse_coefficient_matrix(gauge_images, monom_index)
    candidate_matrix = _kse_coefficient_matrix(candidate_images, monom_index)
    residual =
        size(gauge_matrix, 2) == 0 ?
        candidate_matrix :
        candidate_matrix - gauge_matrix * (gauge_matrix \ candidate_matrix)
    factorization = qr(residual, ColumnNorm())
    diagonal = abs.(diag(factorization.R))
    rank_residual = count(value -> value > 0, diagonal)
    keep = sort(factorization.p[1:rank_residual])
    return w_basis[keep]
end

function _kse_make_ddt_auxiliary(ws, w_basis)
    w, coeffs = _kse_makepoly_from_basis(ws.model, w_basis)
    dwdt = ws.dynamics * sum(
        (-1)^(order - 1) * ws.ddx(differentiate(w, ws.U[order]), order - 1)
        for order in 1:ws.q
    )
    V = _kse_reduce_trig(ws, subs(w, ws.U => ws.u_derivatives))
    LV = _kse_reduce_trig(ws, subs(dwdt, ws.U => ws.u_derivatives))
    return V, LV, coeffs
end

function _kse_quadratic_form(basis_left, block, basis_right = basis_left)
    isempty(basis_left) && return 0
    isempty(basis_right) && return 0
    return basis_left' * block * basis_right
end

function _kse_constrain_zero_polynomial(model, poly)
    coeffs = [DynamicPolynomials.coefficient(poly, monom) for monom in monomials(poly)]
    isempty(coeffs) || @constraint(model, coeffs .== 0)
    return poly
end

function _kse_enforce_nonnegativity(ws, p)
    @assert subs(p, ws.oddvars => -ws.oddvars) == p

    degree_v = round(Int, _kse_max_degree(p, ws.v) / 2, RoundUp)
    degree_c = round(Int, _kse_max_degree(p, [ws.c]) / 2, RoundUp)

    basis_even_no_s = _kse_makebasis(ws, ws.v, degree_v, degree_c; odd = false, max_s_degree = 0)
    basis_even_s = _kse_makebasis(ws, ws.v, degree_v, degree_c - 1; odd = true, max_s_degree = 0)
    basis_odd_no_s = _kse_makebasis(ws, ws.v, degree_v, degree_c; odd = true, max_s_degree = 0)
    basis_odd_s = _kse_makebasis(ws, ws.v, degree_v, degree_c - 1; odd = false, max_s_degree = 0)

    function gram_matrix(basis_no_s, basis_s)
        dimension = length(basis_no_s) + length(basis_s)
        return @variable(ws.model, [1:dimension, 1:dimension], PSD)
    end

    function quadratic_no_s(Q, basis_no_s, basis_s)
        split = length(basis_no_s)
        return _kse_quadratic_form(basis_no_s, Q[1:split, 1:split]) +
               (1 - ws.c^2) * _kse_quadratic_form(basis_s, Q[(split + 1):end, (split + 1):end])
    end

    function quadratic_s(Q, basis_no_s, basis_s)
        split = length(basis_no_s)
        return ws.s * 2 * _kse_quadratic_form(basis_s, Q[(split + 1):end, 1:split], basis_no_s)
    end

    Q_odd = gram_matrix(basis_odd_no_s, basis_odd_s)
    Q_even = gram_matrix(basis_even_no_s, basis_even_s)

    p_with_s = sum(
        DynamicPolynomials.coefficient(p, monom) * monom
        for monom in monomials(p)
        if degree(monom, ws.s) == 1
    )
    p_without_s = p - p_with_s

    expression_without_s =
        p_without_s -
        quadratic_no_s(Q_even, basis_even_no_s, basis_even_s) -
        quadratic_no_s(Q_odd, basis_odd_no_s, basis_odd_s)
    expression_with_s =
        p_with_s -
        quadratic_s(Q_even, basis_even_no_s, basis_even_s) -
        quadratic_s(Q_odd, basis_odd_no_s, basis_odd_s)

    constrained_without_s = _kse_constrain_zero_polynomial(ws.model, expression_without_s)
    constrained_with_s = _kse_constrain_zero_polynomial(ws.model, expression_with_s)

    return (
        expressions = [constrained_without_s, constrained_with_s],
        Q_even = Q_even,
        Q_odd = Q_odd,
    )
end

function build_dualized_kse_timeaverage_model(; phase2_outer_iterations::Int = 4)
    ws = _kse_make_workspace(4, 3 // 4; model = dual_model(Rational{BigInt}))
    set_silent(ws.model)
    set_optimizer_attribute(ws.model, "working_float_type", Float64)
    set_optimizer_attribute(ws.model, "phase2_outer_iterations", phase2_outer_iterations)
    set_optimizer_attribute(ws.model, "dual_postsolve_backend", :hypatia)

    gauge_basis = _kse_makebasis(ws, ws.v[1:4], 4, 4; odd = true)
    f1, _ = _kse_makepoly_from_basis(ws.model, gauge_basis)

    w_basis = _kse_makebasis(ws, ws.U[1:(div(ws.q, 2) + 1)], 3, 3)
    w_basis = _kse_reduce_ddt_basis(ws, gauge_basis, w_basis)
    _, LV, _ = _kse_make_ddt_auxiliary(ws, w_basis)

    @variable(ws.model, B >= 0)
    p1 = _kse_reduce_trig(ws, B - ws.u^2 + LV + ws.ddx(f1))
    certificate = _kse_enforce_nonnegativity(ws, p1)
    @objective(ws.model, Min, B)

    return (
        model = ws.model,
        B = B,
        certificate = certificate,
    )
end

function solve_dualized_kse_timeaverage(; phase2_outer_iterations::Int = 4)
    instance = build_dualized_kse_timeaverage_model(;
        phase2_outer_iterations = phase2_outer_iterations,
    )
    optimize!(instance.model)
    return (
        termination_status = termination_status(instance.model),
        primal_status = primal_status(instance.model),
        B_value = value(instance.B),
        certificate = instance.certificate,
    )
end
