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

function _kse_makepoly_from_basis(model, basis)
    if isempty(basis)
        return 0, VariableRef[]
    end
    coeffs = @variable(model, [1:length(basis)])
    return dot(coeffs, basis), coeffs
end

function _kse_make_workspace(q::Int, k; model = nothing)
    model = isnothing(model) ? GenericModel{Float64}() : model

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
    coeffs = DynamicPolynomials.coefficients(poly) 
    isempty(coeffs) || @constraint(model, coeffs .== 0)
    return poly
end

function explicit_basis_odd_no_s(ws)
    return [
        ws.v[2],
        ws.c * ws.v[2],
        ws.c^2 * ws.v[2],
    ]
end

function explicit_basis_odd_s(ws)
    return [
        one(ws.c),
        ws.v[3],
        ws.v[1],
        ws.c,
        ws.c * ws.v[3],
        ws.c * ws.v[1],
    ]
end

function explicit_basis_even_no_s(ws)
    return [
        one(ws.c),
        ws.v[1],
        ws.c,
        ws.c^2,
        (1 - ws.c^2) * ws.v[3],
        ws.c^2 * ws.v[1],
        ws.c * ws.v[1]
    ]
end

function explicit_basis_even_s(ws)
    return [
        ws.v[2],
        ws.c * ws.v[2],
    ]
end

function _custom_enforce_nonnegativity_from_bases(
    ws,
    p;
    basis_even_no_s,
    basis_even_s,
    basis_odd_no_s,
    basis_odd_s,
)
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
        (
            DynamicPolynomials.coefficient(p, monom) * monom
            for monom in monomials(p)
            if degree(monom, ws.s) == 1
        );
        init = zero(ws.c),
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

    constrained_without_s = _kse_constrain_zero_polynomial(
        ws.model,
        expression_without_s;
    )
    constrained_with_s = _kse_constrain_zero_polynomial(
        ws.model,
        expression_with_s;
    )

    return (
        expressions = [constrained_without_s, constrained_with_s],
        Q_even = Q_even,
        Q_odd = Q_odd,
        basis_even_no_s = basis_even_no_s,
        basis_even_s = basis_even_s,
        basis_odd_no_s = basis_odd_no_s,
        basis_odd_s = basis_odd_s,
    )
end


function build_explicit_kse_model(k, model; basis_even_no_s_builder = explicit_basis_even_no_s)
    ws = _kse_make_workspace(4, k; model = model)

    s = ws.s
    c = ws.c
    v = ws.v
    U = ws.U

    gauge_basis = [
        v[2]*v[3],
        v[1]*v[4],
        v[1]*v[2],
        c*v[4],
        c*v[2],
        s*v[3],
        s*v[1],
        s*c,
        c^2*v[2]*v[3],
        c^2*v[1]*v[4],
        c^2*v[1]*v[2],
        c^3*v[4],
        c^3*v[2],
        s*v[1]^3,
        s*c*v[2]^2,
        s*c*v[1]*v[3],
        s*c*v[1]^2,
        s*c^2*v[3],
        s*c^2*v[1],
        s*c^3,
        c^4*v[1]*v[2],
        s*c^2*v[1]^3,
        s*c^3*v[2]^2,
        s*c^3*v[1]^2,
    ]

    f1, gauge_coeffs = _kse_makepoly_from_basis(ws.model, gauge_basis)

    w_basis = [
        1,
        U[2]^2,
        U[1]*U[2],
        U[1]^2,
        s*c*U[3],
        s*c*U[1],
        c^2*U[2]^2,
        c^2*U[1]*U[3],
        c^2*U[1]^2,
        s*c*U[2]*U[3],
        s*c*U[1]*U[2]
    ]
    _, LV, w_coeffs = _kse_make_ddt_auxiliary(ws, w_basis)

    @variable(ws.model, B >= 0)
    #B = 3
    p1 = _kse_reduce_trig(ws, B - ws.u^2 + LV + ws.ddx(f1))

    certificate = _custom_enforce_nonnegativity_from_bases(
        ws,
        p1;
        basis_even_no_s = basis_even_no_s_builder(ws),
        basis_even_s = explicit_basis_even_s(ws),
        basis_odd_no_s = explicit_basis_odd_no_s(ws),
        basis_odd_s = explicit_basis_odd_s(ws),
    )
    @objective(ws.model, Min, B)

    return (
        model = ws.model,
        ws = ws,
        B = B,
        certificate = certificate,
        gauge_basis = gauge_basis,
        gauge_coeffs = gauge_coeffs,
        w_basis = w_basis,
        w_coeffs = w_coeffs,
    )
end
