# Quasi-convex one-parameter optimization.
#
# This path handles models where the JuMP objective variable multiplies other
# variables in PSD constraints. For a fixed objective value, the model is an
# ordinary SDP, so RationalSDP can solve the explicit optimization by bisection
# over that objective variable.

struct QuasiconvexParameterData{T<:Real}
    parameter::MOI.VariableIndex
    lower::T
    upper::T
    quadratic_psd_constraints::Vector{
        MOI.ConstraintIndex{MOI.VectorQuadraticFunction{T},MOI.PositiveSemidefiniteConeTriangle}
    }
end

function _objective_parameter(storage, ::Type{T}) where {T<:Real}
    MOI.get(storage, MOI.ObjectiveSense()) == MOI.MIN_SENSE || return nothing
    objective_type = MOI.get(storage, MOI.ObjectiveFunctionType())
    if objective_type == MOI.VariableIndex
        return MOI.get(storage, MOI.ObjectiveFunction{MOI.VariableIndex}())
    elseif objective_type <: MOI.ScalarAffineFunction
        func = MOI.get(storage, MOI.ObjectiveFunction{objective_type}())
        iszero(func.constant) || return nothing
        length(func.terms) == 1 || return nothing
        term = only(func.terms)
        term.coefficient == one(T) || return nothing
        return term.variable
    end
    return nothing
end

function _parameter_bounds(storage, parameter::MOI.VariableIndex, ::Type{T}) where {T<:Real}
    lower = nothing
    upper = nothing
    for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.GreaterThan{T}}())
        MOI.get(storage, MOI.ConstraintFunction(), ci) == parameter || continue
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        lower = lower === nothing ? set.lower : max(lower, set.lower)
    end
    for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.LessThan{T}}())
        MOI.get(storage, MOI.ConstraintFunction(), ci) == parameter || continue
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        upper = upper === nothing ? set.upper : min(upper, set.upper)
    end
    for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,MOI.Interval{T}}())
        MOI.get(storage, MOI.ConstraintFunction(), ci) == parameter || continue
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        lower = lower === nothing ? set.lower : max(lower, set.lower)
        upper = upper === nothing ? set.upper : min(upper, set.upper)
    end
    lower === nothing && return nothing
    upper === nothing && return nothing
    lower <= upper || return nothing
    return lower, upper
end

function _is_parameter_quadratic_function(
    func::MOI.VectorQuadraticFunction{T},
    parameter::MOI.VariableIndex,
) where {T<:Real}
    for term in func.quadratic_terms
        scalar = term.scalar_term
        if scalar.variable_1 != parameter && scalar.variable_2 != parameter
            return false
        end
    end
    return true
end

function _detect_quasiconvex_parameter(opt::Optimizer{T}) where {T<:Real}
    storage = opt.storage
    quadratic_psd_constraints = collect(keys(opt.quadratic_psd_functions))
    isempty(quadratic_psd_constraints) && return nothing

    parameter = _objective_parameter(storage, T)
    parameter === nothing && return nothing
    bounds = _parameter_bounds(storage, parameter, T)
    bounds === nothing && return nothing
    lower, upper = bounds

    for ci in quadratic_psd_constraints
        func = opt.quadratic_psd_functions[ci]
        _is_parameter_quadratic_function(func, parameter) || return nothing
    end

    return QuasiconvexParameterData{T}(parameter, lower, upper, quadratic_psd_constraints)
end

_convert_coefficient(::Type{U}, value) where {U<:Real} = convert(U, value)

_remap_variable(variable::MOI.VariableIndex, variable_map) = variable_map[variable]

function _remap_scalar_affine_function(
    func::MOI.ScalarAffineFunction{T},
    variable_map,
    ::Type{U},
) where {T<:Real,U<:Real}
    terms = MOI.ScalarAffineTerm{U}[
        MOI.ScalarAffineTerm(
            _convert_coefficient(U, term.coefficient),
            _remap_variable(term.variable, variable_map),
        ) for term in func.terms
    ]
    return MOI.ScalarAffineFunction(terms, _convert_coefficient(U, func.constant))
end

function _remap_vector_affine_function(
    func::MOI.VectorAffineFunction{T},
    variable_map,
    ::Type{U},
) where {T<:Real,U<:Real}
    terms = MOI.VectorAffineTerm{U}[
        MOI.VectorAffineTerm(
            term.output_index,
            MOI.ScalarAffineTerm(
                _convert_coefficient(U, term.scalar_term.coefficient),
                _remap_variable(term.scalar_term.variable, variable_map),
            ),
        ) for term in func.terms
    ]
    constants = U[_convert_coefficient(U, value) for value in func.constants]
    return MOI.VectorAffineFunction(terms, constants)
end

function _substitute_parameter_quadratic_function(
    func::MOI.VectorQuadraticFunction{T},
    parameter::MOI.VariableIndex,
    fixed_value::U,
    variable_map,
    ::Type{U},
) where {T<:Real,U<:Real}
    terms = MOI.VectorAffineTerm{U}[
        MOI.VectorAffineTerm(
            term.output_index,
            MOI.ScalarAffineTerm(
                _convert_coefficient(U, term.scalar_term.coefficient),
                _remap_variable(term.scalar_term.variable, variable_map),
            ),
        ) for term in func.affine_terms
    ]
    constants = U[_convert_coefficient(U, value) for value in func.constants]

    for term in func.quadratic_terms
        scalar = term.scalar_term
        coefficient = _convert_coefficient(U, scalar.coefficient)
        if scalar.variable_1 == parameter && scalar.variable_2 == parameter
            constants[term.output_index] += coefficient * fixed_value * fixed_value
        elseif scalar.variable_1 == parameter
            push!(
                terms,
                MOI.VectorAffineTerm(
                    term.output_index,
                    MOI.ScalarAffineTerm(
                        coefficient * fixed_value,
                        _remap_variable(scalar.variable_2, variable_map),
                    ),
                ),
            )
        elseif scalar.variable_2 == parameter
            push!(
                terms,
                MOI.VectorAffineTerm(
                    term.output_index,
                    MOI.ScalarAffineTerm(
                        coefficient * fixed_value,
                        _remap_variable(scalar.variable_1, variable_map),
                    ),
                ),
            )
        else
            error("Unsupported quasi-convex quadratic term without objective parameter.")
        end
    end

    return MOI.VectorAffineFunction(terms, constants)
end

function _convert_set(::Type{MOI.EqualTo{U}}, set::MOI.EqualTo{T}) where {T<:Real,U<:Real}
    return MOI.EqualTo(_convert_coefficient(U, set.value))
end

function _convert_set(::Type{MOI.GreaterThan{U}}, set::MOI.GreaterThan{T}) where {T<:Real,U<:Real}
    return MOI.GreaterThan(_convert_coefficient(U, set.lower))
end

function _convert_set(::Type{MOI.LessThan{U}}, set::MOI.LessThan{T}) where {T<:Real,U<:Real}
    return MOI.LessThan(_convert_coefficient(U, set.upper))
end

function _convert_set(::Type{MOI.Interval{U}}, set::MOI.Interval{T}) where {T<:Real,U<:Real}
    return MOI.Interval(_convert_coefficient(U, set.lower), _convert_coefficient(U, set.upper))
end

function _copy_scalar_constraints!(
    dest,
    storage,
    variable_map,
    fixed_parameter::MOI.VariableIndex,
    ::Type{T},
    ::Type{U},
) where {T<:Real,U<:Real}
    scalar_sets = (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval)
    for S in scalar_sets
        for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.ScalarAffineFunction{T},S{T}}())
            func = MOI.get(storage, MOI.ConstraintFunction(), ci)
            set = MOI.get(storage, MOI.ConstraintSet(), ci)
            MOI.add_constraint(
                dest,
                _remap_scalar_affine_function(func, variable_map, U),
                _convert_set(S{U}, set),
            )
        end
    end
    for S in scalar_sets
        for ci in MOI.get(storage, MOI.ListOfConstraintIndices{MOI.VariableIndex,S{T}}())
            func = MOI.get(storage, MOI.ConstraintFunction(), ci)
            func == fixed_parameter && continue
            set = MOI.get(storage, MOI.ConstraintSet(), ci)
            MOI.add_constraint(dest, _remap_variable(func, variable_map), _convert_set(S{U}, set))
        end
    end
    return
end

function _copy_psd_constraints!(
    dest,
    source::Optimizer{T},
    storage,
    variable_map,
    parameter::MOI.VariableIndex,
    fixed_value::U,
    skipped_quadratic_psd_constraints,
    ::Type{T},
    ::Type{U},
) where {T<:Real,U<:Real}
    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorOfVariables,
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        variables = MOI.VariableIndex[
            _remap_variable(variable, variable_map) for variable in func.variables
        ]
        MOI.add_constraint(dest, MOI.VectorOfVariables(variables), set)
    end

    for ci in MOI.get(
        storage,
        MOI.ListOfConstraintIndices{
            MOI.VectorAffineFunction{T},
            MOI.PositiveSemidefiniteConeTriangle,
        }(),
    )
        func = MOI.get(storage, MOI.ConstraintFunction(), ci)
        set = MOI.get(storage, MOI.ConstraintSet(), ci)
        MOI.add_constraint(dest, _remap_vector_affine_function(func, variable_map, U), set)
    end

    skipped = Set(skipped_quadratic_psd_constraints)
    for ci in keys(source.quadratic_psd_functions)
        ci in skipped || error("Unsupported vector quadratic PSD constraint.")
        func = source.quadratic_psd_functions[ci]
        set = source.quadratic_psd_sets[ci]
        MOI.add_constraint(
            dest,
            _substitute_parameter_quadratic_function(func, parameter, fixed_value, variable_map, U),
            set,
        )
    end
    return
end

function _fixed_parameter_optimizer(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    fixed_value::T,
) where {T<:Real}
    child = Optimizer{T}(; (name => getfield(opt.settings, name) for name in _SETTING_FIELDNAMES)...)
    child.silent = true
    storage = opt.storage
    variables = MOI.get(storage, MOI.ListOfVariableIndices())
    variable_map = Dict{MOI.VariableIndex,MOI.VariableIndex}()
    for variable in variables
        variable_map[variable] = MOI.add_variable(child)
    end

    _copy_scalar_constraints!(child, storage, variable_map, data.parameter, T, T)
    MOI.add_constraint(child, variable_map[data.parameter], MOI.EqualTo(fixed_value))
    _copy_psd_constraints!(
        child,
        opt,
        storage,
        variable_map,
        data.parameter,
        fixed_value,
        data.quadratic_psd_constraints,
        T,
        T,
    )
    MOI.set(child, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    return child, variable_map
end

function _fixed_parameter_feasible(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    fixed_value::T,
) where {T<:Real}
    child, variable_map = _fixed_parameter_optimizer(opt, data, fixed_value)
    MOI.optimize!(child)
    return MOI.get(child, MOI.TerminationStatus()) == MOI.OPTIMAL, child, variable_map
end

function _populate_from_quasiconvex_child!(
    opt::Optimizer{T},
    data::QuasiconvexParameterData{T},
    child::Optimizer{T},
    variable_map,
    objective_value::T,
) where {T<:Real}
    for variable in MOI.get(opt.storage, MOI.ListOfVariableIndices())
        opt.variable_primal[variable] = MOI.get(child, MOI.VariablePrimal(), variable_map[variable])
    end
    opt.variable_primal[data.parameter] = objective_value
    opt.objective_value = objective_value
    opt.termination_status = MOI.OPTIMAL
    opt.primal_status = MOI.FEASIBLE_POINT
    opt.dual_status = MOI.NO_SOLUTION
    opt.raw_status = "Solved by quasi-convex parameter search"
    opt.result_count = 1
    return
end

function _try_quasiconvex_parameter_solve!(opt::Optimizer{T}) where {T<:Real}
    data = _detect_quasiconvex_parameter(opt)
    data === nothing && return false

    lower = data.lower
    upper = data.upper
    lower_feasible, lower_child, lower_map = _fixed_parameter_feasible(opt, data, lower)
    if lower_feasible
        _populate_from_quasiconvex_child!(opt, data, lower_child, lower_map, lower)
        return true
    end

    upper_feasible, upper_child, upper_map = _fixed_parameter_feasible(opt, data, upper)
    if !upper_feasible
        opt.termination_status = MOI.INFEASIBLE
        opt.primal_status = MOI.NO_SOLUTION
        opt.dual_status = MOI.NO_SOLUTION
        opt.raw_status = "Quasi-convex parameter upper bound is infeasible"
        return true
    end

    best_child = upper_child
    best_map = upper_map
    for _ in 1:opt.settings.quasiconvex_bisection_iterations
        midpoint = (lower + upper) / 2
        feasible, child, variable_map = _fixed_parameter_feasible(opt, data, midpoint)
        if feasible
            upper = midpoint
            best_child = child
            best_map = variable_map
        else
            lower = midpoint
        end
    end

    _populate_from_quasiconvex_child!(opt, data, best_child, best_map, upper)
    return true
end

function _unsupported_quadratic_psd_error(::Optimizer{T}) where {T<:Real}
    message =
        "RationalSDP only supports vector quadratic PSD constraints for " *
        "one-parameter quasi-convex optimization: the model must minimize a " *
        "single scalar objective variable with finite lower and upper bounds, " *
        "and every quadratic term in each PSD constraint must contain that " *
        "objective variable. General quadratic SDP constraints are not supported."
    return MOI.UnsupportedConstraint{
        MOI.VectorQuadraticFunction{T},
        MOI.PositiveSemidefiniteConeTriangle,
    }(message)
end
