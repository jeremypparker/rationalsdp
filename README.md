# RationalSDP

`RationalSDP.jl` is a small MathOptInterface-compatible SDP solver for JuMP
models where the coefficients are rational and you want an exactly
affine-feasible rational primal solution back at the end.

The solver now supports:

- PSD matrix variables in `PositiveSemidefiniteConeTriangle`
- unconstrained scalar variables alongside PSD blocks
- scalar equalities, inequalities, and interval constraints
- linear objectives
- exact rational recovery for the returned primal point
- user-selectable output types such as `Rational{BigInt}` or `Rational{Int}`

The default algorithm is a two-stage interior-point method:

1. A penalty-barrier Phase I finds a strictly feasible interior point.
2. A feasible barrier path-following Phase II improves the objective.
3. The final point is projected back to an exact rational affine-feasible
   solution.

## Example

```julia
using JuMP
using RationalSDP

model = GenericModel{Rational{BigInt}}(
    () -> RationalSDP.Optimizer{Rational{BigInt}}(verbose = true),
)

@variable(model, X[1:2, 1:2], PSD)
@variable(model, y)
@constraint(model, X[1, 1] == 2//1)
@constraint(model, X[2, 2] == 2//1)
@constraint(model, X[1, 2] - y == 0//1)
@constraint(model, y >= 1//2)
@objective(model, Min, y)

optimize!(model)

println(value.(X))
println(value(y))
```

Set `verbose = false` to suppress iteration logs. In JuMP, `set_silent(model)`
also suppresses the solver output.

## Output Types

`Rational{BigInt}` is the safest choice if you want guaranteed exact output for
general problems. Smaller rational types are supported, but they can only be
used when the exact recovered numerator and denominator fit in the chosen
integer type.

## Current Scope

The implementation is still intentionally focused:

- primal-only solver
- linear objective only
- no dual certificates
- the problem should admit a strictly feasible interior point

## Tests And Benchmarks

The test suite includes:

- mixed PSD and non-PSD models
- scalar inequality and interval constraints
- larger mixed-cone and LP-style instances
- a small SOS-style polynomial optimization example built with
  `DynamicPolynomials`

Run the tests with:

```julia
julia --project=. -e "using Pkg; Pkg.test()"
```

Run the benchmark script with:

```julia
julia benchmark/run_benchmarks.jl
```

Set `RATIONALSDP_BENCH_REPS` to control the number of timed repetitions.
