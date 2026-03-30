# RationalSDP

`RationalSDP.jl` is an experimental semidefinite programming solver for JuMP
models with rational data. Its main goal is not raw speed or broad conic
coverage; it is to take a rational SDP-like model, solve it with an interior
point method, and return an exactly affine-feasible rational primal solution.

## Status

This project has absolutely been vibe-coded.

That is meant literally: most of the implementation was built interactively with
AI assistance rather than through a long, traditional solver-development cycle.
So this is not a battle-hardened industrial optimizer, and you should treat it
as experimental.

That said, it has also been developed in a test-driven way. New features and bug
fixes have generally come with JuMP-based regression tests, and the current test
suite covers:

- PSD-only and mixed PSD/scalar models
- scalar equalities, inequalities, and interval constraints
- exact rational output with multiple rational output types
- SOS-style examples built directly with `DynamicPolynomials`
- Lyapunov-style SOS models, including cases that need PSD face pruning
- larger mixed-cone examples and benchmark-sized regression cases

So the right mental model is: experimental, candidly AI-built, but not random.

## What It Does

`RationalSDP` currently supports:

- PSD matrix variables in `MOI.PositiveSemidefiniteConeTriangle`
- unconstrained scalar variables alongside PSD blocks
- scalar `==`, `>=`, `<=`, and interval constraints
- linear objectives
- exact rational primal recovery at the end of the solve
- user-selectable output types such as `Rational{BigInt}` and `Rational{Int}`
- user-selectable internal working float types, with `Double64` as the default
- a Hypatia-backed centered Phase I solve on the cleaned extracted conic problem
- solver settings exposed through JuMP / MOI optimizer attributes
- threaded PSD barrier assembly for larger blocks

The solver is a two-stage primal interior-point method:

1. Phase I, by default, solves a centered conic feasibility model with Hypatia
   to get a numerically central interior candidate.
2. Phase II follows the barrier path to improve the objective.
3. The final point is projected back to an exact rational primal solution.

## Current Limitations

This is the important section to read before you depend on it.

- It is a primal-only solver, not a primal-dual IPM.
- It does not currently return dual certificates or dual variable values.
- It is aimed at problems that admit a strictly feasible interior point.
- It has some PSD face pruning, but not full general facial reduction.
- It is focused on rational affine data and exact primal recovery, not on being
  a broad high-performance conic solver.

If you want a mature production SDP solver with stronger robustness, better dual
information, and wider problem coverage, you should still look at established
solver stacks.

## Installation

From a local clone of this repository:

```julia
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

If you want to use the package from another Julia environment after cloning:

```julia
using Pkg
Pkg.develop(path = "/path/to/RationalSDP")
```

## Quick Start

```julia
using JuMP
using RationalSDP

model = GenericModel{Rational{BigInt}}(RationalSDP.Optimizer{Rational{BigInt}})
set_optimizer_attribute(model, "verbose", true)

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
println(termination_status(model))
println(primal_status(model))
println(dual_status(model))
```

On success, `value.(...)` returns exact rational values of the model's numeric
type. When Phase II optimization is used, the solver now performs an exact
rational refinement from the strictly interior anchor before returning the final
point, which makes near-boundary optima more robust than a single final
rationalization pass.

## Configuring The Solver

All fields of `RationalSDP.Settings` are exposed as optimizer attributes, so the
preferred JuMP style is:

```julia
model = GenericModel{Rational{BigInt}}(RationalSDP.Optimizer{Rational{BigInt}})
set_optimizer_attribute(model, "verbose", true)
set_optimizer_attribute(model, "phase1_outer_iterations", 100)
set_optimizer_attribute(model, "working_float_type", RationalSDP.Double64)
set_optimizer_attribute(model, "threaded", true)
```

You can also query them back:

```julia
get_optimizer_attribute(model, "phase1_outer_iterations")
```

Some particularly useful settings are:

- `verbose`: print the phase logs
- `verbose_newton`: also print inner Newton progress
- `phase1_backend`: choose `:hypatia` or `:native` for the Phase I feasibility search
- `phase1_hypatia_float_type`: floating-point type used by the Hypatia Phase I backend
- `phase1_hypatia_iter_limit`: iteration cap for the Hypatia Phase I backend
- `phase1_hypatia_margin_upper`: upper bound on the centered Phase I margin variable
- `phase1_outer_iterations`: maximum number of outer Phase I penalty updates
- `phase2_outer_iterations`: maximum number of outer Phase II barrier updates
- `working_float_type`: internal floating-point type used during the numerical solve
- `working_precision`: `BigFloat` precision, used only when `working_float_type == BigFloat`
- `rational_tolerance`: tolerance used during exact rational recovery
- `exact_refinement_bisections`: dyadic refinement budget for exact rational Phase II improvement
- `threaded`: enable threaded PSD barrier assembly
- `threading_min_block_size`: block-size threshold before threading is used

By default, the numerical solve uses `Double64` from `DoubleFloats`, which is a
reasonable middle ground between `Float64` and `BigFloat`. Performance depends
on the problem and on the linear algebra path taken by the chosen type, so it
is worth benchmarking `Float64`, `Double64`, and `BigFloat` on your own models.
If you want a different internal type, for example:

```julia
set_optimizer_attribute(model, "working_float_type", Float64)
set_optimizer_attribute(model, "working_float_type", BigFloat)
```

By default, Phase I uses `Hypatia` with `Float64` on the cleaned conic problem,
even if Phase II uses a different internal type. That split is deliberate:
Phase I only needs to deliver a good numerical interior candidate, and
Hypatia's strongest sparse linear algebra path is currently most attractive in
`Float64`. If you want to force the old in-package Phase I instead:

```julia
set_optimizer_attribute(model, "phase1_backend", :native)
```

## Logging

With `verbose = true`, the solver prints:

- a short solve summary
- a live Phase I log
- a live Phase II table, with one row printed after each outer iteration
- a final completion line with solve time and exact objective

When `phase1_backend == :hypatia`, the Phase I output comes from Hypatia's own
iteration log plus a short RationalSDP summary line with the centered margin
and affine residual. When `phase1_backend == :native`, RationalSDP prints its
own Phase I table.

Use `set_silent(model)` to suppress solver output through JuMP.

## Output Types

`Rational{BigInt}` is the safest output type and the default recommendation.
It can represent the recovered exact solution without overflow.

Smaller types such as `Rational{Int}` also work, but only when the exact
recovered numerators and denominators fit in the chosen integer type.

## Sum-Of-Squares / Polynomial Models

The package is not a dedicated SOS layer, but it does work well with JuMP plus
`DynamicPolynomials` if you build the Gram-matrix constraints yourself.

The test suite includes:

- direct SOS lower-bound examples
- Lyapunov-style SOS feasibility examples
- Lorenz-style polynomial bounds

One practical caveat: if your SOS lift introduces obviously redundant Gram
directions, the feasible set may lie on a face of the PSD cone. The solver has
some automatic pruning for forced nullspace directions, but not full general
facial reduction.

## Testing

Run the full test suite with:

```julia
julia --project=. -e "using Pkg; Pkg.test()"
```

The tests are JuMP-level integration tests, not just unit tests for internal
helpers, so they are a decent indicator that the public workflow still works.

## Benchmarks

Run the benchmark script with:

```julia
julia --project=. benchmark/runbenchmarks.jl
```

Set `RATIONALSDP_BENCH_REPS` to control the number of timed repetitions.

The benchmark suite includes:

- a box-constrained LP-style model
- a larger mixed PSD/scalar model
- a small SOS quartic model
- a quartic Lorenz-style polynomial SDP relaxation

## When This Solver Is A Good Fit

`RationalSDP` is a good fit when:

- you care about exact rational primal output
- your model is naturally written in JuMP / MOI
- your problem is SDP-flavored but still within the current scope
- you are comfortable using an experimental solver with a growing test suite

It is probably not the right fit when:

- you need dual solutions or certificates
- you need the broadest possible conic support
- you need industrial-scale robustness more than exact rational recovery

## License / Maintenance

This repository is currently best viewed as an experimental research / hobby
solver project. Contributions, bug reports, and new regression tests are very
welcome, especially for small failing examples.
