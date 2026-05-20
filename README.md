# RationalSDP.jl

`RationalSDP.jl` is an experimental semidefinite-programming optimizer for
JuMP/MOI models with exact rational data. It solves SDP-like problems
numerically, then returns an exactly affine-feasible rational primal point.

The package is aimed at proof-oriented workflows.

## Status

This is research software. It is not a mature general-purpose SDP solver.

The current implementation is primal-only, has a focused conic surface, and is
optimized for exact rational primal recovery. It is useful for small and
medium-sized SDP/SOS certificate searches, including some quasiconvex
one-parameter problems that are solved by fixed-parameter feasibility searches.

## Supported Model Features

RationalSDP currently supports:

- JuMP/MOI incremental models
- rational scalar variables
- PSD matrix variables
- affine scalar constraints: `==`, `>=`, `<=`, and intervals
- affine PSD constraints
- linear objectives
- exact rational primal values via `value`
- exact rational objective values via `objective_value`
- `Rational{BigInt}` and other rational output types
- selectable numerical working types: `Float64`, `Double64`, `BigFloat`, etc.
- Hypatia-backed or native Phase I feasibility search
- exact affine elimination before the barrier solve
- PSD face pruning for some forced-boundary cases
- optional facial-reduction passes
- threaded PSD barrier assembly
- SumOfSquares.jl models that bridge to supported JuMP/MOI constraints
- one-parameter quasiconvex models where the objective parameter appears
  bilinearly in supported quadratic constraints

The package does not return dual solutions or rigorous dual certificates.

## Minimal JuMP Example

```julia
using JuMP
using RationalSDP

model = GenericModel{Rational{BigInt}}(Optimizer{Rational{BigInt}})

@variable(model, X[1:2, 1:2], PSD)
@variable(model, y)

@constraint(model, X[1, 1] == 1//1)
@constraint(model, X[2, 2] == 1//1)
@constraint(model, X[1, 2] == y)
@constraint(model, y >= 1//2)
@objective(model, Min, y)

optimize!(model)

termination_status(model)
value(y)      # exact Rational{BigInt}
value.(X)     # exact rational matrix
```

Use exact integers and rationals in model data. Avoid float literals such as
`0.1` when you intend a proof-oriented exact model.

## Minimal SumOfSquares Example

`RationalSDP` is compatible with the latest versions of SumOfSquares.jl for SOS optimization.

```julia
using JuMP
using RationalSDP
using DynamicPolynomials
using SumOfSquares

model = GenericModel{Rational{BigInt}}(Optimizer{Rational{BigInt}})

@polyvar z
@variable(model, t)

poly = z^4 - z^2 + t
@constraint(model, poly >= 0, SumOfSquares.SOSCone())
@objective(model, Min, t)

optimize!(model)

termination_status(model)
value(t)      # close to 1//4, returned as an exact rational
```

For larger SOS models, expect performance and robustness to depend strongly on
the Gram basis and on whether the feasible set lies on a PSD face.

## Quasiconvex One-Parameter Problems

RationalSDP has a special path for models that are not affine SDPs because the
objective parameter multiplies other variables, but become ordinary SDP
feasibility problems once that parameter is fixed.

The supported pattern is:

- minimize a single scalar objective variable
- give that objective variable finite lower and upper bounds
- every quadratic term in every quadratic constraint must contain that objective
  variable
- for fixed parameter values, the model must reduce to a supported SDP

Supported quadratic constraints include:

- scalar quadratic constraints such as `gamma * x == 1`
- vector quadratic PSD constraints such as `Symmetric([gamma * x 1; 1 x]) in PSDCone()`

Example:

```julia
using JuMP
using RationalSDP
import MathOptInterface as MOI

model = GenericModel{Rational{BigInt}}(Optimizer{Rational{BigInt}})
set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 12)

@variable(model, 0//1 <= gamma <= 2//1)
@variable(model, 0//1 <= x <= 1//1)

@constraint(model, gamma * x == 1//1)
@objective(model, Min, gamma)

optimize!(model)

value(gamma)
MOI.get(backend(model), MOI.RawStatusString())
```

The solver currently uses a bounded parameter search. It assumes the feasible
set is monotone in the objective parameter; it does not prove monotonicity from
the model.

General quadratic SDP constraints are not supported.

## Solver Settings

All fields of `RationalSDP.Settings` are exposed as JuMP optimizer attributes:

```julia
set_optimizer_attribute(model, "verbose", true)
set_optimizer_attribute(model, "working_float_type", BigFloat)
set_optimizer_attribute(model, "working_precision", 512)
set_optimizer_attribute(model, "phase1_backend", :hypatia)
set_optimizer_attribute(model, "quasiconvex_bisection_iterations", 24)
```

The full set of optimizer attributes is:

- `verbose`
- `verbose_newton`
- `live_progress`
- `inner_log_frequency`
- `max_iterations`
- `phase1_outer_iterations`
- `phase2_outer_iterations`
- `working_float_type`
- `working_precision`
- `phase1_backend`
- `phase1_hypatia_float_type`
- `phase1_hypatia_syssolver`
- `phase1_hypatia_iter_limit`
- `phase1_hypatia_margin_upper`
- `phase1_hypatia_min_margin_upper`
- `phase1_hypatia_margin_shrink`
- `phase1_hypatia_objective_bias`
- `facial_reduction`
- `facial_reduction_max_rounds`
- `facial_reduction_float_type`
- `facial_reduction_exposure_tolerance`
- `facial_reduction_rank_tolerance`
- `facial_reduction_irrational_behavior`
- `feasibility_tolerance`
- `optimality_gap_tolerance`
- `gradient_tolerance`
- `line_search_shrink`
- `armijo_fraction`
- `min_step`
- `initial_scale`
- `initial_penalty`
- `penalty_growth`
- `path_parameter_growth`
- `phase1_center_weight`
- `boundary_fraction`
- `rational_tolerance`
- `exact_refinement_bisections`
- `threaded`
- `threading_min_block_size`
- `iterative_linear_solver`
- `iterative_solver_min_dimension`
- `gc_collect_extraction`
- `gc_collect_full`
- `gc_log`
- `quasiconvex_bisection_iterations`

The default working type is `Double64`. `Float64` is faster but less robust;
`BigFloat` is slower but can help on ill-conditioned models.

## Exactness Model

RationalSDP expects rational model data. The numerical solve is used to locate a
good point, but the returned primal solution is rational and is checked against
the exact affine system.

This does not mean every returned certificate is automatically a complete proof.
For proof use, you should still independently check the final rational matrices,
polynomial identities, and PSD conditions relevant to your argument.

## Limitations

Important current limitations:

- primal solutions only
- no dual variables
- no dual infeasibility or proof certificates
- no full general conic support
- no general nonconvex quadratic support
- quasiconvex support is restricted to one bounded objective parameter
- exact recovery can fail on badly conditioned or nearly infeasible problems
- facial reduction is partial
- large SOS models can be slow

Use established SDP solvers when you need broad conic coverage, dual
information, or production robustness.

## Tests

Fast JuMP/SOS regression tests:

```julia
julia --project=test test/runfasttests.jl
```

Slow regression tests:

```julia
julia --project=test test/runslowtests.jl
```

Package test entry point:

```julia
julia --project=. -e "using Pkg; Pkg.test()"
```

The test suite includes affine SDP models, exact rational output checks,
SumOfSquares integration, PSD face-pruning cases, and quasiconvex parameter
models.
