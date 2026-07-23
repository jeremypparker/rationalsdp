# RationalSDP.jl

`RationalSDP.jl` is an experimental semidefinite-programming optimizer for
JuMP/MOI models with exact rational data. It solves SDP-like problems
numerically, then returns an exactly affine-feasible rational primal point.
Exact primal feasibility does not by itself constitute an exact optimality
proof: ordinary `MOI.OPTIMAL` results use the usual numerical convergence
interpretation unless a particular problem is settled by exact elimination.

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
- selectable numerical working types: `Float64`, `Float64x2`, `BigFloat`, etc.
- Hypatia-backed or native Phase I feasibility search
- exact affine elimination before the barrier solve
- coordinate PSD face pruning for forced-boundary cases
- optional facial-reduction passes with exact evidence certification
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
the Gram basis and on whether the feasible set lies on a PSD face. The solver
can prune some forced zero directions exactly and can run facial reduction when
the exposed face can be certified from exact problem data. For constant-objective
problems, an uncertified numerical face may also be used purely as a search aid.
Any point found that way is accepted only after exact validation of all affine
equations and cones in the original, unreduced SDP; the restriction is never
used to claim infeasibility or an objective bound. For a nonconstant objective,
such a point is returned with `MOI.OTHER_LIMIT` and `MOI.FEASIBLE_POINT`, not
`MOI.OPTIMAL`.

## Facial Reduction

When Phase I finds a numerical point on or near the cone boundary,
RationalSDP tries to move the problem to a smaller exact face before Phase II.
The reduction code treats numerical information as evidence and only applies a
face after exact certification.

Before invoking a numerical oracle, an exact Sieve pass inspects both signs of
each affine equality row. Rows whose multiplier lies in the product dual cone,
has zero right-hand side, and has no free-coordinate component are accepted as
complete exposing certificates. The pass also examines exact row-reduced
combinations while retaining their rational row multipliers, so every accepted
face has explicit provenance.
For very large affine systems, multiplier reconstruction is bounded to avoid
turning preprocessing into a dense quadratic-cost step; individual rows are
still checked exactly and the numerical oracle remains available.

Candidate kernel certificates use an exact row-space cache for
`[Aᵀ; bᵀ]`: each generated form is answered by an exact multiplier `y` with
`Aᵀy = ℓ` and `bᵀy = 0`. This avoids materializing one affine-slice matrix per
nullspace coordinate; the old full-block nullspace construction is retained
only as a bounded small-problem fallback.

The cheap evidence pass tries, in order:

- a cone-dual slack returned by the Hypatia Phase I solve
- kernel directions from the Phase I boundary point

If those do not certify a reducing face, RationalSDP launches a separate
Hypatia exposing-vector oracle. By default this oracle first uses the Phase-I
system solver, then tries a stable alternate solver when recovery is
inconclusive. `facial_reduction_oracle_syssolver` can prefer a solver for the
oracle without changing Phase I, while `facial_reduction_float_type` can
override the oracle float type. An uncertifiable numerical oracle candidate no
longer blocks the alternate solver or configured higher-precision retries.

When both cheap evidence sources certify faces, their exact kernel and scalar
exposures are merged before the reduced problem is built. After each certified
reduction, a bounded rank-expansion oracle pass can search for another exposing
slack normalized on the residual face; control its number of rounds with
`facial_reduction_rank_expansion_rounds` (default `0`). A validated cached face
seeds this expansion before it is applied, so all new directions are accumulated
on the cache record's original formulation. When a save file is configured,
each stronger merged face is checkpointed atomically and replaces the weaker
record that it extends. Fixed-parameter
quasiconvex probes deliberately use the conservative evidence path so that
inconclusive numerical probes do not change the bisection result.

Exact exposing slacks must be nonnegative on scalar cones, PSD on PSD blocks,
expose a nonzero face, vanish on free coordinates, and have an exact affine-row
certificate. PSD kernel directions are accepted as certified facial reductions
only when the exact affine rows or PSD implications prove the direction is
forced to zero. An uncertified rationalized kernel direction may restrict a
feasible-point search, but it is never treated as an equivalent face; any
returned point is validated against the unreduced SDP, and optimization over
that restriction cannot return `MOI.OPTIMAL`.

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
the model. Reaching the configured bisection count returns `MOI.ITERATION_LIMIT`
with the best exact feasible primal point found. A fixed-parameter probe that is
neither feasible nor reported infeasible by the numerical backend returns
`MOI.NUMERICAL_ERROR` rather than being treated as infeasible.

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

- `max_iterations`
- `phase1_outer_iterations`
- `phase2_outer_iterations`
- `phase1_backend`
- `phase1_hypatia_float_type`
- `phase1_hypatia_syssolver`
- `phase1_hypatia_iter_limit`
- `phase1_hypatia_target_margin`
- `phase1_hypatia_margin_upper`
- `phase1_hypatia_min_margin_upper`
- `phase1_hypatia_margin_shrink`
- `phase1_hypatia_boundary_margin_fraction`
- `phase1_hypatia_tol_rel_opt`
- `phase1_hypatia_tol_abs_opt`
- `phase1_hypatia_tol_feas`
- `phase1_hypatia_default_tol_power`
- `phase1_hypatia_default_tol_relax`
- `phase1_hypatia_tol_slow`
- `phase1_candidate_diagnostics`
- `phase1_stop_after_candidate_diagnostics`
- `phase1_exact_recovery_diagnostics`
- `phase1_exact_recovery_pivot_log_frequency`
- `working_float_type`
- `facial_reduction`
- `facial_reduction_max_rounds`
- `facial_reduction_rank_expansion_rounds`
- `facial_reduction_float_type`
- `facial_reduction_exposure_tolerance`
- `facial_reduction_rank_tolerance`
- `facial_reduction_subspace_max_charts`
- `facial_reduction_projector_recovery`
- `facial_reduction_multiblock_recovery`
- `facial_reduction_stable_projective_recovery`
- `facial_reduction_oracle_syssolver`
- `facial_reduction_oracle_precision_escalation_max_retries`
- `facial_reduction_precision_escalation_max_retries`
- `facial_reduction_irrational_behavior`
- `facial_reduction_save_file`
- `facial_reduction_load_file`
- `facial_reduction_row_space_max_entries`
- `facial_reduction_weighted_subspace_max_form_entries`
- `facial_reduction_weighted_subspace_max_affine_products`
- `facial_reduction_cheap_weighted_subspace_max_weight_dimension`
- `facial_reduction_cheap_weighted_subspace_max_form_entries`
- `facial_reduction_cheap_weighted_subspace_max_affine_products`
- `facial_reduction_numeric_weighted_subspace_max_form_entries`
- `facial_reduction_numeric_weighted_subspace_max_affine_products`
- `facial_reduction_individual_max_affine_products`
- `facial_reduction_sparse_affine_validation_max_products`
- `facial_reduction_sieve_transform_max_entries`
- `facial_reduction_affine_compaction_factor`
- `facial_reduction_affine_compaction_max_entries`
- `facial_reduction_tentative_max_directions`
- `facial_reduction_tentative_max_coordinate_entries`
- `facial_reduction_tentative_max_lift_products`
- `facial_reduction_tentative_max_estimated_bytes`
- `facial_reduction_affine_lift_max_output_entries`
- `facial_reduction_affine_lift_max_estimated_bytes`
- `facial_reduction_affine_lift_chunk_columns`
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
- `working_precision`
- `rational_tolerance`
- `recovery_tolerance_shrink`
- `exact_refinement_bisections`
- `verbose`
- `verbose_newton`
- `live_progress`
- `inner_log_frequency`
- `threaded`
- `threading_min_block_size`
- `iterative_linear_solver`
- `iterative_solver_min_dimension`
- `gc_collect_extraction`
- `gc_collect_full`
- `gc_log`
- `quasiconvex_bisection_iterations`
- `quasiconvex_skip_facial_reduction_after_clean_endpoint`

The facial-reduction work limits bound optional certificate, scouting, and
validation calculations whose matrices can become large. They do not relax any
exact check that is needed for correctness: when a limit is exceeded, the
solver skips that optional route and continues with the other facial-reduction
routes. The work limits are nonnegative integers; setting a max-entry/product
limit to `0` disables that optional route. The compaction factor is also
nonnegative, with `0` requesting compaction whenever extra affine equations
are present.

Boundary-kernel recovery is joint and basis invariant. RationalSDP tries up to
`facial_reduction_subspace_max_charts` well-conditioned pivot charts, converts
each exact candidate span to a canonical row-reduced key, and certifies each
distinct span at most once. With `facial_reduction_projector_recovery=true`, an
exact symmetric idempotent rationalization of the numerical orthogonal
projector is also tried. `facial_reduction_precision_escalation_max_retries`
controls the opt-in Phase-I precision ladder and, for compatibility, can raise
the oracle retry budget. The independent
`facial_reduction_oracle_precision_escalation_max_retries` defaults to `1`.
The oracle tries the configured system solver first, then a numerically
different solver. It escalates precision when the available solver attempts
fail to produce an exactly certifiable face; a completed conic solve with no
face does not by itself make the whole recovery pipeline successful.
Higher-precision Phase-I retries use dense QR-Cholesky linear algebra instead
of the Float64 indirect solver.

| Attribute | Default | Work estimate bounded |
| --- | ---: | --- |
| `facial_reduction_row_space_max_entries` | `250_000` | Entries in the augmented affine row-space matrix used to recover optional provenance multipliers. |
| `facial_reduction_weighted_subspace_max_form_entries` | `250_000` | Entries in an exact weighted-subspace exposing-form matrix. |
| `facial_reduction_weighted_subspace_max_affine_products` | `5_000_000` | Exact products needed to test that weighted exposing form on the affine parametrization. |
| `facial_reduction_cheap_weighted_subspace_max_weight_dimension` | `6` | Independent symmetric weights allowed in the early “cheap” exact attempt. |
| `facial_reduction_cheap_weighted_subspace_max_form_entries` | `25_000` | Form entries allowed in the early “cheap” exact attempt. |
| `facial_reduction_cheap_weighted_subspace_max_affine_products` | `500_000` | Affine products allowed in the early “cheap” exact attempt. |
| `facial_reduction_numeric_weighted_subspace_max_form_entries` | `1_000_000` | Entries in the numerical weighted-subspace scout's form matrix. |
| `facial_reduction_numeric_weighted_subspace_max_affine_products` | `25_000_000` | Floating-point products estimated for that numerical scout. |
| `facial_reduction_individual_max_affine_products` | `5_000_000` | Exact products estimated when certifying proposed kernel directions individually. |
| `facial_reduction_sparse_affine_validation_max_products` | `5_000_000` | Exact products in the redundant post-solve validation of a sparse face restriction. |
| `facial_reduction_sieve_transform_max_entries` | `250_000` | Entries in the affine system before the Sieve skips transformed-row provenance. |
| `facial_reduction_affine_compaction_factor` | `4` | Number of original affine equations allowed before redundant-equation compaction is attempted. |
| `facial_reduction_affine_compaction_max_entries` | `5_000_000` | Entries in the augmented exact affine matrix eligible for optional redundant-equation compaction; `0` disables compaction. |
| `facial_reduction_tentative_max_directions` | `8` | Numerically proposed PSD kernel directions admitted in one tentative batch. |
| `facial_reduction_tentative_max_coordinate_entries` | `5_000_000` | Entries in the exact coordinate restriction system for a tentative face. |
| `facial_reduction_tentative_max_lift_products` | `250_000_000` | Estimated exact products needed to lift a tentative affine basis. |
| `facial_reduction_tentative_max_estimated_bytes` | `1_000_000_000` | Estimated retained bytes for a lifted tentative affine basis. |
| `facial_reduction_affine_lift_max_output_entries` | `5_000_000` | Rational entries retained by an exact affine-basis lift; `0` disables optional lifting. |
| `facial_reduction_affine_lift_max_estimated_bytes` | `1_000_000_000` | Estimated retained bytes for any exact affine-basis lift, using current rational coefficient sizes. |
| `facial_reduction_affine_lift_chunk_columns` | `32` | Columns per Nemo multiplication chunk during exact affine lifting. |

When a complete tentative batch exceeds any budget, RationalSDP does not build
it. It tries the best single direction that fits, reruns Phase I on that
restricted problem, and recomputes the remaining numerical kernels. Exact
affine lifting exploits the identity block for newly introduced face
coordinates and never materializes the old extended-nullspace matrix.

Facial-reduction caches can be used to reuse exact faces across related
instances:

```julia
set_optimizer_attribute(model, "facial_reduction_save_file", "faces.rsdpcache")
set_optimizer_attribute(next_model, "facial_reduction_load_file", "faces.rsdpcache")
```

The cache stores exact certified reductions. Loading validates that the current
affine slice is contained in the saved face before applying it, so objective
coefficient changes are fine and incompatible constraint changes fall back to
the normal facial-reduction search.

Each solve records facial-reduction counters and timings. After a solve, obtain
an immutable snapshot with `RationalSDP.facial_reduction_statistics(optimizer)`.
The snapshot includes Phase-I and oracle work, exact RREF dimensions and time,
exact certificate checks, PSD eigendecompositions by block size, certified and
tentative direction outcomes, dimension removed per round, and approximate peak
memory for affine and facial-reduction caches. Tentative faces are never written
to the facial-reduction cache or used as an infeasibility certificate.

The default working type is `Float64x2`. `Float64` is faster but less robust;
`BigFloat` is slower but can help on ill-conditioned models. MultiFloats.jl v3
scalar types are also accepted, for example `Float64x2`, `Float64x3`, and
`Float64x4` via type values or optimizer-attribute strings.

Hypatia Phase I tolerance attributes use negative values to leave Hypatia's
own defaults unchanged. The diagnostic attributes are off by default; enable
`phase1_candidate_diagnostics` or `phase1_exact_recovery_diagnostics` when a
model reaches a numerical boundary point but exact recovery fails.
`phase1_candidate_diagnostics` reports scalar margins and PSD spectra only; it
does not rationalize kernel directions or run exact certificate checks. The
separate facial-reduction oracle reuses the Phase I Hypatia settings by
default. The `recovery_tolerance_shrink` setting controls how aggressively
exact recovery tightens rationalization tolerances between attempts.

An optimal or near-optimal Hypatia Phase I margin at most
`phase1_hypatia_boundary_margin_fraction` times the requested target margin is
treated as a numerical boundary point. The default fraction is `0.01`. Set it
to zero to disable this positive-margin test; nonpositive margins are always
treated as boundary points. Exact recovery is skipped and, when enabled, facial
reduction is attempted directly.

## Exactness Model

RationalSDP expects rational model data. The numerical solve is used to locate a
good point, but the returned primal solution is rational and is checked against
the exact affine system.

For ordinary optimization, `MOI.OPTIMAL` is returned only after Phase II meets
both its numerical stationarity and gap criteria. Iteration or line-search
limits can still return an exact feasible primal point with a non-optimal
termination status.

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
- facial reduction is partial; uncertified face candidates are only feasible-point search aids
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

The opt-in extra-slow regressions include the end-to-end `n=2` SAIRS model:

```sh
RATIONALSDP_EXTRA_SLOW_TESTS=1 julia --project=test test/runslowtests.jl
```

In PowerShell:

```powershell
$env:RATIONALSDP_EXTRA_SLOW_TESTS = "1"; julia --project=test test/runslowtests.jl
```

Package test entry point:

```julia
julia --project=. -e "using Pkg; Pkg.test()"
```

The test suite includes affine SDP models, exact rational output checks,
SumOfSquares integration, PSD face-pruning and facial-reduction cases,
quasiconvex parameter models, and slow SOS regressions from larger log-domain
examples.
