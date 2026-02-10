# Test Suite

MixedHierarchyGames.jl has **450 tests** across **16 test files** (plus 1 shared utility module and 1 shared OLSE reference implementation).

## Running Tests

### Run all tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or directly:

```bash
julia --project=. test/runtests.jl
```

### Run a single test file

```bash
julia --project=. -e 'using Test; using MixedHierarchyGames; include("test/testing_utils.jl"); include("test/test_graph_utils.jl")'
```

Some test files depend on `testing_utils.jl` (provides `make_θ` helper), so always include it first. The OLSE test files (`olse/test_qp_solver.jl`, `olse/test_nonlinear_solver.jl`) also include their own shared module (`olse/olse_closed_form.jl`) internally.

### Run a specific testset

Use Julia's `Test` filtering:

```bash
julia --project=. -e '
using Test, MixedHierarchyGames
include("test/testing_utils.jl")
@testset "Subset" begin
    include("test/test_qp_solver.jl")
end
'
```

## Test File Organization

Tests are organized by implementation phase and listed in the order they run in `runtests.jl`:

### Core Utilities (Phase A)

| File | Tests | Description |
|------|-------|-------------|
| `test_graph_utils.jl` | Graph queries | `is_root`, `is_leaf`, `get_roots`, `get_all_leaders`, `get_all_followers` on chain, tree, mixed, and diamond hierarchies |
| `test_symbolic_utils.jl` | Symbolic variable creation | `make_symbol`, `make_symbolic_vector`, `make_symbolic_matrix` for player and pair variable naming |

### Problem Setup (Phase B)

| File | Tests | Description |
|------|-------|-------------|
| `test_problem_setup.jl` | Variable setup and info vectors | `setup_problem_parameter_variables`, `setup_problem_variables`, information vector construction per hierarchy, coupled constraint rejection |

### QP KKT Construction (Phase C)

| File | Tests | Description |
|------|-------|-------------|
| `test_qp_kkt.jl` | KKT system assembly | `get_qp_kkt_conditions` (leaf/leader KKT, M/N matrices), `strip_policy_constraints` |

### QP Solver (Phase D)

| File | Tests | Description |
|------|-------|-------------|
| `test_qp_solver.jl` | QP solving and `QPSolver` struct | `solve_with_path`, `qp_game_linsolve`, `_run_qp_solver` (linear and PATH backends), `QPSolver` constructor, `solve`/`solve_raw`, configurable solver parameters |

### Linesearch (Phase E)

| File | Tests | Description |
|------|-------|-------------|
| `test_linesearch.jl` | Armijo backtracking | Full step acceptance, backtracking, minimum step size, sufficient decrease condition |

### Nonlinear Solver (Phase F)

| File | Tests | Description |
|------|-------|-------------|
| `test_nonlinear_solver.jl` | Nonlinear Newton solver | `setup_approximate_kkt_solver`, `preoptimize_nonlinear_solver`, `compute_K_evals`, `run_nonlinear_solver` (convergence, max iters, initial guess), failure paths, `NonlinearSolver` input validation |

### KKT Verification

| File | Tests | Description |
|------|-------|-------------|
| `test_kkt_verification.jl` | Solution verification utilities | `evaluate_kkt_residuals` (with/without parameters, enforcement, verbose), `verify_kkt_solution` (single-player, multi-player, invalid solutions) |

### Interface (Phase G)

| File | Tests | Description |
|------|-------|-------------|
| `test_interface.jl` | TrajectoryGamesBase integration | `extract_trajectories`, `solution_to_joint_strategy`, `solve_trajectory_game!` returning `JointStrategy`/`OpenLoopStrategy` for QP and Nonlinear solvers |

### Input Validation

| File | Tests | Description |
|------|-------|-------------|
| `test_input_validation.jl` | Constructor and solve-time validation | Cyclic graphs, self-loops, multiple leaders, dimension mismatches, missing players/parameters, constraint function signatures, `QPPrecomputed` struct |

### Integration Tests

| File | Tests | Description |
|------|-------|-------------|
| `test_integration.jl` | Cross-solver and paper examples | QP vs Nonlinear solver agreement, SIOPT paper OLSE verification (closed-form comparison), single-player edge case, 3-player chain hierarchy |

### OLSE Validation

| File | Tests | Description |
|------|-------|-------------|
| `olse/test_qp_solver.jl` | QP solver against closed-form OLSE | SIOPT 2-player Stackelberg, multiple initial conditions, Nash game (no edges) |
| `olse/test_nonlinear_solver.jl` | Nonlinear solver against closed-form OLSE | Follower response properties, OLSE solution properties (FOC, policy constraints), solver vs closed-form comparison, equilibrium uniqueness, cost optimality, multiple time horizons |

### Type Stability

| File | Tests | Description |
|------|-------|-------------|
| `test_type_stability.jl` | Container type correctness | Verifies `Dict{Int, Vector{Num}}` and `Dict{Int, Union{Matrix{Float64}, Nothing}}` types for problem variables, KKT setup, and K evaluations |

### Timer Integration

| File | Tests | Description |
|------|-------|-------------|
| `test_timer.jl` | `TimerOutputs` instrumentation | Verifies timing sections are recorded for QPSolver/NonlinearSolver construction and solve, backward compatibility without `to` kwarg |

### Shared Files

| File | Description |
|------|-------------|
| `testing_utils.jl` | `make_θ` helper for creating symbolic parameter vectors |
| `olse/olse_closed_form.jl` | `OLSEProblemData` struct, `compute_follower_response`, `compute_olse_solution`, verification functions (shared by both OLSE test files) |

## Adding New Tests

1. Create a new test file `test/test_<feature>.jl`
2. Add `include("test_<feature>.jl")` to `runtests.jl` in the appropriate phase section
3. If your tests need symbolic parameter variables, `include("testing_utils.jl")` is already loaded by `runtests.jl`

Follow TDD as required by `CLAUDE.md`: write a failing test first, then implement, then refactor.

## Test Tolerances

Per `CLAUDE.md`, tolerances must be `1e-6` or tighter:

- General correctness: `1e-6`
- Linear algebra / machine precision: `1e-10` to `1e-14`
- Never loosen tolerances to make failing tests pass; investigate root causes instead
