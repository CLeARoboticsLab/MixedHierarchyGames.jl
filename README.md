# MixedHierarchyGames.jl

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://CLeARoboticsLab.github.io/MixedHierarchyGames.jl/dev/)

A Julia package for solving mixed hierarchy games. This implementation uses the `TrajectoryGamesBase.jl` interface, though the solver is more general and can be used for general equality-constrained games.

Based on: H. Khan, D. H. Lee, J. Li, T. Qiu, C. Ellis, J. Milzman, W. Suttle, and D. Fridovich-Keil, "[Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations](https://arxiv.org/abs/2602.01568)," 2026.

## Citation

If you use this software in your research, please cite:

```bibtex
@article{khan2026mixedhierarchygames,
  title={Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations},
  author={Khan, Hamzah and Lee, Dong Ho and Li, Jingqi and Qiu, Tianyu and Ellis, Christian and Milzman, Jesse and Suttle, Wesley and Fridovich-Keil, David},
  journal={arXiv preprint arXiv:2602.01568},
  year={2026},
  url={https://arxiv.org/abs/2602.01568}
}
```

## Features

- **Flexible hierarchy structures**: Supports arbitrary DAG-based leader-follower relationships (pure Stackelberg, Nash, or mixed)
- **QP Solver**: For linear-quadratic games with equality constraints
- **Nonlinear Solver**: For general nonlinear games using iterative quasi-linear policy approximation
- **TrajectoryGamesBase integration**: Compatible with the TrajectoryGamesBase.jl ecosystem

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/CLeARoboticsLab/MixedHierarchyGames.jl")
```

Or in the package manager (press `]`):
```
add https://github.com/CLeARoboticsLab/MixedHierarchyGames.jl
```

## Quick Start

```julia
# Required packages (install once with: using Pkg; Pkg.add(["Graphs", "Symbolics"]))
using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: @variables
using LinearAlgebra: norm  # Optional, for solution analysis

# Define a 2-player Stackelberg game: Player 1 leads Player 2
G = SimpleDiGraph(2)
add_edge!(G, 1, 2)

# Problem dimensions
state_dim = 2
control_dim = 1
T = 3  # horizon
primal_dims = [(state_dim + control_dim) * T, (state_dim + control_dim) * T]

# Symbolic parameters (θs) - user-defined parameters of arbitrary size per player.
# Common uses: initial states, reference trajectories, obstacle positions.
# These become inputs to solve() and allow the same solver to handle different scenarios.
@variables θ1[1:state_dim] θ2[1:state_dim]
θs = Dict(1 => collect(θ1), 2 => collect(θ2))

# Constraints (initial state constraints)
gs = [
    z -> z[1:state_dim] - collect(θ1),
    z -> z[1:state_dim] - collect(θ2),
]

# Cost functions: Js[i](z1, z2, ..., zN; θ=combined_params) → scalar
# The θ keyword receives all parameter values concatenated [θ1; θ2; ...] during solve.
# For simple cases where costs don't depend on parameters, you can ignore θ.
Js = Dict(
    1 => (z1, z2; θ=nothing) -> sum(z1.^2),
    2 => (z1, z2; θ=nothing) -> sum(z2.^2),
)

# Create solver (precomputes symbolic KKT conditions)
solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

# Solve with specific initial states
strategy = solve(solver, Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0]))

# Access trajectories
player1_states = strategy.substrategies[1].xs
player1_controls = strategy.substrategies[1].us
```

## Examples

See the `experiments/` folder for complete examples:

- **lq_three_player_chain**: 3-player Stackelberg chain with single integrator dynamics
- **nonlinear_lane_change**: 4-vehicle highway scenario with unicycle dynamics
- **pursuer_protector_vip**: 3-agent pursuit-protection game
- **convergence_analysis**: Multi-run convergence analysis for nonlinear solver

Each experiment has a `run.jl` entry point. See `experiments/README.md` for details.

## Hierarchy Graph Structure

The hierarchy is defined using a directed acyclic graph (DAG):
- **Edge i → j** means player i is a leader of player j
- **No edge** between players means they play Nash (simultaneous)
- Players are processed in reverse topological order (followers first)

```julia
# Examples:
# Pure Stackelberg (1 leads 2 leads 3)
G = SimpleDiGraph(3)
add_edge!(G, 1, 2)
add_edge!(G, 2, 3)

# Nash game (no edges)
G = SimpleDiGraph(2)

# Mixed: 1 leads 2 and 3, which both play Nash.
G = SimpleDiGraph(3)
add_edge!(G, 1, 2)
add_edge!(G, 1, 3)
```

## Solvers

### QPSolver

For linear-quadratic games with equality constraints. Supports two backends:
- `:linear` (default) - Direct linear solve of KKT system
- `:path` - PATH solver via ParametricMCPs.jl

**Which solver to use:** Use `:linear` for all current use cases. Both backends solve the same linear KKT system; `:linear` uses direct factorization while `:path` uses the PATH complementarity solver. The `:linear` backend is faster and recommended. The `:path` backend may help with numerically difficult problems.

```julia
solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:linear)
```

### NonlinearSolver

For general nonlinear games using iterative quasi-linear policy approximation with configurable line search.

```julia
solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim;
                         max_iters=100, tol=1e-6, verbose=false, linesearch_method=:geometric)

# Solve with initial guess (optional)
strategy = solve(solver, parameter_values; initial_guess=z0)

# Get raw solution with convergence info
result = solve_raw(solver, parameter_values)
# result.sol, result.converged, result.iterations, result.residual, result.status
```

## Performance Profiling

MixedHierarchyGames.jl includes a conditional timing system built on [TimerOutputs.jl](https://github.com/KristofferC/TimerOutputs.jl). Standard `@timeit` incurs overhead on every call even when profiling is not needed; `@timeit_debug` replaces it with a near-zero-overhead alternative that can be toggled at runtime.

By default, timing is **disabled**. When disabled, the only cost is one atomic boolean check per instrumentation point (~6ns) — no `try/finally` frame, no TimerOutputs bookkeeping.

### Usage

Pass a `TimerOutput` via the `to` keyword argument to capture timing data:

```julia
using MixedHierarchyGames
using TimerOutputs

to = TimerOutput()

# Enable timing and build/solve with the same TimerOutput
enable_timing!()
solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; to)
result = solve(solver, parameter_values; to)
disable_timing!()

# View timing breakdown
show(to)
```

Or use the scoped `with_timing` form to avoid forgetting `disable_timing!()`:

```julia
to = TimerOutput()
with_timing() do
    solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; to)
    solve(solver, parameter_values; to)
end
show(to)
```

### Instrumented Sections

The solver instruments 21 points across construction and solving:

| Phase | Sections |
|-------|----------|
| **QPSolver construction** | `KKT conditions`, `ParametricMCP build`, `linearity check` |
| **QPSolver solve** | `residual evaluation`, `Jacobian evaluation`, `linear solve` |
| **NonlinearSolver construction** | `variable setup`, `approximate KKT setup`, `ParametricMCP build`, `linear solver init` |
| **NonlinearSolver solve** (per iteration) | `compute K evals`, `residual evaluation`, `Jacobian evaluation`, `Newton step`, `line search` |

### Performance Characteristics

Overhead depends on whether timing is enabled (benchmarked on Apple M1):

- **Disabled (default)**: ~6ns per instrumentation point (atomic boolean check only)
- **Enabled**: ~33ns per point (full TimerOutputs instrumentation)

For a typical `QPSolver.solve()` call with 5-10 timing points, disabled overhead is <0.2%. For `NonlinearSolver` with many iterations, each iteration adds ~30-60ns of overhead when disabled.

### Thread Safety

`TIMING_ENABLED` uses `Threads.Atomic{Bool}` for safe concurrent access. However, `TimerOutput` objects are not thread-safe — use separate `TimerOutput` instances when solving concurrently from multiple threads.

## Equilibrium Concept

This solver computes an **Open-Loop Mixed-Hierarchy Equilibrium (OLMHE)**, where:
- Hierarchy structure is defined by a directed acyclic graph following certain assumptions
- Leaders commit to their full trajectory upfront
- Followers observe leader trajectories and respond optimally
- Agents without leader-follower relationships have Nash relationships
- The hierarchy is enforced through KKT conditions with policy constraints

The same mathematical structure can represent feedback Stackelberg equilibrium under appropriate problem formulations. Details on feedback formulations are forthcoming.

## Solver Assumptions

The current implementation makes the following assumptions:

1. **Equality constraints only**: All constraints `g(z) = 0` are equality constraints. Inequality constraints are not yet supported.

2. **Decoupled constraints**: Each player's constraints depend only on their own decision variables: `gs[i](zs[i])`. Coupled constraints (e.g., shared dynamics `x_{t+1} = A*x_t + B1*u1 + B2*u2`) are not directly supported. However, problems with shared dynamics can often be reformulated by "baking" the dynamics into the cost function via trajectory rollout. See `test/olse/` for an example where the OLSE (Open-Loop Stackelberg Equilibrium) problem with shared dynamics is solved by embedding the dynamics in the cost functions rather than as explicit constraints.

3. **DAG hierarchy**: The leader-follower structure must be a directed acyclic graph (DAG). Cyclic dependencies and self-loops are not allowed.

4. **Hierarchy structure assumption**: We assume, for now, that any node has at most one parent. This may change in future versions of the software.

## Requirements

- Julia 1.9+
- See `Project.toml` for package dependencies

**PATHSolver on Apple Silicon (ARM64):** The `:path` backend for `QPSolver` depends on [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl), which only provides x86_64 binaries. It does not run natively on ARM64 (Apple Silicon). To use the `:path` backend on Apple Silicon, run Julia inside the provided Docker container, which uses Rosetta emulation via `platform: linux/amd64` (see [Docker Development Environment](#docker-development-environment) below). The default `:linear` backend works on all platforms without this limitation.

## Docker Development Environment

A Docker container is provided for a consistent development environment with all dependencies pre-installed.

### Prerequisites

- [Docker Desktop](https://docs.docker.com/get-docker/) (macOS; Linux users need to adjust SSH socket path in `docker-compose.yml`)
- SSH key loaded in agent: `ssh-add --apple-use-keychain ~/.ssh/id_rsa`
- Claude Code authenticated on host (via `claude` CLI login or `ANTHROPIC_API_KEY` env var)

### Quick Start

```bash
# Build and start development container
docker compose run --rm dev

# Inside container, you have access to:
julia --project=.     # Julia with all dependencies
git                   # Version control
gh                    # GitHub CLI
claude                # Claude Code CLI
bd                    # Beads work tracking CLI
```

### Running Tests

```bash
# Run all tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run fast tests only (~45s, 264 tests)
FAST_TESTS_ONLY=true julia --project=. -e 'using Pkg; Pkg.test()'
```

The test suite is split into two tiers:

- **Fast tier** (264 tests, ~45s): Unit tests, QP solver, input validation, type stability, OLSE QP
- **Slow tier** (192 tests, ~2min): Nonlinear solver convergence, KKT verification, integration tests, OLSE nonlinear

CI uses `/run-ci` for fast tests and `/run-ci-full` for the complete suite with coverage.

### Development Workflow

The container mounts your local source code at `/workspace`, so changes are reflected immediately:

```bash
# Start development session
docker compose run --rm dev

# Edit code on host, run in container
julia --project=. -e 'using MixedHierarchyGames'

# Use Claude Code for AI-assisted development
claude

# Or with auto-approve enabled (for sandboxed environments)
claude --allow-dangerously-skip-permissions

# Use GitHub CLI for PR management
gh pr create
```

### Rebuilding the Container

After changing `Project.toml` or `Manifest.toml`:

```bash
docker compose build --no-cache dev
```

## Troubleshooting

### PATHSolver fails on Apple Silicon (ARM64)

**Symptom:** `using PATHSolver` or `QPSolver(...; solver=:path)` fails with an error about missing binaries or unsupported platform on an Apple Silicon Mac.

**Cause:** PATHSolver.jl only distributes x86_64 (Intel) binaries. There are no native ARM64 builds.

**Solution:** Use the provided Docker development environment, which forces `platform: linux/amd64` in `docker-compose.yml` to run under Rosetta/QEMU emulation:

```bash
docker compose run --rm dev
# Inside the container:
julia --project=. -e 'using PATHSolver; @info "PATHSolver loaded"'
```

Alternatively, if you only need the QP solver, use the default `:linear` backend, which works natively on all platforms:

```julia
solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:linear)
```

## License

MIT License - see [LICENSE](LICENSE) for details.
