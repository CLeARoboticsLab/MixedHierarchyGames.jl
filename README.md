# MixedHierarchyGames.jl

A Julia package for solving mixed hierarchy games. This implementation uses the `TrajectoryGamesBase.jl` interface, though the solver is more general and can be used for general equality-constrained games.

<!-- TODO: Add paper reference -->
Based on: TBD

## Features

- **Flexible hierarchy structures**: Supports arbitrary DAG-based leader-follower relationships (pure Stackelberg, Nash, or mixed)
- **QP Solver**: For linear-quadratic games with equality constraints
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
using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: @variables

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

# Cost functions (each player minimizes own trajectory cost)
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
- **siopt_stackelberg**: 2-player LQ game from the SIOPT paper
- **pursuer_protector_vip**: 3-agent pursuit-protection game
- **olse_paper_example**: OLSE verification with closed-form comparison

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

### NonlinearSolver (planned)

For general nonlinear games using iterative quasi-linear policy approximation.

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

## Docker Development Environment

A Docker container is provided for a consistent development environment with all dependencies pre-installed.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- GitHub CLI authenticated on host (`gh auth login`)
- Claude Code API key (set `ANTHROPIC_API_KEY` environment variable or authenticate via `claude` CLI)

### Quick Start

```bash
# Build and start development container
docker compose run --rm dev

# Inside container, you have access to:
julia --project=.     # Julia with all dependencies
git                   # Version control
gh                    # GitHub CLI
claude                # Claude Code CLI
```

### Running Tests

```bash
# Run tests in container
docker compose run --rm test

# Or interactively
docker compose run --rm dev
julia --project=. -e 'using Pkg; Pkg.test()'
```

### Development Workflow

The container mounts your local source code at `/workspace`, so changes are reflected immediately:

```bash
# Start development session
docker compose run --rm dev

# Edit code on host, run in container
julia --project=. -e 'using MixedHierarchyGames'

# Use Claude Code for AI-assisted development
claude

# Use GitHub CLI for PR management
gh pr create
```

### Rebuilding the Container

After changing `Project.toml` or `Manifest.toml`:

```bash
docker compose build --no-cache dev
```

## Citation

<!-- TODO: Add proper citation -->
TBD

## License

MIT License - see [LICENSE](LICENSE) for details.
