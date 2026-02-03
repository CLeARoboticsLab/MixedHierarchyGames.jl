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

# Symbolic parameters (initial states)
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

```julia
solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:linear)
```

### NonlinearSolver (planned)

For general nonlinear games using iterative quasi-linear policy approximation.

## Requirements

- Julia 1.9+
- See `Project.toml` for package dependencies

## Citation

<!-- TODO: Add proper citation -->
TBD

## License

MIT License - see [LICENSE](LICENSE) for details.
