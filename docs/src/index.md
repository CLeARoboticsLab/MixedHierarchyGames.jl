# MixedHierarchyGames.jl

A Julia package for solving mixed hierarchy (Stackelberg) trajectory games.

## Overview

MixedHierarchyGames.jl provides solvers for trajectory games with arbitrary leader-follower hierarchies defined by directed acyclic graphs (DAGs). The package implements the `TrajectoryGamesBase.jl` interface, though the solver is more general and can be used for general equality-constrained games.

Based on: H. Khan, D. H. Lee, J. Li, T. Qiu, C. Ellis, J. Milzman, W. Suttle, and D. Fridovich-Keil, "[Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations](https://arxiv.org/abs/2602.01568)," 2026.

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
T = 3
primal_dims = [(state_dim + control_dim) * T, (state_dim + control_dim) * T]

# Symbolic parameters
@variables θ1[1:state_dim] θ2[1:state_dim]
θs = Dict(1 => collect(θ1), 2 => collect(θ2))

# Constraints and costs
gs = [z -> z[1:state_dim] - collect(θ1), z -> z[1:state_dim] - collect(θ2)]
Js = Dict(
    1 => (z1, z2; θ=nothing) -> sum(z1.^2),
    2 => (z1, z2; θ=nothing) -> sum(z2.^2),
)

# Create solver and solve
solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)
strategy = solve(solver, Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0]))
```

See the [API Reference](@ref) for detailed documentation of all exported functions and types.
