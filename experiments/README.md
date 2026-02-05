# Experiments

This folder contains example experiments demonstrating the MixedHierarchyGames.jl package.
All experiments use the NonlinearSolver interface, which works for both LQ and nonlinear problems.

## Structure

```
experiments/
├── README.md                     # This file (overview)
├── common/                       # Shared utilities
│   ├── dynamics.jl              # Common dynamics models
│   ├── collision_avoidance.jl   # Smooth collision cost functions
│   ├── trajectory_utils.jl      # Trajectory generation utilities
│   └── plotting.jl              # Visualization utilities
├── lq_three_player_chain/       # LQ 3-player game
│   ├── README.md                # Detailed documentation
│   ├── config.jl, run.jl, support.jl
├── nonlinear_lane_change/       # Highway merge scenario
│   ├── README.md
│   ├── config.jl, run.jl, support.jl
├── pursuer_protector_vip/       # Pursuit-protection game
│   ├── README.md
│   ├── config.jl, run.jl, support.jl
├── convergence_analysis/        # Solver robustness testing
│   ├── README.md
│   ├── config.jl, run.jl, support.jl
└── three_player_chain_validation.jl  # Validation script
```

## Common Utilities

### dynamics.jl
- `unicycle_dynamics(z, t; Δt)` - Kinematic unicycle model
- `bicycle_dynamics(z, t; Δt, L)` - Kinematic bicycle model
- `double_integrator_2d(z, t; Δt)` - 2D double integrator
- `single_integrator_2d(z, t; Δt)` - 2D single integrator

### collision_avoidance.jl
- `smooth_collision(xsA, xsB; d_safe, α, w)` - Pairwise collision cost
- `smooth_collision_all(xs_all...; d_safe, α, w)` - All-pairs collision cost

### trajectory_utils.jl
- `make_unicycle_traj(T, Δt; R, split, x0)` - Generate unicycle reference trajectory
- `make_straight_traj(T, Δt; x0)` - Generate straight-line trajectory
- `flatten_trajectory(xs, us)` - Flatten trajectories to vector

## Running Experiments

Each experiment can be run independently:

```julia
# From project root
julia --project experiments/lq_three_player_chain/run.jl
```

Or interactively:
```julia
using MixedHierarchyGames
include("experiments/lq_three_player_chain/run.jl")
```

## Experiments Overview

| Experiment | Hierarchy | Dynamics | Description |
|------------|-----------|----------|-------------|
| [lq_three_player_chain](lq_three_player_chain/README.md) | P2 → P1, P2 → P3 | Single integrator | 3-player LQ game, P2 is root leader |
| [nonlinear_lane_change](nonlinear_lane_change/README.md) | P1 → P2 → P4 (P3 Nash) | Unicycle | 4-vehicle highway merge scenario |
| [pursuer_protector_vip](pursuer_protector_vip/README.md) | P2 → P1, P2 → P3 | Single integrator | 3-agent pursuit-protection game |
| [convergence_analysis](convergence_analysis/README.md) | (uses lane change) | Unicycle | Multi-run solver robustness testing |

See individual experiment README files for detailed documentation.

## Adding New Experiments

### Required Structure

Each experiment should follow this structure:

```
experiments/<name>/
├── README.md          # Detailed documentation (required)
├── config.jl          # Parameters: x0, G, N, T, Δt, costs, goals, etc.
├── run.jl             # Main entry point (uses config + support)
└── support.jl         # Experiment-specific helpers (if needed)
```

### Guidelines

1. **config.jl** - Pure data/parameters, no logic:
   ```julia
   # config.jl
   const N = 3
   const T = 10
   const Δt = 0.1
   const x0 = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
   const goals = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
   # ... costs, hierarchy graph, etc.
   ```

2. **run.jl** - Concise entry point that delegates to support functions:
   ```julia
   include("config.jl")
   include("support.jl")

   function run_experiment(; verbose=false)
       # Build problem from config
       # Solve
       # Return results
   end
   ```

3. **support.jl** - Experiment-specific helpers (cost functions, custom dynamics, etc.)

4. **Avoid duplicate code**:
   - Shared dynamics → `experiments/common/dynamics.jl`
   - Shared collision avoidance → `experiments/common/collision_avoidance.jl`
   - Shared plotting → `experiments/common/plotting.jl`
   - Generally useful code → consider adding to `src/`

5. **Update this README** - Add entry to the Experiments Overview table

### Shared Code Organization

```
experiments/
├── common/                    # Shared utilities
│   ├── dynamics.jl           # Single integrator, unicycle, etc.
│   ├── collision_avoidance.jl
│   ├── trajectory_utils.jl
│   └── plotting.jl           # Plotting utilities
├── Project.toml              # Dev dependencies (Plots, JLD2, etc.)
└── <experiment>/             # Individual experiments
```
