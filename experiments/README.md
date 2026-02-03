# Experiments

This folder contains example experiments demonstrating the MixedHierarchyGames.jl package.
All experiments use the NonlinearSolver interface, which works for both LQ and nonlinear problems.

## Structure

```
experiments/
├── README.md                     # This file
├── common/                       # Shared utilities
│   ├── dynamics.jl              # Common dynamics models (unicycle, bicycle, etc.)
│   ├── collision_avoidance.jl   # Smooth collision cost functions
│   └── trajectory_utils.jl      # Trajectory generation utilities
├── lq_three_player_chain/       # LQ example with 3-player Stackelberg chain
│   └── run.jl                   # run_lq_three_player_chain()
├── nonlinear_lane_change/       # Nonlinear lane change scenario
│   └── run.jl                   # run_nonlinear_lane_change()
├── siopt_stackelberg/           # SIOPT paper Stackelberg example
│   └── run.jl                   # run_siopt_stackelberg()
├── pursuer_protector_vip/       # Multi-agent pursuit-protection game
│   └── run.jl                   # run_pursuer_protector_vip()
├── olse_paper_example/          # OLSE paper verification example
│   └── run.jl                   # run_olse_paper_example(), verify_olse_properties()
└── three_player_chain_validation.jl  # Validation against reference
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

## Experiment Descriptions

### LQ Three Player Chain
A linear-quadratic game with 3 players in a Stackelberg chain: P1 → P2 → P3.
Uses single integrator dynamics (2D position + 2D velocity control).
Players have quadratic objectives related to reaching positions and minimizing control.

**Hierarchy:** P1 leads P2, P2 leads P3
**Dynamics:** Single integrator (linear)
**Key function:** `run_lq_three_player_chain(; T=3, Δt=0.5, x0, verbose)`

### Nonlinear Lane Change
Four vehicles with unicycle dynamics on a highway. One vehicle (P3) merges from
an on-ramp following a quarter-circle trajectory, while others maintain lanes.
Tests collision avoidance and nonlinear dynamics.

**Hierarchy:** P1 → P2 → P4 (P3 is Nash)
**Dynamics:** Unicycle (nonlinear)
**Key function:** `run_nonlinear_lane_change(; T=14, Δt=0.4, R=6.0, x0, verbose)`

### SIOPT Stackelberg
Two-player LQ Stackelberg game from the SIOPT paper. Compares our solver's
solution against the closed-form OLSE (Open-Loop Stackelberg Equilibrium).

**Hierarchy:** P1 leads P2
**Dynamics:** Linear state evolution
**Key function:** `run_siopt_stackelberg(; T=2, x0, verbose)`

### Pursuer Protector VIP
Three-agent pursuit-protection game:
- P1 (Pursuer): Chases the VIP
- P2 (Protector): Shields VIP from pursuer
- P3 (VIP): Reaches goal while staying near protector

**Hierarchy:** P1 → P2 → P3
**Dynamics:** Single integrator
**Key function:** `run_pursuer_protector_vip(; T=20, Δt=0.1, x0, x_goal, verbose)`

### OLSE Paper Example
Rigorous verification of Open-Loop Stackelberg Equilibrium properties.
Computes both our solver's solution and the closed-form OLSE solution,
then verifies they match within numerical tolerance.

**Hierarchy:** P1 leads P2
**Key functions:**
- `run_olse_paper_example(; T=2, x0, verbose)` - Single test
- `verify_olse_properties(; num_tests=10, verbose)` - Multi-test verification

## Adding New Experiments

1. Create a new folder under `experiments/`
2. Add a `run.jl` script that demonstrates the experiment
3. Optionally add `test_<name>.jl` for regression tests
4. Update this README with the experiment description
