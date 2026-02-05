# Nonlinear Lane Change

Four vehicles with unicycle dynamics on a highway. One vehicle merges from an on-ramp while others maintain lanes with collision avoidance.

## Scenario

- **P1**: Lead vehicle in right lane (leader)
- **P2**: Following vehicle in right lane
- **P3**: Merging vehicle from on-ramp (Nash with others)
- **P4**: Vehicle in left lane

## Hierarchy

```
P1 (Leader)
 |
 v
P2
 |
 v
P4

P3 (Nash - no hierarchy edges)
```

- **P1** leads P2
- **P2** leads P4
- **P3** plays Nash (not in hierarchy)

**Graph edges:** `P1 → P2`, `P2 → P4`

## Dynamics

Unicycle (nonlinear):
```
x_{t+1} = x_t + Δt * v * cos(θ)
y_{t+1} = y_t + Δt * v * sin(θ)
θ_{t+1} = θ_t + Δt * ω
v_{t+1} = v_t + Δt * a
```

- State dimension: 4 (x, y, θ, v)
- Control dimension: 2 (ω angular velocity, a acceleration)

## Objectives

All players:
- Maintain target velocity
- Stay in lane (or merge for P3)
- Avoid collisions with other vehicles
- Minimize control effort

P3 specifically follows a quarter-circle merge trajectory.

## Usage

```julia
include("experiments/nonlinear_lane_change/run.jl")

# Default parameters (T=14 may require significant memory)
result = run_nonlinear_lane_change(verbose=true)

# Shorter horizon for testing
result = run_nonlinear_lane_change(
    T = 6,           # time horizon (shorter = faster)
    Δt = 0.4,        # time step
    R = 6.0,         # lane width / merge radius
    verbose = true,
    plot = true,
    savepath = "output/lane_change"
)
```

## Output

Returns a named tuple with:
- `trajectories`: Per-player state/control trajectories
- `costs`: Objective values for each player
- `status`: Solver status
- `iterations`: Number of solver iterations
- `R`, `T`, `Δt`: Parameters used
- `plt_traj`, `plt_dist`: Plot objects (if plotting enabled)

## Notes

- T=14 (default) can require significant memory due to symbolic computation
- Use T=6 for quick testing
- The nonlinear solver typically needs 50-70 iterations to converge
