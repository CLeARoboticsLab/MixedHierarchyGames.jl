# LQ Three Player Chain

A linear-quadratic game with 3 players demonstrating a Stackelberg hierarchy.

## Hierarchy

```
    P2 (Leader)
   /    \
  v      v
 P1      P3
```

- **P2** is the root leader
- **P1** and **P3** are followers of P2
- P1 and P3 play Nash with respect to each other

**Graph edges:** `P2 → P1`, `P2 → P3`

## Dynamics

Single integrator (linear): 2D position with 2D velocity control.

```
x_{t+1} = x_t + Δt * u_t
```

- State dimension: 2 (x, y position)
- Control dimension: 2 (vx, vy velocity)

## Objectives

- **P1**: Get close to P2's final position, minimize control effort
- **P2**: Drive P1 and P3 toward the origin, minimize control effort
- **P3**: Get close to P2's final position, minimize control effort

## Usage

```julia
include("experiments/lq_three_player_chain/run.jl")

# Default parameters
result = run_lq_three_player_chain(verbose=true)

# Custom parameters
result = run_lq_three_player_chain(
    T = 5,           # time horizon
    Δt = 0.5,        # time step
    x0 = [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]],  # initial positions
    verbose = true,
    plot = true,
    savepath = "output/lq_chain"
)
```

## Output

Returns a named tuple with:
- `trajectories`: Per-player state/control trajectories
- `costs`: Objective values for each player
- `status`: Solver status (`:solved` or `:failed`)
- `iterations`: Number of solver iterations
- `plt`: Plot object (if plotting enabled)
