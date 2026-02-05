# Pursuer Protector VIP

A three-agent pursuit-protection game demonstrating adversarial dynamics.

## Scenario

- **P1 (Pursuer)**: Tries to catch the VIP
- **P2 (Protector)**: Shields VIP from the pursuer (leader)
- **P3 (VIP)**: Reaches goal while staying near protector

## Hierarchy

```
    P2 (Protector/Leader)
   /    \
  v      v
 P1      P3
(Pursuer) (VIP)
```

- **P2** (Protector) is the root leader
- **P1** (Pursuer) and **P3** (VIP) are followers of P2
- P1 and P3 play Nash with respect to each other

**Graph edges:** `P2 → P1`, `P2 → P3`

## Dynamics

Single integrator (linear): 2D position with 2D velocity control.

## Objectives

- **P1 (Pursuer)**: Minimize distance to VIP, minimize control
- **P2 (Protector)**: Stay between pursuer and VIP, keep pursuer away from VIP
- **P3 (VIP)**: Reach goal position, stay near protector, minimize control

## Usage

```julia
include("experiments/pursuer_protector_vip/run.jl")

# Default parameters
result = run_pursuer_protector_vip(verbose=true)

# Custom parameters
result = run_pursuer_protector_vip(
    T = 20,          # time horizon
    Δt = 0.1,        # time step
    x0 = [[-5.0, 1.0], [-2.0, -2.5], [2.0, -1.0]],  # [pursuer, protector, VIP]
    x_goal = [10.0, -10.0],  # VIP goal position
    verbose = true,
    plot = true,
    savepath = "output/pursuit"
)
```

## Output

Returns a named tuple with:
- `trajectories`: Per-player state/control trajectories
- `costs`: Objective values for each player
- `status`: Solver status
- `iterations`: Number of solver iterations
- `x_goal`: The goal position used
- `plt`: Plot object (if plotting enabled)
