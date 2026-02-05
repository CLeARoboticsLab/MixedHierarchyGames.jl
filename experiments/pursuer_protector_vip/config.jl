#=
    Pursuer-Protector-VIP - Configuration

    Pure parameters for the experiment. No logic, just data.
=#

using Graphs: SimpleDiGraph, add_edge!

# Problem dimensions
const N = 3
const STATE_DIM = 2   # [x, y] position
const CONTROL_DIM = 2 # [vx, vy] velocity

# Default time parameters
const DEFAULT_T = 20
const DEFAULT_DT = 0.1

# Default initial states
const DEFAULT_X0 = [
    [-5.0, 1.0],   # P1: Pursuer
    [-2.0, -2.5],  # P2: Protector (leader)
    [2.0, -4.0],   # P3: VIP
]

# Default goal position for VIP
const DEFAULT_GOAL = [0.0, 0.0]

# Cost weights
const PURSUER_CHASE_WEIGHT = 2.0
const PURSUER_AVOID_WEIGHT = -1.0  # negative = maximize distance
const PURSUER_CONTROL_WEIGHT = 1.25

const PROTECTOR_STAY_WEIGHT = 0.5
const PROTECTOR_PROTECT_WEIGHT = -1.0  # negative = maximize VIP-pursuer distance
const PROTECTOR_CONTROL_WEIGHT = 0.25

const VIP_GOAL_WEIGHT = 10.0
const VIP_STAY_WEIGHT = 1.0
const VIP_CONTROL_WEIGHT = 1.25

# Solver parameters
const MAX_ITERS = 50
const TOLERANCE = 1e-6

"""
    build_hierarchy()

Build the Stackelberg hierarchy graph: P2 → P1, P2 → P3
P2 (Protector) is the root leader.
"""
function build_hierarchy()
    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)  # P2 leads P1
    add_edge!(G, 2, 3)  # P2 leads P3
    return G
end
