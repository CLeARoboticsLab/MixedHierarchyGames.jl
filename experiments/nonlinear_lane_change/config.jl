#=
    Nonlinear Lane Change - Configuration

    Pure parameters for the experiment. No logic, just data.
=#

using Graphs: SimpleDiGraph, add_edge!

# Problem dimensions
const N = 4
const STATE_DIM = 4   # [x, y, ψ, v] - unicycle state
const CONTROL_DIM = 2 # [a, ω] - acceleration, yaw rate

# Default time parameters
const DEFAULT_T = 14
const DEFAULT_DT = 0.4

# Lane/turning radius
const DEFAULT_R = 6.0

# Target velocity
const TARGET_VELOCITY = 2.0

# Cost weights for lane-keeping vehicles (P1, P2, P4)
const CONTROL_WEIGHT_P1 = 10.0
const CONTROL_WEIGHT_P2 = 1.0
const CONTROL_WEIGHT_P4 = 1.0
const Y_DEVIATION_WEIGHT = 5.0
const HEADING_WEIGHT = 1.0
const VELOCITY_WEIGHT = 1.0

# Cost weights for merging vehicle (P3)
const TRACKING_WEIGHT_P3 = 10.0
const CONTROL_WEIGHT_P3 = 1.0
const Y_DEVIATION_WEIGHT_P3 = 5.0

# Solver parameters
const MAX_ITERS = 100
const TOLERANCE = 1e-6

"""
    default_initial_states(R)

Generate default initial states for the 4 vehicles.
"""
function default_initial_states(R)
    return [
        [-1.5R, R, 0.0, TARGET_VELOCITY],      # P1 (LEADER) - in lane
        [-2.0R, R, 0.0, TARGET_VELOCITY],      # P2 (FOLLOWER of P1) - in lane
        [-R, 0.0, π/2, 1.523],                 # P3 (LANE MERGER - Nash) - on ramp
        [-2.5R, R, 0.0, TARGET_VELOCITY],      # P4 (FOLLOWER of P2) - in lane
    ]
end

"""
    build_hierarchy()

Build the Stackelberg hierarchy graph: P1 → P2 → P4, P3 is Nash.
"""
function build_hierarchy()
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)  # P1 leads P2
    add_edge!(G, 2, 4)  # P2 leads P4
    # P3 is Nash (no edges)
    return G
end
