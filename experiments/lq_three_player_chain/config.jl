#=
    LQ Three Player Chain - Configuration

    Pure parameters for the experiment. No logic, just data.
=#

using Graphs: SimpleDiGraph, add_edge!

# Problem dimensions
const N = 3
const STATE_DIM = 2   # [x, y] position
const CONTROL_DIM = 2 # [vx, vy] velocity

# Default time parameters
const DEFAULT_T = 3
const DEFAULT_DT = 0.5

# Default initial states for each player
const DEFAULT_X0 = [
    [0.0, 2.0],  # P1
    [2.0, 4.0],  # P2 (leader)
    [6.0, 8.0],  # P3
]

# Cost weights
const CONTROL_WEIGHT = 0.05
const TERMINAL_WEIGHT = 0.5

# Solver parameters
const MAX_ITERS = 50
const TOLERANCE = 1e-8

"""
    build_hierarchy()

Build the Stackelberg hierarchy graph: P2 → P1, P2 → P3
P2 is the root leader.
"""
function build_hierarchy()
    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)  # P2 leads P1
    add_edge!(G, 2, 3)  # P2 leads P3
    return G
end
