"""
    MixedHierarchyGames

A Julia package for solving mixed hierarchy (Stackelberg) trajectory games.

Provides two solvers:
- `QPSolver` - For quadratic programming problems with linear dynamics
- `NonlinearSolver` - For general nonlinear problems

Both solvers implement the `TrajectoryGamesBase.solve_trajectory_game!` interface.
"""
module MixedHierarchyGames

using TrajectoryGamesBase:
    TrajectoryGamesBase,
    TrajectoryGame,
    OpenLoopStrategy,
    JointStrategy,
    num_players,
    state_dim,
    control_dim,
    horizon

using Graphs: SimpleDiGraph, vertices, edges, inneighbors, outneighbors, topological_sort_by_dfs
using Symbolics: Symbolics, @variables
using SymbolicTracingUtils: SymbolicTracingUtils
using ParametricMCPs: ParametricMCPs
using BlockArrays: BlockArrays, mortar, blocks
using LinearAlgebra: norm

# Types
include("types.jl")
export QPSolver, NonlinearSolver, HierarchyGame

# Problem setup
include("problem_setup.jl")
export setup_problem_variables, setup_problem_parameter_variables

# KKT construction
include("qp_kkt.jl")
export get_lq_kkt_conditions, strip_policy_constraints

include("nonlinear_kkt.jl")
export setup_approximate_kkt_solver, preoptimize_nonlq_solver, compute_K_evals

# Solvers
include("solve.jl")
export solve_with_path, run_lq_solver, run_nonlq_solver

# Utilities
include("utils.jl")
export is_root, is_leaf, get_roots, get_all_leaders, get_all_followers
export make_symbolic_variable
export evaluate_kkt_residuals

end # module
