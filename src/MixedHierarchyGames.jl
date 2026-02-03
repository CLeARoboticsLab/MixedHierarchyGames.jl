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
    horizon,
    unflatten_trajectory

using Graphs: SimpleDiGraph, nv, vertices, edges, inneighbors, outneighbors, topological_sort_by_dfs, is_cyclic, has_self_loops
using Symbolics: Symbolics, @variables
using SymbolicTracingUtils: SymbolicTracingUtils
using ParametricMCPs: ParametricMCPs
using BlockArrays: BlockArrays, mortar, blocks
using LinearAlgebra: norm, I, SingularException, LAPACKException

# Types
include("types.jl")
export QPSolver, NonlinearSolver, HierarchyGame, QPProblem

# Problem setup
include("problem_setup.jl")
export setup_problem_variables, setup_problem_parameter_variables

# KKT construction
include("qp_kkt.jl")
export get_qp_kkt_conditions, strip_policy_constraints

include("nonlinear_kkt.jl")
export setup_approximate_kkt_solver, preoptimize_nonlinear_solver, compute_K_evals

# Solvers
include("solve.jl")
export solve, solve_raw, solve_with_path, solve_qp_linear, qp_game_linsolve, run_qp_solver, run_nonlinear_solver
export extract_trajectories, solution_to_joint_strategy

# Utilities
include("utils.jl")
export is_root, is_leaf, has_leader, get_roots, get_all_leaders, get_all_followers
export make_symbolic_variable, make_symbolic_vector, make_symbolic_matrix
# evaluate_kkt_residuals not exported - not yet implemented

end # module
