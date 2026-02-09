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

using Graphs: SimpleDiGraph, nv, vertices, edges, inneighbors, outneighbors, indegree, topological_sort_by_dfs, is_cyclic, has_self_loops, BFSIterator
using Symbolics: Symbolics, @variables
using SymbolicTracingUtils: SymbolicTracingUtils
using ParametricMCPs: ParametricMCPs
using BlockArrays: BlockArrays, mortar, blocks
using LinearAlgebra: norm, I, SingularException, LAPACKException
using SparseArrays: sparse, spzeros
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using SciMLBase: SciMLBase
using TimerOutputs: TimerOutput, @timeit

# Graph utilities (must come first - used by problem_setup)
include("utils.jl")
export is_root, is_leaf, has_leader, get_roots, get_all_leaders, get_all_followers
export evaluate_kkt_residuals, verify_kkt_solution

# Problem setup (symbolic variable creation)
include("problem_setup.jl")
export setup_problem_variables, setup_problem_parameter_variables
export make_symbolic_vector, make_symbolic_matrix, make_symbol
export default_backend, PLAYER_SYMBOLS, PAIR_SYMBOLS

# Types
include("types.jl")
export QPSolver, NonlinearSolver, HierarchyGame, HierarchyProblem, QPPrecomputed

# KKT construction
include("qp_kkt.jl")
export get_qp_kkt_conditions, strip_policy_constraints

# Line search methods
include("linesearch.jl")
export armijo_backtracking, geometric_reduction

include("nonlinear_kkt.jl")
export setup_approximate_kkt_solver, preoptimize_nonlinear_solver, compute_K_evals

# Solvers
include("solve.jl")
export solve, solve_raw, solve_with_path, solve_qp_linear, qp_game_linsolve, run_nonlinear_solver
export extract_trajectories, solution_to_joint_strategy

end # module
