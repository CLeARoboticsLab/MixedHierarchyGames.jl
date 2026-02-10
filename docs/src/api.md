# API Reference

## Module

```@docs
MixedHierarchyGames
```

## Types

```@docs
QPSolver
NonlinearSolver
HierarchyGame
HierarchyProblem
QPPrecomputed
```

## Solvers

```@docs
solve
solve_raw
solve_with_path
solve_qp_linear
qp_game_linsolve
run_nonlinear_solver
extract_trajectories
solution_to_joint_strategy
```

## KKT Construction

```@docs
get_qp_kkt_conditions
strip_policy_constraints
setup_approximate_kkt_solver
preoptimize_nonlinear_solver
compute_K_evals
```

## Problem Setup

```@docs
setup_problem_variables
setup_problem_parameter_variables
make_symbolic_vector
make_symbolic_matrix
make_symbol
default_backend
```

## Graph Utilities

```@docs
is_root
is_leaf
has_leader
get_roots
get_all_leaders
get_all_followers
evaluate_kkt_residuals
verify_kkt_solution
```

## Internal Functions

```@autodocs
Modules = [MixedHierarchyGames]
Public = false
```
