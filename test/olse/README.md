# OLSE (Open-Loop Stackelberg Equilibrium) Tests

This folder contains tests that validate our solvers against the closed-form OLSE solution from the SIOPT paper.

## Test Files

- `test_qp_solver.jl` - Tests QPSolver against analytical OLSE (PASSING)
- `test_nonlinear_solver.jl` - Tests NonlinearSolver against analytical OLSE (IN PROGRESS)

## Problem Description

The OLSE test problem is a 2-player LQ Stackelberg game with:
- Shared state dynamics: `x_{t+1} = A*x_t + B1*u1_t + B2*u2_t`
- Player 1 (leader) and Player 2 (follower)
- Quadratic costs on states and controls

## Known Results

| Solver | T=2 | T=3 | T=4 |
|--------|-----|-----|-----|
| QPSolver | PASS (1e-16) | PASS (1e-16) | PASS (1e-16) |
| NonlinearSolver | Issues | Issues | Issues |

The QPSolver matches OLSE at machine precision for all time horizons.
The NonlinearSolver has known issues being investigated (see Task #9).
