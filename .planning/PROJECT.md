# MixedHierarchyGames.jl

## What This Is

A Julia package for solving mixed-hierarchy games — multi-player games where players have arbitrary leader-follower relationships represented as DAGs. Implements the solver from "Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations" (Khan et al., 2026). Aimed at robotics researchers and game theory practitioners who need to compute Stackelberg, Nash, or mixed equilibria for trajectory planning problems.

## Core Value

A correct, performant, and extensible solver for mixed-hierarchy games that the research community can build follow-up work on top of.

## Requirements

### Validated

- ✓ QP solver for linear-quadratic games with equality constraints — existing
- ✓ Nonlinear solver using iterative quasi-linear policy approximation — existing
- ✓ Arbitrary DAG-based hierarchy structures (pure Stackelberg, Nash, mixed) — existing
- ✓ TrajectoryGamesBase.jl interface compatibility — existing
- ✓ Symbolic KKT construction with compiled numerical evaluation — existing
- ✓ TimerOutputs instrumentation for performance profiling — existing
- ✓ Comprehensive test suite (450 tests) — existing

### Active

- [ ] Code cleanup: remove dead code, polish public API, clean up examples
- [ ] Documentation: README improvements, docstrings, Documenter.jl site
- [ ] Performance optimization: skip-K-in-line-search (1.63x verified), CSE, sparse solves
- [ ] Inequality constraint support
- [ ] Multiple solver variants with different KKT approximations (full vs approximate Hessian, linearization strategies)
- [ ] Multiple solution concepts (open-loop vs feedback Stackelberg, different equilibrium notions)
- [ ] Hardware/simulation code release (ROS2 integration)
- [ ] Public release: make repo public alongside paper

### Out of Scope

- Real-time MPC controller — solver is a building block, not a full control stack
- GUI or visualization tools — experiments use Plots.jl directly, no dedicated viz
- Non-Julia interfaces — Python/C++ bindings are not planned for v1

## Context

- **Paper:** Khan et al., "Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations," arXiv:2602.01568, 2026
- **Lab:** CLeAR Robotics Lab (UT Austin)
- **Repo is currently private** — will be made public alongside paper
- **Existing codebase:** ~1000 lines in `src/`, 450 tests, 3 experiment configurations (LQ chain, nonlinear lane change, pursuer-protector-VIP)
- **Legacy code in `examples/`:** Original monolithic solver (~1400 lines), being replaced by modular `src/` code
- **Colleague Tianyu** has hardware/simulation code ready for integration (ROS2 controller node)
- **Key dependency:** ParametricMCPs.jl and SymbolicTracingUtils.jl are unregistered packages from a specific source

## Constraints

- **Julia ecosystem:** Must work with Julia 1.9+ and integrate with TrajectoryGamesBase.jl
- **PATHSolver:** Requires license, doesn't work on ARM64 natively (uses Rosetta on Apple Silicon)
- **Memory:** Symbolic KKT construction for 4-player nonlinear games allocates ~159 GiB — needs 24+ GB RAM
- **Construction time:** 345s for 4-player lane change — acceptable for research, not for real-time
- **Unregistered deps:** ParametricMCPs and SymbolicTracingUtils are not in General registry, complicating installation for external users

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Symbolic KKT + compiled evaluation | One-time expensive construction, cheap repeated solves | ✓ Good — enables parameter-varying solves |
| TrajectoryGamesBase interface | Ecosystem compatibility for trajectory games | ✓ Good — but solver is more general than TGB |
| Newton iteration with Armijo line search | Standard approach for nonlinear KKT systems | — Pending — works but convergence can stall |
| Adopt GSD workflow | Context rot across sessions, stale state management | — Pending — just adopted |
| TDD mandatory | Catch regressions early in research code | ✓ Good — 450 tests, caught collision weight bug |

---
*Last updated: 2026-02-07 after initialization*
