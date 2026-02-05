# Changelog

All notable changes to MixedHierarchyGames.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `QPSolver` for linear-quadratic hierarchy games with equality constraints
- `NonlinearSolver` for nonlinear hierarchy games using iterative KKT solving
  - Quasi-linear policy approximation with Newton-like iterations
  - Armijo backtracking line search for globalization
  - Configurable: `max_iters`, `tol`, `verbose`, `use_armijo`
  - Supports warm-starting via `initial_guess` parameter
- `HierarchyGame` type wrapping `TrajectoryGame` with hierarchy graph
- `solve()` function returning `JointStrategy` with extracted trajectories
- `solve_raw()` function for debugging and analysis
- TrajectoryGamesBase.jl integration via `solve_trajectory_game!`
- Input validation with descriptive error messages
- Support for `:linear` (direct solve) and `:path` (PATH solver) backends
- Comprehensive test suite with OLSE validation
- Single-player (N=1) edge case tests
- Type stability tests

### Changed
- Renamed package from FeedbackStackelbergGames to MixedHierarchyGames
- Restructured codebase for cleaner API
- Renamed `z_sol` to `sol` in solver return types for clarity
- Standardized terminology: LQ → QP for quadratic programming problems
- Improved type annotations in internal data structures for better type stability

### Fixed
- Numerical stability: replaced `inv()` with backslash operator for linear solves
- Armijo line search now returns 0.0 on failure instead of last attempted step

### Deprecated
- N/A

### Removed
- N/A

### Security
- N/A
