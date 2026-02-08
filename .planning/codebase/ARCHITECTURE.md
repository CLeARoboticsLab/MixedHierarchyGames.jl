# Architecture

**Analysis Date:** 2026-02-07

## Pattern Overview

**Overall:** Hierarchical game-solving architecture using symbolic computation with KKT-based equilibrium calculation.

**Key Characteristics:**
- Hierarchical leader-follower structure represented as directed acyclic graphs (DAGs)
- Symbolic computation first: KKT conditions constructed symbolically at solver creation
- Dual solver approach: QPSolver for linear-quadratic games, NonlinearSolver for general nonlinear problems
- TrajectoryGamesBase integration for trajectory game interface compatibility
- Separation of symbolic construction (expensive, done once) from numerical solving (cheap, done repeatedly)

## Layers

**Graph/Topology Layer:**
- Purpose: Represent and query hierarchical relationships between players
- Location: `src/utils.jl`
- Contains: Graph traversal, topology query functions (roots, leaves, leaders, followers)
- Depends on: Graphs.jl (SimpleDiGraph)
- Used by: Problem setup, KKT construction, solver creation

**Problem Setup Layer:**
- Purpose: Create symbolic variables for players, parameters, and dual variables
- Location: `src/problem_setup.jl`
- Contains: Symbol naming conventions, symbolic vector/matrix creation, parameter variable setup
- Depends on: Symbolics.jl, SymbolicTracingUtils.jl
- Used by: Both solver types during construction

**Type/Specification Layer:**
- Purpose: Define core data structures for problems and solvers
- Location: `src/types.jl`
- Contains: HierarchyGame, HierarchyProblem, QPSolver, NonlinearSolver, QPPrecomputed type definitions
- Depends on: TrajectoryGamesBase (for TrajectoryGame and SimpleDiGraph)
- Used by: All higher layers

**KKT Construction Layer:**
- Purpose: Transform game specifications into KKT conditions (equality constraints to solve)
- Location: `src/qp_kkt.jl` (QP) and `src/nonlinear_kkt.jl` (nonlinear)
- Contains: KKT condition building, policy constraint handling, M/N matrix computation
- Depends on: Problem setup, graph utilities, Symbolics.jl
- Used by: Solver construction and nonlinear solving

**Solver Interface Layer:**
- Purpose: Implement solve() interface for both solver types
- Location: `src/solve.jl`
- Contains: solve(), solve_raw(), TrajectoryGamesBase.solve_trajectory_game!() implementations
- Depends on: KKT layer, type layer, LinearSolve.jl, PATHSolver.jl
- Used by: External code calling solvers

## Data Flow

**QP Solver Workflow:**

1. **Construction Phase** (done once):
   - User provides: hierarchy graph, cost functions Js, constraint functions gs
   - setup_problem_variables() creates symbolic z, λ, μ, w, y variables
   - get_qp_kkt_conditions() constructs KKT conditions symbolically
   - strip_policy_constraints() extracts equality constraints for solving
   - ParametricMCP built from KKT conditions
   - Linearity verified (Jacobian constant check)

2. **Solving Phase** (done repeatedly):
   - User provides: parameter_values (e.g., initial states)
   - solve_qp_linear() or solve_with_path() solves the KKT system
   - Solution vector extracted and converted to JointStrategy (trajectories)

**Nonlinear Solver Workflow:**

1. **Construction Phase**:
   - User provides same inputs as QP (hierarchy graph, Js, gs)
   - preoptimize_nonlinear_solver() constructs symbolic KKT conditions
   - Instead of K = M \ N symbolically, builds compiled M and N evaluation functions
   - Avoids expression explosion from symbolic K computation

2. **Solving Phase**:
   - run_nonlinear_solver() iterates quasi-linear policy approximation
   - Each iteration: evaluate M(z_current), N(z_current), solve K = M \ N numerically
   - Uses Armijo line search for convergence
   - Converges to local equilibrium

**State Management:**

- **Precomputed state**: Cached in solver instances (symbolic variables, KKT conditions, compiled functions)
- **Iteration state**: During nonlinear solving, tracks z vectors, K policies, step sizes
- **Solution state**: Final z vector from KKT solution, converted to trajectory format

## Key Abstractions

**HierarchyGame:**
- Purpose: Represent a trajectory game with explicit Stackelberg structure
- Examples: `experiments/lq_three_player_chain/run.jl`
- Pattern: Contains TrajectoryGame + SimpleDiGraph hierarchy specification

**HierarchyProblem:**
- Purpose: Low-level specification of a hierarchy game for solver construction
- Examples: Created internally by QPSolver and NonlinearSolver constructors
- Pattern: Stores problem components (graph, costs, constraints, dimensions) used during solving

**QPPrecomputed:**
- Purpose: Cache precomputed symbolic components for efficient repeated solves
- Examples: Used in QPSolver.precomputed
- Pattern: Contains symbolic variables, KKT result, ParametricMCP

**Symbolic Variable Structure:**
- Players represented as indices 1 to N
- Per-player variables: zs[i] (decisions), λs[i] (multipliers), θs[i] (parameters)
- Pair variables: μs[(i,j)] (policy constraint multipliers for leader i, follower j)
- Information variables: ws[i] (follower outputs), ys[i] (leader inputs for policy)

## Entry Points

**QPSolver Construction:**
- Location: `src/types.jl` lines 234-278 and 287-299
- Triggers: User code instantiates QPSolver(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim)
- Responsibilities: Validate inputs, precompute symbolic KKT, build ParametricMCP, verify linearity

**QPSolver.solve():**
- Location: `src/solve.jl` lines 47-83
- Triggers: User calls solve(qp_solver, parameter_values)
- Responsibilities: Validate parameters, select linear or PATH backend, solve KKT system, extract trajectories

**NonlinearSolver Construction:**
- Location: `src/types.jl` lines 339-374
- Triggers: User instantiates NonlinearSolver(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim)
- Responsibilities: Validate inputs, precompute M/N functions, store solver options

**NonlinearSolver.run_nonlinear_solver():**
- Location: `src/nonlinear_kkt.jl`
- Triggers: Internally called from solve() after construction
- Responsibilities: Iterate quasi-linear approximation, manage line search, converge to equilibrium

**TrajectoryGamesBase Integration:**
- Location: `src/solve.jl` lines 136-200+ (solve_trajectory_game! implementations)
- Triggers: External code using TrajectoryGamesBase.solve_trajectory_game! interface
- Responsibilities: Convert HierarchyGame and initial states to solver.solve() call

## Error Handling

**Strategy:** Validation at construction time, explicit error messages at solve time.

**Patterns:**

- **Input Validation** (`src/types.jl` lines 92-143): Check graph structure (DAG, single-parent), dimension consistency
- **Linearity Verification** (`src/types.jl` lines 188-215): Warning if KKT Jacobian varies with z (QP assumption violated)
- **Solver Failure Handling** (`src/solve.jl` lines 78-80): Error on failed linear/PATH solves with diagnostic message
- **Parameter Validation** (`src/solve.jl` line 62): Check all required parameters provided before solving

## Cross-Cutting Concerns

**Logging:** TimerOutput used for performance instrumentation (construction timing, solve timing)
- Usage: @timeit macro in QPSolver constructor and solve functions
- Example: `src/types.jl` line 245: `@timeit to "QPSolver construction"`

**Validation:** Performed at:
- Solver construction: Graph structure, dimensions, function signatures
- Solve time: Parameter values match problem specification

**Symbolic Computation:** Handled uniformly via SymbolicTracingUtils and Symbolics backends
- Symbolic types preserved through construction (minor type instability acceptable at construction time)
- Compiled functions generated for numerical evaluation (stable during solving)

---

*Architecture analysis: 2026-02-07*
