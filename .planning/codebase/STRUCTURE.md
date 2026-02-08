# Codebase Structure

**Analysis Date:** 2026-02-07

## Directory Layout

```
StackelbergHierarchyGames.jl/
├── src/                        # Core package source code
│   ├── MixedHierarchyGames.jl  # Module root, exports API
│   ├── types.jl                # Solver type definitions and constructors
│   ├── problem_setup.jl        # Symbolic variable creation and naming
│   ├── utils.jl                # Graph topology and validation utilities
│   ├── qp_kkt.jl               # QP/LQ KKT condition construction
│   ├── nonlinear_kkt.jl        # Nonlinear game KKT and quasi-linear iteration
│   └── solve.jl                # Solver interface implementations
├── test/                       # Comprehensive test suite
│   ├── runtests.jl             # Test entry point
│   ├── testing_utils.jl        # Shared testing utilities
│   ├── test_*.jl               # Unit/integration tests (16 test files)
│   ├── olse/                   # OLSE baseline comparison tests
│   │   ├── olse_closed_form.jl
│   │   ├── test_qp_solver.jl
│   │   └── test_nonlinear_solver.jl
├── experiments/                # Reproducible experiments and benchmarks
│   ├── Project.toml            # Experiment-specific dependencies
│   ├── common/                 # Shared experiment utilities
│   ├── README.md               # Experiment documentation
│   ├── lq_three_player_chain/  # LQ game with 3-player Stackelberg chain
│   │   ├── config.jl           # Pure parameters
│   │   ├── run.jl              # Main entry point
│   │   └── support.jl          # Experiment-specific helpers
│   ├── convergence_analysis/   # Nonlinear solver convergence tests
│   ├── nonlinear_lane_change/  # Nonlinear traffic scenario
│   ├── pursuer_protector_vip/  # Multi-agent pursuit scenario
│   ├── benchmarks/             # Performance benchmarking
│   └── outputs/                # Generated results (gitignored)
├── docs/                       # Documentation generation
├── Project.toml                # Root package manifest (minimal dependencies)
├── Manifest.toml               # Dependency lock file
├── CLAUDE.md                   # Development workflow rules (TDD, PR requirements)
├── CHANGELOG.md                # Version history
├── RETROSPECTIVES.md           # PR retrospectives for process learning
├── README.md                   # Package overview and quick start
└── .planning/                  # GSD workflow planning documents
    └── codebase/               # Generated architecture/structure docs
```

## Directory Purposes

**src/**
- Purpose: Core package implementation
- Contains: Type definitions, solvers, utilities, KKT construction
- Key files: MixedHierarchyGames.jl (module root), solve.jl (main API)

**test/**
- Purpose: Comprehensive test coverage following TDD patterns
- Contains: Unit tests, integration tests, OLSE validation, type stability checks
- Key files: runtests.jl (test suite entry), testing_utils.jl (shared fixtures)

**experiments/**
- Purpose: Reproducible problem instances demonstrating solver usage
- Contains: Pure config files, run scripts, experiment-specific utilities
- Structure: Each experiment has config.jl (parameters), run.jl (main), support.jl (helpers)
- Key examples: lq_three_player_chain/ (simple 3-player Stackelberg)

**docs/**
- Purpose: Documentation (currently minimal)
- Contains: Generation scripts, API reference (auto-generated from docstrings)

**.planning/codebase/**
- Purpose: GSD workflow analysis documents (ARCHITECTURE.md, STRUCTURE.md, etc.)
- Auto-generated: Yes (created during `/gsd:map-codebase` command)

## Key File Locations

**Entry Points:**
- `src/MixedHierarchyGames.jl`: Module definition, public API exports
- `experiments/lq_three_player_chain/run.jl`: Example usage script
- `test/runtests.jl`: Test suite entry point

**Configuration:**
- `Project.toml`: Root package dependencies (minimal, core only)
- `experiments/Project.toml`: Experiment dependencies (Plots, visualization tools)
- `.JuliaFormatter.toml`: Code formatting configuration

**Core Logic:**
- `src/solve.jl`: Public solve() interface and solver implementations
- `src/types.jl`: Solver type definitions and construction logic
- `src/qp_kkt.jl`: KKT condition building for QP games
- `src/nonlinear_kkt.jl`: KKT and quasi-linear iteration for nonlinear games

**Testing:**
- `test/test_qp_solver.jl`: QPSolver unit tests
- `test/test_nonlinear_solver.jl`: NonlinearSolver unit tests
- `test/test_interface.jl`: TrajectoryGamesBase interface compliance tests
- `test/olse/`: Closed-form validation against OLSE solution

## Naming Conventions

**Files:**
- Source: `snake_case.jl` (e.g., `problem_setup.jl`, `qp_kkt.jl`)
- Tests: `test_<feature>.jl` (e.g., `test_qp_solver.jl`)
- Experiments: `<scenario>/<config|run|support>.jl` (e.g., `lq_three_player_chain/run.jl`)

**Directories:**
- Functional modules: `snake_case` (e.g., `experiments/common/`)
- Scenario-based: `snake_case` (e.g., `nonlinear_lane_change/`)

**Types/Structs:**
- Pascal case: `QPSolver`, `HierarchyGame`, `HierarchyProblem`
- Suffix conventions: `Precomputed` for cached components, `Solver` for solver types

**Variables:**
- Players: integer indices 1 to N
- Collections per player: `zs[i]`, `λs[i]`, `θs[i]` (subscript for player i)
- Pair variables: `μs[(i,j)]` (leader i, follower j)
- Dictionaries: Keys are player indices (1-based)

**Functions:**
- Private (internal): `_function_name()` (leading underscore)
- Public (exported): `function_name()` (no underscore)
- Constructors: Call type directly (e.g., `QPSolver(...)`)

**Symbols:**
- Player variables: `:z`, `:λ`, `:θ`, `:u`, `:x`, `:M`, `:N`, `:K`
- Pair variables: `:μ`
- Formed as: `Symbol("z^2")` for player 2 variable z, `Symbol("μ^(1-2)")` for leader-follower pair

## Where to Add New Code

**New Feature (within QP/Nonlinear solving):**
- Implementation: `src/<feature>.jl` (create new file if substantial)
- Tests: `test/test_<feature>.jl` (required: TDD mandatory per CLAUDE.md)
- Example: If adding new KKT transformation, create `src/kkt_transform.jl`

**New Solver Backend (e.g., different MCP solver):**
- Implementation: `src/solve.jl` (add new solver function)
- Integration: Update `solve()` dispatch in `src/solve.jl`
- Tests: Add test cases to `test/test_qp_solver.jl` or create `test/test_<backend>.jl`

**New Experiment/Scenario:**
- Structure: Create `experiments/<scenario_name>/` directory
- Files required:
  - `config.jl`: Pure parameters (no logic)
  - `run.jl`: Main script (delegates to support functions)
  - `support.jl`: Experiment-specific helpers (if needed)
  - `README.md`: Description and instructions
- Run command: `julia --project=experiments experiments/<scenario_name>/run.jl`

**Shared Experiment Utilities:**
- Location: `experiments/common/` (e.g., common problem builders)
- No duplication: Reuse across experiments rather than copy-pasting

**Utilities and Helpers:**
- Graph utilities: `src/utils.jl`
- Symbolic utilities: `src/problem_setup.jl`
- Testing helpers: `test/testing_utils.jl`

## Special Directories

**debug/**
- Purpose: Temporary debugging scripts and investigation files
- Generated: No (created manually by developers)
- Committed: No (listed in .gitignore)
- Usage: For explorations that don't belong in main codebase

**.julia/, .julia_local/, .julia_depot/**
- Purpose: Julia environment and package caches
- Generated: Yes (auto-created by Julia)
- Committed: No (listed in .gitignore)

**data/**
- Purpose: Problem data files, reference implementations
- Committed: Yes (tracked in git)

**outputs/** (under experiments/)
- Purpose: Generated results, benchmark outputs, experiment artifacts
- Generated: Yes (created during experiment runs)
- Committed: No (typically gitignored)

## Module Organization

**Public API** (exported from `src/MixedHierarchyGames.jl`):

Solvers:
- `QPSolver`: QP/LQ hierarchy game solver
- `NonlinearSolver`: General nonlinear hierarchy game solver

Types:
- `HierarchyGame`: Trajectory game with Stackelberg structure
- `HierarchyProblem`: Low-level problem specification

Setup:
- `setup_problem_variables()`: Create symbolic variables
- `setup_problem_parameter_variables()`: Create parameter variables
- `make_symbolic_vector()`, `make_symbolic_matrix()`: Symbol utilities

Graph utilities:
- `is_root()`, `is_leaf()`, `has_leader()`: Node queries
- `get_roots()`, `get_all_leaders()`, `get_all_followers()`: Traversal
- `verify_kkt_solution()`: Solution validation

Solving:
- `solve()`: Main solver interface
- `solve_raw()`: Raw solution vector (debugging)
- `TrajectoryGamesBase.solve_trajectory_game!()`: Standard interface

KKT (internal):
- `get_qp_kkt_conditions()`: Build QP KKT system
- `strip_policy_constraints()`: Extract solvable constraints

**Stability Note:** Private functions (starting with `_`) subject to change; public functions maintain compatibility.

---

*Structure analysis: 2026-02-07*
