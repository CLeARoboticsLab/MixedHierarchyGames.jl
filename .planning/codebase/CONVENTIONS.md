# Coding Conventions

**Analysis Date:** 2026-02-07

## Naming Patterns

**Files:**
- Lowercase with underscores: `problem_setup.jl`, `nonlinear_kkt.jl`, `qp_kkt.jl`
- Module entry: `MixedHierarchyGames.jl`
- Tests: `test_<module>.jl` format in `/test` directory

**Functions:**
- Lowercase with underscores for public functions: `setup_problem_variables`, `get_qp_kkt_conditions`, `solve_with_path`
- Private functions prefixed with underscore: `_validate_solver_inputs`, `_verify_linear_system`, `_extract_joint_strategy`, `_build_extractor`
- Single-letter or short abbreviations for mathematical objects: `G` (graph), `J` (cost), `θ` (parameters), `z` (decision variables), `λ` (dual variables), `μ` (policy multipliers)
- Boolean predicates use `is_` or `has_` prefix: `is_root`, `is_leaf`, `has_leader`

**Variables:**
- Single letters for mathematical variables: `G` (SimpleDiGraph), `N` (number of players), `T` (time horizon)
- Greek symbols allowed for mathematical clarity: `θs` (parameter dict), `πs` (KKT conditions), `λs` (Lagrange multipliers), `μs` (policy multipliers), `Ks` (policy matrices)
- Dictionary suffixes with `s`: `zs`, `θs`, `λs`, `μs`, `Js`, `gs` (Dict collections)
- CamelCase for types: `HierarchyGame`, `HierarchyProblem`, `QPPrecomputed`, `QPSolver`, `NonlinearSolver`

**Types:**
- PascalCase for struct types: `HierarchyGame`, `HierarchyProblem`, `QPSolver`, `NonlinearSolver`, `QPPrecomputed`
- Type parameters with single uppercase letters: `TV`, `TG`, `TK`, `TP`, `TM` (generic parameters in struct definitions)
- AbstractVector/AbstractMatrix for function parameters to support multiple implementations

## Code Style

**Formatting:**
- Four-space indentation (Julia standard)
- No apparent automated formatter in use (no .prettierrc, eslintrc, or biome.json)
- Long function signatures broken across lines: arguments aligned, keywords on separate lines
- Block comments use `#=` ... `=#` for multi-line, `#` for single-line

**Linting:**
- No linter configuration found (no .eslintrc or similar)
- Convention enforcement relies on peer review and manual adherence

## Import Organization

**Order:**
1. Standard library imports (Test, LinearAlgebra, SparseArrays)
2. External packages (Graphs, Symbolics, ParametricMCPs, BlockArrays)
3. Project internal imports (via `using MixedHierarchyGames`)

**Pattern in `MixedHierarchyGames.jl`:**
```julia
using TrajectoryGamesBase: TrajectoryGamesBase, TrajectoryGame, ...
using Graphs: SimpleDiGraph, nv, vertices, ...
using Symbolics: Symbolics, @variables
using ParametricMCPs: ParametricMCPs
using LinearAlgebra: norm, I, ...
```

**File inclusion order (module initialization):**
1. Graph utilities (`utils.jl`)
2. Problem setup (`problem_setup.jl`)
3. Type definitions (`types.jl`)
4. KKT construction (`qp_kkt.jl`, `nonlinear_kkt.jl`)
5. Solver implementations (`solve.jl`)

See `src/MixedHierarchyGames.jl` lines 36-61 for actual dependency order.

## Error Handling

**Patterns:**
- Explicit validation at construction time in solver constructors
- Use `throw(ArgumentError(...))` for invalid inputs with descriptive messages
- Examples from `src/types.jl` `_validate_solver_inputs` function (lines 92-143):
  - Graph structure validation: self-loops, cycles, single-parent constraint
  - Dimension consistency checks
  - Key existence verification in dictionaries
  - Error messages include context about the specific failure

**Assertion style:**
- Use `@assert` for developer-facing internal checks
- Use `throw()` for user-facing input validation
- Warning via `@warn` for non-fatal issues (see `_verify_linear_system`, line 191)

## Logging

**Framework:** No external logging package. Uses Julia's standard facilities.

**Patterns:**
- `println()` for informational output in debug/verbose mode
- `@warn` macro for warnings
- Verbose output controlled by `verbose::Bool` keyword arguments
- Formatted output in `verify_kkt_solution` uses print statements with separators

**Example from `src/utils.jl` lines 162-172:**
```julia
if verbose
    println("\n" * "="^20 * " KKT Residuals " * "="^20)
    for (idx, val) in enumerate(π_eval)
        if abs(val) >= tol
            println("  π[$idx] = $val")
        end
    end
    println("All KKT conditions satisfied (< $tol)? ", all(abs.(π_eval) .< tol))
    println("‖π‖₂ = ", residual_norm)
    println("="^55)
end
```

## Comments

**When to Comment:**
- Function documentation: Always use docstrings (triple-quoted strings)
- Block-level comments: Use `#=` blocks to section logic
- Inline comments: Sparingly, for non-obvious mathematical operations
- Code organization: Use `#=` section headers to divide files into logical parts

**Docstring Pattern:**
Each public function has a comprehensive docstring including:
- One-line summary
- Detailed description if needed
- `# Arguments` section with type and description
- `# Keyword Arguments` section if applicable
- `# Returns` section with types and description
- `# Example` code block (if complex behavior)
- Cross-references to related functions

See `src/utils.jl` lines 82-111 for `evaluate_kkt_residuals` example.

**Section Headers:**
```julia
#=
    Graph topology utilities
=#

#=
    Solution validation utilities
=#

#=
    Trajectory utilities
=#
```

## Function Design

**Size:**
- Most functions 20-60 lines
- Complex functions like `get_qp_kkt_conditions` exceed 100 lines but remain focused on single concern
- Private helper functions extract repeated logic

**Parameters:**
- Maximum 5-7 positional parameters before requiring keyword args
- Dictionary parameters for variable-length player collections: `Js::Dict`, `θs::Dict`
- Vector parameters for ordered player data: `gs::Vector`, `primal_dims::Vector{Int}`
- Keyword arguments for options: `verbose::Bool=false`, `tol::Float64=1e-6`, `solver::Symbol=:linear`

**Return Values:**
- Single value or tuple for most functions
- Named tuples for multiple related returns: `(; sol, status, info)`
- Dict returns for player-indexed data: `Dict{Int, ...}`
- Use `@timeit` macro from TimerOutputs for performance-critical paths

## Module Design

**Exports:**
From `src/MixedHierarchyGames.jl` (lines 38-61):
- Graph utilities: `is_root`, `is_leaf`, `has_leader`, `get_roots`, `get_all_leaders`, `get_all_followers`
- KKT verification: `evaluate_kkt_residuals`, `verify_kkt_solution`
- Problem setup: `setup_problem_variables`, `setup_problem_parameter_variables`, `make_symbolic_vector`, `make_symbol`, `default_backend`
- Solvers: `QPSolver`, `NonlinearSolver`, `HierarchyGame`, `HierarchyProblem`, `QPPrecomputed`
- Solving: `solve`, `solve_raw`, `solve_with_path`, `solve_qp_linear`, `qp_game_linsolve`, `run_nonlinear_solver`
- Utilities: `extract_trajectories`, `solution_to_joint_strategy`

**Barrel Files:**
- No barrel files used; direct includes in main module
- Each source file (`problem_setup.jl`, `types.jl`, etc.) exports specific functions

**Type Hierarchy:**
- Structs are parameterized for flexibility: `struct QPSolver{TP<:HierarchyProblem, TC<:QPPrecomputed}`
- Allows type-stable code and dispatch on inner types
- See `src/types.jl` for all struct definitions (lines 15-393)

---

*Convention analysis: 2026-02-07*
