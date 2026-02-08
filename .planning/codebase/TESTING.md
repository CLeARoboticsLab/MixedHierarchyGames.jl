# Testing Patterns

**Analysis Date:** 2026-02-07

## Test Framework

**Runner:**
- Julia built-in `Test` module
- Config: `/test/Project.toml` defines test dependencies
- Entry point: `test/runtests.jl`

**Assertion Library:**
- Julia's `Test` module: `@test`, `@testset`, `@test_broken`, `@test_throws`

**Run Commands:**
```bash
# Standard test run (from project root)
julia --project=. -e 'using Pkg; Pkg.test()'

# Direct test runner
julia --project=. test/runtests.jl

# Individual test file
julia --project=. test/test_qp_solver.jl
```

## Test File Organization

**Location:**
- Tests co-located in `/test` directory (separate from source)
- Tests mirror module structure: `src/qp_kkt.jl` → `test/test_qp_kkt.jl`
- Special directories: `test/olse/` for closed-form solution validation tests

**Naming:**
- Pattern: `test_<module>.jl`
- Examples: `test_qp_solver.jl`, `test_nonlinear_solver.jl`, `test_interface.jl`, `test_input_validation.jl`
- Shared utilities: `testing_utils.jl`

**Structure:**
```
test/
├── runtests.jl              # Main test orchestrator
├── testing_utils.jl         # Shared test helpers (make_θ)
├── test_graph_utils.jl      # Graph utility tests
├── test_problem_setup.jl    # Problem setup tests
├── test_qp_kkt.jl           # QP KKT construction tests
├── test_qp_solver.jl        # QP solver tests
├── test_nonlinear_solver.jl # Nonlinear solver tests
├── test_interface.jl        # Public interface tests
├── test_input_validation.jl # Input validation tests
├── test_integration.jl      # Integration tests
├── test_type_stability.jl   # Type stability tests
├── test_timer.jl            # TimerOutputs integration tests
├── olse/                    # OLSE comparison tests (closed-form solutions)
│   ├── test_qp_solver.jl
│   └── test_nonlinear_solver.jl
└── Project.toml             # Test-specific dependencies
```

## Test Structure

**Suite Organization:**

From `test/runtests.jl` (lines 7-48):
```julia
@testset "MixedHierarchyGames.jl" begin
    # Phase A: Utilities (implemented)
    include("test_graph_utils.jl")
    include("test_symbolic_utils.jl")

    # Phase B: Problem Setup (TDD - tests first)
    include("test_problem_setup.jl")

    # Phase C: QP KKT Construction (TDD)
    include("test_qp_kkt.jl")

    # Phase D: QP Solver (TDD)
    include("test_qp_solver.jl")

    # ... more phases
    @testset "Solution Extraction" begin
        @testset "extract_trajectories reshapes correctly" begin
            # Test logic here
        end
    end
end
```

**Nested testsets:**
- Top-level: `@testset "MixedHierarchyGames.jl"` encompasses all tests
- Module level: `@testset "QP Solver - solve_with_path"` for major components
- Feature level: `@testset "Solves simple 1-player QP"` for specific behaviors
- Deep nesting (4-5 levels typical) for comprehensive coverage

**Patterns:**

1. **Arrange-Act-Assert:**
   ```julia
   @testset "extract_trajectories reshapes correctly" begin
       # Arrange
       T = 3
       n_players = 2
       state_dim = 4
       control_dim = 2
       total_dim = (state_dim * (T + 1) + control_dim * T) * n_players
       sol = randn(total_dim)
       dims = (state_dims = [state_dim, state_dim], control_dims = [control_dim, control_dim])

       # Act
       xs, us = extract_trajectories(sol, dims, T, n_players)

       # Assert
       @test xs isa Dict
       @test length(xs[1]) == T + 1
   end
   ```

2. **Multiple assertions per testset:**
   - Related assertions grouped in single testset
   - Setup code shared across multiple `@test` calls

3. **Test helpers in `testing_utils.jl`:**
   ```julia
   """
       make_θ(player::Int, dim::Int)

   Convenience helper to create parameter vectors.
   """
   make_θ(player::Int, dim::Int) = make_symbolic_vector(:θ, player, dim)
   ```

## Mocking

**Framework:** No explicit mocking library (Mocking.jl not in dependencies).

**Patterns:**
- Use symbolic computation directly for function behavior verification
- Create simple test problems instead of mocking dependencies
- Example from `test/test_nonlinear_solver.jl` (lines 13-65):
  ```julia
  function make_two_player_chain_problem(; T=3, state_dim=2, control_dim=2)
      # Creates actual test problem with symbolic costs and dynamics
      G = SimpleDiGraph(N)
      add_edge!(G, 1, 2)

      function J1(z1, z2; θ=nothing)
          (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
          goal = [1.0, 1.0]
          sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
      end
      # ... more setup
  end
  ```

**What to Mock:** Not applicable - use actual implementations and test factories.

**What NOT to Mock:** None - full integration testing with actual solvers and symbolic computation.

## Fixtures and Factories

**Test Data:**
- Shared test problem generators in `test/test_nonlinear_solver.jl`:
  - `make_two_player_chain_problem()` - 2-player hierarchy
  - `make_three_player_chain_problem()` - 3-player hierarchy
  - Parameters: T (horizon), state_dim, control_dim

- Random data generation:
  ```julia
  sol = randn(total_dim)
  z1, z2 = randn(n), randn(n)
  A = randn(n, n); A = A' * A + I  # Positive definite
  ```

**Location:**
- Inline in test files: Most factories are in the same file using them
- Shared utilities: `test/testing_utils.jl` for truly shared helpers
- No separate fixtures directory

## Coverage

**Requirements:** Not explicitly enforced. No coverage targets found in configuration.

**View Coverage:**
```bash
# No standard coverage workflow configured
# Coverage.jl could be used if added to test dependencies
```

**Coverage approach:**
- Manual review of test organization by phase
- Comprehensive testsets for each major component
- Integration tests verify cross-component interactions

## Test Types

**Unit Tests:**
- Scope: Individual functions and types
- Approach: Test function behavior in isolation with controlled inputs
- Examples:
  - `test_graph_utils.jl`: Tests `is_root`, `is_leaf`, `has_leader` (simple predicates)
  - `test_problem_setup.jl`: Tests symbolic variable creation functions
  - Location: `test/test_*.jl` files

**Integration Tests:**
- Scope: Multiple components working together
- Approach: End-to-end problem solving with verification
- Examples:
  - `test/test_integration.jl` (574 lines): Compares QP vs Nonlinear solvers on same problem
  - `test/test_interface.jl`: Tests high-level `solve()` and `solve_trajectory_game!` interface
  - Location: Dedicated integration test files and OLSE comparison tests

**E2E Tests:**
- Framework: Not explicitly used
- Validation approach: Use closed-form analytical solutions (OLSE)
- Location: `test/olse/` directory
  - `olse/test_qp_solver.jl`: Validates QP solver against OLSE solutions
  - `olse/test_nonlinear_solver.jl`: Validates Nonlinear solver against OLSE solutions

## Common Patterns

**Async Testing:**
- Not applicable - no async code in library
- Some tests use `@async` for timeout handling in validation tests
- Example from `test/test_input_validation.jl`:
  ```julia
  function with_timeout(f, timeout_sec::Real)
      result = Channel{Any}(1)
      task = @async put!(result, f())

      if timedwait(() -> isready(result), timeout_sec) == :timed_out
          try
              schedule(task, InterruptException(); error=true)
          catch
          end
          return (nothing, true)
      end

      return (take!(result), false)
  end
  ```

**Error Testing:**
```julia
@testset "Rejects cyclic graph" begin
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 1)  # Creates cycle

    # ... setup ...

    (result, timed_out) = with_timeout(5.0) do
        try
            QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
            return :no_error
        catch e
            return e
        end
    end

    @test !timed_out
    @test result isa ArgumentError  # Verify expected error type
end
```

**Tolerance Testing:**
- Numerical precision: `atol=1e-6` typical default for KKT residuals
- Floating point comparisons: `isapprox(sol[1:2], [1.0, 2.0], atol=1e-6)`
- Linear system solves: `isapprox(A * x, b, atol=1e-8)` for validation
- Type stability tests use `@test_throws` for performance checks
- See `test/test_kkt_verification.jl` for residual tolerance patterns

**Test-Driven Development:**
- TDD phases documented in `test/runtests.jl`:
  - Phase A: Utilities (already implemented)
  - Phase B-G: Each phase has TDD comment indicating tests written first
  - Comment examples (lines 12-30): "# Phase B: Problem Setup (TDD - tests first)"
- Use `@test_broken` for unimplemented features (see CLAUDE.md requirements)

---

*Testing analysis: 2026-02-07*
