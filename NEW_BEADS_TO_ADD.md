# Beads to Add

These beads should be added to the tracking system when `bd` is available.

---

## Bead 1: Fix Armijo Line Search Inconsistency

**Type:** task
**Phase:** Line search improvements

**Description:**
The nonlinear solver has two line search implementations that are inconsistent:

1. **Inline version** (`nonlinear_kkt.jl:584-595`): Simple monotone descent
   - Condition: `norm(F_eval) < F_eval_current_norm` (any decrease)
   - Hardcoded: 10 iterations, 0.5 backtrack factor
   - No sufficient decrease parameter (σ)

2. **Standalone version** (`nonlinear_kkt.jl:639-674`): Proper Armijo
   - Condition: `ϕ_new ≤ ϕ_0 + σ*α*(-2*ϕ_0)` (sufficient decrease)
   - Configurable: α_init, β, σ, max_iters
   - Warns on failure

**Issue:** The inline version is labeled `use_armijo` but doesn't implement true Armijo conditions. This could cause slow convergence in some cases.

**Fix:** Replace the inline loop with a call to `armijo_backtracking_linesearch()`.

---

## Bead 2: Rename `examples/` Folder to `legacy/`

**Type:** task
**Phase:** 5 (cleanup)

**Description:**
The `examples/` folder contains legacy standalone scripts that implement their own solver from scratch. They do NOT use the `MixedHierarchyGames` package:

```
examples/
├── automatic_solver.jl          # Own solver implementation
├── general_kkt_construction.jl  # Own KKT construction
├── solve_kkt_conditions.jl      # Own solve code
├── pursuer_protector_vip.jl     # Uses legacy solver
└── ...
```

This is confusing because:
- `experiments/` folder uses the actual package
- `examples/` folder uses legacy standalone code
- Users may try to run `examples/` expecting package integration

**Options:**
1. Rename `examples/` to `legacy/`
2. Delete `examples/` entirely (if no longer needed)
3. Update `examples/` to use the package (significant work)

**Recommendation:** Rename to `legacy/` with a README explaining it's historical reference code.

---

## Bead 3: Add Progress Bars to NonlinearSolver

**Type:** task
**Phase:** 5 (enhancements)

**Description:**
Use ProgressBars.jl to show solver iteration progress during `run_nonlinear_solver`. The package is already a dependency.

**Implementation:**
- Add a `show_progress::Bool = false` keyword argument to `run_nonlinear_solver`
- Wrap the main iteration loop with `ProgressBar(1:max_iters)` when enabled
- Update progress bar description with current residual norm

**Notes:**
- Should be optional to avoid cluttering output in automated tests
- Consider also adding to QPSolver for consistency

---

## Bead 4: Thread Safety for Nonlinear Solver (Phase 6)

**Type:** task
**Phase:** 6 (low priority)

**Description:**
The nonlinear solver functions (`compute_K_evals`, `run_nonlinear_solver`) are not thread-safe. The precomputed M_fns and N_fns contain shared result buffers that would cause data races if called concurrently from multiple threads.

**Current State:**
- No concurrency issues in single-threaded use
- Precomputed components are read-only except for internal buffers
- No global mutable state

**Required Changes:**
- Add thread-local buffer pools or per-call buffer allocation
- Consider using `Threads.@spawn`-safe patterns
- Document thread safety guarantees in public API

---

## Bead 5: Singular Matrix Protection in K Evaluation (Phase 5)

**Type:** task
**Phase:** 5 (robustness)

**Description:**
Guard `K_evals[ii] = M_evals[ii] \ N_evals[ii]` against singular or ill-conditioned matrices. Add a condition number check, wrap the solve in try-catch, and fall back to a regularized pseudoinverse or fail explicitly instead of producing NaN or Inf.

Apply the same protection to all `M \ N` operations by checking conditioning and using a regularized or pseudoinverse-based solve when κ exceeds a safe threshold (e.g. 1e10).

**Location:** `src/nonlinear_kkt.jl:449`

**Implementation:**
```julia
# Check condition number before solve
κ = cond(M_evals[ii])
if κ > 1e10
    @warn "Ill-conditioned M matrix (κ=$κ) for player $ii, using regularized pseudoinverse"
    K_evals[ii] = pinv(M_evals[ii]) * N_evals[ii]
else
    K_evals[ii] = M_evals[ii] \ N_evals[ii]
end
```

---

## Bead 6: Refactor run_nonlinear_solver for Single Responsibility (Phase 5)

**Type:** task
**Phase:** 5 (code quality)

**Description:**
Refactor `run_nonlinear_solver` to satisfy single-responsibility by extracting helpers for:
- Parameter preparation (`prepare_solver_parameters`)
- Convergence checking (`check_convergence`)
- Newton step computation (`compute_newton_step`)
- Solution update with line search (`update_solution`)

**Current Issues:**
- Function is ~120 lines with multiple responsibilities
- Hard to test individual components
- Difficult to modify individual steps

---

## Bead 7: Convergence Stall Detection (Phase 6)

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Add convergence stall detection to the nonlinear solver by tracking residual history and terminating when improvement falls below a threshold.

**Implementation:**
- Track last N residual values (e.g., N=5)
- Compute relative improvement: `(old_residual - new_residual) / old_residual`
- Terminate with `:stalled` status if improvement < threshold (e.g., 1e-10) for M consecutive iterations
- Report stall in verbose output

---

## Bead 8: Armijo Line Search Explicit Success/Failure (Phase 5)

**Type:** task
**Phase:** 5 (robustness)

**Description:**
Change Armijo line search to return explicit success/failure information instead of silently returning α = 0.0.

**Current Behavior:**
- `armijo_backtracking_linesearch` returns 0.0 on failure with only a warning
- Caller cannot distinguish between "found α=0" and "failed to find step"

**Proposed Change:**
- Return named tuple: `(α=0.5, success=true)` or `(α=0.0, success=false)`
- Or throw an exception on failure
- Update `run_nonlinear_solver` to handle failure explicitly

---

## Bead 9: README Citation and Documentation (Phase 5)

**Type:** task
**Phase:** 5 (documentation)

**Description:**
Replace the "TBD" paper reference in the README with either a concrete citation or "Manuscript in preparation" including author names. Complete the citation section in the README by adding a BibTeX entry or full author and DOI information.

---

## Bead 10: NonlinearSolverOptions Struct (Phase 5)

**Type:** task
**Phase:** 5 (code quality)

**Description:**
Replace the untyped `options::NamedTuple` in NonlinearSolver with a concrete `NonlinearSolverOptions` struct.

**Current:**
```julia
struct NonlinearSolver{TP<:HierarchyProblem, TC}
    problem::TP
    precomputed::TC
    options::NamedTuple  # Untyped!
end
```

**Proposed:**
```julia
struct NonlinearSolverOptions
    max_iters::Int
    tol::Float64
    verbose::Bool
    use_armijo::Bool
end

struct NonlinearSolver{TP<:HierarchyProblem, TC}
    problem::TP
    precomputed::TC
    options::NonlinearSolverOptions
end
```

**Benefits:**
- Type stability
- Better documentation
- IDE autocomplete support
- Validation in constructor

---

## Bead 11: Standardize Verbose Output (Phase 6)

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Standardize verbose output across the codebase by using a single logging mechanism instead of mixing `println` and `@info`.

**Current State:**
- Some functions use `println` for verbose output
- Others use `@info` from Julia's logging system
- Inconsistent formatting and verbosity levels

**Proposed Change:**
- Use `@info`, `@debug`, `@warn` consistently throughout
- Add a `LogLevel` option to solvers for fine-grained control
- Ensure all verbose messages include relevant context (iteration, residual, etc.)

---

## Bead 12: QP Linear Solve Failure Tests (Phase 5)

**Type:** task
**Phase:** 5 (testing)

**Description:**
Add tests that trigger and validate QP linear solve failure behavior. Currently there are no tests verifying that the QP solver correctly handles singular or ill-conditioned KKT systems.

**Tests to Add:**
- Singular KKT matrix (degenerate problem)
- Near-singular matrix with condition number check
- Verify error message clarity
- Verify NaN-filled return vector on failure

---

## Bead 13: Configurable Sparse Factorization (Phase 6)

**Type:** task
**Phase:** 6 (very low priority)

**Description:**
Make sparse factorization strategy configurable instead of hard-coding UMFPACK.

**Current State:**
- Uses Julia's default sparse LU (UMFPACK) for all problems
- No way to switch to alternatives (CHOLMOD for symmetric, iterative solvers for large problems)

**Proposed Change:**
- Add `factorization_method` option to solver constructors
- Support `:auto`, `:umfpack`, `:cholmod`, `:iterative`
- Auto-detect symmetric positive definite systems for CHOLMOD

---

## Bead 14: Zero Jacobian Buffers Before Reuse (Phase 6)

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Explicitly zero Jacobian buffers before reuse to avoid round-off accumulation over many iterations.

**Location:** `src/nonlinear_kkt.jl` in `run_nonlinear_solver`

**Implementation:**
```julia
# Before each Jacobian evaluation
fill!(∇F, 0.0)
mcp_obj.jacobian_z!(∇F, z_est, param_vec)
```

**Risk:** May slightly slow down iterations. Profile before implementing.
