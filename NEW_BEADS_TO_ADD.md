# Beads to Add

These beads should be added to the tracking system when `bd` is available.

---

## Bead 1: Fix Armijo Line Search Inconsistency

**Type:** task
**Phase:** 5 (robustness)

**Description:**
The nonlinear solver has two line search implementations that are inconsistent:

1. **Inline version** (`nonlinear_kkt.jl:604-614`): Simple monotone descent
   - Condition: `norm(F_eval) < F_eval_current_norm` (any decrease)
   - Hardcoded: 10 iterations, 0.5 backtrack factor
   - No sufficient decrease parameter (σ)

2. **Standalone version** (`nonlinear_kkt.jl:665-700`): Proper Armijo
   - Condition: `ϕ_new ≤ ϕ_0 + σ*α*(-2*ϕ_0)` (sufficient decrease)
   - Configurable: α_init, β, σ, max_iters
   - Warns on failure

**Issue:** The inline version is labeled `use_armijo` but doesn't implement true Armijo conditions.

**Fix:** Replace the inline loop with a call to `armijo_backtracking_linesearch()`.

**Also includes:** Magic numbers documentation (LOW issue from review)

---

## Bead 2: Rename `examples/` Folder to `legacy/`

**Type:** task
**Phase:** 5 (cleanup)

**Description:**
The `examples/` folder contains legacy standalone scripts that implement their own solver from scratch. They do NOT use the `MixedHierarchyGames` package.

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
Use ProgressBars.jl to show solver iteration progress during `run_nonlinear_solver`.

**Implementation:**
- Add a `show_progress::Bool = false` keyword argument
- Wrap the main iteration loop with `ProgressBar(1:max_iters)` when enabled
- Update progress bar description with current residual norm

---

## Bead 4: Thread Safety for Nonlinear Solver

**Type:** task
**Phase:** 6 (low priority)

**Description:**
The nonlinear solver functions (`compute_K_evals`, `run_nonlinear_solver`) are not thread-safe. The precomputed M_fns and N_fns contain shared result buffers.

**Required Changes:**
- Add thread-local buffer pools or per-call buffer allocation
- Document thread safety guarantees in public API

---

## Bead 5: Singular Matrix Protection in K Evaluation ⚠️ CRITICAL

**Type:** task
**Phase:** 5 (robustness)
**Priority:** CRITICAL

**Description:**
Guard `K_evals[ii] = M_evals[ii] \ N_evals[ii]` against singular or ill-conditioned matrices. Can silently produce NaN/Inf.

**Location:** `src/nonlinear_kkt.jl:457`

**Implementation:**
```julia
κ = cond(M_evals[ii])
if κ > 1e10
    @warn "Ill-conditioned M matrix (κ=$κ) for player $ii, using regularized pseudoinverse"
    K_evals[ii] = pinv(M_evals[ii]) * N_evals[ii]
else
    K_evals[ii] = M_evals[ii] \ N_evals[ii]
end
```

---

## Bead 6: Refactor run_nonlinear_solver for Single Responsibility

**Type:** task
**Phase:** 5 (code quality)

**Description:**
Refactor `run_nonlinear_solver` to satisfy single-responsibility by extracting helpers for:
- Parameter preparation (`prepare_solver_parameters`)
- Convergence checking (`check_convergence`)
- Newton step computation (`compute_newton_step`)
- Solution update with line search (`update_solution`)

**Also includes:** Code duplication in `solve()` functions (LOW issue from review)

---

## Bead 7: Convergence Stall Detection

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Add convergence stall detection to the nonlinear solver by tracking residual history and terminating when improvement falls below a threshold.

**Implementation:**
- Track last N residual values (e.g., N=5)
- Terminate with `:stalled` status if improvement < 1e-10 for M consecutive iterations

---

## Bead 8: Armijo Line Search Explicit Success/Failure ⚠️ HIGH

**Type:** task
**Phase:** 5 (robustness)
**Priority:** HIGH

**Description:**
Change Armijo line search to return explicit success/failure information instead of silently returning α = 0.0.

**Current Behavior:**
- Returns 0.0 on failure with only a warning
- Caller cannot distinguish "found α=0" from "failed to find step"

**Proposed Change:**
- Return named tuple: `(α=0.5, success=true)` or `(α=0.0, success=false)`
- Update `run_nonlinear_solver` to handle failure explicitly

---

## Bead 9: README Citation and Documentation ✅ DONE

**Type:** task
**Phase:** 5 (documentation)
**Status:** COMPLETED in PR feature/phase-1-nonlinear-kkt

**Description:**
Replace the "TBD" paper reference in the README with "Manuscript in preparation" and add BibTeX placeholder.

---

## Bead 10: NonlinearSolverOptions Struct

**Type:** task
**Phase:** 5 (code quality)

**Description:**
Replace the untyped `options::NamedTuple` in NonlinearSolver with a concrete `NonlinearSolverOptions` struct.

**Proposed:**
```julia
struct NonlinearSolverOptions
    max_iters::Int
    tol::Float64
    verbose::Bool
    use_armijo::Bool
end
```

**Also includes:** Make linear solve residual threshold configurable (MEDIUM issue from review)

---

## Bead 11: Standardize Verbose Output

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Standardize verbose output across the codebase by using a single logging mechanism instead of mixing `println` and `@info`.

**Also includes:** Verbose flag hardcoded in tests (LOW issue from review)

---

## Bead 12: QP Linear Solve Failure Tests ⚠️ HIGH

**Type:** task
**Phase:** 5 (testing)
**Priority:** HIGH

**Description:**
Add tests that trigger and validate QP linear solve failure behavior.

**Tests to Add:**
- Singular KKT matrix (degenerate problem)
- Near-singular matrix with condition number check
- Verify error message clarity
- Verify NaN-filled return vector on failure

---

## Bead 13: Configurable Sparse Factorization

**Type:** task
**Phase:** 6 (very low priority)

**Description:**
Make sparse factorization strategy configurable instead of hard-coding UMFPACK.

**Proposed Change:**
- Add `factorization_method` option to solver constructors
- Support `:auto`, `:umfpack`, `:cholmod`, `:iterative`

---

## Bead 14: Zero Jacobian Buffers Before Reuse

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Explicitly zero Jacobian buffers before reuse to avoid round-off accumulation.

**Location:** `src/nonlinear_kkt.jl` in `run_nonlinear_solver`

```julia
fill!(∇F, 0.0)
mcp_obj.jacobian_z!(∇F, z_est, param_vec)
```

---

## Bead 15: Type Stability in Dict Storage

**Type:** task
**Phase:** 6 (low priority - performance)

**Description:**
Address type instabilities in Dict usage throughout KKT construction:

1. `Dict{Int, Any}` in `src/qp_kkt.jl:91-94`
2. `Dict{Int, Union{Matrix{Float64}, Nothing}}` in `src/nonlinear_kkt.jl:426-428`
3. `Dict{Int, Function}` in `src/nonlinear_kkt.jl:121-122`

**Also includes:**
- Missing type annotation on `compute_K_evals` (LOW)
- Non-idiomatic empty vector construction (LOW)

---

## Bead 16: Extract Player Ordering Helper

**Type:** task
**Phase:** 5 (code quality)

**Description:**
Create `ordered_player_indices(d::Dict)` helper to replace repeated `sort(collect(keys(d)))` pattern.

**Locations:**
- src/solve.jl:160-161
- src/nonlinear_kkt.jl:301-302
- src/utils.jl (multiple)

---

## Bead 17: Pre-allocate params_for_z Buffers

**Type:** task
**Phase:** 6 (low priority - performance)

**Description:**
The `params_for_z` closure in `run_nonlinear_solver` allocates new vectors on each call.

**Also includes:** Dense extractor matrix could be sparse (LOW)

---

## Bead 18: Consolidate Test Helpers

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Move duplicated test helpers to `testing_utils.jl`:
- `make_two_player_chain_problem` appears in multiple test files
- `make_three_player_problem` variations
- Other shared test setup code

---

## Bead 19: Documentation Polish

**Type:** task
**Phase:** 6 (low priority)

**Description:**
Improve documentation consistency:

1. Expand module-level docstring with usage example and key concepts
2. Add consistent detail to all public function docstrings
3. Add symbolic blowup warning to `get_qp_kkt_conditions` docstring
4. Document parameter semantics (`iteration_limit`, `proximal_perturbation`)

---

## Summary by Phase

### Phase 5 (Priority)
| Bead | Description | Priority |
|------|-------------|----------|
| 5 | Singular matrix protection | CRITICAL |
| 8 | Armijo explicit success/failure | HIGH |
| 12 | QP linear solve failure tests | HIGH |
| 1 | Armijo line search unification | Normal |
| 2 | Rename examples/ to legacy/ | Normal |
| 3 | Progress bars | Normal |
| 6 | Refactor run_nonlinear_solver | Normal |
| 10 | NonlinearSolverOptions struct | Normal |
| 16 | Extract player ordering helper | Normal |

### Phase 6 (Lower Priority)
| Bead | Description |
|------|-------------|
| 4 | Thread safety |
| 7 | Convergence stall detection |
| 11 | Standardize verbose output |
| 13 | Configurable sparse factorization |
| 14 | Zero Jacobian buffers |
| 15 | Type stability in Dict storage |
| 17 | Pre-allocate buffers |
| 18 | Consolidate test helpers |
| 19 | Documentation polish |

### Completed
| Bead | Description | PR |
|------|-------------|-----|
| 9 | README citation | feature/phase-1-nonlinear-kkt |
