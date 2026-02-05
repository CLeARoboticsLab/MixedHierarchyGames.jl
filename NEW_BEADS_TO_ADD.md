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
