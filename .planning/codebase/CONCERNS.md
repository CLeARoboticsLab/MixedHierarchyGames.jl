# Codebase Concerns

**Analysis Date:** 2026-02-07

## Tech Debt

**Armijo Line Search Inconsistency:**
- Issue: Two separate implementations of Armijo line search with different iteration limits (10 vs 20)
- Files: `src/nonlinear_kkt.jl` (lines 13-17: `LINESEARCH_MAX_ITERS = 10`), `src/nonlinear_kkt.jl` (line 693: default `max_iters=20` in `armijo_backtracking_linesearch`)
- Impact: Solvers use different step size selection strategies; `run_nonlinear_solver` always uses 10 iterations while the standalone function defaults to 20
- Fix approach: Unify line search implementations into single module-level function with consistent parameters (Bead: ei7)

**Function Complexity Exceeding SRP:**
- Issue: `run_nonlinear_solver` (lines 517-654 in `src/nonlinear_kkt.jl`) is ~120 lines with multiple responsibilities: iteration management, residual evaluation, Jacobian computation, Newton step solving, and line search
- Files: `src/nonlinear_kkt.jl`
- Impact: Hard to test individual components; difficult to swap out line search or stepping strategies; error handling scattered across function
- Fix approach: Extract sub-functions: `_newton_step()`, `_line_search_step()`, `_check_convergence()` (Bead: 3ws)

**Repeated Player Ordering Pattern:**
- Issue: `sort(collect(keys(...)))` pattern repeated 8+ times across codebase without abstraction
- Files: `src/nonlinear_kkt.jl` (lines 157, 308, 536), `src/solve.jl` (lines 379, 449, 513, 599), `src/types.jl` (lines 163, 199, 256)
- Impact: Code duplication; if player ordering semantics change, must update in multiple places; harder to maintain invariants
- Fix approach: Create helper function `_get_player_order(dict_with_player_keys::Dict)` → `Vector{Int}` (Bead: 3mz)

**Type Instability in Symbolic Construction:**
- Issue: `Dict{Int, Any}` used in several places for symbolic KKT construction (`πs`, `Ms`, `Ns`, `Ks` in `src/qp_kkt.jl` lines 91-94)
- Files: `src/qp_kkt.jl` (lines 88-94)
- Impact: Minor type instability during solver construction (not hot path), but violates Julia best practices; makes static analysis harder
- Fix approach: Use concrete symbolic type parameter: `Dict{Int, Vector{Symbolics.Num}}` or union of expected types (Bead: b47 for struct type parameter constraints)

**Magic Numbers for Line Search:**
- Issue: `LINESEARCH_BACKTRACK_FACTOR = 0.5`, `σ = 1e-4` hardcoded; no explanation of why these specific values chosen
- Files: `src/nonlinear_kkt.jl` (lines 16-17, 693)
- Impact: Hard-coded solver behavior; impossible to tune for different problem classes without modifying source
- Fix approach: Move to `NonlinearSolverOptions` struct with documented defaults (Bead: r89)

**No Regularization for Ill-Conditioned Systems:**
- Issue: QP linear solve at `src/solve.jl` lines 531-535 uses bare backslash `J \ (-F)` without regularization for singular/near-singular matrices
- Files: `src/solve.jl` (lines 531-535)
- Impact: Linear solver can fail or produce NaN/Inf for ill-conditioned KKT systems; no recovery mechanism; comment explicitly warns about missing Tikhonov regularization
- Fix approach: Add optional Tikhonov regularization with `damping::Float64` parameter (Bead: ulv)

**Singular Matrix Handling in K Evaluation:**
- Issue: `compute_K_evals` computes `K = M \ N` (line 466 in `src/nonlinear_kkt.jl`) without checking condition number or handling singular M matrices
- Files: `src/nonlinear_kkt.jl` (lines 430-479)
- Impact: If M is singular or near-singular, the backslash operator may fail silently or return NaN; no error message or recovery
- Fix approach: Check `cond(M)` before solve; fall back to pseudoinverse or raise informative error (Bead: txx)

---

## Known Issues

**Non-Thread-Safe Function Evaluation:**
- Issue: `compute_K_evals` uses precomputed M/N evaluation functions (`M_fns`, `N_fns`) with shared result buffers
- Files: `src/nonlinear_kkt.jl` (lines 411-418 in docstring)
- Trigger: Calling `compute_K_evals` from multiple threads will cause data races on the result buffers
- Workaround: Each thread must instantiate its own `NonlinearSolver` instance; do not share precomputed solver across threads
- Root cause: SymbolicTracingUtils functions cache results in mutable buffers for performance
- Planned fix: Thread-local solver instances or buffer-per-thread design (Phase 6 - Bead: performance improvements)

**PATH Solver Unavailable on ARM64:**
- Issue: The PATH solver (from PATHSolver.jl) is not available for ARM64 architecture (Apple Silicon, Graviton, etc.)
- Files: Implicit in `src/solve.jl` solver selection (lines 67-71)
- Impact: Users on ARM64 cannot use `:path` solver backend; forced to use `:linear` backend only
- Workaround: Use `solver=:linear` on ARM64; use `:path` only on x86_64
- Root cause: PATHSolver.jl binary dependencies
- Planned fix: Document limitation and provide fallback solver logic (Bead: e86)

**KKT System Linearity Assumption Not Enforced:**
- Issue: QPSolver assumes KKT system is affine (linear in z) but only checks this with random points at construction time
- Files: `src/types.jl` (lines 176-215)
- Trigger: Passing a nonlinear cost function to QPSolver will produce incorrect results; the warning at line 212-213 may not always fire
- Impact: Silent wrong answers if user constructs QPSolver on problem with nonlinear costs
- Root cause: Check uses finite-difference-like evaluation at random points; adversarial problems could bypass this
- Fix approach: Add problem analysis at construction time; document in CLAUDE.md requirement that QPSolver only accepts quadratic costs (Bead: documentation)

---

## Fragile Areas

**Follower K Matrix Evaluation Sequence:**
- Files: `src/nonlinear_kkt.jl` lines 430-479 (`compute_K_evals`), lines 373-408 (`_build_augmented_z_est`)
- Why fragile: Complex interdependency between z estimates and K evaluations; must process in strict reverse topological order; assumes each player has at most one parent
- Safe modification:
  - Do not change iteration order without updating both `_build_augmented_z_est` and result concatenation
  - Must maintain invariant: K matrices for followers are computed before K matrices for leaders that depend on them
  - Test: Run `test/test_nonlinear_kkt.jl` after any changes to iteration order
- Test coverage: Limited; relies on integration tests in `test/test_solve_nonlinear.jl` rather than unit tests on iteration order

**Policy Constraint Stripping Logic:**
- Files: `src/qp_kkt.jl` lines 200-270 (`strip_policy_constraints`)
- Why fragile: Complex manipulation of KKT condition vectors; assumes strict structure (stationarity, policy constraints, own constraints in order)
- Safe modification:
  - Any change to KKT condition assembly order in `get_qp_kkt_conditions` must be coordinated with `strip_policy_constraints`
  - Verify by running QP solver tests: `julia --project=. test/test_solve_qp.jl`
  - Document changes in both files with cross-references
- Test coverage: Covered by QP solver tests but not isolated unit tests

**Augmented Variable Construction for Nonlinear Solver:**
- Files: `src/nonlinear_kkt.jl` lines 36-56 (`_construct_augmented_variables`)
- Why fragile: Builds variable lists by concatenating K matrix entries; buffer reuse in `_build_augmented_z_est` (lines 387-404) assumes consistent length
- Safe modification:
  - If K matrix dimensions change, must update buffer allocation in `_build_augmented_z_est`
  - Cannot change flatten/unflatten order between K evaluation and symbolic function evaluation
  - Test: Run `experiments/` tests to catch integration failures
- Test coverage: Covered by nonlinear solver integration tests; no isolated unit tests

---

## Scaling Limits

**Symbolic Expression Blowup in QP Solver:**
- Current capacity: Tested up to ~6 players in QP problems without issues
- Limit: K matrix computation `K = M \ N` at `src/qp_kkt.jl` lines 167-170 performed symbolically; M and N matrices grow exponentially with player count
- Symptom: Solver construction becomes extremely slow (>minutes); intermediate expressions can exceed memory
- Scaling path: Nonlinear solver (`src/nonlinear_kkt.jl`) uses numerical K evaluation instead, scales much better. For large problems (>10 players), must use NonlinearSolver
- Recommendation: Document player count limits in README; recommend NonlinearSolver for >8 players

**Memory Usage in Precompilation:**
- Current capacity: Full symbolic precomputation fits in ~2GB RAM for 4-5 player problems
- Limit: `preoptimize_nonlinear_solver` builds all symbolic Jacobians and compiles them to functions
- Symptom: OOM errors during solver construction on systems with <4GB available RAM
- Scaling path: Lazy compilation (compute Jacobians on first solve), but would lose construction-time error detection
- Recommendation: Document minimum RAM requirements; consider implementing incremental compilation (Phase 7)

---

## Test Coverage Gaps

**QP Solver Linear Solve Failure:**
- What's not tested: Behavior when linear solve fails (singular matrix, LAPACK exceptions)
- Files affected: `src/solve.jl` lines 534-560
- Risk: Silent failure modes; NaN values returned but not clearly flagged
- Priority: High (affects correctness)
- Addition needed: Test suite covering ill-conditioned systems, singular matrices, determinant=0 cases (Bead: hxq)

**Nonlinear Solver Non-Convergence:**
- What's not tested: Explicit coverage of max iterations reached, Armijo line search failure, numerical error paths
- Files affected: `src/nonlinear_kkt.jl` lines 577-581, 609-613, 640-644
- Risk: Cannot verify solver gracefully handles failure modes; status codes may not be set correctly
- Priority: High (affects robustness)
- Addition needed: Tests forcing early termination, NaN injection, pathological step sizes (Bead: eeo)

**Policy Constraint Satisfaction Verification:**
- What's not tested: Direct verification that follower policies satisfy policy constraints after solving
- Files affected: `src/qp_kkt.jl` lines 128-142 (policy constraint construction)
- Risk: If policy constraint extraction is wrong, error silent—only manifest in solution quality
- Priority: Medium (affects solution validity)
- Addition needed: Test helper that extracts and verifies `zⱼ ≈ -K * yⱼ` after solve

**Edge Cases in Hierarchy Graph:**
- What's not tested: Single-player games, fully linear hierarchies (chain), disconnected components (not valid but should reject)
- Files affected: `src/types.jl` validation (lines 92-143)
- Risk: Edge cases may produce cryptic error messages or silent failures
- Priority: Low (validation at construction catches most issues)
- Addition needed: Parametrized tests for graph topology edge cases

---

## Numerical Stability Concerns

**Hard-Coded Convergence Tolerance:**
- Issue: Default tolerance `tol=1e-6` (lines in `src/nonlinear_kkt.jl`, `src/types.jl`, `src/solve.jl`)
- Concern: May be loose for ill-conditioned problems, too tight for noisy data
- Impact: No relative tolerance option; cannot adapt to problem scale
- Fix approach: Add `rel_tol::Float64` parameter to solver options (Bead: rac)

**Residual Check Only on Convergence:**
- Issue: `solve_qp_linear` checks `residual > 1e-6 * max(1.0, norm(F))` only AFTER solve (line 545 in `src/solve.jl`)
- Concern: If solve is successful but residual is large, warning is logged but solution is still returned
- Impact: May hide ill-conditioned systems; user may not realize solution quality is poor
- Fix approach: Make residual check configurable; option to fail if residual exceeds threshold

**Line Search Merit Function:**
- Issue: Armijo condition uses `||f||²` norm without scaling to problem size
- Files: `src/nonlinear_kkt.jl` line 696
- Concern: For very small or very large residual norms, line search may behave unpredictably
- Fix approach: Normalize by problem dimension or initial residual

---

## Dependencies at Risk

**Symbolics.jl Version:**
- Risk: Package is rapidly evolving; major versions have breaking API changes
- Current: Uses Symbolics.jl but no pinned major version constraint visible
- Impact: Upgrades could break symbolic function compilation
- Mitigation: Check Project.toml version constraints; add pre-merge test for latest Symbolics.jl version (Bead: upgrade to v7)

**ParametricMCPs.jl Stability:**
- Risk: Relatively new package; internal buffer API could change
- Current usage: Relies on mutable result buffers in jacobian_z! for performance
- Impact: Would need to rewrite K evaluation if buffer API changes
- Mitigation: Encapsulate buffer access in wrapper functions; add comments explaining buffer contract

**PATHSolver.jl Binary Availability:**
- Risk: Binary packages for some architectures may not be available
- Current issue: Not available for ARM64 (documented but impacts users)
- Mitigation: Make `:linear` solver fully functional as fallback; document PATH limitations in README

---

## Missing Critical Features

**Solver Warm-Start Validation:**
- Problem: NonlinearSolver accepts `initial_guess` but no validation that it's in correct format
- Impact: Silent wrong results if user passes incorrectly shaped initial guess
- Files: `src/solve.jl` lines 197, 262
- Fix: Add assertion that `initial_guess` length matches `all_variables` (currently implicit padding at line 541-543)

**Solution Quality Metrics:**
- Problem: Solvers return solution vector but no measure of solution quality/confidence
- Impact: User cannot distinguish between "converged to tolerance" and "ran out of iterations but produced something"
- Files: Affects both QPSolver and NonlinearSolver return signatures
- Fix: Return struct with solution + metrics (optimality gap, constraint violation, condition number)

**Problem Degeneracy Detection:**
- Problem: No detection of degenerate hierarchy structures (duplicate players, isolated components)
- Impact: Produces hard-to-debug errors downstream
- Files: `src/types.jl` validation (lines 92-143)
- Fix: Add explicit checks and clear error messages for degeneracies

---

## Performance Bottlenecks

**K Matrix Recomputation During Line Search:**
- Problem: Line search at `src/nonlinear_kkt.jl` lines 622-633 evaluates K for each trial step
- Current cost: Each K evaluation requires forward pass through all symbolic functions
- Scaling: With L line search iterations and I outer iterations, cost is O(L × I × K_eval_cost)
- Improvement: Cache K for current z; only recompute if full Newton step accepted (Bead: kmv)

**Jacobian Sparsity Not Exploited:**
- Problem: M and N matrices have structure (block-sparse for hierarchies) not utilized
- Files: `src/nonlinear_kkt.jl` lines 204-205 (compute dense Jacobians), lines 462-463 (reshape as dense)
- Impact: Memory and compute cost scales as O(n²) instead of O(n) for sparse problems
- Improvement: Use BlockArrays.jl to preserve sparsity (Bead: 9gy)

**Repeated `sort(collect(keys(...)))` Allocations:**
- Problem: Creates temporary vectors repeatedly instead of caching
- Files: Multiple locations cited in Tech Debt section above
- Impact: Measurable allocation overhead during solve loop
- Improvement: Cache player order during precomputation (Bead: 3mz)

---

## Documentation Gaps

**Architecture Decision Records:**
- Gap: No explanation of why nonlinear solver uses numerical K evaluation instead of symbolic
- Impact: Hard for maintainers to understand trade-offs
- Fix: Create `docs/adr/adr-002-numerical-k-evaluation.md` explaining performance rationale

**Error Message Library:**
- Gap: Error messages at `src/types.jl`, `src/solve.jl` are informative but not centralized
- Impact: Inconsistent tone; hard to test all error paths
- Fix: Create `src/errors.jl` with factory functions for standard failures (Bead: documentation)

**Test Suite Organization:**
- Gap: No README explaining test structure, how to run individual suites, expected failures
- Impact: New contributors cannot quickly understand test coverage
- Fix: Create `test/README.md` documenting structure (Bead: 7w6)

---

*Concerns analysis: 2026-02-07*
