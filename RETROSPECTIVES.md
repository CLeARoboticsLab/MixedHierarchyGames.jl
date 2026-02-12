# Development Retrospectives

This document serves as a learning diary for the MixedHierarchyGames.jl development process. After each PR, we conduct a retrospective to honestly assess how well we followed our development practices (TDD, Clean Code, Clean Architecture, small commits) and identify concrete improvements for future work. The goal is continuous improvement—not perfection, but steady progress toward better engineering discipline.

By documenting what went well and what didn't, we create institutional memory that helps us avoid repeating mistakes and reinforces practices that work. This is especially valuable when context is lost between sessions or when onboarding new contributors.

---

## PR: feature/phase-1-nonlinear-kkt

**Date:** 2026-02-05
**Commits:** 38
**Tests:** 427 passing

### Summary

Implemented the complete nonlinear solver for mixed hierarchy games, including Newton iteration with line search, experiments infrastructure, OLSE verification tests, and comprehensive code review fixes.

### TDD Compliance

**Score: Partial (6/10)**

- **What went well:**
  - OLSE verification tests written before implementing solver comparisons
  - Integration tests defined expected behavior before fixes
  - KKT verification utilities eventually developed with proper TDD (24 tests)

- **What went wrong:**
  - `evaluate_kkt_residuals` and `verify_kkt_solution` were initially implemented WITHOUT tests
  - Had to retroactively write TDD tests mid-PR after this was identified
  - Several nonlinear solver functions were ported from reference code without tests-first approach

- **Root cause:** Pressure to show progress led to "implement first, test later" shortcuts. The reference implementation existed, making it tempting to port directly.

- **Improvement for next PR:**
  - Start each feature by creating the test file FIRST, even if empty
  - Write test names/descriptions before any implementation
  - Use `@test_broken` for tests that define behavior not yet implemented

- **Verifiable Solution:** Before any implementation file is created, run `git status` and verify a corresponding test file exists. If implementing `src/foo.jl`, there must be `test/test_foo.jl` in the staged or committed files first.

### Clean Code Practices

**Score: Fair (7/10)**

- **What went well:**
  - Extracted shared utilities to `experiments/common/`
  - Unified `QPProblem` and `NonlinearProblem` into single `HierarchyProblem`
  - Replaced `_value_or_default` with standard `something()`
  - Added comprehensive docstrings to new functions

- **What went wrong:**
  - `run_nonlinear_solver` is 120+ lines with multiple responsibilities (created Bead 6)
  - Magic numbers for line search parameters (10 vs 20 iterations inconsistency)
  - Some Dict types use `Any` causing type instability
  - Repeated `sort(collect(keys(...)))` pattern not extracted (created Bead 16)

- **Improvement for next PR:**
  - Extract helper functions when any function exceeds 50 lines
  - Define constants at module level with clear names
  - Review for repeated patterns before PR completion

- **Verifiable Solution:** Before marking PR ready, run: `grep -c "^function\|^end" src/*.jl` and manually verify no function body exceeds 50 lines. Add a pre-merge check: search for magic numbers (bare numeric literals) in new code.

### Clean Architecture Practices

**Score: Good (8/10)**

- **What went well:**
  - Clear separation: `src/` (core), `experiments/` (applications), `test/` (verification)
  - Experiments follow consistent `config.jl`/`run.jl`/`support.jl` structure
  - Dependencies point inward (experiments depend on core, not vice versa)
  - Separate `Project.toml` files for different contexts

- **What went wrong:**
  - `examples/` folder contains legacy code that doesn't use the package (confusing)
  - Some coupling between solver internals and interface layer

- **Improvement for next PR:**
  - Complete Bead 2 (rename `examples/` to `legacy/`)
  - Consider whether any experiment code should move to `src/`

- **Verifiable Solution:** Before PR completion, verify no imports go from `src/` to `experiments/`. Run: `grep -r "include.*experiments" src/` should return empty.

### Commit Hygiene

**Score: Poor (5/10)**

- **What went well:**
  - Commit messages generally describe what changed
  - No commits that break tests

- **What went wrong:**
  - Many commits are too large (e.g., "Address CRITICAL and HIGH priority code review issues" touches 9 files)
  - Some commits bundle unrelated changes
  - 38 commits is high for a single PR—suggests scope creep

- **Root cause:** Code review generated many issues that were addressed in batches rather than individually.

- **Improvement for next PR:**
  - Commit after EACH individual fix, not after batches
  - If a commit message needs "and" or lists multiple items, split it
  - Consider feature flags or separate PRs for distinct features

- **Verifiable Solution:** Before committing, run `git diff --stat` and verify fewer than 100 lines changed across no more than 3 files. If commit message contains "and", split into separate commits.

### CLAUDE.md Compliance

**Score: Good (8/10)**

- **What went well:**
  - Added Pre-Merge Retrospective section (meta-improvement!)
  - Added PR requirements documentation
  - Added dependency management guidelines
  - Followed verification checklist before marking complete

- **What went wrong:**
  - TDD section was not followed strictly (see above)
  - Some dead code not immediately removed (deferred to beads)

- **Improvement for next PR:**
  - Review CLAUDE.md at PR START, not just at end
  - Check TDD compliance after each feature, not just at PR end

- **Verifiable Solution:** At PR start, explicitly acknowledge CLAUDE.md review with a comment listing which sections apply. After each feature, run tests and note "TDD verified: [feature name]" in commit message or PR notes.

### Beads Created

17 beads created for future work:
- Phase 5 (Robustness/Quality): Beads 1, 2, 3, 5, 6, 8, 9, 10, 12, 16
- Phase 6 (Performance/Polish): Beads 4, 7, 11, 13, 14, 15, 17

### Key Learnings

1. **TDD discipline requires vigilance.** Even with clear instructions, the temptation to "just port the code" is strong. Mitigation: Create test file first, always.

2. **Code review is valuable but creates batch-fix pressure.** Running comprehensive reviews generates many issues. Addressing them individually with small commits is better than batching.

3. **Scope creep is real.** This PR grew from "implement nonlinear solver" to include experiments refactoring, code review fixes, and documentation updates. Consider splitting earlier.

4. **Retrospectives work.** Writing this identified concrete patterns (batch commits, TDD shortcuts) that weren't obvious during development.

### Action Items for Next PR

- [ ] Create test file before any implementation file
- [ ] Commit after each individual change (target: <50 lines per commit)
- [ ] Review CLAUDE.md at PR start
- [ ] Split PR if scope exceeds original intent
- [ ] Complete Bead 1 (Armijo unification) with strict TDD

---

## PR: feature/docker-gh-token-auth

**Date:** 2026-02-06
**Commits:** 2
**Tests:** 450 passing (no new tests — infrastructure only)

### Summary

Fix Docker container GitHub push auth using Docker Desktop for Mac SSH agent forwarding. Consolidate Dockerfile apt layers. Update README prerequisites.

### TDD Compliance

**Score: N/A** — Infrastructure change, no application code. Manual integration testing (SSH forwarding, git ls-remote, tool verification).

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Consolidated three apt-get layers into one after DevOps review
  - Entrypoint script is minimal (5 lines)
  - Sudoers rule tightly scoped to one command
- **What went wrong:**
  - Initial credential helper had wrong `gh` path (`~/.local/bin/gh` vs `/usr/bin/gh`)

### Clean Architecture Practices

**Score: Good (8/10)**

- **What went well:**
  - Clean separation of build-time config vs runtime entrypoint
  - Socket permissions handled at entrypoint, not baked into image

### Commit Hygiene

**Score: Good (8/10)**

- **What went well:**
  - Two focused commits: Docker auth changes, then README docs
  - Each commit is self-contained and descriptive
- **What went wrong:**
  - Iterative debugging (token approach → SSH pivot) happened in working tree, not captured in commits. Fine for this case but shows the value of committing intermediate states.

### CLAUDE.md Compliance

**Score: Good (8/10)**

- [x] Reviewed CLAUDE.md at PR start
- [x] Pre-merge checklist completed
- [x] Retrospective recorded
- [x] Documentation updated (README prerequisites)
- [x] Security + DevOps review requested and applied

### Key Learnings

1. **Check org PAT policies before creating tokens.** Fine-grained PATs for org repos may need admin approval. Could have saved 30 minutes by checking first.
2. **Docker Desktop for Mac has its own SSH socket path.** `/run/host-services/ssh-auth.sock`, not `$SSH_AUTH_SOCK`. This is a common gotcha.
3. **Host volume mounts override build-time config.** The `~/.gitconfig` mount overwrote our build-time `git config --global` settings. Solved with `GIT_CONFIG_SYSTEM` pointing to a separate file (though ultimately not needed after SSH pivot).

### Action Items for Next PR

- [ ] Revisit fine-grained PAT approach when org admin can approve
- [ ] Consider adding `ssh-add` check to entrypoint with helpful error message

---

## PR: feature/timer-outputs-benchmarking-v2

**Date:** 2026-02-06 – 2026-02-07
**Commits:** 8
**Tests:** 450 passing
**Sessions:** 3 (spanned context window resets)

### Summary

Added TimerOutputs instrumentation to QPSolver and NonlinearSolver, ran comprehensive benchmarks on all three experiments, and confirmed through direct old-vs-new solver comparison that the new `src/` code is algorithmically identical to the old `examples/` code. Discovered and fixed a hidden collision weight mismatch. Moved the weight to caller cost functions. Added two new CLAUDE.md rules from lessons learned.

### TDD Compliance
**Score: 8/10**
- What went well: 23 new timer tests written before/alongside TimerOutputs integration. Tests verify the `to` kwarg flows through correctly and timer sections appear in output.
- What went wrong: Collision weight refactor (moving 0.1 from function to caller) had no new tests. Acceptable since it only affects experiment config, not `src/`.
- Improvement: N/A — this PR is primarily benchmarking/instrumentation, not new solver logic.

### Clean Code Practices
**Score: 8/10**
- What went well: Hidden collision weight discovered and moved to explicit `COLLISION_WEIGHT` constant. This finding was codified as a new CLAUDE.md rule ("no hidden scaling in shared functions").
- What went wrong: `benchmark_all.jl` is ~250 lines that could benefit from helper extraction, but acceptable for a one-off benchmark runner.
- Improvement: Consider breaking benchmark scripts into per-experiment modules if they grow further.

### Clean Architecture Practices
**Score: 9/10**
- What went well: TimerOutputs integration uses optional `to` kwarg — zero overhead when not passed. Clean separation between `src/` instrumentation and `experiments/benchmarks/` runner.
- What went wrong: Nothing significant.

### Commit Hygiene
**Score: 7/10**
- What went well: Later commits (collision weight refactor, CLAUDE.md rules, benchmark re-run) are small and focused.
- What went wrong: Early commits from Session 1 are large — "Add benchmark results" bundles README + script relocation + investigation findings. Spanned 3 sessions, which makes it hard to keep commits granular.
- Improvement: Break large documentation commits into smaller pieces. When a PR spans sessions, review commit history at the start of each new session.

### CLAUDE.md Compliance
**Score: 7/10**
- What went well: TDD for timer tests, experiments structure, retrospective conducted, learnings fed back into CLAUDE.md.
- What went wrong:
  1. **PR description went stale** — after multiple pushes, the description still referenced old benchmark numbers and was missing files from the Changes list. Fixed, and added a new CLAUDE.md rule requiring PR description updates on every push.
  2. **Beads created late** — optimization opportunities (CSE, sparse M\N) discovered mid-investigation but beads not created until user prompted.
  3. **Benchmarks not re-run after code changes** — collision weight refactor changed the cost function but benchmarks weren't re-run until the user asked "are the benchmark results still correct?"
- Improvement: After any code change that affects experiment behavior, immediately re-run benchmarks and update numbers. Don't wait to be asked.
- Verifiable Solution: New CLAUDE.md rule ensures PR descriptions stay current. Benchmark results now match the actual committed code.

### Beads Created
- `0ip` — Explore sparse/block-structured M\N solve
- `zkn` — Enable CSE in compiled symbolic functions
- `ahv` — Re-run allocation benchmarks on nonlinear problem

### Beads Closed
- `b9w` — Verify optimized code performance

### Key Learnings
1. **Shared-process benchmarks are unreliable.** GC warming and JIT artifacts create 15-30% ordering effects. Always benchmark in separate Julia processes.
2. **Bit-for-bit comparison is the gold standard.** Wall-time comparisons are noisy; comparing iteration counts, residuals, and solution vectors to machine precision gives deterministic answers.
3. **Hidden constants in utility functions cause subtle bugs.** The 0.1 inside `smooth_collision_all` caused a long debugging session tracking an apparent convergence difference that was actually a different optimization problem. Now a CLAUDE.md rule.
4. **Matrix dimension scaling dominates per-iteration cost.** Lane change (330×480 M matrix) vs LQ (56×56) explains the 80× per-iteration difference through cubic LU scaling.
5. **PR descriptions rot fast.** Every push should update the description. Stale claims (old benchmark numbers, missing file lists) erode reviewer trust. Now a CLAUDE.md rule.
6. **Re-run benchmarks after code changes.** Even "equivalent" refactors can change Symbolics expression trees, producing different iteration counts. Always verify numbers match the committed code.

### Action Items for Next PR
- [ ] Implement skip-K-in-line-search optimization (bead `kmv`) — 1.63× speedup verified
- [ ] Investigate CSE in Symbolics.jl `build_function` (bead `zkn`)
- [ ] Re-run allocation benchmarks on nonlinear problem (bead `ahv`)

---

## PR: proposal/flexible-callsite-interface (bead mh7)

**Date:** 2026-02-11
**Commits:** 4
**Tests:** 955 passing (34 new)

### Summary

API design proposal evaluating flexible call-site interface for different use patterns (scripting, optimization loops, interactive). After thorough analysis of the existing API, implemented two targeted, backward-compatible improvements: Vector-based parameter passing and iteration callbacks.

### TDD Compliance
**Score: Strong (9/10)**
- What went well: Wrote failing tests first (commit 1), then implemented (commit 2). Clean red-green progression. All 34 new tests were written before any implementation code.
- What could improve: Test for "callback residuals decrease" needed a fix after the green phase — the initial test made assumptions about the callback timing that didn't hold for 1-iteration convergence. This was a test design issue, not a TDD violation.
- Verifiable Solution: Test file created and committed before any `src/` changes. Verified with `git log --oneline`.

### Clean Code Practices
**Score: Strong (9/10)**
- What went well:
  - `_to_parameter_dict` is a minimal 2-method helper with identity dispatch for Dict (zero overhead on existing code paths)
  - Callback is a simple `Union{Nothing, Function}` kwarg — no new types or abstractions
  - All changes are additive (union types on signatures, new kwargs) — no existing code modified
- What could improve: The `Union{Dict, AbstractVector{<:AbstractVector}}` type in 4 function signatures is slightly verbose. Could consider a type alias, but that adds complexity for little gain.

### Clean Architecture Practices
**Score: Strong (9/10)**
- What went well:
  - Changes follow existing patterns exactly (same `something()` override pattern, same kwarg threading)
  - Callback invocation is a single 3-line block in the iteration loop — minimal intrusion
  - No new dependencies, no new files in src/

### Commit Hygiene
**Score: Good (8/10)**
- What went well:
  - 4 commits with clear separation: (1) failing tests, (2) implementation + test fix, (3) test tier config, (4) retrospective + PR
  - Each commit is self-contained
- What could improve: Implementation commit (2) bundles both features + test fix. Could have been 3 commits: vector params, callback, test fix.

### CLAUDE.md Compliance
**Score: Strong (9/10)**
- [x] TDD followed strictly (red-green-refactor)
- [x] Tolerances at 1e-6 or tighter
- [x] Full test suite passed (955/955)
- [x] PR description includes use case analysis
- [x] Retrospective written before PR
- [x] Bead status updated
- [x] Backward compatibility verified

### Beads Created
None — this is a self-contained proposal.

### Key Learnings

1. **Most "flexible interface" requests don't need new abstractions.** The existing API was already well-designed. The actual friction points were minor convenience gaps (Dict vs Vector, no callback hook), not architectural problems.
2. **Callbacks are more useful than result-type wrappers.** Considered a `SolverResult` wrapper type but rejected it — callbacks give users full control over what to track without imposing a fixed structure.
3. **Union types with identity dispatch are the lightest-weight way to add input flexibility.** `_to_parameter_dict(d::Dict) = d` has zero runtime cost.
4. **Proposal PRs benefit from thorough analysis before coding.** Spending 60% of the time reading existing code and experiments prevented over-engineering.

### Action Items for Next PR
- [ ] If this proposal is accepted, update README.md examples to show the new Vector syntax
- [ ] Consider adding callback support to QPSolver (currently only NonlinearSolver, since QPSolver has no iteration loop)

---

*Template for future retrospectives:*

```markdown
## PR: [branch-name]

**Date:** YYYY-MM-DD
**Commits:** N
**Tests:** N passing

### Summary
[1-2 sentences]

### TDD Compliance
**Score: X/10**
- What went well:
- What went wrong:
- Improvement:
- Verifiable Solution:

### Clean Code Practices
**Score: X/10**
- What went well:
- What went wrong:
- Improvement:
- Verifiable Solution:

### Clean Architecture Practices
**Score: X/10**
- What went well:
- What went wrong:
- Improvement:
- Verifiable Solution:

### Commit Hygiene
**Score: X/10**
- What went well:
- What went wrong:
- Improvement:
- Verifiable Solution:

### CLAUDE.md Compliance
**Score: X/10**
- What went well:
- What went wrong:
- Improvement:
- Verifiable Solution:

### Beads Created
[List]

### Key Learnings
[Numbered list]

### Action Items for Next PR
- [ ] Item 1
- [ ] Item 2
```

---

## PR #86: perf/sparse-mn-solve (bead 0ip)

**Date:** 2026-02-09
**Commits:** 2
**Tests:** 481 passing (31 new)

### Summary

Investigated sparse M\N solve for `compute_K_evals`. Added `use_sparse::Bool` flag through the full solver stack. Found sparse solve gives 2-11x speedup for large M matrices (>100 rows) but hurts small matrices. Comprehensive benchmarking across 2/3/4-player chains with sparsity analysis and block structure investigation.

### TDD Compliance

**Score: Good (8/10)**

- **What went well:**
  - Tests written first (`db8abc6` Add sparse M\N solve investigation tests) before implementation (`490683b` Add use_sparse flag)
  - 31 new tests covering numerical equivalence, sparsity analysis, timing, and flag validation
  - Tests verify sparse and dense give identical results to machine epsilon (~1e-15)

- **What could improve:**
  - Only 2 commits — the test commit likely included some implementation scaffolding. Finer commits (test file alone, then implementation, then benchmarks) would be cleaner.

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Flag defaults to `false` preserving backward compatibility
  - Threaded cleanly through entire stack (compute_K_evals → run_nonlinear_solver → NonlinearSolver → solve)
  - Thorough docstrings updated at each level
  - PR description is exemplary — detailed tables, clear recommendation, technical notes

- **What could improve:**
  - Global flag applies to all players equally — a per-player adaptive approach would be better (addressed by follow-up bead sp1)

### Clean Architecture Practices

**Score: Good (8/10)**

- Flag propagation follows existing patterns (same as `use_armijo`)
- No new dependencies in core package (SparseArrays only in test)
- Investigation correctly identified that block elimination is not viable (irregular sparsity)

### Commit Hygiene

**Score: Fair (6/10)**

- Only 2 commits for a non-trivial change. Could have been split:
  1. Test file with failing tests
  2. Implementation of use_sparse flag
  3. Benchmark results and documentation
- Process was interrupted by API stall before retrospective could be written

### CLAUDE.md Compliance

- [x] TDD followed (tests first)
- [x] Tolerances tight (1e-15 verification)
- [x] PR description complete with Summary, Changes, Testing, Changelog
- [ ] Retrospective not written (stalled before completion — written retroactively here)

### Key Learnings

1. Sparse UMFPACK LU beats dense for M matrices >100 rows with <5% fill, but loses for small matrices due to symbolic analysis overhead
2. `sparse(M) \ sparse(N)` is not supported in Julia — must use `sparse(M) \ N` (dense RHS)
3. KKT Jacobian sparsity is structural (independent of operating point), making it predictable

### Action Items for Next PR

- [x] bead sp1: Replace global `use_sparse::Bool` with adaptive `:auto`/`:always`/`:never` that selects per-player based on leader vs leaf
- [x] Benchmark Nash vs Stackelberg chain structures to validate adaptive strategy

---

## PR #87: perf/adaptive-sparse-solve (bead bhz)

**Date:** 2026-02-09 (original), 2026-02-10 (merge with main + re-verification), 2026-02-11 (final merge after PR #106 landed)
**Commits:** 7 (4 original + 1 merge with main + 1 retrospective + 1 final merge)
**Tests:** 951 passing (30 new from this PR)

### Summary

Replaced the global `use_sparse::Bool` flag with an adaptive `use_sparse::Union{Symbol,Bool}=:auto` that selects per-player: sparse LU for non-leaf players (leaders with large M matrices from follower KKT conditions), dense solve for leaf players (small M, no sparse overhead). Merged with main (which had 20 PRs integrated via #99) and resolved 4 conflict files. All benchmarks re-run post-merge to verify behavior.

### TDD Compliance

**Score: Excellent (10/10)**

- **What went well:**
  - Failing tests committed first (`5b34e3f`) — 4 errors, 1 failure confirmed RED phase
  - Implementation committed second (`11da4da`) — all 30 new tests pass (GREEN phase)
  - Tests cover: Symbol mode acceptance, numerical equivalence, Bool backward compatibility, graph structure validation, invalid symbol error, and 5-player chain
  - Clean red-green-refactor cycle followed exactly

- **What could improve:**
  - Nothing — TDD was strictly followed throughout

### Clean Code Practices

**Score: Good (9/10)**

- **What went well:**
  - `Union{Symbol,Bool}` with Bool→Symbol normalization preserves full backward compatibility
  - Per-player decision uses existing `is_leaf(graph, ii)` — no new helpers needed
  - `ArgumentError` for invalid symbols with clear error message
  - Docstrings updated at all 4 levels: compute_K_evals, run_nonlinear_solver, NonlinearSolver, solve/solve_raw
  - Default changed from `false` to `:auto` as recommended by CLAUDE.md ("prefer adaptive defaults over global flags")

- **What could improve:**
  - The `mode` variable could be named `sparse_strategy` for clarity, but `mode` is fine in context

### Clean Architecture Practices

**Score: Good (9/10)**

- **What went well:**
  - Change is localized: only 3 src files touched (nonlinear_kkt.jl, types.jl, solve.jl)
  - No new dependencies — `is_leaf` already available from utils.jl
  - Decision logic stays in compute_K_evals where the M\N solve happens (not leaked upward)
  - Follows the existing `use_armijo` pattern for kwarg propagation

---

## PR #90: perf/optimize-pmcp (beads s6u-a, s6u-b)

**Date:** 2026-02-09
**Commits:** 6
**Tests:** 485 passing (35 new)

### Summary

Two-part bead: profiled ParametricMCPs usage (s6u-a), then implemented buffer pre-allocation optimizations (s6u-b). Found ParametricMCP is already well-cached; pre-allocated z_trial, param_vec, J, F, z0 buffers. Impact modest (1-2% allocation reduction) because dominant allocations are in compute_K_evals, not buffer creation.

### TDD Compliance

**Score: Good (8/10)**

- **What went well:**
  - Tests written first (commit `f60b9d4` "TDD RED") before implementation
  - 35 new tests verifying bit-identical results with buffer reuse
  - Clean red-green progression across commits

- **What could improve:**
  - s6u-a stalled before completing writeup (watchdog killed it)

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Backward compatible: QPSolver buffers are optional kwargs, standalone calls allocate as before
  - Used `mcp_obj.parameter_dimension` instead of extra `compute_K_evals` call (good cleanup)
  - Honest assessment: PR clearly states impact is modest and points to where real gains are

### Commit Hygiene

**Score: Good (8/10)**

- **What went well:**
  - 4 focused commits following TDD cycle:
    1. Failing tests (RED)
    2. Implementation (GREEN)
    3. Benchmarks
    4. Retrospective
  - Each commit has a clear purpose and descriptive message

- **What could improve:**
  - The implementation commit touches 3 files — could arguably be split by file, but they're tightly coupled (changing the type signature requires all three)

### CLAUDE.md Compliance

**Score: Excellent (9/10)**

- [x] CLAUDE.md reviewed at PR start
- [x] TDD mandatory — strictly followed
- [x] Test tolerances 1e-10 (tighter than 1e-6 minimum)
- [x] Full test suite run (511 pass)
- [x] Bead created and tracked
- [x] Pre-merge retrospective written
- [x] "Prefer adaptive defaults over global flags" rule followed — default is `:auto`

### Beads Created
- `bhz` — Adaptive sparse M\N solve (this PR)

### Key Learnings

1. **:auto is a good default.** For the 3-player chain end-to-end, :auto (3604μs) beats both :always (3615μs) and :never (3816μs) at the full solve level.
2. **For deeper chains, :always wins at compute_K_evals level.** With 5 players (4 leaders, 1 leaf), the single dense leaf doesn't offset sparse overhead. But at the full solve level, the difference is small.
3. **Nash games have zero M\N solve cost.** All players are roots, so compute_K_evals is a no-op (~0.4μs). Mode doesn't matter.
4. **Union{Symbol,Bool} with normalization is clean.** Converting Bool to Symbol early avoids branching downstream and maintains full backward compatibility.
5. **Merge conflicts are manageable when changes are localized.** The adaptive sparse feature only touches 3 src files, making conflict resolution straightforward even after 20 PRs landed on main.
6. **All modes produce bit-identical results (sol_diff = 0.00e+00).** Numerical equivalence verified across all problem sizes and hierarchy structures.

### Post-Merge Benchmark Results (2026-02-10)

| Problem | Solver | Structure | :never | :always | :auto | Best? |
|---------|--------|-----------|--------|---------|-------|-------|
| Nash 3P | Nonlinear | Flat | 0.4μs | 0.4μs | 0.4μs | tie |
| Chain 3P (T=3,s=2) | NL (K only) | Hub | 1606μs | 1008μs | 1371μs | :always |
| Chain 3P (T=3,s=2) | NL (e2e) | Hub | 3816μs | 3615μs | 3604μs | :auto |
| Chain 3P (T=3,s=2) | QP | Hub | ~80μs | N/A | N/A | N/A |
| Chain 4P (T=5,s=4) | NL (K only) | Chain | 6529μs | 5337μs | 5786μs | :always |
| Chain 4P (T=5,s=4) | NL (e2e) | Chain | 114ms | 93ms | 98ms | :always |
| Chain 5P (T=3,s=2) | NL (K only) | Chain | 11968μs | 9668μs | 10760μs | :always |
| Chain 5P (T=3,s=2) | NL (e2e) | Chain | 44ms | 38ms | 42ms | :always |

**Key finding**: `:auto` wins at the full solve level for smaller problems (3-player) where the overhead-per-call matters more. For larger problems, `:always` wins because the sparse advantage for leaders outweighs the leaf dense saving. `:auto` is always between `:always` and `:never`, never the worst.

### Final Merge Notes (2026-02-11)

PR #106 (in-place M/N) landed on main before this PR. Merged main into this branch — 3 conflicts in `src/nonlinear_kkt.jl` (docstring + function signature), all straightforward: combined adaptive sparse kwargs with in-place buffer kwargs. 951/951 tests pass. 3-expert code review posted as PR comment (0 issues). Base changed from `perf/inplace-mn-strategy-a` to `main`.

### Action Items for Next PR

- [ ] Consider size-based threshold (use sparse only when M rows > 100) as an alternative/complement to topology-based selection
- [ ] Profile memory allocation differences between sparse and dense paths

---

## PR #90: perf/optimize-pmcp (beads s6u-a, s6u-b)

**Date:** 2026-02-09
**Commits:** 6
**Tests:** 485 passing (35 new)

### Summary

Two-part bead: profiled ParametricMCPs usage (s6u-a), then implemented buffer pre-allocation optimizations (s6u-b). Found ParametricMCP is already well-cached; pre-allocated z_trial, param_vec, J, F, z0 buffers. Impact modest (1-2% allocation reduction) because dominant allocations are in compute_K_evals, not buffer creation.

### TDD Compliance

**Score: Good (8/10)**

- **What went well:**
  - Tests written first (commit `f60b9d4` "TDD RED") before implementation
  - 35 new tests verifying bit-identical results with buffer reuse
  - Clean red-green progression across commits

- **What could improve:**
  - s6u-a stalled before completing writeup (watchdog killed it)

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Backward compatible: QPSolver buffers are optional kwargs, standalone calls allocate as before
  - Used `mcp_obj.parameter_dimension` instead of extra `compute_K_evals` call (good cleanup)
  - Honest assessment: PR clearly states impact is modest and points to where real gains are

- 6 commits with logical separation: profiling → tests → NL buffers → QP buffers → cleanup → benchmarks
- Each commit is self-contained and leaves tests passing

### CLAUDE.md Compliance

- [x] TDD followed
- [x] PR description thorough with benchmark data
- [ ] Retrospective not written by bead (written retroactively)

### Key Learnings

1. Profiling before optimizing is essential — avoided wasting time on ParametricMCP caching (already done)
2. Buffer pre-allocation has diminishing returns when the dominant allocation source is elsewhere (compute_K_evals)
3. Two-part bead (profile then implement) worked well as a pattern

### Action Items for Next PR

- [ ] In-place compute_K_evals (bead mnip/PR #89) addresses the dominant allocation source

---

## PR #91: feature/progress-bar (bead udn)

**Date:** 2026-02-09
**Commits:** 1
**Tests:** 462 passing (6 new)

### Summary

Added `show_progress::Bool=false` option to NonlinearSolver that prints a formatted iteration table (iter, residual, alpha, time) and convergence summary. Threaded through full solver stack. Disabled by default.

### TDD Compliance

**Score: Fair (6/10)**

- **What went well:**
  - 6 tests cover parameter acceptance, output verification, result identity, default behavior

- **What could improve:**
  - Single commit bundles tests + implementation — should be at least 2 (tests first, then impl)

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Follows existing option pattern (`something()` override, stored in options NamedTuple)
  - Disabled by default — no behavioral change for existing code
  - Output format is clean and informative

### Commit Hygiene

**Score: Poor (4/10)**

- Single commit for the entire feature. Should have been: (1) failing tests, (2) implementation, (3) optional formatting polish

### CLAUDE.md Compliance

- [x] TDD followed (tests exist)
- [ ] Commit granularity rule not followed (1 commit)
- [ ] Retrospective not written by bead (written retroactively)

### Key Learnings

1. Simple features still benefit from multi-commit discipline — even a 1-file change should separate tests from implementation

### Action Items for Next PR

- [ ] Enforce multi-commit minimum even for simple features

---

## PR: docs/documenter-setup (#98)

**Date:** 2026-02-09
**Commits:** 2
**Tests:** 450 existing + 11 new docs build tests (all passing)

### Summary

Set up Documenter.jl infrastructure for API documentation. Created docs/ directory with Project.toml, make.jl build script, index.md landing page, and api.md API reference. Added CI workflow for GitHub Pages deployment and a test to verify the docs build.

### TDD Compliance

**Score: Strong (9/10)**

- **What went well:**
  - Wrote `test/docs_build_test.jl` FIRST, confirming RED phase (6 failures)
  - Created docs infrastructure to satisfy tests (GREEN phase)
  - All 11 tests pass, confirming the build works end-to-end

- **What could improve:**
  - The test uses a subprocess to run `docs/make.jl`, which is slightly indirect. Could potentially use `Documenter.makedocs` directly in-process for tighter integration, but the subprocess approach better matches how the build is actually invoked.

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Minimal, focused files with clear purposes
  - API reference organized by logical category (Types, Solvers, KKT, etc.)
  - Used `@autodocs` for internal functions to avoid missing-docs warnings without writing new docstrings
  - `warnonly=[:cross_references]` handles array-notation docstrings cleanly

- **Minor issue:**
  - Cross-reference warnings from `gs[i](z)` in docstrings are cosmetic but noisy. A future PR could escape these in docstrings with backticks.

### Clean Architecture Practices

- Dependencies point correctly: docs depend on the package, not vice versa
- `docs/Project.toml` uses `[sources]` to reference the local package
- CI workflow is independent from the test CI workflow

### Commit Hygiene

**Score: Good (9/10)**

- 2 focused commits:
  1. Docs infrastructure (Project.toml, make.jl, index.md, api.md)
  2. Test and CI workflow
- Each commit is self-contained and leaves the repo in a working state
- Descriptive commit messages explain what and why

### CLAUDE.md Compliance

- [x] TDD followed (red-green-refactor)
- [x] Full test suite verified (450 tests passing)
- [x] PR created with full description
- [x] Commits are logical and focused
- [x] Bead status updated
- [x] Used existing docstrings only (no new docstrings written)

### Key Learnings

1. Documenter.jl parses array-index notation in docstrings (e.g., `Js[i](zs...)`) as markdown links. `warnonly=[:cross_references]` is the standard workaround.
2. GitHub's OAuth tokens don't have `workflow` scope by default — pushing via SSH bypasses this for workflow file changes.
3. `@autodocs` with `Public = false` is a clean way to include documented internal functions without manually listing each one.

### Action Items for Next PR

- [ ] Consider escaping array-index notation in docstrings with backticks (e.g., `` `gs[i]` `` instead of `gs[i]`) to eliminate cross-reference warnings

---

## PR: perf/inplace-mn-strategy-a

**Date:** 2026-02-10
**Commits:** 3
**Tests:** 950 passing (29 new)

### Summary

Implemented in-place M/N evaluation with pre-allocated buffers (Strategy A) for the nonlinear solver's hot path in `compute_K_evals()`. This avoids allocating new matrices on every call to `M_fns[ii]()` and `N_fns[ii]()` during Newton iterations.

### TDD Compliance

**Score: Excellent (10/10)**

- Wrote failing tests FIRST (commit 1: 11 tests, all failing)
- Implemented feature to make all tests pass (commit 2: 29 tests passing)
- Clean Red-Green-Refactor cycle followed precisely
- No implementation code written before tests existed

### Clean Code

**Score: Good (8/10)**

- Functions remain single-purpose
- `inplace_MN` flag threaded cleanly through the full API stack
- Used `var"M_fns!"` Julia naming convention for in-place function dicts
- Pre-allocated buffers have clear ownership (stored in setup_info)
- Minor deduction: test helper functions duplicated from test_nonlinear_solver.jl (acceptable for test isolation)

### Clean Architecture

**Score: Good (8/10)**

- Changes localized to 3 source files with clear separation:
  - `nonlinear_kkt.jl`: core logic (setup + compute)
  - `types.jl`: constructor threading
  - `solve.jl`: API threading
- Default `false` preserves backward compatibility

### Commit Hygiene

**Score: Excellent (9/10)**

- 3 focused commits: RED → GREEN → test tier registration
- Each commit leaves codebase in working state (except RED, which is intentionally failing)
- Commit messages describe why, not just what

### Key Learnings

1. `SymbolicTracingUtils.build_function` with `in_place=true` writes into a matrix buffer directly — the buffer must match the flattened output shape, and the function signature is `fn!(output, input)`.
2. The benchmark shows allocation reduction scales with problem size: 24% for small LQ → 82% for large lane change, confirming M/N allocation dominates for larger problems.
3. Speedups are substantial (4-7x) and consistent across all problem sizes tested.

### Action Items for Next PR

- [ ] Consider making `inplace_MN=true` the default in a follow-up PR after broader testing
- [ ] Profile remaining allocations in the in-place path to find further optimization opportunities

---

## PR: perf/inplace-mn-strategy-a (buffer relocation follow-up)

**Date:** 2026-02-10
**Commits:** 2 (on same branch as above)
**Tests:** 950 passing

### Summary

Moved M_buffers/N_buffers allocation out of `setup_approximate_kkt_solver()` and into the solve path. Buffers are now created lazily in `compute_K_evals()` when `inplace_MN=true`, with `run_nonlinear_solver()` pre-allocating them once and passing through. This makes buffers a solve-time implementation detail rather than a setup-time concern.

### TDD Compliance

**Score: Excellent (10/10)**

- RED: Wrote failing test asserting `!hasproperty(setup_info, :M_buffers)` — failed as expected
- GREEN: Removed buffers from setup_info, added lazy allocation in compute_K_evals — all 950 tests pass
- Clean single-cycle TDD

### Clean Code

**Score: Excellent (9/10)**

- Buffers are no longer leaked into setup_info (separation of concerns)
- `MN_buffers` kwarg is optional — standalone `compute_K_evals` calls work without it
- `run_nonlinear_solver` creates buffer dicts once and reuses across iterations
- get!() pattern provides clean lazy allocation

### Commit Hygiene

**Score: Excellent (9/10)**

- 2 commits: RED (failing test) → GREEN (implementation)
- Each commit is small and focused

### CLAUDE.md Compliance

- [x] TDD followed strictly
- [x] Full test suite verified (950 pass)
- [x] PR description to be updated
- [x] Retrospective recorded

### Key Learnings

1. Named tuple fields in Julia are structural — removing a field from a NamedTuple is a clean breaking change that tests catch immediately
2. The `get!()` pattern with a do-block is ideal for lazy Dict initialization in hot loops

---

## PR: perf/inplace-mn-strategy-a (PR #107 merge — remove inplace_MN flag)

**Date:** 2026-02-11
**Commits:** 1 (merge commit)
**Tests:** 921 passing

### Summary

Merged Copilot's PR #107 (copilot/sub-pr-106) into the Strategy A branch. This removes the `inplace_MN` flag entirely and makes in-place M/N evaluation the default and only code path. The out-of-place path is deleted. Post-merge benchmarks confirm performance is retained:

| Problem | Current (ms) | Main baseline (ms) | Speedup | Alloc Reduction |
|---------|-------------|-------------------|---------|-----------------|
| LQ 3-player chain | 0.42 | 1.76 | 4.2x | 19.0% |
| Pursuer-Protector-VIP | 12.16 | 78.53 | 6.5x | 70.7% |
| Lane Change (4-player) | 8277.21 | 41608.54 | 5.0x | 82.3% |

### TDD Compliance

**Score: N/A (merge, not new feature)**

- No new tests written (this was a merge/simplification, not new functionality)
- All existing 921 tests pass after merge conflict resolution
- Test references updated from M_fns/N_fns to M_fns!/N_fns! to match new API

### Clean Code

**Score: Good (8/10)**

- Removed dead code path (out-of-place M/N evaluation)
- Removed unnecessary flag (`inplace_MN`) from full API stack (types, solve, nonlinear_kkt)
- Deleted test_inplace_strategies.jl (tested the now-removed flag)
- Minor deduction: `var"M_fns!"` naming convention is ugly but unavoidable in Julia for `!` in identifiers

### Clean Architecture

**Score: Excellent (9/10)**

- API simplified — no more flag threading through 4 layers
- Buffers remain a solve-time implementation detail (lazy `get!()` allocation)
- Backward compatible: `compute_K_evals()` works with or without pre-allocated buffers

### Commit Hygiene

**Score: Adequate (6/10)**

- Single large merge commit combining conflict resolution + test fixes
- Could have been split into: (1) merge with conflict resolution, (2) test reference updates
- Justified by the fact this was a manual merge in a worktree during overnight script execution

### CLAUDE.md Compliance

- [x] Retrospective recorded
- [x] PR description updated with fresh benchmark numbers
- [ ] Benchmark script not committed (in gitignored debug/)
- [x] No `@test_broken` left on branch

### Key Learnings

1. **Always re-benchmark after merges that change code paths.** The original benchmarks compared two paths; after merging #107 the old path is gone, so fresh numbers are needed.
2. **Copilot PRs need testing before merge.** PR #107 had no CI and no reviews. The merge introduced two classes of breakage: (a) required kwargs without defaults, (b) removed NamedTuple fields referenced by tests.
3. **`get!()` with empty Dict default is the right pattern for optional pre-allocation.** Making buffers default to `Dict{Int,Matrix{Float64}}()` with lazy `get!()` gives zero-allocation reuse when buffers are passed, and correct-but-allocating behavior when they're not.
4. **Git worktree is essential for parallel work.** Merging in `/tmp/shg-worktree-106` while overnight script ran on main repo avoided any interference.
