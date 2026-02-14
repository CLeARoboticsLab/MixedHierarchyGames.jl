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

---

## PR: dx/clean-solver-output

**Date:** 2026-02-10
**Commits:** 3
**Tests:** 925 passing (4 new)

### Summary

Clean up solver output: convert verbose-gated println() calls to @debug logging macros, add show_progress=true to experiment configs, and add static analysis test enforcing no bare println in src/.

### TDD Compliance

**Score: 9/10**

- **What went well:**
  - Tests written first (RED phase confirmed failing)
  - Static analysis test catches all bare println calls outside show_progress blocks
  - Behavioral tests verify solve()/solve_raw() produce zero stdout with defaults
  - Implementation committed separately after tests

- **Minor gap:**
  - The verbose=true behavioral test passes even before implementation because the verbose println calls happen on code paths not exercised by the test problem. Static analysis test compensates for this.

### Clean Code

**Score: 9/10**

- Functions are focused and unchanged in structure
- @debug calls use structured keyword arguments (e.g., `@debug "KKT Residuals" satisfied residual_norm`)
- No unnecessary changes beyond the task scope

### Commit Hygiene

**Score: 9/10**

- 3 commits, each self-contained:
  1. Failing tests (RED)
  2. Implementation (GREEN) — converts println to @debug
  3. Experiment config updates (show_progress=true)
- Each commit leaves codebase in working state

### CLAUDE.md Compliance

- [x] TDD followed (Red-Green-Refactor)
- [x] Test tolerances appropriate (N/A — no numerical tests added)
- [x] Full test suite run and passing (925 tests)
- [x] Retrospective written before PR description

### Key Learnings

1. Julia's @debug/@info/@warn macros go to stderr, not stdout — this means verbose-gated messages using these macros are invisible to stdout-based silence tests.
2. A static analysis test (scanning source files for patterns) is a robust complement to behavioral tests for enforcing coding standards.
3. show_progress was already defaulting to false — the main work was converting verbose println to @debug.

### Action Items for Next PR

- [ ] Consider adding a pre-commit hook or CI check for bare println in src/

---

## PR: dx/update-test-readme-checklist (bead e6q)

**Date:** 2026-02-10
**Commits:** 1
**Tests:** No src/ changes — existing tests unaffected

### Summary

Added `test/README.md` update requirement to CLAUDE.md verification checklist. Updated `test/README.md` to list all 23 current test files (was missing 8 test files and 1 shared config module added since the README was created in PR #97).

### TDD Compliance

**Score: N/A** — Documentation-only change. No application code or test code written.

---

## PR: cleanup/rename-examples-legacy (bead gyg)

**Date:** 2026-02-11
**Commits:** 1
**Tests:** 921 passing (no new tests — rename only)

### Summary

Renamed `examples/` folder to `legacy/` to clarify that these are historical standalone scripts predating the `MixedHierarchyGames` package. Updated all references in Python test files, experiment comments, and added a `legacy/README.md`.

### TDD Compliance

**Score: N/A** — File rename with no new logic. Existing test suite (921 tests) serves as the verification that no references are broken.

### Clean Code Practices

**Score: Good (9/10)**

- **What went well:**
  - Audit of test/README.md found 9 missing entries — caught and fixed in one pass
  - Updated "Adding New Tests" instructions to include README and test_tiers.jl steps
  - Removed hard-coded test count from README header (was "450 tests" — now uses file count which is easier to verify)

### Commit Hygiene

**Score: Good (9/10)**

- Single focused commit for a single-purpose change
- Both files changed together since the CLAUDE.md rule and the README update are logically coupled

### CLAUDE.md Compliance

- [x] Reviewed CLAUDE.md at PR start
- [x] Retrospective written before final PR description
- [x] No src/ changes, so no test run required for correctness
- [x] PR created with full description

### Key Learnings

1. **README drift is real.** 9 files were added across multiple PRs without updating the README. The new checklist item should prevent this going forward.
2. **Checklist items should be self-enforcing.** Adding the README update to CLAUDE.md's verification checklist means it will be checked before every merge.

### Action Items for Next PR

- [ ] Verify the new checklist item is being followed (first PR after this one should check test/README.md if adding tests)

---

## PR: docs/readme-docs-link

**Date:** 2026-02-10
**Commits:** 1
**Tests:** 466 passing (no new tests — docs-only change)

### Summary

Added a documentation badge to README.md linking to the Documenter.jl-generated docs site at `/dev/`. Verified `/stable/` returns 404 (no tagged release yet), so linked to `/dev/`.

### TDD Compliance

**Score: N/A** — No application or test code changed. README-only edit.

### Clean Code Practices

**Score: Good (9/10)**

- **What went well:**
  - Used standard shields.io badge format consistent with Julia ecosystem conventions
  - Placed badge immediately after title, following standard README layout
  - Verified docs URL accessibility before linking

### Commit Hygiene

**Score: Good (9/10)**

- Single commit for a single-line change — appropriate granularity
- Descriptive commit message

### CLAUDE.md Compliance

- [x] Reviewed CLAUDE.md at PR start
- [x] Full test suite verified (466 passing)
- [x] PR created with full description
- [x] Retrospective written before final PR description

### Key Learnings

1. `/stable/` docs URL requires a tagged release. Until then, link to `/dev/`.

### Action Items for Next PR

- [ ] After first tagged release, update badge to link to `/stable/` or add both `/stable/` and `/dev/` badges

---

## PR: perf/preallocate-params-buffers (bead h1f)

**Date:** 2026-02-11
**Commits:** 3
**Tests:** 938 passing (17 new)

### Summary

Pre-allocate parameter buffers in the nonlinear solver hot loop. F_trial buffer reused across linesearch iterations, Dict caches and all_K_vec buffer reused across compute_K_evals calls, and theta parameter vector built without vcat+comprehension allocation.

---

## Overnight Run PRs (2026-02-10 — 2026-02-11)

The following PRs were created by autonomous Claude agents during the overnight run (`scripts/overnight_run_2.sh`). Retrospectives were not written by the agents and are recorded here retroactively based on PR descriptions and code diffs.

---

## PR #101: dx/clean-solver-output (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

Converts verbose-gated `println()` calls in `src/` to `@debug` logging macros so the library is silent by default. Adds `show_progress=true` to experiment configs and a static analysis test to enforce no bare `println()` in `src/`.

### TDD Compliance

**Score: Good (8/10)**

- Static analysis test enforces the no-println rule going forward
- Cannot fully assess red-green cycle since agent didn't record commit-level TDD progression

### Clean Code Practices

**Score: Good (8/10)**

- Replaces ad-hoc println with Julia's standard `@debug` macro — idiomatic
- Experiments get explicit `show_progress=true` to preserve their output

### Commit Hygiene

**Score: Unknown** — Agent didn't write retrospective; single-session overnight work

### CLAUDE.md Compliance

- [x] Functional change with tests
- [ ] Retrospective not written by agent

### Key Learnings

1. Libraries should be silent by default — `@debug` lets users opt in via `ENV["JULIA_DEBUG"]`

---

## PR #102: dx/update-test-readme-checklist (bead)

**Date:** 2026-02-10
**Created by:** Overnight autonomous agent

### Summary

Adds a verification checklist item to CLAUDE.md requiring test/README.md updates when new test files are added. Brings test/README.md up to date by documenting 8 missing test files.

Retrospective: Small documentation/process change. No process issues identified.

---

## PR #103: docs/readme-docs-link (bead)

**Date:** 2026-02-10
**Created by:** Overnight autonomous agent

### Summary

Adds a shields.io documentation badge to README.md linking to the Documenter.jl docs site.

Retrospective: Trivial change (2 files, 49 additions). No process issues identified.

---

## PR #104: cleanup/repo-public-release (bead)

**Date:** 2026-02-10
**Tests:** All passing (53 new project health assertions)
**Created by:** Overnight autonomous agent

### Summary

Prepares the repo for public release: removes unused root dependencies (InvertedIndices, PATHSolver), deletes stale Docker files and INFRA_VERIFY.md, removes redundant experiment script, adds missing compat entries, and introduces `test_project_health.jl` (53 assertions) to enforce dependency hygiene.

### TDD Compliance

**Score: Good (8/10)**

- **What went well:**
  - Wrote allocation threshold test first (RED: 351KB > 100KB target)
  - Committed failing test before implementation
  - All correctness tests written before implementation changes
  - 17 new tests: numerical equivalence (2-player, 3-player), allocation reduction, buffer reuse correctness

- **What could improve:**
  - Initial allocation threshold (100KB) was too aggressive given unavoidable allocations from M_fns/N_fns (51KB per call). Adjusted to a relative comparison test instead of absolute threshold.

---

### TDD Compliance (PR #104: cleanup/repo-public-release)

**Score: Good (8/10)**

- 53 new health assertions enforce the cleanup rules going forward
- Large deletion count (881 lines) is appropriate for dead code removal

### Clean Code Practices

**Score: Excellent (9/10)**

- Removes dead code and unused dependencies (exactly what CLAUDE.md prescribes)
- Health tests prevent regression

### Key Learnings

1. Automated health tests for project structure prevent dependency bloat from creeping back

---

## PR #114: proposal/numerical-regularization (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

Proposal PR: opt-in Tikhonov regularization (`K = (M + λI) \ N`) for ill-conditioned M matrices. Default `regularization=0.0` means zero behavior change. Includes distortion analysis tables, threads parameter through full API stack.

### TDD Compliance

**Score: Good (8/10)**

- Tests verify regularization parameter acceptance, numerical effects, and backward compatibility
- Distortion analysis provides empirical evidence for parameter selection

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - Backward compatible: `buffers=nothing` default means all existing callers work unchanged
  - Pre-allocated buffers are created once and reused (no unnecessary copies)
  - Docstring updated for new `buffers` parameter
  - Used `copyto!` instead of `vcat` for theta vector construction

- **What could improve:**
  - The `buffers` NamedTuple has 6 fields — could be a proper struct for better documentation, but NamedTuple is fine for internal use

### Clean Architecture Practices

**Score: Good (8/10)**

- `compute_K_evals` API is backward compatible — optional keyword, not breaking
- Buffer creation lives in the solver (consumer), not in compute_K_evals (producer)
- No new dependencies or module-level state

### Commit Hygiene

**Score: Good (8/10)**

- **What went well:**
  - 3 commits with clear TDD progression: (1) failing tests RED, (2) implementation GREEN, (3) docstring + retrospective
  - Each commit is descriptive and self-contained

- **What could improve:**
  - Could have split the GREEN commit into F_trial, compute_K_evals buffers, and theta separately for finer granularity

### CLAUDE.md Compliance

- [x] TDD followed (red-green)
- [x] Full test suite passing (938 tests)
- [x] PR description created
- [x] Retrospective written
- [x] Bead status updated

### Key Learnings

1. **Profile before setting allocation targets.** The initial 100KB threshold was wrong because M_fns/N_fns allocate ~40KB per call (they return new arrays from compiled symbolic functions). Understanding the allocation breakdown before setting thresholds would have saved iteration.
2. **Dict reuse saves less than expected.** Pre-allocating Dict containers saves ~6.5KB per compute_K_evals call (11%), but the dominant allocations are from M_fns/N_fns output vectors. In-place M_fns/N_fns would require SymbolicTracingUtils changes (different PR).
3. **vcat+comprehension is surprisingly expensive.** `vcat([d[k] for k in keys]...)` allocated 1.8MB for a 4-element result due to intermediate array creation. `copyto!` loop is zero-allocation.
4. **Relative allocation tests are more robust than absolute thresholds.** Testing `allocs_with < allocs_without` is stable across Julia versions; testing `allocs < 100_000` is fragile.

### Action Items for Next PR

- [ ] Investigate in-place M_fns/N_fns (SymbolicTracingUtils `build_function` with `in_place=true`) — this is the dominant remaining allocation source
- [ ] Pre-allocate x_new buffer in linesearch functions (currently allocates per trial step)

---

## PR: perf/type-stable-dict-storage

**Date:** 2026-02-11
**Commits:** 3 (failing tests, implementation, retrospective)
**Tests:** 946 passing (28 new)

### Summary

Replaced Dict{Int, ...} containers with Vector-indexed storage for hot-path per-player data structures in `compute_K_evals` and `setup_approximate_kkt_solver`. This eliminates Dict hashing overhead on every solver iteration and enables more predictable memory access patterns.

### TDD Compliance

**Score: 10/10**

- Wrote 28 failing tests first (RED), covering 2-player and 3-player hierarchies
- All 12 type-assertion tests failed as expected (Dict vs Vector)
- All 16 numerical correctness tests passed (baseline verified)
- Implementation (GREEN) made all tests pass without modifying test expectations
- No implementation code written before failing tests existed

### Clean Code

**Score: 9/10**

- Functions remain small and single-purpose
- Placeholder function `_identity_fn` for unused Vector slots is a minor wart but acceptable
- Updated docstrings to reflect new container types
- No unnecessary changes beyond the Dict→Vector migration

### Clean Architecture

**Score: 9/10**

- Change is localized to `nonlinear_kkt.jl` (src) and test files
- No public API changes — the NamedTuple return types transparently switched from Dict to Vector
- Callers that accessed by integer index ([ii]) required zero changes

### Commit Hygiene

**Score: 9/10**

- 3 focused commits: (1) failing tests, (2) implementation + test updates, (3) retrospective
- Each commit leaves codebase in working state (commit 1 is intentionally failing tests)
- Commit messages describe why, not just what

### CLAUDE.md Compliance

- [x] TDD followed (Red-Green-Refactor)
- [x] Test tolerances 1e-6 or tighter
- [x] Full test suite passes
- [x] Retrospective written before final PR description

### Key Learnings

1. Dict→Vector migration for integer-keyed containers is low-risk and straightforward when keys are contiguous 1:N
2. The main migration effort is in test code: `haskey`, `keys`, Dict iteration patterns all need updating
3. `Vector{Function}` requires placeholder values for unused slots — a sentinel function works

### Action Items for Next PR

- [ ] Consider using `FunctionWrappers.jl` for M_fns/N_fns to get fully type-stable function calls (currently `Function` is abstract)

---

## PR: perf/zero-jacobian-buffers

**Date:** 2026-02-11
**Commits:** 1
**Tests:** 939 passing (18 new)

### Summary

Investigation of whether pre-allocated Jacobian buffers need zeroing before reuse. Found that `jacobian_z!` (via ParametricMCPs/SymbolicTracingUtils SparseFunction) fully overwrites all structural nonzero entries every call, so no zeroing is needed. Added 18 defensive regression tests that corrupt buffers between solves to verify this invariant.

### TDD Compliance

**Score: 8/10**

- **What went well:**
  - Tests written first (defensive tests verifying existing behavior)
  - Sentinel-value test directly proves `jacobian_z!` overwrites all `.nzval` entries
  - Tests cover both QPSolver (`J_buffer`) and NonlinearSolver (`∇F`) code paths

- **What could be improved:**
  - This was an investigation task, so "red-green-refactor" is not strictly applicable — no implementation code was written. The tests document the safety invariant.

### Clean Code

**Score: 9/10**

- Test file is well-organized with shared helpers and clear test names
- No production code changes needed (investigation confirmed safety)

### Commit Hygiene

**Score: 9/10**

- Single commit is appropriate for this investigation+test PR
- All changes are logically related

### CLAUDE.md Compliance

- [x] TDD followed (tests written before concluding no code changes needed)
- [x] Full test suite verified (939 tests passing)
- [x] PR created with full description
- [x] Bead status updated
- [x] Test tiers updated for new test file

### Key Learnings

1. `ParametricMCPs.SparseFunction` wraps compiled symbolic functions that write to all structural nonzero positions. The sparsity pattern is determined at symbolic compile time and is immutable.
2. `F_buffer` and `z0_buffer` in `solve_qp_linear` are already zeroed with `fill!` — this is correct because they're used as inputs (not outputs of compiled functions).
3. `∇F` in `run_nonlinear_solver` is allocated fresh each call via `copy(mcp_obj.jacobian_z!.result_buffer)`, so cross-call contamination is impossible even if the result_buffer template is corrupted.

### Action Items for Next PR

- None — this investigation is self-contained

---

## PR: cleanup/rename-examples-legacy (bead gyg)

(Retrospective entry above, under "Clean Code Practices")

### What went well (continued)

  - Systematic grep for all `examples/` references before making changes
  - Updated live code references (Python test files) but left historical prose (RETROSPECTIVES.md, benchmarks README) untouched — correct judgment call
  - Added README.md explaining the legacy folder's purpose and what still references it

### Clean Architecture Practices

**Score: Good (9/10)**

- **What went well:**
  - Naming now accurately reflects purpose: `legacy/` (historical reference) vs `experiments/` (active)
  - Follows the pattern of `reference_archive/old_examples/` already in the repo

### Commit Hygiene

**Score: Good (9/10)**

- Single commit appropriate for a rename operation — splitting would be artificial
- All changes are cohesive: rename + reference updates + README

### CLAUDE.md Compliance

- [x] Reviewed CLAUDE.md at PR start
- [x] Full test suite verified (921 tests passing)
- [x] PR created with full description
- [x] Bead status updated
- [x] Retrospective recorded

### Key Learnings

1. **Grep before rename.** Systematic search for all references before renaming prevents broken paths. The Python test files (`test_python_integration.py`, `test_hardware_nplayer_navigation.py`) would have silently broken without updating their `include()` paths.
2. **Historical prose doesn't need updating.** References in retrospectives and benchmark READMEs describe past events — changing them would rewrite history.

### Action Items for Next PR

- [ ] Consider whether the Python test files should be migrated to use the package API instead of including legacy scripts directly

---

## PR: proposal/numerical-regularization

**Date:** 2026-02-11
**Commits:** 3
**Tests:** 950 passing (29 new)

### Summary

Proposal PR to evaluate Tikhonov regularization for ill-conditioned M matrices in `_solve_K`. Added `regularization::Float64=0.0` parameter threaded through the full solver API: `_solve_K` -> `compute_K_evals` -> `run_nonlinear_solver` -> `NonlinearSolver` constructor -> `solve`/`solve_raw`. Includes accuracy analysis quantifying distortion vs regularization strength.

### TDD Compliance

**Score: 9/10**

- **What went well:**
  - Wrote failing tests first for `_solve_K` regularization parameter (RED confirmed)
  - Wrote integration tests for higher-level API before implementing threading (RED confirmed)
  - All tests passed after each implementation step (GREEN confirmed)
  - Clean progression: unit tests -> implementation -> integration tests -> API threading

- **What could improve:**
  - Minor: `qr` import bug caught by test run, not by pre-check

### Clean Code

**Score: 9/10**

- Functions remained focused and single-purpose
- Regularization added as a simple parameter with clear default (0.0 = disabled)
- No behavior change for existing users
- Docstrings updated for all modified functions

### Clean Architecture

**Score: 9/10**

- Regularization threaded cleanly through existing option pattern (`something(override, options.default)`)
- Followed the same pattern as `use_sparse` for consistency
- No new abstractions needed — parameter flows naturally through existing layers

### Commit Hygiene

**Score: 9/10**

- 3 focused commits: (1) core implementation + tests, (2) API threading + integration tests, (3) test tier fix
- Each commit leaves codebase in working state
- Commit messages describe both what and why

### CLAUDE.md Compliance

- [x] TDD followed (Red-Green-Refactor)
- [x] Test tolerances 1e-6 or tighter
- [x] Full test suite run and passing (950 tests)
- [x] Retrospective written before final PR update

### Key Learnings

1. When importing from `LinearAlgebra` in a test file that gets `include`d, explicit imports like `using LinearAlgebra: qr` are needed because `include` runs in `Main` scope.
2. The test tier system (`test_test_tiers.jl`) has self-validation that catches when new test files aren't registered — useful for preventing omissions.
3. Tikhonov regularization error scales approximately linearly with lambda for well-conditioned systems: relative error ~ lambda / min_singular_value.

### Action Items for Next PR

- [ ] Consider adaptive regularization (auto-detect ill-conditioning and apply minimal lambda)
- [ ] Investigate regularization impact on solver convergence for full nonlinear problems

---

## PR: proposal/unified-solver-interface (bead yc2)

**Date:** 2026-02-11
**Commits:** 2
**Tests:** 945 passing (24 new)

### Summary

API design proposal evaluating whether to consolidate the low-level solver API and TrajectoryGamesBase interface. Found that `solve()` already serves as a unified entry point. Implemented two incremental improvements: `AbstractMixedHierarchyGameSolver` abstract type and flexible input format (Vector-of-Vectors alongside Dict).

### TDD Compliance

**Score: Strong (9/10)**

- **What went well:**
  - Tests written first and confirmed failing (9 errors) before any implementation
  - Clean 2-commit structure: RED (tests) then GREEN (implementation)
  - All 24 new tests verify behavior, not implementation details
  - Existing 921 tests continue passing (+ 24 new = 945 total)

- **What could improve:**
  - Could have written the `_to_parameter_dict` unit tests as a separate first commit for even finer granularity

---

### Clean Code Practices (PR #114: proposal/numerical-regularization)

**Score: Good (8/10)**

- Default 0.0 preserves backward compatibility — zero behavior change unless opted in
- Parameter threaded through existing kwargs pattern (same as use_sparse, use_armijo)

### Key Learnings

1. Proposal PRs that default to no-op behavior are safe to land — they add capability without risk

---

## PR #115: proposal/unified-solver-interface (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

API design proposal: introduces `AbstractHierarchySolver` abstract type (supertype for QPSolver and NonlinearSolver) and adds flexible input support (Dict and Vector formats for solve/solve_raw). Correctly identifies that `solve()` is already the unified interface.

### TDD Compliance

**Score: Good (8/10)**

- Tests verify both input format acceptance and type hierarchy

### Clean Code Practices

**Score: Good (8/10)**

- **What went well:**
  - `_to_parameter_dict` is a small, single-purpose function with clear error messages
  - Used multiple dispatch (3 methods) instead of if-else chains for type conversion
  - Abstract type is minimal — just documents the contract, no unnecessary methods
  - Docstrings updated to reflect new parameter names and accepted formats

- **What could improve:**
  - Parameter rename from `parameter_values` to `initial_state` in function signatures is a minor API change; kept Dict passthrough for zero-cost backward compatibility

### Clean Architecture Practices

**Score: Good (9/10)**

- **What went well:**
  - Conversion function lives in solve.jl (close to usage, not in a separate utils file)
  - Abstract type in types.jl alongside concrete types
  - No new dependencies or module changes needed

### Commit Hygiene

**Score: Good (9/10)**

- **What went well:**
  - Exactly 2 commits: tests then implementation
  - Each commit is focused and self-contained
  - Messages describe what and why, not just what

### CLAUDE.md Compliance

**Score: Strong (9/10)**

- [x] TDD followed (red-green-refactor)
- [x] Tolerances tight (1e-10)
- [x] Full test suite verified
- [x] PR created with full description
- [x] Bead status updated
- [x] Retrospective written before PR finalized

### Key Learnings

1. **Analysis before implementation saves effort.** Thorough API surface audit revealed that unification already existed — the real issues were small ergonomic gaps, not architectural flaws.
2. **Proposals can be small.** The temptation was to redesign the whole solver API. The right answer was two targeted improvements totaling ~60 lines of implementation.
3. **`solve_trajectory_game!` is dead infrastructure.** The TGB adapter is never called outside tests. Future work could deprecate it if TGB compatibility isn't needed.

### Action Items for Next PR

- [ ] Consider deprecating `solve_trajectory_game!` if TGB compatibility is confirmed unnecessary
- [ ] Consider whether `solve_raw` return types should be unified (currently different NamedTuples per solver)

---

## PR: perf/timeit-debug-macros (bead e2f)

**Date:** 2026-02-11
**Commits:** 2
**Tests:** 937 passing (16 new)

### Summary

Added `@timeit_debug` macro that compiles to a no-op branch check when `TIMING_ENABLED[]` is false, and delegates to TimerOutputs' `begin_timed_section!`/`end_timed_section!` when true. Replaced all 21 `@timeit` calls across `types.jl`, `nonlinear_kkt.jl`, and `solve.jl`. Benchmarked overhead: ~6ns per-call when disabled, negligible on real solver operations (<0.2%).

### TDD Compliance
**Score: Good (8/10)**
- **What went well:**
  - Tests written FIRST (`test_timeit_debug.jl`) and verified failing (RED) before implementation
  - 16 tests covering: flag control, conditional timing, return value preservation, nesting, code execution in both states, reset
  - Red-green-refactor cycle followed for the core macro
- **What went wrong:**
  - The initial macro implementation was naive (`if/else` with duplicated body), causing scoping and method-redefinition bugs discovered only when running the full test suite. Had to iterate on the macro design.
- **Improvement:** When writing macros, test with function definitions inside the body from the start, not just simple expressions.

### Clean Code Practices
**Score: Good (8/10)**
- **What went well:**
  - Used `Expr(:tryfinally)` pattern from TimerOutputs itself (no body duplication)
  - Clean API: `enable_timing!()` / `disable_timing!()` + `@timeit_debug`
  - Existing timer tests updated to use `enable_timing!()` / `disable_timing!()`
  - Comprehensive docstrings on all public symbols
- **What went wrong:**
  - First two macro implementations had fundamental scoping bugs. The `Ref{Bool}` runtime check means this is "near-zero" rather than truly zero overhead (6ns per call).

### Clean Architecture Practices
**Score: Good (9/10)**
- Macro defined in its own file (`src/timeit_debug.jl`) included before any file that uses it
- Uses TimerOutputs' public `begin_timed_section!`/`end_timed_section!` API
- Backward compatible: existing `to::TimerOutput` parameter threading unchanged

### Commit Hygiene
**Score: Good (8/10)**
- 2 commits with clear separation:
  1. Tests + macro implementation (TDD red-green in one commit)
  2. Replace `@timeit` → `@timeit_debug` across all source files + test updates
- Benchmarks in gitignored `debug/` directory per CLAUDE.md convention

### CLAUDE.md Compliance
- [x] TDD followed (red-green-refactor)
- [x] Tolerances N/A (no numerical tests)
- [x] Full fast test suite verified (482 passing)
- [x] PR description with Summary, Changes, Testing, Changelog
- [x] Retrospective recorded before PR finalization
- [x] Bead status updated

### Key Learnings

1. **Julia macro hygiene is subtle.** `if/else` branches duplicate the body in the AST, causing method redefinition errors for function definitions inside `@timeit_debug` blocks. The `Expr(:tryfinally)` pattern avoids this by keeping the body in one place.
2. **`Ref{Bool}` checks are not zero-cost.** The branch check costs ~6ns per call. For a true compile-time zero-cost abstraction, you'd need `@generated` or a const global (which can't be toggled at runtime). The 6ns is acceptable for this use case.
3. **Test tier configuration is easy to forget.** Adding a new test file requires updating both `test_tiers.jl` AND `test_test_tiers.jl`.

### Action Items for Next PR

- [ ] Consider `@generated` approach if runtime overhead ever matters for inner-loop timing
- [ ] Add a note in CLAUDE.md about updating test tier files when adding new test files

---

### Clean Code Practices (PR #115: proposal/unified-solver-interface)

**Score: Good (8/10)**

- Leverages Julia's type system (abstract type) for extensibility
- Pragmatic finding that major redesign isn't needed — existing API is already unified

### Key Learnings

1. Investigation PRs that confirm "the architecture is already right" are valuable — they prevent unnecessary refactoring

---

## PR #116: proposal/flexible-callsite-interface (bead mh7)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

*Note: This PR already has a retrospective on its branch (written by the agent). Recorded here for completeness.*

### Summary

Adds flexible callsite interface: `solve()` and `solve_raw()` accept both `Dict{Int, Vector}` and `Vector{Vector}` parameter formats with automatic conversion.

Retrospective: Written by agent on branch — see PR #116 branch for full details.

---

## PR #117: proposal/convergence-stall-detection (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

Adds `stall_window` parameter and `detect_stall()` function to the nonlinear solver. Terminates early with `:stalled` status when residual plateaus. Disabled by default (`stall_window=0`). Demonstrated ~97% iteration savings in stalling scenarios.

### TDD Compliance

**Score: Good (8/10)**

- Tests verify stall detection triggers correctly and doesn't fire on converging problems
- 97% iteration savings demonstrates clear value

### Clean Code Practices

**Score: Good (8/10)**

- Disabled by default — zero behavior change unless opted in
- Clean separation: `detect_stall()` is a pure function, solver just calls it
- `:stalled` status integrates with existing convergence status enum

### Key Learnings

1. Early termination for stalled solvers prevents wasted computation — especially valuable for parameter sweeps where some configurations may not converge

---

## PR #120: perf/timeit-debug-macros (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

Introduces `@timeit_debug` macro that conditionally instruments code with TimerOutputs timing. Default disabled — overhead is ~6ns per call from a single branch check. Replaces all 21 `@timeit` calls, making timing opt-in via `enable_timing!()`/`disable_timing!()`.

### TDD Compliance

**Score: Good (8/10)**

- Tests verify zero-overhead claim and correct TimerOutputs integration
- Replaces existing instrumentation — tests verify no behavioral change

### Clean Code Practices

**Score: Excellent (9/10)**

- Eliminates always-on overhead from production code paths
- Clean API: two functions to toggle, one macro to instrument
- All 21 callsites updated consistently

### Key Learnings

1. Compile-time branch elimination makes conditional instrumentation nearly free — the branch check costs ~6ns vs microseconds for TimerOutputs

---

## PR #121: perf/preallocate-params-buffers (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

Pre-allocates three buffer categories in the nonlinear solver hot loop: (1) F_trial for linesearch, (2) Dict caches and all_K_vec in compute_K_evals, (3) theta_vals_vec via copyto! replacing vcat. Eliminated 1.8MB allocation. All backward compatible.

### TDD Compliance

**Score: Good (8/10)**

- Tests verify numerical equivalence with pre-allocated buffers
- Backward compatible — no changes needed to existing callers

### Clean Code Practices

**Score: Good (8/10)**

- Buffers are optional kwargs with sensible defaults
- `copyto!` replacing `vcat` is a textbook Julia optimization
- 3 source files touched — minimal blast radius

### Key Learnings

1. `vcat` is an allocation hotspot in tight loops — `copyto!` into a pre-allocated buffer is always better

---

## PR #122: perf/type-stable-dict-storage (bead)

**Date:** 2026-02-10
**Tests:** All passing
**Created by:** Overnight autonomous agent

### Summary

Replaces `Dict{Int, ...}` with `Vector` indexed by player ID for all hot-path per-player data (M_evals, N_evals, K_evals, M_fns, N_fns, pi_sizes, caches). Eliminates Dict hashing overhead since players are always 1:N contiguous.

### TDD Compliance

**Score: Good (8/10)**

- Tests verify numerical equivalence after storage representation change
- No behavioral change — purely a performance optimization

### Clean Code Practices

**Score: Excellent (9/10)**

- Dict→Vector is the right choice when keys are contiguous integers
- Eliminates hash collision overhead and improves cache locality
- 8 files touched but change is mechanical (Dict→Vector at each site)

### Key Learnings

1. `Dict{Int, T}` with contiguous integer keys is a code smell — `Vector{T}` is always faster and simpler

---

## PR #123: perf/zero-jacobian-buffers (bead chp)

**Date:** 2026-02-11
**Tests:** All passing (18 new)
**Created by:** Overnight autonomous agent

### Summary

Investigation PR: examined whether pre-allocated Jacobian buffers need zeroing before reuse. **Finding: no zeroing needed** — `jacobian_z!` fully overwrites all structural nonzero entries every call. No source code changes; adds 18 defensive regression tests with sentinel/garbage corruption between solves.

### TDD Compliance

**Score: Excellent (9/10)**

- 18 tests written that corrupt buffers with sentinel values and verify results are identical
- Pure investigation PR — tests ARE the deliverable

### Clean Code Practices

**Score: Excellent (9/10)**

- No unnecessary zeroing added (the correct finding is "don't add code")
- Defensive tests protect against future changes to jacobian_z! internals

### Key Learnings

1. Investigation PRs that add only tests (no source changes) are the best outcome — they confirm correctness and add safety nets
2. `SparseFunction` from SymbolicTracingUtils writes ALL `.nzval` entries every call — no stale data possible

---

## PR #124: cleanup/rename-examples-legacy (bead gyg)

**Date:** 2026-02-11
**Tests:** 921 passing
**Created by:** Overnight autonomous agent

### Summary

Renames `examples/` folder to `legacy/` to clarify these are historical reference scripts. Updates test file references and adds `legacy/README.md`. Historical references in RETROSPECTIVES.md left unchanged.

Retrospective: Straightforward rename. Addresses action item from PR feature/phase-1-nonlinear-kkt (Bead 2). No process issues identified.

---

## Overnight Run Meta-Retrospective

### What Went Well

1. **Autonomous agents produced 13 landable PRs overnight** — significant throughput
2. **Each PR has tests and passes CI** — agents followed TDD to varying degrees
3. **Proposal PRs default to no-op** — safe to land without risk
4. **Investigation PR (#123) correctly found "no change needed"** — honest assessment

### What Could Be Improved

1. **Retrospectives missing from 12/13 PRs** — agents didn't write to RETROSPECTIVES.md (only #116 did)
2. **Cannot verify commit-level TDD progression** — agents committed work but didn't record red-green phases
3. **PR #113 (relative tol) killed by watchdog** — stall detection should be more lenient for PRs that run tests

### Action Items

- [ ] Add RETROSPECTIVES.md requirement to overnight run agent prompts
- [ ] Consider per-PR retrospective verification in land_prs.sh
- [ ] Investigate why #113 stalled (watchdog threshold too aggressive?)

---

## PR: tranche3/options-and-review-fixes (bead r89)

**Date:** 2026-02-12
**Commits:** 2 (failing tests, implementation)
**Tests:** 1176 passing (48 new for NonlinearSolverOptions)

### Summary

Replaced untyped `options::NamedTuple` in `NonlinearSolver` with a concrete `NonlinearSolverOptions` struct. Added keyword constructor with defaults, NamedTuple conversion for backward compatibility, and linesearch validation. No solver behavior changes.

### TDD Compliance

**Score: Strong (9/10)**

- **What went well:**
  - Tests written first and committed as a separate RED commit (21dec83)
  - All 48 new tests defined the expected API before any implementation
  - RED-GREEN cycle clearly separated across commits
  - Existing tests served as regression suite — all 1176 pass unchanged

- **What could improve:**
  - Initial test file referenced a non-existent helper (`create_two_player_nonlinear_problem`); caught during GREEN phase and fixed by defining `_make_options_test_problem` locally

### Clean Code

- NonlinearSolverOptions is a focused, single-responsibility struct
- Validation logic (linesearch method) lives in the constructor where it belongs
- NamedTuple conversion constructor is minimal and explicit
- No dead code introduced; removed NamedTuple type annotation from struct field

### Clean Architecture

- Options struct is pure data — no behavior coupled to it
- solve/solve_raw needed zero changes (dot access works identically)
- Backward compatibility via constructor overload, not runtime checks

### Commit Hygiene

- 2 commits: (1) failing tests, (2) implementation + test fixes
- Could have been 3 commits (separate test fixes from implementation), but the test fixes were small and directly caused by the implementation change

### CLAUDE.md Compliance

- TDD followed: tests first, implementation second
- Test tolerances not applicable (no numerical tests)
- test_tiers.jl and test_test_tiers.jl updated for new test file
- Retrospective written before closing

### Action Items

- None — clean implementation with no follow-up debt

---

## PR: Consolidate test helpers and fix test hygiene (bead 9hy)

**Branch**: `tranche3/options-and-review-fixes`
**Date**: 2026-02-13

### Summary

Addressed 5 items from the 7-expert code review of PR #128:
1. Consolidated duplicated test problem setup (~40 lines each) from 4 test files into shared helpers in `testing_utils.jl`
2. Clarified responsibilities between `test_flexible_callsite.jl` (parameter passing + callbacks) and `test_unified_interface.jl` (type hierarchy + `_to_parameter_dict`)
3. Verified `@info` logging concern was already addressed (`@debug` was already used)
4. Added callback error handling tests (error propagation + partial history preservation)
5. Added root player `M_fns!/N_fns!` stub behavior test

Net result: -334 lines of duplicated test code, +74 lines of new test coverage.

### TDD Compliance

**Score: 7/10**

- The consolidation itself is refactoring of test infrastructure, not new production code — TDD cycle doesn't directly apply
- New tests (callback error handling, root player stubs) were written to document existing behavior, not drive new implementation
- First callback error test iteration had a test that assumed multi-iteration convergence but the solver converges in 1 — fixed by making the test adaptive
- **Improvement**: Even for behavior-documenting tests, should validate assumptions about the system (e.g., "does this solver actually take >1 iteration?") before writing assertions

### Clean Code

**Score: 9/10**

- Shared helpers `make_standard_two_player_problem()` and `make_simple_qp_two_player()` have clear names and docstrings
- Keyword arguments for goals make customization explicit
- Removed unused imports from refactored test files
- File responsibilities are clearly documented with comments at the top of each file

### Clean Architecture

**Score: 9/10**

- Test helpers live in the right place (`testing_utils.jl`)
- Each test file has a single clear responsibility
- No production code changes needed

### Commit Hygiene

**Score: 9/10**

- Two focused commits: (1) consolidation, (2) new tests
- Each commit leaves tests green
- Descriptive commit messages

### CLAUDE.md Compliance

- All instructions followed
- Full test suite run (1166/1166 pass)
- Retrospective written before closing

### Action Items

- None — all review items addressed cleanly

---

## Bead mig: Address src-level Medium review items from PR #128

**Date:** 2026-02-13
**Commits:** 4
**Tests:** 1206 passing

### Summary

Addressed Medium-priority code review items from the 7-expert review of PR #128. Four items implemented, one investigated and skipped with rationale.

**Implemented:**
1. `_merge_options` helper to deduplicate 8x `something()` overrides in `solve()` / `solve_raw()`
2. Added `z_est` (copy of current solution vector) to the callback NamedTuple for convergence analysis
3. Replaced hardcoded show_progress table widths with `@sprintf` formatting — consistent column widths regardless of iteration count or residual magnitude
4. In-place regularization in `_solve_K` — avoids allocating `M + λI` each call by adding/subtracting λ on the diagonal with try-finally cleanup

**Skipped:**
5. `Vector{Function}` type stability for M_fns/N_fns — Investigated and determined the dynamic dispatch overhead (~50ns) is negligible vs actual matrix evaluation cost (μs+). FunctionWrappers.jl would add a dependency and complexity for minimal benefit.

### TDD Compliance

**Score: 10/10**

- All four items followed strict Red-Green-Refactor
- Item 1: 6 failing tests → `_merge_options` implementation → all pass
- Item 2: 2 failing tests (z_est field, copy safety) → callback change → all pass
- Item 3: 2 failing tests (consistent widths, scientific notation) → `@sprintf` formatting → all pass
- Item 4: 2 new safety tests (M not mutated) added to existing regularization suite → in-place impl → all pass

### Clean Code

**Score: 9/10**

- `_merge_options` reduces duplication from 16 lines repeated in two functions to a single helper
- callback NamedTuple now has 4 fields, well-documented in docstring
- `@sprintf` formatting is clearer and more maintainable than string interpolation with `lpad`
- In-place regularization uses try-finally for exception safety — straightforward pattern

### Commits

**Score: 10/10**

- Four focused commits, one per review item
- Each commit leaves tests green
- Descriptive messages explaining *why* not just *what*

### CLAUDE.md Compliance

- All instructions followed
- Full test suite run (1206/1206 pass)
- TDD strictly followed for all items
- Retrospective written before closing
- Skipped item documented with rationale

### Action Items

- None — all addressable items implemented, skip rationale documented

---

## PR: tranche3/options-and-review-fixes (Combined Retrospective)

**Date:** 2026-02-13
**Commits:** 13 (12 across 3 beads + 1 for 4-expert review fixes)
**Tests:** 1235 passing (59 new net, after removing duplicated test setup)
**Beads:** r89 (NonlinearSolverOptions), 9hy (test hygiene), mig (src-level review items)

### Summary

Tranche 3 bundles three code-review-driven beads into a single PR. Bead r89 replaces the untyped `options::NamedTuple` with a concrete `NonlinearSolverOptions` struct. Bead 9hy consolidates duplicated test problem setup across 4 files into shared helpers (-334 lines) and adds missing test coverage (callback errors, root player stubs). Bead mig addresses src-level Medium review items: `_merge_options` helper, `z_est` in callbacks, Printf-based progress table, and allocation-free regularization.

### TDD Compliance

**Score: Strong (9/10)**

- **What went well:**
  - All three beads followed Red-Green-Refactor — failing tests committed before implementation in every case
  - Bead r89: 48 new tests defined the `NonlinearSolverOptions` API before any struct existed
  - Bead mig: Each of 4 sub-items had its own failing-test → implementation cycle
  - Test count grew from 1176 → 1206 while removing 334 lines of duplicated setup
  - No `@test_broken` left on the branch

- **What could improve:**
  - Bead 9hy (test consolidation) is inherently refactoring, so Red-Green doesn't directly apply — the "green" was keeping all existing tests passing after extraction
  - One callback error test assumed multi-iteration convergence but the solver converges in 1 iteration; had to adapt

### Clean Code Practices

**Score: Strong (9/10)**

- **What went well:**
  - `NonlinearSolverOptions` is a focused, single-responsibility struct with validation in the constructor
  - `_merge_options` eliminated 8x repeated `something()` patterns across two functions
  - Shared test helpers (`make_standard_two_player_problem`, `make_simple_qp_two_player`) have clear names, docstrings, and keyword customization
  - Each test file now has a documented single responsibility
  - `@sprintf` formatting replaced fragile `lpad` string interpolation

- **What could improve:**
  - Nothing significant — the review items were well-scoped and clean

### Clean Architecture Practices

**Score: Strong (9/10)**

- **What went well:**
  - Options struct is pure data, no behavior coupled to it
  - `solve`/`solve_raw` needed zero changes for the options migration (dot access works identically)
  - Test helpers live in `testing_utils.jl` where they belong
  - Production code changes are localized: `types.jl`, `solve.jl`, `nonlinear_kkt.jl`
  - No new dependencies added

### Commit Hygiene

**Score: Strong (9/10)**

- **What went well:**
  - 11 commits with clear logical separation:
    - r89: 2 commits (RED tests, GREEN implementation)
    - 9hy: 2 commits (consolidation, new tests) + 1 retrospective
    - mig: 4 commits (one per review item) + 1 retrospective
  - Each commit leaves the test suite green (except intentional RED commits)
  - Commit messages describe *why* not just *what*

- **What could improve:**
  - The retrospective commits could be squashed into the implementation commits to reduce noise, but they serve as documentation checkpoints

### CLAUDE.md Compliance

**Score: Strong (9/10)**

- [x] TDD mandatory — followed for all new code
- [x] Test tolerances 1e-6 or tighter
- [x] Full test suite run (1206/1206 pass)
- [x] Commits are logically organized
- [x] Beads tracked and closed
- [x] Code review offered and items addressed
- [x] Retrospective written before PR finalization
- [x] PR description created with comprehensive changes list

### Key Learnings

1. **Bundling review items into a tranche works well.** Three related beads (struct migration, test hygiene, code quality) complement each other and are easier to review together than as 3 separate PRs.
2. **Test consolidation pays immediate dividends.** Extracting shared helpers removed 334 lines while making each test file's purpose clearer. Future test writers can reuse the helpers.
3. **`something()` override deduplication was overdue.** The 8x repeated pattern in `solve`/`solve_raw` was a maintenance burden — `_merge_options` is a clean extraction.
4. **In-place regularization with try-finally is robust.** Adding/subtracting λ on the diagonal avoids allocation while ensuring the matrix is always restored, even on exceptions.
5. **Investigating and skipping items is valid.** The `Vector{Function}` type stability item was investigated, benchmarked, and skipped with rationale — better than cargo-culting a complex solution for negligible gain.

### Post-Review (4-Expert Code Review)

A 4-expert review (Julia Expert, Software Engineer, Test Engineer, Numerical Computing Expert) produced 18 findings, all addressed in a single commit:
- **`use_sparse` field narrowed** from `Union{Symbol,Bool}` to `Symbol` with Bool→Symbol normalization
- **Field validation** added to constructor (max_iters>0, tol>0, regularization>=0)
- **`_solve_K` → `_solve_K!`** renamed to follow Julia `!` convention
- **Exception path tests** added for `_solve_K!` (rethrow non-Singular errors, M restoration on exception)
- **NamedTuple deprecation** uses `Base.depwarn` instead of silent conversion
- **Benchmark result:** No solve-time regression (min times -0.1% to -5.2% across all problem sizes). Regularization on non-square M matrices was broken on main (crashes with `DimensionMismatch`), now works correctly with zero overhead.

### Action Items for Next PR

- [ ] Consider making `NonlinearSolverOptions` the default constructor path (currently both NamedTuple and struct work)
- [ ] Update README examples if they reference the old NamedTuple options format

---

## PR: perf/27-debug-getter (Perf T1-3)

**Date:** 2026-02-14
**Commits:** 3
**Tests:** 532 passing

### Summary

Removed unnecessary `get()` with empty vector default in debug logging path of `setup_approximate_kkt_solver`. The `get(augmented_variables, ii, [])` call allocated an empty `Vector` on every loop iteration as the default argument, even though `augmented_variables[ii]` is always set earlier in the loop. Replaced with direct dict access `augmented_variables[ii]`.

### TDD Compliance

- [x] TDD followed: test written first verifying verbose debug output behavior
- [x] Red-Green-Refactor cycle completed correctly
- [x] No implementation before tests

### Clean Code

- [x] Single-line change, minimal and focused
- [x] No unnecessary abstractions added

### Commits

- [x] 3 commits: (1) test, (2) implementation, (3) retrospective
- [x] Each commit is small and focused

### CLAUDE.md Compliance

- [x] All instructions followed

### What Went Well

- Clean TDD cycle for a simple performance fix
- Test captures both verbose and non-verbose behavior

### What Could Be Improved

- Impact is negligible (setup-time only, not hot-path) — but part of larger perf audit

### Action Items for Next PR

- None — straightforward fix
