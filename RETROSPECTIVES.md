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

## PR: perf/adaptive-sparse-solve (bead bhz)

**Date:** 2026-02-09
**Commits:** 4
**Tests:** 511 passing (30 new)

### Summary

Replaced the global `use_sparse::Bool` flag with an adaptive `use_sparse::Union{Symbol,Bool}=:auto` that selects per-player: sparse LU for non-leaf players (leaders with large M matrices from follower KKT conditions), dense solve for leaf players (small M, no sparse overhead). Includes benchmark comparison across Nash, 3-player chain, 5-player chain, and larger dimension problems.

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

1. **:auto is the right default.** For the 3-player chain, :auto (1015μs) actually beats both :always (1050μs) and :never (1156μs) because it uses sparse only where it helps.
2. **For deeper chains, :always still wins slightly.** With 5 players (4 have leaders, only 1 leaf), the single dense leaf player doesn't offset the overhead of the auto-check. But the difference is small (1.09x).
3. **Nash games have zero M\N solve cost.** All players are roots with no leaders, so compute_K_evals is essentially a no-op (0.4μs). Mode doesn't matter.
4. **Union{Symbol,Bool} with normalization is clean.** Converting Bool to Symbol early avoids branching downstream and maintains full backward compatibility.

### Action Items for Next PR

- [ ] Consider size-based threshold (use sparse only when M rows > 100) as an alternative/complement to topology-based selection
- [ ] Profile memory allocation differences between sparse and dense paths
