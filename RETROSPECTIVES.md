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
