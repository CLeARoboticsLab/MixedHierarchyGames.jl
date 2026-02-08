# Claude Code Instructions

## Starting a PR

- **Review this file (CLAUDE.md) at the start of every PR.** Skim all sections to refresh on requirements before writing any code.
- **Suggest PR splits if the task involves too many changes.** If a task spans multiple unrelated concerns (e.g., new feature + refactoring + documentation overhaul), propose splitting into separate PRs. Discuss with the user before proceeding with a large-scope PR.

## Git

- Use merge instead of rebase
- Each task must have its own PR unless explicitly allowed by the user (ask if you think combining tasks is needed)
- Each significant chunk of work in a PR should be divided into commits
- Branches associated with merged PRs can be deleted locally (not on the remote repo) after merging

### Pull Request Requirements

**Every `git push` MUST be accompanied by a PR description update.** The PR description is the source of truth for reviewers — if it's stale, the PR is not ready. Update the Changes list, Changelog, and any claims (test counts, file lists, findings) to match the actual state of the branch after every push.

When pushing a branch, always create or update the associated PR with a clear, auditable description:

1. **New branches**: Create a PR immediately after pushing with:
   - Summary of changes made
   - List of files modified with brief descriptions
   - Any breaking changes or migration notes
   - Testing performed

2. **Existing PRs**: When pushing additional commits to a branch with an existing PR:
   - Update the PR description to reflect new changes
   - Add a changelog section at the bottom showing what was added in each push
   - Keep the description current with the actual state of the branch

3. **PR Description Format**:
   ```markdown
   ## Summary
   [Brief description of the overall change]

   ## Changes
   - `path/to/file.jl`: [What changed and why]
   - `another/file.jl`: [What changed and why]

   ## Testing
   - [How the changes were tested]

   ## Changelog
   - [Date]: Initial PR with [features]
   - [Date]: Added [additional features]
   ```

## Test-Driven Development (TDD)

**TDD is mandatory for all new code.** No exceptions without explicit user approval.

### The Red-Green-Refactor Cycle

1. **RED**: Write a failing test that defines the expected behavior
2. **GREEN**: Write the minimal code to make the test pass
3. **REFACTOR**: Clean up the code while keeping tests green
4. **Repeat**: Never skip steps or reverse the order

### TDD Rules

- **Tests come first**: Do NOT write implementation code before a failing test exists
- **One behavior at a time**: Each test should verify one specific behavior
- **Run tests frequently**: After every change, verify the test status
- **Porting existing code**: Even when porting from reference implementations, write tests first that define the expected behavior, then port the code to satisfy those tests
- **Bug fixes**: Write a failing test that reproduces the bug before fixing it
- **Use `@test_broken` to define behavior early**: When planning multiple features, use `@test_broken` to sketch expected behavior before implementation. However, **`@test_broken` must never exist on main**. All broken tests must be resolved before merge unless the user gives explicit approval for an interface design PR.

### Violations

If you find yourself writing implementation code without a failing test:
1. Stop immediately
2. Delete or stash the implementation code
3. Write the test first
4. Then rewrite the implementation

## Testing Standards

- **Tolerance**: Test tolerances should be `1e-6` or tighter unless explicitly approved by the user. Do NOT loosen tolerances to make failing tests pass.
- **Numerical precision**: For linear algebra operations, expect machine precision (~1e-14 to 1e-16). Use `1e-10` for tight tests, `1e-6` for general correctness.
- **Failing tests**: If a test fails, investigate and fix the root cause. Never relax tolerances as a workaround.

## Code Quality

- **Dead code**: When encountering dead code (unused functions, unreachable branches, stub functions that throw errors), mark it for removal. Delete it if clearly unused, or flag for review if uncertain.
- **Duplicated code**: When encountering duplicated logic, mark it for consolidation. Extract shared functionality into helper functions or common modules.
- **No hidden scaling in shared functions**: Constants that affect behavior (weights, thresholds, scaling factors) must be explicit parameters or live in config, never buried inside utility functions.

## Planning

When creating implementation plans:

1. **Every plan must explicitly address TDD**: Include a section specifying:
   - What tests will be written first
   - Expected test structure (test file locations, test names)
   - Order of test-then-implement for each feature

2. **Plans must be granular enough for TDD**: Break features into testable units. If a feature can't be described as a test, break it down further.

3. **Include test verification steps**: After each implementation phase, the plan should specify running tests to confirm the red-green-refactor cycle completed correctly.

## Work Tracking

All work should be tracked in beads (`bd`), including when the user mentions additional work that doesn't immediately get done. Create a task for it.

- Use `bd create "Task title" --type task --body "Description"` for new tasks
- Use `bd dep add <child> <parent>` to set dependencies
- Use `bd update <id> --status in_progress` when starting work
- Use `bd close <id>` when work is complete

### Task Verification

Before beginning a task:
- Summarize what the task requires and what actions will be taken
- Wait for user confirmation before proceeding

Before closing a task:
- Summarize what was accomplished
- List any follow-up items or remaining concerns
- Wait for user confirmation before marking complete

## Code Reviews

When requested, perform code reviews from the perspective of specific expert roles. Available roles:

### Engineering Roles
- **Master Software Engineer**: Architecture, design patterns, code quality, maintainability, SOLID principles
- **Performance Engineer**: Algorithmic complexity, memory usage, profiling, optimization opportunities
- **Test Engineer**: Test coverage, edge cases, test design, mocking strategies

### Infrastructure & Operations
- **DevOps Engineer**: CI/CD, automation, deployment, monitoring, logging
- **Infrastructure Engineer**: Containerization, orchestration, resource management, scalability
- **Site Reliability Engineer (SRE)**: Reliability, observability, incident response, SLOs

### Security
- **Security Engineer**: Vulnerabilities, authentication, authorization, secrets management, OWASP
- **Penetration Tester**: Attack vectors, exploit potential, hardening recommendations

### Domain-Specific
- **Numerical Computing Expert**: Numerical stability, precision, algorithm correctness
- **Game Theory Expert**: Equilibrium concepts, solution quality, mathematical correctness
- **Julia Expert**: Idiomatic Julia, type stability, performance patterns, package conventions

### Documentation & UX
- **Technical Writer**: Documentation clarity, examples, API documentation
- **Developer Experience (DX)**: Onboarding, ergonomics, error messages, discoverability

When reviewing, clearly separate issues by severity (Critical, High, Medium, Low) and provide actionable recommendations.

## Debug Files

All debug scripts should live in the `debug/` folder, which is gitignored. Do not create debug files (e.g., `debug_*.jl`) in the repository root or other tracked locations.

## Experiments Structure

Each experiment in `experiments/` should follow this structure:

```
experiments/<name>/
├── config.jl          # Pure parameters: x0, G, N, T, Δt, costs, goals
├── run.jl             # Main entry point (concise, uses config + support)
├── support.jl         # Experiment-specific helpers (if needed)
└── README.md          # Description (optional)
```

### Guidelines

- **config.jl**: Pure data/parameters, no logic
- **run.jl**: Concise, delegates to support functions
- **No duplicate code**: Use shared utilities in `experiments/common/`
- **Generally useful code**: Consider adding to `src/`

See `experiments/README.md` for full documentation.

## Dependencies

The repository has multiple Project.toml files for different contexts:

- **Root `Project.toml`**: Contains minimal dependencies required for the core package. Keep this lean - only include what's necessary for `src/`.
- **`experiments/Project.toml`**: Contains dependencies needed for running experiments (e.g., Plots, visualization tools). Uses `[sources]` to reference the local MixedHierarchyGames package.
- **`test/Project.toml`** (if exists): Contains test-specific dependencies.

### Guidelines

- **Run experiments** with: `julia --project=experiments experiments/lq_three_player_chain/run.jl`
- **Run tests** with: `julia --project=. -e 'using Pkg; Pkg.test()'`
- **New experiment dependencies**: Add to `experiments/Project.toml`, not the root
- **New package dependencies**: Add to root `Project.toml` only if used by `src/`

## Verification Checklist

Before marking work complete, verify the following:

### Code Quality
- [ ] All tests pass (`julia --project=. -e 'using Pkg; Pkg.test()'` or `julia --project=. test/runtests.jl`)
- [ ] Experiments run successfully (`julia --project=experiments experiments/<name>/run.jl`)
- [ ] No new warnings introduced
- [ ] Type stability checked for performance-critical functions

### Documentation
- [ ] All code examples in README.md are tested and runnable
- [ ] All code examples in docstrings are tested and runnable
- [ ] New public functions have docstrings
- [ ] CHANGELOG.md updated for user-facing changes

### Git
- [ ] Commits are logically organized
- [ ] Commit messages are descriptive
- [ ] Branch is up to date with target branch

## Pre-Merge Retrospective

**Before landing any PR**, conduct a retrospective to identify improvements for future work. This is mandatory.

### Process Adherence Review

Ask these questions and document honest answers:

#### TDD Compliance
- [ ] Was TDD followed for all new code? (Red-Green-Refactor)
- [ ] Were there any instances where implementation was written before tests?
- [ ] If TDD was violated, why? What could prevent this next time?

#### Clean Code Practices
- [ ] Are functions small and single-purpose?
- [ ] Are names clear and intention-revealing?
- [ ] Is there any duplicated code that should be extracted?
- [ ] Are there any magic numbers that should be constants?
- [ ] Is error handling consistent and informative?

#### Clean Architecture Practices
- [ ] Are dependencies pointing inward (toward domain logic)?
- [ ] Is business logic separated from infrastructure concerns?
- [ ] Could any code be more easily tested with better structure?
- [ ] Are module boundaries clear and responsibilities well-defined?

#### Commit Hygiene
- [ ] Are commits small and focused on single changes?
- [ ] Does each commit leave the codebase in a working state?
- [ ] Are commit messages descriptive of *why*, not just *what*?
- [ ] Could any large commits be split for better reviewability?

#### CLAUDE.md Compliance
- [ ] Were all instructions in CLAUDE.md followed?
- [ ] If any were skipped, was there explicit user approval?
- [ ] Should any new instructions be added based on lessons learned?

### Improvement Identification

Document answers to:

1. **What went well?**
   - Practices that worked effectively
   - Patterns worth repeating

2. **What could be improved?**
   - Specific violations or shortcuts taken
   - Areas where quality suffered

3. **Action items for next PR:**
   - Concrete changes to make next time
   - New beads/tasks to create for technical debt

### Example Retrospective Entry

```markdown
## PR Retrospective: feature/phase-1-nonlinear-kkt

### TDD Compliance
- Partially followed. `evaluate_kkt_residuals` was implemented before tests.
- Corrected mid-PR by writing TDD tests retroactively.
- **Improvement**: Start each feature by writing test file first.

### Clean Code
- Some functions (run_nonlinear_solver) exceed 100 lines.
- Created Bead 6 to address this.

### Commits
- Several commits were large (addressing multiple review items).
- **Improvement**: Commit after each individual fix.

### Action Items
- [ ] Bead 6: Refactor run_nonlinear_solver
- [ ] Start next feature with test file creation
```

### When to Skip

This retrospective may be abbreviated (but never skipped) for:
- Trivial PRs (typo fixes, single-line changes)
- Emergency hotfixes (document debt and create follow-up tasks)

Even abbreviated retrospectives should note: "Retrospective: Trivial change, no process issues identified."

### Recording Retrospectives

All retrospectives must be recorded in `RETROSPECTIVES.md` at the repository root. This file serves as a learning diary across PRs. See the template in that file for required sections.
