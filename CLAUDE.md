# Claude Code Instructions

## Git

- Use merge instead of rebase
- Each task must have its own PR unless explicitly allowed by the user (ask if you think combining tasks is needed)
- Each significant chunk of work in a PR should be divided into commits
- Branches associated with merged PRs can be deleted locally (not on the remote repo) after merging

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

## Verification Checklist

Before marking work complete, verify the following:

### Code Quality
- [ ] All tests pass (`julia --project=. -e 'using Pkg; Pkg.test()'`)
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
