# Claude Code Instructions

## Git

- Use merge instead of rebase
- Each task must have its own PR unless explicitly allowed by the user (ask if you think combining tasks is needed)
- Each significant chunk of work in a PR should be divided into commits
- Branches associated with merged PRs can be deleted locally (not on the remote repo) after merging

## Test-Driven Development (TDD)

ALWAYS follow this cycle:

1. Write a failing test that defines the feature
2. Write minimal code to make the test pass
3. Refactor only after tests are green
4. Never skip the red-green-refactor cycle

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
