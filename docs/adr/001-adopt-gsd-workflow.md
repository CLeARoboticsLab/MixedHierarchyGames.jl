# ADR-001: Adopt GSD (Get-Shit-Done) Workflow

**Date:** 2026-02-07
**Status:** Accepted

## Context

Over multiple Claude Code sessions on the `feature/timer-outputs-benchmarking-v2` PR, we hit recurring issues:

1. **Context rot** — 3 session resets due to context window exhaustion, requiring manual summaries and state reconstruction
2. **Stale state** — PR descriptions and benchmark numbers went stale after pushes; had to be caught by the user
3. **Gray area resolution** — A hidden 0.1 collision weight inside `smooth_collision_all` caused hours of debugging an apparent solver divergence that was actually a different optimization problem

These are systematic problems, not one-offs. We evaluated several AI-driven development tools:

## Tools Evaluated

### Loop/Workflow Tools

| Tool | Approach | Complexity | Best For |
|------|----------|-----------|----------|
| **[Ralph Wiggum](https://github.com/fstandhartinger/ralph-wiggum)** | Simple bash retry loop | Minimal | Autonomous task completion |
| **[GSD](https://github.com/glittercowboy/get-shit-done)** | Context engineering + spec-driven | Medium | Complex projects, context rot |
| **[GitHub Spec Kit](https://github.com/github/spec-kit)** | Spec scaffolding, agent-agnostic | Low-medium | Teams wanting structure |
| **[BMAD Method](https://github.com/bmad-code-org/BMAD-METHOD)** | 21 agents, 50+ workflows | High | Enterprise, large teams |

### IDE/Platform Tools

| Tool | Approach | Complexity | Best For |
|------|----------|-----------|----------|
| **[Kiro](https://kiro.dev/)** (AWS) | VS Code fork with spec workflow | Medium | AWS shops, IDE-native |
| **[Tessl](https://tessl.io/)** | Spec registry + framework | Medium | Library-heavy projects |
| **[Codev](https://www.co-dev.ai/)** | Fully autonomous SWE agent | Low (user side) | Autonomous bug fixes |

### Spectrum

```
Lightweight ←──────────────────────────────→ Heavyweight
Ralph Wiggum → GSD → Spec Kit → Kiro → BMAD
```

## Decision

**Adopt GSD** for the following reasons:

1. **Context engineering** — Keeps orchestrator at 30-40% context utilization by delegating to subagents. Directly addresses our context rot problem.
2. **Discuss phase** — Forces gray area resolution before implementation. Would have prevented the collision weight debugging saga.
3. **State management** — `PROJECT.md`, `STATE.md`, and phase files persist decisions across sessions. Complements our existing beads + RETROSPECTIVES.md.
4. **Right complexity level** — More structured than Ralph Wiggum (which is just retry), less enterprise overhead than BMAD (21 agents is overkill for our team size).
5. **Claude Code native** — Works with our existing tooling, installed via `.claude/` directory.

### Why not the others?

- **Ralph Wiggum**: Too simple — doesn't address gray area resolution or context budgeting
- **Spec Kit**: Good but agent-agnostic means less tight integration with Claude Code
- **BMAD**: Overkill for a 1-2 person research project
- **Kiro**: IDE lock-in (VS Code fork), we prefer terminal-based Claude Code
- **Tessl**: More relevant for library-heavy web projects, not numerical computing

## Consequences

- GSD commands installed in `.claude/` (gitignored, personal tooling)
- GSD state files (`PROJECT.md`, `ROADMAP.md`, `STATE.md`, `.planning/`) will be tracked in git
- Existing CLAUDE.md, beads, and RETROSPECTIVES.md continue as-is — GSD complements, doesn't replace
- New PRs should use `/gsd:discuss-phase` before `/gsd:plan-phase` for non-trivial work
