# Technology Stack

**Analysis Date:** 2026-02-07

## Languages

**Primary:**
- Julia 1.11 - Core implementation language for the MixedHierarchyGames package
- Markdown - Documentation and examples

**Build/Config:**
- TOML - Project and dependency configuration
- YAML - GitHub Actions CI/CD workflows
- Shell - Docker entrypoint and build scripts

## Runtime

**Environment:**
- Julia 1.11.7 (current version used in Manifest.toml)
- Minimum: Julia 1.9 (specified in Project.toml compat)
- Deployment: Docker container (linux/amd64 platform, uses julia:1.11 base image)

**Package Manager:**
- Pkg (Julia built-in) - Package dependency management
- Lockfile: `Manifest.toml` (present and tracked)

## Frameworks

**Core Scientific Computing:**
- TrajectoryGamesBase 0.3.11 - Trajectory game interface and utilities
- Symbolics 6.51.0 - Symbolic computation and automatic differentiation
- SciMLBase 2.87.0 - Scientific Machine Learning interface base

**Linear Algebra & Optimization:**
- LinearSolve 3.26.0 - Linear system solving with multiple backends
- PATHSolver 1.7.8 - Complementarity problem solver (requires license)
- ParametricMCPs 0.1.17 - Mixed complementarity problem formulation

**Graph Structures:**
- Graphs 1.13.1 - Directed acyclic graph (DAG) representations for hierarchy

**Numerical & Utilities:**
- BlockArrays 0.16.43 - Block matrix/vector operations
- StaticArrays - High-performance fixed-size arrays
- ForwardDiff - Forward-mode automatic differentiation
- LinearAlgebra - Julia standard library linear algebra
- SparseArrays - Sparse matrix operations

**Symbolic/Code Generation:**
- SymbolicTracingUtils 0.1.3 - Utility for symbolic tracing
- JLD2 - HDF5-based data serialization

**Profiling & Instrumentation:**
- TimerOutputs 0.5.29 - Hierarchical timing and benchmarking

**Testing:**
- Test (Julia stdlib) - Unit testing framework
- BenchmarkTools 1.6.3 - Performance benchmarking (extras/test)
- Infiltrator 5903a43b - Interactive debugging (extras/test)
- Revise 295af30f - Interactive code reloading (extras/test)

**Visualization (Experiments only):**
- Plots 1.x - High-level plotting interface
- StatsPlots - Statistical plotting recipes
- LaTeXStrings - LaTeX string support for labels

**Utilities:**
- FileIO 5789e2e9 - File format agnostic I/O
- FilePathsBase 48062228 - Cross-platform file path handling
- InvertedIndices 1.3.1 - Inverted index utilities
- ProgressBars 49802e3a - Console progress indication

## Key Dependencies

**Critical (Core Solver):**
- Symbolics 6.51.0 - Enables symbolic KKT condition construction and automatic differentiation
- PATHSolver 1.7.8 - Required for complementarity problem solving; **only provides x86_64 binaries** (constrains to linux/amd64 Docker platform)
- TrajectoryGamesBase 0.3.11 - Defines game interface for compatibility with broader ecosystem
- LinearSolve 3.26.0 - Flexible linear system solving backend (used for QP solver)

**Infrastructure:**
- ParametricMCPs 0.1.17 - MCP formulation and management
- BlockArrays 0.16.43 - Efficient block structured problems
- Graphs 1.13.1 - DAG representation and topological ordering for hierarchy

**Development:**
- TimerOutputs 0.5.29 - Performance profiling and timing analysis
- JLD2 - Data serialization for benchmark results and problem snapshots
- Infiltrator, Revise, BenchmarkTools - Interactive development and debugging

## Configuration

**Environment:**
- Julia project configuration: Root `Project.toml` (minimal, core only)
- Experiment configuration: `experiments/Project.toml` (heavier dependencies via `[sources]` reference)
- Code formatting: `.JuliaFormatter.toml` (style: default, indent: 4, margin: 100)
- Docker environment: `docker-compose.yml` with PATH solver license via `PATH_LICENSE_STRING`

**Build & Container:**
- Dockerfile: Multi-stage development container
  - Base: `julia:1.11` (forces linux/amd64 due to PATHSolver x86_64-only binaries)
  - Non-root user: `devuser` (UID 1000, GID 1000)
  - Git and GitHub CLI included
  - SSH agent forwarding support for Git operations
  - Secure sudo access limited to SSH socket permissions fix

**Docker Compose:**
- Two services: `dev` (interactive development), `test` (CI testing)
- Mounts: Source code, git config, SSH keys, Claude auth
- Environment: `PATH_LICENSE_STRING` for solver, `ANTHROPIC_API_KEY` passthrough
- Working dir: `/workspace` (maps to repository root)

## Platform Requirements

**Development:**
- macOS/Linux host with Docker Desktop (ARM64 on Apple Silicon uses Rosetta emulation via --platform flag)
- Julia 1.9+ locally
- Git and GitHub CLI for workflow integration
- PATH solver license (courtesy license provided in docker-compose.yml)

**Production:**
- x86_64 Linux environment (required by PATHSolver binaries)
- Julia 1.9+
- ~100MB disk for Julia depot + dependencies
- PATH solver license when using PATHSolver-based solvers

**CI/CD:**
- GitHub Actions (ubuntu-latest runner)
- Julia Actions: setup-julia, cache, julia-buildpkg, julia-runtest, julia-processcoverage
- Codecov integration for coverage reporting
- Manual trigger via workflow dispatch or PR comment `/run-ci`

---

*Stack analysis: 2026-02-07*
