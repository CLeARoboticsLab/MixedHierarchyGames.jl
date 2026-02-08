# External Integrations

**Analysis Date:** 2026-02-07

## APIs & External Services

**Complementarity Problem Solver:**
- PATHSolver (Ferris-Munson solver) - Solves complementarity problems for KKT conditions
  - SDK/Client: PATHSolver.jl (Julia package wrapper)
  - License: Courtesy academic license, passed via `PATH_LICENSE_STRING` environment variable
  - Integration: Via ParametricMCPs.jl interface
  - Usage: `solve_with_path()` function in `src/solve.jl`
  - Platform constraint: x86_64 only (no ARM64 binaries available)

**Symbolic Computation:**
- Symbolics.jl ecosystem - Symbolic differentiation and expression generation
  - Integration: Direct symbolic KKT condition construction
  - Used in: `src/problem_setup.jl`, `src/qp_kkt.jl`, `src/nonlinear_kkt.jl`
  - No external API calls; local computation only

**Graph Database/Representation:**
- Graphs.jl - DAG representation of player hierarchy
  - No external service; Julia package only
  - Used in: `src/utils.jl` for topological ordering and hierarchy validation

## Data Storage

**Databases:**
- Not detected - No relational or NoSQL databases used

**File Storage:**
- JLD2 - HDF5-based serialization for experiment outputs
  - Stored locally in: `experiments/outputs/` directory
  - Used for: Benchmark results, problem snapshots, convergence data
  - Example: `experiments/benchmarks/` results saved to `.jld2` files
  - No cloud storage integration

**Caching:**
- None - Computations are deterministic; caching not implemented
- TimerOutputs provides in-memory profiling but not persistent caching

## Authentication & Identity

**Auth Provider:**
- None for solver - PATHSolver uses license string (not API token auth)
- Git authentication: Handled via SSH agent forwarding in Docker container
  - Docker mounts: `~/.ssh/` and `/run/host-services/ssh-auth.sock`
  - Enables CI/CD authentication without embedding credentials

**GitHub Integration:**
- GitHub CLI (gh command) - Installed in Docker container
- Used for: PR operations, GitHub API access
- Auth: SSH key passthrough from host machine
- CI: GitHub Actions workflows in `.github/workflows/`

## Monitoring & Observability

**Error Tracking:**
- Not detected - No external error tracking service (Sentry, etc.)
- Local error handling only; errors raised and logged to console

**Logs:**
- Console output - Standard Julia REPL output
- TimerOutputs provides structured timing output (stdout)
- Test framework prints test results to stdout
- No centralized logging service (ELK, CloudWatch, etc.)

**Performance Profiling:**
- TimerOutputs.jl - In-memory hierarchical timing
  - Exported results via `@timeit` macro
  - Used in: `src/solve.jl` for solver phase timing
  - Output: Human-readable console output with timings

## CI/CD & Deployment

**Hosting:**
- GitHub (repository and CI/CD)
- Docker Hub (optional; not currently configured)
- No cloud deployment platform (AWS, GCP, Azure) integration

**CI Pipeline:**
- GitHub Actions (`.github/workflows/ci.yml`)
- Triggers:
  - Manual: Workflow dispatch button
  - PR comment: `/run-ci` comment on PR triggers tests
- Steps:
  1. Checkout branch
  2. Setup Julia 1.x via julia-actions/setup-julia
  3. Cache dependencies (julia-actions/cache)
  4. Build package (julia-actions/julia-buildpkg)
  5. Run tests (julia-actions/julia-runtest)
  6. Process coverage (julia-actions/julia-processcoverage)
  7. Upload to Codecov with token
- Additional workflows: CompatHelper.yml, TagBot.yml for maintenance

**Codecov Integration:**
- Coverage reporting via `codecov/codecov-action@v4`
- Requires: `CODECOV_TOKEN` secret (GitHub Actions secret)
- Output: Coverage data uploaded after test run
- Config: `flags: unittests`, `name: codecov-umbrella`

## Environment Configuration

**Required env vars:**
- `PATH_LICENSE_STRING` - PATHSolver courtesy academic license (example: `1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0`)
- `ANTHROPIC_API_KEY` (optional) - Passed through docker-compose.yml if set on host for Claude integration

**Secrets location:**
- GitHub Actions: Stored as repository secrets (`.github/workflows/`)
  - `CODECOV_TOKEN` - Required for coverage uploads
- Docker: Hardcoded in `docker-compose.yml` (PATH_LICENSE_STRING)
  - Not recommended for production; should use secrets management

**Development Configuration:**
- `.JuliaFormatter.toml` - Code formatting rules
- `.gitignore` - Git ignore patterns (excludes `debug/`, binaries, etc.)
- `.dockerignore` - Docker build context exclusions

## Webhooks & Callbacks

**Incoming:**
- GitHub: PR comment webhook triggers CI (`/run-ci` pattern)
- No custom HTTP webhook endpoints exposed

**Outgoing:**
- Codecov: POST requests to codecov.io with coverage data
- GitHub: Via actions/checkout and github-script actions
- No other HTTP callbacks or webhook integrations

## Data Flow Architecture

**Problem Input:**
1. User defines game structure: hierarchy graph, costs, constraints, dynamics
2. Parameters passed to `solve()` or `solve_raw()` functions
3. Data: Cost functions, constraint functions, DAG structure (in-memory)

**Solver Processing:**
1. Symbolic KKT construction (Symbolics.jl)
2. MCP formulation (ParametricMCPs.jl)
3. Complementarity solve (PATHSolver via ParametricMCPs.jl)
4. Linear system solve (LinearSolve.jl)

**Output:**
1. Trajectories (states and controls per player)
2. Timing data (TimerOutputs)
3. Optional: Results saved to `.jld2` files in experiments

**No Network Communication:** All operations are local; no remote API calls during solving.

## Integration Points with TrajectoryGamesBase

**Interface Implementation:**
- Location: `src/solve.jl`, `src/types.jl`
- Functions: `solve_trajectory_game!()` for TrajectoryGamesBase interface compatibility
- Types: Converts internal solutions to `JointStrategy` and `OpenLoopStrategy`
- Enables: Composition with other TrajectoryGamesBase solvers and environments

## Dependency Management

**Root Project.toml** (`Project.toml`):
- Contains only core package dependencies
- Minimal: TrajectoryGamesBase, Symbolics, Graphs, PATHSolver, etc.
- No experiment-specific dependencies

**Experiments Project.toml** (`experiments/Project.toml`):
- Adds visualization: Plots, StatsPlots, LaTeXStrings
- Adds benchmarking: BenchmarkTools
- References local package via `[sources]`: `MixedHierarchyGames = {path = ".."}`
- Enables isolated experiment environment

---

*Integration audit: 2026-02-07*
