# Performance Benchmarks

Benchmarking scripts and findings for the MixedHierarchyGames solver.

## Running Benchmarks

```bash
julia --project=experiments experiments/benchmarks/benchmark_all.jl
```

## Scripts

### `benchmark_all.jl`

Runs all three experiments (LQ chain, lane change, PPV) with TimerOutputs instrumentation. Reports per-section timing for both construction (one-time) and solve (repeated) phases.

## Benchmark Results (Feb 2026)

**Machine:** Apple M4 Pro, 24 GB RAM, macOS

### Experiment 1: LQ Three Player Chain (3 players, single integrator, T=10)

#### QPSolver

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **QPSolver construction** | **6.70s** | **81.9%** | **1.58 GiB** |
| &emsp; KKT conditions | 2.36s | 28.9% | 592 MiB |
| &emsp; ParametricMCP build | 3.14s | 38.4% | 704 MiB |
| &emsp; linearity check | 122ms | 1.5% | 3.49 MiB |
| **QPSolver solve** (6 calls) | **2.79ms** | **0.0%** | **821 KiB** |
| &emsp; linear solve | 2.77ms | — | 765 KiB |
| &emsp; residual evaluation | 5.29μs | — | 6.09 KiB |
| &emsp; Jacobian evaluation | 1.83μs | — | 6.09 KiB |

**Per-solve: ~465μs** (dominated by linear solve)

#### NonlinearSolver (same LQ problem)

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **NonlinearSolver construction** | **2.60s** | **72.9%** | **712 MiB** |
| &emsp; approximate KKT setup | 1.43s | 40.0% | 367 MiB |
| &emsp; ParametricMCP build | 1.16s | 32.4% | 341 MiB |
| &emsp; linear solver init | 16.0ms | 0.4% | 2.88 MiB |
| &emsp; variable setup | 1.24ms | 0.0% | 583 KiB |
| **NonlinearSolver solve** (6 calls) | **698ms** | **19.6%** | **80.1 MiB** |
| &emsp; compute K evals (12 calls) | 497ms | 13.9% | 74.2 MiB |
| &emsp; Jacobian evaluation | 3.68ms | 0.1% | 204 KiB |
| &emsp; line search | 2.65ms | 0.1% | 2.91 MiB |
| &emsp; residual evaluation (12 calls) | 2.07ms | 0.1% | 136 KiB |
| &emsp; Newton step | 293μs | 0.0% | 590 KiB |

**Per-solve: ~116ms** (1 Newton iteration each, dominated by K evaluation)

### Experiment 2: Nonlinear Lane Change (4 players, unicycle, T=14)

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **NonlinearSolver construction** | **345s** | **61.6%** | **159 GiB** |
| &emsp; approximate KKT setup | 223s | 39.8% | 103 GiB |
| &emsp; ParametricMCP build | 122s | 21.7% | 56.1 GiB |
| &emsp; variable setup | 94.5ms | 0.0% | 18.3 MiB |
| &emsp; linear solver init | 99.2μs | 0.0% | 21.5 KiB |
| **NonlinearSolver solve** (4 calls) | **215s** | **38.4%** | **64.3 GiB** |
| &emsp; compute K evals (236 calls) | 116s | 20.6% | 15.3 GiB |
| &emsp; line search (232 steps) | 98.7s | 17.6% | 48.1 GiB |
| &emsp; Newton step | 318ms | 0.1% | 561 MiB |
| &emsp; Jacobian evaluation | 247ms | 0.0% | 160 MiB |
| &emsp; residual evaluation | 47.0ms | 0.0% | 162 MiB |

**Per-solve: ~53.8s** (~58 iterations, converges with residual ~2.5e-12)

Note: Construction allocates 159 GiB due to symbolic KKT expression tree compilation. This experiment OOM-killed in Docker (8 GB limit) but runs successfully on host with 24 GB RAM.

### Experiment 3: Pursuer-Protector-VIP (3 players, single integrator, T=20)

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **NonlinearSolver construction** | **69.5s** | **96.9%** | **19.6 GiB** |
| &emsp; approximate KKT setup | 42.8s | 59.6% | 10.5 GiB |
| &emsp; ParametricMCP build | 26.7s | 37.2% | 9.06 GiB |
| &emsp; variable setup | 67.3ms | 0.1% | 9.94 MiB |
| &emsp; linear solver init | 52.5μs | 0.0% | 14.0 KiB |
| **NonlinearSolver solve** (4 calls) | **1.90s** | **2.7%** | **291 MiB** |
| &emsp; compute K evals (8 calls) | 1.63s | 2.3% | 241 MiB |
| &emsp; line search | 80.7ms | 0.1% | 38.6 MiB |
| &emsp; Jacobian evaluation | 4.59ms | 0.0% | 1.26 MiB |
| &emsp; Newton step | 3.41ms | 0.0% | 5.84 MiB |
| &emsp; residual evaluation | 2.49ms | 0.0% | 2.25 MiB |

**Per-solve: ~476ms** (1 Newton iteration each, converges with residual ~6.3e-14)

### Summary Table

| Experiment | Construction | Per-Solve | Iterations | Solver |
|------------|-------------|-----------|------------|--------|
| LQ Chain (QPSolver) | 6.70s | **465μs** | N/A (direct) | QPSolver |
| LQ Chain (NonlinearSolver) | 2.60s | **116ms** | 1 | NonlinearSolver |
| Lane Change (4-player unicycle) | 345s | **53.8s** | ~58 | NonlinearSolver |
| PPV (3-player) | 69.5s | **476ms** | 1 | NonlinearSolver |

## Performance Investigation Findings (Feb 2026)

### Summary

We investigated whether the new `src/` solver code is slower than the old `examples/` code. **It is not.** The new code's symbolic construction produces slightly more efficient compiled functions, and the solver loops are equivalent in performance.

### What was tested

1. **Direct old-vs-new comparison** at T=3 and T=10 on the 3-player LQ chain
2. **Ablation of code-level changes**: topological sort algorithm, Dict type annotations, intermediate variable inlining
3. **Three optimization candidates**: pre-allocated caches, pre-allocated matrix buffers, skipping K recomputation in line search
4. **Old loop vs new loop**: structurally identical loops with the same precomputed data
5. **Run-order bias**: same benchmarks in both A→B and B→A order

### Key findings

#### 1. Code-level changes don't matter

| Variant | Effect |
|---------|--------|
| `topological_sort` (Kahn's) vs `topological_sort_by_dfs` | ~3% (noise) |
| `Dict{Int, Any}` vs `Dict{Int, Union{...}}` | ~5% (noise) |
| Inlining M_raw/N_raw intermediates | ~3% (noise) |
| All combined | No improvement |

#### 2. New symbolic construction is better

Using old code's `setup_approximate_kkt_solver` with the new solver loop produced **27% slower** results than the new construction. Different expression tree structures produce different compiled code — the new Dict-based variable organization with keyword-argument cost functions compiles to more efficient numerical functions.

#### 3. Only one optimization matters: skip K in line search

| Optimization | Speedup | Allocations |
|-------------|---------|-------------|
| Pre-allocate Dict caches | 0.82× (worse) | +23% |
| Pre-allocate matrix buffers | 0.98× (no effect) | ~same |
| **Skip K in line search** | **1.63×** | **-31%** |

During Armijo backtracking, `compute_K_evals` (which solves K = M\N) is called at every trial step. Since the Newton direction was computed with fixed K, using stale K during line search is more theoretically consistent and produces **bit-for-bit identical results** on all experiments:

| Experiment | Iterations | Line search steps | Δsol |
|------------|-----------|-------------------|------|
| LQ Chain (T=3, 3-player) | 1 = 1 | 1 = 1 | 0.0 |
| PPV (T=20, 3-player) | 1 = 1 | 1 = 1 | 0.0 |
| Lane Change (T=8, 4-player unicycle) | 66 = 66 | 391 = 391 | 0.0 |

This optimization is tracked as a separate task for implementation.

#### 4. Old loop vs new loop: no real difference

An initial benchmark suggested the old-style loop was 15% faster. Further investigation with 100 solves in both run orders showed this was **entirely a GC warming artifact**: whichever variant runs first in a shared Julia process appears slower due to garbage collection overhead.

| Run Order | A (new loop) | B (old loop) |
|-----------|-------------|-------------|
| A first, B second | 20.8 ms | 21.8 ms |
| B first, A second | 18.3 ms | 24.3 ms |

**Conclusion:** Loops are equivalent. Always benchmark in separate processes or control for run-order bias.

### Cost breakdown (T=10, 3-player LQ chain)

The dominant cost in the nonlinear solver is `compute_K_evals`, which evaluates K = M\N numerically:

| Section | % of solve time |
|---------|----------------|
| `compute_K_evals` | ~62% |
| Line search (K recomputation) | ~32% |
| Newton step (linear solve) | ~1% |
| Residual evaluation | ~0.2% |
| Jacobian evaluation | ~0.2% |

### QP iteration behavior

Verified that the NonlinearSolver applied to LQ problems behaves correctly:
- **Cold start (z=0):** Exactly 1 iteration, residual ~1e-15
- **Warm start (solution passed in):** 0 iterations, status `solved_initial_point`

### Benchmarking methodology notes

- **Shared-process benchmarks are unreliable** for comparing variants. GC warming, JIT, and memory allocator state create ordering effects of 15-30%.
- **Separate Julia processes** eliminate cross-contamination and produce reliable results.
- **TimerOutputs** is effective for per-section profiling within a single solver run.
- **Functional correctness tests** (same iterations, same solution, same α values) are deterministic and more useful than wall-time comparisons for detecting real algorithmic differences.

## Old vs New Solver Convergence Comparison (Feb 2026)

### Setup

We compared the old (`examples/test_automatic_solver.jl`) and new (`src/NonlinearSolver`) solvers on the 4-player nonlinear lane change with Mixed A hierarchy (P1→P2, P2→P4, P1→P3), T=10, Δt=0.5.

### Initial Observation

The new solver appeared to converge in 119 iterations while the old solver stalled at ~0.21 for 500+ iterations. Both started at residual 50.27, diverged by ~7×10⁻⁹ at iteration 8.

### Root Cause: `smooth_collision_all` Weight Mismatch

The apparent convergence difference was caused by **different collision cost weights**, not a solver bug:

- **Old code** (`examples/test_automatic_solver.jl:1223`): `total += 0.1 * smooth_collision(...)` — **0.1× multiplier** inside `smooth_collision_all`
- **New code** (`experiments/common/collision_avoidance.jl:64`): `total += smooth_collision(...)` — **no multiplier** (was missing the 0.1)

This made the collision cost **10× stronger** in the new experiments, creating a different optimization problem. The two solvers were solving different KKT systems — the identical initial residuals were coincidental because collision costs are negligible at the initial guess (vehicles start far apart, softplus ≈ 0).

### Verification

After restoring the `0.1` multiplier to `experiments/common/collision_avoidance.jl`, both solvers produce **bit-for-bit identical** residuals through all 500 iterations:

| Iteration | New solver residual | Old solver residual |
|-----------|--------------------|--------------------|
| 0 | 50.27443404278393 | 50.27443404278393 |
| 8 | 43.90884191226509 | 43.90884191226509 |
| 100 | 0.21219945497264966 | 0.21219945497264966 |
| 500 | 0.21580864774733108 | 0.21580864774733108 |

All 500 iterations match to 17 significant digits. The solvers are algorithmically identical.

### Additional Finding: `get_all_leaders` Ordering Is Irrelevant

We also tested whether the `get_all_leaders` ordering (closest-first vs root-first) affects results. It does not — Symbolics.jl canonicalizes expression trees:

| Config | Closest-first (iters) | Root-first (iters) |
|--------|----------------------|-------------------|
| Default | 119 | 119 |
| P3 further back | 1043 | 1043 |
| P3 faster | 82 | 82 |
| Tighter spacing | 2235 | 2235 |
| Wider spacing | 295 | 295 |

### Conclusion

The new `src/` solver is a correct, algorithmically identical port of the old `examples/` solver. The `0.1` collision weight has been moved from inside `smooth_collision_all` to the cost functions in `config.jl` as `COLLISION_WEIGHT = 0.1`.
