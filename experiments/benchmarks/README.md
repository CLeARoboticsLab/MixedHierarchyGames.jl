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
| **QPSolver construction** | **6.85s** | **83.0%** | **1.58 GiB** |
| &emsp; KKT conditions | 2.40s | 29.2% | 592 MiB |
| &emsp; ParametricMCP build | 3.24s | 39.2% | 704 MiB |
| &emsp; linearity check | 119ms | 1.4% | 3.49 MiB |
| **QPSolver solve** (6 calls) | **3.76ms** | **0.0%** | **821 KiB** |
| &emsp; linear solve | 3.74ms | — | 765 KiB |
| &emsp; residual evaluation | 2.04μs | — | 6.09 KiB |
| &emsp; Jacobian evaluation | 1.62μs | — | 6.09 KiB |

**Per-solve: ~627μs** (dominated by linear solve)

#### NonlinearSolver (same LQ problem)

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **NonlinearSolver construction** | **2.78s** | **73.3%** | **712 MiB** |
| &emsp; approximate KKT setup | 1.57s | 41.5% | 367 MiB |
| &emsp; ParametricMCP build | 1.18s | 31.3% | 342 MiB |
| &emsp; linear solver init | 17.5ms | 0.5% | 2.88 MiB |
| &emsp; variable setup | 1.11ms | 0.0% | 583 KiB |
| **NonlinearSolver solve** (6 calls) | **742ms** | **19.6%** | **80.1 MiB** |
| &emsp; compute K evals (12 calls) | 525ms | 13.8% | 74.2 MiB |
| &emsp; Jacobian evaluation | 3.67ms | 0.1% | 204 KiB |
| &emsp; line search | 2.61ms | 0.1% | 2.91 MiB |
| &emsp; residual evaluation (12 calls) | 2.15ms | 0.1% | 136 KiB |
| &emsp; Newton step | 264μs | 0.0% | 590 KiB |

**Per-solve: ~124ms** (1 Newton iteration each, dominated by K evaluation)

### Experiment 2: Nonlinear Lane Change (4 players, unicycle, T=14)

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **NonlinearSolver construction** | **339s** | **62.6%** | **159 GiB** |
| &emsp; approximate KKT setup | 221s | 40.7% | 103 GiB |
| &emsp; ParametricMCP build | 118s | 21.8% | 56.0 GiB |
| &emsp; variable setup | 85.2ms | 0.0% | 18.3 MiB |
| &emsp; linear solver init | 100μs | 0.0% | 21.5 KiB |
| **NonlinearSolver solve** (4 calls) | **202s** | **37.4%** | **56.5 GiB** |
| &emsp; compute K evals (208 calls) | 114s | 21.0% | 14.3 GiB |
| &emsp; line search (204 steps) | 87.9s | 16.2% | 41.5 GiB |
| &emsp; Newton step | 340ms | 0.1% | 496 MiB |
| &emsp; Jacobian evaluation | 211ms | 0.0% | 140 MiB |
| &emsp; residual evaluation | 36.3ms | 0.0% | 143 MiB |

**Per-solve: ~50.6s** (~51 iterations, converges with residual ~6.6e-11)

Note: Construction allocates 159 GiB due to symbolic KKT expression tree compilation. This experiment OOM-killed in Docker (8 GB limit) but runs successfully on host with 24 GB RAM.

### Experiment 3: Pursuer-Protector-VIP (3 players, single integrator, T=20)

| Section | Time | % of Total | Allocations |
|---------|------|-----------|-------------|
| **NonlinearSolver construction** | **68.4s** | **96.9%** | **19.6 GiB** |
| &emsp; approximate KKT setup | 41.6s | 58.9% | 10.5 GiB |
| &emsp; ParametricMCP build | 26.8s | 37.9% | 9.07 GiB |
| &emsp; variable setup | 65.0ms | 0.1% | 9.94 MiB |
| &emsp; linear solver init | 82.6μs | 0.0% | 14.0 KiB |
| **NonlinearSolver solve** (4 calls) | **1.90s** | **2.7%** | **291 MiB** |
| &emsp; compute K evals (8 calls) | 1.64s | 2.3% | 241 MiB |
| &emsp; line search | 69.3ms | 0.1% | 38.6 MiB |
| &emsp; Jacobian evaluation | 4.50ms | 0.0% | 1.26 MiB |
| &emsp; Newton step | 3.57ms | 0.0% | 5.84 MiB |
| &emsp; residual evaluation | 2.46ms | 0.0% | 2.25 MiB |

**Per-solve: ~476ms** (1 Newton iteration each, converges with residual ~6.3e-14)

### Summary Table

| Experiment | Construction | Per-Solve | Iterations | Solver |
|------------|-------------|-----------|------------|--------|
| LQ Chain (QPSolver) | 6.85s | **627μs** | N/A (direct) | QPSolver |
| LQ Chain (NonlinearSolver) | 2.78s | **124ms** | 1 | NonlinearSolver |
| Lane Change (4-player unicycle) | 339s | **50.6s** | ~51 | NonlinearSolver |
| PPV (3-player) | 68.4s | **476ms** | 1 | NonlinearSolver |

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
