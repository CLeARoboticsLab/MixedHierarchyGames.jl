#!/usr/bin/env bash
#=============================================================================
#  Performance Optimization Implementation Run Plan
#  StackelbergHierarchyGames.jl
#
#  29 PRs across 3 parallel tracks + 1 merge track (Track F).
#  Each PR follows CLAUDE.md: TDD, retrospective, commit hygiene, review.
#
#  Track 1 (10 PRs): Runtime hot path — utils, linesearch, Newton loop
#  Track 2 ( 8 PRs): Dispatch + setup — solve.jl, problem_setup.jl
#  Track 3 ( 5 PRs): KKT setup — qp_kkt.jl, nonlinear_kkt.jl setup
#  Track F ( 6 PRs): Types — runs after Tracks 1-3 all merge to main
#
#  Benchmark: scripts/benchmark_perf_audit.jl (comprehensive audit)
#  Run before starting and after each track merge to track progress.
#
#  Usage:
#    bash scripts/perf_implementation_plan.sh plan     # Print full plan
#    bash scripts/perf_implementation_plan.sh baseline  # Capture baseline
#    bash scripts/perf_implementation_plan.sh bench     # Run audit benchmark
#=============================================================================

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

cmd="${1:-plan}"

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark Commands
# ─────────────────────────────────────────────────────────────────────────────

run_baseline() {
    echo "=== Capturing Baseline Benchmarks ==="
    mkdir -p logs
    local branch
    branch=$(git branch --show-current)
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local prefix="logs/${branch}_${ts}"

    echo "--- Comprehensive performance audit ---"
    julia --project=. scripts/benchmark_perf_audit.jl | tee "${prefix}_perf_audit.txt"

    echo ""
    echo "Results saved to: ${prefix}_perf_audit.txt"
    echo "Run this again after each PR merge to track progress."
}

run_bench() {
    echo "=== Running Performance Audit Benchmark ==="
    julia --project=. scripts/benchmark_perf_audit.jl
}

# ─────────────────────────────────────────────────────────────────────────────
# The 29 PRs — 3 Parallel Tracks + Track F
# ─────────────────────────────────────────────────────────────────────────────

print_plan() {
cat <<'EOF'
===========================================================================
  PERFORMANCE OPTIMIZATION: 29 PRs — 3 PARALLEL TRACKS
  Benchmark: scripts/benchmark_perf_audit.jl
===========================================================================

  ┌─ Track 1 (10 PRs): utils + linesearch + Newton loop ────────────┐
  │  q2p → 7r0 → 9ve → 309 → 311 → 65s → r96 → c0l → bda → l99   │
  ├─ Track 2 ( 8 PRs): solve.jl + problem_setup.jl ────────────────┤
  │  1q1 → qfn → 3ij → dyl → pr8 → 9gl → 40x → e07               │
  ├─ Track 3 ( 5 PRs): qp_kkt.jl + nonlinear_kkt.jl setup ────────┤
  │  4sk → 6te → iyk → br6 → ge7                                   │
  └──────── all 3 tracks merge to main ────────────────────────────┘
                              ↓
  ┌─ Track F ( 6 PRs): types + deep refactors ─────────────────────┐
  │  cbz → ldj → ggt → asz → xhy → vqy                            │
  └─────────────────────────────────────────────────────────────────┘

  Tracks 1, 2, 3 run IN PARALLEL — each branches from main.
  Track F starts AFTER all 3 tracks merge to main.

CLAUDE.md COMPLIANCE (every PR must):
  1. TDD: Write failing test → implement → verify green
  2. Commits: (a) failing tests, (b) implementation, (c) benchmarks
  3. PR description: Summary, Changes, Testing, Changelog
  4. Code review: Offer review before landing (for src/ changes)
  5. Retrospective: Entry in RETROSPECTIVES.md before merge
  6. Verification: All 1235+ tests pass, experiments run clean

MERGE PROTOCOL:
  Within each track: stack PRs (each bases on previous branch)
  Across tracks: each track's first PR bases on main
  Track F: first PR (cbz) bases on main AFTER all 3 tracks merge

BENCHMARK PROTOCOL:
  Before starting:      bash scripts/perf_implementation_plan.sh baseline
  After each PR merge:  bash scripts/perf_implementation_plan.sh bench
  After each full track: compare to baseline, record cumulative gains
  Record results in each PR description under "## Benchmark Results"

===========================================================================
  TRACK 1: Runtime Hot Path (10 PRs)
  Files: src/utils.jl, src/linesearch.jl, src/nonlinear_kkt.jl (solver loop)
  Impact: 8-15% solve time
===========================================================================

  T1-1 │ #20 │ q2p │ perf/20-inline-utils
       │ Add @inline to is_root, is_leaf, has_leader, ordered_player_indices
       │ File: src/utils.jl:14-34
       │ Base: main
       │ TDD: test that @inline functions produce same results
       │ Impact: minor function call overhead reduction
       │
  T1-2 │ #25 │ 7r0 │ perf/25-lazy-warn-linesearch
       │ Lazy string evaluation in linesearch @warn
       │ File: src/linesearch.jl:51,103
       │ Base: perf/20-inline-utils
       │ TDD: test @warn still fires on linesearch failure
       │ Impact: negligible (error path only)
       │
  T1-3 │ #27 │ 9ve │ perf/27-debug-getter
       │ Remove debug getter overhead in setup
       │ File: src/nonlinear_kkt.jl:347
       │ Base: perf/25-lazy-warn-linesearch
       │ TDD: test verbose/debug output unchanged
       │ Impact: negligible (setup debug only)
       │
  T1-4 │ #29 │ 309 │ perf/29-fuse-convergence
       │ Fuse verbose convergence check (single-pass violation detection)
       │ File: src/utils.jl:208
       │ Base: perf/27-debug-getter
       │ TDD: test verbose output matches current behavior
       │ Impact: negligible (verbose only)
       │
  T1-5 │ #26 │ 311 │ perf/26-nan-fill-error
       │ Avoid NaN-fill allocation in _solve_K! error paths
       │ File: src/nonlinear_kkt.jl:741,750
       │ Base: perf/29-fuse-convergence
       │ TDD: test singular matrix handling unchanged
       │ Impact: negligible (error path only)
       │
  T1-6 │ #2  │ 65s │ perf/02-norm-to-dot
       │ Replace norm(f)^2 with dot(f,f) in linesearch
       │ File: src/linesearch.jl:36,41,89,94
       │ Base: perf/26-nan-fill-error
       │ TDD: test dot-based merit == norm-based merit (exact equality)
       │ Impact: 3-8% linesearch speedup (avoids sqrt)
       │
  T1-7 │ #1  │ r96 │ perf/01-linesearch-buffer
       │ Pre-allocate x_new buffer in armijo/geometric linesearch
       │ Files: src/linesearch.jl:40,93 + src/nonlinear_kkt.jl (pass buffer)
       │ Base: perf/02-norm-to-dot
       │ TDD: test linesearch accepts x_buffer kwarg; same results with buffer
       │ Impact: 2-5% solve time (eliminates 20+ allocs per Newton iter)
       │
  T1-8 │ #3  │ c0l │ perf/03-neg-f-buffer
       │ Pre-allocate neg_F buffer for -F_eval
       │ File: src/nonlinear_kkt.jl:957
       │ Base: perf/01-linesearch-buffer
       │ TDD: test Newton step produces same result with pre-allocated neg_F
       │ Impact: 0.5-1% (1 vec alloc per iter)
       │
  T1-9 │ #5  │ bda │ perf/05-closure-outside-loop
       │ Move residual_at_trial closure outside Newton while loop
       │ File: src/nonlinear_kkt.jl:973-981
       │ Base: perf/03-neg-f-buffer
       │ TDD: test convergence unchanged (same iterations, same residual)
       │ Impact: 0.5-1% (closure allocation avoidance)
       │ NOTE: Verify closure still captures updated param_vec correctly
       │
  T1-10│ #4  │ l99 │ perf/04-callback-copy
       │ Avoid unnecessary copy(z_est) when callback is nothing
       │ File: src/nonlinear_kkt.jl:1008
       │ Base: perf/05-closure-outside-loop
       │ TDD: test callback still gets independent copy; no-callback path zero-alloc
       │ Impact: 0-5% depending on callback usage

===========================================================================
  TRACK 2: Dispatch + Setup (8 PRs)
  Files: src/solve.jl, src/problem_setup.jl
  Impact: 2-5% solve time + 30-60% construction time
===========================================================================

  T2-1 │ #28 │ 1q1 │ perf/28-collect-values
       │ Remove collect(values(μs)) before vcat
       │ File: src/problem_setup.jl:266
       │ Base: main
       │ TDD: test all_variables construction unchanged
       │ Impact: negligible
       │
  T2-2 │ #17 │ qfn │ perf/17-merge-options-shortcircuit
       │ Short-circuit _merge_options when no kwargs override
       │ File: src/solve.jl:226-247
       │ Base: perf/28-collect-values
       │ TDD: test _merge_options returns === base when all kwargs nothing
       │ Impact: 0.5-1% (skip Options allocation when no overrides)
       │
  T2-3 │ #16 │ 3ij │ perf/16-parameter-dict-skip
       │ Skip _to_parameter_dict Dict creation when Dict already passed
       │ File: src/solve.jl:20-22
       │ Base: perf/17-merge-options-shortcircuit
       │ TDD: test Dict input returned directly (===), Vector still converted
       │ Impact: 0.5-1% (skip Dict alloc per solve)
       │
  T2-4 │ #13 │ dyl │ perf/13-reduce-vcat-to-copyto
       │ Replace reduce(vcat, generator) with pre-allocated copyto!
       │ File: src/solve.jl:483,552,620,710
       │ Base: perf/16-parameter-dict-skip
       │ TDD: test parameter vector identical to reduce(vcat) output
       │ Impact: 1-3% (eliminate intermediate allocations per solve)
       │
  T2-5 │ #22 │ pr8 │ perf/22-jacobian-buffer-copy
       │ Avoid unnecessary copy of Jacobian result_buffer
       │ File: src/solve.jl:624
       │ Base: perf/13-reduce-vcat-to-copyto
       │ TDD: test Jacobian evaluation produces same result without copy
       │ Impact: minor (one-time copy elimination)
       │
  T2-6 │ #11 │ 9gl │ perf/11-cache-graph-traversals
       │ Cache get_all_followers/get_all_leaders in problem_setup
       │ File: src/problem_setup.jl:206,211,231,250,255
       │ Base: perf/22-jacobian-buffer-copy
       │ TDD: test cached results match uncached for chain/star/nash topologies
       │ Impact: 20-50% setup speedup for N>5
       │
  T2-7 │ #12 │ 40x │ perf/12-ws-single-vcat
       │ Replace repeated vcat in ws construction with single vcat
       │ File: src/problem_setup.jl:235-257
       │ Base: perf/11-cache-graph-traversals
       │ TDD: test ws[i] identical for all players across chain/star/nash
       │ Impact: 5-10% setup speedup (O(N^2) → O(N))
       │
  T2-8 │ #23 │ e07 │ perf/23-all-variables-single-pass
       │ Single-pass all_variables construction
       │ File: src/problem_setup.jl:263-267
       │ Base: perf/12-ws-single-vcat
       │ TDD: test all_variables identical to nested vcat output
       │ Impact: minor (precomputation only)

===========================================================================
  TRACK 3: KKT Setup (5 PRs)
  Files: src/qp_kkt.jl, src/nonlinear_kkt.jl (setup functions only)
  Impact: 5-10% construction time
===========================================================================

  T3-1 │ #24 │ 4sk │ perf/24-symbolics-num-conversion
       │ Remove unnecessary Symbolics.Num conversions
       │ File: src/nonlinear_kkt.jl:462,472
       │ Base: main
       │ TDD: test MCP construction produces same F_sym
       │ Impact: negligible (precomputation only)
       │
  T3-2 │ #14 │ 6te │ perf/14-cache-extractor-matrices
       │ Cache extractor matrices in qp_kkt.jl (built twice per pair)
       │ File: src/qp_kkt.jl:123,140
       │ Base: perf/24-symbolics-num-conversion
       │ TDD: test KKT conditions identical with cached extractors
       │ Impact: 5-10% QP setup speedup
       │
  T3-3 │ #15 │ iyk │ perf/15-dedup-k-flattening
       │ Deduplicate K symbol flattening (computed 3 times)
       │ File: src/nonlinear_kkt.jl:351,449,695
       │ Base: perf/14-cache-extractor-matrices
       │ TDD: test all_K_syms_vec identical when computed once vs three times
       │ Impact: minor (setup redundancy)
       │
  T3-4 │ #18 │ br6 │ perf/18-cache-topo-sort
       │ Cache topological sort result in precomputed
       │ File: src/nonlinear_kkt.jl:238,645
       │ Base: perf/15-dedup-k-flattening
       │ TDD: test reverse_topo_order matches fresh computation
       │ Impact: negligible for small N
       │
  T3-5 │ #19 │ ge7 │ perf/19-optimize-bfs-collect
       │ Optimize collect(BFSIterator)[2:end]
       │ File: src/nonlinear_kkt.jl:521
       │ Base: perf/18-cache-topo-sort
       │ TDD: test follower list identical with optimized path
       │ Impact: negligible (per-player first-iter)

===========================================================================
              ↓ ALL 3 TRACKS MERGE TO MAIN ↓
===========================================================================

  TRACK F: Types + Deep Refactors (6 PRs)
  Files: src/types.jl, src/nonlinear_kkt.jl (type defs), src/qp_kkt.jl
  Impact: 5-10% solve time
  PREREQUISITE: Tracks 1, 2, 3 all merged to main

  WARNING: Track F touches type definitions that affect all source files.
  These PRs are the most invasive and must run AFTER all other tracks.

===========================================================================

  TF-1 │ #21 │ cbz │ perf/21-sparse-pattern-cache
       │ Pre-compute sparse pattern for use_sparse mode
       │ File: src/nonlinear_kkt.jl:732
       │ Base: main (after Tracks 1-3 merged)
       │ TDD: test K solve identical with cached pattern vs fresh sparse()
       │ Impact: 5-10% if use_sparse=:always
       │
  TF-2 │ #10 │ ldj │ perf/10-union-buffer-removal
       │ Replace Union{Matrix,Nothing} buffers with typed containers
       │ File: src/nonlinear_kkt.jl:628-640
       │ Base: perf/21-sparse-pattern-cache
       │ TDD: test K evaluation produces identical results
       │ Impact: ~1-2% per-iteration type inference improvement
       │
  TF-3 │ #8  │ ggt │ perf/08-concrete-precomputed
       │ Replace NamedTuple with concrete struct for precomputed
       │ File: src/types.jl:443 + all files accessing .precomputed
       │ Base: perf/10-union-buffer-removal
       │ TDD: test solver construction and solve produce identical results
       │ Impact: 1-3% per-solve (type-stable field access)
       │ WARNING: Most files will need updating. Run full test suite.
       │
  TF-4 │ #9  │ asz │ perf/09-hierarchy-problem-params
       │ Tighten HierarchyProblem type parameters
       │ File: src/types.jl:47-55
       │ Base: perf/08-concrete-precomputed
       │ TDD: test all existing solver construction paths still work
       │ Impact: 1-3% per-solve
       │
  TF-5 │ #6  │ xhy │ perf/06-typed-function-storage
       │ Replace Vector{Function} with typed function storage
       │ File: src/nonlinear_kkt.jl:245-246
       │ Base: perf/09-hierarchy-problem-params
       │ TDD: test M/N function evaluation identical with typed storage
       │ Impact: 2-5% per K-evaluation (eliminates runtime dispatch)
       │ NOTE: May need FunctionWrappers.jl or concrete closure types
       │
  TF-6 │ #7  │ vqy │ perf/07-typed-kkt-dicts
       │ Replace Dict{Int,Any} with typed KKT dicts (if feasible)
       │ File: src/qp_kkt.jl:92-95
       │ Base: perf/06-typed-function-storage
       │ TDD: test KKT construction produces identical symbolic expressions
       │ Impact: negligible (setup only, may not be feasible)
       │ NOTE: Varying symbolic types may prevent concrete typing

===========================================================================
  PR TEMPLATE (copy for each PR — adheres to CLAUDE.md)
===========================================================================

  ## Summary
  Performance optimization [Track X-N]: [finding title]

  ## Changes
  - `path/to/file.jl`: [what changed and why]

  ## Benchmark Results
  Run: `julia --project=. scripts/benchmark_perf_audit.jl`

  | Metric          | Before    | After     | Change |
  |-----------------|-----------|-----------|--------|
  | solve_raw (2P)  | XXX μs    | XXX μs    | -X.X%  |
  | solve_raw (3P)  | XXX μs    | XXX μs    | -X.X%  |
  | solve_raw (5P)  | XXX μs    | XXX μs    | -X.X%  |
  | construction    | XXX ms    | XXX ms    | -X.X%  |
  | allocations     | XXX KB    | XXX KB    | -X.X%  |

  ## Testing
  - All tests pass: `julia --project=. -e 'using Pkg; Pkg.test()'`
  - Benchmark comparison recorded above

  ## Changelog
  - [Date]: PR created with [finding description]

===========================================================================
  CUMULATIVE IMPACT ESTIMATES
===========================================================================

  Track 1 (runtime hot path):  8-15% solve time
    T1-1..T1-5 (trivials):    <1% combined
    T1-6..T1-10 (hot path):   8-15% solve time

  Track 2 (dispatch + setup):  2-5% solve + 30-60% construction
    T2-1..T2-5 (solve.jl):    2-5% solve time
    T2-6..T2-8 (setup):       30-60% construction time

  Track 3 (KKT setup):         5-10% construction time

  Track F (types):              5-10% solve time

  Total estimated:
    Solve time:        15-30% faster
    Construction time: 30-60% faster
    Allocations:       20-40% fewer

===========================================================================
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

case "$cmd" in
    plan)      print_plan ;;
    baseline)  run_baseline ;;
    bench)     run_bench ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  plan      - Print full 29-PR execution plan (3 parallel tracks + Track F)"
        echo "  baseline  - Capture baseline benchmarks (run before PR #1)"
        echo "  bench     - Run comprehensive benchmark audit"
        echo "  help      - Show this help"
        ;;
esac
