#!/usr/bin/env bash
# Overnight autonomous execution: 3 parallel performance optimization tracks.
# Uses git worktrees so tracks don't fight over the working directory.
#
# Track 1 (10 PRs): utils + linesearch + Newton loop
# Track 2 ( 8 PRs): solve.jl + problem_setup.jl
# Track 3 ( 5 PRs): qp_kkt.jl + nonlinear_kkt.jl setup
#
# Usage:
#   bash scripts/overnight_perf.sh              # all 3 tracks in parallel
#   bash scripts/overnight_perf.sh track1       # just track 1
#   bash scripts/overnight_perf.sh track2       # just track 2
#   bash scripts/overnight_perf.sh track3       # just track 3
#   bash scripts/overnight_perf.sh status       # check progress (from another terminal)
#
# Prerequisites:
#   - PR #130 merged (scripts/benchmark_perf_audit.jl on main)
#   - claude, gh, bd, julia installed
#   - gh auth login completed
#   - Baseline captured: bash scripts/perf_implementation_plan.sh baseline

set -uo pipefail
# NOT set -e: individual PR failures should not halt the track

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR="$REPO_ROOT/logs/overnight-perf-$TS"
mkdir -p "$LOG_DIR"

HEARTBEAT="$LOG_DIR/HEARTBEAT"
START_TIME=$(date +%s)

# Worktree locations (siblings of the repo to avoid nesting)
WT_BASE="$(cd "$REPO_ROOT/.." && pwd)"
WT1="$WT_BASE/perf-track1-$TS"
WT2="$WT_BASE/perf-track2-$TS"
WT3="$WT_BASE/perf-track3-$TS"

# Claude settings — same safe approach as overnight_run.sh
ALLOWED_TOOLS="Read,Write,Edit,Glob,Grep,Task,TaskOutput,Bash"

###############################################################################
# PREAMBLE — injected into every claude prompt
###############################################################################

PREAMBLE='Follow ALL instructions in CLAUDE.md strictly, especially:
- TDD is mandatory (Red-Green-Refactor). Write failing tests FIRST.
- Test tolerances 1e-6 or tighter.
- Run full test suite after changes: julia --project=. -e '"'"'using Pkg; Pkg.test()'"'"'
- Each commit should be logical and self-contained.
- Minimum 3 commits: (1) failing tests, (2) implementation, (3) verification/benchmarks.
- Do NOT run destructive commands (rm -rf, git reset --hard, etc.).
- Check git log first — if this branch already has commits, continue from where it left off.
- If a PR already exists for this branch, update it instead of creating a new one.
- DECOMPOSITION: If the task is too large for one pass, do the smallest sub-task first, commit, then continue.
- RETROSPECTIVE: Add entry to RETROSPECTIVES.md before final push.
- Push your branch and create (or update) a PR when done. Do NOT merge — leave for human review.
- In the PR description, include benchmark results if the change affects hot-path code.
- Update bead status: bd update BEAD_ID --status in_progress at start.'

###############################################################################
# PR DEFINITIONS — compact format per track
# Format: "bead|branch|title|files|description"
# Base branch: first PR in each track bases on main; subsequent PRs base on previous branch.
###############################################################################

TRACK1_PRS=(
"q2p|perf/20-inline-utils|Add @inline to utility functions|src/utils.jl:14-34|Add @inline to is_root(), is_leaf(), has_leader(), ordered_player_indices(). These are trivial wrappers called in hot loops but missing @inline. Test: verify functions return same results. Impact: minor."

"7r0|perf/25-lazy-warn-linesearch|Lazy string eval in linesearch @warn|src/linesearch.jl:51,103|Change @warn \"message \$var\" to @warn lazy\"message \$var\" (or use a closure). Currently string interpolation runs even when warning is suppressed. Test: @warn still fires on linesearch failure. Impact: negligible."

"9ve|perf/27-debug-getter|Remove debug getter overhead|src/nonlinear_kkt.jl:347|Remove unnecessary getter function calls in debug/verbose paths during setup. Test: verbose output unchanged. Impact: negligible."

"309|perf/29-fuse-convergence|Fuse verbose convergence check|src/utils.jl:208|Currently the verbose convergence check iterates twice (once to detect violations, once to format). Fuse into single pass. Test: verbose output matches current behavior. Impact: negligible."

"311|perf/26-nan-fill-error|Avoid NaN-fill allocation in error paths|src/nonlinear_kkt.jl:741,750|In _solve_K! error paths, NaN-filled arrays are allocated but may never be used. Use fill! on existing buffer or lazy NaN sentinel. Test: singular matrix handling unchanged. Impact: negligible."

"65s|perf/02-norm-to-dot|Replace norm(f)^2 with dot(f,f)|src/linesearch.jl:36,41,89,94|norm(f)^2 computes sqrt then squares — dot(f,f) skips the sqrt. Use LinearAlgebra.dot. Test: dot-based merit equals norm-based merit (exact Float64 equality). Impact: 3-8% linesearch speedup."

"r96|perf/01-linesearch-buffer|Pre-allocate x_new buffer in linesearch|src/linesearch.jl:40,93 and src/nonlinear_kkt.jl|Line 40: x_new = x .+ alpha .* d allocates every trial. Add x_buffer kwarg, use @. x_buffer = x + alpha * d. Update caller in nonlinear_kkt.jl to pass pre-allocated buffer. Test: same results with buffer vs without. Impact: 2-5% solve time."

"c0l|perf/03-neg-f-buffer|Pre-allocate neg_F for -F_eval|src/nonlinear_kkt.jl:957|Line 957: neg_F = -F_eval allocates a new vector every Newton iteration. Pre-allocate neg_F buffer at solver construction, use @. neg_F = -F_eval. Test: Newton step produces same result. Impact: 0.5-1%."

"bda|perf/05-closure-outside-loop|Move residual_at_trial closure outside loop|src/nonlinear_kkt.jl:973-981|The residual_at_trial closure is re-created every Newton iteration inside the while loop. Move it outside the loop — it only needs to capture the function F_fn and the current z_est reference. Verify closure still captures updated values correctly. Test: same convergence (iterations and residual). Impact: 0.5-1%."

"l99|perf/04-callback-copy|Avoid copy(z_est) when callback is nothing|src/nonlinear_kkt.jl:1008|Currently copy(z_est) is called every iteration even when callback=nothing. Guard with: if callback !== nothing; callback((; iteration, residual, step_size, z_est=copy(z_est))); end. Test: callback still gets independent copy; no-callback path has zero extra alloc. Impact: 0-5%."
)

TRACK2_PRS=(
"1q1|perf/28-collect-values|Remove collect(values(mus)) before vcat|src/problem_setup.jl:266|collect(values(mus)) creates an intermediate array. Use reduce(vcat, values(mus)) directly or iterate without collecting. Test: all_variables construction unchanged. Impact: negligible."

"qfn|perf/17-merge-options-shortcircuit|Short-circuit _merge_options|src/solve.jl:226-247|When no kwargs override the base options (all nothing), return the base options object directly without constructing a new one. Test: _merge_options returns === base when all kwargs are nothing. Impact: 0.5-1%."

"3ij|perf/16-parameter-dict-skip|Skip Dict creation when Dict passed|src/solve.jl:20-22|_to_parameter_dict currently wraps input in a new Dict even when a Dict{Int,Vector} is already passed. Add a method: _to_parameter_dict(d::Dict{Int}) = d. Test: Dict input returned directly (=== identity), Vector still converted. Impact: 0.5-1%."

"dyl|perf/13-reduce-vcat-to-copyto|Replace reduce(vcat) with copyto!|src/solve.jl:483,552,620,710|reduce(vcat, ...) creates intermediate arrays. Pre-allocate output vector, use copyto! with offsets to fill it. Test: output identical to reduce(vcat). Impact: 1-3%."

"pr8|perf/22-jacobian-buffer-copy|Avoid unnecessary Jacobian copy|src/solve.jl:624|Remove unnecessary copy() of result_buffer after Jacobian evaluation if the buffer is already freshly written. Test: Jacobian evaluation same result without copy. Impact: minor."

"9gl|perf/11-cache-graph-traversals|Cache graph traversals in problem_setup|src/problem_setup.jl:206,211,231,250,255|get_all_followers() and get_all_leaders() are called O(N^2) times during setup. Cache results in a Dict{Int,Vector{Int}} computed once. Test: cached results match uncached for chain/star/nash. Impact: 20-50% setup speedup for N>5."

"40x|perf/12-ws-single-vcat|Single vcat for ws construction|src/problem_setup.jl:235-257|Current ws construction uses repeated vcat in a loop (O(N^2) allocations). Collect all pieces first, then single vcat. Test: ws[i] identical for all players. Impact: 5-10% setup speedup."

"e07|perf/23-all-variables-single-pass|Single-pass all_variables|src/problem_setup.jl:263-267|all_variables currently does nested vcat operations. Construct in a single pass. Test: all_variables identical to current output. Impact: minor."
)

TRACK3_PRS=(
"4sk|perf/24-symbolics-num-conversion|Remove unnecessary Num conversions|src/nonlinear_kkt.jl:462,472|Symbolics.Num() conversions are applied to values that are already Num type. Check with typeof and skip conversion if unnecessary. Test: MCP construction produces same F_sym. Impact: negligible."

"6te|perf/14-cache-extractor-matrices|Cache extractor matrices|src/qp_kkt.jl:123,140|Extractor matrices are built twice per player pair (once for leader KKT, once for follower). Cache in a Dict keyed by (i,j) pair. Test: KKT conditions identical with cached extractors. Impact: 5-10% QP setup speedup."

"iyk|perf/15-dedup-k-flattening|Deduplicate K symbol flattening|src/nonlinear_kkt.jl:351,449,695|The K symbol vector is flattened 3 separate times during setup. Compute once and reuse. Test: all_K_syms_vec identical when computed once vs three times. Impact: minor."

"br6|perf/18-cache-topo-sort|Cache topological sort|src/nonlinear_kkt.jl:238,645|Topological sort is computed multiple times. Compute once during setup and store in precomputed NamedTuple. Test: reverse_topo_order matches fresh computation. Impact: negligible."

"ge7|perf/19-optimize-bfs-collect|Optimize BFS collect|src/nonlinear_kkt.jl:521|collect(BFSIterator(g, v))[2:end] allocates then slices. Use Iterators.drop(BFSIterator(g, v), 1) |> collect or manual iteration. Test: follower list identical. Impact: negligible."
)

###############################################################################
# INFRASTRUCTURE
###############################################################################

heartbeat() {
    while true; do
        sleep 60
        local elapsed=$(( $(date +%s) - START_TIME ))
        local hours=$(( elapsed / 3600 ))
        local mins=$(( (elapsed % 3600) / 60 ))
        {
            echo "Updated: $(date '+%H:%M:%S') | Elapsed: ${hours}h${mins}m"
            for t in 1 2 3; do
                local sf="$LOG_DIR/track${t}.status"
                [ -f "$sf" ] && echo "  Track $t: $(cat "$sf")"
            done
        } > "$HEARTBEAT"
    done
}

setup_worktree() {
    local wt_path="$1"
    echo "Creating worktree: $wt_path"
    cd "$REPO_ROOT"
    git worktree add --detach "$wt_path" HEAD 2>/dev/null
    # Instantiate Julia deps in worktree
    cd "$wt_path"
    julia --project=. -e 'using Pkg; Pkg.instantiate()' 2>/dev/null
}

cleanup_worktree() {
    local wt_path="$1"
    if [ -d "$wt_path" ]; then
        cd "$REPO_ROOT"
        git worktree remove --force "$wt_path" 2>/dev/null || rm -rf "$wt_path"
    fi
}

run_single_pr() {
    local workdir="$1" track_num="$2" pr_num="$3" pr_def="$4" base_branch="$5"

    # Parse PR definition
    local bead branch title files description
    IFS='|' read -r bead branch title files description <<< "$pr_def"

    local log="$LOG_DIR/T${track_num}-${pr_num}_${bead}.log"
    local status_file="$LOG_DIR/track${track_num}.status"
    echo "PR ${pr_num}/${bead}: $title [running]" > "$status_file"

    echo "========================================" | tee "$log"
    echo "[T${track_num}-${pr_num}] Starting: $title" | tee -a "$log"
    echo "Bead: $bead | Branch: $branch | Base: $base_branch" | tee -a "$log"
    echo "$(date)" | tee -a "$log"
    echo "========================================" | tee -a "$log"

    cd "$workdir"

    # Create branch from base
    if [ "$base_branch" = "main" ]; then
        git checkout main >> "$log" 2>&1
        git pull origin main >> "$log" 2>&1
    else
        # For stacked PRs, base on previous branch
        git checkout "$base_branch" >> "$log" 2>&1
    fi

    # Create or reuse branch
    if git ls-remote --heads origin "$branch" 2>/dev/null | grep -q "$branch"; then
        echo "Reusing existing remote branch $branch" >> "$log"
        git checkout -b "$branch" "origin/$branch" >> "$log" 2>&1 || git checkout "$branch" >> "$log" 2>&1
    else
        git checkout -b "$branch" >> "$log" 2>&1 || git checkout "$branch" >> "$log" 2>&1
    fi

    # Build the prompt
    local prompt
    prompt="$PREAMBLE

TASK: Performance optimization T${track_num}-${pr_num}: ${title}
BEAD: ${bead}
BRANCH: ${branch} (base: ${base_branch})
FILE(S): ${files}

DESCRIPTION:
${description}

ADDITIONAL CONTEXT:
- Run: bash scripts/perf_implementation_plan.sh plan — to see the full 29-PR plan
- This is PR T${track_num}-${pr_num} in Track ${track_num}
- Read the target file(s) before making changes
- After implementing, push and create a PR:
    git push -u origin ${branch}
    gh pr create --title \"Perf T${track_num}-${pr_num}: ${title}\" --base ${base_branch}
- If the base is not main, the PR should target the base branch (stacked PR)
- Include benchmark results in the PR description if this is a hot-path change"

    # Run Claude with watchdog (same pattern as overnight_run.sh)
    claude -p "$prompt" \
        --allowedTools "$ALLOWED_TOOLS" \
        >> "$log" 2>&1 &
    local CLAUDE_PID=$!

    # Watchdog: kill if no CPU progress for 5 minutes
    local WATCHDOG_INTERVAL=300
    local STALL_LIMIT=1
    local STALL_COUNT=0
    local LAST_CPU=""
    while kill -0 "$CLAUDE_PID" 2>/dev/null; do
        sleep "$WATCHDOG_INTERVAL"
        if ! kill -0 "$CLAUDE_PID" 2>/dev/null; then break; fi
        local CURRENT_CPU
        CURRENT_CPU=$(
            {
                echo "$CLAUDE_PID"
                pgrep -P "$CLAUDE_PID" 2>/dev/null
            } | sort -u | while read pid; do
                ps -o cputime= -p "$pid" 2>/dev/null
            done | awk -F'[:.]' '{ if (NF==3) s += ($1*60 + $2); else if (NF==4) s += ($1*3600 + $2*60 + $3) } END { print s+0 }'
        )
        if [ "$CURRENT_CPU" = "$LAST_CPU" ]; then
            STALL_COUNT=$((STALL_COUNT + 1))
            echo "[watchdog] No CPU progress for $((STALL_COUNT * WATCHDOG_INTERVAL))s" >> "$log"
            if [ "$STALL_COUNT" -ge "$STALL_LIMIT" ]; then
                echo "[watchdog] Killing stalled process (PID $CLAUDE_PID)" | tee -a "$log"
                kill "$CLAUDE_PID" 2>/dev/null; sleep 2; kill -9 "$CLAUDE_PID" 2>/dev/null
                break
            fi
        else
            STALL_COUNT=0
        fi
        LAST_CPU="$CURRENT_CPU"
    done

    wait "$CLAUDE_PID" 2>/dev/null
    local exit_code=$?

    # Clean working tree for next PR
    cd "$workdir"
    git checkout -- . >> "$log" 2>&1
    git clean -fd --exclude=logs --exclude=scripts >> "$log" 2>&1

    if [ $exit_code -eq 0 ]; then
        echo "PR ${pr_num}/${bead}: $title [DONE]" > "$status_file"
        echo "[T${track_num}-${pr_num}] SUCCESS at $(date)" | tee -a "$log"
        return 0
    else
        echo "PR ${pr_num}/${bead}: $title [FAILED]" > "$status_file"
        echo "[T${track_num}-${pr_num}] FAILED (exit $exit_code) at $(date)" | tee -a "$log"

        # Retry once if no commits were made (likely stalled connection)
        local has_commits=false
        if git log --oneline "${base_branch}..${branch}" 2>/dev/null | head -1 | grep -q .; then
            has_commits=true
        fi

        if [ "$has_commits" = false ]; then
            echo "[T${track_num}-${pr_num}] RETRY: no commits, likely stalled" | tee -a "$log"
            cp "$log" "${log%.log}.attempt1.log"
            sleep 5

            claude -p "$prompt" \
                --allowedTools "$ALLOWED_TOOLS" \
                >> "$log" 2>&1
            exit_code=$?

            cd "$workdir"
            git checkout -- . >> "$log" 2>&1
            git clean -fd --exclude=logs --exclude=scripts >> "$log" 2>&1

            if [ $exit_code -eq 0 ]; then
                echo "PR ${pr_num}/${bead}: $title [DONE after retry]" > "$status_file"
                return 0
            fi
        fi

        return 1
    fi
}

run_track() {
    local track_num="$1" workdir="$2"
    shift 2
    local prs=("$@")

    local status_file="$LOG_DIR/track${track_num}.status"
    local completed=0
    local failed=0
    local total=${#prs[@]}
    local prev_branch="main"

    echo "Starting (${total} PRs)" > "$status_file"

    for i in "${!prs[@]}"; do
        local pr_num=$((i + 1))
        local pr_def="${prs[$i]}"

        # Extract branch name for stacking
        local branch
        branch=$(echo "$pr_def" | cut -d'|' -f2)

        if run_single_pr "$workdir" "$track_num" "$pr_num" "$pr_def" "$prev_branch"; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
            # Continue to next PR even on failure — the branch exists for manual fixup
        fi

        prev_branch="$branch"
        echo "Done ${pr_num}/${total} (ok=$completed fail=$failed)" > "$status_file"
    done

    echo "FINISHED: ${completed}/${total} ok, ${failed} failed" > "$status_file"
    return $failed
}

show_status() {
    local latest
    latest=$(ls -td "$REPO_ROOT/logs/overnight-perf-"* 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "No overnight perf runs found in logs/"
        exit 1
    fi
    echo "=== Latest run: $(basename "$latest") ==="
    if [ -f "$latest/HEARTBEAT" ]; then
        cat "$latest/HEARTBEAT"
    fi
    echo ""
    echo "=== Log files ==="
    ls -lt "$latest"/*.log 2>/dev/null | head -20
    echo ""
    echo "=== Open perf PRs ==="
    gh pr list --state open --search "Perf T" --json number,title,headRefName \
        --template '{{range .}}#{{.number}} [{{.headRefName}}] {{.title}}{{"\n"}}{{end}}' 2>/dev/null || echo "(gh pr list failed)"
}

###############################################################################
# PREFLIGHT CHECKS
###############################################################################

preflight() {
    local errors=0

    # Check tools
    for tool in claude gh julia bd; do
        if ! command -v "$tool" &>/dev/null; then
            echo "ERROR: $tool not found"
            errors=$((errors + 1))
        fi
    done

    # Check we're on main
    cd "$REPO_ROOT"
    local branch
    branch=$(git branch --show-current)
    if [ "$branch" != "main" ]; then
        echo "ERROR: Not on main (on $branch). Switch to main first."
        errors=$((errors + 1))
    fi

    # Check benchmark script exists
    if [ ! -f "$REPO_ROOT/scripts/benchmark_perf_audit.jl" ]; then
        echo "ERROR: scripts/benchmark_perf_audit.jl not found. Merge PR #130 first."
        errors=$((errors + 1))
    fi

    # Check no uncommitted changes
    if ! git diff --quiet || ! git diff --cached --quiet; then
        echo "WARNING: Uncommitted changes detected. Worktrees may not get latest state."
    fi

    if [ $errors -gt 0 ]; then
        echo ""
        echo "Fix the above errors before running."
        exit 1
    fi

    echo "Preflight checks passed."
}

###############################################################################
# MAIN
###############################################################################

cmd="${1:-all}"

case "$cmd" in
    status)
        show_status
        exit 0
        ;;
    track1|track2|track3|all)
        ;;
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  all     - Run all 3 tracks in parallel (default)"
        echo "  track1  - Run Track 1 only (utils + linesearch + Newton loop)"
        echo "  track2  - Run Track 2 only (solve.jl + problem_setup.jl)"
        echo "  track3  - Run Track 3 only (qp_kkt.jl + nonlinear_kkt.jl setup)"
        echo "  status  - Check progress of a running overnight session"
        echo "  help    - Show this help"
        exit 0
        ;;
esac

preflight

echo "==========================================="
echo "  Overnight Perf Run — $(date)"
echo "  Logs: $LOG_DIR"
echo "  Heartbeat: cat $HEARTBEAT"
echo "==========================================="
echo ""

# Start heartbeat
heartbeat &
HEARTBEAT_PID=$!
trap 'kill $HEARTBEAT_PID 2>/dev/null; cleanup_worktree "$WT1"; cleanup_worktree "$WT2"; cleanup_worktree "$WT3"' EXIT

TRACK1_FAIL=0
TRACK2_FAIL=0
TRACK3_FAIL=0

if [ "$cmd" = "all" ]; then
    # Create worktrees
    setup_worktree "$WT1"
    setup_worktree "$WT2"
    setup_worktree "$WT3"

    # Launch all 3 tracks in parallel
    echo "Launching Track 1 (10 PRs) in $WT1..."
    run_track 1 "$WT1" "${TRACK1_PRS[@]}" >> "$LOG_DIR/track1_runner.log" 2>&1 &
    PID1=$!

    echo "Launching Track 2 (8 PRs) in $WT2..."
    run_track 2 "$WT2" "${TRACK2_PRS[@]}" >> "$LOG_DIR/track2_runner.log" 2>&1 &
    PID2=$!

    echo "Launching Track 3 (5 PRs) in $WT3..."
    run_track 3 "$WT3" "${TRACK3_PRS[@]}" >> "$LOG_DIR/track3_runner.log" 2>&1 &
    PID3=$!

    echo ""
    echo "All 3 tracks running. Check progress:"
    echo "  cat $HEARTBEAT"
    echo "  bash scripts/overnight_perf.sh status"
    echo ""

    # Wait for all tracks
    wait $PID1 || TRACK1_FAIL=$?
    wait $PID2 || TRACK2_FAIL=$?
    wait $PID3 || TRACK3_FAIL=$?

elif [ "$cmd" = "track1" ]; then
    setup_worktree "$WT1"
    run_track 1 "$WT1" "${TRACK1_PRS[@]}"
    TRACK1_FAIL=$?

elif [ "$cmd" = "track2" ]; then
    setup_worktree "$WT2"
    run_track 2 "$WT2" "${TRACK2_PRS[@]}"
    TRACK2_FAIL=$?

elif [ "$cmd" = "track3" ]; then
    setup_worktree "$WT3"
    run_track 3 "$WT3" "${TRACK3_PRS[@]}"
    TRACK3_FAIL=$?
fi

# Final report
ELAPSED=$(( $(date +%s) - START_TIME ))
HOURS=$(( ELAPSED / 3600 ))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "==========================================="
echo "  Overnight Perf Run — COMPLETE"
echo "  Elapsed: ${HOURS}h${MINS}m"
echo "  Track 1 failures: $TRACK1_FAIL"
echo "  Track 2 failures: $TRACK2_FAIL"
echo "  Track 3 failures: $TRACK3_FAIL"
echo "==========================================="
echo ""
echo "Track status:"
for t in 1 2 3; do
    sf="$LOG_DIR/track${t}.status"
    [ -f "$sf" ] && echo "  Track $t: $(cat "$sf")"
done
echo ""
echo "Review PRs:"
gh pr list --state open --search "Perf T" --json number,title,headRefName \
    --template '{{range .}}#{{.number}} [{{.headRefName}}] {{.title}}{{"\n"}}{{end}}' 2>/dev/null || echo "(gh pr list failed)"
echo ""
echo "Logs: $LOG_DIR"
