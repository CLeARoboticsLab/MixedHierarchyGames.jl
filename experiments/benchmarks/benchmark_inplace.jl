#=
    Benchmark In-Place Pre-Allocation Strategies

    Tests all 7 combinations of in-place strategies on 3 problem sizes
    for both QP and Nonlinear solvers. Reports timing and allocations.

    Usage:
        julia --project=experiments experiments/benchmarks/benchmark_inplace.jl
=#

using MixedHierarchyGames
using MixedHierarchyGames: compute_K_evals, preoptimize_nonlinear_solver, run_nonlinear_solver
using TimerOutputs: TimerOutput, @timeit
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm
using Printf: @sprintf

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))

# ─── Strategy combinations ───────────────────────────────

const STRATEGIES = [
    (name="Baseline",   inplace_MN=false, inplace_ldiv=false, inplace_lu=false),
    (name="A only",     inplace_MN=true,  inplace_ldiv=false, inplace_lu=false),
    (name="B only",     inplace_MN=false, inplace_ldiv=true,  inplace_lu=false),
    (name="C only",     inplace_MN=false, inplace_ldiv=false, inplace_lu=true),
    (name="A+B",        inplace_MN=true,  inplace_ldiv=true,  inplace_lu=false),
    (name="A+C",        inplace_MN=true,  inplace_ldiv=false, inplace_lu=true),
    (name="A+B+C",      inplace_MN=true,  inplace_ldiv=true,  inplace_lu=true),
]

# ─── Problem setup modules ───────────────────────────────

module SetupLQ
    using MixedHierarchyGames
    using MixedHierarchyGames: preoptimize_nonlinear_solver, setup_problem_parameter_variables
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "config.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "support.jl"))

    function setup()
        G = build_hierarchy()
        Js = make_cost_functions(STATE_DIM, CONTROL_DIM)
        primal_dim = (STATE_DIM + CONTROL_DIM) * (DEFAULT_T + 1)
        primal_dims = fill(primal_dim, N)
        θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

        function make_constraints(i)
            return function (zᵢ)
                dyn = mapreduce(vcat, 1:DEFAULT_T) do t
                    single_integrator_2d(zᵢ, t; Δt=DEFAULT_DT, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
                end
                (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
                ic = xs[1] - θs[i]
                vcat(dyn, ic)
            end
        end
        gs = [make_constraints(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(G, Js, gs, primal_dims, θs;
            state_dim=STATE_DIM, control_dim=CONTROL_DIM)
        params = Dict(i => DEFAULT_X0[i] for i in 1:N)

        return (; precomputed, params, G, name="LQ Chain (NL)")
    end
end

module SetupLaneChange
    using MixedHierarchyGames
    using MixedHierarchyGames: preoptimize_nonlinear_solver, setup_problem_parameter_variables
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
    include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))
    include(joinpath(@__DIR__, "..", "nonlinear_lane_change", "config.jl"))
    include(joinpath(@__DIR__, "..", "nonlinear_lane_change", "support.jl"))

    function setup()
        R = DEFAULT_R; T = DEFAULT_T; Δt = DEFAULT_DT
        G = build_hierarchy()
        Js = make_cost_functions(STATE_DIM, CONTROL_DIM, T, R)
        primal_dim = (STATE_DIM + CONTROL_DIM) * (T + 1)
        primal_dims = fill(primal_dim, N)
        θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

        function make_constraints(i)
            return function (zᵢ)
                dyn = mapreduce(vcat, 1:T) do t
                    unicycle_dynamics(zᵢ, t; Δt, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
                end
                (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
                ic = xs[1] - θs[i]
                vcat(dyn, ic)
            end
        end
        gs = [make_constraints(i) for i in 1:N]
        x0 = default_initial_states(R)
        z0_guess = build_initial_guess(x0, R, T, Δt)

        precomputed = preoptimize_nonlinear_solver(G, Js, gs, primal_dims, θs;
            state_dim=STATE_DIM, control_dim=CONTROL_DIM)
        params = Dict(i => x0[i] for i in 1:N)

        return (; precomputed, params, G, z0_guess, name="Lane Change (NL)")
    end
end

module SetupPPV
    using MixedHierarchyGames
    using MixedHierarchyGames: preoptimize_nonlinear_solver, setup_problem_parameter_variables
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "pursuer_protector_vip", "config.jl"))
    include(joinpath(@__DIR__, "..", "pursuer_protector_vip", "support.jl"))

    function setup()
        G = build_hierarchy()
        Js = make_cost_functions(STATE_DIM, CONTROL_DIM, DEFAULT_T, DEFAULT_GOAL)
        primal_dim = (STATE_DIM + CONTROL_DIM) * (DEFAULT_T + 1)
        primal_dims = fill(primal_dim, N)
        θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

        function make_constraints(i)
            return function (zᵢ)
                dyn = mapreduce(vcat, 1:DEFAULT_T) do t
                    single_integrator_2d(zᵢ, t; Δt=DEFAULT_DT, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
                end
                (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
                ic = xs[1] - θs[i]
                vcat(dyn, ic)
            end
        end
        gs = [make_constraints(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(G, Js, gs, primal_dims, θs;
            state_dim=STATE_DIM, control_dim=CONTROL_DIM)
        params = Dict(i => DEFAULT_X0[i] for i in 1:N)

        return (; precomputed, params, G, name="PPV (NL)")
    end
end

# ─── QP Solver benchmark (separate since QP doesn't use compute_K_evals) ──

module SetupLQ_QP
    using MixedHierarchyGames
    using MixedHierarchyGames: setup_problem_parameter_variables
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "config.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "support.jl"))

    function setup()
        G = build_hierarchy()
        Js = make_cost_functions(STATE_DIM, CONTROL_DIM)
        primal_dim = (STATE_DIM + CONTROL_DIM) * (DEFAULT_T + 1)
        primal_dims = fill(primal_dim, N)
        θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

        function make_constraints(i)
            return function (zᵢ)
                dyn = mapreduce(vcat, 1:DEFAULT_T) do t
                    single_integrator_2d(zᵢ, t; Δt=DEFAULT_DT, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
                end
                (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
                ic = xs[1] - θs[i]
                vcat(dyn, ic)
            end
        end
        gs = [make_constraints(i) for i in 1:N]

        solver = QPSolver(G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM)
        params = Dict(i => DEFAULT_X0[i] for i in 1:N)

        return (; solver, params, name="LQ Chain (QP)")
    end
end

# ─── Benchmark runner ─────────────────────────────────────

function benchmark_solve(setup_result, strategy; num_solves=5)
    precomputed = setup_result.precomputed
    params = setup_result.params
    G = setup_result.G
    initial_guess = hasproperty(setup_result, :z0_guess) ? setup_result.z0_guess : nothing

    # Warmup
    run_nonlinear_solver(precomputed, params, G;
        initial_guess=initial_guess, max_iters=100, tol=1e-6,
        inplace_MN=strategy.inplace_MN, inplace_ldiv=strategy.inplace_ldiv, inplace_lu=strategy.inplace_lu)

    # Timed runs
    times = Float64[]
    allocs_list = Int[]
    bytes_list = Int[]

    for _ in 1:num_solves
        stats = @timed run_nonlinear_solver(precomputed, params, G;
            initial_guess=initial_guess, max_iters=100, tol=1e-6,
            inplace_MN=strategy.inplace_MN, inplace_ldiv=strategy.inplace_ldiv, inplace_lu=strategy.inplace_lu)
        push!(times, stats.time)
        push!(bytes_list, stats.bytes)
    end

    median_time = sort(times)[div(length(times), 2) + 1]
    median_bytes = sort(bytes_list)[div(length(bytes_list), 2) + 1]

    # Check correctness against baseline if not baseline
    if strategy.name != "Baseline"
        result_baseline = run_nonlinear_solver(precomputed, params, G;
            initial_guess=initial_guess, max_iters=100, tol=1e-6)
        result_inplace = run_nonlinear_solver(precomputed, params, G;
            initial_guess=initial_guess, max_iters=100, tol=1e-6,
            inplace_MN=strategy.inplace_MN, inplace_ldiv=strategy.inplace_ldiv, inplace_lu=strategy.inplace_lu)

        sol_diff = norm(result_baseline.sol - result_inplace.sol)
        if sol_diff > 1e-8
            println("  WARNING: $(strategy.name) differs from baseline by $sol_diff")
        end
    end

    return (; median_time, median_bytes, times)
end

function format_time(t)
    if t < 1e-3
        return @sprintf("%.1f μs", t * 1e6)
    elseif t < 1.0
        return @sprintf("%.1f ms", t * 1e3)
    else
        return @sprintf("%.2f s", t)
    end
end

function format_bytes(b)
    if b < 1024
        return "$(b) B"
    elseif b < 1024^2
        return @sprintf("%.1f KiB", b / 1024)
    elseif b < 1024^3
        return @sprintf("%.1f MiB", b / 1024^2)
    else
        return @sprintf("%.2f GiB", b / 1024^3)
    end
end

# ─── Main ─────────────────────────────────────────────────

function main()
    println("\n" * "╌" ^ 70)
    println("  In-Place Strategy Benchmark")
    println("╌" ^ 70)

    # Setup all problems (construction happens once)
    println("\nSetting up problems...")

    print("  LQ Chain (QP)... ")
    lq_qp = SetupLQ_QP.setup()
    println("done")

    print("  LQ Chain (NL)... ")
    lq_nl = SetupLQ.setup()
    println("done")

    print("  PPV (NL)... ")
    ppv = SetupPPV.setup()
    println("done")

    print("  Lane Change (NL)... ")
    lane = SetupLaneChange.setup()
    println("done")

    # QP solver baseline (in-place strategies don't apply to QP)
    println("\n--- QP Solver (baseline only, no in-place strategies apply) ---")
    qp_solver = lq_qp.solver
    qp_params = lq_qp.params
    # Warmup
    solve(qp_solver, qp_params)
    qp_stats = @timed begin
        for _ in 1:5
            solve(qp_solver, qp_params)
        end
    end
    qp_per_solve = qp_stats.time / 5
    println("  LQ (QP): $(format_time(qp_per_solve)) per solve, $(format_bytes(qp_stats.bytes ÷ 5)) per solve")

    # Benchmark matrix
    println("\n" * "=" ^ 80)
    println("  BENCHMARKING ALL COMBINATIONS")
    println("=" ^ 80)

    # Results storage
    results = Dict{String, Dict{String, NamedTuple}}()

    problems = [
        ("LQ (NL)", lq_nl, 10),
        ("PPV (NL)", ppv, 5),
        ("Lane Change (NL)", lane, 3),
    ]

    for (prob_name, prob_setup, num_solves) in problems
        println("\n--- $prob_name ($(num_solves) solves per combination) ---")
        results[prob_name] = Dict{String, NamedTuple}()

        for strat in STRATEGIES
            print("  $(strat.name)... ")
            result = benchmark_solve(prob_setup, strat; num_solves)
            results[prob_name][strat.name] = result
            println("$(format_time(result.median_time)) | $(format_bytes(result.median_bytes))")
        end
    end

    # TimerOutputs detailed breakdown for lane change
    println("\n" * "=" ^ 80)
    println("  TIMEROUTPUTS BREAKDOWN: Lane Change (Baseline vs A+C)")
    println("=" ^ 80)

    for (label, strat) in [("Baseline", STRATEGIES[1]), ("A+C", STRATEGIES[6])]
        println("\n  --- $label ---")
        to = TimerOutput()
        run_nonlinear_solver(lane.precomputed, lane.params, lane.G;
            initial_guess=lane.z0_guess, max_iters=100, tol=1e-6, to=to,
            inplace_MN=strat.inplace_MN, inplace_ldiv=strat.inplace_ldiv, inplace_lu=strat.inplace_lu)
        show(to)
        println()
    end

    # Print results matrix
    println("\n" * "=" ^ 80)
    println("  RESULTS MATRIX (median time)")
    println("=" ^ 80)

    # Header
    prob_names = ["LQ (NL)", "PPV (NL)", "Lane Change (NL)"]
    header = @sprintf("%-15s", "Combination")
    for pn in prob_names
        header *= @sprintf(" | %-20s", pn)
    end
    println(header)
    println("-" ^ length(header))

    for strat in STRATEGIES
        row = @sprintf("%-15s", strat.name)
        for pn in prob_names
            r = results[pn][strat.name]
            row *= @sprintf(" | %-20s", format_time(r.median_time))
        end
        println(row)
    end

    # Allocation matrix
    println("\n" * "=" ^ 80)
    println("  RESULTS MATRIX (median allocations)")
    println("=" ^ 80)

    header = @sprintf("%-15s", "Combination")
    for pn in prob_names
        header *= @sprintf(" | %-20s", pn)
    end
    println(header)
    println("-" ^ length(header))

    for strat in STRATEGIES
        row = @sprintf("%-15s", strat.name)
        for pn in prob_names
            r = results[pn][strat.name]
            row *= @sprintf(" | %-20s", format_bytes(r.median_bytes))
        end
        println(row)
    end

    # Speedup matrix
    println("\n" * "=" ^ 80)
    println("  SPEEDUP vs BASELINE")
    println("=" ^ 80)

    header = @sprintf("%-15s", "Combination")
    for pn in prob_names
        header *= @sprintf(" | %-20s", pn)
    end
    println(header)
    println("-" ^ length(header))

    for strat in STRATEGIES
        row = @sprintf("%-15s", strat.name)
        for pn in prob_names
            baseline_t = results[pn]["Baseline"].median_time
            strat_t = results[pn][strat.name].median_time
            speedup = baseline_t / strat_t
            row *= @sprintf(" | %-20s", @sprintf("%.2fx", speedup))
        end
        println(row)
    end

    println("\n" * "╌" ^ 70)
    println("  Benchmark complete.")
    println("╌" ^ 70)
end

main()
