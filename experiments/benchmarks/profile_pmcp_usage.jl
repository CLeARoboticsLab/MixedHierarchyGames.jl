#=
    Profile ParametricMCPs Usage Patterns

    Targeted profiling script for bead s6u: measures how ParametricMCPs objects
    are used in the solve hot paths. Specifically answers:
      1. Is the ParametricMCP object constructed once or rebuilt per solve?
      2. Are parameter vectors allocated fresh each call or reused?
      3. What fraction of solve time is in ParametricMCPs.f!/jacobian_z! vs our code?

    Usage:
        julia --project=experiments experiments/benchmarks/profile_pmcp_usage.jl
=#

using MixedHierarchyGames
using TimerOutputs: TimerOutput, @timeit, reset_timer!
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "lq_three_player_chain", "config.jl"))
include(joinpath(@__DIR__, "..", "lq_three_player_chain", "support.jl"))

function profile_pmcp_usage(; num_solves=20)
    println("=" ^ 70)
    println("  ParametricMCPs Usage Profiling (bead s6u)")
    println("=" ^ 70)

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
    x0 = DEFAULT_X0

    # ─────────────────────────────────────────────
    # Part 1: QPSolver - verify MCP reuse
    # ─────────────────────────────────────────────
    println("\n--- QPSolver (linear backend) ---")
    to = TimerOutput()

    @timeit to "construction" begin
        solver = QPSolver(G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM; to=to)
    end

    # Get MCP object ID to verify same object is used across solves
    mcp_id = objectid(solver.precomputed.parametric_mcp)
    println("  MCP object ID at construction: $mcp_id")

    params = Dict(i => x0[i] for i in 1:N)

    # Warmup
    solve(solver, params; to=to)

    # Reset timer for clean solve-only measurement
    reset_timer!(to)
    @timeit to "$(num_solves) solves" begin
        for _ in 1:num_solves
            solve(solver, params; to=to)
        end
    end

    println("  MCP object ID after $(num_solves) solves: $(objectid(solver.precomputed.parametric_mcp))")
    println("  Same object? $(mcp_id == objectid(solver.precomputed.parametric_mcp))")

    println("\n  QPSolver solve-only timing ($(num_solves) solves):")
    show(to); println("\n")

    # ─────────────────────────────────────────────
    # Part 2: NonlinearSolver - verify MCP reuse + measure allocation
    # ─────────────────────────────────────────────
    println("\n--- NonlinearSolver ---")
    to_nl = TimerOutput()

    @timeit to_nl "construction" begin
        solver_nl = NonlinearSolver(
            G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
            max_iters = MAX_ITERS, tol = TOLERANCE, to = to_nl
        )
    end

    mcp_nl_id = objectid(solver_nl.precomputed.mcp_obj)
    println("  MCP object ID at construction: $mcp_nl_id")

    # Warmup
    solve(solver_nl, params; to=to_nl)

    # Reset for clean measurement
    reset_timer!(to_nl)
    @timeit to_nl "$(num_solves) solves" begin
        for _ in 1:num_solves
            solve(solver_nl, params; to=to_nl)
        end
    end

    println("  MCP object ID after $(num_solves) solves: $(objectid(solver_nl.precomputed.mcp_obj))")
    println("  Same object? $(mcp_nl_id == objectid(solver_nl.precomputed.mcp_obj))")

    println("\n  NonlinearSolver solve-only timing ($(num_solves) solves):")
    show(to_nl); println("\n")

    # ─────────────────────────────────────────────
    # Part 3: Measure allocation in parameter vector construction
    # ─────────────────────────────────────────────
    println("\n--- Parameter Vector Allocation Analysis ---")

    # QPSolver: measure allocation in solve_qp_linear path
    mcp = solver.precomputed.parametric_mcp
    order = sort(collect(keys(θs)))
    n = size(mcp.jacobian_z!.result_buffer, 1)

    println("\n  QPSolver per-solve allocations:")
    alloc_param = @allocated reduce(vcat, (params[k] for k in order))
    alloc_J = @allocated copy(mcp.jacobian_z!.result_buffer)
    alloc_F = @allocated zeros(n)
    alloc_z0 = @allocated zeros(n)
    println("    all_param_vals_vec (reduce+vcat): $(alloc_param) bytes")
    println("    J buffer (copy): $(alloc_J) bytes")
    println("    F buffer (zeros): $(alloc_F) bytes")
    println("    z0 buffer (zeros): $(alloc_z0) bytes")
    println("    Total per-solve: $(alloc_param + alloc_J + alloc_F + alloc_z0) bytes")

    # NonlinearSolver: measure allocation in params_for_z
    mcp_nl = solver_nl.precomputed.mcp_obj
    problem_vars = solver_nl.precomputed.problem_vars
    setup_info = solver_nl.precomputed.setup_info
    all_variables = solver_nl.precomputed.all_variables

    θs_order = sort(collect(keys(params)))
    θ_vals_vec = vcat([params[k] for k in θs_order]...)
    z_est = zeros(length(all_variables))

    println("\n  NonlinearSolver per-iteration allocations:")
    alloc_theta = @allocated vcat([params[k] for k in θs_order]...)
    println("    θ_vals_vec (vcat): $(alloc_theta) bytes")

    all_K_vec, _ = MixedHierarchyGames.compute_K_evals(z_est, problem_vars, setup_info)
    alloc_params = @allocated vcat(θ_vals_vec, all_K_vec)
    println("    param_vec (vcat θ+K): $(alloc_params) bytes")
    println("    ^ This allocation happens EVERY iteration + EVERY line search step")

    alloc_F_eval = @allocated zeros(length(all_variables))
    alloc_grad = @allocated copy(mcp_nl.jacobian_z!.result_buffer)
    println("    F_eval buffer (zeros, once per solve): $(alloc_F_eval) bytes")
    println("    ∇F buffer (copy, once per solve): $(alloc_grad) bytes")

    alloc_z_trial = @allocated z_est .+ 1.0 .* zeros(length(z_est))
    println("    z_trial (broadcast, per line search step): $(alloc_z_trial) bytes")

    println("\n" * "=" ^ 70)
    println("  Profiling complete.")
    println("=" ^ 70)
end

profile_pmcp_usage()
