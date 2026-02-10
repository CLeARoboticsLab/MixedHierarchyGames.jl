#=
    Allocation Benchmark for Optimization Verification

    Measures per-solve allocations for QPSolver and NonlinearSolver
    to verify that buffer pre-allocation reduces allocation overhead.

    Usage:
        julia --project=experiments experiments/benchmarks/benchmark_allocations.jl
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm

# ─────────────────────────────────────────────────────
# Benchmark 1: QPSolver (LQ problem)
# ─────────────────────────────────────────────────────

module BenchAllocQP
    using MixedHierarchyGames
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "config.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "support.jl"))

    function run()
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
        params = Dict(i => DEFAULT_X0[i] for i in 1:N)

        solver = QPSolver(G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM)

        # Warmup
        solve(solver, params)

        # Measure
        num_solves = 20
        alloc = @allocated for _ in 1:num_solves
            solve(solver, params)
        end
        per_solve = alloc / num_solves

        println("  QPSolver (linear): $(round(per_solve/1024, digits=1)) KiB per solve ($(num_solves) solves)")
        return per_solve
    end
end

# ─────────────────────────────────────────────────────
# Benchmark 2: NonlinearSolver on LQ problem (1 iteration)
# ─────────────────────────────────────────────────────

module BenchAllocNLLQ
    using MixedHierarchyGames
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "config.jl"))
    include(joinpath(@__DIR__, "..", "lq_three_player_chain", "support.jl"))

    function run()
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
        params = Dict(i => DEFAULT_X0[i] for i in 1:N)

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
                                 max_iters=100, tol=1e-6)

        # Warmup
        solve(solver, params)

        # Measure
        num_solves = 20
        alloc = @allocated for _ in 1:num_solves
            solve(solver, params)
        end
        per_solve = alloc / num_solves

        result = solve_raw(solver, params)
        println("  NonlinearSolver (LQ, $(result.iterations) iters): $(round(per_solve/1024, digits=1)) KiB per solve ($(num_solves) solves)")
        return per_solve
    end
end

# ─────────────────────────────────────────────────────
# Benchmark 3: NonlinearSolver on Lane Change (many iterations)
# ─────────────────────────────────────────────────────

module BenchAllocLaneChange
    using MixedHierarchyGames
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!
    using LinearAlgebra: norm

    include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
    include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))
    include(joinpath(@__DIR__, "..", "nonlinear_lane_change", "config.jl"))
    include(joinpath(@__DIR__, "..", "nonlinear_lane_change", "support.jl"))

    function run()
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
        params = Dict(i => x0[i] for i in 1:N)

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
                                 max_iters=MAX_ITERS, tol=TOLERANCE)

        # Warmup
        r0 = solve_raw(solver, params; initial_guess=z0_guess)
        println("  Lane Change warmup: status=$(r0.status), iters=$(r0.iterations)")

        # Measure
        num_solves = 4
        alloc = @allocated for _ in 1:num_solves
            solve_raw(solver, params; initial_guess=z0_guess)
        end
        per_solve = alloc / num_solves

        # Also time it
        t_start = time()
        for _ in 1:num_solves
            solve_raw(solver, params; initial_guess=z0_guess)
        end
        t_elapsed = time() - t_start
        per_solve_time = t_elapsed / num_solves

        result = solve_raw(solver, params; initial_guess=z0_guess)
        println("  NonlinearSolver (Lane Change, $(result.iterations) iters): $(round(per_solve/1024/1024, digits=2)) MiB per solve, $(round(per_solve_time, digits=3))s per solve ($(num_solves) solves)")
        return per_solve
    end
end

# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────

println("\n" * "=" ^ 70)
println("  Allocation Benchmarks")
println("=" ^ 70 * "\n")

qp_alloc = BenchAllocQP.run()
println()
nl_lq_alloc = BenchAllocNLLQ.run()
println()
nl_lc_alloc = BenchAllocLaneChange.run()

println("\n" * "=" ^ 70)
println("  Summary")
println("=" ^ 70)
println("  QPSolver (linear):              $(round(qp_alloc/1024, digits=1)) KiB per solve")
println("  NonlinearSolver (LQ, 1 iter):   $(round(nl_lq_alloc/1024, digits=1)) KiB per solve")
println("  NonlinearSolver (Lane Change):  $(round(nl_lc_alloc/1024/1024, digits=2)) MiB per solve")
println("=" ^ 70)
