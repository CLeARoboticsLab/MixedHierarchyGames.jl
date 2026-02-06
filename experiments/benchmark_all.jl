#=
    Benchmark All Experiments

    Runs each experiment with TimerOutputs to measure and report
    performance of the new src/ code. Separates construction (one-time)
    from solve (repeated) timing.

    Usage:
        julia --project=experiments experiments/benchmark_all.jl
=#

using MixedHierarchyGames
using TimerOutputs: TimerOutput, @timeit
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm

# Include common utilities
include(joinpath(@__DIR__, "common", "dynamics.jl"))
include(joinpath(@__DIR__, "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "common", "trajectory_utils.jl"))

# ─────────────────────────────────────────────────────
# Experiment 1: LQ Three Player Chain
# ─────────────────────────────────────────────────────

module BenchLQ
    using MixedHierarchyGames
    using TimerOutputs: TimerOutput, @timeit
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!
    using LinearAlgebra: norm

    include(joinpath(@__DIR__, "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "lq_three_player_chain", "config.jl"))
    include(joinpath(@__DIR__, "lq_three_player_chain", "support.jl"))

    function run(; num_solves=5)
        println("=" ^ 70)
        println("  Experiment 1: LQ Three Player Chain")
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

        # -- QP Solver --
        to_qp = TimerOutput()
        @timeit to_qp "QP total" begin
            solver = QPSolver(G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM; to=to_qp)
            params = Dict(i => x0[i] for i in 1:N)
            # Warmup
            solve(solver, params; to=to_qp)
            # Timed
            for _ in 1:num_solves
                solve(solver, params; to=to_qp)
            end
        end
        println("\n  QP Solver:")
        show(to_qp); println("\n")

        # -- NonlinearSolver on same LQ problem --
        to_nl = TimerOutput()
        @timeit to_nl "NonlinearSolver total" begin
            solver_nl = NonlinearSolver(
                G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
                max_iters = MAX_ITERS, tol = TOLERANCE, to = to_nl
            )
            params = Dict(i => x0[i] for i in 1:N)
            # Warmup
            solve(solver_nl, params; to=to_nl)
            # Timed
            for _ in 1:num_solves
                solve(solver_nl, params; to=to_nl)
            end
        end
        println("\n  NonlinearSolver on LQ Problem:")
        show(to_nl); println("\n")

        return to_qp, to_nl
    end
end

# ─────────────────────────────────────────────────────
# Experiment 2: Nonlinear Lane Change (4 players)
# ─────────────────────────────────────────────────────

module BenchLaneChange
    using MixedHierarchyGames
    using TimerOutputs: TimerOutput, @timeit
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!
    using LinearAlgebra: norm

    include(joinpath(@__DIR__, "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "common", "collision_avoidance.jl"))
    include(joinpath(@__DIR__, "common", "trajectory_utils.jl"))
    include(joinpath(@__DIR__, "nonlinear_lane_change", "config.jl"))
    include(joinpath(@__DIR__, "nonlinear_lane_change", "support.jl"))

    function run(; num_solves=3)
        println("=" ^ 70)
        println("  Experiment 2: Nonlinear Lane Change (4 players, unicycle)")
        println("=" ^ 70)

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

        to = TimerOutput()
        @timeit to "total" begin
            solver = NonlinearSolver(
                G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
                max_iters = MAX_ITERS, tol = TOLERANCE, to = to
            )
            params = Dict(i => x0[i] for i in 1:N)

            # Warmup
            result = solve_raw(solver, params; initial_guess = z0_guess, to = to)
            println("  Warmup: status=$(result.status), iters=$(result.iterations), residual=$(result.residual)")

            # Timed
            for _ in 1:num_solves
                result = solve_raw(solver, params; initial_guess = z0_guess, to = to)
            end
        end

        println("\n  Nonlinear Lane Change:")
        show(to); println("\n")
        return to
    end
end

# ─────────────────────────────────────────────────────
# Experiment 3: Pursuer-Protector-VIP (3 players)
# ─────────────────────────────────────────────────────

module BenchPPV
    using MixedHierarchyGames
    using TimerOutputs: TimerOutput, @timeit
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!
    using LinearAlgebra: norm

    include(joinpath(@__DIR__, "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "pursuer_protector_vip", "config.jl"))
    include(joinpath(@__DIR__, "pursuer_protector_vip", "support.jl"))

    function run(; num_solves=3)
        println("=" ^ 70)
        println("  Experiment 3: Pursuer-Protector-VIP (3 players)")
        println("=" ^ 70)

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
        x0 = DEFAULT_X0

        to = TimerOutput()
        @timeit to "total" begin
            solver = NonlinearSolver(
                G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
                max_iters = MAX_ITERS, tol = TOLERANCE, to = to
            )
            params = Dict(i => x0[i] for i in 1:N)

            # Warmup
            result = solve_raw(solver, params; to = to)
            println("  Warmup: status=$(result.status), iters=$(result.iterations), residual=$(result.residual)")

            # Timed
            for _ in 1:num_solves
                result = solve_raw(solver, params; to = to)
            end
        end

        println("\n  Pursuer-Protector-VIP:")
        show(to); println("\n")
        return to
    end
end

# ─────────────────────────────────────────────────────
# Main: Run all benchmarks
# ─────────────────────────────────────────────────────

println("\n" * "╌" ^ 70)
println("  MixedHierarchyGames Performance Benchmarks")
println("╌" ^ 70 * "\n")

BenchLQ.run()
println()
BenchLaneChange.run()
println()
BenchPPV.run()

println("\n" * "╌" ^ 70)
println("  All benchmarks complete.")
println("╌" ^ 70)
