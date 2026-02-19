# Tests for the sparse_threshold heuristic in :auto mode.
# The :auto heuristic should use sparse solve based on M matrix size,
# not just graph position (leader vs leaf).

using Test
using LinearAlgebra: norm
using Graphs: SimpleDiGraph, add_edge!, nv
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    compute_K_evals,
    NonlinearSolverOptions,
    setup_problem_parameter_variables,
    has_leader,
    is_leaf

using TrajectoryGamesBase: unflatten_trajectory

# ── Test helpers ──────────────────────────────────────────────────────────

"""
    make_nash_game_large(; N=3, T=5, state_dim=4, control_dim=2)

Nash game with larger dimensions so M matrices exceed sparse threshold.
All players are leaves (no hierarchy edges).
"""
function make_nash_game_large(; N=3, T=5, state_dim=4, control_dim=2)
    G = SimpleDiGraph(N)
    # No edges — pure Nash game, all players are leaves

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end

    goals = [[Float64(i), Float64(i), 0.0, 0.0] for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)

    function make_dynamics_constraint(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                x_next = copy(xs[t])
                x_next[1] += us[t][1]
                x_next[2] += us[t][2]
                push!(constraints, xs[t+1] - x_next)
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end

    gs = [make_dynamics_constraint(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

"""
    make_two_player_small_nash(; T=2, state_dim=1, control_dim=1)

Tiny Nash game where M matrices are small (below any reasonable threshold).
"""
function make_two_player_small_nash(; T=2, state_dim=1, control_dim=1)
    N = 2
    G = SimpleDiGraph(N)
    # No edges — Nash

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end

    Js = Dict(
        1 => make_cost(1, [1.0]),
        2 => make_cost(2, [2.0]),
    )

    function make_dynamics_constraint(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end

    gs = [make_dynamics_constraint(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

# ── Tests ─────────────────────────────────────────────────────────────────

@testset "Sparse Threshold Heuristic" begin

    @testset "NonlinearSolverOptions accepts sparse_threshold" begin
        # Default value
        opts = NonlinearSolverOptions()
        @test opts.sparse_threshold == 50

        # Custom value
        opts2 = NonlinearSolverOptions(sparse_threshold=100)
        @test opts2.sparse_threshold == 100

        # Zero is valid (always use sparse in :auto)
        opts3 = NonlinearSolverOptions(sparse_threshold=0)
        @test opts3.sparse_threshold == 0
    end

    @testset "sparse_threshold validation" begin
        # Negative values are invalid
        @test_throws ArgumentError NonlinearSolverOptions(sparse_threshold=-1)
    end

    @testset "sparse_threshold in _merge_options" begin
        opts = NonlinearSolverOptions(sparse_threshold=50)
        merged = MixedHierarchyGames._merge_options(opts; sparse_threshold=100)
        @test merged.sparse_threshold == 100

        # Nothing preserves base value
        merged2 = MixedHierarchyGames._merge_options(opts; sparse_threshold=nothing)
        @test merged2.sparse_threshold == 50
    end

    @testset "compute_K_evals accepts sparse_threshold kwarg" begin
        # Build a problem with a leader-follower pair
        prob = make_standard_two_player_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )
        z_current = zeros(length(precomputed.all_variables))

        # Should accept sparse_threshold without error
        K_vec, info = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:auto, sparse_threshold=50
        )
        @test info.status == :ok
    end

    @testset ":auto with size-based heuristic — Nash game with large M" begin
        # In a Nash game, ALL players are leaves. Under the old heuristic,
        # :auto would use dense for all. Under the new heuristic, :auto should
        # use sparse for players with M size >= threshold.
        prob = make_nash_game_large()

        # Verify all players are leaves
        for ii in 1:prob.N
            @test is_leaf(prob.G, ii)
        end

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )
        z_current = randn(length(precomputed.all_variables))

        # With threshold=1 (essentially always sparse for any M), :auto should
        # match :always numerically
        K_auto, _ = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:auto, sparse_threshold=1
        )
        K_always, _ = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:always
        )

        @test norm(K_auto - K_always) / max(norm(K_always), 1.0) < 1e-10
    end

    @testset ":auto with high threshold keeps small M dense" begin
        # With a very high threshold, no M matrix should qualify for sparse
        # So :auto should match :never
        prob = make_two_player_small_nash()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )
        z_current = randn(length(precomputed.all_variables))

        K_auto, _ = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:auto, sparse_threshold=10000
        )
        K_never, _ = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:never
        )

        @test norm(K_auto - K_never) / max(norm(K_never), 1.0) < 1e-10
    end

    @testset "NonlinearSolver constructor passes sparse_threshold" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            sparse_threshold=75
        )
        @test solver.options.sparse_threshold == 75
    end

    @testset "Default sparse_threshold flows through solve_raw" begin
        # Verify the full solve path works with the new parameter
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            sparse_threshold=50
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])
        result = solve_raw(solver, params; max_iters=5)
        @test result.iterations <= 5
    end
end
