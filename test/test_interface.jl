using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: @variables
using MixedHierarchyGames: QPSolver, NonlinearSolver, HierarchyGame
using MixedHierarchyGames: extract_trajectories, solution_to_joint_strategy, solve
using TrajectoryGamesBase: solve_trajectory_game!, JointStrategy, OpenLoopStrategy

@testset "Solution Extraction" begin
    @testset "extract_trajectories reshapes correctly" begin
        # Flat solution vector → Dict of trajectories per player
        T = 3  # horizon
        n_players = 2
        state_dim = 4
        control_dim = 2

        # Total size: (state_dim + control_dim) * T * n_players
        total_dim = (state_dim * (T + 1) + control_dim * T) * n_players
        z_sol = randn(total_dim)

        dims = (
            state_dims = [state_dim, state_dim],
            control_dims = [control_dim, control_dim],
        )

        xs, us = extract_trajectories(z_sol, dims, T, n_players)

        # Should return dicts indexed by player
        @test xs isa Dict
        @test us isa Dict
        @test haskey(xs, 1) && haskey(xs, 2)
        @test haskey(us, 1) && haskey(us, 2)

        # Correct trajectory lengths
        @test length(xs[1]) == T + 1  # States: t=0 to t=T
        @test length(us[1]) == T      # Controls: t=0 to t=T-1

        # Correct dimensions per timestep
        @test length(xs[1][1]) == state_dim
        @test length(us[1][1]) == control_dim
    end

    @testset "solution_to_joint_strategy" begin
        T = 3
        n_players = 2
        state_dim = 4
        control_dim = 2

        # Create dummy trajectories
        xs = Dict(
            1 => [randn(state_dim) for _ in 1:(T+1)],
            2 => [randn(state_dim) for _ in 1:(T+1)],
        )
        us = Dict(
            1 => [randn(control_dim) for _ in 1:T],
            2 => [randn(control_dim) for _ in 1:T],
        )

        strategy = solution_to_joint_strategy(xs, us, n_players)

        # Should return JointStrategy
        @test strategy isa JointStrategy

        # Should have correct number of substrategies
        @test length(strategy.substrategies) == n_players

        # Each substrategy should be OpenLoopStrategy
        @test all(s -> s isa OpenLoopStrategy, strategy.substrategies)
    end
end

@testset "TrajectoryGamesBase Interface" begin
    @testset "solve_trajectory_game! returns JointStrategy (QP)" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]  # T=2 timesteps, state_dim=1, control_dim=1
        state_dim = 1
        control_dim = 1

        @variables θ[1:1]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)
        strategy = solve(solver, Dict(1 => [1.0]))

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 1
        @test strategy.substrategies[1] isa OpenLoopStrategy
    end

    @testset "solve_trajectory_game! returns JointStrategy (Nonlinear)" begin
        @test_skip "Implement NonlinearSolver first"
    end

    @testset "Trajectories start from initial state (QP)" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        @variables θ[1:1]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]  # IC constraint: x0 = θ
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # Test with initial state = 3.0
        strategy = solve(solver, Dict(1 => [3.0]))

        # First state should match initial state
        @test strategy.substrategies[1].xs[1][1] ≈ 3.0 atol=1e-6
    end

    @testset "Trajectory dimensions match game specification (QP)" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 1
        T = 3  # timesteps
        primal_dims = [(state_dim + control_dim) * T, (state_dim + control_dim) * T]

        @variables θ1[1:state_dim] θ2[1:state_dim]
        θ1_vec = collect(θ1)
        θ2_vec = collect(θ2)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)

        # IC constraints
        gs = [
            z -> z[1:state_dim] - θ1_vec,
            z -> z[1:state_dim] - θ2_vec,
        ]

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)
        strategy = solve(solver, Dict(1 => [1.0, 2.0], 2 => [3.0, 4.0]))

        # Check dimensions
        @test length(strategy.substrategies) == 2
        @test length(strategy.substrategies[1].xs) == T  # T states
        @test length(strategy.substrategies[1].us) == T  # T controls
        @test length(strategy.substrategies[1].xs[1]) == state_dim
        @test length(strategy.substrategies[1].us[1]) == control_dim
    end
end
