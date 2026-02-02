using Test
using Graphs: SimpleDiGraph, add_edge!
using MixedHierarchyGames: QPSolver, NonlinearSolver, HierarchyGame
using MixedHierarchyGames: extract_trajectories, solution_to_joint_strategy
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
        # Setup minimal game
        @test_skip "Implement full interface first"
    end

    @testset "solve_trajectory_game! returns JointStrategy (Nonlinear)" begin
        @test_skip "Implement full interface first"
    end

    @testset "Trajectories start from initial state" begin
        # First state in trajectory should match given initial state
        @test_skip "Implement full interface first"
    end

    @testset "Trajectory dimensions match game specification" begin
        # State and control dims should match what game defines
        @test_skip "Implement full interface first"
    end
end
