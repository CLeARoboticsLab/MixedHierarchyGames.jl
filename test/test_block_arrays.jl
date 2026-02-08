#=
    Tests for BlockArrays-based solution vector splitting.

    Verifies that BlockArrays utilities produce identical results to manual
    offset-based indexing for solution extraction.
=#

using Test
using BlockArrays: blocks, mortar
using MixedHierarchyGames: _extract_joint_strategy

@testset "BlockArrays Utilities" begin
    @testset "split_solution_vector splits flat vector by player dimensions" begin
        # A flat solution vector with known block sizes
        sol = collect(1.0:12.0)
        primal_dims = [4, 3, 5]

        player_blocks = MixedHierarchyGames.split_solution_vector(sol, primal_dims)

        # Should return iterable of blocks
        blks = collect(player_blocks)
        @test length(blks) == 3
        @test blks[1] == [1.0, 2.0, 3.0, 4.0]
        @test blks[2] == [5.0, 6.0, 7.0]
        @test blks[3] == [8.0, 9.0, 10.0, 11.0, 12.0]
    end

    @testset "split_solution_vector matches manual offset extraction" begin
        # Simulate realistic solution vector dimensions
        # 3 players, each with state_dim=4, control_dim=2, horizon T=3
        # primal_dim per player = (state_dim + control_dim) * T = 6 * 3 = 18
        primal_dims = [18, 18, 18]
        total_dim = sum(primal_dims)
        sol = randn(total_dim)

        # Manual offset extraction (current approach)
        manual_blocks = Vector{Vector{Float64}}(undef, 3)
        offset = 1
        for i in 1:3
            manual_blocks[i] = sol[offset:(offset + primal_dims[i] - 1)]
            offset += primal_dims[i]
        end

        # BlockArrays-based extraction
        block_results = collect(MixedHierarchyGames.split_solution_vector(sol, primal_dims))

        # They should be identical
        for i in 1:3
            @test block_results[i] ≈ manual_blocks[i] atol=1e-14
        end
    end

    @testset "split_solution_vector handles single player" begin
        sol = [1.0, 2.0, 3.0]
        primal_dims = [3]

        blks = collect(MixedHierarchyGames.split_solution_vector(sol, primal_dims))
        @test length(blks) == 1
        @test blks[1] == [1.0, 2.0, 3.0]
    end

    @testset "split_solution_vector handles unequal player dimensions" begin
        # Players with very different dimension sizes (common in mixed hierarchies)
        sol = collect(1.0:15.0)
        primal_dims = [2, 8, 5]

        blks = collect(MixedHierarchyGames.split_solution_vector(sol, primal_dims))
        @test length(blks) == 3
        @test length(blks[1]) == 2
        @test length(blks[2]) == 8
        @test length(blks[3]) == 5
        @test blks[1] == [1.0, 2.0]
        @test blks[2] == collect(3.0:10.0)
        @test blks[3] == collect(11.0:15.0)
    end

    @testset "split_solution_vector rejects mismatched dimensions" begin
        sol = [1.0, 2.0, 3.0, 4.0, 5.0]
        # block_sizes sum (4) < vector length (5) — would silently drop data without check
        @test_throws DimensionMismatch MixedHierarchyGames.split_solution_vector(sol, [2, 2])
        # block_sizes sum (6) > vector length (5)
        @test_throws DimensionMismatch MixedHierarchyGames.split_solution_vector(sol, [3, 3])
    end

    @testset "_extract_joint_strategy uses block splitting correctly" begin
        # unflatten_trajectory expects interleaved format: z = [x0; u0; x1; u1; ...]
        # state_dim=2, control_dim=1, T=2
        # primal_dim = (state_dim + control_dim) * T = 3 * 2 = 6 per player
        state_dim = 2
        control_dim = 1
        T = 2
        primal_dim = (state_dim + control_dim) * T

        # Player 1: [x0_1, x0_2, u0, x1_1, x1_2, u1] = [1,2,3,4,5,6]
        # Player 2: [x0_1, x0_2, u0, x1_1, x1_2, u1] = [7,8,9,10,11,12]
        sol = Float64.(collect(1:12))
        primal_dims = [primal_dim, primal_dim]

        strategy = _extract_joint_strategy(sol, primal_dims, state_dim, control_dim)

        # Player 1: xs = [[1,2], [4,5]], us = [[3], [6]]
        @test strategy.substrategies[1].xs[1] ≈ [1.0, 2.0] atol=1e-14
        @test strategy.substrategies[1].xs[2] ≈ [4.0, 5.0] atol=1e-14
        @test strategy.substrategies[1].us[1] ≈ [3.0] atol=1e-14
        @test strategy.substrategies[1].us[2] ≈ [6.0] atol=1e-14

        # Player 2: xs = [[7,8], [10,11]], us = [[9], [12]]
        @test strategy.substrategies[2].xs[1] ≈ [7.0, 8.0] atol=1e-14
        @test strategy.substrategies[2].xs[2] ≈ [10.0, 11.0] atol=1e-14
        @test strategy.substrategies[2].us[1] ≈ [9.0] atol=1e-14
        @test strategy.substrategies[2].us[2] ≈ [12.0] atol=1e-14
    end

    @testset "extract_trajectories uses block splitting correctly" begin
        # 2 players: state_dim=2, control_dim=1, T=2
        # Layout per player: [x0, x1, x2, u0, u1] (sequential, not interleaved)
        # primal_dim = (T+1)*state_dim + T*control_dim = 3*2 + 2*1 = 8
        state_dim = 2
        control_dim = 1
        T = 2

        # Build solution: player 1 then player 2
        # P1: x0=[1,2], x1=[3,4], x2=[5,6], u0=[7], u1=[8]
        # P2: x0=[9,10], x1=[11,12], x2=[13,14], u0=[15], u1=[16]
        sol = Float64.(collect(1:16))
        dims = (state_dims=[state_dim, state_dim], control_dims=[control_dim, control_dim])
        n_players = 2

        xs, us = MixedHierarchyGames.extract_trajectories(sol, dims, T, n_players)

        # Player 1 states
        @test xs[1][1] ≈ [1.0, 2.0] atol=1e-14
        @test xs[1][2] ≈ [3.0, 4.0] atol=1e-14
        @test xs[1][3] ≈ [5.0, 6.0] atol=1e-14

        # Player 1 controls
        @test us[1][1] ≈ [7.0] atol=1e-14
        @test us[1][2] ≈ [8.0] atol=1e-14

        # Player 2 states
        @test xs[2][1] ≈ [9.0, 10.0] atol=1e-14
        @test xs[2][2] ≈ [11.0, 12.0] atol=1e-14
        @test xs[2][3] ≈ [13.0, 14.0] atol=1e-14

        # Player 2 controls
        @test us[2][1] ≈ [15.0] atol=1e-14
        @test us[2][2] ≈ [16.0] atol=1e-14
    end

    @testset "extract_trajectories with heterogeneous player dimensions" begin
        # 3 players with different state/control dimensions
        # P1: state_dim=4, control_dim=2, T=3
        # P2: state_dim=2, control_dim=1, T=3
        # P3: state_dim=6, control_dim=3, T=3
        T = 3
        state_dims = [4, 2, 6]
        control_dims = [2, 1, 3]
        dims = (state_dims=state_dims, control_dims=control_dims)
        n_players = 3

        # Build per-player blocks manually, then concatenate
        # P1: (T+1)*4 + T*2 = 16 + 6 = 22
        # P2: (T+1)*2 + T*1 = 8 + 3 = 11
        # P3: (T+1)*6 + T*3 = 24 + 9 = 33
        p1_states = randn(16)  # 4 state vectors of dim 4
        p1_controls = randn(6) # 3 control vectors of dim 2
        p2_states = randn(8)   # 4 state vectors of dim 2
        p2_controls = randn(3) # 3 control vectors of dim 1
        p3_states = randn(24)  # 4 state vectors of dim 6
        p3_controls = randn(9) # 3 control vectors of dim 3

        sol = vcat(p1_states, p1_controls, p2_states, p2_controls, p3_states, p3_controls)

        xs, us = MixedHierarchyGames.extract_trajectories(sol, dims, T, n_players)

        # Verify correct number of timesteps per player
        for i in 1:3
            @test length(xs[i]) == T + 1
            @test length(us[i]) == T
        end

        # Verify dimensions of extracted vectors
        for i in 1:3
            for t in 1:(T+1)
                @test length(xs[i][t]) == state_dims[i]
            end
            for t in 1:T
                @test length(us[i][t]) == control_dims[i]
            end
        end

        # Verify actual values for P1
        for t in 1:(T+1)
            @test xs[1][t] ≈ p1_states[((t-1)*4+1):(t*4)] atol=1e-14
        end
        for t in 1:T
            @test us[1][t] ≈ p1_controls[((t-1)*2+1):(t*2)] atol=1e-14
        end

        # Verify actual values for P2
        for t in 1:(T+1)
            @test xs[2][t] ≈ p2_states[((t-1)*2+1):(t*2)] atol=1e-14
        end
        for t in 1:T
            @test us[2][t] ≈ p2_controls[((t-1)*1+1):(t*1)] atol=1e-14
        end

        # Verify actual values for P3
        for t in 1:(T+1)
            @test xs[3][t] ≈ p3_states[((t-1)*6+1):(t*6)] atol=1e-14
        end
        for t in 1:T
            @test us[3][t] ≈ p3_controls[((t-1)*3+1):(t*3)] atol=1e-14
        end
    end

    @testset "split_solution_vector for uniform per-timestep splitting" begin
        # Verify split_solution_vector handles uniform blocks (used for timestep extraction)
        # Simulate extracting T+1 state vectors of dim 4 from a flat state segment
        state_dim = 4
        T = 5
        state_data = collect(1.0:(state_dim * (T + 1)))  # 24 elements

        timestep_blocks = collect(MixedHierarchyGames.split_solution_vector(
            state_data, fill(state_dim, T + 1)
        ))

        @test length(timestep_blocks) == T + 1
        for t in 1:(T+1)
            @test timestep_blocks[t] == state_data[((t-1)*state_dim+1):(t*state_dim)]
        end

        # Same for control vectors: T blocks of control_dim
        control_dim = 2
        control_data = collect(1.0:(control_dim * T))  # 10 elements

        control_blocks = collect(MixedHierarchyGames.split_solution_vector(
            control_data, fill(control_dim, T)
        ))

        @test length(control_blocks) == T
        for t in 1:T
            @test control_blocks[t] == control_data[((t-1)*control_dim+1):(t*control_dim)]
        end
    end
end
