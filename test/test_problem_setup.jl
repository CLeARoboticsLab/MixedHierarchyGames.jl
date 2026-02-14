using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using Symbolics: Num  # Only for type checking
using MixedHierarchyGames: setup_problem_parameter_variables, setup_problem_variables

@testset "Problem Setup" begin
    @testset "setup_problem_parameter_variables" begin
        # 3 players with different parameter dimensions (e.g., initial state dims)
        num_params = [4, 4, 4]  # Each player has 4-dim state

        θs = setup_problem_parameter_variables(num_params)

        # Should return Dict with entry per player
        @test θs isa Dict
        @test length(θs) == 3
        @test haskey(θs, 1) && haskey(θs, 2) && haskey(θs, 3)

        # Each θ should have correct dimension
        @test length(θs[1]) == 4
        @test length(θs[2]) == 4
        @test length(θs[3]) == 4

        # Should be symbolic variables
        @test all(x -> x isa Num, θs[1])
    end

    @testset "setup_problem_variables structure" begin
        # Simple 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        # Each player has 6-dim decision variable (e.g., state + control over horizon)
        primal_dims = [6, 6]

        # Simple constraint functions (dynamics placeholder)
        gs = [
            z -> zeros(Num, 2),  # P1: 2 constraints
            z -> zeros(Num, 2),  # P2: 2 constraints
        ]

        result = setup_problem_variables(G, primal_dims, gs)

        # Should return named tuple or struct with zs, λs, μs, ys, ws
        @test haskey(result, :zs) || hasproperty(result, :zs)
        @test haskey(result, :λs) || hasproperty(result, :λs)
        @test haskey(result, :ys) || hasproperty(result, :ys)
        @test haskey(result, :ws) || hasproperty(result, :ws)
    end

    @testset "setup_problem_variables dimensions" begin
        # 3-player chain: 1→2→3
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [4, 4, 4]
        constraint_dims = [2, 2, 2]

        gs = [
            z -> zeros(Num, constraint_dims[i]) for i in 1:3
        ]

        result = setup_problem_variables(G, primal_dims, gs)

        # zs should have correct primal dimensions
        @test length(result.zs[1]) == primal_dims[1]
        @test length(result.zs[2]) == primal_dims[2]
        @test length(result.zs[3]) == primal_dims[3]

        # λs should have correct constraint dimensions
        @test length(result.λs[1]) == constraint_dims[1]
        @test length(result.λs[2]) == constraint_dims[2]
        @test length(result.λs[3]) == constraint_dims[3]
    end

    @testset "information vectors (ys) follow hierarchy" begin
        # Chain: 1→2→3
        # P1's info: just state (no leaders)
        # P2's info: state + P1's decision
        # P3's info: state + P1's decision + P2's decision
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [4, 4, 4]
        gs = [z -> zeros(Num, 2) for _ in 1:3]

        result = setup_problem_variables(G, primal_dims, gs)

        # P1 (root): ys should be empty or just parameters
        @test length(result.ys[1]) == 0

        # P2: ys should include P1's decisions
        @test length(result.ys[2]) == primal_dims[1]

        # P3: ys should include P1's and P2's decisions
        @test length(result.ys[3]) == primal_dims[1] + primal_dims[2]
    end

    @testset "mixed hierarchy info vectors" begin
        # P1 Nash with P2, P2→P3
        G = SimpleDiGraph(3)
        add_edge!(G, 2, 3)  # Only P2→P3

        primal_dims = [4, 4, 4]
        gs = [z -> zeros(Num, 2) for _ in 1:3]

        result = setup_problem_variables(G, primal_dims, gs)

        # P1 (root, Nash): no leaders
        @test length(result.ys[1]) == 0

        # P2 (root, Nash): no leaders
        @test length(result.ys[2]) == 0

        # P3 (follower of P2): has P2's decisions
        @test length(result.ys[3]) == primal_dims[2]
    end

    @testset "ws construction: exact symbolic content for chain 1→2→3" begin
        # Chain: 1→2→3
        # P1 is leader of P2, P2 is leader of P3
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [3, 4, 5]
        constraint_dims = [2, 3, 2]
        gs = [z -> zeros(Num, constraint_dims[i]) for i in 1:3]

        result = setup_problem_variables(G, primal_dims, gs)
        (; zs, λs, μs, ws, ws_z_indices) = result

        # P1 (root leader): ws[1] = [zs[1], zs[2], zs[3], λs[1], λs[2], λs[3], μs[(1,2)], μs[(1,3)]]
        # P1 has no leaders, so all z variables appear. Followers: 2, 3 (all descendants).
        # μs keys for P1: (1,2) and (1,3) since both are in get_all_followers
        @test length(ws[1]) == primal_dims[1] + primal_dims[2] + primal_dims[3] +
                               constraint_dims[1] + constraint_dims[2] + constraint_dims[3] +
                               primal_dims[2] + primal_dims[3]  # μs[(1,2)] + μs[(1,3)]

        # Verify exact symbolic variables: first block is zs[1]
        @test isequal(ws[1][1:3], zs[1])
        # Next is zs[2] (non-leader, non-self)
        @test isequal(ws[1][4:7], zs[2])
        # Then zs[3]
        @test isequal(ws[1][8:12], zs[3])
        # Then λs[1]
        @test isequal(ws[1][13:14], λs[1])
        # Then λs for followers (2 and 3)
        @test isequal(ws[1][15:17], λs[2])
        @test isequal(ws[1][18:19], λs[3])
        # Then μs[(1,2)] and μs[(1,3)]
        @test isequal(ws[1][20:23], μs[(1, 2)])
        @test isequal(ws[1][24:28], μs[(1, 3)])

        # Verify ws_z_indices for P1
        @test ws_z_indices[1][1] == 1:3
        @test ws_z_indices[1][2] == 4:7
        @test ws_z_indices[1][3] == 8:12

        # P2 (mid-level): leaders = [1]. ws[2] = [zs[2], zs[3], λs[2], λs[3], μs[(2,3)]]
        # P2's leaders: P1. So zs[1] excluded from ws. Non-leader non-self: zs[3].
        @test length(ws[2]) == primal_dims[2] + primal_dims[3] +
                               constraint_dims[2] + constraint_dims[3] +
                               primal_dims[3]  # μs[(2,3)]

        @test isequal(ws[2][1:4], zs[2])
        @test isequal(ws[2][5:9], zs[3])
        @test isequal(ws[2][10:12], λs[2])
        @test isequal(ws[2][13:14], λs[3])
        @test isequal(ws[2][15:19], μs[(2, 3)])

        @test ws_z_indices[2][2] == 1:4
        @test ws_z_indices[2][3] == 5:9

        # P3 (leaf follower): leaders = [1, 2]. ws[3] = [zs[3], λs[3]]
        # All other z variables are leaders, so only zs[3]. No followers → no μs.
        @test length(ws[3]) == primal_dims[3] + constraint_dims[3]
        @test isequal(ws[3][1:5], zs[3])
        @test isequal(ws[3][6:7], λs[3])
        @test ws_z_indices[3][3] == 1:5
    end

    @testset "ws construction: Nash game (no edges)" begin
        # 2-player Nash: no hierarchy edges
        G = SimpleDiGraph(2)
        primal_dims = [3, 4]
        gs = [z -> zeros(Num, 2), z -> zeros(Num, 3)]

        result = setup_problem_variables(G, primal_dims, gs)
        (; zs, λs, ws, ws_z_indices) = result

        # P1: no leaders, no followers. ws[1] = [zs[1], zs[2], λs[1]]
        @test length(ws[1]) == 3 + 4 + 2
        @test isequal(ws[1][1:3], zs[1])
        @test isequal(ws[1][4:7], zs[2])
        @test isequal(ws[1][8:9], λs[1])
        @test ws_z_indices[1][1] == 1:3
        @test ws_z_indices[1][2] == 4:7

        # P2: no leaders, no followers. ws[2] = [zs[2], zs[1], λs[2]]
        @test length(ws[2]) == 4 + 3 + 3
        @test isequal(ws[2][1:4], zs[2])
        @test isequal(ws[2][5:7], zs[1])
        @test isequal(ws[2][8:10], λs[2])
        @test ws_z_indices[2][2] == 1:4
        @test ws_z_indices[2][1] == 5:7
    end

    @testset "coupled constraints are rejected" begin
        using MixedHierarchyGames: default_backend, make_symbolic_vector

        # Create symbolic variables manually to simulate coupled constraints
        backend = default_backend()
        z1_sym = make_symbolic_vector(:z, 1, 4; backend)
        z2_sym = make_symbolic_vector(:z, 2, 4; backend)

        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4]

        # Player 2's constraint incorrectly references player 1's variables (coupled)
        # This captures z1_sym in a closure
        gs_coupled = [
            z -> zeros(Num, 2),  # P1: valid decoupled constraint
            z -> [z[1] - z1_sym[1], z[2] - z1_sym[2]],  # P2: INVALID - references z1
        ]

        # Should throw ArgumentError for coupled constraint
        @test_throws ArgumentError setup_problem_variables(G, primal_dims, gs_coupled)
    end
end
