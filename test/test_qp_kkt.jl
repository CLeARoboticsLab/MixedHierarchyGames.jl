using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: I, norm
using BlockArrays: BlockVector, blocks
using MixedHierarchyGames: get_qp_kkt_conditions, strip_policy_constraints
using MixedHierarchyGames: setup_problem_variables, make_symbolic_vector

@testset "QP KKT Construction" begin
    @testset "Leaf player KKT conditions" begin
        # Single player (leaf) should have:
        # 1. Stationarity: ∇_z L = 0
        # 2. Constraints: g(z) = 0
        G = SimpleDiGraph(1)  # Single player, no edges

        # Setup: 2-dim decision variable, 1 constraint
        primal_dims = [2]
        gs = [z -> [z[1] + z[2] - 1.0]]  # Single equality constraint

        vars = setup_problem_variables(G, primal_dims, gs)

        # Simple quadratic cost: J = z₁² + z₂²
        Js = Dict(1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2))

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices)

        # Should return πs dict with KKT for player 1
        @test haskey(result.πs, 1)

        # Leaf player KKT: stationarity (2) + constraints (1) = 3 conditions
        @test length(result.πs[1]) == 3
    end

    @testset "Leader includes follower policy constraints" begin
        # 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [
            z -> [z[1] + z[2] - 1.0],  # P1 constraint
            z -> [z[1] - z[2]],         # P2 constraint
        ]

        vars = setup_problem_variables(G, primal_dims, gs)

        # Costs
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices)

        # Both players should have KKT conditions
        @test haskey(result.πs, 1)
        @test haskey(result.πs, 2)

        # P2 (leaf): stationarity (2) + constraints (1) = 3
        @test length(result.πs[2]) == 3

        # P1 (leader): stationarity (2+2) + policy constraint (2) + own constraints (1) = 7
        # (leader's KKT includes follower variables and policy)
        @test length(result.πs[1]) >= 5  # At least stationarity + constraints
    end

    @testset "Returns M and N matrices for followers" begin
        # 2-player: 1→2, P2 is follower so needs M, N
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1]], z -> [z[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices)

        # P2 has a leader, so should have M and N matrices
        @test haskey(result.Ms, 2)
        @test haskey(result.Ns, 2)

        # P1 is root, no M/N needed
        @test !haskey(result.Ms, 1)
    end
end

@testset "Strip Policy Constraints" begin
    @testset "Removes policy rows from leader KKT" begin
        # Setup 2-player game
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1]], z -> [z[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices)

        # Strip policy constraints
        πs_stripped = strip_policy_constraints(result.πs, G, vars.zs, gs)

        # Stripped should have fewer or equal conditions
        @test length(πs_stripped[1]) <= length(result.πs[1])

        # Leaf (P2) should be unchanged
        @test length(πs_stripped[2]) == length(result.πs[2])
    end

    @testset "Throws error on malformed πs" begin
        # Setup 2-player game
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1]], z -> [z[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)

        # Manually construct malformed πs with wrong length for leader
        # Leader π should have: grad_self(2) + grad_follower(2) + policy(2) + constraints(1) = 7
        # We give it 5 elements (missing policy constraint rows)
        malformed_πs = Dict(
            1 => make_symbolic_vector(:z, 99, 5),  # Wrong: should be 7
            2 => make_symbolic_vector(:z, 98, 3),  # Correct for leaf
        )

        # Should throw an error for malformed input
        # Note: The exact error type depends on which check fails first (indexing vs dimension).
        # A future improvement would add explicit validation that throws ArgumentError.
        @test_throws Exception strip_policy_constraints(malformed_πs, G, vars.zs, gs)
    end
end

@testset "KKT BlockVector Structure" begin
    @testset "Leader πs has BlockVector with correct block structure (2-player)" begin
        # 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [
            z -> [z[1] + z[2] - 1.0],  # P1: 1 constraint
            z -> [z[1] - z[2]],         # P2: 1 constraint
        ]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs,
                                        vars.ws, vars.ys, vars.ws_z_indices)

        # Leader (P1) KKT should be a BlockVector
        @test result.πs[1] isa BlockVector

        # Block structure: [grad_self(2), grad_f1(2), policy_f1(2), own_constraints(1)]
        blks = collect(blocks(result.πs[1]))
        @test length(blks) == 4
        @test length(blks[1]) == 2   # grad w.r.t. own vars
        @test length(blks[2]) == 2   # grad w.r.t. follower vars
        @test length(blks[3]) == 2   # policy constraint for follower
        @test length(blks[4]) == 1   # own constraints
    end

    @testset "Leaf πs remains a plain vector (not BlockVector)" begin
        # 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1]], z -> [z[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs,
                                        vars.ws, vars.ys, vars.ws_z_indices)

        # Leaf (P2) KKT should NOT be a BlockVector (no block structure needed)
        @test !(result.πs[2] isa BlockVector)
    end

    @testset "Leader πs has correct blocks (3-player chain)" begin
        # 3-player chain: 1→2→3
        # get_all_followers returns transitive followers:
        #   P1's followers: [P2, P3], P2's followers: [P3]
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [3, 2, 2]
        gs = [
            z -> [z[1] - 1.0],     # P1: 1 constraint
            z -> [z[1] + z[2]],     # P2: 1 constraint
            z -> [z[1]],            # P3: 1 constraint
        ]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2) + sum(vars.zs[3].^2),
            3 => (zs...; θ=nothing) -> sum(vars.zs[3].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs,
                                        vars.ws, vars.ys, vars.ws_z_indices)

        # P1 (root leader with 2 transitive followers: P2 and P3)
        # Blocks: [grad_self(3), grad_f1(2), policy_f1(2), grad_f2(2), policy_f2(2), own_constraints(1)]
        @test result.πs[1] isa BlockVector
        blks1 = collect(blocks(result.πs[1]))
        @test length(blks1) == 6
        @test length(blks1[1]) == 3   # grad w.r.t. own vars (dim 3)
        @test length(blks1[2]) == 2   # grad w.r.t. P2 vars (dim 2)
        @test length(blks1[3]) == 2   # policy constraint for P2
        @test length(blks1[4]) == 2   # grad w.r.t. P3 vars (dim 2)
        @test length(blks1[5]) == 2   # policy constraint for P3
        @test length(blks1[6]) == 1   # own constraints

        # P2 (mid-chain: leader of P3, follower of P1)
        # Blocks: [grad_self(2), grad_f1(2), policy_f1(2), own_constraints(1)]
        @test result.πs[2] isa BlockVector
        blks2 = collect(blocks(result.πs[2]))
        @test length(blks2) == 4
        @test length(blks2[1]) == 2   # grad w.r.t. own vars
        @test length(blks2[2]) == 2   # grad w.r.t. P3 vars
        @test length(blks2[3]) == 2   # policy constraint for P3
        @test length(blks2[4]) == 1   # own constraints

        # P3 (leaf): not a BlockVector
        @test !(result.πs[3] isa BlockVector)
    end

    @testset "strip_policy_constraints uses block structure correctly" begin
        # 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [
            z -> [z[1] + z[2] - 1.0],  # P1: 1 constraint
            z -> [z[1] - z[2]],         # P2: 1 constraint
        ]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs,
                                        vars.ws, vars.ys, vars.ws_z_indices)

        πs_stripped = strip_policy_constraints(result.πs, G, vars.zs, gs)

        # Leader stripped KKT: grad_self(2) + grad_follower(2) + own_constraints(1) = 5
        # (policy constraint block of size 2 was removed)
        @test length(πs_stripped[1]) == 5

        # Leaf unchanged
        @test length(πs_stripped[2]) == length(result.πs[2])
    end

    @testset "strip_policy_constraints on 3-player chain preserves only gradient+constraint blocks" begin
        # 3-player chain: 1→2→3
        # P1 has transitive followers [P2, P3], P2 has followers [P3]
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [3, 2, 2]
        gs = [
            z -> [z[1] - 1.0],     # P1: 1 constraint
            z -> [z[1] + z[2]],     # P2: 1 constraint
            z -> [z[1]],            # P3: 1 constraint
        ]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2) + sum(vars.zs[3].^2),
            3 => (zs...; θ=nothing) -> sum(vars.zs[3].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs,
                                        vars.ws, vars.ys, vars.ws_z_indices)

        πs_stripped = strip_policy_constraints(result.πs, G, vars.zs, gs)

        # P1: grad_self(3) + grad_f1(2) + grad_f2(2) + own_constraints(1) = 8
        # (removed policy_f1(2) + policy_f2(2))
        @test length(πs_stripped[1]) == 8

        # P2: grad_self(2) + grad_f1(2) + own_constraints(1) = 5 (removed policy_f1(2))
        @test length(πs_stripped[2]) == 5

        # P3 (leaf): unchanged
        @test length(πs_stripped[3]) == length(result.πs[3])
    end

    @testset "M and N matrices computed correctly from BlockVector πs" begin
        # Verify that M = jacobian(πs, ws) and N = jacobian(πs, ys) work
        # correctly when πs[ii] is a BlockVector
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1] + z[2] - 1.0], z -> [z[1] - z[2]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs,
                                        vars.ws, vars.ys, vars.ws_z_indices)

        # P2 (follower) should have M and N matrices
        @test haskey(result.Ms, 2) && haskey(result.Ns, 2) && haskey(result.Ks, 2)

        # M should be square and match the KKT dimension
        @test size(result.Ms[2], 1) == length(result.πs[2])
        @test size(result.Ms[2], 2) == length(vars.ws[2])

        # N should have rows = KKT dim, cols = ys dim
        @test size(result.Ns[2], 1) == length(result.πs[2])
        @test size(result.Ns[2], 2) == length(vars.ys[2])
    end
end
