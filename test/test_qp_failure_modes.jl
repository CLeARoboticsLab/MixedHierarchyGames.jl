using Test
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm, I, cond, SingularException
using MixedHierarchyGames:
    qp_game_linsolve,
    solve_qp_linear,
    _run_qp_solver,
    QPSolver,
    setup_problem_variables,
    get_qp_kkt_conditions,
    strip_policy_constraints

# make_θ helper is provided by testing_utils.jl (included in runtests.jl)

@testset "QP Solver Failure Modes" begin

    @testset "qp_game_linsolve - singular matrix throws" begin
        # Exactly singular matrix: duplicate rows
        A = [1.0 2.0; 1.0 2.0]
        b = [3.0, 3.0]

        # Julia's \ on a singular dense matrix throws SingularException.
        # qp_game_linsolve is a thin wrapper that does not catch exceptions —
        # exception handling is done by the caller (solve_qp_linear).
        @test_throws SingularException qp_game_linsolve(A, b)
    end

    @testset "qp_game_linsolve - near-singular matrix" begin
        # Near-singular matrix with high condition number
        A = [1.0 1.0; 1.0 1.0 + 1e-15]
        b = [2.0, 2.0]

        x = qp_game_linsolve(A, b)

        # Solution exists but is unreliable due to ill-conditioning
        @test cond(A) > 1e14
        # The solve may produce finite but inaccurate results, or NaN/Inf
        # We just verify the function runs without throwing
    end

    @testset "qp_game_linsolve - zero matrix throws" begin
        # All-zero matrix is singular
        A = zeros(3, 3)
        b = ones(3)

        @test_throws SingularException qp_game_linsolve(A, b)
    end

    @testset "solve_qp_linear - singular KKT from duplicate constraints" begin
        # Single player with duplicate constraints creates a singular KKT system.
        # min z² s.t. z = θ, z = θ (same constraint twice)
        #
        # KKT matrix structure:
        #   [2  -1  -1]   [z ]   [0]
        #   [1   0   0] * [λ1] = [θ]
        #   [1   0   0]   [λ2]   [θ]
        #
        # Rows 2 and 3 are identical → matrix is singular.
        G = SimpleDiGraph(1)
        primal_dims = [1]

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)

        # Two identical constraints: both enforce z = θ
        gs = [z -> [z[1] - θ_vec[1], z[1] - θ_vec[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        parameter_values = Dict(1 => [1.0])

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        @test result.status == :failed
    end

    @testset "solve_qp_linear - returns :failed status with NaN-filled vector" begin
        # Verify the contract: on failure, solution vector is filled with NaN
        G = SimpleDiGraph(1)
        primal_dims = [1]

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)

        # Duplicate constraints → singular KKT
        gs = [z -> [z[1] - θ_vec[1], z[1] - θ_vec[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        parameter_values = Dict(1 => [1.0])

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        @test result.status == :failed
        @test all(isnan, result.sol)
    end

    @testset "QPSolver solve() - throws error with actionable message on singular KKT" begin
        # QPSolver.solve() should throw an informative error when the KKT system is singular.
        # Use state_dim=1, control_dim=0 so primal_dim=1*T but we need at least
        # state_dim + control_dim > 0. With T implied by primal_dim / (state_dim + control_dim).
        G = SimpleDiGraph(1)
        primal_dims = [2]  # e.g., 2 timesteps * (state_dim=1 + control_dim=0) -- but control_dim can't be 0
        state_dim = 1
        control_dim = 1  # primal_dim = T * (1+1) = 2 → T=1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)

        # Two identical constraints to make KKT singular
        gs = [z -> [z[1] - θ_vec[1], z[1] - θ_vec[1]]]

        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        err = try
            MixedHierarchyGames.solve(solver, Dict(1 => [1.0]))
            nothing
        catch e
            e
        end

        @test err !== nothing
        @test err isa ErrorException
        @test occursin("singular", lowercase(err.msg)) || occursin("ill-conditioned", lowercase(err.msg))
        @test occursin("QPSolver", err.msg)
    end

    @testset "QPSolver solve_raw() - returns :failed without throwing" begin
        # solve_raw() should NOT throw, but return :failed status
        G = SimpleDiGraph(1)
        primal_dims = [2]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)

        gs = [z -> [z[1] - θ_vec[1], z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # solve_raw should not throw
        result = MixedHierarchyGames.solve_raw(solver, Dict(1 => [1.0]))

        @test result.status == :failed
        @test all(isnan, result.sol)
        @test result.info === nothing  # linear solver has no extra info
    end

    @testset "Near-singular KKT - high condition number" begin
        # Constraints that are nearly parallel create an ill-conditioned KKT system.
        # g1: z = θ
        # g2: z = θ + ε  (nearly the same constraint)
        #
        # This should still solve (not exactly singular) but produce a warning
        # about high residual when verbose=true.
        G = SimpleDiGraph(1)
        primal_dims = [1]

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)

        ε = 1e-14  # Tiny perturbation makes constraints nearly parallel
        gs = [z -> [z[1] - θ_vec[1], z[1] - θ_vec[1] - ε]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        parameter_values = Dict(1 => [1.0])

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        # Near-singular may either:
        # 1. Return :failed (NaN/Inf detected), or
        # 2. Return :solved with poor accuracy
        # Both are acceptable behaviors for a near-singular system.
        @test result.status ∈ (:solved, :failed)

        if result.status == :solved
            # If it "solved", the solution may be wildly inaccurate
            # due to the ill-conditioned system. We just verify it returned finite values.
            @test all(isfinite, result.sol)
        else
            # If it failed, solution should be NaN-filled
            @test all(isnan, result.sol)
        end
    end

    @testset "Error message mentions KKT system" begin
        # Verify the error message is actionable: it should mention the KKT system
        # so the user knows what to investigate.
        G = SimpleDiGraph(1)
        primal_dims = [2]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)

        gs = [z -> [z[1] - θ_vec[1], z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        @test_throws ErrorException MixedHierarchyGames.solve(solver, Dict(1 => [1.0]))

        # Verify the message content
        try
            MixedHierarchyGames.solve(solver, Dict(1 => [1.0]))
        catch e
            @test occursin("KKT", e.msg)
            @test occursin("singular", lowercase(e.msg)) || occursin("ill-conditioned", lowercase(e.msg))
        end
    end

end
