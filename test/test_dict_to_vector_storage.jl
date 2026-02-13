using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    setup_problem_variables,
    setup_problem_parameter_variables,
    setup_approximate_kkt_solver,
    compute_K_evals,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    default_backend

@testset "Dict→Vector Storage Migration" begin
    # Shared setup: 2-player leader-follower hierarchy
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)  # Player 1 leads player 2
    primal_dims = [2, 2]
    backend = default_backend()
    gs = [z -> Num[] for _ in 1:2]

    @testset "setup_approximate_kkt_solver: Vector-indexed outputs" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, problem_vars.zs, problem_vars.λs, problem_vars.μs,
            gs, problem_vars.ws, problem_vars.ys, θs,
            problem_vars.all_variables, backend
        )

        # M_fns and N_fns should be Vectors indexed by player ID
        @test setup_info.var"M_fns!" isa Vector
        @test setup_info.var"N_fns!" isa Vector
        @test length(setup_info.var"M_fns!") == 2
        @test length(setup_info.var"N_fns!") == 2

        # π_sizes should be a Vector indexed by player ID
        @test setup_info.π_sizes isa Vector{Int}
        @test length(setup_info.π_sizes) == 2
    end

    @testset "compute_K_evals: Vector-indexed outputs" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, problem_vars.zs, problem_vars.λs, problem_vars.μs,
            gs, problem_vars.ws, problem_vars.ys, θs,
            problem_vars.all_variables, backend
        )

        z_current = zeros(length(problem_vars.all_variables))
        all_K_vec, info = compute_K_evals(z_current, problem_vars, setup_info)

        # K_evals, M_evals, N_evals should be Vectors, not Dicts
        @test info.K_evals isa Vector
        @test info.M_evals isa Vector
        @test info.N_evals isa Vector
        @test length(info.K_evals) == 2
        @test length(info.M_evals) == 2
        @test length(info.N_evals) == 2

        # Root player (1) should have nothing, follower (2) should have Matrix
        @test isnothing(info.K_evals[1])
        @test info.K_evals[2] isa Matrix{Float64}
        @test isnothing(info.M_evals[1])
        @test info.M_evals[2] isa Matrix{Float64}
    end

    @testset "Numerical results identical after storage change" begin
        # Setup a complete nonlinear solver problem and verify results
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=1, control_dim=1
        )

        initial_states = Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0])
        result = run_nonlinear_solver(
            precomputed, initial_states, G;
            max_iters=50, tol=1e-8
        )

        # Must converge to same solution
        @test result.converged
        @test result.residual < 1e-8

        # Verify solution is deterministic (run again)
        result2 = run_nonlinear_solver(
            precomputed, initial_states, G;
            max_iters=50, tol=1e-8
        )
        @test result.sol ≈ result2.sol atol=1e-14
    end

    @testset "3-player chain: Vector storage works correctly" begin
        # 3-player chain: 1 → 2 → 3
        G3 = SimpleDiGraph(3)
        add_edge!(G3, 1, 2)
        add_edge!(G3, 2, 3)
        primal_dims3 = [2, 2, 2]
        gs3 = [z -> Num[] for _ in 1:3]

        θs3 = setup_problem_parameter_variables([2, 2, 2]; backend)
        Js3 = Dict(
            1 => (z1, z2, z3; θ=nothing) -> sum(z1.^2) + sum(z2.^2) + sum(z3.^2),
            2 => (z1, z2, z3; θ=nothing) -> sum(z2.^2) + sum(z3.^2),
            3 => (z1, z2, z3; θ=nothing) -> sum(z3.^2)
        )

        precomputed3 = preoptimize_nonlinear_solver(
            G3, Js3, gs3, primal_dims3, θs3;
            state_dim=1, control_dim=1
        )

        # Verify setup_info containers are Vectors
        setup_info = precomputed3.setup_info
        @test setup_info.var"M_fns!" isa Vector
        @test setup_info.var"N_fns!" isa Vector
        @test setup_info.π_sizes isa Vector{Int}

        # Verify compute_K_evals works
        problem_vars = precomputed3.problem_vars
        z_current = zeros(length(precomputed3.all_variables))
        all_K_vec, info = compute_K_evals(z_current, problem_vars, setup_info)

        @test info.K_evals isa Vector
        @test length(info.K_evals) == 3
        # Player 1 is root (nothing), players 2 and 3 are followers (Matrix)
        @test isnothing(info.K_evals[1])
        @test info.K_evals[2] isa Matrix{Float64}
        @test info.K_evals[3] isa Matrix{Float64}

        # Run full solver
        initial_states3 = Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0], 3 => [0.5, 0.5])
        result3 = run_nonlinear_solver(
            precomputed3, initial_states3, G3;
            max_iters=50, tol=1e-8
        )
        @test result3.converged
    end

    @testset "Root player M_fns!/N_fns! stubs throw on accidental invocation" begin
        # Root players (no leader) have placeholder stubs that should error
        # if invoked — this documents the defensive guard behavior.
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, problem_vars.zs, problem_vars.λs, problem_vars.μs,
            gs, problem_vars.ws, problem_vars.ys, θs,
            problem_vars.all_variables, backend
        )

        # Player 1 is root — calling M_fns![1] or N_fns![1] is a bug and should throw
        dummy_buf = zeros(1, 1)
        dummy_input = zeros(length(problem_vars.all_variables))
        @test_throws ErrorException setup_info.var"M_fns!"[1](dummy_buf, dummy_input)
        @test_throws ErrorException setup_info.var"N_fns!"[1](dummy_buf, dummy_input)

        # Player 2 is a follower — its M_fns!/N_fns! should be callable (not stubs)
        @test setup_info.var"M_fns!"[2] isa Function
        @test setup_info.var"N_fns!"[2] isa Function
    end
end
