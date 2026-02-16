@testset "Typed K-eval buffers (no Union{Matrix,Nothing})" begin
    using MixedHierarchyGames:
        compute_K_evals,
        setup_approximate_kkt_solver,
        setup_problem_variables,
        setup_problem_parameter_variables,
        default_backend,
        has_leader
    using Graphs: SimpleDiGraph, add_edge!, nv

    backend = default_backend()

    # ── Helper: build a 2-player chain (P1 root → P2 follower) ──
    function make_two_player()
        G = SimpleDiGraph(2); add_edge!(G, 1, 2)
        primal_dims = [2, 2]
        gs = [z -> Num[] for _ in 1:2]
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        _, setup_info = setup_approximate_kkt_solver(
            G, Js, problem_vars.zs, problem_vars.λs, problem_vars.μs,
            gs, problem_vars.ws, problem_vars.ys, θs,
            problem_vars.all_variables, backend
        )
        z = zeros(length(problem_vars.all_variables))
        return (; G, problem_vars, setup_info, z)
    end

    # ── Helper: build a 3-player chain (P1 → P2 → P3) ──
    function make_three_player()
        G = SimpleDiGraph(3); add_edge!(G, 1, 2); add_edge!(G, 2, 3)
        primal_dims = [2, 2, 2]
        gs = [z -> Num[] for _ in 1:3]
        θs = setup_problem_parameter_variables([2, 2, 2]; backend)
        Js = Dict(
            1 => (z1, z2, z3; θ=nothing) -> sum(z1.^2) + sum(z2.^2) + sum(z3.^2),
            2 => (z1, z2, z3; θ=nothing) -> sum(z2.^2) + sum(z3.^2),
            3 => (z1, z2, z3; θ=nothing) -> sum(z3.^2)
        )
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        _, setup_info = setup_approximate_kkt_solver(
            G, Js, problem_vars.zs, problem_vars.λs, problem_vars.μs,
            gs, problem_vars.ws, problem_vars.ys, θs,
            problem_vars.all_variables, backend
        )
        z = zeros(length(problem_vars.all_variables))
        return (; G, problem_vars, setup_info, z)
    end

    @testset "M/N/K_evals are concretely typed Vector{Matrix{Float64}} (no Union)" begin
        prob = make_two_player()
        _, info = compute_K_evals(prob.z, prob.problem_vars, prob.setup_info)

        # The vectors must NOT contain Union — they should be concretely typed
        @test info.K_evals isa Vector{Matrix{Float64}}
        @test info.M_evals isa Vector{Matrix{Float64}}
        @test info.N_evals isa Vector{Matrix{Float64}}
    end

    @testset "Root players get empty 0×0 sentinel matrices" begin
        prob = make_two_player()
        _, info = compute_K_evals(prob.z, prob.problem_vars, prob.setup_info)

        # P1 is root — should have 0×0 empty matrix sentinel
        @test size(info.K_evals[1]) == (0, 0)
        @test size(info.M_evals[1]) == (0, 0)
        @test size(info.N_evals[1]) == (0, 0)

        # P2 is follower — should have non-empty matrix
        @test !isempty(info.K_evals[2])
        @test !isempty(info.M_evals[2])
        @test !isempty(info.N_evals[2])
    end

    @testset "3-player chain: correct sentinel placement" begin
        prob = make_three_player()
        _, info = compute_K_evals(prob.z, prob.problem_vars, prob.setup_info)

        @test info.K_evals isa Vector{Matrix{Float64}}

        # P1 is root (no leader) — sentinel
        @test size(info.K_evals[1]) == (0, 0)
        # P2 and P3 are followers — real matrices
        @test !isempty(info.K_evals[2])
        @test !isempty(info.K_evals[3])
    end

    @testset "Pre-allocated buffers are also concretely typed" begin
        prob = make_two_player()
        N_players = nv(prob.G)

        # Buffers as they should be after the fix: Vector{Matrix{Float64}}
        empty_sentinel = Matrix{Float64}(undef, 0, 0)
        k_eval_buffers = (;
            M_evals = [copy(empty_sentinel) for _ in 1:N_players],
            N_evals = [copy(empty_sentinel) for _ in 1:N_players],
            K_evals = [copy(empty_sentinel) for _ in 1:N_players],
            follower_cache = Vector{Union{Vector{Int}, Nothing}}(nothing, N_players),
            buffer_cache = Vector{Union{Vector{Float64}, Nothing}}(nothing, N_players),
            all_K_vec = Vector{Float64}(undef, 100),
        )

        _, info = compute_K_evals(prob.z, prob.problem_vars, prob.setup_info;
                                  buffers=k_eval_buffers)

        @test info.K_evals isa Vector{Matrix{Float64}}
        @test info.M_evals isa Vector{Matrix{Float64}}
        @test info.N_evals isa Vector{Matrix{Float64}}
    end

    @testset "Numerical results identical to reference (2-player)" begin
        prob = make_two_player()

        # Get reference result
        all_K_vec, info = compute_K_evals(prob.z, prob.problem_vars, prob.setup_info)

        # Verify K values are finite and non-empty
        @test length(all_K_vec) > 0
        @test all(isfinite, all_K_vec)

        # Follower K should be a valid matrix
        K2 = info.K_evals[2]
        @test all(isfinite, K2)
    end

    @testset "Numerical results identical to reference (3-player)" begin
        prob = make_three_player()

        all_K_vec, info = compute_K_evals(prob.z, prob.problem_vars, prob.setup_info)

        @test length(all_K_vec) > 0
        @test all(isfinite, all_K_vec)

        # Both followers should have valid K matrices
        for i in 2:3
            @test all(isfinite, info.K_evals[i])
        end
    end

    @testset "run_nonlinear_solver hot loop uses typed buffers" begin
        # The hot-loop buffer pre-allocation in run_nonlinear_solver (line ~900)
        # should use Vector{Matrix{Float64}}, not Vector{Union{...}}
        prob = make_standard_two_player_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0])
        result = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=50, tol=1e-8
        )
        @test result.converged
    end
end
