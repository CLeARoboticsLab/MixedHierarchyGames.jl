using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    NonlinearSolver,
    setup_problem_variables,
    setup_problem_parameter_variables,
    setup_approximate_kkt_solver,
    preoptimize_nonlinear_solver,
    compute_K_evals,
    get_qp_kkt_conditions,
    strip_policy_constraints,
    unflatten_trajectory,
    run_nonlinear_solver,
    default_backend,
    MNFunctionWrapper

@testset "Typed KKT Dicts" begin
    # Setup: 2-player Stackelberg hierarchy (player 1 leads player 2)
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    primal_dims = [2, 2]
    backend = default_backend()
    gs = [z -> Num[] for _ in 1:2]

    @testset "Cold-path dicts: QP KKT construction uses Dict{Int, Any} (acceptable)" begin
        # These dicts contain heterogeneous symbolic types (BlockVector vs Vector{Num})
        # and are only accessed during one-time construction, never in the solve loop.
        # Verify they exist and document that Dict{Int, Any} is intentional here.
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )
        θ_all = reduce(vcat, (θs[k] for k in sort(collect(keys(θs)))))
        result = get_qp_kkt_conditions(
            G, Js, problem_vars.zs, problem_vars.λs, problem_vars.μs,
            gs, problem_vars.ws, problem_vars.ys, problem_vars.ws_z_indices;
            θ=θ_all
        )

        # πs, Ms, Ns, Ks are Dict{Int, Any} — acceptable for symbolic construction
        @test result.πs isa Dict{Int, Any}
        @test result.Ms isa Dict{Int, Any}
        @test result.Ns isa Dict{Int, Any}
        @test result.Ks isa Dict{Int, Any}
    end

    @testset "Cold-path: preoptimize π_sizes_trimmed is Dict{Int, Int}" begin
        # π_sizes_trimmed maps player → KKT condition count (Int values only).
        # Should be typed as Dict{Int, Int}, not Dict{Int, Any}.
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        θs = setup_problem_parameter_variables([2, 2]; backend)
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        precomputed = preoptimize_nonlinear_solver(G, Js, gs, primal_dims, θs)
        @test precomputed.π_sizes_trimmed isa Dict{Int, Int}
    end

    @testset "Hot-path: compute_K_evals containers are concretely typed" begin
        # The Newton iteration loop calls compute_K_evals every iteration.
        # Its containers must have concrete value types for type-stable access.
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

        # Hot-path containers: Vector-indexed with concrete Union types
        @test info.K_evals isa Vector{Union{Matrix{Float64}, Nothing}}
        @test info.M_evals isa Vector{Union{Matrix{Float64}, Nothing}}
        @test info.N_evals isa Vector{Union{Matrix{Float64}, Nothing}}

        # all_K_vec should be Vector{Float64} (not Any)
        @test all_K_vec isa Vector{Float64}

        # π_sizes should be Vector{Int} (not Dict)
        @test setup_info.π_sizes isa Vector{Int}

        # M_fns!/N_fns! should be Vector{MNFunctionWrapper}
        @test setup_info.var"M_fns!" isa Vector{MNFunctionWrapper}
        @test setup_info.var"N_fns!" isa Vector{MNFunctionWrapper}
    end

    @testset "Hot-path: M_buffers and N_buffers are Dict{Int, Matrix{Float64}}" begin
        # In run_nonlinear_solver, M_buffers and N_buffers are pre-allocated
        # as Dict{Int, Matrix{Float64}}. Verify this type is maintained when
        # passed through compute_K_evals.
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
        M_buffers = Dict{Int, Matrix{Float64}}()
        N_buffers = Dict{Int, Matrix{Float64}}()

        all_K_vec, info = compute_K_evals(
            z_current, problem_vars, setup_info;
            M_buffers=M_buffers, N_buffers=N_buffers
        )

        # After compute_K_evals, buffers should remain Dict{Int, Matrix{Float64}}
        @test M_buffers isa Dict{Int, Matrix{Float64}}
        @test N_buffers isa Dict{Int, Matrix{Float64}}

        # Buffers should have been populated for follower players
        @test haskey(M_buffers, 2)  # Player 2 has a leader
        @test haskey(N_buffers, 2)
        @test !haskey(M_buffers, 1)  # Player 1 is root (no M/N)
        @test !haskey(N_buffers, 1)
    end

    @testset "Hot-path: pre-allocated k_eval_buffers are correctly typed" begin
        # run_nonlinear_solver pre-allocates k_eval_buffers as a NamedTuple
        # of typed Vectors. Verify compute_K_evals accepts and uses them.
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

        N_players = 2
        z_current = zeros(length(problem_vars.all_variables))

        # Pre-allocate buffers (same pattern as run_nonlinear_solver)
        k_eval_buffers = (;
            M_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            N_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            K_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            follower_cache = Vector{Union{Vector{Int}, Nothing}}(nothing, N_players),
            buffer_cache = Vector{Union{Vector{Float64}, Nothing}}(nothing, N_players),
            all_K_vec = Vector{Float64}(undef, 4),  # 2x2 K matrix for player 2
        )

        all_K_vec, info = compute_K_evals(
            z_current, problem_vars, setup_info;
            buffers=k_eval_buffers
        )

        # Verify buffers were used (not freshly allocated)
        @test info.K_evals === k_eval_buffers.K_evals
        @test info.M_evals === k_eval_buffers.M_evals
        @test info.N_evals === k_eval_buffers.N_evals
    end

    @testset "End-to-end: solver produces correct results with typed containers" begin
        # Build a proper 2-player game and solve it to verify typed containers
        # don't break numerical results.
        state_dim = 2
        control_dim = 2
        T = 2
        primal_dim = (state_dim + control_dim) * (T + 1)
        nl_primal_dims = fill(primal_dim, 2)

        θs = setup_problem_parameter_variables(fill(state_dim, 2); backend)
        Δt = 0.1

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> begin
                (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
                sum(sum(x.^2) for x in xs) + 0.1 * sum(sum(u.^2) for u in us)
            end,
            2 => (z1, z2; θ=nothing) -> begin
                (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
                sum(sum(x.^2) for x in xs) + 0.1 * sum(sum(u.^2) for u in us)
            end
        )

        nl_gs = [
            z -> begin
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t; xs[t+1] - xs[t] - Δt * us[t] end
                ic = xs[1] - θs[1]
                vcat(dyn, ic)
            end,
            z -> begin
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t; xs[t+1] - xs[t] - Δt * us[t] end
                ic = xs[1] - θs[2]
                vcat(dyn, ic)
            end
        ]

        solver = NonlinearSolver(G, Js, nl_gs, nl_primal_dims, θs, state_dim, control_dim)
        initial_states = Dict(1 => [1.0, 0.5], 2 => [-0.5, 1.0])

        result = run_nonlinear_solver(
            solver.precomputed, initial_states, G;
            max_iters=50, tol=1e-6
        )

        @test result.converged
        @test result.residual < 1e-6
        @test result.sol isa Vector{Float64}
    end
end
