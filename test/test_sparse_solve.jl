using Test
using SparseArrays: sparse, nnz
using LinearAlgebra: norm
using Graphs: SimpleDiGraph, add_edge!, nv
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    compute_K_evals,
    setup_approximate_kkt_solver,
    setup_problem_variables,
    setup_problem_parameter_variables,
    default_backend,
    has_leader

using TrajectoryGamesBase: unflatten_trajectory

#=
    Test helpers: Build problems and extract M, N matrices for sparsity analysis
=#

"""
    extract_M_N_matrices(prob; z_current=nothing)

Build a nonlinear solver for `prob` and evaluate M, N matrices at `z_current`.
Returns (M_evals, N_evals, precomputed) where M_evals/N_evals are Dict{Int, Matrix}.
"""
function extract_M_N_matrices(prob; z_current=nothing)
    precomputed = preoptimize_nonlinear_solver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
        state_dim=prob.state_dim,
        control_dim=prob.control_dim,
        verbose=false
    )

    if isnothing(z_current)
        z_current = zeros(length(precomputed.all_variables))
    end

    _, K_info = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info)
    return K_info.M_evals, K_info.N_evals, precomputed
end

"""
    make_two_player_chain(; T=3, state_dim=2, control_dim=2)

Simple 2-player chain: P1 -> P2.
"""
function make_two_player_chain(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        goal = [2.0, 2.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2)

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

"""
    make_three_player_chain(; T=3, state_dim=2, control_dim=2)

3-player chain: P1 -> P2 -> P3.
"""
function make_three_player_chain(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function J1(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        goal = [2.0, 2.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J3(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
        goal = [3.0, 3.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2, 3 => J3)

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

"""
    make_four_player_chain(; T=5, state_dim=4, control_dim=2)

4-player chain: P1 -> P2 -> P3 -> P4, with larger dimensions
to approximate the lane-change problem scale.
"""
function make_four_player_chain(; T=5, state_dim=4, control_dim=2)
    N = 4
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    add_edge!(G, 3, 4)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    # Cost function factory
    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end

    goals = [[1.0, 1.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0],
             [3.0, 3.0, 0.0, 0.0], [4.0, 4.0, 0.0, 0.0]]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)

    function make_dynamics_constraint(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                # Simple integrator: x_{t+1} = x_t + B*u_t
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

#=
    Sparsity measurement utilities
=#

"""
    sparsity_ratio(M::Matrix)

Return the fraction of zeros in M (0.0 = fully dense, 1.0 = fully sparse).
"""
function sparsity_ratio(M::Matrix)
    total = length(M)
    nonzeros = count(!iszero, M)
    return 1.0 - nonzeros / total
end

@testset "Sparse M\\N Solve" begin

    # Note: sparse(M) \ sparse(N) is NOT supported in Julia's SparseArrays.
    # The correct approach is sparse(M) \ N where N remains a dense Matrix.
    # This exploits sparse LU/QR factorization of M while keeping N dense.

    @testset "sparse(M) \\ N matches dense M \\ N — 2-player" begin
        prob = make_two_player_chain()
        M_evals, N_evals, _ = extract_M_N_matrices(prob)

        for (ii, M) in M_evals
            isnothing(M) && continue
            N_mat = N_evals[ii]

            K_dense = M \ N_mat
            K_sparse = sparse(M) \ N_mat

            @test norm(K_sparse - K_dense) / max(norm(K_dense), 1.0) < 1e-10
        end
    end

    @testset "sparse(M) \\ N matches dense M \\ N — 3-player" begin
        prob = make_three_player_chain()
        M_evals, N_evals, _ = extract_M_N_matrices(prob)

        for (ii, M) in M_evals
            isnothing(M) && continue
            N_mat = N_evals[ii]

            K_dense = M \ N_mat
            K_sparse = sparse(M) \ N_mat

            @test norm(K_sparse - K_dense) / max(norm(K_dense), 1.0) < 1e-10
        end
    end

    @testset "sparse(M) \\ N matches dense M \\ N — 4-player chain" begin
        prob = make_four_player_chain()
        M_evals, N_evals, _ = extract_M_N_matrices(prob)

        for (ii, M) in M_evals
            isnothing(M) && continue
            N_mat = N_evals[ii]

            K_dense = M \ N_mat
            K_sparse = sparse(M) \ N_mat

            @test norm(K_sparse - K_dense) / max(norm(K_dense), 1.0) < 1e-10
        end
    end

    @testset "Sparsity analysis — all problem sizes" begin
        problems = [
            ("2-player (T=3, s=2, c=2)", make_two_player_chain()),
            ("3-player (T=3, s=2, c=2)", make_three_player_chain()),
            ("4-player (T=5, s=4, c=2)", make_four_player_chain()),
        ]

        for (label, prob) in problems
            M_evals, N_evals, _ = extract_M_N_matrices(prob)

            for ii in sort(collect(keys(M_evals)))
                M = M_evals[ii]
                isnothing(M) && continue
                N_mat = N_evals[ii]

                sr_M = sparsity_ratio(M)
                sr_N = sparsity_ratio(N_mat)
                nnz_M = count(!iszero, M)
                nnz_N = count(!iszero, N_mat)

                @test 0.0 <= sr_M <= 1.0
                @test 0.0 <= sr_N <= 1.0

                @info "$label — Player $ii: M=$(size(M)) nnz=$(nnz_M)/$(length(M)) sparsity=$(round(sr_M, digits=3)), N=$(size(N_mat)) nnz=$(nnz_N)/$(length(N_mat)) sparsity=$(round(sr_N, digits=3))"
            end
        end
    end

    @testset "Sparsity at non-zero operating point" begin
        # Sparsity might change when evaluated at a non-zero point.
        # Use random z to check if sparsity is inherent to structure, not just z=0.
        prob = make_three_player_chain()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim,
            verbose=false
        )

        z_random = randn(length(precomputed.all_variables))
        _, K_info = compute_K_evals(z_random, precomputed.problem_vars, precomputed.setup_info)

        for ii in sort(collect(keys(K_info.M_evals)))
            M = K_info.M_evals[ii]
            isnothing(M) && continue
            N_mat = K_info.N_evals[ii]

            sr_M = sparsity_ratio(M)
            sr_N = sparsity_ratio(N_mat)

            @test 0.0 <= sr_M <= 1.0
            @test 0.0 <= sr_N <= 1.0

            @info "3-player (random z) — Player $ii: M sparsity=$(round(sr_M, digits=3)), N sparsity=$(round(sr_N, digits=3))"

            # Verify sparse solve still matches dense
            K_dense = M \ N_mat
            K_sparse = sparse(M) \ N_mat
            @test norm(K_sparse - K_dense) / max(norm(K_dense), 1.0) < 1e-10
        end
    end

    @testset "Timing comparison — sparse vs dense" begin
        prob = make_three_player_chain()
        M_evals, N_evals, _ = extract_M_N_matrices(prob)

        for (ii, M) in M_evals
            isnothing(M) && continue
            N_mat = N_evals[ii]
            M_sp = sparse(M)

            # Warmup
            M \ N_mat
            M_sp \ N_mat

            # Benchmark dense
            n_iters = 1000
            t_dense = @elapsed for _ in 1:n_iters
                M \ N_mat
            end

            # Benchmark sparse
            t_sparse = @elapsed for _ in 1:n_iters
                M_sp \ N_mat
            end

            speedup = t_dense / t_sparse
            @info "3-player Player $ii: dense=$(round(t_dense/n_iters*1e6, digits=1))μs, sparse=$(round(t_sparse/n_iters*1e6, digits=1))μs, speedup=$(round(speedup, digits=2))x"

            # Test passes regardless — this is informational
            @test true
        end
    end

    @testset "compute_K_evals with use_sparse flag" begin
        prob = make_three_player_chain()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim,
            verbose=false
        )

        z_current = randn(length(precomputed.all_variables))

        # Dense solve (default)
        K_vec_dense, info_dense = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )

        # Sparse solve
        K_vec_sparse, info_sparse = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=true
        )

        # Results should be numerically identical
        @test norm(K_vec_sparse - K_vec_dense) / max(norm(K_vec_dense), 1.0) < 1e-10

        # Individual K matrices should match
        for ii in 1:prob.N
            K_d = info_dense.K_evals[ii]
            K_s = info_sparse.K_evals[ii]
            if isnothing(K_d)
                @test isnothing(K_s)
            else
                @test norm(K_s - K_d) / max(norm(K_d), 1.0) < 1e-10
            end
        end
    end

    @testset "compute_K_evals use_sparse=true with 4-player chain" begin
        prob = make_four_player_chain()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim,
            verbose=false
        )

        z_current = randn(length(precomputed.all_variables))

        K_vec_dense, _ = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )

        K_vec_sparse, _ = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=true
        )

        @test norm(K_vec_sparse - K_vec_dense) / max(norm(K_vec_dense), 1.0) < 1e-10
    end
end
