using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    compute_K_evals,
    setup_approximate_kkt_solver,
    setup_problem_variables,
    setup_problem_parameter_variables,
    default_backend,
    NonlinearSolver,
    solve_raw,
    has_leader

using TrajectoryGamesBase: unflatten_trajectory

#=
    Test Helpers: Reuse the same problem factories as test_nonlinear_solver.jl
=#

"""
Create a simple 2-player chain hierarchy game for testing.
P1 -> P2 (P1 is leader, P2 is follower)
"""
function make_two_player_chain_problem_inplace(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

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
Create a simple 3-player chain hierarchy game for testing.
P1 -> P2 -> P3 (P1 leads P2, P2 leads P3)
"""
function make_three_player_chain_problem_inplace(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

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

@testset "In-place M/N Evaluation (Strategy A)" begin

    @testset "compute_K_evals: inplace_MN matches out-of-place (2-player)" begin
        prob = make_two_player_chain_problem_inplace()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        # Test at multiple z values to ensure robustness
        z_current = zeros(length(precomputed.all_variables))
        all_K_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )
        
        # Allocate buffers for in-place evaluation
        ws = precomputed.problem_vars.ws
        ys = precomputed.problem_vars.ys
        π_sizes = precomputed.setup_info.π_sizes
        graph = precomputed.setup_info.graph
        M_buffers = Dict{Int, Matrix{Float64}}()
        N_buffers = Dict{Int, Matrix{Float64}}()
        for ii in 1:prob.N
            if has_leader(graph, ii)
                M_buffers[ii] = zeros(Float64, π_sizes[ii], length(ws[ii]))
                N_buffers[ii] = zeros(Float64, π_sizes[ii], length(ys[ii]))
            end
        end
        
        all_K_inplace, info_inplace = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            inplace_MN=true, M_buffers=M_buffers, N_buffers=N_buffers
        )

        # Results must match to machine epsilon
        @test all_K_default ≈ all_K_inplace atol=1e-14
        @test info_default.status == info_inplace.status

        # Check each player's K matrix individually
        for ii in 1:prob.N
            if info_default.K_evals[ii] !== nothing
                @test info_default.K_evals[ii] ≈ info_inplace.K_evals[ii] atol=1e-14
            end
        end
    end

    @testset "compute_K_evals: inplace_MN matches out-of-place (3-player chain)" begin
        prob = make_three_player_chain_problem_inplace()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = randn(length(precomputed.all_variables))
        all_K_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )
        
        # Allocate buffers for in-place evaluation
        ws = precomputed.problem_vars.ws
        ys = precomputed.problem_vars.ys
        π_sizes = precomputed.setup_info.π_sizes
        graph = precomputed.setup_info.graph
        M_buffers = Dict{Int, Matrix{Float64}}()
        N_buffers = Dict{Int, Matrix{Float64}}()
        for ii in 1:prob.N
            if has_leader(graph, ii)
                M_buffers[ii] = zeros(Float64, π_sizes[ii], length(ws[ii]))
                N_buffers[ii] = zeros(Float64, π_sizes[ii], length(ys[ii]))
            end
        end
        
        all_K_inplace, info_inplace = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            inplace_MN=true, M_buffers=M_buffers, N_buffers=N_buffers
        )

        @test all_K_default ≈ all_K_inplace atol=1e-14
        @test info_default.status == info_inplace.status

        for ii in 1:prob.N
            if info_default.K_evals[ii] !== nothing
                @test info_default.K_evals[ii] ≈ info_inplace.K_evals[ii] atol=1e-14
            end
        end
    end

    @testset "compute_K_evals: inplace_MN repeated calls reuse buffers" begin
        # Verify that calling compute_K_evals with inplace_MN=true multiple times
        # with different z values produces correct results each time (buffer reuse is safe)
        prob = make_two_player_chain_problem_inplace()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        # Allocate buffers once
        ws = precomputed.problem_vars.ws
        ys = precomputed.problem_vars.ys
        π_sizes = precomputed.setup_info.π_sizes
        graph = precomputed.setup_info.graph
        M_buffers = Dict{Int, Matrix{Float64}}()
        N_buffers = Dict{Int, Matrix{Float64}}()
        for ii in 1:prob.N
            if has_leader(graph, ii)
                M_buffers[ii] = zeros(Float64, π_sizes[ii], length(ws[ii]))
                N_buffers[ii] = zeros(Float64, π_sizes[ii], length(ys[ii]))
            end
        end

        for trial in 1:5
            z_current = randn(length(precomputed.all_variables))
            all_K_default, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info
            )
            all_K_inplace, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true, M_buffers=M_buffers, N_buffers=N_buffers
            )
            @test all_K_default ≈ all_K_inplace atol=1e-14
        end
    end

    @testset "setup_info contains in-place function dicts" begin
        prob = make_two_player_chain_problem_inplace()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        setup_info = precomputed.setup_info
        # Verify in-place function dictionaries are present
        @test haskey(setup_info, :M_fns!) || hasproperty(setup_info, Symbol("M_fns!"))
        @test haskey(setup_info, :N_fns!) || hasproperty(setup_info, Symbol("N_fns!"))
        # Buffers are no longer in setup_info (allocated per-solve in run_nonlinear_solver)
    end

    @testset "run_nonlinear_solver: inplace_MN produces identical solution (2-player)" begin
        prob = make_two_player_chain_problem_inplace()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        result_default = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6
        )
        result_inplace = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, inplace_MN=true
        )

        @test result_default.converged
        @test result_inplace.converged
        @test result_default.sol ≈ result_inplace.sol atol=1e-10
        @test result_default.iterations == result_inplace.iterations
    end

    @testset "run_nonlinear_solver: inplace_MN produces identical solution (3-player chain)" begin
        prob = make_three_player_chain_problem_inplace()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0], 3 => [0.0, 0.0])

        result_default = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6
        )
        result_inplace = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, inplace_MN=true
        )

        @test result_default.converged
        @test result_inplace.converged
        @test result_default.sol ≈ result_inplace.sol atol=1e-10
        @test result_default.iterations == result_inplace.iterations
    end

    @testset "NonlinearSolver: inplace_MN option threads through" begin
        prob = make_two_player_chain_problem_inplace()

        # Construct solver with inplace_MN=true
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            inplace_MN=true
        )
        @test solver.options.inplace_MN == true

        # Solve and verify convergence
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])
        result = solve_raw(solver, initial_states)
        @test result.converged

        # Compare with default (inplace_MN=false)
        solver_default = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        result_default = solve_raw(solver_default, initial_states)
        @test result.sol ≈ result_default.sol atol=1e-10
    end

    @testset "solve_raw: inplace_MN override at solve time" begin
        prob = make_two_player_chain_problem_inplace()

        # Construct with default (inplace_MN=false)
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        # Override at solve time
        result = solve_raw(solver, initial_states; inplace_MN=true)
        @test result.converged

        # Compare with default
        result_default = solve_raw(solver, initial_states)
        @test result.sol ≈ result_default.sol atol=1e-10
    end

end
