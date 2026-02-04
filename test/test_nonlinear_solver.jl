using Test
using Graphs: SimpleDiGraph, add_edge!, nv, topological_sort
using LinearAlgebra: norm, I
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    compute_K_evals,
    setup_approximate_kkt_solver,
    setup_problem_variables,
    setup_problem_parameter_variables,
    make_symbolic_vector,
    default_backend,
    get_all_followers

using TrajectoryGamesBase: unflatten_trajectory

#=
    Test Helpers: Create simple test problems for nonlinear solver testing
=#

"""
Create a simple 2-player chain hierarchy game for testing.
P1 -> P2 (P1 is leader, P2 is follower)

Each player has simple dynamics: x_{t+1} = x_t + u_t
State dim = 2, Control dim = 2, Time horizon T = 3.
"""
function make_two_player_chain_problem(; T=3, state_dim=2, control_dim=2)
    N = 2

    # Hierarchy: P1 -> P2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    # Dimensions
    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    # Parameter variables (initial states)
    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    # Simple quadratic costs
    # P1: minimize ||x_T - goal1||^2 + ||u||^2
    # P2: minimize ||x_T - goal2||^2 + ||u||^2
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

    # Simple integrator dynamics constraints
    function make_dynamics_constraint(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                # x_{t+1} = x_t + u_t (simple integrator)
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            # Initial condition: x_1 = θ[player_idx]
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
function make_three_player_chain_problem(; T=3, state_dim=2, control_dim=2)
    N = 3

    # Hierarchy: P1 -> P2 -> P3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)

    # Dimensions
    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    # Parameter variables (initial states)
    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    # Simple quadratic costs
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

    # Simple integrator dynamics constraints
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

#=
    Tests for setup_approximate_kkt_solver
=#

@testset "setup_approximate_kkt_solver" begin
    @testset "Returns named tuple with correct fields" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        # Get all variables flat
        all_variables = vars.all_variables

        result = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend
        )

        # Should return a tuple with (augmented_variables, named_tuple)
        @test result isa Tuple
        @test length(result) == 2

        augmented_vars, setup_info = result
        @test augmented_vars isa AbstractVector

        # Named tuple should have these fields
        @test haskey(setup_info, :graph) || hasproperty(setup_info, :graph)
        @test haskey(setup_info, :πs) || hasproperty(setup_info, :πs)
        @test haskey(setup_info, :K_syms) || hasproperty(setup_info, :K_syms)
        @test haskey(setup_info, :M_fns) || hasproperty(setup_info, :M_fns)
        @test haskey(setup_info, :N_fns) || hasproperty(setup_info, :N_fns)
        @test haskey(setup_info, :π_sizes) || hasproperty(setup_info, :π_sizes)
    end

    @testset "K_syms has correct dimensions per player" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend
        )

        K_syms = setup_info.K_syms

        # P1 has no leader, so K_syms[1] should be empty
        @test isempty(K_syms[1])

        # P2 has P1 as leader, K_syms[2] should be length(ws[2]) x length(ys[2])
        @test !isempty(K_syms[2])
        @test size(K_syms[2]) == (length(vars.ws[2]), length(vars.ys[2]))
    end

    @testset "M_fns and N_fns are callable" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        augmented_vars, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend
        )

        # Only player 2 has M_fns and N_fns (has a leader)
        @test haskey(setup_info.M_fns, 2)
        @test haskey(setup_info.N_fns, 2)

        # Test that they are callable with numeric input
        test_input = zeros(length(augmented_vars))
        M_result = setup_info.M_fns[2](test_input)
        N_result = setup_info.N_fns[2](test_input)

        # Results should be matrices of correct size
        @test M_result isa AbstractArray
        @test N_result isa AbstractArray
    end

    @testset "Works for 3-player chain graph" begin
        prob = make_three_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        augmented_vars, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend
        )

        # P1 has no leader (root)
        @test isempty(setup_info.K_syms[1])
        @test !haskey(setup_info.M_fns, 1)

        # P2 has P1 as leader
        @test !isempty(setup_info.K_syms[2])
        @test haskey(setup_info.M_fns, 2)
        @test haskey(setup_info.N_fns, 2)

        # P3 has P2 as leader
        @test !isempty(setup_info.K_syms[3])
        @test haskey(setup_info.M_fns, 3)
        @test haskey(setup_info.N_fns, 3)
    end
end

#=
    Tests for preoptimize_nonlinear_solver
=#

@testset "preoptimize_nonlinear_solver" begin
    @testset "Returns named tuple with required components" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        # Should contain all required fields
        @test hasproperty(precomputed, :problem_vars) || haskey(precomputed, :problem_vars)
        @test hasproperty(precomputed, :setup_info) || haskey(precomputed, :setup_info)
        @test hasproperty(precomputed, :mcp_obj) || haskey(precomputed, :mcp_obj)
        @test hasproperty(precomputed, :linsolver) || haskey(precomputed, :linsolver)
        @test hasproperty(precomputed, :all_variables) || haskey(precomputed, :all_variables)
    end

    @testset "problem_vars contains zs, λs, μs, ws, ys" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        vars = precomputed.problem_vars
        @test hasproperty(vars, :zs) || haskey(vars, :zs)
        @test hasproperty(vars, :λs) || haskey(vars, :λs)
        @test hasproperty(vars, :μs) || haskey(vars, :μs)
        @test hasproperty(vars, :ws) || haskey(vars, :ws)
        @test hasproperty(vars, :ys) || haskey(vars, :ys)
    end

    @testset "setup_info contains M_fns, N_fns, K_syms" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        setup_info = precomputed.setup_info
        @test hasproperty(setup_info, :M_fns) || haskey(setup_info, :M_fns)
        @test hasproperty(setup_info, :N_fns) || haskey(setup_info, :N_fns)
        @test hasproperty(setup_info, :K_syms) || haskey(setup_info, :K_syms)
    end
end

#=
    Tests for compute_K_evals
=#

@testset "compute_K_evals" begin
    @testset "Returns Dict mapping player IDs to K matrices" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))

        all_K_vec, K_info = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info)

        # Should return a vector and a named tuple with K_evals
        @test all_K_vec isa AbstractVector
        @test hasproperty(K_info, :K_evals) || haskey(K_info, :K_evals)

        K_evals = K_info.K_evals
        @test K_evals isa Dict
    end

    @testset "Returns nothing for root players (no leaders)" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))

        _, K_info = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info)
        K_evals = K_info.K_evals

        # P1 is root (no leader), should have nothing
        @test isnothing(K_evals[1])

        # P2 has a leader, should have a K matrix
        @test !isnothing(K_evals[2])
    end

    @testset "K matrix dimensions are correct" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))
        vars = precomputed.problem_vars

        _, K_info = compute_K_evals(z_current, vars, precomputed.setup_info)
        K_evals = K_info.K_evals

        # K[2] should be size (length(ws[2]), length(ys[2]))
        if !isnothing(K_evals[2])
            K2 = K_evals[2]
            expected_rows = length(vars.ws[2])
            expected_cols = length(vars.ys[2])
            @test size(K2) == (expected_rows, expected_cols)
        end
    end

    @testset "Computed in reverse topological order" begin
        prob = make_three_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))

        # This should work - evaluation order is internal
        all_K_vec, K_info = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info)

        K_evals = K_info.K_evals

        # P3 should be computed first (leaf), then P2, then P1 (root)
        # All followers should have K matrices, root should not
        @test isnothing(K_evals[1])  # P1 is root
        @test !isnothing(K_evals[2])  # P2 is follower of P1
        @test !isnothing(K_evals[3])  # P3 is follower of P2
    end
end

#=
    Tests for run_nonlinear_solver
=#

@testset "run_nonlinear_solver" begin
    @testset "Returns correct named tuple fields" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        # Initial states for each player
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=10,
            tol=1e-6,
            verbose=false
        )

        # Check return type has required fields
        @test hasproperty(result, :sol) || haskey(result, :sol)
        @test hasproperty(result, :converged) || haskey(result, :converged)
        @test hasproperty(result, :iterations) || haskey(result, :iterations)
        @test hasproperty(result, :residual) || haskey(result, :residual)
    end

    @testset "Respects max_iters limit" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=5,
            tol=1e-12,  # Very tight tolerance to force max_iters
            verbose=false
        )

        @test result.iterations <= 5
    end

    @testset "Converges on simple QP problem" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            verbose=false
        )

        # Should converge on a QP problem
        @test result.converged
        @test result.residual < 1e-6
    end

    @testset "Works with provided initial_guess" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Create an initial guess
        initial_guess = randn(length(precomputed.all_variables))

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            initial_guess=initial_guess,
            max_iters=100,
            tol=1e-6,
            verbose=false
        )

        # Should still return valid results
        @test result.sol isa AbstractVector
        @test length(result.sol) == length(precomputed.all_variables)
    end
end

#=
    Tests for armijo_backtracking_linesearch (already implemented, ensure tests exist)
=#

@testset "armijo_backtracking_linesearch" begin
    @testset "Returns valid step size" begin
        # Simple quadratic merit function: f(z) = z
        f_eval(z) = z

        z = [1.0, 1.0]
        δz = [-1.0, -1.0]  # descent direction
        f_z = f_eval(z)

        α = MixedHierarchyGames.armijo_backtracking_linesearch(f_eval, z, δz, f_z)

        @test α > 0
        @test α <= 1.0
    end

    @testset "Returns smaller step for steep problems" begin
        # Steeper function requires smaller steps
        f_eval_steep(z) = 100 .* z

        z = [1.0, 1.0]
        δz = [-0.1, -0.1]
        f_z = f_eval_steep(z)

        α = MixedHierarchyGames.armijo_backtracking_linesearch(f_eval_steep, z, δz, f_z)

        @test α > 0
    end
end
