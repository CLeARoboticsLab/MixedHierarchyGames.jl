using Test
using Logging
using Graphs: SimpleDiGraph, add_edge!, nv, topological_sort
using LinearAlgebra: norm, I
using SparseArrays: spzeros, sparse
using LinearSolve: LinearSolve, LinearProblem, init, solve!
using BlockArrays: BlockVector, blocks
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    compute_K_evals,
    compute_newton_step,
    check_convergence,
    perform_linesearch,
    setup_approximate_kkt_solver,
    setup_problem_variables,
    setup_problem_parameter_variables,
    make_symbolic_vector,
    default_backend,
    get_all_followers,
    NonlinearSolver,
    solve,
    solve_raw

using TrajectoryGamesBase: unflatten_trajectory, JointStrategy

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
        @test haskey(setup_info, Symbol("M_fns!")) || hasproperty(setup_info, Symbol("M_fns!"))
        @test haskey(setup_info, Symbol("N_fns!")) || hasproperty(setup_info, Symbol("N_fns!"))
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

    @testset "M_fns! and N_fns! are callable" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        augmented_vars, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend
        )

        # M_fns! and N_fns! are Vector indexed by player ID (all slots exist)
        @test length(setup_info.var"M_fns!") >= 2
        @test length(setup_info.var"N_fns!") >= 2

        # Test that they are callable with numeric input (in-place)
        test_input = zeros(length(augmented_vars))
        M_buffer = zeros(Float64, setup_info.π_sizes[2], length(vars.ws[2]))
        N_buffer = zeros(Float64, setup_info.π_sizes[2], length(vars.ys[2]))
        setup_info.var"M_fns!"[2](M_buffer, test_input)
        setup_info.var"N_fns!"[2](N_buffer, test_input)

        # Results should be matrices of correct size
        @test M_buffer isa AbstractArray
        @test N_buffer isa AbstractArray
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

        # P2 has P1 as leader
        @test !isempty(setup_info.K_syms[2])

        # P3 has P2 as leader
        @test !isempty(setup_info.K_syms[3])

        # M_fns! and N_fns! are Vector indexed by player ID (all slots exist)
        @test length(setup_info.var"M_fns!") == 3
        @test length(setup_info.var"N_fns!") == 3
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

    @testset "setup_info contains M_fns!, N_fns!, K_syms" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        setup_info = precomputed.setup_info
        @test hasproperty(setup_info, Symbol("M_fns!")) || haskey(setup_info, Symbol("M_fns!"))
        @test hasproperty(setup_info, Symbol("N_fns!")) || haskey(setup_info, Symbol("N_fns!"))
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
        @test K_evals isa Vector
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

    @testset "Handles singular M matrix gracefully" begin
        # Create a problem where the follower's cost is constant (zero Hessian),
        # making M (Jacobian of KKT w.r.t. follower's own vars) singular
        state_dim = 2
        control_dim = 2
        T = 3
        N = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
        primal_dims = fill(primal_dim_per_player, N)

        backend = default_backend()
        θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

        # P1 has a normal cost
        function J1(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end

        # P2 has a CONSTANT cost (zero Hessian → singular M)
        function J2_singular(z1, z2; θ=nothing)
            0.0
        end

        Js = Dict(1 => J1, 2 => J2_singular)

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

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim, verbose=false
        )

        z_current = zeros(length(precomputed.all_variables))

        # Should NOT throw - should handle singular M gracefully
        all_K_vec, K_info = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info)

        # Status should indicate singular matrix was encountered
        @test K_info.status == :singular_matrix

        # K values should be NaN (signaling invalid solution)
        @test all(isnan, all_K_vec[all_K_vec .!= 0.0]) || all(isnan, all_K_vec)

        # K_evals for the singular player should contain NaN
        K2 = K_info.K_evals[2]
        @test !isnothing(K2)
        @test all(isnan, K2)
    end

    @testset "Returns :ok status for well-conditioned problems" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))

        all_K_vec, K_info = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info)

        # Well-conditioned problem should have :ok status
        @test K_info.status == :ok

        # K values should be finite
        @test all(isfinite, filter(!iszero, all_K_vec))
    end
end

#=
    Tests for BlockVector structure in nonlinear KKT πs
=#

@testset "Nonlinear KKT BlockVector Structure" begin
    @testset "Leader πs is BlockVector with correct blocks (2-player)" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, vars.all_variables, backend
        )

        πs = setup_info.πs

        # P1 is leader (root) with 1 follower P2
        # Block structure: [grad_self | grad_f1 | policy_f1 | own_constraints]
        @test πs[1] isa BlockVector
        blks = collect(blocks(πs[1]))
        @test length(blks) == 4  # grad_self, grad_f1, policy_f1, constraints
        @test length(blks[1]) == length(vars.zs[1])   # grad w.r.t. own vars
        @test length(blks[2]) == length(vars.zs[2])   # grad w.r.t. follower vars
        @test length(blks[3]) == length(vars.zs[2])   # policy constraint for follower
        constraint_size = length(prob.gs[1](vars.zs[1]))
        @test length(blks[4]) == constraint_size       # own constraints
    end

    @testset "Leaf πs is not BlockVector (2-player)" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, vars.all_variables, backend
        )

        πs = setup_info.πs

        # P2 is a leaf — should not be BlockVector
        @test !(πs[2] isa BlockVector)
    end

    @testset "Leader πs has correct blocks (3-player chain)" begin
        prob = make_three_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, vars.all_variables, backend
        )

        πs = setup_info.πs

        # P1 (root leader, followers: P2 and P3 via BFS)
        # Block structure: [grad_self | grad_f1 | policy_f1 | grad_f2 | policy_f2 | own_constraints]
        @test πs[1] isa BlockVector
        blks1 = collect(blocks(πs[1]))
        @test length(blks1) == 6
        @test length(blks1[1]) == length(vars.zs[1])   # grad w.r.t. own vars
        @test length(blks1[2]) == length(vars.zs[2])   # grad w.r.t. P2 vars
        @test length(blks1[3]) == length(vars.zs[2])   # policy constraint for P2
        @test length(blks1[4]) == length(vars.zs[3])   # grad w.r.t. P3 vars
        @test length(blks1[5]) == length(vars.zs[3])   # policy constraint for P3
        constraint_size_1 = length(prob.gs[1](vars.zs[1]))
        @test length(blks1[6]) == constraint_size_1     # own constraints

        # P2 (mid-chain: leader of P3, follower of P1)
        # Block structure: [grad_self | grad_f1 | policy_f1 | own_constraints]
        @test πs[2] isa BlockVector
        blks2 = collect(blocks(πs[2]))
        @test length(blks2) == 4
        @test length(blks2[1]) == length(vars.zs[2])   # grad w.r.t. own vars
        @test length(blks2[2]) == length(vars.zs[3])   # grad w.r.t. P3 vars
        @test length(blks2[3]) == length(vars.zs[3])   # policy constraint for P3
        constraint_size_2 = length(prob.gs[2](vars.zs[2]))
        @test length(blks2[4]) == constraint_size_2     # own constraints

        # P3 (leaf): not a BlockVector
        @test !(πs[3] isa BlockVector)
    end

    @testset "BlockVector πs total length matches π_sizes" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, vars.all_variables, backend
        )

        πs = setup_info.πs
        π_sizes = setup_info.π_sizes

        for ii in keys(πs)
            @test length(πs[ii]) == π_sizes[ii]
        end
    end

    @testset "strip_policy_constraints works with nonlinear BlockVector πs" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, vars.all_variables, backend
        )

        πs = setup_info.πs

        # Strip policy constraints — should dispatch on BlockVector path for leader
        πs_stripped = strip_policy_constraints(πs, prob.G, vars.zs, prob.gs)

        # P1 (leader with 1 follower): stripped = grad_self + grad_follower + constraints
        expected_stripped_len = length(vars.zs[1]) + length(vars.zs[2]) + length(prob.gs[1](vars.zs[1]))
        @test length(πs_stripped[1]) == expected_stripped_len

        # P2 (leaf): unchanged
        @test length(πs_stripped[2]) == length(πs[2])
    end

    @testset "M and N Jacobians work correctly with BlockVector πs" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)

        _, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, vars.all_variables, backend
        )

        # M_fns! and N_fns! should still be callable (Vector indexed by player ID)
        @test length(setup_info.var"M_fns!") >= 2
        @test length(setup_info.var"N_fns!") >= 2
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
    Tests for armijo_backtracking (moved to src/linesearch.jl)
=#

@testset "armijo_backtracking" begin
    @testset "Returns valid step size" begin
        # Simple quadratic merit function: f(z) = z
        f_eval(z) = z

        z = [1.0, 1.0]
        δz = [-1.0, -1.0]  # descent direction

        α = MixedHierarchyGames.armijo_backtracking(f_eval, z, δz, 1.0)

        @test α > 0
        @test α <= 1.0
    end

    @testset "Returns smaller step for steep problems" begin
        # Steeper function requires smaller steps
        f_eval_steep(z) = 100 .* z

        z = [1.0, 1.0]
        δz = [-0.1, -0.1]

        α = MixedHierarchyGames.armijo_backtracking(f_eval_steep, z, δz, 1.0)

        @test α > 0
    end
end

#=
    Tests for solver failure paths
=#

@testset "Nonlinear Solver Failure Paths" begin
    @testset ":max_iters_reached status" begin
        # Create a problem and run with zero iterations allowed
        # For linear problems, 1 iteration is enough to converge, so we need max_iters=0
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, -0.5])

        # Allow 0 iterations - should hit max_iters immediately
        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=0,
            tol=1e-12,  # Very tight tolerance
            verbose=false
        )

        @test result.status == :max_iters_reached
        @test result.converged == false
        @test result.iterations == 0
    end

    @testset "Solver handles poor initial guess" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, -0.5])

        # Very bad initial guess (large values)
        bad_guess = fill(1000.0, length(precomputed.all_variables))

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            initial_guess=bad_guess,
            max_iters=50,
            tol=1e-6,
            verbose=false
        )

        # Should still return a result (may or may not converge)
        @test result.sol isa Vector{Float64}
        @test result.status in [:solved, :max_iters_reached, :linear_solver_error, :numerical_error, :line_search_failed]
    end

    @testset "Singular K matrix produces :numerical_error status" begin
        # Create a degenerate problem where follower has constant cost (singular M)
        state_dim = 2
        control_dim = 2
        T = 3
        N = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
        primal_dims = fill(primal_dim_per_player, N)

        backend = default_backend()
        θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

        J1(z1, z2; θ=nothing) = begin
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end

        # Constant cost → singular M matrix for follower
        J2_const(z1, z2; θ=nothing) = 0.0

        Js = Dict(1 => J1, 2 => J2_const)

        function make_dyn(player_idx)
            function dyn(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dyn
        end

        gs = [make_dyn(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim, verbose=false
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        # Should not throw - should terminate gracefully
        result = run_nonlinear_solver(
            precomputed, initial_states, G;
            max_iters=10, tol=1e-6, verbose=false
        )

        # NaN from singular K propagates to residual, triggering :numerical_error
        @test result.status == :numerical_error
        @test result.converged == false
    end
end

#=
    Tests for non-convergence scenarios
    Verify that the solver handles failure gracefully across different failure modes.
=#

@testset "Nonlinear Solver Non-Convergence" begin
    @testset "Tight tolerance with few iterations" begin
        # Use standard problem but request machine-precision tolerance with only 1 iteration.
        # The problem normally converges in a few iterations at 1e-6, so 1 iteration at 1e-16
        # should not be enough.
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=1,
            tol=1e-16,  # Extremely tight tolerance
            verbose=false
        )

        # Solver should return gracefully with non-convergence
        @test result.converged == false
        @test result.status == :max_iters_reached
        @test result.iterations == 1
        @test result.residual > 1e-16
        @test result.sol isa Vector{Float64}
        @test length(result.sol) == length(precomputed.all_variables)
    end

    @testset "Conflicting objectives with limited iterations" begin
        # Two players with directly opposing goals and limited iteration budget.
        # P1 wants state at origin, P2 wants state far away — with coupled dynamics
        # this creates tension. With max_iters=2, unlikely to converge.
        T = 3
        state_dim = 2
        control_dim = 2
        N = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
        primal_dims = fill(primal_dim_per_player, N)

        backend = default_backend()
        θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

        # P1 wants state at [0,0], P2 wants state at [100,100]
        # The large goal disparity creates tension in the hierarchy
        function J1_conflict(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            goal = [0.0, 0.0]
            sum((xs[end] .- goal) .^ 2) + 0.01 * sum(sum(u .^ 2) for u in us)
        end

        function J2_conflict(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            goal = [100.0, 100.0]
            sum((xs[end] .- goal) .^ 2) + 0.01 * sum(sum(u .^ 2) for u in us)
        end

        Js = Dict(1 => J1_conflict, 2 => J2_conflict)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dynamics
        end

        gs = [make_dynamics(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            G;
            max_iters=1,
            tol=1e-16,  # Machine-precision tolerance — unreachable in 1 iteration
            verbose=false
        )

        # With only 1 iteration and machine-precision tolerance, solver should not converge
        @test result.converged == false
        @test result.status == :max_iters_reached
        @test result.iterations == 1
        @test result.residual > 1e-16
        # Must still return a valid solution vector
        @test result.sol isa Vector{Float64}
    end

    @testset "Badly scaled problem" begin
        # Costs with wildly different magnitudes create ill-conditioned Jacobians.
        # P1 cost is scaled by 1e8, P2 cost is scaled by 1e-8.
        T = 3
        state_dim = 2
        control_dim = 2
        N = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
        primal_dims = fill(primal_dim_per_player, N)

        backend = default_backend()
        θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

        # Extremely different cost scales
        function J1_scaled(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            goal = [1.0, 1.0]
            1e8 * (sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us))
        end

        function J2_scaled(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            goal = [2.0, 2.0]
            1e-8 * (sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us))
        end

        Js = Dict(1 => J1_scaled, 2 => J2_scaled)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dynamics
        end

        gs = [make_dynamics(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            G;
            max_iters=3,
            tol=1e-12,
            verbose=false
        )

        # Badly scaled problem with few iterations should not converge easily
        # Key assertion: solver returns gracefully regardless of convergence outcome
        @test result.sol isa Vector{Float64}
        @test result.status in [:solved, :max_iters_reached, :linear_solver_error, :numerical_error, :line_search_failed]
        @test result.iterations <= 3
        @test isfinite(result.residual) || result.status == :numerical_error
    end

    @testset "solve_raw() indicates non-convergence" begin
        # Test the high-level NonlinearSolver + solve_raw() API path
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=1,
            tol=1e-16
        )

        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = solve_raw(solver, parameter_values)

        # solve_raw should propagate non-convergence info
        @test result.converged == false
        @test result.status == :max_iters_reached
        @test result.iterations == 1
        @test result.residual > 1e-16
        @test result.sol isa Vector{Float64}
    end

    @testset "solve_raw() overrides solver options for non-convergence" begin
        # Solver has generous defaults, but solve_raw overrides trigger non-convergence
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=100,
            tol=1e-6
        )

        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Override at call site to force non-convergence
        result = solve_raw(solver, parameter_values; max_iters=1, tol=1e-16)

        @test result.converged == false
        @test result.status == :max_iters_reached
    end

    @testset "Verbose mode emits iteration info on non-convergence" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Capture log messages during non-convergent solve with verbose=true
        test_logger = TestLogger(; min_level=Logging.Info)
        result = with_logger(test_logger) do
            run_nonlinear_solver(
                precomputed,
                initial_states,
                prob.G;
                max_iters=2,
                tol=1e-16,
                verbose=true
            )
        end

        # Solver should have emitted @info messages for each iteration
        info_logs = filter(l -> l.level == Logging.Info, test_logger.logs)
        @test length(info_logs) >= 1
        # Check that iteration info contains "residual"
        @test any(occursin("residual", string(l.message)) for l in info_logs)
        # Result should indicate non-convergence
        @test result.converged == false
    end

    @testset "Result struct fields are complete on non-convergence" begin
        # Verify all expected fields are present and have correct types on failure
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=1,
            tol=1e-16,
            verbose=false
        )

        # All fields must be present
        @test haskey(result, :sol)
        @test haskey(result, :converged)
        @test haskey(result, :iterations)
        @test haskey(result, :residual)
        @test haskey(result, :status)

        # Type checks
        @test result.sol isa Vector{Float64}
        @test result.converged isa Bool
        @test result.iterations isa Int
        @test result.residual isa Float64
        @test result.status isa Symbol

        # Non-convergence specific
        @test result.converged == false
        @test result.residual >= 0.0
        @test result.iterations >= 0
    end
end

#=
    Input Validation Tests for NonlinearSolver
=#

@testset "NonlinearSolver Input Validation" begin
    using MixedHierarchyGames: NonlinearSolver, solve, solve_raw

    @testset "Rejects cyclic graph" begin
        # Create a cycle: 1 → 2 → 1
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 1)

        primal_dims = [4, 4]
        θs = setup_problem_parameter_variables([2, 2])
        gs = [z -> [z[1]], z -> [z[1]]]
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        @test_throws ArgumentError NonlinearSolver(G, Js, gs, primal_dims, θs, 2, 1)
    end

    @testset "Rejects self-loop" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 1)  # Self-loop

        primal_dims = [4, 4]
        θs = setup_problem_parameter_variables([2, 2])
        gs = [z -> [z[1]], z -> [z[1]]]
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        @test_throws ArgumentError NonlinearSolver(G, Js, gs, primal_dims, θs, 2, 1)
    end

    @testset "Rejects mismatched primal_dims length" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4, 4]  # 3 elements for 2-player game
        θs = setup_problem_parameter_variables([2, 2])
        gs = [z -> [z[1]], z -> [z[1]]]
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        @test_throws ArgumentError NonlinearSolver(G, Js, gs, primal_dims, θs, 2, 1)
    end

    @testset "Rejects missing player in Js" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4]
        θs = setup_problem_parameter_variables([2, 2])
        gs = [z -> [z[1]], z -> [z[1]]]
        Js = Dict(1 => (z1, z2; θ=nothing) -> sum(z1.^2))  # Missing player 2

        @test_throws ArgumentError NonlinearSolver(G, Js, gs, primal_dims, θs, 2, 1)
    end

    @testset "Rejects missing parameter_values in solve" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        # Use matching state and control dims to avoid dimension mismatch
        state_dim = 2
        control_dim = 2
        T = 2
        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = [primal_dim, primal_dim]

        θs = setup_problem_parameter_variables([state_dim, state_dim])

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        gs = [
            z -> begin
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                vcat([xs[t+1] - xs[t] - 0.1*us[t] for t in 1:T]..., xs[1] - θs[1])
            end,
            z -> begin
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                vcat([xs[t+1] - xs[t] - 0.1*us[t] for t in 1:T]..., xs[1] - θs[2])
            end
        ]

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # Missing player 2 in parameter_values
        parameter_values = Dict(1 => [1.0, 0.0])
        @test_throws ArgumentError solve(solver, parameter_values)
    end
end

#=
    Tests for check_convergence helper
=#

@testset "check_convergence" begin
    @testset "Returns converged=true when residual < tol" begin
        result = check_convergence(1e-8, 1e-6)
        @test result.converged == true
        @test result.status == :solved
    end

    @testset "Returns converged=false when residual >= tol" begin
        result = check_convergence(1e-4, 1e-6)
        @test result.converged == false
        @test result.status == :not_converged
    end

    @testset "Returns converged=true at exact tolerance boundary" begin
        # Residual exactly equal to tol should NOT converge (strict <)
        result = check_convergence(1e-6, 1e-6)
        @test result.converged == false
        @test result.status == :not_converged
    end

    @testset "Returns numerical_error for NaN residual" begin
        result = check_convergence(NaN, 1e-6)
        @test result.converged == false
        @test result.status == :numerical_error
    end

    @testset "Returns numerical_error for Inf residual" begin
        result = check_convergence(Inf, 1e-6)
        @test result.converged == false
        @test result.status == :numerical_error
    end

    @testset "Returns numerical_error for -Inf residual" begin
        result = check_convergence(-Inf, 1e-6)
        @test result.converged == false
        @test result.status == :numerical_error
    end

    @testset "Handles zero residual" begin
        result = check_convergence(0.0, 1e-6)
        @test result.converged == true
        @test result.status == :solved
    end

    @testset "Verbose mode does not error" begin
        # Just verify it doesn't throw when verbose=true
        result = check_convergence(1e-8, 1e-6; verbose=true, iteration=5)
        @test result.converged == true
    end

    @testset "Returns named tuple with correct fields" begin
        result = check_convergence(1e-3, 1e-6)
        @test hasproperty(result, :converged)
        @test hasproperty(result, :status)
        @test result.converged isa Bool
        @test result.status isa Symbol
    end
end

#=
    Tests for compute_newton_step helper
=#

@testset "compute_newton_step" begin
    # Helper to create a LinearSolve solver instance
    function make_linsolver(n)
        algorithm = LinearSolve.UMFPACKFactorization()
        init(LinearProblem(spzeros(n, n), zeros(n)), algorithm)
    end

    @testset "Solves simple 2x2 linear system correctly" begin
        # System: [2 0; 0 3] * δz = [4; 9]  =>  δz = [2; 3]
        linsolver = make_linsolver(2)
        jacobian = sparse([2.0 0.0; 0.0 3.0])
        neg_residual = [4.0, 9.0]

        result = compute_newton_step(linsolver, jacobian, neg_residual)

        @test result.success == true
        @test result.step ≈ [2.0, 3.0] atol=1e-10
    end

    @testset "Returns named tuple with correct fields" begin
        linsolver = make_linsolver(2)
        jacobian = sparse([1.0 0.0; 0.0 1.0])
        neg_residual = [1.0, 1.0]

        result = compute_newton_step(linsolver, jacobian, neg_residual)

        @test hasproperty(result, :step)
        @test hasproperty(result, :success)
        @test result.step isa AbstractVector
        @test result.success isa Bool
    end

    @testset "Solves identity system (δz = -F)" begin
        n = 5
        linsolver = make_linsolver(n)
        jacobian = sparse(Float64.(I(n)))
        neg_residual = randn(n)

        result = compute_newton_step(linsolver, jacobian, neg_residual)

        @test result.success == true
        @test result.step ≈ neg_residual atol=1e-10
    end

    @testset "Solves non-trivial 3x3 system" begin
        # A * x = b where A = [1 2 0; 0 1 1; 1 0 1], b = [5; 3; 4]
        # Solution: x = [7/3; 4/3; 5/3]
        linsolver = make_linsolver(3)
        jacobian = sparse([1.0 2.0 0.0; 0.0 1.0 1.0; 1.0 0.0 1.0])
        neg_residual = [5.0, 3.0, 4.0]

        result = compute_newton_step(linsolver, jacobian, neg_residual)

        @test result.success == true
        @test result.step ≈ [7/3, 4/3, 5/3] atol=1e-10
    end

    @testset "Handles singular matrix gracefully" begin
        linsolver = make_linsolver(2)
        jacobian = sparse([1.0 1.0; 1.0 1.0])  # Singular
        neg_residual = [1.0, 2.0]

        result = compute_newton_step(linsolver, jacobian, neg_residual)

        @test result.success == false
    end
end

#=
    Tests for perform_linesearch helper
=#

@testset "perform_linesearch" begin
    @testset "Returns α=1.0 when use_armijo=false (fixed step)" begin
        # With armijo disabled, should always return α=1.0 regardless of residual
        residual_norm_fn = z -> 1000.0  # Always large residual
        z_est = [1.0, 1.0]
        δz = [-1.0, -1.0]
        current_residual_norm = 10.0

        α = perform_linesearch(residual_norm_fn, z_est, δz, current_residual_norm;
                               use_armijo=false)

        @test α == 1.0
    end

    @testset "Returns α=1.0 when first trial already reduces residual" begin
        # f(z + δz) has smaller norm than f(z), so full step should be accepted
        residual_norm_fn = z -> 0.1  # Trial always has small residual
        z_est = [2.0, 2.0]
        δz = [-1.0, -1.0]
        current_residual_norm = 5.0

        α = perform_linesearch(residual_norm_fn, z_est, δz, current_residual_norm;
                               use_armijo=true)

        @test α == 1.0
    end

    @testset "Backtracks when full step increases residual" begin
        # Full step (α=1): z_trial = [1,1]+1*[1,1] = [2,2] -> norm 10.0 (worse than 5.0)
        # Half step (α=0.5): z_trial = [1,1]+0.5*[1,1] = [1.5,1.5] -> norm 0.1 (better)
        call_count = Ref(0)
        function residual_fn_backtrack(z)
            call_count[] += 1
            # Large z values give large residual, small z values give small residual
            if maximum(abs.(z)) > 1.8
                return 10.0  # Worse than current (5.0)
            else
                return 0.1   # Better than current (5.0)
            end
        end

        z_est = [1.0, 1.0]
        δz = [1.0, 1.0]  # Step that overshoots at full α
        current_residual_norm = 5.0

        α = perform_linesearch(residual_fn_backtrack, z_est, δz, current_residual_norm;
                               use_armijo=true)

        @test α < 1.0
        @test α > 0.0
    end

    @testset "Step size is halved each backtrack iteration" begin
        # Track all α values tried via the z_trial values
        trials = Float64[]
        function residual_fn_track(z)
            push!(trials, z[1])  # Track z_trial[1] = z_est[1] + α * δz[1]
            # Only accept at very small α
            if abs(z[1] - 1.0) < 0.1  # z_est=1, so α*δz must be small
                return 0.01
            end
            return 100.0
        end

        z_est = [1.0]
        δz = [-1.0]
        current_residual_norm = 5.0

        α = perform_linesearch(residual_fn_track, z_est, δz, current_residual_norm;
                               use_armijo=true)

        # Verify backtracking factor of 0.5: trials should show z at α=1, 0.5, 0.25, ...
        # trials[1] = 1.0 + 1.0*(-1.0) = 0.0
        # trials[2] = 1.0 + 0.5*(-1.0) = 0.5
        # trials[3] = 1.0 + 0.25*(-1.0) = 0.75
        @test length(trials) >= 2
        if length(trials) >= 2
            @test trials[1] ≈ 0.0 atol=1e-10   # α=1.0
            @test trials[2] ≈ 0.5 atol=1e-10   # α=0.5
        end
    end

    @testset "Respects max_iters limit" begin
        # Residual never decreases, so should exhaust all iterations
        call_count = Ref(0)
        function residual_fn_never_decrease(z)
            call_count[] += 1
            return 100.0  # Always worse than current
        end

        z_est = [1.0]
        δz = [-1.0]
        current_residual_norm = 5.0

        α = perform_linesearch(residual_fn_never_decrease, z_est, δz, current_residual_norm;
                               use_armijo=true)

        # Should have tried exactly LINESEARCH_MAX_ITERS times (10 by default)
        @test call_count[] == 10
        # α should be 0.5^10 ≈ 9.77e-4 (last tried value)
        @test α ≈ 0.5^10 atol=1e-10
    end

    @testset "Returns named tuple with α field" begin
        # Verify return type matches what the solver expects
        residual_norm_fn = z -> 0.1
        z_est = [1.0]
        δz = [-0.5]
        current_residual_norm = 5.0

        result = perform_linesearch(residual_norm_fn, z_est, δz, current_residual_norm;
                                    use_armijo=true)

        # Should return a scalar Float64 step size
        @test result isa Float64
        @test result > 0.0
        @test result <= 1.0
    end

    @testset "Evaluates residual_norm_fn at correct trial points" begin
        # Verify z_trial = z_est + α * δz is computed correctly
        evaluated_points = Vector{Float64}[]
        function residual_fn_capture(z)
            push!(evaluated_points, copy(z))
            return 0.01  # Accept first trial
        end

        z_est = [3.0, -2.0]
        δz = [1.0, 0.5]
        current_residual_norm = 5.0

        perform_linesearch(residual_fn_capture, z_est, δz, current_residual_norm;
                           use_armijo=true)

        # First (and only) evaluation should be at z_est + 1.0 * δz
        @test length(evaluated_points) == 1
        @test evaluated_points[1] ≈ [4.0, -1.5] atol=1e-10
    end
end

#=
    Tests for linesearch_method option in NonlinearSolver
=#

@testset "NonlinearSolver linesearch_method option" begin
    using MixedHierarchyGames: NonlinearSolver, solve, solve_raw

    @testset "Default linesearch_method is :geometric" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )

        @test solver.options.linesearch_method == :geometric
    end

    @testset "Accepts :armijo linesearch_method" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo
        )

        @test solver.options.linesearch_method == :armijo
    end

    @testset "Accepts :geometric linesearch_method" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric
        )

        @test solver.options.linesearch_method == :geometric
    end

    @testset "Accepts :constant linesearch_method" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:constant
        )

        @test solver.options.linesearch_method == :constant
    end

    @testset "Rejects invalid linesearch_method" begin
        prob = make_two_player_chain_problem()

        @test_throws ArgumentError NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:invalid_method
        )
    end

    @testset "Solver converges with :geometric method" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:geometric
        )

        @test result.converged
        @test result.residual < 1e-6
    end

    @testset "Solver converges with :armijo method" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:armijo
        )

        @test result.converged
        @test result.residual < 1e-6
    end

    @testset "Solver converges with :constant method" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Constant step with α=1.0 (full Newton step) should converge for QP problems
        result = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:constant
        )

        @test result.converged
        @test result.residual < 1e-6
    end

    @testset "All methods produce same solution on QP problem" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result_geo = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, linesearch_method=:geometric
        )

        result_armijo = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, linesearch_method=:armijo
        )

        result_const = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, linesearch_method=:constant
        )

        # All should converge
        @test result_geo.converged
        @test result_armijo.converged
        @test result_const.converged

        # Solutions should match to high precision
        @test result_geo.sol ≈ result_armijo.sol atol=1e-6
        @test result_geo.sol ≈ result_const.sol atol=1e-6
    end

    @testset "linesearch_method propagates through solve interface" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # solve should work without error using the :armijo method
        strategy = solve(solver, initial_states)
        @test strategy isa JointStrategy
    end

    @testset "linesearch_method can be overridden at solve time" begin
        prob = make_two_player_chain_problem()

        # Construct with :geometric
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Override to :armijo at solve time
        result = solve_raw(solver, initial_states; linesearch_method=:armijo)
        @test result.converged
        @test result.residual < 1e-6
    end
end

#=
    Tests for recompute_policy_in_linesearch option
=#

@testset "recompute_policy_in_linesearch option" begin
    @testset "run_nonlinear_solver accepts recompute_policy_in_linesearch kwarg" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Should not throw with recompute_policy_in_linesearch=false (opt-in skip)
        result_skip = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            recompute_policy_in_linesearch=false
        )
        @test result_skip.converged

        # Should not throw with recompute_policy_in_linesearch=true
        result_recompute = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            recompute_policy_in_linesearch=true
        )
        @test result_recompute.converged
    end

    @testset "Default is false (skip K recomputation)" begin
        prob = make_two_player_chain_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        @test solver.options.recompute_policy_in_linesearch == true
    end

    @testset "Constructor stores recompute_policy_in_linesearch option" begin
        prob = make_two_player_chain_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            recompute_policy_in_linesearch=true
        )
        @test solver.options.recompute_policy_in_linesearch == true
    end

    @testset "Solutions match with both settings (2-player chain)" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result_skip = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            recompute_policy_in_linesearch=false
        )

        result_recompute = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            recompute_policy_in_linesearch=true
        )

        @test result_skip.converged
        @test result_recompute.converged
        @test result_skip.sol ≈ result_recompute.sol atol=1e-6
    end

    @testset "Solutions match with both settings (3-player chain)" begin
        prob = make_three_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5], 3 => [1.0, 1.0])

        result_skip = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            recompute_policy_in_linesearch=false
        )

        result_recompute = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            recompute_policy_in_linesearch=true
        )

        @test result_skip.converged
        @test result_recompute.converged
        @test result_skip.sol ≈ result_recompute.sol atol=1e-6
    end

    @testset "solve_raw passes recompute_policy_in_linesearch through" begin
        prob = make_two_player_chain_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            recompute_policy_in_linesearch=false
        )
        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = solve_raw(solver, parameter_values)
        @test result.converged

        # Also test runtime override
        result_override = solve_raw(solver, parameter_values; recompute_policy_in_linesearch=true)
        @test result_override.converged
        @test result.sol ≈ result_override.sol atol=1e-6
    end
end

#=
    Tests for Common Subexpression Elimination (CSE) support
=#

@testset "CSE (Common Subexpression Elimination)" begin
    @testset "setup_approximate_kkt_solver accepts cse keyword" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        # Should accept cse=true without error
        augmented_vars, setup_info = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend;
            cse=true
        )

        @test length(setup_info.var"M_fns!") >= 2
        @test length(setup_info.var"N_fns!") >= 2
    end

    @testset "CSE-compiled M/N functions produce identical results (2-player)" begin
        prob = make_two_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        # Build without CSE
        augmented_vars_no_cse, info_no_cse = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend;
            cse=false
        )

        # Build with CSE
        augmented_vars_cse, info_cse = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend;
            cse=true
        )

        # Evaluate at multiple random points and compare (in-place)
        n_vars = length(augmented_vars_no_cse)
        M_buf_no_cse = zeros(Float64, info_no_cse.π_sizes[2], length(vars.ws[2]))
        M_buf_cse = zeros(Float64, info_cse.π_sizes[2], length(vars.ws[2]))
        N_buf_no_cse = zeros(Float64, info_no_cse.π_sizes[2], length(vars.ys[2]))
        N_buf_cse = zeros(Float64, info_cse.π_sizes[2], length(vars.ys[2]))
        for _ in 1:5
            test_input = randn(n_vars)
            info_no_cse.var"M_fns!"[2](M_buf_no_cse, test_input)
            info_cse.var"M_fns!"[2](M_buf_cse, test_input)
            info_no_cse.var"N_fns!"[2](N_buf_no_cse, test_input)
            info_cse.var"N_fns!"[2](N_buf_cse, test_input)

            @test M_buf_no_cse ≈ M_buf_cse atol=1e-10
            @test N_buf_no_cse ≈ N_buf_cse atol=1e-10
        end
    end

    @testset "CSE-compiled M/N functions produce identical results (3-player)" begin
        prob = make_three_player_chain_problem()
        backend = default_backend()
        vars = setup_problem_variables(prob.G, prob.primal_dims, prob.gs; backend)
        all_variables = vars.all_variables

        # Build without CSE
        augmented_vars_no_cse, info_no_cse = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend;
            cse=false
        )

        # Build with CSE
        augmented_vars_cse, info_cse = setup_approximate_kkt_solver(
            prob.G, prob.Js, vars.zs, vars.λs, vars.μs, prob.gs,
            vars.ws, vars.ys, prob.θs, all_variables, backend;
            cse=true
        )

        # Evaluate at multiple random points for players 2 and 3 (in-place)
        n_vars = length(augmented_vars_no_cse)
        for player in [2, 3]
            M_buf_no_cse = zeros(Float64, info_no_cse.π_sizes[player], length(vars.ws[player]))
            M_buf_cse = zeros(Float64, info_cse.π_sizes[player], length(vars.ws[player]))
            N_buf_no_cse = zeros(Float64, info_no_cse.π_sizes[player], length(vars.ys[player]))
            N_buf_cse = zeros(Float64, info_cse.π_sizes[player], length(vars.ys[player]))
            for _ in 1:5
                test_input = randn(n_vars)
                info_no_cse.var"M_fns!"[player](M_buf_no_cse, test_input)
                info_cse.var"M_fns!"[player](M_buf_cse, test_input)
                info_no_cse.var"N_fns!"[player](N_buf_no_cse, test_input)
                info_cse.var"N_fns!"[player](N_buf_cse, test_input)

                @test M_buf_no_cse ≈ M_buf_cse atol=1e-10
                @test N_buf_no_cse ≈ N_buf_cse atol=1e-10
            end
        end
    end

    @testset "NonlinearSolver accepts cse keyword" begin
        prob = make_two_player_chain_problem()

        # Should construct without error with cse=true
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            cse=true
        )

        @test solver isa NonlinearSolver
    end

    @testset "Solver produces identical results with and without CSE" begin
        prob = make_two_player_chain_problem()
        parameter_values = Dict(1 => [0.1, 0.2], 2 => [0.3, 0.4])

        solver_no_cse = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            cse=false
        )

        solver_cse = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            cse=true
        )

        result_no_cse = solve_raw(solver_no_cse, parameter_values)
        result_cse = solve_raw(solver_cse, parameter_values)

        # Both should converge
        @test result_no_cse.converged
        @test result_cse.converged

        # Solutions should be numerically identical
        @test result_no_cse.sol ≈ result_cse.sol atol=1e-8
    end
end

#=
    Tests for show_progress option
=#

@testset "show_progress option" begin
    @testset "show_progress produces output with expected fields" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Capture stdout to verify progress output
        output = mktemp() do path, io
            redirect_stdout(io) do
                result = run_nonlinear_solver(
                    precomputed,
                    initial_states,
                    prob.G;
                    max_iters=100,
                    tol=1e-6,
                    verbose=false,
                    show_progress=true
                )
                @test result.sol isa AbstractVector
            end
            flush(io)
            read(path, String)
        end

        # Progress output should contain expected fields
        @test contains(output, "iter")
        @test contains(output, "residual")
        @test contains(output, "α")
        @test contains(output, "time")
    end

    @testset "show_progress does not affect solver results" begin
        prob = make_two_player_chain_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Solve without progress
        result_quiet = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            verbose=false,
            show_progress=false
        )

        # Solve with progress (redirect stdout to suppress output)
        result_progress = mktemp() do _, io
            redirect_stdout(io) do
                run_nonlinear_solver(
                    precomputed,
                    initial_states,
                    prob.G;
                    max_iters=100,
                    tol=1e-6,
                    verbose=false,
                    show_progress=true
                )
            end
        end

        # Results must be identical
        @test result_quiet.sol ≈ result_progress.sol atol=1e-14
        @test result_quiet.converged == result_progress.converged
        @test result_quiet.iterations == result_progress.iterations
        @test result_quiet.residual ≈ result_progress.residual atol=1e-14
        @test result_quiet.status == result_progress.status
    end

    @testset "show_progress can be overridden at solve time" begin
        using MixedHierarchyGames: NonlinearSolver, solve_raw

        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            show_progress=false
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # show_progress override at solve time
        output = mktemp() do path, io
            redirect_stdout(io) do
                solve_raw(solver, initial_states; show_progress=true)
            end
            flush(io)
            read(path, String)
        end

        @test contains(output, "iter")
    end
end

#=
    Tests for silent-by-default library behavior
=#

@testset "silent by default" begin
    @testset "solve() produces no stdout with default options" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Capture stdout — default solve should produce zero output
        output = mktemp() do path, io
            redirect_stdout(io) do
                solve(solver, initial_states)
            end
            flush(io)
            read(path, String)
        end

        @test isempty(output)
    end

    @testset "solve_raw() produces no stdout with default options" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        output = mktemp() do path, io
            redirect_stdout(io) do
                solve_raw(solver, initial_states)
            end
            flush(io)
            read(path, String)
        end

        @test isempty(output)
    end

    @testset "verbose=true uses logging macros, not println" begin
        prob = make_two_player_chain_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            verbose=true
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # With verbose=true, stdout should still be empty because verbose
        # messages should use @debug/@info (logging macros go to stderr, not stdout)
        output = mktemp() do path, io
            redirect_stdout(io) do
                solve_raw(solver, initial_states)
            end
            flush(io)
            read(path, String)
        end

        @test isempty(output)
    end

    @testset "no bare println calls in src/ files" begin
        # Verify that no println() calls exist in src/ outside of show_progress blocks.
        # show_progress blocks are the intentional user-facing progress display.
        # All other output should use logging macros (@debug, @info, @warn).
        src_dir = joinpath(dirname(@__DIR__), "src")
        violations = String[]

        for (root, dirs, files) in walkdir(src_dir)
            for f in files
                endswith(f, ".jl") || continue
                filepath = joinpath(root, f)
                in_show_progress_block = 0
                for (lineno, line) in enumerate(eachline(filepath))
                    stripped = lstrip(line)
                    startswith(stripped, "#") && continue
                    # Track if show_progress block nesting
                    if occursin(r"\bif\s+show_progress\b", stripped)
                        in_show_progress_block += 1
                    end
                    if in_show_progress_block > 0 && occursin(r"^\s*end\s*$", line)
                        in_show_progress_block -= 1
                        continue
                    end
                    # Flag println() calls outside show_progress blocks
                    if occursin(r"\bprintln\(", stripped) && in_show_progress_block == 0
                        push!(violations, "$f:$lineno: $stripped")
                    end
                end
            end
        end

        @test isempty(violations)
    end
end

#=
    Tests for pre-allocated parameter buffers
=#

@testset "Pre-allocated parameter buffers" begin
    @testset "F_trial buffer is reused across linesearch iterations (2-player)" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Baseline result
        result_baseline = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:geometric
        )
        @test result_baseline.converged

        # Run again - must produce identical results (buffer reuse must not corrupt)
        result_rerun = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:geometric
        )
        @test result_rerun.sol ≈ result_baseline.sol atol=1e-14
        @test result_rerun.iterations == result_baseline.iterations
        @test result_rerun.residual ≈ result_baseline.residual atol=1e-14
        @test result_rerun.status == result_baseline.status
    end

    @testset "F_trial buffer is reused across linesearch iterations (3-player)" begin
        prob = make_three_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5], 3 => [1.0, 1.0])

        result_baseline = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:geometric
        )
        @test result_baseline.converged

        result_rerun = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:geometric
        )
        @test result_rerun.sol ≈ result_baseline.sol atol=1e-14
        @test result_rerun.iterations == result_baseline.iterations
        @test result_rerun.residual ≈ result_baseline.residual atol=1e-14
    end

    @testset "Armijo linesearch produces identical results with buffer reuse" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result_geometric = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:geometric
        )

        result_armijo = run_nonlinear_solver(
            precomputed,
            initial_states,
            prob.G;
            max_iters=100,
            tol=1e-6,
            linesearch_method=:armijo
        )

        # Both should converge to same solution
        @test result_geometric.converged
        @test result_armijo.converged
        @test result_geometric.sol ≈ result_armijo.sol atol=1e-6
    end

    @testset "all_K_vec buffer reused in compute_K_evals" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_test = zeros(length(precomputed.all_variables))

        # Call compute_K_evals twice - results must be identical
        all_K_vec_1, info_1 = compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info)
        all_K_vec_2, info_2 = compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info)

        @test all_K_vec_1 ≈ all_K_vec_2 atol=1e-14
        @test info_1.status == info_2.status
    end

    @testset "compute_K_evals allocates less with pre-allocated buffers" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_test = zeros(length(precomputed.all_variables))
        mcp_obj = precomputed.mcp_obj
        θ_len = 4  # 2 players × 2 state_dim
        K_len = mcp_obj.parameter_dimension - θ_len

        N_players = 2
        bufs = (;
            M_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            N_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            K_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            follower_cache = Vector{Union{Vector{Int}, Nothing}}(nothing, N_players),
            buffer_cache = Vector{Union{Vector{Float64}, Nothing}}(nothing, N_players),
            all_K_vec = Vector{Float64}(undef, K_len),
        )

        # Warmup both paths
        compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info)
        compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info; buffers=bufs)

        # Measure
        allocs_without = @allocated compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info)
        allocs_with = @allocated compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info; buffers=bufs)

        # Pre-allocated buffers should allocate less than fresh Vectors
        @test allocs_with < allocs_without
    end

    @testset "compute_K_evals with buffers returns identical results" begin
        prob = make_two_player_chain_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_test = randn(length(precomputed.all_variables))
        mcp_obj = precomputed.mcp_obj
        θ_len = 4
        K_len = mcp_obj.parameter_dimension - θ_len

        N_players = 2
        bufs = (;
            M_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            N_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            K_evals = Vector{Union{Matrix{Float64}, Nothing}}(nothing, N_players),
            follower_cache = Vector{Union{Vector{Int}, Nothing}}(nothing, N_players),
            buffer_cache = Vector{Union{Vector{Float64}, Nothing}}(nothing, N_players),
            all_K_vec = Vector{Float64}(undef, K_len),
        )

        all_K_fresh, info_fresh = compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info)
        all_K_buf, info_buf = compute_K_evals(z_test, precomputed.problem_vars, precomputed.setup_info; buffers=bufs)

        @test all_K_fresh ≈ all_K_buf atol=1e-14
        @test info_fresh.status == info_buf.status
    end
end
