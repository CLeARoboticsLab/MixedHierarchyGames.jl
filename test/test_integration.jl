using Test
using LinearAlgebra: norm, I
using Graphs: SimpleDiGraph, add_edge!
using BlockArrays: BlockArray, Block
using Symbolics: Num
using MixedHierarchyGames:
    QPSolver,
    NonlinearSolver,
    solve,
    solve_raw,
    setup_problem_parameter_variables,
    default_backend,
    make_symbolic_vector,
    split_solution_vector

using TrajectoryGamesBase: unflatten_trajectory

#=
    Integration Tests for MixedHierarchyGames.jl

    Tests that verify:
    1. QP and Nonlinear solvers produce same results on QP problems
    2. Paper example regression tests
    3. Multi-player hierarchies work correctly
=#

"""
Create a 2-player QP Stackelberg game for comparison testing.
P1 is leader, P2 is follower.
"""
function make_comparison_problem(; T=3, state_dim=2, control_dim=2)
    N = 2

    # Hierarchy: P1 -> P2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    # Dimensions
    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Parameters
    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    # Simple quadratic costs
    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2, us2 = xs, us
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        xs1, us1 = xs, us

        # P1 wants to minimize distance to goal and control effort
        goal = [1.0, 1.0]
        0.5 * sum((xs1[end] .- goal) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us1)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        xs2, us2 = xs, us

        # P2 wants to minimize distance to different goal and control effort
        goal = [2.0, 2.0]
        0.5 * sum((xs2[end] .- goal) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us2)
    end

    Js = Dict(1 => J1, 2 => J2)

    # Single integrator dynamics: x_{t+1} = x_t + Δt * u_t
    Δt = 0.5
    function make_constraints(i)
        return function (z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            # Dynamics
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - Δt * us[t])
            end
            # Initial condition
            push!(constraints, xs[1] - θs[i])
            return vcat(constraints...)
        end
    end

    gs = [make_constraints(i) for i in 1:N]

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N, Δt)
end

"""
Create the SIOPT paper example problem for OLSE verification.
"""
function make_siopt_problem(; T=2, x0=[1.0, 2.0, 2.0, 1.0])
    nx = 4
    m = 2  # control dimension per player
    A = Matrix(1.0 * I(nx))
    B = Matrix(0.1 * I(nx))
    B1 = B[:, 1:2]
    B2 = B[:, 3:4]
    Q1 = 4.0 * [
        0 0 0 0
        0 0 0 0
        0 0 1 0
        0 0 0 1
    ]
    Q2 = 4.0 * [
        1  0 -1  0
        0  1  0 -1
       -1  0  1  0
        0 -1  0  1
    ]
    R1 = 2 * I(m)
    R2 = 2 * I(m)

    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)  # P1 leads P2

    primal_dim = m * T
    primal_dims = fill(primal_dim, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(nx, N); backend)

    function unpack_u(z)
        us = Vector{Vector{eltype(z)}}(undef, T)
        for t in 1:T
            us[t] = z[(m*(t-1)+1):(m*t)]
        end
        return us
    end

    function rollout_x(u1, u2, x0_val)
        xs = Vector{Vector{eltype(u1[1])}}(undef, T + 1)
        xs[1] = collect(x0_val)
        for t in 1:T
            xs[t+1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
        end
        return xs
    end

    function J1(z1, z2; θ=nothing)
        u1 = unpack_u(z1)
        u2 = unpack_u(z2)
        xs = rollout_x(u1, u2, θs[1])
        x_cost = sum(xs[t+1]' * Q1 * xs[t+1] for t in 1:T)
        u_cost = sum(u1[t]' * R1 * u1[t] for t in 1:T)
        return x_cost + u_cost
    end

    function J2(z1, z2; θ=nothing)
        u1 = unpack_u(z1)
        u2 = unpack_u(z2)
        xs = rollout_x(u1, u2, θs[2])
        x_cost = sum(xs[t+1]' * Q2 * xs[t+1] for t in 1:T)
        u_cost = sum(u2[t]' * R2 * u2[t] for t in 1:T)
        return x_cost + u_cost
    end

    Js = Dict(1 => J1, 2 => J2)

    # No explicit constraints (dynamics embedded in rollout)
    gs = [z -> Num[] for _ in 1:N]

    return (;
        G, Js, gs, primal_dims, θs,
        state_dim=nx, control_dim=m, T, N,
        A, B1, B2, Q1, Q2, R1, R2,
        unpack_u, rollout_x, x0,
    )
end

"""
Compute closed-form OLSE solution for SIOPT problem.
"""
function compute_olse_solution(prob)
    T = prob.T
    nx = prob.state_dim
    m = prob.control_dim
    A, B1, B2 = prob.A, prob.B1, prob.B2
    Q1, Q2, R1, R2 = prob.Q1, prob.Q2, prob.R1, prob.R2
    x0 = prob.x0

    # Follower's KKT system
    M2 = BlockArray(
        zeros((nx + m) * T + nx * T, (nx + m) * T + nx * T),
        vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
        vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
    )
    N2 = BlockArray(
        zeros((nx + m) * T + nx * T, nx + m * T),
        vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
        vcat([nx], m * ones(Int, T)),
    )

    for t in 1:T
        M2[Block(t, t)] = R2
        M2[Block(t, t + T)] = -B2'
    end
    for t in 1:T
        M2[Block(t + T, t + 2 * T)] = Q2
        M2[Block(t + T, t + T)] = I(nx)
        if t > 1
            M2[Block(t + T - 1, t + T)] = -A'
        end
    end
    for t in 1:T
        M2[Block(t + 2 * T, t + 2 * T)] = I(nx)
        M2[Block(t + 2 * T, t)] = -B2
        if t > 1
            M2[Block(t + 2 * T, t + 2 * T - 1)] = -A
        end
        N2[Block(t + 2 * T, t + 1)] = -B1
    end
    N2[Block(2 * T + 1, 1)] = -A

    # Solve for K2: K2 maps [x0; u1] to [u2; λ2; x]
    # Keep as BlockArray to allow Block indexing later
    K2_data = -inv(Array(M2)) * Array(N2)
    K2 = BlockArray(
        K2_data,
        vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
        vcat([nx], m * ones(Int, T)),
    )

    # Leader's KKT system
    M = BlockArray(
        zeros(m * T + m * T + nx * T + nx * T + m * T, m * T + m * T + nx * T + nx * T + m * T),
        vcat(m * ones(Int, T), m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T), m * ones(Int, T)),
        vcat(m * ones(Int, T), m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T), m * ones(Int, T)),
    )
    N_mat = BlockArray(
        zeros(m * T + m * T + nx * T + nx * T + m * T, nx),
        vcat(m * ones(Int, T), m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T), m * ones(Int, T)),
        [nx],
    )

    for t in 1:T
        M[Block(t, t)] = R1
        M[Block(t, t + 3 * T)] = -B1'
        M[Block(t, 1 + 4 * T)] = -K2[Block(t, 2)]'
        M[Block(t, 2 + 4 * T)] = -K2[Block(t, 3)]'
    end
    for t in 1:T
        M[Block(t + T, t + T)] = zeros(m, m)
        M[Block(t + T, t + 3 * T)] = -B2'
        M[Block(t + T, t + 4 * T)] = I(m)
    end
    for t in 1:T
        M[Block(t + 2 * T, t + 2 * T)] = Q1
        M[Block(t + 2 * T, t + 3 * T)] = I(nx)
        if t > 1
            M[Block(t + 2 * T - 1, t + 3 * T)] = -A'
        end
    end
    for t in 1:T
        M[Block(t + 3 * T, t + 2 * T)] = I(nx)
        M[Block(t + 3 * T, t)] = -B1
        M[Block(t + 3 * T, t + T)] = -B2
        if t > 1
            M[Block(t + 3 * T, t + 2 * T - 1)] = -A
        end
    end
    for t in 1:T
        M[Block(t + 4 * T, 1)] = -K2[Block(t, 2)]
        M[Block(t + 4 * T, 2)] = -K2[Block(t, 3)]
        M[Block(t + 4 * T, t + T)] = I(m)
        N_mat[Block(t + 4 * T, 1)] = -K2[Block(t, 1)]
    end
    N_mat[Block(3 * T + 1, 1)] = -A

    sol = -inv(Array(M)) * Array(N_mat) * x0
    u1 = sol[1:(m*T)]
    u2 = K2[1:(m*T), 1:nx] * x0 + K2[1:(m*T), (nx+1):(nx+m*T)] * u1

    u1_traj = prob.unpack_u(u1)
    u2_traj = prob.unpack_u(u2)
    xs = prob.rollout_x(u1_traj, u2_traj, x0)

    return (; u1_traj, u2_traj, xs)
end

"""
Create a single-player (N=1) optimization problem.
No hierarchy, just one player minimizing their cost.
"""
function make_single_player_problem(; T=3, state_dim=2, control_dim=2)
    N = 1

    # No hierarchy edges - single player is a root
    G = SimpleDiGraph(N)

    # Dimensions
    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = [primal_dim]

    # Parameters
    backend = default_backend()
    θs = setup_problem_parameter_variables([state_dim]; backend)

    # Simple quadratic cost: reach goal with minimal effort
    function J1(z1; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        0.5 * sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1)

    # Single integrator dynamics: x_{t+1} = x_t + Δt * u_t
    Δt = 0.5
    function constraints(z)
        (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
        cons = []
        for t in 1:T
            push!(cons, xs[t+1] - xs[t] - Δt * us[t])
        end
        push!(cons, xs[1] - θs[1])
        return vcat(cons...)
    end

    gs = [constraints]

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N, Δt)
end

@testset "Integration Tests" begin
    @testset "Single-Player Edge Case (N=1)" begin
        @testset "QPSolver handles single player" begin
            prob = make_single_player_problem(T=3)

            solver = QPSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim,
            )

            x0 = [0.0, 0.0]
            params = Dict(1 => x0)

            result = solve_raw(solver, params)

            @test result.status == :solved
            # sol includes primal (16) + dual (8 constraints) variables
            @test length(result.sol) >= prob.primal_dims[1]
            # Primal portion should be finite
            @test all(isfinite, result.sol[1:prob.primal_dims[1]])
        end

        @testset "NonlinearSolver handles single player" begin
            prob = make_single_player_problem(T=3)

            solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=50, tol=1e-8,
            )

            x0 = [0.0, 0.0]
            params = Dict(1 => x0)

            result = solve_raw(solver, params)

            @test result.converged
            # sol includes primal (16) + dual (8 constraints) variables
            @test length(result.sol) >= prob.primal_dims[1]
            # Primal portion should be finite
            @test all(isfinite, result.sol[1:prob.primal_dims[1]])
        end
    end

    @testset "QP vs Nonlinear Solver Comparison" begin
        @testset "Solutions match on QP problem" begin
            prob = make_comparison_problem(T=3)

            # Create both solvers
            qp_solver = QPSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim,
            )

            nonlinear_solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-8,
            )

            # Initial states
            x0_vals = [[0.0, 0.0], [1.0, 1.0]]
            params = Dict(1 => x0_vals[1], 2 => x0_vals[2])

            # Solve with both
            qp_result = solve_raw(qp_solver, params)
            nonlinear_result = solve_raw(nonlinear_solver, params)

            # Both should converge
            @test qp_result.status == :solved
            @test nonlinear_result.converged

            # Solutions should match within tolerance
            @test norm(qp_result.sol - nonlinear_result.sol) < 1e-4
        end

        @testset "Different initial conditions" begin
            prob = make_comparison_problem(T=3)

            qp_solver = QPSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim,
            )

            nonlinear_solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-8,
            )

            # Test multiple initial conditions
            test_cases = [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 2.0], [-1.0, 0.5]],
                [[5.0, -5.0], [2.0, 3.0]],
            ]

            for x0_vals in test_cases
                params = Dict(1 => x0_vals[1], 2 => x0_vals[2])

                qp_result = solve_raw(qp_solver, params)
                nonlinear_result = solve_raw(nonlinear_solver, params)

                @test qp_result.status == :solved
                @test nonlinear_result.converged

                err = norm(qp_result.sol - nonlinear_result.sol)
                @test err < 1e-4
            end
        end
    end

    @testset "SIOPT Paper Example (OLSE Verification)" begin
        @testset "Nonlinear solver matches closed-form OLSE" begin
            x0 = [1.0, 2.0, 2.0, 1.0]
            prob = make_siopt_problem(T=2, x0=x0)

            # Compute closed-form solution
            olse = compute_olse_solution(prob)

            # Solve with nonlinear solver
            solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=50, tol=1e-8,
            )

            params = Dict(1 => x0, 2 => x0)
            result = solve_raw(solver, params)

            @test result.converged

            # Extract per-player controls using split_solution_vector
            m = prob.control_dim
            T = prob.T
            u1_sol, u2_sol = collect(split_solution_vector(
                result.sol[1:(2*m*T)], fill(m * T, 2)
            ))

            u1_traj = prob.unpack_u(u1_sol)
            u2_traj = prob.unpack_u(u2_sol)

            # Compare to OLSE
            u1_err = norm(vcat(u1_traj...) - vcat(olse.u1_traj...))
            u2_err = norm(vcat(u2_traj...) - vcat(olse.u2_traj...))

            @test u1_err < 1e-6
            @test u2_err < 1e-6
        end

        @testset "Random initial conditions" begin
            using Random
            rng = MersenneTwister(42)

            for _ in 1:5
                x0 = randn(rng, 4)
                prob = make_siopt_problem(T=2, x0=x0)

                olse = compute_olse_solution(prob)

                solver = NonlinearSolver(
                    prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                    prob.state_dim, prob.control_dim;
                    max_iters=50, tol=1e-8,
                )

                params = Dict(1 => x0, 2 => x0)
                result = solve_raw(solver, params)

                @test result.converged

                m = prob.control_dim
                T = prob.T
                u1_sol, u2_sol = collect(split_solution_vector(
                    result.sol[1:(2*m*T)], fill(m * T, 2)
                ))
                u1_traj = prob.unpack_u(u1_sol)
                u2_traj = prob.unpack_u(u2_sol)

                u1_err = norm(vcat(u1_traj...) - vcat(olse.u1_traj...))
                u2_err = norm(vcat(u2_traj...) - vcat(olse.u2_traj...))

                @test u1_err < 1e-6
                @test u2_err < 1e-6
            end
        end
    end

    @testset "Three-Player Chain Hierarchy" begin
        @testset "Solver converges on 3-player QP problem" begin
            N = 3
            T = 3
            state_dim = 2
            control_dim = 2

            # P1 -> P2 -> P3
            G = SimpleDiGraph(N)
            add_edge!(G, 1, 2)
            add_edge!(G, 2, 3)

            primal_dim = (state_dim + control_dim) * (T + 1)
            primal_dims = fill(primal_dim, N)

            backend = default_backend()
            θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

            # Simple costs
            function J1(z1, z2, z3; θ=nothing)
                (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
                sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
            end

            function J2(z1, z2, z3; θ=nothing)
                (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
                sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
            end

            function J3(z1, z2, z3; θ=nothing)
                (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
                sum((xs[end] .- [3.0, 3.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
            end

            Js = Dict(1 => J1, 2 => J2, 3 => J3)

            Δt = 0.5
            function make_constraints(i)
                return function (z)
                    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                    constraints = []
                    for t in 1:T
                        push!(constraints, xs[t+1] - xs[t] - Δt * us[t])
                    end
                    push!(constraints, xs[1] - θs[i])
                    return vcat(constraints...)
                end
            end

            gs = [make_constraints(i) for i in 1:N]

            solver = NonlinearSolver(
                G, Js, gs, primal_dims, θs, state_dim, control_dim;
                max_iters=100, tol=1e-6,
            )

            params = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.0], 3 => [2.0, 0.0])
            result = solve_raw(solver, params)

            @test result.converged
            @test result.residual < 1e-6
        end
    end
end
