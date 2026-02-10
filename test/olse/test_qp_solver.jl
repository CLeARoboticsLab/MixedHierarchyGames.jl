using Test
using LinearAlgebra: I, norm
using Graphs: SimpleDiGraph, add_edge!
using MixedHierarchyGames: QPSolver, solve_raw, make_symbolic_vector, split_solution_vector

# Import shared OLSE closed-form implementation
include("olse_closed_form.jl")

"""
Create a MixedHierarchyGames QPSolver for the OLSE problem.

Uses a controls-only formulation where dynamics are baked into the cost functions.
"""
function create_qpsolver_for_olse(prob::OLSEProblemData)
    (; T, nx, m, A, B1, B2, Q1, Q2, R1, R2) = prob

    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)

    primal_dim = m * T

    # Use SymbolicTracingUtils via MixedHierarchyGames interface
    θ1_vec = make_symbolic_vector(:θ, 1, nx)
    θ2_vec = make_symbolic_vector(:θ, 2, nx)
    θs = Dict(1 => θ1_vec, 2 => θ2_vec)

    function unpack_u(z)
        return [z[(m * (t - 1) + 1):(m * t)] for t in 1:T]
    end

    function rollout_x(u1, u2, x0_val)
        xs = Vector{typeof(x0_val)}(undef, T + 1)
        xs[1] = x0_val
        for t in 1:T
            xs[t + 1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
        end
        return xs
    end

    Js = Dict(
        1 => (z1, z2; θ=nothing) -> begin
            u1 = unpack_u(z1)
            u2 = unpack_u(z2)
            xs = rollout_x(u1, u2, θ1_vec)
            x_cost = sum(xs[t + 1]' * Q1 * xs[t + 1] for t in 1:T)
            u_cost = sum(u1[t]' * R1 * u1[t] for t in 1:T)
            return x_cost + u_cost
        end,
        2 => (z1, z2; θ=nothing) -> begin
            u1 = unpack_u(z1)
            u2 = unpack_u(z2)
            xs = rollout_x(u1, u2, θ2_vec)
            x_cost = sum(xs[t + 1]' * Q2 * xs[t + 1] for t in 1:T)
            u_cost = sum(u2[t]' * R2 * u2[t] for t in 1:T)
            return x_cost + u_cost
        end,
    )

    gs = [z -> eltype(z)[], z -> eltype(z)[]]

    solver = QPSolver(G, Js, gs, [primal_dim, primal_dim], θs, 1, 1)

    return solver
end

@testset "QP Solver OLSE Validation" begin
    @testset "2-player QP Stackelberg (SIOPT example)" begin
        prob = default_olse_problem(T=2)
        x0 = [1.0, 2.0, 2.0, 1.0]

        # Closed-form solution
        olse = compute_olse_solution(prob, x0)

        # QPSolver solution
        solver = create_qpsolver_for_olse(prob)
        params = Dict(1 => x0, 2 => x0)
        result = solve_raw(solver, params)

        @test result.status == :solved

        m, T = prob.m, prob.T
        u1_solver, u2_solver = collect(split_solution_vector(
            result.sol[1:(2*m*T)], fill(m * T, 2)
        ))

        # Verify solutions match to machine precision
        @test norm(u1_solver - olse.u1) < 1e-10
        @test norm(u2_solver - olse.u2) < 1e-10
    end

    @testset "Different initial conditions" begin
        prob = default_olse_problem(T=2)
        m, T = prob.m, prob.T

        for x0 in [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [2.0, -1.0, 1.0, -2.0]]
            olse = compute_olse_solution(prob, x0)

            solver = create_qpsolver_for_olse(prob)
            params = Dict(1 => x0, 2 => x0)
            result = solve_raw(solver, params)

            @test result.status == :solved

            u1_solver, u2_solver = collect(split_solution_vector(
                result.sol[1:(2*m*T)], fill(m * T, 2)
            ))

            @test norm(u1_solver - olse.u1) < 1e-10
            @test norm(u2_solver - olse.u2) < 1e-10
        end
    end

    # NOTE: 3-player chain test skipped - requires investigation into why the
    # control-only formulation (without explicit dynamics constraints) creates
    # singular KKT systems.

    @testset "Nash game (no hierarchy edges)" begin
        # 2-player Nash game: no edges, players optimize simultaneously
        G = SimpleDiGraph(2)  # No edges

        T = 2
        m = 2
        nx = 4
        primal_dim = m * T

        # Use SymbolicTracingUtils via MixedHierarchyGames interface
        θ1_vec = make_symbolic_vector(:θ, 1, nx)
        θ2_vec = make_symbolic_vector(:θ, 2, nx)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)

        A = Matrix(1.0 * I(nx))
        B1, B2 = 0.1 * I(nx)[:, 1:2], 0.1 * I(nx)[:, 3:4]
        Q, R = 2.0 * Matrix(I(nx)), Matrix(I(m))

        function unpack_u(z)
            return [z[(m * (t - 1) + 1):(m * t)] for t in 1:T]
        end

        function rollout_x(u1, u2, x0_val)
            xs = Vector{typeof(x0_val)}(undef, T + 1)
            xs[1] = x0_val
            for t in 1:T
                xs[t + 1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
            end
            return xs
        end

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> begin
                u1, u2 = unpack_u(z1), unpack_u(z2)
                xs = rollout_x(u1, u2, θ1_vec)
                sum(xs[t+1]' * Q * xs[t+1] for t in 1:T) + sum(u1[t]' * R * u1[t] for t in 1:T)
            end,
            2 => (z1, z2; θ=nothing) -> begin
                u1, u2 = unpack_u(z1), unpack_u(z2)
                xs = rollout_x(u1, u2, θ2_vec)
                sum(xs[t+1]' * Q * xs[t+1] for t in 1:T) + sum(u2[t]' * R * u2[t] for t in 1:T)
            end,
        )

        gs = [z -> eltype(z)[] for _ in 1:2]

        solver = QPSolver(G, Js, gs, fill(primal_dim, 2), θs, 1, 1)

        x0 = [1.0, 2.0, -1.0, -2.0]
        parameter_values = Dict(1 => x0, 2 => x0)
        result = solve_raw(solver, parameter_values)

        @test result.status == :solved
        @test all(isfinite, result.sol)
    end
end
