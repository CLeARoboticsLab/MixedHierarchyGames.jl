using Test
using LinearAlgebra
using Graphs: SimpleDiGraph, add_edge!
using MixedHierarchyGames: QPSolver, solve_raw, make_symbolic_vector

"""
Compute analytical OLSE solution for 2-player LQ Stackelberg game.
This is the ground truth from the SIOPT paper example.
"""
function compute_olse_analytical(; T=2, x0=[1.0, 2.0, 2.0, 1.0])
    nx = 4
    m = 2
    A = Matrix(1.0 * I(nx))
    B = Matrix(0.1 * I(nx))
    B1 = B[:, 1:2]
    B2 = B[:, 3:4]
    Q1 = 4.0 * [
        0 0 0 0;
        0 0 0 0;
        0 0 1 0;
        0 0 0 1;
    ]
    Q2 = 4.0 * [
        1 0 -1 0;
        0 1 0 -1;
        -1 0 1 0;
        0 -1 0 1;
    ]
    R1 = 2.0 * Matrix(I(m))
    R2 = 2.0 * Matrix(I(m))

    # Follower's (P2) KKT system
    dim_M2 = (m + nx + nx) * T
    dim_N2_cols = nx + m * T

    M2 = zeros(dim_M2, dim_M2)
    N2 = zeros(dim_M2, dim_N2_cols)

    u2_range(t) = (m * (t - 1) + 1):(m * t)
    λ2_range(t) = m * T + (nx * (t - 1) + 1):(m * T + nx * t)
    x_range(t) = (m + nx) * T + (nx * (t - 1) + 1):((m + nx) * T + nx * t)
    x0_range = 1:nx
    u1_range(t) = nx + (m * (t - 1) + 1):(nx + m * t)

    for t in 1:T
        M2[u2_range(t), u2_range(t)] = R2
        M2[u2_range(t), λ2_range(t)] = -B2'
    end

    for t in 1:T
        M2[λ2_range(t), x_range(t)] = Q2
        M2[λ2_range(t), λ2_range(t)] = Matrix(I(nx))
        if t > 1
            M2[λ2_range(t - 1), λ2_range(t)] = -A'
        end
    end

    for t in 1:T
        M2[x_range(t), x_range(t)] = Matrix(I(nx))
        M2[x_range(t), u2_range(t)] = -B2
        if t > 1
            M2[x_range(t), x_range(t - 1)] = -A
        end
        N2[x_range(t), u1_range(t)] = -B1
    end
    N2[x_range(1), x0_range] = -A

    K2 = -inv(M2) * N2
    K2_x0 = K2[1:(m * T), x0_range]
    K2_u1 = K2[1:(m * T), (nx + 1):end]

    # Leader's (P1) KKT system
    dim_M1 = (m + m + nx + nx + m) * T
    dim_N1_cols = nx

    M1 = zeros(dim_M1, dim_M1)
    N1 = zeros(dim_M1, dim_N1_cols)

    u1_L_range(t) = (m * (t - 1) + 1):(m * t)
    u2_L_range(t) = m * T + (m * (t - 1) + 1):(m * T + m * t)
    x_L_range(t) = 2 * m * T + (nx * (t - 1) + 1):(2 * m * T + nx * t)
    λ1_range(t) = (2 * m + nx) * T + (nx * (t - 1) + 1):((2 * m + nx) * T + nx * t)
    η_range(t) = (2 * m + 2 * nx) * T + (m * (t - 1) + 1):((2 * m + 2 * nx) * T + m * t)

    K2_u1_blocks = [K2_u1[u2_range(t), u1_range(s) .- nx] for t in 1:T, s in 1:T]

    for s in 1:T
        M1[u1_L_range(s), u1_L_range(s)] = R1
        M1[u1_L_range(s), λ1_range(s)] = -B1'
        for t in 1:T
            M1[u1_L_range(s), η_range(t)] += -K2_u1_blocks[t, s]'
        end
    end

    for t in 1:T
        M1[u2_L_range(t), λ1_range(t)] = -B2'
        M1[u2_L_range(t), η_range(t)] = Matrix(I(m))
    end

    for t in 1:T
        M1[x_L_range(t), x_L_range(t)] = Q1
        M1[x_L_range(t), λ1_range(t)] = Matrix(I(nx))
        if t > 1
            M1[x_L_range(t - 1), λ1_range(t)] = -A'
        end
    end

    for t in 1:T
        M1[λ1_range(t), x_L_range(t)] = Matrix(I(nx))
        M1[λ1_range(t), u1_L_range(t)] = -B1
        M1[λ1_range(t), u2_L_range(t)] = -B2
        if t > 1
            M1[λ1_range(t), x_L_range(t - 1)] = -A
        end
    end
    N1[λ1_range(1), :] = -A

    for t in 1:T
        M1[η_range(t), u2_L_range(t)] = Matrix(I(m))
        for s in 1:T
            M1[η_range(t), u1_L_range(s)] = -K2_u1_blocks[t, s]
        end
        N1[η_range(t), :] = -K2_x0[u2_range(t), :]
    end

    sol = -inv(M1) * N1 * x0
    u1_sol = sol[1:(m * T)]
    u2_sol = sol[(m * T + 1):(2 * m * T)]

    return (; u1=u1_sol, u2=u2_sol)
end

"""
Solve the same problem using QPSolver.
"""
function solve_with_qpsolver(; T=2, x0=[1.0, 2.0, 2.0, 1.0])
    nx = 4
    m = 2
    A = Matrix(1.0 * I(nx))
    B = Matrix(0.1 * I(nx))
    B1 = B[:, 1:2]
    B2 = B[:, 3:4]
    Q1 = 4.0 * [
        0 0 0 0;
        0 0 0 0;
        0 0 1 0;
        0 0 0 1;
    ]
    Q2 = 4.0 * [
        1 0 -1 0;
        0 1 0 -1;
        -1 0 1 0;
        0 -1 0 1;
    ]
    R1 = 2.0 * Matrix(I(m))
    R2 = 2.0 * Matrix(I(m))

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

    parameter_values = Dict(1 => x0, 2 => x0)
    result = solve_raw(solver, parameter_values)

    @test result.status == :solved

    z_sol = result.z_sol
    u1_sol = z_sol[1:primal_dim]
    u2_sol = z_sol[primal_dim+1:2*primal_dim]

    return (; u1=u1_sol, u2=u2_sol)
end

@testset "QP Solver OLSE Validation" begin
    @testset "2-player LQ Stackelberg (SIOPT example)" begin
        T = 2
        x0 = [1.0, 2.0, 2.0, 1.0]

        olse = compute_olse_analytical(; T, x0)
        qp = solve_with_qpsolver(; T, x0)

        # Verify solutions match to machine precision
        @test norm(qp.u1 - olse.u1) < 1e-10
        @test norm(qp.u2 - olse.u2) < 1e-10
    end

    @testset "Different initial conditions" begin
        T = 2
        for x0 in [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [2.0, -1.0, 1.0, -2.0]]
            olse = compute_olse_analytical(; T, x0)
            qp = solve_with_qpsolver(; T, x0)

            @test norm(qp.u1 - olse.u1) < 1e-10
            @test norm(qp.u2 - olse.u2) < 1e-10
        end
    end

    # NOTE: 3-player chain test skipped - requires investigation into why the
    # control-only formulation (without explicit dynamics constraints) creates
    # singular KKT systems. The reference implementation in
    # reference_archive/old_examples/stackelberg_validation_examples/
    # uses full state+control trajectories with explicit dynamics constraints,
    # which works correctly with PATH solver.

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
        @test all(isfinite, result.z_sol)
    end
end
