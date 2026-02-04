#=
    SIOPT Stackelberg Experiment

    Demonstrates a 2-player LQ Stackelberg game matching the formulation
    from the SIOPT paper. Player 1 is the leader, Player 2 is the follower.

    This validates that our solver produces the same results as the
    closed-form OLSE (Open-Loop Stackelberg Equilibrium) solution.
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: I, norm
using BlockArrays: BlockArray, Block
using SymbolicTracingUtils
using Symbolics

"""
    run_siopt_stackelberg(; T=2, x0=[1.0, 2.0, 2.0, 1.0], verbose=false)

Run the SIOPT paper Stackelberg example.

# Arguments
- `T`: Time horizon (number of steps)
- `x0`: Initial state (4D vector)
- `verbose`: Print detailed output

# Returns
Named tuple with solution and comparison to closed-form OLSE.
"""
function run_siopt_stackelberg(;
    T::Integer = 2,
    x0::AbstractVector{<:Real} = [1.0, 2.0, 2.0, 1.0],
    verbose::Bool = false,
)
    # Problem data from OLSE paper
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

    # Each player's decision variable is just their control sequence
    primal_dim = m * T
    primal_dims = fill(primal_dim, N)

    # Set up symbolic parameters
    backend = default_backend()
    θs = setup_problem_parameter_variables(backend, fill(length(x0), N))

    # Helper to unpack control from z
    function unpack_u(z)
        us = Vector{Vector{eltype(z)}}(undef, T)
        for t in 1:T
            us[t] = z[(m*(t-1)+1):(m*t)]
        end
        return us
    end

    # Rollout state trajectory
    function rollout_x(u1, u2)
        xs = Vector{Vector{eltype(u1[1])}}(undef, T + 1)
        xs[1] = collect(x0)
        for t in 1:T
            xs[t+1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
        end
        return xs
    end

    # Player objectives
    function J1(z1, z2, θ)
        u1 = unpack_u(z1)
        u2 = unpack_u(z2)
        xs = rollout_x(u1, u2)
        x_cost = sum(xs[t+1]' * Q1 * xs[t+1] for t in 1:T)
        u_cost = sum(u1[t]' * R1 * u1[t] for t in 1:T)
        return x_cost + u_cost
    end

    function J2(z1, z2, θ)
        u1 = unpack_u(z1)
        u2 = unpack_u(z2)
        xs = rollout_x(u1, u2)
        x_cost = sum(xs[t+1]' * Q2 * xs[t+1] for t in 1:T)
        u_cost = sum(u2[t]' * R2 * u2[t] for t in 1:T)
        return x_cost + u_cost
    end

    Js = Dict{Int,Any}(1 => J1, 2 => J2)

    # No explicit constraints (dynamics embedded in cost via rollout)
    gs = [z -> Symbolics.Num[] for _ in 1:N]

    # Compute closed-form OLSE solution for comparison
    function compute_olse_solution()
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

        u1_traj = unpack_u(u1)
        u2_traj = unpack_u(u2)
        xs = rollout_x(u1_traj, u2_traj)

        return (; u1_traj, u2_traj, xs)
    end

    # Build and solve
    verbose && @info "Building NonlinearSolver..." N T
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, nx, m;  # Using nx as state_dim placeholder
        max_iters = 50, tol = 1e-8, verbose = verbose,
    )

    verbose && @info "Solving..."
    parameter_values = Dict(1 => x0, 2 => x0)
    result = solve_raw(solver, parameter_values; verbose = verbose)

    # Extract solutions
    z_sol = result.z_sol
    u1_sol = z_sol[1:(m*T)]
    u2_sol = z_sol[(m*T+1):(2*m*T)]
    u1_traj = unpack_u(u1_sol)
    u2_traj = unpack_u(u2_sol)
    xs = rollout_x(u1_traj, u2_traj)

    # Compare to OLSE
    olse = compute_olse_solution()
    u1_err = norm(vcat(u1_traj...) - vcat(olse.u1_traj...))
    u2_err = norm(vcat(u2_traj...) - vcat(olse.u2_traj...))
    x_err = norm(vcat(xs...) - vcat(olse.xs...))

    if verbose
        @info "Solution found" status = result.status iterations = result.iterations
        @info "Comparison to OLSE" u1_err u2_err x_err
    end

    return (;
        u1_traj,
        u2_traj,
        xs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        olse,
        u1_err,
        u2_err,
        x_err,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Random
    result = run_siopt_stackelberg(verbose = true)
    println("\nExperiment completed with status: $(result.status)")
    println("OLSE comparison errors: u1=$(result.u1_err), u2=$(result.u2_err), x=$(result.x_err)")

    # Run with random initial conditions
    rng = MersenneTwister(0)
    println("\nRunning with random initial conditions:")
    for k in 1:5
        x0_rand = randn(rng, 4)
        res = run_siopt_stackelberg(x0 = x0_rand, verbose = false)
        println("  Run $k: status=$(res.status), u1_err=$(round(res.u1_err, sigdigits=3)), u2_err=$(round(res.u2_err, sigdigits=3)), x_err=$(round(res.x_err, sigdigits=3))")
    end
end
