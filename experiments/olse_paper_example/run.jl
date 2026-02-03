#=
    OLSE Paper Example

    Verification example from the OLSE (Open-Loop Stackelberg Equilibrium) paper.
    Computes the equilibrium using both our solver and the closed-form OLSE
    solution, then verifies they match.

    This is a 2-player LQ Stackelberg game with:
    - 4-dimensional state (2 positions per player)
    - 2-dimensional control per player
    - Linear dynamics with coupled state evolution
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: I, norm
using BlockArrays: BlockArray, Block
using SymbolicTracingUtils
using Symbolics

"""
    compute_olse_closed_form(x0, T, A, B1, B2, Q1, Q2, R1, R2)

Compute the closed-form OLSE solution using the method from the paper.

# Arguments
- `x0`: Initial state
- `T`: Time horizon
- `A, B1, B2`: Dynamics matrices
- `Q1, Q2, R1, R2`: Cost matrices

# Returns
Named tuple with control trajectories and state trajectory.
"""
function compute_olse_closed_form(x0, T, A, B1, B2, Q1, Q2, R1, R2)
    nx = length(x0)
    m = size(B1, 2)

    # Follower's KKT system: M2 * [u2; λ2; x] + N2 * [x0; u1] = 0
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

    # Gradient w.r.t. u2
    for t in 1:T
        M2[Block(t, t)] = R2
        M2[Block(t, t + T)] = -B2'
    end
    # Gradient w.r.t. x
    for t in 1:T
        M2[Block(t + T, t + 2 * T)] = Q2
        M2[Block(t + T, t + T)] = I(nx)
        if t > 1
            M2[Block(t + T - 1, t + T)] = -A'
        end
    end
    # Dynamics constraints
    for t in 1:T
        M2[Block(t + 2 * T, t + 2 * T)] = I(nx)
        M2[Block(t + 2 * T, t)] = -B2
        if t > 1
            M2[Block(t + 2 * T, t + 2 * T - 1)] = -A
        end
        N2[Block(t + 2 * T, t + 1)] = -B1
    end
    N2[Block(2 * T + 1, 1)] = -A

    K2 = -inv(Array(M2)) * Array(N2)

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

    # Gradient w.r.t. u1
    for t in 1:T
        M[Block(t, t)] = R1
        M[Block(t, t + 3 * T)] = -B1'
        M[Block(t, 1 + 4 * T)] = -K2[Block(t, 2)]'
        M[Block(t, 2 + 4 * T)] = -K2[Block(t, 3)]'
    end
    # Gradient w.r.t. u2
    for t in 1:T
        M[Block(t + T, t + T)] = zeros(m, m)
        M[Block(t + T, t + 3 * T)] = -B2'
        M[Block(t + T, t + 4 * T)] = I(m)
    end
    # Gradient w.r.t. x
    for t in 1:T
        M[Block(t + 2 * T, t + 2 * T)] = Q1
        M[Block(t + 2 * T, t + 3 * T)] = I(nx)
        if t > 1
            M[Block(t + 2 * T - 1, t + 3 * T)] = -A'
        end
    end
    # Dynamics constraints
    for t in 1:T
        M[Block(t + 3 * T, t + 2 * T)] = I(nx)
        M[Block(t + 3 * T, t)] = -B1
        M[Block(t + 3 * T, t + T)] = -B2
        if t > 1
            M[Block(t + 3 * T, t + 2 * T - 1)] = -A
        end
    end
    # Policy constraints
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

    # Unpack into trajectories
    function unpack_u(z)
        us = Vector{Vector{eltype(z)}}(undef, T)
        for t in 1:T
            us[t] = z[(m*(t-1)+1):(m*t)]
        end
        return us
    end

    u1_traj = unpack_u(u1)
    u2_traj = unpack_u(u2)

    # Rollout state
    xs = Vector{Vector{Float64}}(undef, T + 1)
    xs[1] = copy(x0)
    for t in 1:T
        xs[t+1] = A * xs[t] + B1 * u1_traj[t] + B2 * u2_traj[t]
    end

    return (; u1_traj, u2_traj, xs)
end

"""
    run_olse_paper_example(; T=2, x0=[1.0, 2.0, 2.0, 1.0], verbose=false)

Run the OLSE paper verification example.

# Arguments
- `T`: Time horizon
- `x0`: Initial state (4D)
- `verbose`: Print detailed output

# Returns
Named tuple with solutions, comparison errors, and verification status.
"""
function run_olse_paper_example(;
    T::Integer = 2,
    x0::AbstractVector{<:Real} = [1.0, 2.0, 2.0, 1.0],
    verbose::Bool = false,
)
    # Problem data
    nx = 4
    m = 2
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
    θs = setup_problem_parameter_variables(backend, fill(nx, N))

    # Helper functions
    function unpack_u(z)
        us = Vector{Vector{eltype(z)}}(undef, T)
        for t in 1:T
            us[t] = z[(m*(t-1)+1):(m*t)]
        end
        return us
    end

    function rollout_x(u1, u2)
        xs = Vector{Vector{eltype(u1[1])}}(undef, T + 1)
        xs[1] = collect(x0)
        for t in 1:T
            xs[t+1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
        end
        return xs
    end

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
    gs = [z -> Symbolics.Num[] for _ in 1:N]

    # Compute closed-form OLSE
    olse = compute_olse_closed_form(x0, T, A, B1, B2, Q1, Q2, R1, R2)

    # Build and solve with our method
    verbose && @info "Building NonlinearSolver..."
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, nx, m;
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
    u1_err = norm(vcat(u1_traj...) - vcat(olse.u1_traj...))
    u2_err = norm(vcat(u2_traj...) - vcat(olse.u2_traj...))
    x_err = norm(vcat(xs...) - vcat(olse.xs...))

    # Verification
    tol = 1e-6
    verified = u1_err < tol && u2_err < tol && x_err < tol

    if verbose
        @info "Solution found" status = result.status iterations = result.iterations
        @info "OLSE comparison" u1_err u2_err x_err
        @info "Verification" verified tol
        println("\nOLSE trajectories:")
        println("  u1: $(olse.u1_traj)")
        println("  u2: $(olse.u2_traj)")
        println("\nSolver trajectories:")
        println("  u1: $(u1_traj)")
        println("  u2: $(u2_traj)")
    end

    return (;
        # Solver solution
        u1_traj,
        u2_traj,
        xs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        # OLSE solution
        olse,
        # Comparison
        u1_err,
        u2_err,
        x_err,
        verified,
    )
end

"""
    verify_olse_properties(; num_tests=10, verbose=false)

Run multiple verification tests with random initial conditions.

# Arguments
- `num_tests`: Number of random tests to run
- `verbose`: Print detailed output

# Returns
Named tuple with test results.
"""
function verify_olse_properties(; num_tests::Integer = 10, verbose::Bool = false)
    using Random
    rng = MersenneTwister(42)

    results = []
    all_verified = true

    for k in 1:num_tests
        x0 = randn(rng, 4)
        result = run_olse_paper_example(x0 = x0, verbose = false)
        push!(results, result)
        all_verified = all_verified && result.verified

        if verbose
            status_str = result.verified ? "PASS" : "FAIL"
            println("Test $k [$status_str]: u1_err=$(round(result.u1_err, sigdigits=3)), u2_err=$(round(result.u2_err, sigdigits=3)), x_err=$(round(result.x_err, sigdigits=3))")
        end
    end

    return (;
        results,
        all_verified,
        num_passed = sum(r.verified for r in results),
        num_tests,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running OLSE paper example verification...")
    println("=" ^ 60)

    # Single test with verbose output
    result = run_olse_paper_example(verbose = true)
    println("\n" * "=" ^ 60)
    println("Single test: $(result.verified ? "PASSED" : "FAILED")")

    # Multiple random tests
    println("\n" * "=" ^ 60)
    println("Running verification suite...")
    verification = verify_olse_properties(num_tests = 10, verbose = true)
    println("\n" * "=" ^ 60)
    println("Verification suite: $(verification.num_passed)/$(verification.num_tests) passed")
    println("Overall: $(verification.all_verified ? "ALL TESTS PASSED" : "SOME TESTS FAILED")")
end
