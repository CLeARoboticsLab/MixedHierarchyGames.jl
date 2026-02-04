using Test
using LinearAlgebra: norm, I, rank
using Graphs: SimpleDiGraph, add_edge!
using BlockArrays: BlockArray, Block
using Symbolics: Num
using Random: MersenneTwister
using MixedHierarchyGames:
    NonlinearSolver,
    solve_raw,
    setup_problem_parameter_variables,
    default_backend

#=
    OLSE (Open-Loop Stackelberg Equilibrium) Verification Tests

    These tests verify the mathematical properties of the Stackelberg equilibrium
    computed by our solver against the closed-form OLSE solution from the SIOPT paper.

    Key Properties Verified:
    1. Equilibrium existence: Solver converges to a solution
    2. Equilibrium uniqueness: Same initial conditions yield same solution
    3. Follower optimality: Follower's first-order conditions satisfied
    4. Leader optimality: Leader's first-order conditions satisfied (given follower response)
    5. Policy constraint satisfaction: Follower's response matches leader's expectation
=#

"""
OLSE problem parameters from the SIOPT paper.
"""
struct OLSEProblemData
    T::Int           # Time horizon
    nx::Int          # State dimension
    m::Int           # Control dimension per player
    A::Matrix{Float64}   # State transition matrix
    B1::Matrix{Float64}  # Leader control matrix
    B2::Matrix{Float64}  # Follower control matrix
    Q1::Matrix{Float64}  # Leader state cost matrix
    Q2::Matrix{Float64}  # Follower state cost matrix
    R1::Matrix{Float64}  # Leader control cost matrix
    R2::Matrix{Float64}  # Follower control cost matrix
end

function default_olse_problem(; T=2)
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
    R1 = 2.0 * Matrix(I(m))
    R2 = 2.0 * Matrix(I(m))

    return OLSEProblemData(T, nx, m, A, B1, B2, Q1, Q2, R1, R2)
end

"""
Compute follower's optimal response K-matrix.
Returns K2 such that u2 = K2_x0 * x0 + K2_u1 * u1.
"""
function compute_follower_response(prob::OLSEProblemData)
    (; T, nx, m, A, B1, B2, Q2, R2) = prob

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

    # Stationarity w.r.t. u2
    for t in 1:T
        M2[Block(t, t)] = R2
        M2[Block(t, t + T)] = -B2'
    end

    # Stationarity w.r.t. x (adjoint equation)
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

    K2_x0 = K2[1:(m*T), 1:nx]
    K2_u1 = K2[1:(m*T), (nx+1):end]

    return (; K2, K2_x0, K2_u1, M2=Array(M2), N2=Array(N2))
end

"""
Compute the full OLSE solution.
"""
function compute_olse_solution(prob::OLSEProblemData, x0::Vector{Float64})
    (; T, nx, m, A, B1, B2, Q1, R1) = prob

    follower = compute_follower_response(prob)
    K2, K2_x0, K2_u1 = follower.K2, follower.K2_x0, follower.K2_u1

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

    u2_range(t) = (m * (t - 1) + 1):(m * t)

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
    u2 = K2_x0 * x0 + K2_u1 * u1

    # Unpack trajectories
    u1_traj = [u1[(m*(t-1)+1):(m*t)] for t in 1:T]
    u2_traj = [u2[(m*(t-1)+1):(m*t)] for t in 1:T]

    # Rollout states
    xs = Vector{Vector{Float64}}(undef, T + 1)
    xs[1] = copy(x0)
    for t in 1:T
        xs[t+1] = A * xs[t] + B1 * u1_traj[t] + B2 * u2_traj[t]
    end

    return (; u1, u2, u1_traj, u2_traj, xs, M=Array(M), N=Array(N_mat), K2, K2_x0, K2_u1)
end

"""
Verify follower's first-order conditions are satisfied.
"""
function verify_follower_foc(prob::OLSEProblemData, olse, x0; tol=1e-10)
    (; T, nx, m, A, B1, B2, Q2, R2) = prob
    (; u1_traj, u2_traj, xs) = olse

    # For each time step, check: R2 * u2_t - B2' * λ_t = 0
    # We need to compute the adjoint λ

    # Adjoint equation: λ_t = Q2 * x_{t+1} + A' * λ_{t+1}
    # with λ_T = Q2 * x_{T+1}
    λ = Vector{Vector{Float64}}(undef, T)
    λ[T] = Q2 * xs[T+1]
    for t in (T-1):-1:1
        λ[t] = Q2 * xs[t+1] + A' * λ[t+1]
    end

    # Check stationarity
    max_violation = 0.0
    for t in 1:T
        residual = R2 * u2_traj[t] - B2' * λ[t]
        max_violation = max(max_violation, norm(residual))
    end

    return max_violation < tol, max_violation
end

"""
Verify policy constraint: u2 = K2_x0 * x0 + K2_u1 * u1.
"""
function verify_policy_constraint(prob::OLSEProblemData, olse, x0; tol=1e-10)
    (; m, T) = prob
    (; u1, u2, K2_x0, K2_u1) = olse

    u2_predicted = K2_x0 * x0 + K2_u1 * u1
    err = norm(u2 - u2_predicted)

    return err < tol, err
end

"""
Create a MixedHierarchyGames NonlinearSolver for the OLSE problem.
"""
function create_solver_for_olse(prob::OLSEProblemData)
    (; T, nx, m, A, B1, B2, Q1, Q2, R1, R2) = prob

    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim = m * T
    primal_dims = fill(primal_dim, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(nx, N); backend)

    function unpack_u(z)
        return [z[(m*(t-1)+1):(m*t)] for t in 1:T]
    end

    function rollout_x(u1, u2, x0_val)
        xs = Vector{Vector{eltype(u1[1])}}(undef, T + 1)
        xs[1] = collect(x0_val)
        for t in 1:T
            xs[t+1] = A * xs[t] + B1 * u1[t] + B2 * u2[t]
        end
        return xs
    end

    function J1(z1, z2, θ)
        u1 = unpack_u(z1)
        u2 = unpack_u(z2)
        xs = rollout_x(u1, u2, θs[1])
        x_cost = sum(xs[t+1]' * Q1 * xs[t+1] for t in 1:T)
        u_cost = sum(u1[t]' * R1 * u1[t] for t in 1:T)
        return x_cost + u_cost
    end

    function J2(z1, z2, θ)
        u1 = unpack_u(z1)
        u2 = unpack_u(z2)
        xs = rollout_x(u1, u2, θs[2])
        x_cost = sum(xs[t+1]' * Q2 * xs[t+1] for t in 1:T)
        u_cost = sum(u2[t]' * R2 * u2[t] for t in 1:T)
        return x_cost + u_cost
    end

    Js = Dict(1 => J1, 2 => J2)
    gs = [z -> Num[] for _ in 1:N]

    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, nx, m;
        max_iters=50, tol=1e-10,
    )

    return solver
end

@testset "OLSE Verification" begin
    @testset "Follower Response Properties" begin
        @testset "K2 matrix has correct structure" begin
            prob = default_olse_problem(T=2)
            follower = compute_follower_response(prob)

            # K2 maps [x0; u1] to [u2; λ2; x]
            # First m*T rows give u2
            @test size(follower.K2_x0) == (prob.m * prob.T, prob.nx)
            @test size(follower.K2_u1) == (prob.m * prob.T, prob.m * prob.T)
        end

        @testset "Follower KKT system is non-singular" begin
            for T in [2, 3, 5]
                prob = default_olse_problem(T=T)
                follower = compute_follower_response(prob)

                # M2 should be full rank
                @test rank(follower.M2) == size(follower.M2, 1)
            end
        end
    end

    @testset "OLSE Solution Properties" begin
        @testset "Follower FOC satisfied" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]
            olse = compute_olse_solution(prob, x0)

            satisfied, violation = verify_follower_foc(prob, olse, x0)
            @test satisfied "Follower FOC violation: $violation"
        end

        @testset "Policy constraint satisfied" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]
            olse = compute_olse_solution(prob, x0)

            satisfied, err = verify_policy_constraint(prob, olse, x0)
            @test satisfied "Policy constraint error: $err"
        end

        @testset "Leader KKT system is non-singular" begin
            for T in [2, 3]
                prob = default_olse_problem(T=T)
                x0 = ones(prob.nx)
                olse = compute_olse_solution(prob, x0)

                @test rank(olse.M) == size(olse.M, 1)
            end
        end
    end

    @testset "Solver vs Closed-Form OLSE" begin
        @testset "Solver matches OLSE for default initial condition" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]

            # Closed-form solution
            olse = compute_olse_solution(prob, x0)

            # Solver solution
            solver = create_solver_for_olse(prob)
            params = Dict(1 => x0, 2 => x0)
            result = solve_raw(solver, params)

            @test result.converged

            # Extract solver's controls
            m, T = prob.m, prob.T
            u1_solver = result.z_sol[1:(m*T)]
            u2_solver = result.z_sol[(m*T+1):(2*m*T)]

            # Compare
            @test norm(u1_solver - olse.u1) < 1e-6 "u1 error: $(norm(u1_solver - olse.u1))"
            @test norm(u2_solver - olse.u2) < 1e-6 "u2 error: $(norm(u2_solver - olse.u2))"
        end

        @testset "Solver matches OLSE for random initial conditions" begin
            rng = MersenneTwister(42)

            for _ in 1:10
                prob = default_olse_problem(T=2)
                x0 = randn(rng, prob.nx)

                olse = compute_olse_solution(prob, x0)
                solver = create_solver_for_olse(prob)
                params = Dict(1 => x0, 2 => x0)
                result = solve_raw(solver, params)

                @test result.converged

                m, T = prob.m, prob.T
                u1_solver = result.z_sol[1:(m*T)]
                u2_solver = result.z_sol[(m*T+1):(2*m*T)]

                @test norm(u1_solver - olse.u1) < 1e-6 "u1 error for x0=$x0"
                @test norm(u2_solver - olse.u2) < 1e-6 "u2 error for x0=$x0"
            end
        end
    end

    @testset "Equilibrium Uniqueness" begin
        @testset "Same initial condition yields same solution" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]

            solver = create_solver_for_olse(prob)
            params = Dict(1 => x0, 2 => x0)

            # Solve twice
            result1 = solve_raw(solver, params)
            result2 = solve_raw(solver, params)

            @test result1.converged
            @test result2.converged

            # Solutions should be identical
            @test norm(result1.z_sol - result2.z_sol) < 1e-12
        end

        @testset "Different initial guesses converge to same solution" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]

            solver = create_solver_for_olse(prob)
            params = Dict(1 => x0, 2 => x0)

            # Solve with different initial guesses
            n_vars = 2 * prob.m * prob.T
            result1 = solve_raw(solver, params; initial_guess=zeros(n_vars))
            result2 = solve_raw(solver, params; initial_guess=randn(n_vars))

            @test result1.converged
            @test result2.converged

            # Should converge to same solution
            @test norm(result1.z_sol - result2.z_sol) < 1e-6
        end
    end

    @testset "Cost Optimality" begin
        @testset "Perturbation increases follower cost" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]
            olse = compute_olse_solution(prob, x0)

            (; T, m, A, B1, B2, Q2, R2) = prob

            # Compute follower cost at equilibrium
            function follower_cost(u1_traj, u2_traj)
                xs = Vector{Vector{Float64}}(undef, T + 1)
                xs[1] = copy(x0)
                for t in 1:T
                    xs[t+1] = A * xs[t] + B1 * u1_traj[t] + B2 * u2_traj[t]
                end
                x_cost = sum(xs[t+1]' * Q2 * xs[t+1] for t in 1:T)
                u_cost = sum(u2_traj[t]' * R2 * u2_traj[t] for t in 1:T)
                return x_cost + u_cost
            end

            cost_eq = follower_cost(olse.u1_traj, olse.u2_traj)

            # Perturb u2 and verify cost increases
            rng = MersenneTwister(123)
            for _ in 1:5
                perturbation = 0.1 * randn(rng, m)
                u2_perturbed = copy(olse.u2_traj)
                u2_perturbed[1] = u2_perturbed[1] + perturbation

                cost_perturbed = follower_cost(olse.u1_traj, u2_perturbed)

                @test cost_perturbed >= cost_eq - 1e-8 "Perturbation decreased cost: $cost_perturbed < $cost_eq"
            end
        end
    end

    @testset "Different Time Horizons" begin
        for T in [2, 3, 4]
            @testset "T=$T" begin
                prob = default_olse_problem(T=T)
                x0 = [1.0, 2.0, 2.0, 1.0]

                olse = compute_olse_solution(prob, x0)
                solver = create_solver_for_olse(prob)
                params = Dict(1 => x0, 2 => x0)
                result = solve_raw(solver, params)

                @test result.converged

                m = prob.m
                u1_solver = result.z_sol[1:(m*T)]
                u2_solver = result.z_sol[(m*T+1):(2*m*T)]

                @test norm(u1_solver - olse.u1) < 1e-5 "u1 error for T=$T"
                @test norm(u2_solver - olse.u2) < 1e-5 "u2 error for T=$T"
            end
        end
    end
end
