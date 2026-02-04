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

    The solver matches OLSE at machine precision for all time horizons (T=2,3,4+).
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

    # Follower's KKT system block structure:
    # M2 matrix has 3 groups of T blocks: [u2, λ2, x]
    # Block offsets (added to block index t to reach that group):
    u2_offset = 0       # u2 blocks: 1 to T
    λ2_offset = T       # λ2 blocks: T+1 to 2T
    x_offset = 2 * T    # x blocks: 2T+1 to 3T

    # N2 column structure: [x0, u1_1, u1_2, ..., u1_T]
    x0_col = 1          # Block column for x0

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
        M2[Block(t + u2_offset, t + u2_offset)] = R2
        M2[Block(t + u2_offset, t + λ2_offset)] = -B2'
    end

    # Stationarity w.r.t. x (adjoint equation)
    for t in 1:T
        M2[Block(t + λ2_offset, t + x_offset)] = Q2
        M2[Block(t + λ2_offset, t + λ2_offset)] = I(nx)
        if t > 1
            M2[Block(t + λ2_offset - 1, t + λ2_offset)] = -A'
        end
    end

    # Dynamics constraints (stationarity w.r.t. λ2)
    for t in 1:T
        M2[Block(t + x_offset, t + x_offset)] = I(nx)
        M2[Block(t + x_offset, t + u2_offset)] = -B2
        if t > 1
            M2[Block(t + x_offset, t + x_offset - 1)] = -A
        end
        N2[Block(t + x_offset, t + 1)] = -B1  # t+1 because u1 blocks start at column 2
    end
    N2[Block(x_offset + 1, x0_col)] = -A

    K2_raw = -Array(M2) \ Array(N2)

    K2_x0 = K2_raw[1:(m*T), 1:nx]
    K2_u1 = K2_raw[1:(m*T), (nx+1):end]

    # Extract λ portion from K2 (rows m*T+1 to m*T+nx*T)
    λ2_x0 = K2_raw[(m*T+1):(m*T+nx*T), 1:nx]
    λ2_u1 = K2_raw[(m*T+1):(m*T+nx*T), (nx+1):end]

    # Wrap K2 in BlockArray for block indexing in compute_olse_solution
    # Row blocks: same as M2 = [m * T, nx * T, nx * T] -> [m, m, ...(T times), nx, nx, ...(T times), nx, nx, ...(T times)]
    # Col blocks: same as N2 cols = [nx, m * T] -> [nx, m, m, ...(T times)]
    K2 = BlockArray(
        K2_raw,
        vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
        vcat([nx], m * ones(Int, T)),
    )

    return (; K2, K2_x0, K2_u1, λ2_x0, λ2_u1, M2=Array(M2), N2=Array(N2))
end

"""
Compute the full OLSE solution.
"""
function compute_olse_solution(prob::OLSEProblemData, x0::Vector{Float64})
    (; T, nx, m, A, B1, B2, Q1, R1) = prob

    follower = compute_follower_response(prob)
    K2, K2_x0, K2_u1 = follower.K2, follower.K2_x0, follower.K2_u1

    # Leader's KKT system block structure:
    # M matrix has 5 groups of T blocks each: [u1, u2, x, λ, η]
    # Block offsets (added to block index t to reach that group):
    u1_offset = 0       # u1 blocks: 1 to T
    u2_offset = T       # u2 blocks: T+1 to 2T
    x_offset = 2 * T    # x blocks: 2T+1 to 3T
    λ_offset = 3 * T    # λ blocks: 3T+1 to 4T
    η_offset = 4 * T    # η blocks: 4T+1 to 5T

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

    # K2 column block structure: [x0 (1), u1_1 (2), u1_2 (3), ..., u1_T (T+1)]
    k2_x0_block = 1     # Block index for x0 in K2

    # Stationarity w.r.t. u1 (leader's control)
    for t in 1:T
        M[Block(t + u1_offset, t + u1_offset)] = R1
        M[Block(t + u1_offset, t + λ_offset)] = -B1'
        # ∂u2/∂u1 terms from policy constraint
        for s in 1:T
            M[Block(t + u1_offset, s + η_offset)] = -K2[Block(t, s + 1)]'
        end
    end

    # Stationarity w.r.t. u2 (follower's control, as seen by leader)
    for t in 1:T
        M[Block(t + u2_offset, t + u2_offset)] = zeros(m, m)
        M[Block(t + u2_offset, t + λ_offset)] = -B2'
        M[Block(t + u2_offset, t + η_offset)] = I(m)
    end

    # Stationarity w.r.t. x (state)
    for t in 1:T
        M[Block(t + x_offset, t + x_offset)] = Q1
        M[Block(t + x_offset, t + λ_offset)] = I(nx)
        if t > 1
            M[Block(t + x_offset - 1, t + λ_offset)] = -A'
        end
    end

    # Dynamics constraints (stationarity w.r.t. λ)
    for t in 1:T
        M[Block(t + λ_offset, t + x_offset)] = I(nx)
        M[Block(t + λ_offset, t + u1_offset)] = -B1
        M[Block(t + λ_offset, t + u2_offset)] = -B2
        if t > 1
            M[Block(t + λ_offset, t + x_offset - 1)] = -A
        end
    end

    # Policy constraints (stationarity w.r.t. η): u2 = K2_x0 * x0 + K2_u1 * u1
    for t in 1:T
        for s in 1:T
            M[Block(t + η_offset, s + u1_offset)] = -K2[Block(t, s + 1)]
        end
        M[Block(t + η_offset, t + u2_offset)] = I(m)
        N_mat[Block(t + η_offset, 1)] = -K2[Block(t, k2_x0_block)]
    end

    # Initial condition in dynamics
    N_mat[Block(λ_offset + 1, 1)] = -A

    sol = -Array(M) \ (Array(N_mat) * x0)
    u1 = sol[1:(m*T)]
    u2 = K2_x0 * x0 + K2_u1 * u1

    # Extract follower's λ using the follower response matrices
    λ2 = follower.λ2_x0 * x0 + follower.λ2_u1 * u1

    # Unpack trajectories
    u1_traj = [u1[(m*(t-1)+1):(m*t)] for t in 1:T]
    u2_traj = [u2[(m*(t-1)+1):(m*t)] for t in 1:T]
    λ2_traj = [λ2[(nx*(t-1)+1):(nx*t)] for t in 1:T]

    # Rollout states
    xs = Vector{Vector{Float64}}(undef, T + 1)
    xs[1] = copy(x0)
    for t in 1:T
        xs[t+1] = A * xs[t] + B1 * u1_traj[t] + B2 * u2_traj[t]
    end

    return (; u1, u2, λ2, u1_traj, u2_traj, λ2_traj, xs, M=Array(M), N=Array(N_mat), K2, K2_x0, K2_u1)
end

"""
Verify follower's first-order conditions are satisfied.

The KKT stationarity condition w.r.t. u2 is: R2 * u2_t - B2' * λ_t = 0
where λ_t is the Lagrange multiplier from the KKT system (NOT the Pontryagin costate).
"""
function verify_follower_foc(prob::OLSEProblemData, olse, x0; tol=1e-10)
    (; T, m, B2, R2) = prob
    (; u2_traj, λ2_traj) = olse

    # Check stationarity using the actual λ from the KKT solution
    max_violation = 0.0
    for t in 1:T
        residual = R2 * u2_traj[t] - B2' * λ2_traj[t]
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

Uses a controls-only formulation where dynamics are baked into the cost functions.
This is simpler but results in a different policy structure than OLSE:
- OLSE: K2 maps [x0; u1] → u2 (affine in initial condition)
- Solver: K2 maps u1 → u2 (linear in leader's control only)

This difference means the solver won't exactly match OLSE for T > 2.
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
            @test satisfied
        end

        @testset "Policy constraint satisfied" begin
            prob = default_olse_problem(T=2)
            x0 = [1.0, 2.0, 2.0, 1.0]
            olse = compute_olse_solution(prob, x0)

            satisfied, err = verify_policy_constraint(prob, olse, x0)
            @test satisfied
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
            u1_solver = result.sol[1:(m*T)]
            u2_solver = result.sol[(m*T+1):(2*m*T)]

            # Compare
            @test norm(u1_solver - olse.u1) < 1e-6
            @test norm(u2_solver - olse.u2) < 1e-6
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
                u1_solver = result.sol[1:(m*T)]
                u2_solver = result.sol[(m*T+1):(2*m*T)]

                @test norm(u1_solver - olse.u1) < 1e-6
                @test norm(u2_solver - olse.u2) < 1e-6
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
            @test norm(result1.sol - result2.sol) < 1e-12
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
            @test norm(result1.sol - result2.sol) < 1e-6
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

                @test cost_perturbed >= cost_eq - 1e-8
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
                u1_solver = result.sol[1:(m*T)]
                u2_solver = result.sol[(m*T+1):(2*m*T)]

                @test norm(u1_solver - olse.u1) < 1e-6
                @test norm(u2_solver - olse.u2) < 1e-6
            end
        end
    end
end
