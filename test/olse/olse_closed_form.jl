#=
    OLSE (Open-Loop Stackelberg Equilibrium) Closed-Form Solution

    Computes the analytical OLSE solution for the 2-player LQ Stackelberg game
    from the SIOPT paper. This serves as ground truth for verifying our solvers.

    The problem has:
    - 4-dimensional state (2 positions per player)
    - 2-dimensional control per player
    - Linear dynamics with coupled state evolution
=#

using LinearAlgebra: I, norm, rank
using BlockArrays: BlockArray, Block

"""
OLSE problem parameters from the SIOPT paper.
"""
struct OLSEProblemData
    T::Int               # Time horizon
    nx::Int              # State dimension
    m::Int               # Control dimension per player
    A::Matrix{Float64}   # State transition matrix
    B1::Matrix{Float64}  # Leader control matrix
    B2::Matrix{Float64}  # Follower control matrix
    Q1::Matrix{Float64}  # Leader state cost matrix
    Q2::Matrix{Float64}  # Follower state cost matrix
    R1::Matrix{Float64}  # Leader control cost matrix
    R2::Matrix{Float64}  # Follower control cost matrix
end

"""
    default_olse_problem(; T=2)

Create the default OLSE problem from the SIOPT paper.
"""
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
    compute_follower_response(prob::OLSEProblemData)

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

    # Block offsets
    u2_offset = 0
    λ2_offset = T
    x_offset = 2 * T
    x0_col = 1

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

    # Dynamics constraints
    for t in 1:T
        M2[Block(t + x_offset, t + x_offset)] = I(nx)
        M2[Block(t + x_offset, t + u2_offset)] = -B2
        if t > 1
            M2[Block(t + x_offset, t + x_offset - 1)] = -A
        end
        N2[Block(t + x_offset, t + 1)] = -B1
    end
    N2[Block(x_offset + 1, x0_col)] = -A

    K2_raw = -Array(M2) \ Array(N2)

    K2_x0 = K2_raw[1:(m*T), 1:nx]
    K2_u1 = K2_raw[1:(m*T), (nx+1):end]
    λ2_x0 = K2_raw[(m*T+1):(m*T+nx*T), 1:nx]
    λ2_u1 = K2_raw[(m*T+1):(m*T+nx*T), (nx+1):end]

    K2 = BlockArray(
        K2_raw,
        vcat(m * ones(Int, T), nx * ones(Int, T), nx * ones(Int, T)),
        vcat([nx], m * ones(Int, T)),
    )

    return (; K2, K2_x0, K2_u1, λ2_x0, λ2_u1, M2=Array(M2), N2=Array(N2))
end

"""
    compute_olse_solution(prob::OLSEProblemData, x0::Vector{Float64})

Compute the full OLSE solution for given initial state.

Returns named tuple with control trajectories, state trajectory, and intermediate matrices.
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

    # Block offsets
    u1_offset = 0
    u2_offset = T
    x_offset = 2 * T
    λ_offset = 3 * T
    η_offset = 4 * T
    k2_x0_block = 1

    # Stationarity w.r.t. u1
    for t in 1:T
        M[Block(t + u1_offset, t + u1_offset)] = R1
        M[Block(t + u1_offset, t + λ_offset)] = -B1'
        for s in 1:T
            M[Block(t + u1_offset, s + η_offset)] = -K2[Block(t, s + 1)]'
        end
    end

    # Stationarity w.r.t. u2
    for t in 1:T
        M[Block(t + u2_offset, t + u2_offset)] = zeros(m, m)
        M[Block(t + u2_offset, t + λ_offset)] = -B2'
        M[Block(t + u2_offset, t + η_offset)] = I(m)
    end

    # Stationarity w.r.t. x
    for t in 1:T
        M[Block(t + x_offset, t + x_offset)] = Q1
        M[Block(t + x_offset, t + λ_offset)] = I(nx)
        if t > 1
            M[Block(t + x_offset - 1, t + λ_offset)] = -A'
        end
    end

    # Dynamics constraints
    for t in 1:T
        M[Block(t + λ_offset, t + x_offset)] = I(nx)
        M[Block(t + λ_offset, t + u1_offset)] = -B1
        M[Block(t + λ_offset, t + u2_offset)] = -B2
        if t > 1
            M[Block(t + λ_offset, t + x_offset - 1)] = -A
        end
    end

    # Policy constraints
    for t in 1:T
        for s in 1:T
            M[Block(t + η_offset, s + u1_offset)] = -K2[Block(t, s + 1)]
        end
        M[Block(t + η_offset, t + u2_offset)] = I(m)
        N_mat[Block(t + η_offset, 1)] = -K2[Block(t, k2_x0_block)]
    end

    # Initial condition
    N_mat[Block(λ_offset + 1, 1)] = -A

    sol = -Array(M) \ (Array(N_mat) * x0)
    u1 = sol[1:(m*T)]
    u2 = K2_x0 * x0 + K2_u1 * u1

    # Extract follower's λ
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
    verify_follower_foc(prob::OLSEProblemData, olse, x0; tol=1e-10)

Verify follower's first-order conditions are satisfied.
"""
function verify_follower_foc(prob::OLSEProblemData, olse, x0; tol=1e-10)
    (; T, m, B2, R2) = prob
    (; u2_traj, λ2_traj) = olse

    max_violation = 0.0
    for t in 1:T
        residual = R2 * u2_traj[t] - B2' * λ2_traj[t]
        max_violation = max(max_violation, norm(residual))
    end

    return max_violation < tol, max_violation
end

"""
    verify_policy_constraint(prob::OLSEProblemData, olse, x0; tol=1e-10)

Verify policy constraint: u2 = K2_x0 * x0 + K2_u1 * u1.
"""
function verify_policy_constraint(prob::OLSEProblemData, olse, x0; tol=1e-10)
    (; u1, u2, K2_x0, K2_u1) = olse

    u2_predicted = K2_x0 * x0 + K2_u1 * u1
    err = norm(u2 - u2_predicted)

    return err < tol, err
end
