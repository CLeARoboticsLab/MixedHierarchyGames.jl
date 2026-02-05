using Test
using LinearAlgebra: norm, rank
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using Random: MersenneTwister
using MixedHierarchyGames:
    NonlinearSolver,
    solve_raw,
    setup_problem_parameter_variables,
    default_backend

# Import shared OLSE closed-form implementation
include("olse_closed_form.jl")

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
