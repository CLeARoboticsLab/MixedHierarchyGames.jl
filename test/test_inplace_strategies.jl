#=
    Tests for in-place pre-allocation strategies in compute_K_evals.

    Each strategy must produce identical numerical results to the baseline.
    We test on a 2-player chain and a 3-player chain problem.
=#

using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    compute_K_evals,
    setup_problem_parameter_variables,
    default_backend

using TrajectoryGamesBase: unflatten_trajectory

# Reuse test problem constructors from test_nonlinear_solver.jl
# (they are defined in the same test process via include order)

# ─── Helpers ───────────────────────────────────────────────

"""
Create a 2-player chain problem (P1 → P2) for in-place strategy testing.
"""
function make_inplace_two_player_problem(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

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

"""
Create a 3-player chain problem (P1 → P2 → P3) for in-place strategy testing.
"""
function make_inplace_three_player_problem(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

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

# ─── Test Suite ────────────────────────────────────────────

@testset "In-place pre-allocation strategies" begin

    @testset "Strategy A: In-place M/N evaluation" begin
        @testset "2-player chain produces identical K results" begin
            prob = make_inplace_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            # Use a non-trivial z to test with
            z = randn(length(precomputed.all_variables))

            # Baseline
            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            # Strategy A: in-place M/N
            K_vec_A, info_A = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true
            )

            @test K_vec_A ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_A.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end

        @testset "3-player chain produces identical K results" begin
            prob = make_inplace_three_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_A, info_A = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true
            )

            @test K_vec_A ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_A.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end
    end

    @testset "Strategy B: ldiv! for K solve" begin
        @testset "2-player chain produces identical K results" begin
            prob = make_inplace_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_B, info_B = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_ldiv=true
            )

            @test K_vec_B ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_B.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end

        @testset "3-player chain produces identical K results" begin
            prob = make_inplace_three_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_B, info_B = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_ldiv=true
            )

            @test K_vec_B ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_B.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end
    end

    @testset "Strategy C: lu! + ldiv! for K solve" begin
        @testset "2-player chain produces identical K results" begin
            prob = make_inplace_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_C, info_C = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_lu=true
            )

            @test K_vec_C ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_C.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end

        @testset "3-player chain produces identical K results" begin
            prob = make_inplace_three_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_C, info_C = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_lu=true
            )

            @test K_vec_C ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_C.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end
    end

    @testset "Combined: A + B (in-place M/N + ldiv!)" begin
        @testset "2-player chain produces identical K results" begin
            prob = make_inplace_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_AB, info_AB = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true, inplace_ldiv=true
            )

            @test K_vec_AB ≈ K_vec_baseline atol=1e-12
        end
    end

    @testset "Combined: A + C (in-place M/N + lu! + ldiv!)" begin
        @testset "2-player chain produces identical K results" begin
            prob = make_inplace_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_AC, info_AC = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true, inplace_lu=true
            )

            @test K_vec_AC ≈ K_vec_baseline atol=1e-12
        end
    end

    @testset "Combined: A + B + C (all strategies)" begin
        @testset "2-player chain produces identical K results" begin
            prob = make_inplace_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            # A + B + C: inplace_lu implies ldiv! is also used, so
            # inplace_ldiv is redundant when inplace_lu=true, but we test the combination
            K_vec_ABC, info_ABC = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true, inplace_ldiv=true, inplace_lu=true
            )

            @test K_vec_ABC ≈ K_vec_baseline atol=1e-12
        end

        @testset "3-player chain produces identical K results" begin
            prob = make_inplace_three_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            z = randn(length(precomputed.all_variables))

            K_vec_baseline, info_baseline = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_ABC, info_ABC = compute_K_evals(
                z, precomputed.problem_vars, precomputed.setup_info;
                inplace_MN=true, inplace_ldiv=true, inplace_lu=true
            )

            @test K_vec_ABC ≈ K_vec_baseline atol=1e-12
            for ii in 1:prob.N
                if !isnothing(info_baseline.K_evals[ii])
                    @test info_ABC.K_evals[ii] ≈ info_baseline.K_evals[ii] atol=1e-12
                end
            end
        end
    end

    @testset "Solver-level correctness: NonlinearSolver with in-place strategies" begin
        @testset "solve_raw produces identical results with all strategies enabled" begin
            using MixedHierarchyGames: NonlinearSolver, solve_raw, run_nonlinear_solver

            prob = make_inplace_two_player_problem()

            # Baseline solver
            precomputed_baseline = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim
            )

            initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

            result_baseline = run_nonlinear_solver(
                precomputed_baseline, initial_states, prob.G;
                max_iters=50, tol=1e-8, verbose=false
            )

            # In-place solver (all strategies)
            result_inplace = run_nonlinear_solver(
                precomputed_baseline, initial_states, prob.G;
                max_iters=50, tol=1e-8, verbose=false,
                inplace_MN=true, inplace_ldiv=true, inplace_lu=true
            )

            @test result_baseline.converged
            @test result_inplace.converged
            @test result_inplace.sol ≈ result_baseline.sol atol=1e-10
            @test result_inplace.iterations == result_baseline.iterations
        end
    end
end
