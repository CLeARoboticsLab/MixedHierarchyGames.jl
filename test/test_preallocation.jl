using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm, I
using MixedHierarchyGames:
    NonlinearSolver,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    compute_K_evals,
    setup_problem_parameter_variables,
    default_backend

using TrajectoryGamesBase: unflatten_trajectory

@testset "Pre-allocation flag" begin
    @testset "Pre-allocated solve produces identical results (2-player chain)" begin
        # Setup a 2-player chain problem
        N = 2
        T = 3
        state_dim = 2
        control_dim = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = fill(primal_dim, N)

        θs = setup_problem_parameter_variables(fill(state_dim, N))

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

        function make_dynamics(idx)
            return function(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t
                    xs[t+1] - xs[t] - 0.5 * us[t]
                end
                ic = xs[1] - θs[idx]
                vcat(dyn, ic)
            end
        end
        gs = [make_dynamics(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim
        )

        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Default solve
        result_default = run_nonlinear_solver(
            precomputed, params, G;
            max_iters=50, tol=1e-8
        )

        # Pre-allocated solve
        result_prealloc = run_nonlinear_solver(
            precomputed, params, G;
            max_iters=50, tol=1e-8,
            preallocate=true
        )

        @test result_default.converged
        @test result_prealloc.converged
        @test result_default.iterations == result_prealloc.iterations
        @test result_default.status == result_prealloc.status
        @test norm(result_default.sol - result_prealloc.sol) < 1e-12
        @test abs(result_default.residual - result_prealloc.residual) < 1e-12
    end

    @testset "Pre-allocated solve produces identical results (3-player chain)" begin
        N = 3
        T = 5
        state_dim = 2
        control_dim = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 2, 1)
        add_edge!(G, 2, 3)

        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = fill(primal_dim, N)

        θs = setup_problem_parameter_variables(fill(state_dim, N))

        function J1_3p(z1, z2, z3; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            xs1 = xs
            (; xs,) = unflatten_trajectory(z2, state_dim, control_dim)
            0.5 * sum((xs1[end] .- xs[end]) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us)
        end
        function J2_3p(z1, z2, z3; θ=nothing)
            (; xs,) = unflatten_trajectory(z3, state_dim, control_dim)
            xs3 = xs
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            xs2 = xs
            (; xs,) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((0.5 * (xs[end] .+ xs3[end])) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us)
        end
        function J3_3p(z1, z2, z3; θ=nothing)
            (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
            xs3 = xs
            (; xs,) = unflatten_trajectory(z2, state_dim, control_dim)
            0.5 * sum((xs3[end] .- xs[end]) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us)
        end
        Js = Dict(1 => J1_3p, 2 => J2_3p, 3 => J3_3p)

        function make_dynamics_3p(idx)
            return function(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t
                    xs[t+1] - xs[t] - 0.5 * us[t]
                end
                ic = xs[1] - θs[idx]
                vcat(dyn, ic)
            end
        end
        gs = [make_dynamics_3p(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim
        )

        params = Dict(1 => [0.0, 2.0], 2 => [2.0, 4.0], 3 => [6.0, 8.0])

        result_default = run_nonlinear_solver(
            precomputed, params, G;
            max_iters=50, tol=1e-8
        )

        result_prealloc = run_nonlinear_solver(
            precomputed, params, G;
            max_iters=50, tol=1e-8,
            preallocate=true
        )

        @test result_default.converged
        @test result_prealloc.converged
        @test result_default.iterations == result_prealloc.iterations
        @test result_default.status == result_prealloc.status
        @test norm(result_default.sol - result_prealloc.sol) < 1e-12
        @test abs(result_default.residual - result_prealloc.residual) < 1e-12
    end

    @testset "Pre-allocation flag defaults to false" begin
        # Verify that the default behavior doesn't change
        N = 2
        T = 3
        state_dim = 2
        control_dim = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = fill(primal_dim, N)
        θs = setup_problem_parameter_variables(fill(state_dim, N))

        J1(z1, z2; θ=nothing) = sum(z1 .^ 2)
        J2(z1, z2; θ=nothing) = sum(z2 .^ 2)
        Js = Dict(1 => J1, 2 => J2)

        function make_dyn(idx)
            return function(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                dyn = mapreduce(vcat, 1:T) do t
                    xs[t+1] - xs[t] - 0.5 * us[t]
                end
                ic = xs[1] - θs[idx]
                vcat(dyn, ic)
            end
        end
        gs = [make_dyn(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim
        )

        params = Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0])

        # This should work without passing preallocate (default=false)
        result = run_nonlinear_solver(
            precomputed, params, G;
            max_iters=50, tol=1e-6
        )

        @test result.sol isa Vector{Float64}
        @test result.status in [:solved, :solved_initial_point, :max_iters_reached]
    end
end
