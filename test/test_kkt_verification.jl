#=
    Tests for KKT verification utilities: evaluate_kkt_residuals and verify_kkt_solution

    Following TDD - these tests define the expected behavior.
=#

using Test
using MixedHierarchyGames
using MixedHierarchyGames: strip_policy_constraints, compute_K_evals
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm
using Symbolics: @variables
using TrajectoryGamesBase: unflatten_trajectory

@testset "KKT Verification Utilities" begin

    @testset "evaluate_kkt_residuals" begin

        @testset "Basic functionality - simple symbolic expressions" begin
            # Create simple symbolic KKT conditions: π = [z1 - 1, z2 - 2]
            # At solution z = [1, 2], residuals should be [0, 0]
            @variables z1 z2

            πs = Dict(
                1 => [z1 - 1.0],
                2 => [z2 - 2.0]
            )
            all_variables = [z1, z2]
            sol = [1.0, 2.0]  # Exact solution
            θs = Dict{Int, Vector}()  # No parameters
            parameter_values = Dict{Int, Vector{Float64}}()

            residuals = evaluate_kkt_residuals(πs, all_variables, sol, θs, parameter_values)

            @test length(residuals) == 2
            @test norm(residuals) < 1e-10
        end

        @testset "Non-zero residuals for non-solution" begin
            @variables z1 z2

            πs = Dict(
                1 => [z1 - 1.0],
                2 => [z2 - 2.0]
            )
            all_variables = [z1, z2]
            sol = [0.5, 1.5]  # Not the solution
            θs = Dict{Int, Vector}()
            parameter_values = Dict{Int, Vector{Float64}}()

            residuals = evaluate_kkt_residuals(πs, all_variables, sol, θs, parameter_values)

            @test length(residuals) == 2
            @test residuals[1] ≈ -0.5  # z1 - 1 = 0.5 - 1 = -0.5
            @test residuals[2] ≈ -0.5  # z2 - 2 = 1.5 - 2 = -0.5
        end

        @testset "With parameter values" begin
            @variables z1 θ1

            # KKT condition: z1 = θ1 (initial condition constraint)
            πs = Dict(1 => [z1 - θ1])
            all_variables = [z1]
            θs = Dict(1 => [θ1])
            sol = [5.0]  # z1 = 5
            parameter_values = Dict(1 => [5.0])  # θ1 = 5

            residuals = evaluate_kkt_residuals(πs, all_variables, sol, θs, parameter_values)

            @test length(residuals) == 1
            @test abs(residuals[1]) < 1e-10  # z1 - θ1 = 5 - 5 = 0
        end

        @testset "With parameter values - non-zero residual" begin
            @variables z1 θ1

            πs = Dict(1 => [z1 - θ1])
            all_variables = [z1]
            θs = Dict(1 => [θ1])
            sol = [5.0]  # z1 = 5
            parameter_values = Dict(1 => [3.0])  # θ1 = 3

            residuals = evaluate_kkt_residuals(πs, all_variables, sol, θs, parameter_values)

            @test residuals[1] ≈ 2.0  # z1 - θ1 = 5 - 3 = 2
        end

        @testset "Multiple players with multiple conditions" begin
            @variables z1 z2 z3 z4

            # 2 players, 2 conditions each
            πs = Dict(
                1 => [z1 - 1.0, z2 - 2.0],
                2 => [z3 - 3.0, z4 - 4.0]
            )
            all_variables = [z1, z2, z3, z4]
            sol = [1.0, 2.0, 3.0, 4.0]
            θs = Dict{Int, Vector}()
            parameter_values = Dict{Int, Vector{Float64}}()

            residuals = evaluate_kkt_residuals(πs, all_variables, sol, θs, parameter_values)

            @test length(residuals) == 4
            @test norm(residuals) < 1e-10
        end

        @testset "should_enforce throws on bad solution" begin
            @variables z1

            πs = Dict(1 => [z1 - 1.0])
            all_variables = [z1]
            sol = [0.0]  # Wrong solution
            θs = Dict{Int, Vector}()
            parameter_values = Dict{Int, Vector{Float64}}()

            # Should throw AssertionError when should_enforce=true
            @test_throws AssertionError evaluate_kkt_residuals(
                πs, all_variables, sol, θs, parameter_values;
                should_enforce=true, tol=1e-6
            )
        end

        @testset "should_enforce does not throw on good solution" begin
            @variables z1

            πs = Dict(1 => [z1 - 1.0])
            all_variables = [z1]
            sol = [1.0]  # Correct solution
            θs = Dict{Int, Vector}()
            parameter_values = Dict{Int, Vector{Float64}}()

            # Should NOT throw when solution is valid
            residuals = evaluate_kkt_residuals(
                πs, all_variables, sol, θs, parameter_values;
                should_enforce=true, tol=1e-6
            )
            @test norm(residuals) < 1e-6
        end

        @testset "verbose output does not error" begin
            @variables z1

            πs = Dict(1 => [z1 - 1.0])
            all_variables = [z1]
            sol = [1.0]
            θs = Dict{Int, Vector}()
            parameter_values = Dict{Int, Vector{Float64}}()

            # Should not error with verbose=true
            residuals = evaluate_kkt_residuals(
                πs, all_variables, sol, θs, parameter_values;
                verbose=true
            )
            @test length(residuals) == 1
        end

        @testset "Player ordering is consistent" begin
            @variables z1 z2

            # Define players in reverse order
            πs = Dict(
                2 => [z2 - 2.0],
                1 => [z1 - 1.0]
            )
            all_variables = [z1, z2]
            sol = [1.0, 2.0]
            θs = Dict{Int, Vector}()
            parameter_values = Dict{Int, Vector{Float64}}()

            residuals = evaluate_kkt_residuals(πs, all_variables, sol, θs, parameter_values)

            # Should still work correctly regardless of Dict insertion order
            @test norm(residuals) < 1e-10
        end

    end

    @testset "verify_kkt_solution" begin
        # Use the 3-player chain setup which is known to work
        # Hierarchy: P2 → P1, P2 → P3 (P2 is leader)

        function setup_three_player_problem()
            G = SimpleDiGraph(3)
            add_edge!(G, 2, 1)  # P2 leads P1
            add_edge!(G, 2, 3)  # P2 leads P3

            state_dim = 2
            control_dim = 2
            T = 3
            N = 3

            primal_dim = (state_dim + control_dim) * (T + 1)
            primal_dims = fill(primal_dim, N)

            θs = setup_problem_parameter_variables(fill(state_dim, N))

            # Cost functions - each player wants to minimize own state/control
            function make_cost(player_idx)
                return function (z1, z2, z3; θ=nothing)
                    zs = [z1, z2, z3]
                    z = zs[player_idx]
                    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                    state_cost = sum(sum(x.^2) for x in xs)
                    control_cost = sum(sum(u.^2) for u in us)
                    return state_cost + 0.1 * control_cost
                end
            end
            Js = Dict(i => make_cost(i) for i in 1:N)

            # Single integrator dynamics
            Δt = 0.1
            function make_constraints(i)
                return function (z)
                    (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                    dyn = mapreduce(vcat, 1:T) do t
                        xs[t+1] - xs[t] - Δt * us[t]
                    end
                    ic = xs[1] - θs[i]
                    vcat(dyn, ic)
                end
            end
            gs = [make_constraints(i) for i in 1:N]

            x0 = Dict(
                1 => [1.0, 0.0],
                2 => [-1.0, 1.0],
                3 => [0.0, -1.0]
            )

            return G, Js, gs, primal_dims, θs, state_dim, control_dim, x0
        end

        @testset "Basic verification with NonlinearSolver" begin
            G, Js, gs, primal_dims, θs, state_dim, control_dim, x0 = setup_three_player_problem()

            solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; verbose=false)
            result = solve_raw(solver, x0; verbose=false)

            # Verify - should have small residuals for a converged solution
            residuals = verify_kkt_solution(solver, result.sol, θs, x0)

            @test length(residuals) > 0
            @test norm(residuals) < 1e-6
        end

        @testset "should_enforce works with solver" begin
            G, Js, gs, primal_dims, θs, state_dim, control_dim, x0 = setup_three_player_problem()

            solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; verbose=false)
            result = solve_raw(solver, x0; verbose=false)

            # Should NOT throw for valid solution
            residuals = verify_kkt_solution(solver, result.sol, θs, x0; should_enforce=true, tol=1e-6)
            @test norm(residuals) < 1e-6
        end

        @testset "Returns correct residual vector length" begin
            G, Js, gs, primal_dims, θs, state_dim, control_dim, x0 = setup_three_player_problem()

            solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; verbose=false)
            result = solve_raw(solver, x0; verbose=false)

            residuals = verify_kkt_solution(solver, result.sol, θs, x0)

            # Residuals should be a non-empty vector of Float64
            @test residuals isa Vector{Float64}
            @test length(residuals) > 0
        end

    end

end
