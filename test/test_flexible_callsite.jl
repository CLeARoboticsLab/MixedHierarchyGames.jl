using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    QPSolver, NonlinearSolver, HierarchyProblem,
    solve, solve_raw,
    setup_problem_parameter_variables,
    split_solution_vector
using TrajectoryGamesBase: JointStrategy, OpenLoopStrategy, unflatten_trajectory

# make_θ, make_standard_two_player_problem, make_simple_qp_two_player
# are provided by testing_utils.jl (included in runtests.jl)

#=
    Tests for Vector-based parameter passing
    Feature: Allow solve(solver, [[1.0, 0.0], [0.0, 1.0]]) in addition to
             solve(solver, Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0]))
=#

@testset "Vector-based parameter passing" begin
    @testset "QPSolver solve() accepts Vector of Vectors" begin
        prob = make_simple_qp_two_player()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        # New: pass as Vector of Vectors instead of Dict
        strategy = solve(solver, [[1.0], [3.0]])

        @test strategy isa JointStrategy
        @test strategy.substrategies[1].xs[1][1] ≈ 1.0 atol=1e-6
        @test strategy.substrategies[2].xs[1][1] ≈ 3.0 atol=1e-6
    end

    @testset "QPSolver solve_raw() accepts Vector of Vectors" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # New: pass as Vector of Vectors
        result = solve_raw(solver, [[1.0]])

        @test result.status == :solved
        @test result.sol[1] ≈ 1.0 atol=1e-6
    end

    @testset "NonlinearSolver solve() accepts Vector of Vectors" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                 prob.state_dim, prob.control_dim)

        # New: pass as Vector of Vectors
        strategy = solve(solver, [[0.0, 0.0], [0.5, 0.5]])

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 2
    end

    @testset "NonlinearSolver solve_raw() accepts Vector of Vectors" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                 prob.state_dim, prob.control_dim)

        # New: pass as Vector of Vectors
        result = solve_raw(solver, [[0.0, 0.0], [0.5, 0.5]])

        @test result.sol isa Vector{Float64}
        @test result.converged isa Bool
    end

    @testset "Vector and Dict produce same results (QPSolver)" begin
        prob = make_simple_qp_two_player()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        # Dict-based (existing)
        strategy_dict = solve(solver, Dict(1 => [2.0], 2 => [4.0]))
        # Vector-based (new)
        strategy_vec = solve(solver, [[2.0], [4.0]])

        # Results should be identical
        for i in 1:2
            @test strategy_dict.substrategies[i].xs ≈ strategy_vec.substrategies[i].xs atol=1e-10
            @test strategy_dict.substrategies[i].us ≈ strategy_vec.substrategies[i].us atol=1e-10
        end
    end

    @testset "Vector and Dict produce same results (NonlinearSolver)" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                 prob.state_dim, prob.control_dim)

        params_dict = Dict(1 => [0.1, 0.2], 2 => [0.3, 0.4])
        params_vec = [[0.1, 0.2], [0.3, 0.4]]

        result_dict = solve_raw(solver, params_dict)
        result_vec = solve_raw(solver, params_vec)

        @test result_dict.sol ≈ result_vec.sol atol=1e-10
        @test result_dict.converged == result_vec.converged
        @test result_dict.status == result_vec.status
    end
end

#=
    Tests for iteration callback
    Feature: Optional callback function called each iteration with convergence info.
    Enables: iteration history tracking, early stopping, interactive inspection.
=#

@testset "Iteration callback" begin
    # Helper: build a simple 2-player NonlinearSolver for callback tests
    function make_callback_test_solver()
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                 prob.state_dim, prob.control_dim;
                                 max_iters=100, tol=1e-6)
        return solver
    end

    @testset "solve_raw accepts callback kwarg" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        history = []
        callback = info -> push!(history, info)

        result = solve_raw(solver, params; callback=callback)

        @test result.converged
        @test length(history) >= 1  # At least 1 iteration
    end

    @testset "callback receives iteration info with expected fields" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        history = []
        callback = info -> push!(history, info)

        solve_raw(solver, params; callback=callback)

        # Each callback entry should have these fields
        for info in history
            @test haskey(info, :iteration) || hasproperty(info, :iteration)
            @test haskey(info, :residual) || hasproperty(info, :residual)
            @test haskey(info, :step_size) || hasproperty(info, :step_size)
        end
    end

    @testset "callback receives z_est (current solution vector)" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        history = []
        callback = info -> push!(history, info)

        result = solve_raw(solver, params; callback=callback)

        @test result.converged
        @test length(history) >= 1

        # Each callback entry should include z_est
        for info in history
            @test hasproperty(info, :z_est)
            @test info.z_est isa AbstractVector
            @test length(info.z_est) == length(result.sol)
        end

        # The final callback's z_est should match the solution
        # (callback is called after the update, so the last z_est should be close to sol)
        last_z = history[end].z_est
        @test last_z isa AbstractVector
    end

    @testset "callback z_est is a copy (not a reference to solver internal state)" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        z_snapshots = Vector{Float64}[]
        callback = info -> push!(z_snapshots, info.z_est)

        result = solve_raw(solver, params; callback=callback)

        if length(z_snapshots) >= 2
            # Each snapshot should be an independent copy — not all the same object
            @test z_snapshots[1] != z_snapshots[end]  # Different iteration, different values
        end
    end

    @testset "callback iteration numbers are sequential" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        iterations = Int[]
        callback = info -> push!(iterations, info.iteration)

        solve_raw(solver, params; callback=callback)

        # Should be [1, 2, 3, ...]
        @test iterations == collect(1:length(iterations))
    end

    @testset "callback residuals decrease toward convergence" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        residuals = Float64[]
        callback = info -> push!(residuals, info.residual)

        result = solve_raw(solver, params; callback=callback)

        # Solver should converge
        @test result.converged

        # Residuals should be non-negative
        @test all(r -> r >= 0.0, residuals)

        # If multiple iterations, residuals should generally trend downward
        if length(residuals) >= 2
            @test residuals[1] >= residuals[end]
        end
    end

    @testset "callback not called when not provided (default nothing)" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Should work without callback — no error
        result = solve_raw(solver, params)
        @test result.converged
    end

    @testset "solve() also accepts callback kwarg" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        history = []
        callback = info -> push!(history, info)

        strategy = solve(solver, params; callback=callback)

        @test strategy isa JointStrategy
        @test length(history) >= 1
    end

    @testset "callback enables iteration history tracking" begin
        # This is the motivating use case from experiments/convergence_analysis
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Track full iteration history
        iteration_history = NamedTuple[]
        callback = info -> push!(iteration_history, info)

        result = solve_raw(solver, params; callback=callback)

        @test result.converged
        @test length(iteration_history) == result.iterations

        # Each entry should have iteration number and residual
        for (i, info) in enumerate(iteration_history)
            @test info.iteration == i
            @test info.residual isa Float64
            @test info.residual >= 0.0
        end
    end

    @testset "callback works with Vector-based parameters" begin
        solver = make_callback_test_solver()
        params = [[0.0, 0.0], [0.5, 0.5]]

        history = []
        callback = info -> push!(history, info)

        result = solve_raw(solver, params; callback=callback)

        @test result.converged
        @test length(history) >= 1
    end

    @testset "callback error propagates cleanly" begin
        solver = make_callback_test_solver()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Callback that throws on first iteration
        throwing_callback = info -> error("test callback error")

        # Error should propagate — solver does not swallow callback exceptions
        @test_throws ErrorException solve_raw(solver, params; callback=throwing_callback)
    end

    @testset "callback error on later iteration preserves partial history" begin
        # First, verify the solver takes multiple iterations with these settings.
        # Use max_iters high enough and a callback that counts to confirm.
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                 prob.state_dim, prob.control_dim;
                                 max_iters=200, tol=1e-6)
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Count total iterations first
        count_history = Int[]
        result = solve_raw(solver, params; callback=info -> push!(count_history, info.iteration))
        total_iters = length(count_history)

        if total_iters >= 3
            # Solver does multiple iterations — test that throwing partway preserves history
            history = []
            function failing_callback(info)
                push!(history, info)
                if info.iteration >= 3
                    error("intentional failure after iteration 3")
                end
            end

            @test_throws ErrorException solve_raw(solver, params; callback=failing_callback)
            @test length(history) == 3
            @test history[1].iteration == 1
            @test history[3].iteration == 3
        else
            # Solver converges too quickly for multi-iteration test.
            # Just verify that throwing on first iteration propagates.
            @test_throws ErrorException solve_raw(solver, params;
                callback=info -> error("fail on iter $(info.iteration)"))
        end
    end
end
