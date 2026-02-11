using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    QPSolver, NonlinearSolver, HierarchyProblem,
    solve, solve_raw,
    setup_problem_parameter_variables,
    split_solution_vector
using TrajectoryGamesBase: JointStrategy, OpenLoopStrategy, unflatten_trajectory

# make_θ helper is provided by testing_utils.jl (included in runtests.jl)

#=
    Tests for Vector-based parameter passing
    Feature: Allow solve(solver, [[1.0, 0.0], [0.0, 1.0]]) in addition to
             solve(solver, Dict(1 => [1.0, 0.0], 2 => [0.0, 1.0]))
=#

@testset "Vector-based parameter passing" begin
    @testset "QPSolver solve() accepts Vector of Vectors" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4]
        state_dim = 1
        control_dim = 1

        θ1_vec = make_θ(1, 1)
        θ2_vec = make_θ(2, 1)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)

        gs = [
            z -> [z[1] - θ1_vec[1]],
            z -> [z[1] - θ2_vec[1]],
        ]

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

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
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 3
        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = [primal_dim, primal_dim]

        θs = setup_problem_parameter_variables([state_dim, state_dim])

        function J1(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        function J2(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        Js = Dict(1 => J1, 2 => J2)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dynamics
        end
        gs = [make_dynamics(i) for i in 1:2]

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # New: pass as Vector of Vectors
        strategy = solve(solver, [[0.0, 0.0], [0.5, 0.5]])

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 2
    end

    @testset "NonlinearSolver solve_raw() accepts Vector of Vectors" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 3
        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = [primal_dim, primal_dim]

        θs = setup_problem_parameter_variables([state_dim, state_dim])

        function J1(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        function J2(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        Js = Dict(1 => J1, 2 => J2)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dynamics
        end
        gs = [make_dynamics(i) for i in 1:2]

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # New: pass as Vector of Vectors
        result = solve_raw(solver, [[0.0, 0.0], [0.5, 0.5]])

        @test result.sol isa Vector{Float64}
        @test result.converged isa Bool
    end

    @testset "Vector and Dict produce same results (QPSolver)" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4]
        state_dim = 1
        control_dim = 1

        θ1_vec = make_θ(1, 1)
        θ2_vec = make_θ(2, 1)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)

        gs = [
            z -> [z[1] - θ1_vec[1]],
            z -> [z[1] - θ2_vec[1]],
        ]
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

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
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 3
        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = [primal_dim, primal_dim]

        θs = setup_problem_parameter_variables([state_dim, state_dim])

        function J1(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        function J2(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        Js = Dict(1 => J1, 2 => J2)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dynamics
        end
        gs = [make_dynamics(i) for i in 1:2]

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

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
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 3
        primal_dim = (state_dim + control_dim) * (T + 1)
        primal_dims = [primal_dim, primal_dim]

        θs = setup_problem_parameter_variables([state_dim, state_dim])

        function J1(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        function J2(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        Js = Dict(1 => J1, 2 => J2)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dynamics
        end
        gs = [make_dynamics(i) for i in 1:2]

        solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim;
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

        # Final residual should be below tolerance
        @test result.converged
        @test residuals[end] < 1e-6

        # First residual should be larger than last
        @test residuals[1] > residuals[end]
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
end
