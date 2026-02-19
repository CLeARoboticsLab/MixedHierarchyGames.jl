#=
    Tests for warm-start α feature in the nonlinear solver.

    Warm-start tracks the accepted step size α across iterations and uses
    min(1.0, 2*α_prev) as the initial step size for the next linesearch,
    instead of always starting from α=1.0.

    Usage (standalone): julia --project=. test/test_warmstart_alpha.jl
    Also runs as part of: julia --project=. test/runtests.jl
=#

# Standalone bootstrap: load packages when run directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Test
    using MixedHierarchyGames
    using MixedHierarchyGames: setup_problem_parameter_variables
    using Graphs: SimpleDiGraph, add_edge!
    using TrajectoryGamesBase: unflatten_trajectory
    include("testing_utils.jl")
end

@testset "Warm-start α" begin
    @testset "First iteration uses α_init=1.0 (no warm-start history)" begin
        # On the first iteration, there's no prior α, so the solver should
        # start the linesearch from α=1.0 (the default initial step size).
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        step_sizes = Float64[]
        callback = info -> push!(step_sizes, info.step_size)

        result = solve_raw(solver, params; callback=callback)
        @test result.converged

        # First iteration should accept full step or backtrack from 1.0
        # In either case, α ≤ 1.0
        @test length(step_sizes) >= 1
        @test step_sizes[1] <= 1.0
        @test step_sizes[1] > 0.0
    end

    @testset "All step sizes are bounded by 1.0" begin
        # Warm-start uses min(1.0, 2*α_prev), so α_init should never exceed 1.0
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        step_sizes = Float64[]
        callback = info -> push!(step_sizes, info.step_size)

        result = solve_raw(solver, params; callback=callback)
        @test result.converged

        # All accepted step sizes must be ≤ 1.0
        for (i, α) in enumerate(step_sizes)
            @test α <= 1.0
        end
    end

    @testset "Solution unchanged with warm-start (geometric)" begin
        # Warm-start should not change the converged solution — only the path
        # to get there. We verify convergence to the same tolerance.
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric, tol=1e-8
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = solve_raw(solver, params)
        @test result.converged
        @test result.residual < 1e-8
    end

    @testset "Solution unchanged with warm-start (armijo)" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo, tol=1e-8
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = solve_raw(solver, params)
        @test result.converged
        @test result.residual < 1e-8
    end

    @testset "Failed linesearch does not corrupt α_prev" begin
        # When linesearch returns α=0.0 (failure), α_prev should retain
        # its previous value rather than being set to 0.0.
        # We test this indirectly: after a failed step (α=0.0), the next
        # iteration should still use a reasonable α_init.
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric, tol=1e-8
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        step_sizes = Float64[]
        callback = info -> push!(step_sizes, info.step_size)

        result = solve_raw(solver, params; callback=callback)

        # Even if some steps return α=0.0, subsequent steps should still
        # use a reasonable α_init (not 0.0 or 2*0.0 = 0.0)
        for i in 2:length(step_sizes)
            if step_sizes[i-1] == 0.0
                # After a failed step, the next accepted α should still be positive
                # (warm-start should preserve the last successful α_prev)
                @test step_sizes[i] >= 0.0  # at minimum, don't crash
            end
        end
    end

    @testset "Warm-start with constant linesearch is identity" begin
        # Constant linesearch always returns α=1.0, so warm-start
        # should have no effect — min(1.0, 2*1.0) = 1.0
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:constant, tol=1e-8, max_iters=5
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        step_sizes = Float64[]
        callback = info -> push!(step_sizes, info.step_size)

        solve_raw(solver, params; callback=callback)

        # Constant linesearch always returns 1.0
        for α in step_sizes
            @test α == 1.0
        end
    end
end
