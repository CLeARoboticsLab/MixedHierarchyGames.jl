using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory

"""Helper to create a minimal 2-player nonlinear problem for options integration tests."""
function _make_options_test_problem(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = state_dim * (T + 1) + control_dim * (T + 1)
    primal_dims = fill(primal_dim_per_player, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
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

@testset "NonlinearSolverOptions" begin
    @testset "Default construction" begin
        opts = NonlinearSolverOptions()
        @test opts.max_iters == 100
        @test opts.tol == 1e-6
        @test opts.verbose == false
        @test opts.linesearch_method == :geometric
        @test opts.recompute_policy_in_linesearch == true
        @test opts.use_sparse == :auto
        @test opts.show_progress == false
        @test opts.regularization == 0.0
    end

    @testset "Custom construction with keyword arguments" begin
        opts = NonlinearSolverOptions(
            max_iters=200,
            tol=1e-8,
            verbose=true,
            linesearch_method=:armijo,
            recompute_policy_in_linesearch=false,
            use_sparse=:always,
            show_progress=true,
            regularization=1e-4
        )
        @test opts.max_iters == 200
        @test opts.tol == 1e-8
        @test opts.verbose == true
        @test opts.linesearch_method == :armijo
        @test opts.recompute_policy_in_linesearch == false
        @test opts.use_sparse == :always
        @test opts.show_progress == true
        @test opts.regularization == 1e-4
    end

    @testset "Partial keyword construction uses defaults for unspecified fields" begin
        opts = NonlinearSolverOptions(max_iters=50, verbose=true)
        @test opts.max_iters == 50
        @test opts.verbose == true
        # Remaining fields use defaults
        @test opts.tol == 1e-6
        @test opts.linesearch_method == :geometric
        @test opts.recompute_policy_in_linesearch == true
        @test opts.use_sparse == :auto
        @test opts.show_progress == false
        @test opts.regularization == 0.0
    end

    @testset "Concrete type" begin
        opts = NonlinearSolverOptions()
        @test isconcretetype(typeof(opts))
        @test opts isa NonlinearSolverOptions
    end

    @testset "Linesearch method validation" begin
        @test_nowarn NonlinearSolverOptions(linesearch_method=:armijo)
        @test_nowarn NonlinearSolverOptions(linesearch_method=:geometric)
        @test_nowarn NonlinearSolverOptions(linesearch_method=:constant)
        @test_throws ArgumentError NonlinearSolverOptions(linesearch_method=:invalid)
    end

    @testset "use_sparse accepts Bool for backward compatibility" begin
        opts_true = NonlinearSolverOptions(use_sparse=true)
        @test opts_true.use_sparse === true

        opts_false = NonlinearSolverOptions(use_sparse=false)
        @test opts_false.use_sparse === false

        opts_auto = NonlinearSolverOptions(use_sparse=:auto)
        @test opts_auto.use_sparse === :auto

        opts_always = NonlinearSolverOptions(use_sparse=:always)
        @test opts_always.use_sparse === :always

        opts_never = NonlinearSolverOptions(use_sparse=:never)
        @test opts_never.use_sparse === :never
    end

    @testset "NonlinearSolver stores NonlinearSolverOptions" begin
        prob = _make_options_test_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        @test solver.options isa NonlinearSolverOptions
    end

    @testset "NonlinearSolver constructor passes options through" begin
        prob = _make_options_test_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=42,
            tol=1e-10,
            verbose=true,
            linesearch_method=:armijo,
            regularization=1e-3,
            show_progress=true
        )
        @test solver.options.max_iters == 42
        @test solver.options.tol == 1e-10
        @test solver.options.verbose == true
        @test solver.options.linesearch_method == :armijo
        @test solver.options.regularization == 1e-3
        @test solver.options.show_progress == true
    end

    @testset "NamedTuple backward compatibility" begin
        # NonlinearSolverOptions can be constructed from a NamedTuple
        nt = (;
            max_iters=100, tol=1e-6, verbose=false,
            linesearch_method=:geometric, recompute_policy_in_linesearch=true,
            use_sparse=:auto, show_progress=false, regularization=0.0
        )
        opts = NonlinearSolverOptions(nt)
        @test opts isa NonlinearSolverOptions
        @test opts.max_iters == 100
        @test opts.tol == 1e-6
        @test opts.linesearch_method == :geometric
    end

    @testset "NamedTuple property access still works" begin
        # The options struct should support property access like NamedTuple did
        opts = NonlinearSolverOptions(max_iters=50)
        @test opts.max_iters == 50
        @test opts.tol == 1e-6  # default
    end
end

@testset "_merge_options" begin
    @testset "no overrides returns original options" begin
        opts = NonlinearSolverOptions(max_iters=42, tol=1e-8)
        merged = MixedHierarchyGames._merge_options(opts)
        @test merged.max_iters == 42
        @test merged.tol == 1e-8
        @test merged.verbose == false
        @test merged.linesearch_method == :geometric
    end

    @testset "single override replaces that field" begin
        opts = NonlinearSolverOptions()
        merged = MixedHierarchyGames._merge_options(opts; max_iters=200)
        @test merged.max_iters == 200
        # Remaining fields unchanged
        @test merged.tol == 1e-6
        @test merged.verbose == false
        @test merged.linesearch_method == :geometric
    end

    @testset "multiple overrides replace those fields" begin
        opts = NonlinearSolverOptions()
        merged = MixedHierarchyGames._merge_options(opts;
            max_iters=50, tol=1e-10, verbose=true, linesearch_method=:armijo)
        @test merged.max_iters == 50
        @test merged.tol == 1e-10
        @test merged.verbose == true
        @test merged.linesearch_method == :armijo
        # Unchanged
        @test merged.recompute_policy_in_linesearch == true
        @test merged.use_sparse == :auto
        @test merged.show_progress == false
        @test merged.regularization == 0.0
    end

    @testset "nothing overrides are ignored (use base option)" begin
        opts = NonlinearSolverOptions(max_iters=42, verbose=true)
        merged = MixedHierarchyGames._merge_options(opts;
            max_iters=nothing, verbose=nothing, tol=1e-10)
        @test merged.max_iters == 42       # nothing → kept from opts
        @test merged.verbose == true       # nothing → kept from opts
        @test merged.tol == 1e-10          # overridden
    end

    @testset "all fields can be overridden" begin
        opts = NonlinearSolverOptions()
        merged = MixedHierarchyGames._merge_options(opts;
            max_iters=200, tol=1e-12, verbose=true,
            linesearch_method=:constant,
            recompute_policy_in_linesearch=false,
            use_sparse=:always, show_progress=true,
            regularization=0.01)
        @test merged.max_iters == 200
        @test merged.tol == 1e-12
        @test merged.verbose == true
        @test merged.linesearch_method == :constant
        @test merged.recompute_policy_in_linesearch == false
        @test merged.use_sparse == :always
        @test merged.show_progress == true
        @test merged.regularization == 0.01
    end

    @testset "returns NonlinearSolverOptions type" begin
        opts = NonlinearSolverOptions()
        merged = MixedHierarchyGames._merge_options(opts; max_iters=10)
        @test merged isa NonlinearSolverOptions
    end
end
