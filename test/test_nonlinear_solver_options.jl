# make_standard_two_player_problem is provided by testing_utils.jl (included in runtests.jl)

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

    @testset "use_sparse normalizes Bool to Symbol" begin
        # Bool values are normalized to Symbol at construction
        opts_true = NonlinearSolverOptions(use_sparse=true)
        @test opts_true.use_sparse === :always

        opts_false = NonlinearSolverOptions(use_sparse=false)
        @test opts_false.use_sparse === :never

        # Symbol values stored directly
        opts_auto = NonlinearSolverOptions(use_sparse=:auto)
        @test opts_auto.use_sparse === :auto

        opts_always = NonlinearSolverOptions(use_sparse=:always)
        @test opts_always.use_sparse === :always

        opts_never = NonlinearSolverOptions(use_sparse=:never)
        @test opts_never.use_sparse === :never
    end

    @testset "use_sparse validates Symbol values" begin
        @test_throws ArgumentError NonlinearSolverOptions(use_sparse=:bogus)
        @test_throws ArgumentError NonlinearSolverOptions(use_sparse=:sparse)
    end

    @testset "Field validation" begin
        # max_iters must be positive
        @test_throws ArgumentError NonlinearSolverOptions(max_iters=0)
        @test_throws ArgumentError NonlinearSolverOptions(max_iters=-1)

        # tol must be positive
        @test_throws ArgumentError NonlinearSolverOptions(tol=0.0)
        @test_throws ArgumentError NonlinearSolverOptions(tol=-1e-6)

        # regularization must be non-negative
        @test_throws ArgumentError NonlinearSolverOptions(regularization=-0.001)
        @test_nowarn NonlinearSolverOptions(regularization=0.0)  # zero is ok
    end

    @testset "NonlinearSolver stores NonlinearSolverOptions" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        @test solver.options isa NonlinearSolverOptions
    end

    @testset "NonlinearSolver constructor passes options through" begin
        prob = make_standard_two_player_problem()
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
        # NonlinearSolverOptions can be constructed from a NamedTuple (deprecated path)
        nt = (;
            max_iters=100, tol=1e-6, verbose=false,
            linesearch_method=:geometric, recompute_policy_in_linesearch=true,
            use_sparse=:auto, show_progress=false, regularization=0.0
        )
        opts = @test_deprecated NonlinearSolverOptions(nt)
        @test opts isa NonlinearSolverOptions
        @test opts.max_iters == 100
        @test opts.tol == 1e-6
        @test opts.linesearch_method == :geometric
    end

    @testset "NamedTuple constructor validates through keyword path" begin
        # Invalid linesearch should be caught even via NamedTuple constructor
        bad_nt = (;
            max_iters=100, tol=1e-6, verbose=false,
            linesearch_method=:bogus, recompute_policy_in_linesearch=true,
            use_sparse=:auto, show_progress=false, regularization=0.0
        )
        @test_throws ArgumentError @test_deprecated NonlinearSolverOptions(bad_nt)
    end

    @testset "NamedTuple constructor handles partial NamedTuples with defaults" begin
        # Only specify a subset of fields — rest use defaults
        partial_nt = (; max_iters=50, tol=1e-8)
        opts = @test_deprecated NonlinearSolverOptions(partial_nt)
        @test opts.max_iters == 50
        @test opts.tol == 1e-8
        @test opts.linesearch_method == :geometric  # default
        @test opts.use_sparse == :auto  # default
    end

    @testset "NamedTuple property access still works" begin
        # The options struct should support property access like NamedTuple did
        opts = NonlinearSolverOptions(max_iters=50)
        @test opts.max_iters == 50
        @test opts.tol == 1e-6  # default
    end

    @testset "Base.show produces readable output" begin
        opts = NonlinearSolverOptions()
        s = sprint(show, opts)
        @test contains(s, "NonlinearSolverOptions")
        @test contains(s, "max_iters=100")
        @test contains(s, "tol=1e-6") || contains(s, "tol=1.0e-6")
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

    @testset "_merge_options validates overrides" begin
        opts = NonlinearSolverOptions()
        @test_throws ArgumentError MixedHierarchyGames._merge_options(opts; linesearch_method=:bogus)
        @test_throws ArgumentError MixedHierarchyGames._merge_options(opts; max_iters=-1)
        @test_throws ArgumentError MixedHierarchyGames._merge_options(opts; tol=-1.0)
        @test_throws ArgumentError MixedHierarchyGames._merge_options(opts; regularization=-0.01)
        @test_throws ArgumentError MixedHierarchyGames._merge_options(opts; use_sparse=:bogus)
    end

    @testset "_merge_options normalizes Bool use_sparse" begin
        opts = NonlinearSolverOptions()
        merged = MixedHierarchyGames._merge_options(opts; use_sparse=true)
        @test merged.use_sparse === :always
        merged2 = MixedHierarchyGames._merge_options(opts; use_sparse=false)
        @test merged2.use_sparse === :never
    end

    @testset "_merge_options integration: overrides work through solve_raw" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=200
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Override max_iters to 3 — should respect the override
        result = solve_raw(solver, params; max_iters=3)
        @test result.iterations <= 3
    end
end
