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
        prob = create_two_player_nonlinear_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        @test solver.options isa NonlinearSolverOptions
    end

    @testset "NonlinearSolver constructor passes options through" begin
        prob = create_two_player_nonlinear_problem()
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
