using Test
using MixedHierarchyGames: NonlinearSolverOptions

@testset "NonlinearSolverOptions" begin
    @testset "Default construction" begin
        opts = NonlinearSolverOptions()
        @test opts.max_iters == 100
        @test opts.tol == 1e-6
        @test opts.verbose == false
        @test opts.use_armijo == true
    end

    @testset "Custom construction" begin
        opts = NonlinearSolverOptions(max_iters=500, tol=1e-8, verbose=true, use_armijo=false)
        @test opts.max_iters == 500
        @test opts.tol == 1e-8
        @test opts.verbose == true
        @test opts.use_armijo == false
    end

    @testset "Partial custom construction" begin
        opts = NonlinearSolverOptions(max_iters=200)
        @test opts.max_iters == 200
        @test opts.tol == 1e-6       # default
        @test opts.verbose == false   # default
        @test opts.use_armijo == true # default
    end

    @testset "Is a concrete struct (not abstract or NamedTuple)" begin
        opts = NonlinearSolverOptions()
        @test isconcretetype(typeof(opts))
        @test opts isa NonlinearSolverOptions
        @test !(opts isa NamedTuple)
    end

    @testset "Field types are enforced" begin
        @test fieldtype(NonlinearSolverOptions, :max_iters) == Int
        @test fieldtype(NonlinearSolverOptions, :tol) == Float64
        @test fieldtype(NonlinearSolverOptions, :verbose) == Bool
        @test fieldtype(NonlinearSolverOptions, :use_armijo) == Bool
    end

    @testset "Type conversion in constructor" begin
        # Int tol should be converted to Float64
        opts = NonlinearSolverOptions(tol=1)
        @test opts.tol === 1.0
        @test opts.tol isa Float64
    end

    @testset "NonlinearSolver stores NonlinearSolverOptions" begin
        using Graphs: SimpleDiGraph, add_edge!
        using MixedHierarchyGames: NonlinearSolver, HierarchyProblem

        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)
        prob = HierarchyProblem(
            G, Dict(1 => identity), [identity, identity],
            [2, 2], Dict(1 => [1.0]), 1, 1
        )
        # Direct construction with NonlinearSolverOptions
        opts = NonlinearSolverOptions(max_iters=50, tol=1e-4)
        solver = NonlinearSolver(prob, (;), opts)
        @test solver.options isa NonlinearSolverOptions
        @test solver.options.max_iters == 50
        @test solver.options.tol == 1e-4
    end

    @testset "Backward compatibility: NamedTuple converted to NonlinearSolverOptions" begin
        using Graphs: SimpleDiGraph, add_edge!
        using MixedHierarchyGames: NonlinearSolver, HierarchyProblem

        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)
        prob = HierarchyProblem(
            G, Dict(1 => identity), [identity, identity],
            [2, 2], Dict(1 => [1.0]), 1, 1
        )
        # Legacy construction with NamedTuple should still work
        nt = (; max_iters=75, tol=1e-5, verbose=true, use_armijo=false)
        solver = NonlinearSolver(prob, (;), nt)
        @test solver.options isa NonlinearSolverOptions
        @test solver.options.max_iters == 75
        @test solver.options.tol == 1e-5
        @test solver.options.verbose == true
        @test solver.options.use_armijo == false
    end
end
