using Test
using TOML

@testset "Dependency Hygiene" begin
    root_toml = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))

    @testset "Root Project.toml only contains src/ dependencies" begin
        root_deps = Set(keys(root_toml["deps"]))

        # These are the only packages that src/MixedHierarchyGames.jl actually imports.
        # Stdlib packages (LinearAlgebra, SparseArrays) are included because they must
        # be declared in [deps] for the package to load correctly.
        expected_deps = Set([
            "BlockArrays",
            "Graphs",
            "LinearAlgebra",
            "LinearSolve",
            "ParametricMCPs",
            "SciMLBase",
            "SparseArrays",
            "SymbolicTracingUtils",
            "Symbolics",
            "TimerOutputs",
            "TrajectoryGamesBase",
        ])

        extra_deps = setdiff(root_deps, expected_deps)
        missing_deps = setdiff(expected_deps, root_deps)

        @test isempty(extra_deps) || error("Unexpected deps in root Project.toml: $extra_deps — these likely belong in experiments/Project.toml or test/Project.toml")
        @test isempty(missing_deps) || error("Missing expected deps in root Project.toml: $missing_deps")
    end

    @testset "Root [extras] only contains test targets" begin
        root_extras = Set(keys(get(root_toml, "extras", Dict())))
        # Only Test should be in extras (for [targets] test = ["Test"])
        @test root_extras == Set(["Test"]) || root_extras == Set()
    end
end
