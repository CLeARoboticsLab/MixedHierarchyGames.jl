# Test that documentation builds successfully
# This test verifies the Documenter.jl infrastructure works correctly.

using Test

@testset "Documentation Build" begin
    docs_dir = joinpath(@__DIR__, "..", "docs")
    make_jl = joinpath(docs_dir, "make.jl")
    project_toml = joinpath(docs_dir, "Project.toml")
    src_dir = joinpath(docs_dir, "src")
    build_dir = joinpath(docs_dir, "build")

    @testset "Documentation files exist" begin
        @test isfile(make_jl)
        @test isfile(project_toml)
        @test isfile(joinpath(src_dir, "index.md"))
        @test isfile(joinpath(src_dir, "api.md"))
    end

    @testset "Documentation builds without errors" begin
        # Run the docs build in a subprocess to isolate it
        result = run(
            `$(Base.julia_cmd()) --project=$docs_dir $make_jl`;
            wait=true,
        )
        @test result.exitcode == 0

        # Verify build output exists
        @test isdir(build_dir)
        @test isfile(joinpath(build_dir, "index.html"))
        @test isfile(joinpath(build_dir, "api", "index.html"))
    end

    @testset "API reference contains exported symbols" begin
        api_html = read(joinpath(build_dir, "api", "index.html"), String)
        # Check that key exported types/functions are documented
        @test occursin("QPSolver", api_html)
        @test occursin("NonlinearSolver", api_html)
        @test occursin("HierarchyGame", api_html)
    end
end
