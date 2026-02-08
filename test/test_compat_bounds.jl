using Test
using TOML

@testset "Project.toml compat bounds" begin
    # Standard library packages that don't need compat entries
    stdlib_packages = Set([
        "LinearAlgebra", "SparseArrays", "Random", "Test", "Distributed",
        "Printf", "Logging", "Markdown", "Libdl", "InteractiveUtils",
        "Statistics", "Pkg", "Dates", "UUIDs", "Mmap", "Profile",
    ])

    @testset "Root Project.toml" begin
        project = TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
        deps = keys(get(project, "deps", Dict()))
        compat = keys(get(project, "compat", Dict()))

        non_stdlib_deps = setdiff(deps, stdlib_packages)

        for dep in sort(collect(non_stdlib_deps))
            @test dep in compat || dep == "julia"
        end
    end

    @testset "experiments/Project.toml" begin
        exp_toml = joinpath(@__DIR__, "..", "experiments", "Project.toml")
        if isfile(exp_toml)
            project = TOML.parsefile(exp_toml)
            deps = keys(get(project, "deps", Dict()))
            compat = keys(get(project, "compat", Dict()))

            # Exclude stdlib and local source packages
            sources = keys(get(project, "sources", Dict()))
            non_stdlib_deps = setdiff(deps, stdlib_packages, sources)

            for dep in sort(collect(non_stdlib_deps))
                @test dep in compat || dep == "julia"
            end
        end
    end
end
