#=
    Project health tests - verify dependency hygiene and repo structure.

    These tests ensure that:
    1. All root [deps] are actually imported in src/
    2. All non-stdlib [deps] have [compat] entries
    3. No stale entries exist in [extras]
    4. experiments/Project.toml deps have compat entries
=#

using Test

@testset "Project Health" begin
    root_dir = joinpath(@__DIR__, "..")

    # Parse a minimal TOML-like section from Project.toml
    # We avoid adding TOML as a dep by parsing the simple structure ourselves
    function parse_toml_section(filepath, section)
        lines = readlines(filepath)
        in_section = false
        entries = Dict{String,String}()
        for line in lines
            stripped = strip(line)
            if startswith(stripped, "[$section]")
                in_section = true
                continue
            elseif startswith(stripped, "[") && in_section
                break
            elseif in_section && contains(stripped, "=") && !isempty(stripped) && !startswith(stripped, "#")
                key, val = split(stripped, "="; limit=2)
                entries[strip(key)] = strip(val)
            end
        end
        return entries
    end

    @testset "Root Project.toml dependency hygiene" begin
        root_toml = joinpath(root_dir, "Project.toml")
        deps = parse_toml_section(root_toml, "deps")
        compat = parse_toml_section(root_toml, "compat")

        # Standard library packages that don't need compat entries
        stdlib_packages = Set(["LinearAlgebra", "SparseArrays", "Random", "Test", "Logging"])

        # Packages actually imported in src/MixedHierarchyGames.jl
        expected_deps = Set([
            "TrajectoryGamesBase",
            "Graphs",
            "Symbolics",
            "SymbolicTracingUtils",
            "ParametricMCPs",
            "BlockArrays",
            "LinearAlgebra",
            "SparseArrays",
            "LinearSolve",
            "SciMLBase",
            "TimerOutputs",
        ])

        @testset "No dead dependencies in [deps]" begin
            for dep in keys(deps)
                @test dep in expected_deps
            end
        end

        @testset "All expected deps are present" begin
            for dep in expected_deps
                @test dep in keys(deps)
            end
        end

        @testset "All non-stdlib [deps] have [compat] entries" begin
            for dep in keys(deps)
                if dep in stdlib_packages
                    continue
                end
                @test haskey(compat, dep)
            end
        end

        @testset "No stale [compat] entries" begin
            for key in keys(compat)
                if key == "julia"
                    continue
                end
                @test key in keys(deps)
            end
        end
    end

    @testset "Root Project.toml [extras] hygiene" begin
        root_toml = joinpath(root_dir, "Project.toml")
        extras = parse_toml_section(root_toml, "extras")

        # Only Test should remain in [extras] (used in [targets])
        @test Set(keys(extras)) == Set(["Test"])
    end

    @testset "No loose .jl files in experiments/" begin
        experiments_dir = joinpath(root_dir, "experiments")
        if isdir(experiments_dir)
            for f in readdir(experiments_dir)
                if endswith(f, ".jl")
                    @test false  # Loose .jl file found: $f
                end
            end
        end
    end

    @testset "experiments/Project.toml compat coverage" begin
        exp_toml = joinpath(root_dir, "experiments", "Project.toml")
        if isfile(exp_toml)
            deps = parse_toml_section(exp_toml, "deps")
            compat = parse_toml_section(exp_toml, "compat")

            stdlib_packages = Set(["LinearAlgebra", "SparseArrays", "Random", "Test", "Logging"])

            for dep in keys(deps)
                if dep in stdlib_packages
                    continue
                end
                # MixedHierarchyGames is local, no compat needed
                if dep == "MixedHierarchyGames"
                    continue
                end
                @test haskey(compat, dep)
            end
        end
    end

    @testset "No Docker artifacts at repo root" begin
        @test !isfile(joinpath(root_dir, "Dockerfile"))
        @test !isfile(joinpath(root_dir, "docker-compose.yml"))
        @test !isfile(joinpath(root_dir, ".dockerignore"))
    end

    @testset ".gitignore covers logs/" begin
        gitignore = joinpath(root_dir, ".gitignore")
        content = read(gitignore, String)
        @test contains(content, "logs/")
    end
end
