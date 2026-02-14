using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    setup_problem_variables,
    setup_problem_parameter_variables,
    setup_approximate_kkt_solver,
    preoptimize_nonlinear_solver,
    default_backend

@testset "Symbolics.Num conversion optimization" begin
    # Setup a simple 2-player hierarchy
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    primal_dims = [2, 2]
    backend = default_backend()
    gs = [z -> Num[] for _ in 1:2]

    problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
    zs = problem_vars.zs
    λs = problem_vars.λs
    μs = problem_vars.μs
    ws = problem_vars.ws
    ys = problem_vars.ys
    all_variables = problem_vars.all_variables

    θs = setup_problem_parameter_variables([2, 2]; backend)

    Js = Dict(
        1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
        2 => (z1, z2; θ=nothing) -> sum(z2.^2)
    )

    @testset "KKT condition vectors are already Num-typed" begin
        _, setup_info = setup_approximate_kkt_solver(
            G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
        )

        # πs values should contain Num elements (either Vector{Num} or BlockVector with Num)
        for (k, v) in setup_info.πs
            collected = collect(v)
            @test eltype(collected) == Num
        end
    end

    @testset "Parameter variables are already Num-typed" begin
        # θs values should be Vector{Num}
        for (k, v) in θs
            @test v isa Vector{Num}
            @test eltype(v) == Num
        end

        # K_syms values should have Num elements
        _, setup_info = setup_approximate_kkt_solver(
            G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
        )
        for (k, v) in setup_info.K_syms
            if !isempty(v)
                @test eltype(v) == Num
            end
        end
    end

    @testset "_ensure_num_vec skips conversion for Num vectors" begin
        # The helper should be a no-op when input is already Vector{Num}
        num_vec = Num.(ones(3))  # Create a Vector{Num}
        result = MixedHierarchyGames._ensure_num_vec(num_vec)
        @test result === num_vec  # Same object, no allocation

        # For non-Num input, it should convert
        int_vec = [1, 2, 3]
        result_converted = MixedHierarchyGames._ensure_num_vec(int_vec)
        @test eltype(result_converted) == Num
    end

    @testset "MCP construction produces identical F_sym" begin
        # Build the full preoptimized solver and verify F_sym is Vector{Num}
        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            backend, verbose=false
        )

        @test precomputed.F_sym isa Vector{Num}
        @test eltype(precomputed.F_sym) == Num
    end
end
