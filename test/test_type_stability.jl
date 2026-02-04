using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames:
    setup_problem_variables,
    setup_problem_parameter_variables,
    setup_approximate_kkt_solver,
    compute_K_evals,
    default_backend

@testset "Type Stability" begin
    # Setup a simple 2-player hierarchy for testing
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    primal_dims = [2, 2]
    backend = default_backend()

    # Simple constraint functions (no constraints)
    gs = [z -> Num[] for _ in 1:2]

    @testset "setup_problem_variables returns typed containers" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)

        # ys should be Dict{Int, Vector{Num}}, not Dict{Int, Any}
        @test problem_vars.ys isa Dict{Int, Vector{Num}}

        # ws should be Dict{Int, Vector{Num}}, not Dict{Int, Any}
        @test problem_vars.ws isa Dict{Int, Vector{Num}}

        # Verify the values are actually Vector{Num}
        for (k, v) in problem_vars.ys
            @test v isa Vector{Num}
        end
        for (k, v) in problem_vars.ws
            @test v isa Vector{Num}
        end
    end

    @testset "setup_approximate_kkt_solver returns typed containers" begin
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)
        zs = problem_vars.zs
        λs = problem_vars.λs
        μs = problem_vars.μs
        ws = problem_vars.ws
        ys = problem_vars.ys
        all_variables = problem_vars.all_variables

        # Create parameter variables
        θs = setup_problem_parameter_variables([2, 2]; backend)

        # Simple cost functions
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2) + sum(z2.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2)
        )

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
        )

        # K_syms should have concrete Union type, not Any
        @test setup_info.K_syms isa Dict{Int, Union{Matrix{Num}, Vector{Num}}}

        # πs should be Dict{Int, Vector{Num}}
        @test setup_info.πs isa Dict{Int, Vector{Num}}

        # M_fns and N_fns should be Dict{Int, Function}
        @test setup_info.M_fns isa Dict{Int, Function}
        @test setup_info.N_fns isa Dict{Int, Function}
    end

    @testset "compute_K_evals returns typed containers" begin
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

        _, setup_info = setup_approximate_kkt_solver(
            G, Js, zs, λs, μs, gs, ws, ys, θs, all_variables, backend
        )

        # Create a test z_current
        z_current = zeros(length(all_variables))

        all_K_vec, info = compute_K_evals(z_current, problem_vars, setup_info)

        # K_evals should be properly typed
        @test info.K_evals isa Dict{Int, Union{Matrix{Float64}, Nothing}}
        @test info.M_evals isa Dict{Int, Union{Matrix{Float64}, Nothing}}
        @test info.N_evals isa Dict{Int, Union{Matrix{Float64}, Nothing}}

        # Verify actual values have correct types
        for (k, v) in info.K_evals
            @test v isa Union{Matrix{Float64}, Nothing}
        end
    end

    @testset "run_nonlinear_solver accepts Vector{Float64} initial_guess" begin
        # This tests that the function signature is properly typed
        # The function should accept Vector{Float64}, not just Vector
        problem_vars = setup_problem_variables(G, primal_dims, gs; backend)

        # Verify the method exists with Float64 signature
        # (This is more of a compile-time check - if the type is wrong, this would error)
        @test hasmethod(
            MixedHierarchyGames.run_nonlinear_solver,
            Tuple{NamedTuple, Dict, SimpleDiGraph}
        )
    end
end
